from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.annotation import (
    SYSTEM_PROMPT,
    AnnotationStats,
    QwenAnnotationClient,
    annotate_clips,
    annotation_system_prompt,
    sanitize_annotation_payload,
    sanitize_background,
    sanitize_entity_candidates,
)
from r2v_data_v2.v3.config import (
    DebugConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.schemas import (
    MAX_ANNOTATION_ENTITIES,
    AnnotationEntity,
    AnnotationState,
    ClipRecord,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    RawAnnotationPayload,
    ReferencesState,
    render_annotation_plain_text,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug: bool = False,
    repair_retries: int = 1,
    source_start_index: int = 0,
    source_limit: int | None = 100,
    source_allow_full_run: bool = False,
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(v3_config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(v3_config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(v3_config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    annotation_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_root / "source.json",
        run_root=writable / "runs" / "annotation",
        export_root=writable / "datasets" / "annotation-v1",
        source=SourceConfig(
            start_index=source_start_index,
            limit=source_limit,
            allow_full_run=source_allow_full_run,
        ),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(
                model=str(annotation_model),
                repair_retries=repair_retries,
            ),
            instruction_writer=QwenServiceConfig(model=str(annotation_model)),
            candidate_judge=QwenServiceConfig(model=str(annotation_model)),
            background_remove_judge=QwenServiceConfig(model=str(annotation_model)),
        ),
        sam3=Sam3Config(model_path=user_models / "sam3" / "checkpoint.pt"),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=(user_models / "Qwen-Image-Edit-2511-Object-Remover"),
        ),
        debug=DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _config_path(config: V3Config, tmp_path: Path) -> Path:
    path = tmp_path / "v3-annotation.yaml"
    source_limit = (
        "" if config.source.limit is None else f"  limit: {config.source.limit}\n"
    )
    path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        f"  start_index: {config.source.start_index}\n"
        f"{source_limit}"
        "  allow_full_run: "
        f"{str(config.source.allow_full_run).lower()}\n"
        "qwen:\n"
        "  annotation:\n"
        f"    model: {config.qwen.annotation.model}\n"
        f"    repair_retries: {config.qwen.annotation.repair_retries}\n"
        "  candidate_judge:\n"
        f"    model: {config.qwen.candidate_judge.model}\n"
        "  background_remove_judge:\n"
        f"    model: {config.qwen.background_remove_judge.model}\n"
        "debug:\n"
        f"  save_diagnostics: {str(config.debug.save_diagnostics).lower()}\n",
        encoding="utf-8",
    )
    return path


def _video(config: V3Config, name: str = "scene_1_0.mp4") -> Path:
    path = config.dataset_json.parent / "videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mock video")
    return path


def _write_source(
    config: V3Config,
    records: list[dict[str, object]],
) -> None:
    config.dataset_json.write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )


def _entity(
    phrase: str,
    *,
    reference_type: str = "subject",
    grounding_prompt: str | None = None,
) -> dict[str, object]:
    return {
        "reference_type": reference_type,
        "phrase": phrase,
        "grounding_prompt": (
            phrase.casefold() if grounding_prompt is None else grounding_prompt
        ),
    }


def _payload(
    *,
    template: str | None = None,
    caption: str | None = None,
    entities: list[object] | None = None,
    background: object = ...,
) -> dict[str, object]:
    caption_was_supplied = caption is not None
    text_was_supplied = template is not None or caption is not None
    if template is None:
        template = caption
    if entities is None and not text_was_supplied:
        entities = [
            _entity("A woman in a yellow coat"),
            _entity("a wooden table", reference_type="object"),
        ]
    elif entities is None:
        entities = []
    if background is ...:
        background = (
            {
                "phrase": "a sunlit plaza",
                "grounding_prompt": "the empty sunlit plaza",
            }
            if not text_was_supplied
            else None
        )
    if template is None:
        mentions = [
            f"{{{{entity_{index}}}}} remains present."
            for index in range(1, len(entities) + 1)
        ]
        if background is not None:
            mentions.append("The scene remains {{background}}.")
        template = " ".join(mentions) or "A quiet scene remains visible."
    elif caption_was_supplied:
        marker_sentences = [
            f"{{{{entity_{index}}}}} remains visible."
            for index in range(1, len(entities) + 1)
        ]
        if background is not None:
            marker_sentences.append("The environment remains {{background}}.")
        if marker_sentences:
            template = f"{template} {' '.join(marker_sentences)}"
    return {
        "entities": entities,
        "background": background,
        "instruction_template": template,
    }


class _FakeQwenClient(QwenAnnotationClient):
    def __init__(
        self,
        config: QwenAnnotationConfig,
        responses: list[str | Exception],
    ) -> None:
        self.config = config
        self._responses: Iterator[str | Exception] = iter(responses)
        self.requests: list[list[dict[str, object]]] = []

    def _request(self, messages: list[dict[str, object]]) -> str:
        self.requests.append(messages)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _storage_with_manifest(
    config: V3Config,
) -> tuple[RunStorage, str]:
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")
    stats = build_manifest(config, storage)
    assert stats.processed == 1
    clip_uid = next(storage.iter_clips()).clip_uid
    return storage, clip_uid


def _annotate_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, object] | str],
    *,
    repair_retries: int = 0,
) -> tuple[AnnotationStats, ClipRecord, _FakeQwenClient]:
    config = _config(
        tmp_path,
        monkeypatch,
        repair_retries=repair_retries,
    )
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    responses = [
        payload if isinstance(payload, str) else json.dumps(payload)
        for payload in payloads
    ]
    client = _FakeQwenClient(config.qwen.annotation, responses)
    stats = annotate_clips(config, storage, client=client)
    return stats, storage.read_clip(clip_uid), client


def _seed_ready_downstream(storage: RunStorage, clip_uid: str) -> None:
    clip = storage.read_clip(clip_uid)
    assert clip.annotation is not None
    summaries = {}
    for index, entity in enumerate(clip.annotation.entities):
        visible_frame_count = 7 if index == 0 else 3
        summaries[entity.entity_id] = EntityVisibilitySummary(
            status="ready",
            visible_frame_slots=list(range(visible_frame_count)),
            visible_frame_count=visible_frame_count,
            coverage_ratio=visible_frame_count / 10,
            qualifies=index == 0,
            per_frame_area_ratio=(
                [0.1] * visible_frame_count + [0.0] * (10 - visible_frame_count)
            ),
            per_frame_confidence=(
                [0.9] * visible_frame_count + [None] * (10 - visible_frame_count)
            ),
        )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary=summaries,
        ),
    )
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=[
                EntityReferenceState(
                    entity_id="e1",
                    status="ready",
                    reference_scope="local",
                    visible_region="upper_body",
                    whole_entity_recognizable=False,
                    identity_features_visible=True,
                    scope_reason="coherent upper body",
                    image_path=f"clips/{clip_uid}/selected/e1.png",
                    source_frame_index=2,
                )
            ]
        ),
    )
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
        ),
    )
    body = "\u4f7f\u7528{{image_1}}\u751f\u6210\u8fde\u7eed\u955c\u5934\u3002"
    legend = [
        InstructionLegendEntry(
            image_id="image_1",
            description="\u9ec4\u8272\u5916\u5957\u5973\u5b50",
        )
    ]
    storage.write_instruction(
        clip_uid,
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    storage.write_export(
        clip_uid,
        ExportState(accepted=True, reason=None),
    )


def test_manifest_creates_one_clip_json_without_stage_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video), "text": "draft"}])
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    stats = build_manifest(config, storage)

    clip = storage.read_clip(parse_clip_identity(video).clip_uid)
    assert stats.to_dict() == {
        "processed": 1,
        "skipped_existing": 0,
        "failed": 0,
    }
    assert clip.schema_version == "r2v.v3.clip.2"
    assert clip.source.video_path == str(video.resolve())
    assert clip.source.caption_raw == "draft"
    assert clip.source.metadata == {"text": "draft"}
    assert list(config.resolved_run_root.rglob("clip.json")) == [
        storage.clip_path(clip.clip_uid)
    ]
    assert not list(config.resolved_run_root.rglob("*manifest*.jsonl"))


def test_manifest_selection_limit_and_rerun_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, source_limit=5)
    records = [
        {"file_path": str(_video(config, f"scene_{index}_0.mp4"))} for index in range(8)
    ]
    _write_source(config, records)
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    first = build_manifest(config, storage)
    second = build_manifest(config, storage)

    assert first.processed == 5
    assert second.skipped_existing == 5
    assert len(list(storage.iter_clips())) == 5


def test_bad_manifest_source_does_not_block_valid_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [{"text": "missing path"}, {"file_path": str(video)}],
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    stats = build_manifest(config, storage)

    assert stats.processed == 1
    assert stats.failed == 1
    assert len(list(storage.iter_clips())) == 1


def test_manifest_source_identity_conflict_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    payload = storage.read_clip(clip_uid).model_dump(mode="json")
    payload["source"]["parent_video_id"] = "conflicting-parent"
    write_json_atomic(storage.clip_path(clip_uid), payload)

    stats = build_manifest(config, storage)

    assert stats.failed == 1
    assert stats.skipped_existing == 0


def test_source_selection_is_validated_and_fingerprinted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        source=replace(config.source, start_index=2, limit=5),
    )
    assert changed.fingerprint() != config.fingerprint()
    with pytest.raises(ValueError, match="source.limit is required"):
        _config(
            tmp_path,
            monkeypatch,
            source_limit=None,
            source_allow_full_run=False,
        )


def test_minimal_raw_annotation_schema_contains_only_expected_fields() -> None:
    schema = RawAnnotationPayload.model_json_schema()

    assert list(schema["properties"]) == [
        "entities",
        "background",
        "instruction_template",
    ]
    entity_schema = schema["$defs"]["RawAnnotationEntity"]
    background_schema = schema["$defs"]["RawBackgroundAnnotation"]
    assert set(entity_schema["properties"]) == {
        "reference_type",
        "phrase",
        "grounding_prompt",
    }
    assert set(background_schema["properties"]) == {
        "phrase",
        "grounding_prompt",
    }
    assert "entity_id" not in json.dumps(schema)
    assert "relations" not in json.dumps(schema)


@pytest.mark.parametrize(
    "field",
    [
        "category",
        "salience",
        "localization_scope",
        "selection_reason",
        "entity_id",
    ],
)
def test_raw_annotation_schema_strictly_rejects_old_entity_fields(
    field: str,
) -> None:
    payload = _payload()
    entity = payload["entities"][0]
    assert isinstance(entity, dict)
    entity[field] = "legacy"

    with pytest.raises(ValidationError, match=field):
        RawAnnotationPayload.model_validate(payload)


def test_raw_annotation_schema_strictly_rejects_relations() -> None:
    payload = _payload()
    payload["relations"] = []

    with pytest.raises(ValidationError, match="relations"):
        RawAnnotationPayload.model_validate(payload)


def test_entity_sanitizer_deduplicates_truncates_and_assigns_ids() -> None:
    candidates = [
        _entity("  Woman.  "),
        _entity("invalid", reference_type="scene"),
        _entity("woman"),
        _entity("Table", reference_type="object"),
        _entity("Band", reference_type="group"),
        _entity("Car", reference_type="object"),
        _entity("Bicycle", reference_type="object"),
        _entity("Boat", reference_type="object"),
    ]

    entities, warnings = sanitize_entity_candidates(candidates)

    assert [entity.entity_id for entity in entities] == [
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
    ]
    assert [entity.phrase for entity in entities] == [
        "Woman",
        "Table",
        "Band",
        "Car",
        "Bicycle",
    ]
    assert [entity.reference_type for entity in entities] == [
        "subject",
        "object",
        "group",
        "object",
        "object",
    ]
    assert "dropped_entity_reference_type:1" in warnings
    assert "dropped_duplicate_entity_phrase:2" in warnings
    assert "truncated_entity_candidates:5" in warnings


def test_annotation_state_accepts_schema_capacity_of_eight_contiguous_entities() -> (
    None
):
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type="object",
            phrase=f"object {index}",
            grounding_prompt=f"visible object {index}",
        )
        for index in range(1, MAX_ANNOTATION_ENTITIES + 1)
    ]

    annotation = AnnotationState(
        status="ready",
        instruction_template=" ".join(
            f"Object {index} {{{{entity_{index}}}}} remains visible."
            for index in range(1, MAX_ANNOTATION_ENTITIES + 1)
        ),
        entities=entities,
    )

    assert [entity.entity_id for entity in annotation.entities] == [
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "e8",
    ]


def test_annotation_state_loads_legacy_caption_without_template() -> None:
    annotation = AnnotationState.model_validate(
        {
            "status": "ready",
            "t2v_caption": "A legacy caption remains available.",
            "entities": [],
            "background": None,
            "reason": None,
        }
    )

    assert annotation.instruction_template == ""
    assert annotation.t2v_caption == "A legacy caption remains available."


def test_annotation_state_rejects_two_published_text_sources() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AnnotationState(
            status="ready",
            instruction_template="A new template is visible.",
            t2v_caption="A second caption is visible.",
        )


def test_failed_annotation_cannot_publish_instruction_template() -> None:
    with pytest.raises(ValidationError, match="must not publish"):
        AnnotationState(
            status="failed",
            instruction_template="A leaked template.",
            reason="annotation_failed",
        )


def test_annotation_state_rejects_more_than_eight_entities() -> None:
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type="object",
            phrase=f"object {index}",
            grounding_prompt=f"visible object {index}",
        )
        for index in range(1, MAX_ANNOTATION_ENTITIES + 2)
    ]

    with pytest.raises(ValidationError, match="at most 8 entities"):
        AnnotationState(
            status="ready",
            instruction_template="Six objects remain visible.",
            entities=entities,
        )


def test_annotation_state_rejects_noncontiguous_five_entity_ids() -> None:
    entity_ids = ["e1", "e2", "e3", "e4", "e6"]
    entities = [
        AnnotationEntity(
            entity_id=entity_id,
            reference_type="object",
            phrase=f"object {index}",
            grounding_prompt=f"visible object {index}",
        )
        for index, entity_id in enumerate(entity_ids, start=1)
    ]

    with pytest.raises(ValidationError, match="contiguous and ordered"):
        AnnotationState(
            status="ready",
            instruction_template="Five objects remain visible.",
            entities=entities,
        )


@pytest.mark.parametrize(
    ("candidate", "warning"),
    [
        (None, "dropped_entity_not_object:0"),
        (_entity("", grounding_prompt="person"), "dropped_entity_phrase:0"),
        (_entity("person", grounding_prompt=""), "dropped_entity_grounding_prompt:0"),
        (_entity("person", reference_type="scene"), "dropped_entity_reference_type:0"),
        (_entity("person <ref_subject_1>"), "dropped_entity_reference_token:0"),
    ],
)
def test_invalid_entity_candidate_is_dropped_locally(
    candidate: object,
    warning: str,
) -> None:
    entities, warnings = sanitize_entity_candidates([candidate])

    assert entities == []
    assert warning in warnings


def test_zero_valid_entities_still_produces_ready_annotation() -> None:
    annotation, issues, _ = sanitize_annotation_payload(_payload(entities=[]))

    assert issues == []
    assert annotation is not None
    assert annotation.status == "ready"
    assert annotation.entities == []


def test_background_null_is_valid() -> None:
    background, warnings = sanitize_background(None)

    assert background is None
    assert warnings == ()


@pytest.mark.parametrize(
    "background",
    [
        "room",
        {"phrase": "", "grounding_prompt": "room"},
        {"phrase": "room", "grounding_prompt": ""},
        {"phrase": "room <ref_bg_1>", "grounding_prompt": "room"},
        {
            "phrase": "room",
            "grounding_prompt": "room",
            "reference_worthy": True,
        },
    ],
)
def test_invalid_background_is_dropped_without_failing_annotation(
    background: object,
) -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(background=background)
    )

    assert issues == []
    assert annotation is not None
    assert annotation.status == "ready"
    assert annotation.background is None
    assert warnings


def test_annotation_writes_minimal_semantic_fields_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [_payload()],
    )

    assert stats.processed == 1
    assert clip.annotation is not None
    assert [entity.entity_id for entity in clip.annotation.entities] == [
        "e1",
        "e2",
    ]
    assert clip.annotation.background is not None
    serialized = clip.annotation.model_dump(mode="json")
    assert set(serialized) == {
        "status",
        "instruction_template",
        "t2v_caption",
        "entities",
        "background",
        "reason",
    }
    assert set(serialized["entities"][0]) == {
        "entity_id",
        "reference_type",
        "phrase",
        "grounding_prompt",
    }
    assert "relations" not in json.dumps(serialized)
    assert "<ref_" not in json.dumps(serialized)
    assert clip.annotation.instruction_template
    assert clip.annotation.t2v_caption == ""


def test_phrase_and_grounding_prompt_need_not_appear_in_template() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses a courtyard.",
            entities=[
                _entity(
                    "a red-haired woman",
                    grounding_prompt="woman with long red hair and a green coat",
                )
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.entities[0].phrase == "a red-haired woman"
    assert annotation.entities[0].grounding_prompt.endswith("green coat")
    assert (
        render_annotation_plain_text(
            annotation.instruction_template,
            annotation.entities,
            annotation.background,
        )
        == "a red-haired woman crosses a courtyard."
    )


def test_placeholder_represents_the_complete_entity_mention() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} kneels.",
            entities=[_entity("a bald monk in a robe")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert (
        render_annotation_plain_text(
            annotation.instruction_template,
            annotation.entities,
            annotation.background,
        )
        == "a bald monk in a robe kneels."
    )


def test_three_entity_markers_are_valid_once_each() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=("{{entity_1}} stands beside {{entity_2}} near {{entity_3}}."),
            entities=[
                _entity("woman"),
                _entity("boat", reference_type="object"),
                _entity("tower", reference_type="object"),
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.instruction_template.count("{{entity_") == 3


@pytest.mark.parametrize(
    ("template", "entities", "issue_code"),
    [
        (
            "A woman {{entity_0}} walks.",
            [_entity("woman")],
            "invalid_entity_marker",
        ),
        (
            "A woman {{ entity_1 }} walks.",
            [_entity("woman")],
            "invalid_entity_marker",
        ),
        (
            "A woman {{image_1}} walks.",
            [_entity("woman")],
            "invalid_annotation_image_marker",
        ),
        (
            "A woman <Image 1> walks.",
            [_entity("woman")],
            "invalid_annotation_image_marker",
        ),
        (
            "A woman <ref_subject_1> walks.",
            [_entity("woman")],
            "invalid_annotation_reference_token",
        ),
        (
            "A woman {{unknown_marker}} walks.",
            [_entity("woman")],
            "invalid_entity_marker",
        ),
        (
            "A woman {{entity_1 walks.",
            [_entity("woman")],
            "invalid_entity_marker",
        ),
    ],
)
def test_unsafe_internal_marker_structures_are_rejected(
    template: str,
    entities: list[object],
    issue_code: str,
) -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=template,
            entities=entities,
            background=None,
        )
    )

    assert annotation is None
    assert issue_code in {issue.code for issue in issues}


def test_missing_entity_marker_drops_only_that_proposal_and_renumbers() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template=("{{entity_1}} stands beside a dog while {{entity_3}} passes."),
            entities=[
                _entity("woman"),
                _entity("dog"),
                _entity("boat", reference_type="object"),
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.phrase for entity in annotation.entities] == ["woman", "boat"]
    assert [entity.entity_id for entity in annotation.entities] == ["e1", "e2"]
    assert annotation.instruction_template == (
        "{{entity_1}} stands beside a dog while {{entity_2}} passes."
    )
    assert "dropped_entity_missing_marker:2" in warnings


def test_duplicate_entity_placeholder_keeps_first_and_expands_later() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=("{{entity_1}} walks. Later {{entity_1}} stops."),
            entities=[_entity("a man in blue")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.phrase for entity in annotation.entities] == ["a man in blue"]
    assert annotation.instruction_template == (
        "{{entity_1}} walks. Later a man in blue stops."
    )
    assert annotation.instruction_template.count("{{entity_1}}") == 1


def test_three_entity_placeholder_occurrences_keep_only_first() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=(
                "{{entity_1}} walks. {{entity_1}} stops. {{entity_1}} looks left."
            ),
            entities=[_entity("a man in blue")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.instruction_template == (
        "{{entity_1}} walks. a man in blue stops. a man in blue looks left."
    )


def test_two_duplicate_entity_placeholders_are_canonicalized_independently() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=(
                "{{entity_1}} greets {{entity_2}}. Later {{entity_1}} follows "
                "{{entity_2}}."
            ),
            entities=[
                _entity("a man in blue"),
                _entity("a white boat", reference_type="object"),
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.phrase for entity in annotation.entities] == [
        "a man in blue",
        "a white boat",
    ]
    assert annotation.instruction_template == (
        "{{entity_1}} greets {{entity_2}}. Later a man in blue follows a white boat."
    )


@pytest.mark.parametrize(
    "template",
    [
        "{{entity_1}} stands beside dog{{entity_2}}.",
        "{{entity_1}} stands beside {{entity_2}}nearby.",
    ],
)
def test_invalid_entity_marker_position_drops_only_that_proposal(
    template: str,
) -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template=template,
            entities=[_entity("woman"), _entity("dog")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.phrase for entity in annotation.entities] == ["woman"]
    assert "{{entity_2}}" not in annotation.instruction_template
    assert "dropped_entity_embedded_placeholder:2" in warnings


def test_unexpected_entity_marker_is_removed_without_extra_entity() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="A woman {{entity_1}} stands near a tower {{entity_4}}.",
            entities=[_entity("woman")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.phrase for entity in annotation.entities] == ["woman"]
    assert annotation.instruction_template == (
        "A woman {{entity_1}} stands near a tower."
    )
    assert "removed_unexpected_entity_marker:4" in warnings


def test_invalid_entity_candidate_removes_its_marker_and_renumbers() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template=("{{entity_1}} stands beside {{entity_2}} and {{entity_3}}."),
            entities=[
                _entity("woman"),
                _entity("region", reference_type="scene"),
                _entity("boat", reference_type="object"),
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.entity_id for entity in annotation.entities] == ["e1", "e2"]
    assert [entity.phrase for entity in annotation.entities] == ["woman", "boat"]
    assert annotation.instruction_template == (
        "{{entity_1}} stands beside region and {{entity_2}}."
    )
    assert "dropped_entity_reference_type:1" in warnings


def test_missing_and_duplicate_entities_leave_contiguous_placeholders() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template=(
                "{{entity_1}} sees a dog and {{entity_3}} twice before "
                "{{entity_3}} stops beside {{entity_4}}."
            ),
            entities=[
                _entity("woman"),
                _entity("dog"),
                _entity("sign", reference_type="object"),
                _entity("boat", reference_type="object"),
            ],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert [entity.entity_id for entity in annotation.entities] == [
        "e1",
        "e2",
        "e3",
    ]
    assert [entity.phrase for entity in annotation.entities] == [
        "woman",
        "sign",
        "boat",
    ]
    assert annotation.instruction_template.count("{{entity_") == 3
    assert "{{entity_1}}" in annotation.instruction_template
    assert "{{entity_2}}" in annotation.instruction_template
    assert "{{entity_3}}" in annotation.instruction_template
    assert "sign stops" in annotation.instruction_template
    assert "dropped_entity_missing_marker:2" in warnings


def test_duplicate_placeholder_then_grounding_failure_expands_every_occurrence() -> (
    None
):
    grounding = " ".join(f"feature{index}" for index in range(25))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} walks. Later {{entity_1}} stops.",
            entities=[_entity("a man in blue", grounding_prompt=grounding)],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.entities == []
    assert annotation.instruction_template == (
        "a man in blue walks. Later a man in blue stops."
    )
    assert "{{entity_" not in annotation.instruction_template
    assert "dropped_entity_grounding_prompt_too_long:1" in warnings


def test_zero_entities_removes_unexpected_entity_marker() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="A plaza contains a stray marker {{entity_1}}.",
            entities=[],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.entities == []
    assert annotation.instruction_template == "A plaza contains a stray marker."
    assert "removed_unexpected_entity_marker:1" in warnings


def test_annotation_word_count_excludes_internal_markers() -> None:
    words = " ".join(f"visible{index}" for index in range(179))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template=f"{words} {{{{entity_1}}}}.",
            entities=[_entity("candidate")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    assert "instruction_template_over_preferred_length:180" in warnings


def test_annotation_word_count_uses_substituted_entity_phrase() -> None:
    words = " ".join(f"visible{index}" for index in range(210))
    phrase = "one two three four five six seven eight nine ten eleven twelve"
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=f"{words} {{{{entity_1}}}}.",
            entities=[_entity(phrase)],
            background=None,
        )
    )

    assert annotation is None
    assert {issue.code for issue in issues} == {"instruction_template_too_long"}


def test_background_marker_matches_background_presence() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses {{background}}.",
            entities=[_entity("woman")],
            background={
                "phrase": "a sunlit plaza",
                "grounding_prompt": "broad limestone plaza with arched colonnades",
            },
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.background is not None

    missing, missing_issues, missing_warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses a sunlit plaza.",
            entities=[_entity("woman")],
            background={"phrase": "plaza", "grounding_prompt": "sunlit plaza"},
        )
    )
    assert missing_issues == []
    assert missing is not None
    assert missing.background is None
    assert "dropped_background_missing_marker" in missing_warnings

    unexpected, unexpected_issues, unexpected_warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses a plaza beside {{background}}.",
            entities=[_entity("woman")],
            background=None,
        )
    )
    assert unexpected_issues == []
    assert unexpected is not None
    assert unexpected.background is None
    assert "{{background}}" not in unexpected.instruction_template
    assert "removed_unexpected_background_marker" in unexpected_warnings


def test_duplicate_background_placeholder_keeps_first_and_expands_later() -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            template=(
                "{{entity_1}} crosses {{background}} before "
                "{{background}} appears again."
            ),
            entities=[_entity("woman")],
            background={
                "phrase": "a sunlit plaza",
                "grounding_prompt": "sunlit plaza",
            },
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.background is not None
    assert annotation.instruction_template == (
        "{{entity_1}} crosses {{background}} before a sunlit plaza appears again."
    )
    assert annotation.instruction_template.count("{{background}}") == 1


def test_embedded_background_placeholder_drops_background() -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses abc{{background}}def.",
            entities=[_entity("woman")],
            background={"phrase": "plaza", "grounding_prompt": "sunlit plaza"},
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.background is None
    assert "{{background}}" not in annotation.instruction_template
    assert "dropped_background_embedded_placeholder" in warnings


def test_invalid_background_grounding_preserves_safe_phrase_in_text() -> None:
    grounding = " ".join(f"detail{index}" for index in range(25))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} crosses {{background}}.",
            entities=[_entity("a woman")],
            background={
                "phrase": "a sunlit plaza",
                "grounding_prompt": grounding,
            },
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.background is None
    assert annotation.instruction_template == ("{{entity_1}} crosses a sunlit plaza.")
    assert "dropped_background_grounding_prompt_too_long" in warnings
    assert (
        render_annotation_plain_text(
            annotation.instruction_template,
            annotation.entities,
            annotation.background,
        )
        == "a woman crosses a sunlit plaza."
    )


def test_missing_marker_is_sanitized_without_repair_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _payload(
        template="A table remains visible.",
        entities=[
            _entity(
                "dark wooden altar with carvings",
                reference_type="object",
            )
        ],
    )
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [invalid],
        repair_retries=1,
    )

    assert stats.processed == 1
    assert stats.repaired == 0
    assert len(client.requests) == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"
    assert clip.annotation.entities == []
    assert clip.annotation.instruction_template == "A table remains visible."


@pytest.mark.parametrize(
    "caption",
    [
        "Thin branches sway slightly while the camera remains still.",
        "Bright natural daylight illuminates a quiet courtyard.",
        "Diffuse overcast lighting covers the street.",
        "A woman speaks beneath a cloudy sky.",
        "Two people are talking in a sunny plaza.",
    ],
)
def test_directly_visible_caption_language_is_allowed(caption: str) -> None:
    annotation, issues, _ = sanitize_annotation_payload(_payload(caption=caption))

    assert issues == []
    assert annotation is not None
    assert annotation.status == "ready"


@pytest.mark.parametrize("word_count", [120, 121, 180, 181, 196, 220])
def test_instruction_template_warning_boundaries(
    word_count: int,
) -> None:
    caption = " ".join(f"visible{index}" for index in range(word_count))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(caption=caption)
    )

    assert issues == []
    assert annotation is not None
    if word_count > 180:
        assert f"instruction_template_over_target_length:{word_count}" in warnings
        assert not any(
            warning.startswith("instruction_template_over_preferred_length:")
            for warning in warnings
        )
    elif word_count > 120:
        assert f"instruction_template_over_preferred_length:{word_count}" in warnings
    else:
        assert not any(
            warning.startswith("instruction_template_over_preferred_length:")
            for warning in warnings
        )


def test_instruction_template_over_220_words_enters_repair_validation() -> None:
    caption = " ".join(f"visible{index}" for index in range(221))
    annotation, issues, _ = sanitize_annotation_payload(_payload(caption=caption))

    assert annotation is None
    assert {issue.code for issue in issues} == {"instruction_template_too_long"}


@pytest.mark.parametrize("word_count", [12, 13, 14, 18])
def test_entity_phrase_preferred_and_absolute_boundaries(
    word_count: int,
) -> None:
    phrase = " ".join(f"detail{index}" for index in range(word_count))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            template="{{entity_1}} remains visible.",
            entities=[_entity(phrase, grounding_prompt="stable visible object")],
            background=None,
        )
    )

    assert issues == []
    assert annotation is not None
    warning = f"entity_phrase_over_preferred_length:1:{word_count}"
    if word_count > 12:
        assert warning in warnings
    else:
        assert warning not in warnings


def test_entity_phrase_over_18_words_enters_repair_validation() -> None:
    phrase = " ".join(f"detail{index}" for index in range(19))
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            caption=f"A {phrase} remains visible.",
            entities=[_entity(phrase, grounding_prompt="stable visible object")],
        )
    )

    assert annotation is None
    assert {issue.code for issue in issues} == {"entity_phrase_too_long"}


@pytest.mark.parametrize("word_count", [13, 14, 18])
def test_soft_entity_phrase_length_does_not_request_qwen_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    word_count: int,
) -> None:
    phrase = " ".join(f"detail{index}" for index in range(word_count))
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [
            _payload(
                template="{{entity_1}} remains visible.",
                entities=[_entity(phrase, grounding_prompt="stable visible object")],
                background=None,
            )
        ],
        repair_retries=1,
    )

    assert stats.processed == 1
    assert stats.repaired == 0
    assert len(client.requests) == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


def test_grounding_prompt_over_24_words_drops_only_entity() -> None:
    grounding = " ".join(f"feature{index}" for index in range(25))
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            caption="A stable object remains visible.",
            entities=[_entity("a stable object", grounding_prompt=grounding)],
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.entities == []
    assert "a stable object" in annotation.instruction_template
    assert "dropped_entity_grounding_prompt_too_long:1" in warnings


def test_grounding_prompt_fail_soft_does_not_request_qwen_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounding = " ".join(f"feature{index}" for index in range(25))
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [
            _payload(
                template="{{entity_1}} remains visible.",
                entities=[_entity("a stable object", grounding_prompt=grounding)],
                background=None,
            )
        ],
        repair_retries=1,
    )

    assert stats.processed == 1
    assert stats.repaired == 0
    assert len(client.requests) == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"
    assert clip.annotation.entities == []
    assert clip.annotation.instruction_template == ("a stable object remains visible.")


@pytest.mark.parametrize(
    "grounding_prompt",
    [
        "the woman gesturing beside the table",
        "the man speaking near the podium",
        "the person holding up a sign",
        "the woman moving her hand near the chair",
    ],
)
def test_transient_action_in_grounding_prompt_drops_only_entity(
    grounding_prompt: str,
) -> None:
    annotation, issues, warnings = sanitize_annotation_payload(
        _payload(
            caption="A visible person stands near a window.",
            entities=[
                _entity(
                    "a visible person",
                    grounding_prompt=grounding_prompt,
                )
            ],
        )
    )

    assert issues == []
    assert annotation is not None
    assert annotation.entities == []
    assert "a visible person" in annotation.instruction_template
    assert "dropped_entity_transient_grounding_action:1" in warnings


@pytest.mark.parametrize(
    "grounding_prompt",
    [
        "the seated woman in a yellow coat",
        "the standing metal floor lamp near the window",
    ],
)
def test_stable_short_grounding_prompt_is_allowed(grounding_prompt: str) -> None:
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(
            caption="A stable foreground reference stands near a window.",
            entities=[
                _entity(
                    "a stable foreground reference",
                    grounding_prompt=grounding_prompt,
                )
            ],
        )
    )

    assert issues == []
    assert annotation is not None


@pytest.mark.parametrize(
    "caption",
    [
        "Wind causes the branches to sway beside the road.",
        "The bright light is suggesting a sunny day.",
        "An enemy figure crosses the courtyard.",
        "The statues have expressions of determination.",
        "The branch movement is caused by wind.",
    ],
)
def test_unsupported_caption_inference_is_rejected(caption: str) -> None:
    annotation, issues, _ = sanitize_annotation_payload(_payload(caption=caption))

    assert annotation is None
    assert [issue.code for issue in issues] == ["unsupported_caption_inference"]


def test_caption_semantic_issue_can_be_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [
            _payload(caption="Wind causes the branches to sway beside the road."),
            _payload(caption="Thin branches sway slightly beside the road."),
        ],
        repair_retries=1,
    )

    assert stats.processed == 1
    assert stats.repaired == 1
    assert len(client.requests) == 2
    assert "unsupported_caption_inference" in str(client.requests[1])
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"
    assert clip.annotation.instruction_template.startswith("Thin branches")
    assert clip.annotation.t2v_caption == ""


def test_annotation_text_limit_issue_can_be_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_caption = " ".join(f"visible{index}" for index in range(221))
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [_payload(caption=long_caption), _payload()],
        repair_retries=1,
    )

    assert stats.repaired == 1
    assert len(client.requests) == 2
    assert "instruction_template_too_long" in str(client.requests[1])
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


def test_annotation_prompt_defines_concise_text_limits() -> None:
    lowered = " ".join(SYSTEM_PROMPT.lower().split())
    for phrase in (
        "prefer roughly 60 to 120 english content words",
        "target no more than 180 english content words",
        "absolute validation ceiling is 220 words",
        "at or below 12 words as the target",
        "absolute maximum of 18 words",
        "must not exceed 24 words",
        "do not include transient actions",
        "phrase is a stable, natural english noun phrase",
        "grounding_prompt need not occur in instruction_template",
        "placeholder represents that entity's complete phrase",
        "after every placeholder is replaced by its phrase",
    ):
        assert phrase in lowered


def test_dense_prompt_preserves_recall_with_semantic_type_boundaries() -> None:
    prompt = annotation_system_prompt(
        QwenAnnotationConfig(entity_selection_mode="reference_dense_v1")
    ).lower()
    for contract in (
        "animals are subjects, never objects",
        "body parts are not independent objects",
        "amorphous materials, liquids, sauces, smoke, shadows, lighting",
        "buildings, room architecture, bridges, trees, landscape elements",
        "screen, painting, photograph, poster",
        "worn or attached objects may be selected",
        "plate of meatballs",
        "loaf of bread",
    ):
        assert contract in prompt


@pytest.mark.parametrize(
    "phrase",
    (
        "object",
        "a large black object with cutouts",
        "small dark floating object",
        "an unidentified item beside the chair",
        "a strange thing with holes",
    ),
)
def test_dense_generic_object_pattern_drops_modified_vague_heads(
    phrase: str,
) -> None:
    entities, warnings = sanitize_entity_candidates(
        [
            {
                "reference_type": "object",
                "phrase": phrase,
                "grounding_prompt": "stable visible foreground shape near center",
            }
        ],
        max_entities=8,
        reject_generic_object_phrases=True,
    )

    assert entities == []
    assert warnings == ("dropped_generic_object_phrase:0",)


@pytest.mark.parametrize(
    "phrase",
    (
        "a red shoulder bag",
        "a silver SUV",
        "a steel cutting tool",
        "a plate of meatballs",
        "an ocean-patterned dress",
    ),
)
def test_dense_generic_object_pattern_preserves_concrete_objects(
    phrase: str,
) -> None:
    entities, warnings = sanitize_entity_candidates(
        [
            {
                "reference_type": "object",
                "phrase": phrase,
                "grounding_prompt": "stable visible foreground object near center",
            }
        ],
        max_entities=8,
        reject_generic_object_phrases=True,
    )

    assert len(entities) == 1
    assert "dropped_generic_object_phrase:0" not in warnings


def test_empty_caption_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [_payload(caption="  "), _payload()],
        repair_retries=1,
    )

    assert stats.processed == 1
    assert stats.repaired == 1
    assert len(client.requests) == 2
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


def test_empty_caption_after_repair_marks_annotation_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [_payload(caption=""), _payload(caption=" ")],
        repair_retries=1,
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "failed"
    assert clip.annotation.reason == "empty_instruction_template"


def test_invalid_json_enters_repair_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        ["not json", _payload()],
        repair_retries=1,
    )

    assert stats.repaired == 1
    assert len(client.requests) == 2
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


def test_caption_reference_token_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [_payload(caption="A woman <ref_subject_1> walks.")],
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.reason == "invalid_annotation_reference_token"


def test_changed_annotation_invalidates_all_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    raw = json.dumps(_payload())
    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )
    _seed_ready_downstream(storage, clip_uid)

    changed = _payload(
        caption=(
            "A woman in a yellow coat walks beside a wooden table through a "
            "sunlit plaza, then pauses as the camera pans left."
        )
    )
    annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(changed)],
        ),
    )

    clip = storage.read_clip(clip_uid)
    assert clip.coverage is None
    assert clip.references == ReferencesState()
    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_identical_annotation_rerun_preserves_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    raw = json.dumps(_payload())
    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )
    _seed_ready_downstream(storage, clip_uid)
    before = storage.read_clip(clip_uid)
    clip_bytes = storage.clip_path(clip_uid).read_bytes()

    annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )

    assert storage.read_clip(clip_uid) == before
    assert storage.clip_path(clip_uid).read_bytes() == clip_bytes


def test_failed_annotation_overwrite_invalidates_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=0)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(_payload())],
        ),
    )
    _seed_ready_downstream(storage, clip_uid)

    stats = annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [RuntimeError("endpoint unavailable")],
        ),
    )

    clip = storage.read_clip(clip_uid)
    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "failed"
    assert clip.coverage is None
    assert clip.references == ReferencesState()
    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_qwen_request_failure_uses_unified_failure_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=0)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)

    stats = annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [RuntimeError("endpoint unavailable")],
        ),
    )

    assert stats.failed == 1
    failure = json.loads(storage.failures_path.read_text(encoding="utf-8"))
    assert failure["clip_uid"] == clip_uid
    assert failure["stage"] == "annotate"
    assert failure["reason"] == "qwen_request_failed"
    assert not list(config.resolved_run_root.rglob("qwen_failed.jsonl"))


def test_annotation_uses_clip_source_evidence_without_rescanning_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [
            {
                "file_path": str(video),
                "text": "Stored draft evidence.",
                "title": "Stored metadata evidence",
            }
        ],
    )
    storage, clip_uid = _storage_with_manifest(config)
    config.dataset_json.unlink()
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.processed == 1
    request = str(client.requests[0])
    assert "Stored draft evidence." in request
    assert "Stored metadata evidence" in request
    assert storage.read_clip(clip_uid).annotation is not None


def test_raw_responses_are_saved_only_when_debug_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, debug=True)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    raw = json.dumps(_payload())

    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )

    diagnostic = json.loads(
        (storage.clip_dir(clip_uid) / "debug" / "annotation_raw.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["raw_responses"] == [raw]


class _CompletionsStub:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.raw))]
        )


class _StrictFallbackCompletionsStub(_CompletionsStub):
    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise BadRequestError(
                "strict schema unsupported",
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST",
                        "http://127.0.0.1:8000/v1/chat/completions",
                    ),
                ),
                body={},
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.raw))]
        )


class _SequencedCompletionsStub:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))
            ]
        )


def test_qwen_request_uses_full_video_minimal_schema_and_no_resampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    completions = _CompletionsStub(json.dumps(_payload()))
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.annotate(
        video_path=video,
        caption_raw="draft",
        metadata={"title": "evidence"},
    )

    assert result.annotation.status == "ready"
    request = completions.calls[0]
    assert request["extra_body"] == {
        "mm_processor_kwargs": {
            "fps": 2.0,
            "do_sample_frames": False,
        }
    }
    response_format = request["response_format"]
    assert response_format["json_schema"]["strict"] is True
    assert list(response_format["json_schema"]["schema"]["properties"]) == [
        "entities",
        "background",
        "instruction_template",
    ]
    messages = request["messages"]
    assert messages[-1]["content"][0]["video_url"]["url"] == (video.resolve().as_uri())


def test_qwen_falls_back_to_json_object_when_strict_schema_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    completions = _StrictFallbackCompletionsStub(json.dumps(_payload()))
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.annotate(
        video_path=video,
        caption_raw="draft",
        metadata={},
    )

    assert result.annotation.status == "ready"
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert completions.calls[1]["response_format"] == {"type": "json_object"}


def test_annotation_repair_profiles_each_real_http_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=1)
    video = _video(config)
    completions = _SequencedCompletionsStub(["{}", json.dumps(_payload())])
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")

    with active_profiler(profiler):
        result = client.annotate(
            video_path=video,
            caption_raw="draft",
            metadata={},
        )

    assert result.annotation.status == "ready"
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["component"] for event in events] == [
        "qwen_annotation",
        "qwen_annotation",
    ]
    assert [event["retry_index"] for event in events] == [0, 1]
    assert [event["operation"] for event in events] == ["initial", "repair"]
    assert all(event["input_image_count"] == 0 for event in events)
    assert all(event["metadata"]["video_input"] is True for event in events)


def test_system_prompt_describes_minimal_candidate_contract() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()

    assert "return at most five entities" in normalized
    assert "return at most three entities" not in normalized
    assert "stable, discrete foreground reference candidates" in normalized
    assert "do not output relations" in normalized
    assert "do not include <ref_...> tokens" in normalized
    assert "name ontology" in normalized
    assert "instruction_template" in normalized
    assert "{{entity_1}}" in SYSTEM_PROMPT
    assert "{{background}}" in SYSTEM_PROMPT


def test_system_prompt_plans_references_before_writing_template() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()

    step_1 = normalized.index("step 1: select the reference entity proposals")
    step_2 = normalized.index(
        "step 2: decide whether one stable background reference is useful"
    )
    step_3 = normalized.index(
        "step 3: after the entity and background proposals are fixed"
    )
    assert step_1 < step_2 < step_3
    assert (
        "do not output an entity proposal unless you can place its corresponding "
        "placeholder exactly once in instruction_template"
    ) in normalized
    assert (
        "do not output a non-null background unless you can place "
        "{{background}} exactly once in instruction_template"
    ) in normalized
    assert "every listed entity must have its placeholder" in normalized
    assert "first clear natural mention of that entity" in normalized
    assert "later mentions must use pronouns or ordinary natural-language" in (
        normalized
    )
    assert "never repeat the same entity placeholder" in normalized
    assert "first natural environment mention" in normalized
    assert "duplicate entity or background placeholders are forbidden" in (normalized)


def test_system_prompt_rejects_inferred_causes_and_multiscene_backgrounds() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()

    assert "describe visible motion directly without assigning an unseen cause" in (
        normalized
    )
    assert 'write "branches sway slightly"' in normalized
    assert "major scene transition between different environments" in normalized
    assert "return background=null" in normalized
    assert "stable, natural english noun phrase rather than an action" in normalized


def test_do_sample_frames_true_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        qwen=replace(
            config.qwen,
            annotation=replace(
                config.qwen.annotation,
                video=replace(
                    config.qwen.annotation.video,
                    do_sample_frames=True,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="disable HF video re-sampling"):
        changed.validate()


def test_pipeline_runs_manifest_and_annotation_with_fake_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    config_path = _config_path(config, tmp_path)
    assert "sam3:" not in config_path.read_text(encoding="utf-8")
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("manifest", "annotate"),
        git_commit="pipeline-test",
        annotation_client=client,
    )

    assert result["completed_stages"] == ["manifest", "annotate"]
    assert result["manifest"]["processed"] == 1
    assert result["annotate"]["processed"] == 1
