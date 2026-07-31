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
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.schemas import (
    ClipRecord,
    CoverageState,
    EntityReferenceState,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    RawAnnotationPayload,
    ReferencesState,
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
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=(
                user_models / "Qwen-Image-Edit-2511-Object-Remover"
            ),
        ),
        debug=DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _config_path(config: V3Config, tmp_path: Path) -> Path:
    path = tmp_path / "v3-annotation.yaml"
    source_limit = (
        ""
        if config.source.limit is None
        else f"  limit: {config.source.limit}\n"
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
            phrase.casefold()
            if grounding_prompt is None
            else grounding_prompt
        ),
    }


def _payload(
    *,
    caption: str = (
        "A woman in a yellow coat walks beside a wooden table through a "
        "sunlit plaza as the camera tracks backward."
    ),
    entities: list[object] | None = None,
    background: object = ...,
) -> dict[str, object]:
    if entities is None:
        entities = [
            _entity("A woman in a yellow coat"),
            _entity("a wooden table", reference_type="object"),
        ]
    if background is ...:
        background = {
            "phrase": "a sunlit plaza",
            "grounding_prompt": "the empty sunlit plaza",
        }
    return {
        "t2v_caption": caption,
        "entities": entities,
        "background": background,
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
    storage.write_coverage(
        clip_uid,
        CoverageState(passed=True, qualifying_entity_ids=["e1"]),
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
        {"file_path": str(_video(config, f"scene_{index}_0.mp4"))}
        for index in range(8)
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

    assert set(schema["properties"]) == {
        "t2v_caption",
        "entities",
        "background",
    }
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
    ]

    entities, warnings = sanitize_entity_candidates(candidates)

    assert [entity.entity_id for entity in entities] == ["e1", "e2", "e3"]
    assert [entity.phrase for entity in entities] == ["Woman", "Table", "Band"]
    assert [entity.reference_type for entity in entities] == [
        "subject",
        "object",
        "group",
    ]
    assert "dropped_entity_reference_type:1" in warnings
    assert "dropped_duplicate_entity_phrase:2" in warnings
    assert "truncated_entity_candidates:3" in warnings


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
    annotation, issues, _ = sanitize_annotation_payload(
        _payload(entities=[None, _entity("", grounding_prompt="person")])
    )

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
    assert clip.annotation.reason == "empty_t2v_caption"


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
    assert clip.annotation.reason == "reference_token_in_annotation"


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
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.raw))
            ]
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
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.raw))
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
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
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
    assert set(response_format["json_schema"]["schema"]["properties"]) == {
        "t2v_caption",
        "entities",
        "background",
    }
    messages = request["messages"]
    assert messages[-1]["content"][0]["video_url"]["url"] == (
        video.resolve().as_uri()
    )


def test_qwen_falls_back_to_json_object_when_strict_schema_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    completions = _StrictFallbackCompletionsStub(json.dumps(_payload()))
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
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


def test_system_prompt_describes_minimal_candidate_contract() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()

    assert "return at most three entities" in normalized
    assert "stable, discrete foreground reference candidates" in normalized
    assert "do not output relations" in normalized
    assert "do not include <ref_...> tokens" in normalized
    assert "name ontology" in normalized


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
