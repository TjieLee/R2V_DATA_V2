from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.annotation import (
    COMPOSITION_BALANCED_ENTITY_SELECTION_PROMPT,
    SYSTEM_PROMPT,
    QwenAnnotationClient,
)
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.entity_composition_audit import audit_entity_composition
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.rank import build_coverage_state
from r2v_data_v2.v3.sam3_backend import BackendMaskObservation, EntityTrackResult
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    EntityReferenceState,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    RawAnnotationPayload,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedMasksArtifact,
    render_inline_instruction_text,
)
from r2v_data_v2.v3.segment import segment_clips
from r2v_data_v2.v3.storage import RunStorage


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_name: str = "composition",
    source: SourceConfig | None = None,
    annotation_mode: str = "default",
    rescue_mode: str = "off",
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
    model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_root / f"{run_name}.json",
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / run_name,
        source=source or SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(
                model=str(model),
                entity_selection_mode=annotation_mode,
            ),
            instruction_writer=QwenServiceConfig(model=str(model)),
            candidate_judge=QwenServiceConfig(model=str(model)),
            background_remove_judge=QwenServiceConfig(model=str(model)),
        ),
        sam3=Sam3Config(
            model_path=user_models / "sam3" / "checkpoint.pt",
            object_rescue_mode=rescue_mode,
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "Qwen-Image-Edit-2511-Object-Remover",
        ),
    )
    config.dataset_json.write_text("[]", encoding="utf-8")
    config.validate()
    return config


def _entity(
    entity_id: str,
    reference_type: str,
) -> AnnotationEntity:
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type=reference_type,
        phrase=f"the concise {reference_type} {entity_id}",
        grounding_prompt=f"the distinct {reference_type} {entity_id} near center",
    )


def _mask(x1: int, x2: int) -> np.ndarray:
    value = np.zeros((12, 16), dtype=bool)
    value[2:10, x1:x2] = True
    return value


def _ready(mask: np.ndarray, *, slots: int = 7) -> EntityTrackResult:
    return EntityTrackResult(
        status="ready",
        observations=tuple(
            BackendMaskObservation(
                slot=slot,
                mask=mask,
                confidence=0.9,
                object_id="track-1",
            )
            for slot in range(slots)
        ),
    )


def _not_found(reason: str = "not found") -> EntityTrackResult:
    return EntityTrackResult(status="not_found", reason=reason)


@dataclass
class PromptBackend:
    results: dict[tuple[str, str], list[EntityTrackResult | Exception]]
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
    ) -> EntityTrackResult:
        del frame_paths
        self.calls.append((entity_id, reference_type, grounding_prompt))
        values = self.results[(entity_id, grounding_prompt)]
        value = values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _storage_with_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entities: list[AnnotationEntity],
    rescue_mode: str,
    run_name: str,
) -> RunStorage:
    config = _config(
        tmp_path,
        monkeypatch,
        run_name=run_name,
        rescue_mode=rescue_mode,
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="composition-test")
    video = config.dataset_json.parent / "videos" / "pilot_0.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    storage.create_clip(
        clip_uid="clip-1",
        source=ClipSource(
            video_path=str(video),
            parent_video_id="pilot",
            clip_suffix="0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    placeholders = " and ".join(
        f"{{{{entity_{index}}}}}" for index in range(1, len(entities) + 1)
    )
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            entities=entities,
            instruction_template=f"{placeholders} remain visible.",
        ),
    )
    frames_dir = storage.frames_dir("clip-1")
    frames_dir.mkdir(parents=True)
    frame_records: list[SampledFrame] = []
    for slot in range(10):
        path = frames_dir / f"{slot:02d}.jpg"
        Image.new("RGB", (16, 12), (slot, slot, slot)).save(path)
        frame_records.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot,
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    write_json_atomic(
        storage.frames_manifest_path("clip-1"),
        SampledFramesArtifact(
            clip_uid="clip-1",
            width=16,
            height=12,
            frames=frame_records,
        ).model_dump(mode="json"),
    )
    return storage


def test_default_annotation_mode_preserves_exact_system_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    client = QwenAnnotationClient(config.qwen.annotation, client=object())

    messages = client._messages(
        video_path=config.dataset_json.parent / "pilot_0.mp4",
        request_text="annotate",
    )

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert COMPOSITION_BALANCED_ENTITY_SELECTION_PROMPT not in SYSTEM_PROMPT


def test_composition_annotation_mode_is_additive_and_not_a_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        annotation_mode="composition_balanced_v1",
    )
    client = QwenAnnotationClient(config.qwen.annotation, client=object())
    system = str(
        client._messages(
            video_path=config.dataset_json.parent / "pilot_0.mp4",
            request_text="annotate",
        )[0]["content"]
    )

    assert system == f"{SYSTEM_PROMPT}\n\n{COMPOSITION_BALANCED_ENTITY_SELECTION_PROMPT}"
    assert "preference, not a quota" in system
    assert "never invent" in system
    subject_only = RawAnnotationPayload.model_validate(
        {
            "entities": [
                {
                    "reference_type": "subject",
                    "phrase": "a visible woman",
                    "grounding_prompt": "a visible woman near the center",
                }
            ],
            "background": None,
            "instruction_template": "{{entity_1}} walks forward.",
        }
    )
    assert [entity.reference_type for entity in subject_only.entities] == ["subject"]
    assert "entity_selection_mode" not in RawAnnotationPayload.model_json_schema()[
        "properties"
    ]


def test_invalid_annotation_and_rescue_modes_fail_config_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="entity_selection_mode"):
        replace(
            config,
            qwen=replace(
                config.qwen,
                annotation=replace(
                    config.qwen.annotation,
                    entity_selection_mode="quota",
                ),
            ),
        ).validate()
    with pytest.raises(ValueError, match="object_rescue_mode"):
        replace(
            config,
            sam3=replace(config.sam3, object_rescue_mode="all_entities"),
        ).validate()
    assert replace(
        config,
        source=replace(
            config.source,
            selection_mode="parent_stratified_random_v1",
            random_seed=20260812,
        ),
    ).fingerprint() != config.fingerprint()


def _write_source_dataset(config: V3Config, parent_count: int = 8) -> dict[str, bytes]:
    records: list[dict[str, str]] = []
    for parent_index in range(parent_count):
        for clip_index in range(2):
            path = (
                config.dataset_json.parent
                / "videos"
                / f"parent-{parent_index}_{clip_index}.mp4"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{parent_index}:{clip_index}".encode())
            records.append({"file_path": str(path)})
    config.dataset_json.write_text(json.dumps(records), encoding="utf-8")
    return {
        str(path.relative_to(config.dataset_json.parent)): path.read_bytes()
        for path in sorted(config.dataset_json.parent.rglob("*"))
        if path.is_file()
    }


def _run_random_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int,
    run_name: str,
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    config = _config(
        tmp_path,
        monkeypatch,
        run_name=run_name,
        source=SourceConfig(
            limit=4,
            selection_mode="parent_stratified_random_v1",
            random_seed=seed,
            max_clips_per_parent=1,
        ),
    )
    before = _write_source_dataset(config)
    storage = RunStorage(config)
    storage.initialize(git_commit="composition-test")
    build_manifest(config, storage)
    provenance = json.loads(
        (storage.root / "source_selection.json").read_text(encoding="utf-8")
    )
    assert provenance["selection_mode"] == "parent_stratified_random_v1"
    assert provenance["random_seed"] == seed
    assert provenance["max_clips_per_parent"] == 1
    assert provenance["requested_limit"] == 4
    assert provenance["selected_count"] == 4
    after = {
        str(path.relative_to(config.dataset_json.parent)): path.read_bytes()
        for path in sorted(config.dataset_json.parent.rglob("*"))
        if path.is_file()
    }
    assert after == before
    return provenance["selected"], before


def test_parent_stratified_random_selection_is_deterministic_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = _run_random_manifest(
        tmp_path,
        monkeypatch,
        seed=20260812,
        run_name="random-a",
    )
    second, _ = _run_random_manifest(
        tmp_path,
        monkeypatch,
        seed=20260812,
        run_name="random-b",
    )
    changed, _ = _run_random_manifest(
        tmp_path,
        monkeypatch,
        seed=17,
        run_name="random-c",
    )

    assert first == second
    assert first != changed
    assert len({item["parent_video_id"] for item in first}) == len(first)
    assert len({item["clip_uid"] for item in first}) == len(first)


def test_random_selection_fails_when_parent_cap_cannot_satisfy_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        run_name="random-unsatisfied",
        source=SourceConfig(
            limit=3,
            selection_mode="parent_stratified_random_v1",
            random_seed=1,
            max_clips_per_parent=1,
        ),
    )
    _write_source_dataset(config, parent_count=2)
    storage = RunStorage(config)
    storage.initialize(git_commit="composition-test")

    with pytest.raises(ValueError, match="cannot be satisfied"):
        build_manifest(config, storage)


def test_sequential_selection_preserves_start_and_limit_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        run_name="sequential",
        source=SourceConfig(start_index=2, limit=3),
    )
    _write_source_dataset(config, parent_count=4)
    storage = RunStorage(config)
    storage.initialize(git_commit="composition-test")

    build_manifest(config, storage)

    assert sorted(clip.source.source_index for clip in storage.iter_clips()) == [2, 3, 4]
    assert not (storage.root / "source_selection.json").exists()


def test_object_not_found_retries_phrase_once_and_publishes_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1", "object")
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
        rescue_mode="phrase_retry_v1",
        run_name="not-found-ready",
    )
    backend = PromptBackend(
        {
            (entity.entity_id, entity.grounding_prompt): [_not_found()],
            (entity.entity_id, entity.phrase): [_ready(_mask(2, 7))],
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)

    assert [call[2] for call in backend.calls] == [
        entity.grounding_prompt,
        entity.phrase,
    ]
    assert storage.read_masks("clip-1").entities["e1"].status == "ready"
    assert stats.object_rescue_attempted == 1
    assert stats.object_not_found_retry_attempted == 1
    assert stats.object_not_found_retry_ready == 1


@pytest.mark.parametrize("reference_type", ["subject", "group"])
def test_non_object_not_found_never_uses_object_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_type: str,
) -> None:
    entity = _entity("e1", reference_type)
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
        rescue_mode="phrase_retry_v1",
        run_name=f"no-rescue-{reference_type}",
    )
    backend = PromptBackend(
        {(entity.entity_id, entity.grounding_prompt): [_not_found()]}
    )

    stats = segment_clips(storage.config, storage, backend=backend)

    assert len(backend.calls) == 1
    assert stats.object_rescue_attempted == 0


def test_failed_object_retry_preserves_original_not_found_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1", "object")
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
        rescue_mode="phrase_retry_v1",
        run_name="not-found-failed",
    )
    backend = PromptBackend(
        {
            (entity.entity_id, entity.grounding_prompt): [_not_found("initial")],
            (entity.entity_id, entity.phrase): [RuntimeError("retry failed")],
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    track = storage.read_masks("clip-1").entities["e1"]

    assert len(backend.calls) == 2
    assert track.status == "not_found"
    assert track.reason == "initial"
    assert stats.object_not_found_retry_ready == 0


def test_collision_retry_keeps_subject_and_localized_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _entity("e1", "subject")
    object_entity = _entity("e2", "object")
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[subject, object_entity],
        rescue_mode="phrase_retry_v1",
        run_name="collision-ready",
    )
    backend = PromptBackend(
        {
            (subject.entity_id, subject.grounding_prompt): [_ready(_mask(1, 8))],
            (object_entity.entity_id, object_entity.grounding_prompt): [
                _ready(_mask(1, 8))
            ],
            (object_entity.entity_id, object_entity.phrase): [_ready(_mask(10, 15))],
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")

    assert artifact.entities["e1"].status == "ready"
    assert artifact.entities["e2"].status == "ready"
    assert backend.calls[-1] == ("e2", "object", object_entity.phrase)
    assert stats.cross_type_collision_detected == 1
    assert stats.cross_type_collision_retry_attempted == 1
    assert stats.cross_type_collision_retry_ready == 1


def test_unresolved_collision_rejects_only_object_and_records_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _entity("e1", "subject")
    object_entity = _entity("e2", "object")
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[subject, object_entity],
        rescue_mode="phrase_retry_v1",
        run_name="collision-unresolved",
    )
    same = _mask(1, 8)
    backend = PromptBackend(
        {
            (subject.entity_id, subject.grounding_prompt): [_ready(same)],
            (object_entity.entity_id, object_entity.grounding_prompt): [_ready(same)],
            (object_entity.entity_id, object_entity.phrase): [_ready(same)],
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")
    diagnostic = json.loads(
        (
            storage.root
            / "diagnostics"
            / "object_rescue"
            / "clip-1.json"
        ).read_text(encoding="utf-8")
    )

    assert artifact.entities["e1"].status == "ready"
    assert artifact.entities["e2"].status == "failed"
    assert artifact.entities["e2"].reason == (
        "cross_type_subject_object_collision_unresolved:e1"
    )
    assert stats.cross_type_collision_unresolved == 1
    assert diagnostic["attempts"][0]["subject_collision_entity_id"] == "e1"
    assert diagnostic["attempts"][0]["duplicate_metrics_before_retry"]
    assert diagnostic["attempts"][0]["duplicate_metrics_after_retry"]


def test_rescue_disabled_preserves_existing_cross_entity_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _entity("e1", "subject")
    object_entity = _entity("e2", "object")
    storage = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[subject, object_entity],
        rescue_mode="off",
        run_name="collision-disabled",
    )
    same = _mask(1, 8)
    backend = PromptBackend(
        {
            (subject.entity_id, subject.grounding_prompt): [_ready(same)],
            (object_entity.entity_id, object_entity.grounding_prompt): [_ready(same)],
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")

    assert len(backend.calls) == 2
    assert artifact.entities["e1"].status == "ready"
    assert artifact.entities["e2"].reason == "duplicate_cross_entity_track:e1"
    assert stats.object_rescue_attempted == 0
    assert not (storage.root / "diagnostics" / "object_rescue.jsonl").exists()


def _publish_audit_clip(
    storage: RunStorage,
    *,
    clip_uid: str,
    source_index: int,
    types: list[str],
) -> None:
    video = storage.config.dataset_json.parent / "videos" / f"audit-{clip_uid}_0.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(video),
            parent_video_id=f"audit-{clip_uid}",
            clip_suffix="0",
            source_index=source_index,
            caption_raw="",
            metadata={},
        ),
    )
    entities = [
        _entity(f"e{index}", reference_type)
        for index, reference_type in enumerate(types, start=1)
    ]
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            entities=entities,
            instruction_template=" and ".join(
                f"{{{{entity_{index}}}}}" for index in range(1, len(entities) + 1)
            ),
        ),
    )
    tracks = {}
    for entity in entities:
        backend_result = _ready(_mask(2, 7))
        from r2v_data_v2.v3.segment import _entity_masks_from_result

        tracks[entity.entity_id] = _entity_masks_from_result(
            entity,
            backend_result,
            height=12,
            width=16,
        )
    masks = TrackedMasksArtifact(
        clip_uid=clip_uid,
        height=12,
        width=16,
        entities=tracks,
    )
    storage.write_masks(clip_uid, masks)
    storage.write_coverage(
        clip_uid,
        build_coverage_state(
            artifact=masks,
            entities=entities,
            required_visible_frames=7,
        ),
    )
    references = []
    counters = {"subject": 0, "object": 0, "group": 0}
    tokens = {}
    for entity in entities:
        counters[entity.reference_type] += 1
        tokens[entity.entity_id] = (
            f"<ref_{entity.reference_type}_{counters[entity.reference_type]}>"
        )
        references.append(
            EntityReferenceState(
                entity_id=entity.entity_id,
                status="ready",
                reference_scope="full",
                visible_region="whole",
                whole_entity_recognizable=True,
                identity_features_visible=True,
                scope_reason="complete visible entity",
                image_path=f"clips/{clip_uid}/selected/{entity.entity_id}.png",
                source_frame_index=0,
            )
        )
    storage.write_references_and_pairing(
        clip_uid,
        ReferencesState(entities=references),
        PairingState(
            status="ready",
            retained_entity_ids=[entity.entity_id for entity in entities],
            tokens=tokens,
        ),
    )
    body = " and ".join(
        f"{{{{image_{index}}}}}" for index in range(1, len(entities) + 1)
    )
    storage.write_instruction(
        clip_uid,
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=[
                InstructionLegendEntry(
                    image_id=f"image_{index}",
                    description=f"reference {index}",
                )
                for index in range(1, len(entities) + 1)
            ],
            r2v_instruction=render_inline_instruction_text(body),
        ),
    )
    storage.write_export(clip_uid, ExportState(accepted=True, reason=None))


def test_composition_audit_counts_subject_object_samples_and_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, run_name="audit")
    storage = RunStorage(config)
    storage.initialize(git_commit="composition-test")
    _publish_audit_clip(storage, clip_uid="subject", source_index=0, types=["subject"])
    _publish_audit_clip(storage, clip_uid="object", source_index=1, types=["object"])
    _publish_audit_clip(
        storage,
        clip_uid="mixed",
        source_index=2,
        types=["subject", "object"],
    )
    before = {
        str(path.relative_to(storage.root)): path.read_bytes()
        for path in storage.root.rglob("*")
        if path.is_file()
    }

    summary = audit_entity_composition(
        run_root=storage.root,
        output_root=tmp_path / "composition-audit-output",
    )
    after = {
        str(path.relative_to(storage.root)): path.read_bytes()
        for path in storage.root.rglob("*")
        if path.is_file()
    }

    assert summary["sample_count"] == 3
    assert summary["compositions"]["subject_only"] == 1
    assert summary["compositions"]["object_only"] == 1
    assert summary["compositions"]["subject_object"] == 1
    assert summary["samples_with_subject_and_object"] == 1
    assert summary["entity_counts"] == {"subject": 2, "object": 2, "group": 0}
    assert before == after
