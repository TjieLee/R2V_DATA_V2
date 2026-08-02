from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.v3.config import (
    CoverageConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.rank import (
    build_coverage_state,
    rank_temporal_coverage,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    CoverageState,
    ReferencesState,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    required_visible_frames: int = 7,
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(v3_config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(
        v3_config_module,
        "ALLOWED_PRETRAINED_ROOT",
        pretrained,
    )
    monkeypatch.setattr(
        v3_config_module,
        "ALLOWED_USER_MODEL_ROOT",
        user_models,
    )
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "rank",
        export_root=writable / "datasets" / "rank-v1",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(model)),
            instruction_writer=QwenServiceConfig(model=str(model)),
            candidate_judge=QwenServiceConfig(model=str(model)),
            background_remove_judge=QwenServiceConfig(model=str(model)),
        ),
        coverage=CoverageConfig(
            required_visible_frames=required_visible_frames
        ),
        sam3=Sam3Config(
            model_path=user_models / "sam3" / "checkpoint.pt"
        ),
        remove=RemoveConfig(
            base_model_path=pretrained
            / "Qwen"
            / "Qwen-Image-Edit-2511",
            adapter_path=user_models
            / "Qwen-Image-Edit-2511-Object-Remover",
        ),
    )
    config.validate()
    return config


def _entity(
    entity_id: str,
    *,
    reference_type: str = "subject",
) -> AnnotationEntity:
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type=reference_type,
        phrase=f"entity {entity_id}",
        grounding_prompt=f"grounding prompt {entity_id}",
    )


def _tracked_entity(
    entity: AnnotationEntity,
    *,
    visible_count: int,
    status: str = "ready",
) -> TrackedEntityMasks:
    if status != "ready":
        visible_count = 0
    present_mask = np.zeros((4, 5), dtype=bool)
    present_mask[1:3, 2:4] = True
    empty_mask = np.zeros_like(present_mask)
    object_id = f"track-{entity.entity_id}"
    frames = [
        TrackedMaskFrame(
            slot=slot,
            present=status == "ready" and slot < visible_count,
            confidence=(
                0.7 + slot * 0.01
                if status == "ready" and slot < visible_count
                else None
            ),
            backend_confidences=(
                [0.7 + slot * 0.01]
                if status == "ready" and slot < visible_count
                else []
            ),
            backend_object_ids=(
                [object_id]
                if status == "ready" and slot < visible_count
                else []
            ),
            area_pixels=(
                int(present_mask.sum())
                if status == "ready" and slot < visible_count
                else 0
            ),
            area_ratio=(
                float(present_mask.mean())
                if status == "ready" and slot < visible_count
                else 0.0
            ),
            bbox_xyxy=(
                (2, 1, 4, 3)
                if status == "ready" and slot < visible_count
                else None
            ),
            rle=encode_binary_mask(
                present_mask
                if status == "ready" and slot < visible_count
                else empty_mask
            ),
        )
        for slot in range(10)
    ]
    return TrackedEntityMasks(
        status=status,
        reference_type=entity.reference_type,
        grounding_prompt=entity.grounding_prompt,
        backend_object_ids=[object_id] if visible_count else [],
        frames=frames,
        reason="tracking failed" if status == "failed" else None,
    )


def _artifact(
    entities: list[AnnotationEntity],
    visible_counts: list[int],
    *,
    statuses: list[str] | None = None,
) -> TrackedMasksArtifact:
    entity_statuses = statuses or ["ready"] * len(entities)
    return TrackedMasksArtifact(
        clip_uid="clip-1",
        height=4,
        width=5,
        entities={
            entity.entity_id: _tracked_entity(
                entity,
                visible_count=visible_count,
                status=status,
            )
            for entity, visible_count, status in zip(
                entities,
                visible_counts,
                entity_statuses,
            )
        },
    )


def _storage_with_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entities: list[AnnotationEntity],
    visible_counts: list[int],
    statuses: list[str] | None = None,
    required_visible_frames: int = 7,
) -> RunStorage:
    storage = RunStorage(
        _config(
            tmp_path,
            monkeypatch,
            required_visible_frames=required_visible_frames,
        )
    )
    storage.initialize(git_commit="rank-test")
    video_path = storage.config.dataset_json.parent / "videos" / "clip-1.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake-video")
    storage.create_clip(
        clip_uid="clip-1",
        source=ClipSource(
            video_path=str(video_path),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            t2v_caption="Visible entities move through the scene.",
            entities=entities,
        ),
    )
    storage.write_masks(
        "clip-1",
        _artifact(
            entities,
            visible_counts,
            statuses=statuses,
        ),
    )
    return storage


@pytest.mark.parametrize(
    ("visible_count", "qualifies"),
    [(6, False), (7, True), (8, True)],
)
def test_temporal_coverage_uses_configured_integer_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_count: int,
    qualifies: bool,
) -> None:
    entity = _entity("e1")
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=[entity],
        visible_counts=[visible_count],
    )

    stats = rank_temporal_coverage(storage.config, storage)
    coverage = storage.read_clip("clip-1").coverage

    assert stats.processed == 1
    assert coverage is not None
    assert coverage.passed is qualifies
    assert coverage.qualifying_entity_ids == (["e1"] if qualifies else [])
    assert coverage.required_visible_frames == 7
    summary = coverage.entity_visibility_summary["e1"]
    assert summary.visible_frame_count == visible_count
    assert summary.visible_frame_slots == list(range(visible_count))
    assert summary.coverage_ratio == pytest.approx(visible_count / 10)
    assert len(summary.per_frame_area_ratio) == 10
    assert len(summary.per_frame_confidence) == 10


def test_clip_coverage_uses_any_entity_and_preserves_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2", reference_type="object")]
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=entities,
        visible_counts=[7, 3],
    )

    rank_temporal_coverage(storage.config, storage)
    clip = storage.read_clip("clip-1")

    assert clip.coverage is not None
    assert clip.coverage.passed is True
    assert clip.coverage.qualifying_entity_ids == ["e1"]
    assert [entity.entity_id for entity in clip.annotation.entities] == [
        "e1",
        "e2",
    ]


def test_all_entities_below_threshold_reject_clip_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2", reference_type="object")]
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=entities,
        visible_counts=[6, 5],
    )

    stats = rank_temporal_coverage(storage.config, storage)
    coverage = storage.read_clip("clip-1").coverage

    assert stats.rejected == 1
    assert stats.failed == 0
    assert coverage is not None
    assert coverage.passed is False
    assert coverage.qualifying_entity_ids == []


def test_zero_entity_clip_gets_normal_failed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=[],
        visible_counts=[],
    )

    stats = rank_temporal_coverage(storage.config, storage)
    coverage = storage.read_clip("clip-1").coverage

    assert stats.processed == 1
    assert stats.rejected == 1
    assert coverage == CoverageState(
        passed=False,
        required_visible_frames=7,
    )


def test_not_found_and_failed_entities_do_not_qualify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2", reference_type="object")]
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=entities,
        visible_counts=[10, 10],
        statuses=["not_found", "failed"],
    )

    rank_temporal_coverage(storage.config, storage)
    coverage = storage.read_clip("clip-1").coverage

    assert coverage is not None
    assert coverage.passed is False
    assert {
        entity_id: summary.visible_frame_count
        for entity_id, summary in coverage.entity_visibility_summary.items()
    } == {"e1": 0, "e2": 0}


def test_same_masks_support_seven_and_eight_frame_gates() -> None:
    entity = _entity("e1")
    artifact = _artifact([entity], [7])

    seven = build_coverage_state(
        artifact=artifact,
        entities=[entity],
        required_visible_frames=7,
    )
    eight = build_coverage_state(
        artifact=artifact,
        entities=[entity],
        required_visible_frames=8,
    )

    assert seven.passed is True
    assert eight.passed is False
    assert seven.entity_visibility_summary["e1"].visible_frame_count == 7
    assert eight.entity_visibility_summary["e1"].visible_frame_count == 7


@pytest.mark.parametrize("required", [0, 11, True])
def test_coverage_config_rejects_out_of_range_or_boolean_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required: int,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match="coverage.required_visible_frames",
    ):
        replace(
            config,
            coverage=CoverageConfig(
                required_visible_frames=required
            ),
        ).validate()


def test_coverage_state_no_longer_has_literal_eight_constraint() -> None:
    assert CoverageState(
        passed=False,
        required_visible_frames=7,
    ).required_visible_frames == 7
    assert CoverageState(
        passed=False,
        required_visible_frames=8,
    ).required_visible_frames == 8
    with pytest.raises(ValidationError):
        CoverageState(passed=False, required_visible_frames=11)


def test_coverage_threshold_loads_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config_path = tmp_path / "coverage.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 10\n"
        "qwen:\n"
        "  candidate_judge:\n"
        f"    model: {config.qwen.candidate_judge.model}\n"
        "  background_remove_judge:\n"
        f"    model: {config.qwen.background_remove_judge.model}\n"
        "sam3:\n"
        f"  model_path: {config.sam3.model_path}\n"
        "coverage:\n"
        "  required_visible_frames: 8\n",
        encoding="utf-8",
    )

    assert load_config(
        config_path
    ).coverage.required_visible_frames == 8


def test_rank_does_not_generate_references_or_canonical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
        visible_counts=[7],
    )

    rank_temporal_coverage(storage.config, storage)
    clip = storage.read_clip("clip-1")

    assert clip.references == ReferencesState()
    assert not (storage.clip_dir("clip-1") / "selected").exists()


def test_missing_masks_is_rank_failure_and_clears_stale_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_masks(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
        visible_counts=[7],
    )
    rank_temporal_coverage(storage.config, storage)
    storage.masks_path("clip-1").unlink()

    stats = rank_temporal_coverage(
        storage.config,
        storage,
        overwrite=True,
    )

    assert stats.failed == 1
    assert storage.read_clip("clip-1").coverage is None
    failure = json.loads(
        storage.failures_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert failure["stage"] == "rank"
