from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.background as background_module
import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.config import (
    BackgroundConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3

WIDTH = 5
HEIGHT = 4


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    max_ratio: float = 0.50,
    run_name: str = "run",
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / f"{run_name}-dataset",
        source=SourceConfig(limit=10),
        background=BackgroundConfig(
            enabled=enabled,
            max_pending_remove_area_ratio=max_ratio,
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "object-remover",
        ),
    )
    config.validate()
    return config


def _entity(index: int) -> AnnotationEntity:
    entity_id = f"e{index}"
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type="subject" if index == 1 else "object",
        phrase=f"entity {index}",
        grounding_prompt=f"entity {index}",
    )


def _empty_mask() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=bool)


def _mask(*coordinates: tuple[int, int]) -> np.ndarray:
    value = _empty_mask()
    for y, x in coordinates:
        value[y, x] = True
    return value


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _tracked_masks(
    clip_uid: str,
    masks_by_entity: list[list[np.ndarray]],
    *,
    statuses: list[str] | None = None,
    invalid_slots: list[set[int]] | None = None,
) -> TrackedMasksArtifact:
    resolved_statuses = statuses or ["ready"] * len(masks_by_entity)
    resolved_invalid = invalid_slots or [set() for _ in masks_by_entity]
    entities: dict[str, TrackedEntityMasks] = {}
    for index, slot_masks in enumerate(masks_by_entity, start=1):
        status = resolved_statuses[index - 1]
        invalid = resolved_invalid[index - 1]
        frames: list[TrackedMaskFrame] = []
        for slot, requested_mask in enumerate(slot_masks):
            mask = requested_mask if status == "ready" else _empty_mask()
            present = bool(mask.any())
            frames.append(
                TrackedMaskFrame(
                    slot=slot,
                    present=present,
                    track_valid=slot not in invalid,
                    confidence=0.9 if present else None,
                    backend_confidences=[0.9] if present else [],
                    backend_object_ids=[f"obj-{index}"] if present else [],
                    area_pixels=int(mask.sum()),
                    area_ratio=float(mask.mean()),
                    bbox_xyxy=_bbox(mask) if present else None,
                    rle=encode_binary_mask(mask),
                )
            )
        entities[f"e{index}"] = TrackedEntityMasks(
            status=status,
            reference_type="subject" if index == 1 else "object",
            grounding_prompt=f"entity {index}",
            backend_object_ids=(
                [f"obj-{index}"]
                if any(mask.any() for mask in slot_masks) and status == "ready"
                else []
            ),
            frames=frames,
            reason="tracking failed" if status == "failed" else None,
        )
    return TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=WIDTH,
        height=HEIGHT,
        entities=entities,
    )


def _write_frames(storage: RunStorage, clip_uid: str) -> None:
    frames: list[SampledFrame] = []
    for slot in range(10):
        path = storage.frame_path(clip_uid, slot)
        Image.new("RGB", (WIDTH, HEIGHT), (slot, 20, 30)).save(
            path,
            format="JPEG",
        )
        frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    write_json_atomic(
        storage.frames_manifest_path(clip_uid),
        SampledFramesArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            frames=frames,
        ).model_dump(mode="json"),
    )


def _coverage(entity_count: int, *, passed: bool = True) -> CoverageState:
    if not passed:
        return CoverageState(
            passed=False,
            entity_visibility_summary={
                f"e{index}": EntityVisibilitySummary(
                    status="not_found",
                    visible_frame_slots=[],
                    visible_frame_count=0,
                    coverage_ratio=0.0,
                    qualifies=False,
                    per_frame_area_ratio=[0.0] * 10,
                    per_frame_confidence=[None] * 10,
                )
                for index in range(1, entity_count + 1)
            },
        )
    summaries = {
        f"e{index}": EntityVisibilitySummary(
            status="ready",
            visible_frame_slots=list(range(7)),
            visible_frame_count=7,
            coverage_ratio=0.7,
            qualifies=True,
            per_frame_area_ratio=[0.1] * 7 + [0.0] * 3,
            per_frame_confidence=[0.9] * 7 + [None] * 3,
        )
        for index in range(1, entity_count + 1)
    }
    return CoverageState(
        passed=True,
        qualifying_entity_ids=list(summaries),
        entity_visibility_summary=summaries,
    )


def _storage_with_clip(
    config: V3Config,
    masks: TrackedMasksArtifact,
    *,
    clip_uid: str = "clip-1",
    background: bool = True,
    coverage: bool | None = True,
    artifacts: bool = True,
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    video_path = config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(video_path),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entity_count = len(masks.entities)
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="A subject moves through the scene.",
            entities=[_entity(index) for index in range(1, entity_count + 1)],
            background=(
                BackgroundAnnotation(
                    phrase="open plaza",
                    grounding_prompt="open plaza",
                )
                if background
                else None
            ),
        ),
    )
    if artifacts:
        _write_frames(storage, clip_uid)
        storage.write_masks(clip_uid, masks)
    if coverage is not None:
        storage.write_coverage(
            clip_uid,
            _coverage(entity_count, passed=coverage),
        )
    return storage


def _uniform_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [mask.copy() for _ in range(10)]


def _background(
    storage: RunStorage, clip_uid: str = "clip-1"
) -> BackgroundReferenceState:
    state = storage.read_clip(clip_uid).references.background
    assert state is not None
    return state


def _read_mask(storage: RunStorage, state: BackgroundReferenceState) -> np.ndarray:
    assert state.source_mask_path is not None
    with Image.open(storage.root / state.source_mask_path) as image:
        return np.asarray(image.convert("L"))


def test_disabled_background_writes_none_without_frame_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, enabled=False)
    masks = _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))])
    storage = _storage_with_clip(config, masks, artifacts=False)

    stats = build_background_candidates(config, storage)

    assert stats.processed == stats.none == 1
    assert _background(storage).reason == "background_stage_disabled"


def test_missing_annotation_background_writes_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    masks = _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))])
    storage = _storage_with_clip(
        config,
        masks,
        background=False,
        artifacts=False,
    )

    stats = build_background_candidates(config, storage)

    assert stats.processed == stats.none == 1
    assert _background(storage).reason == "annotation_has_no_background"


@pytest.mark.parametrize("coverage", [None, False])
def test_missing_or_rejected_coverage_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coverage: bool | None,
) -> None:
    config = _config(tmp_path, monkeypatch)
    masks = _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))])
    storage = _storage_with_clip(
        config,
        masks,
        coverage=coverage,
        artifacts=False,
    )

    stats = build_background_candidates(config, storage)

    assert stats.skipped_not_ready == 1
    assert storage.read_clip("clip-1").references.background is None
    assert not storage.failures_path.exists()


def test_empty_union_selects_clean_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    slot_masks = [_mask((0, 0))] + [_empty_mask() for _ in range(9)]
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [slot_masks]),
    )

    stats = build_background_candidates(config, storage)
    state = _background(storage)

    assert stats.clean_raw == 1
    assert state.status == "clean_raw"
    assert state.source_frame_slot == 5
    assert state.source_foreground_area_pixels == 0
    assert state.source_foreground_area_ratio == 0.0


def test_clean_raw_reuses_sampled_jpeg_without_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    slot_masks = [_mask((0, 0))] + [_empty_mask() for _ in range(9)]
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [slot_masks]),
    )

    build_background_candidates(config, storage)
    state = _background(storage)

    assert state.source_image_path == "frames/05.jpg"
    assert state.output_image_path == state.source_image_path
    assert not storage.background_dir("clip-1").exists()


def test_single_entity_nonempty_mask_is_pending_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((1, 2)))]),
    )

    stats = build_background_candidates(config, storage)
    state = _background(storage)

    assert stats.pending_remove == 1
    assert state.status == "pending_remove"
    assert state.source_foreground_area_pixels == 1
    assert state.output_image_path is None
    assert state.generation_mask_path is None


def test_multi_entity_union_uses_logical_or(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    first = _mask((0, 0), (0, 1), (1, 0), (1, 1))
    second = _mask((1, 0), (1, 1), (2, 0), (2, 1))
    storage = _storage_with_clip(
        config,
        _tracked_masks(
            "clip-1",
            [_uniform_masks(first), _uniform_masks(second)],
        ),
    )

    build_background_candidates(config, storage)

    assert _background(storage).source_foreground_area_pixels == 6


def test_pending_png_exactly_matches_union_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    expected = _mask((0, 0), (1, 2), (3, 4))
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(expected)]),
    )

    build_background_candidates(config, storage)
    pixels = _read_mask(storage, _background(storage))

    assert np.array_equal(pixels, expected.astype(np.uint8) * 255)


def test_pending_png_is_single_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )

    build_background_candidates(config, storage)
    state = _background(storage)
    assert state.source_mask_path is not None
    with Image.open(storage.root / state.source_mask_path) as image:
        assert image.mode == "L"
        assert image.size == (WIDTH, HEIGHT)


def test_pending_png_has_no_mask_transforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    expected = _mask((0, 0), (3, 4))
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(expected)]),
    )

    build_background_candidates(config, storage)
    pixels = _read_mask(storage, _background(storage))

    assert set(np.unique(pixels)) == {0, 255}
    assert int(np.count_nonzero(pixels)) == 2
    assert pixels[0, 0] == pixels[3, 4] == 255


def test_source_selection_uses_minimum_union_area(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    masks = _uniform_masks(_mask((0, 0), (0, 1)))
    masks[3] = _mask((0, 0))
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [masks]),
    )

    build_background_candidates(config, storage)

    assert _background(storage).source_frame_slot == 3


def test_equal_area_uses_fixed_center_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )

    build_background_candidates(config, storage)

    assert _background(storage).source_frame_slot == 5


def test_invalid_slot_is_not_a_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    masks = _uniform_masks(_mask((0, 0), (0, 1)))
    masks[5] = _empty_mask()
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [masks], invalid_slots=[{5}]),
    )

    build_background_candidates(config, storage)

    assert _background(storage).source_frame_slot == 4


def test_all_invalid_slots_reject_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    first = [_empty_mask() for _ in range(10)]
    second = [_empty_mask() for _ in range(10)]
    first[5] = _mask((0, 0))
    second[0] = _mask((0, 1))
    storage = _storage_with_clip(
        config,
        _tracked_masks(
            "clip-1",
            [first, second],
            invalid_slots=[set(range(5)), set(range(5, 10))],
        ),
    )

    build_background_candidates(config, storage)
    state = _background(storage)

    assert state.status == "rejected"
    assert state.reason == "no_valid_background_source_frame"
    assert state.source_image_path is None


@pytest.mark.parametrize("status", ["not_found", "failed"])
def test_incomplete_tracking_rejects_only_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    requested = _uniform_masks(_mask((0, 0)))
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [requested], statuses=[status]),
    )
    reference = EntityReferenceState(
        entity_id="e1",
        status="rejected",
        reference_scope="reject",
        visible_region="central",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason="not suitable",
    )
    storage.write_references("clip-1", ReferencesState(entities=[reference]))

    build_background_candidates(config, storage)
    clip = storage.read_clip("clip-1")

    assert clip.references.entities == [reference]
    assert clip.references.background is not None
    assert clip.references.background.reason == "incomplete_foreground_tracking"
    assert clip.references.background.source_mask_path is None


def test_ratio_above_threshold_is_rejected_with_audit_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    large = _empty_mask()
    large.reshape(-1)[:11] = True
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(large)]),
    )

    build_background_candidates(config, storage)
    state = _background(storage)

    assert state.status == "rejected"
    assert state.reason == "foreground_mask_too_large"
    assert state.source_foreground_area_pixels == 11
    assert state.source_mask_path is not None


def test_ratio_equal_to_threshold_is_pending_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    boundary = _empty_mask()
    boundary.reshape(-1)[:10] = True
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(boundary)]),
    )

    build_background_candidates(config, storage)

    assert _background(storage).status == "pending_remove"


def test_raw_foreground_ratio_must_remain_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="raw_foreground_area_ratio"):
        replace(
            config,
            background=BackgroundConfig(raw_foreground_area_ratio=0.01),
        ).validate()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf"), 1])
def test_max_pending_remove_ratio_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="max_pending_remove_area_ratio"):
        replace(
            config,
            background=BackgroundConfig(
                max_pending_remove_area_ratio=value,  # type: ignore[arg-type]
            ),
        ).validate()


def test_background_threshold_changes_config_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        background=BackgroundConfig(max_pending_remove_area_ratio=0.25),
    )

    assert config.fingerprint() != changed.fingerprint()


def test_config_loader_reads_background_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, run_name="loaded")
    config_path = tmp_path / "loaded.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 10\n"
        "background:\n"
        "  enabled: true\n"
        "  raw_foreground_area_ratio: 0.0\n"
        "  max_pending_remove_area_ratio: 0.25\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.background.max_pending_remove_area_ratio == 0.25


@pytest.mark.parametrize("corruption", ["rle_size", "area"])
def test_corrupt_mask_input_is_isolated_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    path = storage.masks_path("clip-1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = payload["entities"]["e1"]["frames"][0]
    if corruption == "rle_size":
        frame["rle"]["size"] = [3, 5]
    else:
        frame["area_pixels"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    stats = build_background_candidates(config, storage)

    assert stats.failed == 1
    assert storage.read_clip("clip-1").references.background is None
    assert '"stage":"background"' in storage.failures_path.read_text(encoding="utf-8")


def test_existing_valid_state_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    build_background_candidates(config, storage)

    stats = build_background_candidates(config, storage)

    assert stats.skipped_existing == 1
    assert stats.processed == 0


def test_existing_corrupt_mask_fails_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    build_background_candidates(config, storage)
    before = _background(storage)
    assert before.source_mask_path is not None
    Image.new("L", (WIDTH, HEIGHT), 0).save(storage.root / before.source_mask_path)

    stats = build_background_candidates(config, storage)

    assert stats.failed == 1
    assert _background(storage) == before


def test_clip_write_failure_preserves_existing_hashed_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    build_background_candidates(config, storage)
    before = _background(storage)
    assert before.source_mask_path is not None
    before_path = storage.root / before.source_mask_path
    before_bytes = before_path.read_bytes()
    replacement_masks = _tracked_masks(
        "clip-1",
        [_uniform_masks(_mask((0, 0), (0, 1)))],
    )
    write_json_atomic(
        storage.masks_path("clip-1"),
        replacement_masks.model_dump(mode="json"),
    )

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated clip.json write failure")

    monkeypatch.setattr(storage, "write_references", fail_write)

    stats = build_background_candidates(config, storage, overwrite=True)

    assert stats.failed == 1
    assert _background(storage) == before
    assert before_path.read_bytes() == before_bytes
    assert list(storage.background_dir("clip-1").glob("*.png")) == [before_path]


def test_overwrite_recomputes_and_cleans_stale_hash_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    build_background_candidates(config, storage)
    stale = storage.background_source_mask_path("clip-1", "f" * 64)
    Image.new("L", (WIDTH, HEIGHT), 255).save(stale)

    stats = build_background_candidates(config, storage, overwrite=True)

    assert stats.processed == 1
    assert not stale.exists()
    assert len(list(storage.background_dir("clip-1").glob("*.png"))) == 1


def test_background_update_preserves_entity_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [_uniform_masks(_mask((0, 0)))]),
    )
    reference = EntityReferenceState(
        entity_id="e1",
        status="rejected",
        reference_scope="reject",
        visible_region="central",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason="not suitable",
    )
    storage.write_references("clip-1", ReferencesState(entities=[reference]))

    build_background_candidates(config, storage)

    assert storage.read_clip("clip-1").references.entities == [reference]


def test_background_update_invalidates_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    slot_masks = [_mask((0, 0))] + [_empty_mask() for _ in range(9)]
    storage = _storage_with_clip(
        config,
        _tracked_masks("clip-1", [slot_masks]),
    )
    reference = EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="complete",
        image_path="clips/clip-1/selected/e1.png",
        source_frame_index=0,
    )
    storage.write_references("clip-1", ReferencesState(entities=[reference]))
    storage.write_pairing(
        "clip-1",
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
        ),
    )
    body = "Use {{image_1}}."
    legend = [InstructionLegendEntry(image_id="image_1", description="entity")]
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    storage.write_export("clip-1", ExportState(accepted=True, reason=None))

    build_background_candidates(config, storage)
    clip = storage.read_clip("clip-1")

    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_one_clip_failure_does_not_block_other_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    bad_masks = _tracked_masks("bad", [_uniform_masks(_mask((0, 0)))])
    storage = _storage_with_clip(config, bad_masks, clip_uid="bad")
    good_masks = _tracked_masks("good", [_uniform_masks(_mask((0, 0)))])
    _storage_with_clip(config, good_masks, clip_uid="good")
    bad_path = storage.masks_path("bad")
    bad_path.write_text("{bad json", encoding="utf-8")

    stats = build_background_candidates(config, storage)

    assert stats.failed == 1
    assert stats.processed == 1
    assert _background(storage, "good").status == "pending_remove"


def test_background_module_has_no_sam3_or_qwen_dependency() -> None:
    source = inspect.getsource(background_module).lower()

    assert "sam3" not in source
    assert "qwen" not in source


def test_pipeline_executes_background_stage_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, run_name="pipeline")
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 10\n",
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("background",),
        git_commit="abc123",
    )

    assert result["completed_stages"] == ["background"]
    assert result["background"]["processed"] == 0


def test_pair_stage_loads_configuration(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pipeline_v3(
            config_path=tmp_path / "unused.yaml",
            stages=("pair",),
        )


def test_none_schema_cannot_reference_artifacts() -> None:
    with pytest.raises(ValidationError, match="cannot reference artifacts"):
        BackgroundReferenceState(
            status="none",
            source_image_path="frames/00.jpg",
        )


def test_clean_raw_schema_requires_zero_area() -> None:
    with pytest.raises(ValidationError, match="zero foreground area"):
        BackgroundReferenceState(
            status="clean_raw",
            source_image_path="frames/00.jpg",
            output_image_path="frames/00.jpg",
            source_frame_slot=0,
            source_frame_index=0,
            source_foreground_area_pixels=1,
            source_foreground_area_ratio=0.05,
        )


def test_pending_remove_schema_requires_positive_area() -> None:
    with pytest.raises(ValidationError, match="positive foreground area"):
        BackgroundReferenceState(
            status="pending_remove",
            source_image_path="frames/00.jpg",
            source_frame_slot=0,
            source_frame_index=0,
            source_mask_path="clips/clip-1/background/source_mask_deadbeef.png",
            source_foreground_area_pixels=0,
            source_foreground_area_ratio=0.0,
        )
