from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.pair as pair_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import (
    PairConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceScopeConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.cross_pair_judge import CrossPairDecisionAttempt
from r2v_data_v2.v3.instruction import build_instruction_bindings
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.pair import (
    build_candidate_context_image,
    build_entity_reference_candidates,
    build_reference_crop,
    pair_clips,
    validate_entity_reference_artifact,
)
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    ClipSource,
    CoverageState,
    EntityVisibilitySummary,
    PairingState,
    RawCrossPairDecision,
    RawEntityReferenceDecision,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3

WIDTH = 12
HEIGHT = 9


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_local: bool = True,
    pair: PairConfig | None = None,
    debug: bool = False,
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
    model = str(pretrained / "Qwen" / "judge")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "pair-test",
        export_root=writable / "datasets" / "pair-test",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=model),
            instruction_writer=QwenServiceConfig(model=model),
            candidate_judge=QwenServiceConfig(model=model),
            background_remove_judge=QwenServiceConfig(model=model),
cross_pair_judge=(
                QwenServiceConfig(model=model)
                if pair is not None and pair.same_parent_fallback_enabled
                else None
            ),
        ),
        reference_scope=ReferenceScopeConfig(allow_local=allow_local),
        pair=pair or PairConfig(),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "object-remover",
        ),
        debug=config_module.DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _tracked_frame(slot: int, mask: np.ndarray) -> TrackedMaskFrame:
    area = int(np.count_nonzero(mask))
    if not area:
        return TrackedMaskFrame(
            slot=slot,
            present=False,
            track_valid=True,
            area_pixels=0,
            area_ratio=0.0,
            rle=encode_binary_mask(mask),
        )
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        track_valid=True,
        confidence=0.9,
        backend_confidences=[0.9],
        backend_object_ids=["object-1"],
        area_pixels=area,
        area_ratio=area / mask.size,
        bbox_xyxy=_bbox(mask),
        rle=encode_binary_mask(mask),
    )


def _visibility(qualifies: bool) -> EntityVisibilitySummary:
    count = 7 if qualifies else 3
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(count)),
        visible_frame_count=count,
        coverage_ratio=count / 10,
        qualifies=qualifies,
        per_frame_area_ratio=[0.1] * count + [0.0] * (10 - count),
        per_frame_confidence=[0.9] * count + [None] * (10 - count),
    )


def _add_ready_clip(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str = "clip-1",
    entity_types: tuple[str, ...] = ("subject", "object"),
    tracking_status: dict[str, str] | None = None,
    parent_video_id: str = "parent",
    clip_suffix: str | None = None,
) -> None:
    tracking_status = tracking_status or {}
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(
                config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"
            ),
            parent_video_id=parent_video_id,
            clip_suffix=clip_suffix or clip_uid,
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=reference_type,
            phrase=f"entity {index}",
            grounding_prompt=f"visible entity {index}",
        )
        for index, reference_type in enumerate(entity_types, start=1)
    ]
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="Several entities move through a scene.",
            entities=entities,
        ),
    )
    sampled: list[SampledFrame] = []
    for slot in range(10):
        pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        pixels[..., 0] = 20 + slot
        pixels[..., 1] = np.arange(WIDTH, dtype=np.uint8)
        pixels[..., 2] = np.arange(HEIGHT, dtype=np.uint8)[:, None]
        path = storage.frame_path(clip_uid, slot)
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        sampled.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot + 1),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=_sha256(path),
            )
        )
    frames = SampledFramesArtifact(
        clip_uid=clip_uid,
        width=WIDTH,
        height=HEIGHT,
        frames=sampled,
    )
    write_json_atomic(
        storage.frames_manifest_path(clip_uid),
        frames.model_dump(mode="json"),
    )
    tracked_entities: dict[str, TrackedEntityMasks] = {}
    for index, entity in enumerate(entities, start=1):
        status = tracking_status.get(entity.entity_id, "ready")
        frame_masks: list[TrackedMaskFrame] = []
        for slot in range(10):
            mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            if status == "ready":
                left = min(index + (slot % 2), WIDTH - 5)
                mask[2:7, left : left + 4] = True
            frame_masks.append(_tracked_frame(slot, mask))
        tracked_entities[entity.entity_id] = TrackedEntityMasks(
            status=status,
            reference_type=entity.reference_type,
            grounding_prompt=entity.grounding_prompt,
            backend_object_ids=["object-1"] if status == "ready" else [],
            frames=frame_masks,
            reason="tracking failed" if status == "failed" else None,
        )
    storage.write_masks(
        clip_uid,
        TrackedMasksArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            entities=tracked_entities,
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={
                entity.entity_id: _visibility(entity.entity_id == "e1")
                for entity in entities
            },
        ),
    )


def _storage(
    config: V3Config,
    *,
    entity_types: tuple[str, ...] = ("subject", "object"),
    tracking_status: dict[str, str] | None = None,
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="pair-test")
    _add_ready_clip(
        config,
        storage,
        entity_types=entity_types,
        tracking_status=tracking_status,
    )
    return storage


def _decision(scope: str = "full") -> RawEntityReferenceDecision:
    if scope == "reject":
        return RawEntityReferenceDecision(
            selected_candidate_id=None,
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            scope_reason="not reusable",
        )
    return RawEntityReferenceDecision(
        selected_candidate_id="candidate_1",
        reference_scope=scope,
        visible_region="whole" if scope == "full" else "central",
        whole_entity_recognizable=scope == "full",
        identity_features_visible=True,
        scope_reason="clear visual identity",
    )


class _Judge:
    def __init__(self, scopes: dict[str, str] | None = None) -> None:
        self.scopes = scopes or {}
        self.calls: list[tuple[str, list[str]]] = []
        self.close_calls = 0

    def decide(self, *, entity, candidates, source_images):
        self.calls.append(
            (entity.entity_id, [item.candidate_id for item in candidates])
        )
        assert set(source_images) == {item.image_path for item in candidates}
        return EntityReferenceDecisionAttempt(
            decision=_decision(self.scopes.get(entity.entity_id, "full")),
            raw_responses=("{}",),
            repair_attempts=1 if entity.entity_id == "e1" else 0,
        )

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    "pair_config",
    [
        PairConfig(enabled=1),
        PairConfig(max_candidates_per_entity=True),
        PairConfig(max_candidates_per_entity=0),
        PairConfig(max_candidates_per_entity=11),
        PairConfig(crop_padding_ratio=0),
        PairConfig(crop_padding_ratio=float("inf")),
        PairConfig(repair_retries=True),
        PairConfig(repair_retries=-1),
    ],
)
def test_pair_config_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pair_config: PairConfig,
) -> None:
    config = _config(tmp_path, monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        replace(config, pair=pair_config).validate()


def test_pair_config_changes_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        pair=replace(config.pair, max_candidates_per_entity=4),
    )
    assert config.fingerprint() != changed.fingerprint()


def test_context_and_crop_preserve_exact_pixels_without_resize() -> None:
    pixels = np.arange(8 * 7 * 3, dtype=np.uint8).reshape(7, 8, 3)
    source = Image.fromarray(pixels, mode="RGB")
    mask = np.zeros((7, 8), dtype=bool)
    mask[1:5, 2:6] = True
    mask[2, 3] = False

    context = np.asarray(build_candidate_context_image(source, mask))
    crop, crop_box = build_reference_crop(
        source,
        mask,
        crop_padding_ratio=0.25,
    )
    rgba = np.asarray(crop)

    assert np.array_equal(context[mask], pixels[mask])
    assert np.array_equal(
        context[6, 7], pixels[6, 7].astype(np.uint16) * 35 // 100
    )
    assert crop_box == (1, 0, 7, 6)
    assert crop.size == (6, 6)
    expected_mask = mask[0:6, 1:7]
    assert set(np.unique(rgba[..., 3])) == {0, 255}
    assert np.array_equal(rgba[..., 3] == 255, expected_mask)
    assert np.array_equal(rgba[..., :3][expected_mask], pixels[0:6, 1:7][expected_mask])
    assert np.all(rgba[..., :3][~expected_mask] == 255)


def test_crop_padding_clips_to_image_boundary() -> None:
    source = Image.new("RGB", (6, 5), (1, 2, 3))
    mask = np.zeros((5, 6), dtype=bool)
    mask[:2, :3] = True
    crop, crop_box = build_reference_crop(
        source,
        mask,
        crop_padding_ratio=0.5,
    )
    assert crop_box == (0, 0, 5, 4)
    assert crop.size == (5, 4)


def test_candidate_shortlist_is_deterministic_and_renumbered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(max_candidates_per_entity=3),
    )
    storage = _storage(config)
    clip = storage.read_clip("clip-1")
    frames = storage.read_frames("clip-1")
    masks = storage.read_masks("clip-1")
    assert clip.annotation is not None

    first = build_entity_reference_candidates(
        config,
        storage,
        clip_uid="clip-1",
        entity=clip.annotation.entities[0],
        frames=frames,
        masks=masks,
    )
    second = build_entity_reference_candidates(
        config,
        storage,
        clip_uid="clip-1",
        entity=clip.annotation.entities[0],
        frames=frames,
        masks=masks,
    )

    assert [item.candidate_id for item in first] == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
    ]
    assert [item.frame_slot for item in first] == [item.frame_slot for item in second]
    assert len(first) == 3


def test_pair_publishes_all_ready_references_and_per_type_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "subject"))
    judge = _Judge()

    stats = pair_clips(config, storage, judge=judge)
    clip = storage.read_clip("clip-1")

    assert stats.to_dict() == {
        "processed": 1,
        "skipped_existing": 0,
        "skipped_not_ready": 0,
        "failed": 0,
        "ready": 1,
        "rejected": 0,
        "entities_ready": 3,
        "entities_rejected": 0,
        "backgrounds_bound": 0,
        "repaired": 1,
        "cross_pair_attempted": 0,
        "cross_pair_ready": 0,
        "cross_pair_repaired": 0,
    }
    assert clip.pairing == PairingState(
        status="ready",
        retained_entity_ids=["e1", "e2", "e3"],
        tokens={
            "e1": "<ref_subject_1>",
            "e2": "<ref_object_1>",
            "e3": "<ref_subject_2>",
        },
    )
    assert [item.entity_id for item in clip.references.entities] == [
        "e1",
        "e2",
        "e3",
    ]
    for entity, state in zip(clip.annotation.entities, clip.references.entities):
        validate_entity_reference_artifact(
            config,
            storage,
            "clip-1",
            entity,
            state,
            storage.read_frames("clip-1"),
            storage.read_masks("clip-1"),
        )
        with Image.open(storage.selected_entity_path("clip-1", entity.entity_id)) as image:
            assert image.mode == "RGBA"
    assert not (storage.clip_dir("clip-1") / "debug" / "pair").exists()


def test_tracking_not_ready_and_local_disabled_become_content_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, allow_local=False)
    storage = _storage(config, tracking_status={"e2": "not_found"})

    stats = pair_clips(config, storage, judge=_Judge({"e1": "local"}))
    clip = storage.read_clip("clip-1")

    assert stats.rejected == 1
    assert [state.scope_reason for state in clip.references.entities] == [
        "local_reference_disabled",
        "tracking_not_ready:not_found",
    ]
    assert clip.pairing is not None
    assert clip.pairing.reason == "no_qualifying_ready_reference"
    assert not list((storage.clip_dir("clip-1") / "selected").glob("e*.png"))


def test_nonqualifying_ready_reference_is_kept_but_cannot_rescue_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)

    stats = pair_clips(
        config,
        storage,
        judge=_Judge({"e1": "reject", "e2": "full"}),
    )
    clip = storage.read_clip("clip-1")

    assert stats.rejected == 1
    assert clip.pairing is not None
    assert clip.pairing.reason == "no_qualifying_ready_reference"
    assert clip.references.entities[1].status == "ready"
    assert storage.selected_entity_path("clip-1", "e2").is_file()


def test_missing_candidate_judge_fails_config_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    invalid = replace(
        config,
        qwen=replace(config.qwen, candidate_judge=None),
    )

    with pytest.raises(
        ValueError,
        match="qwen.candidate_judge is required when pair.enabled is true",
    ):
        invalid.validate()


@pytest.mark.parametrize("status", ["pending_remove", "rejected"])
def test_nonready_background_does_not_block_or_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    if status == "pending_remove":
        background = BackgroundReferenceState(
            status="pending_remove",
            source_image_path="clips/clip-1/frames/00.jpg",
            source_frame_slot=0,
            source_frame_index=0,
            source_mask_path="clips/clip-1/background/source_mask_fake.png",
            source_foreground_area_pixels=1,
            source_foreground_area_ratio=1 / (WIDTH * HEIGHT),
        )
    else:
        background = BackgroundReferenceState(
            status="rejected",
            reason="not reusable",
        )
    storage.write_references(
        "clip-1",
        ReferencesState(background=background),
    )

    pair_clips(config, storage, judge=_Judge())

    assert storage.read_clip("clip-1").pairing.background_token is None


def test_clean_background_is_validated_and_bound_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    background = BackgroundReferenceState(
        status="clean_raw",
        source_image_path="clips/clip-1/frames/00.jpg",
        output_image_path="clips/clip-1/frames/00.jpg",
        source_frame_slot=0,
        source_frame_index=0,
        source_foreground_area_pixels=0,
        source_foreground_area_ratio=0.0,
    )
    storage.write_references(
        "clip-1",
        ReferencesState(background=background),
    )

    stats = pair_clips(config, storage, judge=_Judge())

    assert stats.backgrounds_bound == 1
    assert storage.read_clip("clip-1").pairing.background_token == "<ref_bg_1>"


def test_valid_existing_is_skipped_and_corruption_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    pair_clips(config, storage, judge=_Judge())

    skipped = pair_clips(config, storage, judge=_Judge())
    assert skipped.skipped_existing == 1
    artifact = storage.selected_entity_path("clip-1", "e1")
    with Image.open(artifact) as opened:
        pixels = np.asarray(opened).copy()
    pixels[0, 0, 3] = 127
    Image.fromarray(pixels, mode="RGBA").save(artifact, format="PNG")
    before = storage.clip_path("clip-1").read_bytes()

    failed = pair_clips(config, storage, judge=_Judge())

    assert failed.failed == 1
    assert storage.clip_path("clip-1").read_bytes() == before


def test_overwrite_removes_stale_entities_but_preserves_other_selected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    pair_clips(config, storage, judge=_Judge())
    background = storage.selected_background_output_path("clip-1")
    background.write_bytes(b"background")
    other = storage.selected_path("clip-1", "other.png")
    other.write_bytes(b"other")

    stats = pair_clips(
        config,
        storage,
        overwrite=True,
        judge=_Judge({"e2": "reject"}),
    )

    assert stats.ready == 1
    assert storage.selected_entity_path("clip-1", "e1").is_file()
    assert not storage.selected_entity_path("clip-1", "e2").exists()
    assert background.read_bytes() == b"background"
    assert other.read_bytes() == b"other"


def test_publication_failure_rolls_back_old_artifacts_and_clip_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    pair_clips(config, storage, judge=_Judge())
    old_clip = storage.clip_path("clip-1").read_bytes()
    old_artifacts = {
        entity_id: storage.selected_entity_path("clip-1", entity_id).read_bytes()
        for entity_id in ("e1", "e2")
    }

    def fail_write(*args, **kwargs):
        raise OSError("clip write failed")

    monkeypatch.setattr(storage, "write_references_and_pairing", fail_write)
    stats = pair_clips(config, storage, overwrite=True, judge=_Judge())

    assert stats.failed == 1
    assert storage.clip_path("clip-1").read_bytes() == old_clip
    assert {
        entity_id: storage.selected_entity_path("clip-1", entity_id).read_bytes()
        for entity_id in ("e1", "e2")
    } == old_artifacts
    assert not list(
        storage.selected_entity_path("clip-1", "e1").parent.glob(
            ".*-pair-*.png"
        )
    )


def test_pair_skips_ineligible_without_writing_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="pair-test")
    storage.create_clip(
        clip_uid="clip-1",
        source=ClipSource(
            video_path=str(config.dataset_json.parent / "videos" / "clip.mp4"),
            parent_video_id="parent",
            clip_suffix="1",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            t2v_caption="One entity moves.",
            entities=[
                AnnotationEntity(
                    entity_id="e1",
                    reference_type="subject",
                    phrase="entity",
                    grounding_prompt="visible entity",
                )
            ],
        ),
    )
    storage.write_coverage(
        "clip-1",
        CoverageState(
            passed=False,
            entity_visibility_summary={"e1": _visibility(False)},
        ),
    )

    stats = pair_clips(config, storage, judge=_Judge())

    assert stats.skipped_not_ready == 1
    assert storage.read_clip("clip-1").pairing is None


def test_disabled_pair_stage_fails_before_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(enabled=False),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="pair-test")
    with pytest.raises(ValueError, match="pair stage is disabled"):
        pair_clips(config, storage, judge=_Judge())


def test_owned_judge_is_lazy_and_closed_but_injected_judge_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    owned = _Judge()
    constructions: list[object] = []

    def factory(*args, **kwargs):
        constructions.append((args, kwargs))
        return owned

    monkeypatch.setattr(pair_module, "QwenEntityReferenceJudge", factory)
    stats = pair_clips(config, storage)
    assert stats.ready == 1
    assert len(constructions) == 1
    assert owned.close_calls == 1

    second_config = replace(
        config,
        run_root=config.run_root.parent / "injected",
        export_root=config.export_root.parent / "injected",
    )
    second_storage = _storage(second_config)
    injected = _Judge()
    pair_clips(second_config, second_storage, judge=injected)
    assert injected.close_calls == 0


def test_no_eligible_clip_does_not_construct_owned_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="pair-test")

    def fail_factory(*args, **kwargs):
        raise AssertionError("judge must remain lazy")

    monkeypatch.setattr(pair_module, "QwenEntityReferenceJudge", fail_factory)
    assert pair_clips(config, storage).processed == 0


def test_pipeline_pair_integration_uses_injected_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    _storage(config)
    config_path = tmp_path / "pair.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  limit: 10",
                "qwen:",
                "  annotation:",
                f"    model: {config.qwen.annotation.model}",
                "  instruction_writer:",
                f"    model: {config.qwen.instruction_writer.model}",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("pair",),
        git_commit="pair-test",
        entity_reference_judge=_Judge(),
    )

    assert result["completed_stages"] == ["pair"]
    assert result["pair"]["ready"] == 1


def test_selected_entity_path_rejects_unsafe_or_noncanonical_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    for entity_id in ("../e1", "e0", "candidate_1", "e01"):
        with pytest.raises(ValueError, match="entity_id"):
            storage.selected_entity_path("clip-1", entity_id)


def test_clip_schema_enforces_exact_ready_pair_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config)
    pair_clips(config, storage, judge=_Judge())
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    payload["pairing"]["tokens"] = {
        "e1": "<ref_subject_2>",
        "e2": "<ref_object_1>",
    }
    with pytest.raises(ValidationError, match="deterministic per-type"):
        pair_module.ClipRecord.model_validate(payload)


def _cross_decision(*, accept: bool) -> RawCrossPairDecision:
    return RawCrossPairDecision(
        verdict="accept" if accept else "reject",
        same_physical_entity=accept,
        identity_features_match=accept,
        reference_is_usable=accept,
        reason=(
            "The visible identity matches."
            if accept
            else "The visible identity does not match."
        ),
    )


class _ClipJudge:
    def __init__(
        self,
        scopes: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.scopes = scopes or {}
        self.calls: list[tuple[str, str]] = []

    def decide(self, *, entity, candidates, source_images):
        clip_uid = Path(candidates[0].image_path).parts[1]
        self.calls.append((clip_uid, entity.entity_id))
        assert set(source_images) == {item.image_path for item in candidates}
        return EntityReferenceDecisionAttempt(
            decision=_decision(
                self.scopes.get((clip_uid, entity.entity_id), "full")
            ),
            raw_responses=("{}",),
            repair_attempts=0,
        )


class _CrossJudge:
    def __init__(
        self,
        decisions: list[bool] | None = None,
        *,
        repair_attempts: int = 0,
    ) -> None:
        self.decisions = iter(decisions or [True])
        self.repair_attempts = repair_attempts
        self.calls: list[dict[str, object]] = []

    def decide(
        self,
        *,
        target_clip_uid,
        target_entity,
        target_context_image,
        target_entity_crop,
        donor_clip_uid,
        donor_entity,
        donor_reference_image,
    ):
        self.calls.append(
            {
                "target_clip_uid": target_clip_uid,
                "target_entity_id": target_entity.entity_id,
                "target_reference_type": target_entity.reference_type,
                "target_phrase": target_entity.phrase,
                "donor_clip_uid": donor_clip_uid,
                "donor_entity_id": donor_entity.entity_id,
                "donor_reference_type": donor_entity.reference_type,
                "context_mode": target_context_image.mode,
                "crop_mode": target_entity_crop.mode,
                "donor_mode": donor_reference_image.mode,
            }
        )
        return CrossPairDecisionAttempt(
            decision=_cross_decision(accept=next(self.decisions)),
            raw_responses=("{}",),
            repair_attempts=self.repair_attempts,
        )


def _same_parent_storage(
    config: V3Config,
    *,
    donor_suffix: str = "2",
    target_parent: str = "parent",
    target_tracking_status: str = "ready",
    donor_type: str = "subject",
    target_type: str = "subject",
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-pair-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix=donor_suffix,
        entity_types=(donor_type,),
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="20",
        parent_video_id=target_parent,
        entity_types=(target_type,),
        tracking_status={"e1": target_tracking_status},
    )
    return storage


def test_same_parent_fallback_copies_exact_donor_and_target_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    target = storage.read_clip("target")
    assert target.annotation is not None
    updated_annotation = target.annotation.model_copy(
        update={
            "background": BackgroundAnnotation(
                phrase="a city street",
                grounding_prompt="the visible city street",
            )
        }
    )
    write_json_atomic(
        storage.clip_path("target"),
        target.model_copy(update={"annotation": updated_annotation}).model_dump(
            mode="json"
        ),
    )
    background = BackgroundReferenceState(
        status="clean_raw",
        source_image_path="clips/target/frames/00.jpg",
        output_image_path="clips/target/frames/00.jpg",
        source_frame_slot=0,
        source_frame_index=0,
        source_foreground_area_pixels=0,
        source_foreground_area_ratio=0.0,
    )
    storage.write_references(
        "target",
        ReferencesState(background=background),
    )
    phase_a = _ClipJudge({("target", "e1"): "reject"})
    cross = _CrossJudge(repair_attempts=1)

    stats = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=cross,
    )

    donor = storage.read_clip("donor")
    target = storage.read_clip("target")
    donor_path = storage.selected_entity_path("donor", "e1")
    target_path = storage.selected_entity_path("target", "e1")
    target_reference = target.references.entities[0]
    assert target.pairing == PairingState(
        status="ready",
        retained_entity_ids=["e1"],
        tokens={"e1": "<ref_subject_1>"},
        background_token="<ref_bg_1>",
    )
    assert target_reference.source_clip_uid == "donor"
    assert target_reference.source_entity_id == "e1"
    assert target_reference.source_frame_index == (
        donor.references.entities[0].source_frame_index
    )
    assert donor.references.entities[0].source_clip_uid == "donor"
    assert donor.references.entities[0].source_entity_id == "e1"
    assert target_path.read_bytes() == donor_path.read_bytes()
    with Image.open(target_path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
    bindings = build_instruction_bindings(target)
    assert bindings[0].entity_id == "e1"
    assert bindings[0].phrase == target.annotation.entities[0].phrase
    assert bindings[1].reference_type == "background"
    assert target.references.background == background
    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 1
    assert stats.cross_pair_repaired == 1
    assert stats.backgrounds_bound == 1
    assert cross.calls[0] == {
        "target_clip_uid": "target",
        "target_entity_id": "e1",
        "target_reference_type": "subject",
        "target_phrase": "entity 1",
        "donor_clip_uid": "donor",
        "donor_entity_id": "e1",
        "donor_reference_type": "subject",
        "context_mode": "RGB",
        "crop_mode": "RGBA",
        "donor_mode": "RGBA",
    }


@pytest.mark.parametrize(
    "target_parent,target_tracking_status,donor_type,target_type,donor_scope",
    [
        ("other-parent", "ready", "subject", "subject", "full"),
        ("parent", "failed", "subject", "subject", "full"),
        ("parent", "ready", "object", "subject", "full"),
        ("parent", "ready", "subject", "subject", "local"),
    ],
)
def test_fallback_rejects_ineligible_donors_or_text_only_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_parent: str,
    target_tracking_status: str,
    donor_type: str,
    target_type: str,
    donor_scope: str,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(
        config,
        target_parent=target_parent,
        target_tracking_status=target_tracking_status,
        donor_type=donor_type,
        target_type=target_type,
    )
    phase_a = _ClipJudge(
        {
            ("donor", "e1"): donor_scope,
            ("target", "e1"): "reject",
        }
    )
    cross = _CrossJudge()

    pair_clips(config, storage, judge=phase_a, cross_pair_judge=cross)

    assert cross.calls == []
    assert storage.read_clip("target").references.entities[0].status == "rejected"
    assert not storage.selected_entity_path("target", "e1").exists()


def test_fallback_uses_natural_donor_order_and_stops_at_first_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(
            same_parent_fallback_enabled=True,
            same_parent_max_donor_references=2,
        ),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-order-test")
    for uid, suffix in (("donor-10", "10"), ("donor-2", "2")):
        _add_ready_clip(
            config,
            storage,
            clip_uid=uid,
            clip_suffix=suffix,
            entity_types=("subject",),
        )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="20",
        entity_types=("subject",),
    )
    phase_a = _ClipJudge({("target", "e1"): "reject"})
    cross = _CrossJudge([False, True])

    stats = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == [
        "donor-2",
        "donor-10",
    ]
    assert stats.cross_pair_attempted == 2
    assert stats.cross_pair_ready == 1
    assert storage.read_clip("target").references.entities[0].source_clip_uid == (
        "donor-10"
    )


@pytest.mark.parametrize("target_scope", ["full", "local"])
def test_fallback_never_replaces_an_existing_ready_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_scope: str,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    phase_a = _ClipJudge({("target", "e1"): "local"})
    cross = _CrossJudge()

    pair_clips(config, storage, judge=phase_a, cross_pair_judge=cross)

    target_reference = storage.read_clip("target").references.entities[0]
    assert target_reference.status == "ready"
    assert target_reference.reference_scope == "local"
    assert target_reference.source_clip_uid == "target"
    assert cross.calls == []


def test_cross_pair_publication_rolls_back_when_clip_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    phase_a = _ClipJudge({("target", "e1"): "reject"})
    cross = _CrossJudge()
    original_write = storage.write_references_and_pairing
    write_calls = 0

    def fail_cross_write(clip_uid, references, pairing):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 3:
            raise OSError("simulated cross-pair clip write failure")
        return original_write(clip_uid, references, pairing)

    monkeypatch.setattr(storage, "write_references_and_pairing", fail_cross_write)
    donor_path = storage.selected_entity_path("donor", "e1")

    stats = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=cross,
    )

    target = storage.read_clip("target")
    assert target.references.entities[0].status == "rejected"
    assert not storage.selected_entity_path("target", "e1").exists()
    assert donor_path.exists()
    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 0
    assert stats.failed == 1


def test_existing_cross_pair_validates_and_overwrite_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    phase_a = _ClipJudge({("target", "e1"): "reject"})
    first_cross = _CrossJudge()
    pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=first_cross,
    )
    donor_bytes = storage.selected_entity_path("donor", "e1").read_bytes()

    unused_cross = _CrossJudge()
    skipped = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=unused_cross,
    )
    assert skipped.skipped_existing == 2
    assert unused_cross.calls == []

    storage.selected_entity_path("target", "e1").write_bytes(b"corrupt")
    corrupt_cross = _CrossJudge()
    corrupt = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=corrupt_cross,
    )
    assert corrupt.failed == 1
    assert corrupt.skipped_existing == 1
    assert corrupt_cross.calls == []

    overwritten = pair_clips(
        config,
        storage,
        overwrite=True,
        judge=phase_a,
        cross_pair_judge=_CrossJudge(),
    )
    target_reference = storage.read_clip("target").references.entities[0]
    assert target_reference.source_clip_uid == "donor"
    assert storage.selected_entity_path("target", "e1").read_bytes() == donor_bytes
    assert overwritten.cross_pair_ready == 1


def test_legacy_in_pair_reference_without_provenance_still_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    reference = payload["references"]["entities"][0]
    reference.pop("source_clip_uid")
    reference.pop("source_entity_id")
    storage.clip_path("clip-1").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    stats = pair_clips(config, storage, judge=_Judge())

    assert stats.skipped_existing == 1
    loaded = storage.read_clip("clip-1").references.entities[0]
    assert loaded.source_clip_uid is None
    assert loaded.source_entity_id is None

def test_donor_limit_caps_judge_calls_and_all_reject_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(
            same_parent_fallback_enabled=True,
            same_parent_max_donor_references=2,
        ),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-limit-test")
    for uid, suffix in (
        ("donor-10", "10"),
        ("donor-2", "2"),
        ("donor-1", "1"),
    ):
        _add_ready_clip(
            config,
            storage,
            clip_uid=uid,
            clip_suffix=suffix,
            entity_types=("subject",),
        )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="20",
        entity_types=("subject",),
    )
    cross = _CrossJudge([False, False])

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == [
        "donor-1",
        "donor-2",
    ]
    assert stats.cross_pair_attempted == 2
    assert stats.cross_pair_ready == 0
    target = storage.read_clip("target")
    assert target.pairing is not None
    assert target.pairing.status == "rejected"
    assert target.references.entities[0].status == "rejected"
    assert not storage.selected_entity_path("target", "e1").exists()

def test_pipeline_pair_integration_passes_injected_cross_pair_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    _same_parent_storage(config)
    config_path = tmp_path / "cross-pair.yaml"
    assert config.qwen.candidate_judge is not None
    assert config.qwen.background_remove_judge is not None
    assert config.qwen.cross_pair_judge is not None
    config_path.write_text(
        "\n".join(
            [
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  limit: 10",
                "qwen:",
                "  annotation:",
                f"    model: {config.qwen.annotation.model}",
                "  instruction_writer:",
                f"    model: {config.qwen.instruction_writer.model}",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "  cross_pair_judge:",
                f"    model: {config.qwen.cross_pair_judge.model}",
                "pair:",
                "  same_parent_fallback_enabled: true",
                "  same_parent_max_donor_references: 8",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
            ]
        ),
        encoding="utf-8",
    )
    cross = _CrossJudge()

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("pair",),
        git_commit="cross-pair-test",
        entity_reference_judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert result["pair"]["cross_pair_ready"] == 1
    assert cross.calls[0]["target_clip_uid"] == "target"

def test_cross_pair_debug_is_json_only_and_disabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_config = _config(
        tmp_path / "enabled",
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
        debug=True,
    )
    enabled_storage = _same_parent_storage(enabled_config)
    pair_clips(
        enabled_config,
        enabled_storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=_CrossJudge(),
    )
    debug_root = (
        enabled_storage.clip_dir("target")
        / "debug"
        / "pair"
        / "e1"
        / "cross_pair"
    )
    artifact = debug_root / "donor-e1" / "decision.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert set(payload) == {
        "target",
        "donor",
        "raw_responses",
        "issues",
        "repair_attempts",
        "decision",
    }
    assert not list(debug_root.rglob("*.png"))

    disabled_config = _config(
        tmp_path / "disabled",
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    disabled_storage = _same_parent_storage(disabled_config)
    pair_clips(
        disabled_config,
        disabled_storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=_CrossJudge(),
    )
    assert not (disabled_storage.clip_dir("target") / "debug").exists()


def test_donor_index_scans_once_and_validates_each_artifact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-index-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="1",
        entity_types=("subject", "object"),
    )
    for target_uid, suffix in (("target-a", "2"), ("target-b", "3")):
        _add_ready_clip(
            config,
            storage,
            clip_uid=target_uid,
            clip_suffix=suffix,
            entity_types=("subject", "object"),
        )

    iter_calls = 0
    original_iter_clips = storage.iter_clips

    def counted_iter_clips():
        nonlocal iter_calls
        iter_calls += 1
        yield from original_iter_clips()

    index_builds = 0
    original_build_index = pair_module._build_same_parent_donor_index

    def counted_build_index(config, storage):
        nonlocal index_builds
        index_builds += 1
        return original_build_index(config, storage)

    validation_calls: dict[tuple[str, str], int] = {}
    original_validate = pair_module.validate_entity_reference_artifact

    def counted_validate(
        config,
        storage,
        clip_uid,
        annotation_entity,
        reference_state,
        frames,
        masks,
    ):
        key = (clip_uid, annotation_entity.entity_id)
        validation_calls[key] = validation_calls.get(key, 0) + 1
        return original_validate(
            config,
            storage,
            clip_uid,
            annotation_entity,
            reference_state,
            frames,
            masks,
        )

    monkeypatch.setattr(storage, "iter_clips", counted_iter_clips)
    monkeypatch.setattr(
        pair_module,
        "_build_same_parent_donor_index",
        counted_build_index,
    )
    monkeypatch.setattr(
        pair_module,
        "validate_entity_reference_artifact",
        counted_validate,
    )
    phase_a = _ClipJudge(
        {
            (target_uid, entity_id): "reject"
            for target_uid in ("target-a", "target-b")
            for entity_id in ("e1", "e2")
        }
    )

    stats = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=_CrossJudge([True, True, True, True]),
    )

    assert iter_calls == 2  # Phase A once, donor index once.
    assert index_builds == 1
    assert validation_calls == {("donor", "e1"): 1, ("donor", "e2"): 1}
    assert stats.cross_pair_ready == 4


def test_target_self_is_excluded_before_donor_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(
            same_parent_fallback_enabled=True,
            same_parent_max_donor_references=1,
        ),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-self-limit-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="1",
        entity_types=("subject", "subject"),
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="2",
        entity_types=("subject",),
    )
    cross = _CrossJudge()

    pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e2"): "reject"}),
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == ["donor"]
    target_reference = storage.read_clip("target").references.entities[1]
    assert target_reference.source_clip_uid == "donor"


def test_disabled_fallback_never_builds_donor_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _same_parent_storage(config)

    def unexpected_index_build(config, storage):
        del config, storage
        raise AssertionError("disabled fallback must not build a donor index")

    monkeypatch.setattr(
        pair_module,
        "_build_same_parent_donor_index",
        unexpected_index_build,
    )
    cross = _CrossJudge()

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert stats.cross_pair_attempted == 0
    assert cross.calls == []


def test_cross_pair_does_not_double_count_existing_background_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-background-count-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="1",
        entity_types=("subject",),
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="2",
        entity_types=("subject", "subject"),
    )
    background = BackgroundReferenceState(
        status="clean_raw",
        source_image_path="clips/target/frames/00.jpg",
        output_image_path="clips/target/frames/00.jpg",
        source_frame_slot=0,
        source_frame_index=0,
        source_foreground_area_pixels=0,
        source_foreground_area_ratio=0.0,
    )
    storage.write_references(
        "target",
        ReferencesState(background=background),
    )

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e2"): "reject"}),
        cross_pair_judge=_CrossJudge(),
    )

    target = storage.read_clip("target")
    assert target.pairing is not None
    assert target.pairing.status == "ready"
    assert target.pairing.background_token == "<ref_bg_1>"
    assert target.references.entities[1].source_clip_uid == "donor"
    assert stats.cross_pair_ready == 1
    assert stats.backgrounds_bound == 1
