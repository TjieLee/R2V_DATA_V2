from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.pair as pair_module
import r2v_data_v2.v3.reference_completion as completion_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import (
    PairConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceEditConfig,
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
    build_cross_pair_target_contact_sheet,
    build_entity_reference_candidates,
    build_reference_crop,
    pair_clips,
    validate_entity_reference_artifact,
)
from r2v_data_v2.v3.reference_completion_benchmark import (
    ReferenceCompletionReview,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    QWEN_LOCALIZED_PROMPT_EN_SHORT,
    QwenImageEdit2511CompletionConfig,
    QwenImageEdit2511ReferenceCompletionBackend,
)
from r2v_data_v2.v3.reference_edit import reference_edit_clips
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguBackgroundReview,
    BooguCompletionReview,
    BooguEditOutput,
    BooguSamReview,
)
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
)
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    ClipSource,
    CoverageState,
    EntityVisibilitySummary,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    RawCrossPairDecision,
    RawEntityReferenceDecision,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage
from run_pipeline_v3 import STAGE_ORDER, run_pipeline_v3

WIDTH = 12
HEIGHT = 9


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_local: bool = True,
    allow_synthetic_completion: bool = False,
    pair: PairConfig | None = None,
    debug: bool = False,
    reference_edit_enabled: bool = False,
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
            reference_edit_judge=(
                QwenServiceConfig(model=model) if reference_edit_enabled else None
            ),
        ),
        reference_scope=ReferenceScopeConfig(
            allow_local=allow_local,
            allow_synthetic_completion=allow_synthetic_completion,
        ),
        pair=pair or PairConfig(),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "object-remover",
        ),
        reference_edit=(
            ReferenceEditConfig(
                enabled=True,
                python_executable=writable / "venvs" / "boogu" / "python",
                code_root=writable / "vendor" / "Boogu-Image",
                model_path=writable / "models" / "Boogu-Image-0.1-Edit-Turbo",
            )
            if reference_edit_enabled
            else ReferenceEditConfig()
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
            video_path=str(config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"),
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
                top = 0 if config.reference_scope.allow_synthetic_completion else 2
                left = min(index + (slot % 2), WIDTH - 5)
                mask[top : top + 5, left : left + 4] = True
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
            image_quality="acceptable",
            completeness="fragmented",
            reference_scope="reject",
            visible_region="custom",
            whole_entity_recognizable=False,
            identity_features_visible=False,
            scope_reason="not reusable",
        )
    return RawEntityReferenceDecision(
        selected_candidate_id="candidate_1",
        image_quality="high",
        completeness="complete" if scope == "full" else "local_usable",
        reference_scope=scope,
        visible_region="whole" if scope == "full" else "central",
        whole_entity_recognizable=(scope in {"full", "local"}),
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


def test_cross_pair_target_contact_sheet_is_bounded_and_source_safe() -> None:
    frame_images = [
        (
            slot,
            Image.new(
                "RGB",
                (80, 40),
                ((slot * 23) % 256, 100, 150),
            ),
        )
        for slot in range(10)
    ]
    source_pixels = [np.asarray(image).copy() for _, image in frame_images]

    sheet = build_cross_pair_target_contact_sheet(
        frame_images,
        panel_max_side=40,
    )

    assert sheet.mode == "RGB"
    assert sheet.size == (200, 136)
    sheet_pixels = np.asarray(sheet)
    for slot, image in frame_images:
        column = slot % 5
        row = slot // 5
        origin_x = column * 40
        origin_y = row * 68
        assert sheet.getpixel((origin_x + 20, origin_y + 48)) == (
            (slot * 23) % 256,
            100,
            150,
        )
        assert sheet.getpixel((origin_x + 20, origin_y + 67)) == (
            255,
            255,
            255,
        )
        assert np.any(
            sheet_pixels[
                origin_y : origin_y + 28,
                origin_x : origin_x + 40,
            ]
            != 255
        )
        assert np.array_equal(np.asarray(image), source_pixels[slot])

    with pytest.raises(ValueError, match="ordered slots 0 through 9"):
        build_cross_pair_target_contact_sheet(frame_images[:-1])


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
    assert np.array_equal(context[6, 7], pixels[6, 7].astype(np.uint16) * 35 // 100)
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
        "completion_attempted": 0,
        "completion_ready": 0,
        "completion_rejected": 0,
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
        with Image.open(
            storage.selected_entity_path("clip-1", entity.entity_id)
        ) as image:
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
        storage.selected_entity_path("clip-1", "e1").parent.glob(".*-pair-*.png")
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
        target_entity_visible=accept,
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
            decision=_decision(self.scopes.get((clip_uid, entity.entity_id), "full")),
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
        target_evidence_mode,
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
                "target_evidence_mode": target_evidence_mode,
                "donor_clip_uid": donor_clip_uid,
                "donor_entity_id": donor_entity.entity_id,
                "donor_reference_type": donor_entity.reference_type,
                "context_mode": target_context_image.mode,
                "context_size": target_context_image.size,
                "crop_mode": (
                    target_entity_crop.mode if target_entity_crop is not None else None
                ),
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
        "target_evidence_mode": "masked_candidate",
        "donor_clip_uid": "donor",
        "donor_entity_id": "e1",
        "donor_reference_type": "subject",
        "context_mode": "RGB",
        "context_size": (WIDTH, HEIGHT),
        "crop_mode": "RGBA",
        "donor_mode": "RGBA",
    }


@pytest.mark.parametrize(
    "target_parent,target_tracking_status,donor_type,target_type,donor_scope",
    [
        ("other-parent", "ready", "subject", "subject", "full"),
        ("parent", "ready", "object", "subject", "full"),
        ("parent", "ready", "subject", "subject", "local"),
    ],
)
def test_fallback_rejects_ineligible_donors(
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


@pytest.mark.parametrize("tracking_status", ["not_found", "failed"])
def test_fallback_uses_sampled_frames_without_a_target_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracking_status: str,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(
        config,
        target_tracking_status=tracking_status,
    )
    phase_a = _ClipJudge()
    cross = _CrossJudge()

    stats = pair_clips(
        config,
        storage,
        judge=phase_a,
        cross_pair_judge=cross,
    )

    assert ("target", "e1") not in phase_a.calls
    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 1
    assert cross.calls[0]["target_evidence_mode"] == "sampled_frames"
    assert cross.calls[0]["context_mode"] == "RGB"
    assert cross.calls[0]["context_size"] == (1920, 824)
    assert cross.calls[0]["crop_mode"] is None
    target = storage.read_clip("target")
    assert target.references.entities[0].source_clip_uid == "donor"
    assert (
        storage.selected_entity_path(
            "target",
            "e1",
        ).read_bytes()
        == storage.selected_entity_path(
            "donor",
            "e1",
        ).read_bytes()
    )


def test_ready_tracking_without_a_valid_candidate_uses_sampled_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    original_builder = pair_module.build_entity_reference_candidates

    def without_target_candidates(
        config,
        storage,
        *,
        clip_uid,
        entity,
        frames,
        masks,
    ):
        if clip_uid == "target":
            return []
        return original_builder(
            config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            frames=frames,
            masks=masks,
        )

    monkeypatch.setattr(
        pair_module,
        "build_entity_reference_candidates",
        without_target_candidates,
    )
    cross = _CrossJudge()

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge(),
        cross_pair_judge=cross,
    )

    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 1
    assert cross.calls[0]["target_evidence_mode"] == "sampled_frames"
    assert cross.calls[0]["crop_mode"] is None


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
def test_real_donor_replaces_only_lower_priority_local_reference(
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
    phase_a = _ClipJudge({("target", "e1"): target_scope})
    cross = _CrossJudge()

    pair_clips(config, storage, judge=phase_a, cross_pair_judge=cross)

    target_reference = storage.read_clip("target").references.entities[0]
    assert target_reference.status == "ready"
    assert target_reference.reference_scope == "full"
    if target_scope == "full":
        assert target_reference.source_clip_uid == "target"
        assert cross.calls == []
    else:
        assert target_reference.source_clip_uid == "donor"
        assert len(cross.calls) == 1


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


def test_context_only_existing_cross_pair_requires_complete_sampled_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(
        config,
        target_tracking_status="not_found",
    )
    first_cross = _CrossJudge()
    pair_clips(
        config,
        storage,
        judge=_ClipJudge(),
        cross_pair_judge=first_cross,
    )
    assert first_cross.calls[0]["target_evidence_mode"] == "sampled_frames"

    unused_cross = _CrossJudge()
    skipped = pair_clips(
        config,
        storage,
        judge=_ClipJudge(),
        cross_pair_judge=unused_cross,
    )
    assert skipped.skipped_existing == 2
    assert unused_cross.calls == []

    storage.frame_path("target", 9).unlink()
    incomplete_cross = _CrossJudge()
    invalid = pair_clips(
        config,
        storage,
        judge=_ClipJudge(),
        cross_pair_judge=incomplete_cross,
    )
    assert invalid.failed == 1
    assert invalid.skipped_existing == 1
    assert incomplete_cross.calls == []


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


def test_masked_cross_pair_debug_is_json_only_and_disabled_by_default(
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
        enabled_storage.clip_dir("target") / "debug" / "pair" / "e1" / "cross_pair"
    )
    artifact = debug_root / "donor-e1" / "decision.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert set(payload) == {
        "target_evidence_mode",
        "target_frame_slots",
        "target",
        "donor",
        "raw_responses",
        "issues",
        "repair_attempts",
        "decision",
    }
    assert payload["target_evidence_mode"] == "masked_candidate"
    assert len(payload["target_frame_slots"]) == 1
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


def test_sampled_frame_debug_saves_contact_sheet_only_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_config = _config(
        tmp_path / "enabled-sampled",
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
        debug=True,
    )
    enabled_storage = _same_parent_storage(
        enabled_config,
        target_tracking_status="not_found",
    )
    pair_clips(
        enabled_config,
        enabled_storage,
        judge=_ClipJudge(),
        cross_pair_judge=_CrossJudge(),
    )
    debug_directory = (
        enabled_storage.clip_dir("target")
        / "debug"
        / "pair"
        / "e1"
        / "cross_pair"
        / "donor-e1"
    )
    payload = json.loads(
        (debug_directory / "decision.json").read_text(encoding="utf-8")
    )
    assert payload["target_evidence_mode"] == "sampled_frames"
    assert payload["target_frame_slots"] == list(range(10))
    contact_sheet = debug_directory / "target_contact_sheet.png"
    with Image.open(contact_sheet) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (1920, 824)
    assert not list(
        enabled_storage.selected_entity_path("target", "e1").parent.glob("*contact*")
    )

    disabled_config = _config(
        tmp_path / "disabled-sampled",
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    disabled_storage = _same_parent_storage(
        disabled_config,
        target_tracking_status="not_found",
    )
    pair_clips(
        disabled_config,
        disabled_storage,
        judge=_ClipJudge(),
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


def test_four_context_only_targets_attempt_judge_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-context-smoke-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="1",
        entity_types=("subject",),
    )
    target_uids = [f"target-{index}" for index in range(4)]
    for index, target_uid in enumerate(target_uids, start=2):
        _add_ready_clip(
            config,
            storage,
            clip_uid=target_uid,
            clip_suffix=str(index),
            entity_types=("subject",),
            tracking_status={"e1": "not_found" if index % 2 == 0 else "failed"},
        )
    cross = _CrossJudge([False, False, False, False])

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge(),
        cross_pair_judge=cross,
    )

    assert stats.cross_pair_attempted == 4
    assert stats.cross_pair_ready == 0
    assert len(cross.calls) == 4
    assert {call["target_evidence_mode"] for call in cross.calls} == {"sampled_frames"}
    for target_uid in target_uids:
        target = storage.read_clip(target_uid)
        assert target.references.entities[0].status == "rejected"
        assert not storage.selected_entity_path(target_uid, "e1").exists()


class _LocalizedCompletionBackend:
    def __init__(
        self,
        *,
        fail: Exception | None = None,
        return_input_unchanged: bool = False,
    ) -> None:
        self.fail = fail
        self.return_input_unchanged = return_input_unchanged
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        input_rgb,
        entity_phrase,
        seed,
        prompt,
        negative_prompt,
    ):
        self.calls.append(
            {
                "input_size": input_rgb.size,
                "entity_phrase": entity_phrase,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )
        if self.fail is not None:
            raise self.fail
        if self.return_input_unchanged:
            return input_rgb.copy()
        pixels = np.asarray(input_rgb, dtype=np.uint8).copy()
        visible = np.any(pixels < 250, axis=2)
        rows, columns = np.nonzero(visible)
        assert rows.size
        top = int(rows.min())
        left = int(columns.min())
        right = int(columns.max()) + 1
        if top > 0:
            pixels[top - 1, left:right] = pixels[top, left:right]
        else:
            bottom = int(rows.max()) + 1
            assert bottom < pixels.shape[0]
            pixels[bottom, left:right] = pixels[bottom - 1, left:right]
        return Image.fromarray(pixels, mode="RGB")


class _CompletionGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _CompletionGenerator:
        self.seed = seed
        return self


class _PaddedLocalizedCompletionPipeline:
    vae_scale_factor = 8

    def __init__(self, *, returned_size: tuple[int, int] | None = None) -> None:
        self.returned_size = returned_size
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        image,
        prompt,
        negative_prompt,
        height,
        width,
        generator,
        true_cfg_scale,
        guidance_scale,
        num_inference_steps,
        num_images_per_prompt,
    ):
        self.calls.append(
            {
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
            }
        )
        if self.returned_size is not None:
            return SimpleNamespace(
                images=[Image.new("RGB", self.returned_size, "white")]
            )
        model_input = image[0]
        assert isinstance(model_input, Image.Image)
        assert model_input.mode == "RGB"
        assert model_input.size == (width, height)
        pixels = np.asarray(model_input, dtype=np.uint8).copy()
        visible = np.any(pixels < 250, axis=2)
        rows, columns = np.nonzero(visible)
        assert rows.size
        top = int(rows.min())
        left = int(columns.min())
        right = int(columns.max()) + 1
        if top > 0:
            pixels[top - 1, left:right] = pixels[top, left:right]
        else:
            bottom = int(rows.max()) + 1
            assert bottom < pixels.shape[0]
            pixels[bottom, left:right] = pixels[bottom - 1, left:right]
        return SimpleNamespace(images=[Image.fromarray(pixels, mode="RGB")])


def _production_completion_backend(
    config: V3Config,
    pipeline: _PaddedLocalizedCompletionPipeline,
) -> QwenImageEdit2511ReferenceCompletionBackend:
    config.remove.base_model_path.mkdir(parents=True, exist_ok=True)
    return QwenImageEdit2511ReferenceCompletionBackend(
        QwenImageEdit2511CompletionConfig(
            model_path=config.remove.base_model_path,
            device=config.remove.device,
            dtype=config.remove.dtype,
            num_inference_steps=config.remove.num_inference_steps,
            true_cfg_scale=config.remove.true_cfg_scale,
            guidance_scale=config.remove.guidance_scale,
            mode="localized_raw",
            force_input_size=True,
        ),
        pipeline=pipeline,
        torch_module=SimpleNamespace(Generator=_CompletionGenerator),
    )


class _LocalizedCompletionJudge:
    def __init__(
        self,
        *,
        verdict: str = "accept",
        reason: str = "The local missing region is repaired cleanly.",
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.verdict = verdict
        self.reason = reason

    def review(
        self,
        *,
        source_rgba,
        candidate_rgb,
        entity_phrase,
        reference_type,
    ):
        assert source_rgba.mode == "RGBA"
        assert candidate_rgb.mode == "RGB"
        self.calls.append((entity_phrase, reference_type))
        accepted = self.verdict == "accept"
        return ReferenceCompletionReview(
            verdict=self.verdict,
            visible_source_preserved=accepted,
            same_entity_continued=accepted,
            identity_preserved=accepted,
            exactly_one_entity=accepted,
            completion_plausible=accepted,
            completion_useful=accepted,
            no_occluder_reconstructed=accepted,
            no_new_salient_entity=accepted,
            boundary_clean=accepted,
            reference_usable=accepted,
            reason=self.reason,
        )


class _SingleFrameSegmentationBackend:
    def __init__(self, status: str = "ready") -> None:
        self.status = status
        self.calls: list[dict[str, object]] = []

    def track(
        self,
        *,
        frame_paths,
        entity_id,
        reference_type,
        grounding_prompt,
    ):
        assert len(frame_paths) == 1
        isolated_frame_path = frame_paths[0]
        assert isolated_frame_path.name == "00.jpg"
        assert isolated_frame_path.is_file()
        session_entries = sorted(
            path.name for path in isolated_frame_path.parent.iterdir()
        )
        assert session_entries == ["00.jpg"]
        with Image.open(isolated_frame_path) as opened:
            opened.load()
            assert opened.format == "JPEG"
            assert opened.mode == "RGB"
            isolated_pixels = np.asarray(opened, dtype=np.uint8)
        working_directory = isolated_frame_path.parent.parent
        working_png_names = sorted(
            path.name for path in working_directory.glob("*.png")
        )
        assert working_png_names == [
            "candidate_rgb.png",
            "input_rgb.png",
            "source_rgba.png",
        ]
        candidate_path = working_directory / "candidate_rgb.png"
        assert candidate_path.is_file()
        self.calls.append(
            {
                "frame_paths": list(frame_paths),
                "session_directory": isolated_frame_path.parent,
                "session_entries": session_entries,
                "working_png_names": working_png_names,
                "candidate_bytes": candidate_path.read_bytes(),
                "entity_id": entity_id,
                "reference_type": reference_type,
                "grounding_prompt": grounding_prompt,
            }
        )
        if self.status == "exception":
            raise RuntimeError("segmentation crashed")
        if self.status == "not_found":
            return EntityTrackResult(status="not_found", reason="not found")
        if self.status == "failed":
            return EntityTrackResult(status="failed", reason="segmentation failed")
        if self.status == "missing_slot":
            return EntityTrackResult(
                status="ready",
                observations=(
                    BackendMaskObservation(
                        slot=1,
                        mask=np.ones((HEIGHT, WIDTH), dtype=bool),
                        confidence=0.97,
                        object_id="wrong-slot",
                    ),
                ),
            )
        mask = (
            np.ones(isolated_pixels.shape[:2], dtype=bool)
            if self.status == "mask_gate"
            else np.any(isolated_pixels < 240, axis=2)
        )
        return EntityTrackResult(
            status="ready",
            observations=(
                BackendMaskObservation(
                    slot=0,
                    mask=mask,
                    confidence=0.97,
                    object_id="generated-entity",
                ),
            ),
        )


class _LocalThenGeneratedFullJudge:
    def __init__(
        self,
        *,
        generated_scope: str = "full",
        source_recognizable: bool = True,
    ) -> None:
        self.calls: list[str] = []
        self.generated_scope = generated_scope
        self.source_recognizable = source_recognizable

    def decide(self, *, entity, candidates, source_images):
        del entity
        assert set(source_images) == {item.image_path for item in candidates}
        generated = "reference_completion" in candidates[0].image_path
        self.calls.append("generated" if generated else "source")
        decision = _decision(self.generated_scope if generated else "local")
        if not generated and not self.source_recognizable:
            decision = decision.model_copy(update={"whole_entity_recognizable": False})
        return EntityReferenceDecisionAttempt(
            decision=decision,
            raw_responses=("{}",),
            repair_attempts=0,
        )


def _completion_rejection_path(
    storage: RunStorage,
    *,
    clip_uid: str = "clip-1",
    entity_id: str = "e1",
) -> Path:
    return (
        storage.clip_dir(clip_uid)
        / "reference_completion"
        / entity_id
        / "rejection.json"
    )


def _read_completion_rejection(storage: RunStorage) -> dict[str, object]:
    return json.loads(_completion_rejection_path(storage).read_text(encoding="utf-8"))


def _assert_completion_rejection_artifacts(
    directory: Path,
    *,
    expected_filename: str = "rejection.json",
) -> None:
    assert {path.name for path in directory.iterdir()} == {
        "diagnostics",
        expected_filename,
    }
    diagnostic_names = {path.name for path in (directory / "diagnostics").iterdir()}
    assert {
        "cleaned_source.png",
        "protected_component_mask.png",
        "recovery_corridor_mask.png",
    }.issubset(diagnostic_names)
    assert not list(directory.parent.glob(".tmp-*"))
    assert not list(directory.parent.glob(".backup-*"))


def test_generated_fallback_uses_existing_components_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    completion = _LocalizedCompletionBackend()
    completion_judge = _LocalizedCompletionJudge()
    segmenter = _SingleFrameSegmentationBackend()
    reference_judge = _LocalThenGeneratedFullJudge()

    stats = pair_clips(
        config,
        storage,
        judge=reference_judge,
        completion_backend=completion,
        completion_judge=completion_judge,
        completion_segmentation_backend=segmenter,
    )

    clip = storage.read_clip("clip-1")
    state = clip.references.entities[0]
    assert stats.completion_attempted == 1
    assert stats.completion_ready == 1
    assert stats.completion_rejected == 0
    assert state.status == "ready"
    assert state.reference_scope == "full"
    assert state.synthetic is True
    assert state.source_clip_uid == "clip-1"
    assert state.source_entity_id == "e1"
    assert state.generation_metadata_path is not None
    assert completion.calls[0]["entity_phrase"] == "entity 1"
    assert completion.calls[0]["prompt"] == QWEN_LOCALIZED_PROMPT_EN_SHORT
    assert completion.calls[0]["prompt"] == (
        "Complete the missing parts in this image. Do not generate a new instance."
    )
    assert segmenter.calls[0]["grounding_prompt"] == "entity 1"
    assert segmenter.calls[0]["session_entries"] == ["00.jpg"]
    assert segmenter.calls[0]["working_png_names"] == [
        "candidate_rgb.png",
        "input_rgb.png",
        "source_rgba.png",
    ]
    assert not segmenter.calls[0]["session_directory"].exists()
    assert reference_judge.calls == ["source", "generated"]
    final_path = storage.selected_entity_path("clip-1", "e1")
    with Image.open(final_path) as opened:
        assert opened.format == "PNG"
        assert opened.mode == "RGBA"
        pixels = np.asarray(opened, dtype=np.uint8)
    assert set(np.unique(pixels[..., 3])).issubset({0, 255})
    assert np.all(pixels[..., :3][pixels[..., 3] == 0] == 255)
    metadata_path = storage.root / state.generation_metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "accepted"
    assert metadata["completion"]["mode"] == "localized_raw"
    assert metadata["completion"]["prompt_language"] == "en"
    assert metadata["completion"]["prompt"] == QWEN_LOCALIZED_PROMPT_EN_SHORT
    assert metadata["segmentation"]["backend"] == "sam3"
    assert metadata["fallback_used"] is False
    assert metadata["recovery_direction"] == ["top"]
    assert metadata["protected_component_ids"] == ["component_1"]
    assert metadata["candidate_new_region_stats"]["matching_recovery_directions"] == [
        "top"
    ]
    diagnostic_directory = metadata_path.parent / "diagnostics"
    assert {path.name for path in diagnostic_directory.iterdir()} == {
        "cleaned_source.png",
        "protected_component_mask.png",
        "recovery_corridor_mask.png",
        "sam3_candidate_mask.png",
    }
    assert not _completion_rejection_path(storage).exists()
    assert not (metadata_path.parent / ".sam3_single_frame").exists()
    candidate_path = metadata_path.parent / "candidate_rgb.png"
    assert candidate_path.read_bytes() == segmenter.calls[0]["candidate_bytes"]
    assert _sha256(candidate_path) == metadata["candidate_rgb_sha256"]
    assert metadata["production_gates"] == {
        **metadata["production_gates"],
        "mask_passed": True,
        "improvement_passed": True,
        "background_passed": True,
    }
    validate_entity_reference_artifact(
        config,
        storage,
        "clip-1",
        clip.annotation.entities[0],
        state,
        storage.read_frames("clip-1"),
        storage.read_masks("clip-1"),
    )

    unused_completion = _LocalizedCompletionBackend(
        fail=AssertionError("existing generated fallback must be skipped")
    )
    skipped = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=unused_completion,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )
    assert skipped.skipped_existing == 1
    assert skipped.completion_attempted == 0
    assert unused_completion.calls == []

    body = "Keep {{image_1}} visually consistent throughout the clip."
    legend = [
        InstructionLegendEntry(
            image_id="image_1",
            description="The completed subject reference.",
        )
    ]
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    dataset = DatasetExporter(config, storage).export()
    assert dataset.sample_count == 1
    sample = json.loads(
        (config.resolved_export_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    exported_reference = sample["references"][0]
    assert exported_reference["synthetic"] is True
    assert _sha256(config.resolved_export_root / exported_reference["image_path"]) == (
        state.generation_output_sha256
    )


def test_production_qwen_backend_accepts_arbitrary_policy_canvas_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pipeline = _PaddedLocalizedCompletionPipeline()
    backend = _production_completion_backend(config, pipeline)

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=backend,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    assert stats.completion_attempted == 1
    assert stats.completion_ready == 1
    assert stats.completion_rejected == 0
    size_diagnostics = backend.last_size_diagnostics
    assert size_diagnostics is not None
    original_size = size_diagnostics["original_model_input_size"]
    padded_size = size_diagnostics["padded_model_input_size"]
    assert isinstance(original_size, list)
    assert isinstance(padded_size, list)
    assert any(value % 16 for value in original_size)
    assert all(value % 16 == 0 for value in padded_size)
    assert size_diagnostics["model_multiple"] == 16
    assert size_diagnostics["returned_model_output_size"] == padded_size
    request = pipeline.calls[0]
    assert [request["width"], request["height"]] == padded_size
    model_input = request["image"][0]
    assert isinstance(model_input, Image.Image)
    assert list(model_input.size) == padded_size

    state = storage.read_clip("clip-1").references.entities[0]
    assert state.generation_metadata_path is not None
    metadata = json.loads(
        (storage.root / state.generation_metadata_path).read_text(encoding="utf-8")
    )
    completion_metadata = metadata["completion"]
    for field_name, expected in size_diagnostics.items():
        assert completion_metadata[field_name] == expected
    assert (
        completion_metadata["original_model_input_size"]
        == metadata["completion_input_size"]
    )


def test_production_qwen_size_failure_records_rejection_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pipeline = _PaddedLocalizedCompletionPipeline(returned_size=(1, 1))
    backend = _production_completion_backend(config, pipeline)

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=backend,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    assert stats.completion_attempted == 1
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 1
    rejection = _read_completion_rejection(storage)
    assert rejection["stage"] == "generation"
    assert rejection["exception_type"] == "RuntimeError"
    assert rejection["original_model_input_size"]
    assert rejection["padded_model_input_size"]
    assert rejection["model_multiple"] == 16
    assert rejection["returned_model_output_size"] == [1, 1]
    assert rejection["padding_right"] >= 0
    assert rejection["padding_bottom"] >= 0
    assert "original_size=" in rejection["reason"]
    assert "padded_size=" in rejection["reason"]
    assert "returned_size=(1, 1)" in rejection["reason"]
    assert "model_multiple=16" in rejection["reason"]
    state = storage.read_clip("clip-1").references.entities[0]
    assert state.status == "ready"
    assert state.reference_scope == "local"
    assert state.synthetic is False


def test_generated_fallback_is_disabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    completion = _LocalizedCompletionBackend(
        fail=AssertionError("disabled completion must not call Qwen")
    )

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=completion,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    assert state.status == "ready"
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert stats.completion_attempted == 0
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 0
    assert completion.calls == []


def test_ineligible_local_reference_skips_completion_and_keeps_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    completion = _LocalizedCompletionBackend(
        fail=AssertionError("ineligible source must not call Qwen")
    )
    completion_judge = _LocalizedCompletionJudge()
    segmenter = _SingleFrameSegmentationBackend("exception")

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(source_recognizable=False),
        completion_backend=completion,
        completion_judge=completion_judge,
        completion_segmentation_backend=segmenter,
    )

    state = storage.read_clip("clip-1").references.entities[0]
    assert state.status == "ready"
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert state.whole_entity_recognizable is False
    assert stats.completion_attempted == 0
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 0
    assert completion.calls == []
    assert completion_judge.calls == []
    assert segmenter.calls == []
    metadata_path = _completion_rejection_path(storage).with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "skipped"
    assert metadata["completion_skipped_reason"] == (
        "completion_source_not_whole_recognizable"
    )
    assert metadata["fallback_used"] is True
    assert not _completion_rejection_path(storage).exists()
    _assert_completion_rejection_artifacts(
        metadata_path.parent,
        expected_filename="metadata.json",
    )


def test_completion_generation_failure_keeps_original_local_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    segmenter = _SingleFrameSegmentationBackend(
        "exception",
    )

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(
            fail=RuntimeError("completion failed"),
        ),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=segmenter,
    )

    state = storage.read_clip("clip-1").references.entities[0]
    assert stats.failed == 0
    assert stats.completion_attempted == 1
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 1
    assert state.status == "ready"
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert segmenter.calls == []
    rejection_path = _completion_rejection_path(storage)
    rejection = _read_completion_rejection(storage)
    assert rejection["schema_version"] == ("r2v.v3.reference_completion.rejection.1")
    assert rejection["status"] == "rejected"
    assert rejection["clip_uid"] == "clip-1"
    assert rejection["entity_id"] == "e1"
    assert rejection["reference_type"] == "subject"
    assert rejection["stage"] == "generation"
    assert rejection["reason"] == "completion failed"
    assert rejection["exception_type"] == "RuntimeError"
    assert rejection["source_reference"] == state.model_dump(mode="json")
    assert rejection["source_image_sha256"] == _sha256(
        storage.selected_entity_path("clip-1", "e1")
    )
    assert rejection["config_hash"] == config.fingerprint()
    assert rejection["git_commit"] == "pair-test"
    assert rejection["fallback_used"] is True
    assert rejection["protected_component_ids"] == ["component_1"]
    _assert_completion_rejection_artifacts(rejection_path.parent)
    completion_root = storage.clip_dir("clip-1") / "reference_completion"
    assert not list(completion_root.glob(".tmp-*"))
    assert not list(completion_root.glob(".backup-*"))
    with Image.open(storage.selected_entity_path("clip-1", "e1")) as opened:
        assert opened.mode == "RGBA"

    body = "Keep {{image_1}} visually consistent throughout the clip."
    legend = [
        InstructionLegendEntry(
            image_id="image_1",
            description="The original local subject reference.",
        )
    ]
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    dataset = DatasetExporter(config, storage).export()
    assert dataset.sample_count == 1
    sample = json.loads(
        (config.resolved_export_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    assert sample["references"][0]["synthetic"] is False


@pytest.mark.parametrize(
    "segmentation_status",
    ["not_found", "failed", "missing_slot", "exception"],
)
def test_generated_fallback_failure_keeps_original_local_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segmentation_status: str,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    reference_judge = _LocalThenGeneratedFullJudge()
    segmenter = _SingleFrameSegmentationBackend(segmentation_status)

    stats = pair_clips(
        config,
        storage,
        judge=reference_judge,
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=segmenter,
    )

    clip = storage.read_clip("clip-1")
    state = clip.references.entities[0]
    assert stats.failed == 0
    assert stats.completion_attempted == 1
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 1
    assert state.status == "ready"
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert state.source_clip_uid == "clip-1"
    assert reference_judge.calls == ["source"]
    assert not segmenter.calls[0]["session_directory"].exists()
    rejection = _read_completion_rejection(storage)
    assert rejection["stage"] == "segmentation"
    assert rejection["exception_type"] == (
        "RuntimeError" if segmentation_status == "exception" else None
    )
    if segmentation_status == "not_found":
        assert rejection["reason"] == ("SAM3 track status is not_found: not found")
        assert rejection["segmentation"] == {
            "track_status": "not_found",
            "track_reason": "not found",
        }
    rejection_directory = _completion_rejection_path(storage).parent
    _assert_completion_rejection_artifacts(rejection_directory)
    with Image.open(storage.selected_entity_path("clip-1", "e1")) as opened:
        assert opened.mode == "RGBA"


def test_completion_hard_check_rejection_records_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    segmenter = _SingleFrameSegmentationBackend("exception")

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(return_input_unchanged=True),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=segmenter,
    )

    state = storage.read_clip("clip-1").references.entities[0]
    rejection = _read_completion_rejection(storage)
    assert stats.completion_rejected == 1
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert segmenter.calls == []
    assert rejection["stage"] == "localized_hard_check"
    assert rejection["exception_type"] is None
    assert rejection["localized_hard_check"]["status"] == "failed"
    assert (
        "candidate_unchanged_from_input"
        in (rejection["localized_hard_check"]["reasons"])
    )


def test_completion_mask_gate_rejection_records_reasons_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend("mask_gate"),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    rejection = _read_completion_rejection(storage)
    assert stats.completion_rejected == 1
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert rejection["stage"] == "improvement_gate"
    assert rejection["exception_type"] is None
    assert rejection["reason"] == "completion_growth_outside_recovery_corridor"
    candidate_stats = rejection["candidate_new_region_stats"]
    assert "completion_growth_outside_recovery_corridor" in candidate_stats["reasons"]
    assert candidate_stats["outside_recovery_corridor_pixels"] > 0
    assert rejection["fallback_used"] is True


def test_completion_localized_judge_rejection_records_verdict_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    review_reason = "The completion changes the subject identity."

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(
            verdict="reject",
            reason=review_reason,
        ),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    rejection = _read_completion_rejection(storage)
    assert stats.completion_rejected == 1
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert rejection["stage"] == "localized_judge"
    assert rejection["exception_type"] is None
    assert rejection["localized_judge"] == {
        "verdict": "reject",
        "reason": review_reason,
    }


def test_completion_non_full_ranking_records_decision_and_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(generated_scope="local"),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    rejection = _read_completion_rejection(storage)
    assert stats.completion_rejected == 1
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert rejection["stage"] == "reference_ranking"
    assert rejection["exception_type"] is None
    assert rejection["reference_ranking"]["decision"]["reference_scope"] == ("local")
    assert rejection["reference_ranking"]["issues"] == []


def test_completion_publication_failure_records_stage_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    original_write = storage.write_references_and_pairing
    write_calls = 0

    def fail_fallback_publication(clip_uid, references, pairing):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise RuntimeError("completion publication failed")
        return original_write(clip_uid, references, pairing)

    monkeypatch.setattr(
        storage,
        "write_references_and_pairing",
        fail_fallback_publication,
    )

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    rejection = _read_completion_rejection(storage)
    assert write_calls == 2
    assert stats.failed == 0
    assert stats.completion_rejected == 1
    assert stats.completion_ready == 0
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert rejection["stage"] == "publication"
    assert rejection["reason"] == "completion publication failed"
    assert rejection["exception_type"] == "RuntimeError"
    completion_directory = _completion_rejection_path(storage).parent
    _assert_completion_rejection_artifacts(completion_directory)
    completion_root = completion_directory.parent
    assert not list(completion_root.glob(".tmp-*"))
    assert not list(completion_root.glob(".backup-*"))
    validate_entity_reference_artifact(
        config,
        storage,
        "clip-1",
        storage.read_clip("clip-1").annotation.entities[0],
        state,
        storage.read_frames("clip-1"),
        storage.read_masks("clip-1"),
    )


def test_completion_rejection_and_success_replace_each_other_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))

    first = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(
            fail=RuntimeError("first generation failure")
        ),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )
    first_rejection = _read_completion_rejection(storage)
    assert first.completion_rejected == 1
    assert first_rejection["stage"] == "generation"

    second = pair_clips(
        config,
        storage,
        overwrite=True,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )
    completion_directory = _completion_rejection_path(storage).parent
    assert second.completion_ready == 1
    assert not _completion_rejection_path(storage).exists()
    assert (completion_directory / "metadata.json").is_file()
    assert (completion_directory / "generated_reference.png").is_file()

    third = pair_clips(
        config,
        storage,
        overwrite=True,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend("not_found"),
    )
    third_rejection = _read_completion_rejection(storage)
    assert third.completion_rejected == 1
    assert third_rejection["stage"] == "segmentation"
    assert third_rejection["reason"] != first_rejection["reason"]
    _assert_completion_rejection_artifacts(completion_directory)
    completion_root = completion_directory.parent
    assert not list(completion_root.glob(".tmp-*"))
    assert not list(completion_root.glob(".backup-*"))
    final_state = storage.read_clip("clip-1").references.entities[0]
    assert final_state.reference_scope == "local"
    assert final_state.synthetic is False


def test_completion_rejection_diagnostic_write_failure_is_fail_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    original_write_json = completion_module.write_json_atomic

    def fail_rejection_write(path, value):
        if path.name == "rejection.json":
            raise OSError("diagnostic disk failure")
        return original_write_json(path, value)

    monkeypatch.setattr(
        completion_module,
        "write_json_atomic",
        fail_rejection_write,
    )

    stats = pair_clips(
        config,
        storage,
        judge=_LocalThenGeneratedFullJudge(),
        completion_backend=_LocalizedCompletionBackend(
            fail=RuntimeError("completion failed")
        ),
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    assert stats.completion_rejected == 1
    assert state.reference_scope == "local"
    assert state.synthetic is False
    assert not _completion_rejection_path(storage).exists()
    failures = [
        json.loads(line)
        for line in storage.failures_path.read_text(encoding="utf-8").splitlines()
    ]
    failure = failures[-1]
    assert failure["stage"] == "pair"
    assert failure["reason"] == "completion rejection diagnostic write failed"
    assert failure["details"]["completion_stage"] == "generation"
    assert failure["details"]["diagnostic_error_type"] == "OSError"
    completion_root = storage.clip_dir("clip-1") / "reference_completion"
    assert not list(completion_root.glob(".tmp-*"))
    assert not list(completion_root.glob(".backup-*"))


def test_real_donor_precedes_generated_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
        pair=PairConfig(same_parent_fallback_enabled=True),
    )
    storage = _same_parent_storage(config)
    completion = _LocalizedCompletionBackend(
        fail=AssertionError("real donor must precede Qwen fallback")
    )

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "local"}),
        cross_pair_judge=_CrossJudge(),
        completion_backend=completion,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    target = storage.read_clip("target").references.entities[0]
    assert target.source_clip_uid == "donor"
    assert target.synthetic is False
    assert stats.cross_pair_ready == 1
    assert stats.completion_attempted == 0
    assert completion.calls == []


def test_full_real_self_precedes_all_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        allow_synthetic_completion=True,
    )
    storage = _storage(config, entity_types=("subject",))
    completion = _LocalizedCompletionBackend(
        fail=AssertionError("full real self reference must not call Qwen")
    )

    stats = pair_clips(
        config,
        storage,
        judge=_Judge(),
        completion_backend=completion,
        completion_judge=_LocalizedCompletionJudge(),
        completion_segmentation_backend=_SingleFrameSegmentationBackend(),
    )

    state = storage.read_clip("clip-1").references.entities[0]
    assert state.reference_scope == "full"
    assert state.synthetic is False
    assert stats.completion_attempted == 0
    assert completion.calls == []


class _ReferenceEditBackend:
    def __init__(self) -> None:
        self.start_calls = 0
        self.close_calls = 0
        self.calls: list[dict[str, object]] = []

    def start(self, *, stderr_log_path: Path) -> None:
        self.start_calls += 1
        self.stderr_log_path = stderr_log_path

    def edit(self, **kwargs: object) -> BooguEditOutput:
        self.calls.append(kwargs)
        buffer = io.BytesIO()
        Image.new(
            "RGB",
            (int(kwargs["width"]), int(kwargs["height"])),
            (73, 91, 127),
        ).save(buffer, format="PNG")
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=buffer.getvalue(),
            original_instruction=instruction,
            rewritten_instruction=(
                instruction if bool(kwargs["thinking_enabled"]) else None
            ),
            effective_instruction=instruction,
        )

    def close(self) -> None:
        self.close_calls += 1


class _FailingReferenceEditBackend(_ReferenceEditBackend):
    def edit(self, **kwargs: object) -> BooguEditOutput:
        del kwargs
        raise RuntimeError("worker protocol failed")


class _ReferenceEditJudge:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.calls: list[str] = []

    def review(self, **kwargs: object) -> object:
        operation = str(kwargs["operation"])
        self.calls.append(operation)
        values = {
            field: self.accept
            for field in (
                "same_physical_entity",
                "identity_preserved",
                "original_visible_attributes_preserved",
                "exactly_one_entity",
                "missing_parts_plausibly_completed",
                "no_duplicate_entity",
                "no_unrelated_entity",
                "no_severe_structure_artifact",
                "style_coherent",
                "resolution_usable",
                "reference_usable",
                "certain",
            )
        }
        if operation == "complete_entity":
            return BooguCompletionReview(
                verdict="accept" if self.accept else "reject",
                reason="usable" if self.accept else "rejected",
                **values,
            )
        background_values = {
            field: self.accept
            for field in (
                "exactly_one_target_entity",
                "identity_preserved",
                "entity_appearance_consistent",
                "no_duplicate_entity",
                "no_added_salient_entity",
                "no_unintended_completion_or_extension",
                "background_coherent",
                "background_style_consistent",
                "no_halo_or_seam",
                "subject_not_severely_redrawn",
                "reference_usable",
                "certain",
            )
        }
        return BooguBackgroundReview(
            verdict="accept" if self.accept else "reject",
            reason="usable" if self.accept else "rejected",
            **background_values,
        )


class _ReferenceEditSamReviewer:
    def review(self, **kwargs: object) -> BooguSamReview:
        del kwargs
        return BooguSamReview(
            passed=True,
            target_entity_present=True,
            exactly_one_target_instance=True,
            area_growth_acceptable=True,
            fragmentation_acceptable=True,
            reason="valid",
        )


def test_reference_edit_stage_reuses_one_worker_and_exports_native_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config)
    pair_clips(config, storage, judge=_Judge())
    canonical_hashes = {
        entity_id: _sha256(storage.selected_entity_path("clip-1", entity_id))
        for entity_id in ("e1", "e2")
    }
    backend = _ReferenceEditBackend()

    stats = reference_edit_clips(
        config,
        storage,
        backend=backend,
        judge=_ReferenceEditJudge(),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    clip = storage.read_clip("clip-1")
    assert stats.entities_eligible == 2
    assert stats.entities_accepted == 2
    assert stats.worker_starts == 1
    assert backend.start_calls == 1
    assert backend.close_calls == 1
    assert len(backend.calls) == 2
    assert clip.reference_edit is not None
    assert clip.reference_edit.status == "ready"
    for reference in clip.references.entities:
        assert reference.status == "ready"
        assert reference.synthetic is True
        assert reference.image_path is not None
        assert reference.image_path.endswith("/final_reference_1k.png")
        assert (
            _sha256(storage.selected_entity_path("clip-1", reference.entity_id))
            == (canonical_hashes[reference.entity_id])
        )
        validate_entity_reference_artifact(
            config,
            storage,
            "clip-1",
            next(
                entity
                for entity in clip.annotation.entities
                if entity.entity_id == reference.entity_id
            ),
            reference,
            storage.read_frames("clip-1"),
            storage.read_masks("clip-1"),
        )

    body = "Keep {{image_1}} and {{image_2}} visually consistent."
    legend = [
        InstructionLegendEntry(image_id="image_1", description="First entity."),
        InstructionLegendEntry(image_id="image_2", description="Second entity."),
    ]
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    dataset = DatasetExporter(config, storage).export()
    assert dataset.sample_count == 1
    sample = json.loads(
        (config.resolved_export_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    for exported, reference in zip(sample["references"], clip.references.entities):
        assert (config.resolved_export_root / exported["image_path"]).read_bytes() == (
            (storage.root / reference.image_path).read_bytes()
        )
    unused_backend = _ReferenceEditBackend()
    skipped = reference_edit_clips(
        config,
        storage,
        backend=unused_backend,
        judge=_ReferenceEditJudge(),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )
    assert skipped.skipped_existing == 1
    assert skipped.worker_starts == 0
    assert unused_backend.start_calls == 0


def test_reference_edit_stage_is_between_pair_and_instruct() -> None:
    pair_index = STAGE_ORDER.index("pair")
    assert STAGE_ORDER[pair_index : pair_index + 3] == (
        "pair",
        "reference_edit",
        "instruct",
    )


def test_pipeline_runs_formal_reference_edit_stage_with_injected_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    config_path = tmp_path / "reference-edit.yaml"
    assert config.qwen.candidate_judge is not None
    assert config.qwen.background_remove_judge is not None
    assert config.qwen.reference_edit_judge is not None
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
                "  reference_edit_judge:",
                f"    model: {config.qwen.reference_edit_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
                "reference_edit:",
                "  enabled: true",
                (
                    "  python_executable: "
                    f"{config.reference_edit.python_executable}"
                ),
                f"  code_root: {config.reference_edit.code_root}",
                f"  model_path: {config.reference_edit.model_path}",
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("reference_edit",),
        git_commit="pair-test",
        reference_edit_backend=_ReferenceEditBackend(),
        reference_edit_judge=_ReferenceEditJudge(),
        reference_edit_sam_reviewer=_ReferenceEditSamReviewer(),
    )

    assert result["reference_edit"]["entities_accepted"] == 1
    assert result["completed_stages"] == ["reference_edit"]


def test_reference_edit_does_not_start_worker_without_eligible_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge({"e1": "local"}))
    backend = _ReferenceEditBackend()

    stats = reference_edit_clips(
        config,
        storage,
        backend=backend,
        judge=_ReferenceEditJudge(),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    clip = storage.read_clip("clip-1")
    assert stats.entities_eligible == 0
    assert stats.worker_starts == 0
    assert backend.start_calls == 0
    assert backend.close_calls == 0
    assert clip.reference_edit is not None
    assert clip.reference_edit.entities[0].status == "not_required"


def test_rejected_reference_edit_uses_explicit_keep_source_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    source_path = storage.selected_entity_path("clip-1", "e1")
    source_bytes = source_path.read_bytes()

    stats = reference_edit_clips(
        config,
        storage,
        backend=_ReferenceEditBackend(),
        judge=_ReferenceEditJudge(accept=False),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    clip = storage.read_clip("clip-1")
    reference = clip.references.entities[0]
    assert stats.entities_fallback == 1
    assert clip.pairing.status == "ready"
    assert clip.reference_edit.entities[0].status == "fallback"
    assert clip.reference_edit.entities[0].fallback_policy == "keep_source"
    assert reference.synthetic is False
    assert reference.image_path == storage.relative_artifact_path(source_path)
    assert source_path.read_bytes() == source_bytes


def test_reference_edit_overwrite_restores_immutable_source_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    source_path = storage.selected_entity_path("clip-1", "e1")
    source_bytes = source_path.read_bytes()
    reference_edit_clips(
        config,
        storage,
        backend=_ReferenceEditBackend(),
        judge=_ReferenceEditJudge(),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    stats = reference_edit_clips(
        config,
        storage,
        overwrite=True,
        backend=_ReferenceEditBackend(),
        judge=_ReferenceEditJudge(accept=False),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    clip = storage.read_clip("clip-1")
    reference = clip.references.entities[0]
    assert stats.entities_fallback == 1
    assert reference.synthetic is False
    assert reference.image_path == storage.relative_artifact_path(source_path)
    assert clip.reference_edit.entities[0].source_reference == reference
    assert source_path.read_bytes() == source_bytes
    assert not (
        storage.reference_edit_dir("clip-1") / "e1" / "final_reference_1k.png"
    ).exists()


def test_reference_edit_worker_failure_is_logged_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())

    stats = reference_edit_clips(
        config,
        storage,
        backend=_FailingReferenceEditBackend(),
        judge=_ReferenceEditJudge(),
        sam_reviewer=_ReferenceEditSamReviewer(),
    )

    clip = storage.read_clip("clip-1")
    assert stats.entities_failed == 1
    assert stats.entities_fallback == 1
    assert clip.references.entities[0].synthetic is False
    failures = [
        json.loads(line)
        for line in storage.failures_path.read_text(encoding="utf-8").splitlines()
    ]
    assert failures[-1]["stage"] == "reference_edit"
    assert failures[-1]["details"]["entity_id"] == "e1"
    assert failures[-1]["reason"].startswith("boogu_reference_edit_failed:")


def test_production_pair_bypasses_legacy_completion_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_edit_enabled=True,
    )
    storage = _storage(config, entity_types=("subject",))

    def fail_legacy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("legacy completion must not run")

    monkeypatch.setattr(pair_module, "run_reference_completion_fallbacks", fail_legacy)
    stats = pair_clips(config, storage, judge=_Judge({"e1": "local"}))

    assert stats.completion_attempted == 0
    assert storage.read_clip("clip-1").references.entities[0].completeness == (
        "local_usable"
    )


def test_sharpness_score_controls_reference_shortlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(max_candidates_per_entity=1),
    )
    storage = _storage(config, entity_types=("subject",))
    Image.new("RGB", (WIDTH, HEIGHT), (80, 80, 80)).save(
        storage.frame_path("clip-1", 5),
        format="JPEG",
        quality=100,
        subsampling=0,
    )
    checker = np.indices((HEIGHT, WIDTH)).sum(axis=0) % 2
    pixels = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    Image.fromarray(pixels, mode="RGB").save(
        storage.frame_path("clip-1", 4),
        format="JPEG",
        quality=100,
        subsampling=0,
    )
    clip = storage.read_clip("clip-1")
    candidates = build_entity_reference_candidates(
        config,
        storage,
        clip_uid="clip-1",
        entity=clip.annotation.entities[0],
        frames=storage.read_frames("clip-1"),
        masks=storage.read_masks("clip-1"),
    )

    assert len(candidates) == 1
    assert candidates[0].frame_slot == 4
    assert candidates[0].sharpness_score > 0
