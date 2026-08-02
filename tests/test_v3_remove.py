from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.remove as remove_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.config import (
    BackgroundConfig,
    DebugConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.remove import (
    build_generation_mask,
    composite_candidate,
    remove_backgrounds,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    BackgroundRemovalAttempt,
    BackgroundRemovalReview,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3

WIDTH = 6
HEIGHT = 5


def _accept(reason: str = "clean") -> BackgroundRemovalReview:
    return BackgroundRemovalReview(
        verdict="accept",
        foreground_absent=True,
        foreground_not_reconstructed=True,
        no_new_salient_entity=True,
        background_only_in_repaired_region=True,
        background_continuity_ok=True,
        no_visible_artifacts=True,
        reason=reason,
    )


def _reject(reason: str = "visible seam") -> BackgroundRemovalReview:
    return BackgroundRemovalReview(
        verdict="reject",
        foreground_absent=True,
        foreground_not_reconstructed=True,
        no_new_salient_entity=True,
        background_only_in_repaired_region=True,
        background_continuity_ok=True,
        no_visible_artifacts=False,
        reason=reason,
    )


class _Backend:
    def __init__(
        self,
        responses: list[Image.Image | Exception] | None = None,
    ) -> None:
        self.responses: Iterator[Image.Image | Exception] = iter(responses or [])
        self.seeds: list[int] = []
        self.phrases: list[list[str]] = []
        self.closed = False

    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        self.seeds.append(seed)
        self.phrases.append(list(removal_phrases))
        try:
            response = next(self.responses)
        except StopIteration:
            response = Image.new("RGB", image.size, (220, 30, 40))
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class _Judge:
    def __init__(
        self,
        responses: list[BackgroundRemovalReview | Exception] | None = None,
    ) -> None:
        self.responses: Iterator[BackgroundRemovalReview | Exception] = iter(
            responses or []
        )
        self.calls = 0
        self.closed = False

    def review(self, **kwargs: object) -> BackgroundRemovalReview:
        self.calls += 1
        try:
            response = next(self.responses)
        except StopIteration:
            response = _accept()
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    seeds: tuple[int, ...] = (0, 17),
    dilation: int = 0,
    max_generation_ratio: float = 1.0,
    save_rejected: bool = False,
    debug: bool = False,
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
    dataset = dataset_root / "source.jsonl"
    dataset.write_text("", encoding="utf-8")
    config = V3Config(
        dataset_json=dataset,
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / f"{run_name}-dataset",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            candidate_judge=QwenServiceConfig(
                model=str(pretrained / "Qwen" / "judge")
            ),
            background_remove_judge=QwenServiceConfig(
                model=str(pretrained / "Qwen" / "judge")
            ),
        ),
        background=BackgroundConfig(max_pending_remove_area_ratio=0.5),
        remove=RemoveConfig(
            enabled=enabled,
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "object-remover",
            candidate_seeds=seeds,
            generation_mask_dilation_pixels=dilation,
            max_generation_mask_area_ratio=max_generation_ratio,
            save_rejected_candidates=save_rejected,
        ),
        debug=DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _mask(y: int = 2, x: int = 2) -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=bool)
    value[y, x] = True
    return value


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _tracked_masks(clip_uid: str) -> TrackedMasksArtifact:
    mask = _mask()
    frames = [
        TrackedMaskFrame(
            slot=slot,
            present=True,
            track_valid=True,
            confidence=0.9,
            backend_confidences=[0.9],
            backend_object_ids=["obj-1"],
            area_pixels=1,
            area_ratio=float(mask.mean()),
            bbox_xyxy=_bbox(mask),
            rle=encode_binary_mask(mask),
        )
        for slot in range(10)
    ]
    return TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=WIDTH,
        height=HEIGHT,
        entities={
            "e1": TrackedEntityMasks(
                status="ready",
                reference_type="subject",
                grounding_prompt="person in red",
                backend_object_ids=["obj-1"],
                frames=frames,
            )
        },
    )


def _coverage() -> CoverageState:
    return CoverageState(
        passed=True,
        qualifying_entity_ids=["e1"],
        entity_visibility_summary={
            "e1": EntityVisibilitySummary(
                status="ready",
                visible_frame_slots=list(range(10)),
                visible_frame_count=10,
                coverage_ratio=1.0,
                qualifies=True,
                per_frame_area_ratio=[1 / (WIDTH * HEIGHT)] * 10,
                per_frame_confidence=[0.9] * 10,
            )
        },
    )


def _write_frames(storage: RunStorage, clip_uid: str) -> None:
    frames: list[SampledFrame] = []
    for slot in range(10):
        path = storage.frame_path(clip_uid, slot)
        Image.new("RGB", (WIDTH, HEIGHT), (10 + slot, 20, 30)).save(
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


def _pending_storage(config: V3Config, *, clip_uid: str = "clip-1") -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(config.dataset_json.parent / f"{clip_uid}.mp4"),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="A person crosses an open plaza.",
            entities=[
                AnnotationEntity(
                    entity_id="e1",
                    reference_type="subject",
                    phrase="person in red",
                    grounding_prompt="person in red",
                )
            ],
            background=BackgroundAnnotation(
                phrase="open plaza",
                grounding_prompt="open plaza",
            ),
        ),
    )
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=[
                EntityReferenceState(
                    entity_id="e1",
                    status="ready",
                    reference_scope="full",
                    visible_region="whole",
                    whole_entity_recognizable=True,
                    identity_features_visible=True,
                    scope_reason="complete",
                    image_path="selected/entity_e1.png",
                    source_frame_index=0,
                )
            ]
        ),
    )
    _write_frames(storage, clip_uid)
    storage.write_masks(clip_uid, _tracked_masks(clip_uid))
    storage.write_coverage(clip_uid, _coverage())
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=[
                EntityReferenceState(
                    entity_id="e1",
                    status="ready",
                    reference_scope="full",
                    visible_region="whole",
                    whole_entity_recognizable=True,
                    identity_features_visible=True,
                    scope_reason="complete",
                    image_path="selected/entity_e1.png",
                    source_frame_index=0,
                )
            ]
        ),
    )
    stats = build_background_candidates(config, storage)
    assert stats.pending_remove == 1
    return storage


def _background(
    storage: RunStorage,
    clip_uid: str = "clip-1",
) -> BackgroundReferenceState:
    state = storage.read_clip(clip_uid).references.background
    assert state is not None
    return state



def test_generation_mask_dilation_zero_is_exact_copy() -> None:
    source = _mask()
    result = build_generation_mask(source, dilation_pixels=0)
    assert np.array_equal(result, source)
    assert result is not source


def test_generation_mask_dilation_is_deterministic_and_contains_source() -> None:
    source = _mask()
    first = build_generation_mask(source, dilation_pixels=1)
    second = build_generation_mask(source, dilation_pixels=1)
    assert np.array_equal(first, second)
    assert np.all(first[source])
    assert int(first.sum()) == 9


def test_generation_mask_clips_at_edges_without_wraparound() -> None:
    result = build_generation_mask(_mask(0, 0), dilation_pixels=1)
    assert int(result.sum()) == 4
    assert result[0, 0]
    assert not result[-1, -1]


@pytest.mark.parametrize(
    ("source", "dilation", "error"),
    [
        (np.zeros((2, 2), dtype=np.uint8), 0, TypeError),
        (np.zeros((0, 2), dtype=bool), 0, ValueError),
        (np.zeros((2, 2), dtype=bool), 0, ValueError),
        (_mask(), -1, ValueError),
        (_mask(), True, ValueError),
    ],
)
def test_generation_mask_rejects_invalid_inputs(
    source: np.ndarray,
    dilation: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_generation_mask(source, dilation_pixels=dilation)


def test_composite_preserves_outside_and_uses_edited_inside() -> None:
    source = Image.new("RGB", (WIDTH, HEIGHT), (1, 2, 3))
    edited = Image.new("RGB", (WIDTH, HEIGHT), (9, 8, 7))
    mask = _mask()
    result = np.asarray(
        composite_candidate(
            source_image=source,
            edited_image=edited,
            generation_mask=mask,
        )
    )
    assert np.all(result[~mask] == (1, 2, 3))
    assert np.all(result[mask] == (9, 8, 7))


def test_composite_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        composite_candidate(
            source_image=Image.new("RGB", (3, 2)),
            edited_image=Image.new("RGB", (4, 2)),
            generation_mask=np.ones((2, 3), dtype=bool),
        )


def test_composite_rejects_unchanged_masked_region() -> None:
    source = Image.new("RGB", (WIDTH, HEIGHT), (1, 2, 3))
    with pytest.raises(ValueError, match="candidate_did_not_modify"):
        composite_candidate(
            source_image=source,
            edited_image=source.copy(),
            generation_mask=_mask(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {**_accept().model_dump(), "verdict": "reject"},
        {**_reject().model_dump(), "verdict": "accept"},
        {**_accept().model_dump(), "foreground_absent": False},
        {**_accept().model_dump(), "reason": " "},
        {**_accept().model_dump(), "extra": True},
    ],
)
def test_review_schema_rejects_contradictions_and_extra_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        BackgroundRemovalReview.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "seed": 0,
            "status": "accepted",
            "runtime_seconds": 1.0,
            "candidate_sha256": None,
            "reason": None,
            "review": _accept(),
        },
        {
            "seed": 0,
            "status": "rejected",
            "runtime_seconds": 1.0,
            "candidate_sha256": "a" * 64,
            "reason": None,
            "review": _reject(),
        },
        {
            "seed": 0,
            "status": "failed",
            "runtime_seconds": 1.0,
            "candidate_sha256": None,
            "reason": "failed",
            "review": _reject(),
        },
        {
            "seed": 0,
            "status": "failed",
            "runtime_seconds": float("inf"),
            "candidate_sha256": None,
            "reason": "failed",
            "review": None,
        },
    ],
)
def test_removal_attempt_schema_rejects_invalid_states(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        BackgroundRemovalAttempt.model_validate(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"candidate_seeds": ()}, "one or two"),
        ({"candidate_seeds": (0, 1, 2)}, "one or two"),
        ({"candidate_seeds": (0, 0)}, "unique"),
        ({"candidate_seeds": (-1,)}, "non-negative"),
        ({"candidate_seeds": (True,)}, "integers"),
        ({"device": ""}, "non-empty"),
        ({"dtype": "float64"}, "dtype"),
        ({"num_inference_steps": 0}, "positive"),
        ({"true_cfg_scale": float("inf")}, "finite"),
        ({"guidance_scale": -1.0}, "non-negative"),
        ({"generation_mask_dilation_pixels": True}, "non-negative integer"),
        ({"max_generation_mask_area_ratio": 0.0}, "greater than 0"),
        ({"max_generation_mask_area_ratio": 1}, "finite float"),
        ({"adapter_weight_name": ""}, "non-empty string"),
        ({"save_rejected_candidates": 1}, "boolean"),
    ],
)
def test_remove_config_validates_all_new_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(config, remove=replace(config.remove, **changes))
    with pytest.raises((TypeError, ValueError), match=message):
        changed.validate()


def test_config_fingerprint_and_identifiers_cover_new_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        remove=replace(config.remove, num_inference_steps=41),
    )
    assert changed.fingerprint() != config.fingerprint()
    assert config.model_identifiers()["remove.num_inference_steps"] == "40"
    assert changed.model_identifiers()["remove.num_inference_steps"] == "41"


def test_remove_yaml_parser_loads_new_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, run_name="yaml")
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  limit: 10",
                "qwen:",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
                "  adapter_weight_name: object.safetensors",
                "  candidate_seeds: [5]",
                "  dtype: float16",
                "  num_inference_steps: 33",
                "  generation_mask_dilation_pixels: 4",
                "  max_generation_mask_area_ratio: 0.7",
                "  save_rejected_candidates: true",
            ]
        ),
        encoding="utf-8",
    )
    loaded = config_module.load_config(path)
    assert loaded.remove.adapter_weight_name == "object.safetensors"
    assert loaded.remove.candidate_seeds == (5,)
    assert loaded.remove.dtype == "float16"
    assert loaded.remove.num_inference_steps == 33
    assert loaded.remove.generation_mask_dilation_pixels == 4
    assert loaded.remove.max_generation_mask_area_ratio == 0.7
    assert loaded.remove.save_rejected_candidates is True



def test_disabled_remove_does_not_call_injected_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, enabled=False)
    storage = _pending_storage(config)
    backend = _Backend()
    stats = remove_backgrounds(config, storage, backend=backend, judge=_Judge())
    assert stats.skipped_disabled == 1
    assert backend.seeds == []
    assert _background(storage).status == "pending_remove"


def test_first_seed_accepts_and_publishes_strict_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    backend = _Backend()
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge([_accept()]),
    )
    state = _background(storage)
    assert stats.processed == stats.ready_removed == 1
    assert stats.candidates_generated == 1
    assert backend.seeds == [0]
    assert state.status == "ready_removed"
    assert state.removal_seed == 0
    assert state.removal_backend == "qwen_image_edit_2511_object_remover"
    assert state.output_image_path == "clips/clip-1/selected/bg_removed.png"
    assert state.generation_mask_path is not None
    assert state.generation_mask_path.startswith(
        "clips/clip-1/background/generation_mask_"
    )
    assert state.output_image_path is not None
    output = storage.root / state.output_image_path
    assert hashlib.sha256(output.read_bytes()).hexdigest() == state.output_sha256
    accepted = [
        attempt
        for attempt in state.removal_attempts
        if attempt.status == "accepted"
    ]
    assert len(accepted) == 1
    assert storage.read_clip("clip-1").references.entities[0].entity_id == "e1"


def test_first_reject_second_accept_preserves_seed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    backend = _Backend()
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge([_reject("bad first"), _accept("good second")]),
    )
    state = _background(storage)
    assert backend.seeds == [0, 17]
    assert [attempt.status for attempt in state.removal_attempts] == [
        "rejected",
        "accepted",
    ]
    assert state.removal_seed == 17
    assert stats.candidates_rejected == 1


def test_first_failure_second_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    backend = _Backend([RuntimeError("generation failed")])
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge([_accept()]),
    )
    assert backend.seeds == [0, 17]
    assert [attempt.status for attempt in _background(storage).removal_attempts] == [
        "failed",
        "accepted",
    ]
    assert stats.candidates_failed == 1


@pytest.mark.parametrize(
    ("judge_responses", "backend_responses", "expected_reason"),
    [
        (
            [_reject("one"), _reject("two")],
            None,
            "all_removal_candidates_rejected",
        ),
        (
            None,
            [RuntimeError("one"), RuntimeError("two")],
            "all_removal_candidates_failed",
        ),
        (
            [_reject("one")],
            [
                Image.new("RGB", (WIDTH, HEIGHT), (200, 1, 1)),
                RuntimeError("two"),
            ],
            "no_acceptable_removal_candidate",
        ),
    ],
)
def test_no_accepted_candidate_is_controlled_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    judge_responses: list[BackgroundRemovalReview] | None,
    backend_responses: list[Image.Image | Exception] | None,
    expected_reason: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(backend_responses),
        judge=_Judge(judge_responses),
    )
    state = _background(storage)
    assert stats.processed == stats.rejected == 1
    assert stats.failed == 0
    assert state.status == "rejected"
    assert state.reason == expected_reason
    assert state.output_image_path is None
    assert state.generation_mask_path is None
    assert state.output_sha256 is None
    assert all(
        attempt.status != "accepted"
        for attempt in state.removal_attempts
    )
    assert storage.read_clip("clip-1").references.entities[0].entity_id == "e1"


def test_generation_mask_too_large_rejects_before_backend_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        dilation=2,
        max_generation_ratio=0.2,
    )
    storage = _pending_storage(config)
    backend = _Backend()
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge(),
    )
    assert stats.rejected == 1
    assert backend.seeds == []
    assert _background(storage).reason == "generation_mask_too_large"


def test_injected_backend_does_not_require_real_adapter_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    assert not config.remove.base_model_path.exists()
    assert config.remove.adapter_path is not None
    assert not config.remove.adapter_path.exists()
    storage = _pending_storage(config)
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge(),
    )
    assert stats.ready_removed == 1


def test_missing_background_remove_judge_fails_config_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    invalid = replace(
        config,
        qwen=replace(config.qwen, background_remove_judge=None),
    )

    with pytest.raises(
        ValueError,
        match=(
            "qwen.background_remove_judge is required when "
            "remove.enabled is true"
        ),
    ):
        invalid.validate()


def test_judge_failure_is_current_candidate_failure_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge([ValueError("malformed review"), _accept()]),
    )
    assert stats.candidates_failed == 1
    assert _background(storage).status == "ready_removed"


def test_source_mask_union_mismatch_is_clip_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    state = _background(storage)
    assert state.source_mask_path is not None
    mask_path = storage.root / state.source_mask_path
    Image.fromarray(
        np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255
    ).save(mask_path)
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge(),
    )
    assert stats.failed == 1
    assert _background(storage).status == "pending_remove"


def test_rejected_candidate_debug_saving_is_disabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge([_reject(), _reject()]),
    )
    assert not (storage.clip_dir("clip-1") / "debug" / "remove").exists()


@pytest.mark.parametrize(("save_rejected", "debug"), [(True, False), (False, True)])
def test_rejected_candidate_debug_saving_is_explicitly_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_rejected: bool,
    debug: bool,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        save_rejected=save_rejected,
        debug=debug,
    )
    storage = _pending_storage(config)
    remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge([_reject(), _reject()]),
    )
    directory = storage.clip_dir("clip-1") / "debug" / "remove"
    assert (directory / "candidate_seed_0.png").is_file()
    assert (directory / "review_seed_0.json").is_file()



def test_existing_valid_ready_removed_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())
    backend = _Backend()
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge(),
    )
    assert stats.skipped_existing == 1
    assert backend.seeds == []


def test_existing_corrupt_ready_removed_fails_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())
    state = _background(storage)
    assert state.output_image_path is not None
    (storage.root / state.output_image_path).write_bytes(b"corrupt")
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge(),
    )
    assert stats.failed == 1
    assert _background(storage) == state


def test_overwrite_replaces_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())
    old = _background(storage).output_sha256
    stats = remove_backgrounds(
        config,
        storage,
        overwrite=True,
        backend=_Backend(
            [Image.new("RGB", (WIDTH, HEIGHT), (1, 240, 2))]
        ),
        judge=_Judge(),
    )
    assert stats.ready_removed == 1
    assert _background(storage).output_sha256 != old


def test_overwrite_publication_failure_restores_old_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())
    before = _background(storage)
    assert before.output_image_path is not None
    output = storage.root / before.output_image_path
    old_bytes = output.read_bytes()

    def fail_write(*args: object, **kwargs: object) -> object:
        raise RuntimeError("clip write failed")

    monkeypatch.setattr(storage, "write_references", fail_write)
    stats = remove_backgrounds(
        config,
        storage,
        overwrite=True,
        backend=_Backend(
            [Image.new("RGB", (WIDTH, HEIGHT), (1, 240, 2))]
        ),
        judge=_Judge(),
    )
    assert stats.failed == 1
    assert output.read_bytes() == old_bytes
    assert storage.read_clip("clip-1").references.background == before


def test_injected_backend_and_judge_are_not_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    backend = _Backend()
    judge = _Judge()
    remove_backgrounds(config, storage, backend=backend, judge=judge)
    assert backend.closed is False
    assert judge.closed is False


def test_owned_backend_and_judge_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    backend = _Backend()
    judge = _Judge()
    monkeypatch.setattr(
        remove_module,
        "QwenImageEditRemovalBackend",
        lambda config: backend,
    )
    monkeypatch.setattr(
        remove_module,
        "QwenBackgroundRemovalJudge",
        lambda config: judge,
    )
    service = config_module.QwenServiceConfig(model="local")
    config = replace(
        config,
        qwen=replace(config.qwen, background_remove_judge=service),
    )
    storage.config = config
    remove_backgrounds(config, storage)
    assert backend.closed is True
    assert judge.closed is True


def test_non_pending_state_skips_without_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    clip = storage.read_clip("clip-1")
    storage.write_references(
        "clip-1",
        ReferencesState(
            entities=list(clip.references.entities),
            background=BackgroundReferenceState(
                status="rejected",
                reason="earlier rejection",
            ),
        ),
    )
    backend = _Backend()
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=_Judge(),
    )
    assert stats.skipped_not_pending == 1
    assert backend.seeds == []


def test_ready_validation_detects_output_outside_mask_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())
    state = _background(storage)
    assert state.output_image_path is not None
    output = storage.root / state.output_image_path
    with Image.open(output) as opened:
        pixels = np.asarray(opened.convert("RGB")).copy()
    pixels[0, 0] = (255, 255, 255)
    Image.fromarray(pixels).save(output, format="PNG")
    state = state.model_copy(
        update={"output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    )
    state = state.model_copy(
        update={
            "removal_attempts": [
                state.removal_attempts[-1].model_copy(
                    update={"candidate_sha256": state.output_sha256}
                )
            ]
        }
    )
    clip = storage.read_clip("clip-1")
    storage.write_references(
        "clip-1",
        ReferencesState(
            entities=list(clip.references.entities),
            background=state,
        ),
    )
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge(),
    )
    assert stats.failed == 1


def test_pipeline_executes_remove_and_pair_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, run_name="pipeline")
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  limit: 10",
                "qwen:",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
                "  generation_mask_dilation_pixels: 0",
                "  max_generation_mask_area_ratio: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    result = run_pipeline_v3(
        config_path=config_path,
        stages=("remove",),
        git_commit="abc123",
        background_removal_backend=_Backend(),
        background_removal_judge=_Judge(),
    )
    assert result["remove"]["processed"] == 0
    pair_result = run_pipeline_v3(
        config_path=config_path,
        stages=("pair",),
        git_commit="abc123",
    )
    assert pair_result["pair"]["processed"] == 0
