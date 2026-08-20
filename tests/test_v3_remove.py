from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.boogu_remove_backend as boogu_remove_module
import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.remove as remove_module
from r2v_data_v2.v3.boogu_remove_backend import (
    BooguBackgroundRemovalBackend,
    build_boogu_background_removal_prompt,
    create_boogu_background_removal_backend,
)
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.config import (
    BOOGU_REMOVE_BACKEND,
    BackgroundConfig,
    DebugConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput
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
        self.candidate_modes: list[object] = []
        self.closed = False

    def review(self, **kwargs: object) -> BackgroundRemovalReview:
        self.calls += 1
        self.candidate_modes.append(kwargs.get("candidate_mode"))
        try:
            response = next(self.responses)
        except StopIteration:
            response = _accept()
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class _BooguWorker:
    def __init__(self, color: tuple[int, int, int] = (220, 30, 40)) -> None:
        self.color = color
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def edit(self, **kwargs: object) -> BooguEditOutput:
        self.calls.append(dict(kwargs))
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        output = BytesIO()
        Image.new("RGB", (width, height), self.color).save(output, format="PNG")
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=output.getvalue(),
            original_instruction=instruction,
            rewritten_instruction=None,
            effective_instruction=instruction,
            worker_metadata={"seed": kwargs.get("seed")},
        )

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


def _boogu_config(config: V3Config, *, target_area: int = 32 * 32) -> V3Config:
    writable = config.run_root.parents[1]
    changed = replace(
        config,
        remove=replace(
            config.remove,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
            adapter_path=None,
        ),
        reference_edit=replace(
            config.reference_edit,
            python_executable=writable / "venvs" / "boogu" / "python",
            code_root=writable / "vendor" / "Boogu-Image",
            model_path=writable / "models" / "Boogu-Image",
            target_area=target_area,
        ),
    )
    changed.validate()
    return changed


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


def test_boogu_remove_prompt_is_short_and_uses_only_entity_phrases() -> None:
    prompt = build_boogu_background_removal_prompt(
        ["person in red", "small brown dog"]
    )

    assert prompt == (
        "Remove the following foreground entities from the image: person in red, "
        "small brown dog.\n"
        "Fill the removed areas naturally with the surrounding background."
    )
    assert "open plaza" not in prompt
    assert "mask" not in prompt.casefold()


def test_boogu_remove_adapter_forwards_seed_disables_thinking_and_resizes_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _BooguWorker()
    backend = BooguBackgroundRemovalBackend(
        worker,  # type: ignore[arg-type]
        target_area=32 * 32,
        alignment=16,
    )
    source = Image.new("RGB", (63, 35), (10, 20, 30))
    original_bytes = source.tobytes()
    prompt = build_boogu_background_removal_prompt(["person", "dog"])
    resize_calls: list[tuple[tuple[int, int], object]] = []
    original_resize = Image.Image.resize

    def tracked_resize(
        image: Image.Image,
        size: tuple[int, int],
        resample: object = None,
        box: object = None,
        reducing_gap: object = None,
    ) -> Image.Image:
        resize_calls.append((size, resample))
        return original_resize(
            image,
            size,
            resample=resample,
            box=box,
            reducing_gap=reducing_gap,
        )

    monkeypatch.setattr(Image.Image, "resize", tracked_resize)

    result = backend.remove(
        image=source,
        removal_phrases=["person", "dog"],
        background_phrase="open plaza",
        prompt=prompt,
        seed=17,
    )

    assert result.size == source.size
    assert result.mode == "RGB"
    assert source.size == (63, 35)
    assert source.tobytes() == original_bytes
    assert len(worker.calls) == 1
    call = worker.calls[0]
    assert call["thinking_enabled"] is False
    assert call["seed"] == 17
    assert call["instruction"] == prompt
    assert (call["width"], call["height"]) != source.size
    assert (call["width"], call["height"]) == backend.generation_size(source)
    assert resize_calls == [(source.size, Image.Resampling.LANCZOS)]


def test_boogu_remove_worker_routes_one_physical_gpu_as_local_cuda_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))
    storage = _pending_storage(config)
    created: list[object] = []

    class FakeSubprocessBackend:
        def __init__(self, worker_config: object) -> None:
            self.config = worker_config
            self.started_with: Path | None = None
            self.closed = False
            created.append(self)

        def start(self, *, stderr_log_path: Path) -> None:
            self.started_with = stderr_log_path

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        boogu_remove_module,
        "BooguSubprocessBackend",
        FakeSubprocessBackend,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    backend = create_boogu_background_removal_backend(config, storage)

    assert len(created) == 1
    worker = created[0]
    assert isinstance(worker, FakeSubprocessBackend)
    assert worker.config.cuda_visible_devices == "4"
    assert worker.config.device == "cuda:0"
    assert worker.started_with == storage.boogu_remove_worker_log_path()
    backend.close()
    assert worker.closed is True

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    with pytest.raises(ValueError, match="one visible physical GPU"):
        create_boogu_background_removal_backend(config, storage)


def test_boogu_remove_uses_authoritative_full_frame_candidate_without_composite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch), target_area=32 * 32)
    storage = _pending_storage(config)
    worker = _BooguWorker(color=(240, 1, 2))
    backend = BooguBackgroundRemovalBackend(
        worker,  # type: ignore[arg-type]
        target_area=config.reference_edit.target_area,
        alignment=config.reference_edit.alignment,
    )

    class CapturingJudge(_Judge):
        def __init__(self) -> None:
            super().__init__([_reject(), _accept()])
            self.candidates: list[Image.Image] = []
            self.sources: list[Image.Image] = []

        def review(self, **kwargs: object) -> BackgroundRemovalReview:
            candidate = kwargs["candidate_image"]
            assert isinstance(candidate, Image.Image)
            self.candidates.append(candidate.copy())
            source = kwargs["source_image"]
            assert isinstance(source, Image.Image)
            self.sources.append(source.copy())
            return super().review(**kwargs)

    judge = CapturingJudge()

    def reject_composite(**kwargs: object) -> Image.Image:
        del kwargs
        raise AssertionError("Boogu full-frame mode must not composite")

    monkeypatch.setattr(remove_module, "composite_candidate", reject_composite)
    stats = remove_backgrounds(config, storage, backend=backend, judge=judge)

    assert stats.ready_removed == 1
    assert stats.candidates_generated == 2
    assert [call["seed"] for call in worker.calls] == [0, 17]
    assert all(call["thinking_enabled"] is False for call in worker.calls)
    assert len(judge.candidates) == 2
    assert judge.candidate_modes == ["full_frame", "full_frame"]
    state = _background(storage)
    assert state.output_image_path is not None
    assert state.source_mask_path is not None
    assert state.generation_mask_path is not None
    assert state.removal_seed == 17
    assert state.output_sha256 == hashlib.sha256(
        (storage.root / state.output_image_path).read_bytes()
    ).hexdigest()
    with Image.open(storage.root / state.output_image_path) as loaded:
        published = loaded.convert("RGB")
    assert np.all(np.asarray(published) == np.array([240, 1, 2]))


def test_boogu_remove_profiling_uses_dedicated_component_and_safe_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch, seeds=(23,)))
    storage = _pending_storage(config)
    worker = _BooguWorker()
    backend = BooguBackgroundRemovalBackend(
        worker,  # type: ignore[arg-type]
        target_area=config.reference_edit.target_area,
        alignment=config.reference_edit.alignment,
    )
    profiler = V3Profiler(storage.root, git_commit="abc123")

    with active_profiler(profiler):
        stats = remove_backgrounds(config, storage, backend=backend, judge=_Judge())

    assert stats.ready_removed == 1
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event = next(item for item in events if item["kind"] == "model_call")
    assert event["component"] == "boogu_background_remove"
    assert event["operation"] == "background_remove"
    assert event["metadata"] == {
        "clip_uid": "clip-1",
        "generation_height": worker.calls[0]["height"],
        "generation_width": worker.calls[0]["width"],
        "seed": 23,
        "source_height": HEIGHT,
        "source_width": WIDTH,
        "thinking_enabled": False,
    }


def test_legacy_remove_profiling_keeps_object_remover_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, seeds=(0,))
    storage = _pending_storage(config)
    profiler = V3Profiler(storage.root, git_commit="abc123")

    with active_profiler(profiler):
        remove_backgrounds(config, storage, backend=_Backend(), judge=_Judge())

    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    event = next(item for item in events if item["kind"] == "model_call")
    assert event["component"] == "qwen_image_edit_object_remover"


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
        ({"inference_profile": "fast"}, "inference_profile"),
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
        remove=replace(
            config.remove,
            inference_profile="experimental_override",
            num_inference_steps=41,
        ),
    )
    assert changed.fingerprint() != config.fingerprint()
    assert config.model_identifiers()["remove.num_inference_steps"] == "4"
    assert (
        config.model_identifiers()["remove.inference_profile"]
        == "object_remover_4step_v1"
    )
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
                "  inference_profile: experimental_override",
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
    assert loaded.remove.inference_profile == "experimental_override"
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
    judge = _Judge([_accept()])
    composite_calls = 0
    original_composite = remove_module.composite_candidate

    def tracked_composite(**kwargs: object) -> Image.Image:
        nonlocal composite_calls
        composite_calls += 1
        return original_composite(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(remove_module, "composite_candidate", tracked_composite)
    stats = remove_backgrounds(
        config,
        storage,
        backend=backend,
        judge=judge,
    )
    state = _background(storage)
    assert stats.processed == stats.ready_removed == 1
    assert stats.candidates_generated == 1
    assert backend.seeds == [0]
    assert state.status == "ready_removed"
    assert state.removal_seed == 0
    assert state.removal_backend == "qwen_image_edit_2511_object_remover"
    assert judge.candidate_modes == ["masked_local"]
    assert composite_calls == 2
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


@pytest.mark.parametrize(
    ("wrong_size", "wrong_hash", "mode", "error"),
    [
        (True, False, "RGB", "dimensions are invalid"),
        (False, False, "RGBA", "must be RGB"),
        (False, True, "RGB", "hash does not match candidate bytes"),
    ],
)
def test_publish_ready_full_frame_fails_closed_on_dimensions_or_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_size: bool,
    wrong_hash: bool,
    mode: str,
    error: str,
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))
    storage = _pending_storage(config)
    state = _background(storage)
    inputs = remove_module._prepare_inputs(config, storage, "clip-1", state)
    size = (2, 2) if wrong_size else inputs.source_image.size
    color = (7, 8, 9, 255) if mode == "RGBA" else (7, 8, 9)
    candidate = Image.new(mode, size, color)
    candidate_bytes = remove_module._png_bytes(candidate)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if wrong_hash:
        candidate_sha = "0" * 64

    with pytest.raises(ValueError, match=error):
        remove_module._publish_ready(
            config,
            storage,
            clip_uid="clip-1",
            original=state,
            inputs=inputs,
            candidate=candidate,
            candidate_bytes=candidate_bytes,
            candidate_sha256=candidate_sha,
            seed=0,
            attempts=[],
            candidate_mode="full_frame",
        )

    assert _background(storage).status == "pending_remove"
    assert not storage.selected_background_output_path("clip-1").exists()


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


def test_all_quality_rejected_candidates_are_permanent_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge([_reject("one"), _reject("two")]),
    )
    state = _background(storage)
    assert stats.processed == stats.rejected == 1
    assert stats.failed == 0
    assert state.status == "rejected"
    assert state.reason == "all_removal_candidates_rejected"
    assert state.output_image_path is None
    assert state.generation_mask_path is None
    assert state.output_sha256 is None
    assert all(
        attempt.status != "accepted"
        for attempt in state.removal_attempts
    )
    assert storage.read_clip("clip-1").references.entities[0].entity_id == "e1"


@pytest.mark.parametrize(
    ("judge_responses", "backend_responses"),
    [
        (None, [RuntimeError("one"), RuntimeError("two")]),
        (
            [_reject("one")],
            [
                Image.new("RGB", (WIDTH, HEIGHT), (200, 1, 1)),
                RuntimeError("two"),
            ],
        ),
    ],
)
def test_backend_failure_keeps_background_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    judge_responses: list[BackgroundRemovalReview] | None,
    backend_responses: list[Image.Image | Exception],
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
    assert stats.processed == 1
    assert stats.retryable_pending == 1
    assert stats.rejected == 0
    assert state.status == "pending_remove"
    assert state.reason == "removal_infrastructure_failure"
    assert any(attempt.status == "failed" for attempt in state.removal_attempts)


def test_retryable_pending_can_be_retried_without_background_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    remove_backgrounds(
        config,
        storage,
        backend=_Backend([RuntimeError("cuda failed"), RuntimeError("cuda failed")]),
        judge=_Judge(),
    )

    stats = remove_backgrounds(
        config,
        storage,
        backend=_Backend(),
        judge=_Judge([_accept()]),
    )

    assert stats.ready_removed == 1
    assert _background(storage).status == "ready_removed"


def test_default_remove_profile_is_validated_four_step_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    assert config.remove.inference_profile == "object_remover_4step_v1"
    assert config.remove.num_inference_steps == 4
    with pytest.raises(ValueError, match="requires num_inference_steps=4"):
        replace(
            config,
            remove=replace(config.remove, num_inference_steps=40),
        ).validate()
    replace(
        config,
        remove=replace(
            config.remove,
            inference_profile="experimental_override",
            num_inference_steps=40,
        ),
    ).validate()


def test_boogu_remove_profile_is_strict_four_step_without_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))

    assert config.remove.backend == BOOGU_REMOVE_BACKEND
    assert config.remove.adapter_path is None
    assert config.remove.num_inference_steps == 4
    with pytest.raises(ValueError, match="requires num_inference_steps=4"):
        replace(
            config,
            remove=replace(config.remove, num_inference_steps=5),
        ).validate()


@pytest.mark.parametrize(
    ("backend", "profile"),
    [
        (BOOGU_REMOVE_BACKEND, "object_remover_4step_v1"),
        (config_module.REMOVE_BACKEND, "boogu_4step_v1"),
    ],
)
def test_remove_backend_and_profile_must_not_contradict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    profile: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    if backend == BOOGU_REMOVE_BACKEND:
        config = _boogu_config(config)

    with pytest.raises(ValueError, match="inference_profile"):
        replace(
            config,
            remove=replace(
                config.remove,
                backend=backend,
                inference_profile=profile,
            ),
        ).validate()


def test_enabled_remove_requires_object_remover_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="adapter_path is required"):
        replace(
            config,
            remove=replace(config.remove, adapter_path=None),
        ).validate()


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
    assert (directory / "source.png").is_file()
    assert (directory / "source_mask.png").is_file()
    assert (directory / "generation_mask.png").is_file()



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
