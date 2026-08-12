# ruff: noqa: BLE001
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import (
    build_union_foreground_mask,
    validate_background_inputs,
    validate_background_reference,
)
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.qwen_image_edit_backend import (
    BackgroundRemovalBackend,
    QwenImageEditRemovalBackend,
    build_background_removal_prompt,
)
from r2v_data_v2.v3.removal_judge import (
    BackgroundRemovalJudge,
    QwenBackgroundRemovalJudge,
)
from r2v_data_v2.v3.schemas import (
    BackgroundReferenceState,
    BackgroundRemovalAttempt,
    BackgroundRemovalReview,
    ReferencesState,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class RemoveStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_pending: int = 0
    skipped_disabled: int = 0
    failed: int = 0
    ready_removed: int = 0
    rejected: int = 0
    retryable_pending: int = 0
    candidates_generated: int = 0
    candidates_rejected: int = 0
    candidates_failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _RemovalInputs:
    frames: SampledFramesArtifact
    masks: TrackedMasksArtifact
    source_image: Image.Image
    source_mask: np.ndarray
    generation_mask: np.ndarray
    removal_phrases: list[str]
    background_phrase: str


def build_generation_mask(
    source_mask: np.ndarray,
    *,
    dilation_pixels: int,
) -> np.ndarray:
    if not isinstance(source_mask, np.ndarray):
        raise TypeError("source_mask must be a numpy array")
    if source_mask.ndim != 2 or source_mask.size == 0:
        raise ValueError("source_mask must be a non-empty two-dimensional mask")
    if source_mask.dtype != np.bool_:
        raise TypeError("source_mask must use boolean dtype")
    if not np.any(source_mask):
        raise ValueError("source_mask must contain foreground pixels")
    if (
        not isinstance(dilation_pixels, int)
        or isinstance(dilation_pixels, bool)
        or dilation_pixels < 0
    ):
        raise ValueError("dilation_pixels must be a non-negative integer")
    if dilation_pixels == 0:
        return source_mask.copy()

    radius = dilation_pixels
    padded = np.pad(
        source_mask.astype(np.uint8),
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=0,
    )
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    window_sums = (
        integral[diameter:, diameter:]
        - integral[:-diameter, diameter:]
        - integral[diameter:, :-diameter]
        + integral[:-diameter, :-diameter]
    )
    result = window_sums > 0
    if result.shape != source_mask.shape or not np.all(result[source_mask]):
        raise RuntimeError("generation mask dilation violated its contract")
    return result


def composite_candidate(
    *,
    source_image: Image.Image,
    edited_image: Image.Image,
    generation_mask: np.ndarray,
) -> Image.Image:
    if not isinstance(source_image, Image.Image) or not isinstance(
        edited_image, Image.Image
    ):
        raise TypeError("source_image and edited_image must be PIL images")
    if source_image.size != edited_image.size:
        raise ValueError("edited image dimensions do not match source image")
    if (
        not isinstance(generation_mask, np.ndarray)
        or generation_mask.ndim != 2
        or generation_mask.dtype != np.bool_
    ):
        raise TypeError("generation_mask must be a two-dimensional boolean array")
    width, height = source_image.size
    if generation_mask.shape != (height, width):
        raise ValueError("generation mask dimensions do not match source image")
    source = np.asarray(source_image.convert("RGB"))
    edited = np.asarray(edited_image.convert("RGB"))
    changed_inside = np.any(source[generation_mask] != edited[generation_mask])
    if not changed_inside:
        raise ValueError("candidate_did_not_modify_masked_region")
    composite = source.copy()
    composite[generation_mask] = edited[generation_mask]
    if not np.array_equal(
        composite[~generation_mask],
        source[~generation_mask],
    ):
        raise RuntimeError("candidate modified pixels outside generation mask")
    return Image.fromarray(composite, mode="RGB")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _mask_png_bytes(mask: np.ndarray) -> bytes:
    return _png_bytes(Image.fromarray(mask.astype(np.uint8) * 255, mode="L"))


def _write_bytes_atomic(path: Path, value: bytes, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{prefix}-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_run_path(storage: RunStorage, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        raise ValueError("V3 remove artifact paths must be relative to run_root")
    resolved = (storage.root / path).resolve(strict=False)
    try:
        resolved.relative_to(storage.root.resolve())
    except ValueError as exc:
        raise ValueError("V3 remove artifact must remain inside run_root") from exc
    return resolved


def _resolve_source_image_path(
    storage: RunStorage,
    clip_uid: str,
    value: str,
) -> Path:
    run_relative = _resolve_run_path(storage, value)
    if run_relative.is_file():
        return run_relative
    clip_relative = (storage.clip_dir(clip_uid) / value).resolve(strict=False)
    try:
        clip_relative.relative_to(storage.root.resolve())
    except ValueError as exc:
        raise ValueError("background source image must remain inside run_root") from exc
    if not clip_relative.is_file():
        raise FileNotFoundError(
            f"background source image is missing: {value}"
        )
    return clip_relative


def _read_source_image(
    storage: RunStorage,
    clip_uid: str,
    state: BackgroundReferenceState,
) -> Image.Image:
    if state.source_image_path is None:
        raise ValueError("background source image path is missing")
    path = _resolve_source_image_path(
        storage,
        clip_uid,
        state.source_image_path,
    )
    with Image.open(path) as opened:
        opened.load()
        return opened.convert("RGB")


def _read_binary_mask(
    storage: RunStorage,
    value: str,
    *,
    expected_size: tuple[int, int],
) -> np.ndarray:
    path = _resolve_run_path(storage, value)
    if not path.is_file():
        raise FileNotFoundError(f"background mask is missing: {value}")
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("background mask must be a PNG")
        if opened.size != expected_size:
            raise ValueError("background mask dimensions do not match source image")
        pixels = np.asarray(opened.convert("L"))
    if pixels.ndim != 2 or not np.isin(pixels, (0, 255)).all():
        raise ValueError("background mask must contain only 0 and 255")
    return pixels == 255


def _pending_view(state: BackgroundReferenceState) -> BackgroundReferenceState:
    if state.status == "pending_remove":
        return state
    return BackgroundReferenceState(
        status="pending_remove",
        source_image_path=state.source_image_path,
        source_frame_slot=state.source_frame_slot,
        source_frame_index=state.source_frame_index,
        source_mask_path=state.source_mask_path,
        source_foreground_area_pixels=state.source_foreground_area_pixels,
        source_foreground_area_ratio=state.source_foreground_area_ratio,
    )


def _prepare_inputs(
    config: V3Config,
    storage: RunStorage,
    clip_uid: str,
    state: BackgroundReferenceState,
) -> _RemovalInputs:
    clip = storage.read_clip(clip_uid)
    annotation = clip.annotation
    if annotation is None or annotation.status != "ready":
        raise ValueError("pending_remove background requires a ready annotation")
    if annotation.background is None:
        raise ValueError("pending_remove background is missing background annotation")

    pending = _pending_view(state)
    frames = validate_sampled_frames(storage, clip_uid)
    validate_background_reference(
        storage,
        clip_uid,
        pending,
        frames=frames,
    )
    masks = storage.read_masks(clip_uid)
    validate_background_inputs(
        clip_uid=clip_uid,
        annotation=annotation,
        frames=frames,
        masks=masks,
    )
    if pending.source_frame_slot is None or pending.source_mask_path is None:
        raise ValueError("pending_remove background source provenance is incomplete")
    slot = pending.source_frame_slot

    removal_phrases: list[str] = []
    for entity in annotation.entities:
        tracked = masks.entities[entity.entity_id]
        frame = tracked.frames[slot]
        if (
            tracked.status == "ready"
            and frame.track_valid
            and frame.present
            and frame.area_pixels > 0
        ):
            removal_phrases.append(entity.phrase)
    if not removal_phrases:
        raise ValueError("background source slot has no removable foreground phrase")

    union_mask = build_union_foreground_mask(masks, slot)
    if union_mask is None:
        raise ValueError("background source slot contains an invalid tracked mask")
    source_image = _read_source_image(storage, clip_uid, pending)
    source_mask = _read_binary_mask(
        storage,
        pending.source_mask_path,
        expected_size=source_image.size,
    )
    if not np.array_equal(union_mask, source_mask):
        raise ValueError(
            "background source mask does not match tracked foreground union"
        )
    generation_mask = build_generation_mask(
        source_mask,
        dilation_pixels=config.remove.generation_mask_dilation_pixels,
    )
    return _RemovalInputs(
        frames=frames,
        masks=masks,
        source_image=source_image,
        source_mask=source_mask,
        generation_mask=generation_mask,
        removal_phrases=removal_phrases,
        background_phrase=annotation.background.phrase,
    )


def _debug_enabled(config: V3Config) -> bool:
    return (
        config.debug.save_diagnostics
        or config.remove.save_rejected_candidates
    )


def _write_candidate_debug(
    storage: RunStorage,
    *,
    clip_uid: str,
    seed: int,
    candidate: Image.Image,
    review: BackgroundRemovalReview | None,
) -> None:
    directory = storage.remove_debug_dir(clip_uid)
    candidate_path = directory / f"candidate_seed_{seed}.png"
    _write_bytes_atomic(
        candidate_path,
        _png_bytes(candidate),
        prefix="tmp-remove-candidate",
    )
    if review is not None:
        write_json_atomic(
            directory / f"review_seed_{seed}.json",
            review.model_dump(mode="json"),
        )


def _publish_rejected(
    storage: RunStorage,
    *,
    clip_uid: str,
    original: BackgroundReferenceState,
    reason: str,
    removal_backend: str | None = None,
    attempts: list[BackgroundRemovalAttempt] | None = None,
) -> None:
    state = BackgroundReferenceState(
        status="rejected",
        source_image_path=original.source_image_path,
        source_frame_slot=original.source_frame_slot,
        source_frame_index=original.source_frame_index,
        source_mask_path=original.source_mask_path,
        source_foreground_area_pixels=original.source_foreground_area_pixels,
        source_foreground_area_ratio=original.source_foreground_area_ratio,
        removal_backend=removal_backend,
        removal_attempts=list(attempts or ()),
        reason=reason,
    )
    clip = storage.read_clip(clip_uid)
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=list(clip.references.entities),
            background=state,
        ),
    )
    (
        storage.clip_dir(clip_uid) / "selected" / "bg_removed.png"
    ).unlink(missing_ok=True)
    storage.cleanup_remove_artifacts(clip_uid)


def _publish_retryable(
    storage: RunStorage,
    *,
    clip_uid: str,
    original: BackgroundReferenceState,
    attempts: list[BackgroundRemovalAttempt],
    reason: str,
    removal_backend: str,
) -> None:
    state = BackgroundReferenceState(
        status="pending_remove",
        source_image_path=original.source_image_path,
        source_frame_slot=original.source_frame_slot,
        source_frame_index=original.source_frame_index,
        source_mask_path=original.source_mask_path,
        source_foreground_area_pixels=original.source_foreground_area_pixels,
        source_foreground_area_ratio=original.source_foreground_area_ratio,
        removal_backend=removal_backend,
        removal_attempts=attempts,
        reason=reason,
    )
    clip = storage.read_clip(clip_uid)
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=list(clip.references.entities),
            background=state,
        ),
    )
    storage.selected_background_output_path(clip_uid).unlink(missing_ok=True)
    storage.cleanup_remove_artifacts(clip_uid)


def _validate_output_png(
    path: Path,
    *,
    expected_size: tuple[int, int],
) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("removed background output must be a PNG")
        if opened.size != expected_size:
            raise ValueError("removed background output dimensions are invalid")
        return opened.convert("RGB")


def _publish_ready(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    original: BackgroundReferenceState,
    inputs: _RemovalInputs,
    candidate: Image.Image,
    candidate_bytes: bytes,
    candidate_sha256: str,
    seed: int,
    attempts: list[BackgroundRemovalAttempt],
) -> None:
    output = storage.selected_background_output_path(clip_uid)
    temporary_output = storage.remove_output_temporary_path(clip_uid)
    backup: Path | None = None
    generation_bytes = _mask_png_bytes(inputs.generation_mask)
    generation_sha = _sha256_bytes(generation_bytes)
    generation_path = storage.background_generation_mask_path(
        clip_uid,
        generation_sha,
    )
    created_generation = False
    try:
        with temporary_output.open("wb") as handle:
            handle.write(candidate_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        validated_candidate = _validate_output_png(
            temporary_output,
            expected_size=inputs.source_image.size,
        )
        recomposited = composite_candidate(
            source_image=inputs.source_image,
            edited_image=validated_candidate,
            generation_mask=inputs.generation_mask,
        )
        if _png_bytes(recomposited) != candidate_bytes:
            raise ValueError("published candidate failed exact local compositing")

        if generation_path.is_file():
            if generation_path.read_bytes() != generation_bytes:
                raise ValueError("generation mask content hash collision")
        else:
            temporary_mask = (
                storage.background_dir(clip_uid)
                / f".tmp-remove-{uuid.uuid4().hex}.png"
            )
            try:
                with temporary_mask.open("wb") as handle:
                    handle.write(generation_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary_mask.replace(generation_path)
                created_generation = True
            finally:
                temporary_mask.unlink(missing_ok=True)

        if output.is_file():
            backup = storage.remove_output_backup_path(clip_uid)
            output.replace(backup)
        temporary_output.replace(output)

        generation_pixels = int(np.count_nonzero(inputs.generation_mask))
        generation_ratio = generation_pixels / inputs.generation_mask.size
        state = BackgroundReferenceState(
            status="ready_removed",
            source_image_path=original.source_image_path,
            output_image_path=storage.relative_artifact_path(output),
            source_frame_slot=original.source_frame_slot,
            source_frame_index=original.source_frame_index,
            source_mask_path=original.source_mask_path,
            generation_mask_path=storage.relative_artifact_path(generation_path),
            source_foreground_area_pixels=original.source_foreground_area_pixels,
            source_foreground_area_ratio=original.source_foreground_area_ratio,
            removal_backend=config.remove.backend,
            removal_seed=seed,
            generation_mask_dilation_pixels=(
                config.remove.generation_mask_dilation_pixels
            ),
            generation_mask_area_pixels=generation_pixels,
            generation_mask_area_ratio=generation_ratio,
            output_sha256=candidate_sha256,
            removal_attempts=list(attempts),
            reason=None,
        )
        validate_background_reference(
            storage,
            clip_uid,
            state,
            frames=inputs.frames,
        )
        clip = storage.read_clip(clip_uid)
        storage.write_references(
            clip_uid,
            ReferencesState(
                entities=list(clip.references.entities),
                background=state,
            ),
        )
    except Exception:
        if output.is_file():
            output.unlink()
        if backup is not None and backup.is_file():
            backup.replace(output)
        if created_generation:
            generation_path.unlink(missing_ok=True)
        raise
    else:
        if backup is not None:
            backup.unlink(missing_ok=True)
        storage.cleanup_remove_artifacts(
            clip_uid,
            keep_generation_mask=generation_path,
        )
    finally:
        temporary_output.unlink(missing_ok=True)


def _exception_reason(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


def remove_backgrounds(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    backend: BackgroundRemovalBackend | None = None,
    judge: BackgroundRemovalJudge | None = None,
) -> RemoveStats:
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "skipped_not_pending": 0,
        "skipped_disabled": 0,
        "failed": 0,
        "ready_removed": 0,
        "rejected": 0,
        "retryable_pending": 0,
        "candidates_generated": 0,
        "candidates_rejected": 0,
        "candidates_failed": 0,
    }
    active_backend = backend
    active_judge = judge
    owned_backend = False
    owned_judge = False
    try:
        for listed_clip in storage.iter_clips():
            clip_uid = listed_clip.clip_uid
            state = listed_clip.references.background
            if state is None or state.status in {
                "none",
                "clean_raw",
                "rejected",
            }:
                counters["skipped_not_pending"] += 1
                continue
            if state.status == "ready_removed" and not overwrite:
                try:
                    validate_background_reference(storage, clip_uid, state)
                except Exception as exc:
                    storage.append_failure(
                        clip_uid=clip_uid,
                        stage="remove",
                        reason=_exception_reason(exc),
                    )
                    counters["failed"] += 1
                else:
                    counters["skipped_existing"] += 1
                continue
            if not config.remove.enabled:
                counters["skipped_disabled"] += 1
                continue

            try:
                inputs = _prepare_inputs(
                    config,
                    storage,
                    clip_uid,
                    state,
                )
                generation_pixels = int(
                    np.count_nonzero(inputs.generation_mask)
                )
                generation_ratio = generation_pixels / inputs.generation_mask.size
                if generation_ratio > config.remove.max_generation_mask_area_ratio:
                    _publish_rejected(
                        storage,
                        clip_uid=clip_uid,
                        original=state,
                        reason="generation_mask_too_large",
                    )
                    counters["processed"] += 1
                    counters["rejected"] += 1
                    continue

                if active_backend is None:
                    active_backend = QwenImageEditRemovalBackend(config.remove)
                    owned_backend = True
                if active_judge is None:
                    service = config.qwen.background_remove_judge
                    if service is None:
                        raise RuntimeError(
                            "background remove judge is not configured"
                        )
                    active_judge = QwenBackgroundRemovalJudge(service)
                    owned_judge = True

                prompt = build_background_removal_prompt(
                    removal_phrases=inputs.removal_phrases,
                    background_phrase=inputs.background_phrase,
                )
                source_mask_image = Image.fromarray(
                    inputs.source_mask.astype(np.uint8) * 255,
                    mode="L",
                )
                generation_mask_image = Image.fromarray(
                    inputs.generation_mask.astype(np.uint8) * 255,
                    mode="L",
                )
                attempts: list[BackgroundRemovalAttempt] = []
                accepted: tuple[Image.Image, bytes, str, int] | None = None
                for seed in config.remove.candidate_seeds:
                    started = time.monotonic()
                    candidate: Image.Image | None = None
                    candidate_sha: str | None = None
                    review: BackgroundRemovalReview | None = None
                    try:
                        edited = active_backend.remove(
                            image=inputs.source_image,
                            removal_phrases=inputs.removal_phrases,
                            background_phrase=inputs.background_phrase,
                            prompt=prompt,
                            seed=seed,
                        )
                        if not isinstance(edited, Image.Image):
                            raise TypeError(
                                "background removal backend did not return a PIL image"
                            )
                        counters["candidates_generated"] += 1
                        candidate = composite_candidate(
                            source_image=inputs.source_image,
                            edited_image=edited,
                            generation_mask=inputs.generation_mask,
                        )
                        candidate_bytes = _png_bytes(candidate)
                        candidate_sha = _sha256_bytes(candidate_bytes)
                        review = active_judge.review(
                            source_image=inputs.source_image,
                            candidate_image=candidate,
                            source_mask=source_mask_image,
                            generation_mask=generation_mask_image,
                            removal_phrases=inputs.removal_phrases,
                            background_phrase=inputs.background_phrase,
                        )
                        runtime = time.monotonic() - started
                        if review.verdict == "accept":
                            attempt = BackgroundRemovalAttempt(
                                seed=seed,
                                status="accepted",
                                runtime_seconds=runtime,
                                candidate_sha256=candidate_sha,
                                reason=None,
                                review=review,
                            )
                            attempts.append(attempt)
                            accepted = (
                                candidate,
                                candidate_bytes,
                                candidate_sha,
                                seed,
                            )
                            break
                        attempt = BackgroundRemovalAttempt(
                            seed=seed,
                            status="rejected",
                            runtime_seconds=runtime,
                            candidate_sha256=candidate_sha,
                            reason=review.reason,
                            review=review,
                        )
                        attempts.append(attempt)
                        counters["candidates_rejected"] += 1
                        if _debug_enabled(config):
                            _write_candidate_debug(
                                storage,
                                clip_uid=clip_uid,
                                seed=seed,
                                candidate=candidate,
                                review=review,
                            )
                    except Exception as exc:
                        runtime = time.monotonic() - started
                        attempts.append(
                            BackgroundRemovalAttempt(
                                seed=seed,
                                status="failed",
                                runtime_seconds=runtime,
                                candidate_sha256=candidate_sha,
                                reason=_exception_reason(exc),
                                review=None,
                            )
                        )
                        counters["candidates_failed"] += 1
                        if candidate is not None and _debug_enabled(config):
                            _write_candidate_debug(
                                storage,
                                clip_uid=clip_uid,
                                seed=seed,
                                candidate=candidate,
                                review=None,
                            )

                if accepted is None:
                    statuses = {attempt.status for attempt in attempts}
                    if statuses == {"rejected"}:
                        reason = "all_removal_candidates_rejected"
                        _publish_rejected(
                            storage,
                            clip_uid=clip_uid,
                            original=state,
                            reason=reason,
                            removal_backend=config.remove.backend,
                            attempts=attempts,
                        )
                        counters["rejected"] += 1
                    else:
                        reason = "removal_infrastructure_failure"
                        if state.status == "pending_remove":
                            _publish_retryable(
                                storage,
                                clip_uid=clip_uid,
                                original=state,
                                attempts=[*state.removal_attempts, *attempts],
                                reason=reason,
                                removal_backend=config.remove.backend,
                            )
                            counters["retryable_pending"] += 1
                        else:
                            storage.append_failure(
                                clip_uid=clip_uid,
                                stage="remove",
                                reason=reason,
                            )
                            counters["failed"] += 1
                    counters["processed"] += 1
                    continue

                candidate, candidate_bytes, candidate_sha, seed = accepted
                _publish_ready(
                    config,
                    storage,
                    clip_uid=clip_uid,
                    original=state,
                    inputs=inputs,
                    candidate=candidate,
                    candidate_bytes=candidate_bytes,
                    candidate_sha256=candidate_sha,
                    seed=seed,
                    attempts=[*state.removal_attempts, *attempts],
                )
                counters["processed"] += 1
                counters["ready_removed"] += 1
            except Exception as exc:
                storage.append_failure(
                    clip_uid=clip_uid,
                    stage="remove",
                    reason=_exception_reason(exc),
                )
                counters["failed"] += 1
    finally:
        if owned_judge and active_judge is not None:
            close = getattr(active_judge, "close", None)
            if callable(close):
                close()
        if owned_backend and active_backend is not None:
            close = getattr(active_backend, "close", None)
            if callable(close):
                close()

    stats = RemoveStats(**counters)
    storage.update_stage_counts("remove", stats.to_dict())
    return stats
