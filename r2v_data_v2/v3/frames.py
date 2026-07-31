from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.schemas import (
    SampledFrame,
    SampledFramesArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class DecodedFrameInfo:
    source_frame_index: int
    timestamp_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class DecodedVideoFrame:
    source_frame_index: int
    image: Image.Image


class FrameDecoder(Protocol):
    def inspect(self, video_path: Path) -> list[DecodedFrameInfo]: ...

    def decode_indices(
        self,
        video_path: Path,
        source_frame_indices: list[int],
    ) -> list[DecodedVideoFrame]: ...


class OpenCvFrameDecoder:
    def inspect(self, video_path: Path) -> list[DecodedFrameInfo]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open source video: {video_path}")
        frames: list[DecodedFrameInfo] = []
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                timestamp_seconds = (
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                )
                if (
                    not math.isfinite(timestamp_seconds)
                    or timestamp_seconds < 0
                ):
                    raise ValueError(
                        "video decoder returned an invalid frame timestamp"
                    )
                frames.append(
                    DecodedFrameInfo(
                        source_frame_index=len(frames),
                        timestamp_seconds=timestamp_seconds,
                        width=int(width),
                        height=int(height),
                    )
                )
        finally:
            capture.release()
        return frames

    def decode_indices(
        self,
        video_path: Path,
        source_frame_indices: list[int],
    ) -> list[DecodedVideoFrame]:
        import cv2

        requested = set(source_frame_indices)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open source video: {video_path}")
        decoded: dict[int, DecodedVideoFrame] = {}
        source_frame_index = 0
        try:
            while requested:
                ok, image = capture.read()
                if not ok:
                    break
                if source_frame_index in requested:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    decoded[source_frame_index] = DecodedVideoFrame(
                        source_frame_index=source_frame_index,
                        image=Image.fromarray(rgb),
                    )
                    requested.remove(source_frame_index)
                source_frame_index += 1
        finally:
            capture.release()
        missing = sorted(requested)
        if missing:
            raise ValueError(
                f"could not decode selected source frame indices: {missing}"
            )
        return [decoded[index] for index in source_frame_indices]


@dataclass(frozen=True)
class FrameSamplingStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def deterministic_frame_indices(
    total_decodable_frames: int,
    *,
    count: int = 10,
) -> list[int]:
    if count < 2:
        raise ValueError("frame sample count must be at least two")
    if total_decodable_frames < count:
        raise ValueError(
            f"video has {total_decodable_frames} decodable frames; "
            f"{count} unique frames are required"
        )
    last_index = total_decodable_frames - 1
    denominator = count - 1
    indices = [
        (slot * last_index * 2 + denominator) // (2 * denominator)
        for slot in range(count)
    ]
    if (
        indices[0] != 0
        or indices[-1] != last_index
        or any(
            indices[index] >= indices[index + 1]
            for index in range(len(indices) - 1)
        )
    ):
        raise ValueError("deterministic frame selection was not strictly increasing")
    return indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_frames(
    infos: list[DecodedFrameInfo],
    selected_indices: list[int],
) -> tuple[int, int, list[DecodedFrameInfo]]:
    selected = [infos[index] for index in selected_indices]
    dimensions = {(frame.width, frame.height) for frame in selected}
    if len(dimensions) != 1:
        raise ValueError("sampled video frames must have stable dimensions")
    timestamps = [frame.timestamp_seconds for frame in selected]
    if any(
        not math.isfinite(timestamp) or timestamp < 0
        for timestamp in timestamps
    ):
        raise ValueError("sampled video frame timestamps must be finite")
    if any(
        timestamps[index] >= timestamps[index + 1]
        for index in range(len(timestamps) - 1)
    ):
        raise ValueError(
            "sampled video frame timestamps must be strictly increasing"
        )
    width, height = dimensions.pop()
    return width, height, selected


def _save_jpeg(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(
        path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def _validate_existing_frames(
    storage: RunStorage,
    clip_uid: str,
) -> SampledFramesArtifact:
    artifact = storage.read_frames(clip_uid)
    if artifact.clip_uid != clip_uid:
        raise ValueError("frames artifact clip_uid does not match its clip")
    clip_dir = storage.clip_dir(clip_uid).resolve(strict=False)
    for frame in artifact.frames:
        image_path = (clip_dir / frame.image_path).resolve(strict=False)
        if clip_dir not in image_path.parents or not image_path.is_file():
            raise ValueError(
                f"sampled frame is missing or outside clip directory: "
                f"{frame.image_path}"
            )
        if _sha256(image_path) != frame.sha256:
            raise ValueError(
                f"sampled frame hash mismatch: {frame.image_path}"
            )
        with Image.open(image_path) as image:
            if image.size != (artifact.width, artifact.height):
                raise ValueError(
                    f"sampled frame dimensions mismatch: {frame.image_path}"
                )
    return artifact


def _build_clip_frames(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    video_path: Path,
    decoder: FrameDecoder,
) -> SampledFramesArtifact:
    infos = decoder.inspect(video_path)
    selected_indices = deterministic_frame_indices(
        len(infos),
        count=config.frames.count,
    )
    width, height, selected_infos = _validate_source_frames(
        infos,
        selected_indices,
    )
    decoded = decoder.decode_indices(video_path, selected_indices)
    if [frame.source_frame_index for frame in decoded] != selected_indices:
        raise ValueError("frame decoder returned frames in an unexpected order")

    frames_dir = storage.frames_dir(clip_uid)
    frames_dir.mkdir(parents=True, exist_ok=True)
    temporary = frames_dir / f".tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        sampled_frames: list[SampledFrame] = []
        for slot, (info, decoded_frame) in enumerate(
            zip(selected_infos, decoded)
        ):
            if decoded_frame.image.size != (width, height):
                raise ValueError("decoded frame dimensions changed during sampling")
            temporary_path = temporary / f"{slot:02d}.jpg"
            _save_jpeg(decoded_frame.image, temporary_path)
            sampled_frames.append(
                SampledFrame(
                    slot=slot,
                    source_frame_index=info.source_frame_index,
                    timestamp_seconds=info.timestamp_seconds,
                    image_path=f"frames/{slot:02d}.jpg",
                    sha256=_sha256(temporary_path),
                )
            )
        artifact = SampledFramesArtifact(
            clip_uid=clip_uid,
            width=width,
            height=height,
            frames=sampled_frames,
        )

        storage.prepare_frames_publication(clip_uid)
        for frame in artifact.frames:
            source = temporary / f"{frame.slot:02d}.jpg"
            destination = storage.frame_path(clip_uid, frame.slot)
            source.replace(destination)
        write_json_atomic(
            storage.frames_manifest_path(clip_uid),
            artifact.model_dump(mode="json"),
        )
        return artifact
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def sample_frames(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    decoder: FrameDecoder | None = None,
) -> FrameSamplingStats:
    frame_decoder = decoder or OpenCvFrameDecoder()
    processed = skipped_existing = skipped_not_ready = failed = 0
    for clip in storage.iter_clips():
        if clip.annotation is None or clip.annotation.status != "ready":
            skipped_not_ready += 1
            continue
        manifest_path = storage.frames_manifest_path(clip.clip_uid)
        if manifest_path.is_file() and not overwrite:
            try:
                _validate_existing_frames(storage, clip.clip_uid)
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clip artifacts
                storage.append_failure(
                    clip_uid=clip.clip_uid,
                    stage="frames",
                    reason=str(exc),
                )
                failed += 1
            else:
                skipped_existing += 1
            continue
        try:
            _build_clip_frames(
                config,
                storage,
                clip_uid=clip.clip_uid,
                video_path=Path(clip.source.video_path),
                decoder=frame_decoder,
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-clip decode failures
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="frames",
                reason=str(exc),
            )
            failed += 1
    stats = FrameSamplingStats(
        processed=processed,
        skipped_existing=skipped_existing,
        skipped_not_ready=skipped_not_ready,
        failed=failed,
    )
    storage.update_stage_counts("frames", stats.to_dict())
    return stats
