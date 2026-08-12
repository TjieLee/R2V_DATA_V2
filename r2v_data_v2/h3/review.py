from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from r2v_data_v2.h3.pilot_schemas import LRASDNativeArtifact
from r2v_data_v2.h3.schemas import (
    AudioBindingSidecar,
    AudioEntityBinding,
    EntityFaceAssociation,
)


class ReviewMediaBackend(Protocol):
    def render_visualization(
        self,
        *,
        source_video_path: Path,
        timeline_path: Path,
        destination_path: Path,
    ) -> None: ...

    def extract_audio(
        self,
        *,
        source_audio_path: Path,
        start_time: float,
        end_time: float,
        destination_path: Path,
    ) -> None: ...


class ExternalReviewMediaBackend:
    def __init__(
        self,
        *,
        python_path: Path,
        ffmpeg_path: str = "ffmpeg",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.python_path = python_path.expanduser().resolve(strict=True)
        self.ffmpeg_path = ffmpeg_path
        self.timeout_seconds = timeout_seconds
        if timeout_seconds <= 0:
            raise ValueError("review media timeout must be positive")

    def _run(self, command: list[str], *, destination_path: Path) -> None:
        stdout_path = destination_path.with_suffix(destination_path.suffix + ".stdout")
        stderr_path = destination_path.with_suffix(destination_path.suffix + ".stderr")
        with (
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"review media subprocess failed: {type(exc).__name__}: {exc}"
                ) from exc
        if result.returncode != 0 or not destination_path.is_file():
            raise RuntimeError(
                f"review media subprocess exited with code {result.returncode}; "
                f"stderr={stderr_path}"
            )

    def render_visualization(
        self,
        *,
        source_video_path: Path,
        timeline_path: Path,
        destination_path: Path,
    ) -> None:
        helper = Path(__file__).resolve().parents[2] / "tools" / (
            "render_h3_audio_binding_review.py"
        )
        self._run(
            [
                str(self.python_path),
                str(helper),
                "--source",
                str(source_video_path),
                "--timeline",
                str(timeline_path),
                "--output",
                str(destination_path),
                "--ffmpeg",
                self.ffmpeg_path,
            ],
            destination_path=destination_path,
        )

    def extract_audio(
        self,
        *,
        source_audio_path: Path,
        start_time: float,
        end_time: float,
        destination_path: Path,
    ) -> None:
        self._run(
            [
                self.ffmpeg_path,
                "-y",
                "-ss",
                f"{start_time:.6f}",
                "-i",
                str(source_audio_path),
                "-t",
                f"{end_time - start_time:.6f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination_path),
                "-loglevel",
                "error",
            ],
            destination_path=destination_path,
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _binding_at(
    bindings: list[AudioEntityBinding],
    timestamp_seconds: float,
) -> AudioEntityBinding | None:
    return next(
        (
            binding
            for binding in bindings
            if binding.start_time <= timestamp_seconds < binding.end_time
        ),
        None,
    )


def build_review_timeline(
    *,
    native: LRASDNativeArtifact,
    associations: list[EntityFaceAssociation],
    bindings: list[AudioEntityBinding],
) -> dict[str, object]:
    association_by_track = {item.face_track_id: item for item in associations}
    samples = []
    for track in native.tracks:
        association = association_by_track.get(track.face_track_id)
        entity_id = (
            association.entity_id
            if association is not None and association.status == "matched"
            else None
        )
        for sample in track.samples:
            binding = _binding_at(bindings, sample.timestamp_seconds)
            samples.append(
                {
                    "face_track_id": track.face_track_id,
                    "entity_id": entity_id,
                    "frame_index": sample.frame_index,
                    "timestamp_seconds": sample.timestamp_seconds,
                    "bbox_xyxy": list(sample.bbox_xyxy),
                    "raw_class1_logit": sample.raw_class1_logit,
                    "backend_native_active": sample.backend_native_active,
                    "binding_status": (
                        binding.status if binding is not None else "ambiguous"
                    ),
                }
            )
    samples.sort(
        key=lambda item: (
            float(item["timestamp_seconds"]),
            str(item["face_track_id"]),
        )
    )
    return {
        "schema_version": "r2v.h3.audio_binding_review_timeline.1",
        "clip_uid": native.clip_uid,
        "model_fps": native.model_fps,
        "score_semantics": native.score_semantics,
        "active_decision_rule": native.active_decision_rule,
        "samples": samples,
        "bindings": [binding.model_dump(mode="json") for binding in bindings],
    }


def _bound_review_segments(
    bindings: list[AudioEntityBinding],
) -> list[tuple[str, float, float]]:
    segments: list[tuple[str, float, float]] = []
    for binding in bindings:
        if binding.status != "bound" or binding.entity_id is None:
            continue
        if (
            segments
            and segments[-1][0] == binding.entity_id
            and abs(segments[-1][2] - binding.start_time) <= 1e-9
        ):
            entity_id, start_time, _ = segments[-1]
            segments[-1] = (entity_id, start_time, binding.end_time)
        else:
            segments.append(
                (binding.entity_id, binding.start_time, binding.end_time)
            )
    return segments


def write_review_bundle(
    *,
    destination: Path,
    source_video_path: Path,
    native: LRASDNativeArtifact,
    associations: list[EntityFaceAssociation],
    sidecar: AudioBindingSidecar,
    media_backend: ReviewMediaBackend,
    source_audio_path: Path | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    source_copy = destination / "source.mp4"
    shutil.copyfile(source_video_path, source_copy)
    _write_json(
        destination / "audio_binding.json",
        sidecar.model_dump(mode="json"),
    )
    _write_json(
        destination / "lr_asd_native.json",
        native.model_dump(mode="json"),
    )
    _write_json(
        destination / "face_entity_association.json",
        {
            "schema_version": "r2v.h3.face_entity_association.1",
            "clip_uid": native.clip_uid,
            "associations": [
                association.model_dump(mode="json")
                for association in associations
            ],
        },
    )
    timeline = build_review_timeline(
        native=native,
        associations=associations,
        bindings=sidecar.bindings,
    )
    timeline_path = destination / "timeline.json"
    _write_json(timeline_path, timeline)
    media_backend.render_visualization(
        source_video_path=source_copy,
        timeline_path=timeline_path,
        destination_path=destination / "visualization.mp4",
    )
    bound_root = destination / "bound_audio"
    bound_root.mkdir()
    per_entity_counter: dict[str, int] = {}
    review_audio_path = (
        source_audio_path if source_audio_path is not None else Path(native.audio_path)
    ).resolve(strict=True)
    for entity_id, start_time, end_time in _bound_review_segments(sidecar.bindings):
        per_entity_counter[entity_id] = per_entity_counter.get(entity_id, 0) + 1
        media_backend.extract_audio(
            source_audio_path=review_audio_path,
            start_time=start_time,
            end_time=end_time,
            destination_path=(
                bound_root
                / f"{entity_id}_{per_entity_counter[entity_id]:03d}.wav"
            ),
        )
