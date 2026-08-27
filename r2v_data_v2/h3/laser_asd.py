from __future__ import annotations

import hashlib
import math
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from r2v_data_v2.h3.lr_asd import _launch_executable, _run_logged_command
from r2v_data_v2.h3.pilot_schemas import (
    LaserASDNativeArtifact,
    LaserASDNativeSample,
    SpeechActivityArtifact,
)
from r2v_data_v2.h3.schemas import (
    ActiveSpeakerFaceScore,
    ActiveSpeakerInterval,
    AudioBindingEvidence,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceGeometrySample,
    FaceTrack,
)

LASER_ASD_UPSTREAM_REPOSITORY = "https://github.com/plnguyen2908/LASER_ASD"
LASER_ASD_UPSTREAM_COMMIT = "3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9"


class LaserASDRuntimeError(RuntimeError):
    pass


class LaserASDBackend(Protocol):
    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LaserASDNativeArtifact: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LaserASDRuntimeConfig:
    code_root: Path
    python_path: Path
    model_path: Path
    config_path: Path
    landmark_model_path: Path
    timeout_seconds: float = 1800.0

    @classmethod
    def from_environment(cls) -> LaserASDRuntimeConfig:
        names = (
            "LASER_ASD_CODE_ROOT",
            "LASER_ASD_PYTHON",
            "LASER_ASD_MODEL_PATH",
            "LASER_ASD_CONFIG_PATH",
            "LASER_ASD_LANDMARK_MODEL_PATH",
        )
        values = {name: os.environ.get(name) for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"missing LASER ASD runtime inputs: {', '.join(missing)}")
        return cls(
            code_root=Path(values["LASER_ASD_CODE_ROOT"] or ""),
            python_path=Path(values["LASER_ASD_PYTHON"] or ""),
            model_path=Path(values["LASER_ASD_MODEL_PATH"] or ""),
            config_path=Path(values["LASER_ASD_CONFIG_PATH"] or ""),
            landmark_model_path=Path(
                values["LASER_ASD_LANDMARK_MODEL_PATH"] or ""
            ),
        )

    def validate(self) -> None:
        code_root = self.code_root.expanduser().resolve(strict=True)
        required_source_files = (
            "README.md",
            "LoCoNet/demoLoCoNet_landmark.py",
            "LoCoNet/landmark_loconet.py",
            "create_landmark.py",
        )
        if any(not (code_root / relative).is_file() for relative in required_source_files):
            raise ValueError("LASER_ASD_CODE_ROOT is not the expected upstream checkout")
        if not _launch_executable(self.python_path).is_file():
            raise ValueError("LASER_ASD_PYTHON must be a local executable file")
        for label, path in (
            ("LASER_ASD_MODEL_PATH", self.model_path),
            ("LASER_ASD_CONFIG_PATH", self.config_path),
            ("LASER_ASD_LANDMARK_MODEL_PATH", self.landmark_model_path),
        ):
            if not path.expanduser().resolve(strict=True).is_file():
                raise ValueError(f"{label} must be a local file")
        if self.timeout_seconds <= 0:
            raise ValueError("LASER ASD timeout must be positive")
        try:
            result = subprocess.run(
                ["git", "-C", str(code_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"cannot verify LASER upstream commit: {exc}") from exc
        if result.returncode != 0 or result.stdout.strip() != LASER_ASD_UPSTREAM_COMMIT:
            raise ValueError(
                "LASER checkout must be pinned to " + LASER_ASD_UPSTREAM_COMMIT
            )


class LaserASDSubprocessBackend:
    def __init__(self, config: LaserASDRuntimeConfig) -> None:
        config.validate()
        self.config = config
        self._checkpoint_sha256 = _sha256(config.model_path.resolve())
        self._config_sha256 = _sha256(config.config_path.resolve())
        self._landmark_model_sha256 = _sha256(config.landmark_model_path.resolve())

    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LaserASDNativeArtifact:
        source = source_video_path.expanduser().resolve(strict=True)
        destination = work_dir.expanduser().resolve(strict=False)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination / "laser_asd_native.json"
        bridge = Path(__file__).resolve().parents[2] / "tools" / (
            "run_h3_laser_loconet_bridge.py"
        )
        command = [
            str(_launch_executable(self.config.python_path)),
            str(bridge),
            "--clip-uid",
            clip_uid,
            "--source-video",
            str(source),
            "--work-dir",
            str(destination),
            "--code-root",
            str(self.config.code_root.resolve()),
            "--model-path",
            str(self.config.model_path.resolve()),
            "--config-path",
            str(self.config.config_path.resolve()),
            "--landmark-model-path",
            str(self.config.landmark_model_path.resolve()),
            "--checkpoint-sha256",
            self._checkpoint_sha256,
            "--config-sha256",
            self._config_sha256,
            "--landmark-model-sha256",
            self._landmark_model_sha256,
            "--upstream-commit",
            LASER_ASD_UPSTREAM_COMMIT,
            "--output",
            str(output),
        ]
        _run_logged_command(
            command,
            cwd=self.config.code_root.resolve(),
            stdout_path=destination / "laser_asd.stdout.log",
            stderr_path=destination / "laser_asd.stderr.log",
            timeout_seconds=self.config.timeout_seconds,
            error_type=LaserASDRuntimeError,
        )
        try:
            artifact = LaserASDNativeArtifact.model_validate_json(
                output.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise LaserASDRuntimeError(
                f"LASER bridge returned invalid strict JSON: {exc}"
            ) from exc
        if artifact.clip_uid != clip_uid:
            raise LaserASDRuntimeError("LASER bridge returned the wrong clip UID")
        if artifact.source_video_path != str(source):
            raise LaserASDRuntimeError("LASER bridge returned the wrong source video")
        expected_provenance = (
            str(self.config.model_path.resolve()),
            self._checkpoint_sha256,
            str(self.config.config_path.resolve()),
            self._config_sha256,
            str(self.config.landmark_model_path.resolve()),
            self._landmark_model_sha256,
        )
        actual_provenance = (
            artifact.model_provenance.checkpoint_path,
            artifact.model_provenance.checkpoint_sha256,
            artifact.config_path,
            artifact.config_sha256,
            artifact.landmark_model_path,
            artifact.landmark_model_sha256,
        )
        if actual_provenance != expected_provenance:
            raise LaserASDRuntimeError("LASER bridge runtime provenance does not match")
        return artifact


def _bounded_detection_samples(
    samples: Sequence[LaserASDNativeSample],
    *,
    maximum_samples: int = 32,
) -> list[LaserASDNativeSample]:
    detected = [sample for sample in samples if sample.detection_confidence is not None]
    if not detected:
        raise ValueError("LASER track has no raw S3FD confidence samples")
    if len(detected) <= maximum_samples:
        return detected
    indices = [
        round(index * (len(detected) - 1) / (maximum_samples - 1))
        for index in range(maximum_samples)
    ]
    return [detected[index] for index in indices]


def normalize_laser_face_tracks(native: LaserASDNativeArtifact) -> list[FaceTrack]:
    tracks: list[FaceTrack] = []
    for native_track in native.tracks:
        samples = _bounded_detection_samples(native_track.samples)
        confidences = [
            float(sample.detection_confidence)
            for sample in samples
            if sample.detection_confidence is not None
        ]
        tracks.append(
            FaceTrack(
                face_track_id=native_track.face_track_id,
                start_time=native_track.samples[0].timestamp_seconds,
                end_time=min(
                    native.duration_seconds,
                    native_track.samples[-1].timestamp_seconds
                    + 1 / native.model_fps,
                ),
                sample_count=len(native_track.samples),
                mean_detection_confidence=sum(confidences) / len(confidences),
                geometry_samples=[
                    FaceGeometrySample(
                        frame_index=sample.frame_index,
                        timestamp=sample.timestamp_seconds,
                        bbox_xyxy=sample.bbox_xyxy,
                        confidence=float(sample.detection_confidence),
                    )
                    for sample in samples
                ],
            )
        )
    return tracks


def _speech_present(
    speech: SpeechActivityArtifact,
    *,
    start_time: float,
    end_time: float,
) -> bool:
    return any(
        interval.start_time < end_time and interval.end_time > start_time
        for interval in speech.intervals
    )


def normalize_laser_asd_evidence(
    native: LaserASDNativeArtifact,
    speech: SpeechActivityArtifact,
    associations: list[EntityFaceAssociation],
) -> AudioBindingEvidence:
    if native.clip_uid != speech.clip_uid:
        raise ValueError("LASER and speech evidence clip IDs must match")
    if not math.isclose(
        native.duration_seconds,
        speech.duration_seconds,
        rel_tol=0,
        abs_tol=1 / native.model_fps,
    ):
        raise ValueError("LASER and speech evidence durations do not match")
    samples_by_frame: dict[int, list[tuple[str, LaserASDNativeSample]]] = {}
    for track in native.tracks:
        for sample in track.samples:
            samples_by_frame.setdefault(sample.frame_index, []).append(
                (track.face_track_id, sample)
            )
    frame_count = math.ceil(native.duration_seconds * native.model_fps)
    intervals: list[ActiveSpeakerInterval] = []
    for frame_index in range(frame_count):
        start_time = frame_index / native.model_fps
        end_time = min((frame_index + 1) / native.model_fps, native.duration_seconds)
        if end_time <= start_time:
            continue
        samples = sorted(samples_by_frame.get(frame_index, []), key=lambda item: item[0])
        visible_ids = [face_track_id for face_track_id, _ in samples]
        intervals.append(
            ActiveSpeakerInterval(
                start_time=start_time,
                end_time=end_time,
                speech_present=_speech_present(
                    speech,
                    start_time=start_time,
                    end_time=end_time,
                ),
                visible_face_track_ids=visible_ids,
                face_scores=[
                    ActiveSpeakerFaceScore(
                        face_track_id=face_track_id,
                        raw_backend_score=sample.raw_backend_score,
                        backend_native_active=sample.backend_native_active,
                        score_semantics="laser_loconet_native_score",
                    )
                    for face_track_id, sample in samples
                ],
                face_speaking_probabilities={},
                asd_coverage_ratio=1.0,
                model_provenance=native.model_provenance,
                audio_quality_usable=True,
                synchronization_plausible=True,
            )
        )
    return AudioBindingEvidence(
        clip_uid=native.clip_uid,
        audio=AudioTrackMetadata(
            status="ready",
            source_video_path=native.source_video_path,
            full_audio_path=native.audio_path,
            duration_seconds=native.duration_seconds,
            sample_rate_hz=native.audio_sample_rate_hz,
            channels=native.audio_channels,
        ),
        face_tracks=normalize_laser_face_tracks(native),
        associations=associations,
        active_speaker_intervals=intervals,
    )


class PrecomputedLaserASDBackend:
    def __init__(self, artifacts: dict[str, LaserASDNativeArtifact]) -> None:
        self.artifacts = artifacts

    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LaserASDNativeArtifact:
        del work_dir
        artifact = self.artifacts[clip_uid]
        if artifact.source_video_path != str(source_video_path):
            raise ValueError("precomputed LASER source video does not match")
        return artifact
