from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from r2v_data_v2.h3.pilot_schemas import (
    LRASDNativeArtifact,
    LRASDNativeSample,
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


class LRASDRuntimeError(RuntimeError):
    pass


class SpeechActivityRuntimeError(RuntimeError):
    pass


def _launch_executable(path: Path) -> Path:
    return path.expanduser().absolute()


class LRASDBackend(Protocol):
    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LRASDNativeArtifact: ...


class SpeechActivityBackend(Protocol):
    def detect(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
        duration_seconds: float,
        work_dir: Path,
    ) -> SpeechActivityArtifact: ...


@dataclass(frozen=True)
class LRASDRuntimeConfig:
    code_root: Path
    python_path: Path
    model_path: Path
    timeout_seconds: float = 1800.0

    @classmethod
    def from_environment(cls) -> LRASDRuntimeConfig:
        required = {
            "LR_ASD_CODE_ROOT": os.environ.get("LR_ASD_CODE_ROOT"),
            "LR_ASD_PYTHON": os.environ.get("LR_ASD_PYTHON"),
            "LR_ASD_MODEL_PATH": os.environ.get("LR_ASD_MODEL_PATH"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing LR-ASD runtime inputs: {', '.join(missing)}")
        return cls(
            code_root=Path(required["LR_ASD_CODE_ROOT"] or ""),
            python_path=Path(required["LR_ASD_PYTHON"] or ""),
            model_path=Path(required["LR_ASD_MODEL_PATH"] or ""),
        )

    def validate(self) -> None:
        code_root = self.code_root.expanduser().resolve(strict=True)
        python_path = _launch_executable(self.python_path)
        model_path = self.model_path.expanduser().resolve(strict=True)
        if not (code_root / "Columbia_test.py").is_file():
            raise ValueError("LR_ASD_CODE_ROOT must contain Columbia_test.py")
        if not python_path.is_file():
            raise ValueError("LR_ASD_PYTHON must be a local executable file")
        if not model_path.is_file():
            raise ValueError("LR_ASD_MODEL_PATH must be a local checkpoint file")
        if self.timeout_seconds <= 0:
            raise ValueError("LR-ASD timeout must be positive")


@dataclass(frozen=True)
class SileroVADRuntimeConfig:
    python_path: Path
    model_path: Path
    timeout_seconds: float = 300.0

    def validate(self) -> None:
        if not _launch_executable(self.python_path).is_file():
            raise ValueError("Silero VAD python path must be a local file")
        if not self.model_path.expanduser().resolve(strict=True).is_file():
            raise ValueError("Silero VAD model path must be a local file")
        if self.timeout_seconds <= 0:
            raise ValueError("Silero VAD timeout must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    error_type: type[RuntimeError],
    environment_override: dict[str, str] | None = None,
) -> None:
    environment = None
    if environment_override is not None:
        environment = os.environ.copy()
        environment.update(environment_override)
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise error_type(
                f"subprocess execution failed: {type(exc).__name__}: {exc}"
            ) from exc
    if result.returncode != 0:
        raise error_type(
            f"subprocess exited with code {result.returncode}; stderr={stderr_path}"
        )


class LRASDSubprocessBackend:
    """Run the unmodified official raw-video demo once, then normalize its pickle output."""

    def __init__(self, config: LRASDRuntimeConfig) -> None:
        config.validate()
        self.config = config
        self._checkpoint_sha256 = _sha256(config.model_path.resolve())

    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LRASDNativeArtifact:
        return self._analyze(
            clip_uid=clip_uid,
            source_video_path=source_video_path,
            work_dir=work_dir,
            command_environment=None,
        )

    def analyze_on_gpu(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
        gpu_id: str,
    ) -> LRASDNativeArtifact:
        if not gpu_id.strip() or "," in gpu_id:
            raise ValueError("LR-ASD assigned GPU must be one physical device ID")
        return self._analyze(
            clip_uid=clip_uid,
            source_video_path=source_video_path,
            work_dir=work_dir,
            command_environment={"CUDA_VISIBLE_DEVICES": gpu_id},
        )

    def _analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
        command_environment: dict[str, str] | None,
    ) -> LRASDNativeArtifact:
        source = source_video_path.expanduser().resolve(strict=True)
        destination = work_dir.expanduser().resolve(strict=False)
        destination.mkdir(parents=True, exist_ok=False)
        source_copy = destination / f"source{source.suffix.lower() or '.mp4'}"
        shutil.copyfile(source, source_copy)
        stdout_path = destination / "lr_asd.stdout.log"
        stderr_path = destination / "lr_asd.stderr.log"
        command = [
            str(_launch_executable(self.config.python_path)),
            str((self.config.code_root / "Columbia_test.py").resolve()),
            "--videoName",
            "source",
            "--videoFolder",
            str(destination),
            "--pretrainModel",
            str(self.config.model_path.resolve()),
        ]
        command_arguments = {
            "cwd": self.config.code_root.resolve(),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "timeout_seconds": self.config.timeout_seconds,
            "error_type": LRASDRuntimeError,
        }
        if command_environment is not None:
            command_arguments["environment_override"] = command_environment
        _run_logged_command(command, **command_arguments)
        native_path = destination / "lr_asd_native.json"
        converter = Path(__file__).resolve().parents[2] / "tools" / (
            "convert_lr_asd_native.py"
        )
        converter_command = [
            str(_launch_executable(self.config.python_path)),
            str(converter),
            "--clip-uid",
            clip_uid,
            "--source-video",
            str(source),
            "--vendor-output",
            str(destination / "source"),
            "--code-root",
            str(self.config.code_root.resolve()),
            "--model-path",
            str(self.config.model_path.resolve()),
            "--checkpoint-sha256",
            self._checkpoint_sha256,
            "--output",
            str(native_path),
        ]
        _run_logged_command(
            converter_command,
            cwd=self.config.code_root.resolve(),
            stdout_path=destination / "converter.stdout.log",
            stderr_path=destination / "converter.stderr.log",
            timeout_seconds=min(self.config.timeout_seconds, 300.0),
            error_type=LRASDRuntimeError,
        )
        try:
            return LRASDNativeArtifact.model_validate_json(
                native_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise LRASDRuntimeError(
                f"LR-ASD bridge returned invalid strict JSON: {exc}"
            ) from exc


class SileroVADSubprocessBackend:
    def __init__(self, config: SileroVADRuntimeConfig) -> None:
        config.validate()
        self.config = config

    def detect(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
        duration_seconds: float,
        work_dir: Path,
    ) -> SpeechActivityArtifact:
        destination = work_dir.expanduser().resolve(strict=False)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "speech_activity.json"
        bridge = Path(__file__).resolve().parents[2] / "tools" / (
            "run_h3_silero_vad_bridge.py"
        )
        command = [
            str(_launch_executable(self.config.python_path)),
            str(bridge),
            "--clip-uid",
            clip_uid,
            "--audio",
            str(audio_path.expanduser().resolve(strict=True)),
            "--duration-seconds",
            str(duration_seconds),
            "--model-path",
            str(self.config.model_path.resolve()),
            "--output",
            str(output),
        ]
        _run_logged_command(
            command,
            cwd=destination,
            stdout_path=destination / "silero.stdout.log",
            stderr_path=destination / "silero.stderr.log",
            timeout_seconds=self.config.timeout_seconds,
            error_type=SpeechActivityRuntimeError,
        )
        try:
            return SpeechActivityArtifact.model_validate_json(
                output.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise SpeechActivityRuntimeError(
                f"Silero VAD bridge returned invalid strict JSON: {exc}"
            ) from exc


def _bounded_detection_samples(
    samples: Sequence[LRASDNativeSample],
    *,
    maximum_samples: int = 32,
) -> list[LRASDNativeSample]:
    detected = [sample for sample in samples if sample.detection_confidence is not None]
    if not detected:
        raise ValueError("LR-ASD track has no raw S3FD confidence samples")
    if len(detected) <= maximum_samples:
        return detected
    indices = [
        round(index * (len(detected) - 1) / (maximum_samples - 1))
        for index in range(maximum_samples)
    ]
    return [detected[index] for index in indices]


def normalize_face_tracks(native: LRASDNativeArtifact) -> list[FaceTrack]:
    tracks: list[FaceTrack] = []
    for native_track in native.tracks:
        samples = _bounded_detection_samples(native_track.samples)
        confidences = [
            sample.detection_confidence
            for sample in samples
            if sample.detection_confidence is not None
        ]
        tracks.append(
            FaceTrack(
                face_track_id=native_track.face_track_id,
                start_time=native_track.samples[0].timestamp_seconds,
                end_time=min(
                    native.duration_seconds,
                    native_track.samples[-1].timestamp_seconds + 1 / native.model_fps,
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


def normalize_lr_asd_evidence(
    native: LRASDNativeArtifact,
    speech: SpeechActivityArtifact,
    associations: list[EntityFaceAssociation],
) -> AudioBindingEvidence:
    if native.clip_uid != speech.clip_uid:
        raise ValueError("LR-ASD and speech evidence clip IDs must match")
    if not math.isclose(
        native.duration_seconds,
        speech.duration_seconds,
        rel_tol=0,
        abs_tol=1 / native.model_fps,
    ):
        raise ValueError("LR-ASD and speech evidence durations do not match")
    samples_by_frame: dict[int, list[tuple[str, LRASDNativeSample]]] = {}
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
        frame_samples = list(samples_by_frame.get(frame_index, []))
        frame_samples.sort(key=lambda item: item[0])
        visible_ids = [face_track_id for face_track_id, _ in frame_samples]
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
                        raw_backend_score=sample.raw_class1_logit,
                        backend_native_active=sample.backend_native_active,
                        score_semantics="lr_asd_native_class_1_logit",
                    )
                    for face_track_id, sample in frame_samples
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
        face_tracks=normalize_face_tracks(native),
        associations=associations,
        active_speaker_intervals=intervals,
    )


class PrecomputedLRASDBackend:
    def __init__(self, artifacts: dict[str, LRASDNativeArtifact]) -> None:
        self.artifacts = artifacts

    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LRASDNativeArtifact:
        del work_dir
        artifact = self.artifacts[clip_uid]
        if artifact.source_video_path != str(source_video_path):
            raise ValueError("precomputed LR-ASD source video does not match")
        return artifact


class PrecomputedSpeechActivityBackend:
    def __init__(self, artifacts: dict[str, SpeechActivityArtifact]) -> None:
        self.artifacts = artifacts

    def detect(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
        duration_seconds: float,
        work_dir: Path,
    ) -> SpeechActivityArtifact:
        del work_dir
        artifact = self.artifacts[clip_uid]
        if artifact.source_audio_path != str(audio_path):
            raise ValueError("precomputed speech source audio does not match")
        if not math.isclose(
            artifact.duration_seconds,
            duration_seconds,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("precomputed speech duration does not match")
        return artifact
