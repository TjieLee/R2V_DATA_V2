from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import subprocess
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, Self

import numpy as np
from PIL import Image

from r2v_data_v2.h3.audio_schemas import AudioStreamProvenance


@dataclass(frozen=True)
class MaterializedMedia:
    path: Path
    stream: AudioStreamProvenance
    sample_rate_hz: int
    channels: int
    output_format: str


@dataclass(frozen=True)
class EmbeddingResult:
    vector: np.ndarray
    model_identifier: str
    checkpoint_sha256: str | None = None
    backend_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class FaceEmbeddingResult:
    status: str
    embedding: EmbeddingResult | None = None
    face_crop: Image.Image | None = None
    reason: str | None = None


def fingerprint_local_model_path(path: Path) -> str:
    """Hash one explicit local model file/directory without resolving a cache ID."""
    source = path.expanduser().resolve(strict=True)
    files = [source] if source.is_file() else sorted(
        item for item in source.rglob("*") if item.is_file()
    )
    if not files:
        raise ValueError("local embedding model path contains no files")
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if source.is_file() else item.relative_to(source).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class FaceEmbeddingBackend(Protocol):
    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult: ...


class SpeakerEmbeddingBackend(Protocol):
    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult: ...


class IdentityTextEmbeddingBackend(Protocol):
    def embed_text(
        self,
        *,
        entity_occurrence_id: str,
        text: str,
    ) -> EmbeddingResult: ...


class TranscriptBackend(Protocol):
    def transcript(self, *, clip_uid: str, audio_path: Path) -> list[dict[str, object]]: ...


class AudioMediaBackend(Protocol):
    def materialize_full_audio(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        destination: Path,
        sample_rate_hz: int,
        channels: int,
        output_format: str,
    ) -> MaterializedMedia: ...

    def extract_voice_reference(
        self,
        *,
        clip_uid: str,
        entity_id: str,
        full_audio_path: Path,
        start_time: float,
        end_time: float,
        destination: Path,
        sample_rate_hz: int,
        output_format: str,
        source_audio_path: Path | None = None,
        source_start_sample: int | None = None,
        source_end_sample: int | None = None,
    ) -> Path: ...


class CandidateIndexBackend(Protocol):
    def search(
        self,
        vectors: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class PrecomputedEmbeddingBackend:
    """GPU-free embedding backend keyed by entity occurrence ID."""

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        *,
        model_identifier: str,
        checkpoint_sha256: str | None = None,
        face_crops: dict[str, Image.Image] | None = None,
    ) -> None:
        self.vectors = vectors
        self.model_identifier = model_identifier
        self.checkpoint_sha256 = checkpoint_sha256
        self.face_crops = face_crops or {}

    def _result(self, entity_occurrence_id: str) -> EmbeddingResult:
        try:
            vector = np.asarray(self.vectors[entity_occurrence_id], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(
                f"missing precomputed embedding for {entity_occurrence_id}"
            ) from exc
        return EmbeddingResult(
            vector=vector,
            model_identifier=self.model_identifier,
            checkpoint_sha256=self.checkpoint_sha256,
            backend_metadata={"backend": "precomputed"},
        )

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        del image_path
        if entity_occurrence_id not in self.vectors:
            return FaceEmbeddingResult(status="unavailable", reason="face_not_found")
        crop = self.face_crops.get(entity_occurrence_id)
        if crop is None:
            crop = Image.new("RGB", (16, 16), "white")
        return FaceEmbeddingResult(
            status="available",
            embedding=self._result(entity_occurrence_id),
            face_crop=crop.copy(),
        )

    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult:
        del audio_path
        return self._result(entity_occurrence_id)

    def embed_text(
        self,
        *,
        entity_occurrence_id: str,
        text: str,
    ) -> EmbeddingResult:
        if not text.strip():
            raise ValueError("identity text must not be empty")
        return self._result(entity_occurrence_id)


class PrecomputedAudioMediaBackend:
    """Copies precomputed lossless fixtures while preserving explicit provenance."""

    def __init__(
        self,
        full_audio_by_clip: dict[str, Path],
        voice_audio_by_occurrence: dict[str, Path],
        stream_by_clip: dict[str, AudioStreamProvenance],
    ) -> None:
        self.full_audio_by_clip = full_audio_by_clip
        self.voice_audio_by_occurrence = voice_audio_by_occurrence
        self.stream_by_clip = stream_by_clip

    @staticmethod
    def _copy(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        return destination

    def materialize_full_audio(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        destination: Path,
        sample_rate_hz: int,
        channels: int,
        output_format: str,
    ) -> MaterializedMedia:
        del source_video_path
        source = self.full_audio_by_clip[clip_uid]
        self._copy(source, destination)
        return MaterializedMedia(
            path=destination,
            stream=self.stream_by_clip[clip_uid],
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            output_format=output_format,
        )

    def extract_voice_reference(
        self,
        *,
        clip_uid: str,
        entity_id: str,
        full_audio_path: Path,
        start_time: float,
        end_time: float,
        destination: Path,
        sample_rate_hz: int,
        output_format: str,
        source_audio_path: Path | None = None,
        source_start_sample: int | None = None,
        source_end_sample: int | None = None,
    ) -> Path:
        del (
            full_audio_path,
            start_time,
            end_time,
            sample_rate_hz,
            output_format,
            source_audio_path,
            source_start_sample,
            source_end_sample,
        )
        return self._copy(
            self.voice_audio_by_occurrence[f"{clip_uid}/{entity_id}"],
            destination,
        )


class PrecomputedTranscriptBackend:
    def __init__(self, segments_by_clip: dict[str, list[dict[str, object]]]) -> None:
        self.segments_by_clip = segments_by_clip

    def transcript(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
    ) -> list[dict[str, object]]:
        del audio_path
        return [dict(item) for item in self.segments_by_clip.get(clip_uid, [])]


class FFmpegAudioMediaBackend:
    """Explicit local ffmpeg adapter; it never downloads or loads a model."""

    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def _probe(self, source: Path) -> AudioStreamProvenance:
        completed = subprocess.run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index,codec_name,sample_rate,channels,duration,time_base",
                "-of",
                "json",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(completed.stdout).get("streams", [])
        if len(streams) != 1:
            raise ValueError("source video must expose exactly one selected audio stream")
        stream = streams[0]
        return AudioStreamProvenance(
            stream_index=int(stream["index"]),
            codec_name=str(stream["codec_name"]),
            original_sample_rate_hz=int(stream["sample_rate"]),
            original_channels=int(stream["channels"]),
            duration_seconds=float(stream["duration"]),
            time_base=str(stream["time_base"]),
        )

    @staticmethod
    def _publish_command(command: list[str], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.tmp-{uuid.uuid4().hex}{destination.suffix}"
        )
        try:
            subprocess.run([*command, str(temporary)], check=True, capture_output=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_pcm16_mono_slice(
        *,
        source: Path,
        destination: Path,
        sample_rate_hz: int,
        start_sample: int,
        end_sample: int,
    ) -> None:
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError("voice-reference sample extent is invalid")
        with wave.open(str(source), "rb") as input_audio:
            if (
                input_audio.getnchannels() != 1
                or input_audio.getsampwidth() != 2
                or input_audio.getframerate() != sample_rate_hz
                or input_audio.getcomptype() != "NONE"
            ):
                raise ValueError(
                    "exact voice-reference extraction requires mono PCM16 source audio"
                )
            if end_sample > input_audio.getnframes():
                raise ValueError("voice-reference sample extent exceeds source audio")
            input_audio.setpos(start_sample)
            frames = input_audio.readframes(end_sample - start_sample)
        if len(frames) != (end_sample - start_sample) * 2:
            raise ValueError("voice-reference source audio ended unexpectedly")
        with wave.open(str(destination), "wb") as output_audio:
            output_audio.setnchannels(1)
            output_audio.setsampwidth(2)
            output_audio.setframerate(sample_rate_hz)
            output_audio.writeframes(frames)

    def materialize_full_audio(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        destination: Path,
        sample_rate_hz: int,
        channels: int,
        output_format: str,
    ) -> MaterializedMedia:
        del clip_uid
        stream = self._probe(source_video_path)
        codec = "flac" if output_format == "flac" else "pcm_s16le"
        self._publish_command(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-i",
                str(source_video_path),
                "-map",
                f"0:{stream.stream_index}",
                "-ar",
                str(sample_rate_hz),
                "-ac",
                str(channels),
                "-c:a",
                codec,
                "-y",
            ],
            destination,
        )
        return MaterializedMedia(
            path=destination,
            stream=stream,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            output_format=output_format,
        )

    def extract_voice_reference(
        self,
        *,
        clip_uid: str,
        entity_id: str,
        full_audio_path: Path,
        start_time: float,
        end_time: float,
        destination: Path,
        sample_rate_hz: int,
        output_format: str,
        source_audio_path: Path | None = None,
        source_start_sample: int | None = None,
        source_end_sample: int | None = None,
    ) -> Path:
        del clip_uid, entity_id
        codec = "flac" if output_format == "flac" else "pcm_s16le"
        exact_values = (
            source_audio_path,
            source_start_sample,
            source_end_sample,
        )
        if any(value is not None for value in exact_values):
            if not all(value is not None for value in exact_values):
                raise ValueError("exact voice-reference extraction requires all sample inputs")
            assert source_audio_path is not None
            assert source_start_sample is not None
            assert source_end_sample is not None
            destination.parent.mkdir(parents=True, exist_ok=True)
            slice_path = destination.with_name(
                f".{destination.stem}.pcm-slice-{uuid.uuid4().hex}.wav"
            )
            try:
                self._write_pcm16_mono_slice(
                    source=source_audio_path,
                    destination=slice_path,
                    sample_rate_hz=sample_rate_hz,
                    start_sample=source_start_sample,
                    end_sample=source_end_sample,
                )
                self._publish_command(
                    [
                        self.ffmpeg,
                        "-v",
                        "error",
                        "-i",
                        str(slice_path),
                        "-ar",
                        str(sample_rate_hz),
                        "-ac",
                        "1",
                        "-c:a",
                        codec,
                        "-y",
                    ],
                    destination,
                )
            finally:
                slice_path.unlink(missing_ok=True)
            return destination
        self._publish_command(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-ss",
                f"{start_time:.9f}",
                "-to",
                f"{end_time:.9f}",
                "-i",
                str(full_audio_path),
                "-ar",
                str(sample_rate_hz),
                "-ac",
                "1",
                "-c:a",
                codec,
                "-y",
            ],
            destination,
        )
        return destination


class ExternalSubprocessEmbeddingBackend:
    """One-request JSON contract for optional server-only embedding adapters."""

    def __init__(
        self,
        *,
        executable: list[str],
        model_identifier: str,
        checkpoint_sha256: str | None,
        timeout_seconds: float,
        diagnostics_root: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not executable or timeout_seconds <= 0:
            raise ValueError("external embedding backend requires command and timeout")
        self.executable = executable
        self.model_identifier = model_identifier
        self.checkpoint_sha256 = checkpoint_sha256
        self.timeout_seconds = timeout_seconds
        self.diagnostics_root = diagnostics_root
        self.environment = environment or {}

    def _invoke(
        self,
        payload: dict[str, object],
        request_id: str,
    ) -> tuple[EmbeddingResult, dict[str, object]]:
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            self.executable,
            input=json.dumps(payload, separators=(",", ":")) + "\n",
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env={**os.environ, **self.environment},
            check=False,
        )
        (self.diagnostics_root / f"{request_id}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (self.diagnostics_root / f"{request_id}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"embedding subprocess failed for {request_id}")
        response = json.loads(completed.stdout)
        if response.get("request_id") != request_id:
            raise ValueError("embedding response request_id mismatch")
        vector = np.asarray(response["embedding"], dtype=np.float32)
        vector_norm = float(np.linalg.norm(vector.reshape(-1)))
        if (
            response.get("dimension") != vector.size
            or response.get("dtype") != "float32"
            or response.get("normalized") is not True
            or vector.size == 0
            or not np.isfinite(vector).all()
            or not np.isclose(vector_norm, 1.0, rtol=0, atol=1e-4)
        ):
            raise ValueError("embedding response metadata mismatch")
        return (
            EmbeddingResult(
                vector=vector,
                model_identifier=self.model_identifier,
                checkpoint_sha256=self.checkpoint_sha256,
                backend_metadata={
                    "backend": "external_subprocess",
                    "executable": list(self.executable),
                },
            ),
            response,
        )

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        request_id = hashlib.sha256(
            f"face\0{entity_occurrence_id}".encode()
        ).hexdigest()[:20]
        embedding, response = self._invoke(
            {
                "request_id": request_id,
                "operation": "face_embedding",
                "image_path": str(image_path),
                "model_identifier": self.model_identifier,
            },
            request_id,
        )
        crop_path_value = response.get("face_crop_path")
        if not isinstance(crop_path_value, str) or not crop_path_value.strip():
            raise ValueError("face embedding response requires face_crop_path")
        crop_path = Path(crop_path_value).resolve(strict=True)
        with Image.open(crop_path) as opened:
            opened.load()
            crop = opened.convert("RGB")
        return FaceEmbeddingResult(
            status="available",
            embedding=embedding,
            face_crop=crop,
        )

    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult:
        request_id = hashlib.sha256(entity_occurrence_id.encode()).hexdigest()[:20]
        return self._invoke(
            {
                "request_id": request_id,
                "operation": "speaker_embedding",
                "audio_path": str(audio_path),
                "model_identifier": self.model_identifier,
            },
            request_id,
        )[0]

    def embed_text(
        self,
        *,
        entity_occurrence_id: str,
        text: str,
    ) -> EmbeddingResult:
        request_id = hashlib.sha256(
            f"{entity_occurrence_id}\0{text}".encode()
        ).hexdigest()[:20]
        return self._invoke(
            {
                "request_id": request_id,
                "operation": "identity_text_embedding",
                "text": text,
                "model_identifier": self.model_identifier,
            },
            request_id,
        )[0]


class PersistentSubprocessEmbeddingBackend:
    """Long-lived JSONL embedding adapter that loads one model per process."""

    def __init__(
        self,
        *,
        executable: list[str],
        model_identifier: str,
        checkpoint_sha256: str,
        timeout_seconds: float,
        diagnostics_root: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not executable or timeout_seconds <= 0:
            raise ValueError("persistent embedding backend requires command and timeout")
        if len(checkpoint_sha256) != 64:
            raise ValueError("persistent embedding backend requires model fingerprint")
        executable_path = Path(executable[0]).expanduser()
        if executable_path.parent != Path(".") and not executable_path.exists():
            raise FileNotFoundError(f"embedding Python executable is missing: {executable[0]}")
        self.executable = list(executable)
        self.model_identifier = model_identifier
        self.checkpoint_sha256 = checkpoint_sha256
        self.timeout_seconds = timeout_seconds
        self.diagnostics_root = diagnostics_root
        self.environment = environment or {}
        self._process: subprocess.Popen[str] | None = None
        self._stderr_stream: IO[str] | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        stderr_stream = (self.diagnostics_root / "worker.stderr.log").open(
            "w",
            encoding="utf-8",
        )
        try:
            process = subprocess.Popen(
                self.executable,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_stream,
                text=True,
                bufsize=1,
                env={**os.environ, **self.environment},
            )
        except Exception:
            stderr_stream.close()
            raise
        if process.stdin is None or process.stdout is None:
            process.terminate()
            stderr_stream.close()
            raise RuntimeError("persistent embedding worker pipes are unavailable")
        self._process = process
        self._stderr_stream = stderr_stream

    def _readline(self) -> str:
        assert self._process is not None
        assert self._process.stdout is not None
        ready, _, _ = select.select(
            [self._process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            raise TimeoutError("persistent embedding worker response timed out")
        line = self._process.stdout.readline()
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(
                "persistent embedding worker exited without a response "
                f"(returncode={return_code})"
            )
        return line

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        request_id = str(payload["request_id"])
        try:
            self._process.stdin.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("persistent embedding worker request failed") from exc
        try:
            response = json.loads(self._readline())
        except json.JSONDecodeError as exc:
            raise ValueError("persistent embedding worker returned invalid JSON") from exc
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise ValueError("persistent embedding worker response request_id mismatch")
        return response

    def _embedding(self, response: dict[str, object]) -> EmbeddingResult:
        if response.get("status") != "available":
            reason = str(response.get("reason") or "embedding_runtime_failed")
            raise RuntimeError(reason)
        if response.get("model_identifier") != self.model_identifier:
            raise ValueError("persistent embedding worker model identifier mismatch")
        if response.get("model_fingerprint") != self.checkpoint_sha256:
            raise ValueError("persistent embedding worker model fingerprint mismatch")
        vector = np.asarray(response.get("embedding"), dtype=np.float32).reshape(-1)
        if (
            vector.size == 0
            or not np.isfinite(vector).all()
            or response.get("dimension") != vector.size
            or response.get("dtype") != "float32"
        ):
            raise ValueError("persistent embedding worker vector is invalid")
        metadata = response.get("backend_metadata")
        if not isinstance(metadata, dict):
            raise TypeError("persistent embedding worker metadata is invalid")
        return EmbeddingResult(
            vector=vector,
            model_identifier=self.model_identifier,
            checkpoint_sha256=self.checkpoint_sha256,
            backend_metadata={
                **metadata,
                "backend": "persistent_subprocess",
            },
        )

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        request_id = hashlib.sha256(
            f"face\0{entity_occurrence_id}".encode()
        ).hexdigest()[:20]
        crop_path = self.diagnostics_root / "face_crops" / f"{request_id}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        response = self._request(
            {
                "request_id": request_id,
                "operation": "face_embedding",
                "entity_occurrence_id": entity_occurrence_id,
                "image_path": str(image_path),
                "face_crop_output_path": str(crop_path),
                "model_identifier": self.model_identifier,
            }
        )
        status = response.get("status")
        if status == "unavailable":
            return FaceEmbeddingResult(
                status="unavailable",
                reason=str(response.get("reason") or "face_not_found"),
            )
        embedding = self._embedding(response)
        returned_crop = Path(str(response.get("face_crop_path", ""))).resolve(
            strict=True
        )
        if returned_crop != crop_path.resolve(strict=True):
            raise ValueError("persistent face worker crop path mismatch")
        with Image.open(returned_crop) as opened:
            opened.load()
            crop = opened.convert("RGB")
        return FaceEmbeddingResult(
            status="available",
            embedding=embedding,
            face_crop=crop,
        )

    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult:
        request_id = hashlib.sha256(
            f"speaker\0{entity_occurrence_id}".encode()
        ).hexdigest()[:20]
        response = self._request(
            {
                "request_id": request_id,
                "operation": "speaker_embedding",
                "entity_occurrence_id": entity_occurrence_id,
                "audio_path": str(audio_path),
                "model_identifier": self.model_identifier,
            }
        )
        return self._embedding(response)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._request(
                        {
                            "request_id": "shutdown",
                            "operation": "shutdown",
                        }
                    )
                except (
                    BrokenPipeError,
                    OSError,
                    RuntimeError,
                    TimeoutError,
                    ValueError,
                ):
                    process.terminate()
                try:
                    process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=self.timeout_seconds)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if self._stderr_stream is not None:
                self._stderr_stream.close()
            self._process = None
            self._stderr_stream = None


class ExternalSubprocessTranscriptBackend:
    """Optional ASR bridge with bounded execution and explicit diagnostics."""

    def __init__(
        self,
        *,
        executable: list[str],
        model_identifier: str,
        timeout_seconds: float,
        diagnostics_root: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not executable or timeout_seconds <= 0 or not model_identifier.strip():
            raise ValueError("external transcript backend configuration is invalid")
        self.executable = executable
        self.model_identifier = model_identifier
        self.timeout_seconds = timeout_seconds
        self.diagnostics_root = diagnostics_root
        self.environment = environment or {}

    def transcript(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
    ) -> list[dict[str, object]]:
        request_id = hashlib.sha256(f"asr\0{clip_uid}".encode()).hexdigest()[:20]
        payload = {
            "request_id": request_id,
            "operation": "transcript",
            "audio_path": str(audio_path),
            "model_identifier": self.model_identifier,
        }
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            self.executable,
            input=json.dumps(payload, separators=(",", ":")) + "\n",
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env={**os.environ, **self.environment},
            check=False,
        )
        (self.diagnostics_root / f"{request_id}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (self.diagnostics_root / f"{request_id}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"transcript subprocess failed for {clip_uid}")
        response = json.loads(completed.stdout)
        if response.get("request_id") != request_id:
            raise ValueError("transcript response request_id mismatch")
        segments = response.get("segments")
        if not isinstance(segments, list):
            raise TypeError("transcript response segments must be a list")
        return [dict(item) for item in segments if isinstance(item, dict)]
