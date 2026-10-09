"""Thin stage adapters for V3 Group inputs; no legacy inventory/media scans."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoReferenceSelection,
    MimoSegmentEvidence,
)
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract
from r2v_data_v2.h3.ra2va_group_source import GroupTask
from r2v_data_v2.h3.schemas import SchemaModel


class GroupPicture(SchemaModel):
    image_index: int
    picture_label: str
    source_image_index: int
    source_image_id: str
    source_image_label: str
    image_artifact_path: str
    kind: str
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None


class GroupFrozenJob(SchemaModel):
    clip_uid: str
    r2v_instruction: str
    target_video_path: str
    target_full_audio_path: str
    target_duration_seconds: float
    reference_images: list[GroupPicture]
    reference_subjects: list[RecaptionSubjectContract]
    reference_selection: MimoReferenceSelection
    segments: list[MimoSegmentEvidence]
    source_h3_sample_ids: list[str]
    binding_evidence_mode: str = "none"


def frozen_job(task: GroupTask, canonical: dict, segments: list[dict]) -> GroupFrozenJob:
    pictures = [GroupPicture(
        **{key: p.get(key) for key in ("image_index", "picture_label", "kind", "entity_id",
                                      "attribute_id", "owner_entity_id", "attribute_type")},
        source_image_index=p["image_index"], source_image_id=p["image_id"],
        source_image_label=f"<Image {p['image_index']}>", image_artifact_path=p["image_path"],
    ) for p in task.pictures]
    selection = MimoReferenceSelection(
        original_picture_count=len(pictures), selected_picture_count=len(pictures),
        selected_source_image_ids=[p.source_image_id for p in pictures],
        selected_source_image_indexes=[p.source_image_index for p in pictures],
        dropped_references=[], face_promotions=[],
    )
    return GroupFrozenJob(
        clip_uid=task.clip_uid, r2v_instruction=task.sample["r2v_instruction"],
        target_video_path=task.target_video_path, target_full_audio_path=canonical["path"],
        target_duration_seconds=canonical["frame_count"] / 32000,
        reference_images=pictures, reference_subjects=task.subjects,
        reference_selection=selection, segments=segments, source_h3_sample_ids=[task.sample_id],
    )


def audio_metadata(path: Path) -> dict:
    import soundfile as sf

    info = sf.info(path)
    return {"path": str(path), "frame_count": info.frames, "sample_rate_hz": info.samplerate,
            "channels": info.channels, "sample_format": info.subtype}


class GroupStageWorker:
    """One persistent backend context per stage; shared control is not written here."""

    def __init__(self, stage, artifact_root, *, backend_factory, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.stage, self.root = stage, Path(artifact_root)
        self.backend_factory = backend_factory
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    def __enter__(self):
        self.context = (nullcontext(None) if self.stage == "resolve"
                        else self.backend_factory(self.stage))
        self.backend = self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def _stem(self, raw: Path, destination: Path) -> dict:
        FFmpegAudioMediaBackend._publish_command([
            self.ffmpeg, "-nostdin", "-v", "error", "-i", str(raw), "-map", "0:a:0",
            "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", "-y",
        ], destination)
        return audio_metadata(destination)

    def process(self, task: GroupTask, upstream: dict[str, dict]) -> dict:
        output = self.root / self.stage / task.clip_uid
        output.mkdir(parents=True, exist_ok=True)
        if self.stage == "canonical":
            path = output / "full.flac"
            self.backend.materialize_full_audio(
                clip_uid=task.clip_uid, source_video_path=Path(task.target_video_path), destination=path,
                sample_rate_hz=32000, channels=2, output_format="flac",
            )
            return audio_metadata(path)
        if self.stage == "sam":
            source = Path(upstream["canonical"]["path"])
            outcomes = []
            for prompt, target, residual in (("Music", "music", "residual"), ("Speech", "speech", "sfx")):
                raw, remainder = output / f"{target}.raw.wav", output / f"{residual}.raw.wav"
                result = self.backend.separate(
                    clip_uid=task.clip_uid, source_audio_path=source, prompt=prompt,
                    target_path=raw, residual_path=remainder,
                )
                if result.verification_state != "success":
                    raise ValueError(f"SAM {prompt} separation: {result.verification_state}")
                outcomes.append(result.verification_state)
                source = remainder
            return {**{kind: self._stem(output / f"{kind}.raw.wav", output / f"{kind}.wav")
                       for kind in ("speech", "music", "sfx")}, "separation_state": "success",
                    "model_call_count": len(outcomes)}
        if self.stage == "auk":
            from r2v_data_v2.h3.auk_speech_shadow import canonicalize_speech

            canonical = upstream["canonical"]
            source = SimpleNamespace(clip_uid=task.clip_uid, source_audio_path=canonical["path"],
                source_audio_sha256=None, source_duration_seconds=canonical["frame_count"] / 32000)
            raw = output / "speech.raw.wav"
            response = self.backend.generate(source, raw)
            if response["status"] != "ok":
                raise ValueError(response.get("reason", "AuK inference failed"))
            path = output / "speech.wav"
            adjustment = canonicalize_speech(raw, path, canonical["frame_count"], ffmpeg=self.ffmpeg)
            return {"speech": audio_metadata(path), "response": response,
                    "canonical_adjustment_samples": adjustment, "model_call_count": 1}
        if self.stage == "resolve":
            sam, auk = upstream["sam"], upstream["auk"]
            return {"speech": auk["speech"], "music": sam["music"], "sfx": sam["sfx"],
                    "separation_state": sam["separation_state"]}
        if self.stage == "diarizen":
            from r2v_data_v2.h3.diarization_binding import DiarizationBackendSegment

            audio = upstream["resolve"]["speech"]
            rows = [DiarizationBackendSegment.model_validate(row) for row in self.backend.diarize(
                clip_uid=task.clip_uid, audio_path=Path(audio["path"]),
            )]
            clusters, segments = {}, []
            for row in sorted(rows, key=lambda r: (r.start_time, r.end_time, r.speaker_label)):
                start, end = round(row.start_time * 32000), min(round(row.end_time * 32000), audio["frame_count"])
                if start >= audio["frame_count"] or end <= start:
                    raise ValueError("diarization segment has no positive canonical source intersection")
                group = clusters.setdefault(row.speaker_label, f"speaker_{len(clusters)}")
                segments.append({"segment_id": f"segment_{len(segments) + 1:04d}",
                    "start_time": start / 32000, "end_time": end / 32000,
                    "source_start_sample": start, "source_end_sample": end, "source_sample_rate_hz": 32000,
                    "source_speaker_cluster_id": group, "identity_scope": "unresolved",
                    "direct_anchor_seconds": 0, "cluster_binding_status": "unbound",
                    "overlapping_visible_entities": [], "direct_support_seconds_by_entity": {},
                    "competing_visible_speaker_evidence": []})
            return {"segments": segments}
        if self.stage == "asr":
            from r2v_data_v2.h3.qwen3_asr import load_qwen3_asr_model_input

            segments = []
            for segment in upstream["diarizen"]["segments"]:
                waveform, rate = load_qwen3_asr_model_input(Path(upstream["resolve"]["speech"]["path"]),
                    segment["start_time"], segment["end_time"], ffmpeg=self.ffmpeg)
                text, language = self.backend.transcribe(waveform=waveform, sample_rate_hz=rate)
                segments.append({**segment, "asr_status": "transcribed" if text.strip() else "empty",
                    "asr_text": text if text.strip() else None, "asr_language": language if text.strip() else None})
            return {"job": frozen_job(task, upstream["canonical"], segments).model_dump(mode="json")}
        raise ValueError(f"unsupported Group stage: {self.stage}")
