from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from r2v_data_v2.h3.schemas import (
    ActiveSpeakerInterval,
    AudioBindingEvidence,
    AudioEntityBinding,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceTrack,
    PrecomputedEvidenceFile,
    VoiceReferenceCandidate,
)
from r2v_data_v2.h3.visual_clip_contract import VisualClipRecord


class AudioPreprocessorBackend(Protocol):
    def inspect(self, *, clip_uid: str, video_path: Path) -> AudioTrackMetadata: ...


class FaceTrackingBackend(Protocol):
    def track(
        self,
        *,
        clip_uid: str,
        video_path: Path,
    ) -> Sequence[FaceTrack]: ...


class EntityFaceAssociationBackend(Protocol):
    def associate(
        self,
        *,
        clip: VisualClipRecord,
        source_run_root: Path,
        tracked_masks_path: Path,
        face_tracks: Sequence[FaceTrack],
    ) -> Sequence[EntityFaceAssociation]: ...


class ActiveSpeakerBackend(Protocol):
    def score(
        self,
        *,
        clip_uid: str,
        video_path: Path,
        audio: AudioTrackMetadata,
        face_tracks: Sequence[FaceTrack],
    ) -> Sequence[ActiveSpeakerInterval]: ...


class VoiceReferenceBackend(Protocol):
    def extract(
        self,
        *,
        clip_uid: str,
        audio: AudioTrackMetadata,
        clean_bindings: Sequence[AudioEntityBinding],
    ) -> Sequence[VoiceReferenceCandidate]: ...


class AudioBindingEvidenceBackend(Protocol):
    def collect(self, clip: VisualClipRecord) -> AudioBindingEvidence: ...


class PrecomputedEvidenceBackend:
    """Strict GPU-free evidence provider for tests and sidecar replay."""

    def __init__(self, evidence: PrecomputedEvidenceFile) -> None:
        self._by_clip = {item.evidence.clip_uid: item for item in evidence.clips}

    @classmethod
    def from_path(cls, path: Path) -> PrecomputedEvidenceBackend:
        return cls(PrecomputedEvidenceFile.model_validate_json(path.read_text("utf-8")))

    def collect(self, clip: VisualClipRecord) -> AudioBindingEvidence:
        try:
            return self._by_clip[clip.clip_uid].evidence
        except KeyError as exc:
            raise KeyError(f"missing audio evidence for clip {clip.clip_uid}") from exc

    def extract(
        self,
        *,
        clip_uid: str,
        audio: AudioTrackMetadata,
        clean_bindings: Sequence[AudioEntityBinding],
    ) -> Sequence[VoiceReferenceCandidate]:
        try:
            record = self._by_clip[clip_uid]
        except KeyError as exc:
            raise KeyError(f"missing voice evidence for clip {clip_uid}") from exc
        if record.evidence.audio != audio:
            raise ValueError(
                "voice extraction audio does not match precomputed evidence"
            )
        if any(
            binding.status != "bound" or not binding.evidence.clean_training_eligible
            for binding in clean_bindings
        ):
            raise ValueError("voice extraction requires clean bound intervals")
        return record.voice_reference_candidates
