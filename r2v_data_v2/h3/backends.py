from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from r2v_data_v2.h3.schemas import (
    ActiveSpeakerInterval,
    AudioBindingEvidence,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceTrack,
    PrecomputedEvidenceFile,
    VoiceReferenceCandidate,
)
from r2v_data_v2.v3.schemas import AnnotationEntity, ClipRecord


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
        clip_uid: str,
        entities: Sequence[AnnotationEntity],
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
    def propose(
        self,
        *,
        clip_uid: str,
        audio: AudioTrackMetadata,
        intervals: Sequence[ActiveSpeakerInterval],
    ) -> Sequence[VoiceReferenceCandidate]: ...


class AudioBindingEvidenceBackend(Protocol):
    def collect(self, clip: ClipRecord) -> AudioBindingEvidence: ...


class PrecomputedEvidenceBackend:
    """Strict GPU-free evidence provider for tests and sidecar replay."""

    def __init__(self, evidence: PrecomputedEvidenceFile) -> None:
        self._by_clip = {item.clip_uid: item for item in evidence.clips}

    @classmethod
    def from_path(cls, path: Path) -> PrecomputedEvidenceBackend:
        return cls(PrecomputedEvidenceFile.model_validate_json(path.read_text("utf-8")))

    def collect(self, clip: ClipRecord) -> AudioBindingEvidence:
        try:
            return self._by_clip[clip.clip_uid]
        except KeyError as exc:
            raise KeyError(f"missing audio evidence for clip {clip.clip_uid}") from exc
