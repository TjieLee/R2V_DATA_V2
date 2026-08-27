from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from r2v_data_v2.h3.backends import VoiceReferenceBackend
from r2v_data_v2.h3.schemas import (
    ActiveSpeakerInterval,
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioEntityBinding,
    BindingEvidence,
    BindingStatus,
    H3AudioAsset,
    H3AudioBindingIR,
    H3TaskSpecification,
    PictureAsset,
    SemanticSubject,
    VoiceReference,
    VoiceReferenceCandidate,
)
from r2v_data_v2.h3.visual_clip_contract import VisualClipRecord


@dataclass(frozen=True)
class AudioBindingPolicy:
    active_speaker_probability: float = 0.80
    offscreen_probability_ceiling: float = 0.20
    minimum_top_score_margin: float = 0.15
    minimum_association_confidence: float = 0.80
    minimum_clean_speech_seconds: float = 0.50
    minimum_voice_reference_quality: float = 0.70
    minimum_asd_coverage: float = 1.0

    def __post_init__(self) -> None:
        probabilities = (
            self.active_speaker_probability,
            self.offscreen_probability_ceiling,
            self.minimum_top_score_margin,
            self.minimum_association_confidence,
            self.minimum_voice_reference_quality,
            self.minimum_asd_coverage,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("audio binding probability thresholds must be in [0, 1]")
        if self.offscreen_probability_ceiling >= self.active_speaker_probability:
            raise ValueError("offscreen ceiling must be below active threshold")
        if self.minimum_clean_speech_seconds <= 0:
            raise ValueError("minimum clean speech duration must be positive")


def _binding(
    interval: ActiveSpeakerInterval,
    *,
    status: BindingStatus,
    confidence: float,
    active_face_track_ids: list[str],
    reason_codes: list[str],
    entity_id: str | None = None,
    face_track_id: str | None = None,
    association_confidence: float | None = None,
    clean_training_eligible: bool = False,
) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=interval.start_time,
        end_time=interval.end_time,
        entity_id=entity_id,
        face_track_id=face_track_id,
        status=status,
        confidence=max(0.0, min(1.0, confidence)),
        evidence=BindingEvidence(
            active_face_track_ids=active_face_track_ids,
            face_speaking_probabilities=dict(
                sorted(interval.face_speaking_probabilities.items())
            ),
            association_confidence=association_confidence,
            audio_quality_usable=interval.audio_quality_usable,
            synchronization_plausible=interval.synchronization_plausible,
            clean_training_eligible=clean_training_eligible,
            reason_codes=reason_codes,
        ),
    )


def _native_decision_reason(
    interval: ActiveSpeakerInterval,
) -> str | None:
    semantics = {score.score_semantics for score in interval.face_scores}
    if not semantics:
        return None
    if semantics == {"laser_loconet_native_score"}:
        return "laser_asd_native_decision_unvalidated"
    return "lr_asd_native_decision_unvalidated"


def fuse_audio_entity_bindings(
    evidence: AudioBindingEvidence,
    *,
    known_entity_ids: set[str],
    policy: AudioBindingPolicy | None = None,
) -> list[AudioEntityBinding]:
    active_policy = policy or AudioBindingPolicy()
    associations = {item.face_track_id: item for item in evidence.associations}
    unknown_entities = {
        item.entity_id
        for item in evidence.associations
        if item.status == "matched"
        and item.entity_id is not None
        and item.entity_id not in known_entity_ids
    }
    if unknown_entities:
        raise ValueError(
            f"face associations reference unknown V3 entities: {sorted(unknown_entities)}"
        )
    bindings: list[AudioEntityBinding] = []
    for interval in evidence.active_speaker_intervals:
        probabilities = sorted(
            interval.face_speaking_probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if not interval.speech_present:
            bindings.append(
                _binding(
                    interval,
                    status="no_speech",
                    confidence=1.0,
                    active_face_track_ids=[],
                    reason_codes=["no_speech_detected"],
                )
            )
            continue
        if interval.asd_coverage_ratio < active_policy.minimum_asd_coverage:
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=0.0,
                    active_face_track_ids=[],
                    reason_codes=["asd_visible_face_coverage_incomplete"],
                )
            )
            continue
        native_scores = {
            score.face_track_id: score for score in interval.face_scores
        }
        uses_native_decision = bool(native_scores)
        native_decision_reason = _native_decision_reason(interval)
        if uses_native_decision:
            active = sorted(
                face_track_id
                for face_track_id, score in native_scores.items()
                if score.backend_native_active
            )
        else:
            active = [
                face_track_id
                for face_track_id, score in probabilities
                if score >= active_policy.active_speaker_probability
            ]
        if len(active) >= 2:
            if uses_native_decision:
                normalized_scores = [
                    native_scores[face_track_id].normalized_score
                    for face_track_id in active
                ]
                confidence = (
                    min(score for score in normalized_scores if score is not None)
                    if all(score is not None for score in normalized_scores)
                    else 0.0
                )
            else:
                confidence = min(
                    interval.face_speaking_probabilities[face_track_id]
                    for face_track_id in active
                )
            bindings.append(
                _binding(
                    interval,
                    status="overlap",
                    confidence=confidence,
                    active_face_track_ids=active,
                    reason_codes=[
                        "multiple_visible_speakers_active",
                        *(
                            [native_decision_reason]
                            if native_decision_reason is not None
                            else []
                        ),
                    ],
                )
            )
            continue
        if uses_native_decision and not active:
            bindings.append(
                _binding(
                    interval,
                    status="offscreen",
                    confidence=0.0,
                    active_face_track_ids=[],
                    reason_codes=[
                        "speech_without_visible_active_speaker",
                        native_decision_reason,
                    ],
                )
            )
            continue
        if uses_native_decision:
            top_face = active[0]
            normalized_score = native_scores[top_face].normalized_score
            top_score = normalized_score if normalized_score is not None else 0.0
            second_score = 0.0
        else:
            top_face, top_score = probabilities[0] if probabilities else (None, 0.0)
            second_score = probabilities[1][1] if len(probabilities) > 1 else 0.0
        if (
            not uses_native_decision
            and top_score <= active_policy.offscreen_probability_ceiling
        ):
            bindings.append(
                _binding(
                    interval,
                    status="offscreen",
                    confidence=1.0 - top_score,
                    active_face_track_ids=[],
                    reason_codes=["speech_without_visible_active_speaker"],
                )
            )
            continue
        if not uses_native_decision and (
            top_face is None
            or top_score < active_policy.active_speaker_probability
            or top_score - second_score < active_policy.minimum_top_score_margin
        ):
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=top_score,
                    active_face_track_ids=[] if top_face is None else [top_face],
                    reason_codes=["active_speaker_scores_ambiguous"],
                )
            )
            continue
        association = associations.get(top_face)
        if association is None:
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=top_score,
                    active_face_track_ids=[top_face],
                    reason_codes=["face_track_entity_association_missing"],
                )
            )
            continue
        if association.status != "matched" or association.entity_id is None:
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=top_score,
                    active_face_track_ids=[top_face],
                    reason_codes=[
                        "face_track_entity_association_ambiguous"
                        if association.status == "ambiguous"
                        else "face_track_entity_association_unmatched"
                    ],
                )
            )
            continue
        if association.confidence < active_policy.minimum_association_confidence:
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=min(top_score, association.confidence),
                    active_face_track_ids=[top_face],
                    association_confidence=association.confidence,
                    reason_codes=["face_track_entity_association_low_confidence"],
                )
            )
            continue
        quality_reasons: list[str] = []
        if not interval.audio_quality_usable:
            quality_reasons.append("audio_quality_unusable")
        if not interval.synchronization_plausible:
            quality_reasons.append("audiovisual_sync_implausible")
        if quality_reasons:
            bindings.append(
                _binding(
                    interval,
                    status="ambiguous",
                    confidence=min(top_score, association.confidence),
                    active_face_track_ids=[top_face],
                    association_confidence=association.confidence,
                    reason_codes=quality_reasons,
                )
            )
            continue
        duration = interval.end_time - interval.start_time
        clean = duration >= active_policy.minimum_clean_speech_seconds
        binding_confidence = (
            association.confidence
            if uses_native_decision
            else min(top_score, association.confidence)
        )
        bindings.append(
            _binding(
                interval,
                status="bound",
                confidence=binding_confidence,
                active_face_track_ids=[top_face],
                association_confidence=association.confidence,
                reason_codes=[
                    *([] if clean else ["speech_interval_too_short"]),
                    *(
                        [native_decision_reason]
                        if native_decision_reason is not None
                        else []
                    ),
                ],
                entity_id=association.entity_id,
                face_track_id=top_face,
                clean_training_eligible=clean,
            )
        )
    return bindings


def select_voice_references(
    candidates: Sequence[VoiceReferenceCandidate],
    bindings: list[AudioEntityBinding],
    *,
    entity_order: list[str],
    eligible_entity_ids: set[str],
    policy: AudioBindingPolicy | None = None,
) -> list[VoiceReference]:
    active_policy = policy or AudioBindingPolicy()
    clean_by_entity: dict[str, list[AudioEntityBinding]] = {}
    for binding in bindings:
        if binding.entity_id is not None and binding.evidence.clean_training_eligible:
            clean_by_entity.setdefault(binding.entity_id, []).append(binding)
    selected: list[VoiceReference] = []
    for entity_id in entity_order:
        if entity_id not in eligible_entity_ids:
            continue
        clean_bindings = clean_by_entity.get(entity_id, [])
        matches = []
        for candidate in candidates:
            if (
                candidate.quality_score < active_policy.minimum_voice_reference_quality
                or candidate.source_end - candidate.source_start
                < active_policy.minimum_clean_speech_seconds
            ):
                continue
            matching = next(
                (
                    binding
                    for binding in clean_bindings
                    if candidate.source_start >= binding.start_time
                    and candidate.source_end <= binding.end_time
                ),
                None,
            )
            if matching is not None:
                matches.append((candidate, matching))
        if not matches:
            continue
        candidate, matching = min(
            matches,
            key=lambda item: (
                -item[0].quality_score,
                -(item[0].source_end - item[0].source_start),
                item[0].source_start,
                item[0].path,
            ),
        )
        selected.append(
            VoiceReference(
                voice_reference_id=f"voice_reference_{len(selected) + 1}",
                entity_id=entity_id,
                path=candidate.path,
                source_start=candidate.source_start,
                source_end=candidate.source_end,
                quality_score=candidate.quality_score,
                quality_metadata=candidate.quality_metadata,
                binding_start=matching.start_time,
                binding_end=matching.end_time,
            )
        )
    return selected


def build_h3_audio_ir(
    clip: VisualClipRecord,
    evidence: AudioBindingEvidence,
    bindings: list[AudioEntityBinding],
    voice_references: list[VoiceReference],
    *,
    task: H3TaskSpecification,
) -> H3AudioBindingIR:
    if clip.annotation is None or clip.annotation.status != "ready":
        raise ValueError("H3 audio IR requires ready annotation")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise ValueError("H3 audio IR requires ready pairing")
    references = {
        reference.entity_id: reference
        for reference in clip.references.entities
        if reference.status == "ready"
    }
    entities = {entity.entity_id: entity for entity in clip.annotation.entities}
    pictures: list[PictureAsset] = []
    subjects: list[SemanticSubject] = []
    for index, entity_id in enumerate(clip.pairing.retained_entity_ids, start=1):
        reference = references.get(entity_id)
        entity = entities.get(entity_id)
        if reference is None or reference.image_path is None or entity is None:
            raise ValueError("paired H3 entity is missing reference evidence")
        picture_id = f"picture_{index}"
        pictures.append(
            PictureAsset(
                picture_id=picture_id,
                entity_id=entity_id,
                path=reference.image_path,
            )
        )
        subjects.append(
            SemanticSubject(
                subject_id=f"subject_{index}",
                entity_id=entity_id,
                reference_type=entity.reference_type,
                phrase=entity.phrase,
                source_assets=[picture_id],
            )
        )
    audio_assets: list[H3AudioAsset] = []
    if "audio_reference" in task.components:
        for voice in voice_references:
            audio_assets.append(
                H3AudioAsset(
                    audio_id=f"audio_{len(audio_assets) + 1}",
                    role="voice_reference",
                    voice_reference_id=voice.voice_reference_id,
                    entity_id=voice.entity_id,
                    path=voice.path,
                    source_start=voice.source_start,
                    source_end=voice.source_end,
                )
            )
    if "audio_reuse" in task.components:
        if evidence.audio.full_audio_path is None:
            raise ValueError("audio_reuse requires full-audio provenance")
        audio_assets.append(
            H3AudioAsset(
                audio_id=f"audio_{len(audio_assets) + 1}",
                role="full_audio",
                path=evidence.audio.full_audio_path,
            )
        )
    return H3AudioBindingIR(
        clip_uid=clip.clip_uid,
        task=task,
        picture_assets=pictures,
        subjects=subjects,
        audio_assets=audio_assets,
        bindings=bindings,
    )


def render_h3_audio_instruction(value: H3AudioBindingIR) -> str:
    lines = [
        f"<Subject {index}> is {subject.phrase} in <Picture {index}>."
        for index, subject in enumerate(value.subjects, start=1)
    ]
    subject_index = {
        subject.entity_id: index
        for index, subject in enumerate(value.subjects, start=1)
    }
    for index, audio in enumerate(value.audio_assets, start=1):
        if audio.role == "voice_reference":
            assert audio.entity_id is not None
            subject = subject_index[audio.entity_id]
            lines.append(
                f"<Audio {index}> is the voice-timbre reference for "
                f"<Subject {subject}> (S{subject})."
            )
        else:
            lines.append(f"<Audio {index}> preserves the full source audio.")
    mode = " + ".join(
        component.replace("_", " ") for component in value.task.components
    )
    lines.extend(("", "summary:", f"[{mode}] deterministic H3 audio binding"))
    return "\n".join(lines)


def build_audio_binding_sidecar(
    clip: VisualClipRecord,
    evidence: AudioBindingEvidence,
    *,
    source_run_root: str,
    voice_reference_backend: VoiceReferenceBackend | None = None,
    task: H3TaskSpecification | None = None,
    policy: AudioBindingPolicy | None = None,
) -> AudioBindingSidecar:
    if evidence.clip_uid != clip.clip_uid:
        raise ValueError("audio evidence clip_uid does not match V3 clip")
    if evidence.audio.source_video_path != clip.source.video_path:
        raise ValueError("audio evidence source video does not match V3 clip")
    if (
        clip.annotation is None
        or clip.annotation.status != "ready"
        or clip.pairing is None
        or clip.pairing.status != "ready"
    ):
        return AudioBindingSidecar(
            clip_uid=clip.clip_uid,
            source_run_root=source_run_root,
            source_video_path=clip.source.video_path,
            status="ineligible",
            reason="visual_state_not_ready",
            evidence=evidence,
        )
    if evidence.audio.status != "ready":
        return AudioBindingSidecar(
            clip_uid=clip.clip_uid,
            source_run_root=source_run_root,
            source_video_path=clip.source.video_path,
            status="failed",
            reason=f"audio_{evidence.audio.status}",
            evidence=evidence,
        )
    known_entities = {entity.entity_id for entity in clip.annotation.entities}
    active_policy = policy or AudioBindingPolicy()
    active_task = task or H3TaskSpecification(
        components=["reference_generation", "audio_reference"]
    )
    bindings = fuse_audio_entity_bindings(
        evidence,
        known_entity_ids=known_entities,
        policy=active_policy,
    )
    voice_references: list[VoiceReference] = []
    if "audio_reference" in active_task.components:
        eligible_voice_entities = {
            entity.entity_id
            for entity in clip.annotation.entities
            if entity.reference_type == "subject"
        }
        clean_bindings = [
            binding
            for binding in bindings
            if binding.entity_id in eligible_voice_entities
            and binding.evidence.clean_training_eligible
        ]
        if not clean_bindings:
            return AudioBindingSidecar(
                clip_uid=clip.clip_uid,
                source_run_root=source_run_root,
                source_video_path=clip.source.video_path,
                status="ineligible",
                reason="no_clean_entity_bound_audio",
                evidence=evidence,
                bindings=bindings,
            )
        if voice_reference_backend is None:
            raise ValueError("audio_reference task requires a voice reference backend")
        candidates = voice_reference_backend.extract(
            clip_uid=clip.clip_uid,
            audio=evidence.audio,
            clean_bindings=clean_bindings,
        )
        assert evidence.audio.duration_seconds is not None
        if any(
            candidate.source_end > evidence.audio.duration_seconds
            for candidate in candidates
        ):
            raise ValueError("voice candidate exceeds source audio duration")
        voice_references = select_voice_references(
            candidates,
            bindings,
            entity_order=clip.pairing.retained_entity_ids,
            eligible_entity_ids=eligible_voice_entities,
            policy=active_policy,
        )
        if not voice_references:
            return AudioBindingSidecar(
                clip_uid=clip.clip_uid,
                source_run_root=source_run_root,
                source_video_path=clip.source.video_path,
                status="ineligible",
                reason="no_usable_voice_reference",
                evidence=evidence,
                bindings=bindings,
            )
    h3_ir = build_h3_audio_ir(
        clip,
        evidence,
        bindings,
        voice_references,
        task=active_task,
    )
    return AudioBindingSidecar(
        clip_uid=clip.clip_uid,
        source_run_root=source_run_root,
        source_video_path=clip.source.video_path,
        status="ready",
        reason=None,
        evidence=evidence,
        bindings=bindings,
        voice_references=voice_references,
        h3_ir=h3_ir,
    )
