"""Opt-in Visual then joint Speech/AV, acoustic profile and audio finalization."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from r2v_data_v2.h3.mimo25_backend import (
    _DIALOGUE,
    AUDIO_FINALIZE_SYSTEM_PROMPT,
    MIMO25_SCHEMA_VERSION,
    SPEAKER_PROFILE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VISUAL_SYSTEM_PROMPT,
    MimoAudioFinalizeDraft,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendJob,
    MimoBackendProvenance,
    MimoBackendResult,
    MimoCompletionDiagnostic,
    MimoSpeakerMarkerPolishAudit,
    MimoSpeakerVoiceProfile,
    MimoSpeechAVAssemblyDraft,
    MimoUsage,
    MimoVisualBlock,
    MimoVisualDraft,
    MimoVisualObservation,
    _canonicalize_raw_annotation_payload,
    _canonicalize_visual_draft_payload,
    _compact_json,
    _completion_diagnostic,
    _official_detailed_description_icl_messages,
    _sha256_text,
    _speaker_profile_targets,
    _validate_av_observation_usage,
    _validate_finish_reason,
    _value,
    direct_speech_facts,
    protect_direct_dialogue,
    validate_annotation,
)
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speaker_ownership import (
    neutralize_two_step_caption_identity,
    speaker_ownership_reasons,
    unconfirmed_visible_binding_segments,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

TWO_STEP_BACKEND_VERSION = "r2v.h3.mimo25_backend.85"
TWO_STEP_PROMPT_VERSION = "h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"
JOINT_INPUT_MODALITY = "target_video_joint_av_audio"


class MimoJointAVAudioDraft(SchemaModel):
    speech_av: MimoSpeechAVAssemblyDraft
    speaker_voice_profiles: list[MimoSpeakerVoiceProfile]
    audio_finalize: MimoAudioFinalizeDraft


# Specialize stage ownership only; frozen Multi prompt constants stay untouched.
_JOINT_SPEECH_PROMPT = SYSTEM_PROMPT.replace(
    "Return one compact MimoSpeechAVAssemblyDraft JSON object.",
    "Return one MimoJointAVAudioDraft JSON object with speech_av, speaker_voice_profiles and audio_finalize.",
).replace(
    "Return only audio_observation, av_grounding, summary, shot1_caption and warnings.",
    "speech_av contains audio_observation, av_grounding, summary, shot1_caption and warnings.",
).replace(
    "Defer ALL non-dialogue audio to Turn 4.",
    "Put ALL non-dialogue audio in audio_finalize, not shot1_caption.",
).replace(
    "Number by first ACTUAL vocal appearance: first actual vocal source -> S1; second distinct vocal source -> S2.",
    "gN and Sx are different namespaces: gN follows all acoustic sources' first appearance; "
    "S1, S2, ... follow distinct final groups' first transcribed appearance. "
    "Non-transcribed-only groups consume no Sx.",
)
_JOINT_PROFILE_PROMPT = SPEAKER_PROFILE_SYSTEM_PROMPT.replace(
    "Listen to the supplied localized speech snippets and return ONLY speaker_voice_profiles for the already-supplied speaker groups, in target order.",
    "Listen to the full speech stem using the frozen segment time windows. "
    "After determining speech_av groups, output speaker_voice_profiles only for distinct non-null "
    "primary_speaker_group values in transcribed segments, in first transcribed appearance order.",
).replace(
    "Each labeled snippet is an exact speech-stem crop for the named speaker_group/speaker_id and interval. ",
    "Use the corresponding segment windows within that stem as localized acoustic evidence. ",
).replace(
    "For each supplied target", "For each resulting profile target",
).replace("supplied snippet", "corresponding speech-stem windows")
_JOINT_AUDIO_PROMPT = AUDIO_FINALIZE_SYSTEM_PROMPT.replace(
    "Return ONLY overall_soundscape and non_diegetic_music.",
    "audio_finalize contains overall_soundscape and non_diegetic_music.",
).replace(
    "Do not output visual prose, dialogue, Subject labels, Sx, summary or shot1_caption.",
    "Do not put visual prose, dialogue, Subject labels, Sx, summary or shot1_caption in audio_finalize.",
)
JOINT_SYSTEM_PROMPT = (
    _JOINT_SPEECH_PROMPT
    + """

SOURCE RESPONSIBILITIES
Frozen Pictures (<Picture N>) establish the stable appearance of their assigned Subjects, Objects and Background. <Subject N> names a frozen visual entity or attribute, gN names an acoustic speaker group, and (Sx) names a final dialogue speaker; these are different namespaces.
Turn 1 Visual supplies the observed scene, composition, actions, expressions, interactions and temporal ordering. DiariZen + Qwen3-ASR supply frozen segments, time windows and exact dialogue. Original AV determines actual vocal sources, speaker binding and spatial presentation. Speech stem supports acoustic traits; Music/SFX stems support soundscape/music judgment, not automatic truth from separator residuals.
Reference images do not determine actions or speaker identity. Do not reclassify Pictures or reassign Subjects, or turn incidental reference-image background into an independent reference requirement.

VISUAL PERFORMANCE FIDELITY
Treat the Turn 1 shot1_visual_description as the authoritative visual action sequence. Preserve all meaningful observable actions, gestures, gaze shifts, expressions, object interactions, body and head movements, interactions between people, camera behavior, composition changes and scene transitions, and their temporal ordering.
Integrate speech, pauses and observable reactions into that sequence with only minimal local grammatical changes. Do not replace a sequence of actions with a generic scene summary or omit intermediate actions merely to shorten the caption. Do not invent unsupported events, psychology, relationships or environment details to increase detail. No minimum word count; do not mechanically repeat stable appearance traits.

VOICE-SOURCE NARRATION CONSISTENCY
Speaker identity and changes come from speech_av grounding, never Picture order, Subject numbering or prominence, or grammatical proximity.
When two dialogue events belong to different final speaker groups, introduce the second event as a distinct vocal source. Never use "he continues", "she continues", "the same speaker", or equivalent continuation language across a speaker change.
Attach (Sx) to a visible Subject only when the corresponding AV grounding explicitly supports that visible entity. Describe the offscreen or unresolved vocal source separately from visible Subjects. When identity is uncertain, use a neutral voice description; do not guess gender or identity.
Do not change, merge or renumber gN/Sx to improve the caption. Preserve the established Sx correspondence and exact ASR text, language markers, punctuation and dialogue order.
shot1_caption owns visual action, speaker presentation and exact dialogue; summary stays a short target-video overview. Environment and physical sounds belong in overall_soundscape, audience-only music in non_diegetic_music, never in shot1_caption. For clips without transcribed ASR, use the Turn 1 Visual caption; do not invent (Sx) or <d>. Continue returning the joint JSON, not rendered six-section H3 text.
"""
    + "\nACOUSTIC VOICE PROFILES\n" + _JOINT_PROFILE_PROMPT
    + "\nWhen supported traits cannot be reliably described, use voice_characteristics=null; never invent a profile. "
    "Profiling must not change speech_av grouping, binding or speaker markers. "
    "If there are no transcribed ASR segments, speaker_voice_profiles=[]; do not create Sx or dialogue. "
    "A non-silent speech stem, laughter or breath is not proof of transcribed speech. "
    "If there are no segments, keep both audio/AV inventories empty and still finalize soundscape/music. "
    "Transcribed ASR remains exact even if acoustic characterization is uncertain."
    + "\n\nNON-DIALOGUE AUDIO FINALIZATION\n" + _JOINT_AUDIO_PROMPT
)


_PICTURE_LABEL = re.compile(r"<Picture [1-9]\d*>")
_PICTURE_SOURCE_PHRASE = re.compile(
    r"(?:,?\s+|^)(?:as shown in|shown in|depicted in|with its visual detail sourced from)\s+<Picture [1-9]\d*>"
)
_CLOTHING_SOURCE_PREFIX = re.compile(
    r"^The upper-clothing attribute (?:shown|depicted) in <Picture [1-9]\d*>,\s+(.+)$"
)
_FACE_SOURCE_PHRASE = re.compile(r"^(The face of [^<>]+?) in <Picture [1-9]\d*>(, showing .+)$")


def _canonicalize_two_step_visual_payload(raw: str, job: MimoBackendJob) -> tuple[str, dict[str, int]]:
    canonical, prior_corrections = _canonicalize_visual_draft_payload(raw)
    try:
        payload = json.loads(normalize_structured_json_envelope(canonical))
    except (TypeError, ValueError):
        return canonical, prior_corrections
    definitions = payload.get("subject_definitions") if isinstance(payload, dict) else None
    if not isinstance(definitions, list):
        return canonical, prior_corrections
    subjects = {subject.subject_label: subject for subject in job.reference_subjects}
    corrections = Counter(prior_corrections)
    for row in definitions:
        if not isinstance(row, dict) or not isinstance(row.get("description"), str):
            continue
        subject = subjects.get(row.get("subject_label"))
        description = row["description"]
        pictures = set(_PICTURE_LABEL.findall(description))
        if subject is None or not pictures or not pictures.issubset(subject.source_picture_labels):
            continue
        cleaned = description
        if subject.kind == "attribute":
            prefix = _CLOTHING_SOURCE_PREFIX.fullmatch(cleaned)
            if prefix:
                cleaned = prefix[1][0].upper() + prefix[1][1:]
            cleaned = _FACE_SOURCE_PHRASE.sub(r"\1\2", cleaned)
        cleaned = _PICTURE_SOURCE_PHRASE.sub("", cleaned)
        if cleaned != description:
            cleaned = cleaned.strip(" ,;:\t\r\n")
            row["description"] = cleaned if any(char.isalnum() for char in cleaned) else ""
            corrections["visual_subject_picture_provenance_removed"] += 1
    return _compact_json(payload), dict(corrections)


def _canonicalize_joint_payload(
    raw: str, job: MimoBackendJob, *, warnings: list[str],
) -> tuple[str, dict[str, int]]:
    try:
        payload = json.loads(normalize_structured_json_envelope(raw))
    except (TypeError, ValueError):
        return raw, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("speech_av"), dict):
        return raw, {}
    audio = payload["speech_av"].get("audio_observation")
    if isinstance(audio, dict) and isinstance(audio.get("segment_decisions"), list):
        seen = {}
        for row in audio["segment_decisions"]:
            if not isinstance(row, dict) or not isinstance(row.get("segment_id"), str):
                continue  # Malformed rows still go through the existing schema parser.
            segment_id = row["segment_id"]
            if segment_id in seen and row != seen[segment_id]:
                raise MimoBackendFailure(
                    code="mimo_structured_output_failed", reason="MiMo joint has conflicting Audio segment decisions",
                    issues=(ValidationIssue("segment_inventory_mismatch", segment_id,
                                            "conflicting duplicate Audio segment decisions"),),
                )
            seen[segment_id] = row
    canonical, corrections = _canonicalize_raw_annotation_payload(_compact_json(payload["speech_av"]))
    payload["speech_av"] = json.loads(canonical)
    rows = payload["speech_av"].get("warnings")
    if not job.segments and isinstance(rows, list):
        retained = [row for row in rows if row != {"code": "possible_asr_conflict", "segment_id": None}]
        removed = len(rows) - len(retained)
        if removed:
            payload["speech_av"]["warnings"] = retained
            corrections["unscoped_possible_asr_conflict_no_segments"] = removed
            warnings.append("unscoped_possible_asr_conflict_no_segments")
    return _compact_json(payload), corrections


def _required_dialogue_blocks(job: MimoBackendJob) -> list[str]:
    return [
        f"<d>[{segment.asr_language or 'Unknown'}] {segment.asr_text}</d>"
        for segment in job.segments if segment.asr_status == "transcribed"
    ]


def _normalize_joint_draft(
    joint: MimoJointAVAudioDraft, job: MimoBackendJob, corrections: Counter[str], *, warnings: list[str],
) -> None:
    assembly = joint.speech_av
    audio = assembly.audio_observation
    unique = {}
    for decision in audio.segment_decisions:
        if decision.segment_id in unique:
            # Original JSON conflicts were rejected before common evidence canonicalization.
            corrections["joint_audio_segment_duplicate_removed"] += 1
        else:
            unique[decision.segment_id] = decision
    audio.segment_decisions = list(unique.values())
    required_ids = [segment.segment_id for segment in job.segments]
    for field, rows in (
        ("audio_observation.segment_decisions", audio.segment_decisions),
        ("av_grounding.segment_groundings", assembly.av_grounding.segment_groundings),
    ):
        if [row.segment_id for row in rows] != required_ids:
            raise MimoBackendFailure(
                code="mimo_structured_output_failed", reason="MiMo joint segment inventory differs",
                issues=(ValidationIssue("segment_inventory_mismatch", field,
                                        "joint decisions must exactly follow frozen chronological segment IDs"),),
            )

    required_groups = [target["speaker_group"] for target in _speaker_profile_targets(assembly, job)]
    profiles = joint.speaker_voice_profiles
    actual_groups = [profile.speaker_group for profile in profiles]
    non_transcribed_ids = {s.segment_id for s in job.segments if s.asr_status != "transcribed"}
    non_transcribed_only = {
        decision.primary_speaker_group for decision in audio.segment_decisions
        if decision.segment_id in non_transcribed_ids and decision.primary_speaker_group is not None
    } - set(required_groups)
    extras = [profile for profile in profiles if profile.speaker_group not in required_groups]
    if (
        len(actual_groups) != len(set(actual_groups))
        or not set(required_groups).issubset(actual_groups)
        or any(profile.speaker_group not in non_transcribed_only for profile in extras)
    ):
        raise MimoBackendFailure(
            code="mimo_structured_output_failed", reason="MiMo joint voice profile inventory differs",
            issues=(ValidationIssue("speaker_voice_profile_inventory_mismatch", "speaker_voice_profiles",
                                    "required groups must be unique; only known non-transcribed-only extras may be excluded"),),
        )
    for profile in extras:
        if profile.voice_characteristics is None:
            corrections["joint_non_transcribed_null_profile_removed"] += 1
        else:
            corrections["joint_non_transcribed_non_null_profile_excluded"] += 1
            warnings.append(
                f"non_transcribed_only_voice_profile_excluded:{profile.speaker_group}:"
                "acoustic traits preserved in speech_av_raw_response; not published as an identity-related voice profile"
            )
    retained = [profile for profile in profiles if profile.speaker_group in required_groups]
    moved = sum(profile.speaker_group != group for profile, group in zip(retained, required_groups, strict=True))
    if moved:
        corrections["joint_voice_profile_order_normalization"] += moved
    by_group = {profile.speaker_group: profile for profile in retained}
    joint.speaker_voice_profiles = [by_group[group] for group in required_groups]

    caption = assembly.shot1_caption
    blocks = list(_DIALOGUE.finditer(caption))
    required_dialogue = _required_dialogue_blocks(job)
    transcripts = [s for s in job.segments if s.asr_status == "transcribed"]
    if (
        len(blocks) != len(required_dialogue)
        or caption.count("<d>") != len(blocks) or caption.count("</d>") != len(blocks)
        or any("<d>" in block.group(1) for block in blocks)
        or any(
            block.group(0) != locked and block.group(1) != segment.asr_text
            for block, locked, segment in zip(blocks, required_dialogue, transcripts, strict=True)
        )
    ):
        raise MimoBackendFailure(
            code="mimo_structured_output_failed", reason="MiMo joint exact dialogue inventory differs",
            issues=(ValidationIssue("direct_transcribed_dialogue_missing", "shot1_caption",
                                    "dialogue count, order and exact text must match frozen ASR; only missing language markers are repairable"),),
        )
    # All blocks match before any edit. Preserve every transcript byte and all surrounding prose.
    for block, locked in reversed(list(zip(blocks, required_dialogue, strict=True))):
        if block.group(0) != locked:
            caption = caption[:block.start()] + locked + caption[block.end():]
            corrections["joint_dialogue_language_marker_restored"] += 1
    assembly.shot1_caption = caption


def _normalize_unambiguous_speaker_markers(annotation: MimoAVAnnotationDraft, job: MimoBackendJob) -> int:
    speech = direct_speech_facts(annotation, list(job.segments))
    groups = {d.primary_speaker_group for d in annotation.audio_observation.segment_decisions
              if d.primary_speaker_group is not None}
    transcribed_ids = {fact["segment_id"] for fact in speech}
    transcribed_groups = {d.primary_speaker_group for d in annotation.audio_observation.segment_decisions
                          if d.segment_id in transcribed_ids}
    if groups != transcribed_groups or any(
        speaker_ownership_reasons(d) for d in annotation.audio_observation.segment_decisions if d.segment_id in transcribed_ids
    ):
        return 0
    caption = annotation.h3_semantics.shot1_caption
    blocks = list(_DIALOGUE.finditer(caption))
    if len(blocks) != len(speech):
        return 0
    known = {fact["speaker_id"] for fact in speech}
    markers = re.compile(r"\(S[1-9]\d*\)")
    edits = []
    previous_end = 0
    for block, fact in zip(blocks, speech, strict=True):
        lead_in = caption[previous_end:block.start()]
        for marker in markers.finditer(lead_in):
            if marker.group()[1:-1] in known:
                continue
            # Only an explicit dialogue lead-in; unrelated narrative markers stay hard errors.
            if not re.fullmatch(r"\s*(?:(?:says|asks|replies|adds|continues)[,:]?\s*)?", lead_in[marker.end():]):
                return 0
            edits.append((previous_end + marker.start(), previous_end + marker.end(), f"({fact['speaker_id']})"))
        previous_end = block.end()
    if any(marker.group()[1:-1] not in known for marker in markers.finditer(caption[previous_end:])):
        return 0
    for start, end, marker in reversed(edits):
        caption = caption[:start] + marker + caption[end:]
    annotation.h3_semantics.shot1_caption = caption
    return len(edits)


class TwoStepOpenAIMimo26Backend(StemAwareOpenAIMimo25Backend):
    def __init__(self, config: MimoBackendConfig, **kwargs: Any) -> None:
        if config.transport != "sglang":
            raise ValueError("two-step RA2VA requires SGLang structured output")
        super().__init__(config, **kwargs)

    @property
    def provenance(self) -> MimoBackendProvenance:
        values = self.config.provenance().model_dump(mode="json", exclude={"configuration_fingerprint"})
        values.update(schema_version=TWO_STEP_BACKEND_VERSION, prompt_version=TWO_STEP_PROMPT_VERSION)
        return MimoBackendProvenance(
            **values, configuration_fingerprint=_sha256_text(_compact_json(values)),
        )

    def _maybe_polish_speaker_markers(self, annotation: MimoAVAnnotationDraft, **_: Any) -> tuple[
        MimoAVAnnotationDraft, MimoSpeakerMarkerPolishAudit, None,
    ]:
        return annotation, MimoSpeakerMarkerPolishAudit(), None

    def reconcile(self, job: MimoBackendJob, **kwargs: Any) -> MimoBackendResult:
        try:
            return super().reconcile(job, **kwargs)
        except MimoBackendFailure as exc:
            if exc.annotation is None or not exc.issues:
                raise
            segments = unconfirmed_visible_binding_segments(exc.annotation)
            if any(issue.code != "visible_entity_binding_not_permitted" or issue.field not in segments for issue in exc.issues):
                raise
            groundings = {g.segment_id: g for g in exc.annotation.av_grounding.segment_groundings}
            exc.diagnostics[-1].warnings.extend(
                f"{'sparse_lip_motion_unconfirmed' if 'no_visible_lip_motion' in groundings[segment].evidence_codes else 'visible_entity_binding_missing_positive_cue'}:{segment}:identity_publication_restricted"
                for segment in sorted(segments)
            )
            corrections = {}
            for diagnostic in exc.diagnostics:
                for warning in diagnostic.warnings:
                    if warning.startswith("deterministic_correction_count:"):
                        key, count = warning.removeprefix("deterministic_correction_count:").rsplit("=", 1)
                        corrections[key] = int(count)
            return MimoBackendResult(
                annotation=exc.annotation, raw_responses=exc.raw_responses, diagnostics=exc.diagnostics,
                model_call_count=exc.model_call_count, http_attempt_count=exc.http_attempt_count,
                http_retry_count=0, recheck_count=0, input_modality=self._input_modality,
                visual_model_call_count=exc.visual_model_call_count, visual_raw_response=exc.visual_raw_response,
                speech_av_raw_response=exc.speech_av_raw_response,
                deterministic_correction_counts=corrections,
            )

    def _prompt(self, job: MimoBackendJob, *, allowed_reference_labels: set[str]) -> str:
        prefix, encoded = super()._prompt(job, allowed_reference_labels=allowed_reference_labels).split("AUTHORITATIVE INPUT:\n", 1)
        contract = json.loads(encoded)
        contract["required_segment_ids_in_order"] = [s.segment_id for s in job.segments]
        contract["required_dialogue_blocks_in_order"] = _required_dialogue_blocks(job)
        return prefix + "AUTHORITATIVE INPUT:\n" + _compact_json(contract)

    def _request(
        self, job: MimoBackendJob, *, allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> tuple[str, tuple[str | None, ...], list[MimoCompletionDiagnostic], dict[str, int]]:
        diagnostics: list[MimoCompletionDiagnostic] = []
        raws: list[str | None] = [None, None, None, None]
        corrections: Counter[str] = Counter()

        def audit_corrections() -> None:
            if diagnostics:
                diagnostics[-1].warnings.extend(
                    f"deterministic_correction_count:{key}={count}"
                    for key, count in sorted(corrections.items()) if count
                )

        def request_turn(messages: list[dict[str, Any]], schema: type[SchemaModel], *, turn_index: int) -> str:
            visual_only = turn_index == 0
            modality = "target_video_visual_only" if visual_only else JOINT_INPUT_MODALITY
            payload: dict[str, Any] = {
                "model": self.config.model, "messages": messages,
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_completion_tokens, "stream": False,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True,
                }},
                "extra_body": {
                    "use_audio_in_video": not visual_only,
                    "chat_template_kwargs": {
                        "thinking": self.config.thinking == "enabled",
                        "enable_thinking": self.config.thinking == "enabled",
                    },
                },
            }
            if self.config.thinking == "disabled":
                payload["reasoning_effort"] = "none"
            diagnostic = MimoCompletionDiagnostic(input_modality=modality, usage=MimoUsage(), http_attempt_count=1)
            diagnostics.append(diagnostic)
            try:
                completion, _, _ = self._call(payload)
                choices = _value(completion, "choices")
                if not isinstance(choices, list) or not choices:
                    raise TypeError("MiMo response has no choices")
                choice = choices[0]
                raw = _value(_value(choice, "message"), "content")
                if not isinstance(raw, str):
                    raise TypeError("MiMo response content must be text")
                raws[turn_index] = raw
                diagnostic = _completion_diagnostic(
                    completion, choice, modality=modality, http_attempt_count=1, thinking=self.config.thinking,
                )
                diagnostics[-1] = diagnostic
            except Exception as exc:
                diagnostic.request_error = f"{type(exc).__name__}: {exc}"
                raise
            _validate_finish_reason(diagnostic)
            _validate_av_observation_usage(diagnostic, require_explicit_audio=False, require_reference_images=visual_only)
            if visual_only:
                diagnostic.warnings = [w for w in diagnostic.warnings if w != "audio_tokens_unavailable"]
            elif diagnostic.usage.audio_tokens == 0:
                diagnostic.warnings.append("embedded_audio_tokens_zero")
            return raw

        try:
            if auxiliary_audio_paths is None or set(auxiliary_audio_paths) != {"speech", "music", "sfx"}:
                raise ValueError("two-step RA2VA requires all three resolved stems")
            # Keep Multi Turn 1's request construction identical, including official ICL.
            full_contract = self.build_compact_task_contract(job)
            visual_contract = {key: full_contract[key] for key in (
                "clip_uid", "target_duration_seconds", "reference_selection",
                "reference_image_mapping", "subject_definition_requirements",
            )}
            visual_contract["segment_windows"] = [
                {"segment_id": s.segment_id, "start_time": s.start_time, "end_time": s.end_time}
                for s in job.segments
            ]
            visual_contract["allowed_visible_entity_ids"] = full_contract["allowed_speaker_bindable_entity_ids"]
            visual_contract["allowed_h3_reference_labels"] = sorted(
                label for label in allowed_reference_labels if label.startswith(("<Subject ", "<Picture "))
            )
            content = self._media_content(job)
            content.append({"type": "text", "text": "VISUAL-ONLY INPUT:\n" + _compact_json(visual_contract)})
            visual_raw = request_turn([
                {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
                *(_official_detailed_description_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                {"role": "user", "content": content},
            ], MimoVisualDraft, turn_index=0)
            canonical_visual, visual_corrections = _canonicalize_two_step_visual_payload(visual_raw, job)
            corrections.update(visual_corrections)
            visual, issues = parse_structured_json_issues(canonical_visual, MimoVisualDraft)
            if visual is None or issues:
                raise MimoBackendFailure(code="mimo_visual_structured_output_failed",
                                         reason="MiMo visual draft failed structured validation", issues=tuple(issues))
            target_video = next(item for item in content if item["type"] == "video_url")
            joint_content = [target_video]
            for kind in ("speech", "music", "sfx"):
                joint_content.extend([
                    {"type": "text", "text": f"resolved {kind} stem: separated audio of the SAME target"},
                    {"type": "audio_url", "audio_url": {"url": self.config.media_resolver.resolve(auxiliary_audio_paths[kind])}},
                ])
            joint_content.append({"type": "text", "text": (
                self._prompt(job, allowed_reference_labels=allowed_reference_labels)
                + "\nTURN 1 VISUAL DRAFT:\n" + visual.model_dump_json()
            )})
            prompt = JOINT_SYSTEM_PROMPT
            if getattr(job, "binding_evidence_mode", "legacy_lr_asd") == "none":
                prompt += (
                    "\nNO PRIOR BINDING EVIDENCE: binding_evidence_mode=none. No LR-ASD or current speaker/entity proposal is supplied. "
                    "Neutral prior fields mean unavailable proposals, not measured LR-ASD negatives. "
                    "Use Turn 1 exact-window visible entities, original target AV, DiariZen segments and acoustic clusters. "
                    "Original target AV remains final speaker-binding authority. Never claim lr_asd_support or lr_asd_conflict. "
                    "Do not assume a visible salient person is speaking; do not add a lip-motion prerequisite."
                )
            joint_raw = request_turn([
                {"role": "system", "content": prompt}, {"role": "user", "content": joint_content},
            ], MimoJointAVAudioDraft, turn_index=1)
            canonical_joint, joint_corrections = _canonicalize_joint_payload(joint_raw, job, warnings=diagnostics[-1].warnings)
            corrections.update(joint_corrections)
            joint, issues = parse_structured_json_issues(canonical_joint, MimoJointAVAudioDraft)
            if joint is None or issues:
                raise MimoBackendFailure(code="mimo_structured_output_failed",
                                         reason="MiMo joint AV/audio draft failed structured validation", issues=tuple(issues))
            assembly = joint.speech_av
            _normalize_joint_draft(joint, job, corrections, warnings=diagnostics[-1].warnings)
            shot1_caption = assembly.shot1_caption
            visual_description = visual.shot1_visual_description
            if not any(s.asr_status == "transcribed" for s in job.segments):
                visual_description, removed = re.subn(r"\s*\(S[1-9]\d*\)", "", visual_description)
                if removed:
                    corrections["no_transcript_visual_speaker_markers_removed"] = removed
                shot1_caption = visual_description
                if shot1_caption != assembly.shot1_caption:
                    corrections["no_transcript_visual_caption_projection"] = 1
            final = {
                "schema_version": MIMO25_SCHEMA_VERSION,
                "visual_observation": MimoVisualObservation(
                    visual_blocks=[MimoVisualBlock(block_id="v1", text=visual_description)],
                    segment_views=visual.segment_views,
                ).model_dump(mode="json"),
                "audio_observation": {
                    "segment_decisions": [d.model_dump(mode="json") for d in assembly.audio_observation.segment_decisions],
                    "speaker_voice_profiles": [p.model_dump(mode="json") for p in joint.speaker_voice_profiles],
                },
                "av_grounding": assembly.av_grounding.model_dump(mode="json"),
                "warnings": [w.model_dump(mode="json") for w in assembly.warnings],
                "h3_semantics": {
                    "summary": assembly.summary, "shot1_caption": shot1_caption,
                    **joint.audio_finalize.model_dump(mode="json"),
                    **visual.model_dump(mode="json", exclude={"segment_views", "shot1_visual_description"}),
                },
            }
            annotation, issues = parse_structured_json_issues(_compact_json(final), MimoAVAnnotationDraft)
            if annotation is None or issues:
                raise MimoBackendFailure(
                    code="mimo_structured_output_failed", reason="MiMo joint assembled annotation failed structured validation",
                    issues=tuple(issues),
                )
            # Expose identity contradictions before the shared normalizer can merge gN.
            unconfirmed_segments = unconfirmed_visible_binding_segments(annotation)
            strict_issues = [issue for issue in validate_annotation(
                annotation, segment_ids=[s.segment_id for s in job.segments],
                segment_intervals={s.segment_id: (s.start_time, s.end_time) for s in job.segments},
                transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
                authoritative_transcripts=[s.asr_text for s in job.segments if s.asr_status == "transcribed"],
                allowed_entity_ids=set(full_contract["allowed_speaker_bindable_entity_ids"]),
                allowed_reference_labels=allowed_reference_labels, reference_subjects=job.reference_subjects,
                target_duration_seconds=job.target_duration_seconds,
                binding_evidence_mode=getattr(job, "binding_evidence_mode", "legacy_lr_asd"),
            ) if issue.code in {
                "visible_entity_speaker_group_contradiction", "speaker_group_entity_contradiction",
                "visible_entity_absent_from_visual_segment", "unknown_entity", "av_audio_speaker_group_mismatch",
                "visible_speaker_evidence_presentation_contradiction", "visible_entity_requires_resolved_audio",
            } or (issue.code == "visible_entity_binding_not_permitted"
                  and issue.field not in unconfirmed_segments)]
            if strict_issues:
                _, marker_issues, marker_warnings = protect_direct_dialogue(
                    annotation.h3_semantics.shot1_caption, direct_speech_facts(annotation, list(job.segments)),
                    allowed_labels=allowed_reference_labels,
                )
                diagnostics[-1].warnings.extend(marker_warnings)
                raise MimoBackendFailure(
                    code="mimo_structured_output_failed", reason="MiMo joint speaker binding contradiction",
                    issues=(*strict_issues, *marker_issues), annotation=annotation,
                )
            caption, summary, identity_corrections, publishable = neutralize_two_step_caption_identity(
                annotation, job.segments, self.provenance, reference_subjects=job.reference_subjects,
            )
            if publishable:
                annotation.h3_semantics.shot1_caption = caption
                annotation.h3_semantics.summary = summary
                corrections.update(identity_corrections)
                final = annotation.model_dump(mode="json")
            else:
                diagnostics[-1].warnings.append("two_step_identity_caption_publication_restricted")
            marker_count = _normalize_unambiguous_speaker_markers(annotation, job)
            if marker_count:
                corrections["joint_unambiguous_speaker_marker_corrected"] += marker_count
                final = annotation.model_dump(mode="json")
                diagnostics[-1].warnings.append("joint_speaker_marker_only_correction:review_voice_source_narration")
            audit_corrections()
            return _compact_json(final), tuple(raws), diagnostics, dict(corrections)
        except Exception as exc:
            audit_corrections()
            raise MimoBackendFailure(
                code=exc.code if isinstance(exc, MimoBackendFailure) else "mimo_request_failed",
                reason=exc.reason if isinstance(exc, MimoBackendFailure) else f"{type(exc).__name__}: {exc}",
                issues=exc.issues if isinstance(exc, MimoBackendFailure) else (),
                annotation=exc.annotation if isinstance(exc, MimoBackendFailure) else None,
                raw_responses=tuple(raw for raw in raws if raw is not None), diagnostics=tuple(diagnostics),
                model_call_count=len(diagnostics), http_attempt_count=len(diagnostics),
                visual_model_call_count=sum(d.input_modality == "target_video_visual_only" for d in diagnostics),
                visual_raw_response=raws[0], speech_av_raw_response=raws[1],
            ) from exc
