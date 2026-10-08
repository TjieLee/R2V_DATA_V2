"""Opt-in Visual then joint Speech/AV, acoustic profile and audio finalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r2v_data_v2.h3.mimo25_backend import (
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
)
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

TWO_STEP_BACKEND_VERSION = "r2v.h3.mimo25_backend.80"
TWO_STEP_PROMPT_VERSION = "h3_mimo26_ra2va_two_step_joint_v1"
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
    _JOINT_SPEECH_PROMPT + "\n\nACOUSTIC VOICE PROFILES\n" + _JOINT_PROFILE_PROMPT
    + "\nWhen supported traits cannot be reliably described, use voice_characteristics=null; never invent a profile. "
    "Profiling must not change speech_av grouping, binding or speaker markers. "
    "If there are no transcribed ASR segments, speaker_voice_profiles=[]; do not create Sx or dialogue. "
    "A non-silent speech stem, laughter or breath is not proof of transcribed speech. "
    "If there are no segments, keep both audio/AV inventories empty and still finalize soundscape/music. "
    "Transcribed ASR remains exact even if acoustic characterization is uncertain."
    + "\n\nNON-DIALOGUE AUDIO FINALIZATION\n" + _JOINT_AUDIO_PROMPT
)


def _canonicalize_joint_payload(raw: str) -> tuple[str, dict[str, int]]:
    try:
        payload = json.loads(normalize_structured_json_envelope(raw))
    except (TypeError, ValueError):
        return raw, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("speech_av"), dict):
        return raw, {}
    canonical, corrections = _canonicalize_raw_annotation_payload(_compact_json(payload["speech_av"]))
    payload["speech_av"] = json.loads(canonical)
    return _compact_json(payload), corrections


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

    def _request(
        self, job: MimoBackendJob, *, allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> tuple[str, tuple[str | None, ...], list[MimoCompletionDiagnostic], dict[str, int]]:
        diagnostics: list[MimoCompletionDiagnostic] = []
        raws: list[str | None] = [None, None, None, None]

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
            canonical_visual, visual_corrections = _canonicalize_visual_draft_payload(visual_raw)
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
            canonical_joint, corrections = _canonicalize_joint_payload(joint_raw)
            corrections.update(visual_corrections)
            joint, issues = parse_structured_json_issues(canonical_joint, MimoJointAVAudioDraft)
            if joint is None or issues:
                raise MimoBackendFailure(code="mimo_structured_output_failed",
                                         reason="MiMo joint AV/audio draft failed structured validation", issues=tuple(issues))
            assembly = joint.speech_av
            required_groups = [target["speaker_group"] for target in _speaker_profile_targets(assembly, job)]
            if [p.speaker_group for p in joint.speaker_voice_profiles] != required_groups:
                raise MimoBackendFailure(
                    code="mimo_structured_output_failed", reason="MiMo joint voice profile inventory differs",
                    issues=(ValidationIssue("speaker_voice_profile_inventory_mismatch", "speaker_voice_profiles",
                                            "voice profiles must exactly follow transcribed primary speaker groups"),),
                )
            if not job.segments and (assembly.audio_observation.segment_decisions or assembly.av_grounding.segment_groundings):
                raise MimoBackendFailure(
                    code="mimo_structured_output_failed", reason="MiMo joint draft invented an audio/AV segment",
                    issues=(ValidationIssue("segment_inventory_mismatch", "speech_av", "empty segment input requires empty inventories"),),
                )
            shot1_caption = assembly.shot1_caption
            if not any(s.asr_status == "transcribed" for s in job.segments):
                shot1_caption = visual.shot1_visual_description
                if shot1_caption != assembly.shot1_caption:
                    corrections["no_transcript_visual_caption_projection"] = 1
            final = {
                "schema_version": MIMO25_SCHEMA_VERSION,
                "visual_observation": MimoVisualObservation(
                    visual_blocks=[MimoVisualBlock(block_id="v1", text=visual.shot1_visual_description)],
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
            return _compact_json(final), tuple(raws), diagnostics, corrections
        except Exception as exc:
            raise MimoBackendFailure(
                code=exc.code if isinstance(exc, MimoBackendFailure) else "mimo_request_failed",
                reason=exc.reason if isinstance(exc, MimoBackendFailure) else f"{type(exc).__name__}: {exc}",
                issues=exc.issues if isinstance(exc, MimoBackendFailure) else (),
                raw_responses=tuple(raw for raw in raws if raw is not None), diagnostics=tuple(diagnostics),
                model_call_count=len(diagnostics), http_attempt_count=len(diagnostics),
                visual_model_call_count=sum(d.input_modality == "target_video_visual_only" for d in diagnostics),
                visual_raw_response=raws[0], speech_av_raw_response=raws[1],
            ) from exc
