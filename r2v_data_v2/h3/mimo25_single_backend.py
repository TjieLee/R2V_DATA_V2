"""One-request RA2VA shadow adapter over the frozen final annotation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendProvenance,
    MimoCompletionDiagnostic,
    MimoSpeakerMarkerPolishAudit,
    MimoUsage,
    _compact_json,
    _completion_diagnostic,
    _official_detailed_description_icl_messages,
    _sha256_text,
    _validate_av_observation_usage,
    _validate_finish_reason,
    _value,
)
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend

SINGLE_PROMPT_VERSION = "h3_mimo26_ra2va_single_v3"
SINGLE_BACKEND_VERSION = "r2v.h3.mimo25_backend.69"

SINGLE_SYSTEM_PROMPT = """Return exactly one MimoAVAnnotationDraft JSON object. Complete these responsibilities within one observation of the original target AV and the supplied frozen Pictures and resolved stems. The original AV is the final audiovisual authority; stems aid acoustic recall. Official ICL shows H3 writing and field boundaries, not the response schema or this clip's facts.

STEP 1 — VISUAL OBSERVATION
Observe the entire video and frozen references first. Fill visual_observation with exact-window visible entities and observable visual states. Give each required Subject one concise stable appearance description in subject_definitions, concise visual_retention_analysis, one-sentence style_opening, and one detailed chronological visual description covering composition, setting, lighting, camera, actions, expressions, and progression through the end. Cite visibly preserved entity/background <Subject N> naturally at first appearance. Do not decide who speaks yet. Do not put <Picture N> in model-authored definition descriptions: the materializer adds frozen Picture provenance and attribute ownership.

STEP 2 — ACOUSTIC SPEAKER STRUCTURE
Use original audio and the speech stem. Fill audio_observation.segment_decisions for EVERY supplied DiariZen segment, including non-transcribed ones, in order and without changing boundaries. Identify stable acoustic sources g1, g2, ... by first actual vocal appearance; ASR sentences, pauses, language changes, and segment boundaries alone do not create new groups. Record vocal composition, secondary activity, and transcribed delivery without visual identity.

STEP 3 — AV SPEAKER GROUNDING
Compare each acoustic group with Step-1 visibility in the exact segment and the original AV. Fill ordered av_grounding.segment_groundings, retaining each primary group. Bind visible_entity only when that entity appears in the segment's visible_entity_ids and the AV supports that speaker; visible presence alone does not assign speech. Otherwise distinguish genuine offscreen evidence from no_reliable_entity or uncertain, with null entity_id as appropriate. No LR-ASD evidence or upstream entity proposals are supplied. Confirmed overlapping or sequential multiple-speaker speech is not a clean single-identity claim.

STEP 4 — DIALOGUE PROJECTION
Build h3_semantics.shot1_caption from the Step-1 visual description, preserving its visual content and playback order; insert only local speaker presentation and locked dialogue. Each transcribed segment supplies an immutable required_dialogue_block: place it exactly once in chronological order, without translation, paraphrase, normalization, merging, or omission. Non-transcribed segments have no dialogue block. Subject N is visual identity, gN is an acoustic group, and Sx is the consumer speaker ID; gN and Sx are different namespaces. Never infer Sx from Subject N or the numeric suffix of gN. Number Sx by each distinct final speaker source/group's first transcribed appearance: first distinct speaker appearing in transcribed dialogue -> S1, next distinct transcribed speaker -> S2. A group occurring only in non-transcribed vocal activity consumes no Sx. A silent visible Subject consumes no Sx; offscreen or unbound speakers receive Sx only when they have transcribed dialogue. Establish (Sx) before the first dialogue and before a changed speaker's dialogue; a continuing speaker may inherit its established marker. Keep grounding rationale out of caption. Write a concise target-video summary without task prefix.

STEP 5 — VOICE CHARACTERISTICS
For transcribed segments only, output speaker_voice_profiles for each distinct non-null primary_speaker_group, ordered by first transcribed appearance. Describe supported acoustic pitch/register, timbre/texture, cadence/rate, and energy/delivery from the corresponding segment-window speech-stem evidence. A group appearing only in non-transcribed activity gets no profile. Do not re-decide grouping or entity binding.

STEP 6 — NON-DIALOGUE AUDIO
Listen to original target AV with the music and SFX stems. Put continuous ambience and physical/environmental sound in overall_soundscape, and audience-only score in non_diegetic_music; keep dialogue out of both. Do not add non-dialogue audio to shot1_caption; it contains visual prose, speaker presentation, and exact dialogue. The materializer alone creates <Audio N> definitions, relationships, retention, Picture provenance, and final H3 formatting.

CROSS-STEP CONSISTENCY
Later steps consume earlier-step state; do not independently reinterpret it. For each segment, av_grounding must agree with the exact same visual_observation.segment_views entry. If an entity has speech_correlated_articulation="observed", do not emit no_visible_lip_motion for that entity in that segment. If the original AV also supports temporal correspondence with that visible entity, do not classify the same vocal source as offscreen. binding_status="visible_entity" requires speech_presentation="onscreen_spoken", entity_id from that segment's visible_entity_ids, and compatible positive AV evidence. binding_status="offscreen" requires speech_presentation="offscreen_spoken", entity_id=null, and genuine offscreen_audio evidence. no_reliable_entity or uncertain requires entity_id=null and must not claim a visible Subject as speaker. shot1_caption must follow the FINAL av_grounding: render a visible bound speaker with its mapped <Subject N> and (Sx), an offscreen speaker as a natural offscreen source with (Sx), and an unresolved speaker without Subject identity. Do not make contradictory speaker claims across visual_observation, av_grounding, and shot1_caption.

OUTPUT INVENTORY
Treat required_output_inventory as exact and ordered. subject_definitions and visual_retention_analysis each contain exactly one row per supplied Subject in supplied order, without invented Subjects. visual_observation.segment_views, audio_observation.segment_decisions, and av_grounding.segment_groundings each contain exactly one row per supplied segment in chronological order, without missing, duplicated, or invented segments. Use only labels and IDs from the authoritative input.

FINAL
Return only the supplied annotation schema, not intermediate steps or reasoning."""


def _single_semantic_icl_messages() -> list[dict[str, str]]:
    example = {
        "schema_version": "r2v.h3.mimo25_av_annotation.20",
        "visual_observation": {
            "visual_blocks": [{
                "block_id": "v1",
                "text": "<Subject 1> watches quietly beside the doorway while <Subject 2> turns toward the camera side. Their positions remain separate in the steady composition.",
            }],
            "segment_views": [
                {
                    "segment_id": "segment_0001",
                    "visible_entity_ids": ["e1", "e2"],
                    "entity_observations": [
                        {
                            "entity_id": "e1", "visibility": "visible", "orientation": "three_quarter",
                            "face_visibility": "clear", "mouth_visibility": "clear",
                            "speech_correlated_articulation": "not_observed",
                        },
                        {
                            "entity_id": "e2", "visibility": "visible", "orientation": "three_quarter",
                            "face_visibility": "clear", "mouth_visibility": "clear",
                            "speech_correlated_articulation": "observed",
                        },
                    ],
                },
                {
                    "segment_id": "segment_0002",
                    "visible_entity_ids": ["e1", "e2"],
                    "entity_observations": [
                        {
                            "entity_id": "e1", "visibility": "visible", "orientation": "three_quarter",
                            "face_visibility": "clear", "mouth_visibility": "clear",
                            "speech_correlated_articulation": "not_observed",
                        },
                        {
                            "entity_id": "e2", "visibility": "visible", "orientation": "three_quarter",
                            "face_visibility": "clear", "mouth_visibility": "clear",
                            "speech_correlated_articulation": "not_observed",
                        },
                    ],
                },
            ],
        },
        "audio_observation": {
            "segment_decisions": [
                {
                    "segment_id": "segment_0001", "vocal_composition": "single_speaker",
                    "resolution": "resolved", "primary_speaker_group": "g1",
                    "delivery_style": "calm conversational delivery",
                    "secondary_vocal_activity": {"present": False, "speaker_relation": "none", "kind": None},
                    "confidence": "high", "audio_evidence_codes": ["source_cluster_support"],
                },
                {
                    "segment_id": "segment_0002", "vocal_composition": "single_speaker",
                    "resolution": "resolved", "primary_speaker_group": "g2",
                    "delivery_style": "brief questioning delivery",
                    "secondary_vocal_activity": {"present": False, "speaker_relation": "none", "kind": None},
                    "confidence": "medium", "audio_evidence_codes": ["speaker_turn_change"],
                },
            ],
            "speaker_voice_profiles": [
                {"speaker_group": "g1", "voice_characteristics": "a mid-register voice with a clear timbre and measured cadence"},
                {"speaker_group": "g2", "voice_characteristics": "a lower voice with restrained questioning delivery"},
            ],
        },
        "av_grounding": {
            "segment_groundings": [
                {
                    "segment_id": "segment_0001", "primary_speaker_group": "g1",
                    "binding_status": "visible_entity", "speech_presentation": "onscreen_spoken",
                    "entity_id": "e2", "confidence": "high",
                    "evidence_codes": ["visible_lip_motion", "av_temporal_alignment"],
                },
                {
                    "segment_id": "segment_0002", "primary_speaker_group": "g2",
                    "binding_status": "offscreen", "speech_presentation": "offscreen_spoken",
                    "entity_id": None, "confidence": "medium", "evidence_codes": ["offscreen_audio"],
                },
            ],
        },
        "h3_semantics": {
            "subject_definitions": [
                {"subject_label": "<Subject 1>", "description": "an adult woman with dark shoulder-length hair and a dark green coat"},
                {"subject_label": "<Subject 2>", "description": "an adult man with short dark hair and a gray jacket"},
            ],
            "summary": "Two people pause near an entrance; the visible man answers first, followed by an offscreen speaker.",
            "style_opening": "Naturalistic live-action cinematography with a steady camera and soft practical lighting.",
            "shot1_caption": "<Subject 1> watches quietly beside the doorway while <Subject 2> turns toward the camera side. <Subject 2> (S1) replies, <d>[Chinese] 好，我知道了。</d> He pauses as an offscreen voice (S2) asks, <d>[Chinese] 你去哪？</d>",
            "overall_soundscape": "A quiet interior ambience with faint footsteps and subtle doorway movement.",
            "non_diegetic_music": "N/A",
            "visual_retention_analysis": [
                {"subject_label": "<Subject 1>", "marker": "fully_preserved", "description": "Her hairstyle, coat, and visible proportions remain consistent."},
                {"subject_label": "<Subject 2>", "marker": "fully_preserved", "description": "His short hair, jacket, and visible proportions remain consistent."},
            ],
        },
        "warnings": [],
    }
    return [
        {"role": "user", "content": (
            "This is a compact synthetic demonstration of cross-field semantic consistency "
            "for the single-call annotation schema. It demonstrates Subject/gN/Sx "
            "relationships and field dependencies only. Do not copy its people, dialogue, "
            "entities, visual style, or factual content into the real task."
        )},
        {"role": "assistant", "content": _compact_json(example)},
    ]


class SingleCallOpenAIMimo25Backend(StemAwareOpenAIMimo25Backend):
    def __init__(self, config: MimoBackendConfig, **kwargs: Any) -> None:
        if config.transport != "sglang":
            raise ValueError("single RA2VA requires SGLang structured output")
        super().__init__(config, **kwargs)

    @property
    def provenance(self) -> MimoBackendProvenance:
        values = self.config.provenance().model_dump(
            mode="json", exclude={"configuration_fingerprint"},
        )
        values.update(schema_version=SINGLE_BACKEND_VERSION, prompt_version=SINGLE_PROMPT_VERSION)
        return MimoBackendProvenance(
            **values, configuration_fingerprint=_sha256_text(_compact_json(values)),
        )

    @property
    def _input_modality(self) -> str:
        return "target_av_with_auxiliary_raw_audio"

    def _request(
        self, job: Any, *, allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> tuple[str, tuple[str | None, ...], list[MimoCompletionDiagnostic], dict[str, int]]:
        diagnostic = MimoCompletionDiagnostic(
            input_modality="target_av_with_auxiliary_raw_audio",
            usage=MimoUsage(), http_attempt_count=1,
        )
        raw: str | None = None
        attempted = False
        try:
            if auxiliary_audio_paths is None or set(auxiliary_audio_paths) != {"speech", "music", "sfx"}:
                raise ValueError("single RA2VA requires all three resolved stems")
            contract = self.build_compact_task_contract(job)
            segments = []
            for source in contract["segments"]:
                segment = {
                    key: value for key, value in source.items()
                    if key in {
                        "segment_id", "start_time", "end_time", "source_speaker_cluster_id",
                        "asr_status", "asr_text", "asr_language",
                    }
                }
                if segment["asr_status"] == "transcribed":
                    segment["required_dialogue_block"] = (
                        f"<d>[{segment['asr_language'] or 'Unknown'}] "
                        f"{segment['asr_text']}</d>"
                    )
                segments.append(segment)
            contract["segments"] = segments
            contract["required_output_inventory"] = {
                "subject_labels_in_order": [subject.subject_label for subject in job.reference_subjects],
                "segment_ids_in_order": [segment.segment_id for segment in job.segments],
                "transcribed_dialogue_blocks_in_order": [
                    {"segment_id": segment["segment_id"],
                     "required_dialogue_block": segment["required_dialogue_block"]}
                    for segment in segments if segment["asr_status"] == "transcribed"
                ],
            }
            contract["allowed_h3_reference_labels"] = sorted(allowed_reference_labels)
            content = self._media_content(job)
            for kind in ("speech", "music", "sfx"):
                content.extend([
                    {"type": "text", "text": f"resolved {kind} stem"},
                    {"type": "audio_url", "audio_url": {
                        "url": self.config.media_resolver.resolve(auxiliary_audio_paths[kind]),
                    }},
                ])
            content.append({"type": "text", "text": "AUTHORITATIVE INPUT:\n" + _compact_json(contract)})
            payload: dict[str, object] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SINGLE_SYSTEM_PROMPT},
                    *(_official_detailed_description_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                    *(_single_semantic_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_completion_tokens,
                "stream": False,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "MimoAVAnnotationDraft",
                    "schema": MimoAVAnnotationDraft.model_json_schema(), "strict": True,
                }},
                "extra_body": {
                    "use_audio_in_video": True,
                    "chat_template_kwargs": {
                        "thinking": self.config.thinking == "enabled",
                        "enable_thinking": self.config.thinking == "enabled",
                    },
                },
            }
            if self.config.thinking == "disabled":
                payload["reasoning_effort"] = "none"
            attempted = True
            completion, _, _ = self._call(payload)
            choices = _value(completion, "choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("MiMo response has no choices")
            choice = choices[0]
            raw = _value(_value(choice, "message"), "content")
            if not isinstance(raw, str):
                raise TypeError("MiMo response content must be text")
            diagnostic = _completion_diagnostic(
                completion, choice, modality="target_av_with_auxiliary_raw_audio",
                http_attempt_count=1, thinking=self.config.thinking,
            )
            _validate_finish_reason(diagnostic)
            _validate_av_observation_usage(diagnostic, require_explicit_audio=False)
            if diagnostic.usage.audio_tokens == 0:
                diagnostic.warnings.append("embedded_audio_tokens_zero")
            return raw, (None, raw, None, None), [diagnostic], {}
        except Exception as exc:
            diagnostic.request_error = f"{type(exc).__name__}: {exc}"
            raise MimoBackendFailure(
                code=exc.code if isinstance(exc, MimoBackendFailure) else "mimo_request_failed",
                reason=exc.reason if isinstance(exc, MimoBackendFailure) else diagnostic.request_error,
                issues=exc.issues if isinstance(exc, MimoBackendFailure) else (),
                raw_responses=(raw,) if raw is not None else (),
                diagnostics=(diagnostic,) if attempted else (),
                model_call_count=int(attempted), http_attempt_count=int(attempted),
                speech_av_raw_response=raw,
            ) from exc

    def _maybe_polish_speaker_markers(self, annotation: MimoAVAnnotationDraft, **_: Any) -> tuple[
        MimoAVAnnotationDraft, MimoSpeakerMarkerPolishAudit, None,
    ]:
        return annotation, MimoSpeakerMarkerPolishAudit(), None
