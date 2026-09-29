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

SINGLE_PROMPT_VERSION = "h3_mimo26_ra2va_single_v2"
SINGLE_BACKEND_VERSION = "r2v.h3.mimo25_backend.68"

SINGLE_SYSTEM_PROMPT = """Return exactly one MimoAVAnnotationDraft JSON object. Complete these responsibilities within one observation of the original target AV and the supplied frozen Pictures and resolved stems. The original AV is the final audiovisual authority; stems aid acoustic recall. Official ICL shows H3 writing and field boundaries, not the response schema or this clip's facts.

STEP 1 — VISUAL OBSERVATION
Observe the entire video and frozen references first. Fill visual_observation with exact-window visible entities and observable visual states. Give each required Subject one concise stable appearance description in subject_definitions, concise visual_retention_analysis, one-sentence style_opening, and one detailed chronological visual description covering composition, setting, lighting, camera, actions, expressions, and progression through the end. Cite visibly preserved entity/background <Subject N> naturally at first appearance. Do not decide who speaks yet. Do not put <Picture N> in model-authored definition descriptions: the materializer adds frozen Picture provenance and attribute ownership.

STEP 2 — ACOUSTIC SPEAKER STRUCTURE
Use original audio and the speech stem. Fill audio_observation.segment_decisions for EVERY supplied DiariZen segment, including non-transcribed ones, in order and without changing boundaries. Identify stable acoustic sources g1, g2, ... by first actual vocal appearance; ASR sentences, pauses, language changes, and segment boundaries alone do not create new groups. Record vocal composition, secondary activity, and transcribed delivery without visual identity.

STEP 3 — AV SPEAKER GROUNDING
Compare each acoustic group with Step-1 visibility in the exact segment and the original AV. Fill ordered av_grounding.segment_groundings, retaining each primary group. Bind visible_entity only when that entity appears in the segment's visible_entity_ids and the AV supports that speaker; visible presence alone does not assign speech. Otherwise distinguish genuine offscreen evidence from no_reliable_entity or uncertain, with null entity_id as appropriate. No LR-ASD evidence or upstream entity proposals are supplied. Confirmed overlapping or sequential multiple-speaker speech is not a clean single-identity claim.

STEP 4 — DIALOGUE PROJECTION
Build h3_semantics.shot1_caption from the Step-1 visual description, preserving its visual content and playback order; insert only local speaker presentation and locked dialogue. Each transcribed segment supplies an immutable required_dialogue_block: place it exactly once in chronological order, without translation, paraphrase, normalization, merging, or omission. Non-transcribed segments have no dialogue block. Subject N is visual identity, gN is an acoustic group, and Sx is the consumer speaker ID; never infer Sx from Subject N. Number by actual vocal-source order: first actual vocal source -> S1, second distinct source -> S2. A silent visible Subject consumes no Sx; offscreen or unbound actual speakers still receive Sx. Establish (Sx) before the first dialogue and before a changed speaker's dialogue; a continuing speaker may inherit its established marker. Keep grounding rationale out of caption. Write a concise target-video summary without task prefix.

STEP 5 — VOICE CHARACTERISTICS
For already decided speaker groups, fill speaker_voice_profiles with supported acoustic pitch/register, timbre/texture, cadence/rate, and energy/delivery from speech evidence. Do not re-decide grouping or entity binding.

STEP 6 — NON-DIALOGUE AUDIO
Listen to original target AV with the music and SFX stems. Put continuous ambience and physical/environmental sound in overall_soundscape, and audience-only score in non_diegetic_music; keep dialogue out of both. Localized diegetic sound may appear at its chronological point in shot1_caption. The materializer alone creates <Audio N> definitions, relationships, retention, Picture provenance, and final H3 formatting.

FINAL
Return only the supplied annotation schema, not intermediate steps or reasoning."""


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
