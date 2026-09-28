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
    _sha256_text,
    _validate_av_observation_usage,
    _validate_finish_reason,
    _value,
)
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend

SINGLE_PROMPT_VERSION = "h3_mimo26_ra2va_single_v1"
SINGLE_BACKEND_VERSION = "r2v.h3.mimo25_backend.67"

SINGLE_SYSTEM_PROMPT = """Return exactly one MimoAVAnnotationDraft JSON object for the complete original target AV.
Observe the entire video, its embedded audio, the ordered frozen Pictures and their Subject ownership, and the labeled speech/music/SFX stems together. The original target AV is the final sound and AV authority; separated stems aid acoustic recall but cannot override it.
Preserve the supplied Picture/Subject labels, order and ownership. Give detailed chronological visual prose, concise natural Subject definitions and retention analysis, AV speaker grouping/binding, speaker voice profiles, exact authoritative dialogue, soundscape and music in their existing annotation fields.
Every supplied DiariZen segment needs one ordered acoustic decision and one ordered AV grounding, including non-transcribed segments. Preserve exact segment boundaries. Only transcribed segments receive chronological <d>[Language] exact ASR text</d> in shot1_caption. Never translate or paraphrase ASR. Assign stable (Sx) by first actual vocal appearance and place each marker before its dialogue. Do not invent a speaker, entity, reference label, transcript or audio fact.
Visible speaker binding must agree with the same segment's visible_entity_ids. If the AV cannot reliably bind a voice, keep entity_id null and the presentation uncertain; absence of visible lip motion alone does not prove offscreen speech. Do not use LR-ASD or upstream entity proposals. Confirmed multiple-speaker speech cannot publish a clean identity.
Keep localized sound in chronological shot prose, continuous ambience in overall_soundscape, and audience-only music in non_diegetic_music. Return only the supplied schema; no extra calls or repair text."""


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
            contract["segments"] = [
                {
                    key: value for key, value in segment.items()
                    if key in {
                        "segment_id", "start_time", "end_time", "source_speaker_cluster_id",
                        "asr_status", "asr_text", "asr_language",
                    }
                }
                for segment in contract["segments"]
            ]
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
