from __future__ import annotations

from r2v_data_v2.h3.mimo25_stem_shadow import (
    MusicStemFacts,
    OpenAIStemFactsBackend,
    SFXStemFacts,
    SpeechStemFacts,
    StemFactJob,
    StemFactsBackendResult,
    StemType,
    StemView,
)


def _normalized_lexical_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _matches_authoritative_asr(description: str, job: StemFactJob) -> bool:
    observed = _normalized_lexical_text(description)
    if not observed:
        return False
    for segment in job.segments:
        if segment.asr_status != "transcribed" or not segment.asr_text:
            continue
        authoritative = _normalized_lexical_text(segment.asr_text)
        if len(authoritative) >= 4 and authoritative in observed:
            return True
    return False


def canonicalize_stem_facts(
    facts: SpeechStemFacts | MusicStemFacts | SFXStemFacts,
    job: StemFactJob,
) -> tuple[SpeechStemFacts | MusicStemFacts | SFXStemFacts, int]:
    """Apply code-owned timeline authority and remove obvious ASR leakage from SFX."""

    if isinstance(facts, SpeechStemFacts):
        values = facts.model_dump(mode="python")
        values["clip_duration_seconds"] = job.clip_duration_seconds
        return SpeechStemFacts.model_validate(values), 0

    if isinstance(facts, MusicStemFacts):
        values = facts.model_dump(mode="python")
        values["clip_duration_seconds"] = job.clip_duration_seconds
        return MusicStemFacts.model_validate(values), 0

    kept_layers = [
        item
        for item in facts.continuous_layers
        if not _matches_authoritative_asr(item.description, job)
    ]
    kept_events = [
        item for item in facts.events if not _matches_authoritative_asr(item.description, job)
    ]
    suppressed_count = (
        len(facts.continuous_layers)
        - len(kept_layers)
        + len(facts.events)
        - len(kept_events)
    )
    residual_note = facts.residual_artifact_notes
    if suppressed_count:
        suppression_note = (
            f"Suppressed {suppressed_count} residual item(s) whose lexical content "
            "matched authoritative ASR; treated as separator speech leakage."
        )
        residual_note = (
            suppression_note
            if not residual_note
            else f"{residual_note.rstrip()} {suppression_note}"
        )
    values = facts.model_dump(mode="python")
    values.update(
        clip_duration_seconds=job.clip_duration_seconds,
        continuous_layers=[item.model_dump(mode="python") for item in kept_layers],
        events=[item.model_dump(mode="python") for item in kept_events],
        residual_artifact_notes=residual_note,
    )
    return SFXStemFacts.model_validate(values), suppressed_count


class AuthoritativeOpenAIStemFactsBackend(OpenAIStemFactsBackend):
    """Keep model semantics while making deterministic upstream facts authoritative."""

    def extract(
        self,
        *,
        job: StemFactJob,
        stem_type: StemType,
        view: StemView,
    ) -> StemFactsBackendResult:
        result = super().extract(job=job, stem_type=stem_type, view=view)
        reported_duration = result.facts.clip_duration_seconds
        facts, suppressed_count = canonicalize_stem_facts(result.facts, job)
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "reported_clip_duration_seconds": reported_duration,
                "authoritative_clip_duration_seconds": job.clip_duration_seconds,
                "clip_duration_authority": "stem_fact_job_over_model_echo_v1",
                "sfx_asr_leakage_suppressed_count": suppressed_count,
            }
        )
        return StemFactsBackendResult(
            facts=facts,
            raw_response=result.raw_response,
            diagnostics=diagnostics,
        )


__all__ = [
    "AuthoritativeOpenAIStemFactsBackend",
    "canonicalize_stem_facts",
]
