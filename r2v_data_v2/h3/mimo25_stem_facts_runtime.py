from __future__ import annotations

import hashlib
import json
import re

from pydantic import Field

from r2v_data_v2.h3.mimo25_stem_shadow import (
    Confidence,
    MusicStemFacts,
    OpenAIStemFactsBackend,
    SFXCategory,
    SFXStemFacts,
    SpeechStemFacts,
    StemFactJob,
    StemFactsBackendResult,
    StemType,
    StemView,
)
from r2v_data_v2.h3.schemas import SchemaModel

SFX_STRUCTURED_OUTPUT_POLICY = "stem_fact_sfx_bounded_normalization_v2"
SFX_MAX_CONTINUOUS_LAYERS = 8
SFX_MAX_EVENTS = 16
_SFX_MUSIC_SEMANTICS = re.compile(
    r"\b(?:bgm|music|song|soundtrack|score|melody|instrumental)\b"
    r"|背景音乐|音乐|歌曲|配乐",
    flags=re.IGNORECASE,
)


class BoundedSFXContinuousLayerDraft(SchemaModel):
    """Pre-publication SFX layer schema that permits zero-length model artifacts."""

    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    confidence: Confidence


class BoundedSFXEventDraft(BoundedSFXContinuousLayerDraft):
    category: SFXCategory


class BoundedSFXStemFactsDraft(SchemaModel):
    """Bounded structured-output schema before deterministic SFX canonicalization."""

    clip_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    continuous_layers: list[BoundedSFXContinuousLayerDraft] = Field(
        max_length=SFX_MAX_CONTINUOUS_LAYERS
    )
    events: list[BoundedSFXEventDraft] = Field(max_length=SFX_MAX_EVENTS)
    residual_artifact_notes: str | None = None


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def _contains_music_semantics(description: str) -> bool:
    return _SFX_MUSIC_SEMANTICS.search(description) is not None


def _item_key(
    item: BoundedSFXContinuousLayerDraft | BoundedSFXEventDraft,
) -> tuple[object, ...]:
    return (
        item.start_time,
        item.end_time,
        " ".join(str(item.description).split()).casefold(),
        item.category if isinstance(item, BoundedSFXEventDraft) else None,
    )


def _canonicalize_sfx_draft(
    facts: BoundedSFXStemFactsDraft,
    job: StemFactJob,
) -> tuple[SFXStemFacts, dict[str, int]]:
    nonpositive_count = 0
    duplicate_count = 0
    asr_leakage_count = 0
    music_leakage_count = 0
    model_note_discarded_count = int(bool(facts.residual_artifact_notes))

    def clean(
        values: list[BoundedSFXContinuousLayerDraft | BoundedSFXEventDraft],
    ) -> list[BoundedSFXContinuousLayerDraft | BoundedSFXEventDraft]:
        nonlocal (
            nonpositive_count,
            duplicate_count,
            asr_leakage_count,
            music_leakage_count,
        )
        output: list[BoundedSFXContinuousLayerDraft | BoundedSFXEventDraft] = []
        seen: set[tuple[object, ...]] = set()
        for item in values:
            if item.end_time <= item.start_time:
                nonpositive_count += 1
                continue
            key = _item_key(item)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            if _matches_authoritative_asr(str(item.description), job):
                asr_leakage_count += 1
                continue
            if _contains_music_semantics(str(item.description)):
                music_leakage_count += 1
                continue
            output.append(item)
        return output

    kept_layers = clean(list(facts.continuous_layers))
    kept_events = clean(list(facts.events))
    return (
        SFXStemFacts(
            clip_duration_seconds=job.clip_duration_seconds,
            continuous_layers=[
                {
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "description": item.description,
                    "confidence": item.confidence,
                }
                for item in kept_layers
            ],
            events=[
                {
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "description": item.description,
                    "confidence": item.confidence,
                    "category": item.category,
                }
                for item in kept_events
                if isinstance(item, BoundedSFXEventDraft)
            ],
            residual_artifact_notes=None,
        ),
        {
            "nonpositive": nonpositive_count,
            "duplicate": duplicate_count,
            "asr_leakage": asr_leakage_count,
            "music_leakage": music_leakage_count,
            "model_note_discarded": model_note_discarded_count,
        },
    )


def canonicalize_stem_facts(
    facts: SpeechStemFacts | MusicStemFacts | SFXStemFacts,
    job: StemFactJob,
) -> tuple[SpeechStemFacts | MusicStemFacts | SFXStemFacts, int]:
    """Apply code-owned timeline authority and remove cross-stem leakage from SFX."""

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
        and not _contains_music_semantics(item.description)
    ]
    kept_events = [
        item
        for item in facts.events
        if not _matches_authoritative_asr(item.description, job)
        and not _contains_music_semantics(item.description)
    ]
    suppressed_count = (
        len(facts.continuous_layers)
        - len(kept_layers)
        + len(facts.events)
        - len(kept_events)
    )
    values = facts.model_dump(mode="python")
    values.update(
        clip_duration_seconds=job.clip_duration_seconds,
        continuous_layers=[item.model_dump(mode="python") for item in kept_layers],
        events=[item.model_dump(mode="python") for item in kept_events],
        residual_artifact_notes=None,
    )
    return SFXStemFacts.model_validate(values), suppressed_count


class AuthoritativeOpenAIStemFactsBackend(OpenAIStemFactsBackend):
    """Keep model semantics while making deterministic upstream facts authoritative."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        values = self.provenance.model_dump(
            mode="json", exclude={"configuration_fingerprint"}
        )
        values["stem_fact_runtime_policy"] = SFX_STRUCTURED_OUTPUT_POLICY
        self.provenance = self.provenance.model_copy(
            update={
                "configuration_fingerprint": hashlib.sha256(
                    _compact_json(values).encode()
                ).hexdigest()
            }
        )

    @staticmethod
    def _schema(stem_type: StemType) -> type[SchemaModel]:
        if stem_type == "sfx":
            return BoundedSFXStemFactsDraft
        return OpenAIStemFactsBackend._schema(stem_type)

    @staticmethod
    def _prompt(job: StemFactJob, stem_type: StemType) -> str:
        prompt = OpenAIStemFactsBackend._prompt(job, stem_type)
        if stem_type != "sfx":
            return prompt
        return prompt + (
            " Report only non-musical, non-lexical sounds directly audible in this "
            "SFX stem audio. Never infer sounds from the visual setting, ASR text, "
            "the presence or name of another stem, or general scene expectations. "
            "Do not report music, songs, soundtrack, score, BGM, or lexical speech "
            "as SFX layers/events. If no qualifying SFX is directly audible, return "
            "empty continuous_layers and events. residual_artifact_notes may describe "
            "separator artifacts only and must not contain scene-based sound guesses."
        )

    def extract(
        self,
        *,
        job: StemFactJob,
        stem_type: StemType,
        view: StemView,
    ) -> StemFactsBackendResult:
        result = super().extract(job=job, stem_type=stem_type, view=view)
        reported_duration = result.facts.clip_duration_seconds
        diagnostics = dict(result.diagnostics)
        diagnostics["stem_fact_runtime_policy"] = SFX_STRUCTURED_OUTPUT_POLICY

        if isinstance(result.facts, BoundedSFXStemFactsDraft):
            facts, counts = _canonicalize_sfx_draft(result.facts, job)
            suppressed_count = counts["asr_leakage"]
            diagnostics.update(
                {
                    "sfx_nonpositive_items_suppressed_count": counts["nonpositive"],
                    "sfx_duplicate_items_suppressed_count": counts["duplicate"],
                    "sfx_music_leakage_suppressed_count": counts["music_leakage"],
                    "sfx_model_residual_note_discarded_count": counts[
                        "model_note_discarded"
                    ],
                }
            )
        else:
            facts, suppressed_count = canonicalize_stem_facts(result.facts, job)
            diagnostics.update(
                {
                    "sfx_nonpositive_items_suppressed_count": 0,
                    "sfx_duplicate_items_suppressed_count": 0,
                    "sfx_music_leakage_suppressed_count": 0,
                    "sfx_model_residual_note_discarded_count": 0,
                }
            )

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
    "SFX_MAX_CONTINUOUS_LAYERS",
    "SFX_MAX_EVENTS",
    "SFX_STRUCTURED_OUTPUT_POLICY",
    "AuthoritativeOpenAIStemFactsBackend",
    "BoundedSFXStemFactsDraft",
    "canonicalize_stem_facts",
]
