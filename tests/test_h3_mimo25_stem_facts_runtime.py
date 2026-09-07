from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.mimo25_stem_facts_runtime import (
    SFX_MAX_CONTINUOUS_LAYERS,
    SFX_MAX_EVENTS,
    BoundedSFXEventDraft,
    BoundedSFXStemFactsDraft,
    _canonicalize_sfx_draft,
    canonicalize_stem_facts,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MusicStemFacts,
    SFXContinuousLayer,
    SFXEvent,
    SFXStemFacts,
    SpeechStemFacts,
    SpeechStemSegmentFact,
)


def _job(*, clip_duration_seconds: float = 4.38140625) -> SimpleNamespace:
    return SimpleNamespace(
        clip_duration_seconds=clip_duration_seconds,
        segments=[
            SimpleNamespace(
                asr_status="transcribed",
                asr_text="当时我们一起出差的时候，你不是。",
            )
        ],
    )


def test_model_echo_duration_is_replaced_by_authoritative_job_duration() -> None:
    job = _job()
    speech, speech_suppressed = canonicalize_stem_facts(
        SpeechStemFacts(
            clip_duration_seconds=4.381406,
            segment_facts=[
                SpeechStemSegmentFact(
                    segment_id="segment_0001",
                    vocal_composition="single speaker",
                    confidence="high",
                )
            ],
        ),
        job,
    )
    music, music_suppressed = canonicalize_stem_facts(
        MusicStemFacts(
            clip_duration_seconds=4.381406,
            music_status="absent",
            intervals=[],
        ),
        job,
    )

    assert speech.clip_duration_seconds == job.clip_duration_seconds
    assert music.clip_duration_seconds == job.clip_duration_seconds
    assert speech_suppressed == 0
    assert music_suppressed == 0


def test_sfx_asr_leakage_is_suppressed_without_dropping_real_sfx() -> None:
    job = _job()
    facts, suppressed = canonicalize_stem_facts(
        SFXStemFacts(
            clip_duration_seconds=4.381406,
            continuous_layers=[
                SFXContinuousLayer(
                    start_time=0.0,
                    end_time=4.381406,
                    description="quiet room tone",
                    confidence="high",
                )
            ],
            events=[
                SFXEvent(
                    start_time=1.0,
                    end_time=1.2,
                    description="a short door click",
                    confidence="medium",
                    category="physical",
                ),
                SFXEvent(
                    start_time=2.2925,
                    end_time=4.381406,
                    description=(
                        "Female speech in Chinese: '当时我们一起出差的时候，你不是。'"
                    ),
                    confidence="high",
                    category="human_non_speech",
                ),
            ],
            residual_artifact_notes="model guessed from the scene",
        ),
        job,
    )

    assert isinstance(facts, SFXStemFacts)
    assert facts.clip_duration_seconds == job.clip_duration_seconds
    assert suppressed == 1
    assert [item.description for item in facts.events] == ["a short door click"]
    assert facts.residual_artifact_notes is None


def test_bounded_sfx_schema_publishes_hard_array_limits() -> None:
    schema = BoundedSFXStemFactsDraft.model_json_schema()
    assert schema["properties"]["continuous_layers"]["maxItems"] == (
        SFX_MAX_CONTINUOUS_LAYERS
    )
    assert schema["properties"]["events"]["maxItems"] == SFX_MAX_EVENTS

    event = BoundedSFXEventDraft(
        start_time=0.1,
        end_time=0.2,
        description="click",
        confidence="medium",
        category="physical",
    )
    with pytest.raises(ValidationError):
        BoundedSFXStemFactsDraft(
            clip_duration_seconds=1.0,
            continuous_layers=[],
            events=[event] * (SFX_MAX_EVENTS + 1),
        )


def test_sfx_draft_drops_nonpositive_duplicates_and_asr_leakage() -> None:
    draft = BoundedSFXStemFactsDraft(
        clip_duration_seconds=4.381406,
        continuous_layers=[],
        events=[
            BoundedSFXEventDraft(
                start_time=0.0,
                end_time=0.0,
                description="faint breath",
                confidence="medium",
                category="human_non_speech",
            ),
            BoundedSFXEventDraft(
                start_time=1.0,
                end_time=1.2,
                description="door click",
                confidence="medium",
                category="physical",
            ),
            BoundedSFXEventDraft(
                start_time=1.0,
                end_time=1.2,
                description="door click",
                confidence="medium",
                category="physical",
            ),
            BoundedSFXEventDraft(
                start_time=2.2925,
                end_time=4.381406,
                description="当时我们一起出差的时候，你不是。",
                confidence="high",
                category="human_non_speech",
            ),
        ],
    )

    facts, counts = _canonicalize_sfx_draft(draft, _job())

    assert facts.clip_duration_seconds == _job().clip_duration_seconds
    assert [item.description for item in facts.events] == ["door click"]
    assert counts == {
        "nonpositive": 1,
        "duplicate": 1,
        "asr_leakage": 1,
        "music_leakage": 0,
        "model_note_discarded": 0,
    }
    assert facts.residual_artifact_notes is None


def test_sfx_draft_drops_music_cross_stem_leakage_and_speculative_notes() -> None:
    job = _job(clip_duration_seconds=5.16175)
    draft = BoundedSFXStemFactsDraft(
        clip_duration_seconds=5.16175,
        continuous_layers=[
            {
                "start_time": 0.0,
                "end_time": 5.16175,
                "description": (
                    "Background music and ambient noise (club/bar atmosphere) "
                    "playing continuously."
                ),
                "confidence": "high",
            },
            {
                "start_time": 0.0,
                "end_time": 5.16175,
                "description": "steady room ambience and glass clinks",
                "confidence": "medium",
            },
        ],
        events=[],
        residual_artifact_notes=(
            "The audio likely contains background music typical of a bar, as suggested "
            "by the visual context."
        ),
    )

    facts, counts = _canonicalize_sfx_draft(draft, job)

    assert facts.clip_duration_seconds == job.clip_duration_seconds
    assert [item.description for item in facts.continuous_layers] == [
        "steady room ambience and glass clinks"
    ]
    assert facts.residual_artifact_notes is None
    assert counts == {
        "nonpositive": 0,
        "duplicate": 0,
        "asr_leakage": 0,
        "music_leakage": 1,
        "model_note_discarded": 1,
    }
