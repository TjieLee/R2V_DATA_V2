from __future__ import annotations

from types import SimpleNamespace

from r2v_data_v2.h3.mimo25_stem_facts_runtime import canonicalize_stem_facts
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MusicStemFacts,
    SFXContinuousLayer,
    SFXEvent,
    SFXStemFacts,
    SpeechStemFacts,
    SpeechStemSegmentFact,
)


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        clip_duration_seconds=4.38140625,
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
                    start_time=2.2925,
                    end_time=4.381406,
                    description=(
                        "Female speech in Chinese: '当时我们一起出差的时候，你不是。'"
                    ),
                    confidence="high",
                    category="human_non_speech",
                ),
                SFXEvent(
                    start_time=1.0,
                    end_time=1.2,
                    description="a short door click",
                    confidence="medium",
                    category="physical",
                ),
            ],
        ),
        job,
    )

    assert isinstance(facts, SFXStemFacts)
    assert facts.clip_duration_seconds == job.clip_duration_seconds
    assert suppressed == 1
    assert [item.description for item in facts.events] == ["a short door click"]
    assert facts.residual_artifact_notes is not None
    assert "separator speech leakage" in facts.residual_artifact_notes
