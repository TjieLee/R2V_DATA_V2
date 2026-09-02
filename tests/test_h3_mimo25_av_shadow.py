from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import r2v_data_v2.h3.mimo25_h3_materializer as mimo25_materializer
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalQwen3SpeechSegment,
    FinalSubjectVoice,
    FinalVisualReference,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MIMO25_FAILURE_VERSION,
    MIMO25_INVENTORY_VERSION,
    MIMO25_RECORD_VERSION,
    MimoClipJob,
    MimoFailure,
    MimoRecord,
    MimoReferenceImage,
    MimoSegmentEvidence,
    _inventory,
    _job,
    _validate_clip_segment_inventory,
    _validate_h3_variant_observations,
    run_mimo25_av_reconcile,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MODEL,
    MIMO25_POLICY_VERSION,
    MIMO25_PROMPT_VERSION,
    MIMO25_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    MimoAnnotationWarning,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendResult,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    MimoUsage,
    OpenAIMimo25Backend,
    validate_annotation,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    _materialize_sample,
    materialize_mimo25_h3_shadow,
)
from r2v_data_v2.h3.mimo25_human_review import (
    MimoHumanReviewAnnotation,
    MimoReviewCase,
    MimoReviewStore,
    make_review_server,
    render_review_html,
)
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract
from r2v_data_v2.structured_output import ValidationIssue


def _annotation(
    *,
    group: str = "g1",
    entity_id: str | None = "e1",
    composition: str = "single_speaker",
    resolution: str = "resolved",
) -> MimoAVAnnotationDraft:
    return MimoAVAnnotationDraft.model_validate(
        {
            "schema_version": MIMO25_SCHEMA_VERSION,
            "segment_decisions": [
                {
                    "segment_id": "segment_1",
                    "vocal_composition": composition,
                    "resolution": resolution,
                    "primary_speaker_group": group,
                    "binding_status": (
                        "visible_entity" if entity_id is not None else "offscreen"
                    ),
                    "entity_id": entity_id,
                    "secondary_vocal_activity": {
                        "present": composition != "single_speaker",
                        "speaker_relation": (
                            "none"
                            if composition == "single_speaker"
                            else
                            "same_speaker"
                            if composition == "same_speaker_nonlexical"
                            else "different_speaker"
                        ),
                        "kind": (
                            None
                            if composition == "single_speaker"
                            else "interjection"
                            if composition == "same_speaker_nonlexical"
                            else "speech"
                        ),
                    },
                    "confidence": "high",
                    "evidence_codes": ["av_temporal_alignment"],
                }
            ],
            "audio_semantics": {
                "temporal_non_speech_events": [
                    {
                        "event_id": "ae1",
                        "approximate_start_time": 0.1,
                        "approximate_end_time": 0.2,
                        "category": "physical",
                        "pattern": "repeated",
                        "description": "A short repeated clink is audible.",
                        "source_grounding": "audiovisually_grounded",
                    }
                ],
                "speaker_delivery": [
                    {"speaker_group": group, "delivery_style": "calm and clear"}
                ],
                "overall_soundscape": "Quiet speech with a short clink.",
                "non_diegetic_music_status": "absent",
                "non_diegetic_music": None,
                "audiovisual_summary": "One visible speaker talks in a quiet scene.",
            },
            "h3_draft": {
                "subject_definitions": [
                    "<Subject 1> is the person shown in <Picture 1>."
                ],
                "summary": "A person speaks while remaining visible.",
                "visual_retention_analysis": [
                    "<Subject 1>: fully_preserved - the person remains visible."
                ],
                "shots": [
                    {
                        "shot_index": 1,
                        "start_time": None,
                        "description_template": (
                            "<Subject 1> faces the camera. [[audio_event:ae1]] "
                            "[[segment:segment_1]]"
                        ),
                    }
                ],
            },
            "warnings": [],
        }
    )


def _job_fixture(tmp_path: Path) -> MimoClipJob:
    video = tmp_path / "target.mp4"
    audio = tmp_path / "full.flac"
    image = tmp_path / "reference.png"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    image.write_bytes(b"image")
    reference = MimoReferenceImage(
        image_index=1,
        picture_label="<Picture 1>",
        kind="subject",
        entity_id="e1",
        image_artifact_path=str(image.resolve()),
        image_sha256="1" * 64,
    )
    segment = MimoSegmentEvidence(
        segment_id="segment_1",
        start_time=0.0,
        end_time=1.0,
        source_start_sample=0,
        source_end_sample=16000,
        source_sample_rate_hz=16000,
        source_speaker_cluster_id="speaker_0",
        current_entity_id="e1",
        entity_occurrence_id="clip-1/e1",
        identity_scope="direct_anchor_present",
        direct_anchor_seconds=0.5,
        cluster_binding_status="candidate_mapped",
        overlapping_visible_entities=["e1"],
        direct_support_seconds_by_entity={"e1": 0.5},
        competing_visible_speaker_evidence=[],
        asr_status="transcribed",
        asr_text="Exact, text!",
        asr_language="English",
    )
    values = {
        "clip_uid": "clip-1",
        "r2v_instruction": "Use Image 1 to recreate the observed target clip.",
        "target_video_path": str(video.resolve()),
        "target_video_sha256": "2" * 64,
        "target_full_audio_path": str(audio.resolve()),
        "target_full_audio_sha256": "3" * 64,
        "target_duration_seconds": 1.0,
        "reference_images": [reference.model_dump(mode="json")],
        "reference_subjects": [
            RecaptionSubjectContract(
                subject_index=1,
                subject_label="<Subject 1>",
                kind="entity",
                entity_id="e1",
                source_picture_labels=["<Picture 1>"],
            ).model_dump(mode="json")
        ],
        "segments": [segment.model_dump(mode="json")],
        "source_h3_sample_ids": ["clip-1/in_pair"],
    }
    return _job(values)


def _validate(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str] | None = None,
    segment_intervals: dict[str, tuple[float, float]] | None = None,
    transcribed_segment_ids: list[str] | None = None,
    allowed_entity_ids: set[str] | None = None,
    allowed_reference_labels: set[str] | None = None,
    authoritative_transcripts: list[str] | None = None,
    reference_subjects: list[RecaptionSubjectContract] | None = None,
    target_duration_seconds: float = 1.0,
) -> list[ValidationIssue]:
    return validate_annotation(
        annotation,
        segment_ids=segment_ids or ["segment_1"],
        segment_intervals=(
            {"segment_1": (0.0, 1.0)}
            if segment_intervals is None
            else segment_intervals
        ),
        transcribed_segment_ids=(
            ["segment_1"]
            if transcribed_segment_ids is None
            else transcribed_segment_ids
        ),
        authoritative_transcripts=(
            ["Exact, text!"]
            if authoritative_transcripts is None
            else authoritative_transcripts
        ),
        allowed_entity_ids=allowed_entity_ids or {"e1"},
        allowed_reference_labels=(
            {"<Picture 1>", "<Subject 1>"}
            if allowed_reference_labels is None
            else allowed_reference_labels
        ),
        reference_subjects=(
            [
                RecaptionSubjectContract(
                    subject_index=1,
                    subject_label="<Subject 1>",
                    kind="entity",
                    entity_id="e1",
                    source_picture_labels=["<Picture 1>"],
                )
            ]
            if reference_subjects is None
            else reference_subjects
        ),
        target_duration_seconds=target_duration_seconds,
    )


class _Completions:
    def __init__(
        self,
        responses: list[
            tuple[str, int | None]
            | tuple[str, int | None, str]
            | tuple[str, int | None, str, int | None]
            | tuple[str, int | None, str, int | None, int | None]
            | tuple[
                str,
                int | None,
                str,
                int | None,
                int | None,
                int | None,
            ]
        ],
    ) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        raw, audio_tokens = response[:2]
        finish_reason = response[2] if len(response) >= 3 else "stop"
        reasoning_tokens = response[3] if len(response) >= 4 else 0
        video_tokens = response[4] if len(response) >= 5 else 10
        image_tokens = response[5] if len(response) >= 6 else 6
        details = {
            key: value
            for key, value in {
                "audio_tokens": audio_tokens,
                "video_tokens": video_tokens,
                "image_tokens": image_tokens,
            }.items()
            if value is not None
        } or None
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=raw), finish_reason=finish_reason
                )
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": details,
                "completion_tokens_details": (
                    None
                    if reasoning_tokens is None
                    else {"reasoning_tokens": reasoning_tokens}
                ),
            },
        )


class _RetryingCompletions(_Completions):
    def __init__(self, responses: list[tuple[str, int | None]]) -> None:
        super().__init__(responses)
        self.failures_remaining = 2

    def create(self, **kwargs: object) -> object:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("temporary connection reset")
        return super().create(**kwargs)


def _backend(
    tmp_path: Path,
    responses: list[
        tuple[str, int | None]
        | tuple[str, int | None, str]
        | tuple[str, int | None, str, int | None]
        | tuple[str, int | None, str, int | None, int | None]
        | tuple[str, int | None, str, int | None, int | None, int | None]
    ],
) -> tuple[OpenAIMimo25Backend, _Completions]:
    completions = _Completions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(media_resolver=resolver, api_key="secret"),
        client=client,
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    return backend, completions


def test_mimo_request_contract_and_embedded_audio(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    raw = _annotation().model_dump_json()
    backend, completions = _backend(tmp_path, [(raw, 8)])
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 1
    request = completions.requests[0]
    assert request["model"] == MIMO25_MODEL
    assert request["response_format"] == {"type": "json_object"}
    assert request["stream"] is False
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    content = request["messages"][1]["content"]  # type: ignore[index]
    video = next(item for item in content if item["type"] == "video_url")  # type: ignore[union-attr]
    assert video["video_url"] == {"url": video["video_url"]["url"]}
    assert video["fps"] == 4.0
    assert video["media_resolution"] == "default"
    assert "fps" not in video["video_url"]
    assert "media_resolution" not in video["video_url"]
    assert not any(item["type"] in {"audio_url", "input_audio"} for item in content)  # type: ignore[union-attr]
    assert [item["type"] for item in content[:2]] == ["text", "image_url"]  # type: ignore[index]
    assert "secret" not in json.dumps(request)
    assert MIMO25_PROMPT_VERSION in backend.provenance.model_dump_json()
    assert MIMO25_POLICY_VERSION in backend.provenance.model_dump_json()
    assert job.r2v_instruction in content[-1]["text"]  # type: ignore[index]


def test_mimo_v3_prompt_restores_dense_visual_and_audio_authority_contract() -> None:
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_unified_av_reconcile_v3"
    assert MIMO25_POLICY_VERSION == "h3_mimo25_av_authority_contract_v3"
    assert MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.3"
    for phrase in (
        "shot scale and framing",
        "foreground, midground, and background composition",
        "body, arm, hand, and head motion",
        "temporal progression through early, middle, and late portions",
        "300-450 English words",
        "attribute_transfer is forbidden",
        "Pictures are content references, not first frames, last frames, or keyframes",
        "must not quote or paraphrase dialogue",
    ):
        assert phrase in SYSTEM_PROMPT
    assert "SUPPLIED <Picture N> AND <Subject N> LABELS" in SYSTEM_PROMPT


def test_audio_token_zero_uses_one_canonical_audio_fallback(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    raw = _annotation().model_dump_json()
    backend, completions = _backend(tmp_path, [(raw, 0), (raw, 4)])
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 2
    assert result.input_modality == "target_video_plus_canonical_full_audio_fallback"
    assert len(completions.requests) == 2
    content = completions.requests[1]["messages"][1]["content"]  # type: ignore[index]
    audio_items = [item for item in content if item["type"] == "input_audio"]  # type: ignore[union-attr]
    assert len(audio_items) == 1
    assert set(audio_items[0]) == {"type", "input_audio"}
    assert audio_items[0]["input_audio"]["data"].startswith("data:audio/")
    assert ";base64," in audio_items[0]["input_audio"]["data"]
    assert not any(item["type"] == "audio_url" for item in content)  # type: ignore[union-attr]
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_http_audio_fallback_uses_input_audio_data(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    completions = _Completions(
        [(_annotation().model_dump_json(), 0), (_annotation().model_dump_json(), 3)]
    )
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=MimoMediaResolver(
                mode="http",
                media_root=tmp_path,
                media_base_url="https://media.example.test/root/",
            ),
            api_key="secret",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    content = completions.requests[1]["messages"][1]["content"]  # type: ignore[index]
    audio = next(item for item in content if item["type"] == "input_audio")
    assert audio == {
        "type": "input_audio",
        "input_audio": {"data": "https://media.example.test/root/full.flac"},
    }


def test_unknown_audio_token_usage_does_not_loop(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(tmp_path, [(_annotation().model_dump_json(), None)])
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 1
    assert len(completions.requests) == 1
    assert "audio_tokens_unavailable" in result.diagnostics[0].warnings


def test_unknown_av_token_usage_warns_without_loop(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), None, "stop", 0, None, None)],
    )
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert len(completions.requests) == 1
    assert {
        "prompt_tokens_details_unavailable",
        "image_tokens_unavailable",
        "video_tokens_unavailable",
        "audio_tokens_unavailable",
    } <= set(result.diagnostics[0].warnings)


def test_zero_reference_image_tokens_fail_closed(tmp_path: Path) -> None:
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 5, "stop", 0, 10, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_reference_images_not_observed"
    assert len(completions.requests) == 1


def test_unknown_reference_image_tokens_warn_without_retry(tmp_path: Path) -> None:
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 5, "stop", 0, 10, None)],
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert len(completions.requests) == 1
    assert "image_tokens_unavailable" in result.diagnostics[0].warnings


def test_audio_fallback_zero_reference_image_tokens_fail_closed(
    tmp_path: Path,
) -> None:
    raw = _annotation().model_dump_json()
    backend, completions = _backend(
        tmp_path,
        [(raw, 0, "stop", 0, 10, 6), (raw, 4, "stop", 0, 10, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_reference_images_not_observed"
    assert len(completions.requests) == 2


def test_explicit_zero_video_tokens_fail_closed(tmp_path: Path) -> None:
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 5, "stop", 0, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_target_video_not_observed"
    assert len(completions.requests) == 1


def test_audio_fallback_zero_video_tokens_fail_closed(tmp_path: Path) -> None:
    raw = _annotation().model_dump_json()
    backend, _ = _backend(
        tmp_path,
        [(raw, 0, "stop", 0, 10), (raw, 4, "stop", 0, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_target_video_not_observed"


def test_explicit_audio_fallback_zero_audio_tokens_fail_closed(tmp_path: Path) -> None:
    raw = _annotation().model_dump_json()
    backend, completions = _backend(tmp_path, [(raw, 0), (raw, 0)])
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_target_audio_not_observed"
    assert len(completions.requests) == 2


def test_explicit_audio_fallback_unknown_usage_warns_without_loop(
    tmp_path: Path,
) -> None:
    raw = _annotation().model_dump_json()
    backend, completions = _backend(
        tmp_path,
        [(raw, 0), (raw, None, "stop", 0, 10)],
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert len(completions.requests) == 2
    assert "audio_tokens_unavailable_after_explicit_fallback" in (
        result.diagnostics[1].warnings
    )


def test_http_retry_is_bounded_and_diagnostic(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    completions = _RetryingCompletions([(_annotation().model_dump_json(), 3)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
            api_key="secret",
        ),
        client=client,
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.http_retry_count == 2
    assert result.diagnostics[0].http_attempt_count == 3
    assert result.model_call_count == 1
    assert result.http_attempt_count == 3


def test_http_retry_failure_preserves_logical_and_http_counts(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)

    class AlwaysFails:
        def create(self, **_: object) -> object:
            raise OSError("offline")

    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
            api_key="secret",
        ),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=AlwaysFails())
        ),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            job,
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    failure = exc_info.value
    assert failure.model_call_count == 1
    assert failure.http_attempt_count == 3
    assert failure.http_retry_count == 2
    assert failure.diagnostics[0].http_attempt_count == 3


def test_retried_response_contract_failure_preserves_http_counts(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)

    class RetryThenMalformed:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object) -> object:
            self.calls += 1
            if self.calls < 3:
                raise OSError("temporary")
            return SimpleNamespace(choices=[], usage={})

    completions = RetryThenMalformed()
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
            api_key="secret",
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            job,
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    failure = exc_info.value
    assert failure.model_call_count == 1
    assert failure.http_attempt_count == 3
    assert failure.http_retry_count == 2
    assert failure.diagnostics[0].request_error is not None


def test_reasoning_token_diagnostic_under_disabled_thinking(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, _ = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 8, "stop", 17)],
    )
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.diagnostics[0].usage.reasoning_tokens == 17
    assert "reasoning_tokens_nonzero_under_disabled_thinking" in (
        result.diagnostics[0].warnings
    )


def test_length_finish_reason_fails_without_full_av_recheck(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 8, "length")],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            job,
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_output_truncated"
    assert exc_info.value.recheck_count == 0
    assert len(completions.requests) == 1


def test_audio_fallback_length_also_fails_without_full_av_recheck(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)
    raw = _annotation().model_dump_json()
    backend, completions = _backend(
        tmp_path,
        [(raw, 0, "stop"), (raw, 4, "length")],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            job,
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_output_truncated"
    assert exc_info.value.model_call_count == 2
    assert len(completions.requests) == 2


@pytest.mark.parametrize(
    "finish_reason",
    ["content_filter", "tool_calls", "function_call", "cancelled", "custom_reason"],
)
def test_explicit_non_stop_finish_reason_fails_without_recheck(
    tmp_path: Path,
    finish_reason: str,
) -> None:
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 5, finish_reason)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_incomplete_finish_reason"
    assert exc_info.value.recheck_count == 0
    assert len(completions.requests) == 1


def test_missing_finish_reason_warns_and_accepts(tmp_path: Path) -> None:
    backend, _ = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 5, None)],  # type: ignore[list-item]
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert "finish_reason_unavailable" in result.diagnostics[0].warnings


def test_true_malformed_output_uses_exactly_one_full_av_recheck(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(
        tmp_path,
        [("not json", 5), (_annotation().model_dump_json(), 5)],
    )
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 2
    assert result.recheck_count == 1
    assert len(completions.requests) == 2
    content = completions.requests[1]["messages"][1]["content"]  # type: ignore[index]
    assert any(item["type"] == "video_url" for item in content)
    assert any(item["type"] == "image_url" for item in content)
    assert "Reinspect the SAME audiovisual evidence" in content[-1]["text"]
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    recheck = content[-1]["text"]
    assert job.r2v_instruction in recheck
    assert job.target_video_path not in recheck


def test_both_malformed_responses_fail_with_nullable_issue_field(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(
        tmp_path,
        [("not json", 5), ("still not json", 5)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            job,
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    failure = exc_info.value
    assert failure.code == "mimo_structured_output_failed"
    assert failure.model_call_count == 2
    assert failure.recheck_count == 1
    assert len(completions.requests) == 2
    assert failure.issues[0].field is None


def test_semantic_issue_triggers_full_media_recheck(tmp_path: Path) -> None:
    invalid = _annotation().model_dump(mode="json")
    invalid["segment_decisions"][0]["entity_id"] = "e9"
    backend, completions = _backend(
        tmp_path,
        [
            (json.dumps(invalid), 5),
            (_annotation().model_dump_json(), 5),
        ],
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.recheck_count == 1
    assert result.diagnostics[-1].input_modality == "full_av_recheck_embedded_audio"
    content = completions.requests[-1]["messages"][1]["content"]  # type: ignore[index]
    assert {item["type"] for item in content} >= {"image_url", "video_url", "text"}


def test_audio_fallback_recheck_preserves_explicit_audio_modality(
    tmp_path: Path,
) -> None:
    invalid = _annotation().model_dump(mode="json")
    invalid["segment_decisions"][0]["entity_id"] = "e9"
    raw = _annotation().model_dump_json()
    backend, completions = _backend(
        tmp_path,
        [(raw, 0), (json.dumps(invalid), 4), (raw, 4)],
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 3
    assert result.recheck_count == 1
    assert result.diagnostics[-1].input_modality == (
        "full_av_recheck_with_canonical_audio"
    )
    content = completions.requests[-1]["messages"][1]["content"]  # type: ignore[index]
    assert any(item["type"] == "video_url" for item in content)
    assert any(item["type"] == "input_audio" for item in content)
    assert all(isinstance(request["messages"][1]["content"], list) for request in completions.requests)  # type: ignore[index]


def test_full_av_recheck_zero_video_tokens_fails_closed(tmp_path: Path) -> None:
    backend, _ = _backend(
        tmp_path,
        [("not json", 5), (_annotation().model_dump_json(), 5, "stop", 0, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_target_video_not_observed"
    assert exc_info.value.recheck_count == 1


def test_full_av_recheck_zero_reference_image_tokens_fails_closed(
    tmp_path: Path,
) -> None:
    backend, _ = _backend(
        tmp_path,
        [
            ("not json", 5, "stop", 0, 10, 6),
            (_annotation().model_dump_json(), 5, "stop", 0, 10, 0),
        ],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_reference_images_not_observed"
    assert exc_info.value.recheck_count == 1


def test_explicit_audio_recheck_zero_audio_tokens_fails_closed(
    tmp_path: Path,
) -> None:
    invalid = _annotation().model_dump(mode="json")
    invalid["segment_decisions"][0]["entity_id"] = "e9"
    raw = _annotation().model_dump_json()
    backend, _ = _backend(
        tmp_path,
        [(raw, 0), (json.dumps(invalid), 4), (raw, 0)],
    )
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert exc_info.value.code == "mimo_target_audio_not_observed"
    assert exc_info.value.recheck_count == 1


def test_speaker_entity_contradiction_triggers_full_media_recheck(
    tmp_path: Path,
) -> None:
    invalid = _two_segment_annotation(second_group="g1", second_entity="e2")
    valid = _two_segment_annotation(second_group="g1", second_entity=None)
    backend, completions = _backend(
        tmp_path,
        [(invalid.model_dump_json(), 5), (valid.model_dump_json(), 5)],
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
        allowed_entity_ids={"e1", "e2"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.recheck_count == 1
    content = completions.requests[1]["messages"][1]["content"]  # type: ignore[index]
    assert any(item["type"] == "video_url" for item in content)
    assert "speaker_group_entity_contradiction" in content[-1]["text"]


def test_base64_oversize_fails_closed(tmp_path: Path) -> None:
    media = tmp_path / "large.mp4"
    media.write_bytes(b"1234")
    resolver = MimoMediaResolver(
        mode="base64", media_root=tmp_path, maximum_base64_bytes=4
    )
    with pytest.raises(MimoBackendFailure, match="Base64 media exceeds"):
        resolver.resolve(media)


def test_r2v_instruction_is_fingerprinted_and_variant_disagreement_fails(
    tmp_path: Path,
) -> None:
    first = _job_fixture(tmp_path)
    values = first.model_dump(mode="json", exclude={"request_fingerprint"})
    values["r2v_instruction"] = "A different deterministic task intent."
    second = _job(values)
    assert first.request_fingerprint != second.request_fingerprint

    sample = _sample(tmp_path)
    changed = sample.model_copy(update={"r2v_instruction": "Different instruction."})
    with pytest.raises(ValueError, match="variants disagree"):
        _validate_h3_variant_observations("clip-1", [sample, changed])


def test_only_subject_reference_entities_are_speaker_bindable(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    image = tmp_path / "object.png"
    image.write_bytes(b"object")
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["reference_images"].append(
        MimoReferenceImage(
            image_index=2,
            picture_label="<Picture 2>",
            kind="object",
            entity_id="e2",
            image_artifact_path=str(image.resolve()),
            image_sha256="4" * 64,
        ).model_dump(mode="json")
    )
    with_object = _job(values)
    contract = OpenAIMimo25Backend.build_compact_task_contract(with_object)
    assert contract["allowed_speaker_bindable_entity_ids"] == ["e1"]

    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["entity_id"] = "e2"
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        allowed_entity_ids={"e1"},
    )
    assert "unknown_entity" in {item.code for item in issues}


def test_segment_inventory_requires_exact_four_way_coverage() -> None:
    def row(segment_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            target_clip_uid="clip-1",
            clip_uid="clip-1",
            segment_id=segment_id,
        )

    complete = [row("segment_1"), row("segment_2")]
    _validate_clip_segment_inventory(
        "clip-1",
        raw=complete,
        bound=complete,
        asr=complete,
        audits=complete,
    )
    with pytest.raises(ValueError, match="segment inventories differ"):
        _validate_clip_segment_inventory(
            "clip-1",
            raw=complete,
            bound=complete,
            asr=complete[:1],
            audits=complete,
        )


def test_segment_contract_preserves_multi_vocal_segments() -> None:
    same = _annotation(composition="same_speaker_nonlexical")
    assert same.segment_decisions[0].primary_speaker_group == "g1"
    assert same.segment_decisions[0].resolution == "resolved"
    overlap = _annotation(
        composition="overlapping_secondary_speech",
        resolution="needs_acoustic_refinement",
    )
    assert len(overlap.segment_decisions) == 1
    with pytest.raises(ValidationError, match="requires acoustic refinement"):
        _annotation(composition="sequential_multi_speaker_speech")
    assert "Multiple vocal sounds inside one segment never make that segment invalid" in SYSTEM_PROMPT
    assert "transcript" not in MimoAVAnnotationDraft.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("composition", "relation", "kind", "resolution"),
    [
        ("single_speaker", "none", None, "resolved"),
        ("same_speaker_nonlexical", "same_speaker", "sigh", "resolved"),
        (
            "secondary_non_speech_vocalization",
            "different_speaker",
            "laughter",
            "resolved",
        ),
        (
            "secondary_non_speech_vocalization",
            "uncertain",
            "cough",
            "resolved",
        ),
        (
            "overlapping_secondary_speech",
            "different_speaker",
            "speech",
            "needs_acoustic_refinement",
        ),
        (
            "sequential_multi_speaker_speech",
            "different_speaker",
            "speech",
            "needs_acoustic_refinement",
        ),
    ],
)
def test_vocal_composition_cross_field_contract_accepts_valid_combinations(
    composition: str,
    relation: str,
    kind: str | None,
    resolution: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision["vocal_composition"] = composition
    decision["resolution"] = resolution
    decision["secondary_vocal_activity"] = {
        "present": composition != "single_speaker",
        "speaker_relation": relation,
        "kind": kind,
    }
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert len(annotation.segment_decisions) == 1


@pytest.mark.parametrize(
    ("composition", "relation", "kind", "resolution"),
    [
        ("single_speaker", "same_speaker", "sigh", "resolved"),
        ("same_speaker_nonlexical", "different_speaker", "sigh", "resolved"),
        (
            "secondary_non_speech_vocalization",
            "different_speaker",
            "speech",
            "resolved",
        ),
        (
            "overlapping_secondary_speech",
            "different_speaker",
            "speech",
            "resolved",
        ),
    ],
)
def test_vocal_composition_cross_field_contract_rejects_contradictions(
    composition: str,
    relation: str,
    kind: str,
    resolution: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision["vocal_composition"] = composition
    decision["resolution"] = resolution
    decision["secondary_vocal_activity"] = {
        "present": True,
        "speaker_relation": relation,
        "kind": kind,
    }
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)


def _two_segment_annotation(
    *,
    second_group: str,
    second_entity: str | None,
) -> MimoAVAnnotationDraft:
    payload = _annotation().model_dump(mode="json")
    second = dict(payload["segment_decisions"][0])
    second["segment_id"] = "segment_2"
    second["primary_speaker_group"] = second_group
    second["binding_status"] = (
        "visible_entity" if second_entity is not None else "offscreen"
    )
    second["entity_id"] = second_entity
    payload["segment_decisions"].append(second)
    payload["h3_draft"]["shots"][0]["description_template"] += (
        " Then [[segment:segment_2]]"
    )
    if second_group != "g1":
        payload["audio_semantics"]["speaker_delivery"].append(
            {"speaker_group": second_group, "delivery_style": "brief and clear"}
        )
    return MimoAVAnnotationDraft.model_validate(payload)


def test_speaker_group_entity_consistency_rules() -> None:
    same_group_offscreen = _two_segment_annotation(
        second_group="g1", second_entity=None
    )
    assert not _validate(
        same_group_offscreen,
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
    )

    group_changes_entity = _two_segment_annotation(
        second_group="g1", second_entity="e2"
    )
    issues = _validate(
        group_changes_entity,
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
        allowed_entity_ids={"e1", "e2"},
    )
    assert "speaker_group_entity_contradiction" in {item.code for item in issues}

    entity_changes_group = _two_segment_annotation(
        second_group="g2", second_entity="e1"
    )
    issues = _validate(
        entity_changes_group,
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
    )
    assert "visible_entity_speaker_group_contradiction" in {
        item.code for item in issues
    }


def test_cross_reference_validation_rejects_inventory_drift() -> None:
    annotation = _annotation()
    issues = _validate(
        annotation,
        segment_ids=["segment_1", "missing"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e2"},
    )
    assert {item.code for item in issues} == {"segment_inventory_mismatch", "unknown_entity"}


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    [
        ("summary", "A summary [[segment:segment_1]]", "speech_placeholder_outside_shot"),
        (
            "visual_retention_analysis",
            ["<Picture 1>: fully_preserved [[segment:segment_1]]"],
            "speech_placeholder_outside_shot",
        ),
    ],
)
def test_speech_placeholders_are_allowed_only_in_shots(
    field: str,
    value: object,
    issue_code: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["description_template"] = (
        "<Subject 1> remains visible."
    )
    payload["h3_draft"][field] = value
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert issue_code in {item.code for item in issues}


@pytest.mark.parametrize(
    ("draft_text", "issue_code"),
    [
        ("<Video 1> shows the target.", "draft_contains_pipeline_owned_syntax"),
        ("[Shot 1] A person stands.", "draft_contains_pipeline_owned_syntax"),
        (
            "<Picture 1> is the target keyframe.",
            "unassigned_picture_keyframe_role",
        ),
        (
            "The person says [[segment:segment_1]]",
            "draft_prefixes_complete_speech_placeholder",
        ),
    ],
)
def test_h3_draft_rejects_pipeline_owned_or_ambiguous_syntax(
    draft_text: str,
    issue_code: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    if "[[segment:" in draft_text:
        payload["h3_draft"]["shots"][0]["description_template"] = draft_text
    else:
        payload["h3_draft"]["summary"] = draft_text
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert issue_code in {item.code for item in issues}


@pytest.mark.parametrize(
    "marker",
    ["fully_preserved", "partially_preserved", "weak_reference"],
)
def test_allowed_visual_retention_markers_pass(marker: str) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        f"<Subject 1>: {marker} - observed appearance remains grounded."
    ]
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert not {
        "unassigned_attribute_transfer",
        "unknown_visual_retention_marker",
    } & {item.code for item in issues}


def test_attribute_transfer_and_unknown_retention_marker_reject() -> None:
    for marker, expected in (
        ("attribute_transfer", "unassigned_attribute_transfer"),
        ("invented_marker", "unknown_visual_retention_marker"),
    ):
        payload = _annotation().model_dump(mode="json")
        payload["h3_draft"]["visual_retention_analysis"] = [
            f"<Subject 1>: {marker} - invalid."
        ]
        issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
        assert expected in {item.code for item in issues}


def _two_audio_event_annotation() -> MimoAVAnnotationDraft:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"].append(
        {
            "event_id": "ae2",
            "approximate_start_time": 0.3,
            "approximate_end_time": 0.4,
            "category": "environmental",
            "pattern": "single",
            "description": "A brief distant horn is audible.",
            "source_grounding": "audible_only",
        }
    )
    payload["h3_draft"]["shots"][0]["description_template"] = (
        "<Subject 1> remains visible. [[audio_event:ae1]] "
        "[[segment:segment_1]] [[audio_event:ae2]]"
    )
    return MimoAVAnnotationDraft.model_validate(payload)


def test_audio_event_ids_are_contiguous_and_chronological() -> None:
    annotation = _two_audio_event_annotation()
    assert [
        item.event_id for item in annotation.audio_semantics.temporal_non_speech_events
    ] == ["ae1", "ae2"]
    payload = annotation.model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"][1]["event_id"] = "ae3"
    with pytest.raises(ValidationError, match="IDs must be contiguous"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_simultaneous_double_digit_audio_event_ids_are_chronological() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"] = [
        {
            "event_id": f"ae{index}",
            "approximate_start_time": 0.1,
            "approximate_end_time": 0.2,
            "category": "physical",
            "pattern": "single",
            "description": f"Audible event {index} occurs.",
            "source_grounding": "audible_only",
        }
        for index in range(1, 11)
    ]
    payload["h3_draft"]["shots"][0]["description_template"] = " ".join(
        [*(f"[[audio_event:ae{index}]]" for index in range(1, 11)), "[[segment:segment_1]]"]
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert len(annotation.audio_semantics.temporal_non_speech_events) == 10


@pytest.mark.parametrize(
    ("template", "issue_code"),
    [
        (
            "[[audio_event:ae1]] [[segment:segment_1]]",
            "missing_audio_event_placeholder",
        ),
        (
            "[[audio_event:ae1]] [[audio_event:ae1]] "
            + "[[segment:segment_1]] [[audio_event:ae2]]",
            "duplicate_audio_event_placeholder",
        ),
        (
            "[[audio_event:ae1]] [[segment:segment_1]] [[audio_event:ae3]]",
            "unknown_audio_event_placeholder",
        ),
        (
            "[[audio_event:ae2]] [[segment:segment_1]] [[audio_event:ae1]]",
            "audio_event_placeholder_order_mismatch",
        ),
    ],
)
def test_audio_event_placeholder_coverage_is_exact(
    template: str,
    issue_code: str,
) -> None:
    payload = _two_audio_event_annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["description_template"] = template
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert issue_code in {item.code for item in issues}


def test_audio_event_placeholder_is_allowed_only_in_shots() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["summary"] += " [[audio_event:ae1]]"
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "audio_event_placeholder_outside_shot" in {item.code for item in issues}


def test_audiovisual_summary_is_required_and_nonempty() -> None:
    payload = _annotation().model_dump(mode="json")
    del payload["audio_semantics"]["audiovisual_summary"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)
    payload["audio_semantics"]["audiovisual_summary"] = " "
    with pytest.raises(ValidationError, match="summary must not be empty"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_resolved_transcribed_speaker_delivery_requires_full_group_coverage() -> None:
    annotation = _two_segment_annotation(second_group="g2", second_entity="e2")
    payload = annotation.model_dump(mode="json")
    payload["audio_semantics"]["speaker_delivery"] = payload["audio_semantics"][
        "speaker_delivery"
    ][:1]
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
        allowed_entity_ids={"e1", "e2"},
    )
    assert "missing_resolved_transcribed_speaker_delivery" in {
        item.code for item in issues
    }


def test_every_supplied_subject_requires_exactly_one_retention_line() -> None:
    subjects = [
        RecaptionSubjectContract(
            subject_index=1,
            subject_label="<Subject 1>",
            kind="entity",
            entity_id="e1",
            source_picture_labels=["<Picture 1>"],
        ),
        RecaptionSubjectContract(
            subject_index=2,
            subject_label="<Subject 2>",
            kind="entity",
            entity_id="e2",
            source_picture_labels=["<Picture 2>"],
        ),
    ]
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"].append(
        "<Subject 2> is sourced from <Picture 2>."
    )
    payload["h3_draft"]["visual_retention_analysis"].append(
        "<Subject 2>: weak_reference - only limited observed structure is retained."
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert not _validate(
        annotation,
        allowed_entity_ids={"e1", "e2"},
        allowed_reference_labels={
            "<Picture 1>",
            "<Picture 2>",
            "<Subject 1>",
            "<Subject 2>",
        },
        reference_subjects=subjects,
    )
    payload = annotation.model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        "<Subject 1>: fully_preserved - retained.",
        "<Subject 1>: weak_reference - duplicated.",
    ]
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        allowed_reference_labels={
            "<Picture 1>",
            "<Picture 2>",
            "<Subject 1>",
            "<Subject 2>",
        },
        reference_subjects=subjects,
    )
    assert "subject_retention_contract_mismatch" in {item.code for item in issues}


@pytest.mark.parametrize(
    "extra_definition",
    [
        "An extra arbitrary definition.",
        "<Picture 1> defines the person.",
    ],
)
def test_subject_definitions_reject_noncanonical_extra_rows(
    extra_definition: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"].append(extra_definition)
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "subject_definition_contract_mismatch" in {item.code for item in issues}


@pytest.mark.parametrize(
    "retention_rows",
    [
        ["<Picture 1>: fully_preserved - visible."],
        ["fully_preserved - visible."],
        ["<Subject 1>: fully_preserved weak_reference - conflicting markers."],
        ["<Subject 1> and <Subject 2>: fully_preserved - invalid owner."],
    ],
)
def test_visual_retention_requires_exact_canonical_subject_rows(
    retention_rows: list[str],
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = retention_rows
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        allowed_reference_labels={"<Picture 1>", "<Subject 1>", "<Subject 2>"},
    )
    assert "subject_retention_contract_mismatch" in {item.code for item in issues}


def test_significant_authoritative_transcript_cannot_leak_into_draft() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["summary"] = "请把这扇门轻轻关上"
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        authoritative_transcripts=["请把这扇门轻轻关上"],
    )
    assert "draft_contains_authoritative_transcript" in {item.code for item in issues}
    assert "draft_contains_authoritative_transcript" not in {
        item.code
        for item in _validate(
            _annotation(),
            authoritative_transcripts=["嗯"],
        )
    }


@pytest.mark.parametrize(
    "field",
    ["event", "delivery", "soundscape", "music", "summary"],
)
def test_significant_authoritative_transcript_cannot_leak_into_audio_semantics(
    field: str,
) -> None:
    transcript = "请把这扇门轻轻关上"
    payload = _annotation().model_dump(mode="json")
    semantics = payload["audio_semantics"]
    if field == "event":
        semantics["temporal_non_speech_events"][0]["description"] = transcript
    elif field == "delivery":
        semantics["speaker_delivery"][0]["delivery_style"] = transcript
    elif field == "soundscape":
        semantics["overall_soundscape"] = transcript
    elif field == "music":
        semantics["non_diegetic_music_status"] = "present"
        semantics["non_diegetic_music"] = transcript
    else:
        semantics["audiovisual_summary"] = transcript
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        authoritative_transcripts=[transcript],
    )
    assert "audio_semantics_contains_authoritative_transcript" in {
        item.code for item in issues
    }


@pytest.mark.parametrize("transcript", ["嗯", "OK"])
def test_short_authoritative_text_does_not_trigger_audio_semantics_leakage(
    transcript: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["audiovisual_summary"] = transcript
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        authoritative_transcripts=[transcript],
    )
    assert "audio_semantics_contains_authoritative_transcript" not in {
        item.code for item in issues
    }


def _two_shot_annotation(
    *,
    speech_shot: int = 1,
    event_shot: int = 1,
    event_interval: tuple[float, float] = (1.0, 1.5),
) -> MimoAVAnnotationDraft:
    payload = _annotation().model_dump(mode="json")
    event = payload["audio_semantics"]["temporal_non_speech_events"][0]
    event["approximate_start_time"], event["approximate_end_time"] = event_interval
    shot_text = {1: ["<Subject 1> remains visible."], 2: ["The scene continues."]}
    shot_text[speech_shot].append("[[segment:segment_1]]")
    shot_text[event_shot].append("[[audio_event:ae1]]")
    payload["h3_draft"]["shots"] = [
        {
            "shot_index": 1,
            "start_time": None,
            "description_template": " ".join(shot_text[1]),
        },
        {
            "shot_index": 2,
            "start_time": 5.0,
            "description_template": " ".join(shot_text[2]),
        },
    ]
    return MimoAVAnnotationDraft.model_validate(payload)


def test_speech_placeholder_must_overlap_its_shot() -> None:
    issues = _validate(
        _two_shot_annotation(speech_shot=2),
        segment_intervals={"segment_1": (1.0, 2.0)},
        target_duration_seconds=10.0,
    )
    assert "speech_placeholder_wrong_shot" in {item.code for item in issues}


@pytest.mark.parametrize("speech_shot", [1, 2])
def test_speech_placeholder_crossing_cut_may_use_either_overlapping_shot(
    speech_shot: int,
) -> None:
    issues = _validate(
        _two_shot_annotation(speech_shot=speech_shot),
        segment_intervals={"segment_1": (4.8, 5.2)},
        target_duration_seconds=10.0,
    )
    assert "speech_placeholder_wrong_shot" not in {item.code for item in issues}


def test_audio_event_placeholder_must_overlap_its_shot() -> None:
    issues = _validate(
        _two_shot_annotation(event_shot=2),
        segment_intervals={"segment_1": (1.0, 2.0)},
        target_duration_seconds=10.0,
    )
    assert "audio_event_placeholder_wrong_shot" in {item.code for item in issues}


@pytest.mark.parametrize("event_shot", [1, 2])
def test_audio_event_crossing_cut_may_use_either_overlapping_shot(
    event_shot: int,
) -> None:
    issues = _validate(
        _two_shot_annotation(event_shot=event_shot, event_interval=(4.9, 5.1)),
        segment_intervals={"segment_1": (1.0, 2.0)},
        target_duration_seconds=10.0,
    )
    assert "audio_event_placeholder_wrong_shot" not in {
        item.code for item in issues
    }


def test_subject_definition_and_shot_bounds_fail_closed() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"] = [
        "<Subject 1> has no supplied source Picture."
    ]
    payload["h3_draft"]["shots"].append(
        {
            "shot_index": 2,
            "start_time": 1.0,
            "description_template": "A hard cut occurs.",
        }
    )
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        target_duration_seconds=1.0,
    )
    assert {item.code for item in issues} >= {
        "subject_definition_contract_mismatch",
        "shot_start_outside_target",
    }


def test_subject_definition_rejects_wrong_supplied_picture() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"] = [
        "<Subject 1> is shown only in <Picture 2>."
    ]
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        allowed_reference_labels={"<Picture 1>", "<Picture 2>", "<Subject 1>"},
    )
    assert "subject_definition_contract_mismatch" in {item.code for item in issues}


def test_warning_requires_known_segment_and_never_contains_replacement_text() -> None:
    warning = MimoAnnotationWarning(
        code="possible_asr_conflict", segment_id="segment_1"
    )
    assert warning.model_dump(mode="json") == {
        "code": "possible_asr_conflict",
        "segment_id": "segment_1",
    }
    assert "replacement_text" not in MimoAnnotationWarning.model_json_schema()[
        "properties"
    ]
    payload = _annotation().model_dump(mode="json")
    payload["warnings"] = [
        {"code": "possible_asr_conflict", "segment_id": "unknown"}
    ]
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "warning_unknown_segment" in {item.code for item in issues}


def _sample(tmp_path: Path) -> FinalH3SampleV2:
    video = tmp_path / "target.mp4"
    audio = tmp_path / "full.flac"
    voice = tmp_path / "voice.flac"
    image = tmp_path / "reference.png"
    for path, value in ((video, b"v"), (audio, b"a"), (voice, b"x"), (image, b"i")):
        path.write_bytes(value)
    return FinalH3SampleV2(
        sample_id="clip-1/in_pair",
        pair_id="in_pair/clip-1",
        pair_type="in_pair",
        clip_uid="clip-1",
        clip_display_path="01/show/episode/clip-1",
        media_collection_relpath="01/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-1",
        shard_id="shard",
        target_video=str(video),
        target_full_audio_path=str(audio),
        r2v_instruction="Use Image 1.",
        visual_references=[
            FinalVisualReference(
                image_id="image_1",
                image_index=1,
                kind="subject",
                image_path="selected/reference.png",
                image_artifact_path=str(image),
                entity_id="e1",
                source_frame_index=0,
                scope="full",
                visible_region="whole",
                synthetic=False,
            )
        ],
        subject_voices=[
            FinalSubjectVoice(
                subject_index=1,
                entity_id="e1",
                target_occurrence_id="clip-1/e1",
                voice_reference_path=str(voice),
                voice_source="target",
            )
        ],
        speech_segments=[
            FinalQwen3SpeechSegment(
                segment_id="segment_1",
                speaker_cluster_id="speaker_0",
                entity_id="e1",
                entity_occurrence_id="clip-1/e1",
                source_start_sample=0,
                source_end_sample=16000,
                source_sample_rate_hz=16000,
                start_time=0.0,
                end_time=1.0,
                text="Exact, text!",
                language="English",
            )
        ],
    )


def _record_fixture(
    tmp_path: Path,
    annotation: MimoAVAnnotationDraft,
    *,
    job: MimoClipJob | None = None,
    inventory_fingerprint: str = "a" * 64,
) -> MimoRecord:
    active_job = job or _job_fixture(tmp_path)
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    provenance = MimoBackendConfig(media_resolver=resolver, api_key="secret").provenance()
    values = {
        "schema_version": MIMO25_RECORD_VERSION,
        "clip_uid": active_job.clip_uid,
        "request_fingerprint": active_job.request_fingerprint,
        "inventory_fingerprint": inventory_fingerprint,
        "status": "ready",
        "backend_provenance": provenance.model_dump(mode="json"),
        "annotation": annotation.model_dump(mode="json"),
        "failure": None,
        "input_modality": "target_video_with_embedded_audio",
        "model_call_count": 1,
        "http_attempt_count": 1,
        "raw_response_count": 1,
        "http_retry_count": 0,
        "recheck_count": 0,
    }
    fingerprint = __import__("hashlib").sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MimoRecord(**values, record_fingerprint=fingerprint)


def _replace_record(record: MimoRecord, **changes: object) -> MimoRecord:
    values = record.model_dump(mode="json", exclude={"record_fingerprint"})
    values.update(changes)
    fingerprint = __import__("hashlib").sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return MimoRecord(**values, record_fingerprint=fingerprint)


def _write_models_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def _materializer_fixture(
    tmp_path: Path,
    *,
    source_sample_count: int = 1,
    declared_sample_ids: list[str] | None = None,
) -> SimpleNamespace:
    media_root = tmp_path / "media"
    media_root.mkdir()
    sample = _sample(media_root)
    raw_job = _job_fixture(media_root)
    samples = [sample]
    for index in range(2, source_sample_count + 1):
        payload = sample.model_dump(mode="python")
        payload["sample_id"] = f"clip-1/variant-{index}"
        payload["pair_id"] = f"in_pair/clip-1-variant-{index}"
        samples.append(FinalH3SampleV2.model_validate(payload))
    source_h3 = tmp_path / "source-h3"
    samples_path = source_h3 / "samples.jsonl"
    _write_models_jsonl(samples_path, samples)
    job_values = raw_job.model_dump(mode="json", exclude={"request_fingerprint"})
    job_values.update(
        {
            "target_video_sha256": _file_sha256(Path(raw_job.target_video_path)),
            "target_full_audio_sha256": _file_sha256(
                Path(raw_job.target_full_audio_path)
            ),
            "source_h3_sample_ids": (
                [item.sample_id for item in samples]
                if declared_sample_ids is None
                else declared_sample_ids
            ),
        }
    )
    job_values["reference_images"][0]["image_sha256"] = _file_sha256(
        Path(raw_job.reference_images[0].image_artifact_path)
    )
    job = _job(job_values)
    inventory_values = {
        "schema_version": MIMO25_INVENTORY_VERSION,
        "inventory_scope": "current_diarization_asr_target_inventory",
        "canonical_wide_coverage": False,
        "source_visual_inventory_sha256": "1" * 64,
        "source_canonical_audio_manifest_sha256": "2" * 64,
        "source_diarization_raw_segments_sha256": "3" * 64,
        "source_diarization_bound_segments_sha256": "4" * 64,
        "source_qwen3_asr_segments_sha256": "5" * 64,
        "source_binding_audit_segments_sha256": "6" * 64,
        "source_h3_samples_sha256": _file_sha256(samples_path),
        "clip_count": 1,
        "jobs": [job.model_dump(mode="json")],
    }
    inventory = _inventory(inventory_values)
    record = _record_fixture(
        media_root,
        _annotation(),
        job=job,
        inventory_fingerprint=inventory.inventory_fingerprint,
    )
    mimo_root = tmp_path / "mimo"
    mimo_root.mkdir()
    (mimo_root / "inventory.json").write_text(
        inventory.model_dump_json(), encoding="utf-8"
    )
    _write_models_jsonl(mimo_root / "records.jsonl", [record])
    return SimpleNamespace(
        mimo_root=mimo_root,
        source_h3=source_h3,
        samples_path=samples_path,
        output_root=tmp_path / "materialized",
        job=job,
        inventory=inventory,
        record=record,
        samples=samples,
    )


def test_materializer_preserves_exact_asr_and_segment(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    record = _record_fixture(tmp_path, _annotation(composition="same_speaker_nonlexical"))
    corrected, rendered, warnings = _materialize_sample(sample, job, record)
    assert len(corrected) == 1
    assert corrected[0].text == "Exact, text!"
    assert corrected[0].language == "English"
    assert corrected[0].start_time == 0.0
    assert corrected[0].end_time == 1.0
    assert "<d>[English] Exact, text!</d>" in rendered
    assert "A short repeated clink is audible." in rendered
    assert "[[audio_event:" not in rendered
    assert "<Subject 1> (S1)" in rendered
    assert not warnings or all("words" in item for item in warnings)


def test_materializer_inserts_every_audio_event_description_exactly_once(
    tmp_path: Path,
) -> None:
    annotation = _two_audio_event_annotation()
    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )
    for event in annotation.audio_semantics.temporal_non_speech_events:
        assert rendered.count(event.description) == 1
    assert "[[audio_event:" not in rendered


def test_materializer_retains_refinement_segment_without_entity(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    annotation = _annotation(
        entity_id=None,
        composition="overlapping_secondary_speech",
        resolution="needs_acoustic_refinement",
    )
    record = _record_fixture(tmp_path, annotation)
    corrected, rendered, warnings = _materialize_sample(sample, job, record)
    assert len(corrected) == 1
    assert corrected[0].entity_id is None
    assert "(S1) says, <d>[English] Exact, text!</d>" in rendered
    assert "<Subject 1> (S1)" not in rendered
    assert "segment_1:acoustic_refinement_unresolved" in warnings


def test_materializer_rejects_source_speaker_cluster_drift(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    payload = sample.model_dump(mode="python")
    payload["speech_segments"][0]["speaker_cluster_id"] = "speaker_changed"
    changed = FinalH3SampleV2.model_validate(payload)
    with pytest.raises(ValueError, match="authoritative Qwen3-ASR"):
        _materialize_sample(
            changed,
            _job_fixture(tmp_path),
            _record_fixture(tmp_path, _annotation()),
        )


def test_materializer_preserves_structured_asr_warning_location(tmp_path: Path) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["warnings"] = [
        {"code": "possible_asr_conflict", "segment_id": "segment_1"}
    ]
    _, _, warnings = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, MimoAVAnnotationDraft.model_validate(payload)),
    )
    assert "segment_1:possible_asr_conflict" in warnings


def test_materializer_provenance_preflight_accepts_unchanged_inputs(
    tmp_path: Path,
) -> None:
    fixture = _materializer_fixture(tmp_path)
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
    )
    assert summary.ready_count == 1
    assert summary.failed_count == 0


def test_materializer_rejects_stale_source_h3_samples_sha(tmp_path: Path) -> None:
    fixture = _materializer_fixture(tmp_path)
    fixture.samples_path.write_bytes(fixture.samples_path.read_bytes() + b"\n")
    with pytest.raises(
        ValueError,
        match="MiMo source H3 inventory changed after AV annotation",
    ):
        materialize_mimo25_h3_shadow(
            mimo_root=fixture.mimo_root,
            source_h3_root=fixture.source_h3,
            output_root=fixture.output_root,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("inventory_fingerprint", "record inventory fingerprint mismatch"),
        ("request_fingerprint", "record request fingerprint mismatch"),
    ],
)
def test_materializer_rejects_record_inventory_or_request_drift(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    fixture = _materializer_fixture(tmp_path)
    changed = _replace_record(fixture.record, **{field: "f" * 64})
    _write_models_jsonl(fixture.mimo_root / "records.jsonl", [changed])
    with pytest.raises(ValueError, match=message):
        materialize_mimo25_h3_shadow(
            mimo_root=fixture.mimo_root,
            source_h3_root=fixture.source_h3,
            output_root=fixture.output_root,
        )


@pytest.mark.parametrize(
    ("source_sample_count", "declared_sample_ids"),
    [
        (2, ["clip-1/in_pair"]),
        (1, ["clip-1/in_pair", "clip-1/removed-variant"]),
    ],
)
def test_materializer_rejects_source_h3_sample_id_addition_or_removal(
    tmp_path: Path,
    source_sample_count: int,
    declared_sample_ids: list[str],
) -> None:
    fixture = _materializer_fixture(
        tmp_path,
        source_sample_count=source_sample_count,
        declared_sample_ids=declared_sample_ids,
    )
    with pytest.raises(ValueError, match="source H3 sample IDs changed"):
        materialize_mimo25_h3_shadow(
            mimo_root=fixture.mimo_root,
            source_h3_root=fixture.source_h3,
            output_root=fixture.output_root,
        )


@pytest.mark.parametrize(
    ("path_kind", "message"),
    [
        ("video", "target video changed"),
        ("audio", "target full audio changed"),
        ("image", "frozen reference image changed"),
    ],
)
def test_materializer_rejects_media_byte_drift(
    tmp_path: Path,
    path_kind: str,
    message: str,
) -> None:
    fixture = _materializer_fixture(tmp_path)
    paths = {
        "video": Path(fixture.job.target_video_path),
        "audio": Path(fixture.job.target_full_audio_path),
        "image": Path(fixture.job.reference_images[0].image_artifact_path),
    }
    paths[path_kind].write_bytes(b"changed")
    with pytest.raises(ValueError, match=message):
        materialize_mimo25_h3_shadow(
            mimo_root=fixture.mimo_root,
            source_h3_root=fixture.source_h3,
            output_root=fixture.output_root,
        )


def test_materializer_hashes_clip_media_once_across_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _materializer_fixture(tmp_path, source_sample_count=2)
    original = mimo25_materializer._sha256_file
    calls: dict[Path, int] = {}

    def counted(path: Path) -> str:
        resolved = path.resolve()
        calls[resolved] = calls.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(mimo25_materializer, "_sha256_file", counted)
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
    )
    assert summary.ready_count == 2
    for path in (
        Path(fixture.job.target_video_path),
        Path(fixture.job.target_full_audio_path),
        Path(fixture.job.reference_images[0].image_artifact_path),
    ):
        assert calls[path.resolve()] == 1


class _FakeBackend:
    def __init__(self, tmp_path: Path) -> None:
        resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
        self._provenance = MimoBackendConfig(
            media_resolver=resolver, api_key="never-persist-this"
        ).provenance()
        self.calls: list[str] = []

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self._provenance

    def reconcile(self, job: MimoClipJob, **_: object) -> MimoBackendResult:
        self.calls.append(job.clip_uid)
        return MimoBackendResult(
            annotation=_annotation(),
            raw_responses=(_annotation().model_dump_json(),),
            diagnostics=(
                MimoCompletionDiagnostic(
                    input_modality="target_video_with_embedded_audio",
                    finish_reason="stop",
                    usage=MimoUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        image_tokens=2,
                        video_tokens=4,
                        audio_tokens=3,
                        cached_tokens=1,
                    ),
                    http_attempt_count=1,
                ),
            ),
            model_call_count=1,
            http_attempt_count=1,
            http_retry_count=0,
            recheck_count=0,
            input_modality="target_video_with_embedded_audio",
        )


class _FailingBackend(_FakeBackend):
    def reconcile(self, job: MimoClipJob, **_: object) -> MimoBackendResult:
        self.calls.append(job.clip_uid)
        raise MimoBackendFailure(
            code="mimo_structured_output_failed",
            reason="invalid twice",
            raw_responses=("not json", "still not json"),
            diagnostics=(
                MimoCompletionDiagnostic(
                    input_modality="target_video_with_embedded_audio",
                    finish_reason="stop",
                    usage=MimoUsage(reasoning_tokens=3),
                    http_attempt_count=1,
                    warnings=[
                        "reasoning_tokens_nonzero_under_disabled_thinking"
                    ],
                ),
                MimoCompletionDiagnostic(
                    input_modality="full_av_recheck_embedded_audio",
                    finish_reason="stop",
                    usage=MimoUsage(reasoning_tokens=0),
                    http_attempt_count=1,
                ),
            ),
            issues=(ValidationIssue("invalid_json", None, "bad JSON"),),
            model_call_count=2,
            http_attempt_count=2,
            recheck_count=1,
        )


def test_shadow_runner_is_atomic_and_does_not_modify_inputs(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    inventory_values = {
        "schema_version": MIMO25_INVENTORY_VERSION,
        "inventory_scope": "current_diarization_asr_target_inventory",
        "canonical_wide_coverage": False,
        "source_visual_inventory_sha256": "1" * 64,
        "source_canonical_audio_manifest_sha256": "2" * 64,
        "source_diarization_raw_segments_sha256": "3" * 64,
        "source_diarization_bound_segments_sha256": "4" * 64,
        "source_qwen3_asr_segments_sha256": "5" * 64,
        "source_binding_audit_segments_sha256": "6" * 64,
        "source_h3_samples_sha256": "7" * 64,
        "clip_count": 1,
        "jobs": [job.model_dump(mode="json")],
    }
    inventory = _inventory(inventory_values)
    before = {
        path: path.read_bytes()
        for path in (
            Path(job.target_video_path),
            Path(job.target_full_audio_path),
            Path(job.reference_images[0].image_artifact_path),
        )
    }
    backend = _FakeBackend(tmp_path)
    output = tmp_path / "mimo-output"
    summary = run_mimo25_av_reconcile(
        inventory=inventory,
        backend=backend,
        output_root=output,
    )
    assert backend.calls == ["clip-1"]
    assert summary.ready_count == 1
    assert summary.usage_totals["image_tokens"] == 2
    assert summary.production_binding_modified is False
    assert summary.production_diarization_modified is False
    assert summary.production_asr_modified is False
    assert summary.production_h3_modified is False
    assert all(path.read_bytes() == value for path, value in before.items())
    published = "".join(path.read_text() for path in output.rglob("*.json*"))
    assert "never-persist-this" not in published
    assert "data:video" not in published


def test_failed_issue_with_null_field_persists_and_reasoning_is_summarized(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)
    inventory_values = {
        "schema_version": MIMO25_INVENTORY_VERSION,
        "inventory_scope": "current_diarization_asr_target_inventory",
        "canonical_wide_coverage": False,
        "source_visual_inventory_sha256": "1" * 64,
        "source_canonical_audio_manifest_sha256": "2" * 64,
        "source_diarization_raw_segments_sha256": "3" * 64,
        "source_diarization_bound_segments_sha256": "4" * 64,
        "source_qwen3_asr_segments_sha256": "5" * 64,
        "source_binding_audit_segments_sha256": "6" * 64,
        "source_h3_samples_sha256": "7" * 64,
        "clip_count": 1,
        "jobs": [job.model_dump(mode="json")],
    }
    output = tmp_path / "failed-output"
    summary = run_mimo25_av_reconcile(
        inventory=_inventory(inventory_values),
        backend=_FailingBackend(tmp_path),
        output_root=output,
    )
    failure = MimoFailure.model_validate_json(
        (output / "failures.jsonl").read_text(encoding="utf-8")
    )
    assert failure.schema_version == MIMO25_FAILURE_VERSION
    assert failure.issues == [
        {"code": "invalid_json", "field": None, "message": "bad JSON"}
    ]
    assert summary.model_call_count == 2
    assert summary.http_attempt_count == 2
    assert summary.responses_with_nonzero_reasoning_tokens == 1


def test_review_persistence_and_stale_fingerprint(tmp_path: Path) -> None:
    case = MimoReviewCase("clip-1", "a" * 64, {"clip_uid": "clip-1"})
    store = MimoReviewStore(tmp_path / "human_review", [case])
    summary = store.save(
        MimoHumanReviewAnnotation(
            clip_uid="clip-1",
            record_fingerprint="a" * 64,
            decision="PASS",
            issue_tags=[],
            notes="good",
            reviewed_at="2026-09-01T00:00:00Z",
        )
    )
    assert summary.reviewed_count == 1
    assert "clip-1" in (tmp_path / "human_review" / "annotations.csv").read_text()
    stale_store = MimoReviewStore(
        tmp_path / "human_review",
        [MimoReviewCase("clip-1", "b" * 64, {"clip_uid": "clip-1"})],
    )
    stale = stale_store.publish_derived()
    assert stale.reviewed_count == 0
    assert stale.stale_annotation_count == 1
    assert "clip-1" not in (tmp_path / "human_review" / "annotations.csv").read_text()


def test_review_html_contains_required_panels() -> None:
    payload = {
        "clip_uid": "clip-1",
        "target_video_url": "/media/token",
        "references": [],
        "source_segments": [],
        "mimo_record": {"record_fingerprint": "a" * 64, "annotation": None},
        "shadow_variants": [],
        "legacy_qwen38": {},
    }
    page = render_review_html([MimoReviewCase("clip-1", "a" * 64, payload)], {})
    assert "speaker_grouping_issue" in page
    assert "MiMo H3 shadow" in page
    assert "Legacy Qwen3.8" in page
    assert "Cache-Control" not in page


def test_review_server_allowlist_traversal_and_no_store(tmp_path: Path) -> None:
    media_file = tmp_path / "target.mp4"
    media_file.write_bytes(b"0123456789")
    payload = {
        "clip_uid": "clip-1",
        "target_video_url": "/media/abc",
        "references": [],
        "source_segments": [],
        "mimo_record": {"record_fingerprint": "a" * 64, "annotation": None},
        "shadow_variants": [],
        "legacy_qwen38": {},
    }
    cases = [MimoReviewCase("clip-1", "a" * 64, payload)]
    store = MimoReviewStore(tmp_path / "review", cases)
    server = make_review_server(
        host="127.0.0.1",
        port=0,
        cases=cases,
        media={"abcdef012345abcdef012345": media_file},
        store=store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(
            base + "/media/abcdef012345abcdef012345"
        ) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == b"0123456789"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(base + "/media/../../etc/passwd")
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
