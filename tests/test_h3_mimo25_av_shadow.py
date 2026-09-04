from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args

import numpy as np
import pytest
from pydantic import ValidationError

import r2v_data_v2.h3.mimo25_av_reconcile as mimo25_reconcile
import r2v_data_v2.h3.mimo25_h3_materializer as mimo25_materializer
from r2v_data_v2.h3.audio_backends import AudioFileProbe
from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    DiarizationTargetClip,
)
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    jea_production_paths,
)
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
    MIMO25_REFERENCE_SELECTION_POLICY_VERSION,
    MimoCaseManifest,
    MimoClipJob,
    MimoFailure,
    MimoRawResponse,
    MimoRecord,
    MimoReferenceImage,
    MimoReferenceSelection,
    MimoSegmentEvidence,
    _inventory,
    _job,
    _validate_clip_segment_inventory,
    _validate_h3_variant_observations,
    build_mimo25_inventory,
    project_mimo_h3_sample_references,
    run_mimo25_av_reconcile,
    select_mimo_reference_projection,
)
from r2v_data_v2.h3.mimo25_backend import (
    _CONSERVATIVE_VISIBLE_SPEAKER_ISSUES,
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
    SpeechPresentation,
    _canonicalize_same_visible_entity_speaker_groups,
    _normalize_speaker_voice_profiles,
    validate_annotation,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    _materialize_sample,
    _render_subject_definition,
    materialize_mimo25_h3_shadow,
)
from r2v_data_v2.h3.mimo25_human_review import (
    MimoHumanReviewAnnotation,
    MimoReviewCase,
    MimoReviewStore,
    _review_case_fingerprint,
    build_review_cases,
    make_review_server,
    render_review_html,
)
from r2v_data_v2.h3.mimo25_recovered_voice import (
    CanonicalAudioAnalysis,
    MimoRecoveredVoiceQualityPolicy,
    recover_mimo_target_voices,
)
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract
from r2v_data_v2.structured_output import ValidationIssue
from tools.materialize_h3_mimo25_shadow import _parser as _materializer_parser


def _subject_definition(subject_label: str, description: str) -> dict[str, str]:
    return {"subject_label": subject_label, "description": description}


def _retention(
    subject_label: str,
    marker: str,
    description: str,
) -> dict[str, str]:
    return {
        "subject_label": subject_label,
        "marker": marker,
        "description": description,
    }


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
                    "speech_presentation": (
                        "onscreen_spoken"
                        if entity_id is not None
                        else "offscreen_spoken"
                    ),
                    "entity_id": entity_id,
                    "delivery_style": "calm and clear",
                    "secondary_vocal_activity": {
                        "present": composition != "single_speaker",
                        "speaker_relation": (
                            "none"
                            if composition == "single_speaker"
                            else "same_speaker"
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
                    "evidence_codes": (
                        ["visible_lip_motion", "av_temporal_alignment"]
                        if entity_id is not None
                        else ["offscreen_audio"]
                    ),
                }
            ],
            "speaker_voice_profiles": (
                [
                    {
                        "speaker_group": group,
                        "voice_characteristics": (
                            "clear mid-register timbre with measured cadence"
                        ),
                    }
                ]
                if resolution == "resolved"
                else []
            ),
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
                "overall_soundscape_status": "present",
                "overall_soundscape": "Quiet speech with a short clink.",
                "non_diegetic_music_status": "absent",
                "non_diegetic_music": None,
                "audiovisual_summary": "One visible speaker talks in a quiet scene.",
            },
            "h3_draft": {
                "subject_definitions": [
                    _subject_definition(
                        "<Subject 1>",
                        "a person with a clearly visible appearance.",
                    )
                ],
                "summary": "A person speaks while remaining visible.",
                "visual_retention_analysis": [
                    _retention(
                        "<Subject 1>",
                        "fully_preserved",
                        "the person remains visible.",
                    )
                ],
                "shots": [
                    {
                        "shot_index": 1,
                        "start_time": None,
                        "timeline_parts": [
                            {
                                "type": "prose",
                                "text": "<Subject 1> faces the camera.",
                            },
                            {"type": "audio_event", "event_id": "ae1"},
                            {"type": "speech", "segment_id": "segment_1"},
                        ],
                    }
                ],
            },
            "warnings": [],
        }
    )


def _prose(text: str) -> dict[str, str]:
    return {"type": "prose", "text": text}


def _speech(segment_id: str) -> dict[str, str]:
    return {"type": "speech", "segment_id": segment_id}


def _audio_event(event_id: str) -> dict[str, str]:
    return {"type": "audio_event", "event_id": event_id}


def _segment_decision_branch_schemas() -> dict[str, dict[str, object]]:
    schema = MimoAVAnnotationDraft.model_json_schema()
    item_schema = schema["properties"]["segment_decisions"]["items"]
    return {
        resolution: schema["$defs"][reference.rsplit("/", 1)[-1]]
        for resolution, reference in item_schema["discriminator"]["mapping"].items()
    }


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
        source_image_index=1,
        source_image_id="image_1",
        source_image_label="<Image 1>",
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
        source_end_sample=32000,
        source_sample_rate_hz=32000,
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
        "reference_selection": MimoReferenceSelection(
            original_picture_count=1,
            selected_picture_count=1,
            selected_source_image_indexes=[1],
            selected_source_image_ids=["image_1"],
            dropped_references=[],
        ).model_dump(mode="json"),
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


def _multi_picture_job_fixture(tmp_path: Path) -> MimoClipJob:
    job = _job_fixture(tmp_path)
    second_image = tmp_path / "reference-2.png"
    second_image.write_bytes(b"image-2")
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["reference_images"].append(
        MimoReferenceImage(
            image_index=2,
            picture_label="<Picture 2>",
            source_image_index=2,
            source_image_id="image_2",
            source_image_label="<Image 2>",
            kind="subject",
            entity_id="e1",
            image_artifact_path=str(second_image.resolve()),
            image_sha256="4" * 64,
        ).model_dump(mode="json")
    )
    values["reference_selection"].update(
        original_picture_count=2,
        selected_picture_count=2,
        selected_source_image_indexes=[1, 2],
        selected_source_image_ids=["image_1", "image_2"],
    )
    values["reference_subjects"][0]["source_picture_labels"] = [
        "<Picture 1>",
        "<Picture 2>",
    ]
    return _job(values)


def _job_with_non_transcribed_segment(tmp_path: Path) -> MimoClipJob:
    job = _job_fixture(tmp_path)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["target_duration_seconds"] = 2.0
    values["segments"].append(
        MimoSegmentEvidence(
            segment_id="segment_2",
            start_time=1.0,
            end_time=2.0,
            source_start_sample=32000,
            source_end_sample=64000,
            source_sample_rate_hz=32000,
            source_speaker_cluster_id="speaker_1",
            identity_scope="unresolved",
            direct_anchor_seconds=0.0,
            cluster_binding_status="unbound",
            overlapping_visible_entities=[],
            direct_support_seconds_by_entity={},
            competing_visible_speaker_evidence=[],
            asr_status="empty",
        ).model_dump(mode="json")
    )
    return _job(values)


def _three_transcribed_segment_job(tmp_path: Path) -> MimoClipJob:
    job = _job_fixture(tmp_path)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["target_duration_seconds"] = 3.0
    first = values["segments"][0]
    for index in (2, 3):
        segment = dict(first)
        segment["segment_id"] = f"segment_{index}"
        segment["start_time"] = float(index - 1)
        segment["end_time"] = float(index)
        segment["source_start_sample"] = (index - 1) * 32000
        segment["source_end_sample"] = index * 32000
        segment["source_speaker_cluster_id"] = f"speaker_{index - 1}"
        segment["asr_text"] = f"Exact text {index}."
        values["segments"].append(segment)
    return _job(values)


def _three_segment_annotation(
    assignments: list[tuple[str, str]],
    *,
    profiles: list[dict[str, str | None]] | None = None,
    reliable: bool = True,
) -> MimoAVAnnotationDraft:
    payload = _annotation().model_dump(mode="json")
    first = payload["segment_decisions"][0]
    payload["segment_decisions"] = []
    for index, (group, entity_id) in enumerate(assignments, start=1):
        decision = dict(first)
        decision["segment_id"] = f"segment_{index}"
        decision["primary_speaker_group"] = group
        decision["entity_id"] = entity_id
        decision["delivery_style"] = f"measured delivery {index}"
        decision["evidence_codes"] = (
            ["visible_lip_motion", "av_temporal_alignment"]
            if reliable
            else ["voice_continuity"]
        )
        payload["segment_decisions"].append(decision)
    if profiles is None:
        profiles = [
            {"speaker_group": group, "voice_characteristics": f"profile {group}"}
            for group in dict.fromkeys(group for group, _ in assignments)
        ]
    payload["speaker_voice_profiles"] = profiles
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        _prose("A visible conversation unfolds."),
        _audio_event("ae1"),
        *(_speech(f"segment_{index}") for index in range(1, 4)),
    ]
    return MimoAVAnnotationDraft.model_validate(payload)


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
    *,
    transport: Literal["xiaomi", "sglang"] = "xiaomi",
) -> tuple[OpenAIMimo25Backend, _Completions]:
    completions = _Completions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=resolver,
            api_key="secret",
            transport=transport,
        ),
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
    assert "reasoning_effort" not in request
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
    assert backend.provenance.transport == "xiaomi"
    assert backend.provenance.backend == "xiaomi_openai_compatible"
    assert job.r2v_instruction in content[-1]["text"]  # type: ignore[index]


def test_first_shot_zero_does_not_trigger_full_av_recheck(tmp_path: Path) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["start_time"] = 0
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    backend, completions = _backend(tmp_path, [(annotation.model_dump_json(), 8)])

    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.annotation.h3_draft.shots[0].start_time == 0
    assert result.model_call_count == 1
    assert result.recheck_count == 0
    assert len(completions.requests) == 1


def test_sglang_primary_uses_embedded_video_audio_and_non_thinking_contract(
    tmp_path: Path,
) -> None:
    backend, completions = _backend(
        tmp_path,
        [(_annotation().model_dump_json(), 53)],
        transport="sglang",
    )
    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.model_call_count == 1
    request = completions.requests[0]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "MimoAVAnnotationDraft"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == (
        MimoAVAnnotationDraft.model_json_schema()
    )
    decision_schemas = _segment_decision_branch_schemas()
    for decision_schema in decision_schemas.values():
        assert "entity_id" in decision_schema["required"]
        assert "delivery_style" in decision_schema["required"]
        assert (
            "direct_anchor_present"
            not in decision_schema["properties"]["evidence_codes"]["items"]["enum"]
        )
        assert (
            "visible_lip_motion"
            in decision_schema["properties"]["evidence_codes"]["items"]["enum"]
        )
    assert request["reasoning_effort"] == "none"
    assert request["extra_body"] == {
        "use_audio_in_video": True,
        "chat_template_kwargs": {
            "thinking": False,
            "enable_thinking": False,
        },
    }
    content = request["messages"][1]["content"]  # type: ignore[index]
    assert any(item["type"] == "video_url" for item in content)  # type: ignore[union-attr]
    assert not any(item["type"] in {"audio_url", "input_audio"} for item in content)  # type: ignore[union-attr]
    assert backend.provenance.transport == "sglang"
    assert backend.provenance.backend == "sglang_openai_compatible"
    assert backend.provenance.response_format == "json_schema"
    prompt = content[-1]["text"]  # type: ignore[index]
    assert '"$defs"' not in prompt
    assert "constrained by the supplied response_format" in prompt


def test_xiaomi_prompt_retains_textual_json_schema(tmp_path: Path) -> None:
    backend, _ = _backend(tmp_path, [])
    prompt = backend._prompt(_job_fixture(tmp_path))
    assert json.dumps(
        MimoAVAnnotationDraft.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) in prompt


def test_transport_changes_backend_configuration_fingerprint(tmp_path: Path) -> None:
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    xiaomi = MimoBackendConfig(
        media_resolver=resolver,
        api_key="secret",
    ).provenance()
    sglang = MimoBackendConfig(
        media_resolver=resolver,
        api_key="secret",
        transport="sglang",
    ).provenance()

    assert xiaomi.transport == "xiaomi"
    assert sglang.transport == "sglang"
    assert xiaomi.configuration_fingerprint != sglang.configuration_fingerprint


def test_current_backend_schema_keeps_existing_materializer_v6_provenance_readable(
    tmp_path: Path,
) -> None:
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    current = MimoBackendConfig(
        media_resolver=resolver,
        api_key="secret",
    ).provenance()
    values = current.model_dump(mode="json", exclude={"configuration_fingerprint"})
    values["materializer_version"] = "h3_mimo25_materializer_v6"
    values["configuration_fingerprint"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in values.items() if key != "configuration_fingerprint"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    historical = type(current).model_validate(values)
    assert historical.materializer_version == "h3_mimo25_materializer_v6"
    assert current.materializer_version == "h3_mimo25_materializer_v13"


def test_mimo_v19_prompt_preserves_dense_visual_and_audio_authority_contract() -> None:
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_unified_av_reconcile_v19"
    assert MIMO25_POLICY_VERSION == "h3_mimo25_av_authority_contract_v13"
    assert MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.12"
    for phrase in (
        "shot scale and framing",
        "foreground, midground, and background composition",
        "body, arm, hand, and head motion",
        "temporal progression through early, middle, and late portions",
        "300-450 English words",
        "attribute_transfer is forbidden",
        "Pictures are content references, not first frames, last frames, or keyframes",
        "Never transcribe, quote, correct, paraphrase",
        "Voice continuity may preserve speaker-group identity",
        "cannot alone bind a visible entity",
        "Use absent only for verified complete silence of the soundscape",
        "must not summarize or repeat spoken dialogue or speech",
        "narration or voice-over content",
        "singing, diegetic music, or non-diegetic music/BGM/score",
        "diegetic music belongs in detailed_description",
        "non-diegetic music belongs in non_diegetic_music",
        "never create room tone or another sound from visual context",
        "Subject definitions use typed subject_label",
        "Keep description visual-only",
        "Visual retention uses typed subject_label, marker, and description",
        "Use acoustic-first wording",
        "Supported audible descriptors such as male, female, youthful, or mature",
        "Frozen Subject-to-Picture provenance is pipeline-owned",
        "do not repeat any Subject or Picture label",
        "Decide clip-local speaker identity independently",
        "articulation in an adjacent segment does not count",
        "a visible listener must never inherit the audible speaker identity",
        "must never carry entity_id into a later segment",
        "self-audit every visible_entity decision",
        "Perform a full-timeline pass",
        "door opening, closing, and latch sounds",
        "must not be omitted merely because it is brief",
        "Visible motion alone never creates sound",
        "localized physical actions or object interactions as physical",
        "traffic ambience, crowd ambience, and outdoor layers as environmental",
        "machinery or device mechanisms operating as an audible source",
        "beeps, alarms, tones, and device signals",
        "laughter, coughing, sighs, breaths, gasps, crying",
        "pattern=single means one localized transient occurrence",
        "pattern=repeated means several distinct repetitions",
        "pattern=continuous means a sustained audible layer or operation",
        "not continuous merely because its approximate event window spans multiple video frames",
        "1-4 natural English sentences",
        "ordinary observed target-video sounds never create <Audio N>",
        "reference_selection maps each surviving source Image",
        "Create no Picture or Subject for a dropped source Image",
        "never reinterpret one surviving source Image as another",
    ):
        assert phrase in SYSTEM_PROMPT
    assert "natural official MiniMax H3 Ref2VA visual description" in (
        SYSTEM_PROMPT
    )


def test_primary_prompt_includes_exact_subject_picture_contract(
    tmp_path: Path,
) -> None:
    job = _multi_picture_job_fixture(tmp_path)
    backend, _ = _backend(tmp_path, [])

    prompt = backend._prompt(job)
    contract = backend.build_mandatory_h3_draft_contract(job)

    assert contract["subject_definition_requirements"] == [
        {
            "subject_label": "<Subject 1>",
            "required_source_picture_labels": ["<Picture 1>", "<Picture 2>"],
        }
    ]
    assert contract["reference_selection"]["selected_source_image_indexes"] == [
        1,
        2,
    ]
    assert [
        (item["source_image_label"], item["picture_label"])
        for item in contract["reference_image_mapping"]
    ] == [("<Image 1>", "<Picture 1>"), ("<Image 2>", "<Picture 2>")]
    assert (
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        in prompt
    )
    assert "pipeline owns and materializes exact Subject-to-Picture provenance" in prompt
    assert "do not put Picture labels in description" in prompt


def test_primary_prompt_separates_decision_and_typed_speech_inventories(
    tmp_path: Path,
) -> None:
    job = _job_with_non_transcribed_segment(tmp_path)
    backend, _ = _backend(tmp_path, [])

    prompt = backend._prompt(job)
    contract = backend.build_mandatory_h3_draft_contract(job)

    assert contract["allowed_segment_ids"] == ["segment_1", "segment_2"]
    assert contract["transcribed_segment_ids"] == ["segment_1"]
    assert "required_speech_segment_sequence" not in contract
    assert "forbidden_speech_segment_ids" not in contract
    assert "Use allowed_segment_ids for all decisions" in prompt
    assert "transcribed_segment_ids" in prompt
    assert "timeline_parts" in prompt
    assert "[[segment:" not in prompt
    assert "[[audio_event:" not in prompt


def test_prompt_defines_primary_speaker_group_as_identity() -> None:
    for phrase in (
        "identify speakers by first appearance, not turns",
        "must not by itself create a new group",
        "same resolved visible entity reuses one group",
        "split or merge only with AV support",
    ):
        assert phrase in SYSTEM_PROMPT


def test_mimo_prompt_separates_voice_identity_from_visible_speech() -> None:
    for phrase in (
        "speaker_visible_mouth_occluded",
        "LR-ASD/direct anchors are neither sufficient nor required",
        "no binding",
        "Every segment receives one decision",
        "Silent phone reading/typing",
        "message_voice_over",
        "Never delete a segment for non-onscreen presentation",
        "Non-onscreen speech must not create visible speaking/lip motion",
    ):
        assert phrase in SYSTEM_PROMPT


def test_segment_decision_requires_speech_presentation() -> None:
    assert get_args(SpeechPresentation) == (
        "onscreen_spoken",
        "offscreen_spoken",
        "voice_over",
        "message_voice_over",
        "device_playback",
        "uncertain",
    )
    payload = _annotation().model_dump(mode="json")
    del payload["segment_decisions"][0]["speech_presentation"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)


def test_segment_decision_entity_id_is_required_but_nullable() -> None:
    assert all(
        "entity_id" in schema["required"]
        for schema in _segment_decision_branch_schemas().values()
    )

    payload = _annotation().model_dump(mode="json")
    del payload["segment_decisions"][0]["entity_id"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)

    assert _annotation(entity_id="e1").segment_decisions[0].entity_id == "e1"
    assert _annotation(entity_id=None).segment_decisions[0].entity_id is None

    invalid = _annotation(entity_id=None).model_dump(mode="json")
    invalid["segment_decisions"][0]["entity_id"] = "e1"
    with pytest.raises(ValidationError, match="only visible_entity"):
        MimoAVAnnotationDraft.model_validate(invalid)


def test_speaker_voice_profiles_are_exact_ordered_and_nullable() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = None
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert not _validate(annotation)

    for profiles in (
        [],
        payload["speaker_voice_profiles"] * 2,
        [{"speaker_group": "g9", "voice_characteristics": None}],
    ):
        changed_payload = annotation.model_dump(mode="json")
        changed_payload["speaker_voice_profiles"] = profiles
        changed = MimoAVAnnotationDraft.model_validate(changed_payload)
        assert "speaker_voice_profile_inventory_mismatch" in {
            issue.code for issue in _validate(changed)
        }


def test_speaker_voice_profile_rejects_significant_transcript_copy() -> None:
    transcript = "This authoritative transcript must never be copied verbatim."
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = transcript
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        authoritative_transcripts=[transcript],
    )
    assert "audio_semantics_contains_authoritative_transcript" in {
        issue.code for issue in issues
    }


@pytest.mark.parametrize(
    "description",
    [
        "A mid-range male voice with a slightly raspy timbre.",
        "A mature male voice with a slightly raspy mid-low timbre.",
    ],
)
def test_speaker_voice_profile_accepts_supported_audible_demographics(
    description: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = description

    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert "speaker_voice_profile_contains_identity_claim" not in {
        issue.code for issue in issues
    }


@pytest.mark.parametrize(
    "description",
    [
        "A doctor speaking with a measured low cadence.",
        "The Chinese man has a clear mid-register voice.",
        "The voice of Alice is calm and lightly raspy.",
    ],
)
def test_speaker_voice_profile_rejects_role_nationality_or_named_identity(
    description: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = description

    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert "speaker_voice_profile_contains_identity_claim" in {
        issue.code for issue in issues
    }


def test_speaker_voice_profile_accepts_acoustic_characteristics() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = (
        "A slightly raspy mid-to-low register voice with measured cadence."
    )

    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert "speaker_voice_profile_contains_identity_claim" not in {
        issue.code for issue in issues
    }


def test_music_event_and_global_status_consistency() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"][0]["category"] = (
        "non_diegetic_music"
    )
    with pytest.raises(ValidationError, match="present global music"):
        MimoAVAnnotationDraft.model_validate(payload)

    payload["audio_semantics"].update(
        non_diegetic_music_status="present",
        non_diegetic_music="A soft acoustic guitar score plays.",
    )
    assert MimoAVAnnotationDraft.model_validate(payload)

    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"][0]["category"] = (
        "diegetic_music"
    )
    assert MimoAVAnnotationDraft.model_validate(payload)

    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"].update(
        non_diegetic_music_status="present",
        non_diegetic_music="A continuous orchestral score is audible.",
    )
    assert MimoAVAnnotationDraft.model_validate(payload)


def test_segment_decision_delivery_is_required_but_nullable() -> None:
    assert all(
        "delivery_style" in schema["required"]
        for schema in _segment_decision_branch_schemas().values()
    )

    payload = _annotation().model_dump(mode="json")
    del payload["segment_decisions"][0]["delivery_style"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)

    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["delivery_style"] = " "
    with pytest.raises(ValidationError, match="delivery must be non-empty"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_resolution_discriminator_structures_primary_speaker_group() -> None:
    schema = MimoAVAnnotationDraft.model_json_schema()
    item_schema = schema["properties"]["segment_decisions"]["items"]
    assert item_schema["discriminator"]["propertyName"] == "resolution"
    assert set(item_schema["discriminator"]["mapping"]) == {
        "resolved",
        "needs_acoustic_refinement",
        "uncertain",
    }

    branches = _segment_decision_branch_schemas()
    resolved = branches["resolved"]
    refinement = branches["needs_acoustic_refinement"]
    assert "primary_speaker_group" in resolved["required"]
    assert resolved["properties"]["primary_speaker_group"] == {
        "pattern": r"^g[1-9]\d*$",
        "title": "Primary Speaker Group",
        "type": "string",
    }
    assert "primary_speaker_group" in refinement["required"]
    assert {
        item["type"]
        for item in refinement["properties"]["primary_speaker_group"]["anyOf"]
    } == {
        "string",
        "null",
    }

    resolved_payload = _annotation().model_dump(mode="json")
    del resolved_payload["segment_decisions"][0]["primary_speaker_group"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(resolved_payload)

    resolved_payload = _annotation().model_dump(mode="json")
    resolved_payload["segment_decisions"][0]["primary_speaker_group"] = None
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(resolved_payload)
    assert _annotation().segment_decisions[0].primary_speaker_group == "g1"

    refinement_payload = _annotation(resolution="needs_acoustic_refinement").model_dump(
        mode="json"
    )
    refinement_payload["segment_decisions"][0]["primary_speaker_group"] = None
    assert (
        MimoAVAnnotationDraft.model_validate(refinement_payload)
        .segment_decisions[0]
        .primary_speaker_group
        is None
    )
    del refinement_payload["segment_decisions"][0]["primary_speaker_group"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(refinement_payload)


def test_evidence_codes_are_bounded_in_strict_json_schema() -> None:
    branches = _segment_decision_branch_schemas()
    for branch in branches.values():
        evidence = branch["properties"]["evidence_codes"]
        assert evidence["minItems"] == 1
        assert evidence["maxItems"] == 8

    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = [
        "visible_lip_motion",
        "speaker_visible_mouth_occluded",
        "av_temporal_alignment",
        "voice_continuity",
        "speaker_turn_change",
        "offscreen_audio",
        "lr_asd_support",
        "source_cluster_support",
        "message_text_alignment",
    ]
    with pytest.raises(ValidationError, match="at most 8 items"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_secondary_vocal_activity_is_schema_discriminated_by_presence() -> None:
    schema = MimoAVAnnotationDraft.model_json_schema()
    decision = _segment_decision_branch_schemas()["resolved"]
    secondary = decision["properties"]["secondary_vocal_activity"]
    assert secondary["discriminator"]["propertyName"] == "present"

    absent = schema["$defs"]["MimoAbsentSecondaryVocalActivity"]
    assert set(absent["required"]) == {"present", "speaker_relation", "kind"}
    assert absent["properties"]["present"]["const"] is False
    assert absent["properties"]["speaker_relation"]["const"] == "none"
    assert absent["properties"]["kind"]["type"] == "null"

    present = schema["$defs"]["MimoPresentSecondaryVocalActivity"]
    assert present["properties"]["present"]["const"] is True
    assert "none" not in present["properties"]["speaker_relation"]["enum"]

    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["secondary_vocal_activity"]["kind"] = (
        "non_lyrical_singing"
    )
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)


def test_visible_entity_structural_relationships_remain_model_validated() -> None:
    payload = _annotation().model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision["speech_presentation"] = "message_voice_over"
    with pytest.raises(ValidationError, match="onscreen_spoken"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize(
    "evidence_codes",
    [
        ["av_temporal_alignment", "voice_continuity"],
        [
            "av_temporal_alignment",
            "voice_continuity",
            "source_cluster_support",
        ],
    ],
)
def test_indirect_continuity_parses_then_fails_visible_speaker_validation(
    evidence_codes: list[str],
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = evidence_codes

    annotation = MimoAVAnnotationDraft.model_validate(payload)
    issues = _validate(annotation)

    assert annotation.segment_decisions[0].entity_id == "e1"
    assert {item.code for item in issues} == {
        "visible_entity_requires_confirmed_onscreen_speech",
        "onscreen_speech_requires_reliable_visible_speaker_evidence",
    }
    assert {item.field for item in issues} == {"segment_1"}


@pytest.mark.parametrize(
    "evidence_codes",
    [
        ["av_temporal_alignment"],
        ["voice_continuity"],
        ["source_cluster_support"],
    ],
)
def test_single_indirect_signal_has_segment_level_semantic_issues(
    evidence_codes: list[str],
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = evidence_codes
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    expected = {
        "visible_entity_requires_confirmed_onscreen_speech",
        "onscreen_speech_requires_reliable_visible_speaker_evidence",
    }
    assert {item.code for item in issues} == expected
    assert {item.field for item in issues} == {"segment_1"}


@pytest.mark.parametrize(
    "conflict_code", ["lr_asd_conflict", "source_cluster_conflict"]
)
def test_explicit_conflict_does_not_make_indirect_continuity_visible_evidence(
    conflict_code: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = [
        "av_temporal_alignment",
        "voice_continuity",
        conflict_code,
    ]
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert {item.code for item in issues} == {
        "visible_entity_requires_confirmed_onscreen_speech",
        "onscreen_speech_requires_reliable_visible_speaker_evidence",
    }


def test_visible_lip_motion_remains_valid_onscreen_evidence() -> None:
    annotation = _annotation()
    assert annotation.segment_decisions[0].evidence_codes == [
        "visible_lip_motion",
        "av_temporal_alignment",
    ]
    assert not _validate(annotation)


@pytest.mark.parametrize(
    "evidence_codes",
    [
        ["speaker_visible_mouth_occluded", "av_temporal_alignment"],
        ["speaker_visible_mouth_occluded", "voice_continuity"],
    ],
)
def test_visible_entity_allows_mouth_occluded_continuity_evidence(
    evidence_codes: list[str],
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = evidence_codes
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert annotation.segment_decisions[0].entity_id == "e1"
    assert not _validate(annotation)
    assert "lr_asd_support" not in evidence_codes


def test_mouth_occluded_path_allows_explicit_no_visible_lip_motion() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = [
        "speaker_visible_mouth_occluded",
        "voice_continuity",
        "no_visible_lip_motion",
    ]

    annotation = MimoAVAnnotationDraft.model_validate(payload)

    assert not _validate(annotation)


@pytest.mark.parametrize(
    "contradiction",
    ["offscreen_audio", "voice_over_context", "device_playback_context"],
)
def test_visible_speaker_rejects_explicit_presentation_contradiction(
    contradiction: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"].append(contradiction)

    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert "visible_speaker_evidence_presentation_contradiction" in {
        item.code for item in issues
    }


def test_mouth_occlusion_without_continuity_is_not_onscreen_evidence() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["evidence_codes"] = [
        "speaker_visible_mouth_occluded"
    ]
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert {item.code for item in _validate(annotation)} == {
        "visible_entity_requires_confirmed_onscreen_speech",
        "onscreen_speech_requires_reliable_visible_speaker_evidence",
    }


def test_onscreen_speech_without_known_entity_is_valid() -> None:
    payload = _annotation(entity_id=None).model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision.update(
        binding_status="no_reliable_entity",
        speech_presentation="onscreen_spoken",
        evidence_codes=["visible_lip_motion"],
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert annotation.segment_decisions[0].entity_id is None

    decision["binding_status"] = "offscreen"
    with pytest.raises(ValidationError, match="cannot use offscreen"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_unbound_zero_anchor_segment_can_be_recovered_by_mimo(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    source = values["segments"][0]
    source.update(
        current_entity_id=None,
        entity_occurrence_id=None,
        identity_scope="unresolved",
        direct_anchor_seconds=0.0,
        cluster_binding_status="unbound",
        direct_support_seconds_by_entity={},
    )
    unbound_job = _job(values)
    response = _annotation().model_dump(mode="json")
    response["segment_decisions"][0]["evidence_codes"] = [
        "speaker_visible_mouth_occluded",
        "voice_continuity",
    ]
    backend, completions = _backend(
        tmp_path,
        [(json.dumps(response), 8)],
    )

    result = backend.reconcile(
        unbound_job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.annotation.segment_decisions[0].entity_id == "e1"
    assert result.model_call_count == 1
    prompt = completions.requests[0]["messages"][1]["content"][-1]["text"]
    assert '"cluster_binding_status":"unbound"' in prompt
    assert '"direct_anchor_seconds":0.0' in prompt
    assert '"allowed_segment_ids":["segment_1"]' in prompt


def test_non_onscreen_presentations_cannot_claim_visible_entity() -> None:
    for presentation in (
        "offscreen_spoken",
        "voice_over",
        "message_voice_over",
        "device_playback",
        "uncertain",
    ):
        payload = _annotation().model_dump(mode="json")
        payload["segment_decisions"][0]["speech_presentation"] = presentation
        with pytest.raises(ValidationError):
            MimoAVAnnotationDraft.model_validate(payload)


def test_message_voice_over_preserves_resolved_speech_without_entity() -> None:
    payload = _annotation(entity_id=None).model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision.update(
        binding_status="offscreen",
        speech_presentation="message_voice_over",
        evidence_codes=["no_visible_lip_motion", "message_text_alignment"],
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert annotation.segment_decisions[0].resolution == "resolved"
    assert not _validate(annotation)


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
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_sglang_audio_token_zero_uses_audio_url_fallback(tmp_path: Path) -> None:
    raw = _annotation().model_dump_json()
    backend, completions = _backend(
        tmp_path,
        [(raw, 0), (raw, 4)],
        transport="sglang",
    )

    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.model_call_count == 2
    assert result.input_modality == "target_video_plus_canonical_full_audio_fallback"
    fallback = completions.requests[1]
    content = fallback["messages"][1]["content"]  # type: ignore[index]
    assert any(item["type"] == "video_url" for item in content)  # type: ignore[union-attr]
    audio_items = [item for item in content if item["type"] == "audio_url"]  # type: ignore[union-attr]
    assert len(audio_items) == 1
    assert audio_items[0]["audio_url"]["url"].startswith("data:audio/")
    assert not any(item["type"] == "input_audio" for item in content)  # type: ignore[union-attr]
    assert fallback["reasoning_effort"] == "none"
    assert fallback["extra_body"]["use_audio_in_video"] is True


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
        client=SimpleNamespace(chat=SimpleNamespace(completions=AlwaysFails())),
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
    assert "Reinspect the same full audiovisual evidence" in content[-1]["text"]
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    recheck = content[-1]["text"]
    assert job.r2v_instruction in recheck
    assert job.target_video_path not in recheck


def test_full_av_recheck_repeats_exact_draft_contract(tmp_path: Path) -> None:
    job = _multi_picture_job_fixture(tmp_path)
    backend, _ = _backend(tmp_path, [])
    issues = [
        ValidationIssue(
            "subject_definition_contract_mismatch",
            "h3_draft.subject_definitions",
            "definition differs",
        ),
        ValidationIssue(
            "speech_placeholder_inventory_mismatch",
            "h3_draft.shots",
            "placeholder inventory differs",
        ),
    ]

    prompt = backend._full_av_recheck_prompt(
        job,
        invalid_response="{}",
        issues=issues,
    )
    contract = backend.build_mandatory_h3_draft_contract(job)

    assert (
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        in prompt
    )
    assert "Repair typed Subject definition or retention rows" in prompt
    assert "pipeline materializes frozen Subject-to-Picture ownership" in prompt
    assert "rebuild all typed speech timeline parts" in prompt
    assert "transcribed_segment_ids" in prompt
    assert "exactly equals" in prompt


def test_full_av_recheck_targets_visual_audio_and_voice_identity_leakage(
    tmp_path: Path,
) -> None:
    backend, _ = _backend(tmp_path, [])
    prompt = backend._full_av_recheck_prompt(
        _job_fixture(tmp_path),
        invalid_response="{}",
        issues=[
            ValidationIssue(
                "subject_definition_contains_audio_profile",
                "h3_draft.subject_definitions",
                "visual definition contains timbre",
            ),
            ValidationIssue(
                "speaker_voice_profile_contains_identity_claim",
                "speaker_voice_profiles",
                "voice profile contains demographic wording",
            ),
        ],
    )

    assert "describe only visible appearance" in prompt
    assert "Remove demographic, identity, nationality, and role claims" in prompt


def test_full_av_recheck_explains_speaker_identity_contradiction(
    tmp_path: Path,
) -> None:
    job = _job_fixture(tmp_path)
    backend, _ = _backend(tmp_path, [])
    prompt = backend._full_av_recheck_prompt(
        job,
        invalid_response="{}",
        issues=[
            ValidationIssue(
                "visible_entity_speaker_group_contradiction",
                "segment_decisions",
                "e1 maps to multiple groups",
            )
        ],
    )

    assert "reconsider clip-local speaker identity" in prompt
    assert "A group represents identity, not a turn" in prompt
    assert "Do not blindly merge groups" in prompt


def test_full_av_recheck_targets_unreliable_onscreen_speaker_evidence(
    tmp_path: Path,
) -> None:
    backend, _ = _backend(tmp_path, [])
    prompt = backend._full_av_recheck_prompt(
        _job_fixture(tmp_path),
        invalid_response="{}",
        issues=[
            ValidationIssue(
                "visible_entity_requires_confirmed_onscreen_speech",
                "segment_1",
                "insufficient onscreen evidence",
            ),
            ValidationIssue(
                "onscreen_speech_requires_reliable_visible_speaker_evidence",
                "segment_1",
                "insufficient onscreen evidence",
            ),
        ],
    )

    for phrase in (
        "reinspect every affected segment in its exact audiovisual interval",
        "synchronized mouth, lip, or jaw motion",
        "not merely an adjacent segment",
        "include visible_lip_motion",
        "speaker_visible_mouth_occluded",
        "av_temporal_alignment and/or voice_continuity",
        "If neither A nor B is supported, (C) do not claim visible_entity",
        "set entity_id=null",
        "offscreen with offscreen_spoken",
        "may preserve primary_speaker_group identity continuity",
        "do not establish which visible entity is speaking",
        "Never invent lip motion or mouth occlusion",
        "never preserve visible_entity merely to satisfy validation",
        "never bind a person merely because the same voice continues",
        "never transfer the audible speaker to a visible listener",
        "source_cluster_support or the current binding proposes it",
    ):
        assert phrase in prompt


def test_full_av_recheck_explains_presentation_evidence_contradiction(
    tmp_path: Path,
) -> None:
    backend, _ = _backend(tmp_path, [])
    prompt = backend._full_av_recheck_prompt(
        _job_fixture(tmp_path),
        invalid_response="{}",
        issues=[
            ValidationIssue(
                "visible_speaker_evidence_presentation_contradiction",
                "segment_1",
                "visible claim conflicts with offscreen evidence",
            )
        ],
    )

    assert "reinspect the exact segment" in prompt
    assert "offscreen_audio, voice_over_context, or device_playback_context" in prompt
    assert "Speaker-group identity may continue" in prompt


def test_unreliable_onscreen_evidence_can_be_corrected_by_one_recheck(
    tmp_path: Path,
) -> None:
    invalid_payload = _annotation().model_dump(mode="json")
    invalid_payload["segment_decisions"][0]["evidence_codes"] = [
        "av_temporal_alignment",
        "source_cluster_support",
    ]
    backend, completions = _backend(
        tmp_path,
        [
            (json.dumps(invalid_payload), 5),
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
    assert result.model_call_count == 2
    assert len(completions.requests) == 2
    assert result.annotation.segment_decisions[0].evidence_codes == [
        "visible_lip_motion",
        "av_temporal_alignment",
    ]


def _annotation_for_938_review(*, corrected_offscreen: bool) -> MimoAVAnnotationDraft:
    payload = _annotation().model_dump(mode="json")
    first = payload["segment_decisions"][0]
    first["segment_id"] = "segment_0001"
    later = []
    for segment_id in ("segment_0004", "segment_0005"):
        decision = json.loads(json.dumps(first))
        decision.update(
            segment_id=segment_id,
            primary_speaker_group="g2",
            binding_status=("offscreen" if corrected_offscreen else "visible_entity"),
            speech_presentation=(
                "offscreen_spoken" if corrected_offscreen else "onscreen_spoken"
            ),
            entity_id=None if corrected_offscreen else "e4",
            evidence_codes=(
                ["voice_continuity", "source_cluster_support", "offscreen_audio"]
                if corrected_offscreen
                else [
                    "av_temporal_alignment",
                    "voice_continuity",
                    "source_cluster_support",
                ]
            ),
        )
        later.append(decision)
    payload["segment_decisions"] = [first, *later]
    payload["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "clear measured voice"},
        {"speaker_group": "g2", "voice_characteristics": None},
    ]
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        _prose("The first visible speaker talks."),
        _audio_event("ae1"),
        _speech("segment_0001"),
        _prose("An offscreen voice continues while another person remains visible."),
        _speech("segment_0004"),
        _speech("segment_0005"),
    ]
    return MimoAVAnnotationDraft.model_validate(payload)


def _validate_938_review(annotation: MimoAVAnnotationDraft) -> list[ValidationIssue]:
    return _validate(
        annotation,
        segment_ids=["segment_0001", "segment_0004", "segment_0005"],
        segment_intervals={
            "segment_0001": (0.0, 1.0),
            "segment_0004": (1.0, 2.0),
            "segment_0005": (2.0, 3.0),
        },
        transcribed_segment_ids=[
            "segment_0001",
            "segment_0004",
            "segment_0005",
        ],
        authoritative_transcripts=["First", "Fourth", "Fifth"],
        allowed_entity_ids={"e1", "e4"},
        target_duration_seconds=3.0,
    )


def test_938_style_strong_indirect_visible_binding_is_rejected_by_segment() -> None:
    annotation = _annotation_for_938_review(corrected_offscreen=False)
    issues = _validate_938_review(annotation)

    assert [(item.field, item.code) for item in issues] == [
        ("segment_0004", "visible_entity_requires_confirmed_onscreen_speech"),
        (
            "segment_0004",
            "onscreen_speech_requires_reliable_visible_speaker_evidence",
        ),
        ("segment_0005", "visible_entity_requires_confirmed_onscreen_speech"),
        (
            "segment_0005",
            "onscreen_speech_requires_reliable_visible_speaker_evidence",
        ),
    ]


def test_938_corrected_offscreen_segments_reuse_speaker_group() -> None:
    annotation = _annotation_for_938_review(corrected_offscreen=True)

    assert not _validate_938_review(annotation)
    later = annotation.segment_decisions[1:]
    assert [item.primary_speaker_group for item in later] == ["g2", "g2"]
    assert [item.binding_status for item in later] == ["offscreen", "offscreen"]
    assert [item.speech_presentation for item in later] == [
        "offscreen_spoken",
        "offscreen_spoken",
    ]
    assert [item.entity_id for item in later] == [None, None]
    assert all("voice_continuity" in item.evidence_codes for item in later)


@pytest.mark.parametrize(
    ("evidence_codes", "binding_status", "speech_presentation"),
    [
        (["voice_continuity"], "no_reliable_entity", "uncertain"),
        (
            ["voice_continuity", "offscreen_audio"],
            "offscreen",
            "offscreen_spoken",
        ),
    ],
)
def test_repeated_unreliable_onscreen_evidence_is_conservatively_downgraded(
    tmp_path: Path,
    evidence_codes: list[str],
    binding_status: str,
    speech_presentation: str,
) -> None:
    invalid_payload = _annotation().model_dump(mode="json")
    invalid_payload["segment_decisions"][0]["evidence_codes"] = evidence_codes
    raw = json.dumps(invalid_payload)
    backend, completions = _backend(tmp_path, [(raw, 5), (raw, 5)])

    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    decision = result.annotation.segment_decisions[0]
    assert result.recheck_count == 1
    assert result.model_call_count == 2
    assert len(completions.requests) == 2
    assert decision.binding_status == binding_status
    assert decision.speech_presentation == speech_presentation
    assert decision.entity_id is None
    assert decision.confidence == "low"
    assert decision.primary_speaker_group == "g1"
    assert decision.vocal_composition == "single_speaker"
    assert decision.delivery_style == "calm and clear"
    assert set(decision.evidence_codes) == {
        *evidence_codes,
        "insufficient_evidence",
    }
    assert result.deterministic_correction_counts == {
        "conservative_visible_speaker_downgrade": 1
    }


def test_mixed_a073_issues_allow_downgrade_and_profile_normalization(
    tmp_path: Path,
) -> None:
    job = _three_transcribed_segment_job(tmp_path)
    invalid_payload = _annotation().model_dump(mode="json")
    first = invalid_payload["segment_decisions"][0]
    first["evidence_codes"] = ["voice_continuity"]
    invalid_payload["segment_decisions"] = []
    for index in (1, 2, 3):
        decision = dict(first)
        decision["segment_id"] = f"segment_{index}"
        decision["primary_speaker_group"] = f"g{index}"
        decision["delivery_style"] = f"measured delivery {index}"
        invalid_payload["segment_decisions"].append(decision)
    invalid_payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        _prose("A visible conversation unfolds."),
        _audio_event("ae1"),
        _speech("segment_1"),
        _speech("segment_2"),
        _speech("segment_3"),
    ]
    invalid_annotation = MimoAVAnnotationDraft.model_validate(invalid_payload)
    issue_codes = {
        issue.code
        for issue in _validate(
            invalid_annotation,
            segment_ids=["segment_1", "segment_2", "segment_3"],
            segment_intervals={
                "segment_1": (0.0, 1.0),
                "segment_2": (1.0, 2.0),
                "segment_3": (2.0, 3.0),
            },
            transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
            authoritative_transcripts=[
                "Exact, text!",
                "Exact text 2.",
                "Exact text 3.",
            ],
            target_duration_seconds=3.0,
        )
    }
    assert _CONSERVATIVE_VISIBLE_SPEAKER_ISSUES <= issue_codes
    assert "visible_entity_speaker_group_contradiction" in issue_codes
    assert "speaker_voice_profile_inventory_mismatch" in issue_codes
    raw = json.dumps(invalid_payload)
    backend, completions = _backend(tmp_path, [(raw, 5), (raw, 5)])

    result = backend.reconcile(
        job,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert len(completions.requests) == 2
    assert result.recheck_count == 1
    assert [item.primary_speaker_group for item in result.annotation.segment_decisions] == [
        "g1",
        "g2",
        "g3",
    ]
    assert [item.entity_id for item in result.annotation.segment_decisions] == [
        None,
        None,
        None,
    ]
    assert [item.binding_status for item in result.annotation.segment_decisions] == [
        "no_reliable_entity",
        "no_reliable_entity",
        "no_reliable_entity",
    ]
    assert [item.speech_presentation for item in result.annotation.segment_decisions] == [
        "uncertain",
        "uncertain",
        "uncertain",
    ]
    profiles = result.annotation.speaker_voice_profiles
    assert [item.speaker_group for item in profiles] == ["g1", "g2", "g3"]
    assert profiles[0].voice_characteristics == (
        "clear mid-register timbre with measured cadence"
    )
    assert [item.voice_characteristics for item in profiles[1:]] == [None, None]
    assert result.deterministic_correction_counts == {
        "conservative_visible_speaker_downgrade": 3,
        "speaker_voice_profile_inventory_normalization": 1,
    }


def test_a073_reliable_same_entity_groups_are_safely_canonicalized(
    tmp_path: Path,
) -> None:
    job = _three_transcribed_segment_job(tmp_path)
    invalid_annotation = _three_segment_annotation(
        [("g1", "e1"), ("g2", "e1"), ("g2", "e1")],
        profiles=[
            {
                "speaker_group": "g2",
                "voice_characteristics": "steady measured cadence",
            }
        ],
    )
    issue_codes = {
        issue.code
        for issue in _validate(
            invalid_annotation,
            segment_ids=["segment_1", "segment_2", "segment_3"],
            segment_intervals={
                "segment_1": (0.0, 1.0),
                "segment_2": (1.0, 2.0),
                "segment_3": (2.0, 3.0),
            },
            transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
            target_duration_seconds=3.0,
        )
    }
    assert "visible_entity_speaker_group_contradiction" in issue_codes
    assert "speaker_voice_profile_inventory_mismatch" in issue_codes
    raw = invalid_annotation.model_dump_json()
    backend, completions = _backend(tmp_path, [(raw, 5), (raw, 5)])

    result = backend.reconcile(
        job,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert len(completions.requests) == 2
    assert [item.primary_speaker_group for item in result.annotation.segment_decisions] == [
        "g1",
        "g1",
        "g1",
    ]
    assert [item.entity_id for item in result.annotation.segment_decisions] == [
        "e1",
        "e1",
        "e1",
    ]
    assert [item.speaker_group for item in result.annotation.speaker_voice_profiles] == [
        "g1"
    ]
    assert result.annotation.speaker_voice_profiles[0].voice_characteristics == (
        "steady measured cadence"
    )
    assert result.deterministic_correction_counts == {
        "same_visible_entity_speaker_group_merge": 1,
        "speaker_voice_profile_inventory_normalization": 1,
    }


def test_distinct_visible_entities_do_not_merge_speaker_groups() -> None:
    annotation = _three_segment_annotation(
        [("g1", "e1"), ("g2", "e2"), ("g2", "e2")]
    )

    corrected, counts, sources = _canonicalize_same_visible_entity_speaker_groups(
        annotation,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1", "e2"},
    )

    assert corrected == annotation
    assert counts == {}
    assert sources is None


def test_group_resolved_to_different_visible_entity_blocks_same_entity_merge(
    tmp_path: Path,
) -> None:
    annotation = _three_segment_annotation(
        [("g1", "e1"), ("g2", "e1"), ("g2", "e2")]
    )

    corrected, counts, sources = _canonicalize_same_visible_entity_speaker_groups(
        annotation,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1", "e2"},
    )

    assert corrected == annotation
    assert counts == {}
    assert sources is None
    issue_codes = {
        issue.code
        for issue in _validate(
            annotation,
            segment_ids=["segment_1", "segment_2", "segment_3"],
            segment_intervals={
                "segment_1": (0.0, 1.0),
                "segment_2": (1.0, 2.0),
                "segment_3": (2.0, 3.0),
            },
            transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
            authoritative_transcripts=["First", "Second", "Third"],
            allowed_entity_ids={"e1", "e2"},
            target_duration_seconds=3.0,
        )
    }
    assert "speaker_group_entity_contradiction" in issue_codes
    assert "visible_entity_speaker_group_contradiction" in issue_codes

    raw = annotation.model_dump_json()
    backend, _ = _backend(tmp_path, [(raw, 5), (raw, 5)])
    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _three_transcribed_segment_job(tmp_path),
            segment_ids=["segment_1", "segment_2", "segment_3"],
            transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
            allowed_entity_ids={"e1", "e2"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )
    assert "speaker_group_entity_contradiction" in {
        issue.code for issue in exc_info.value.issues
    }


def test_uncertain_visible_entity_binding_blocks_same_entity_merge() -> None:
    payload = _three_segment_annotation(
        [("g1", "e1"), ("g2", "e1"), ("g3", "e1")]
    ).model_dump(mode="json")
    payload["segment_decisions"][2]["resolution"] = "uncertain"
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    corrected, counts, sources = _canonicalize_same_visible_entity_speaker_groups(
        annotation,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1"},
    )

    assert corrected == annotation
    assert counts == {}
    assert sources is None


def test_group_merge_recanonicalizes_remaining_groups_and_profiles() -> None:
    annotation = _three_segment_annotation(
        [("g1", "e1"), ("g2", "e1"), ("g3", "e2")],
        profiles=[
            {"speaker_group": "g1", "voice_characteristics": "canonical profile"},
            {"speaker_group": "g2", "voice_characteristics": "merged profile"},
            {"speaker_group": "g3", "voice_characteristics": "second profile"},
        ],
    )

    corrected, counts, sources = _canonicalize_same_visible_entity_speaker_groups(
        annotation,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        allowed_entity_ids={"e1", "e2"},
    )
    corrected, profile_count = _normalize_speaker_voice_profiles(
        corrected,
        transcribed_segment_ids={"segment_1", "segment_2", "segment_3"},
        source_groups_by_final_group=sources,
    )

    assert [item.primary_speaker_group for item in corrected.segment_decisions] == [
        "g1",
        "g1",
        "g2",
    ]
    assert counts == {
        "same_visible_entity_speaker_group_merge": 1,
        "speaker_group_id_recanonicalization": 1,
    }
    assert profile_count == 1
    assert [item.model_dump(mode="json") for item in corrected.speaker_voice_profiles] == [
        {"speaker_group": "g1", "voice_characteristics": "canonical profile"},
        {"speaker_group": "g2", "voice_characteristics": "second profile"},
    ]
    assert not _validate(
        corrected,
        segment_ids=["segment_1", "segment_2", "segment_3"],
        segment_intervals={
            "segment_1": (0.0, 1.0),
            "segment_2": (1.0, 2.0),
            "segment_3": (2.0, 3.0),
        },
        transcribed_segment_ids=["segment_1", "segment_2", "segment_3"],
        authoritative_transcripts=["First", "Second", "Third"],
        allowed_entity_ids={"e1", "e2"},
        target_duration_seconds=3.0,
    )


def test_recheck_visible_lip_motion_positive_stays_visible(tmp_path: Path) -> None:
    invalid = "not json"
    valid = _annotation().model_dump_json()
    backend, _ = _backend(tmp_path, [(invalid, 5), (valid, 5)])

    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.annotation.segment_decisions[0].binding_status == "visible_entity"
    assert result.annotation.segment_decisions[0].entity_id == "e1"
    assert result.deterministic_correction_counts == {}


def test_visible_speaker_downgrade_does_not_hide_other_semantic_failures(
    tmp_path: Path,
) -> None:
    invalid_payload = _annotation().model_dump(mode="json")
    invalid_payload["segment_decisions"][0]["evidence_codes"] = [
        "voice_continuity"
    ]
    invalid_payload["h3_draft"]["subject_definitions"].append(
        _subject_definition("<Subject 2>", "an invented extra person.")
    )
    raw = json.dumps(invalid_payload)
    backend, _ = _backend(tmp_path, [(raw, 5), (raw, 5)])

    with pytest.raises(MimoBackendFailure) as exc_info:
        backend.reconcile(
            _job_fixture(tmp_path),
            segment_ids=["segment_1"],
            transcribed_segment_ids=["segment_1"],
            allowed_entity_ids={"e1"},
            allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        )

    assert exc_info.value.code == "mimo_structured_output_failed"
    assert "subject_definition_contract_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }
    assert not (
        _CONSERVATIVE_VISIBLE_SPEAKER_ISSUES
        & {issue.code for issue in exc_info.value.issues}
    )


def test_sglang_full_av_recheck_preserves_primary_transport_contract(
    tmp_path: Path,
) -> None:
    backend, completions = _backend(
        tmp_path,
        [("not json", 5), (_annotation().model_dump_json(), 5)],
        transport="sglang",
    )

    result = backend.reconcile(
        _job_fixture(tmp_path),
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )

    assert result.model_call_count == 2
    assert result.recheck_count == 1
    for request in completions.requests:
        assert request["reasoning_effort"] == "none"
        assert request["extra_body"]["use_audio_in_video"] is True
        content = request["messages"][1]["content"]  # type: ignore[index]
        assert any(item["type"] == "video_url" for item in content)
        assert not any(item["type"] in {"audio_url", "input_audio"} for item in content)


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
    assert all(
        isinstance(request["messages"][1]["content"], list)
        for request in completions.requests
    )  # type: ignore[index]


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
    canonical_payload = sample.model_dump(mode="python")
    canonical_payload.update(
        {
            "sample_id": "clip-1/canonical",
            "pair_id": "canonical/clip-1",
            "pair_type": "canonical",
            "subject_voices": [],
        }
    )
    canonical = FinalH3SampleV2.model_validate(canonical_payload)
    with pytest.raises(ValueError, match="variants disagree"):
        _validate_h3_variant_observations("clip-1", [canonical, changed])


def test_only_subject_reference_entities_are_speaker_bindable(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    image = tmp_path / "object.png"
    image.write_bytes(b"object")
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["reference_images"].append(
        MimoReferenceImage(
            image_index=2,
            picture_label="<Picture 2>",
            source_image_index=2,
            source_image_id="image_2",
            source_image_label="<Image 2>",
            kind="object",
            entity_id="e2",
            image_artifact_path=str(image.resolve()),
            image_sha256="4" * 64,
        ).model_dump(mode="json")
    )
    values["reference_selection"].update(
        original_picture_count=2,
        selected_picture_count=2,
        selected_source_image_indexes=[1, 2],
        selected_source_image_ids=["image_1", "image_2"],
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
    assert (
        "Multiple vocal sounds inside one segment never make that segment invalid"
        in SYSTEM_PROMPT
    )
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
    second["speech_presentation"] = (
        "onscreen_spoken" if second_entity is not None else "offscreen_spoken"
    )
    second["evidence_codes"] = (
        ["visible_lip_motion"] if second_entity is not None else ["offscreen_audio"]
    )
    second["entity_id"] = second_entity
    second["delivery_style"] = "brief and clear"
    payload["segment_decisions"].append(second)
    if second_group != "g1":
        payload["speaker_voice_profiles"].append(
            {"speaker_group": second_group, "voice_characteristics": None}
        )
    payload["h3_draft"]["shots"][0]["timeline_parts"].extend(
        [_prose("Then the action continues."), _speech("segment_2")]
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
    assert {item.code for item in issues} == {
        "segment_inventory_mismatch",
        "unknown_entity",
    }


@pytest.mark.parametrize(
    ("field", "value", "issue_code"),
    [
        (
            "summary",
            "A summary [[segment:segment_1]]",
            "speech_placeholder_outside_shot",
        ),
        (
            "visual_retention_analysis",
            [
                _retention(
                    "<Subject 1>",
                    "fully_preserved",
                    "appearance remains [[segment:segment_1]]",
                )
            ],
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
        ("The person says", "draft_prefixes_complete_speech_placeholder"),
    ],
)
def test_h3_draft_rejects_pipeline_owned_or_ambiguous_syntax(
    draft_text: str,
    issue_code: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    if issue_code == "draft_prefixes_complete_speech_placeholder":
        payload["h3_draft"]["shots"][0]["timeline_parts"] = [
            _prose(draft_text),
            _speech("segment_1"),
            _audio_event("ae1"),
        ]
    else:
        payload["h3_draft"]["summary"] = draft_text
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert issue_code in {item.code for item in issues}


def test_timeline_parts_use_a_discriminated_union() -> None:
    shot_schema = MimoAVAnnotationDraft.model_json_schema()["$defs"]["MimoH3Shot"]
    item_schema = shot_schema["properties"]["timeline_parts"]["items"]
    assert item_schema["discriminator"]["propertyName"] == "type"
    assert set(item_schema["discriminator"]["mapping"]) == {
        "prose",
        "speech",
        "audio_event",
    }


def test_subject_and_retention_fields_are_structured_in_json_schema() -> None:
    schema = MimoAVAnnotationDraft.model_json_schema()
    subject = schema["$defs"]["MimoSubjectDefinitionDraft"]
    retention = schema["$defs"]["MimoVisualRetentionDraft"]

    assert subject["properties"]["subject_label"]["pattern"] == (
        r"^<Subject [1-9]\d*>$"
    )
    assert set(subject["required"]) == {"subject_label", "description"}
    assert retention["properties"]["marker"]["enum"] == [
        "fully_preserved",
        "partially_preserved",
        "weak_reference",
    ]
    assert set(retention["required"]) == {
        "subject_label",
        "marker",
        "description",
    }


@pytest.mark.parametrize(
    "forbidden_text",
    [
        "[[segment:segment_1]]",
        "[[audio_event:ae1]]",
        "(S1) speaks.",
        "<d>[English] text</d>",
        "[Shot 1] A person stands.",
        "<Video 1> supplies motion.",
        "<Audio 1> supplies sound.",
        "<Unknown 1> supplies an invented reference.",
    ],
)
def test_timeline_prose_rejects_pipeline_owned_syntax(forbidden_text: str) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["timeline_parts"][0]["text"] = forbidden_text
    with pytest.raises(ValidationError, match="pipeline-owned syntax"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize(
    "marker",
    ["fully_preserved", "partially_preserved", "weak_reference"],
)
def test_allowed_visual_retention_markers_pass(marker: str) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        _retention(
            "<Subject 1>",
            marker,
            "observed appearance remains grounded.",
        )
    ]
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "subject_retention_contract_mismatch" not in {
        item.code for item in issues
    }


@pytest.mark.parametrize("marker", ["attribute_transfer", "invented_marker"])
def test_attribute_transfer_and_unknown_retention_marker_reject(marker: str) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        _retention("<Subject 1>", marker, "invalid.")
    ]
    with pytest.raises(ValidationError, match="marker"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize(
    ("marker", "repeated"),
    [
        ("fully_preserved", "fully_preserved"),
        ("fully_preserved", "fully preserved"),
        ("partially_preserved", "partially preserved"),
        ("weak_reference", "weak reference"),
    ],
)
def test_retention_description_cannot_repeat_marker(
    marker: str,
    repeated: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        _retention("<Subject 1>", marker, f"the result is {repeated}.")
    ]
    with pytest.raises(ValidationError, match="repeats a retention marker"):
        MimoAVAnnotationDraft.model_validate(payload)


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
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        _prose("<Subject 1> remains visible."),
        _audio_event("ae1"),
        _speech("segment_1"),
        _audio_event("ae2"),
    ]
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
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        *(_audio_event(f"ae{index}") for index in range(1, 11)),
        _speech("segment_1"),
    ]
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert len(annotation.audio_semantics.temporal_non_speech_events) == 10


@pytest.mark.parametrize(
    ("timeline_parts", "issue_code"),
    [
        (
            [_audio_event("ae1"), _speech("segment_1")],
            "missing_audio_event_placeholder",
        ),
        (
            [
                _audio_event("ae1"),
                _audio_event("ae1"),
                _speech("segment_1"),
                _audio_event("ae2"),
            ],
            "duplicate_audio_event_placeholder",
        ),
        (
            [_audio_event("ae1"), _speech("segment_1"), _audio_event("ae3")],
            "unknown_audio_event_placeholder",
        ),
        (
            [_audio_event("ae2"), _speech("segment_1"), _audio_event("ae1")],
            "audio_event_placeholder_order_mismatch",
        ),
    ],
)
def test_audio_event_placeholder_coverage_is_exact(
    timeline_parts: list[dict[str, str]],
    issue_code: str,
) -> None:
    payload = _two_audio_event_annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["timeline_parts"] = timeline_parts
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert issue_code in {item.code for item in issues}


def test_free_form_timeline_syntax_is_rejected_outside_shots() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["summary"] += " [[audio_event:ae1]]"
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "speech_placeholder_outside_shot" in {item.code for item in issues}


def test_audiovisual_summary_is_required_and_nonempty() -> None:
    payload = _annotation().model_dump(mode="json")
    del payload["audio_semantics"]["audiovisual_summary"]
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(payload)
    payload["audio_semantics"]["audiovisual_summary"] = " "
    with pytest.raises(ValidationError, match="summary must not be empty"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_segment_delivery_matches_authoritative_transcription_status() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["segment_decisions"][0]["delivery_style"] = None
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "missing_transcribed_segment_delivery" in {item.code for item in issues}

    payload["segment_decisions"][0]["delivery_style"] = "calm and clear"
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        transcribed_segment_ids=[],
    )
    assert "non_transcribed_segment_delivery" in {item.code for item in issues}


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
        _subject_definition("<Subject 2>", "a second person with a distinct coat.")
    )
    payload["h3_draft"]["visual_retention_analysis"].append(
        _retention(
            "<Subject 2>",
            "weak_reference",
            "only limited observed structure is retained.",
        )
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
        _retention("<Subject 1>", "fully_preserved", "retained."),
        _retention("<Subject 1>", "weak_reference", "duplicated."),
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
    "extra_description",
    [
        "An extra arbitrary definition.",
        "A second arbitrary visual definition.",
    ],
)
def test_subject_definitions_reject_noncanonical_extra_rows(
    extra_description: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"].append(
        _subject_definition("<Subject 2>", extra_description)
    )
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "subject_definition_contract_mismatch" in {item.code for item in issues}


def test_visual_retention_requires_exact_canonical_subject_rows() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["visual_retention_analysis"] = [
        _retention("<Subject 2>", "fully_preserved", "visible.")
    ]
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
        payload["segment_decisions"][0]["delivery_style"] = transcript
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
    shot_parts = {
        1: [_prose("<Subject 1> remains visible.")],
        2: [_prose("The scene continues.")],
    }
    shot_parts[speech_shot].append(_speech("segment_1"))
    shot_parts[event_shot].append(_audio_event("ae1"))
    payload["h3_draft"]["shots"] = [
        {
            "shot_index": 1,
            "start_time": None,
            "timeline_parts": shot_parts[1],
        },
        {
            "shot_index": 2,
            "start_time": 5.0,
            "timeline_parts": shot_parts[2],
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
    assert "audio_event_placeholder_wrong_shot" not in {item.code for item in issues}


def test_subject_definition_picture_omission_is_valid_but_shot_bounds_fail_closed() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"] = [
        _subject_definition("<Subject 1>", "has no supplied source Picture.")
    ]
    payload["h3_draft"]["shots"].append(
        {
            "shot_index": 2,
            "start_time": 1.0,
            "timeline_parts": [_prose("A hard cut occurs.")],
        }
    )
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        target_duration_seconds=1.0,
    )
    assert {item.code for item in issues} == {"shot_start_outside_target"}


@pytest.mark.parametrize("start_time", [None, 0])
def test_first_shot_implicit_or_explicit_zero_is_accepted(
    start_time: float | None,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["start_time"] = start_time
    draft = MimoAVAnnotationDraft.model_validate(payload)
    assert draft.h3_draft.shots[0].start_time == start_time


def test_first_shot_nonzero_start_time_is_rejected() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["start_time"] = 0.1
    with pytest.raises(ValidationError, match="start implicitly or at zero"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize("start_time", [None, 0])
def test_later_shot_missing_or_zero_start_time_is_rejected(
    start_time: float | None,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"].append(
        {
            "shot_index": 2,
            "start_time": start_time,
            "timeline_parts": [_prose("A hard cut reveals another angle.")],
        }
    )
    with pytest.raises(ValidationError, match="later MiMo H3"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_later_positive_hard_cuts_remain_strictly_ordered() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"].extend(
        [
            {
                "shot_index": 2,
                "start_time": 0.25,
                "timeline_parts": [_prose("A hard cut reveals another angle.")],
            },
            {
                "shot_index": 3,
                "start_time": 0.75,
                "timeline_parts": [_prose("A final hard cut returns to the subject.")],
            },
        ]
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert [shot.start_time for shot in annotation.h3_draft.shots] == [
        None,
        0.25,
        0.75,
    ]
    payload["h3_draft"]["shots"][2]["start_time"] = 0.2
    with pytest.raises(ValidationError, match="strictly increase"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_subject_definition_rejects_model_authored_picture_provenance() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"] = [
        _subject_definition("<Subject 1>", "is shown only in <Picture 2>.")
    ]
    with pytest.raises(ValidationError, match="cannot own Picture provenance"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize("subject_label", ["Subject 1", "<Subject 0>"])
def test_subject_definition_requires_exact_structured_subject_label(
    subject_label: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0]["subject_label"] = subject_label

    with pytest.raises(ValidationError, match="subject_label"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_subject_definition_description_rejects_bare_subject_label() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0]["description"] = (
        "Subject 1 is the person with dark hair."
    )

    with pytest.raises(ValidationError, match="bare Subject label"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_subject_definition_description_rejects_bracketed_subject_label() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0]["description"] = (
        "<Subject 1> is the person with dark hair."
    )

    with pytest.raises(ValidationError, match="repeats a Subject label"):
        MimoAVAnnotationDraft.model_validate(payload)


def test_subject_definition_rejects_even_duplicate_model_picture_labels() -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0]["description"] = (
        "is shown in <Picture 1>, with details repeated from <Picture 1>."
    )

    with pytest.raises(ValidationError, match="cannot own Picture provenance"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize("audio_term", ["timbre", "cadence", "articulation"])
def test_visual_subject_definition_rejects_audio_profile_leakage(
    audio_term: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0]["description"] = (
        f"a clearly visible face with a steady {audio_term}."
    )

    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))

    assert "subject_definition_contains_audio_profile" in {
        issue.code for issue in issues
    }


@pytest.mark.parametrize(
    "speech_segment_ids",
    [
        ["segment_1"],
        ["segment_1", "segment_1", "segment_2"],
        ["segment_2", "segment_1"],
    ],
)
def test_speech_placeholder_inventory_rejects_missing_duplicate_or_reordered(
    speech_segment_ids: list[str],
) -> None:
    payload = _two_segment_annotation(second_group="g1", second_entity=None).model_dump(
        mode="json"
    )
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        _prose("<Subject 1> remains visible."),
        _audio_event("ae1"),
        *(_speech(segment_id) for segment_id in speech_segment_ids),
    ]
    issues = _validate(
        MimoAVAnnotationDraft.model_validate(payload),
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
    )
    assert "speech_placeholder_inventory_mismatch" in {item.code for item in issues}


def test_non_transcribed_segment_cannot_receive_speech_placeholder() -> None:
    annotation = _two_segment_annotation(second_group="g1", second_entity=None)
    issues = _validate(
        annotation,
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1"],
    )
    assert "speech_placeholder_inventory_mismatch" in {item.code for item in issues}


def test_exact_transcribed_speech_placeholder_inventory_is_accepted() -> None:
    annotation = _two_segment_annotation(second_group="g1", second_entity=None)
    issues = _validate(
        annotation,
        segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"],
    )
    assert "speech_placeholder_inventory_mismatch" not in {item.code for item in issues}


def test_warning_requires_known_segment_and_never_contains_replacement_text() -> None:
    warning = MimoAnnotationWarning(
        code="possible_asr_conflict", segment_id="segment_1"
    )
    assert warning.model_dump(mode="json") == {
        "code": "possible_asr_conflict",
        "segment_id": "segment_1",
    }
    assert (
        "replacement_text"
        not in MimoAnnotationWarning.model_json_schema()["properties"]
    )
    payload = _annotation().model_dump(mode="json")
    payload["warnings"] = [{"code": "possible_asr_conflict", "segment_id": "unknown"}]
    issues = _validate(MimoAVAnnotationDraft.model_validate(payload))
    assert "warning_unknown_segment" in {item.code for item in issues}


def _sample(tmp_path: Path) -> FinalH3SampleV2:
    video = tmp_path / "target.mp4"
    audio = tmp_path / "full.flac"
    voice = tmp_path / "voice.flac"
    image = tmp_path / "reference.png"
    for path, value in (
        (video, b"v"),
        (audio, b"a"),
        (voice, b"x"),
        (image, b"i"),
    ):
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
        target_full_audio_sha256=_file_sha256(audio),
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
                voice_reference_sha256=_file_sha256(voice),
                source_start=0.0,
                source_end=1.0,
                source_start_sample=0,
                source_end_sample=32000,
                sample_mapping_policy="round_time_seconds_times_32000_v1",
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
                source_end_sample=32000,
                source_sample_rate_hz=32000,
                start_time=0.0,
                end_time=1.0,
                text="Exact, text!",
                language="English",
            )
        ],
    )


def _visual_reference_inventory(
    tmp_path: Path,
    kinds: list[str],
    *,
    same_entity: bool = False,
) -> list[FinalVisualReference]:
    references = []
    for index, value in enumerate(kinds, start=1):
        image = tmp_path / f"selection-reference-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        is_attribute = value not in {"subject", "object", "group", "background"}
        kind = "attribute" if is_attribute else value
        references.append(
            FinalVisualReference(
                image_id=f"image_{index}",
                image_index=index,
                kind=kind,
                image_path=f"selected/reference-{index}.png",
                image_artifact_path=str(image.resolve()),
                entity_id=(
                    None
                    if kind in {"attribute", "background"}
                    else "e1"
                    if same_entity
                    else f"e{index}"
                ),
                attribute_id=f"a{index}" if is_attribute else None,
                owner_entity_id="e1" if is_attribute else None,
                attribute_type=(
                    "upper_clothing" if value == "clothing" else value
                )
                if is_attribute
                else None,
                source_frame_index=index - 1,
                scope=(
                    None
                    if is_attribute
                    else "scene"
                    if kind == "background"
                    else "full"
                ),
                visible_region=None if is_attribute else "whole",
                synthetic=False,
            )
        )
    return references


def _sample_with_reference_inventory(
    tmp_path: Path,
    references: list[FinalVisualReference],
) -> FinalH3SampleV2:
    sample = _sample(tmp_path)
    values = sample.model_dump(mode="python")
    values.update(
        sample_id="clip-1/canonical",
        pair_id="canonical/clip-1",
        pair_type="canonical",
        subject_voices=[],
    )
    values["visual_references"] = [
        item.model_dump(mode="python") for item in references
    ]
    return FinalH3SampleV2.model_validate(values)


def _job_for_reference_inventory(
    tmp_path: Path,
    sample: FinalH3SampleV2,
) -> MimoClipJob:
    base = _job_fixture(tmp_path)
    selection, images = select_mimo_reference_projection(
        sample.clip_uid,
        sample.visual_references,
    )
    projected = project_mimo_h3_sample_references(
        sample,
        reference_images=images,
        reference_selection=selection,
    )
    contract = mimo25_materializer.build_reference_contract(projected, "visual_only")
    values = base.model_dump(mode="json", exclude={"request_fingerprint"})
    values.update(
        r2v_instruction=sample.r2v_instruction,
        reference_selection=selection.model_dump(mode="json"),
        reference_images=[item.model_dump(mode="json") for item in images],
        reference_subjects=[
            item.model_dump(mode="json") for item in contract.subjects
        ],
        source_h3_sample_ids=[sample.sample_id],
    )
    return _job(values)


@pytest.mark.parametrize("picture_count", [8, 9])
def test_mimo_reference_selection_is_exact_noop_at_or_below_limit(
    tmp_path: Path,
    picture_count: int,
) -> None:
    references = _visual_reference_inventory(
        tmp_path,
        ["subject"] * picture_count,
        same_entity=True,
    )
    selection, projected = select_mimo_reference_projection("clip-1", references)

    assert selection.policy_version == MIMO25_REFERENCE_SELECTION_POLICY_VERSION
    assert selection.original_picture_count == picture_count
    assert selection.selected_picture_count == picture_count
    assert selection.dropped_references == []
    assert [item.source_image_id for item in projected] == [
        item.image_id for item in references
    ]
    assert [item.image_index for item in projected] == list(
        range(1, picture_count + 1)
    )


def test_mimo_reference_selection_drops_one_hair_deterministically(
    tmp_path: Path,
) -> None:
    kinds = ["subject", "subject", "subject", "hair"] + ["subject"] * 6
    references = _visual_reference_inventory(tmp_path, kinds, same_entity=True)

    first = select_mimo_reference_projection("clip-1", references)
    second = select_mimo_reference_projection("clip-1", list(reversed(references)))

    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert [item.model_dump(mode="json") for item in first[1]] == [
        item.model_dump(mode="json") for item in second[1]
    ]
    assert first[0].selected_picture_count == 9
    assert [item.drop_reason for item in first[0].dropped_references] == [
        "hair_capacity_trim"
    ]
    assert first[0].dropped_references[0].source_image_index == 4
    assert [item.source_image_index for item in first[1]] == [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert [item.picture_label for item in first[1]] == [
        f"<Picture {index}>" for index in range(1, 10)
    ]


@pytest.mark.parametrize(
    ("kinds", "expected_reasons"),
    [
        (
            ["subject"] * 9 + ["hair", "face"],
            ["hair_capacity_trim", "face_capacity_trim"],
        ),
        (
            ["subject"] * 9 + ["hair", "face", "clothing"],
            [
                "hair_capacity_trim",
                "face_capacity_trim",
                "attribute_capacity_trim",
            ],
        ),
        (["subject"] * 9 + ["face"], ["face_capacity_trim"]),
        (["subject"] * 9 + ["clothing"], ["attribute_capacity_trim"]),
    ],
)
def test_mimo_reference_selection_uses_exact_priority_cycle(
    tmp_path: Path,
    kinds: list[str],
    expected_reasons: list[str],
) -> None:
    references = _visual_reference_inventory(tmp_path, kinds, same_entity=True)
    selection, projected = select_mimo_reference_projection("clip-1", references)

    assert selection.selected_picture_count == 9
    assert len(projected) == 9
    assert [item.drop_reason for item in selection.dropped_references] == (
        expected_reasons
    )
    assert [item.source_image_index for item in projected] == sorted(
        item.source_image_index for item in projected
    )


def test_mimo_reference_selection_fails_closed_without_attributes(
    tmp_path: Path,
) -> None:
    references = _visual_reference_inventory(
        tmp_path,
        ["subject", "object", "group", "background", "subject"] * 2,
    )
    with pytest.raises(ValueError, match="without a droppable attribute reference"):
        select_mimo_reference_projection("clip-1", references)


def test_selected_reference_mapping_controls_media_and_materialization(
    tmp_path: Path,
) -> None:
    kinds = ["subject", "subject", "subject", "hair"] + ["subject"] * 6
    references = _visual_reference_inventory(tmp_path, kinds, same_entity=True)
    sample = _sample_with_reference_inventory(tmp_path, references)
    source_sample_snapshot = sample.model_dump_json()
    source_image_snapshots = {
        Path(item.image_artifact_path): Path(item.image_artifact_path).read_bytes()
        for item in sample.visual_references
    }
    job = _job_for_reference_inventory(tmp_path, sample)
    dropped = job.reference_selection.dropped_references[0]

    assert dropped.source_image_index == 4
    assert [item.source_image_index for item in job.reference_images] == [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert [item.image_index for item in job.reference_images] == list(range(1, 10))

    backend, _ = _backend(tmp_path, [])
    content = backend._media_content(job, include_audio_fallback=False)
    image_items = [item for item in content if item["type"] == "image_url"]
    metadata = [
        json.loads(str(item["text"]))
        for item in content
        if item["type"] == "text"
    ]
    assert len(image_items) == 9
    assert [item["source_image_index"] for item in metadata] == [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert dropped.source_image_id not in {
        item["source_image_id"] for item in metadata
    }

    projected = project_mimo_h3_sample_references(
        sample,
        reference_images=job.reference_images,
        reference_selection=job.reference_selection,
    )
    with pytest.raises(ValidationError, match="per-modality reference limit"):
        mimo25_materializer.build_reference_contract(sample, "visual_only")
    contract = mimo25_materializer.build_reference_contract(projected, "visual_only")
    assert len(contract.pictures) == 9
    assert [item.image_id for item in contract.pictures] == [
        f"image_{index}" for index in range(1, 10)
    ]

    annotation = _annotation()
    _, rendered, _ = _materialize_sample(
        sample,
        job,
        _record_fixture(tmp_path, annotation, job=job),
    )
    assert "<Picture 9>" in rendered
    assert "<Picture 10>" not in rendered
    assert dropped.source_image_id not in rendered
    assert "<Audio " not in rendered
    assert sample.model_dump_json() == source_sample_snapshot
    assert all(path.read_bytes() == content for path, content in source_image_snapshots.items())


def _record_fixture(
    tmp_path: Path,
    annotation: MimoAVAnnotationDraft,
    *,
    job: MimoClipJob | None = None,
    inventory_fingerprint: str = "a" * 64,
) -> MimoRecord:
    active_job = job or _job_fixture(tmp_path)
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    provenance = MimoBackendConfig(
        media_resolver=resolver, api_key="secret"
    ).provenance()
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
        "deterministic_correction_counts": {},
    }
    fingerprint = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        .hexdigest()
    )
    return MimoRecord(**values, record_fingerprint=fingerprint)


def _replace_record(record: MimoRecord, **changes: object) -> MimoRecord:
    values = record.model_dump(mode="json", exclude={"record_fingerprint"})
    values.update(changes)
    fingerprint = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                values,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        .hexdigest()
    )
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


def test_mimo_inventory_builds_one_job_per_canonical_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_root = tmp_path / "production"
    paths = jea_production_paths(production_root)
    visual_root = tmp_path / "visual"
    visual_runs_root = tmp_path / "visual-runs"
    visual_root.mkdir()
    visual_runs_root.mkdir()
    (visual_root / "samples.jsonl").write_text("{}\n", encoding="utf-8")

    sample_roots = [tmp_path / "clip-1", tmp_path / "clip-2"]
    for root in sample_roots:
        root.mkdir()
    seed_samples = [_sample(root) for root in sample_roots]

    def variant(
        seed: FinalH3SampleV2,
        *,
        clip_uid: str,
        pair_type: str,
        suffix: str,
    ) -> FinalH3SampleV2:
        payload = seed.model_dump(mode="json")
        payload.update(
            {
                "sample_id": f"{clip_uid}/{suffix}",
                "pair_id": f"{pair_type}/{clip_uid}",
                "pair_type": pair_type,
                "clip_uid": clip_uid,
                "clip_display_path": f"01/show/episode/{clip_uid}",
                "clip_name": clip_uid,
                "speech_segments": [],
                "subject_voices": (
                    [] if pair_type == "canonical" else payload["subject_voices"]
                ),
            }
        )
        if pair_type == "cross_pair":
            payload["subject_voices"][0].update(
                {
                    "voice_source": "cross_donor",
                    "donor_occurrence_id": "donor/e1",
                    "donor_clip_uid": "donor",
                    "donor_clip_display_path": "01/show/episode/donor",
                }
            )
        return FinalH3SampleV2.model_validate(payload)

    canonical_1 = variant(
        seed_samples[0],
        clip_uid="clip-1",
        pair_type="canonical",
        suffix="z-canonical",
    )
    in_pair_1 = variant(
        seed_samples[0],
        clip_uid="clip-1",
        pair_type="in_pair",
        suffix="a-in-pair",
    )
    cross_pair_1 = variant(
        seed_samples[0],
        clip_uid="clip-1",
        pair_type="cross_pair",
        suffix="b-cross-pair",
    )
    canonical_2 = variant(
        seed_samples[1],
        clip_uid="clip-2",
        pair_type="canonical",
        suffix="canonical",
    )
    h3_samples = [in_pair_1, cross_pair_1, canonical_1, canonical_2]

    canonical_audio = []
    targets = []
    visual_clips = []
    for sample in (canonical_1, canonical_2):
        video = Path(sample.target_video).resolve(strict=True)
        audio = Path(sample.target_full_audio_path).resolve(strict=True)
        identity = SimpleNamespace(clip_uid=sample.clip_uid)
        visual_clips.append(
            SimpleNamespace(
                identity=identity, sample=SimpleNamespace(target_video=str(video))
            )
        )
        canonical_audio.append(
            CanonicalAudioClip(
                clip_uid=sample.clip_uid,
                clip_display_path=sample.clip_display_path,
                media_collection_relpath=sample.media_collection_relpath,
                media_collection_name=sample.media_collection_name,
                episode_name=sample.episode_name,
                clip_name=sample.clip_name,
                shard_id=sample.shard_id,
                target_video_path=str(video),
                target_video_sha256=_file_sha256(video),
                target_full_audio_path=str(audio),
                target_full_audio_sha256=_file_sha256(audio),
                frame_count=32000,
                target_duration_seconds=1.0,
                subject_reference_count=1,
            )
        )
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=sample.clip_uid,
                target_video_path=str(video),
                source_audio_path=str(audio),
                source_audio_sha256=_file_sha256(audio),
                source_sample_rate_hz=32000,
                source_channels=2,
                source_frame_count=32000,
                visual_references=[],
            )
        )

    _write_models_jsonl(paths.audio / "canonical_clips.jsonl", canonical_audio)
    canonical_manifest_sha256 = _file_sha256(paths.audio / "canonical_clips.jsonl")
    paths.diarization.mkdir(parents=True)
    diarization_inventory = DiarizationInventory(
        mode="production",
        source_inventory_kind="canonical_audio_manifest",
        source_visual_production_root=str(visual_root),
        source_visual_inventory_path=str(visual_root / "samples.jsonl"),
        source_visual_inventory_sha256=_file_sha256(visual_root / "samples.jsonl"),
        source_canonical_audio_manifest_path=str(paths.audio / "canonical_clips.jsonl"),
        source_canonical_audio_manifest_sha256=canonical_manifest_sha256,
        inventory_fingerprint="a" * 64,
        source_target_count=2,
        selected_target_count=2,
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )
    (paths.diarization / "inventory.json").write_text(
        diarization_inventory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    for path in (
        paths.diarization / "raw_segments.jsonl",
        paths.diarization / "bound_segments.jsonl",
        paths.asr / "segments.jsonl",
        production_root / "binding_audit_v1/segments.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    _write_models_jsonl(paths.h3 / "samples.jsonl", h3_samples)

    visual_inventory = SimpleNamespace(canonical_clips=visual_clips)
    monkeypatch.setattr(
        mimo25_reconcile,
        "load_visual_production_inventory",
        lambda **_kwargs: visual_inventory,
    )
    inventory = build_mimo25_inventory(
        visual_production_root=visual_root,
        visual_runs_root=visual_runs_root,
        audio_production_root=production_root,
    )

    assert inventory.inventory_scope == "canonical_visual_target_inventory"
    assert inventory.canonical_wide_coverage is True
    assert [job.clip_uid for job in inventory.jobs] == ["clip-1", "clip-2"]
    assert inventory.jobs[0].source_h3_sample_ids == [
        "clip-1/a-in-pair",
        "clip-1/b-cross-pair",
        "clip-1/z-canonical",
    ]
    assert inventory.jobs[1].source_h3_sample_ids == ["clip-2/canonical"]

    subset = build_mimo25_inventory(
        visual_production_root=visual_root,
        visual_runs_root=visual_runs_root,
        audio_production_root=production_root,
        case_manifest=MimoCaseManifest(clip_uids=["clip-2"]),
    )
    assert subset.inventory_scope == "explicit_case_subset"
    assert subset.canonical_wide_coverage is False
    assert [job.clip_uid for job in subset.jobs] == ["clip-2"]


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
    canonical_payload = sample.model_dump(mode="python")
    canonical_payload.update(
        {
            "sample_id": "clip-1/canonical",
            "pair_id": "canonical/clip-1",
            "pair_type": "canonical",
            "subject_voices": [],
        }
    )
    samples = [FinalH3SampleV2.model_validate(canonical_payload)]
    if source_sample_count >= 2:
        samples.append(sample)
    for index in range(3, source_sample_count + 1):
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
        _presentation_annotation("offscreen_spoken"),
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


class _RecoveredVoiceAnalyzer:
    def __init__(self, *, seconds: float = 3.0, speech_amplitude: int = 3000) -> None:
        frame_count = round(seconds * 32000)
        samples = np.full(frame_count, 100, dtype=np.int16)
        samples[: min(32000, frame_count)] = speech_amplitude
        self.analysis = CanonicalAudioAnalysis(
            probe=AudioFileProbe(
                sample_rate_hz=32000,
                channels=2,
                frame_count=frame_count,
                duration_seconds=seconds,
                format_name="flac",
            ),
            mono_pcm16=samples,
        )
        self.calls: list[Path] = []

    def load(self, path: Path) -> CanonicalAudioAnalysis:
        self.calls.append(path)
        return self.analysis


class _RecoveredVoiceMediaBackend:
    def __init__(self) -> None:
        self.extractions: list[dict[str, object]] = []

    def probe_audio_file(self, path: Path) -> AudioFileProbe:
        del path
        raise AssertionError("recovery uses the injected analyzer probe")

    def materialize_full_audio(self, **_: object) -> object:
        raise AssertionError("recovery never materializes canonical full audio")

    def extract_voice_reference(self, **kwargs: object) -> Path:
        self.extractions.append(dict(kwargs))
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"deterministic-32k-stereo-flac")
        return destination


class _ConditioningAudioBackend(_RecoveredVoiceMediaBackend):
    def __init__(self, *, source_duration_seconds: float) -> None:
        super().__init__()
        self.source_duration_seconds = source_duration_seconds

    def probe_audio_file(self, path: Path) -> AudioFileProbe:
        for extraction in self.extractions:
            if Path(extraction["destination"]) == path:
                start = int(extraction["source_start_sample"])
                end = int(extraction["source_end_sample"])
                return AudioFileProbe(
                    sample_rate_hz=32000,
                    channels=2,
                    frame_count=end - start,
                    duration_seconds=(end - start) / 32000,
                    format_name="flac",
                )
        return AudioFileProbe(
            sample_rate_hz=32000,
            channels=2,
            frame_count=round(self.source_duration_seconds * 32000),
            duration_seconds=self.source_duration_seconds,
            format_name="flac",
        )


def _constant_conditioning_analyzer(amplitude: int) -> _RecoveredVoiceAnalyzer:
    analyzer = _RecoveredVoiceAnalyzer(seconds=3.0)
    analyzer.analysis = CanonicalAudioAnalysis(
        probe=analyzer.analysis.probe,
        mono_pcm16=np.full(3 * 32000, amplitude, dtype=np.int16),
    )
    return analyzer


def _configure_music_materializer_fixture(
    fixture: SimpleNamespace,
    *,
    event_start: float = 1.2,
    event_end: float = 2.4,
    category: str = "non_diegetic_music",
    music_status: str = "present",
) -> SimpleNamespace:
    job_values = fixture.job.model_dump(mode="json", exclude={"request_fingerprint"})
    job_values["target_duration_seconds"] = 3.0
    job = _job(job_values)
    annotation_values = _presentation_annotation("offscreen_spoken").model_dump(
        mode="json"
    )
    event = annotation_values["audio_semantics"]["temporal_non_speech_events"][0]
    event.update(
        approximate_start_time=event_start,
        approximate_end_time=event_end,
        category=category,
        pattern="continuous",
        description="A sparse acoustic-guitar texture moves at a restrained pace.",
        source_grounding="audible_only",
    )
    annotation_values["audio_semantics"]["non_diegetic_music_status"] = music_status
    annotation_values["audio_semantics"]["non_diegetic_music"] = (
        "Sparse acoustic guitar continues softly."
        if music_status == "present"
        else None
    )
    annotation_values["h3_draft"]["shots"][0]["timeline_parts"] = [
        {"type": "prose", "text": "<Subject 1> remains in view."},
        {"type": "speech", "segment_id": "segment_1"},
        {"type": "audio_event", "event_id": "ae1"},
    ]
    annotation = MimoAVAnnotationDraft.model_validate(annotation_values)
    inventory_values = fixture.inventory.model_dump(
        mode="json", exclude={"inventory_fingerprint"}
    )
    inventory_values["jobs"] = [job.model_dump(mode="json")]
    if inventory_values["source_diarization_inventory_sha256"] is None:
        inventory_values.pop("source_diarization_inventory_sha256")
    inventory = _inventory(inventory_values)
    record = _record_fixture(
        Path(job.target_video_path).parent,
        annotation,
        job=job,
        inventory_fingerprint=inventory.inventory_fingerprint,
    )
    (fixture.mimo_root / "inventory.json").write_text(
        inventory.model_dump_json(), encoding="utf-8"
    )
    _write_models_jsonl(fixture.mimo_root / "records.jsonl", [record])
    fixture.job = job
    fixture.inventory = inventory
    fixture.record = record
    return fixture


def _as_asd_miss(job: MimoClipJob) -> MimoClipJob:
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["target_duration_seconds"] = 3.0
    values["segments"][0].update(
        current_entity_id=None,
        entity_occurrence_id=None,
        identity_scope="unresolved",
        direct_anchor_seconds=0.0,
        cluster_binding_status="unbound",
        direct_support_seconds_by_entity={},
    )
    return _job(values)


def _configure_recovery_materializer_fixture(
    fixture: SimpleNamespace,
    *,
    recover_entity_id: str,
) -> SimpleNamespace:
    samples = fixture.samples
    job_values = fixture.job.model_dump(mode="json", exclude={"request_fingerprint"})
    if recover_entity_id == "e2":
        image = Path(fixture.job.target_video_path).parent / "reference-e2.png"
        image.write_bytes(b"image-e2")
        reference = FinalVisualReference(
            image_id="image_2",
            image_index=2,
            kind="subject",
            image_path="selected/reference-e2.png",
            image_artifact_path=str(image),
            entity_id="e2",
            source_frame_index=1,
            scope="full",
            visible_region="whole",
            synthetic=False,
        )
        samples = [
            FinalH3SampleV2.model_validate(
                {
                    **item.model_dump(mode="python"),
                    "visual_references": [
                        *item.model_dump(mode="python")["visual_references"],
                        reference.model_dump(mode="python"),
                    ],
                }
            )
            for item in samples
        ]
        job_values["reference_images"].append(
            MimoReferenceImage(
                image_index=2,
                picture_label="<Picture 2>",
                source_image_index=2,
                source_image_id="image_2",
                source_image_label="<Image 2>",
                kind="subject",
                entity_id="e2",
                image_artifact_path=str(image),
                image_sha256=_file_sha256(image),
            ).model_dump(mode="json")
        )
        job_values["reference_selection"].update(
            original_picture_count=2,
            selected_picture_count=2,
            selected_source_image_indexes=[1, 2],
            selected_source_image_ids=["image_1", "image_2"],
        )
        job_values["reference_subjects"].append(
            RecaptionSubjectContract(
                subject_index=2,
                subject_label="<Subject 2>",
                kind="entity",
                entity_id="e2",
                source_picture_labels=["<Picture 2>"],
            ).model_dump(mode="json")
        )
    job = _as_asd_miss(_job(job_values))
    _write_models_jsonl(fixture.samples_path, samples)
    inventory_values = fixture.inventory.model_dump(
        mode="json", exclude={"inventory_fingerprint"}
    )
    inventory_values["source_h3_samples_sha256"] = _file_sha256(fixture.samples_path)
    inventory_values["jobs"] = [job.model_dump(mode="json")]
    if inventory_values["source_diarization_inventory_sha256"] is None:
        inventory_values.pop("source_diarization_inventory_sha256")
    inventory = _inventory(inventory_values)
    annotation_payload = _annotation(entity_id=recover_entity_id).model_dump(
        mode="json"
    )
    annotation_payload["segment_decisions"][0]["evidence_codes"] = [
        "speaker_visible_mouth_occluded",
        "voice_continuity",
    ]
    if recover_entity_id == "e2":
        annotation_payload["h3_draft"]["subject_definitions"].append(
            _subject_definition(
                "<Subject 2>",
                "a second person with a distinct visible appearance.",
            )
        )
        annotation_payload["h3_draft"]["visual_retention_analysis"].append(
            _retention(
                "<Subject 2>",
                "fully_preserved",
                "the person remains visible.",
            )
        )
    record = _record_fixture(
        Path(job.target_video_path).parent,
        MimoAVAnnotationDraft.model_validate(annotation_payload),
        job=job,
        inventory_fingerprint=inventory.inventory_fingerprint,
    )
    (fixture.mimo_root / "inventory.json").write_text(
        inventory.model_dump_json(), encoding="utf-8"
    )
    _write_models_jsonl(fixture.mimo_root / "records.jsonl", [record])
    fixture.job = job
    fixture.inventory = inventory
    fixture.record = record
    fixture.samples = samples
    return fixture


def _review_fixture(
    tmp_path: Path,
    *,
    source_sample_count: int = 1,
) -> SimpleNamespace:
    fixture = _materializer_fixture(
        tmp_path,
        source_sample_count=source_sample_count,
    )
    materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
    )
    raw = MimoRawResponse(
        clip_uid=fixture.job.clip_uid,
        request_fingerprint=fixture.job.request_fingerprint,
        raw_responses=["raw response must not enter review HTML"],
        diagnostics=[
            MimoCompletionDiagnostic(
                input_modality="target_video_with_embedded_audio",
                finish_reason="stop",
                usage=MimoUsage(
                    image_tokens=2,
                    video_tokens=8,
                    audio_tokens=3,
                    reasoning_tokens=0,
                ),
                http_attempt_count=1,
                warnings=["image_tokens_unavailable"],
            ).model_dump(mode="json")
        ],
    )
    raw_path = fixture.mimo_root / "raw_responses" / f"{fixture.job.clip_uid}.json"
    raw_path.parent.mkdir()
    raw_path.write_text(raw.model_dump_json(), encoding="utf-8")
    return fixture


def _shadow_records(fixture: SimpleNamespace) -> list[object]:
    return [
        mimo25_materializer.MimoH3ShadowRecord.model_validate(row)
        for row in (
            json.loads(line)
            for line in (fixture.output_root / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    ]


def _presentation_annotation(presentation: str) -> MimoAVAnnotationDraft:
    onscreen = presentation == "onscreen_spoken"
    payload = _annotation(entity_id="e1" if onscreen else None).model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision["binding_status"] = "visible_entity" if onscreen else "offscreen"
    decision["speech_presentation"] = presentation
    decision["entity_id"] = "e1" if onscreen else None
    decision["evidence_codes"] = (
        ["visible_lip_motion", "av_temporal_alignment"]
        if onscreen
        else ["no_visible_lip_motion"]
    )
    if presentation == "message_voice_over":
        decision["evidence_codes"].append("message_text_alignment")
        payload["h3_draft"]["summary"] = "A person silently handles a phone."
        payload["h3_draft"]["shots"][0]["timeline_parts"] = [
            _prose("<Subject 1> looks at the phone and types without visible speech."),
            _audio_event("ae1"),
            _speech("segment_1"),
        ]
    return MimoAVAnnotationDraft.model_validate(payload)


def _write_shadow_records(
    fixture: SimpleNamespace,
    records: list[object],
) -> None:
    _write_models_jsonl(fixture.output_root / "records.jsonl", records)
    summary_path = fixture.output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sample_count"] = len(records)
    summary["ready_count"] = sum(record.status == "ready" for record in records)
    summary["failed_count"] = sum(record.status == "failed" for record in records)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_materializer_preserves_exact_asr_and_segment(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    record = _record_fixture(
        tmp_path, _annotation(composition="same_speaker_nonlexical")
    )
    corrected, rendered, warnings = _materialize_sample(sample, job, record)
    assert len(corrected) == 1
    assert corrected[0].text == "Exact, text!"
    assert corrected[0].language == "English"
    assert corrected[0].start_time == 0.0
    assert corrected[0].end_time == 1.0
    assert "<d>[English] Exact, text!</d>" in rendered
    assert "A short repeated clink is audible." in rendered
    assert rendered.index("A short repeated clink is audible.") < rendered.index(
        "<d>[English] Exact, text!</d>"
    )
    assert "[[audio_event:" not in rendered
    assert "<Subject 1> (S1)" in rendered
    assert not warnings or all("words" in item for item in warnings)


def test_materializer_canonicalizes_first_shot_zero_to_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["start_time"] = 0
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    assert annotation.h3_draft.shots[0].start_time == 0
    _, expected_rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, _annotation()),
    )

    original = mimo25_materializer.materialize_h3_draft
    observed: dict[str, float | None] = {}

    def capture_start_time(draft: object, *args: object, **kwargs: object) -> object:
        observed["start_time"] = draft.shots[0].start_time  # type: ignore[attr-defined]
        return original(draft, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        mimo25_materializer,
        "materialize_h3_draft",
        capture_start_time,
    )
    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    assert observed["start_time"] is None
    assert rendered == expected_rendered


def test_materializer_audio_facts_use_segment_level_delivery(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    record = _record_fixture(tmp_path, _annotation())
    facts = mimo25_materializer._audio_facts(
        sample=sample,
        corrected=sample.speech_segments,
        record=record,
        contract=mimo25_materializer.build_reference_contract(
            sample,
            mimo25_materializer._variant(sample),
        ),
    )

    assert facts.speech[0].delivery == "calm and clear"


def _repeated_speech_inputs(
    tmp_path: Path,
    *,
    second_shot: bool,
) -> tuple[FinalH3SampleV2, MimoClipJob, MimoRecord]:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    annotation = _annotation()
    sample_values = sample.model_dump(mode="python")
    job_values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    annotation_values = annotation.model_dump(mode="json")
    sample_segments = []
    job_segments = []
    decisions = []
    for index in range(4):
        segment_id = f"segment_{index + 1}"
        start = float(index)
        end = start + 0.75
        sample_segment = sample.speech_segments[0].model_dump(mode="python")
        sample_segment.update(
            segment_id=segment_id,
            source_start_sample=index * 32000,
            source_end_sample=index * 32000 + 24000,
            start_time=start,
            end_time=end,
            text=f"Exact line {index + 1}.",
        )
        source_segment = job.segments[0].model_dump(mode="json")
        source_segment.update(
            segment_id=segment_id,
            source_start_sample=index * 32000,
            source_end_sample=index * 32000 + 24000,
            start_time=start,
            end_time=end,
            asr_text=f"Exact line {index + 1}.",
        )
        decision = dict(annotation.segment_decisions[0].model_dump(mode="json"))
        decision["segment_id"] = segment_id
        sample_segments.append(sample_segment)
        job_segments.append(source_segment)
        decisions.append(decision)
    sample_values["speech_segments"] = sample_segments
    job_values["segments"] = job_segments
    job_values["target_duration_seconds"] = 4.0
    annotation_values["segment_decisions"] = decisions
    first_parts = [_prose("The speaker remains visible."), _audio_event("ae1")]
    if second_shot:
        annotation_values["h3_draft"]["shots"] = [
            {
                "shot_index": 1,
                "start_time": None,
                "timeline_parts": [*first_parts, _speech("segment_1"), _speech("segment_2")],
            },
            {
                "shot_index": 2,
                "start_time": 2.0,
                "timeline_parts": [
                    _prose("A real hard cut changes the view."),
                    _speech("segment_3"),
                    _speech("segment_4"),
                ],
            },
        ]
    else:
        annotation_values["h3_draft"]["shots"][0]["timeline_parts"] = [
            *first_parts,
            *(_speech(f"segment_{index}") for index in range(1, 5)),
        ]
    typed_sample = FinalH3SampleV2.model_validate(sample_values)
    typed_job = _job(job_values)
    typed_annotation = MimoAVAnnotationDraft.model_validate(annotation_values)
    return (
        typed_sample,
        typed_job,
        _record_fixture(tmp_path, typed_annotation, job=typed_job),
    )


def test_same_speaker_group_can_move_offscreen_without_entity_propagation(
    tmp_path: Path,
) -> None:
    sample, job, record = _repeated_speech_inputs(tmp_path, second_shot=False)
    assert record.annotation is not None
    payload = record.annotation.model_dump(mode="json")
    second = payload["segment_decisions"][1]
    second.update(
        binding_status="offscreen",
        speech_presentation="offscreen_spoken",
        entity_id=None,
        evidence_codes=["offscreen_audio", "voice_continuity"],
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    intervals = {
        segment.segment_id: (segment.start_time, segment.end_time)
        for segment in job.segments
    }

    assert not _validate(
        annotation,
        segment_ids=[segment.segment_id for segment in job.segments],
        segment_intervals=intervals,
        transcribed_segment_ids=[segment.segment_id for segment in job.segments],
        authoritative_transcripts=[segment.asr_text or "" for segment in job.segments],
        target_duration_seconds=job.target_duration_seconds,
    )
    corrected, rendered, _ = _materialize_sample(
        sample,
        job,
        _record_fixture(tmp_path, annotation, job=job),
    )

    assert corrected[0].speaker_cluster_id == corrected[1].speaker_cluster_id == "g1"
    assert corrected[0].entity_id == "e1"
    assert corrected[1].entity_id is None
    assert "<Subject 1> (S1)" in rendered
    assert "(S1), speaking offscreen: <d>[English] Exact line 2.</d>" in rendered


def test_visible_listener_prose_does_not_reattach_continuing_speaker_entity(
    tmp_path: Path,
) -> None:
    sample, job, record = _repeated_speech_inputs(tmp_path, second_shot=False)
    assert record.annotation is not None
    payload = record.annotation.model_dump(mode="json")
    payload["h3_draft"]["shots"][0]["timeline_parts"][0] = _prose(
        "A visible listener keeps a closed mouth while the established voice continues."
    )
    second = payload["segment_decisions"][1]
    second.update(
        binding_status="no_reliable_entity",
        speech_presentation="uncertain",
        entity_id=None,
        evidence_codes=["voice_continuity", "insufficient_evidence"],
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    corrected, rendered, _ = _materialize_sample(
        sample,
        job,
        _record_fixture(tmp_path, annotation, job=job),
    )

    assert corrected[0].speaker_cluster_id == corrected[1].speaker_cluster_id == "g1"
    assert corrected[1].entity_id is None
    assert "(S1), with speech presentation uncertain" in rendered
    assert "<Subject 1> (S1) says, <d>[English] Exact line 2.</d>" not in rendered


def test_materializer_cites_voice_audio_once_per_speaker_per_shot(
    tmp_path: Path,
) -> None:
    sample, job, record = _repeated_speech_inputs(tmp_path, second_shot=False)
    _, rendered, _ = _materialize_sample(sample, job, record)
    assert rendered.count("referenced from <Audio 1>") == 1
    assert rendered.count("<d>[English] Exact line") == 4


def test_materializer_cites_voice_audio_again_in_new_shot(tmp_path: Path) -> None:
    sample, job, record = _repeated_speech_inputs(tmp_path, second_shot=True)
    _, rendered, _ = _materialize_sample(sample, job, record)
    assert rendered.count("referenced from <Audio 1>") == 2
    assert rendered.count("<d>[English] Exact line") == 4


def test_materializer_voice_profile_and_null_fallback(tmp_path: Path) -> None:
    _, profiled, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, _annotation()),
    )
    assert "featuring clear mid-register timbre with measured cadence" in profiled

    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = None
    _, fallback, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, MimoAVAnnotationDraft.model_validate(payload)),
    )
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)." in fallback


def test_materializer_preserves_official_ref2va_format(tmp_path: Path) -> None:
    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, _annotation()),
    )
    sections = [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]

    assert [rendered.index(f"{section}:\n") for section in sections] == sorted(
        rendered.index(f"{section}:\n") for section in sections
    )
    assert "[[segment:" not in rendered
    assert "[[audio_event:" not in rendered
    assert "[[" not in rendered
    assert "<Subject 1> (S1)" in rendered
    assert "<d>[English] Exact, text!</d>" in rendered


def test_materializer_renders_typed_subject_and_retention_as_official_h3(
    tmp_path: Path,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["h3_draft"]["subject_definitions"][0] = _subject_definition(
        "<Subject 1>",
        "the person with dark wavy hair and a blue jacket.",
    )
    payload["h3_draft"]["visual_retention_analysis"][0] = _retention(
        "<Subject 1>",
        "fully_preserved",
        "the referenced visual appearance remains intact.",
    )

    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(
            tmp_path,
            MimoAVAnnotationDraft.model_validate(payload),
        ),
    )

    assert (
        "<Subject 1> is the person with dark wavy hair and a blue jacket, shown in "
        "<Picture 1>."
        in rendered
    )
    assert (
        "<Subject 1>: fully_preserved - the referenced visual appearance remains "
        "intact."
    ) in rendered
    assert "Subject 1 is" not in rendered
    assert "is fully preserved" not in rendered


def test_materializer_owns_exact_multi_picture_subject_provenance(
    tmp_path: Path,
) -> None:
    job = _multi_picture_job_fixture(tmp_path)
    draft = _annotation().h3_draft.subject_definitions[0]

    rendered = _render_subject_definition(draft, job.reference_subjects[0])

    assert rendered.startswith("<Subject 1> is ")
    assert rendered.count("<Picture 1>") == 1
    assert rendered.count("<Picture 2>") == 1
    assert rendered.endswith("shown in <Picture 1> and <Picture 2>.")


@pytest.mark.parametrize(
    ("status", "description", "expected"),
    [
        (
            "present",
            "A low room hum and a light clink are audible.",
            "A low room hum and a light clink are audible.",
        ),
        ("absent", None, "N/A"),
    ],
)
def test_materializer_renders_soundscape_status_without_ungrounded_prose(
    tmp_path: Path,
    status: str,
    description: str | None,
    expected: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    semantics = payload["audio_semantics"]
    semantics["overall_soundscape_status"] = status
    semantics["overall_soundscape"] = description
    if status == "absent":
        semantics["temporal_non_speech_events"] = []
        payload["h3_draft"]["shots"][0]["timeline_parts"] = [
            item
            for item in payload["h3_draft"]["shots"][0]["timeline_parts"]
            if item["type"] != "audio_event"
        ]
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    assert f"overall_soundscape:\n{expected}" in rendered
    assert "No additional soundscape is established" not in rendered


def test_materializer_rejects_unknown_soundscape_as_confirmed_silence(
    tmp_path: Path,
) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["temporal_non_speech_events"] = []
    payload["audio_semantics"]["overall_soundscape_status"] = "unknown"
    payload["audio_semantics"]["overall_soundscape"] = None
    payload["h3_draft"]["shots"][0]["timeline_parts"] = [
        item
        for item in payload["h3_draft"]["shots"][0]["timeline_parts"]
        if item["type"] != "audio_event"
    ]
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    with pytest.raises(
        ValueError,
        match="unknown MiMo soundscape cannot be materialized as confirmed silence",
    ):
        _materialize_sample(
            _sample(tmp_path),
            _job_fixture(tmp_path),
            _record_fixture(tmp_path, annotation),
        )


@pytest.mark.parametrize("status", ["absent", "unknown"])
def test_audible_non_speech_event_requires_present_soundscape(status: str) -> None:
    payload = _annotation().model_dump(mode="json")
    payload["audio_semantics"]["overall_soundscape_status"] = status
    payload["audio_semantics"]["overall_soundscape"] = None

    with pytest.raises(
        ValidationError,
        match="audible non-speech event requires present global soundscape semantics",
    ):
        MimoAVAnnotationDraft.model_validate(payload)


def test_indoor_dialogue_with_low_room_ambience_renders_soundscape(
    tmp_path: Path,
) -> None:
    payload = _annotation().model_dump(mode="json")
    event = payload["audio_semantics"]["temporal_non_speech_events"][0]
    event.update(
        {
            "category": "environmental",
            "pattern": "continuous",
            "description": "A faint steady indoor room ambience is audible.",
            "source_grounding": "audible_only",
        }
    )
    payload["audio_semantics"]["overall_soundscape"] = (
        "A faint steady indoor room ambience sits beneath the dialogue."
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    assert (
        "overall_soundscape:\n"
        "A faint steady indoor room ambience sits beneath the dialogue."
    ) in rendered
    assert "overall_soundscape:\nN/A" not in rendered


def test_non_diegetic_music_does_not_substitute_for_soundscape() -> None:
    payload = _annotation().model_dump(mode="json")
    event = payload["audio_semantics"]["temporal_non_speech_events"][0]
    event["category"] = "non_diegetic_music"
    event["description"] = "A faint audience-only string score is audible."
    payload["audio_semantics"]["overall_soundscape_status"] = "absent"
    payload["audio_semantics"]["overall_soundscape"] = None
    payload["audio_semantics"]["non_diegetic_music_status"] = "present"
    payload["audio_semantics"]["non_diegetic_music"] = (
        "A faint audience-only string score continues underneath."
    )

    annotation = MimoAVAnnotationDraft.model_validate(payload)

    assert annotation.audio_semantics.overall_soundscape_status == "absent"
    assert annotation.audio_semantics.non_diegetic_music_status == "present"


@pytest.mark.parametrize(
    ("status", "description", "expected"),
    [
        (
            "present",
            "Faint non-diegetic strings continue underneath.",
            "Faint non-diegetic strings continue underneath.",
        ),
        ("absent", None, "N/A"),
        ("unknown", None, "N/A"),
    ],
)
def test_materializer_renders_music_status_without_ungrounded_prose(
    tmp_path: Path,
    status: str,
    description: str | None,
    expected: str,
) -> None:
    payload = _annotation().model_dump(mode="json")
    semantics = payload["audio_semantics"]
    semantics["non_diegetic_music_status"] = status
    semantics["non_diegetic_music"] = description
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    assert f"non_diegetic_music:\n{expected}" in rendered
    assert "Non-diegetic music is not established" not in rendered


@pytest.mark.parametrize(
    ("presentation", "expected_phrase"),
    [
        ("offscreen_spoken", "speaking offscreen"),
        ("voice_over", "as a voice-over rather than visible speech"),
        ("message_voice_over", "as a message voice-over rather than visible speech"),
        (
            "device_playback",
            "heard through an in-scene device rather than visible speech",
        ),
        ("uncertain", "with speech presentation uncertain"),
    ],
)
def test_materializer_renders_non_onscreen_speech_without_visible_subject(
    tmp_path: Path,
    presentation: str,
    expected_phrase: str,
) -> None:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    annotation = _presentation_annotation(presentation)
    corrected, rendered, _ = _materialize_sample(
        sample, job, _record_fixture(tmp_path, annotation)
    )
    segment = corrected[0]
    assert segment.entity_id is None
    assert segment.entity_occurrence_id is None
    assert expected_phrase in rendered
    assert "<Subject 1> (S1)" not in rendered
    assert " says," not in rendered
    assert "<d>[English] Exact, text!</d>" in rendered
    assert (
        segment.start_time,
        segment.end_time,
        segment.source_start_sample,
        segment.source_end_sample,
        segment.source_sample_rate_hz,
        segment.language,
    ) == (0.0, 1.0, 0, 32000, 32000, "English")


def test_materializer_preserves_existing_onscreen_speech_clause(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    payload = sample.model_dump(mode="python")
    payload.update(
        sample_id="clip-1/canonical",
        pair_id="canonical/clip-1",
        pair_type="canonical",
        subject_voices=[],
    )
    canonical = FinalH3SampleV2.model_validate(payload)
    _, rendered, _ = _materialize_sample(
        canonical,
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, _presentation_annotation("onscreen_spoken")),
    )
    assert ("<Subject 1> (S1) says, <d>[English] Exact, text!</d>") in rendered


def test_materializer_keeps_onscreen_speech_with_unknown_entity_unbound(
    tmp_path: Path,
) -> None:
    payload = _annotation(entity_id=None).model_dump(mode="json")
    decision = payload["segment_decisions"][0]
    decision.update(
        binding_status="no_reliable_entity",
        speech_presentation="onscreen_spoken",
        evidence_codes=["visible_lip_motion"],
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    corrected, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )
    assert corrected[0].entity_id is None
    assert "(S1) says, <d>[English] Exact, text!</d>" in rendered
    assert "<Subject 1> (S1)" not in rendered


def test_message_voice_over_draft_keeps_phone_action_without_visible_speech(
    tmp_path: Path,
) -> None:
    annotation = _presentation_annotation("message_voice_over")
    assert (
        "looks at the phone and types"
        in annotation.h3_draft.shots[0].timeline_parts[0].text
    )
    corrected, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )
    assert corrected[0].entity_id is None
    assert "message voice-over" in rendered
    assert "<Subject 1> (S1) says" not in rendered


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


def test_observed_door_close_renders_once_without_creating_audio_reference(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    sample_payload = sample.model_dump(mode="python")
    sample_payload.update(
        sample_id="clip-1/canonical",
        pair_id="canonical/clip-1",
        pair_type="canonical",
        subject_voices=[],
    )
    canonical = FinalH3SampleV2.model_validate(sample_payload)
    payload = _annotation().model_dump(mode="json")
    event = payload["audio_semantics"]["temporal_non_speech_events"][0]
    event.update(
        approximate_start_time=0.45,
        approximate_end_time=0.55,
        category="physical",
        pattern="single",
        description="A door closes with a brief solid thud.",
        source_grounding="audiovisually_grounded",
    )
    payload["audio_semantics"]["overall_soundscape"] = (
        "Low indoor ambience is punctuated by the solid thud of a closing door."
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    observed_event = annotation.audio_semantics.temporal_non_speech_events[0]
    assert observed_event.category == "physical"
    assert observed_event.pattern == "single"
    assert observed_event.approximate_start_time == 0.45
    assert observed_event.approximate_end_time == 0.55

    _, rendered, _ = _materialize_sample(
        canonical,
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    assert rendered.count("A door closes with a brief solid thud.") == 1
    assert "overall_soundscape:\nLow indoor ambience is punctuated" in rendered
    assert "ae1" not in rendered
    assert "0.45" not in rendered
    assert "<audio_event>" not in rendered
    assert "<sound_event>" not in rendered
    assert "<Audio " not in rendered
    assert rendered.index("A door closes with a brief solid thud.") < rendered.index(
        "<d>[English] Exact, text!</d>"
    )


def test_multiple_physical_events_keep_chronological_materialized_order(
    tmp_path: Path,
) -> None:
    payload = _two_audio_event_annotation().model_dump(mode="json")
    first, second = payload["audio_semantics"]["temporal_non_speech_events"]
    first.update(
        category="physical",
        description="A door latch clicks and the door closes.",
    )
    second.update(
        category="physical",
        description="Two quick footsteps cross the floor.",
    )
    payload["audio_semantics"]["overall_soundscape"] = (
        "Quiet room ambience includes a closing door and quick footsteps."
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    _, rendered, _ = _materialize_sample(
        _sample(tmp_path),
        _job_fixture(tmp_path),
        _record_fixture(tmp_path, annotation),
    )

    first_text = "A door latch clicks and the door closes."
    second_text = "Two quick footsteps cross the floor."
    assert rendered.count(first_text) == rendered.count(second_text) == 1
    assert rendered.index(first_text) < rendered.index(second_text)
    assert "ae1" not in rendered and "ae2" not in rendered


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
    assert "(S1), speaking offscreen: <d>[English] Exact, text!</d>" in rendered
    assert "<Subject 1> (S1)" not in rendered
    assert "segment_1:acoustic_refinement_unresolved" in warnings


def test_materializer_validates_bound_speech_with_long_voice_profile(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    job = _job_fixture(tmp_path)
    payload = _annotation().model_dump(mode="json")
    payload["speaker_voice_profiles"][0]["voice_characteristics"] = (
        "a controlled mid-register delivery with softly rounded consonants, "
        "measured pauses, restrained energy, and a steady unhurried cadence "
        "that remains consistent through the utterance "
    )
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    corrected, rendered, _ = _materialize_sample(
        sample,
        job,
        _record_fixture(tmp_path, annotation, job=job),
    )

    assert corrected[0].entity_id == "e1"
    dialogue = "<d>[English] Exact, text!</d>"
    dialogue_index = rendered.index(dialogue)
    source_index = rendered.rindex("<Subject 1> (S1)", 0, dialogue_index)
    assert dialogue_index - source_index > 180


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
    payload["warnings"] = [{"code": "possible_asr_conflict", "segment_id": "segment_1"}]
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
    assert summary.full_audio_reuse_count == 0
    assert summary.music_reference_count == 0


def test_materializer_isolates_one_final_contract_failure_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _materializer_fixture(tmp_path, source_sample_count=2)
    original_validate = mimo25_materializer.validate_h3_response

    def validate_with_one_bad_sample(response: object, request: object) -> object:
        if request.sample.sample_id == "clip-1/in_pair":  # type: ignore[attr-defined]
            return (
                [
                    ValidationIssue(
                        "locked_dialogue_source_mismatch",
                        "detailed_description",
                        "segment_0001 requires <Subject 1> (S1)",
                    )
                ],
                [],
            )
        return original_validate(response, request)  # type: ignore[arg-type]

    monkeypatch.setattr(
        mimo25_materializer,
        "validate_h3_response",
        validate_with_one_bad_sample,
    )
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_full_audio_reuse=True,
        audio_backend=_ConditioningAudioBackend(source_duration_seconds=1.0),
    )
    records = _shadow_records(fixture)

    assert {item.sample_id: item.status for item in records} == {
        "clip-1/audio_reuse": "ready",
        "clip-1/canonical": "ready",
        "clip-1/in_pair": "failed",
    }
    failed = next(item for item in records if item.status == "failed")
    assert failed.rendered_h3_prompt is None
    assert failed.corrected_speech_segments == []
    assert failed.effective_subject_voices == []
    assert failed.audio_references == []
    assert failed.recovered_voice_references == []
    assert failed.warnings == []
    failure = json.loads(failed.failure_reason)
    assert failure == {
        "category": "materialization_contract_failed",
        "issues": [
            {
                "code": "locked_dialogue_source_mismatch",
                "field": "detailed_description",
                "message": "segment_0001 requires <Subject 1> (S1)",
            }
        ],
    }
    assert summary.sample_count == 3
    assert summary.ready_count == 2
    assert summary.failed_count == 1
    assert summary.schema_version == "r2v.h3.mimo25_h3_shadow_summary.12"
    assert summary.materialization_failure_count == 1
    assert summary.materialization_failure_code_counts == {
        "locked_dialogue_source_mismatch": 1
    }
    assert {item.schema_version for item in records} == {
        "r2v.h3.mimo25_h3_shadow.11"
    }
    assert {item.materializer_version for item in records} == {
        "h3_mimo25_materializer_v13"
    }


def test_materializer_does_not_isolate_authoritative_asr_drift(tmp_path: Path) -> None:
    fixture = _materializer_fixture(tmp_path)
    sample_values = fixture.samples[0].model_dump(mode="python")
    sample_values["speech_segments"][0]["text"] = "Mutated source text."
    changed_sample = FinalH3SampleV2.model_validate(sample_values)
    _write_models_jsonl(fixture.samples_path, [changed_sample])

    inventory_values = fixture.inventory.model_dump(
        mode="json", exclude={"inventory_fingerprint"}
    )
    inventory_values["source_h3_samples_sha256"] = _file_sha256(fixture.samples_path)
    if inventory_values["source_diarization_inventory_sha256"] is None:
        inventory_values.pop("source_diarization_inventory_sha256")
    inventory = _inventory(inventory_values)
    changed_record = _replace_record(
        fixture.record,
        inventory_fingerprint=inventory.inventory_fingerprint,
    )
    (fixture.mimo_root / "inventory.json").write_text(
        inventory.model_dump_json(), encoding="utf-8"
    )
    _write_models_jsonl(fixture.mimo_root / "records.jsonl", [changed_record])

    with pytest.raises(
        ValueError,
        match="source H3 speech differs from authoritative Qwen3-ASR",
    ):
        materialize_mimo25_h3_shadow(
            mimo_root=fixture.mimo_root,
            source_h3_root=fixture.source_h3,
            output_root=fixture.output_root,
        )
    assert not fixture.output_root.exists()


def test_materializer_audio_variant_flags_default_false_and_are_explicit() -> None:
    defaults = _materializer_parser().parse_args(
        ["--audio-production-root", "/tmp/production"]
    )
    enabled = _materializer_parser().parse_args(
        [
            "--audio-production-root",
            "/tmp/production",
            "--enable-full-audio-reuse",
            "--enable-music-reference",
        ]
    )
    assert defaults.enable_full_audio_reuse is False
    assert defaults.enable_music_reference is False
    assert enabled.enable_full_audio_reuse is True
    assert enabled.enable_music_reference is True


def test_full_audio_reuse_is_opt_in_and_uses_canonical_flac(tmp_path: Path) -> None:
    fixture = _materializer_fixture(tmp_path)
    source_paths = [
        fixture.samples_path,
        fixture.mimo_root / "records.jsonl",
        Path(fixture.job.target_full_audio_path),
    ]
    before = {path: path.read_bytes() for path in source_paths}
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_full_audio_reuse=True,
        audio_backend=_ConditioningAudioBackend(source_duration_seconds=1.0),
    )
    records = _shadow_records(fixture)
    reuse = next(item for item in records if item.conditioning_variant == "full_audio_reuse")
    assert len(records) == 2
    assert reuse.sample_id == "clip-1/audio_reuse"
    assert reuse.pair_type == "canonical"
    assert reuse.effective_subject_voices == []
    assert len(reuse.audio_references) == 1
    audio = reuse.audio_references[0]
    assert audio.role == "full_audio_reuse"
    assert audio.audio_path == fixture.job.target_full_audio_path
    assert audio.audio_sha256 == fixture.job.target_full_audio_sha256
    assert audio.speaker_id is None
    assert "<Audio 1>: fully_copy" in (reuse.rendered_h3_prompt or "")
    assert "[reference generation + audio reuse]" in (reuse.rendered_h3_prompt or "")
    assert summary.full_audio_reuse_count == 1
    assert summary.music_reference_count == 0
    assert all(path.read_bytes() == before[path] for path in source_paths)


def test_clean_non_diegetic_music_derives_one_real_reference(tmp_path: Path) -> None:
    fixture = _configure_music_materializer_fixture(_materializer_fixture(tmp_path))
    source_paths = [
        fixture.samples_path,
        fixture.mimo_root / "records.jsonl",
        Path(fixture.job.target_full_audio_path),
    ]
    before = {path: path.read_bytes() for path in source_paths}
    media = _ConditioningAudioBackend(source_duration_seconds=3.0)
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_music_reference=True,
        audio_backend=media,
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(seconds=3.0),
    )
    music = next(
        item
        for item in _shadow_records(fixture)
        if item.conditioning_variant == "music_reference"
    )
    audio = music.audio_references[0]
    assert music.sample_id == "clip-1/music_reference"
    assert audio.role == "music_reference"
    assert audio.source_event_id == "ae1"
    assert audio.source_start_sample == round(1.2 * 32000)
    assert audio.source_end_sample == round(2.4 * 32000)
    assert audio.interval_provenance == (
        "mimo_approximate_event_times_rounded_to_32000_v1"
    )
    assert audio.subject_index is None
    assert audio.speaker_id is None
    assert Path(audio.audio_path).is_file()
    assert media.probe_audio_file(Path(audio.audio_path)).sample_rate_hz == 32000
    assert media.probe_audio_file(Path(audio.audio_path)).channels == 2
    rendered = music.rendered_h3_prompt or ""
    assert "<Audio 1> is a music-style reference" in rendered
    assert "<Audio 1>: reference" in rendered
    assert "without directly reusing the source signal" in rendered
    assert "[[" not in rendered
    assert summary.music_reference_count == 1
    assert summary.full_audio_reuse_count == 0
    assert len(media.extractions) == 1
    assert all(path.read_bytes() == before[path] for path in source_paths)


@pytest.mark.parametrize(
    ("event_start", "event_end", "category", "music_status", "reason"),
    [
        (0.5, 1.8, "non_diegetic_music", "present", "music_reference_overlaps_speech"),
        (1.2, 2.4, "diegetic_music", "absent", None),
        (1.2, 2.4, "physical", "unknown", None),
        (1.2, 1.8, "non_diegetic_music", "present", "music_reference_too_short"),
        (
            1.2,
            3.1,
            "non_diegetic_music",
            "present",
            "music_reference_invalid_sample_range",
        ),
    ],
)
def test_music_reference_rejects_ineligible_events_without_publishing(
    tmp_path: Path,
    event_start: float,
    event_end: float,
    category: str,
    music_status: str,
    reason: str | None,
) -> None:
    fixture = _configure_music_materializer_fixture(
        _materializer_fixture(tmp_path),
        event_start=event_start,
        event_end=event_end,
        category=category,
        music_status=music_status,
    )
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_music_reference=True,
        audio_backend=_ConditioningAudioBackend(source_duration_seconds=3.0),
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(seconds=3.0),
    )
    assert summary.music_reference_count == 0
    assert all(
        item.conditioning_variant != "music_reference"
        for item in _shadow_records(fixture)
    )
    if reason is not None:
        assert summary.music_reference_rejection_reason_counts[reason] == 1
    else:
        assert summary.music_reference_rejection_reason_counts == {}


@pytest.mark.parametrize(
    ("amplitude", "reason"),
    [
        (0, "music_reference_rms_unusable"),
        (32767, "music_reference_clipping_excessive"),
    ],
)
def test_music_reference_rejects_unusable_audio_quality(
    tmp_path: Path,
    amplitude: int,
    reason: str,
) -> None:
    fixture = _configure_music_materializer_fixture(_materializer_fixture(tmp_path))
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_music_reference=True,
        audio_backend=_ConditioningAudioBackend(source_duration_seconds=3.0),
        recovered_voice_analyzer=_constant_conditioning_analyzer(amplitude),
    )
    assert summary.music_reference_count == 0
    assert summary.music_reference_rejection_reason_counts[reason] == 1


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
        (2, ["clip-1/canonical"]),
        (1, ["clip-1/canonical", "clip-1/removed-variant"]),
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


def test_mimo_recovery_uses_asd_independent_exact_canonical_samples(
    tmp_path: Path,
) -> None:
    job = _as_asd_miss(_job_fixture(tmp_path))
    record = _record_fixture(tmp_path, _annotation(), job=job)
    analyzer = _RecoveredVoiceAnalyzer()
    media = _RecoveredVoiceMediaBackend()

    recovered, temporary_voices, published_voices = recover_mimo_target_voices(
        job=job,
        record=record,
        subject_index_by_entity={"e1": 1},
        existing_entity_ids=set(),
        reference_capacity=3,
        temporary_root=tmp_path / "temporary",
        final_root=tmp_path / "final",
        audio_backend=media,
        analyzer=analyzer,
    )

    assert len(recovered) == 1
    assert recovered[0].status == "selected"
    assert recovered[0].reason_codes == []
    assert recovered[0].quality_policy_version == (
        "h3_mimo25_recovered_voice_quality_v1"
    )
    assert recovered[0].source_start_sample == 0
    assert recovered[0].source_end_sample == 32000
    assert len(temporary_voices) == len(published_voices) == 1
    extraction = media.extractions[0]
    assert extraction["source_start_sample"] == 0
    assert extraction["source_end_sample"] == 32000
    assert extraction["sample_rate_hz"] == 32000
    assert extraction["channels"] == 2
    assert extraction["source_audio_path"] == Path(job.target_full_audio_path)
    assert "lr_asd" not in json.dumps(recovered[0].model_dump(mode="json"))


def test_mimo_recovery_preserves_existing_entity_voice(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    analyzer = _RecoveredVoiceAnalyzer()
    recovered, temporary, published = recover_mimo_target_voices(
        job=job,
        record=_record_fixture(tmp_path, _annotation(), job=job),
        subject_index_by_entity={"e1": 1},
        existing_entity_ids={"e1"},
        reference_capacity=2,
        temporary_root=tmp_path / "temporary",
        final_root=tmp_path / "final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=analyzer,
    )
    assert recovered == temporary == published == []
    assert analyzer.calls == []


@pytest.mark.parametrize(
    ("annotation", "expected_reason"),
    [
        (_presentation_annotation("offscreen_spoken"), None),
        (
            MimoAVAnnotationDraft.model_validate(
                {
                    **_annotation().model_dump(mode="json"),
                    "segment_decisions": [
                        {
                            **_annotation().model_dump(mode="json")[
                                "segment_decisions"
                            ][0],
                            "resolution": "uncertain",
                        }
                    ],
                }
            ),
            "unresolved",
        ),
        (
            _annotation(
                composition="overlapping_secondary_speech",
                resolution="needs_acoustic_refinement",
            ),
            "unresolved",
        ),
    ],
)
def test_mimo_recovery_rejects_unsafe_speaker_semantics(
    tmp_path: Path,
    annotation: MimoAVAnnotationDraft,
    expected_reason: str | None,
) -> None:
    job = _as_asd_miss(_job_fixture(tmp_path))
    analyzer = _RecoveredVoiceAnalyzer()
    recovered, _, voices = recover_mimo_target_voices(
        job=job,
        record=_record_fixture(tmp_path, annotation, job=job),
        subject_index_by_entity={"e1": 1},
        existing_entity_ids=set(),
        reference_capacity=3,
        temporary_root=tmp_path / "temporary",
        final_root=tmp_path / "final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=analyzer,
    )
    assert voices == []
    if expected_reason is None:
        assert recovered == []
        assert analyzer.calls == []
    else:
        assert expected_reason in recovered[0].reason_codes


def test_938_offscreen_group_never_recovers_e4_voice(tmp_path: Path) -> None:
    job = _as_asd_miss(_job_fixture(tmp_path))
    analyzer = _RecoveredVoiceAnalyzer()
    recovered, _, voices = recover_mimo_target_voices(
        job=job,
        record=_record_fixture(
            tmp_path,
            _presentation_annotation("offscreen_spoken"),
            job=job,
        ),
        subject_index_by_entity={"e1": 1, "e4": 2},
        existing_entity_ids=set(),
        reference_capacity=3,
        temporary_root=tmp_path / "temporary",
        final_root=tmp_path / "final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=analyzer,
    )
    assert recovered == []
    assert voices == []
    assert analyzer.calls == []


def test_mimo_recovery_acoustic_and_reference_limit_rejections(
    tmp_path: Path,
) -> None:
    job = _as_asd_miss(_job_fixture(tmp_path))
    record = _record_fixture(tmp_path, _annotation(), job=job)
    quiet, _, quiet_voices = recover_mimo_target_voices(
        job=job,
        record=record,
        subject_index_by_entity={"e1": 1},
        existing_entity_ids=set(),
        reference_capacity=3,
        temporary_root=tmp_path / "quiet-temporary",
        final_root=tmp_path / "quiet-final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=_RecoveredVoiceAnalyzer(speech_amplitude=0),
    )
    assert "rms_too_low" in quiet[0].reason_codes
    assert quiet_voices == []

    limited, _, limited_voices = recover_mimo_target_voices(
        job=job,
        record=record,
        subject_index_by_entity={"e1": 1},
        existing_entity_ids=set(),
        reference_capacity=0,
        temporary_root=tmp_path / "limited-temporary",
        final_root=tmp_path / "limited-final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=_RecoveredVoiceAnalyzer(),
    )
    assert limited[0].reason_codes == ["reference_limit"]
    assert limited_voices == []


def test_canonical_only_asd_miss_derives_real_in_pair_without_mutating_source(
    tmp_path: Path,
) -> None:
    fixture = _configure_recovery_materializer_fixture(
        _materializer_fixture(tmp_path), recover_entity_id="e1"
    )
    before = {
        path: path.read_bytes()
        for path in (
            fixture.samples_path,
            fixture.mimo_root / "records.jsonl",
            Path(fixture.job.target_full_audio_path),
        )
    }
    media = _RecoveredVoiceMediaBackend()
    summary = materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        audio_backend=media,
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(),
    )
    records = _shadow_records(fixture)
    canonical = next(item for item in records if item.pair_type == "canonical")
    derived = next(
        item for item in records if item.derived_from_pair_type == "canonical"
    )

    assert canonical.effective_subject_voices == []
    assert canonical.audio_references == []
    assert derived.pair_type == "in_pair"
    assert len(derived.effective_subject_voices) == 1
    assert derived.audio_references[0].source_type == "mimo_recovered_target_voice"
    assert derived.audio_references[0].source_segment_id == "segment_1"
    assert (
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S1), "
        "featuring clear mid-register timbre with measured cadence."
    ) in (
        derived.rendered_h3_prompt or ""
    )
    assert (
        "using the clear mid-register timbre with measured cadence voice "
        "referenced from <Audio 1>"
    ) in (
        derived.rendered_h3_prompt or ""
    )
    assert "mimo_recovered" not in (derived.rendered_h3_prompt or "")
    assert summary.derived_in_pair_count == 1
    assert summary.recovered_voice_accepted_count == 1
    assert summary.existing_target_voice_reference_count == 0
    assert all(path.read_bytes() == content for path, content in before.items())


def test_existing_in_pair_is_enriched_in_subject_order_and_cross_is_unchanged(
    tmp_path: Path,
) -> None:
    fixture = _configure_recovery_materializer_fixture(
        _materializer_fixture(tmp_path, source_sample_count=2),
        recover_entity_id="e2",
    )
    materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        audio_backend=_RecoveredVoiceMediaBackend(),
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(),
    )
    records = _shadow_records(fixture)
    in_pair = next(item for item in records if item.pair_type == "in_pair")
    canonical = next(item for item in records if item.pair_type == "canonical")

    assert [item.entity_id for item in in_pair.effective_subject_voices] == ["e1", "e2"]
    assert [item.source_type for item in in_pair.audio_references] == [
        "existing_target_voice",
        "mimo_recovered_target_voice",
    ]
    assert (
        "<Audio 2> is the voice-timbre reference for <Subject 2> (S1), "
        "featuring clear mid-register timbre with measured cadence."
    ) in (
        in_pair.rendered_h3_prompt or ""
    )
    assert canonical.effective_subject_voices == []
    assert not any(item.derived_from_pair_type == "canonical" for item in records)


def test_mimo_recovery_candidate_selection_is_deterministic(tmp_path: Path) -> None:
    job_values = _job_fixture(tmp_path).model_dump(
        mode="json", exclude={"request_fingerprint"}
    )
    job_values["target_duration_seconds"] = 3.0
    first = job_values["segments"][0]
    first.update(
        end_time=0.4,
        source_end_sample=12800,
        current_entity_id=None,
        entity_occurrence_id=None,
        identity_scope="unresolved",
        direct_anchor_seconds=0.0,
        cluster_binding_status="unbound",
        direct_support_seconds_by_entity={},
    )
    second = dict(first)
    second.update(
        segment_id="segment_2",
        start_time=0.5,
        end_time=0.9,
        source_start_sample=16000,
        source_end_sample=28800,
        source_speaker_cluster_id="speaker_1",
        asr_text="Second exact text.",
    )
    job_values["segments"] = [first, second]
    job = _job(job_values)
    annotation_values = _annotation().model_dump(mode="json")
    second_decision = dict(annotation_values["segment_decisions"][0])
    second_decision.update(segment_id="segment_2", primary_speaker_group="g2")
    annotation_values["segment_decisions"] = [
        annotation_values["segment_decisions"][0],
        second_decision,
    ]
    annotation = MimoAVAnnotationDraft.model_validate(annotation_values)
    policy = MimoRecoveredVoiceQualityPolicy(minimum_duration_seconds=0.25)

    recovered, _, voices = recover_mimo_target_voices(
        job=job,
        record=_record_fixture(tmp_path, annotation, job=job),
        subject_index_by_entity={"e1": 1},
        existing_entity_ids=set(),
        reference_capacity=3,
        temporary_root=tmp_path / "temporary",
        final_root=tmp_path / "final",
        audio_backend=_RecoveredVoiceMediaBackend(),
        analyzer=_RecoveredVoiceAnalyzer(),
        policy=policy,
    )
    assert [item.source_segment_id for item in recovered] == ["segment_1", "segment_2"]
    assert recovered[0].status == "selected"
    assert recovered[1].reason_codes == ["not_selected_better_candidate"]
    assert voices[0].source_start_sample == 0


def test_review_exposes_recovered_audio_asset_and_provenance(tmp_path: Path) -> None:
    fixture = _configure_recovery_materializer_fixture(
        _materializer_fixture(tmp_path), recover_entity_id="e1"
    )
    materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        audio_backend=_RecoveredVoiceMediaBackend(),
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(),
    )
    raw_path = fixture.mimo_root / "raw_responses" / f"{fixture.job.clip_uid}.json"
    raw_path.parent.mkdir()
    raw_path.write_text(
        MimoRawResponse(
            clip_uid=fixture.job.clip_uid,
            request_fingerprint=fixture.job.request_fingerprint,
            raw_responses=[],
            diagnostics=[],
        ).model_dump_json(),
        encoding="utf-8",
    )
    cases, media = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    derived = next(
        item
        for item in cases[0].payload["shadow_variants"]
        if item["derived_from_pair_type"] == "canonical"
    )
    audio = derived["audio_references"][0]
    assert audio["source_type"] == "mimo_recovered_target_voice"
    assert audio["source_segment_id"] == "segment_1"
    assert audio["media_url"].removeprefix("/media/") in media
    page = render_review_html(cases, {})
    assert '<audio controls preload="none"' in page
    assert "mimo_recovered_target_voice" in page


def test_review_exposes_music_and_full_audio_roles_without_subject_binding(
    tmp_path: Path,
) -> None:
    fixture = _configure_music_materializer_fixture(_materializer_fixture(tmp_path))
    materialize_mimo25_h3_shadow(
        mimo_root=fixture.mimo_root,
        source_h3_root=fixture.source_h3,
        output_root=fixture.output_root,
        enable_full_audio_reuse=True,
        enable_music_reference=True,
        audio_backend=_ConditioningAudioBackend(source_duration_seconds=3.0),
        recovered_voice_analyzer=_RecoveredVoiceAnalyzer(seconds=3.0),
    )
    raw_path = fixture.mimo_root / "raw_responses" / f"{fixture.job.clip_uid}.json"
    raw_path.parent.mkdir()
    raw_path.write_text(
        MimoRawResponse(
            clip_uid=fixture.job.clip_uid,
            request_fingerprint=fixture.job.request_fingerprint,
            raw_responses=[],
            diagnostics=[],
        ).model_dump_json(),
        encoding="utf-8",
    )
    cases, media = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    audio_rows = [
        audio
        for variant in cases[0].payload["shadow_variants"]
        for audio in variant["audio_references"]
    ]
    assert {item["role"] for item in audio_rows} == {
        "music_reference",
        "full_audio_reuse",
    }
    assert all(item["subject_index"] is None for item in audio_rows)
    assert all(item["media_url"].removeprefix("/media/") in media for item in audio_rows)
    page = render_review_html(cases, {})
    assert "music_reference" in page
    assert "full_audio_reuse" in page
    assert "music_description" in page
    assert "interval_provenance" in page


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
                    warnings=["reasoning_tokens_nonzero_under_disabled_thinking"],
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


class _WarningBackend(_FakeBackend):
    def reconcile(self, job: MimoClipJob, **_: object) -> MimoBackendResult:
        self.calls.append(job.clip_uid)
        annotation = _annotation()
        return MimoBackendResult(
            annotation=annotation,
            raw_responses=(annotation.model_dump_json(), annotation.model_dump_json()),
            diagnostics=(
                MimoCompletionDiagnostic(
                    input_modality="target_video_with_embedded_audio",
                    finish_reason="stop",
                    usage=MimoUsage(image_tokens=2, video_tokens=4, audio_tokens=3),
                    http_attempt_count=1,
                    warnings=[
                        "image_tokens_unavailable",
                        "audio_tokens_unavailable",
                    ],
                ),
                MimoCompletionDiagnostic(
                    input_modality="full_av_recheck_embedded_audio",
                    finish_reason="stop",
                    usage=MimoUsage(image_tokens=2, video_tokens=4, audio_tokens=3),
                    http_attempt_count=1,
                    warnings=["image_tokens_unavailable"],
                ),
            ),
            model_call_count=2,
            http_attempt_count=2,
            http_retry_count=0,
            recheck_count=1,
            input_modality="target_video_with_embedded_audio",
            deterministic_correction_counts={
                "conservative_visible_speaker_downgrade": 2
            },
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
    backend = _WarningBackend(tmp_path)
    output = tmp_path / "mimo-output"
    summary = run_mimo25_av_reconcile(
        inventory=inventory,
        backend=backend,
        output_root=output,
    )
    assert backend.calls == ["clip-1"]
    assert summary.ready_count == 1
    assert summary.usage_totals["image_tokens"] == 4
    assert summary.diagnostic_warning_counts == {
        "audio_tokens_unavailable": 1,
        "image_tokens_unavailable": 2,
    }
    assert summary.correction_counts["conservative_visible_speaker_downgrade"] == 2
    published_records = [
        MimoRecord.model_validate(json.loads(line))
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert published_records[0].deterministic_correction_counts == {
        "conservative_visible_speaker_downgrade": 2
    }
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
    assert summary.diagnostic_warning_counts == {
        "reasoning_tokens_nonzero_under_disabled_thinking": 1
    }


def test_review_cases_validate_provenance_and_include_runtime_diagnostics(
    tmp_path: Path,
) -> None:
    fixture = _review_fixture(tmp_path)
    cases, _ = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    repeated, _ = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    assert cases[0].record_fingerprint == repeated[0].record_fingerprint
    assert cases[0].payload["mimo_runtime_diagnostics"] == [
        {
            "input_modality": "target_video_with_embedded_audio",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "image_tokens": 2,
                "video_tokens": 8,
                "audio_tokens": 3,
                "cached_tokens": None,
                "reasoning_tokens": 0,
            },
            "http_attempt_count": 1,
            "warnings": ["image_tokens_unavailable"],
            "request_error": None,
        }
    ]
    assert "raw response must not enter review HTML" not in json.dumps(cases[0].payload)


def test_review_fingerprint_changes_with_mimo_or_shadow_record(
    tmp_path: Path,
) -> None:
    fixture = _review_fixture(tmp_path)
    variants = _shadow_records(fixture)
    original = _review_case_fingerprint(fixture.record, variants)
    changed_mimo = _replace_record(fixture.record, model_call_count=2)
    changed_mimo_fingerprint = _review_case_fingerprint(changed_mimo, variants)
    assert changed_mimo_fingerprint != original
    review_root = tmp_path / "mimo-fingerprint-review"
    original_store = MimoReviewStore(
        review_root,
        [MimoReviewCase("clip-1", original, {"clip_uid": "clip-1"})],
    )
    original_store.save(
        MimoHumanReviewAnnotation(
            clip_uid="clip-1",
            record_fingerprint=original,
            decision="PASS",
            issue_tags=[],
            notes="current",
            reviewed_at="2026-09-02T00:00:00Z",
        )
    )
    changed_store = MimoReviewStore(
        review_root,
        [
            MimoReviewCase(
                "clip-1",
                changed_mimo_fingerprint,
                {"clip_uid": "clip-1"},
            )
        ],
    )
    assert changed_store.current_annotations() == {}
    shadow_values = variants[0].model_dump(mode="json", exclude={"record_fingerprint"})
    shadow_values["warnings"] = ["rerendered"]
    changed_shadow = mimo25_materializer._record(shadow_values)
    assert _review_case_fingerprint(fixture.record, [changed_shadow]) != original


@pytest.mark.parametrize(
    "field",
    ["inventory_fingerprint", "request_fingerprint"],
)
def test_review_rejects_stale_av_record_provenance(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _review_fixture(tmp_path)
    changed = _replace_record(fixture.record, **{field: "f" * 64})
    _write_models_jsonl(fixture.mimo_root / "records.jsonl", [changed])
    with pytest.raises(ValueError, match="AV annotation provenance mismatch"):
        build_review_cases(
            mimo_root=fixture.mimo_root,
            shadow_root=fixture.output_root,
        )


def test_changed_shadow_fingerprint_makes_old_review_stale(tmp_path: Path) -> None:
    fixture = _review_fixture(tmp_path)
    cases, _ = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    review_root = tmp_path / "human-review"
    store = MimoReviewStore(review_root, cases)
    store.save(
        MimoHumanReviewAnnotation(
            clip_uid="clip-1",
            record_fingerprint=cases[0].record_fingerprint,
            decision="PASS",
            issue_tags=[],
            notes="current",
            reviewed_at="2026-09-02T00:00:00Z",
        )
    )
    records = _shadow_records(fixture)
    values = records[0].model_dump(mode="json", exclude={"record_fingerprint"})
    values["warnings"] = ["rerendered"]
    _write_shadow_records(fixture, [mimo25_materializer._record(values)])
    changed_cases, _ = build_review_cases(
        mimo_root=fixture.mimo_root,
        shadow_root=fixture.output_root,
    )
    assert changed_cases[0].record_fingerprint != cases[0].record_fingerprint
    stale_store = MimoReviewStore(review_root, changed_cases)
    assert stale_store.current_annotations() == {}
    assert stale_store.publish_derived().stale_annotation_count == 1


def test_review_rejects_stale_shadow_record_provenance(tmp_path: Path) -> None:
    fixture = _review_fixture(tmp_path)
    records = _shadow_records(fixture)
    values = records[0].model_dump(mode="json", exclude={"record_fingerprint"})
    values["source_mimo_record_fingerprint"] = "f" * 64
    _write_shadow_records(fixture, [mimo25_materializer._record(values)])
    with pytest.raises(
        ValueError,
        match="MiMo review shadow provenance differs from current AV annotation",
    ):
        build_review_cases(
            mimo_root=fixture.mimo_root,
            shadow_root=fixture.output_root,
        )


@pytest.mark.parametrize("variant_change", ["missing", "duplicate"])
def test_review_rejects_missing_or_extra_shadow_variant(
    tmp_path: Path,
    variant_change: str,
) -> None:
    fixture = _review_fixture(tmp_path)
    records = _shadow_records(fixture)
    changed = [] if variant_change == "missing" else [records[0], records[0]]
    _write_shadow_records(fixture, changed)
    with pytest.raises(
        ValueError,
        match="MiMo review shadow provenance differs from current AV annotation",
    ):
        build_review_cases(
            mimo_root=fixture.mimo_root,
            shadow_root=fixture.output_root,
        )


@pytest.mark.parametrize("summary_change", ["inventory", "count"])
def test_review_rejects_stale_shadow_summary(
    tmp_path: Path,
    summary_change: str,
) -> None:
    fixture = _review_fixture(tmp_path)
    summary_path = fixture.output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_change == "inventory":
        summary["source_mimo_inventory_fingerprint"] = "f" * 64
    else:
        summary["sample_count"] = 2
        summary["ready_count"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="MiMo review shadow provenance differs from current AV annotation",
    ):
        build_review_cases(
            mimo_root=fixture.mimo_root,
            shadow_root=fixture.output_root,
        )


def test_review_rejects_runtime_diagnostics_provenance_drift(tmp_path: Path) -> None:
    fixture = _review_fixture(tmp_path)
    raw_path = fixture.mimo_root / "raw_responses" / "clip-1.json"
    raw = MimoRawResponse.model_validate_json(raw_path.read_text(encoding="utf-8"))
    payload = raw.model_dump(mode="json")
    payload["request_fingerprint"] = "f" * 64
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime diagnostics provenance mismatch"):
        build_review_cases(
            mimo_root=fixture.mimo_root,
            shadow_root=fixture.output_root,
        )


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
        "mimo_runtime_diagnostics": [
            {
                "input_modality": "target_video_with_embedded_audio",
                "finish_reason": "stop",
                "usage": {"image_tokens": 2, "video_tokens": 8, "audio_tokens": 3},
                "warnings": ["image_tokens_unavailable"],
            }
        ],
        "shadow_variants": [],
        "legacy_qwen38": {},
    }
    page = render_review_html([MimoReviewCase("clip-1", "a" * 64, payload)], {})
    assert "speaker_grouping_issue" in page
    assert "MiMo H3 shadow" in page
    assert "Legacy Qwen3.8" in page
    assert "MiMo runtime diagnostics" in page
    assert "image_tokens" in page
    assert "review_case_fingerprint" in page
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
