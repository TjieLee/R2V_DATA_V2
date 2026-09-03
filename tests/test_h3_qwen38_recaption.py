from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalQwen3SpeechSegment,
    FinalSubjectVoice,
    FinalVisualReference,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    QWEN38_RECAPTION_DRAFT_VERSION,
    QWEN38_RECAPTION_MATERIALIZER_VERSION,
    QWEN38_RECAPTION_POLICY_VERSION,
    QWEN38_RECAPTION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNGROUNDED_NON_DIEGETIC_MUSIC,
    UNGROUNDED_OVERALL_SOUNDSCAPE,
    AudioFactAuditItem,
    OpenAIQwen38RecaptionBackend,
    Qwen38BackendProvenance,
    Qwen38BackendResult,
    Qwen38DraftShot,
    Qwen38H3DraftResponse,
    Qwen38H3StructuredResponse,
    Qwen38RecaptionConfig,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRecord,
    Qwen38RecaptionRequest,
    RecaptionCompletionDiagnostic,
    RecaptionNonSpeechFact,
    build_audio_facts,
    build_qwen38_full_manifest,
    build_qwen38_pilot_manifest,
    build_reference_contract,
    materialize_h3_draft,
    render_h3_prompt,
    run_qwen38_h3_recaption_pilot,
    validate_h3_draft,
    validate_h3_response,
)
from r2v_data_v2.h3.qwen38_human_review import (
    Qwen38HumanReviewAnnotation,
    current_human_review_summary,
    current_human_reviews,
    make_review_handler,
    save_human_review,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from tools.run_h3_qwen38_recaption import _parser


def _reference(
    tmp_path: Path,
    index: int,
    kind: str,
    *,
    entity_id: str | None = None,
    attribute_id: str | None = None,
    owner_entity_id: str | None = None,
    attribute_type: str | None = None,
) -> FinalVisualReference:
    artifact = tmp_path / f"image-{index}.png"
    artifact.write_bytes(f"image-{index}".encode())
    values = {
        "image_id": f"image_{index}",
        "image_index": index,
        "kind": kind,
        "image_path": f"selected/image-{index}.png",
        "image_artifact_path": str(artifact),
        "source_frame_index": index,
        "synthetic": False,
    }
    if kind in {"subject", "object", "group"}:
        values.update(entity_id=entity_id or "e1", scope="full", visible_region="whole")
    elif kind == "attribute":
        values.update(
            attribute_id=attribute_id or "a1",
            owner_entity_id=owner_entity_id or "e1",
            attribute_type=attribute_type or "upper_clothing",
        )
    else:
        values.update(scope="scene")
    return FinalVisualReference.model_validate(values)


def _sample(
    tmp_path: Path,
    *,
    voice_source: str | None = "target",
    reference_count: int = 3,
) -> FinalH3SampleV2:
    video = tmp_path / "target.mp4"
    audio = tmp_path / "full.flac"
    voice = tmp_path / "voice.flac"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    voice.write_bytes(b"voice")
    references = [
        _reference(tmp_path, 1, "subject", entity_id="e1"),
        _reference(tmp_path, 2, "attribute", owner_entity_id="e1"),
        _reference(tmp_path, 3, "background"),
    ]
    for index in range(4, reference_count + 1):
        references.append(
            _reference(tmp_path, index, "object", entity_id=f"e{index}")
        )
    voices = []
    if voice_source is not None:
        voices.append(
            FinalSubjectVoice(
                subject_index=1,
                entity_id="e1",
                target_occurrence_id="clip-1/e1",
                voice_reference_path=str(voice),
                voice_reference_sha256=hashlib.sha256(voice.read_bytes()).hexdigest(),
                source_start=0.0,
                source_end=1.0,
                source_start_sample=0,
                source_end_sample=32000,
                sample_mapping_policy="round_time_seconds_times_32000_v1",
                voice_source=voice_source,
                donor_occurrence_id=("donor/e1" if voice_source == "cross_donor" else None),
                donor_clip_uid=("donor" if voice_source == "cross_donor" else None),
                donor_clip_display_path=(
                    "01/show/season/episode/donor"
                    if voice_source == "cross_donor"
                    else None
                ),
            )
        )
    speech = [
        FinalQwen3SpeechSegment(
            segment_id="segment-early",
            speaker_cluster_id="cluster-b",
            source_start_sample=3200,
            source_end_sample=6400,
            source_sample_rate_hz=32000,
            start_time=0.1,
            end_time=0.2,
            text="Wait here.",
            language="English",
        ),
        FinalQwen3SpeechSegment(
            segment_id="segment-late",
            speaker_cluster_id="cluster-a",
            entity_id="e1",
            entity_occurrence_id="clip-1/e1",
            source_start_sample=9600,
            source_end_sample=12800,
            source_sample_rate_hz=32000,
            start_time=0.3,
            end_time=0.4,
            text="I am ready.",
            language="English",
        ),
    ]
    return FinalH3SampleV2(
        sample_id="clip-1/in_pair",
        pair_id="in_pair/clip-1",
        pair_type="in_pair",
        clip_uid="clip-1",
        clip_display_path="01/show/season/episode/clip-1",
        media_collection_relpath="01/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-1",
        shard_id="shard-1",
        target_video=str(video),
        target_full_audio_path=str(audio),
        target_full_audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        r2v_instruction="Use Image 1, Image 2, and Image 3 as frozen references.",
        visual_references=references,
        subject_voices=voices,
        speech_segments=speech,
    )


def _provenance(tmp_path: Path) -> Qwen38BackendProvenance:
    return Qwen38RecaptionConfig(
        base_url="http://127.0.0.1:8000/v1",
        media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
    ).provenance()


def _request(
    tmp_path: Path,
    *,
    variant: str = "visual_only",
    voice_source: str | None = "target",
    with_event: bool = False,
) -> Qwen38RecaptionRequest:
    sample = _sample(tmp_path, voice_source=voice_source)
    contract = build_reference_contract(sample, variant)  # type: ignore[arg-type]
    facts = build_audio_facts(
        sample,
        contract,
        None,
        semantics_records_sha256=None,
    )
    if with_event:
        facts = facts.model_copy(
            update={
                "non_speech_events": [
                    RecaptionNonSpeechFact(
                        fact_id="non_speech_1",
                        start_time=0.2,
                        end_time=0.25,
                        category="temporal_audio_event",
                        description="the visible woman slams the door",
                        source_attribution="visible woman",
                        provenance="fixture",
                    )
                ]
            }
        )
    return Qwen38RecaptionRequest(
        sample=sample,
        case=Qwen38RecaptionManifestCase(
            sample_id=sample.sample_id,
            conditioning_variant=variant,  # type: ignore[arg-type]
        ),
        reference_contract=contract,
        audio_facts=facts,
        request_fingerprint="a" * 64,
    )


def _same_speaker_request(
    tmp_path: Path,
    *,
    with_event: bool = False,
) -> Qwen38RecaptionRequest:
    request = _request(tmp_path, with_event=with_event)
    first, second = request.audio_facts.speech
    speech = [
        first.model_copy(
            update={
                "speaker_cluster_id": second.speaker_cluster_id,
                "entity_id": second.entity_id,
                "entity_subject_label": second.entity_subject_label,
                "speaker_id": "S1",
            }
        ),
        second.model_copy(update={"speaker_id": "S1"}),
    ]
    return Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=request.audio_facts.model_copy(update={"speech": speech}),
        request_fingerprint=request.request_fingerprint,
    )


def _response(request: Qwen38RecaptionRequest) -> Qwen38H3StructuredResponse:
    return materialize_h3_draft(_draft(request), request)


def _draft(request: Qwen38RecaptionRequest) -> Qwen38H3DraftResponse:
    definitions = []
    for subject in request.reference_contract.subjects:
        definitions.append(
            f"{subject.subject_label} is the frozen referenced content sourced from "
            + " and ".join(subject.source_picture_labels)
            + "."
        )
    retention = [
        f"{subject.subject_label} (appears in [Shot 1]): fully_preserved - the referenced content is retained."
        for subject in request.reference_contract.subjects
    ]
    audits = [
        AudioFactAuditItem(fact_id=item.fact_id, action="preserved")
        for item in request.audio_facts.non_speech_events
    ]
    placeholders = " ".join(
        f"A visible action continues. [[{fact.fact_id}]]"
        for fact in request.audio_facts.speech
    )
    return Qwen38H3DraftResponse(
        subject_definitions=definitions,
        summary=(
            {
                "visual_only": "[reference generation]",
                "target_voice_reference": "[reference generation + audio reference]",
                "cross_voice_reference": "[reference generation + audio reference]",
                "full_audio_reuse": "[reference generation + audio reuse]",
            }[request.case.conditioning_variant]
            + " The target preserves all frozen referenced content."
        ),
        retention_analysis=retention,
        shots=[
            Qwen38DraftShot(
                shot_index=1,
                description_template=(
                    "The target uses a natural observational style. " + placeholders
                ),
            )
        ],
        overall_soundscape=UNGROUNDED_OVERALL_SOUNDSCAPE,
        non_diegetic_music=UNGROUNDED_NON_DIEGETIC_MUSIC,
        audio_fact_audit=audits,
    )


def _codes(
    response: Qwen38H3StructuredResponse,
    request: Qwen38RecaptionRequest,
) -> set[str]:
    issues, _ = validate_h3_response(response, request)
    return {item.code for item in issues}


def test_exact_six_section_renderer_order(tmp_path: Path) -> None:
    request = _request(tmp_path)
    rendered = render_h3_prompt(_response(request))
    labels = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    assert [rendered.index(label) for label in labels] == sorted(
        rendered.index(label) for label in labels
    )
    assert "audio_fact_audit" not in rendered


def test_reference_contract_preserves_picture_and_subject_mapping(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert [item.picture_label for item in request.reference_contract.pictures] == [
        "<Picture 1>",
        "<Picture 2>",
        "<Picture 3>",
    ]
    assert [item.kind for item in request.reference_contract.subjects] == [
        "entity",
        "attribute",
        "background",
    ]
    assert request.reference_contract.subjects[1].owner_entity_id == "e1"
    assert request.reference_contract.h3_reference_video_count == 0


def test_subject_order_is_entity_then_attribute_then_background(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    sample = sample.model_copy(
        update={
            "visual_references": [
                sample.visual_references[0],
                sample.visual_references[2].model_copy(
                    update={"image_id": "image_2", "image_index": 2}
                ),
                sample.visual_references[1].model_copy(
                    update={"image_id": "image_3", "image_index": 3}
                ),
            ]
        }
    )
    contract = build_reference_contract(sample, "visual_only")
    assert [item.kind for item in contract.subjects] == [
        "entity",
        "attribute",
        "background",
    ]
    assert contract.subjects[1].source_picture_labels == ["<Picture 3>"]
    assert contract.subjects[2].source_picture_labels == ["<Picture 2>"]


@pytest.mark.parametrize(
    ("variant", "voice_source", "prefix", "marker"),
    [
        ("visual_only", "target", "[reference generation]", None),
        (
            "target_voice_reference",
            "target",
            "[reference generation + audio reference]",
            "reference",
        ),
        (
            "cross_voice_reference",
            "cross_donor",
            "[reference generation + audio reference]",
            "reference",
        ),
        (
            "full_audio_reuse",
            "target",
            "[reference generation + audio reuse]",
            "fully_copy",
        ),
    ],
)
def test_conditioning_variants(
    tmp_path: Path,
    variant: str,
    voice_source: str,
    prefix: str,
    marker: str | None,
) -> None:
    request = _request(tmp_path, variant=variant, voice_source=voice_source)
    response = _response(request)
    assert response.summary.startswith(prefix)
    assert _codes(response, request) == set()
    if marker is None:
        assert request.reference_contract.audios == []
        assert "<Audio " not in render_h3_prompt(response)
    else:
        assert [item.retention_marker for item in request.reference_contract.audios] == [
            marker
        ]
    kinds = {item.kind for item in request.reference_contract.audios}
    assert not ({"target_voice", "cross_voice"} & kinds and "full_audio_reuse" in kinds)


def test_speaker_ids_follow_first_cluster_appearance_and_binding(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert [item.speaker_id for item in request.audio_facts.speech] == ["S1", "S2"]
    assert request.audio_facts.speech[0].entity_subject_label is None
    assert request.audio_facts.speech[1].entity_subject_label == "<Subject 1>"
    response = _response(request)
    assert "(S1) says," in response.detailed_description
    assert "<Subject 1> (S2)" in response.detailed_description
    assert _codes(response, request) == set()


def test_default_materializer_is_unchanged_without_speech_transform(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    draft = _draft(request)
    default = materialize_h3_draft(draft, request)
    identity_transform = materialize_h3_draft(
        draft,
        request,
        speech_clause_transform=lambda _speech, clause: clause,
    )
    assert default.model_dump(mode="json") == identity_transform.model_dump(mode="json")


@pytest.mark.parametrize(
    ("variant", "voice_source"),
    [
        ("target_voice_reference", "target"),
        ("cross_voice_reference", "cross_donor"),
    ],
)
def test_voice_audio_contract_is_materialized_deterministically(
    tmp_path: Path,
    variant: str,
    voice_source: str,
) -> None:
    request = _request(tmp_path, variant=variant, voice_source=voice_source)
    draft = _draft(request)
    assert all("<Audio " not in item for item in draft.subject_definitions)
    assert all("<Audio " not in item for item in draft.retention_analysis)
    response = materialize_h3_draft(draft, request)
    assert (
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S2)."
        in response.subject_definitions
    )
    retention = next(
        item for item in response.retention_analysis if item.startswith("<Audio 1>")
    )
    assert retention == (
        "<Audio 1>: reference - its voice timbre and delivery guide "
        "<Subject 1>'s speech without copying the source signal."
    )
    assert "(S2)" not in retention
    assert (
        "<Subject 1> (S2), using the voice timbre referenced from <Audio 1>, "
        "says,"
    ) in response.detailed_description
    first_dialogue_prefix = response.detailed_description.split(
        request.audio_facts.speech[0].locked_dialogue_block, maxsplit=1
    )[0]
    assert "<Audio 1>" not in first_dialogue_prefix
    assert _codes(response, request) == set()


def test_voice_audio_definition_does_not_invent_unknown_speaker_id(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, variant="target_voice_reference")
    audio = request.reference_contract.audios[0].model_copy(
        update={"speaker_id": None}
    )
    contract = request.reference_contract.model_copy(update={"audios": [audio]})
    adjusted = Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=contract,
        audio_facts=request.audio_facts,
        request_fingerprint=request.request_fingerprint,
    )
    response = materialize_h3_draft(_draft(adjusted), adjusted)
    definition = next(
        item for item in response.subject_definitions if item.startswith("<Audio 1>")
    )
    assert definition == "<Audio 1> is the voice-timbre reference for <Subject 1>."
    assert "referenced from <Audio 1>" in response.detailed_description
    assert _codes(response, adjusted) == set()


def test_full_audio_reuse_materializes_canonical_definition_and_retention(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, variant="full_audio_reuse")
    response = _response(request)
    assert (
        "<Audio 1> is the supplied full-audio reference for the target."
        in response.subject_definitions
    )
    assert (
        "<Audio 1>: fully_copy - the supplied full audio is reused in full."
        in response.retention_analysis
    )
    assert "referenced from <Audio 1>" not in response.detailed_description
    assert _codes(response, request) == set()


def test_draft_cannot_emit_pipeline_owned_audio_lines(tmp_path: Path) -> None:
    request = _request(tmp_path, variant="target_voice_reference")
    draft = _draft(request).model_copy(
        update={
            "subject_definitions": [
                *_draft(request).subject_definitions,
                "<Audio 1> is a model-authored Audio definition.",
            ]
        }
    )
    assert "draft_contains_pipeline_owned_audio_reference" in {
        item.code for item in validate_h3_draft(draft, request)
    }


def test_music_absence_distinguishes_grounded_from_unknown(tmp_path: Path) -> None:
    request = _request(tmp_path)
    grounded_request = Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=request.audio_facts.model_copy(
            update={"audio_grounding_complete": True}
        ),
        request_fingerprint=request.request_fingerprint,
    )
    grounded = materialize_h3_draft(_draft(grounded_request), grounded_request)
    assert grounded.non_diegetic_music == "N/A"
    assert _codes(grounded, grounded_request) == set()
    unknown = materialize_h3_draft(_draft(request), request)
    assert unknown.non_diegetic_music == UNGROUNDED_NON_DIEGETIC_MUSIC
    assert unknown.non_diegetic_music != "N/A"


def test_speech_timestamps_remain_internal_placement_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    response = _response(request)
    for speech in request.audio_facts.speech:
        assert str(speech.start_time) not in response.detailed_description
        assert str(speech.end_time) not in response.detailed_description
    assert "start_time" in request.audio_facts.model_dump(mode="json")["speech"][0]


def test_every_same_speaker_turn_repeats_exact_source(tmp_path: Path) -> None:
    request = _same_speaker_request(tmp_path)
    response = materialize_h3_draft(_draft(request), request)
    source = "<Subject 1> (S1)"
    assert response.detailed_description.count(source) == 2
    assert _codes(response, request) == set()


def test_same_speaker_materialization_preserves_mixed_binding_per_turn(
    tmp_path: Path,
) -> None:
    request = _same_speaker_request(tmp_path)
    first, second = request.audio_facts.speech
    speech = [
        first,
        second.model_copy(
            update={
                "fact_id": "speech_2",
                "entity_id": None,
                "entity_subject_label": None,
            }
        ),
        first.model_copy(
            update={
                "fact_id": "speech_3",
                "segment_id": "segment-third",
                "start_time": 0.5,
                "end_time": 0.6,
                "text": "Third line.",
                "locked_dialogue_block": "<d>[English] Third line.</d>",
            }
        ),
    ]
    mixed_request = Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=request.audio_facts.model_copy(update={"speech": speech}),
        request_fingerprint=request.request_fingerprint,
    )
    response = materialize_h3_draft(_draft(mixed_request), mixed_request)
    description = response.detailed_description
    first_position = description.index("<d>[English] Wait here.</d>")
    second_position = description.index("<d>[English] I am ready.</d>")
    third_position = description.index("<d>[English] Third line.</d>")
    assert "<Subject 1> (S1) says," in description[:first_position]
    assert description[first_position:second_position].rstrip().endswith("(S1) says,")
    assert "<Subject 1> (S1) says," in description[second_position:third_position]
    assert _codes(response, mixed_request) == set()


def test_pronoun_on_second_same_speaker_turn_rejects_source_mismatch(
    tmp_path: Path,
) -> None:
    request = _same_speaker_request(tmp_path)
    response = _response(request)
    second = request.audio_facts.speech[1]
    exact_clause = (
        f"<Subject 1> (S1) says, {second.locked_dialogue_block}"
    )
    changed = response.model_copy(
        update={
            "detailed_description": response.detailed_description.replace(
                exact_clause,
                f"She says, {second.locked_dialogue_block}",
                1,
            )
        }
    )
    assert "locked_dialogue_source_mismatch" in _codes(changed, request)


@pytest.mark.parametrize(
    ("templates", "expected_code"),
    [
        (["Only [[speech_1]] appears."], "missing_speech_placeholder"),
        (
            ["[[speech_1]] then [[speech_1]] then [[speech_2]]"],
            "duplicate_speech_placeholder",
        ),
        (["[[speech_2]] then [[speech_1]]"], "speech_placeholder_order_mismatch"),
        (
            ["[[speech_1]] then [[speech_2]] then [[speech_99]]"],
            "unknown_speech_placeholder",
        ),
        (
            ["[Shot 1] [[speech_1]] then [[speech_2]]"],
            "draft_contains_shot_header",
        ),
        (
            ["<d>[English] copied</d> [[speech_1]] then [[speech_2]]"],
            "draft_contains_dialogue_markup",
        ),
        (
            ["She says [[speech_1]] then pauses before [[speech_2]]"],
            "draft_prefixes_complete_speech_placeholder",
        ),
    ],
)
def test_draft_rejects_pipeline_owned_or_invalid_placeholder_content(
    tmp_path: Path,
    templates: list[str],
    expected_code: str,
) -> None:
    request = _same_speaker_request(tmp_path)
    draft = _draft(request).model_copy(
        update={
            "shots": [
                Qwen38DraftShot(
                    shot_index=index,
                    start_time=None if index == 1 else float(index),
                    description_template=template,
                )
                for index, template in enumerate(templates, start=1)
            ]
        }
    )
    codes = {item.code for item in validate_h3_draft(draft, request)}
    assert expected_code in codes


def test_materializer_owns_single_and_later_shot_headers(tmp_path: Path) -> None:
    request = _request(tmp_path)
    draft = _draft(request)
    single = materialize_h3_draft(draft, request)
    assert single.detailed_description.startswith("[Shot 1] ")
    second = Qwen38DraftShot(
        shot_index=2,
        start_time=3.125,
        description_template="A hard cut reveals the opposite side of the room.",
    )
    two_shots = draft.model_copy(update={"shots": [draft.shots[0], second]})
    materialized = materialize_h3_draft(two_shots, request)
    assert "[Shot 2] At 00:03.125," in materialized.detailed_description
    assert _codes(materialized, request) == set()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda value: value.replace("<Subject 1>", "<Subject 9>", 1), "unknown_reference_label"),
        (lambda value: value + " <Video 1>", "unknown_reference_label"),
        (lambda value: value + " (S9)", "unknown_speaker_id"),
        (lambda value: value + " <Picture 9>", "unknown_reference_label"),
        (lambda value: value.replace("Wait here.", "Wait there.", 1), "locked_dialogue_mismatch"),
        (lambda value: value + " <Picture 1> is a keyframe.", "unassigned_picture_keyframe_role"),
    ],
)
def test_validator_rejects_unknown_or_changed_contract(
    tmp_path: Path,
    mutation: object,
    expected_code: str,
) -> None:
    request = _request(tmp_path)
    response = _response(request)
    changed = response.model_copy(
        update={"detailed_description": mutation(response.detailed_description)}  # type: ignore[operator]
    )
    assert expected_code in _codes(changed, request)


def test_reference_limit_fails_without_dropping_images(tmp_path: Path) -> None:
    sample = _sample(tmp_path, reference_count=10)
    with pytest.raises(ValueError, match="per-modality reference limit"):
        build_reference_contract(sample, "visual_only")


def test_audio_attribution_can_generalize_but_not_delete(tmp_path: Path) -> None:
    request = _request(tmp_path, with_event=True)
    response = _response(request).model_copy(
        update={
            "audio_fact_audit": [
                AudioFactAuditItem(
                    fact_id="non_speech_1",
                    action="attribution_generalized",
                    rewritten_description="a door slam is heard",
                )
            ]
        }
    )
    assert _codes(response, request) == set()
    with pytest.raises(ValidationError):
        AudioFactAuditItem.model_validate(
            {"fact_id": "non_speech_1", "action": "deleted", "rewritten_description": None}
        )


def test_speech_fact_cannot_enter_empty_non_speech_audit(tmp_path: Path) -> None:
    request = _request(tmp_path)
    response = _response(request).model_copy(
        update={
            "audio_fact_audit": [
                AudioFactAuditItem(fact_id="speech_1", action="preserved")
            ]
        }
    )
    assert "audio_fact_audit_mismatch" in _codes(response, request)


@pytest.mark.parametrize("claim", ["N/A", "none", "no music", "no background music"])
def test_ungrounded_music_cannot_claim_absence(tmp_path: Path, claim: str) -> None:
    request = _request(tmp_path)
    response = _response(request).model_copy(update={"non_diegetic_music": claim})
    assert "ungrounded_non_diegetic_music" in _codes(response, request)


def test_ungrounded_soundscape_cannot_invent_office_ambience(tmp_path: Path) -> None:
    request = _request(tmp_path)
    response = _response(request).model_copy(
        update={
            "overall_soundscape": (
                "Subtle ambient room tone typical of an office environment."
            )
        }
    )
    assert "ungrounded_overall_soundscape" in _codes(response, request)


def test_grounded_audio_semantics_remain_eligible_for_specific_soundscape(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    grounded_facts = request.audio_facts.model_copy(
        update={
            "audio_grounding_complete": True,
            "overall_soundscape_hint": "Office room tone is audible.",
            "non_diegetic_music_hint": "Soft instrumental music is audible.",
        }
    )
    grounded_request = Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=grounded_facts,
        request_fingerprint=request.request_fingerprint,
    )
    response = _response(grounded_request).model_copy(
        update={
            "overall_soundscape": "Office room tone remains audible under speech.",
            "non_diegetic_music": "Soft instrumental music continues in the background.",
        }
    )
    assert _codes(response, grounded_request) == set()


def test_missing_audio_semantics_stays_explicit_and_invents_nothing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert request.audio_facts.audio_grounding_complete is False
    assert request.audio_facts.non_speech_events == []
    assert request.audio_facts.overall_soundscape_hint is None
    assert request.audio_facts.non_diegetic_music_hint is None


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )


def test_openai_request_labels_media_and_never_sends_audio(tmp_path: Path) -> None:
    request = _request(tmp_path)
    valid = _draft(request).model_dump_json()
    completions = _FakeCompletions([valid])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIQwen38RecaptionBackend(
        Qwen38RecaptionConfig(
            base_url="http://127.0.0.1:8000/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        ),
        client=client,
    )
    result = backend.recaption(request)
    assert result.model_call_count == 1
    assert result.response.detailed_description.startswith("[Shot 1] ")
    assert "[[speech_" not in result.response.detailed_description
    assert request.audio_facts.speech[0].locked_dialogue_block in (
        result.response.detailed_description
    )
    raw_draft = json.loads(result.raw_responses[0])
    assert "shots" in raw_draft
    assert "detailed_description" not in raw_draft
    payload = completions.requests[0]
    content = payload["messages"][1]["content"]  # type: ignore[index]
    types = [item["type"] for item in content]  # type: ignore[index]
    assert types[:2] == ["text", "video_url"]
    assert types.count("image_url") == 3
    assert "audio_url" not in types
    user_prompt = content[-1]["text"]  # type: ignore[index]
    assert "SPEECH PLACEHOLDER CONTRACT:" in user_prompt
    assert "speech_1: [[speech_1]]" in user_prompt
    assert "allowed_fact_ids=[]" in user_prompt
    assert "audio_fact_audit MUST be exactly []." in user_prompt
    assert payload["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert "mm_processor_kwargs" not in payload["extra_body"]
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["presence_penalty"] == 1.5
    assert payload["max_tokens"] == 8192
    schema = payload["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    assert "shots" in schema["properties"]
    assert "detailed_description" not in schema["properties"]


def test_sglang_provenance_records_non_thinking_sampling(tmp_path: Path) -> None:
    provenance = _provenance(tmp_path)
    assert provenance.backend == "sglang"
    assert provenance.temperature == 0.7
    assert provenance.top_p == 0.8
    assert provenance.top_k == 20
    assert provenance.min_p == 0.0
    assert provenance.presence_penalty == 1.5
    assert provenance.repetition_penalty == 1.0
    assert provenance.enable_thinking is False
    assert provenance.draft_schema_version == QWEN38_RECAPTION_DRAFT_VERSION
    assert provenance.materializer_version == QWEN38_RECAPTION_MATERIALIZER_VERSION
    assert "video_fps" not in provenance.model_dump(mode="json")


def test_cli_exposes_sglang_sampling_without_video_fps(tmp_path: Path) -> None:
    arguments = _parser().parse_args(["--audio-production-root", str(tmp_path)])
    assert arguments.min_p == 0.0
    assert arguments.repetition_penalty == 1.0
    assert "video_fps" not in vars(arguments)


def test_cli_exposes_full_inventory_manifest_policy(tmp_path: Path) -> None:
    arguments = _parser().parse_args(
        [
            "--audio-production-root",
            str(tmp_path),
            "--prepare-all-manifest",
            str(tmp_path / "all.jsonl"),
            "--conditioning-policy",
            "sample_pair_type",
        ]
    )
    assert arguments.prepare_all_manifest == tmp_path / "all.jsonl"
    assert arguments.conditioning_policy == "sample_pair_type"


def test_legacy_vllm_backend_literal_remains_parseable(tmp_path: Path) -> None:
    values = _provenance(tmp_path).model_dump(
        mode="json", exclude={"configuration_fingerprint"}
    )
    values["backend"] = "vllm"
    values.pop("min_p")
    values.pop("repetition_penalty")
    values["video_fps"] = 4.0
    fingerprint = hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    provenance = Qwen38BackendProvenance.model_validate(
        {**values, "configuration_fingerprint": fingerprint}
    )
    assert provenance.backend == "vllm"
    assert provenance.min_p is None
    assert provenance.repetition_penalty is None
    assert "video_fps" not in provenance.model_dump(mode="json")


def test_true_malformed_response_uses_exactly_one_repair(tmp_path: Path) -> None:
    request = _request(tmp_path)
    valid = _draft(request).model_dump_json()
    completions = _FakeCompletions(["not-json", valid])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIQwen38RecaptionBackend(
        Qwen38RecaptionConfig(
            base_url="http://127.0.0.1:8000/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        ),
        client=client,
    )
    assert backend.recaption(request).model_call_count == 2
    assert len(completions.requests) == 2


def test_repair_repeats_exact_speech_and_audit_contracts(tmp_path: Path) -> None:
    request = _same_speaker_request(tmp_path, with_event=True)
    valid = _draft(request)
    invalid_shot = valid.shots[0].model_copy(
        update={
            "description_template": valid.shots[0].description_template.replace(
                "[[speech_2]]", ""
            )
        }
    )
    invalid = valid.model_copy(update={"shots": [invalid_shot]})
    completions = _FakeCompletions([invalid.model_dump_json(), valid.model_dump_json()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIQwen38RecaptionBackend(
        Qwen38RecaptionConfig(
            base_url="http://127.0.0.1:8000/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        ),
        client=client,
    )
    assert backend.recaption(request).model_call_count == 2
    repair_content = completions.requests[1]["messages"][1]["content"]  # type: ignore[index]
    repair_prompt = repair_content[-1]["text"]  # type: ignore[index]
    assert "SPEECH PLACEHOLDER CONTRACT:" in repair_prompt
    assert "speech_1: [[speech_1]]" in repair_prompt
    assert "speech_2: [[speech_2]]" in repair_prompt
    assert 'allowed_fact_ids=["non_speech_1"]' in repair_prompt
    assert "Speech fact IDs are never audit entries." in repair_prompt


@dataclass
class _FakeBackend:
    provenance: Qwen38BackendProvenance
    calls: list[str]

    def recaption(self, request: Qwen38RecaptionRequest) -> Qwen38BackendResult:
        self.calls.append(request.sample.sample_id)
        response = _response(request)
        issues, warnings = validate_h3_response(response, request)
        assert not issues
        return Qwen38BackendResult(
            response=response,
            raw_responses=(response.model_dump_json(),),
            diagnostics=(RecaptionCompletionDiagnostic(finish_reason="stop"),),
            model_call_count=1,
            validation_warnings=tuple(warnings),
        )


def _write_samples(root: Path, samples: list[FinalH3SampleV2]) -> Path:
    path = root / "h3/samples.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in samples
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_helper_and_sidecar_are_read_only(tmp_path: Path) -> None:
    production = tmp_path / "production"
    sample = _sample(tmp_path)
    samples_path = _write_samples(production, [sample])
    before = samples_path.read_bytes()
    manifest = tmp_path / "pilot.jsonl"
    cases = build_qwen38_pilot_manifest(
        h3_samples_path=samples_path,
        output_path=manifest,
        size=1,
        conditioning_variant="visual_only",
    )
    assert [item.sample_id for item in cases] == [sample.sample_id]
    backend = _FakeBackend(provenance=_provenance(tmp_path), calls=[])
    output = tmp_path / "pilot-output"
    summary = run_qwen38_h3_recaption_pilot(
        audio_production_root=production,
        case_manifest_path=manifest,
        backend=backend,
        output_root=output,
    )
    assert summary.ready_count == 1
    assert summary.target_video_reference_count == 0
    assert summary.checkpoint_written is False
    assert samples_path.read_bytes() == before
    assert backend.calls == [sample.sample_id]
    assert (output / "manifest.jsonl").is_file()
    assert (output / "records.jsonl").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "review.html").is_file()
    for name in ("annotations.jsonl", "annotations.csv", "summary.json", "report.md"):
        assert (output / "human_review" / name).is_file()
    assert (output / "human_review/annotations.jsonl").read_text() == ""
    assert list((output / "raw_responses").glob("*.json"))
    review = (output / "review.html").read_text(encoding="utf-8")
    assert "media/0000/" not in review
    target_sha = hashlib.sha256(b"video").hexdigest()
    assert f"target-{target_sha[:12]}.mp4?v={target_sha}" in review
    assert "picture-1-" in review
    assert "?v=" in review
    assert "file://" not in review


def test_full_manifest_preserves_all_pair_samples_and_maps_conditioning(
    tmp_path: Path,
) -> None:
    in_root = tmp_path / "in"
    cross_root = tmp_path / "cross"
    in_root.mkdir()
    cross_root.mkdir()
    in_sample = _sample(in_root, voice_source="target")
    cross_sample = _sample(cross_root, voice_source="cross_donor").model_copy(
        update={
            "sample_id": "clip-1/cross_pair/1",
            "pair_id": "cross_pair/clip-1/1",
            "pair_type": "cross_pair",
        }
    )
    canonical_sample = in_sample.model_copy(
        update={
            "sample_id": "clip-1/canonical",
            "pair_id": "canonical/clip-1",
            "pair_type": "canonical",
            "subject_voices": [],
        }
    )
    production = tmp_path / "production"
    samples_path = _write_samples(
        production, [canonical_sample, in_sample, cross_sample]
    )
    before = samples_path.read_bytes()
    manifest_path = tmp_path / "all.jsonl"
    cases = build_qwen38_full_manifest(
        h3_samples_path=samples_path,
        output_path=manifest_path,
        conditioning_policy="sample_pair_type",
    )
    assert [item.sample_id for item in cases] == [
        canonical_sample.sample_id,
        in_sample.sample_id,
        cross_sample.sample_id,
    ]
    assert [item.conditioning_variant for item in cases] == [
        "visual_only",
        "target_voice_reference",
        "cross_voice_reference",
    ]
    assert all("canonical-wide" in (item.note or "") for item in cases)
    assert samples_path.read_bytes() == before


def test_full_manifest_fails_closed_without_required_voice_contract(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path).model_copy(update={"subject_voices": []})
    canonical = sample.model_copy(
        update={
            "sample_id": "clip-1/canonical",
            "pair_id": "canonical/clip-1",
            "pair_type": "canonical",
        }
    )
    samples_path = _write_samples(tmp_path / "production", [canonical, sample])
    with pytest.raises(ValueError, match="cannot satisfy target_voice_reference"):
        build_qwen38_full_manifest(
            h3_samples_path=samples_path,
            output_path=tmp_path / "all.jsonl",
            conditioning_policy="sample_pair_type",
        )


def test_prompt_version_is_frozen() -> None:
    assert QWEN38_RECAPTION_PROMPT_VERSION == "h3_qwen38_ref2va_recaption_v6"
    assert QWEN38_RECAPTION_POLICY_VERSION == "h3_qwen38_ref2va_contract_v4"
    assert QWEN38_RECAPTION_DRAFT_VERSION == "r2v.h3.qwen38_recaption_draft.1"
    assert QWEN38_RECAPTION_MATERIALIZER_VERSION == "h3_qwen38_materializer_v2"


def test_prompt_allows_only_current_visible_retention_markers() -> None:
    allowed_contract = (
        "the only allowed visible retention markers are:\n"
        "- fully_preserved\n"
        "- partially_preserved\n"
        "- weak_reference"
    )
    assert allowed_contract in SYSTEM_PROMPT
    assert (
        "attribute_transfer is not assigned by the current conditioning contract "
        "and MUST\nNOT be emitted"
    ) in SYSTEM_PROMPT
    assert "markers are fully_preserved, partially_preserved, attribute_transfer" not in (
        SYSTEM_PROMPT
    )


@pytest.mark.parametrize(
    "marker",
    ["fully_preserved", "partially_preserved", "weak_reference"],
)
def test_current_visible_retention_markers_remain_valid(
    tmp_path: Path,
    marker: str,
) -> None:
    request = _request(tmp_path)
    response = _response(request)
    retention = list(response.retention_analysis)
    retention[0] = retention[0].replace("fully_preserved", marker)
    normalized = response.model_copy(update={"retention_analysis": retention})
    assert _codes(normalized, request) == set()


def test_attribute_transfer_remains_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    response = _response(request)
    retention = list(response.retention_analysis)
    retention[0] = retention[0].replace("fully_preserved", "attribute_transfer")
    assert "unassigned_attribute_transfer" in _codes(
        response.model_copy(update={"retention_analysis": retention}), request
    )


def test_prompt_requires_dense_observed_visual_recaption() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    required_dimensions = (
        "visual style",
        "shot scale and framing",
        "camera angle",
        "foreground, midground, and background composition",
        "every salient visible subject's appearance",
        "spatial positions and relationships",
        "body, arm, and hand motion",
        "head motion",
        "gaze",
        "facial expression and visible expression changes",
        "object state and state changes",
        "environment, materials, and readable text",
        "lighting and color",
        "explicit stable/static camera",
        "early through middle to late portions of the shot",
        "speech placeholders at their correct observed temporal positions",
    )
    assert "generation-quality dense video description" in prompt
    for dimension in required_dimensions:
        assert dimension in prompt
    assert "300-450 English words" in prompt
    assert "Do not pad a visually simple clip" in prompt


def test_prompt_forbids_unsupported_visual_filler() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "Use observed evidence, not plausible filler" in prompt
    for forbidden_inference in (
        "psychology or emotion",
        "intent",
        "causality",
        "relationships",
        "identity not supplied upstream",
        "sounds from visible actions",
        "invisible or offscreen events",
        "invented object details",
    ):
        assert forbidden_inference in prompt


def _response_with_word_count(
    response: Qwen38H3StructuredResponse,
    word_count: int,
) -> Qwen38H3StructuredResponse:
    current = len(re.findall(r"\b[\w'-]+\b", response.detailed_description))
    assert current <= word_count
    padding = " ".join(f"detail{index}" for index in range(word_count - current))
    description = response.detailed_description
    if padding:
        description += " " + padding
    assert len(re.findall(r"\b[\w'-]+\b", description)) == word_count
    return response.model_copy(update={"detailed_description": description})


@pytest.mark.parametrize(
    ("word_count", "expected_warnings"),
    [
        (249, ["detailed_description_below_250_words"]),
        (250, []),
        (500, []),
        (501, ["detailed_description_above_500_words"]),
    ],
)
def test_description_word_count_warnings_are_non_blocking_boundaries(
    tmp_path: Path,
    word_count: int,
    expected_warnings: list[str],
) -> None:
    request = _request(tmp_path)
    response = _response_with_word_count(_response(request), word_count)
    issues, warnings = validate_h3_response(response, request)
    assert issues == []
    assert warnings == expected_warnings


def _review_output_fixture(
    tmp_path: Path,
    *,
    variant: str = "target_voice_reference",
) -> tuple[Path, Path, _FakeBackend]:
    production = tmp_path / "production"
    sample = _sample(
        tmp_path,
        voice_source=("cross_donor" if variant == "cross_voice_reference" else "target"),
    )
    samples_path = _write_samples(production, [sample])
    manifest = tmp_path / "manifest.jsonl"
    build_qwen38_pilot_manifest(
        h3_samples_path=samples_path,
        output_path=manifest,
        size=1,
        conditioning_variant=variant,  # type: ignore[arg-type]
    )
    backend = _FakeBackend(provenance=_provenance(tmp_path), calls=[])
    output = tmp_path / "review-output"
    run_qwen38_h3_recaption_pilot(
        audio_production_root=production,
        case_manifest_path=manifest,
        backend=backend,
        output_root=output,
    )
    return output, manifest, backend


def _review_identity(output: Path) -> tuple[str, str, str]:
    record = json.loads((output / "records.jsonl").read_text(encoding="utf-8"))
    return record["sample_id"], record["clip_uid"], record["request_fingerprint"]


def test_human_review_annotation_schema_is_strict() -> None:
    annotation = Qwen38HumanReviewAnnotation(
        sample_id="sample",
        clip_uid="clip",
        request_fingerprint="a" * 64,
        decision="issue",
        severity="major",
        issue_tags=["audio_hallucination"],
        notes="Repeated unsupported events.",
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert annotation.schema_version == "r2v.h3.qwen38_human_review.1"
    with pytest.raises(ValidationError):
        Qwen38HumanReviewAnnotation.model_validate(
            {**annotation.model_dump(mode="json"), "decision": "accept"}
        )
    with pytest.raises(ValidationError):
        Qwen38HumanReviewAnnotation.model_validate(
            {
                **annotation.model_dump(mode="json"),
                "decision": "issue",
                "severity": None,
            }
        )


def test_human_review_save_reload_update_and_reports(tmp_path: Path) -> None:
    output, _, _ = _review_output_fixture(tmp_path)
    sample_id, clip_uid, fingerprint = _review_identity(output)
    first_time = datetime(2026, 9, 1, 1, 2, 3, tzinfo=UTC)
    annotation, summary = save_human_review(
        output,
        {
            "sample_id": sample_id,
            "clip_uid": clip_uid,
            "request_fingerprint": fingerprint,
            "decision": "pass",
            "severity": None,
            "issue_tags": [],
            "notes": "Looks correct.",
        },
        reviewed_at=first_time,
    )
    assert annotation.reviewed_at == first_time
    assert summary.human_review["pass"] == 1
    second_time = datetime(2026, 9, 1, 2, 3, 4, tzinfo=UTC)
    updated, summary = save_human_review(
        output,
        {
            "sample_id": sample_id,
            "clip_uid": clip_uid,
            "request_fingerprint": fingerprint,
            "decision": "issue",
            "severity": "major",
            "issue_tags": ["audio_event_oversegmentation", "audio_hallucination"],
            "notes": "Repeated unsupported events.",
        },
        reviewed_at=second_time,
    )
    assert updated.reviewed_at == second_time
    annotations = (output / "human_review/annotations.jsonl").read_text().splitlines()
    assert len(annotations) == 1
    assert current_human_reviews(output)["annotations"][0]["decision"] == "issue"
    assert summary.human_review["pass"] == 0
    assert summary.human_review["issue"] == 1
    assert summary.human_review["severity_counts"] == {"major": 1}
    assert summary.human_review["issue_tag_counts"] == {
        "audio_event_oversegmentation": 1,
        "audio_hallucination": 1,
    }
    assert summary.batch["prompt_version"] == QWEN38_RECAPTION_PROMPT_VERSION
    assert summary.batch["policy_version"] == QWEN38_RECAPTION_POLICY_VERSION
    assert summary.batch["materializer_version"] == (
        QWEN38_RECAPTION_MATERIALIZER_VERSION
    )
    assert summary.scope["inventory_scope"] == "current_h3_samples_inventory_only"
    assert summary.scope["canonical_wide_coverage"] is False
    csv_text = (output / "human_review/annotations.csv").read_text()
    assert "request_fingerprint" in csv_text
    assert "audio_event_oversegmentation;audio_hallucination" in csv_text
    report = (output / "human_review/report.md").read_text()
    assert "H3 Full Review Progress" in report
    assert "Issue #11 canonical-wide migration pending" in report


def test_stale_fingerprint_is_not_counted_as_current_review(tmp_path: Path) -> None:
    output, _, _ = _review_output_fixture(tmp_path)
    sample_id, clip_uid, fingerprint = _review_identity(output)
    save_human_review(
        output,
        {
            "sample_id": sample_id,
            "clip_uid": clip_uid,
            "request_fingerprint": fingerprint,
            "decision": "pass",
            "severity": None,
            "issue_tags": [],
            "notes": "Old output.",
        },
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    record = json.loads((output / "records.jsonl").read_text())
    record["request_fingerprint"] = "b" * 64
    (output / "records.jsonl").write_text(json.dumps(record) + "\n")
    reviews = current_human_reviews(output)
    summary = current_human_review_summary(output)
    assert reviews == {"annotations": [], "stale_annotation_count": 1}
    assert summary["human_review"]["reviewed"] == 0
    assert summary["human_review"]["unreviewed"] == 1
    with pytest.raises(ValueError, match="fingerprint is stale"):
        save_human_review(
            output,
            {
                "sample_id": sample_id,
                "clip_uid": clip_uid,
                "request_fingerprint": fingerprint,
                "decision": "pass",
                "severity": None,
                "issue_tags": [],
                "notes": "Must not bind to stale output.",
            },
        )


def test_review_media_urls_are_content_and_sample_addressed(tmp_path: Path) -> None:
    from r2v_data_v2.h3.qwen38_human_review import materialize_review_media

    output, _, _ = _review_output_fixture(tmp_path)
    record = Qwen38RecaptionRecord.model_validate(
        json.loads((output / "records.jsonl").read_text())
    )
    first = materialize_review_media(tmp_path / "media-a", [record])[record.sample_id]
    repeated = materialize_review_media(tmp_path / "media-b", [record])[record.sample_id]
    assert first == repeated
    other_sample = record.model_copy(update={"sample_id": "different/sample"})
    other = materialize_review_media(tmp_path / "media-c", [other_sample])[
        other_sample.sample_id
    ]
    assert first["target"] != other["target"]
    changed_target = tmp_path / "changed.mp4"
    changed_target.write_bytes(b"changed-video")
    changed_sha = hashlib.sha256(b"changed-video").hexdigest()
    changed_record = record.model_copy(
        update={
            "target_video_path": str(changed_target),
            "target_video_sha256": changed_sha,
        }
    )
    changed = materialize_review_media(tmp_path / "media-d", [changed_record])[
        record.sample_id
    ]
    assert first["target"] != changed["target"]
    assert changed_sha in changed["target"]
    assert first["audios"]


def test_review_html_controls_and_collapsed_audit_are_present(tmp_path: Path) -> None:
    output, _, _ = _review_output_fixture(tmp_path)
    review = (output / "review.html").read_text()
    for label in ("Reviewed", "Pass", "Issue", "Skip", "Unreviewed"):
        assert label in review
    assert "Model failed" in review
    assert "Issue tag" in review
    assert "Save Review" in review
    assert "audio_event_oversegmentation" in review
    assert "<details><summary>Audio fact audit:" in review
    assert "<details open><summary>Audio fact audit:" not in review
    assert "This batch covers the current H3 samples inventory only" in review
    assert "audio_fact_audit:" not in render_h3_prompt(
        _response(_request(tmp_path))
    )
    assert "raw_responses" not in render_h3_prompt(_response(_request(tmp_path)))


def test_generated_review_javascript_is_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for JavaScript syntax validation")
    output, _, _ = _review_output_fixture(tmp_path)
    review = (output / "review.html").read_text(encoding="utf-8")
    match = re.search(r"<script>(.*?)</script>", review, flags=re.DOTALL)
    assert match is not None
    script_path = tmp_path / "qwen38-review.js"
    script_path.write_text(match.group(1), encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_review_server_cache_headers_and_persisted_post(tmp_path: Path) -> None:
    output, _, _ = _review_output_fixture(tmp_path)
    sample_id, clip_uid, fingerprint = _review_identity(output)
    server = HTTPServer(("127.0.0.1", 0), make_review_handler(output))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/review.html", timeout=5) as response:
            review = response.read().decode()
            assert response.headers["Cache-Control"] == (
                "no-store, no-cache, must-revalidate"
            )
            assert response.headers["Pragma"] == "no-cache"
            assert response.headers["Expires"] == "0"
        target_url = re.search(r"<video[^>]+src='([^']+)'", review)
        assert target_url is not None
        with urlopen(base + "/" + target_url.group(1), timeout=5) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"].startswith("no-store")
        payload = json.dumps(
            {
                "sample_id": sample_id,
                "clip_uid": clip_uid,
                "request_fingerprint": fingerprint,
                "decision": "issue",
                "severity": "minor",
                "issue_tags": ["visual_caption_issue"],
                "notes": "Small visual omission.",
            }
        ).encode()
        request = Request(
            base + "/api/review",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            saved = json.loads(response.read())
        assert saved["annotation"]["decision"] == "issue"
        with urlopen(base + "/api/reviews", timeout=5) as response:
            reviews = json.loads(response.read())
        assert reviews["annotations"][0]["notes"] == "Small visual omission."
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_review_writes_use_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import r2v_data_v2.h3.qwen38_human_review as review_module

    output, _, _ = _review_output_fixture(tmp_path)
    sample_id, clip_uid, fingerprint = _review_identity(output)
    replacements: list[tuple[Path, Path]] = []
    original_replace = review_module.os.replace

    def recording_replace(source: object, destination: object) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(review_module.os, "replace", recording_replace)
    save_human_review(
        output,
        {
            "sample_id": sample_id,
            "clip_uid": clip_uid,
            "request_fingerprint": fingerprint,
            "decision": "skip",
            "severity": None,
            "issue_tags": [],
            "notes": "Not reviewable.",
        },
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert {destination.name for _, destination in replacements} == {
        "annotations.jsonl",
        "annotations.csv",
        "summary.json",
        "report.md",
    }
    assert not list((output / "human_review").glob(".*.tmp-*"))


def test_reviewed_output_cannot_be_overwritten(tmp_path: Path) -> None:
    output, manifest, _ = _review_output_fixture(tmp_path)
    sample_id, clip_uid, fingerprint = _review_identity(output)
    save_human_review(
        output,
        {
            "sample_id": sample_id,
            "clip_uid": clip_uid,
            "request_fingerprint": fingerprint,
            "decision": "pass",
            "severity": None,
            "issue_tags": [],
            "notes": "Approved.",
        },
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    records_before = (output / "records.jsonl").read_bytes()
    reviews_before = (output / "human_review/annotations.jsonl").read_bytes()
    blocked_backend = _FakeBackend(provenance=_provenance(tmp_path), calls=[])
    with pytest.raises(ValueError, match="reviewed Qwen3.8 output"):
        run_qwen38_h3_recaption_pilot(
            audio_production_root=tmp_path / "production",
            case_manifest_path=manifest,
            backend=blocked_backend,
            output_root=output,
            overwrite=True,
        )
    assert blocked_backend.calls == []
    assert (output / "records.jsonl").read_bytes() == records_before
    assert (output / "human_review/annotations.jsonl").read_bytes() == reviews_before
