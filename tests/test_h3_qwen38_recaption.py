from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalQwen3SpeechSegment,
    FinalSubjectVoice,
    FinalVisualReference,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    QWEN38_RECAPTION_PROMPT_VERSION,
    AudioFactAuditItem,
    OpenAIQwen38RecaptionBackend,
    Qwen38BackendProvenance,
    Qwen38BackendResult,
    Qwen38H3StructuredResponse,
    Qwen38RecaptionConfig,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRequest,
    RecaptionCompletionDiagnostic,
    RecaptionNonSpeechFact,
    build_audio_facts,
    build_qwen38_pilot_manifest,
    build_reference_contract,
    render_h3_prompt,
    run_qwen38_h3_recaption_pilot,
    validate_h3_response,
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
            source_start_sample=1600,
            source_end_sample=3200,
            source_sample_rate_hz=16000,
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
            source_start_sample=4800,
            source_end_sample=6400,
            source_sample_rate_hz=16000,
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


def _response(request: Qwen38RecaptionRequest) -> Qwen38H3StructuredResponse:
    definitions = []
    for subject in request.reference_contract.subjects:
        definitions.append(
            f"{subject.subject_label} is the frozen referenced content sourced from "
            + " and ".join(subject.source_picture_labels)
            + "."
        )
    for audio in request.reference_contract.audios:
        owner = "" if audio.subject_label is None else f" for {audio.subject_label}"
        speaker = "" if audio.speaker_id is None else f" ({audio.speaker_id})"
        definitions.append(
            f"{audio.audio_label} is the supplied audio condition{owner}{speaker}."
        )
    retention = [
        f"{subject.subject_label} (appears in [Shot 1]): fully_preserved - the referenced content is retained."
        for subject in request.reference_contract.subjects
    ]
    retention.extend(
        f"{audio.audio_label}: {audio.retention_marker} - the declared audio role is retained."
        for audio in request.reference_contract.audios
    )
    speech_parts = []
    for fact in request.audio_facts.speech:
        source = (
            f"{fact.entity_subject_label} ({fact.speaker_id})"
            if fact.entity_subject_label is not None
            else f"An unidentified voice ({fact.speaker_id})"
        )
        speech_parts.append(f"{source} says, {fact.locked_dialogue_block}")
    audits = [
        AudioFactAuditItem(fact_id=item.fact_id, action="preserved")
        for item in request.audio_facts.non_speech_events
    ]
    return Qwen38H3StructuredResponse(
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
        detailed_description=(
            "The target uses a natural observational style.\n[Shot 1] "
            + " ".join(speech_parts)
        ),
        overall_soundscape="Only the supplied speech and grounded events are asserted.",
        non_diegetic_music="Audio grounding is incomplete; music is not asserted.",
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
    assert "An unidentified voice (S1)" in response.detailed_description
    assert "<Subject 1> (S2)" in response.detailed_description
    assert _codes(response, request) == set()


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
    valid = _response(request).model_dump_json()
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
    payload = completions.requests[0]
    content = payload["messages"][1]["content"]  # type: ignore[index]
    types = [item["type"] for item in content]  # type: ignore[index]
    assert types[:2] == ["text", "video_url"]
    assert types.count("image_url") == 3
    assert "audio_url" not in types
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
    assert "video_fps" not in provenance.model_dump(mode="json")


def test_cli_exposes_sglang_sampling_without_video_fps(tmp_path: Path) -> None:
    arguments = _parser().parse_args(["--audio-production-root", str(tmp_path)])
    assert arguments.min_p == 0.0
    assert arguments.repetition_penalty == 1.0
    assert "video_fps" not in vars(arguments)


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
    valid = _response(request).model_dump_json()
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
    assert list((output / "raw_responses").glob("*.json"))
    review = (output / "review.html").read_text(encoding="utf-8")
    assert "media/0000/target.mp4" in review
    assert "media/0000/picture-1.png" in review
    assert "file://" not in review


def test_prompt_version_is_frozen() -> None:
    assert QWEN38_RECAPTION_PROMPT_VERSION == "h3_qwen38_ref2va_recaption_v1"
