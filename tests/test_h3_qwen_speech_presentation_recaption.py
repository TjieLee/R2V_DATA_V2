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
    FinalVisualReference,
)
from r2v_data_v2.h3.mimo25_backend import SpeechPresentation
from r2v_data_v2.h3.qwen38_h3_recaption import (
    QWEN38_RECAPTION_BACKEND_VERSION,
    QWEN38_RECAPTION_DRAFT_VERSION,
    QWEN38_RECAPTION_MATERIALIZER_VERSION,
    QWEN38_RECAPTION_POLICY_VERSION,
    QWEN38_RECAPTION_PROMPT_VERSION,
    UNGROUNDED_NON_DIEGETIC_MUSIC,
    UNGROUNDED_OVERALL_SOUNDSCAPE,
    Qwen38DraftShot,
    Qwen38H3DraftResponse,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRequest,
    RecaptionCompletionDiagnostic,
    build_audio_facts,
    build_reference_contract,
    materialize_h3_draft,
)
from r2v_data_v2.h3.qwen_speech_presentation_ab_review import (
    build_qwen_speech_presentation_ab_review,
)
from r2v_data_v2.h3.qwen_speech_presentation_recaption import (
    QWEN_PRESENTATION_BACKEND_VERSION,
    QWEN_PRESENTATION_DRAFT_VERSION,
    QWEN_PRESENTATION_MATERIALIZER_VERSION,
    QWEN_PRESENTATION_POLICY_VERSION,
    QWEN_PRESENTATION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    OpenAIQwenPresentationBackend,
    QwenPresentationAwareDraftResponse,
    QwenPresentationBackendResult,
    QwenPresentationConfig,
    QwenPresentationRequest,
    QwenSpeechPresentationDecision,
    build_corrected_audio_facts,
    materialize_presentation_draft,
    run_qwen_speech_presentation_recaption,
    validate_presentation_draft,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.speech_presentation import (
    render_speech_presentation_clause,
)
from tools.run_h3_qwen_speech_presentation_recaption import _parser


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample(tmp_path: Path, *, sample_id: str = "clip-1/canonical") -> FinalH3SampleV2:
    suffix = hashlib.sha256(sample_id.encode()).hexdigest()[:8]
    video = tmp_path / f"target-{suffix}.mp4"
    audio = tmp_path / f"audio-{suffix}.flac"
    image = tmp_path / f"image-{suffix}.png"
    video.write_bytes(b"video-" + sample_id.encode())
    audio.write_bytes(b"audio-" + sample_id.encode())
    image.write_bytes(b"image-" + sample_id.encode())
    clip_uid = sample_id.split("/", 1)[0]
    return FinalH3SampleV2(
        sample_id=sample_id,
        pair_id=f"canonical/{clip_uid}",
        pair_type="canonical",
        clip_uid=clip_uid,
        clip_display_path=f"01/show/season/episode/{clip_uid}",
        media_collection_relpath="01/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name=clip_uid,
        shard_id="shard-1",
        target_video=str(video),
        target_full_audio_path=str(audio),
        target_full_audio_sha256=_sha(audio.read_bytes()),
        r2v_instruction="Use Image 1 as the frozen visual reference.",
        visual_references=[
            FinalVisualReference(
                image_id="image_1",
                image_index=1,
                kind="subject",
                image_path="selected/image.png",
                image_artifact_path=str(image),
                entity_id="e1",
                source_frame_index=1,
                scope="full",
                visible_region="whole",
                synthetic=False,
            )
        ],
        subject_voices=[],
        speech_segments=[
            FinalQwen3SpeechSegment(
                segment_id="segment_1",
                speaker_cluster_id="cluster_a",
                entity_id="e1",
                entity_occurrence_id=f"{clip_uid}/e1",
                source_start_sample=3200,
                source_end_sample=19200,
                source_sample_rate_hz=32000,
                start_time=0.1,
                end_time=0.6,
                text="Keep the exact words.",
                language="English",
            )
        ],
    )


def _request(tmp_path: Path) -> QwenPresentationRequest:
    sample = _sample(tmp_path)
    case = Qwen38RecaptionManifestCase(
        sample_id=sample.sample_id,
        conditioning_variant="visual_only",
    )
    contract = build_reference_contract(sample, "visual_only")
    facts = build_audio_facts(sample, contract, None, semantics_records_sha256=None)
    facts = facts.model_copy(
        update={
            "non_speech_events": [],
            "overall_soundscape_hint": None,
            "non_diegetic_music_hint": None,
            "audio_grounding_complete": False,
            "provenance": {"speech": "current_h3_final_sample.speech_segments"},
        }
    )
    return QwenPresentationRequest(
        sample=sample,
        case=case,
        reference_contract=contract,
        original_audio_facts=facts,
        request_fingerprint="a" * 64,
    )


def _decision(
    *,
    presentation: SpeechPresentation = "onscreen_spoken",
    entity_id: str | None = "e1",
    evidence: list[str] | None = None,
) -> QwenSpeechPresentationDecision:
    return QwenSpeechPresentationDecision(
        fact_id="speech_1",
        segment_id="segment_1",
        speech_presentation=presentation,
        visible_entity_id=entity_id,
        evidence_codes=(
            ["visible_lip_motion"] if evidence is None else evidence
        ),
        confidence="high",
    )


def _draft(
    request: QwenPresentationRequest,
    decisions: list[QwenSpeechPresentationDecision] | None = None,
) -> QwenPresentationAwareDraftResponse:
    return QwenPresentationAwareDraftResponse(
        subject_definitions=["<Subject 1> is sourced from <Picture 1>."],
        summary="[reference generation] A person remains in the observed scene.",
        retention_analysis=[
            "<Subject 1> (appears in [Shot 1]): fully_preserved - the reference is retained."
        ],
        shots=[
            Qwen38DraftShot(
                shot_index=1,
                description_template="The person looks toward a phone. [[speech_1]]",
            )
        ],
        overall_soundscape=UNGROUNDED_OVERALL_SOUNDSCAPE,
        non_diegetic_music=UNGROUNDED_NON_DIEGETIC_MUSIC,
        audio_fact_audit=[],
        speech_presentations=[_decision()] if decisions is None else decisions,
    )


def _issue_codes(
    draft: QwenPresentationAwareDraftResponse,
    request: QwenPresentationRequest,
) -> set[str]:
    return {item.code for item in validate_presentation_draft(draft, request)}


def test_speech_presentation_reuses_mimo_enum_and_versions_are_independent() -> None:
    assert SpeechPresentation.__args__ == (
        "onscreen_spoken",
        "offscreen_spoken",
        "voice_over",
        "message_voice_over",
        "device_playback",
        "uncertain",
    )
    assert QWEN_PRESENTATION_PROMPT_VERSION == "h3_qwen_ref2va_speech_presentation_v1"
    assert QWEN_PRESENTATION_POLICY_VERSION.endswith("contract_v1")
    assert QWEN_PRESENTATION_DRAFT_VERSION.endswith("draft.1")
    assert QWEN_PRESENTATION_BACKEND_VERSION.endswith("backend.1")
    assert QWEN_PRESENTATION_MATERIALIZER_VERSION.endswith("materializer_v1")
    assert QWEN38_RECAPTION_PROMPT_VERSION == "h3_qwen38_ref2va_recaption_v6"
    assert QWEN38_RECAPTION_POLICY_VERSION == "h3_qwen38_ref2va_contract_v4"
    assert QWEN38_RECAPTION_DRAFT_VERSION == "r2v.h3.qwen38_recaption_draft.1"
    assert QWEN38_RECAPTION_BACKEND_VERSION == "r2v.h3.qwen38_recaption_backend.1"
    assert QWEN38_RECAPTION_MATERIALIZER_VERSION == "h3_qwen38_materializer_v2"


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        ([], "speech_presentation_inventory_mismatch"),
        ([_decision(), _decision()], "speech_presentation_inventory_mismatch"),
        (
            [
                _decision().model_copy(
                    update={"fact_id": "unknown", "segment_id": "unknown"}
                )
            ],
            "speech_presentation_inventory_mismatch",
        ),
    ],
)
def test_decision_inventory_must_be_exact(
    tmp_path: Path,
    decisions: list[QwenSpeechPresentationDecision],
    expected: str,
) -> None:
    request = _request(tmp_path)
    assert expected in _issue_codes(_draft(request, decisions), request)


def test_reordered_decisions_fail(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first = request.original_audio_facts.speech[0]
    second = first.model_copy(
        update={"fact_id": "speech_2", "segment_id": "segment_2", "start_time": 0.7, "end_time": 0.8}
    )
    request = QwenPresentationRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        original_audio_facts=request.original_audio_facts.model_copy(
            update={"speech": [first, second]}
        ),
        request_fingerprint=request.request_fingerprint,
    )
    decisions = [
        _decision().model_copy(update={"fact_id": "speech_2", "segment_id": "segment_2"}),
        _decision(),
    ]
    assert "speech_presentation_inventory_mismatch" in _issue_codes(
        _draft(request, decisions), request
    )


def test_presentation_hard_consistency() -> None:
    with pytest.raises(ValidationError, match="visible_lip_motion"):
        _decision(evidence=["visual_insufficient"])
    with pytest.raises(ValidationError, match="non-onscreen"):
        _decision(presentation="offscreen_spoken", entity_id="e1", evidence=["offscreen_visual_context"])
    with pytest.raises(ValidationError, match="non-onscreen"):
        _decision(presentation="uncertain", entity_id="e1", evidence=["visual_insufficient"])
    decision = _decision(entity_id=None)
    assert decision.speech_presentation == "onscreen_spoken"


def test_visible_entity_must_be_supplied(tmp_path: Path) -> None:
    request = _request(tmp_path)
    draft = _draft(request, [_decision(entity_id="e9")])
    assert "unknown_visible_entity" in _issue_codes(draft, request)


@pytest.mark.parametrize(
    ("presentation", "expected"),
    [
        ("onscreen_spoken", "<Subject 1> (S1) says, <d>[English] Keep the exact words.</d>"),
        ("offscreen_spoken", "(S1), speaking offscreen: <d>[English] Keep the exact words.</d>"),
        ("voice_over", "(S1), as a voice-over rather than visible speech: <d>[English] Keep the exact words.</d>"),
        ("message_voice_over", "(S1), as a message voice-over rather than visible speech: <d>[English] Keep the exact words.</d>"),
        ("device_playback", "(S1), heard through an in-scene device rather than visible speech: <d>[English] Keep the exact words.</d>"),
        ("uncertain", "(S1) says, <d>[English] Keep the exact words.</d>"),
    ],
)
def test_corrected_facts_and_materializer_preserve_authority(
    tmp_path: Path,
    presentation: SpeechPresentation,
    expected: str,
) -> None:
    request = _request(tmp_path)
    entity = "e1" if presentation == "onscreen_spoken" else None
    evidence = (
        ["visible_lip_motion"]
        if presentation == "onscreen_spoken"
        else ["visual_insufficient"]
    )
    decision = _decision(
        presentation=presentation,
        entity_id=entity,
        evidence=evidence,
    )
    draft = _draft(request, [decision])
    corrected, structured, _ = materialize_presentation_draft(draft, request)
    original = request.original_audio_facts.speech[0]
    updated = corrected.speech[0]
    assert (
        original.fact_id,
        original.segment_id,
        original.speaker_cluster_id,
        original.speaker_id,
        original.start_time,
        original.end_time,
        original.text,
        original.language,
        original.locked_dialogue_block,
    ) == (
        updated.fact_id,
        updated.segment_id,
        updated.speaker_cluster_id,
        updated.speaker_id,
        updated.start_time,
        updated.end_time,
        updated.text,
        updated.language,
        updated.locked_dialogue_block,
    )
    assert expected in structured.detailed_description
    if presentation == "uncertain":
        assert "presentation uncertain" not in structured.detailed_description
    if presentation != "onscreen_spoken":
        assert updated.entity_id is None
        assert "<Subject 1> (S1) says" not in structured.detailed_description


def test_onscreen_may_change_only_entity_binding(tmp_path: Path) -> None:
    request = _request(tmp_path)
    second_subject = request.reference_contract.subjects[0].model_copy(
        update={
            "subject_index": 2,
            "subject_label": "<Subject 2>",
            "entity_id": "e2",
        }
    )
    request = QwenPresentationRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract.model_copy(
            update={"subjects": [*request.reference_contract.subjects, second_subject]}
        ),
        original_audio_facts=request.original_audio_facts,
        request_fingerprint=request.request_fingerprint,
    )
    corrected = build_corrected_audio_facts(request, [_decision(entity_id="e2")])
    assert corrected.speech[0].entity_id == "e2"
    assert corrected.speech[0].entity_subject_label == "<Subject 2>"
    assert corrected.speech[0].speaker_id == "S1"


def test_shared_renderer_matches_mimo_v4_wording() -> None:
    speech = SimpleNamespace(
        speaker_id="S2",
        locked_dialogue_block="<d>[English] Exact.</d>",
    )
    assert render_speech_presentation_clause(
        speech=speech,
        base_clause="<Subject 1> (S2) says, <d>[English] Exact.</d>",
        presentation="message_voice_over",
    ) == "(S2), as a message voice-over rather than visible speech: <d>[English] Exact.</d>"


def test_existing_qwen_materializer_default_is_unchanged(tmp_path: Path) -> None:
    request = _request(tmp_path)
    old_request = Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=request.original_audio_facts,
        request_fingerprint=request.request_fingerprint,
    )
    parent = _draft(request)
    old_draft = Qwen38H3DraftResponse.model_validate(
        parent.model_dump(mode="json", exclude={"schema_version", "speech_presentations"})
    )
    default = materialize_h3_draft(old_draft, old_request)
    identity = materialize_h3_draft(
        old_draft,
        old_request,
        speech_clause_transform=lambda _speech, clause: clause,
    )
    assert default.model_dump(mode="json") == identity.model_dump(mode="json")


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.responses.pop(0)),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )


def _config(tmp_path: Path, model: str = "arbitrary/model") -> QwenPresentationConfig:
    return QwenPresentationConfig(
        base_url="http://127.0.0.1:8000/v1",
        media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        served_model_name=model,
        checkpoint_id=f"/models/{model}",
    )


def test_openai_request_is_model_neutral_visual_only_and_uses_one_repair(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    valid = _draft(request).model_dump_json()
    completions = _FakeCompletions(["not json", valid])
    backend = OpenAIQwenPresentationBackend(
        _config(tmp_path),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    result = backend.recaption(request)
    assert result.model_call_count == 2
    assert backend.provenance.served_model_name == "arbitrary/model"
    payload = completions.requests[0]
    content = payload["messages"][1]["content"]  # type: ignore[index]
    assert [item["type"] for item in content][:2] == ["text", "video_url"]  # type: ignore[index]
    assert sum(item["type"] == "image_url" for item in content) == 1  # type: ignore[index]
    assert all(item["type"] != "audio_url" for item in content)  # type: ignore[index]
    assert payload["modalities"] == ["text"]
    assert payload["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert "A visible person is NOT a visible speaker" in SYSTEM_PROMPT
    assert "no acoustic observation" in SYSTEM_PROMPT


def test_cli_defaults_match_old_ab_sampling(tmp_path: Path) -> None:
    arguments = _parser().parse_args(
        [
            "--audio-production-root",
            str(tmp_path),
            "--case-manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output-root",
            str(tmp_path / "out"),
            "--served-model-name",
            "model",
            "--checkpoint-id",
            "checkpoint",
            "--media-root",
            str(tmp_path),
        ]
    )
    assert (
        arguments.temperature,
        arguments.top_p,
        arguments.top_k,
        arguments.min_p,
        arguments.presence_penalty,
        arguments.repetition_penalty,
        arguments.max_tokens,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0, 8192)


@dataclass
class _FakeBackend:
    config: QwenPresentationConfig
    presentation: SpeechPresentation = "message_voice_over"

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def recaption(self, request: QwenPresentationRequest) -> QwenPresentationBackendResult:
        entity_id = "e1" if self.presentation == "onscreen_spoken" else None
        evidence = [
            "visible_lip_motion"
            if self.presentation == "onscreen_spoken"
            else "message_text_alignment"
        ]
        draft = _draft(
            request,
            [
                _decision(
                    presentation=self.presentation,
                    entity_id=entity_id,
                    evidence=evidence,
                )
            ],
        )
        corrected, response, warnings = materialize_presentation_draft(draft, request)
        return QwenPresentationBackendResult(
            draft=draft,
            corrected_audio_facts=corrected,
            response=response,
            raw_responses=(draft.model_dump_json(),),
            diagnostics=(RecaptionCompletionDiagnostic(finish_reason="stop"),),
            model_call_count=1,
            validation_warnings=tuple(warnings),
        )


def _write_stage_inputs(root: Path, samples: list[FinalH3SampleV2]) -> Path:
    h3 = root / "h3"
    h3.mkdir(parents=True)
    (h3 / "samples.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in samples
        ),
        encoding="utf-8",
    )
    manifest = root / "cases.jsonl"
    manifest.write_text(
        "".join(
            Qwen38RecaptionManifestCase(
                sample_id=item.sample_id,
                conditioning_variant="visual_only",
            ).model_dump_json()
            + "\n"
            for item in samples
        ),
        encoding="utf-8",
    )
    return manifest


def test_stage_preserves_manifest_order_and_never_modifies_h3(tmp_path: Path) -> None:
    production = tmp_path / "production"
    samples = [_sample(tmp_path, sample_id="clip-b/canonical"), _sample(tmp_path, sample_id="clip-a/canonical")]
    manifest = _write_stage_inputs(production, samples)
    source = production / "h3/samples.jsonl"
    before = source.read_bytes()
    output = tmp_path / "new"
    summary = run_qwen_speech_presentation_recaption(
        audio_production_root=production,
        case_manifest_path=manifest,
        output_root=output,
        backend=_FakeBackend(_config(tmp_path)),
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["clip-b/canonical", "clip-a/canonical"]
    assert source.read_bytes() == before
    assert summary.production_h3_modified is False
    assert summary.mimo_annotation_created is False
    assert summary.visible_entity_binding_removed_count == 2
    assert (output / "review.html").is_file()


def test_stage_rejects_noncanonical_or_nonvisual_manifest(tmp_path: Path) -> None:
    production = tmp_path / "production"
    sample = _sample(tmp_path)
    manifest = _write_stage_inputs(production, [sample])
    manifest.write_text(
        Qwen38RecaptionManifestCase(
            sample_id=sample.sample_id,
            conditioning_variant="target_voice_reference",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical visual_only"):
        run_qwen_speech_presentation_recaption(
            audio_production_root=production,
            case_manifest_path=manifest,
            output_root=tmp_path / "out",
            backend=_FakeBackend(_config(tmp_path)),
        )


def _write_old_root(
    root: Path,
    *,
    manifest: Path,
    new_root: Path,
    model_name: str,
) -> None:
    root.mkdir()
    (root / "manifest.jsonl").write_bytes(manifest.read_bytes())
    new_record = json.loads((new_root / "records.jsonl").read_text().splitlines()[0])
    old_record = {
        "sample_id": new_record["sample_id"],
        "clip_uid": new_record["clip_uid"],
        "status": "ready",
        "target_video_path": new_record["target_video_path"],
        "target_video_sha256": new_record["target_video_sha256"],
        "reference_contract": new_record["reference_contract"],
        "rendered_h3_prompt": f"{model_name} old prompt",
    }
    (root / "records.jsonl").write_text(json.dumps(old_record) + "\n")
    (root / "summary.json").write_text(
        json.dumps({"backend_provenance": {"served_model_name": model_name}})
    )


def test_four_way_review_validates_inventory_and_displays_real_models(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production"
    sample = _sample(tmp_path)
    manifest = _write_stage_inputs(production, [sample])
    q35_new = tmp_path / "q35-new"
    q38_new = tmp_path / "q38-new"
    run_qwen_speech_presentation_recaption(
        audio_production_root=production,
        case_manifest_path=manifest,
        output_root=q35_new,
        backend=_FakeBackend(_config(tmp_path, "Qwen3.5-397B-A17B")),
    )
    run_qwen_speech_presentation_recaption(
        audio_production_root=production,
        case_manifest_path=manifest,
        output_root=q38_new,
        backend=_FakeBackend(_config(tmp_path, "Qwen3.8-Flash-Next"), "onscreen_spoken"),
    )
    q35_old = tmp_path / "q35-old"
    q38_old = tmp_path / "q38-old"
    _write_old_root(q35_old, manifest=manifest, new_root=q35_new, model_name="Qwen3.5-397B-A17B")
    _write_old_root(q38_old, manifest=manifest, new_root=q38_new, model_name="Qwen3.8-Flash-Next")
    output = tmp_path / "review"
    summary = build_qwen_speech_presentation_ab_review(
        qwen35_old_root=q35_old,
        qwen35_new_root=q35_new,
        qwen38_old_root=q38_old,
        qwen38_new_root=q38_new,
        output_root=output,
    )
    page = (output / "review.html").read_text(encoding="utf-8")
    assert summary.case_count == 1
    assert "Qwen3.5-397B-A17B OLD" in page
    assert "Qwen3.8-Flash-Next NEW" in page
    assert "message_voice_over" in page
    assert "onscreen_spoken" in page
    assert "&lt;Subject 1&gt;" in page
    assert (output / "comparisons.jsonl").is_file()


def test_four_way_review_rejects_manifest_mismatch(tmp_path: Path) -> None:
    production = tmp_path / "production"
    sample = _sample(tmp_path)
    manifest = _write_stage_inputs(production, [sample])
    q35_new = tmp_path / "q35-new"
    q38_new = tmp_path / "q38-new"
    for root, model in ((q35_new, "m35"), (q38_new, "m38")):
        run_qwen_speech_presentation_recaption(
            audio_production_root=production,
            case_manifest_path=manifest,
            output_root=root,
            backend=_FakeBackend(_config(tmp_path, model)),
        )
    q35_old = tmp_path / "q35-old"
    q38_old = tmp_path / "q38-old"
    _write_old_root(q35_old, manifest=manifest, new_root=q35_new, model_name="m35")
    _write_old_root(q38_old, manifest=manifest, new_root=q38_new, model_name="m38")
    (q38_old / "manifest.jsonl").write_text(
        Qwen38RecaptionManifestCase(
            sample_id="other/canonical",
            conditioning_variant="visual_only",
        ).model_dump_json()
        + "\n"
    )
    with pytest.raises(ValueError, match="manifests differ"):
        build_qwen_speech_presentation_ab_review(
            qwen35_old_root=q35_old,
            qwen35_new_root=q35_new,
            qwen38_old_root=q38_old,
            qwen38_new_root=q38_new,
            output_root=tmp_path / "review",
        )
