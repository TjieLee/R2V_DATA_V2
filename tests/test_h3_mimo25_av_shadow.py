from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
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
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoRecord,
    MimoReferenceImage,
    MimoSegmentEvidence,
    _inventory,
    _job,
    run_mimo25_av_reconcile,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MODEL,
    MIMO25_POLICY_VERSION,
    MIMO25_PROMPT_VERSION,
    MIMO25_SCHEMA_VERSION,
    SYSTEM_PROMPT,
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
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from r2v_data_v2.h3.mimo25_human_review import (
    MimoHumanReviewAnnotation,
    MimoReviewCase,
    MimoReviewStore,
    make_review_server,
    render_review_html,
)
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract


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
                    "<Picture 1>: fully_preserved - the person remains visible."
                ],
                "shots": [
                    {
                        "shot_index": 1,
                        "start_time": None,
                        "description_template": (
                            "<Subject 1> faces the camera. [[segment:segment_1]]"
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


class _Completions:
    def __init__(self, responses: list[tuple[str, int | None]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        raw, audio_tokens = self.responses.pop(0)
        details = None if audio_tokens is None else {"audio_tokens": audio_tokens, "video_tokens": 10}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=raw), finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": details,
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


def _backend(tmp_path: Path, responses: list[tuple[str, int | None]]) -> tuple[OpenAIMimo25Backend, _Completions]:
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
    assert request["extra_body"] == {"thinking": "disabled"}
    content = request["messages"][1]["content"]  # type: ignore[index]
    video = next(item for item in content if item["type"] == "video_url")  # type: ignore[union-attr]
    assert video["video_url"]["fps"] == 4.0
    assert video["video_url"]["media_resolution"] == "default"
    assert not any(item["type"] == "audio_url" for item in content)  # type: ignore[union-attr]
    assert [item["type"] for item in content[:2]] == ["text", "image_url"]  # type: ignore[index]
    assert "secret" not in json.dumps(request)
    assert MIMO25_PROMPT_VERSION in backend.provenance.model_dump_json()
    assert MIMO25_POLICY_VERSION in backend.provenance.model_dump_json()


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
    assert sum(item["type"] == "audio_url" for item in content) == 1  # type: ignore[union-attr]


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
    assert result.diagnostics[0].warning == "prompt_tokens_details_unavailable"


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


def test_true_malformed_output_uses_exactly_one_structure_repair(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    backend, completions = _backend(
        tmp_path,
        [("not json", 5), (_annotation().model_dump_json(), 0)],
    )
    result = backend.reconcile(
        job,
        segment_ids=["segment_1"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
    )
    assert result.model_call_count == 2
    assert result.repair_count == 1
    assert len(completions.requests) == 2
    assert completions.requests[1]["messages"][1]["content"].startswith(  # type: ignore[index]
        "Repair only JSON structure"
    )


def test_base64_oversize_fails_closed(tmp_path: Path) -> None:
    media = tmp_path / "large.mp4"
    media.write_bytes(b"1234")
    resolver = MimoMediaResolver(
        mode="base64", media_root=tmp_path, maximum_base64_bytes=4
    )
    with pytest.raises(MimoBackendFailure, match="Base64 media exceeds"):
        resolver.resolve(media)


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


def test_cross_reference_validation_rejects_inventory_drift() -> None:
    annotation = _annotation()
    issues = validate_annotation(
        annotation,
        segment_ids=["segment_1", "missing"],
        transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e2"},
        allowed_reference_labels={"<Picture 1>", "<Subject 1>"},
        target_duration_seconds=1.0,
    )
    assert {item.code for item in issues} == {"segment_inventory_mismatch", "unknown_entity"}


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


def _record_fixture(tmp_path: Path, annotation: MimoAVAnnotationDraft) -> MimoRecord:
    job = _job_fixture(tmp_path)
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    provenance = MimoBackendConfig(media_resolver=resolver, api_key="secret").provenance()
    values = {
        "schema_version": "r2v.h3.mimo25_record.1",
        "clip_uid": job.clip_uid,
        "request_fingerprint": job.request_fingerprint,
        "inventory_fingerprint": "a" * 64,
        "status": "ready",
        "backend_provenance": provenance.model_dump(mode="json"),
        "annotation": annotation.model_dump(mode="json"),
        "failure": None,
        "input_modality": "target_video_with_embedded_audio",
        "model_call_count": 1,
        "raw_response_count": 1,
        "http_retry_count": 0,
        "repair_count": 0,
    }
    fingerprint = __import__("hashlib").sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MimoRecord(**values, record_fingerprint=fingerprint)


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
    assert "<Subject 1> (S1)" in rendered
    assert not warnings or all("words" in item for item in warnings)


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
                        video_tokens=4,
                        audio_tokens=3,
                        cached_tokens=1,
                    ),
                    http_attempt_count=1,
                ),
            ),
            model_call_count=1,
            http_retry_count=0,
            repair_count=0,
            input_modality="target_video_with_embedded_audio",
        )


def test_shadow_runner_is_atomic_and_does_not_modify_inputs(tmp_path: Path) -> None:
    job = _job_fixture(tmp_path)
    inventory_values = {
        "schema_version": "r2v.h3.mimo25_inventory.1",
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
    assert summary.production_binding_modified is False
    assert summary.production_diarization_modified is False
    assert summary.production_asr_modified is False
    assert summary.production_h3_modified is False
    assert all(path.read_bytes() == value for path, value in before.items())
    published = "".join(path.read_text() for path in output.rglob("*.json*"))
    assert "never-persist-this" not in published
    assert "data:video" not in published


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
