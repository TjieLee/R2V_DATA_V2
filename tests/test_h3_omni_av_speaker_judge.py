from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationBoundaryReconciliation,
    DiarizationClusterBinding,
    DiarizationEntitySupport,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.omni_av_speaker_judge import (
    DEFAULT_MODEL,
    OmniAVCompletionDiagnostic,
    OmniAVSpeakerBackendProvenance,
    OmniAVSpeakerJudgeConfig,
    OmniAVSpeakerJudgeFailure,
    OmniAVSpeakerJudgeRequest,
    OmniAVSpeakerJudgeResult,
    OmniAVSpeakerObservation,
    OpenAIOmniAVSpeakerJudge,
    build_neutral_face_timeline,
    run_omni_av_speaker_judge_pilot,
)
from r2v_data_v2.h3.pilot_schemas import (
    LRASDNativeArtifact,
    LRASDNativeSample,
    LRASDNativeTrack,
)
from r2v_data_v2.h3.schemas import (
    ASDModelProvenance,
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceGeometrySample,
    FaceTrack,
    H3AudioBindingIR,
    H3TaskSpecification,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver

_HASH = "a" * 64
_SAMPLE_RATE = 16000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw(
    segment_id: str,
    cluster_id: str,
    start: float,
    end: float,
    source_audio: Path,
) -> RawDiarizationSegment:
    start_sample = round(start * _SAMPLE_RATE)
    end_sample = round(end * _SAMPLE_RATE)
    return RawDiarizationSegment(
        target_clip_uid="clip-a",
        segment_id=segment_id,
        speaker_cluster_id=cluster_id,
        backend_speaker_label=cluster_id,
        backend_reported_start_time=start,
        backend_reported_end_time=end,
        backend_reported_start_sample=start_sample,
        backend_reported_end_sample=end_sample,
        start_time=start,
        end_time=end,
        source_start_sample=start_sample,
        source_end_sample=end_sample,
        source_audio_path=str(source_audio),
        source_audio_sha256=_sha256(source_audio),
        source_sample_rate_hz=_SAMPLE_RATE,
        backend="fixture",
        model_identifier="fixture/diarizen",
        model_fingerprint=_HASH,
        backend_configuration_fingerprint="b" * 64,
        boundary_reconciliation=DiarizationBoundaryReconciliation(
            adjusted=False,
            end_clamped=False,
            end_overrun_samples=0,
            end_overrun_seconds=0,
        ),
    )


def _cluster(
    raw: RawDiarizationSegment,
    *,
    status: str,
    entity_id: str | None,
) -> DiarizationClusterBinding:
    support = (
        []
        if entity_id is None
        else [
            DiarizationEntitySupport(
                entity_id=entity_id,
                direct_support_samples=1600,
                direct_support_seconds=0.1,
                weighted_support=1520,
                contributing_binding_count=1,
            )
        ]
    )
    return DiarizationClusterBinding(
        target_clip_uid="clip-a",
        speaker_cluster_id=raw.speaker_cluster_id,
        status=status,
        entity_id=entity_id,
        cluster_segment_count=1,
        cluster_speaker_seconds=raw.end_time - raw.start_time,
        usable_anchor_sample_count=1600 if support else 0,
        usable_anchor_duration=0.1 if support else 0,
        contested_anchor_sample_count=0,
        contested_anchor_duration=0,
        unmatched_anchor_sample_count=0,
        unmatched_anchor_duration=0,
        entity_supports=support,
        top1_entity_id=entity_id,
        top1_support=1520 if support else 0,
        top2_support=0,
        top1_share=1 if support else None,
        top1_top2_margin=1520 if support else None,
        visual_anchor_coverage_ratio=0.25 if support else 0,
        reason_codes=[] if status == "candidate_mapped" else ["fixture_conflict"],
    )


def _bound(
    raw: RawDiarizationSegment,
    *,
    status: str,
    entity_id: str | None,
) -> BoundDiarizationSegment:
    return BoundDiarizationSegment(
        target_clip_uid="clip-a",
        segment_id=raw.segment_id,
        speaker_cluster_id=raw.speaker_cluster_id,
        start_time=raw.start_time,
        end_time=raw.end_time,
        source_start_sample=raw.source_start_sample,
        source_end_sample=raw.source_end_sample,
        cluster_binding_status=status,
        entity_id=entity_id,
        entity_occurrence_id=(None if entity_id is None else f"clip-a/{entity_id}"),
        direct_anchor_samples=1600 if entity_id else 0,
        direct_anchor_seconds=0.1 if entity_id else 0,
        identity_scope="direct_anchor_present" if entity_id else "unresolved",
    )


def _face_track(track_id: str) -> FaceTrack:
    return FaceTrack(
        face_track_id=track_id,
        start_time=0,
        end_time=5,
        sample_count=1,
        mean_detection_confidence=0.9,
        geometry_samples=[
            FaceGeometrySample(
                frame_index=0,
                timestamp=0,
                bbox_xyxy=(1, 1, 20, 20),
                confidence=0.9,
            )
        ],
    )


def _native_track(track_id: str, bbox: tuple[float, float, float, float]) -> LRASDNativeTrack:
    return LRASDNativeTrack(
        face_track_id=track_id,
        samples=[
            LRASDNativeSample(
                frame_index=frame,
                timestamp_seconds=frame / 25,
                bbox_xyxy=bbox,
                detection_confidence=0.9,
                raw_class1_logit=(1 if frame % 2 else -1),
                backend_native_active=bool(frame % 2),
            )
            for frame in range(125)
        ],
    )


def _fixture(root: Path) -> tuple[Path, list[RawDiarizationSegment], dict[str, bytes]]:
    audio = root / "audio"
    diarization = root / "diarization"
    runtime = audio / "runtime" / "clip-a" / "lr_asd"
    clip_root = audio / "clips" / "clip-a"
    for path in (audio, diarization, runtime, clip_root):
        path.mkdir(parents=True, exist_ok=True)
    target_video = root / "target.mp4"
    model_video = root / "model.mp4"
    canonical_audio = root / "canonical.flac"
    source_audio = root / "diarization.wav"
    target_video.write_bytes(b"target-video")
    model_video.write_bytes(b"model-video")
    canonical_audio.write_bytes(b"canonical-audio")
    source_audio.write_bytes(b"source-audio")

    specs = [
        ("segment_0001", "speaker_0", 0.5, 0.9, "candidate_mapped", "e1"),
        ("segment_0002", "speaker_1", 1.2, 1.6, "candidate_mapped", "e2"),
        ("segment_0003", "speaker_2", 2.0, 2.4, "candidate_mapped", "e2"),
        ("segment_0004", "speaker_3", 3.0, 3.4, "conflict", None),
    ]
    raws = [_raw(*spec[:4], source_audio) for spec in specs]
    clusters = [
        _cluster(raw, status=spec[4], entity_id=spec[5])
        for raw, spec in zip(raws, specs, strict=True)
    ]
    bounds = [
        _bound(raw, status=spec[4], entity_id=spec[5])
        for raw, spec in zip(raws, specs, strict=True)
    ]
    _write_jsonl(diarization / "raw_segments.jsonl", raws)
    _write_jsonl(diarization / "cluster_bindings.jsonl", clusters)
    _write_jsonl(diarization / "bound_segments.jsonl", bounds)

    associations = [
        EntityFaceAssociation(
            face_track_id="face_1",
            status="matched",
            entity_id="e1",
            confidence=0.95,
            method="test_fixture",
        ),
        EntityFaceAssociation(
            face_track_id="face_2",
            status="matched",
            entity_id="e2",
            confidence=0.95,
            method="test_fixture",
        ),
        EntityFaceAssociation(
            face_track_id="face_3",
            status="unmatched",
            confidence=0,
            method="test_fixture",
            reason="no match",
        ),
    ]
    sidecar = AudioBindingSidecar(
        clip_uid="clip-a",
        source_run_root="/fixture/visual",
        source_video_path=str(target_video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid="clip-a",
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(target_video),
                full_audio_path=str(source_audio),
                duration_seconds=5,
                sample_rate_hz=_SAMPLE_RATE,
                channels=1,
            ),
            face_tracks=[
                _face_track("face_1"),
                _face_track("face_2"),
                _face_track("face_3"),
            ],
            associations=associations,
        ),
        bindings=[],
        h3_ir=H3AudioBindingIR(
            clip_uid="clip-a",
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[],
            subjects=[],
            audio_assets=[],
            bindings=[],
        ),
    )
    sidecar_path = clip_root / "audio_binding.json"
    sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    native = LRASDNativeArtifact(
        clip_uid="clip-a",
        source_video_path=str(target_video),
        model_video_path=str(model_video),
        audio_path=str(source_audio),
        model_provenance=ASDModelProvenance(
            backend="lr_asd",
            model_identifier="fixture/lr-asd",
            checkpoint_path="/fixture/model.pth",
            checkpoint_sha256=_HASH,
        ),
        width=100,
        height=80,
        duration_seconds=5,
        tracks=[
            _native_track("face_1", (1, 1, 20, 30)),
            _native_track("face_2", (30, 1, 50, 30)),
            _native_track("face_3", (60, 1, 80, 30)),
        ],
    )
    native_path = runtime / "lr_asd_native.json"
    native_path.write_text(native.model_dump_json(indent=2) + "\n", encoding="utf-8")
    canonical = CanonicalAudioClip(
        clip_uid="clip-a",
        clip_display_path="01/show/episode/clip-a.mp4",
        media_collection_relpath="01/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-a.mp4",
        shard_id="01",
        target_video_path=str(target_video),
        target_video_sha256=_sha256(target_video),
        target_full_audio_path=str(canonical_audio),
        target_full_audio_sha256=_sha256(canonical_audio),
        target_duration_seconds=5,
        subject_reference_count=2,
        target_audio_binding_path=str(sidecar_path),
        target_audio_binding_sha256=_sha256(sidecar_path),
    )
    _write_jsonl(audio / "canonical_clips.jsonl", [canonical])
    sources = [
        diarization / "raw_segments.jsonl",
        diarization / "cluster_bindings.jsonl",
        diarization / "bound_segments.jsonl",
        audio / "canonical_clips.jsonl",
        sidecar_path,
        native_path,
        target_video,
        model_video,
        canonical_audio,
        source_audio,
    ]
    return root, raws, {str(path): path.read_bytes() for path in sources}


def _manifest(path: Path, segment_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "r2v.h3.omni_av_speaker_judge_manifest.1",
                    "clip_uid": "clip-a",
                    "segment_id": segment_id,
                    "human_label": (
                        {"decision": "visible_entity", "entity_id": "e1"}
                        if segment_id == "segment_0002"
                        else None
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for segment_id in segment_ids
        ),
        encoding="utf-8",
    )


def _provenance() -> OmniAVSpeakerBackendProvenance:
    return OmniAVSpeakerBackendProvenance(
        served_model_name=DEFAULT_MODEL,
        checkpoint_id=DEFAULT_MODEL,
        base_url="http://127.0.0.1:8091/v1",
        media_mode="file",
        media_root="/fixture",
        max_tokens=512,
        configuration_fingerprint="c" * 64,
    )


class _FakeMedia:
    def __init__(self) -> None:
        self.video_calls: list[dict[str, object]] = []
        self.audio_calls: list[dict[str, object]] = []

    def render_neutral_video(self, **kwargs: object) -> None:
        self.video_calls.append(kwargs)
        Path(kwargs["destination_path"]).write_bytes(b"neutral-video")

    def trim_canonical_audio(self, **kwargs: object) -> None:
        self.audio_calls.append(kwargs)
        Path(kwargs["destination_path"]).write_bytes(b"trimmed-audio")


class _FakeBackend:
    def __init__(self, responses: list[OmniAVSpeakerObservation | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[OmniAVSpeakerJudgeRequest, bool]] = []

    @property
    def provenance(self) -> OmniAVSpeakerBackendProvenance:
        return _provenance()

    def decide(
        self,
        request: OmniAVSpeakerJudgeRequest,
        *,
        verification: bool,
    ) -> OmniAVSpeakerJudgeResult:
        self.calls.append((request, verification))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return OmniAVSpeakerJudgeResult(
            observation=response,
            raw_responses=(response.model_dump_json(),),
            completion_diagnostics=(OmniAVCompletionDiagnostic(finish_reason="stop"),),
            model_call_count=1,
        )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_read_only_pilot_runs_blind_two_pass_policy_and_preserves_sources(
    tmp_path: Path,
) -> None:
    root, _, source_bytes = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(
        manifest,
        ["segment_0001", "segment_0002", "segment_0003", "segment_0004"],
    )
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(decision="visible_entity", entity_id="e1"),
            OmniAVSpeakerObservation(decision="visible_entity", entity_id="e1"),
            OmniAVSpeakerObservation(decision="visible_entity", entity_id="e1"),
            OmniAVSpeakerObservation(decision="visible_entity", entity_id="e1"),
            OmniAVSpeakerObservation(decision="uncertain", entity_id=None),
            OmniAVSpeakerObservation(decision="offscreen", entity_id=None),
            OmniAVSpeakerObservation(decision="offscreen", entity_id=None),
        ]
    )
    media = _FakeMedia()

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=None,
        backend=backend,
        media_backend=media,
    )

    output = root / "omni_av_speaker_judge_pilot_v1"
    records = _records(output / "records.jsonl")
    assert summary.case_count == 4
    assert summary.pass2_case_count == 3
    assert summary.model_call_count == 7
    assert [verification for _, verification in backend.calls] == [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert records[0]["comparison"] == "agree"
    assert not records[0]["pass2_called"]
    assert records[1]["comparison"] == "disagree"
    assert records[1]["stable_observation"]
    assert records[1]["proposed_entity_id"] == "e1"
    assert records[1]["draft_entity_id"] == "e2"
    assert records[2]["comparison"] == "unresolved"
    assert not records[2]["stable_observation"]
    assert records[2]["proposed_entity_id"] is None
    assert records[3]["comparison"] == "draft_unresolved"
    assert records[3]["proposed_non_entity_class"] == "offscreen"
    assert all(call[0].visible_candidate_entity_ids == ("e1", "e2") for call in backend.calls)
    assert all(not hasattr(call[0], "draft_entity_id") for call in backend.calls)
    assert all(not hasattr(call[0], "human_label") for call in backend.calls)
    assert all(not hasattr(call[0], "asr_transcript") for call in backend.calls)
    for video_call, audio_call in zip(media.video_calls, media.audio_calls, strict=True):
        assert video_call["window_start"] == audio_call["window_start"]
        assert video_call["window_end"] == audio_call["window_end"]
        timeline = video_call["timeline"]
        payload = timeline.model_dump(mode="json")
        assert {sample["label"] for sample in payload["samples"]} == {
            "e1",
            "e2",
            "OTHER",
        }
        serialized = json.dumps(payload)
        assert "backend_native_active" not in serialized
        assert "raw_class1_logit" not in serialized
        assert "draft" not in serialized
    assert all(Path(path).read_bytes() == content for path, content in source_bytes.items())
    assert summary.bindings_modified_count == 0
    assert summary.diarization_bindings_modified_count == 0


def test_per_case_model_failure_does_not_destroy_other_successes(tmp_path: Path) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0001", "segment_0002"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerJudgeFailure(
                code="omni_av_structured_output_failed",
                reason="fixture failure",
                raw_responses=("", "bad"),
                model_call_count=2,
            ),
            OmniAVSpeakerObservation(decision="visible_entity", entity_id="e2"),
        ]
    )

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=_FakeMedia(),
    )

    records = _records(root / "pilot" / "records.jsonl")
    assert summary.failed_case_count == 1
    assert summary.succeeded_case_count == 1
    assert records[0]["status"] == "failed"
    assert records[0]["failure"]["code"] == "omni_av_structured_output_failed"
    assert records[1]["status"] == "succeeded"
    raw = json.loads(
        next((root / "pilot" / "raw").glob("0001_*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert raw["failure"]["raw_responses"] == ["", "bad"]


def test_openai_request_is_blind_synchronized_text_only_and_repairs_once(
    tmp_path: Path,
) -> None:
    video = tmp_path / "neutral.mp4"
    audio = tmp_path / "canonical.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    calls: list[dict[str, object]] = []

    class _Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            content = (
                '{"decision":"visible_entity","entity_id":"unknown"}'
                if len(calls) == 1
                else '{"decision":"visible_entity","entity_id":"e1"}'
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    backend = OpenAIOmniAVSpeakerJudge(
        OmniAVSpeakerJudgeConfig(
            base_url="http://127.0.0.1:8091/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        ),
        client=client,
    )
    result = backend.decide(
        OmniAVSpeakerJudgeRequest(
            neutral_video_path=video,
            canonical_audio_path=audio,
            target_start_in_window=0.75,
            target_end_in_window=1.15,
            visible_candidate_entity_ids=("e1", "e2"),
        ),
        verification=False,
    )

    assert result.model_call_count == 2
    assert result.observation.entity_id == "e1"
    for call in calls:
        assert call["temperature"] == 0
        assert call["modalities"] == ["text"]
        assert call["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        content = call["messages"][1]["content"]
        assert [item["type"] for item in content] == [
            "text",
            "video_url",
            "audio_url",
        ]
        request_text = content[0]["text"].lower()
        for forbidden in (
            "draft",
            "candidate_mapped",
            "lr-asd",
            "active flag",
            "human_label",
            "transcript",
            "asr",
        ):
            assert forbidden not in request_text


def test_neutral_timeline_and_renderer_never_expose_speaking_state(tmp_path: Path) -> None:
    root, raws, _ = _fixture(tmp_path / "production")
    sidecar = AudioBindingSidecar.model_validate_json(
        (root / "audio/clips/clip-a/audio_binding.json").read_text(encoding="utf-8")
    )
    native = LRASDNativeArtifact.model_validate_json(
        (root / "audio/runtime/clip-a/lr_asd/lr_asd_native.json").read_text(
            encoding="utf-8"
        )
    )
    timeline = build_neutral_face_timeline(
        native=native,
        sidecar=sidecar,
        window_start=raws[0].start_time,
        window_end=raws[0].end_time,
    )
    payload = timeline.model_dump(mode="json")
    assert {item["label"] for item in payload["samples"]} == {"e1", "e2", "OTHER"}
    assert "backend_native_active" not in json.dumps(payload)
    helper = (
        Path(__file__).resolve().parents[1]
        / "tools/render_h3_omni_av_speaker_media.py"
    ).read_text(encoding="utf-8")
    assert "backend_native_active" not in helper
    assert "raw_class1_logit" not in helper
    assert "(0, 210, 0)" not in helper
