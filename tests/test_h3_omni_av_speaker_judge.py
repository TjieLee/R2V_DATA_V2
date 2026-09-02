from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    OMNI_AV_SPEAKER_POLICY_VERSION,
    PASS1_PROMPT_VERSION,
    PASS1_SYSTEM_PROMPT,
    PASS2_PROMPT_VERSION,
    PASS2_SYSTEM_PROMPT,
    OmniAVCompletionDiagnostic,
    OmniAVSpeakerBackendProvenance,
    OmniAVSpeakerJudgeConfig,
    OmniAVSpeakerJudgeFailure,
    OmniAVSpeakerJudgeRequest,
    OmniAVSpeakerJudgeResult,
    OmniAVSpeakerObservation,
    OmniAVSpeakerPilotRecord,
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
from tools.render_h3_omni_av_speaker_media import _is_target_frame

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
        source_channels=1,
        backend="fixture",
        model_identifier="fixture/diarizen",
        model_fingerprint=_HASH,
        backend_configuration_fingerprint="b" * 64,
        input_preprocessing="h3_diarizen_native_16k_mono_passthrough_v1",
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
        source_sample_rate_hz=_SAMPLE_RATE,
        source_channels=1,
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
        source_sample_rate_hz=raw.source_sample_rate_hz,
        source_channels=raw.source_channels,
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
        frame_count=160000,
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


def _replace_segment_times(
    root: Path,
    *,
    segment_id: str,
    start_time: float,
    end_time: float,
) -> None:
    raw_path = root / "diarization/raw_segments.jsonl"
    bound_path = root / "diarization/bound_segments.jsonl"
    raws = [
        RawDiarizationSegment.model_validate_json(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    bounds = [
        BoundDiarizationSegment.model_validate_json(line)
        for line in bound_path.read_text(encoding="utf-8").splitlines()
    ]
    original = next(row for row in raws if row.segment_id == segment_id)
    replacement = _raw(
        segment_id,
        original.speaker_cluster_id,
        start_time,
        end_time,
        Path(original.source_audio_path),
    )
    raws = [replacement if row.segment_id == segment_id else row for row in raws]
    bounds = [
        BoundDiarizationSegment.model_validate(
            {
                **row.model_dump(mode="json"),
                "start_time": replacement.start_time,
                "end_time": replacement.end_time,
                "source_start_sample": replacement.source_start_sample,
                "source_end_sample": replacement.source_end_sample,
            }
        )
        if row.segment_id == segment_id
        else row
        for row in bounds
    ]
    _write_jsonl(raw_path, raws)
    _write_jsonl(bound_path, bounds)


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


def _openai_backend(
    tmp_path: Path,
    responses: list[str],
) -> tuple[OpenAIOmniAVSpeakerJudge, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    class _Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            content = responses.pop(0)
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
    return (
        OpenAIOmniAVSpeakerJudge(
            OmniAVSpeakerJudgeConfig(
                base_url="http://127.0.0.1:8091/v1",
                media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
            ),
            client=client,
        ),
        calls,
    )


def _judge_request(tmp_path: Path) -> OmniAVSpeakerJudgeRequest:
    video = tmp_path / "neutral.mp4"
    audio = tmp_path / "canonical.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    return OmniAVSpeakerJudgeRequest(
        neutral_video_path=video,
        canonical_audio_path=audio,
        target_start_in_window=0.75,
        target_end_in_window=1.15,
        visible_candidate_entity_ids=("e1", "e2"),
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
            OmniAVSpeakerObservation(
                decision="visible_entity", entity_id="e1", secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="visible_entity", entity_id="e1", secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="visible_entity", entity_id="e1", secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="visible_entity", entity_id="e1", secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="uncertain", entity_id=None, secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="offscreen", entity_id=None, secondary_speech_status="none"
            ),
            OmniAVSpeakerObservation(
                decision="offscreen", entity_id=None, secondary_speech_status="none"
            ),
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
    assert not records[0]["multiple_speakers_confirmed"]
    assert not records[0]["subject_entity_binding_excluded"]
    assert not records[0]["identity_specific_voice_products_excluded"]
    assert (
        records[0]["media_provenance"]["media_construction_policy"]
        == "neutral_faces_with_target_interval_marker_v1"
    )
    assert records[1]["comparison"] == "disagree"
    assert records[1]["primary_observation_stable"]
    assert records[1]["proposed_entity_id"] == "e1"
    assert records[1]["draft_entity_id"] == "e2"
    assert records[2]["comparison"] == "unresolved"
    assert not records[2]["primary_observation_stable"]
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
        assert video_call["window_start"] <= video_call["target_start"]
        assert video_call["target_end"] <= video_call["window_end"]
        assert "target_start" not in audio_call
        assert "target_end" not in audio_call
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
            OmniAVSpeakerObservation(
                decision="visible_entity", entity_id="e2", secondary_speech_status="none"
            ),
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


def test_clear_visible_primary_with_confirmed_incidental_speech_keeps_binding(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0002"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e2",
                secondary_speech_status="incidental",
            ),
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e2",
                secondary_speech_status="incidental",
            ),
        ]
    )

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=_FakeMedia(),
    )

    record = _records(root / "pilot/records.jsonl")[0]
    assert summary.pass2_case_count == 1
    assert record["primary_observation_stable"]
    assert record["secondary_speech_stable"]
    assert record["confirmed_secondary_speech_status"] == "incidental"
    assert record["proposed_entity_id"] == "e2"
    assert not record["subject_entity_binding_excluded"]
    assert record["identity_specific_voice_products_excluded"]


def test_non_speech_vocalization_does_not_create_secondary_speech() -> None:
    assert set(OmniAVSpeakerObservation.model_json_schema()["required"]) == {
        "decision",
        "entity_id",
        "secondary_speech_status",
    }
    observation = OmniAVSpeakerObservation(
        decision="visible_entity",
        entity_id="e1",
        secondary_speech_status="none",
    )
    assert observation.decision == "visible_entity"
    assert observation.secondary_speech_status == "none"
    for prompt in (PASS1_SYSTEM_PROMPT, PASS2_SYSTEM_PROMPT):
        normalized = prompt.lower()
        assert "non-linguistic" in normalized
        assert "sighing" in normalized
        assert "speech-turn ownership" in normalized
        assert "briefly audible" in normalized
        assert "frames marked target" in normalized
        assert "context only" in normalized
        assert "full context window" in normalized
    assert PASS1_PROMPT_VERSION.endswith("_v4")
    assert PASS2_PROMPT_VERSION.endswith("_v4")
    assert OMNI_AV_SPEAKER_POLICY_VERSION.endswith("_v4")


def test_true_competing_speech_excludes_subject_and_voice_products(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0002"])
    competing = OmniAVSpeakerObservation(
        decision="multiple_speakers",
        entity_id=None,
        secondary_speech_status="competing",
    )
    run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=_FakeBackend([competing, competing]),
        media_backend=_FakeMedia(),
    )
    record = _records(root / "pilot/records.jsonl")[0]

    assert record["multiple_speakers_confirmed"]
    assert record["subject_entity_binding_excluded"]
    assert record["identity_specific_voice_products_excluded"]
    assert record["proposed_entity_id"] is None
    for field in (
        "subject_entity_binding_excluded",
        "identity_specific_voice_products_excluded",
    ):
        with pytest.raises(ValueError, match="derived observation state"):
            OmniAVSpeakerPilotRecord.model_validate({**record, field: False})
    with pytest.raises(ValueError, match="derived observation state"):
        OmniAVSpeakerPilotRecord.model_validate(
            {**record, "proposed_entity_id": "e2"}
        )


def test_multiple_offscreen_speakers_remain_offscreen(tmp_path: Path) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0004"])
    offscreen = OmniAVSpeakerObservation(
        decision="offscreen",
        entity_id=None,
        secondary_speech_status="competing",
    )
    run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=_FakeBackend([offscreen, offscreen]),
        media_backend=_FakeMedia(),
    )
    record = _records(root / "pilot/records.jsonl")[0]

    assert record["primary_observation_stable"]
    assert record["proposed_non_entity_class"] == "offscreen"
    assert not record["multiple_speakers_confirmed"]
    assert not record["subject_entity_binding_excluded"]
    assert record["identity_specific_voice_products_excluded"]


def test_primary_agreement_survives_secondary_status_disagreement(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0002"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e2",
                secondary_speech_status="incidental",
            ),
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e2",
                secondary_speech_status="none",
            ),
        ]
    )
    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=_FakeMedia(),
    )
    record = _records(root / "pilot/records.jsonl")[0]

    assert summary.primary_observation_stable_count == 1
    assert summary.secondary_speech_stable_count == 0
    assert record["primary_observation_stable"]
    assert not record["secondary_speech_stable"]
    assert record["confirmed_secondary_speech_status"] is None
    assert record["proposed_entity_id"] == "e2"
    assert record["comparison"] == "agree"
    assert not record["subject_entity_binding_excluded"]
    assert not record["identity_specific_voice_products_excluded"]


@pytest.mark.parametrize(
    ("decision", "entity_id", "secondary_status"),
    [
        ("multiple_speakers", None, "none"),
        ("multiple_speakers", None, "incidental"),
        ("multiple_speakers", "e1", "competing"),
        ("visible_entity", None, "none"),
        ("visible_entity", "e1", "competing"),
    ],
)
def test_observation_schema_rejects_inconsistent_combinations(
    decision: str,
    entity_id: str | None,
    secondary_status: str,
) -> None:
    with pytest.raises(ValueError):
        OmniAVSpeakerObservation.model_validate(
            {
                "decision": decision,
                "entity_id": entity_id,
                "secondary_speech_status": secondary_status,
            }
        )


@pytest.mark.parametrize("overflow", [0.028118, 0.0366825, 0.0423195])
def test_synchronized_media_tail_overflow_is_clipped_for_model_only(
    tmp_path: Path,
    overflow: float,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    _replace_segment_times(
        root,
        segment_id="segment_0001",
        start_time=4.5,
        end_time=5.0 + overflow,
    )
    source_paths = [
        root / "diarization/raw_segments.jsonl",
        root / "diarization/bound_segments.jsonl",
    ]
    source_bytes = {path: path.read_bytes() for path in source_paths}
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0001"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e1",
                secondary_speech_status="none",
            )
        ]
    )
    media = _FakeMedia()

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=media,
    )

    record = _records(root / "pilot/records.jsonl")[0]
    request = backend.calls[0][0]
    assert summary.case_count == 1
    assert summary.succeeded_case_count == 1
    assert summary.failed_case_count == 0
    assert summary.model_call_count == 1
    assert record["absolute_segment_end"] == pytest.approx(5.0 + overflow)
    assert record["effective_absolute_segment_end"] == pytest.approx(5.0)
    assert record["target_boundary_clipped"]
    assert record["target_boundary_clip_seconds"] == pytest.approx(overflow)
    assert record["window_end"] == pytest.approx(5.0)
    assert request.target_end_in_window == pytest.approx(
        record["effective_absolute_segment_end"] - record["window_start"]
    )
    assert media.video_calls[0]["window_end"] == pytest.approx(5.0)
    assert media.video_calls[0]["target_start"] == pytest.approx(4.5)
    assert media.video_calls[0]["target_end"] == pytest.approx(5.0)
    assert media.audio_calls[0]["window_end"] == pytest.approx(5.0)
    assert all(path.read_bytes() == content for path, content in source_bytes.items())


def test_synchronized_media_in_boundary_target_is_not_clipped(tmp_path: Path) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    _replace_segment_times(
        root,
        segment_id="segment_0001",
        start_time=4.5,
        end_time=5.0,
    )
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0001"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e1",
                secondary_speech_status="none",
            )
        ]
    )

    run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=_FakeMedia(),
    )

    record = _records(root / "pilot/records.jsonl")[0]
    assert not record["target_boundary_clipped"]
    assert record["target_boundary_clip_seconds"] == 0
    assert record["absolute_segment_end"] == 5.0
    assert record["effective_absolute_segment_end"] == 5.0
    assert backend.calls[0][0].target_end_in_window == pytest.approx(
        5.0 - record["window_start"]
    )


def test_invalid_synchronized_boundary_fails_one_case_and_continues(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    _replace_segment_times(
        root,
        segment_id="segment_0001",
        start_time=4.5,
        end_time=5.5,
    )
    source_paths = [
        root / "diarization/raw_segments.jsonl",
        root / "diarization/bound_segments.jsonl",
    ]
    source_bytes = {path: path.read_bytes() for path in source_paths}
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0001", "segment_0002"])
    backend = _FakeBackend(
        [
            OmniAVSpeakerObservation(
                decision="visible_entity",
                entity_id="e2",
                secondary_speech_status="none",
            )
        ]
    )
    media = _FakeMedia()

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=media,
    )

    records = _records(root / "pilot/records.jsonl")
    assert summary.case_count == 2
    assert summary.failed_case_count == 1
    assert summary.succeeded_case_count == 1
    assert summary.model_call_count == 1
    assert records[0]["status"] == "failed"
    assert (
        records[0]["failure"]["code"]
        == "omni_av_synchronized_media_boundary_invalid"
    )
    assert records[0]["effective_absolute_segment_end"] is None
    assert records[0]["window_end"] is None
    assert records[1]["status"] == "succeeded"
    assert [call[0].target_end_in_window for call in backend.calls] == [
        pytest.approx(1.15)
    ]
    assert len(media.video_calls) == 1
    assert len(media.audio_calls) == 1
    assert all(path.read_bytes() == content for path, content in source_bytes.items())


def test_segment_starting_at_synchronized_end_fails_without_model_call(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    _replace_segment_times(
        root,
        segment_id="segment_0001",
        start_time=5.0,
        end_time=5.05,
    )
    manifest = tmp_path / "cases.jsonl"
    _manifest(manifest, ["segment_0001"])
    backend = _FakeBackend([])
    media = _FakeMedia()

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "pilot",
        backend=backend,
        media_backend=media,
    )

    record = _records(root / "pilot/records.jsonl")[0]
    assert summary.case_count == 1
    assert summary.failed_case_count == 1
    assert summary.succeeded_case_count == 0
    assert summary.model_call_count == 0
    assert record["status"] == "failed"
    assert "starts at or after" in record["failure"]["reason"]
    assert backend.calls == []
    assert media.video_calls == []
    assert media.audio_calls == []


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
                '{"decision":"visible_entity","entity_id":"unknown",'
                '"secondary_speech_status":"none"}'
                if len(calls) == 1
                else '{"decision":"visible_entity","entity_id":"e1",'
                '"secondary_speech_status":"none"}'
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


def test_exact_entity_decision_alias_normalizes_without_model_repair(
    tmp_path: Path,
) -> None:
    for entity_id in ("e1", "e2"):
        case_root = tmp_path / entity_id
        case_root.mkdir()
        backend, calls = _openai_backend(
            case_root,
            [
                json.dumps(
                    {
                        "decision": entity_id,
                        "entity_id": entity_id,
                        "secondary_speech_status": "none",
                    }
                )
            ],
        )

        result = backend.decide(_judge_request(case_root), verification=False)

        assert result.observation == OmniAVSpeakerObservation(
            decision="visible_entity",
            entity_id=entity_id,
            secondary_speech_status="none",
        )
        assert result.model_call_count == 1
        assert len(calls) == 1
        assert result.normalization_applied
        assert (
            result.normalization_kind
            == "entity_decision_alias_to_visible_entity"
        )


def test_entity_decision_alias_does_not_guess_conflicts_or_invisible_entities(
    tmp_path: Path,
) -> None:
    invalid_aliases = [
        {
            "decision": "e1",
            "entity_id": "e2",
            "secondary_speech_status": "none",
        },
        {
            "decision": "e9",
            "entity_id": "e9",
            "secondary_speech_status": "none",
        },
    ]
    for index, invalid in enumerate(invalid_aliases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        backend, calls = _openai_backend(
            case_root,
            [
                json.dumps(invalid),
                (
                    '{"decision":"uncertain","entity_id":null,'
                    '"secondary_speech_status":"none"}'
                ),
            ],
        )

        result = backend.decide(_judge_request(case_root), verification=False)

        assert result.observation == OmniAVSpeakerObservation(
            decision="uncertain",
            entity_id=None,
            secondary_speech_status="none",
        )
        assert result.model_call_count == 2
        assert len(calls) == 2
        assert not result.normalization_applied
        assert result.normalization_kind is None


def test_canonical_response_is_unchanged_and_true_malformed_output_repairs(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    backend, calls = _openai_backend(
        canonical_root,
        [
            (
                '{"decision":"visible_entity","entity_id":"e1",'
                '"secondary_speech_status":"none"}'
            )
        ],
    )
    canonical = backend.decide(_judge_request(canonical_root), verification=False)
    assert canonical.observation.entity_id == "e1"
    assert canonical.model_call_count == 1
    assert len(calls) == 1
    assert not canonical.normalization_applied

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    backend, calls = _openai_backend(
        malformed_root,
        [
            "not json",
            (
                '{"decision":"offscreen","entity_id":null,'
                '"secondary_speech_status":"none"}'
            ),
        ],
    )
    repaired = backend.decide(_judge_request(malformed_root), verification=False)
    assert repaired.observation.decision == "offscreen"
    assert repaired.model_call_count == 2
    assert len(calls) == 2
    assert not repaired.normalization_applied


def test_real_positive_alias_shape_uses_two_total_calls_and_stable_e1(
    tmp_path: Path,
) -> None:
    root, _, _ = _fixture(tmp_path / "production")
    manifest = tmp_path / "positive.jsonl"
    _manifest(manifest, ["segment_0002"])
    backend, calls = _openai_backend(
        tmp_path,
        [
            (
                '{"decision":"e1","entity_id":"e1",'
                '"secondary_speech_status":"none"}'
            ),
            (
                '{"decision":"e1","entity_id":"e1",'
                '"secondary_speech_status":"none"}'
            ),
        ],
    )

    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=root,
        case_manifest_path=manifest,
        output_root=root / "positive-pilot",
        backend=backend,
        media_backend=_FakeMedia(),
    )

    record = _records(root / "positive-pilot/records.jsonl")[0]
    assert len(calls) == 2
    assert summary.model_call_count == 2
    assert record["pass1_decision"] == "visible_entity"
    assert record["pass1_entity_id"] == "e1"
    assert record["pass2_decision"] == "visible_entity"
    assert record["pass2_entity_id"] == "e1"
    assert record["primary_observation_stable"]
    assert record["comparison"] == "disagree"
    assert record["proposed_entity_id"] == "e1"
    raw = json.loads(
        next((root / "positive-pilot/raw").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert raw["pass1"]["normalization_applied"]
    assert raw["pass2"]["normalization_applied"]


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
    assert '"TARGET"' in helper
    assert "draft_entity_id" not in helper


def test_target_marker_uses_exact_half_open_authoritative_interval() -> None:
    assert not _is_target_frame(0.749999, target_start=0.75, target_end=1.15)
    assert _is_target_frame(0.75, target_start=0.75, target_end=1.15)
    assert _is_target_frame(1.149999, target_start=0.75, target_end=1.15)
    assert not _is_target_frame(1.15, target_start=0.75, target_end=1.15)

    # A raw terminal overrun is clipped to the authoritative synchronized end.
    assert _is_target_frame(4.999999, target_start=4.5, target_end=5.0)
    assert not _is_target_frame(5.0, target_start=4.5, target_end=5.0)
