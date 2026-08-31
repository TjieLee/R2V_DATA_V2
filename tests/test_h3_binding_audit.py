from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.audit_h3_speaker_bindings as audit_cli
from r2v_data_v2.h3.binding_audit import run_speaker_binding_audit
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationBoundaryReconciliation,
    DiarizationClusterBinding,
    DiarizationEntitySupport,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.pilot_schemas import (
    LRASDNativeArtifact,
    LRASDNativeSample,
    LRASDNativeTrack,
)
from r2v_data_v2.h3.schemas import (
    ASDModelProvenance,
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioEntityBinding,
    AudioTrackMetadata,
    BindingEvidence,
    EntityFaceAssociation,
    FaceGeometrySample,
    FaceTrack,
    H3AudioBindingIR,
    H3TaskSpecification,
)

_SAMPLE_RATE = 16000
_HASH = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw(
    cluster_id: str,
    segment_id: str,
    start: float,
    end: float,
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
        source_audio_path="/fixture/audio.wav",
        source_audio_sha256=_HASH,
        source_sample_rate_hz=_SAMPLE_RATE,
        backend="fixture_diarizen",
        model_identifier="fixture/model",
        model_fingerprint=_HASH,
        backend_configuration_fingerprint="b" * 64,
        boundary_reconciliation=DiarizationBoundaryReconciliation(
            adjusted=False,
            end_clamped=False,
            end_overrun_samples=0,
            end_overrun_seconds=0,
        ),
    )


def _support(entity_id: str, seconds: float) -> DiarizationEntitySupport:
    samples = round(seconds * _SAMPLE_RATE)
    return DiarizationEntitySupport(
        entity_id=entity_id,
        direct_support_samples=samples,
        direct_support_seconds=samples / _SAMPLE_RATE,
        weighted_support=samples * 0.95,
        contributing_binding_count=1,
    )


def _cluster(
    cluster_id: str,
    *,
    segments: list[RawDiarizationSegment],
    status: str,
    entity_id: str | None,
    supports: list[DiarizationEntitySupport],
    contested: float = 0.0,
) -> DiarizationClusterBinding:
    duration = sum(item.end_time - item.start_time for item in segments)
    direct = sum(item.direct_support_seconds for item in supports)
    total_weight = sum(item.weighted_support for item in supports)
    top1 = supports[0] if supports else None
    top2 = supports[1] if len(supports) > 1 else None
    return DiarizationClusterBinding(
        target_clip_uid="clip-a",
        speaker_cluster_id=cluster_id,
        status=status,
        entity_id=entity_id,
        cluster_segment_count=len(segments),
        cluster_speaker_seconds=duration,
        usable_anchor_sample_count=round(direct * _SAMPLE_RATE),
        usable_anchor_duration=direct,
        contested_anchor_sample_count=round(contested * _SAMPLE_RATE),
        contested_anchor_duration=contested,
        unmatched_anchor_sample_count=0,
        unmatched_anchor_duration=0,
        entity_supports=supports,
        top1_entity_id=top1.entity_id if top1 else None,
        top1_support=top1.weighted_support if top1 else 0,
        top2_support=top2.weighted_support if top2 else 0,
        top1_share=top1.weighted_support / total_weight if top1 else None,
        top1_top2_margin=(
            top1.weighted_support - (top2.weighted_support if top2 else 0)
            if top1
            else None
        ),
        visual_anchor_coverage_ratio=direct / duration if duration else 0,
        warnings=[],
        reason_codes=([] if status == "candidate_mapped" else [f"fixture_{status}"]),
    )


def _bound(
    raw: RawDiarizationSegment,
    *,
    status: str,
    entity_id: str | None,
    direct: float,
) -> BoundDiarizationSegment:
    return BoundDiarizationSegment(
        target_clip_uid=raw.target_clip_uid,
        segment_id=raw.segment_id,
        speaker_cluster_id=raw.speaker_cluster_id,
        start_time=raw.start_time,
        end_time=raw.end_time,
        source_start_sample=raw.source_start_sample,
        source_end_sample=raw.source_end_sample,
        cluster_binding_status=status,
        entity_id=entity_id,
        entity_occurrence_id=(f"clip-a/{entity_id}" if entity_id else None),
        direct_anchor_samples=round(direct * _SAMPLE_RATE),
        direct_anchor_seconds=direct,
        identity_scope=(
            "unresolved"
            if entity_id is None
            else "direct_anchor_present"
            if direct
            else "cluster_propagated_only"
        ),
    )


def _binding(
    start: float,
    end: float,
    entity_id: str,
    face_track_id: str,
) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        entity_id=entity_id,
        face_track_id=face_track_id,
        status="bound",
        confidence=0.95,
        evidence=BindingEvidence(
            active_face_track_ids=[face_track_id],
            face_speaking_probabilities={face_track_id: 0.95},
            association_confidence=0.95,
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=True,
        ),
    )


def _track(track_id: str, frame_index: int) -> FaceTrack:
    return FaceTrack(
        face_track_id=track_id,
        start_time=0,
        end_time=7,
        sample_count=1,
        mean_detection_confidence=0.95,
        geometry_samples=[
            FaceGeometrySample(
                frame_index=frame_index,
                timestamp=frame_index / 25,
                bbox_xyxy=(1, 1, 10, 10),
                confidence=0.95,
            )
        ],
    )


def _native_track(track_id: str, frames: list[int]) -> LRASDNativeTrack:
    return LRASDNativeTrack(
        face_track_id=track_id,
        samples=[
            LRASDNativeSample(
                frame_index=frame,
                timestamp_seconds=frame / 25,
                bbox_xyxy=(1, 1, 10, 10),
                detection_confidence=0.95,
                raw_class1_logit=1,
                backend_native_active=True,
            )
            for frame in frames
        ],
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[Path, dict[str, bytes]]:
    audio = root / "audio"
    diarization = root / "diarization"
    (audio / "clips" / "clip-a").mkdir(parents=True)
    (audio / "runtime" / "clip-a" / "lr_asd").mkdir(parents=True)
    diarization.mkdir(parents=True)

    raw = [
        _raw("speaker_0", "segment_0001", 0, 1),
        _raw("speaker_1", "segment_0002", 1, 1.5),
        _raw("speaker_1", "segment_0009", 1.5, 2),
        _raw("speaker_2", "segment_0003", 2, 3),
        _raw("speaker_3", "segment_0004", 3, 3.2),
        _raw("speaker_3", "segment_0005", 3.2, 4),
        _raw("speaker_4", "segment_0006", 4, 5),
        _raw("speaker_5", "segment_0007", 5, 6),
        _raw("speaker_6", "segment_0008", 6, 7),
    ]
    raw_by_cluster: dict[str, list[RawDiarizationSegment]] = {}
    for item in raw:
        raw_by_cluster.setdefault(item.speaker_cluster_id, []).append(item)
    clusters = [
        _cluster(
            "speaker_0",
            segments=raw_by_cluster["speaker_0"],
            status="candidate_mapped",
            entity_id="e1",
            supports=[_support("e1", 0.4)],
        ),
        _cluster(
            "speaker_1",
            segments=raw_by_cluster["speaker_1"],
            status="ambiguous",
            entity_id=None,
            supports=[_support("e1", 0.2), _support("e2", 0.2)],
        ),
        _cluster(
            "speaker_2",
            segments=raw_by_cluster["speaker_2"],
            status="candidate_mapped",
            entity_id="e1",
            supports=[_support("e1", 0.2)],
        ),
        _cluster(
            "speaker_3",
            segments=raw_by_cluster["speaker_3"],
            status="candidate_mapped",
            entity_id="e1",
            supports=[_support("e1", 0.2)],
        ),
        _cluster(
            "speaker_4",
            segments=raw_by_cluster["speaker_4"],
            status="candidate_mapped",
            entity_id="e1",
            supports=[_support("e1", 0.04)],
        ),
        _cluster(
            "speaker_5",
            segments=raw_by_cluster["speaker_5"],
            status="conflict",
            entity_id=None,
            supports=[_support("e2", 0.2)],
            contested=0.04,
        ),
        _cluster(
            "speaker_6",
            segments=raw_by_cluster["speaker_6"],
            status="unbound",
            entity_id=None,
            supports=[],
        ),
    ]
    bound = [
        _bound(raw[0], status="candidate_mapped", entity_id="e1", direct=0.4),
        _bound(raw[1], status="ambiguous", entity_id=None, direct=0.2),
        _bound(raw[2], status="ambiguous", entity_id=None, direct=0.2),
        _bound(raw[3], status="candidate_mapped", entity_id="e1", direct=0.2),
        _bound(raw[4], status="candidate_mapped", entity_id="e1", direct=0.2),
        _bound(raw[5], status="candidate_mapped", entity_id="e1", direct=0),
        _bound(raw[6], status="candidate_mapped", entity_id="e1", direct=0.04),
        _bound(raw[7], status="conflict", entity_id=None, direct=0.2),
        _bound(raw[8], status="unbound", entity_id=None, direct=0),
    ]
    bindings = [
        _binding(0, 0.4, "e1", "face_1"),
        _binding(1, 1.2, "e1", "face_1"),
        _binding(1.5, 1.7, "e2", "face_2"),
        _binding(2, 2.2, "e1", "face_1"),
        _binding(3, 3.2, "e1", "face_1"),
        _binding(4, 4.04, "e1", "face_1"),
        _binding(5, 5.2, "e2", "face_2"),
    ]
    tracks = [_track("face_1", 0), _track("face_2", 30), _track("face_3", 60)]
    associations = [
        EntityFaceAssociation(
            face_track_id="face_1",
            entity_id="e1",
            confidence=0.95,
            method="test_fixture",
            status="matched",
        ),
        EntityFaceAssociation(
            face_track_id="face_2",
            entity_id="e2",
            confidence=0.95,
            method="test_fixture",
            status="matched",
        ),
        EntityFaceAssociation(
            face_track_id="face_3",
            confidence=0,
            method="test_fixture",
            status="unmatched",
            reason="no entity match",
        ),
    ]
    sidecar = AudioBindingSidecar(
        clip_uid="clip-a",
        source_run_root="/fixture/visual",
        source_video_path="/fixture/video.mp4",
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid="clip-a",
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path="/fixture/video.mp4",
                full_audio_path="/fixture/audio.wav",
                duration_seconds=7,
                sample_rate_hz=_SAMPLE_RATE,
                channels=1,
            ),
            face_tracks=tracks,
            associations=associations,
        ),
        bindings=bindings,
        h3_ir=H3AudioBindingIR(
            clip_uid="clip-a",
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[],
            subjects=[],
            audio_assets=[],
            bindings=bindings,
        ),
    )
    sidecar_path = audio / "clips" / "clip-a" / "audio_binding.json"
    sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    native = LRASDNativeArtifact(
        clip_uid="clip-a",
        source_video_path="/fixture/video.mp4",
        model_video_path="/fixture/model.mp4",
        audio_path="/fixture/audio.wav",
        model_provenance=ASDModelProvenance(
            backend="lr_asd",
            model_identifier="fixture/lr-asd",
            checkpoint_path="/fixture/model.pth",
            checkpoint_sha256=_HASH,
        ),
        width=100,
        height=100,
        duration_seconds=7,
        tracks=[
            _native_track("face_1", [*range(13), *range(25, 37), *range(50, 57)]),
            _native_track("face_2", [30, *range(37, 50), *range(125, 130)]),
            _native_track("face_3", [30, 52, *range(57, 75)]),
        ],
    )
    native_path = audio / "runtime" / "clip-a" / "lr_asd" / "lr_asd_native.json"
    native_path.write_text(native.model_dump_json(indent=2) + "\n", encoding="utf-8")
    source_paths = {
        "raw": diarization / "raw_segments.jsonl",
        "clusters": diarization / "cluster_bindings.jsonl",
        "bound": diarization / "bound_segments.jsonl",
        "summary": diarization / "summary.json",
        "sidecar": sidecar_path,
        "native": native_path,
    }
    _write_jsonl(source_paths["raw"], raw)
    _write_jsonl(source_paths["clusters"], clusters)
    _write_jsonl(source_paths["bound"], bound)
    source_paths["summary"].write_text('{"fixture":true}\n', encoding="utf-8")
    return root, {str(path): path.read_bytes() for path in source_paths.values()}


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_model_free_binding_audit_emits_structural_evidence_without_mutation(
    tmp_path: Path,
) -> None:
    root, source_bytes = _fixture(tmp_path / "audio-run")

    summary = run_speaker_binding_audit(audio_production_root=root)

    output = root / "binding_audit_v1"
    paths = jea_production_paths(root)
    assert paths.audio == root / "audio"
    assert paths.diarization == root / "diarization"
    assert not (root / "production").exists()
    clusters = {
        row["speaker_cluster_id"]: row for row in _rows(output / "clusters.jsonl")
    }
    segments = {row["segment_id"]: row for row in _rows(output / "segments.jsonl")}

    clean = clusters["speaker_0"]
    assert clean["exclusive_active_faces"] == [
        {
            "face_track_id": "face_1",
            "entity_id": "e1",
            "overlap_seconds": pytest.approx(0.52),
        }
    ]
    assert not clean["flags"]["exclusive_active_entity_contradiction"]

    contradiction = clusters["speaker_1"]
    assert contradiction["flags"]["multiple_exclusive_active_face_tracks"]
    assert contradiction["flags"]["exclusive_active_entity_contradiction"]
    assert contradiction["flags"]["multiple_direct_entity_support"]
    assert {item["entity_id"] for item in contradiction["exclusive_active_entities"]} == {
        "e1",
        "e2",
    }

    unmatched = clusters["speaker_2"]
    assert unmatched["unmatched_exclusive_active_face_track_ids"] == ["face_3"]
    assert unmatched["flags"]["mapped_entity_vs_unmatched_face_contradiction"]
    assert unmatched["current_mapping_status"] == "candidate_mapped"
    assert unmatched["current_entity_id"] == "e1"
    assert unmatched["multi_active_lr_asd_seconds"] == pytest.approx(0.04)
    assert unmatched["multi_active_frame_count"] == 1
    assert unmatched["max_simultaneous_active_face_count"] == 2
    assert unmatched["multi_active_face_track_ids"] == ["face_1", "face_3"]
    assert unmatched["multi_active_mapped_entity_ids"] == ["e1"]
    assert unmatched["multi_active_unmatched_face_track_ids"] == ["face_3"]
    assert unmatched["flags"]["has_multi_active_lr_asd_frames"]
    assert unmatched["flags"][
        "multi_active_contains_current_entity_and_other_face"
    ]
    face_1_exclusive = next(
        item
        for item in unmatched["exclusive_active_faces"]
        if item["face_track_id"] == "face_1"
    )
    assert face_1_exclusive["overlap_seconds"] == pytest.approx(0.24)

    assert contradiction["multi_active_lr_asd_seconds"] == pytest.approx(0.04)
    assert contradiction["multi_active_frame_count"] == 1
    assert contradiction["max_simultaneous_active_face_count"] == 3
    assert contradiction["multi_active_face_track_ids"] == [
        "face_1",
        "face_2",
        "face_3",
    ]
    assert contradiction["multi_active_mapped_entity_ids"] == ["e1", "e2"]
    assert contradiction["multi_active_unmatched_face_track_ids"] == ["face_3"]
    assert segments["segment_0002"]["flags"]["has_multi_active_lr_asd_frames"]
    assert segments["segment_0002"]["max_simultaneous_active_face_count"] == 3
    assert not segments["segment_0009"]["flags"]["has_multi_active_lr_asd_frames"]
    assert segments["segment_0009"]["multi_active_frame_count"] == 0

    propagated = clusters["speaker_3"]
    assert propagated["fully_propagated_seconds"] == pytest.approx(0.8)
    assert propagated["flags"]["has_fully_propagated_segments"]
    assert segments["segment_0005"]["flags"]["fully_propagated_segment"]
    assert segments["segment_0005"]["direct_anchor_seconds"] == 0
    assert not segments["segment_0002"]["flags"]["multiple_direct_entity_support"]
    assert not segments["segment_0009"]["flags"]["multiple_direct_entity_support"]
    assert segments["segment_0002"]["direct_support_seconds_by_entity"] == {
        "e1": pytest.approx(0.2)
    }
    assert segments["segment_0009"]["direct_support_seconds_by_entity"] == {
        "e2": pytest.approx(0.2)
    }

    tiny = clusters["speaker_4"]
    assert tiny["current_mapping_status"] == "candidate_mapped"
    assert tiny["direct_anchor_seconds"] == pytest.approx(0.04)
    assert tiny["direct_support_ratio"] == pytest.approx(0.04)
    assert tiny["longest_direct_anchor_seconds"] == pytest.approx(0.04)
    assert tiny["shortest_direct_anchor_seconds"] == pytest.approx(0.04)

    assert clusters["speaker_5"]["current_mapping_status"] == "conflict"
    assert clusters["speaker_5"]["flags"]["contested_anchor"]
    assert not segments["segment_0007"]["flags"]["contested_anchor"]
    assert segments["segment_0007"]["contested_anchor_seconds"] == 0
    assert clusters["speaker_6"]["current_mapping_status"] == "unbound"
    assert summary.model_call_count == 0
    assert summary.bindings_modified_count == 0
    assert summary.cluster_count == 7
    assert summary.segment_count == 9
    assert summary.clusters_with_exclusive_active_entity_contradiction == 1
    assert summary.clusters_with_mapped_entity_vs_unmatched_face == 1
    assert summary.clusters_with_fully_propagated_segments == 1
    assert summary.clusters_with_multi_active_lr_asd_frames == 2
    assert summary.clusters_with_current_entity_plus_other_active_face == 1
    assert summary.multi_active_lr_asd_speaker_seconds == pytest.approx(0.08)
    assert len(summary.audio_binding_input_set_sha256) == 64
    assert len(summary.lr_asd_native_input_set_sha256) == 64
    assert "cluster_propagated_only" not in summary.review_priority_counts
    sidecar_path = root / "audio" / "clips" / "clip-a" / "audio_binding.json"
    native_path = (
        root / "audio" / "runtime" / "clip-a" / "lr_asd" / "lr_asd_native.json"
    )
    assert summary.audio_binding_input_set_sha256 == hashlib.sha256(
        json.dumps(
            [{"clip_uid": "clip-a", "artifact_sha256": _sha256(sidecar_path)}],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert summary.lr_asd_native_input_set_sha256 == hashlib.sha256(
        json.dumps(
            [{"clip_uid": "clip-a", "artifact_sha256": _sha256(native_path)}],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    for path, before in source_bytes.items():
        assert Path(path).read_bytes() == before


def test_binding_audit_review_order_is_deterministic_and_non_gating(
    tmp_path: Path,
) -> None:
    root, _ = _fixture(tmp_path / "audio-run")
    run_speaker_binding_audit(audio_production_root=root)
    output = root / "binding_audit_v1"

    review = _rows(output / "review_manifest.jsonl")

    assert [row["review_index"] for row in review] == list(range(1, 8))
    assert review[0]["review_priority"] == "conflict"
    assert review[1]["review_priority"] == "exclusive_active_entity_contradiction"
    assert review[2]["review_priority"] == (
        "mapped_entity_vs_unmatched_face_contradiction"
    )
    assert review[3]["review_priority"] == "unbound_or_ambiguous"
    candidate_rows = [
        row for row in _rows(output / "clusters.jsonl")
        if row["current_mapping_status"] == "candidate_mapped"
    ]
    assert len(candidate_rows) == 4
    assert all(row["current_entity_id"] == "e1" for row in candidate_rows)


def test_binding_audit_cli_prints_summary_and_requires_explicit_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _fixture(tmp_path / "audio-run")

    result = audit_cli.main(["--audio-production-root", str(root)])

    assert result["model_call_count"] == 0
    assert json.loads(capsys.readouterr().out)["cluster_count"] == 7
    with pytest.raises(FileExistsError, match="already exists"):
        audit_cli.main(["--audio-production-root", str(root)])
    audit_cli.main(["--audio-production-root", str(root), "--overwrite"])


def test_binding_audit_fails_closed_on_cross_artifact_identity_drift(
    tmp_path: Path,
) -> None:
    root, _ = _fixture(tmp_path / "audio-run")
    bound_path = root / "diarization" / "bound_segments.jsonl"
    rows = _rows(bound_path)
    rows.pop()
    bound_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw and bound segment identities differ"):
        run_speaker_binding_audit(audio_production_root=root)
