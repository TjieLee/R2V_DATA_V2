from __future__ import annotations

import builtins
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3.asr_transcription import build_asr_inventory
from r2v_data_v2.h3.audio_production import (
    H3ProductionInPair,
    H3ProductionInPairSubject,
)
from r2v_data_v2.h3.diarization_binding import (
    DIARIZATION_HUMAN_QA_LABELS,
    DiarizationBackendFailure,
    DiarizationBackendProvenance,
    DiarizationBackendSegment,
    DiarizationBoundaryReconciliation,
    DiarizationHumanQAExport,
    DiarizationHumanQALabel,
    DiarizationTargetClip,
    PersistentDiariZenBackend,
    RawDiarizationSegment,
    _mapped_identity_duration_metrics,
    _normalize_segments,
    bind_diarization_segments,
    build_diarization_inventory,
    diarization_output_root,
    run_diarization_binding_pilot,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioEntityBinding,
    AudioTrackMetadata,
    BindingEvidence,
    H3AudioBindingIR,
    H3TaskSpecification,
    PictureAsset,
    SemanticSubject,
)
from tools.run_h3_diarization_binding import main as diarization_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, *, frames: int = 16000, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (np.arange(frames, dtype=np.int32) % 2000 - 1000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(values.tobytes())


def _evidence(
    *,
    synchronization_plausible: bool = True,
    audio_quality_usable: bool = True,
    clean_training_eligible: bool = True,
) -> BindingEvidence:
    return BindingEvidence(
        active_face_track_ids=["face_1"],
        face_speaking_probabilities={"face_1": 0.9},
        association_confidence=0.95,
        audio_quality_usable=audio_quality_usable,
        synchronization_plausible=synchronization_plausible,
        clean_training_eligible=clean_training_eligible,
    )


def _binding(
    start: float,
    end: float,
    entity_id: str = "e1",
    *,
    confidence: float = 0.93,
    synchronization_plausible: bool = True,
    audio_quality_usable: bool = True,
    clean_training_eligible: bool = True,
) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        entity_id=entity_id,
        face_track_id="face_1",
        status="bound",
        confidence=confidence,
        evidence=_evidence(
            synchronization_plausible=synchronization_plausible,
            audio_quality_usable=audio_quality_usable,
            clean_training_eligible=clean_training_eligible,
        ),
    )


def _nonbound(start: float, end: float, status: str) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        status=status,
        confidence=0.9,
        evidence=BindingEvidence(
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=False,
        ),
    )


def _write_pair(root: Path, clip_uid: str) -> H3ProductionInPair:
    media = root / "fixture_media"
    video = media / f"{clip_uid}.mp4"
    audio = media / f"{clip_uid}.wav"
    visual = media / f"{clip_uid}.png"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f"video:{clip_uid}".encode())
    visual.write_bytes(b"fixture-image")
    _write_wav(audio)
    bindings = [_binding(0.2, 0.4)]
    sidecar = AudioBindingSidecar(
        clip_uid=clip_uid,
        source_run_root="/frozen/visual",
        source_video_path=str(video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid=clip_uid,
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(video),
                full_audio_path=str(audio),
                duration_seconds=1.0,
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
        bindings=bindings,
        h3_ir=H3AudioBindingIR(
            clip_uid=clip_uid,
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[
                PictureAsset(
                    picture_id="picture_1",
                    entity_id="e1",
                    path=str(visual),
                )
            ],
            subjects=[
                SemanticSubject(
                    subject_id="subject_1",
                    entity_id="e1",
                    reference_type="subject",
                    phrase="person",
                    source_assets=["picture_1"],
                )
            ],
            audio_assets=[],
            bindings=bindings,
        ),
    )
    sidecar_path = root / "production" / "audio" / clip_uid / "audio_binding.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return H3ProductionInPair(
        pair_id=f"in_pair/{clip_uid}",
        target_clip_uid=clip_uid,
        target_video_path=str(video),
        target_full_audio_path=str(audio),
        target_audio_binding_path=str(sidecar_path),
        subjects=[
            H3ProductionInPairSubject(
                subject_index=1,
                target_occurrence_id=f"{clip_uid}/e1",
                target_entity_id="e1",
                target_visual_reference_path=str(visual),
                target_primary_voice_reference_path=f"voice/{clip_uid}/e1.flac",
            )
        ],
    )


def _audio_run(tmp_path: Path) -> tuple[Path, list[str]]:
    pairs_root = tmp_path / "production" / "pairs"
    pairs_root.mkdir(parents=True)
    clip_ids = [f"clip-{index:03d}" for index in range(75)]
    pairs = [_write_pair(tmp_path, clip_uid) for clip_uid in clip_ids]
    (pairs_root / "in_pairs.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in reversed(pairs)),
        encoding="utf-8",
    )
    (pairs_root / "cross_pairs.jsonl").write_text(
        json.dumps(
            {
                "target_clip_uid": "clip-000",
                "donor_clip_uid": "must-not-be-diarized",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    asr = build_asr_inventory(pairs_root=pairs_root, mode="pilot20")
    asr_root = tmp_path / "asr_pilot20"
    asr_root.mkdir()
    (asr_root / "inventory.json").write_text(
        asr.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return tmp_path, [item.target_clip_uid for item in asr.targets]


def _provenance() -> DiarizationBackendProvenance:
    return DiarizationBackendProvenance(
        backend="fake_diarizen",
        model_identifier="fixture/diarizen-v2",
        model_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )


class _FakeBackend:
    def __init__(
        self,
        responses: dict[str, list[DiarizationBackendSegment]] | None = None,
    ) -> None:
        self.provenance = _provenance()
        self.responses = responses or {}
        self.calls: list[tuple[str, Path]] = []

    def diarize(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
    ) -> list[DiarizationBackendSegment]:
        self.calls.append((clip_uid, audio_path))
        return list(
            self.responses.get(
                clip_uid,
                [
                    DiarizationBackendSegment(
                        start_time=0.0,
                        end_time=0.8,
                        speaker_label="vendor-speaker-A",
                    )
                ],
            )
        )


def _target(*, clip_uid: str = "clip-a", rate: int = 100) -> DiarizationTargetClip:
    return DiarizationTargetClip(
        target_clip_uid=clip_uid,
        target_video_path=f"/{clip_uid}.mp4",
        source_audio_path=f"/{clip_uid}.wav",
        source_audio_sha256="c" * 64,
        source_sample_rate_hz=rate,
        source_channels=1,
        source_frame_count=1000,
        target_audio_binding_path=f"/{clip_uid}.json",
        visual_references=[],
    )


def _raw(
    cluster: str,
    start: float,
    end: float,
    *,
    index: int,
    target: DiarizationTargetClip | None = None,
) -> RawDiarizationSegment:
    active = target or _target()
    return RawDiarizationSegment(
        target_clip_uid=active.target_clip_uid,
        segment_id=f"segment_{index:04d}",
        speaker_cluster_id=cluster,
        backend_speaker_label=cluster,
        backend_reported_start_time=start,
        backend_reported_end_time=end,
        backend_reported_start_sample=round(start * active.source_sample_rate_hz),
        backend_reported_end_sample=round(end * active.source_sample_rate_hz),
        start_time=start,
        end_time=end,
        source_start_sample=round(start * active.source_sample_rate_hz),
        source_end_sample=round(end * active.source_sample_rate_hz),
        source_audio_path=active.source_audio_path,
        source_audio_sha256=active.source_audio_sha256,
        source_sample_rate_hz=active.source_sample_rate_hz,
        backend="fake_diarizen",
        model_identifier="fixture/diarizen-v2",
        model_fingerprint="a" * 64,
        backend_configuration_fingerprint="b" * 64,
        boundary_reconciliation=DiarizationBoundaryReconciliation(
            adjusted=False,
            end_clamped=False,
            end_overrun_samples=0,
            end_overrun_seconds=0.0,
        ),
    )


def test_inventory_reuses_exact_asr_pilot_order_and_production_is_complete(
    tmp_path: Path,
) -> None:
    root, expected = _audio_run(tmp_path)

    pilot = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    production = build_diarization_inventory(
        audio_run_root=root,
        mode="production",
    )

    assert [item.target_clip_uid for item in pilot.targets] == expected
    assert pilot.selected_target_count == 20
    assert pilot.source_target_count == 75
    assert pilot.source_asr_inventory_fingerprint
    assert production.selected_target_count == 75
    assert production.parent_quota_applied is False
    assert production.production_blocked is True


def test_dry_run_imports_no_diarizen_and_real_production_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _audio_run(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "diarizen" or name.startswith("diarizen."):
            raise AssertionError("dry-run imported DiariZen")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = diarization_main(
        ["--audio-run-root", str(root), "--mode", "production", "--dry-run"]
    )
    assert result["selected_target_count"] == 75
    assert result["parent_quota_applied"] is False
    assert result["cross_pair_jobs_created"] == 0
    with pytest.raises(
        ValueError,
        match="production_blocked_pending_diarization_binding_calibration",
    ):
        diarization_main(["--audio-run-root", str(root), "--mode", "production"])


def test_pilot_calls_backend_once_per_target_and_never_for_cross_pairs(
    tmp_path: Path,
) -> None:
    root, expected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    backend = _FakeBackend()
    frozen_before = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=diarization_output_root(root, mode="pilot20"),
        backend=backend,
    )

    assert [item[0] for item in backend.calls] == expected
    assert summary.backend_call_count == 20
    assert summary.schema_version == "r2v.h3.diarization_summary.2"
    assert summary.boundary_adjusted_segment_count == 0
    assert summary.boundary_adjusted_clip_count == 0
    assert summary.total_end_overrun_seconds == 0
    assert summary.max_end_overrun_seconds == 0
    assert summary.max_end_overrun_samples == 0
    assert summary.median_positive_end_overrun_seconds is None
    assert all(item[0] != "must-not-be-diarized" for item in backend.calls)
    frozen_after = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "diarization_pilot20/" not in path.relative_to(root).as_posix()
    }
    assert frozen_after == frozen_before
    raw_rows = [
        json.loads(line)
        for line in (root / "diarization_pilot20" / "raw_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        (item["target_clip_uid"], item["segment_id"]) for item in raw_rows
    ] == sorted((item["target_clip_uid"], item["segment_id"]) for item in raw_rows)


def test_overlapping_segments_are_preserved_and_cluster_ids_are_clip_local() -> None:
    target = _target()
    normalized = _normalize_segments(
        target=target,
        segments=[
            DiarizationBackendSegment(
                start_time=0.0, end_time=2.0, speaker_label="alpha"
            ),
            DiarizationBackendSegment(
                start_time=1.0, end_time=3.0, speaker_label="beta"
            ),
        ],
        provenance=_provenance(),
    )
    other = _normalize_segments(
        target=_target(clip_uid="clip-b"),
        segments=[
            DiarizationBackendSegment(
                start_time=0.0, end_time=1.0, speaker_label="unrelated-label"
            )
        ],
        provenance=_provenance(),
    )

    assert [(item.start_time, item.end_time) for item in normalized] == [
        (0.0, 2.0),
        (1.0, 3.0),
    ]
    assert [item.speaker_cluster_id for item in normalized] == [
        "speaker_0",
        "speaker_1",
    ]
    assert other[0].speaker_cluster_id == "speaker_0"


def test_terminal_segments_intersect_canonical_source_and_preserve_provenance() -> None:
    target = _target()
    normalized = _normalize_segments(
        target=target,
        segments=[
            DiarizationBackendSegment(
                start_time=1.0,
                end_time=2.0,
                speaker_label="inside",
            ),
            DiarizationBackendSegment(
                start_time=2.0,
                end_time=10.0,
                speaker_label="exact-eof",
            ),
            DiarizationBackendSegment(
                start_time=7.92,
                end_time=30.0,
                speaker_label="crossing-a",
            ),
            DiarizationBackendSegment(
                start_time=8.0,
                end_time=11.25,
                speaker_label="crossing-b",
            ),
        ],
        provenance=_provenance(),
    )

    inside, exact, crossing_a, crossing_b = normalized
    assert inside.source_end_sample == 200
    assert inside.boundary_reconciliation.adjusted is False
    assert exact.source_end_sample == target.source_frame_count
    assert exact.boundary_reconciliation.adjusted is False

    assert crossing_a.schema_version == "r2v.h3.diarization_segment.2"
    assert crossing_a.backend_reported_start_time == 7.92
    assert crossing_a.backend_reported_end_time == 30.0
    assert crossing_a.backend_reported_start_sample == 792
    assert crossing_a.backend_reported_end_sample == 3000
    assert crossing_a.source_start_sample == 792
    assert crossing_a.source_end_sample == target.source_frame_count
    assert crossing_a.end_time == 10.0
    assert crossing_a.boundary_reconciliation.model_dump() == {
        "policy_version": "canonical_source_intersection_v1",
        "adjusted": True,
        "end_clamped": True,
        "end_overrun_samples": 2000,
        "end_overrun_seconds": 20.0,
        "reason": "end_clamped_to_canonical_source",
    }
    assert crossing_b.source_start_sample == 800
    assert crossing_b.source_end_sample == 1000
    assert crossing_b.boundary_reconciliation.end_overrun_samples == 125
    assert max(crossing_a.source_start_sample, crossing_b.source_start_sample) < min(
        crossing_a.source_end_sample,
        crossing_b.source_end_sample,
    )


def test_segment_without_positive_canonical_intersection_fails_closed() -> None:
    target = _target()
    with pytest.raises(
        DiarizationBackendFailure,
        match="starts at or after canonical source EOF",
    ):
        _normalize_segments(
            target=target,
            segments=[
                DiarizationBackendSegment(
                    start_time=10.0,
                    end_time=11.0,
                    speaker_label="after-eof",
                )
            ],
            provenance=_provenance(),
        )

    with pytest.raises(
        DiarizationBackendFailure,
        match="no positive canonical source intersection",
    ):
        _normalize_segments(
            target=target,
            segments=[
                DiarizationBackendSegment(
                    start_time=0.001,
                    end_time=0.004,
                    speaker_label="rounds-to-zero",
                )
            ],
            provenance=_provenance(),
        )


def test_sparse_raw_anchor_maps_cluster_and_propagates_to_zero_overlap_segment() -> (
    None
):
    target = _target()
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=[
            _raw("speaker_0", 0.0, 1.0, index=1),
            _raw("speaker_0", 3.0, 5.0, index=2),
        ],
        frozen_bindings=[_binding(0.2, 0.3)],
    )

    cluster = mapping.bindings[0]
    assert cluster.status == "candidate_mapped"
    assert cluster.entity_id == "e1"
    assert cluster.visual_anchor_coverage_ratio < 0.04
    assert cluster.visual_anchor_coverage_is_diagnostic_only is True
    assert [item.identity_scope for item in mapping.bound_segments] == [
        "direct_anchor_present",
        "cluster_propagated_only",
    ]
    assert mapping.bound_segments[1].entity_id == "e1"


def test_identity_propagation_metrics_include_unanchored_part_of_mixed_segment() -> (
    None
):
    target = _target()
    raw = [_raw("speaker_0", 0.0, 4.0, index=1)]
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=raw,
        frozen_bindings=[_binding(0.8, 1.2)],
    )

    direct, propagated, fully_propagated = _mapped_identity_duration_metrics(
        raw_segments=raw,
        cluster_bindings=mapping.bindings,
        bound_segments=mapping.bound_segments,
    )

    assert mapping.bindings[0].cluster_speaker_seconds == pytest.approx(4.0)
    assert direct == pytest.approx(0.4)
    assert propagated == pytest.approx(3.6)
    assert fully_propagated == 0.0
    assert direct + propagated == pytest.approx(4.0)


def test_identity_propagation_metrics_are_union_safe_and_status_scoped() -> None:
    target = _target()
    overlapping_raw = [
        _raw("speaker_0", 0.0, 2.0, index=1),
        _raw("speaker_0", 1.0, 3.0, index=2),
    ]
    overlapping = bind_diarization_segments(
        target=target,
        raw_segments=overlapping_raw,
        frozen_bindings=[_binding(0.0, 0.4)],
    )
    direct, propagated, fully = _mapped_identity_duration_metrics(
        raw_segments=overlapping_raw,
        cluster_bindings=overlapping.bindings,
        bound_segments=overlapping.bound_segments,
    )
    assert overlapping.bindings[0].cluster_speaker_seconds == pytest.approx(3.0)
    assert direct == pytest.approx(0.4)
    assert propagated == pytest.approx(2.6)
    assert fully == pytest.approx(2.0)

    zero_anchor_overlap_raw = [
        _raw("speaker_0", 0.0, 2.0, index=1),
        _raw("speaker_0", 1.0, 3.0, index=2),
        _raw("speaker_0", 4.0, 5.0, index=3),
    ]
    zero_anchor_overlap = bind_diarization_segments(
        target=target,
        raw_segments=zero_anchor_overlap_raw,
        frozen_bindings=[_binding(4.0, 5.0)],
    )
    assert _mapped_identity_duration_metrics(
        raw_segments=zero_anchor_overlap_raw,
        cluster_bindings=zero_anchor_overlap.bindings,
        bound_segments=zero_anchor_overlap.bound_segments,
    ) == pytest.approx((1.0, 3.0, 3.0))

    fully_anchored_raw = [_raw("speaker_0", 0.0, 1.0, index=1)]
    fully_anchored = bind_diarization_segments(
        target=target,
        raw_segments=fully_anchored_raw,
        frozen_bindings=[_binding(0.0, 1.0)],
    )
    assert _mapped_identity_duration_metrics(
        raw_segments=fully_anchored_raw,
        cluster_bindings=fully_anchored.bindings,
        bound_segments=fully_anchored.bound_segments,
    ) == pytest.approx((1.0, 0.0, 0.0))

    ambiguous_raw = [_raw("speaker_0", 0.0, 1.0, index=1)]
    ambiguous = bind_diarization_segments(
        target=target,
        raw_segments=ambiguous_raw,
        frozen_bindings=[_binding(0.0, 0.4, "e1"), _binding(0.4, 0.8, "e2")],
    )
    assert ambiguous.bindings[0].status == "ambiguous"
    assert _mapped_identity_duration_metrics(
        raw_segments=ambiguous_raw,
        cluster_bindings=ambiguous.bindings,
        bound_segments=ambiguous.bound_segments,
    ) == pytest.approx((0.0, 0.0, 0.0))

    conflict_raw = [
        _raw("speaker_0", 0.0, 2.0, index=1),
        _raw("speaker_1", 1.0, 3.0, index=2),
    ]
    conflict = bind_diarization_segments(
        target=target,
        raw_segments=conflict_raw,
        frozen_bindings=[_binding(0.0, 0.4, "e1"), _binding(2.4, 2.8, "e1")],
    )
    assert [item.status for item in conflict.bindings] == ["conflict", "conflict"]
    assert _mapped_identity_duration_metrics(
        raw_segments=conflict_raw,
        cluster_bindings=conflict.bindings,
        bound_segments=conflict.bound_segments,
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_only_frozen_bound_sync_identity_evidence_contributes() -> None:
    target = _target()
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.0, 2.0, index=1)],
        frozen_bindings=[
            _nonbound(0.0, 0.1, "offscreen"),
            _nonbound(0.1, 0.2, "ambiguous"),
            _nonbound(0.2, 0.3, "overlap"),
            _nonbound(0.3, 0.4, "no_speech"),
            _binding(0.4, 0.5, confidence=0.79),
            _binding(0.5, 0.6, synchronization_plausible=False),
            _binding(
                0.6,
                0.8,
                audio_quality_usable=False,
                clean_training_eligible=False,
            ),
        ],
    )

    cluster = mapping.bindings[0]
    assert cluster.status == "candidate_mapped"
    assert cluster.usable_anchor_sample_count == 20
    assert cluster.entity_supports[0].contributing_binding_count == 1


def test_direct_support_counts_raw_bindings_instead_of_coalesced_turns() -> None:
    target = _target()
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.0, 1.0, index=1)],
        frozen_bindings=[
            _binding(0.10, 0.20, "e1"),
            _binding(0.24, 0.34, "e1"),
        ],
    )

    support = mapping.bindings[0].entity_supports[0]
    assert support.direct_support_samples == 20
    assert support.contributing_binding_count == 2


def test_no_support_is_unbound_and_multiple_entities_are_ambiguous() -> None:
    target = _target()
    no_support = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.0, 1.0, index=1)],
        frozen_bindings=[],
    )
    ambiguous = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.0, 1.0, index=1)],
        frozen_bindings=[_binding(0.1, 0.2, "e1"), _binding(0.3, 0.4, "e2")],
    )

    assert no_support.bindings[0].status == "unbound"
    assert no_support.bindings[0].entity_id is None
    assert ambiguous.bindings[0].status == "ambiguous"
    assert ambiguous.bindings[0].entity_id is None
    assert [item.entity_id for item in ambiguous.bindings[0].entity_supports] == [
        "e1",
        "e2",
    ]


def test_same_entity_nonoverlap_clusters_allowed_but_temporal_overlap_conflicts() -> (
    None
):
    target = _target()
    nonoverlap = bind_diarization_segments(
        target=target,
        raw_segments=[
            _raw("speaker_0", 0.0, 1.0, index=1),
            _raw("speaker_1", 2.0, 3.0, index=2),
        ],
        frozen_bindings=[_binding(0.1, 0.2, "e1"), _binding(2.1, 2.2, "e1")],
    )
    overlap = bind_diarization_segments(
        target=target,
        raw_segments=[
            _raw("speaker_0", 0.0, 2.0, index=1),
            _raw("speaker_1", 1.0, 3.0, index=2),
        ],
        frozen_bindings=[_binding(0.1, 0.2, "e1"), _binding(2.1, 2.2, "e1")],
    )

    assert [item.status for item in nonoverlap.bindings] == [
        "candidate_mapped",
        "candidate_mapped",
    ]
    assert [item.status for item in overlap.bindings] == ["conflict", "conflict"]
    assert all(item.entity_id is None for item in overlap.bindings)


def test_different_entities_may_overlap_and_contested_span_is_not_double_counted() -> (
    None
):
    target = _target()
    valid = bind_diarization_segments(
        target=target,
        raw_segments=[
            _raw("speaker_0", 0.0, 2.0, index=1),
            _raw("speaker_1", 1.0, 3.0, index=2),
        ],
        frozen_bindings=[_binding(0.1, 0.2, "e1"), _binding(2.1, 2.2, "e2")],
    )
    contested = bind_diarization_segments(
        target=target,
        raw_segments=[
            _raw("speaker_0", 0.0, 2.0, index=1),
            _raw("speaker_1", 1.0, 3.0, index=2),
        ],
        frozen_bindings=[_binding(1.2, 1.4, "e1")],
    )

    assert [item.status for item in valid.bindings] == [
        "candidate_mapped",
        "candidate_mapped",
    ]
    assert [item.entity_id for item in valid.bindings] == ["e1", "e2"]
    assert contested.contested_samples == 20
    assert contested.usable_samples == 0
    assert all(not item.entity_supports for item in contested.bindings)


def test_summary_distinguishes_speaker_wallclock_and_propagated_seconds(
    tmp_path: Path,
) -> None:
    root, selected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    responses = {
        selected[0]: [
            DiarizationBackendSegment(start_time=0.0, end_time=0.5, speaker_label="a"),
            DiarizationBackendSegment(start_time=0.5, end_time=0.8, speaker_label="b"),
            DiarizationBackendSegment(start_time=0.6, end_time=1.0, speaker_label="a"),
        ]
    }
    backend = _FakeBackend(responses)

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=diarization_output_root(root, mode="pilot20"),
        backend=backend,
    )

    # Remaining 19 clips each contribute 0.8 speaker/wall-clock seconds.
    assert summary.diarized_speaker_seconds == pytest.approx(16.4)
    assert summary.diarized_wallclock_speech_seconds == pytest.approx(16.2)
    assert summary.mapped_speaker_seconds == pytest.approx(16.1)
    assert summary.mapped_direct_anchor_speaker_seconds == pytest.approx(4.0)
    assert summary.identity_propagated_speaker_seconds == pytest.approx(12.1)
    assert summary.fully_propagated_segment_speaker_seconds == pytest.approx(0.4)


def test_source_hash_change_fails_only_that_clip_without_backend_call(
    tmp_path: Path,
) -> None:
    root, selected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    first = inventory.targets[0]
    Path(first.source_audio_path).write_bytes(b"changed")
    backend = _FakeBackend()

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=diarization_output_root(root, mode="pilot20"),
        backend=backend,
    )

    assert summary.failed_clip_count == 1
    assert summary.backend_call_count == 19
    assert selected[0] not in [item[0] for item in backend.calls]


def test_review_preserves_overlap_lanes_and_qa_fingerprint(tmp_path: Path) -> None:
    root, selected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    backend = _FakeBackend(
        {
            selected[0]: [
                DiarizationBackendSegment(
                    start_time=0.0, end_time=0.6, speaker_label="left"
                ),
                DiarizationBackendSegment(
                    start_time=0.3, end_time=0.8, speaker_label="right"
                ),
            ]
        }
    )
    output = diarization_output_root(root, mode="pilot20")
    run_diarization_binding_pilot(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    review = (output / "review.html").read_text(encoding="utf-8")
    assert "speaker_0" in review and "speaker_1" in review
    assert 'class="lane"' in review
    assert "Export QA JSON" in review
    assert inventory.inventory_fingerprint in review
    assert inventory.source_asr_inventory_fingerprint in review
    assert "localStorage.clear()" not in review


def test_boundary_summary_review_and_audio_use_effective_canonical_intervals(
    tmp_path: Path,
) -> None:
    root, selected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    backend = _FakeBackend(
        {
            selected[0]: [
                DiarizationBackendSegment(
                    start_time=0.0,
                    end_time=1.13,
                    speaker_label="left",
                ),
                DiarizationBackendSegment(
                    start_time=0.7,
                    end_time=1.2,
                    speaker_label="right",
                ),
            ]
        }
    )
    output = diarization_output_root(root, mode="pilot20")

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    assert summary.failed_clip_count == 0
    assert summary.boundary_adjusted_segment_count == 2
    assert summary.boundary_adjusted_clip_count == 1
    assert summary.end_clamped_segment_count == 2
    assert summary.total_end_overrun_seconds == pytest.approx(0.33)
    assert summary.max_end_overrun_seconds == pytest.approx(0.2)
    assert summary.max_end_overrun_samples == 3200
    assert summary.median_positive_end_overrun_seconds == pytest.approx(0.165)

    rows = [
        json.loads(line)
        for line in (output / "raw_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["target_clip_uid"] == selected[0]
    ]
    assert [item["backend_reported_end_time"] for item in rows] == [1.13, 1.2]
    assert [item["source_end_sample"] for item in rows] == [16000, 16000]
    review = (output / "review.html").read_text(encoding="utf-8")
    assert review.count("EOF CLAMP") >= 4
    assert "REPORTED: 0.000-1.130s" in review
    assert "CANONICAL: 0.000-1.000s" in review
    assert "EOF CLAMP: +0.130s" in review

    first_audio = (
        output / "review_media" / "segments" / selected[0] / "segment_0001.wav"
    )
    second_audio = (
        output / "review_media" / "segments" / selected[0] / "segment_0002.wav"
    )
    with wave.open(str(first_audio), "rb") as stream:
        assert stream.getnframes() == 16000
    with wave.open(str(second_audio), "rb") as stream:
        assert stream.getnframes() == 4800


def test_qa_schema_allows_only_fixed_labels_and_deterministic_order() -> None:
    assert DIARIZATION_HUMAN_QA_LABELS == ("CORRECT", "WRONG", "UNCERTAIN")
    labels = [
        DiarizationHumanQALabel(
            target_clip_uid="clip-a",
            speaker_cluster_id="speaker_0",
            predicted_binding_status="candidate_mapped",
            predicted_entity_id="e1",
            label="CORRECT",
        ),
        DiarizationHumanQALabel(
            target_clip_uid="clip-b",
            speaker_cluster_id="speaker_0",
            predicted_binding_status="unbound",
            label="UNCERTAIN",
        ),
    ]
    export = DiarizationHumanQAExport(
        inventory_fingerprint="a" * 64,
        source_asr_inventory_fingerprint="b" * 64,
        label_count=2,
        total_cluster_count=3,
        counts={"CORRECT": 1, "WRONG": 0, "UNCERTAIN": 1, "UNLABELED": 1},
        labels=labels,
    )
    assert export.inventory_fingerprint == "a" * 64
    with pytest.raises(ValueError):
        DiarizationHumanQAExport.model_validate(
            {**export.model_dump(mode="python"), "labels": list(reversed(labels))}
        )


def test_persistent_backend_reuses_one_worker_for_multiple_clips(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_worker.py"
    load_marker = tmp_path / "loads.txt"
    worker.write_text(
        """import json, os, sys
marker = os.environ['LOAD_MARKER']
with open(marker, 'a', encoding='utf-8') as out: out.write('loaded\\n')
model = os.environ['MODEL_ID']
fingerprint = os.environ['MODEL_FP']
print(json.dumps({'request_id':'startup','status':'ready','model_identifier':model,'model_fingerprint':fingerprint}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['operation'] == 'shutdown':
        print(json.dumps({'request_id':request['request_id'],'status':'shutdown'}), flush=True)
        break
    print(json.dumps({'request_id':request['request_id'],'status':'ready','model_identifier':model,'segments':[{'start_time':0.0,'end_time':1.0,'speaker_label':'a'}]}), flush=True)
""",
        encoding="utf-8",
    )
    provenance = _provenance()
    backend = PersistentDiariZenBackend(
        executable=[sys.executable, str(worker)],
        provenance=provenance,
        timeout_seconds=5,
        diagnostics_root=tmp_path / "diagnostics",
        environment={
            "LOAD_MARKER": str(load_marker),
            "MODEL_ID": provenance.model_identifier,
            "MODEL_FP": provenance.model_fingerprint,
        },
    )
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    try:
        first = backend.diarize(clip_uid="clip-a", audio_path=audio)
        process = backend._process
        second = backend.diarize(clip_uid="clip-b", audio_path=audio)
        assert backend._process is process
    finally:
        backend.close()

    assert len(first) == len(second) == 1
    assert load_marker.read_text(encoding="utf-8").splitlines() == ["loaded"]


def test_backend_failure_is_isolated_and_empty_output_is_not_failure(
    tmp_path: Path,
) -> None:
    root, selected = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="pilot20")

    class Backend(_FakeBackend):
        def diarize(
            self,
            *,
            clip_uid: str,
            audio_path: Path,
        ) -> list[DiarizationBackendSegment]:
            self.calls.append((clip_uid, audio_path))
            if clip_uid == selected[0]:
                raise DiarizationBackendFailure("fixture failure")
            if clip_uid == selected[1]:
                return []
            return [
                DiarizationBackendSegment(
                    start_time=0.0,
                    end_time=0.8,
                    speaker_label="vendor-speaker-A",
                )
            ]

    backend = Backend()
    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=diarization_output_root(root, mode="pilot20"),
        backend=backend,
    )

    assert summary.failed_clip_count == 1
    assert summary.empty_clip_count == 1
    assert summary.ready_clip_count == 18
