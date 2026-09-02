from __future__ import annotations

import builtins
import hashlib
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

import tools.run_h3_diarization_binding as diarization_cli
from r2v_data_v2.h3.asr_transcription import build_asr_inventory
from r2v_data_v2.h3.audio_production import (
    H3ProductionInPair,
    H3ProductionInPairSubject,
)
from r2v_data_v2.h3.diarization_binding import (
    DIARIZATION_CANONICAL_ANCHOR_BOUNDARY_POLICY_VERSION,
    DIARIZATION_HUMAN_QA_LABELS,
    DIARIZATION_REQUEST_VERSION,
    DiarizationBackendFailure,
    DiarizationBackendProvenance,
    DiarizationBackendSegment,
    DiarizationBoundaryReconciliation,
    DiarizationHumanQAExport,
    DiarizationHumanQALabel,
    DiarizationInventory,
    DiarizationTargetClip,
    PersistentDiariZenBackend,
    RawDiarizationSegment,
    _legacy_metrics,
    _mapped_identity_duration_metrics,
    _normalize_segments,
    _usable_anchors,
    bind_diarization_segments,
    build_complete_diarization_inventory,
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
from tools.diarizen_worker import (
    _all_inactive_reconstruct_guard,
    _prepare_analysis_audio,
)


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
        input_profile="legacy_16k_mono",
        input_preprocessing="h3_diarizen_native_16k_mono_passthrough_v1",
        source_sample_rate_hz=16000,
        source_channels=1,
    )


class _FakeBackend:
    def __init__(
        self,
        responses: dict[str, list[DiarizationBackendSegment]] | None = None,
    ) -> None:
        self.provenance = _provenance()
        self.responses = responses or {}
        self.calls: list[tuple[str, Path]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

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


def _target(
    *,
    clip_uid: str = "clip-a",
    rate: int = 100,
    channels: int = 1,
    frame_count: int = 1000,
) -> DiarizationTargetClip:
    return DiarizationTargetClip(
        target_clip_uid=clip_uid,
        target_video_path=f"/{clip_uid}.mp4",
        source_audio_path=f"/{clip_uid}.wav",
        source_audio_sha256="c" * 64,
        source_sample_rate_hz=rate,
        source_channels=channels,
        source_frame_count=frame_count,
        target_audio_binding_path=f"/{clip_uid}.json",
        visual_references=[],
    )


def test_canonical_targets_without_sidecars_still_diarize_and_bind_unbound(
    tmp_path: Path,
) -> None:
    targets = []
    for clip_uid in ("clip-speech", "clip-empty"):
        audio = tmp_path / f"{clip_uid}.flac"
        audio.write_bytes(f"audio:{clip_uid}".encode())
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=f"/{clip_uid}.mp4",
                source_audio_path=str(audio),
                source_audio_sha256=_sha256(audio),
                source_sample_rate_hz=16000,
                source_channels=1,
                source_frame_count=16000,
                target_audio_binding_path=None,
                visual_references=[],
            )
        )
    pairs = tmp_path / "legacy-pairs.jsonl"
    pairs.write_text("{}\n", encoding="utf-8")
    inventory = build_complete_diarization_inventory(
        source_pairs_path=pairs,
        targets=sorted(targets, key=lambda item: item.target_clip_uid),
    )
    backend = _FakeBackend(
        {
            "clip-speech": [
                DiarizationBackendSegment(
                    start_time=0.0,
                    end_time=0.5,
                    speaker_label="speaker-a",
                )
            ],
            "clip-empty": [],
        }
    )
    output = tmp_path / "diarization"

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    results = {
        row["target_clip_uid"]: row
        for row in (
            json.loads(line)
            for line in (output / "clip_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    clusters = [
        json.loads(line)
        for line in (output / "cluster_bindings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    bound = [
        json.loads(line)
        for line in (output / "bound_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item[0] for item in backend.calls] == ["clip-empty", "clip-speech"]
    assert results["clip-empty"]["status"] == "empty"
    assert results["clip-speech"]["status"] == "ready"
    assert results["clip-speech"]["legacy_bound_interval_count"] == 0
    assert clusters[0]["status"] == "unbound"
    assert clusters[0]["entity_id"] is None
    assert bound[0]["identity_scope"] == "unresolved"
    assert bound[0]["entity_id"] is None
    assert summary.target_clip_count == 2
    assert summary.ready_clip_count == 1
    assert summary.empty_clip_count == 1
    assert summary.failed_clip_count == 0


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
        source_channels=active.source_channels,
        backend="fake_diarizen",
        model_identifier="fixture/diarizen-v2",
        model_fingerprint="a" * 64,
        backend_configuration_fingerprint="b" * 64,
        input_preprocessing="h3_diarizen_native_16k_mono_passthrough_v1",
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
    (root / "asr_pilot20" / "inventory.json").unlink()
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
    assert production.cross_pair_jobs_created == 0
    assert production.production_blocked is False
    assert production.source_asr_inventory_path is None
    assert production.source_asr_inventory_fingerprint is None
    assert production.mapping_policy_validated is True
    assert production.numeric_mapping_thresholds_used is False
    assert production.mapping_policy_version == "h3_diarizen_sparse_anchor_policy_v1"
    assert production.schema_version == "r2v.h3.diarization_inventory.4"
    assert production.calibration_inventory_fingerprint == (
        "776761abc1ffa1822766eb29c1ecf61f9e32beda35f2246cb3ef6dc3f096e7b7"
    )


def test_dry_run_imports_no_diarizen_and_real_production_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _audio_run(tmp_path)
    frozen_before = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "diarizen" or name.startswith("diarizen."):
            raise AssertionError("dry-run imported DiariZen")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = diarization_cli.main(
        ["--audio-run-root", str(root), "--mode", "production", "--dry-run"]
    )
    assert result["selected_target_count"] == 75
    assert result["parent_quota_applied"] is False
    assert result["cross_pair_jobs_created"] == 0
    assert result["production_blocked"] is False
    assert result["output_root"] == str(root / "production" / "diarization")

    backend = _FakeBackend()
    hidden_diagnostics = root / ".fake-diarization-worker"
    monkeypatch.setattr(
        diarization_cli,
        "_runtime_backend",
        lambda *, output_root: (backend, hidden_diagnostics),
    )
    completed = diarization_cli.main(
        ["--audio-run-root", str(root), "--mode", "production"]
    )

    assert completed["stage_status"] == "completed"
    assert len(backend.calls) == 75
    assert [item[0] for item in backend.calls] == sorted(
        item[0] for item in backend.calls
    )
    output = root / "production" / "diarization"
    assert output.is_dir()
    assert not (output / "review.html").exists()
    assert not (output / "review_media").exists()
    assert sum(1 for _ in (output / "raw_segments.jsonl").open()) == 75
    first_cluster = json.loads(
        (output / "cluster_bindings.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_cluster["schema_version"] == "r2v.h3.diarization_cluster_binding.3"
    assert first_cluster["mapping_policy_version"] == (
        "h3_diarizen_sparse_anchor_policy_v1"
    )
    summary = completed["summary"]
    assert isinstance(summary, dict)
    assert summary["schema_version"] == "r2v.h3.diarization_summary.4"
    assert summary["mode"] == "production"
    assert summary["target_clip_count"] == 75
    assert summary["backend_call_count"] == 75
    assert summary["mapping_policy_validated"] is True
    assert summary["mapping_policy_version"] == "h3_diarizen_sparse_anchor_policy_v1"
    assert summary["thresholds_calibrated"] is False
    assert summary["numeric_mapping_thresholds_used"] is False
    assert summary["parent_quota_applied"] is False
    assert summary["visual_anchor_coverage_used_as_gate"] is False
    assert summary["mapped_speaker_seconds"] == pytest.approx(
        summary["mapped_direct_anchor_speaker_seconds"]
        + summary["identity_propagated_speaker_seconds"]
    )
    frozen_after = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.relative_to(root).as_posix().startswith("production/diarization/")
    }
    assert frozen_after == frozen_before


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
    assert summary.schema_version == "r2v.h3.diarization_summary.4"
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

    assert crossing_a.schema_version == "r2v.h3.diarization_segment.3"
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


def test_canonical_anchor_small_eof_overrun_is_clamped_for_direct_support() -> None:
    target = _target(rate=32000, channels=2, frame_count=32000)
    binding = _binding(0.9, 1.05)
    anchors = _usable_anchors(
        [binding],
        target=target,
        input_profile="canonical_32k_stereo",
    )

    assert [(item.start, item.end) for item in anchors] == [(28800, 32000)]
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.8, 1.0, index=1, target=target)],
        frozen_bindings=[binding],
        input_profile="canonical_32k_stereo",
    )

    assert mapping.bindings[0].status == "candidate_mapped"
    assert mapping.bindings[0].usable_anchor_sample_count == 3200
    assert mapping.bindings[0].entity_supports[0].direct_support_samples == 3200
    assert mapping.bound_segments[0].direct_anchor_samples == 3200


def test_canonical_anchor_large_eof_overrun_fails_closed() -> None:
    target = _target(rate=32000, channels=2, frame_count=32000)

    with pytest.raises(
        ValueError,
        match="frozen Audio binding exceeds canonical source audio",
    ):
        _usable_anchors(
            [_binding(0.8, 1.10003125)],
            target=target,
            input_profile="canonical_32k_stereo",
        )


def test_canonical_anchor_starting_at_eof_fails_closed() -> None:
    target = _target(rate=32000, channels=2, frame_count=32000)

    with pytest.raises(
        ValueError,
        match="frozen Audio binding exceeds canonical source audio",
    ):
        _usable_anchors(
            [_binding(1.0, 1.02)],
            target=target,
            input_profile="canonical_32k_stereo",
        )


def test_legacy_anchor_small_eof_overrun_remains_strict() -> None:
    target = _target(rate=16000, channels=1, frame_count=16000)

    with pytest.raises(
        ValueError,
        match="frozen Audio binding exceeds canonical source audio",
    ):
        _usable_anchors(
            [_binding(0.9, 1.01)],
            target=target,
            input_profile="legacy_16k_mono",
        )


def test_legacy_metrics_and_mapping_share_effective_canonical_anchors() -> None:
    target = _target(rate=32000, channels=2, frame_count=32000)
    binding = _binding(0.9, 1.05)
    anchors, durations = _legacy_metrics(
        target,
        SimpleNamespace(bindings=[binding]),
        input_profile="canonical_32k_stereo",
    )
    mapping = bind_diarization_segments(
        target=target,
        raw_segments=[_raw("speaker_0", 0.8, 1.0, index=1, target=target)],
        frozen_bindings=[binding],
        input_profile="canonical_32k_stereo",
    )

    assert [(item.start, item.end) for item in anchors] == [(28800, 32000)]
    assert durations == pytest.approx([0.15])
    assert mapping.usable_samples == anchors[0].end - anchors[0].start


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


def test_production_source_hash_change_fails_only_that_target(tmp_path: Path) -> None:
    root, _ = _audio_run(tmp_path)
    inventory = build_diarization_inventory(audio_run_root=root, mode="production")
    first = inventory.targets[0]
    Path(first.source_audio_path).write_bytes(b"changed")
    backend = _FakeBackend()

    summary = run_diarization_binding_pilot(
        inventory=inventory,
        output_root=diarization_output_root(root, mode="production"),
        backend=backend,
    )

    assert summary.mode == "production"
    assert summary.target_clip_count == 75
    assert summary.failed_clip_count == 1
    assert summary.backend_call_count == 74
    assert first.target_clip_uid not in [item[0] for item in backend.calls]
    assert summary.failure_reason_counts == {"ValueError:source_audio_hash_mismatch": 1}


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
    print(json.dumps({'request_id':request['request_id'],'status':'ready','model_identifier':model,'backend_metadata':{'input_profile':'legacy_16k_mono','input_preprocessing':'h3_diarizen_native_16k_mono_passthrough_v1','source_sample_rate_hz':16000,'source_channels':1,'model_input_sample_rate_hz':16000,'model_input_channels':1},'segments':[{'start_time':0.0,'end_time':1.0,'speaker_label':'a'}]}), flush=True)
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


def test_diarizen_runtime_preprocessing_downmixes_and_resamples_ephemerally(
    tmp_path: Path,
) -> None:
    class _Truth:
        @staticmethod
        def item() -> bool:
            return True

    class _Finite:
        @staticmethod
        def all() -> _Truth:
            return _Truth()

    class _Waveform:
        ndim = 2
        shape = (2, 64000)

        def mean(self, *, dim: int, keepdim: bool) -> str:
            assert (dim, keepdim) == (0, True)
            return "mono-32k"

    calls: list[tuple[str, object]] = []

    class _Functional:
        @staticmethod
        def resample(value: object, **kwargs: object) -> str:
            calls.append(("resample", (value, kwargs)))
            return "mono-16k"

    class _Audio:
        functional = _Functional()

        @staticmethod
        def load(path: str) -> tuple[_Waveform, int]:
            calls.append(("load", path))
            return _Waveform(), 32000

        @staticmethod
        def save(path: str, value: object, rate: int, **kwargs: object) -> None:
            calls.append(("save", (path, value, rate, kwargs)))

    torch = SimpleNamespace(isfinite=lambda _value: _Finite())
    source = tmp_path / "canonical.flac"
    destination = tmp_path / "ephemeral" / "analysis.wav"
    _prepare_analysis_audio(
        source=source,
        destination=destination,
        torch=torch,
        torchaudio=_Audio(),
    )

    assert calls[0] == ("load", str(source))
    assert calls[1][0] == "resample"
    value, options = calls[1][1]
    assert value == "mono-32k"
    assert options["orig_freq"] == 32000
    assert options["new_freq"] == 16000
    assert options["resampling_method"] == "sinc_interp_kaiser"
    assert calls[2] == (
        "save",
        (
            str(destination),
            "mono-16k",
            16000,
            {"encoding": "PCM_S", "bits_per_sample": 16},
        ),
    )


def test_diarizen_legacy_runtime_profile_is_native_16k_mono_passthrough(
    tmp_path: Path,
) -> None:
    class _Truth:
        @staticmethod
        def item() -> bool:
            return True

    class _Finite:
        @staticmethod
        def all() -> _Truth:
            return _Truth()

    class _Waveform:
        ndim = 2
        shape = (1, 16000)

    calls: list[tuple[str, object]] = []

    class _Functional:
        @staticmethod
        def resample(*args: object, **kwargs: object) -> None:
            calls.append(("resample", (args, kwargs)))

    class _Audio:
        functional = _Functional()

        @staticmethod
        def load(path: str) -> tuple[_Waveform, int]:
            calls.append(("load", path))
            return _Waveform(), 16000

        @staticmethod
        def save(*args: object, **kwargs: object) -> None:
            calls.append(("save", (args, kwargs)))

    source = tmp_path / "legacy.wav"
    destination = tmp_path / "unused.wav"
    result = _prepare_analysis_audio(
        source=source,
        destination=destination,
        torch=SimpleNamespace(isfinite=lambda _value: _Finite()),
        torchaudio=_Audio(),
        input_profile="legacy_16k_mono",
    )

    assert result == source
    assert calls == [("load", str(source))]
    assert not destination.exists()


class _Segmentations:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _ReconstructPipeline:
    def __init__(self, *, unrelated_failure: bool = False) -> None:
        self.calls: list[np.ndarray] = []
        self.unrelated_failure = unrelated_failure

    def reconstruct(
        self,
        segmentations: _Segmentations,
        hard_clusters: np.ndarray,
    ) -> np.ndarray:
        del segmentations
        self.calls.append(hard_clusters.copy())
        if self.unrelated_failure:
            raise ValueError("unrelated reconstruction failure")
        return np.zeros((1, int(np.max(hard_clusters)) + 1))


def test_diarizen_all_inactive_reconstruct_guard_returns_empty_activity() -> None:
    pipeline = _ReconstructPipeline()
    segmentations = _Segmentations(np.zeros((1, 4, 2), dtype=np.int64))
    hard_clusters = np.full((1, 2), -2, dtype=np.int64)

    with _all_inactive_reconstruct_guard(pipeline, numpy=np) as state:
        reconstructed = pipeline.reconstruct(segmentations, hard_clusters)

    assert state == {"applied": True}
    assert pipeline.calls[0].tolist() == [[0, 0]]
    assert reconstructed.shape == (1, 1)
    assert "reconstruct" not in vars(pipeline)
    with pytest.raises(ValueError, match="negative dimensions"):
        pipeline.reconstruct(segmentations, hard_clusters)


def test_diarizen_reconstruct_guard_leaves_other_states_unchanged() -> None:
    pipeline = _ReconstructPipeline()
    segmentations = _Segmentations(
        np.array([[[1, 0], [0, 0]]], dtype=np.int64)
    )
    hard_clusters = np.array([[0, -2]], dtype=np.int64)

    with _all_inactive_reconstruct_guard(pipeline, numpy=np) as state:
        pipeline.reconstruct(segmentations, hard_clusters)

    assert state == {"applied": False}
    assert [item.tolist() for item in pipeline.calls] == [hard_clusters.tolist()]


def test_diarizen_reconstruct_guard_requires_zero_inactive_segmentations() -> None:
    pipeline = _ReconstructPipeline()
    segmentations = _Segmentations(np.ones((1, 2, 1), dtype=np.int64))
    hard_clusters = np.full((1, 1), -2, dtype=np.int64)

    with _all_inactive_reconstruct_guard(
        pipeline, numpy=np
    ) as state, pytest.raises(ValueError, match="negative dimensions"):
        pipeline.reconstruct(segmentations, hard_clusters)

    assert state == {"applied": False}
    assert [item.tolist() for item in pipeline.calls] == [hard_clusters.tolist()]


def test_diarizen_reconstruct_guard_propagates_unrelated_value_error() -> None:
    pipeline = _ReconstructPipeline(unrelated_failure=True)
    segmentations = _Segmentations(np.zeros((1, 2, 1), dtype=np.int64))
    hard_clusters = np.full((1, 1), -2, dtype=np.int64)

    with _all_inactive_reconstruct_guard(
        pipeline, numpy=np
    ) as state, pytest.raises(ValueError, match="unrelated reconstruction failure"):
        pipeline.reconstruct(segmentations, hard_clusters)

    assert state == {"applied": True}


def test_diarization_request_v3_changes_configuration_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = {
        "model_identifier": "fixture/diarizen",
        "model_fingerprint": "a" * 64,
        "requested_device": "cuda:0",
        "input_profile": "canonical_32k_stereo",
    }
    current = diarization_cli._configuration_fingerprint(**arguments)
    monkeypatch.setattr(
        diarization_cli,
        "DIARIZATION_REQUEST_VERSION",
        "h3_diarizen_clip_diarization_v2",
    )
    previous = diarization_cli._configuration_fingerprint(**arguments)

    assert DIARIZATION_REQUEST_VERSION == "h3_diarizen_clip_diarization_v3"
    assert _provenance().request_contract_version == DIARIZATION_REQUEST_VERSION
    assert current != previous
    assert (
        DIARIZATION_CANONICAL_ANCHOR_BOUNDARY_POLICY_VERSION
        == "canonical_audio_anchor_source_intersection_v1"
    )


def test_current_diarization_inventory_rejects_old_sample_domain_schema(
    tmp_path: Path,
) -> None:
    root, _selected = _audio_run(tmp_path)
    current = build_diarization_inventory(audio_run_root=root, mode="pilot20")
    legacy = current.model_dump(mode="json")
    legacy["schema_version"] = "r2v.h3.diarization_inventory.3"

    with pytest.raises(ValueError, match="diarization_inventory.4"):
        DiarizationInventory.model_validate(legacy)


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
