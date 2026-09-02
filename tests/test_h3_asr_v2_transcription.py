from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import r2v_data_v2.h3.asr_v2_transcription as asr_v2_module
import tools.run_h3_asr_v2_transcription as asr_v2_cli
from r2v_data_v2.h3.asr_transcription import (
    ASRBackendResult,
    ASRDecoderDiagnostics,
    ASRHumanQACounts,
    ASRInventory,
    ASRPreprocessingProvenance,
    ASRSummary,
    ASRTargetClip,
    ASRTurnJob,
    ASRTurnRecord,
    ASRTurnSegmentationProvenance,
    WhisperASRConfig,
)
from r2v_data_v2.h3.asr_transcription import (
    _inventory_fingerprint as _asr_v1_inventory_fingerprint,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT,
    ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT,
    ASRV2HumanQAExport,
    ASRV2HumanQALabel,
    ASRV2ProductionInventory,
    ASRV2ProductionSummary,
    ASRV2SegmentJob,
    ASRV2SegmentRecord,
    asr_v2_output_root,
    build_asr_v2_backend_provenance,
    build_asr_v2_inventory,
    regenerate_asr_v2_review,
    run_asr_v2_transcription,
)
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationBackendProvenance,
    DiarizationBoundaryReconciliation,
    DiarizationClusterBinding,
    DiarizationEntitySupport,
    DiarizationInventory,
    DiarizationSummary,
    DiarizationTargetClip,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in values
        ),
        encoding="utf-8",
    )


def _write_wav(path: Path, *, seconds: float = 3.0) -> None:
    sample_count = round(seconds * 16000)
    samples = (np.arange(sample_count, dtype=np.int32) % 20000 - 10000).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        destination.writeframes(samples.tobytes())


def _asr_diagnostics() -> ASRDecoderDiagnostics:
    return ASRDecoderDiagnostics(
        detected_language="en",
        language_probability=0.96,
        avg_log_probability=-0.2,
        no_speech_probability=0.02,
        compression_ratio=1.1,
        decoder_segment_count=1,
    )


def _write_asr_v1(root: Path, *, ordered_clip_ids: list[str]) -> ASRInventory:
    asr_root = root / "asr_pilot20"
    pairs = root / "production" / "pairs" / "in_pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_text("fixture-pairs\n", encoding="utf-8")
    targets: list[ASRTargetClip] = []
    jobs: list[ASRTurnJob] = []
    for clip_uid in ordered_clip_ids:
        media = root / "fixture_media"
        targets.append(
            ASRTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(media / f"{clip_uid}.mp4"),
                source_audio_path=str(media / f"{clip_uid}.wav"),
                source_audio_sha256="0" * 64,
                target_audio_binding_path=f"/fixture/{clip_uid}/audio_binding.json",
                subject_count=1,
                turn_count=1,
            )
        )
        jobs.append(
            ASRTurnJob(
                target_clip_uid=clip_uid,
                turn_id="turn_1",
                entity_id="e1",
                entity_occurrence_id=f"{clip_uid}/e1",
                start_time=0.0,
                end_time=0.5,
                source_audio_path=str(media / f"{clip_uid}.wav"),
                source_audio_sha256="0" * 64,
                source_sample_rate_hz=16000,
                source_channels=1,
                source_start_sample=0,
                source_end_sample=8000,
                segment_provenance=ASRTurnSegmentationProvenance(
                    boundary_source="frozen_audio_binding_turns_v1",
                    source_segment_id="turn_1",
                    speaker_cluster_id=None,
                    entity_binding_source="lr_asd_visual_entity_binding_v1",
                ),
            )
        )
    pairs_sha = _sha256(pairs)
    inventory = ASRInventory(
        source_pairs_path=str(pairs),
        source_pairs_sha256=pairs_sha,
        inventory_fingerprint=_asr_v1_inventory_fingerprint(
            source_pairs_sha256=pairs_sha,
            mode="pilot20",
            targets=targets,
            jobs=jobs,
        ),
        mode="pilot20",
        source_target_count=75,
        selected_target_count=20,
        selected_turn_count=20,
        selection_mode="multi_subject_first_then_clip_uid_v1",
        bounded_selection_applied=True,
        targets=targets,
        jobs=jobs,
    )
    provenance = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="a" * 64,
        device="cuda:0",
        compute_type="float16",
    ).provenance()
    records = [
        ASRTurnRecord(
            **job.model_dump(mode="python"),
            model_identifier=provenance.model_identifier,
            backend_provenance=provenance,
            preprocessing=ASRPreprocessingProvenance(
                source_sample_rate_hz=16000,
                source_channels=1,
                resampled=False,
                downmixed=False,
            ),
            request_fingerprint=f"{index:064x}",
            status="transcribed",
            text=f"baseline {index}",
            language="en",
            diagnostics=_asr_diagnostics(),
        )
        for index, job in enumerate(jobs, start=1)
    ]
    summary = ASRSummary(
        mode="pilot20",
        inventory_fingerprint=inventory.inventory_fingerprint,
        backend_provenance=provenance,
        source_target_count=75,
        target_clip_count=20,
        turn_count=20,
        transcribed_count=20,
        uncertain_count=0,
        failed_count=0,
        backend_call_count=20,
        failure_reason_counts={},
        bounded_selection_applied=True,
    )
    _write_json(asr_root / "inventory.json", inventory.model_dump(mode="json"))
    _write_jsonl(asr_root / "turns.jsonl", records)
    _write_json(asr_root / "summary.json", summary.model_dump(mode="json"))
    return inventory


def _cluster_binding(
    *,
    clip_uid: str,
    cluster_id: str,
    status: str,
) -> DiarizationClusterBinding:
    entity_id = "e1" if status == "candidate_mapped" else None
    support_ids = (
        [] if status == "unbound" else ["e1", "e2"] if status == "ambiguous" else ["e1"]
    )
    supports = [
        DiarizationEntitySupport(
            entity_id=entity,
            direct_support_samples=8000 - index,
            direct_support_seconds=(8000 - index) / 16000,
            weighted_support=float(8000 - index),
            contributing_binding_count=1,
        )
        for index, entity in enumerate(support_ids)
    ]
    total = sum(item.direct_support_samples for item in supports)
    return DiarizationClusterBinding(
        target_clip_uid=clip_uid,
        speaker_cluster_id=cluster_id,
        source_sample_rate_hz=16000,
        source_channels=1,
        status=status,
        entity_id=entity_id,
        cluster_segment_count=1,
        cluster_speaker_seconds=1.0,
        usable_anchor_sample_count=total,
        usable_anchor_duration=total / 16000,
        contested_anchor_sample_count=0,
        contested_anchor_duration=0.0,
        unmatched_anchor_sample_count=0,
        unmatched_anchor_duration=0.0,
        entity_supports=supports,
        top1_entity_id=support_ids[0] if support_ids else None,
        top1_support=float(supports[0].direct_support_samples) if supports else 0.0,
        top2_support=(
            float(supports[1].direct_support_samples) if len(supports) > 1 else 0.0
        ),
        top1_share=(supports[0].direct_support_samples / total if supports else None),
        top1_top2_margin=(
            float(
                supports[0].direct_support_samples - supports[1].direct_support_samples
            )
            if len(supports) > 1
            else None
        ),
        visual_anchor_coverage_ratio=0.5 if supports else 0.0,
    )


def _write_diarization_source(root: Path) -> DiarizationInventory:
    diar_root = root / "production" / "diarization"
    media_root = root / "fixture_media"
    pairs = root / "production" / "pairs" / "in_pairs.jsonl"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    if not pairs.exists():
        pairs.write_text("fixture-pairs\n", encoding="utf-8")
    targets: list[DiarizationTargetClip] = []
    raw_segments: list[RawDiarizationSegment] = []
    bound_segments: list[BoundDiarizationSegment] = []
    clusters: list[DiarizationClusterBinding] = []
    special_statuses = {
        ("clip-018", "segment_001"): "ambiguous",
        ("clip-017", "segment_001"): "conflict",
        ("clip-019", "segment_002"): "unbound",
    }
    for index in range(75):
        clip_uid = f"clip-{index:03d}"
        video = media_root / f"{clip_uid}.mp4"
        audio = media_root / f"{clip_uid}.wav"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video:{clip_uid}".encode())
        _write_wav(audio)
        audio_sha = _sha256(audio)
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video),
                source_audio_path=str(audio),
                source_audio_sha256=audio_sha,
                source_sample_rate_hz=16000,
                source_channels=1,
                source_frame_count=48000,
                target_audio_binding_path=f"/fixture/{clip_uid}/audio_binding.json",
                visual_references=[],
            )
        )
        segment_count = 3 if index < 10 or 20 <= index < 39 else 2
        segment_specs = [
            ("segment_001", "speaker_1", 0.0, 1.0),
            ("segment_002", "speaker_2", 0.5, 1.5),
            ("segment_003", "speaker_3", 1.5, 2.5),
        ][:segment_count]
        for segment_id, cluster_id, start, end in segment_specs:
            status = special_statuses.get(
                (clip_uid, segment_id),
                "candidate_mapped",
            )
            clamped = clip_uid == "clip-015" and segment_id == "segment_001"
            effective_end = 1.0 if clamped else end
            reported_end = 1.05 if clamped else end
            adjusted = reported_end != effective_end
            reconciliation = DiarizationBoundaryReconciliation(
                adjusted=adjusted,
                end_clamped=adjusted,
                end_overrun_samples=round((reported_end - effective_end) * 16000),
                end_overrun_seconds=reported_end - effective_end,
                reason="end_clamped_to_canonical_source" if adjusted else None,
            )
            raw_segments.append(
                RawDiarizationSegment(
                    target_clip_uid=clip_uid,
                    segment_id=segment_id,
                    speaker_cluster_id=cluster_id,
                    backend_speaker_label=cluster_id,
                    backend_reported_start_time=start,
                    backend_reported_end_time=reported_end,
                    backend_reported_start_sample=round(start * 16000),
                    backend_reported_end_sample=round(reported_end * 16000),
                    start_time=start,
                    end_time=effective_end,
                    source_start_sample=round(start * 16000),
                    source_end_sample=round(effective_end * 16000),
                    source_audio_path=str(audio),
                    source_audio_sha256=audio_sha,
                    source_sample_rate_hz=16000,
                    source_channels=1,
                    backend="fixture_diarizen",
                    model_identifier="fixture/diarizen",
                    model_fingerprint="b" * 64,
                    backend_configuration_fingerprint="c" * 64,
                    input_preprocessing="h3_diarizen_native_16k_mono_passthrough_v1",
                    boundary_reconciliation=reconciliation,
                )
            )
            propagated = clip_uid == "clip-016"
            mapped = status == "candidate_mapped"
            bound_segments.append(
                BoundDiarizationSegment(
                    target_clip_uid=clip_uid,
                    segment_id=segment_id,
                    speaker_cluster_id=cluster_id,
                    start_time=start,
                    end_time=effective_end,
                    source_start_sample=round(start * 16000),
                    source_end_sample=round(effective_end * 16000),
                    source_sample_rate_hz=16000,
                    source_channels=1,
                    cluster_binding_status=status,
                    entity_id="e1" if mapped else None,
                    entity_occurrence_id=f"{clip_uid}/e1" if mapped else None,
                    direct_anchor_samples=0 if propagated or not mapped else 8000,
                    direct_anchor_seconds=0.0 if propagated or not mapped else 0.5,
                    identity_scope=(
                        "cluster_propagated_only"
                        if propagated
                        else "direct_anchor_present"
                        if mapped
                        else "unresolved"
                    ),
                )
            )
            clusters.append(
                _cluster_binding(
                    clip_uid=clip_uid,
                    cluster_id=cluster_id,
                    status=status,
                )
            )

    pairs_sha = _sha256(pairs)
    inventory_fp = _diarization_inventory_fingerprint(
        source_pairs_sha256=pairs_sha,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
    )
    inventory = DiarizationInventory(
        mode="production",
        source_pairs_path=str(pairs),
        source_pairs_sha256=pairs_sha,
        inventory_fingerprint=inventory_fp,
        source_target_count=75,
        selected_target_count=75,
        selection_mode="complete_in_pair_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )
    status_counts = {
        status: sum(item.status == status for item in clusters)
        for status in ("candidate_mapped", "ambiguous", "unbound", "conflict")
    }
    mapped = float(status_counts["candidate_mapped"])
    summary = DiarizationSummary(
        mode="production",
        inventory_fingerprint=inventory_fp,
        backend_provenance=DiarizationBackendProvenance(
            backend="fixture_diarizen",
            model_identifier="fixture/diarizen",
            model_fingerprint="b" * 64,
            configuration_fingerprint="c" * 64,
            input_profile="legacy_16k_mono",
            input_preprocessing="h3_diarizen_native_16k_mono_passthrough_v1",
            source_sample_rate_hz=16000,
            source_channels=1,
        ),
        target_clip_count=75,
        ready_clip_count=75,
        empty_clip_count=0,
        failed_clip_count=0,
        backend_call_count=75,
        raw_segment_count=len(raw_segments),
        speaker_cluster_count=len(clusters),
        candidate_mapped_cluster_count=status_counts["candidate_mapped"],
        unbound_cluster_count=status_counts["unbound"],
        ambiguous_cluster_count=status_counts["ambiguous"],
        conflict_cluster_count=status_counts["conflict"],
        diarized_speaker_seconds=float(len(raw_segments)),
        diarized_wallclock_speech_seconds=float(len(raw_segments)),
        legacy_lr_asd_bound_seconds=40.0,
        usable_direct_anchor_seconds=mapped - 1.0,
        contested_anchor_seconds=0.0,
        mapped_speaker_seconds=mapped,
        mapped_direct_anchor_speaker_seconds=mapped - 1.0,
        identity_propagated_speaker_seconds=1.0,
        fully_propagated_segment_speaker_seconds=1.0,
        unbound_speaker_seconds=float(status_counts["unbound"]),
        ambiguous_speaker_seconds=float(status_counts["ambiguous"]),
        conflict_speaker_seconds=float(status_counts["conflict"]),
        legacy_coalesced_bound_turn_count=75,
        legacy_bound_turn_median_duration=0.5,
        diarization_segment_median_duration=1.0,
        boundary_adjusted_segment_count=1,
        boundary_adjusted_clip_count=1,
        end_clamped_segment_count=1,
        total_end_overrun_seconds=0.05,
        max_end_overrun_seconds=0.05,
        max_end_overrun_samples=800,
        median_positive_end_overrun_seconds=0.05,
        failure_reason_counts={},
    )
    _write_json(diar_root / "inventory.json", inventory.model_dump(mode="json"))
    _write_jsonl(diar_root / "raw_segments.jsonl", raw_segments)
    _write_jsonl(diar_root / "bound_segments.jsonl", bound_segments)
    _write_jsonl(diar_root / "cluster_bindings.jsonl", clusters)
    _write_json(diar_root / "summary.json", summary.model_dump(mode="json"))
    return inventory


def _audio_run(tmp_path: Path) -> tuple[Path, ASRInventory, DiarizationInventory]:
    root = tmp_path / "audio-run"
    ordered = [f"clip-{index:03d}" for index in reversed(range(20))]
    asr_v1 = _write_asr_v1(root, ordered_clip_ids=ordered)
    diarization = _write_diarization_source(root)
    return root, asr_v1, diarization


def _build(
    root: Path,
    *,
    mode: str,
    asr_v1: ASRInventory,
    diarization: DiarizationInventory,
):
    return build_asr_v2_inventory(
        audio_run_root=root,
        mode=mode,  # type: ignore[arg-type]
        expected_diarization_inventory_fingerprint=diarization.inventory_fingerprint,
        expected_asr_v1_inventory_fingerprint=asr_v1.inventory_fingerprint,
        expected_calibration_checkpoint_fingerprint="a" * 64,
    )


class _FakeBackend:
    def __init__(self, *, device: str = "cuda:3") -> None:
        self.calls: list[tuple[np.ndarray, int]] = []
        self.provenance = WhisperASRConfig(
            model_identifier="fixture/whisper-large-v3",
            checkpoint_fingerprint="a" * 64,
            device=device,
            compute_type="float16",
        ).provenance()

    def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
    ) -> ASRBackendResult:
        self.calls.append((audio.copy(), sample_rate_hz))
        return ASRBackendResult(
            text=f"v2 transcript {len(self.calls)}",
            language="en",
            diagnostics=_asr_diagnostics(),
        )


def test_inventory_uses_exact_pilot_order_and_every_diarizen_segment(
    tmp_path: Path,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    source_before = _tree_hashes(root)

    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )

    assert [item.target_clip_uid for item in inventory.targets] == [
        item.target_clip_uid for item in asr_v1.targets
    ]
    assert inventory.selected_target_count == 20
    assert inventory.selected_segment_count == 50
    assert inventory.source_target_count == 75
    assert inventory.schema_version == "r2v.h3.asr_v2_inventory.1"
    assert inventory.bounded_selection_applied is True
    assert inventory.parent_quota_applied is False
    assert inventory.donor_media_used is False
    assert inventory.cross_pair_jobs_created == 0
    assert [item.segment_id for item in inventory.jobs[:2]] == [
        "segment_001",
        "segment_002",
    ]
    assert _tree_hashes(root) == source_before
    assert "coalesce_audio_bindings" not in Path(asr_v2_module.__file__).read_text(
        encoding="utf-8"
    )


def test_production_inventory_is_complete_and_fixed_root(tmp_path: Path) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)

    inventory = _build(
        root,
        mode="production",
        asr_v1=asr_v1,
        diarization=diarization,
    )

    assert inventory.selected_target_count == inventory.source_target_count == 75
    assert isinstance(inventory, ASRV2ProductionInventory)
    assert inventory.schema_version == "r2v.h3.asr_v2_inventory.2"
    assert inventory.selected_segment_count == 179
    assert inventory.bounded_selection_applied is False
    assert inventory.production_inference_enabled is True
    assert inventory.asr_v2_policy_validated is True
    assert inventory.calibration_inventory_fingerprint == (
        ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT
    )
    assert inventory.calibration_checkpoint_fingerprint == (
        ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    )
    assert inventory.calibration_human_qa_total == 50
    assert inventory.calibration_human_qa_correct == 41
    assert inventory.calibration_human_qa_wrong == 3
    assert inventory.calibration_human_qa_uncertain == 6
    assert inventory.calibration_human_qa_unlabeled == 0
    assert inventory.text_usability_gate_applied is False
    assert inventory.transcript_confidence_threshold_used is False
    assert asr_v2_output_root(root, mode="pilot20") == root / "asr_v2_pilot20"
    assert asr_v2_output_root(root, mode="production") == (
        root / "production" / "asr_v2"
    )


def test_identity_statuses_remain_jobs_and_effective_eof_is_authoritative(
    tmp_path: Path,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    by_status = {item.cluster_binding_status: item for item in inventory.jobs}

    mapped = by_status["candidate_mapped"]
    assert mapped.speaker_cluster_id
    assert mapped.entity_id == "e1"
    assert mapped.entity_occurrence_id == f"{mapped.target_clip_uid}/e1"
    for status in ("ambiguous", "unbound", "conflict"):
        item = by_status[status]
        assert item.speaker_cluster_id
        assert item.entity_id is None
        assert item.entity_occurrence_id is None
        assert item.identity_scope == "unresolved"
    clamped = next(
        item for item in inventory.jobs if item.target_clip_uid == "clip-015"
    )
    assert clamped.source_end_sample == 16000
    assert clamped.boundary_reconciliation.end_clamped is True
    assert clamped.boundary_reconciliation.end_overrun_samples == 800
    first, second = inventory.jobs[:2]
    assert first.speaker_cluster_id != second.speaker_cluster_id
    assert first.start_time < second.end_time and second.start_time < first.end_time


def test_segment_schema_requires_speaker_and_enforces_nullable_identity(
    tmp_path: Path,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    payload = inventory.jobs[0].model_dump(mode="json")
    payload.pop("speaker_cluster_id")
    with pytest.raises(ValidationError):
        ASRV2SegmentJob.model_validate(payload)
    unresolved = next(
        item for item in inventory.jobs if item.cluster_binding_status == "unbound"
    ).model_dump(mode="json")
    unresolved["entity_id"] = "e9"
    unresolved["entity_occurrence_id"] = f"{unresolved['target_clip_uid']}/e9"
    with pytest.raises(ValidationError):
        ASRV2SegmentJob.model_validate(unresolved)


def test_pilot_transcribes_all_statuses_with_exact_crops_and_no_metadata(
    tmp_path: Path,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    backend = _FakeBackend()
    output = root / "asr_v2_pilot20"
    asr_v1_before = _tree_hashes(root / "asr_pilot20")
    diarization_before = _tree_hashes(root / "production" / "diarization")

    summary = run_asr_v2_transcription(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    assert len(backend.calls) == len(inventory.jobs) == 50
    assert all(sample_rate == 16000 for _, sample_rate in backend.calls)
    assert [len(audio) for audio, _ in backend.calls] == [
        item.source_end_sample - item.source_start_sample for item in inventory.jobs
    ]
    assert summary.candidate_mapped_segment_count == 47
    assert summary.ambiguous_segment_count == 1
    assert summary.unbound_segment_count == 1
    assert summary.conflict_segment_count == 1
    assert summary.unresolved_segment_count == 3
    assert summary.transcribed_count == 50
    assert summary.backend_call_count == 50
    assert summary.transcript_confidence_threshold_used is False
    assert summary.backend_provenance.device == "cuda:3"
    assert summary.backend_provenance.task == "transcribe"
    assert summary.backend_provenance.condition_on_previous_text is False
    assert summary.backend_provenance.vad_filter is False
    assert summary.backend_provenance.word_timestamps is False
    records = [
        ASRV2SegmentRecord.model_validate(json.loads(line))
        for line in (output / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {item.cluster_binding_status for item in records} == {
        "candidate_mapped",
        "ambiguous",
        "unbound",
        "conflict",
    }
    assert all(item.preprocessing.padding_seconds == 0 for item in records)
    review = (output / "review.html").read_text(encoding="utf-8")
    assert "ASR V1 REFERENCE - NOT GROUND TRUTH" in review
    assert "Export QA JSON" in review
    assert "UNRESOLVED" in review
    assert _tree_hashes(root / "asr_pilot20") == asr_v1_before
    assert _tree_hashes(root / "production" / "diarization") == diarization_before


def test_model_integrity_enforces_checkpoint_and_ignores_device() -> None:
    baseline = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="a" * 64,
        device="cuda:0",
        compute_type="float16",
    ).provenance()
    runtime = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="a" * 64,
        device="cuda:7",
        compute_type="float16",
    ).provenance()
    assert (
        build_asr_v2_backend_provenance(
            runtime=runtime,
            baseline=baseline,
        ).device
        == "cuda:7"
    )
    changed = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="b" * 64,
        device="cuda:0",
        compute_type="float16",
    ).provenance()
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        build_asr_v2_backend_provenance(runtime=changed, baseline=baseline)


def test_missing_v1_checkpoint_remains_explicitly_unverified() -> None:
    baseline = WhisperASRConfig(
        model_identifier="large-v3",
        checkpoint_fingerprint=None,
        device="cuda:0",
        compute_type="float16",
    ).provenance()
    runtime = WhisperASRConfig(
        model_identifier="large-v3",
        checkpoint_fingerprint="b" * 64,
        device="cuda:1",
        compute_type="float16",
    ).provenance()
    provenance = build_asr_v2_backend_provenance(runtime=runtime, baseline=baseline)
    assert provenance.checkpoint_fingerprint is None
    assert provenance.checkpoint_comparison == "unavailable_in_asr_v1"


def test_production_dry_run_is_model_free_and_real_execution_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    for name in ("primary_voice", "embedding"):
        path = root / "production" / name / "sentinel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    protected_roots = [
        root / "asr_pilot20",
        root / "production" / "diarization",
        root / "production" / "pairs",
        root / "production" / "primary_voice",
        root / "production" / "embedding",
    ]
    protected_before = {str(path): _tree_hashes(path) for path in protected_roots}
    inventory = _build(
        root,
        mode="production",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    monkeypatch.setattr(asr_v2_cli, "build_asr_v2_inventory", lambda **_: inventory)

    class _ForbiddenBackend:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("production proof must not load Whisper")

    monkeypatch.setattr(asr_v2_cli, "FasterWhisperASRBackend", _ForbiddenBackend)
    plan = asr_v2_cli.main(
        ["--audio-run-root", str(root), "--mode", "production", "--dry-run"]
    )
    assert plan["selected_target_count"] == 75
    assert plan["selected_segment_count"] == 179
    assert plan["production_inference_enabled"] is True
    assert plan["asr_v2_policy_validated"] is True
    assert plan["calibration_inventory_fingerprint"] == (
        ASR_V2_CALIBRATION_INVENTORY_FINGERPRINT
    )
    assert plan["calibration_checkpoint_fingerprint"] == (
        ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    )
    assert plan["calibration_human_qa_total"] == 50
    assert plan["calibration_human_qa_correct"] == 41
    assert plan["calibration_human_qa_wrong"] == 3
    assert plan["calibration_human_qa_uncertain"] == 6
    assert plan["calibration_human_qa_unlabeled"] == 0
    assert plan["text_usability_gate_applied"] is False
    assert plan["transcript_confidence_threshold_used"] is False

    backend = _FakeBackend(device="cuda:7")
    monkeypatch.setattr(
        asr_v2_cli,
        "FasterWhisperASRBackend",
        lambda _: backend,
    )
    result = asr_v2_cli.main(
        [
            "--audio-run-root",
            str(root),
            "--mode",
            "production",
            "--model",
            "fixture/whisper-large-v3",
            "--model-fingerprint",
            "a" * 64,
            "--device",
            "cuda:7",
        ]
    )
    assert result["stage_status"] == "completed"
    assert len(backend.calls) == 179
    summary = ASRV2ProductionSummary.model_validate(result["summary"])
    assert summary.schema_version == "r2v.h3.asr_v2_summary.2"
    assert summary.segment_count == 179
    assert summary.backend_call_count == 179
    assert summary.asr_v2_policy_validated is True
    assert summary.calibration_checkpoint_fingerprint == (
        ASR_V2_CALIBRATION_CHECKPOINT_FINGERPRINT
    )
    assert summary.text_usability_gate_applied is False
    assert summary.transcript_confidence_threshold_used is False
    assert summary.production_inference_enabled is True
    production_output = root / "production" / "asr_v2"
    assert (production_output / "segments.jsonl").is_file()
    inference_bytes = {
        name: (production_output / name).read_bytes()
        for name in ("inventory.json", "segments.jsonl", "summary.json")
    }
    regenerated = regenerate_asr_v2_review(
        output_root=production_output,
        expected_mode="production",
    )
    assert regenerated["backend_calls"] == 0
    assert all(
        (production_output / name).read_bytes() == value
        for name, value in inference_bytes.items()
    )
    assert {str(path): _tree_hashes(path) for path in protected_roots} == (
        protected_before
    )


def test_review_regeneration_is_model_free_and_preserves_inference_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    output = root / "asr_v2_pilot20"
    run_asr_v2_transcription(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(),
    )
    protected = {
        name: (output / name).read_bytes()
        for name in ("inventory.json", "segments.jsonl", "summary.json")
    }
    missing = output / "review_media" / inventory.jobs[0].target_clip_uid
    missing /= f"{inventory.jobs[0].segment_id}.wav"
    missing.unlink()
    result = regenerate_asr_v2_review(
        output_root=output,
        expected_mode="pilot20",
    )
    assert result["backend_calls"] == 0
    assert result["model_loaded"] is False
    assert result["regenerated_segment_media_count"] == 1
    assert missing.is_file()
    assert all(
        (output / name).read_bytes() == value for name, value in protected.items()
    )

    class _ForbiddenBackend:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("review regeneration must not load Whisper")

    monkeypatch.setattr(asr_v2_cli, "FasterWhisperASRBackend", _ForbiddenBackend)
    cli_result = asr_v2_cli.main(
        ["--audio-run-root", str(root), "--mode", "pilot20", "--regenerate-review"]
    )
    assert cli_result["backend_calls"] == 0


def test_frozen_pilot_v1_output_remains_reusable_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    output = root / "asr_v2_pilot20"
    run_asr_v2_transcription(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(),
    )
    protected = {
        name: (output / name).read_bytes()
        for name in ("inventory.json", "segments.jsonl", "summary.json")
    }
    monkeypatch.setattr(asr_v2_cli, "build_asr_v2_inventory", lambda **_: inventory)

    class _ForbiddenBackend:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("frozen pilot reuse must not load Whisper")

    monkeypatch.setattr(asr_v2_cli, "FasterWhisperASRBackend", _ForbiddenBackend)
    result = asr_v2_cli.main(
        [
            "--audio-run-root",
            str(root),
            "--mode",
            "pilot20",
            "--model",
            "fixture/whisper-large-v3",
            "--model-fingerprint",
            "a" * 64,
            "--device",
            "cuda:3",
        ]
    )
    assert result["stage_status"] == "reused"
    assert all(
        (output / name).read_bytes() == value for name, value in protected.items()
    )


def test_human_qa_contract_supports_unresolved_segments() -> None:
    export = ASRV2HumanQAExport(
        inventory_fingerprint="a" * 64,
        mode="pilot20",
        label_count=1,
        total_segment_count=2,
        counts=ASRHumanQACounts(CORRECT=1, WRONG=0, UNCERTAIN=0, UNLABELED=1),
        labels=[
            ASRV2HumanQALabel(
                target_clip_uid="clip-a",
                segment_id="segment_1",
                speaker_cluster_id="speaker_1",
                label="CORRECT",
            )
        ],
    )
    assert export.schema_version == "r2v.h3.asr_v2_human_qa.1"
    with pytest.raises(ValidationError):
        ASRV2HumanQALabel(
            target_clip_uid="clip-a",
            segment_id="segment_1",
            speaker_cluster_id="speaker_1",
            label="MAYBE",  # type: ignore[arg-type]
        )


def test_source_fingerprints_fail_closed_before_inference(tmp_path: Path) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    with pytest.raises(ValueError, match="DiariZen production fingerprint"):
        build_asr_v2_inventory(
            audio_run_root=root,
            mode="pilot20",
            expected_diarization_inventory_fingerprint="f" * 64,
            expected_asr_v1_inventory_fingerprint=asr_v1.inventory_fingerprint,
        )
    with pytest.raises(ValueError, match="ASR V1 pilot fingerprint"):
        build_asr_v2_inventory(
            audio_run_root=root,
            mode="pilot20",
            expected_diarization_inventory_fingerprint=diarization.inventory_fingerprint,
            expected_asr_v1_inventory_fingerprint="f" * 64,
        )
    with pytest.raises(ValueError, match="calibrated baseline"):
        build_asr_v2_inventory(
            audio_run_root=root,
            mode="production",
            expected_diarization_inventory_fingerprint=(
                diarization.inventory_fingerprint
            ),
            expected_asr_v1_inventory_fingerprint=asr_v1.inventory_fingerprint,
        )


def test_changed_source_audio_fails_without_backend_call(tmp_path: Path) -> None:
    root, asr_v1, diarization = _audio_run(tmp_path)
    inventory = _build(
        root,
        mode="pilot20",
        asr_v1=asr_v1,
        diarization=diarization,
    )
    changed_job = inventory.jobs[0]
    Path(changed_job.source_audio_path).write_bytes(b"changed")
    backend = _FakeBackend()

    summary = run_asr_v2_transcription(
        inventory=inventory,
        output_root=root / "asr_v2_pilot20",
        backend=backend,
    )

    assert summary.failed_count == 2
    assert summary.backend_call_count == len(inventory.jobs) - 2
    assert summary.failure_reason_counts == {"source_audio_changed": 2}


def test_v2_backend_receives_only_waveform_and_sample_rate() -> None:
    signature = asr_v2_module.ASRBackend.transcribe
    assert list(__import__("inspect").signature(signature).parameters) == [
        "self",
        "audio",
        "sample_rate_hz",
    ]
    assert asr_v2_module.ASR_V2_REQUEST_CONTRACT_VERSION == (
        "h3_whisper_diarizen_segment_asr_v2"
    )
    assert asr_v2_module.ASR_V2_PREPROCESSING_VERSION == (
        "pcm16_exact_diarizen_segment_crop_v1"
    )


def test_fake_backend_does_not_import_real_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = SimpleNamespace(
        __getattr__=lambda *_: (_ for _ in ()).throw(
            AssertionError("real model import is forbidden")
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", forbidden)
    monkeypatch.setitem(__import__("sys").modules, "diarizen", forbidden)
    backend = _FakeBackend(device="cpu")
    assert backend.calls == []
