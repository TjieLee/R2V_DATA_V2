from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import r2v_data_v2.h3.asr_v2_text_calibration as calibration_module
from r2v_data_v2.h3.asr_transcription import (
    ASRDecoderDiagnostics,
    ASRHumanQACounts,
    WhisperASRConfig,
)
from r2v_data_v2.h3.asr_v2_text_calibration import (
    CALIBRATION_OUTPUT_DIRECTORY,
    CalibrationCondition,
    TextCalibrationSegment,
    analyze_asr_v2_text_usability,
    build_calibration_sweep,
    raw_text_available,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    ASRV2HumanQAExport,
    ASRV2HumanQALabel,
    ASRV2Inventory,
    ASRV2PreprocessingProvenance,
    ASRV2ProductionInventory,
    ASRV2SegmentJob,
    ASRV2SegmentRecord,
    ASRV2TargetClip,
    build_asr_v2_backend_provenance,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    _inventory_fingerprint as _asr_v2_inventory_fingerprint,
)
from r2v_data_v2.h3.asr_v2_transcription import _summary as _asr_v2_summary
from r2v_data_v2.h3.diarization_binding import DiarizationBoundaryReconciliation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
                value.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _backend_provenance():
    baseline = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="a" * 64,
        device="cuda:0",
        compute_type="float16",
    ).provenance()
    runtime = WhisperASRConfig(
        model_identifier="fixture/whisper-large-v3",
        checkpoint_fingerprint="a" * 64,
        device="cuda:1",
        compute_type="float16",
    ).provenance()
    return baseline, build_asr_v2_backend_provenance(runtime=runtime, baseline=baseline)


def _job(clip_uid: str, segment_index: int) -> ASRV2SegmentJob:
    start = float(segment_index - 1)
    end = start + 1.25
    return ASRV2SegmentJob(
        target_clip_uid=clip_uid,
        segment_id=f"segment_{segment_index:03d}",
        speaker_cluster_id=f"speaker_{(segment_index % 2) + 1}",
        cluster_binding_status="candidate_mapped",
        entity_id="e1",
        entity_occurrence_id=f"{clip_uid}/e1",
        identity_scope=(
            "direct_anchor_present" if segment_index % 2 else "cluster_propagated_only"
        ),
        start_time=start,
        end_time=end,
        source_start_sample=round(start * 16000),
        source_end_sample=round(end * 16000),
        source_audio_path=f"/fixture/{clip_uid}.wav",
        source_audio_sha256="b" * 64,
        source_sample_rate_hz=16000,
        source_channels=1,
        source_diarization_inventory_fingerprint="c" * 64,
        source_segment_id=f"segment_{segment_index:03d}",
        boundary_reconciliation=DiarizationBoundaryReconciliation(
            adjusted=False,
            end_clamped=False,
            end_overrun_samples=0,
            end_overrun_seconds=0.0,
        ),
    )


def _targets_and_jobs(
    *, target_count: int, segment_count: int
) -> tuple[list[ASRV2TargetClip], list[ASRV2SegmentJob]]:
    base = segment_count // target_count
    extra = segment_count % target_count
    targets: list[ASRV2TargetClip] = []
    jobs: list[ASRV2SegmentJob] = []
    for index in range(target_count):
        clip_uid = f"clip-{index:03d}"
        count = base + (1 if index < extra else 0)
        targets.append(
            ASRV2TargetClip(
                target_clip_uid=clip_uid,
                target_video_path=f"/fixture/{clip_uid}.mp4",
                source_audio_path=f"/fixture/{clip_uid}.wav",
                source_audio_sha256="b" * 64,
                source_sample_rate_hz=16000,
                source_channels=1,
                source_frame_count=160000,
                segment_count=count,
            )
        )
        jobs.extend(_job(clip_uid, segment_index + 1) for segment_index in range(count))
    return targets, jobs


def _inventory_values(*, mode: str) -> dict[str, object]:
    baseline, _ = _backend_provenance()
    target_count, segment_count = (20, 50) if mode == "pilot20" else (75, 179)
    targets, jobs = _targets_and_jobs(
        target_count=target_count, segment_count=segment_count
    )
    return {
        "mode": mode,
        "source_diarization_root": "/fixture/diarization",
        "source_diarization_inventory_path": "/fixture/diarization/inventory.json",
        "source_diarization_inventory_sha256": "1" * 64,
        "source_diarization_inventory_fingerprint": "c" * 64,
        "source_diarization_raw_segments_path": "/fixture/diarization/raw.jsonl",
        "source_diarization_raw_segments_sha256": "2" * 64,
        "source_diarization_bound_segments_path": "/fixture/diarization/bound.jsonl",
        "source_diarization_bound_segments_sha256": "3" * 64,
        "source_diarization_cluster_bindings_path": "/fixture/diarization/clusters.jsonl",
        "source_diarization_cluster_bindings_sha256": "4" * 64,
        "source_diarization_summary_path": "/fixture/diarization/summary.json",
        "source_diarization_summary_sha256": "5" * 64,
        "source_asr_v1_inventory_path": "/fixture/asr_v1/inventory.json",
        "source_asr_v1_inventory_sha256": "6" * 64,
        "source_asr_v1_inventory_fingerprint": "d" * 64,
        "source_asr_v1_turns_path": "/fixture/asr_v1/turns.jsonl",
        "source_asr_v1_turns_sha256": "7" * 64,
        "source_asr_v1_summary_path": "/fixture/asr_v1/summary.json",
        "source_asr_v1_summary_sha256": "8" * 64,
        "baseline_asr_v1_backend_provenance": baseline,
        "baseline_asr_v1_turn_count": 82,
        "baseline_asr_v1_turn_median_duration": 1.02,
        "source_target_count": 75,
        "selected_target_count": target_count,
        "selected_segment_count": segment_count,
        "targets": targets,
        "jobs": jobs,
    }


def _inventory(*, mode: str):
    values = _inventory_values(mode=mode)
    if mode == "pilot20":
        values.update(
            {
                "selection_mode": "same_ordered_asr_v1_pilot20_targets_v1",
                "bounded_selection_applied": True,
                "production_inference_blocked": False,
            }
        )
        provisional = ASRV2Inventory(**values, inventory_fingerprint="0" * 64)
        payload = provisional.model_dump(mode="json", exclude={"inventory_fingerprint"})
        return ASRV2Inventory(
            **payload,
            inventory_fingerprint=_asr_v2_inventory_fingerprint(payload),
        )
    values.update(
        {
            "selection_mode": "complete_production_diarization_targets_v1",
            "bounded_selection_applied": False,
            "production_inference_enabled": True,
            "asr_v2_policy_validated": True,
        }
    )
    provisional = ASRV2ProductionInventory(**values, inventory_fingerprint="0" * 64)
    payload = provisional.model_dump(mode="json", exclude={"inventory_fingerprint"})
    return ASRV2ProductionInventory(
        **payload,
        inventory_fingerprint=_asr_v2_inventory_fingerprint(payload),
    )


def _diagnostics(index: int) -> ASRDecoderDiagnostics:
    if index < 41:
        avg_log = -0.12 - index * 0.006
        no_speech = 0.01 + index * 0.001
        language = 0.99 - index * 0.002
        compression = 1.02 + index * 0.004
    elif index < 44:
        avg_log = -1.4 - (index - 41) * 0.1
        no_speech = 0.50 + (index - 41) * 0.05
        language = 0.35 - (index - 41) * 0.05
        compression = 2.2 + (index - 41) * 0.1
    else:
        avg_log = -0.8 - (index - 44) * 0.03
        no_speech = 0.18 + (index - 44) * 0.02
        language = 0.72 - (index - 44) * 0.02
        compression = 1.45 + (index - 44) * 0.03
    return ASRDecoderDiagnostics(
        detected_language="zh",
        language_probability=language,
        avg_log_probability=avg_log,
        no_speech_probability=no_speech,
        compression_ratio=compression,
        decoder_segment_count=1,
    )


def _records(inventory, *, uncertain_count: int) -> list[ASRV2SegmentRecord]:
    _, provenance = _backend_provenance()
    records = []
    uncertain_start = len(inventory.jobs) - uncertain_count
    for index, job in enumerate(inventory.jobs):
        uncertain = index >= uncertain_start
        records.append(
            ASRV2SegmentRecord(
                **job.model_dump(mode="python"),
                model_identifier=provenance.model_identifier,
                backend_provenance=provenance,
                preprocessing=ASRV2PreprocessingProvenance(
                    source_sample_rate_hz=16000,
                    source_channels=1,
                    resampled=False,
                    downmixed=False,
                ),
                request_fingerprint=f"{index + 1:064x}",
                status="uncertain" if uncertain else "transcribed",
                text=None if uncertain else f"transcript {index}",
                language=None if uncertain else "zh",
                diagnostics=_diagnostics(index % 50),
                warnings=["empty_transcript"] if uncertain else [],
            )
        )
    return records


def _write_asr_root(
    root: Path, *, mode: str
) -> tuple[object, list[ASRV2SegmentRecord]]:
    inventory = _inventory(mode=mode)
    records = _records(inventory, uncertain_count=2 if mode == "pilot20" else 3)
    summary = _asr_v2_summary(
        inventory=inventory,
        records=records,
        backend_provenance=records[0].backend_provenance,
        backend_call_count=len(records),
        failure_counts=Counter(),
    )
    _write_json(root / "inventory.json", inventory.model_dump(mode="json"))
    _write_jsonl(root / "segments.jsonl", records)
    _write_json(root / "summary.json", summary.model_dump(mode="json"))
    return inventory, records


def _write_qa(
    path: Path,
    *,
    inventory_fingerprint: str,
    records: list[ASRV2SegmentRecord],
) -> ASRV2HumanQAExport:
    labels = []
    for index, record in enumerate(records):
        label = "CORRECT" if index < 41 else "WRONG" if index < 44 else "UNCERTAIN"
        labels.append(
            ASRV2HumanQALabel(
                target_clip_uid=record.target_clip_uid,
                segment_id=record.segment_id,
                speaker_cluster_id=record.speaker_cluster_id,
                label=label,
            )
        )
    qa = ASRV2HumanQAExport(
        inventory_fingerprint=inventory_fingerprint,
        mode="pilot20",
        label_count=50,
        total_segment_count=50,
        counts=ASRHumanQACounts(CORRECT=41, WRONG=3, UNCERTAIN=6, UNLABELED=0),
        labels=labels,
    )
    _write_json(path, qa.model_dump(mode="json"))
    return qa


@pytest.fixture
def source_run(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "audio-run"
    pilot_inventory, pilot_records = _write_asr_root(
        root / "asr_v2_pilot20", mode="pilot20"
    )
    _write_asr_root(root / "production" / "asr_v2", mode="production")
    qa_path = tmp_path / "qa.json"
    _write_qa(
        qa_path,
        inventory_fingerprint=pilot_inventory.inventory_fingerprint,
        records=pilot_records,
    )
    return root, qa_path, pilot_inventory.inventory_fingerprint


def _analyze(source_run, *, overwrite: bool = False, before_publish=None):
    root, qa_path, fingerprint = source_run
    return analyze_asr_v2_text_usability(
        audio_run_root=root,
        qa_json=qa_path,
        overwrite=overwrite,
        expected_pilot_inventory_fingerprint=fingerprint,
        before_publish=before_publish,
    )


def _source_hashes(root: Path) -> dict[str, str]:
    paths = [
        *(root / "asr_v2_pilot20").glob("*"),
        *(root / "production" / "asr_v2").glob("*"),
    ]
    return {str(path): _sha256(path) for path in paths if path.is_file()}


def test_analyzer_publishes_fixed_model_free_read_only_report(source_run) -> None:
    root, qa_path, _ = source_run
    before = _source_hashes(root)
    qa_bytes = qa_path.read_bytes()
    summary = _analyze(source_run)

    output = root / CALIBRATION_OUTPUT_DIRECTORY
    assert output.name == "asr_v2_text_calibration"
    assert sorted(path.name for path in output.iterdir()) == [
        "human_qa.json",
        "inventory.json",
        "joined_segments.jsonl",
        "report.html",
        "summary.json",
        "sweep.json",
    ]
    assert (output / "human_qa.json").read_bytes() == qa_bytes
    assert summary.pilot_segment_count == 50
    assert summary.production_segment_count == 179
    assert summary.hard_case_count == 9
    assert summary.text_usability_policy_validated is False
    assert summary.text_usability_gate_applied is False
    assert summary.transcript_confidence_threshold_used is False
    assert _source_hashes(root) == before

    inventory = json.loads((output / "inventory.json").read_text())
    assert inventory["model_calls"] == inventory["gpu_calls"] == 0
    assert inventory["production_asr_modified"] is False
    assert inventory["diarization_modified"] is False
    assert inventory["voice_pair_embedding_modified"] is False
    assert inventory["human_qa_sha256"] == hashlib.sha256(qa_bytes).hexdigest()


def test_qa_fingerprint_duplicate_missing_and_extra_fail_closed(source_run) -> None:
    root, qa_path, fingerprint = source_run
    payload = json.loads(qa_path.read_text())

    payload["inventory_fingerprint"] = "f" * 64
    _write_json(qa_path, payload)
    with pytest.raises(ValueError, match="fingerprint"):
        _analyze((root, qa_path, fingerprint))

    _, records = _write_asr_root(root / "asr_v2_pilot20", mode="pilot20")
    _write_qa(qa_path, inventory_fingerprint=fingerprint, records=records)
    payload = json.loads(qa_path.read_text())
    payload["labels"][1] = payload["labels"][0]
    with pytest.raises(ValueError, match="unique"):
        _write_json(qa_path, payload)
        _analyze((root, qa_path, fingerprint))

    _write_qa(qa_path, inventory_fingerprint=fingerprint, records=records)
    payload = json.loads(qa_path.read_text())
    payload["labels"][0]["segment_id"] = "unknown-segment"
    _write_json(qa_path, payload)
    with pytest.raises(ValueError, match="one-to-one"):
        _analyze((root, qa_path, fingerprint))

    _write_qa(qa_path, inventory_fingerprint=fingerprint, records=records)
    payload = json.loads(qa_path.read_text())
    payload["labels"] = payload["labels"][:-1]
    payload["label_count"] = 49
    payload["counts"]["UNCERTAIN"] = 5
    payload["counts"]["UNLABELED"] = 1
    _write_json(qa_path, payload)
    with pytest.raises(ValueError, match="cover every"):
        _analyze((root, qa_path, fingerprint))

    _write_qa(qa_path, inventory_fingerprint=fingerprint, records=records)
    payload = json.loads(qa_path.read_text())
    payload["counts"]["CORRECT"] = 40
    payload["counts"]["WRONG"] = 4
    payload["labels"][0]["label"] = "WRONG"
    _write_json(qa_path, payload)
    with pytest.raises(ValueError, match="frozen counts"):
        _analyze((root, qa_path, fingerprint))


def test_raw_text_baseline_and_null_diagnostics_are_explicit(source_run) -> None:
    _analyze(source_run)
    root, _, _ = source_run
    output = root / CALIBRATION_OUTPUT_DIRECTORY
    joined = [
        json.loads(line)
        for line in (output / "joined_segments.jsonl").read_text().splitlines()
    ]
    assert len(joined) == 50
    assert sum(row["qa_label"] == "WRONG" for row in joined) == 3
    assert sum(row["qa_label"] == "UNCERTAIN" for row in joined) == 6
    assert all(not row["raw_text_available"] for row in joined[-2:])
    assert raw_text_available(TextCalibrationSegment.model_validate(joined[0]))

    sweep = json.loads((output / "sweep.json").read_text())
    assert sweep["baseline"]["rule"] == "raw_text_available_only"
    assert sweep["baseline"]["retained_segment_count"] == 48
    assert all(
        condition["missing_value_behavior"] == "not_retained"
        for rule in sweep["one_dimensional_rules"]
        for condition in rule["conditions"]
    )


def test_uncertain_and_missing_diagnostics_never_pass_a_candidate_gate() -> None:
    base = {
        "target_clip_uid": "clip",
        "speaker_cluster_id": "speaker",
        "duration_seconds": 2.0,
        "avg_log_probability": -0.01,
        "no_speech_probability": 0.0,
        "compression_ratio": 1.0,
        "decoder_segment_count": 1,
        "cluster_binding_status": "candidate_mapped",
        "identity_scope": "direct_anchor_present",
    }
    backend_uncertain = TextCalibrationSegment(
        **base,
        segment_id="uncertain",
        qa_label="UNCERTAIN",
        status="uncertain",
        text=None,
        text_present=False,
        raw_text_available=False,
        language_probability=1.0,
    )
    missing = TextCalibrationSegment(
        **base,
        segment_id="missing",
        qa_label="CORRECT",
        status="transcribed",
        text="visible text",
        text_present=True,
        raw_text_available=True,
        language_probability=None,
    )
    condition = CalibrationCondition(
        diagnostic="language_probability", operator="ge", threshold=0.9
    )
    assert calibration_module._rule_passes(backend_uncertain, [condition]) is False
    assert calibration_module._rule_passes(missing, [condition]) is False
    result = calibration_module._evaluate_rule(
        rows=[backend_uncertain, missing],
        conditions=[condition],
        ranges={name: 1.0 for name in calibration_module.DIAGNOSTIC_ORDER},
    )
    assert result.retained_segment_count == 0
    assert result.missing_value_count == 1


def test_sweeps_shortlists_and_pareto_are_deterministic(source_run) -> None:
    _analyze(source_run)
    root, _, _ = source_run
    first = (root / CALIBRATION_OUTPUT_DIRECTORY / "sweep.json").read_bytes()
    _analyze(source_run, overwrite=True)
    second = (root / CALIBRATION_OUTPUT_DIRECTORY / "sweep.json").read_bytes()
    assert first == second

    sweep = json.loads(second)
    assert sweep["learned_classifier_used"] is False
    assert sweep["fixed_half_threshold_used"] is False
    assert all(len(rule["conditions"]) == 1 for rule in sweep["one_dimensional_rules"])
    assert all(len(rule["conditions"]) == 2 for rule in sweep["two_condition_rules"])
    assert all(rule["wrong_retained"] == 0 for rule in sweep["zero_wrong_shortlist"])
    assert all(
        rule["wrong_retained"] <= 1 for rule in sweep["at_most_one_wrong_shortlist"]
    )
    assert sweep["pareto_frontier"]
    representations = [
        rule["rule"]
        for rule in [
            *sweep["one_dimensional_rules"],
            *sweep["two_condition_rules"],
        ]
    ]
    assert all("cluster_binding_status" not in value for value in representations)
    assert all("identity_scope" not in value for value in representations)


def test_thresholds_derive_from_observed_values_without_fixed_half() -> None:
    greater = calibration_module._threshold_candidates([0.1, 0.2], operator="ge")
    lower = calibration_module._threshold_candidates([0.1, 0.2], operator="le")
    assert greater == [
        0.1,
        pytest.approx(0.15),
    ]
    assert lower == [pytest.approx(0.15), 0.2]
    assert 0.5 not in greater
    assert 0.5 not in lower
    conditions = [
        CalibrationCondition(
            diagnostic="language_probability", operator="ge", threshold=value
        )
        for value in greater
    ]
    rows = [
        TextCalibrationSegment(
            target_clip_uid="c",
            segment_id=f"s{index}",
            speaker_cluster_id="spk",
            qa_label="CORRECT",
            status="transcribed",
            text="text",
            text_present=True,
            raw_text_available=True,
            duration_seconds=1.0,
            language_probability=value,
            avg_log_probability=-0.1,
            no_speech_probability=value,
            compression_ratio=1.0,
            decoder_segment_count=1,
            cluster_binding_status="candidate_mapped",
            identity_scope="direct_anchor_present",
        )
        for index, value in enumerate((0.1, 0.2))
    ]
    decisions = [
        tuple(calibration_module._condition_passes(row, condition) for row in rows)
        for condition in conditions
    ]
    assert len(decisions) == len(set(decisions))

    lower_conditions = [
        CalibrationCondition(
            diagnostic="no_speech_probability", operator="le", threshold=value
        )
        for value in lower
    ]
    lower_decisions = [
        tuple(calibration_module._condition_passes(row, condition) for row in rows)
        for condition in lower_conditions
    ]
    assert len(lower_decisions) == len(set(lower_decisions))


def test_production_shadow_is_dynamic_coverage_only_and_identity_is_not_gate(
    source_run,
) -> None:
    summary = _analyze(source_run)
    root, _, _ = source_run
    sweep = json.loads((root / CALIBRATION_OUTPUT_DIRECTORY / "sweep.json").read_text())
    assert summary.production_segment_count == 179
    assert sweep["production_shadow"]
    assert all(
        row["production_segment_count"] == 179
        and row["production_precision_claimed"] is False
        for row in sweep["production_shadow"]
    )
    assert all(
        set(row["retained_by_cluster_binding_status"])
        == {"candidate_mapped", "ambiguous", "unbound", "conflict"}
        for row in sweep["production_shadow"]
    )
    report = (root / CALIBRATION_OUTPUT_DIRECTORY / "report.html").read_text()
    assert "language_probability" in report
    assert "not transcript correctness probability" in report
    assert "Coverage only" in report


def test_atomic_failure_preserves_existing_output(source_run) -> None:
    _analyze(source_run)
    root, _, _ = source_run
    output = root / CALIBRATION_OUTPUT_DIRECTORY
    before = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}

    def fail() -> None:
        raise RuntimeError("publication probe")

    with pytest.raises(RuntimeError, match="publication probe"):
        _analyze(source_run, overwrite=True, before_publish=fail)
    after = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    assert after == before
    assert not list(root.glob(f".{CALIBRATION_OUTPUT_DIRECTORY}.tmp-*"))


def test_no_backend_or_gpu_contract_is_encoded(source_run) -> None:
    _analyze(source_run)
    root, _, _ = source_run
    inventory = json.loads(
        (root / CALIBRATION_OUTPUT_DIRECTORY / "inventory.json").read_text()
    )
    summary = json.loads(
        (root / CALIBRATION_OUTPUT_DIRECTORY / "summary.json").read_text()
    )
    assert inventory["model_calls"] == 0
    assert inventory["gpu_calls"] == 0
    assert summary["raw_asr_preserved"] is True
    assert summary["speaker_identity_independent"] is True
    assert summary["voice_reference_quality_independent"] is True
    source = Path(calibration_module.__file__).read_text()
    assert "FasterWhisperASRBackend" not in source
    assert "DiariZen" not in source


def test_build_sweep_accepts_cpu_only_data() -> None:
    rows = [
        TextCalibrationSegment(
            target_clip_uid="clip",
            segment_id=f"segment-{index}",
            speaker_cluster_id="speaker",
            qa_label="CORRECT" if index == 0 else "WRONG",
            status="transcribed",
            text="text",
            text_present=True,
            raw_text_available=True,
            duration_seconds=1.0 + index,
            language_probability=0.9 - index * 0.4,
            avg_log_probability=-0.1 - index,
            no_speech_probability=0.1 + index * 0.4,
            compression_ratio=1.0 + index,
            decoder_segment_count=1,
            cluster_binding_status="candidate_mapped",
            identity_scope="direct_anchor_present",
        )
        for index in range(2)
    ]
    production = _records(_inventory(mode="production"), uncertain_count=3)
    sweep, distributions = build_calibration_sweep(
        joined=rows, production_records=production
    )
    assert sweep.zero_wrong_shortlist
    assert distributions["duration_seconds"]["CORRECT"].count == 1
