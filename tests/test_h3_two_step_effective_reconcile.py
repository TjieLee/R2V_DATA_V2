from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.audio_reuse_prepared import _hash
from r2v_data_v2.h3.mimo25_av_reconcile import _job
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    MimoStemReconcileSummary,
)
from tests.h3_audio_reuse_prepared_helpers import stem_record_values
from tests.test_h3_mimo26_two_step import _reconcile, _setup


def _inputs(tmp_path, monkeypatch):
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    template = jobs[0]
    annotation = _reconcile(backend, template, stems).annotation
    source = stem_record_values(tmp_path, template, annotation, stems[0])
    source.update(
        backend_provenance=backend.provenance.model_dump(mode="json"),
        raw_responses=["visual", "joint"], visual_raw_response="visual", speech_av_raw_response="joint",
        audio_finalize_raw_response=None, av_model_call_count=1, model_call_count=2,
        diagnostics=[
            {**source["diagnostics"][0], "input_modality": modality}
            for modality in ("target_video_visual_only", "target_video_joint_av_audio")
        ],
    )
    inventory, rows, stem_rows = [], [], []
    for index in range(20):
        values = template.model_dump(mode="json", exclude={"request_fingerprint"})
        values["clip_uid"] = f"{index:024x}"
        values["binding_evidence_mode"] = "none"
        job = _job(values)
        inventory.append(job)
        stem = stems[0].model_copy(update={"clip_uid": job.clip_uid})
        stem_rows.append(stem)
        row = copy.deepcopy(source)
        row.update(clip_uid=job.clip_uid, source_job_fingerprint=job.request_fingerprint)
        if index >= 11:
            row.update(status="failed", failure_code="original_failure", failure_reason=f"failure {index}")
        row["record_fingerprint"] = _hash({k: v for k, v in row.items() if k != "record_fingerprint"})
        rows.append(row)
    base, override = tmp_path / "original-reconcile", tmp_path / "override-reconcile"
    _write_root(base, inventory, rows)
    replacement = copy.deepcopy(rows[15:17])
    for row in replacement:
        row.update(status="ready", failure_code=None, failure_reason=None)
        row["record_fingerprint"] = _hash({k: v for k, v in row.items() if k != "record_fingerprint"})
    _write_root(override, inventory[15:17], replacement)
    return base, override, inventory, rows, stem_rows


def _write_root(root, jobs, rows):
    root.mkdir()
    (root / "source_contract.json").write_text(json.dumps({
        "schema_version": "r2v.h3.mimo25_stem_source_contract.1", "binding_evidence_mode": "none",
        "clip_uids": [j.clip_uid for j in jobs], "jobs": [j.model_dump(mode="json") for j in jobs],
    }))
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "summary.json").write_text(MimoStemReconcileSummary(
        schema_version="r2v.h3.mimo25_stem_reconcile_summary.17", route="resolved",
        clip_count=len(jobs), processed_clip_count=len(rows), skipped_clip_count=0,
        clip_uids=[j.clip_uid for j in jobs], processed_clip_uids=[j.clip_uid for j in jobs],
        skipped_clips=[], diarization_failed_clips=[],
        ready_count=sum(row["status"] == "ready" for row in rows),
        failed_count=sum(row["status"] == "failed" for row in rows),
        model_call_count=sum(row["model_call_count"] for row in rows),
        av_model_call_count=sum(row["av_model_call_count"] for row in rows),
        visual_model_call_count=sum(row["visual_model_call_count"] for row in rows),
        audio_model_call_count=0, text_model_call_count=0,
    ).model_dump_json())


def test_effective20_overlay_preserves_order_audit_and_three_failures(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import two_step_effective_reconcile as effective

    base, override, jobs, rows, stems = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(effective, "load_stem_source", lambda path: (None, stems, None))
    replayed = jobs[11:15]

    def replay(job, record):
        values = record.model_dump(mode="json", exclude={"record_fingerprint"})
        values.update(status="ready", failure_code=None, failure_reason=None)
        return MimoStemReconcileRecord(**values, record_fingerprint=_hash(values))

    monkeypatch.setattr(effective, "replay_two_step_record", replay)
    before = {p: p.read_bytes() for root in (base, override) for p in root.iterdir()}
    output = tmp_path / "effective"
    result = effective.build_effective_reconcile(
        base_root=base, override_root=override, output_root=output,
        replay_clip_uids=[j.clip_uid for j in replayed],
    )
    assert (result.clip_count, result.ready_count, result.failed_count) == (20, 17, 3)
    assert (output / "source_contract.json").read_bytes() == before[base / "source_contract.json"]
    actual = [MimoStemReconcileRecord.model_validate_json(line)
              for line in (output / "records.jsonl").read_text().splitlines()]
    assert [r.clip_uid for r in actual] == [j.clip_uid for j in jobs]
    assert [r.failure_reason for r in actual[-3:]] == ["failure 17", "failure 18", "failure 19"]
    audit = json.loads((output / "effective_sources.json").read_text())
    assert audit["new_model_call_count"] == 0
    assert audit["retained_inference_model_call_count"] == result.model_call_count == 40
    assert len(audit["records"]) == 20
    original_by_clip = {row["clip_uid"]: row for row in rows}
    override_by_clip = {row["clip_uid"]: row for row in map(json.loads, (override / "records.jsonl").read_text().splitlines())}
    for item, row in zip(audit["records"], actual, strict=True):
        source = override if item["operation"] == "real_override" else base
        assert item["source_records_path"] == str(source / "records.jsonl")
        expected = override_by_clip if source == override else original_by_clip
        assert item["source_record_fingerprint"] == expected[row.clip_uid]["record_fingerprint"]
        assert item["postprocess_backend_version"] == (
            "r2v.h3.mimo25_backend.85" if item["operation"] == "cpu_replay" else None
        )
        assert item["effective_validation_backend_version"] == "r2v.h3.mimo25_backend.85"
    assert before == {p: p.read_bytes() for p in before}
    with pytest.raises(FileExistsError):
        effective.build_effective_reconcile(base_root=base, override_root=override, output_root=output,
                                            replay_clip_uids=[j.clip_uid for j in replayed])


@pytest.mark.parametrize("field", ["source_job_fingerprint", "source_stem_record_fingerprint", "record_fingerprint"])
def test_effective_rejects_invalid_source_lineage(tmp_path, monkeypatch, field):
    from r2v_data_v2.h3 import two_step_effective_reconcile as effective

    base, override, _, _, stems = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(effective, "load_stem_source", lambda path: (None, stems, None))
    path = override / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = "e" * 64
    if field != "record_fingerprint":
        rows[0]["record_fingerprint"] = _hash({k: v for k, v in rows[0].items() if k != "record_fingerprint"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="fingerprint"):
        effective.build_effective_reconcile(base_root=base, override_root=override,
                                            output_root=tmp_path / "effective", replay_clip_uids=[])
    assert not (tmp_path / "effective").exists()


def test_cached_replay_preserves_actual_inference_audit_without_media_or_http(tmp_path, monkeypatch):
    from r2v_data_v2.h3.two_step_effective_reconcile import replay_two_step_record

    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    result = _reconcile(backend, jobs[0], stems)
    values = stem_record_values(tmp_path, jobs[0], result.annotation, stems[0])
    values.update(
        backend_provenance=backend.provenance.model_dump(mode="json"),
        status="failed", failure_code="previous_failure", failure_reason="format",
        raw_responses=list(result.raw_responses), visual_raw_response=result.visual_raw_response,
        speech_av_raw_response=result.speech_av_raw_response, audio_finalize_raw_response=None,
        diagnostics=[d.model_dump(mode="json") for d in result.diagnostics],
        av_model_call_count=1, model_call_count=2,
    )
    values["diagnostics"][1]["warnings"].append("saved_inference_warning")
    values.pop("record_fingerprint")
    original = MimoStemReconcileRecord(**values, record_fingerprint=_hash(values))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU replay must not read media or make HTTP requests")

    class ForbiddenClient:
        __init__ = forbidden

    monkeypatch.setattr("r2v_data_v2.h3.mimo25_backend.MimoMediaResolver.resolve", forbidden)
    monkeypatch.setattr("r2v_data_v2.h3.mimo25_backend.OpenAI", ForbiddenClient)
    restored = replay_two_step_record(jobs[0], original)
    assert restored.status == "ready"
    assert restored.annotation == result.annotation
    assert restored.raw_responses == original.raw_responses
    assert restored.model_call_count == original.model_call_count == 2
    for saved, current in zip(original.diagnostics, restored.diagnostics, strict=True):
        assert saved.model_dump(exclude={"warnings"}) == current.model_dump(exclude={"warnings"})
        assert set(saved.warnings) <= set(current.warnings)
    MimoStemReconcileRecord.model_validate(restored.model_dump())


def test_effective_rejects_false_ready_dialogue_and_bad_summary(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import two_step_effective_reconcile as effective

    base, override, _, rows, stems = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(effective, "load_stem_source", lambda path: (None, stems, None))
    summary_path = base / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(ready_count=10, failed_count=10)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary"):
        effective.build_effective_reconcile(base_root=base, override_root=override,
                                            output_root=tmp_path / "invalid", replay_clip_uids=[])
    summary.update(ready_count=11, failed_count=9)
    summary_path.write_text(json.dumps(summary))
    rows[0]["annotation"]["h3_semantics"]["shot1_caption"] = "<Subject 1> stands still."
    rows[0]["record_fingerprint"] = _hash({k: v for k, v in rows[0].items() if k != "record_fingerprint"})
    (base / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="ready annotation validation failed"):
        effective.build_effective_reconcile(base_root=base, override_root=override,
                                            output_root=tmp_path / "invalid", replay_clip_uids=[])
    assert not (tmp_path / "invalid").exists()


def test_cli_reports_actual_unexpected_counts_without_publishing_fake_ready(tmp_path, monkeypatch, capsys):
    from tools import build_h3_two_step_effective_reconcile as cli

    base, _, _, _, _ = _inputs(tmp_path, monkeypatch)
    actual = MimoStemReconcileSummary.model_validate_json((base / "summary.json").read_text())
    monkeypatch.setattr(cli, "build_effective_reconcile", lambda **kwargs: actual)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--base-root", str(base), "--override-root", str(base), "--output-root", str(tmp_path / "out")])
    assert caught.value.code == 2
    assert json.loads(capsys.readouterr().out)["ready_count"] == 11


def _payload_job(template, case):
    values = template.model_dump(mode="json", exclude={"request_fingerprint"})
    values.update(clip_uid=case["clip_uid"], target_duration_seconds=case["target_duration_seconds"],
                  binding_evidence_mode="none")
    segment = values["segments"][0]
    values["segments"] = [{
        **segment, **s, "segment_id": f"segment_{index:04d}",
        "source_start_sample": round(s["start_time"] * 32000),
        "source_end_sample": round(s["end_time"] * 32000),
    } for index, s in enumerate(case.get("segments", []), 1)]
    subjects, images = [], []
    for index, (kind, entity, picture, *attribute) in enumerate(case["subjects"], 1):
        picture_index = int(picture.removeprefix("<Picture ").removesuffix(">"))
        subject = {**values["reference_subjects"][0], "subject_index": index, "subject_label": f"<Subject {index}>",
                   "kind": kind, "entity_id": entity if kind == "entity" else None, "source_picture_labels": [picture]}
        image = {**values["reference_images"][0], "image_index": picture_index, "picture_label": picture,
                 "source_image_index": picture_index, "source_image_id": f"image_{picture_index}",
                 "source_image_label": f"<Image {picture_index}>",
                 "kind": "subject" if kind == "entity" else kind, "entity_id": subject["entity_id"]}
        if attribute:
            details = {"owner_entity_id": entity, "attribute_id": f"a{index}", "attribute_type": attribute[0]}
            subject.update(details)
            image.update(details)
        subjects.append(subject)
        images.append(image)
    images.sort(key=lambda image: image["image_index"])
    values.update(reference_images=images, reference_subjects=subjects)
    values["reference_selection"].update(original_picture_count=len(images), selected_picture_count=len(images),
                                         selected_source_image_indexes=list(range(1, len(images) + 1)),
                                         selected_source_image_ids=[r["source_image_id"] for r in images])
    return _job(values)


_CASES = [case for name in ("binding", "format") for case in json.loads(
    (Path(__file__).parent / f"fixtures/h3_two_step_random20_{name}_cases.json").read_text()
) if case["joint"] is not None]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["clip_uid"])
def test_saved_random20_payloads_replay_preserves_facts_and_identity_restrictions(tmp_path, monkeypatch, case):
    from r2v_data_v2.h3.speaker_ownership import two_step_identity_restricted_groups
    from r2v_data_v2.h3.two_step_effective_reconcile import (
        _validate_ready_record,
        replay_two_step_record,
    )

    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    result = _reconcile(backend, jobs[0], stems)
    job = _payload_job(jobs[0], case)
    values = stem_record_values(tmp_path, job, result.annotation, stems[0])
    provenance = backend.provenance.model_dump(mode="json", exclude={"configuration_fingerprint"})
    provenance["schema_version"] = "r2v.h3.mimo25_backend.82"
    provenance["configuration_fingerprint"] = _hash(provenance)
    visual, joint = json.dumps(case["visual"]), json.dumps(case["joint"])
    values.update(
        backend_provenance=provenance, status="failed", annotation=None,
        failure_code="mimo_structured_output_failed", failure_reason="historical failure",
        raw_responses=[visual, joint], visual_raw_response=visual, speech_av_raw_response=joint,
        audio_finalize_raw_response=None, diagnostics=[d.model_dump(mode="json") for d in result.diagnostics],
        av_model_call_count=1, model_call_count=2,
    )
    values.pop("record_fingerprint")
    original = MimoStemReconcileRecord(**values, record_fingerprint=_hash(values))
    replayed = replay_two_step_record(job, original)
    assert replayed.raw_responses == original.raw_responses
    assert replayed.backend_provenance.schema_version.endswith(".85")
    assert replayed.model_call_count == original.model_call_count == 2
    if case["clip_uid"].startswith("25755"):
        assert replayed.status == "failed"
        assert "direct_unknown_speaker" in {i.code for i in replayed.failure_issues}
        return
    assert replayed.status == "ready", replayed.failure_issues
    _validate_ready_record(job, replayed)
    annotation = replayed.annotation
    assert annotation.av_grounding.model_dump(mode="json") == case["joint"]["speech_av"]["av_grounding"]
    assert [d.model_dump(mode="json") for d in annotation.audio_observation.segment_decisions] == (
        case["joint"]["speech_av"]["audio_observation"]["segment_decisions"]
    )
    if case["clip_uid"].startswith(("4e0506", "009a05")):
        assert two_step_identity_restricted_groups(annotation, replayed.backend_provenance) == {"g1"}
        assert "A voice (S1) says," in annotation.h3_semantics.shot1_caption
        assert "<Subject 1> (S1)" not in annotation.h3_semantics.shot1_caption
        assert "He says, <d>" not in annotation.h3_semantics.shot1_caption
    else:
        assert "(S" not in annotation.h3_semantics.shot1_caption
        assert "<d>" not in annotation.h3_semantics.shot1_caption
