from __future__ import annotations

import json

import pytest

pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse_prepared as prep
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import MimoInventory
from r2v_data_v2.h3.mimo25_stem_shadow import MIMO25_STEM_RECONCILE_STAGE
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file
from tests.h3_audio_reuse_prepared_helpers import stem_record_values
from tests.test_h3_audio_reuse import _fixture, _seal
from tests.test_h3_audio_reuse_materializer import _case
from tests.test_h3_audio_shadow_qa import _fixture as _shadow_fixture
from tests.test_h3_audio_shadow_qa import qa


def _replace(record, **updates):
    return _seal(prep.AudioReusePreparedSource, {**record.model_dump(mode="json"), **updates}, "prepared_fingerprint")


def test_exact_134_overlay_preserves_order_and_failure_replacement(tmp_path):
    args = _fixture(tmp_path)
    root = tmp_path / "base"
    root.mkdir()
    raw = stem_record_values(tmp_path, args["job"], args["annotation"], args["stem_record"], backend_version=60)
    (root / "records.jsonl").write_text(json.dumps(raw))
    template = prep.load_reconcile_sources(root)[0]
    base = [_replace(template, clip_uid=f"clip-{133-i}") for i in range(134)]
    override = [_replace(
        r, status="failed" if i < 9 else "ready",
        source_reconcile_record_fingerprint=f"{i+1:064x}",
        failure_code="original_failure" if i < 9 else None,
        failure_reason="Original issue" if i < 9 else None,
    ) for i, r in enumerate(base[30:55])]
    result = prep.overlay_reconcile_sources(base, override)
    assert [r.clip_uid for r in result] == [r.clip_uid for r in base]
    assert len(result) == 134 and sum(r.status == "ready" for r in result) == 125
    assert sum(r.status == "failed" for r in result) == 9
    assert result[:30] == base[:30] and result[55:] == base[55:]
    assert result[30:55] == override


@pytest.mark.parametrize("change,message", [
    ({"clip_uid": "unknown"}, "unknown"),
    ({"source_job_fingerprint": "0" * 64}, "source job"),
    ({"source_stem_record_fingerprint": "0" * 64}, "source stem"),
])
def test_overlay_rejects_foreign_provenance(tmp_path, change, message):
    _, _, source = _case(tmp_path)
    with pytest.raises(ValueError, match=message):
        prep.overlay_reconcile_sources([source.record], [_replace(source.record, **change)])
    with pytest.raises(ValueError, match="duplicate"):
        prep.overlay_reconcile_sources([source.record, source.record], [])
    with pytest.raises(ValueError, match="duplicate"):
        prep.overlay_reconcile_sources([source.record], [source.record, source.record])


@pytest.mark.parametrize("version", [60, 61, 62, 63])
def test_frozen_source_preserves_real_schema_provenance_and_annotation(tmp_path, version):
    args = _fixture(tmp_path)
    raw = stem_record_values(tmp_path, args["job"], args["annotation"], args["stem_record"], backend_version=version)
    if version < 63:
        provenance = raw["backend_provenance"]
        provenance["prompt_version"] = f"h3_mimo25_speech_assembly_v{46 if version < 62 else 47}"
        provenance["visual_prompt_version"] = "h3_mimo25_visual_only_v4"
        provenance["materializer_version"] = "h3_mimo25_materializer_v26"
        provenance["configuration_fingerprint"] = prep._hash({
            k: v for k, v in provenance.items() if k != "configuration_fingerprint"
        })
        raw["record_fingerprint"] = prep._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (tmp_path / "records.jsonl").write_text(json.dumps(raw))
    before = (tmp_path / "records.jsonl").read_bytes()
    record = prep.load_reconcile_sources(tmp_path)[0]
    assert record.source_backend_provenance.model_dump(mode="json") == raw["backend_provenance"]
    assert record.source_reconcile_record_fingerprint == raw["record_fingerprint"]
    assert record.annotation.model_dump(mode="json") == raw["annotation"]
    assert "raw_responses" not in record.model_dump()
    assert (tmp_path / "records.jsonl").read_bytes() == before
    raw["raw_responses"][0] = "tampered"
    (tmp_path / "records.jsonl").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        prep.load_reconcile_sources(tmp_path)


def test_projection_preserves_cross_donor_and_exact_asr_without_identity(tmp_path):
    _, sample, source = _case(tmp_path, ("g1", "g2"), text="Exact punctuation, preserved!")
    values = sample.model_dump(mode="json")
    values["pair_type"] = "cross_pair"
    values["subject_voices"][0].update(
        voice_source="cross_donor", donor_clip_uid="donor",
        donor_occurrence_id="donor/e2", donor_clip_display_path="01/show/donor",
    )
    values["speech_segments"] = []
    sample = FinalH3SampleV2.model_validate(values)
    projected = prep.project_prepared_samples([source.job], [sample])[0]
    assert projected.model_dump(exclude={"speech_segments"}) == sample.model_dump(exclude={"speech_segments"})
    assert projected.subject_voices == sample.subject_voices
    for speech, segment in zip(projected.speech_segments, source.job.segments, strict=True):
        assert speech.segment_id == segment.segment_id
        assert speech.speaker_cluster_id == segment.source_speaker_cluster_id
        assert (speech.source_start_sample, speech.source_end_sample, speech.source_sample_rate_hz) == (
            segment.source_start_sample, segment.source_end_sample, 32000,
        )
        assert (speech.start_time, speech.end_time, speech.text, speech.language) == (
            segment.start_time, segment.end_time, segment.asr_text, segment.asr_language,
        )
        assert speech.entity_id is None and speech.entity_occurrence_id is None
    job_values = source.job.model_dump(mode="json")
    job_values["segments"][1].update(asr_status="empty", asr_text=None, asr_language=None)
    job = _seal(type(source.job), job_values, "request_fingerprint")
    assert len(prep.project_prepared_samples([job], [sample])[0].speech_segments) == 1


def _prepare_fixture(tmp_path, monkeypatch):
    args, shadow = _shadow_fixture(tmp_path, monkeypatch, variants=True)
    base_inventory = qa.build_mimo25_inventory()
    calls = []

    def build(**kwargs):
        calls.append(kwargs["case_manifest"].clip_uids)
        return base_inventory

    monkeypatch.setattr(prep, "build_mimo25_inventory", build)
    production = args["audio_production_root"]
    # The fixture mocks only the production inventory reader; stem job rebuilding is real.
    for relative in ("diarization/inventory.json", "diarization/raw_segments.jsonl",
                     "diarization/bound_segments.jsonl", "asr/segments.jsonl",
                     "binding_audit_v1/segments.jsonl"):
        path = production / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (args["visual_production_root"] / "samples.jsonl").write_text("{}\n")
    base_root = shadow / MIMO25_STEM_RECONCILE_STAGE
    rows = [json.loads(line) for line in (base_root / "records.jsonl").read_text().splitlines()]
    for row in rows:
        provenance = row["backend_provenance"]
        provenance["schema_version"] = "r2v.h3.mimo25_backend.60"
        provenance["configuration_fingerprint"] = prep._hash({k: v for k, v in provenance.items() if k != "configuration_fingerprint"})
        row["record_fingerprint"] = prep._hash({k: v for k, v in row.items() if k != "record_fingerprint"})
    (base_root / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    override = shadow / "override61"
    override.mkdir()
    replacement = json.loads(json.dumps(rows[0]))
    provenance = replacement["backend_provenance"]
    provenance["schema_version"] = "r2v.h3.mimo25_backend.61"
    provenance["configuration_fingerprint"] = prep._hash({k: v for k, v in provenance.items() if k != "configuration_fingerprint"})
    replacement["record_fingerprint"] = prep._hash({k: v for k, v in replacement.items() if k != "record_fingerprint"})
    (override / "records.jsonl").write_text(json.dumps(replacement) + "\n")
    return {
        "audio_production_root": production, "visual_production_root": args["visual_production_root"],
        "visual_runs_root": args["visual_runs_root"], "stem_shadow_root": shadow,
        "base_reconcile_root": base_root, "override_reconcile_root": override, "source_h3_root": production / "h3",
        "prepared_root": shadow / "prepared",
    }, base_inventory, calls


def test_prepare_cli_real_stem_job_reconstruction_and_atomic_publication(tmp_path, monkeypatch):
    args, base_inventory, calls = _prepare_fixture(tmp_path, monkeypatch)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    from tools.prepare_h3_audio_reuse_sources import main

    result = main([part for k, v in args.items() for part in ("--" + k.replace("_", "-"), str(v))])
    output = args["prepared_root"]
    inventory = MimoInventory.model_validate_json((output / "inventory.json").read_text())
    records = [prep.AudioReusePreparedSource.model_validate_json(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert result["model_call_count"] == 0 and result["production_artifacts_modified"] is False
    assert result["clip_uids"] == calls[0] == ["clip-z", "clip-a", "clip-m"]
    assert [r.source_backend_provenance.schema_version for r in records] == [
        "r2v.h3.mimo25_backend.61", "r2v.h3.mimo25_backend.60", "r2v.h3.mimo25_backend.60",
    ]
    assert inventory.source_h3_samples_sha256 == sha256_file(output / "h3/samples.jsonl")
    provenance, _, _ = prep.validate_stem_diarization_lineage(args["stem_shadow_root"] / "diarization")
    rebuilt = prep.build_stem_reconcile_jobs(
        base_inventory=prep.usable_stem_reconcile_inventory(base_inventory, provenance),
        stem_diarization_root=args["stem_shadow_root"] / "diarization",
        stem_asr_root=args["stem_shadow_root"] / "asr", route=provenance.route,
    )
    assert inventory.jobs == rebuilt
    assert [j.request_fingerprint for j in inventory.jobs] == [r.source_job_fingerprint for r in records]
    assert before == {p: sha256_file(p) for p in before}
    with pytest.raises(FileExistsError):
        prep.prepare_audio_reuse_sources(**args)


def test_job_reconstruction_mismatch_fails_without_publication(tmp_path, monkeypatch):
    args, _, _ = _prepare_fixture(tmp_path, monkeypatch)
    original = prep.build_stem_reconcile_jobs

    def changed(**kwargs):
        jobs = original(**kwargs)
        values = jobs[0].model_dump(mode="json")
        values["segments"][0]["asr_text"] = "different frozen transcript"
        jobs[0] = _seal(type(jobs[0]), values, "request_fingerprint")
        return jobs

    monkeypatch.setattr(prep, "build_stem_reconcile_jobs", changed)
    with pytest.raises(ValueError, match="source job"):
        prep.prepare_audio_reuse_sources(**args)
    assert not args["prepared_root"].exists()


def test_legacy_record_rendering_is_identical(tmp_path):
    from r2v_data_v2.h3.mimo25_av_reconcile import MimoRecord
    from r2v_data_v2.h3.mimo25_h3_materializer import (
        MIMO25_MATERIALIZER_VERSION,
        _materialize_sample,
    )
    from tests.test_h3_mimo25_av_shadow import _record_fixture

    _, sample, source = _case(tmp_path)
    legacy = _record_fixture(tmp_path, source.record.annotation, job=source.job)
    assert MimoRecord.model_validate_json(legacy.model_dump_json()) == legacy
    assert MIMO25_MATERIALIZER_VERSION == "h3_mimo25_materializer_v27"
    assert _materialize_sample(sample, source.job, legacy, conditioning_variant="visual_only") == (
        _materialize_sample(sample, source.job, source.record, conditioning_variant="visual_only")
    )
    with pytest.raises(ValueError):
        MimoRecord.model_validate_json(source.record.model_dump_json())


def test_prepare_late_input_mutation_keeps_destination_absent(tmp_path, monkeypatch):
    args, _, _ = _prepare_fixture(tmp_path, monkeypatch)
    original = prep.project_prepared_samples

    def change_input(*a, **kw):
        result = original(*a, **kw)
        (args["base_reconcile_root"] / "records.jsonl").write_text("changed")
        return result

    monkeypatch.setattr(prep, "project_prepared_samples", change_input)
    with pytest.raises(ValueError, match="input changed"):
        prep.prepare_audio_reuse_sources(**args)
    assert not args["prepared_root"].exists()
    assert not list(args["prepared_root"].parent.glob(".reuse-prepare-*"))


def test_prepare_cannot_publish_into_source_or_production_stage(tmp_path, monkeypatch):
    args, _, _ = _prepare_fixture(tmp_path, monkeypatch)
    for output in (args["audio_production_root"] / "audio/new", args["base_reconcile_root"] / "new",
                   args["stem_shadow_root"] / "diarization/new", args["source_h3_root"] / "new"):
        with pytest.raises(ValueError, match="overlaps"):
            prep.prepare_audio_reuse_sources(**{**args, "prepared_root": output})
