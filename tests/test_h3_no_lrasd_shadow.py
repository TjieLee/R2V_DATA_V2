import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.h3 import mimo25_av_reconcile as reconcile
from r2v_data_v2.h3.mimo25_stem_shadow import build_stem_reconcile_jobs
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    run_stem_diarization_shadow,
    run_stem_qwen3_asr_shadow,
    validate_stem_asr_lineage,
)
from tests import test_h3_resolved_audio_stems as resolved_tests
from tests.test_h3_resolved_audio_stems import _resolve
from tests.test_h3_sam_audio_stem_shadow import (
    _Diarization,
    _mimo_inventory_for_clip_order,
    _Qwen,
)

setup = resolved_tests.setup
ffmpeg = resolved_tests.ffmpeg


def _none_inventory(base):
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
    values.update(schema_version="r2v.h3.mimo25_inventory.5", binding_evidence_mode="none")
    for name in ("source_diarization_bound_segments_sha256", "source_binding_audit_segments_sha256"):
        values[name] = None
    values["jobs"] = [reconcile._job({
        **job.model_dump(mode="json", exclude={"request_fingerprint"}),
        "binding_evidence_mode": "none", "segments": [],
    }).model_dump(mode="json") for job in base.jobs]
    return reconcile._inventory(values)


def _deny_binding_reads(monkeypatch):
    original = Path.open

    def guarded(path, mode="r", *args, **kwargs):
        if "r" in mode:
            assert path.name not in {"bound_segments.jsonl", "cluster_bindings.jsonl"}
            assert "binding_audit_v1" not in path.parts
            assert "lrasd_native" not in str(path)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def test_resolved_raw_asr_without_production_binding(setup, tmp_path, ffmpeg, monkeypatch):
    root, (inventory, _, _) = _resolve(setup, tmp_path, ffmpeg)
    production = Path(setup.audio_production_root)
    assert not (production / "diarization").exists()
    visual = tmp_path / "visual"
    visual.mkdir()
    (visual / "samples.jsonl").write_text("{}\n")
    diari = root.parent / "diarization_no_lrasd_v1"
    asr = root.parent / "asr_no_lrasd_v1"
    backend = _Diarization()
    provenance = run_stem_diarization_shadow(
        stem_root=root, production_diarization_root=production / "diarization",
        output_root=diari, backend=backend, route="resolved", allow_unverified=True,
        binding_evidence_mode="none", visual_production_root=visual,
    )
    assert provenance.binding_evidence_mode == "none"
    assert provenance.bound_segments_sha256 is None
    assert provenance.cluster_bindings_sha256 is None
    targets = json.loads((diari / "inventory.json").read_text())["targets"]
    assert all(t["target_audio_binding_path"] is None for t in targets)
    assert all(t["target_audio_binding_sha256"] is None for t in targets)
    assert all(t["visual_references"] == [] for t in targets)
    # Neutral compatibility outputs are never dependencies of this mode.
    (diari / "bound_segments.jsonl").unlink()
    (diari / "cluster_bindings.jsonl").unlink()
    _deny_binding_reads(monkeypatch)
    summary, asr_provenance = run_stem_qwen3_asr_shadow(
        stem_diarization_root=diari, source_visual_production_root=str(visual),
        output_root=asr, backend=_Qwen(), route="resolved", allow_unverified=True,
        segment_audio_loader=lambda p, s, e: (np.zeros(round((e-s)*16000), dtype=np.float32), 16000),
    )
    assert summary.transcribed_count == 3
    assert asr_provenance.binding_evidence_mode == "none"
    validate_stem_asr_lineage(asr, expected_diarization_root=diari)
    base = _none_inventory(_mimo_inventory_for_clip_order(tmp_path / "base", inventory.clip_uids))
    from tools import run_h3_mimo25_stem_reconcile_shadow as cli
    from tools import run_h3_stem_diarization_shadow as diari_cli
    from tools import run_h3_stem_qwen3_asr_shadow as asr_cli
    common = ["--audio-production-root", str(production), "--shadow-run-id", setup.shadow_run_id,
              "--case-manifest", setup.source_case_manifest_path, "--allow-unverified", "--dry-run"]
    dry = diari_cli.main([*common, "--binding-evidence-mode", "none", "--visual-production-root", str(visual)])
    assert dry["output_root"] == str(diari)
    assert asr_cli.main([*common, "--diarization-root", str(diari), "--visual-production-root", str(visual)])["output_root"] == str(asr)
    monkeypatch.setattr(cli, "build_mimo25_reference_inventory", lambda **kw: base)
    result = cli.main([
        *common, "--binding-evidence-mode", "none", "--stem-diarization-root", str(diari),
        "--stem-asr-root", str(asr), "--visual-production-root", str(visual),
        "--visual-runs-root", str(visual),
    ])
    assert result["output_root"] == str(root.parent / "mimo_reconcile_no_lrasd_v1")
    assert result["clip_uids"] == inventory.clip_uids
    assert not result["model_called"]
    jobs = build_stem_reconcile_jobs(
        base_inventory=base, stem_diarization_root=diari, stem_asr_root=asr,
        route="resolved", binding_evidence_mode="none",
    )
    assert [j.clip_uid for j in jobs] == ["c", "a", "b"]
    for job in jobs:
        for segment in job.segments:
            assert segment.current_entity_id is None
            assert segment.entity_occurrence_id is None
            assert segment.identity_scope == "unresolved"
            assert segment.direct_anchor_seconds == 0
            assert segment.cluster_binding_status == "unbound"
            assert segment.direct_support_seconds_by_entity == {}
            assert segment.competing_visible_speaker_evidence == []
            assert segment.asr_text
    with pytest.raises(ValueError, match="provenance differs"):
        build_stem_reconcile_jobs(base_inventory=base, stem_diarization_root=diari, stem_asr_root=asr)
    from r2v_data_v2.h3 import mimo25_stem_shadow as stem_module
    read = stem_module._read_jsonl
    for field, value in (("speaker_cluster_id", "speaker_99"), ("source_audio_path", "/wrong.wav"),
                         ("segment_id", "segment_9999"), ("source_start_sample", 1), ("start_time", 0.2)):
        def changed(path, field=field, value=value):
            rows = read(path)
            if path == asr / "segments.jsonl":
                rows[0][field] = value
            return rows
        with monkeypatch.context() as context:
            context.setattr(stem_module, "_read_jsonl", changed)
            with pytest.raises(ValueError):
                build_stem_reconcile_jobs(
                    base_inventory=base, stem_diarization_root=diari, stem_asr_root=asr,
                    binding_evidence_mode="none", route="resolved",
                )


def test_reference_builder_never_reads_speaker_evidence(tmp_path, monkeypatch):
    from tests.test_h3_audio_shadow_qa import _fixture
    args, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    (args["visual_production_root"] / "samples.jsonl").write_text("{}\n")
    samples = [json.loads(line) for line in (args["audio_production_root"] / "h3/samples.jsonl").read_text().splitlines()]
    canonical = [s for s in samples if s["pair_type"] == "canonical"]
    monkeypatch.setattr(reconcile, "load_visual_production_inventory", lambda **kw: SimpleNamespace(
        canonical_clips=[SimpleNamespace(identity=SimpleNamespace(clip_uid=s["clip_uid"]),
                                        sample=SimpleNamespace(target_video=s["target_video"])) for s in canonical],
    ))
    _deny_binding_reads(monkeypatch)
    inventory = reconcile.build_mimo25_reference_inventory(
        visual_production_root=args["visual_production_root"],
        visual_runs_root=args["visual_runs_root"], audio_production_root=args["audio_production_root"],
    )
    assert inventory.binding_evidence_mode == "none"
    assert all(j.segments == [] and j.reference_subjects for j in inventory.jobs)
    assert inventory.source_binding_audit_segments_sha256 is None
    assert reconcile.MimoInventory.model_validate_json(inventory.model_dump_json()) == inventory
    from r2v_data_v2.h3.audio_reuse_prepared import prepare_audio_reuse_sources
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
    from r2v_data_v2.h3.sam_audio_stem_shadow import load_stem_shadow
    from tests.test_h3_audio_shadow_qa import _Reconcile
    diari, asr = shadow / "diarization_no_lrasd_v1", shadow / "asr_no_lrasd_v1"
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
              and p.name not in {"bound_segments.jsonl", "cluster_bindings.jsonl"}}
    run_stem_diarization_shadow(
        stem_root=shadow / "separation", production_diarization_root=tmp_path / "nonexistent",
        output_root=diari, backend=_Diarization(), route="music_first", allow_unverified=True,
        binding_evidence_mode="none", visual_production_root=args["visual_production_root"],
    )
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=diari, source_visual_production_root=str(args["visual_production_root"]),
        output_root=asr, backend=_Qwen(), route="music_first", allow_unverified=True,
        segment_audio_loader=lambda p, s, e: (np.zeros(round((e-s)*16000), dtype=np.float32), 16000),
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=inventory, stem_diarization_root=diari, stem_asr_root=asr,
        binding_evidence_mode="none", route="music_first",
    )
    output = shadow / "mimo_reconcile_no_lrasd_v1"
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_records=load_stem_shadow(shadow / "separation")[1],
        backend=_Reconcile(tmp_path), output_root=output, route="music_first",
        allow_unverified=True, binding_evidence_mode="none",
    )
    contract = json.loads((output / "source_contract.json").read_text())
    assert contract["binding_evidence_mode"] == "none"
    assert contract["jobs"] == [j.model_dump(mode="json") for j in jobs]
    prepared = shadow / "prepared_no_lrasd_v1"
    summary = prepare_audio_reuse_sources(
        audio_production_root=args["audio_production_root"],
        visual_production_root=args["visual_production_root"], visual_runs_root=args["visual_runs_root"],
        stem_shadow_root=shadow, base_reconcile_root=output, override_reconcile_root=None,
        source_h3_root=args["audio_production_root"] / "h3", prepared_root=prepared,
        binding_evidence_mode="none", stem_diarization_root=diari, stem_asr_root=asr,
    )
    assert not summary.production_artifacts_modified
    published = reconcile.MimoInventory.model_validate_json((prepared / "inventory.json").read_text())
    assert published.binding_evidence_mode == "none"
    assert all(p.read_bytes() == contents for p, contents in before.items())


def test_legacy_serialization_preserves_hash_contract(tmp_path):
    base = _mimo_inventory_for_clip_order(tmp_path, ["a", "b"])
    before = base.model_dump(mode="json")
    assert "binding_evidence_mode" not in before
    assert all("binding_evidence_mode" not in j for j in before["jobs"])
    loaded = reconcile.MimoInventory.model_validate(before)
    assert loaded.binding_evidence_mode == "legacy_lr_asd"
    assert loaded.model_dump(mode="json") == before
    explicit = {**before, "binding_evidence_mode": "legacy_lr_asd"}
    assert reconcile.MimoInventory.model_validate(explicit).model_dump(mode="json") == before
    new = _none_inventory(base)
    assert new.inventory_fingerprint != base.inventory_fingerprint
    assert all(a.request_fingerprint != b.request_fingerprint for a, b in zip(base.jobs, new.jobs, strict=True))


@pytest.mark.parametrize("code", ["lr_asd_support", "lr_asd_conflict"])
def test_unsupplied_lr_asd_evidence_remains_hard(code, tmp_path):
    from r2v_data_v2.h3.mimo25_backend import validate_annotation
    from tests.test_h3_mimo25_av_shadow import _annotation, _job_fixture
    annotation = _annotation()
    annotation.av_grounding.segment_groundings[0].evidence_codes = [code]
    job = _job_fixture(tmp_path)
    kwargs = {
        "segment_ids": [s.segment_id for s in job.segments],
        "transcribed_segment_ids": [s.segment_id for s in job.segments],
        "segment_intervals": {s.segment_id: (s.start_time, s.end_time) for s in job.segments},
        "authoritative_transcripts": [s.asr_text for s in job.segments],
        "reference_subjects": job.reference_subjects,
        "target_duration_seconds": job.target_duration_seconds,
        "allowed_entity_ids": {"e1"},
        "allowed_reference_labels": {"<Subject 1>", "<Picture 1>"},
    }
    assert "unsupplied_lr_asd_evidence" in {i.code for i in validate_annotation(annotation, **kwargs, binding_evidence_mode="none")}
    assert "unsupplied_lr_asd_evidence" not in {i.code for i in validate_annotation(annotation, **kwargs)}


def test_none_mode_preserves_four_stages_and_legacy_request(tmp_path):
    from tests.h3_mimo_two_turn_helpers import split_annotation
    from tests.test_h3_mimo25_av_shadow import _annotation, _backend, _job_fixture
    from tests.test_h3_mimo25_two_turn import _run
    job = _job_fixture(tmp_path)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["binding_evidence_mode"] = "none"
    for s in values["segments"]:
        s.update(current_entity_id=None, entity_occurrence_id=None, identity_scope="unresolved",
                 direct_anchor_seconds=0.0, cluster_binding_status="unbound", overlapping_visible_entities=[],
                 direct_support_seconds_by_entity={}, competing_visible_speaker_evidence=[])
    neutral = reconcile._job(values)
    raw = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(r, 8) for r in raw])
    result = _run(backend, neutral)
    assert result.annotation is not None
    assert result.model_call_count == 4
    assert len(calls.requests) == 4
    assert "binding_evidence_mode=none" in json.dumps(calls.requests[1])
    assert "No LR-ASD" in json.dumps(calls.requests[1])
    old_backend, old_calls = _backend(tmp_path, [(r, 8) for r in raw])
    assert _run(old_backend, job).model_call_count == 4
    assert "binding_evidence_mode" not in json.dumps(old_calls.requests)
    # The same model-only response and final semantics are used; only input priors differ.
    assert calls.requests[0]["response_format"] == old_calls.requests[0]["response_format"]


@pytest.mark.parametrize("code", ["lr_asd_support", "lr_asd_conflict"])
def test_none_rejects_model_claims_even_after_normalization(tmp_path, code):
    from r2v_data_v2.h3.mimo25_backend import MimoBackendFailure
    from tests.h3_mimo_two_turn_helpers import split_annotation
    from tests.test_h3_mimo25_av_shadow import _annotation, _backend, _job_fixture
    from tests.test_h3_mimo25_two_turn import _run
    job = _job_fixture(tmp_path)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["binding_evidence_mode"] = "none"
    for s in values["segments"]:
        s.update(current_entity_id=None, entity_occurrence_id=None, identity_scope="unresolved",
                 direct_anchor_seconds=0.0, cluster_binding_status="unbound", overlapping_visible_entities=[],
                 direct_support_seconds_by_entity={}, competing_visible_speaker_evidence=[])
    neutral = reconcile._job(values)
    annotation = _annotation()
    annotation.av_grounding.segment_groundings[0].evidence_codes.append(code)
    raw = split_annotation(annotation.model_dump_json())
    backend, _ = _backend(tmp_path, [(r, 8) for r in raw])
    with pytest.raises(MimoBackendFailure) as error:
        _run(backend, neutral)
    assert "unsupplied_lr_asd_evidence" in {i.code for i in error.value.issues}
