from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse_prepared as prepared
from r2v_data_v2.h3 import frame_conditioning_projection as frame
from r2v_data_v2.h3.audio_reuse import AudioReuseManifest
from r2v_data_v2.h3.audio_reuse_materializer import AudioReuseProduct
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_single_backend import SingleCallOpenAIMimo25Backend
from r2v_data_v2.h3.resolved_audio_stems import (
    ResolvedStem,
    ResolvedStemInventory,
    ResolvedStemRecord,
    ResolvedStemSummary,
)
from r2v_data_v2.h3.training_task_review import (
    TrainingTaskReviewStore,
    build_review_cases,
    export_reviewed_training_manifests,
)
from tests.h3_audio_reuse_prepared_helpers import stem_record_values
from tests.test_h3_audio_reuse import _seal
from tests.test_h3_audio_reuse_materializer import _case
from tests.test_h3_training_task_review import _annotation


def _rows(path, model=None):
    return [model.model_validate_json(line) if model else json.loads(line)
            for line in path.read_text().splitlines() if line.strip()]


def _fixture(tmp_path, monkeypatch, *, offscreen=False, music="Quiet piano.", no_speech=False):
    uid = "a" * 24
    args, sample, _ = _case(tmp_path / "case", clip=uid, offscreen=offscreen, music=music,
                            target_subtype="PCM_24")
    legacy_in_pair = sample
    sample = sample.model_copy(update={
        "sample_id": f"{uid}/canonical", "pair_id": f"canonical/{uid}",
        "pair_type": "canonical", "subject_voices": [], "speech_segments": [],
    })
    values = args["job"].model_dump(mode="json")
    values.update(binding_evidence_mode="none", source_h3_sample_ids=[sample.sample_id])
    for segment in values["segments"]:
        segment.update(current_entity_id=None, entity_occurrence_id=None, identity_scope="unresolved",
                       direct_anchor_seconds=0.0, cluster_binding_status="unbound", overlapping_visible_entities=[],
                       direct_support_seconds_by_entity={}, competing_visible_speaker_evidence=[])
        if no_speech:
            segment.update(asr_status="empty", asr_text=None, asr_language=None)
    job = _seal(MimoClipJob, values, "request_fingerprint")
    annotation = args["annotation"]
    if no_speech:
        values = annotation.model_dump(mode="json")
        values["h3_semantics"]["shot1_caption"] = "<Subject 1> stands in a room."
        annotation = type(annotation).model_validate(values)
    audio_root = args["audio_production_root"]
    run = audio_root / "sam_audio_stem_shadow_v1/runs/frozen"
    reconcile = run / "reconcile"
    reconcile.mkdir(parents=True)
    stems = {}
    for source in args["stem_record"].stems:
        kind = source.stem_type
        stems[kind] = ResolvedStem(
            kind=kind, source_backend="auk" if kind == "speech" else "sam_audio",
            canonical_stem_path=source.canonical_stem_path, canonical_stem_sha256=source.canonical_stem_sha256,
            canonical_frame_count=source.canonical_frame_count, source_audio_path=source.source_audio_path,
            source_audio_sha256=source.source_audio_sha256, source_end_sample=source.source_end_sample,
        ).model_dump(mode="json")
    stem = _seal(ResolvedStemRecord, {
        "schema_version": "r2v.h3.resolved_stem_record.1", "clip_uid": uid,
        "inventory_fingerprint": "a" * 64, "source_sam_record_fingerprint": "b" * 64,
        "source_auk_record_fingerprint": "c" * 64, "status": "ready", "verification_state": "unverified",
        "failure_reason": None, **stems,
    }, "record_fingerprint")
    manifest = audio_root / "audio/canonical_clips.jsonl"
    manifest.parent.mkdir()
    manifest.write_text('{"fixture":"canonical metadata"}\n')
    inventory = _seal(ResolvedStemInventory, {
        "schema_version": "r2v.h3.resolved_stem_inventory.1", "speech_source": "auk",
        "music_source": "sam_audio", "sfx_source": "sam_audio", "source_sam_root": str(run / "separation"),
        "source_auk_root": str(run / "auk_speech_v1"), "source_sam_inventory_fingerprint": "a" * 64,
        "source_auk_inventory_fingerprint": "b" * 64, "source_sam_records_sha256": "c" * 64,
        "source_auk_records_sha256": "d" * 64, "source_canonical_audio_manifest_path": str(manifest),
        "source_canonical_audio_manifest_sha256": frame.sha256_file(manifest), "clip_uids": [uid],
        "jobs": [{"clip_uid": uid, "clip_display_path": sample.clip_display_path,
                  "target_video_path": job.target_video_path, "target_video_sha256": job.target_video_sha256,
                  "source_audio_path": job.target_full_audio_path, "source_audio_sha256": job.target_full_audio_sha256,
                  "source_sample_rate_hz": 32000, "source_channels": 2,
                  "source_frame_count": 64000, "source_duration_seconds": 2.0}],
    }, "inventory_fingerprint")
    root = run / "resolved_stems_v1"
    root.mkdir()
    (root / "inventory.json").write_text(inventory.model_dump_json())
    (root / "records.jsonl").write_text(stem.model_dump_json() + "\n")
    (root / "summary.json").write_text(ResolvedStemSummary(
        inventory_fingerprint=inventory.inventory_fingerprint, clip_uids=[uid], ready_count=1, failed_count=0,
    ).model_dump_json())
    raw = stem_record_values(tmp_path, job, annotation, stem)
    raw.update(backend_provenance=SingleCallOpenAIMimo25Backend(MimoBackendConfig(
        api_key="fixture", transport="sglang",
        media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
    ), single_synthetic_icl_variant="action_v2", stem_records_by_clip={uid: stem}).provenance.model_dump(mode="json"),
        raw_responses=["compact"], visual_raw_response=None, speech_av_raw_response="compact",
        audio_finalize_raw_response=None, visual_model_call_count=0, av_model_call_count=1, model_call_count=1,
        diagnostics=[{**raw["diagnostics"][0], "input_modality": "target_av_with_auxiliary_raw_audio"}])
    raw["record_fingerprint"] = prepared._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (reconcile / "records.jsonl").write_text(json.dumps(raw) + "\n")
    (reconcile / "source_contract.json").write_text(json.dumps({
        "schema_version": "r2v.h3.mimo25_stem_source_contract.1", "binding_evidence_mode": "none",
        "clip_uids": [uid], "jobs": [job.model_dump(mode="json")],
    }))
    (reconcile / "summary.json").write_text(json.dumps({
        "schema_version": "r2v.h3.mimo25_stem_reconcile_summary.17", "route": "resolved",
        "clip_count": 1, "processed_clip_count": 1, "skipped_clip_count": 0, "clip_uids": [uid],
        "processed_clip_uids": [uid], "skipped_clips": [], "diarization_failed_clips": [],
        "ready_count": 1, "failed_count": 0, "av_model_call_count": 1, "visual_model_call_count": 0,
        "audio_model_call_count": 0, "text_model_call_count": 0, "model_call_count": 1,
    }))
    h3 = audio_root / "h3"
    h3.mkdir()
    (h3 / "samples.jsonl").write_text(sample.model_dump_json() + "\n" + legacy_in_pair.model_dump_json() + "\n")

    def forbidden(*a, **kw):
        raise AssertionError("frozen preparation must not reconstruct Visual or speech")

    for name in ("build_mimo25_inventory", "build_mimo25_reference_inventory", "load_mimo25_reference_sources",
                 "build_stem_reconcile_jobs", "validate_stem_diarization_lineage"):
        monkeypatch.setattr(prepared, name, forbidden)
    monkeypatch.setattr(frame, "validate_stem_diarization_lineage", forbidden)

    def extract(job, stage, published, *, ffmpeg):
        from PIL import Image

        stage.mkdir(parents=True)
        for name, color in (("first", "red"), ("last", "green")):
            Image.new("RGB", (8, 8), color).save(stage / f"{name}.png")
        metadata = frame.FrameMetadata(
            clip_uid=job.clip_uid, source_video_path=job.target_video_path, source_video_sha256=job.target_video_sha256,
            source_duration_seconds=job.target_duration_seconds, first_frame_path=str(published / "first.png"),
            first_frame_sha256=frame.sha256_file(stage / "first.png"), last_frame_path=str(published / "last.png"),
            last_frame_sha256=frame.sha256_file(stage / "last.png"),
        )
        (stage / "metadata.json").write_text(metadata.model_dump_json() + "\n")
        return metadata

    monkeypatch.setattr(frame, "extract_frames", extract)
    return reconcile, job, sample, stem


@pytest.mark.parametrize("offscreen,music,no_speech,task_count", [
    (False, "Quiet piano.", False, 12), (True, "N/A", False, 12), (False, "Quiet piano.", True, 8),
])
def test_frozen_bundle_real_pcm_materialization_frames_and_pass_only(tmp_path, monkeypatch, offscreen, music, no_speech, task_count):
    reconcile, job, canonical, stem = _fixture(tmp_path, monkeypatch, offscreen=offscreen, music=music, no_speech=no_speech)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    from tools.materialize_h3_ra2va_training_bundle import main

    output = tmp_path / "bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    assert result["model_call_count"] == 0 and result["production_artifacts_modified"] is False
    assert sum(result["task_counts"].values()) == task_count
    assert before == {p: p.read_bytes() for p in before}
    assert not (reconcile.parent / "audio_reuse_assets_v1").exists()
    inventory = prepared.load_prepared_inventory(output / "audio_reuse_prepared_v1/inventory.json")
    assert inventory.jobs == [job]
    assert inventory.source_stem_root == str(reconcile.parent / "resolved_stems_v1")
    sample = _rows(output / "audio_reuse_prepared_v1/h3/samples.jsonl", FinalH3SampleV2)[0]
    assert sample.subject_voices == [] and sample.pair_type == "canonical"
    assert sample.visual_references == canonical.visual_references
    if no_speech:
        assert sample.speech_segments == []
    else:
        assert sample.speech_segments[0].text == job.segments[0].asr_text
        assert sample.speech_segments[0].entity_id is None
    manifest = AudioReuseManifest.model_validate_json((output / f"audio_reuse_assets_v1/{job.clip_uid}/manifest.json").read_text())
    assert manifest.source_stem_record_fingerprint == stem.record_fingerprint
    assert bool(manifest.speakers) is not no_speech
    if manifest.speakers:
        asset = manifest.speakers[0]
        assert sf.info(asset.output_path).format == "FLAC" and sf.info(asset.output_path).subtype == "PCM_16"
        pcm, _ = sf.read(asset.output_path, dtype="int16", always_2d=True)
        source, _ = sf.read(stem.speech.canonical_stem_path, dtype="int16", always_2d=True)
        expected = np.zeros((64000, 2), dtype=np.int16)
        for r in asset.source_sample_ranges:
            expected[r.target_start_sample:r.target_end_sample] = source[r.target_start_sample:r.target_end_sample]
        np.testing.assert_array_equal(pcm, expected)
        if offscreen:
            assert asset.entity_id is None and asset.subject_index is None
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert all(p.status == "ready" for p in products)
    full = next(p for p in products if p.conditioning_variant == "full_audio_reuse")
    assert full.audio_references[0].contract.path == job.target_full_audio_path
    assert sf.info(job.target_full_audio_path).subtype == "PCM_24"
    speech = next((p for p in products if p.conditioning_variant == "target_speech_reuse"), None)
    if speech:
        assert [r.contract.kind for r in speech.audio_references] == (
            ["speaker_speech_reuse", "music_reuse"] if music != "N/A" else ["speaker_speech_reuse"]
        )
    for p in products:
        if no_speech:
            assert "<d>" not in p.rendered_h3_prompt
        else:
            assert p.rendered_h3_prompt.count(f"<d>[English] {job.segments[0].asr_text}</d>") == 1
    projected, _ = frame.load_frame_conditioned_products(output / frame.FRAME_STAGE, output)
    assert len(projected) == 3 * len(products)
    cases = build_review_cases(output)
    assert len(cases) == task_count
    assert all(set(c.row) == {"video", "images", "audios", "caption"} for c in cases)
    assert all(Path(p).is_file() for c in cases for p in [c.row["video"], *c.row["images"], *c.row["audios"]])
    store = TrainingTaskReviewStore(tmp_path / "review", cases)
    assert store.publish_derived().reviewed_count == 0
    store.save(_annotation(cases[0], "PASS"))
    store.save(_annotation(cases[1], "ISSUE", tags=["speaker_binding_issue"]))
    export = export_reviewed_training_manifests(cases=cases, store=store, output_root=tmp_path / "accepted")
    assert sum(len(_rows(export / f"{task}.jsonl")) for task in result["task_counts"]) == 1
    assert set(_rows(export / "videos.jsonl")[0]) == {"video", "tasks"}
    with pytest.raises(FileExistsError):
        main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])


def test_frozen_preparation_missing_canonical_metadata_reports_source(tmp_path, monkeypatch):
    reconcile, _, _, _ = _fixture(tmp_path, monkeypatch)
    (reconcile.parents[3] / "h3/samples.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="canonical H3.*source-h3-root"):
        prepared.prepare_frozen_audio_reuse_sources(reconcile_root=reconcile, prepared_root=tmp_path / "prepared")


def test_clip_subset_and_nonready_selection(tmp_path, monkeypatch):
    reconcile, job, _, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unknown"):
        prepared.prepare_frozen_audio_reuse_sources(
            reconcile_root=reconcile, prepared_root=tmp_path / "unknown", clip_uids=["unknown"],
        )
    summary = prepared.prepare_frozen_audio_reuse_sources(
        reconcile_root=reconcile, prepared_root=tmp_path / "selected", clip_uids=[job.clip_uid],
    )
    assert summary.clip_uids == [job.clip_uid]
    raw = _rows(reconcile / "records.jsonl")[0]
    raw.update(status="failed", annotation=None, failure_code="semantic_failure", failure_reason="frozen failure")
    raw["record_fingerprint"] = prepared._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (reconcile / "records.jsonl").write_text(json.dumps(raw) + "\n")
    summary_raw = json.loads((reconcile / "summary.json").read_text())
    summary_raw.update(ready_count=0, failed_count=1)
    (reconcile / "summary.json").write_text(json.dumps(summary_raw))
    with pytest.raises(ValueError, match="no ready"):
        prepared.prepare_frozen_audio_reuse_sources(reconcile_root=reconcile, prepared_root=tmp_path / "failed")


def test_frozen_projection_preserves_recorded_face_promotion_without_sampling(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import mimo25_av_reconcile as reconcile_module
    from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract
    from tests.test_h3_mimo25_av_shadow import (
        _augmentation_draws,
        _visual_reference_inventory,
    )

    reconcile, job, canonical, _ = _fixture(tmp_path, monkeypatch)
    refs = _visual_reference_inventory(tmp_path, ["subject", "face", "hair"], same_entity=True)
    canonical = canonical.model_copy(update={"visual_references": refs})
    _augmentation_draws(monkeypatch, [.8, .9])
    selection, images = reconcile_module.select_mimo_reference_projection(job.clip_uid, refs)
    projected = reconcile_module.project_mimo_h3_sample_references(
        canonical, reference_images=images, reference_selection=selection,
    )
    values = job.model_dump(mode="json")
    values.update(reference_selection=selection.model_dump(mode="json"),
                  reference_images=[r.model_dump(mode="json") for r in images],
                  reference_subjects=[s.model_dump(mode="json") for s in build_reference_contract(projected, "visual_only").subjects])
    job = _seal(MimoClipJob, values, "request_fingerprint")
    path = reconcile / "source_contract.json"
    contract = json.loads(path.read_text())
    contract["jobs"] = [job.model_dump(mode="json")]
    path.write_text(json.dumps(contract))
    raw = _rows(reconcile / "records.jsonl")[0]
    raw["source_job_fingerprint"] = job.request_fingerprint
    raw["record_fingerprint"] = prepared._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (reconcile / "records.jsonl").write_text(json.dumps(raw) + "\n")
    (reconcile.parents[3] / "h3/samples.jsonl").write_text(canonical.model_dump_json() + "\n")

    def forbidden(*a, **kw):
        raise AssertionError("frozen references must not be sampled again")

    monkeypatch.setattr(reconcile_module, "select_mimo_reference_projection", forbidden)
    output = tmp_path / "projected"
    prepared.prepare_frozen_audio_reuse_sources(reconcile_root=reconcile, prepared_root=output)
    actual = _rows(output / "h3/samples.jsonl", FinalH3SampleV2)[0]
    assert actual.visual_references == projected.visual_references
    assert len(actual.visual_references) == 1 and actual.visual_references[0].kind == "subject"
    assert actual.visual_references[0].image_artifact_path == refs[1].image_artifact_path
    assert actual.r2v_instruction == job.r2v_instruction
    assert _rows(output / "records.jsonl")[0]["annotation"] == raw["annotation"]
