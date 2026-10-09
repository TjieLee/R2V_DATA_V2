from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
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
from tests.test_h3_mimo25_av_shadow import _Completions
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


def _single_v3_raw(job, *, binding="visible_subject", composition="single_speaker", music="Quiet piano."):
    rows = []
    blocks = []
    transcribed_groups = []
    for index, segment in enumerate(job.segments):
        group = f"g{index + 1}"
        transcribed = segment.asr_status == "transcribed"
        multi = composition in {"overlapping_secondary_speech", "sequential_multi_speaker_speech"}
        same = composition == "same_speaker_nonlexical"
        rows.append({
            "segment_id": segment.segment_id, "primary_speaker_group": group,
            "delivery_style": "calm conversational delivery" if transcribed else None,
            "binding_status": binding, "speaker_subject_label": "<Subject 1>" if binding == "visible_subject" else None,
            "resolution": "needs_acoustic_refinement" if multi else "resolved",
            "vocal_composition": composition,
            "secondary_vocal_activity": (
                {"present": True, "speaker_relation": "different_speaker", "kind": "speech"} if multi
                else {"present": True, "speaker_relation": "same_speaker", "kind": "sigh"} if same
                else {"present": False, "speaker_relation": "none", "kind": None}
            ),
            "confidence": "medium", "audio_evidence_codes": ["voice_continuity"],
            "visible_subject_labels": ["<Subject 1>"],
            "speech_presentation": "onscreen_spoken" if binding == "visible_subject" else (
                "offscreen_spoken" if binding == "offscreen" else "uncertain"),
            "evidence_codes": ["av_temporal_alignment" if binding == "visible_subject" else (
                "offscreen_audio" if binding == "offscreen" else "insufficient_evidence")],
        })
        if transcribed:
            transcribed_groups.append(group)
            lead = "<Subject 1>" if binding == "visible_subject" else "An offscreen voice" if binding == "offscreen" else "A visible woman"
            blocks.append(f"{lead} says, <d>[{segment.asr_language}] {segment.asr_text}</d>")
    return {
        "schema_version": "r2v.h3.mimo26_single_compact.3",
        "visual_blocks": [{"block_id": "v1", "text": "<Subject 1> stands in a room."}],
        "segments": rows,
        "speaker_voice_profiles": [
            {"speaker_group": group, "voice_characteristics": "a mid-register voice with an even cadence"}
            for group in transcribed_groups
        ],
        "h3_semantics": {
            "subject_definitions": [{"subject_label": "<Subject 1>", "description": "an adult woman in a blue coat"}],
            "summary": "A woman stands in a room.",
            "style_opening": "Naturalistic live-action cinematography with a fixed camera.",
            "shot1_caption": "<Subject 1> stands in a room. " + " ".join(blocks),
            "overall_soundscape": "Low interior ambience.", "non_diegetic_music": music,
            "visual_retention_analysis": [{"subject_label": "<Subject 1>", "marker": "fully_preserved",
                                           "description": "The blue coat remains recognizable."}],
        },
    }


def _write_single_v3(reconcile, job, stem, payload):
    completions = _Completions([(json.dumps(payload), 8)])
    backend = SingleCallOpenAIMimo25Backend(
        MimoBackendConfig(api_key="fixture", transport="sglang", model="mimo-v2.6-flash-rl",
                          media_resolver=MimoMediaResolver(mode="http", media_root=reconcile.parents[5],
                                                           media_base_url="http://media.invalid/")),
        single_contract="compact3", stem_records_by_clip={job.clip_uid: stem},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=[job], stem_records=[stem], backend=backend, output_root=reconcile,
        route="resolved", allow_unverified=True, overwrite=True, binding_evidence_mode="none",
    )
    assert len(completions.requests) == summary.model_call_count == 1
    assert summary.ready_count == 1, _rows(reconcile / "records.jsonl")
    record = _rows(reconcile / "records.jsonl")[0]
    assert json.loads(record["raw_responses"][0]) == payload
    assert record["backend_provenance"]["schema_version"] == "r2v.h3.mimo25_backend.79"
    return record


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
    assert result["human_review_required"] is False
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


@pytest.mark.parametrize("sparse_lip_motion", [False, True])
def test_two_step_unconfirmed_identity_exports_available_products_without_review(tmp_path, monkeypatch, sparse_lip_motion):
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
    from tools.export_h3_training_manifests import main as export
    from tools.materialize_h3_ra2va_training_bundle import main

    reconcile, job, _, _ = _fixture(tmp_path, monkeypatch)
    raw = _rows(reconcile / "records.jsonl")[0]
    raw["annotation"]["av_grounding"]["segment_groundings"][0]["evidence_codes"] = (
        ["av_temporal_alignment", "no_visible_lip_motion"] if sparse_lip_motion else ["voice_continuity"]
    )
    raw["annotation"]["visual_observation"]["segment_views"][0]["entity_observations"][0]["speech_correlated_articulation"] = "not_assessable"
    raw.update(
        backend_provenance=TwoStepOpenAIMimo26Backend(MimoBackendConfig(
            api_key="fixture", transport="sglang",
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        ), stem_records_by_clip={}).provenance.model_dump(mode="json"),
        raw_responses=["visual", "joint"], visual_raw_response="visual", speech_av_raw_response="joint",
        visual_model_call_count=1, model_call_count=2,
        diagnostics=[
            {**raw["diagnostics"][0], "input_modality": "target_video_visual_only"},
            {**raw["diagnostics"][0], "input_modality": "target_video_joint_av_audio"},
        ],
    )
    raw["record_fingerprint"] = prepared._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (reconcile / "records.jsonl").write_text(json.dumps(raw) + "\n")
    summary = json.loads((reconcile / "summary.json").read_text())
    summary.update(visual_model_call_count=1, model_call_count=2)
    (reconcile / "summary.json").write_text(json.dumps(summary))

    output = tmp_path / "bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    assert result["human_review_required"] is False
    assert result["model_call_count"] == result["product_failed_count"] == 0
    manifest = AudioReuseManifest.model_validate_json((output / f"audio_reuse_assets_v1/{job.clip_uid}/manifest.json").read_text())
    assert manifest.speakers == []
    assert {e.reason for e in manifest.exclusions} == {"unconfirmed_visible_identity"}
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert {p.conditioning_variant for p in products} == {"visual_only", "full_audio_reuse"}
    assert all(p.status == "ready" for p in products)
    assert all("g1:identity_publication_restricted" in p.warnings for p in products)
    assert all(s.entity_id is None and s.entity_occurrence_id is None for p in products for s in p.corrected_speech_segments)
    assert all("<Subject 1> (S1)" not in p.rendered_h3_prompt for p in products)
    assert all("A voice (S1) says," in p.rendered_h3_prompt for p in products)
    assert _rows(output / "audio_reuse_prepared_v1/records.jsonl")[0]["annotation"] == raw["annotation"]
    assert _rows(reconcile / "records.jsonl")[0] == raw
    # Historical bundles remain directly exportable without changing their metadata.
    historical_summary = {**result, "human_review_required": True}
    (output / "summary.json").write_text(json.dumps(historical_summary))
    exported = export(["--ra2va-shadow-root", str(output), "--output-root", str(tmp_path / "training")])
    rows = [row for task in result["task_counts"] for row in _rows(exported / f"{task}.jsonl")]
    assert len(rows) == 8
    assert all(set(row) == {"video", "images", "audios", "caption"} for row in rows)
    assert json.loads((output / "summary.json").read_text()) == historical_summary
    assert not (output / "review").exists()


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


def test_frozen_uncertain_composition_remains_ineligible_for_speaker_reuse(tmp_path, monkeypatch):
    from tools.materialize_h3_ra2va_training_bundle import main

    reconcile, job, _, _ = _fixture(tmp_path, monkeypatch)
    raw = _rows(reconcile / "records.jsonl")[0]
    decision = raw["annotation"]["audio_observation"]["segment_decisions"][0]
    decision["vocal_composition"] = "uncertain"
    raw["record_fingerprint"] = prepared._hash({k: v for k, v in raw.items() if k != "record_fingerprint"})
    (reconcile / "records.jsonl").write_text(json.dumps(raw) + "\n")
    output = tmp_path / "bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    manifest = AudioReuseManifest.model_validate_json((output / f"audio_reuse_assets_v1/{job.clip_uid}/manifest.json").read_text())
    assert manifest.speakers == []
    assert [(e.segment_id, e.reason) for e in manifest.exclusions] == [(job.segments[0].segment_id, "non_single_speaker")]
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert {p.conditioning_variant for p in products} == {"visual_only", "full_audio_reuse"}
    assert all(p.status == "ready" for p in products)
    assert result["unavailable_target_speech_reuse_clip_uids"] == [job.clip_uid]
    assert _rows(output / "audio_reuse_prepared_v1/records.jsonl")[0]["annotation"] == raw["annotation"]


@pytest.mark.parametrize("promote_face", [False, True])
@pytest.mark.parametrize("single_v3", [False, True])
def test_frozen_bundle_projects_selected_references_only_at_materialization(tmp_path, monkeypatch, promote_face, single_v3):
    from r2v_data_v2.h3 import mimo25_av_reconcile as reconcile_module
    from r2v_data_v2.h3.mimo25_h3_materializer import _prepare_materialization_context
    from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract
    from tests.test_h3_mimo25_av_shadow import (
        _augmentation_draws,
        _visual_reference_inventory,
    )

    reconcile, job, canonical, stem = _fixture(tmp_path, monkeypatch)
    refs = _visual_reference_inventory(tmp_path, ["subject", "face", "hair"], same_entity=True)
    canonical = canonical.model_copy(update={"visual_references": refs})
    _augmentation_draws(monkeypatch, [.8 if promote_face else .0, .9])
    selection, images = reconcile_module.select_mimo_reference_projection(job.clip_uid, refs)
    assert selection.original_picture_count == 3 and selection.selected_picture_count == 1
    assert bool(selection.face_promotions) is promote_face
    assert selection.selected_source_image_indexes == [2 if promote_face else 1]
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
    if single_v3:
        raw = _write_single_v3(reconcile, job, stem, _single_v3_raw(job))
    (reconcile.parents[3] / "h3/samples.jsonl").write_text(canonical.model_dump_json() + "\n")

    def forbidden(*a, **kw):
        raise AssertionError("frozen references must not be sampled again")

    monkeypatch.setattr(reconcile_module, "select_mimo_reference_projection", forbidden)
    from tools.materialize_h3_ra2va_training_bundle import main

    output = tmp_path / "bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert all(p.status == "ready" for p in products), [p.failure_reason for p in products]
    assert result["product_failed_count"] == 0 and result["model_call_count"] == 0
    stage = output / "audio_reuse_prepared_v1"
    actual = _rows(stage / "h3/samples.jsonl", FinalH3SampleV2)[0]
    assert actual.visual_references == refs
    assert [r.image_index for r in actual.visual_references] == [1, 2, 3]
    assert [r.owner_entity_id for r in actual.visual_references] == [None, "e1", "e1"]
    assert actual.r2v_instruction == job.r2v_instruction
    record = _rows(stage / "records.jsonl", prepared.AudioReusePreparedSource)[0]
    assert record.annotation.model_dump(mode="json") == raw["annotation"]
    context = _prepare_materialization_context(actual, job, record, conditioning_variant="visual_only")
    assert context.sample.visual_references == projected.visual_references
    assert context.contract.subjects == job.reference_subjects
    picture = context.contract.pictures[0]
    selected = refs[1 if promote_face else 0]
    assert picture.picture_label == "<Picture 1>" and picture.image_index == 1
    assert picture.image_id == projected.visual_references[0].image_id == "image_1"
    assert job.reference_images[0].source_image_id == selected.image_id
    assert picture.image_path == selected.image_artifact_path
    assert Path(picture.image_path).read_bytes() == Path(selected.image_artifact_path).read_bytes()
    assert picture.kind == "subject" and picture.entity_id == "e1"
    assert picture.owner_entity_id is None and picture.attribute_id is None
    assert context.contract.subjects[0].entity_id == "e1"
    assert context.contract.subjects[0].source_picture_labels == ["<Picture 1>"]
    for product in products:
        assert "<Picture 1>" in product.rendered_h3_prompt
        assert "<Picture 2>" not in product.rendered_h3_prompt and "<Picture 3>" not in product.rendered_h3_prompt
    references = [c.row["images"] for c in build_review_cases(output) if c.task in {
        "r2va_reference", "ra2va_reference_full_audio", "ra2va_reference_speech_bgm",
    }]
    assert references and all(paths == [selected.image_artifact_path] for paths in references)


@pytest.mark.parametrize(("binding", "composition", "no_speech", "task_count"), [
    ("visible_subject", "single_speaker", False, 12),
    ("visible_subject", "same_speaker_nonlexical", False, 12),
    ("offscreen", "single_speaker", False, 12),
    ("no_reliable_subject", "single_speaker", False, 12),
    ("no_reliable_subject", "uncertain", False, 8),
    ("no_reliable_subject", "uncertain", True, 8),
])
def test_single_v3_raw_to_assets_h3_frames_and_training_keeps_common_contract(
    tmp_path, monkeypatch, binding, composition, no_speech, task_count,
):
    from tools.materialize_h3_ra2va_training_bundle import main

    reconcile, job, _, stem = _fixture(tmp_path, monkeypatch, no_speech=no_speech)
    payload = _single_v3_raw(job, binding=binding, composition=composition)
    if binding == "no_reliable_subject":
        payload["h3_semantics"]["subject_definitions"][0]["description"] = "a white pedestal table"
    raw = _write_single_v3(reconcile, job, stem, payload)
    output = tmp_path / "v3-bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    assert result["model_call_count"] == 0
    assert result["product_failed_count"] == 0
    assert sum(result["task_counts"].values()) == task_count
    manifest = AudioReuseManifest.model_validate_json((output / "audio_reuse_assets_v1" / job.clip_uid / "manifest.json").read_text())
    eligible = not no_speech and composition in {"single_speaker", "same_speaker_nonlexical"}
    assert len(manifest.speakers) == int(eligible)
    if eligible:
        assert manifest.speakers[0].entity_id == ("e1" if binding == "visible_subject" else None)
        expected, _ = sf.read(stem.speech.canonical_stem_path, dtype="int16", always_2d=True)
        actual, _ = sf.read(manifest.speakers[0].output_path, dtype="int16", always_2d=True)
        segment = job.segments[0]
        assert np.array_equal(actual[segment.source_start_sample:segment.source_end_sample],
                              expected[segment.source_start_sample:segment.source_end_sample])
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert all(p.status == "ready" and p.schema_version == "r2v.h3.audio_reuse_product.1" for p in products)
    for product in products:
        if no_speech:
            assert "<d>" not in product.rendered_h3_prompt
        else:
            assert "(S1)" in product.rendered_h3_prompt
            assert f"<d>[{job.segments[0].asr_language}] {job.segments[0].asr_text}</d>" in product.rendered_h3_prompt
    speech_products = [p for p in products if p.conditioning_variant == "target_speech_reuse"]
    assert len(speech_products) == int(eligible)
    if eligible:
        speech_audio = speech_products[0].audio_references[0].contract
        projected_record = _rows(output / "audio_reuse_prepared_v1/records.jsonl", prepared.AudioReusePreparedSource)[0]
        assert projected_record.annotation.speaker_voice_profiles[0].voice_characteristics == (
            raw["annotation"]["audio_observation"]["speaker_voice_profiles"][0]["voice_characteristics"]
        )
        # Target reuse currently omits profiles for both backends; do not change its contract here.
        assert speech_audio.voice_characteristics is None
    assert len(_rows(output / frame.FRAME_STAGE / "records.jsonl")) == 3 * len(products)
    for case in build_review_cases(output):
        assert set(case.row) == {"video", "images", "audios", "caption"}
        assert all(Path(path).is_file() for path in case.row["images"] + case.row["audios"])
    if binding == "no_reliable_subject":
        assert raw["annotation"]["av_grounding"]["segment_groundings"][0]["entity_id"] is None
        assert "white pedestal table" in products[0].rendered_h3_prompt


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_single_v3_multi_speaker_excludes_reuse_preserves_asr_and_real_h3_failure(tmp_path, monkeypatch, composition):
    from tools.materialize_h3_ra2va_training_bundle import main

    reconcile, job, _, stem = _fixture(tmp_path, monkeypatch)
    record = _write_single_v3(reconcile, job, stem, _single_v3_raw(job, binding="no_reliable_subject", composition=composition))
    assert job.segments[0].asr_text in record["annotation"]["h3_semantics"]["shot1_caption"]
    output = tmp_path / "multi-bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    manifest = AudioReuseManifest.model_validate_json((output / "audio_reuse_assets_v1" / job.clip_uid / "manifest.json").read_text())
    assert manifest.speakers == []
    assert {e.reason for e in manifest.exclusions} >= {"non_single_speaker", "secondary_vocal_activity"}
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert products and all(p.status == "failed" for p in products)
    assert all("multi_speaker_segment_requires_turn_refinement" in p.failure_reason for p in products)
    assert result["product_failed_count"] == len(products)


def test_single_v3_overlapping_distinct_groups_never_copy_mixed_waveform(tmp_path, monkeypatch):
    from tools.materialize_h3_ra2va_training_bundle import main

    reconcile, job, _, stem = _fixture(tmp_path, monkeypatch)
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    first = values["segments"][0]
    second = {**first, "segment_id": "segment_0002", "source_speaker_cluster_id": "cluster_b",
              "start_time": first["start_time"] + .05, "end_time": first["end_time"] + .05,
              "source_start_sample": first["source_start_sample"] + 1600,
              "source_end_sample": first["source_end_sample"] + 1600, "asr_text": "A second voice answers."}
    values["segments"].append(second)
    job = _seal(MimoClipJob, values, "request_fingerprint")
    record = _write_single_v3(reconcile, job, stem, _single_v3_raw(job, binding="offscreen"))
    assert [d["primary_speaker_group"] for d in record["annotation"]["audio_observation"]["segment_decisions"]] == ["g1", "g2"]
    output = tmp_path / "overlap-bundle"
    result = main(["--reconcile-root", str(reconcile), "--output-root", str(output), "--allow-unverified"])
    manifest = AudioReuseManifest.model_validate_json((output / f"audio_reuse_assets_v1/{job.clip_uid}/manifest.json").read_text())
    assert manifest.speakers == []
    assert [(e.segment_id, e.reason) for e in manifest.exclusions] == [
        (s.segment_id, "cross_speaker_overlap") for s in job.segments
    ]
    assert result["product_failed_count"] == 0
    products = _rows(output / "h3_audio_reuse_products_v1/records.jsonl", AudioReuseProduct)
    assert {p.conditioning_variant for p in products} == {"visual_only", "full_audio_reuse"}
    for product in products:
        for segment in job.segments:
            assert f"<d>[{segment.asr_language}] {segment.asr_text}</d>" in product.rendered_h3_prompt
