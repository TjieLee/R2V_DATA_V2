from __future__ import annotations

import copy
import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse_finalizer as finalizer
from r2v_data_v2.h3 import audio_reuse_prepared as prepared
from r2v_data_v2.h3.audio_reuse import AudioReuseManifest
from r2v_data_v2.h3.audio_reuse_materializer import AudioReuseProduct
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file
from tests import test_h3_audio_shadow_qa as qa_fixture
from tests import test_h3_sam_audio_stem_shadow as sam_fixture
from tests.test_h3_audio_reuse import _signal
from tools.finalize_h3_audio_reuse_shadow import main


def _fixture(tmp_path, monkeypatch, *, music=True, multiple=False, cross=False, canonical_only=False, one_clip=False, mixed=False):
    # Use genuine production media formats before any fixture hashes are sealed.
    def write(path, *, frame_count=32000, sample_rate_hz=32000, channels=2):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), _signal(frame_count)[:, :channels], sample_rate_hz,
                 format="FLAC" if path.suffix == ".flac" else "WAV", subtype="PCM_16")

    def probe(path):
        info = sf.info(str(path))
        return sam_fixture.AudioFileProbe(
            sample_rate_hz=info.samplerate, channels=info.channels,
            frame_count=info.frames, duration_seconds=info.duration, format_name=info.format.lower(),
        )

    monkeypatch.setattr(sam_fixture, "_write_wav", write)
    monkeypatch.setattr(sam_fixture, "_probe", probe)
    monkeypatch.setattr(sam_fixture._Diarization, "diarize", lambda *a, **kw: [
        sam_fixture.DiarizationBackendSegment(start_time=.1, end_time=.3, speaker_label="A"),
        sam_fixture.DiarizationBackendSegment(start_time=.5, end_time=.8, speaker_label="B"),
    ])
    if mixed:
        def mixed_diarize(self, *, clip_uid, audio_path):
            if clip_uid == "clip-a":
                raise RuntimeError("synthetic per-clip DiariZen failure")
            if clip_uid == "clip-m":
                return []
            return sam_fixture._Diarization.diarize(self, clip_uid=clip_uid, audio_path=audio_path)
        monkeypatch.setattr(sam_fixture._MixedDiarization, "diarize", mixed_diarize)
    original_annotation = qa_fixture._annotation

    def annotation():
        values = original_annotation().model_dump(mode="json")
        for section, field in (("visual_observation", "segment_views"),
                               ("audio_observation", "segment_decisions"),
                               ("av_grounding", "segment_groundings")):
            rows = values[section][field]
            rows.append(copy.deepcopy(rows[0]))
            rows[1]["segment_id"] = "segment_0002"
            if section != "visual_observation":
                rows[1]["primary_speaker_group"] = "g2"
        values["av_grounding"]["segment_groundings"][0].update(
            binding_status="offscreen", entity_id=None, speech_presentation="offscreen_spoken",
            evidence_codes=["offscreen_audio"],
        )
        if multiple:
            decision = values["audio_observation"]["segment_decisions"][0]
            decision.update(vocal_composition="sequential_multi_speaker_speech", resolution="needs_acoustic_refinement",
                            secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"})
        profile = values["audio_observation"]["speaker_voice_profiles"][0]
        values["audio_observation"]["speaker_voice_profiles"].append({**profile, "speaker_group": "g2"})
        values["h3_semantics"]["shot1_caption"] = (
            "An offscreen voice (S1) says, <d>[English] exact transcript</d>. "
            "<Subject 1> (S2) replies, <d>[English] exact transcript</d>."
        )
        values["h3_semantics"]["non_diegetic_music"] = "A gentle piano melody." if music else "N/A"
        return qa_fixture.MimoAVAnnotationDraft.model_validate(values)

    monkeypatch.setattr(qa_fixture, "_annotation", annotation)
    original_run = qa_fixture.run_mimo25_stem_reconcile_shadow

    def run(**kw):
        # Freeze source variants before fixture reconcile, never modify them in the finalizer.
        base = qa_fixture.qa.build_mimo25_inventory()
        samples_path = tmp_path / "production/h3/samples.jsonl"
        samples = _rows(samples_path, qa_fixture.FinalH3SampleV2)
        if canonical_only:
            samples = [s for s in samples if s.pair_type == "canonical"]
        if cross:
            for sample in samples:
                if sample.pair_type == "cross_pair":
                    donor = "clip-m" if sample.clip_uid != "clip-m" else "clip-z"
                    sample.subject_voices[0].donor_clip_uid = donor
                    sample.subject_voices[0].donor_occurrence_id = f"{donor}/e1"
        samples_path.write_text("".join(s.model_dump_json() + "\n" for s in samples))
        values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
        values["source_h3_samples_sha256"] = sha256_file(samples_path)
        for job in values["jobs"]:
            job["source_h3_sample_ids"] = [s.sample_id for s in samples if s.clip_uid == job["clip_uid"]]
            job.update(qa_fixture._job({k: v for k, v in job.items() if k != "request_fingerprint"}).model_dump(mode="json"))
        kw["jobs"] = [qa_fixture._job({
            **j.model_dump(mode="json", exclude={"request_fingerprint"}),
            "source_h3_sample_ids": [s.sample_id for s in samples if s.clip_uid == j.clip_uid],
        }) for j in kw["jobs"]]
        if one_clip:
            values["jobs"] = values["jobs"][:1]
            values["clip_count"] = 1
            kw["jobs"] = kw["jobs"][:1]
            kw["source_clip_uids"] = ["clip-z"]
            (tmp_path / "production/case.json").write_text('{"clip_uids":["clip-z"]}')
        base = qa_fixture._inventory(values)
        monkeypatch.setattr(qa_fixture.qa, "build_mimo25_inventory", lambda **kw: base)
        return original_run(**kw)

    monkeypatch.setattr(qa_fixture, "run_mimo25_stem_reconcile_shadow", run)
    args, shadow = qa_fixture._fixture(tmp_path, monkeypatch, variants=True, cross_variant=cross, mixed=mixed)
    base = qa_fixture.qa.build_mimo25_inventory()
    monkeypatch.setattr(finalizer, "build_mimo25_inventory", lambda **kw: base)
    monkeypatch.setattr(prepared, "build_mimo25_inventory", lambda **kw: base)
    production = args["audio_production_root"]
    for relative in ("diarization/inventory.json", "diarization/raw_segments.jsonl",
                     "diarization/bound_segments.jsonl", "asr/segments.jsonl",
                     "binding_audit_v1/segments.jsonl"):
        path = production / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (args["visual_production_root"] / "samples.jsonl").write_text("{}\n")
    return {**args, "allow_unverified": True}, shadow


def _rows(path, model):
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def test_named_finalizer_real_stages_pcm_subject_s2_and_qa(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    # Fixture setup may simulate inference; finalization itself must not invoke it.
    def forbidden(*a, **kw):
        raise AssertionError("no inference during finalization")

    monkeypatch.setattr(qa_fixture._Reconcile, "reconcile", forbidden)
    argv = [part for key, value in args.items() if key != "allow_unverified"
            for part in ("--" + key.replace("_", "-"), str(value))] + ["--allow-unverified"]
    result = main(argv)
    summary = finalizer.AudioReuseFinalizationSummary.model_validate(result)
    assert summary.clip_uids == ["clip-z", "clip-a", "clip-m"]
    assert (summary.reconcile_ready_count, summary.reconcile_failed_count) == (2, 1)
    assert summary.unavailable_clips == {"clip-a": "synthetic_failed"}
    assert (summary.reuse_asset_clip_count, summary.speaker_speech_asset_count, summary.music_asset_count) == (2, 4, 2)
    assert (summary.prepared_ready_count, summary.prepared_failed_count) == (2, 1)
    assert (summary.product_sample_count, summary.product_ready_count, summary.product_failed_count) == (9, 6, 3)
    assert summary.audio_kind_counts == {"full_audio_reuse": 2, "music_reuse": 2, "speaker_speech_reuse": 4}
    assert summary.model_call_count == 0 and summary.production_artifacts_modified is False
    assert not (shadow / finalizer.ASSETS_STAGE / "clip-a").exists()
    manifest = AudioReuseManifest.model_validate_json((shadow / finalizer.ASSETS_STAGE / "clip-z/manifest.json").read_text())
    assert [(s.speaker_id, s.entity_id, s.subject_index) for s in manifest.speakers] == [
        ("S1", None, None), ("S2", "e1", 1),
    ]
    for asset in manifest.speakers:
        pcm, rate = sf.read(asset.output_path, dtype="int16", always_2d=True)
        source, _ = sf.read(asset.source_stem_path, dtype="int16", always_2d=True)
        assert rate == 32000 and len(pcm) == sf.info(asset.target_audio_path).frames == 32000
        expected = np.zeros_like(pcm)
        for interval in asset.source_sample_ranges:
            expected[interval.target_start_sample:interval.target_end_sample] = source[interval.target_start_sample:interval.target_end_sample]
        assert np.any(expected)
        np.testing.assert_array_equal(pcm, expected)
    products = _rows(shadow / finalizer.PRODUCTS_STAGE / "records.jsonl", AudioReuseProduct)
    target = next(p for p in products if p.sample_id == "clip-z/in_pair")
    assert [a.contract.kind for a in target.audio_references] == ["speaker_speech_reuse", "speaker_speech_reuse", "music_reuse"]
    assert (target.audio_references[1].contract.speaker_id, target.audio_references[1].contract.subject_label) == ("S2", "<Subject 1>")
    assert all(not p.audio_references for p in products if p.status == "failed")
    assert before == {p: sha256_file(p) for p in before}
    qa_args = {k: v for k, v in args.items() if k != "allow_unverified"}
    qa_fixture.qa.build_audio_shadow_qa(**qa_args)
    data = json.loads((shadow / "qa/data.json").read_text())
    variants = data["clips"][0]["final_h3"]["variants"]
    assert any(v["source_kind"] == "audio_reuse_product" for v in variants)
    full = [v for v in variants if v["conditioning_variant"] == "full_audio_reuse"]
    assert len(full) == 1 and full[0]["source_kind"] == "audio_reuse_product"
    with pytest.raises(FileExistsError):
        finalizer.finalize_audio_reuse_shadow(**args)


@pytest.mark.parametrize("stage", [finalizer.ASSETS_STAGE, finalizer.PREPARED_STAGE, finalizer.PRODUCTS_STAGE, finalizer.FINALIZATION_STAGE])
def test_preflight_refuses_any_existing_output_before_assets(tmp_path, monkeypatch, stage):
    args, shadow = _fixture(tmp_path, monkeypatch)
    (shadow / stage).mkdir()
    sentinel = shadow / stage / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        finalizer.finalize_audio_reuse_shadow(**args)
    assert sentinel.read_text() == "keep"
    assert not (shadow / finalizer.ASSETS_STAGE).exists() or stage == finalizer.ASSETS_STAGE


def test_dangling_output_symlink_cannot_redirect_publication(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch)
    redirected = shadow / "separation/new-output"
    (shadow / finalizer.ASSETS_STAGE).symlink_to(redirected, target_is_directory=True)
    with pytest.raises(FileExistsError):
        finalizer.finalize_audio_reuse_shadow(**args)
    assert not redirected.exists()


@pytest.mark.parametrize("corruption", ["samples", "stem", "reconcile", "route", "case", "unverified"])
def test_stale_or_incompatible_inputs_fail_before_publication(tmp_path, monkeypatch, corruption):
    args, shadow = _fixture(tmp_path, monkeypatch)
    if corruption == "samples":
        with (args["audio_production_root"] / "h3/samples.jsonl").open("a") as stream:
            stream.write("\n")
    elif corruption in {"stem", "reconcile"}:
        root = shadow / ("separation" if corruption == "stem" else finalizer.MIMO25_STEM_RECONCILE_STAGE)
        with (root / "records.jsonl").open("a") as stream:
            stream.write('{"stale":true}\n')
    elif corruption == "route":
        args["sam_route"] = "voice_first"
    elif corruption == "case":
        args["case_manifest"].write_text('{"clip_uids":["clip-m","clip-z"]}')
    else:
        args["allow_unverified"] = False
    with pytest.raises(ValueError):
        finalizer.finalize_audio_reuse_shadow(**args)
    assert all(not (shadow / stage).exists() for stage in (
        finalizer.ASSETS_STAGE, finalizer.PREPARED_STAGE, finalizer.PRODUCTS_STAGE, finalizer.FINALIZATION_STAGE,
    ))


def test_music_absence(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch, music=False)
    result = finalizer.finalize_audio_reuse_shadow(**args)
    assert "music_reuse" not in result.audio_kind_counts
    assert result.audio_kind_counts["full_audio_reuse"] == 2
    assert result.speaker_speech_asset_count == 4


def test_multiple_speaker_exclusion_keeps_existing_materialization_failure(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch, multiple=True)
    result = finalizer.finalize_audio_reuse_shadow(**args)
    assert result.reuse_exclusion_reason_counts == {"non_single_speaker": 2, "secondary_vocal_activity": 2}
    assert result.speaker_speech_asset_count == 2
    products = _rows(shadow / finalizer.PRODUCTS_STAGE / "records.jsonl", AudioReuseProduct)
    assert all(p.status == "failed" and not p.audio_references for p in products)
    assert "multi_speaker_segment_requires_turn_refinement" in products[0].failure_reason


def test_one_ready_canonical_derives_target_and_full_audio(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch, canonical_only=True, one_clip=True)
    result = finalizer.finalize_audio_reuse_shadow(**args)
    assert result.clip_uids == ["clip-z"]
    assert result.product_ready_count == 3 and result.product_failed_count == 0
    products = _rows(shadow / finalizer.PRODUCTS_STAGE / "records.jsonl", AudioReuseProduct)
    assert [p.conditioning_variant for p in products] == ["visual_only", "target_speech_reuse", "full_audio_reuse"]
    assert products[0].audio_references == []
    assert [r.contract.kind for r in products[1].audio_references] == ["speaker_speech_reuse", "speaker_speech_reuse", "music_reuse"]
    assert products[2].audio_references[0].contract.path == products[1].audio_references[0].reuse_asset.target_audio_path


def test_cross_pair_uses_only_frozen_donor_reuse_track(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch, cross=True)
    result = finalizer.finalize_audio_reuse_shadow(**args)
    assert result.audio_kind_counts["cross_voice"] == 2
    products = _rows(shadow / finalizer.PRODUCTS_STAGE / "records.jsonl", AudioReuseProduct)
    cross = next(p for p in products if p.sample_id == "clip-z/cross_pair")
    voice = next(r for r in cross.audio_references if r.contract.kind == "cross_voice")
    assert voice.source_clip_uid == "clip-m"
    assert voice.donor_occurrence_id == "clip-m/e1"
    assert voice.contract.speaker_id == "S2" and voice.contract.subject_label == "<Subject 1>"
    assert voice.reuse_asset.speaker_id == "S2"
    assert "/clip-m/" in voice.contract.path


def test_upstream_diarization_failure_remains_explicit(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch, mixed=True)
    result = finalizer.finalize_audio_reuse_shadow(**args)
    assert result.upstream_unavailable_count == 1
    assert result.unavailable_clips == {"clip-a": "stem_diarization_failed"}
    assert result.reconcile_ready_count == 2 and result.reconcile_failed_count == 0
    assert result.product_ready_count == 6 and result.product_failed_count == 0
    assert not (shadow / finalizer.ASSETS_STAGE / "clip-a").exists()


def test_late_source_mutation_cannot_publish_completion(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch)
    original = finalizer.materialize_audio_reuse_products

    def changed(**kw):
        result = original(**kw)
        with args["case_manifest"].open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(finalizer, "materialize_audio_reuse_products", changed)
    with pytest.raises(ValueError, match="input changed"):
        finalizer.finalize_audio_reuse_shadow(**args)
    assert not (shadow / finalizer.FINALIZATION_STAGE).exists()
    assert not list(shadow.glob(".reuse-finalize-*"))


def test_later_stage_failure_preserves_published_assets_without_final_summary(tmp_path, monkeypatch):
    args, shadow = _fixture(tmp_path, monkeypatch)
    def fail(**kw):
        raise ValueError("synthetic prepare failure")
    monkeypatch.setattr(finalizer, "prepare_audio_reuse_sources", fail)
    with pytest.raises(ValueError, match="synthetic"):
        finalizer.finalize_audio_reuse_shadow(**args)
    assert (shadow / finalizer.ASSETS_STAGE / "clip-z/manifest.json").is_file()
    assert not (shadow / finalizer.FINALIZATION_STAGE).exists()
    assert not (shadow / finalizer.PRODUCTS_STAGE).exists()
