from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from r2v_data_v2.h3.audio_reuse import AudioReuseManifest, build_audio_reuse_assets
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    FrozenAudioReuseInventory,
)
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from tests.h3_audio_reuse_prepared_helpers import prepared_record
from tests.test_h3_audio_reuse import _fixture, _seal


def _export(root):
    from r2v_data_v2.h3.voice_donor_reserve import export_voice_donor_reserve

    return export_voice_donor_reserve(bundle_root=root)


def _bundle(tmp_path, *, intervals=((.1, .3, "g1"), (.9, 1.1, "g1")), all_visible=False,
            profile="A clear, low-register voice.", offscreen=False, mixed=False):
    args = _fixture(tmp_path, intervals, offscreen=offscreen, mixed=mixed)
    payload = args["annotation"].model_dump(mode="json")
    if all_visible:
        for grounding in payload["av_grounding"]["segment_groundings"]:
            grounding.update(binding_status="visible_entity", entity_id="e1", speech_presentation="onscreen_spoken",
                             evidence_codes=["visible_lip_motion"])
    payload["audio_observation"]["speaker_voice_profiles"] = [
        {"speaker_group": group, "voice_characteristics": profile}
        for group in dict.fromkeys(group for _, _, group in intervals)
    ]
    args["annotation"] = MimoAVAnnotationDraft.model_validate(payload)
    root = tmp_path / "bundle"
    prepared = root / "audio_reuse_prepared_v1"
    prepared.mkdir(parents=True)
    job = args["job"]
    args["output_root"] = root / "audio_reuse_assets_v1" / job.clip_uid
    manifest = build_audio_reuse_assets(**args)
    record = prepared_record(tmp_path, job, args["annotation"], args["stem_record"])
    (prepared / "records.jsonl").write_text(record.model_dump_json() + "\n")
    inventory = _seal(FrozenAudioReuseInventory, {
        "schema_version": "r2v.h3.audio_reuse_frozen_inventory.1",
        "source_reconcile_root": str(tmp_path / "unavailable-reconcile"),
        "source_stem_root": str(tmp_path / "unavailable-stems"),
        "source_h3_root": str(prepared / "h3"), "source_h3_samples_sha256": "a" * 64,
        "clip_count": 1, "jobs": [job.model_dump(mode="json")],
    }, "inventory_fingerprint")
    (prepared / "inventory.json").write_text(inventory.model_dump_json() + "\n")
    return root, job, manifest, record


def _manifest_path(root, job):
    return root / "audio_reuse_assets_v1" / job.clip_uid / "manifest.json"


def _write_manifest(root, job, payload):
    _manifest_path(root, job).write_text(json.dumps(payload) + "\n")


def _write_record(root, payload, *, update_manifest=False):
    record = _seal(AudioReusePreparedSource, payload, "prepared_fingerprint")
    (root / "audio_reuse_prepared_v1/records.jsonl").write_text(record.model_dump_json() + "\n")
    if update_manifest:
        from r2v_data_v2.h3.audio_reuse import _fingerprint

        path = root / "audio_reuse_assets_v1" / record.clip_uid / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_annotation_sha256"] = _fingerprint(record.annotation)
        for asset in manifest["speakers"]:
            asset["source_annotation_sha256"] = manifest["source_annotation_sha256"]
        path.write_text(json.dumps(manifest) + "\n")


def test_cli_exports_all_segments_sample_exact_with_real_paths(tmp_path):
    root, job, manifest, _ = _bundle(tmp_path)
    asset = manifest.speakers[0]
    timeline, _ = sf.read(asset.output_path, dtype="int16", always_2d=True)
    # Published order need not be chronological; the reserve always is.
    payload = manifest.model_dump(mode="json")
    payload["speakers"][0]["source_segment_ids"].reverse()
    payload["speakers"][0]["source_sample_ranges"].reverse()
    _write_manifest(root, job, payload)
    # The standalone exporter must not read the original mix or source stems.
    Path(job.target_full_audio_path).unlink()
    Path(job.target_video_path).unlink()
    for path in (tmp_path / "production").glob("*.wav"):
        path.unlink()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    from tools.export_h3_voice_donor_reserve import main

    result = main(["--bundle-root", str(root)])
    assert json.loads((root / "voice_donor_reserve.json").read_text()) == result
    assert result["schema_version"] == "r2v.h3.voice_donor_reserve.1"
    assert len(result["donors"]) == 1
    donor = result["donors"][0]
    assert {k: donor[k] for k in (
        "clip_uid", "entity_id", "donor_occurrence_id", "subject_label", "speaker_group", "speaker_id",
        "voice_characteristics", "timeline_speech_asset_path", "segment_count",
    )} == {
        "clip_uid": job.clip_uid, "entity_id": "e1", "donor_occurrence_id": f"{job.clip_uid}/e1",
        "subject_label": "<Subject 1>", "speaker_group": "g1", "speaker_id": "S1",
        "voice_characteristics": "A clear, low-register voice.", "timeline_speech_asset_path": asset.output_path,
        "segment_count": 2,
    }
    assert donor["total_speech_duration_seconds"] == pytest.approx(.4)
    assert [s["segment_id"] for s in donor["segments"]] == ["segment_1", "segment_2"]
    for row, sid, start, end, source_start, source_end, t0, t1 in zip(
        donor["segments"], ["segment_1", "segment_2"], [3200, 28800], [9600, 35200],
        [1600, 14400], [4800, 17600], [.1, .9], [.3, 1.1], strict=True,
    ):
        relative = f"voice_donor_segments_v1/{job.clip_uid}/g1/{sid}.flac"
        assert row == {
            "segment_id": sid, "audio_path": str(root / relative), "relative_path": relative,
            "start_time": t0, "end_time": t1, "source_start_sample": source_start,
            "source_end_sample": source_end, "source_sample_rate_hz": 16000,
        }
        pcm, rate = sf.read(row["audio_path"], dtype="int16", always_2d=True)
        assert rate == 32000 and pcm.dtype == np.int16 and pcm.shape == (6400, 2)
        assert sf.info(row["audio_path"]).subtype == "PCM_16"
        np.testing.assert_array_equal(pcm, timeline[start:end])
    assert before == {p: p.read_bytes() for p in before}
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize("missing_profile", [False, True])
def test_nullable_or_absent_voice_profile_is_not_an_eligibility_gate(tmp_path, missing_profile):
    root, _, _, record = _bundle(tmp_path, profile=None)
    if missing_profile:
        payload = record.model_dump(mode="json")
        payload["annotation"]["audio_observation"]["speaker_voice_profiles"] = []
        _write_record(root, payload, update_manifest=True)
    assert _export(root)["donors"][0]["voice_characteristics"] is None


def test_same_entity_different_speaker_groups_are_not_merged(tmp_path):
    root, job, _, _ = _bundle(tmp_path, intervals=((.1, .2, "g1"), (.3, .4, "g2")), all_visible=True)
    donors = _export(root)["donors"]
    assert [(d["entity_id"], d["speaker_group"], d["speaker_id"]) for d in donors] == [
        ("e1", "g1", "S1"), ("e1", "g2", "S2"),
    ]
    assert [d["donor_occurrence_id"] for d in donors] == [f"{job.clip_uid}/e1"] * 2
    assert [d["segment_count"] for d in donors] == [1, 1]
    assert len({s["audio_path"] for d in donors for s in d["segments"]}) == 2


@pytest.mark.parametrize("reason", ["offscreen", "mixed", "failed", "restricted", "sparse_restricted"])
def test_ineligible_or_failed_annotations_do_not_publish_donors(tmp_path, reason):
    root, job, manifest, record = _bundle(tmp_path, offscreen=reason == "offscreen", mixed=reason == "mixed")
    if reason == "failed":
        payload = record.model_dump(mode="json")
        payload.update(status="failed", failure_code="semantic_failure", failure_reason="Frozen failure")
        _write_record(root, payload)
        _manifest_path(root, job).unlink()  # Failed annotations do not require an asset manifest.
    elif reason in {"restricted", "sparse_restricted"}:
        from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
        from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend

        payload = record.model_dump(mode="json")
        payload["source_backend_provenance"] = TwoStepOpenAIMimo26Backend(MimoBackendConfig(
            api_key="fixture", transport="sglang", media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        ), stem_records_by_clip={}).provenance.model_dump(mode="json")
        for view in payload["annotation"]["visual_observation"]["segment_views"]:
            view["entity_observations"][0]["speech_correlated_articulation"] = "not_assessable"
        for grounding in payload["annotation"]["av_grounding"]["segment_groundings"]:
            grounding["evidence_codes"] = (["av_temporal_alignment", "no_visible_lip_motion"]
                                         if reason == "sparse_restricted" else ["voice_continuity"])
        _write_record(root, payload, update_manifest=True)
        assert manifest.speakers  # Historical assets must still obey the current restriction.
    result = _export(root)
    if reason == "mixed":
        assert result["donors"][0]["segment_count"] == 1
        assert result["donors"][0]["segments"][0]["segment_id"] == "segment_2"
    else:
        assert result["donors"] == []
        assert not list((root / "voice_donor_segments_v1").rglob("*.flac"))


def test_null_entity_group_is_excluded_without_dropping_legal_group(tmp_path):
    root, _, _, _ = _bundle(tmp_path, intervals=((.1, .2, "g1"), (.3, .4, "g2")))
    assert [d["speaker_group"] for d in _export(root)["donors"]] == ["g1"]


@pytest.mark.parametrize("stale", ["annotation", "asset_annotation", "asset_job", "asset_stem"])
def test_stale_manifest_ownership_cannot_publish_donors(tmp_path, stale):
    root, job, manifest, record = _bundle(tmp_path)
    if stale == "annotation":
        payload = record.model_dump(mode="json")
        for decision in payload["annotation"]["audio_observation"]["segment_decisions"]:
            decision["primary_speaker_group"] = "g2"
        for grounding in payload["annotation"]["av_grounding"]["segment_groundings"]:
            grounding["primary_speaker_group"] = "g2"
        payload["annotation"]["audio_observation"]["speaker_voice_profiles"][0]["speaker_group"] = "g2"
        _write_record(root, payload)
    else:
        field = {"asset_annotation": "source_annotation_sha256", "asset_job": "source_job_fingerprint",
                 "asset_stem": "source_stem_record_fingerprint"}[stale]
        payload = manifest.model_dump(mode="json")
        payload["speakers"][0][field] = "0" * 64
        _write_manifest(root, job, payload)
    with pytest.raises(ValueError, match="manifest|ownership"):
        _export(root)
    assert not (root / "voice_donor_reserve.json").exists()
    assert not list((root / "voice_donor_segments_v1").rglob("*.flac"))


@pytest.mark.parametrize("corruption", ["unknown", "missing", "missing_boundary", "mapping", "frozen_boundary", "short_timeline"])
def test_invalid_boundaries_fail_closed_without_publishing_json(tmp_path, corruption):
    root, job, manifest, _ = _bundle(tmp_path)
    payload = manifest.model_dump(mode="json")
    asset = payload["speakers"][0]
    if corruption == "unknown":
        asset["source_segment_ids"][0] = "unknown_segment"
        asset["source_sample_ranges"][0]["segment_id"] = "unknown_segment"
    elif corruption == "missing":
        asset["source_sample_ranges"].pop()
    elif corruption == "missing_boundary":
        del asset["source_sample_ranges"][0]["source_end_sample"]
    elif corruption == "mapping":
        asset["source_sample_ranges"][0]["target_start_sample"] += 1
    elif corruption == "frozen_boundary":
        asset["source_sample_ranges"][0]["source_end_sample"] += 1
        asset["source_sample_ranges"][0]["target_end_sample"] += 2
    else:
        pcm, _ = sf.read(asset["output_path"], dtype="int16", always_2d=True)
        sf.write(asset["output_path"], pcm[:34000], 32000, format="FLAC", subtype="PCM_16")
    _write_manifest(root, job, payload)
    with pytest.raises(ValueError):
        _export(root)
    assert not (root / "voice_donor_reserve.json").exists()


@pytest.mark.parametrize("format,subtype,rate,channels", [
    ("FLAC", "PCM_24", 32000, 2), ("FLAC", "PCM_16", 16000, 2),
    ("FLAC", "PCM_16", 32000, 1), ("WAV", "PCM_16", 32000, 2),
])
def test_timeline_requires_pcm16_32k_stereo_flac(tmp_path, format, subtype, rate, channels):
    root, _, manifest, _ = _bundle(tmp_path)
    path = manifest.speakers[0].output_path
    pcm, _ = sf.read(path, dtype="int16", always_2d=True)
    sf.write(path, pcm[:, :channels], rate, format=format, subtype=subtype)
    with pytest.raises(ValueError, match="PCM16"):
        _export(root)
    assert not (root / "voice_donor_reserve.json").exists()


def test_existing_json_is_never_overwritten_even_when_inputs_are_missing(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    sidecar = root / "voice_donor_reserve.json"
    sidecar.write_bytes(b"existing sidecar\n")
    with pytest.raises(FileExistsError):
        _export(root)
    assert sidecar.read_bytes() == b"existing sidecar\n"
    assert list(root.iterdir()) == [sidecar]


def test_completed_export_cannot_be_overwritten(tmp_path):
    root, _, _, _ = _bundle(tmp_path)
    _export(root)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        _export(root)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_fresh_crops_are_not_decoded_again_at_runtime(tmp_path, monkeypatch):
    root, _, _, _ = _bundle(tmp_path)
    real_read = sf.read

    def read_source_only(path, *args, **kwargs):
        if "voice_donor_segments_v1" in Path(path).parts:
            raise AssertionError("fresh crops must not be decoded again")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(sf, "read", read_source_only)
    assert _export(root)["donors"][0]["segment_count"] == 2


def test_interrupted_export_reuses_existing_exact_crop_without_overwrite(tmp_path, monkeypatch):
    root, job, manifest, _ = _bundle(tmp_path)
    real_write = sf.write

    def interrupt(path, *args, **kwargs):
        if "segment_2" in Path(path).name:
            raise OSError("simulated interruption")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(sf, "write", interrupt)
    with pytest.raises(OSError, match="interruption"):
        _export(root)
    existing = root / f"voice_donor_segments_v1/{job.clip_uid}/g1/segment_1.flac"
    assert existing.is_file()
    before = existing.read_bytes(), existing.stat().st_mtime_ns
    assert not (root / "voice_donor_reserve.json").exists()
    monkeypatch.setattr(sf, "write", real_write)
    result = _export(root)
    assert before == (existing.read_bytes(), existing.stat().st_mtime_ns)
    assert result["donors"][0]["segment_count"] == 2
    timeline, _ = sf.read(manifest.speakers[0].output_path, dtype="int16", always_2d=True)
    np.testing.assert_array_equal(sf.read(existing, dtype="int16", always_2d=True)[0], timeline[3200:9600])
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize("corrupt", [False, True])
def test_conflicting_existing_crop_is_never_overwritten(tmp_path, corrupt):
    root, job, _, _ = _bundle(tmp_path)
    path = root / f"voice_donor_segments_v1/{job.clip_uid}/g1/segment_1.flac"
    path.parent.mkdir(parents=True)
    if corrupt:
        path.write_bytes(b"not a FLAC")
    else:
        sf.write(path, np.zeros((6400, 2), dtype=np.int16), 32000, format="FLAC", subtype="PCM_16")
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises((FileExistsError, ValueError)):
        _export(root)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    assert not (root / "voice_donor_reserve.json").exists()


def test_segment_path_cannot_escape_the_bundle(tmp_path):
    root, job, manifest, _ = _bundle(tmp_path)
    payload = manifest.model_dump(mode="json")
    payload["speakers"][0]["source_segment_ids"][0] = "../../escape"
    payload["speakers"][0]["source_sample_ranges"][0]["segment_id"] = "../../escape"
    _write_manifest(root, job, payload)
    with pytest.raises(ValueError):
        _export(root)
    assert not (tmp_path / "escape.flac").exists()
    assert not (root / "voice_donor_reserve.json").exists()


def test_two_cpu_fixture_bundles_keep_24_training_rows_unchanged(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_training_bundle import materialize_ra2va_training_bundle
    from r2v_data_v2.h3.training_manifest_export import build_ra2va_training_task_rows
    from tests.test_h3_ra2va_training_bundle import _fixture as training_fixture

    counts_before, counts_after = [], []
    for index in range(2):
        case = tmp_path / f"case-{index}"
        reconcile, job, _, _ = training_fixture(case, monkeypatch, offscreen=bool(index))
        root = case / "bundle"
        summary = materialize_ra2va_training_bundle(reconcile_root=reconcile, output_root=root, allow_unverified=True)
        assert summary["product_failed_count"] == 0
        before = build_ra2va_training_task_rows(root)
        counts_before.append(sum(len(rows) for rows in before.values()))
        original_files = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        result = _export(root)
        after = build_ra2va_training_task_rows(root)
        counts_after.append(sum(len(rows) for rows in after.values()))
        assert after == before
        assert original_files == {p: p.read_bytes() for p in original_files}
        if index == 0:
            manifest = AudioReuseManifest.model_validate_json(_manifest_path(root, job).read_text())
            donor = result["donors"][0]
            assert donor["timeline_speech_asset_path"] == manifest.speakers[0].output_path
            assert donor["donor_occurrence_id"] == f"{job.clip_uid}/e1"
            assert Path(donor["segments"][0]["audio_path"]).is_file()
        else:
            assert result["donors"] == []
    assert counts_before == counts_after == [12, 12]
    assert sum(counts_before) == sum(counts_after) == 24
