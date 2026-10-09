from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from r2v_data_v2.h3.ra2va_group_source import adapt_production_sample
from r2v_data_v2.v3.production_export import ProductionSample
from tests.test_h3_audio_reuse_materializer import _case
from tests.test_h3_ra2va_group_pipeline import ffmpeg


def export_case(tmp_path, *, restricted=False):
    from r2v_data_v2.h3.ra2va_group_pipeline import frozen_job
    args, sample, _ = _case(tmp_path / "source", groups=("g1", "g1"), music="Gentle piano.", all_visible=True)
    video = tmp_path / "video.mp4"
    subprocess.run([ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x24:rate=4:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
    references = []
    for reference in sample.visual_references:
        row = reference.model_dump(mode="json", exclude={"image_artifact_path"})
        row["image_path"] = str(Path(reference.image_artifact_path).relative_to(tmp_path))
        references.append(row)
    source = ProductionSample(schema_version="r2v.v3.production_sample.1", sample_id=sample.clip_uid,
        clip_uid=sample.clip_uid, target_video=str(video), t2v_caption="A person in a room.",
        r2v_instruction="Person <Image 1>", references=references,
        source={"parent_video_id": "parent", "clip_suffix": "01", "shard_id": "shard"})
    task = adapt_production_sample(source, tmp_path, 0)
    job = frozen_job(task, {"path": args["job"].target_full_audio_path, "frame_count": 64000},
                     [s.model_dump(mode="json") for s in args["job"].segments])
    annotation = args["annotation"].model_dump(mode="json")
    if restricted:
        annotation["h3_semantics"]["shot1_caption"] = (
            "<Subject 1> stands in a room. He (S1) says, <d>[English] Exact, text!</d>. "
            "He (S1) says, <d>[English] Exact, text!</d>."
        )
        annotation["h3_semantics"]["summary"] = "A voice speaks while <Subject 1> stands in a room."
        for grounding in annotation["av_grounding"]["segment_groundings"]:
            grounding["evidence_codes"] = ["av_temporal_alignment", "no_visible_lip_motion"]
        for view in annotation["visual_observation"]["segment_views"]:
            for observation in view["entity_observations"]:
                observation["speech_correlated_articulation"] = "not_assessable"
    result = {"status": "ready", "clip_uid": job.clip_uid, "annotation": annotation,
        "backend_provenance": {"prompt_version": "h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"},
        "model_call_count": 2, "raw_responses": ["original visual", "original joint"]}
    stems = {s.stem_type: {"path": s.canonical_stem_path, "frame_count": s.canonical_frame_count}
             for s in args["stem_record"].stems}
    stems["separation_state"] = "success"
    return task, job, result, stems


def test_group_export_uses_real_h3_and_all_twelve_four_field_tasks(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_export import export_group_clip
    task, job, result, stems = export_case(tmp_path)
    before = copy.deepcopy(result)

    def no_source_scan(*args, **kwargs):
        raise AssertionError("legacy source media scan/reselection must not run")

    monkeypatch.setattr("r2v_data_v2.h3.mimo25_h3_materializer.build_reference_contract", no_source_scan)
    root = tmp_path / "bundle"
    summary = export_group_clip(root=root, task=task, job=job, result=result, stems=stems, ffmpeg=ffmpeg())
    assert summary["product_ready_count"] == 3 and summary["product_failed_count"] == 0, summary
    assert len(summary["task_counts"]) == 12 and set(summary["task_counts"].values()) == {1}
    assert summary["donor_count"] == 1 and summary["donor_segment_count"] == 2
    for path in (root / "training").glob("*.jsonl"):
        if path.name == "videos.jsonl":
            continue
        row = json.loads(path.read_text())
        assert set(row) == {"video", "images", "audios", "caption"}
        assert row["caption"].count("<d>[English] Exact, text!</d>") == 2
        assert all(Path(p).is_file() for p in row["images"] + row["audios"])
        assert row["caption"].count("voice characteristics described by") <= 1
    reserve = json.loads((root / "voice_donor_reserve.json").read_text())
    donor = reserve["donors"][0]
    assert donor["entity_id"] == "e1" and donor["speaker_group"] == "g1" and donor["speaker_id"] == "S1"
    assert donor["segment_count"] == 2
    original, _ = sf.read(donor["timeline_speech_asset_path"], dtype="int16", always_2d=True)
    for segment in donor["segments"]:
        pcm, rate = sf.read(segment["audio_path"], dtype="int16", always_2d=True)
        assert rate == 32000
        assert np.array_equal(pcm, original[segment["source_start_sample"]:segment["source_end_sample"]])
    assert result == before
    files = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert export_group_clip(root=root, task=task, job=job, result=result, stems=stems,
                             ffmpeg="must-not-run") == summary
    assert files == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_native_identity_restriction_blocks_speech_and_donors_not_full_audio(tmp_path):
    from r2v_data_v2.h3.ra2va_group_export import export_group_clip
    task, job, result, stems = export_case(tmp_path, restricted=True)
    root = tmp_path / "restricted"
    summary = export_group_clip(root=root, task=task, job=job, result=result, stems=stems, ffmpeg=ffmpeg())
    assert summary["product_ready_count"] == 2 and summary["product_failed_count"] == 0, summary
    assert summary["donor_count"] == summary["donor_segment_count"] == 0
    assert all(value == 0 for key, value in summary["task_counts"].items() if "speech_bgm" in key)
    for path in (root / "training").glob("*.jsonl"):
        if path.name == "videos.jsonl":
            continue
        for line in path.read_text().splitlines():
            assert "<Subject 1> (S1)" not in json.loads(line)["caption"]
    assert {e["reason"] for e in summary["identity_exclusions"]} == {"unconfirmed_visible_identity"}


def test_failed_annotation_cannot_publish_group_assets_or_donors(tmp_path):
    from r2v_data_v2.h3.ra2va_group_export import export_group_clip
    task, job, result, stems = export_case(tmp_path)
    result["status"] = "failed"
    root = tmp_path / "failed"
    import pytest
    with pytest.raises(ValueError, match="ready annotation"):
        export_group_clip(root=root, task=task, job=job, result=result, stems=stems, ffmpeg=ffmpeg())
    assert not root.exists()


def test_no_music_annotation_does_not_condition_on_music_stem(tmp_path):
    from r2v_data_v2.h3.ra2va_group_export import export_group_clip
    task, job, result, stems = export_case(tmp_path)
    result["annotation"]["h3_semantics"]["non_diegetic_music"] = "N/A"
    stems["separation_state"] = "unverified"
    root = tmp_path / "no-music"
    summary = export_group_clip(root=root, task=task, job=job, result=result, stems=stems, ffmpeg=ffmpeg())
    assert summary["product_failed_count"] == 0, summary
    products = [json.loads(line) for line in (root / "h3_audio_reuse_products_v1/records.jsonl").read_text().splitlines()]
    speech = next(p for p in products if p["conditioning_variant"] == "target_speech_reuse")
    assert [a["contract"]["kind"] for a in speech["audio_references"]] == ["speaker_speech_reuse"]
