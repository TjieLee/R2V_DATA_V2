from __future__ import annotations

import json

import pytest

from tests.test_h3_ra2va_group_local import fixture_group
from tests.test_h3_ra2va_group_pipeline import ffmpeg


def test_offline_group_replay_uses_saved_responses_and_real_export_without_network(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_local import run_cpu_pipeline
    from tools.replay_h3_ra2va_group_pilot import replay_group

    source, fixtures = fixture_group(tmp_path, count=1)
    old = tmp_path / "original"
    run_cpu_pipeline(source_root=source, run_root=old, fixtures_path=fixtures,
                     workers=1, limit=1, max_groups=1, ffmpeg=ffmpeg())
    group = old / "groups/group-000000"
    protected = {str(p): p.read_bytes() for p in group.rglob("*") if p.is_file()}

    def no_network(*args, **kwargs):
        raise AssertionError("offline replay must not create an HTTP client")

    monkeypatch.setattr("openai.OpenAI", no_network)
    output = tmp_path / "replay"
    summary = replay_group(source_group=group, output_root=output, ffmpeg=ffmpeg())
    assert summary["video_count"] == summary["reconcile_ready_count"] == 1
    assert summary["new_request_count"] == 0
    assert summary["response_replay_count"] == summary["historical_model_call_count"] == 2
    assert summary["product_ready_count"] == 3 and summary["product_failed_count"] == 0
    assert set(summary["task_counts"].values()) == {1}
    assert summary["donor_count"] == 1 and summary["donor_segment_count"] == 2
    assert protected == {str(p): p.read_bytes() for p in group.rglob("*") if p.is_file()}
    old_record = json.loads((group / "bundle/records.jsonl").read_text())
    record = json.loads((output / "bundle/records.jsonl").read_text())
    for field in ("annotation", "visual_raw_response", "speech_av_raw_response", "backend_provenance"):
        assert record[field] == old_record[field]
    assert record["postprocessing_version"] == "ra2va_group_durable_two_step_v2"
    assert record["source_record_path"]
    with pytest.raises(FileExistsError):
        replay_group(source_group=group, output_root=output, ffmpeg=ffmpeg())
