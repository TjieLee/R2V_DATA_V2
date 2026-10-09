from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_auk_speech_shadow import setup as auk_setup  # noqa: F401
from tests.test_h3_ra2va_group_export import export_case
from tests.test_h3_ra2va_group_pipeline import ffmpeg, task_at
from tests.test_h3_ra2va_group_source import publish_group


def fixture_group(root, count=20):
    task, job, result, _ = export_case(root / "media")
    source = root / "post_mask"
    directory, marker = publish_group(source, count=count, declared=count + 5)
    for p in task.pictures:
        (directory / f"reference-{p['image_index']}.png").symlink_to(p["image_path"])
    samples, clips = [], {}
    visual, speech, profiles, finalized = map(json.loads, split_annotation(json.dumps(result["annotation"])))
    for rows in (visual["segment_views"], speech["audio_observation"]["segment_decisions"],
                 speech["av_grounding"]["segment_groundings"]):
        for index, row in enumerate(rows, 1):
            row["segment_id"] = f"segment_{index:04d}"
    for index in range(count):
        uid = f"clip-{index:03d}"
        sample = {**task.sample, "sample_id": uid, "clip_uid": uid,
            "references": [{**r, "image_path": f"reference-{p['image_index']}.png"}
                           for r, p in zip(task.sample["references"], task.pictures, strict=True)]}
        samples.append(sample)
        clips[uid] = {"canonical_audio_path": job.target_full_audio_path,
            "diarization": [{"start_time": s.start_time, "end_time": s.end_time, "speaker_label": "voice-a"}
                            for s in job.segments],
            "asr": [{"text": s.asr_text, "language": s.asr_language} for s in job.segments],
            "visual": visual, "joint": {"speech_av": speech, **profiles, "audio_finalize": finalized}}
    (directory / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    assert json.loads(marker.read_text())["sample_count"] > count
    fixtures = root / "cpu-fixtures.json"
    fixtures.write_text(json.dumps({"media_root": str(root), "clips": clips}))
    return source, fixtures


def test_twenty_clip_two_worker_cpu_pipeline_and_terminal_resume(tmp_path):
    from r2v_data_v2.h3.ra2va_group_local import run_cpu_pipeline
    source, fixtures = fixture_group(tmp_path)
    root = tmp_path / "run"
    summary = run_cpu_pipeline(source_root=source, run_root=root, fixtures_path=fixtures,
                               workers=2, limit=20, max_groups=1, ffmpeg=ffmpeg())
    assert len(summary["stages"]) == 8
    assert all(s["ready"] == 20 and s["failed"] == s["skipped"] == 0 for s in summary["stages"]), summary
    assert summary["model_call_count"] == 0 and summary["simulated_mimo_request_count"] == 40
    group = summary["exports"][0]
    assert group["product_ready_count"] == 60 and group["product_failed_count"] == 0
    assert set(group["task_counts"].values()) == {20}
    assert group["donor_count"] == 20 and group["donor_segment_count"] == 40
    bundle = root / "groups/group-000000/bundle"
    contract = json.loads((bundle / "source_contract.json").read_text())
    assert [j["clip_uid"] for j in contract["jobs"]] == [f"clip-{i:03d}" for i in range(20)]
    records = [json.loads(line) for line in (bundle / "records.jsonl").read_text().splitlines()]
    assert len(records) == 20 and all(r["execution"] == "cpu_fixture" for r in records)
    results = list((root / "groups/group-000000/stages").glob("*/results/*.json"))
    assert len(results) == 160
    assert len({json.loads(p.read_text())["worker_id"] for p in results}) == 2
    assert (root / "groups/group-000000/PILOT_COMPLETE").exists()
    assert not list(root.rglob("COMPLETE"))
    before = {str(p): p.read_bytes() for p in results}
    resumed = run_cpu_pipeline(source_root=source, run_root=root, fixtures_path=fixtures,
                               workers=2, limit=20, max_groups=1, ffmpeg="must-not-run")
    assert resumed["exports"] == summary["exports"]
    assert before == {str(p): p.read_bytes() for p in results}
    assert len(list(root.glob("groups/group-*"))) == 1


def test_local_pipeline_run_is_separate_from_old_fake_mode(tmp_path):
    from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
    GroupCoordinator(tmp_path / "run", tmp_path / "source", 20)
    with pytest.raises(ValueError, match="mode differs"):
        GroupCoordinator(tmp_path / "run", tmp_path / "source", 20, mode="cpu_pipeline_pilot")


def test_native_auk_worker_skips_only_legacy_source_digest(auk_setup, tmp_path):  # noqa: F811
    from r2v_data_v2.h3.ra2va_group_runtime import GroupAukBackend
    configuration = auk_setup.model_configuration
    source = auk_setup.jobs[0]
    native = SimpleNamespace(clip_uid="native", source_audio_path=source.source_audio_path,
                             source_duration_seconds=source.source_duration_seconds)
    with GroupAukBackend(configuration) as backend:
        result = backend.generate(native, tmp_path / "native.wav")
        legacy = backend.__class__.__mro__[1].generate(backend,
            SimpleNamespace(**vars(native), source_audio_sha256="0" * 64), tmp_path / "legacy.wav")
    assert result["status"] == "ok"
    assert legacy["status"] == "failed" and "hash differs" in legacy["reason"]
    assert (tmp_path / "native.wav").is_file() and not (tmp_path / "legacy.wav").exists()


def test_pipeline_semantic_failure_stays_terminal_and_is_not_exported(tmp_path):
    from r2v_data_v2.h3.ra2va_group_local import run_cpu_pipeline
    source, fixtures = fixture_group(tmp_path, count=2)
    data = json.loads(fixtures.read_text())
    data["clips"]["clip-000"]["asr"][0]["text"] = "Different frozen ASR."
    fixtures.write_text(json.dumps(data))
    kwargs = {"source_root": source, "run_root": tmp_path / "run", "fixtures_path": fixtures,
              "workers": 2, "limit": 2, "max_groups": 1, "ffmpeg": ffmpeg()}
    result = run_cpu_pipeline(**kwargs)
    mimo, exported = result["stages"][-2:]
    assert mimo["failed"] == 1 and mimo["ready"] == 1
    assert exported["skipped"] == 1 and exported["ready"] == 1
    assert result["exports"][0]["product_ready_count"] == 3
    assert result["exports"][0]["donor_count"] == 1
    saved = (tmp_path / "run/groups/group-000000/artifacts/mimo/clip-000/joint.response.json").read_bytes()
    resumed = run_cpu_pipeline(**{**kwargs, "ffmpeg": "must-not-run"})
    assert resumed["exports"] == result["exports"]
    assert saved == (tmp_path / "run/groups/group-000000/artifacts/mimo/clip-000/joint.response.json").read_bytes()


def test_native_auk_partial_output_can_be_reclaimed(auk_setup, tmp_path):  # noqa: F811
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    from r2v_data_v2.h3.ra2va_group_runtime import GroupAukBackend
    configuration, source = auk_setup.model_configuration, auk_setup.jobs[0]

    class InterruptedAuk(GroupAukBackend):
        def generate(self, job, raw_path):
            assert super().generate(job, raw_path)["status"] == "ok"
            raise SystemExit("after AuK raw, before stage publication")

    upstream = {"canonical": {"path": source.source_audio_path, "frame_count": 32000}}
    root, task = tmp_path / "stage", task_at(tmp_path)
    with (GroupStageWorker("auk", root, backend_factory=lambda _: InterruptedAuk(configuration), ffmpeg=ffmpeg()) as worker,
          pytest.raises(SystemExit)):
        worker.process(task, upstream)
    with GroupStageWorker("auk", root, backend_factory=lambda _: GroupAukBackend(configuration), ffmpeg=ffmpeg()) as worker:
        result = worker.process(task, upstream)
    assert result["response"]["status"] == "ok"
    assert result["speech"]["frame_count"] == 32000
    assert len(list((root / "auk/clip-a").glob("attempt-*/speech.raw.wav"))) == 2
