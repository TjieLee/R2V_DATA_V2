from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import ra2va_production as production


def _fake_runner(calls: list[str], *, bad: set[str] = frozenset(), interrupt: set[str] = frozenset()):
    def run(uid: str, destination: Path) -> dict:
        calls.append(uid)
        if uid in interrupt:
            raise KeyboardInterrupt
        destination.mkdir(parents=True)
        record = {
            "clip_uid": uid,
            "status": "failed" if uid in bad else "ready",
            "failure_code": "semantic_failure" if uid in bad else None,
            "failure_reason": "invalid annotation" if uid in bad else None,
            "annotation": None if uid in bad else {"clip_uid": uid},
        }
        (destination / "records.jsonl").write_text(json.dumps(record) + "\n")
        return record
    return run


def test_ra2va_shard_ownership_reuses_t2va_schedule():
    orders = [production.shard_order(start=0, end=19, seed=84, rank=r, world_size=4) for r in range(4)]
    assert all(not set(left).intersection(right) for i, left in enumerate(orders) for right in orders[i + 1:])
    assert set().union(*map(set, orders)) == set(range(20))
    assert production.shard_order(start=0, end=19, seed=84, rank=1, world_size=4) == orders[1]


def test_ra2va_semantic_failure_is_per_clip_and_ready_annotation_is_durable(tmp_path):
    calls: list[str] = []
    order = ["a", "b", "c"]
    states = production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=order,
        run_clip=_fake_runner(calls, bad={"b"}), request_workers=1,
    )
    assert calls == order
    assert [states[uid]["status"] for uid in order] == ["ready", "failed", "ready"]
    assert [row["clip_uid"] for row in production.ready_reconcile_records(tmp_path, 0)] == ["a", "c"]
    calls.clear()
    states = production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=order,
        run_clip=_fake_runner(calls), request_workers=1,
    )
    assert calls == ["b"]
    assert all(states[uid]["status"] == "ready" for uid in order)
    assert len(production.ready_reconcile_records(tmp_path, 0)) == 3


def test_ra2va_interruption_recovers_published_clip_without_second_call(tmp_path):
    calls: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        production.process_reconcile_shard(
            root=tmp_path, shard_id=0, clip_uids=["a", "b", "c"],
            run_clip=_fake_runner(calls, interrupt={"b"}), request_workers=1,
        )
    assert calls == ["a", "b"]
    calls.clear()
    states = production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b", "c"],
        run_clip=_fake_runner(calls), request_workers=1,
    )
    assert calls == ["b", "c"]
    assert all(states[uid]["status"] == "ready" for uid in ("a", "b", "c"))


def test_ra2va_upstream_skip_never_calls_model(tmp_path):
    calls: list[str] = []
    states = production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b"],
        run_clip=_fake_runner(calls), skipped={"a": "upstream_failed"}, request_workers=1,
    )
    assert calls == ["b"]
    assert states["a"]["status"] == "skipped"
    assert states["a"]["failure_reason"] == "upstream_failed"


def test_ra2va_adapter_reuses_frozen_reconcile_result(tmp_path, monkeypatch):
    from r2v_data_v2.h3.audio_reuse_prepared import load_reconcile_sources
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
    from tests.test_h3_audio_shadow_qa import _fixture
    from tests.test_h3_mimo25_raw_stems import _backend, _raw

    _, shadow = _fixture(tmp_path, monkeypatch)
    backend_a, _, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    backend_b, _, _, _ = _backend(tmp_path, shadow, [(_raw(), 8)])
    direct_root = shadow / "direct-one"
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs[:1], stem_records=stems[:1], backend=backend_a,
        output_root=direct_root, source_clip_uids=[jobs[0].clip_uid],
        route="music_first", allow_unverified=True,
    )
    expected = json.loads((direct_root / "records.jsonl").read_text())
    actual = production.run_reconcile_clip(
        job=jobs[0], stem_record=stems[0], backend=backend_b,
        destination=shadow / "production-one", route="music_first",
        allow_unverified=True,
    )
    assert actual["annotation"] == expected["annotation"]
    assert actual["status"] == expected["status"]
    assert actual["status"] == "ready"

    def published_record(_uid: str, destination: Path) -> dict:
        shutil.copytree(direct_root, destination)
        return expected

    production_root = tmp_path / "production"
    production.process_reconcile_shard(
        root=production_root, shard_id=0, clip_uids=[jobs[0].clip_uid],
        run_clip=published_record, request_workers=1,
    )
    ready_view = tmp_path / "ready-view"
    production.publish_ready_reconcile_view(production_root, ready_view, [0])
    downstream = load_reconcile_sources(ready_view)
    assert len(downstream) == 1
    assert downstream[0].clip_uid == jobs[0].clip_uid
    assert downstream[0].annotation.model_dump(mode="json") == expected["annotation"]


def test_ra2va_production_dry_run_uses_frozen_inputs_without_video_hashing(tmp_path, monkeypatch):
    from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
    from r2v_data_v2.h3.resolved_audio_stems import downstream_stem_root
    from tools import run_h3_ra2va_production as cli

    manifest = tmp_path / "cases.json"
    manifest.write_text(MimoCaseManifest(clip_uids=["a", "b"]).model_dump_json())
    audio_root = tmp_path / "audio-production"
    stem_root = downstream_stem_root(audio_root, "pilot")
    monkeypatch.setattr(cli, "load_stem_source", lambda path: (
        SimpleNamespace(clip_uids=["a", "b"]), [], None,
    ))
    monkeypatch.setattr(cli, "validate_stem_diarization_lineage", lambda *args, **kwargs: (
        SimpleNamespace(binding_evidence_mode="none", source_stem_root=str(stem_root),
                        skipped_clips=[], diarization_failed_clips=[]), None, None,
    ))

    def build_reference_inventory(**kwargs):
        assert kwargs["verify_video"] is False
        assert kwargs["verify_audio"] is False
        assert kwargs["case_manifest"].clip_uids == ["a", "b"]
        return object()

    monkeypatch.setattr(cli, "build_mimo25_reference_inventory", build_reference_inventory)
    monkeypatch.setattr(cli, "usable_stem_reconcile_inventory", lambda base, provenance: base)
    monkeypatch.setattr(cli, "build_stem_reconcile_jobs", lambda **kwargs: [
        SimpleNamespace(clip_uid="a"), SimpleNamespace(clip_uid="b"),
    ])
    args = [
        "--visual-production-root", str(tmp_path / "visual"),
        "--visual-runs-root", str(tmp_path / "visual-runs"),
        "--audio-production-root", str(audio_root),
        "--production-root", str(tmp_path / "output"),
        "--media-root", str(tmp_path), "--shadow-run-id", "pilot",
        "--case-manifest", str(manifest), "--mimo-model", "mimo-v2.6-flash-rl",
        "--mimo-call-mode", "single", "--mimo-gpu-groups", "0,1,2,3;4,5,6,7",
        "--mimo-ports", "8094,8095", "--dry-run",
    ]
    result = cli.main(args)
    assert result["shards"] == [0]
    assert result["request_workers"] == 2
    assert result["model_called"] is False
    assert not (tmp_path / "output").exists()
    assert result["single_contract"] == "compact2"
    assert cli.main([*args, "--single-contract", "compact3"])["single_contract"] == "compact3"
    default = cli.main(args[:args.index("--mimo-model")] + ["--dry-run"])
    assert (default["model"], default["call_mode"], default["request_workers"], default["endpoints"]) == (
        "mimo-v2.5", "multi", 1, [],
    )
    assert "single_contract" not in default
    with pytest.raises(ValueError, match="single-call"):
        cli.main(args[:args.index("--mimo-model")] + ["--dry-run", "--single-contract", "compact3"])


def test_ra2va_endpoint_failure_stops_new_work_and_resumes(tmp_path):
    from r2v_data_v2.h3.t2va_production import EndpointUnavailable

    calls: list[str] = []

    def unavailable(uid: str, destination: Path) -> dict:
        calls.append(uid)
        if uid == "b":
            raise ConnectionError("connection refused")
        return _fake_runner([])(uid, destination)

    with pytest.raises(EndpointUnavailable):
        production.process_reconcile_shard(
            root=tmp_path, shard_id=0, clip_uids=["a", "b", "c"],
            run_clip=unavailable, request_workers=1,
        )
    assert calls == ["a", "b"]
    calls.clear()
    production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b", "c"],
        run_clip=_fake_runner(calls), request_workers=1,
    )
    assert calls == ["b", "c"]


def test_ra2va_stage_startup_failure_is_infrastructure_not_clip_failure(tmp_path):
    from r2v_data_v2.h3.t2va_production import EndpointUnavailable

    @contextmanager
    def broken_stage(_shard):
        raise ConnectionError("connection refused: MiMo stage port busy")
        yield

    calls: list[str] = []
    with pytest.raises(EndpointUnavailable):
        production.process_reconcile_shard(
            root=tmp_path, shard_id=0, clip_uids=["a", "b"],
            run_clip=_fake_runner(calls), stage=broken_stage, request_workers=1,
        )
    assert calls == []
    assert production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b"],
        run_clip=_fake_runner(calls), request_workers=1,
    )["a"]["status"] == "ready"


def test_ra2va_two_workers_overlap_inside_one_mimo_stage(tmp_path):
    barrier = Barrier(2, timeout=2)
    stages: list[str] = []

    @contextmanager
    def stage(_shard):
        stages.append("start")
        yield
        stages.append("stop")

    def run(uid: str, destination: Path) -> dict:
        barrier.wait()
        return _fake_runner([])(uid, destination)

    states = production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b"],
        run_clip=run, request_workers=2, stage=stage,
    )
    assert stages == ["start", "stop"]
    assert all(item["status"] == "ready" for item in states.values())


def test_ra2va_ready_view_reuses_one_annotation_for_downstream_products(tmp_path):
    from tools.build_h3_ra2va_reconcile_view import main as export_main

    calls: list[str] = []
    production.process_reconcile_shard(
        root=tmp_path, shard_id=0, clip_uids=["a", "b", "c"],
        run_clip=_fake_runner(calls, bad={"b"}), request_workers=1,
    )
    view = tmp_path / "ready-view"
    production.publish_ready_reconcile_view(tmp_path, view, [0])
    assert [json.loads(line)["clip_uid"] for line in (view / "records.jsonl").read_text().splitlines()] == ["a", "c"]
    assert calls == ["a", "b", "c"]
    with pytest.raises(FileExistsError):
        production.publish_ready_reconcile_view(tmp_path, view, [0])
    via_cli = export_main([
        "--production-root", str(tmp_path), "--output-root", str(tmp_path / "ready-view-cli"),
        "--shards", "0",
    ])
    assert via_cli["ready_count"] == 2
    assert (tmp_path / "ready-view-cli/records.jsonl").read_bytes() == (view / "records.jsonl").read_bytes()
