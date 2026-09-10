from __future__ import annotations

import builtins
import io
import json
import os
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest
from PIL import Image

from r2v_data_v2.v3 import frames as frames_module
from r2v_data_v2.v3 import pre_qwen_production as stage2
from tests.test_v3_pre_qwen_production import (
    FakeBackend,
    FakeDecoder,
    _paths,
    _row,
    _write_shard,
)
from tools import entity_mask_recovery_plan as campaign
from tools import run_v3_entity_mask_auto as production
from tools import run_v3_entity_mask_recovery as recovery


def _fixture(tmp_path, monkeypatch, *, background=True, visible_slots=10, decoder=None):
    fixture = _paths(tmp_path, monkeypatch)
    paths = [
        _write_shard(
            fixture,
            [
                _row(fixture, start + offset, background=background)
                for offset in range(count)
            ],
            nominal_end=start + 9999,
        )
        for start, count in [(0, 2), (10_000, 1), (20_000, 1)]
    ]
    monkeypatch.setattr(
        production, "validate_entity_mask_production_config", lambda config: None
    )
    monkeypatch.setattr(recovery, "load_config_identity", lambda path: fixture.identity)
    stage2.ensure_execution_identity(fixture.output_root, chunk_rows=100)
    stage2.ensure_sam3_session_reuse_identity(fixture.output_root, mode="clip_reset_v1")
    stage2.write_json_atomic(
        production._static_topology_path(fixture.output_root),
        production.expected_static_topology(
            input_root=fixture.input_root,
            shard_paths=paths,
            world_size=5,
            local_gpu_count=8,
        ),
    )
    for path in paths[:2]:
        stage2.process_shard(
            path,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=FakeBackend(visible_slots=visible_slots),
            decoder=decoder or FakeDecoder(),
        )
    # Missing canonical shards are counted, not inspected or repaired.
    ignored = fixture.output_root / "artifacts" / paths[2].stem / "state.json"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("intentionally not a checkpoint")
    workspace = fixture.output_root / "artifacts" / paths[0].stem / "clip-1"
    _, storage = stage2._stage2_workspace_storage(
        fixture.config, workspace, output_root=fixture.output_root
    )
    return fixture, paths, workspace, storage


def _snapshot(root):
    return {
        str(path.relative_to(root)): (
            path.lstat().st_mode,
            path.lstat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def _forbid_writes_and_models(monkeypatch, *, no_images=False):
    def forbidden(*args, **kwargs):
        pytest.fail("canonical audit attempted a write, model, or recovery operation")

    for module, names in [
        (
            stage2,
            [
                "default_backend_factory",
                "build_sam3_segment_backend",
                "segment_clips",
                "process_shard",
                "write_json_atomic",
                "shard_lock",
            ],
        ),
        (production, ["check_stage2_candidate_judge_health"]),
        (campaign, ["build_plan", "obtain_plan", "weighted_assignment"]),
        (recovery, ["parse_gpus", "run_cluster_recovery", "run_workers"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for name in ["mkdir", "unlink", "rename", "replace", "touch", "chmod"]:
        monkeypatch.setattr(Path, name, forbidden)
    for module in (builtins, io):
        original = module.open

        def readonly_open(file, mode="r", *args, _original=original, **kwargs):
            assert not set(mode) & set("wax+"), "audit opened a writable file"
            if no_images and isinstance(file, (str, os.PathLike)):
                assert Path(file).suffix.lower() not in {".jpg", ".jpeg", ".png"}
            return _original(file, mode, *args, **kwargs)

        monkeypatch.setattr(module, "open", readonly_open)
    original_os_open = os.open

    def readonly_os_open(path, flags, *args, **kwargs):
        assert not flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", readonly_os_open)
    if no_images:
        monkeypatch.setattr(Image, "open", forbidden)
        monkeypatch.setattr(frames_module, "_sha256", forbidden)
        monkeypatch.setattr(frames_module, "validate_sampled_frames", forbidden)
        monkeypatch.setattr(stage2, "validate_sampled_frames", forbidden)
        monkeypatch.setattr(stage2, "_terminal_from_storage", forbidden)


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "clip",
        "masks",
        "background",
        "background_mask",
        "frames",
        "workspace",
        "checkpoint",
        "coverage",
        "row_count",
        "row_order",
        "row_provenance",
        "torn_tail",
    ],
)
@pytest.mark.parametrize("full", [False, True])
def test_canonical_audit_closure_is_read_only(
    tmp_path, monkeypatch, capsys, damage, full
):
    fixture, paths, workspace, storage = _fixture(tmp_path, monkeypatch)
    part = fixture.output_root / "parts" / paths[0].name
    if damage == "clip":
        storage.clip_path("clip-1").unlink()
    elif damage == "masks":
        storage.masks_path("clip-1").unlink()
    elif damage == "background":
        path = storage.clip_path("clip-1")
        payload = json.loads(path.read_text())
        payload["references"]["background"] = None
        path.write_text(json.dumps(payload))
    elif damage == "background_mask":
        background = storage.read_clip("clip-1").references.background
        assert background.source_mask_path is not None
        (storage.root / background.source_mask_path).unlink()
    elif damage == "frames":
        (storage.frames_dir("clip-1") / "00.jpg").unlink()
    elif damage == "workspace":
        workspace.rename(workspace.with_name("lost-workspace"))
    elif damage == "checkpoint":
        path = workspace / "state.json"
        payload = json.loads(path.read_text())
        payload["base_config_sha256"] = "f" * 64
        path.write_text(json.dumps(payload))
    elif damage == "coverage":
        path = storage.clip_path("clip-1")
        payload = json.loads(path.read_text())
        payload["coverage"] = None
        path.write_text(json.dumps(payload))
    elif damage == "row_count":
        part.write_bytes(part.read_bytes().splitlines(keepends=True)[0])
    elif damage == "row_order":
        part.write_bytes(
            b"".join(reversed(part.read_bytes().splitlines(keepends=True)))
        )
    elif damage == "row_provenance":
        values = [json.loads(line) for line in part.read_text().splitlines()]
        values[1]["input_row_sha256"] = "f" * 64
        part.write_text("".join(json.dumps(row) + "\n" for row in values))
    elif damage == "torn_tail":
        part.write_bytes(part.read_bytes().rstrip(b"\n"))
    # Neither an existing plan nor an absent plan should be touched by audit.
    plan = fixture.writable / "recovery-plan.json"
    if damage is not None:
        plan.write_text("not read or normalized by canonical audit")
    before = _snapshot(tmp_path)
    validations = []
    original_validate = stage2._validate_completed_shard

    def validate(path, **kwargs):
        validations.append((path.name, kwargs["validate_artifacts"]))
        return original_validate(path, **kwargs)

    monkeypatch.setattr(stage2, "_validate_completed_shard", validate)
    argv = [
        "--audit-canonical-full" if full else "--audit-canonical",
        "--audit-workers",
        "1" if full else "4",
        "--input-root",
        str(fixture.input_root),
        "--output-root",
        str(fixture.output_root),
        "--base-config",
        str(fixture.base_path),
        "--plan",
        str(plan),
        "--log-root",
        str(fixture.writable / "must-not-exist"),
    ]
    with monkeypatch.context() as guard:
        _forbid_writes_and_models(guard, no_images=not full)
        if damage is None:
            assert recovery.main(argv)["invalid_canonical"] == 0
        else:
            with pytest.raises(SystemExit) as exit_info:
                recovery.main(argv)
            assert exit_info.value.code == 1
    assert _snapshot(tmp_path) == before
    assert sorted(validations) == [(path.name, full) for path in paths[:2]]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    results = [event for event in events if event["event"] == "canonical_audit_shard"]
    summary = events[-1]
    assert summary["event"] == "canonical_audit_completed"
    assert summary["audit_mode"] == ("full" if full else "fast")
    assert summary["canonical_total"] == 2
    assert summary["missing_canonical"] == 1
    assert summary["valid_canonical"] == (2 if damage is None else 1)
    assert (
        summary["invalid_canonical"]
        == len(summary["invalid"])
        == int(damage is not None)
    )
    assert results[1]["status"] == "VALID_CANONICAL"  # Continue after a broken shard.
    if damage is not None:
        failure = summary["invalid"][0]
        assert failure["shard"] == paths[0].name
        assert failure["status"] == results[0]["status"] == "INVALID_CANONICAL"
        assert failure["exception_type"] and failure["message"]
        assert len(failure["message"]) <= 500 and "\n" not in failure["message"]
        if damage not in {"row_count", "row_order", "row_provenance", "torn_tail"}:
            assert failure["clip_uid"] == "clip-1"
            assert failure["source_index"] == 1
        else:
            assert "clip_uid" not in failure  # Do not guess a structurally invalid row.


@pytest.mark.parametrize("mode", ["--cluster-auto", "--single-shard"])
@pytest.mark.parametrize("audit", ["--audit-canonical", "--audit-canonical-full"])
def test_audit_mode_is_exclusive_with_recovery_modes(mode, audit):
    args = [audit, mode]
    if mode == "--single-shard":
        args.append("shard-000000000-000009999.jsonl")
    with pytest.raises(SystemExit) as exc:
        recovery.main(args)
    assert exc.value.code == 2


def test_orphan_canonical_is_reported_not_silently_omitted(tmp_path, monkeypatch):
    fixture, paths, _, _ = _fixture(tmp_path, monkeypatch)
    orphan = fixture.output_root / "parts" / "shard-000030000-000039999.jsonl"
    orphan.write_bytes((fixture.output_root / "parts" / paths[0].name).read_bytes())
    before = _snapshot(tmp_path)
    with monkeypatch.context() as guard:
        _forbid_writes_and_models(guard)
        report = recovery.audit_canonical(
            input_root=fixture.input_root,
            output_root=fixture.output_root,
            identity=fixture.identity,
        )
    assert report["canonical_total"] == 3
    assert report["valid_canonical"] == 2
    assert report["invalid_canonical"] == report["missing_canonical"] == 1
    assert report["invalid"][0]["shard"] == orphan.name
    assert "no matching completed Stage1 shard" in report["invalid"][0]["message"]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("image", ["sampled_jpeg", "background_mask"])
def test_fast_skips_image_content_but_full_still_detects_corruption(
    tmp_path, monkeypatch, image
):
    fixture, _, _, storage = _fixture(tmp_path, monkeypatch)
    path = storage.frames_dir("clip-1") / "00.jpg"
    if image == "background_mask":
        path = (
            storage.root
            / storage.read_clip("clip-1").references.background.source_mask_path
        )
    path.write_bytes(b"existing file with deliberately invalid image content")
    before = _snapshot(tmp_path)
    for full in (False, True):
        with monkeypatch.context() as guard:
            _forbid_writes_and_models(guard, no_images=not full)
            report = recovery.audit_canonical(
                input_root=fixture.input_root,
                output_root=fixture.output_root,
                identity=fixture.identity,
                full=full,
            )
        assert report["invalid_canonical"] == int(full)
        assert report["audit_workers"] == 16
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_run",
        "missing_state",
        "missing_frames_manifest",
        "invalid_frames_manifest",
        "invalid_masks_manifest",
        "invalid_clip",
        "run_identity",
        "clip_identity",
        "frames_identity",
        "frame_symlink",
        "background_frame",
        "workspace_swap",
    ],
)
def test_fast_metadata_existence_and_path_identity_fail_closed(
    tmp_path, monkeypatch, damage
):
    fixture, paths, workspace, storage = _fixture(tmp_path, monkeypatch)
    targets = {
        "missing_run": storage.run_path,
        "missing_state": workspace / "state.json",
        "missing_frames_manifest": storage.frames_manifest_path("clip-1"),
        "invalid_frames_manifest": storage.frames_manifest_path("clip-1"),
        "invalid_masks_manifest": storage.masks_path("clip-1"),
        "invalid_clip": storage.clip_path("clip-1"),
    }
    if damage.startswith("missing_"):
        targets[damage].unlink()
    elif damage.startswith("invalid_"):
        targets[damage].write_text("not valid JSON")
    elif damage == "frame_symlink":
        frame = storage.frames_dir("clip-1") / "00.jpg"
        frame.unlink()
        peer = fixture.output_root / "artifacts" / paths[0].stem / "clip-0"
        frame.symlink_to(peer / "run" / "clips" / "clip-0" / "frames" / "00.jpg")
    elif damage == "workspace_swap":
        part = fixture.output_root / "parts" / paths[0].name
        rows = [json.loads(line) for line in part.read_text().splitlines()]
        rows[1]["artifact_root"] = rows[0]["artifact_root"]
        part.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = storage.clip_path("clip-1")
        if damage == "run_identity":
            path = storage.run_path
        elif damage == "frames_identity":
            path = storage.frames_manifest_path("clip-1")
        payload = json.loads(path.read_text())
        if damage == "run_identity":
            payload["config_hash"] = "wrong identity"
        elif damage == "background_frame":
            payload["references"]["background"]["source_frame_index"] += 1
        else:
            payload["clip_uid"] = "wrong-clip"
        path.write_text(json.dumps(payload))
    before = _snapshot(tmp_path)
    with monkeypatch.context() as guard:
        _forbid_writes_and_models(guard, no_images=True)
        report = recovery.audit_canonical(
            input_root=fixture.input_root,
            output_root=fixture.output_root,
            identity=fixture.identity,
        )
    assert report["invalid_canonical"] == report["valid_canonical"] == 1
    assert report["invalid"][0]["clip_uid"] == "clip-1"
    assert _snapshot(tmp_path) == before


def test_fast_shard_parallelism_preserves_result_and_failure_order(
    tmp_path, monkeypatch, capsys
):
    fixture, paths, _, _ = _fixture(tmp_path, monkeypatch)
    barrier, second_done, lock = Barrier(2), Event(), Lock()
    active = peak = 0
    completed = []
    original = stage2._validate_completed_shard

    def validate(path, **kwargs):
        nonlocal active, peak
        assert kwargs["validate_artifacts"] is False
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=5)  # A serial implementation cannot pass.
            original(path, **kwargs)
            if path.name == paths[0].name:
                assert second_done.wait(timeout=5)
            with lock:
                completed.append(path.name)
            if path.name == paths[1].name:
                second_done.set()
            raise ValueError("synthetic shard-level audit error")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(stage2, "_validate_completed_shard", validate)
    with monkeypatch.context() as guard:
        _forbid_writes_and_models(guard, no_images=True)
        report = recovery.audit_canonical(
            input_root=fixture.input_root,
            output_root=fixture.output_root,
            identity=fixture.identity,
            workers=2,
        )
    assert peak == 2
    assert completed == [paths[1].name, paths[0].name]
    assert [item["shard"] for item in report["invalid"]] == [p.name for p in paths[:2]]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        item["shard"] for item in events if item["event"] == "canonical_audit_shard"
    ] == [path.name for path in paths[:2]]


@pytest.mark.parametrize("workers", [0, -1])
def test_audit_workers_must_be_positive(workers):
    with pytest.raises(ValueError, match="audit-workers must be positive"):
        recovery.audit_canonical(
            input_root=Path("unused"),
            output_root=Path("unused"),
            identity=None,
            workers=workers,
        )


def test_fast_and_full_cli_modes_are_exclusive():
    with pytest.raises(SystemExit) as exc:
        recovery.main(["--audit-canonical", "--audit-canonical-full"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "status", ["ready_no_background", "coverage_rejected", "failed_frames"]
)
def test_fast_preserves_legitimate_terminal_closures(tmp_path, monkeypatch, status):
    class BrokenFrames(FakeDecoder):
        def inspect(self, video_path):
            raise ValueError("insufficient usable frames")

    fixture, paths, _, _ = _fixture(
        tmp_path,
        monkeypatch,
        background=status != "ready_no_background",
        visible_slots=1 if status == "coverage_rejected" else 10,
        decoder=BrokenFrames() if status == "failed_frames" else None,
    )
    assert (
        json.loads(
            (fixture.output_root / "parts" / paths[0].name).read_text().splitlines()[0]
        )["status"]
        == status
    )
    with monkeypatch.context() as guard:
        _forbid_writes_and_models(guard, no_images=True)
        report = recovery.audit_canonical(
            input_root=fixture.input_root,
            output_root=fixture.output_root,
            identity=fixture.identity,
        )
    assert report["valid_canonical"] == 2 and report["invalid_canonical"] == 0
