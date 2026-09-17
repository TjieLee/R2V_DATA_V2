from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from r2v_data_v2.v3.config import RuntimeGpuWorkersConfig
from r2v_data_v2.v3.schemas import (
    AnnotationState,
    BackgroundReferenceState,
    CoverageState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
)
from r2v_data_v2.v3.storage import RunStorage
from tests.test_v3_storage import (
    _clip_source,
    _config,
    _entity,
    _tracked_masks,
    _visibility_summary,
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)
    root = config.dataset_json.parent / "entity_mask"
    shard = root / "parts" / "shard-00000.jsonl"
    shard.parent.mkdir(parents=True)
    return config, root, shard


def _ready(case, uid="clip-0", index=0, background="none"):
    config, root, shard = case
    # Build frozen input using normal storage, then move the fixture under input.
    storage = RunStorage(replace(config, run_root=config.run_root.parent / uid))
    storage.initialize(git_commit="stage2")
    source = _clip_source(storage, uid).model_copy(update={"source_index": index})
    Path(source.video_path).parent.mkdir(parents=True, exist_ok=True)
    Path(source.video_path).write_bytes(b"target")
    storage.create_clip(clip_uid=uid, source=source)
    storage.write_annotation(
        uid,
        AnnotationState(
            status="ready",
            t2v_caption="A woman walks.",
            entities=[_entity("e1", "a woman")],
        ),
    )
    frames = SampledFramesArtifact(
        clip_uid=uid,
        width=5,
        height=4,
        frames=[
            SampledFrame(
                slot=i,
                source_frame_index=i,
                timestamp_seconds=float(i),
                image_path=f"frames/{i:02d}.jpg",
                sha256="0" * 64,
            )
            for i in range(10)
        ],
    )
    for frame in frames.frames:
        storage.frame_path(uid, frame.slot).write_bytes(b"not decoded or hashed")
    storage.frames_manifest_path(uid).write_text(frames.model_dump_json())
    masks = _tracked_masks(uid)
    first = masks.entities["e1"].frames[0]
    masks.entities["e1"].frames = [
        first.model_copy(update={"slot": i}) for i in range(10)
    ]
    storage.write_masks(uid, masks)
    storage.write_coverage(
        uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={"e1": _visibility_summary(10, qualifies=True)},
        ),
    )
    if background == "pending_remove":
        mask = storage.background_source_mask_path(uid, "a" * 64)
        mask.write_bytes(b"source mask")
        bg = BackgroundReferenceState(
            status=background,
            source_image_path=f"clips/{uid}/frames/00.jpg",
            source_frame_slot=0,
            source_frame_index=0,
            source_mask_path=storage.relative_artifact_path(mask),
            source_foreground_area_pixels=4,
            source_foreground_area_ratio=0.2,
        )
    else:
        bg = BackgroundReferenceState(status=background, reason="no valid background")
    storage.write_references(uid, ReferencesState(background=bg))
    extra = storage.clip_dir(uid) / "candidates" / "candidate.png"
    extra.parent.mkdir()
    extra.write_bytes(b"candidate asset")
    destination = root / "artifacts" / shard.stem / uid / "run"
    destination.parent.mkdir(parents=True)
    storage.root.rename(destination)
    status = {
        "none": "ready_no_background",
        "rejected": "ready_background_rejected",
        "pending_remove": "ready_background_pending_remove",
    }[background]
    return {
        "schema_version": "r2v.v3.pre_qwen_stage2.1",
        "source_index": index,
        "clip_uid": uid,
        "status": status,
        "coverage_passed": True,
        "background_status": background,
        "reason": None,
        "artifact_root": f"artifacts/{shard.stem}/{uid}",
    }


def _write_rows(case, rows):
    case[2].write_text("".join(json.dumps(row) + "\n" for row in rows))


def _start(case):
    api = importlib.import_module("r2v_data_v2.v3.post_mask_production")
    config, _root, shard = case
    paths = api.ShardPaths.for_shard(config.run_root.parent / "campaign", shard)
    storage = api.initialize_shard(config, paths, git_commit="first")
    return api, paths, storage


def _hydrate(case, api, paths, storage):
    return api.hydrate_shard(
        storage, entity_mask_root=case[1], shard_path=case[2], paths=paths
    )


@pytest.mark.parametrize("background", ["none", "rejected", "pending_remove"])
def test_ready_hydration_preserves_facts_and_assets_without_image_audit(
    case, background
):
    row = _ready(case, background=background)
    _write_rows(case, [row])
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.clip_uids, result.ready, result.excluded, result.corrupt) == (
        ("clip-0",),
        1,
        0,
        0,
    )
    source = case[1] / row["artifact_root"] / "run" / "clips" / "clip-0"
    for path in source.rglob("*"):
        if path.is_file():
            copied = storage.clip_dir("clip-0") / path.relative_to(source)
            assert copied.read_bytes() == path.read_bytes()
            assert copied.stat().st_ino != path.stat().st_ino
    storage.clip_path("clip-0").write_text(
        storage.clip_path("clip-0").read_text() + "\n"
    )
    assert (
        source.joinpath("clip.json").read_bytes()
        != storage.clip_path("clip-0").read_bytes()
    )


def test_excluded_rows_retain_identity_and_do_not_require_artifacts(case):
    rows = [
        {
            "source_index": i,
            "clip_uid": f"c{i}",
            "status": status,
            "reason": "upstream reason",
            "artifact_root": "missing",
        }
        for i, status in enumerate(
            [
                "coverage_rejected",
                "skipped_no_entities",
                "skipped_annotation_failed",
                "failed_input",
                "failed_frames",
            ]
        )
    ]
    _write_rows(case, rows)
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.ready, result.excluded, result.corrupt) == (0, 5, 0)
    records = [
        json.loads(line) for line in paths.exclusions_path.read_text().splitlines()
    ]
    assert [
        {key: record[key] for key in row} for record, row in zip(records, rows)
    ] == rows
    assert all(record["canonical_shard"] == str(case[2]) for record in records)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_frame",
        "bad_masks",
        "bad_clip",
        "wrong_identity",
        "unsafe_uid",
        "unsafe_artifact",
        "symlink",
        "missing_background_mask",
    ],
)
def test_corrupt_ready_clip_isolated_before_models(case, damage):
    bad = _ready(case, background="pending_remove")
    good = _ready(case, uid="clip-1", index=1)
    clip = case[1] / bad["artifact_root"] / "run" / "clips" / "clip-0"
    if damage == "missing_frame":
        (clip / "frames/00.jpg").unlink()
    elif damage == "bad_masks":
        (clip / "masks.rle.json").write_text("{}")
    elif damage == "bad_clip":
        (clip / "clip.json").write_text("{")
    elif damage == "wrong_identity":
        bad["source_index"] = 9
    elif damage == "unsafe_uid":
        bad["clip_uid"] = "../clip-1"
    elif damage == "unsafe_artifact":
        bad["artifact_root"] = "../outside"
    elif damage == "symlink":
        (clip / "frames/00.jpg").unlink()
        (clip / "frames/00.jpg").symlink_to(clip / "frames/01.jpg")
    else:
        next((clip / "background").glob("*.png")).unlink()
    _write_rows(case, [bad, good])
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.clip_uids, result.ready, result.corrupt) == (("clip-1",), 1, 1)
    first = paths.input_failures_path.read_bytes()
    _hydrate(case, api, paths, storage)
    assert paths.input_failures_path.read_bytes() == first


def test_duplicate_clip_identity_excludes_both_not_first_wins(case):
    row = _ready(case)
    _write_rows(case, [row, {**row, "source_index": 2}])
    api, paths, storage = _start(case)
    assert _hydrate(case, api, paths, storage).corrupt == 2
    assert not storage.clip_path("clip-0").exists()


def test_resume_preserves_destination_downstream_work_and_rebuilds_owned_staging(case):
    _write_rows(case, [_ready(case)])
    api, paths, storage = _start(case)
    partial = paths.staging_root / "clip-0"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("interrupted copy")
    _hydrate(case, api, paths, storage)
    assert not partial.exists()
    original = storage.clip_path("clip-0").read_text()
    storage.clip_path("clip-0").write_text(original + "\n")
    owned = storage.clip_dir("clip-0") / "downstream.txt"
    owned.write_text("keep")
    _hydrate(case, api, paths, storage)
    assert storage.clip_path("clip-0").read_text() == original + "\n"
    assert owned.read_text() == "keep"


def test_historical_commit_and_physical_remap_keep_identity(case):
    _write_rows(case, [_ready(case)])
    api, paths, storage = _start(case)
    cfg = case[0]
    moved = replace(
        cfg,
        sam3=replace(cfg.sam3, device="cuda:6"),
        remove=replace(cfg.remove, device="cuda:7"),
        reference_edit=replace(cfg.reference_edit, cuda_visible_devices="7"),
        runtime=replace(
            cfg.runtime, gpu_workers=RuntimeGpuWorkersConfig(segment="6", remove="7")
        ),
    )
    restarted = api.initialize_shard(moved, paths, git_commit="new-code")
    assert restarted.read_run() == storage.read_run()
    assert restarted.read_run().git_commit == "first"
    assert "r2v_v3_runs/production/jea_motion_v1/campaign/shard-00000" in str(
        paths.run_root
    )


@pytest.mark.parametrize("change", ["policy", "source", "model", "attributes", "shard"])
def test_restart_identity_mismatch_fails_closed(case, change):
    _write_rows(case, [_ready(case)])
    api, paths, storage = _start(case)
    cfg = case[0]
    if change == "policy":
        cfg = replace(cfg, coverage=replace(cfg.coverage, required_visible_frames=8))
    elif change == "source":
        cfg = replace(cfg, dataset_json=cfg.dataset_json.with_name("different.jsonl"))
    elif change == "model":
        cfg = replace(
            cfg,
            sam3=replace(
                cfg.sam3, model_path=cfg.sam3.model_path.with_name("other.pt")
            ),
        )
    elif change == "attributes":
        cfg = replace(
            cfg,
            subject_attributes=replace(
                cfg.subject_attributes,
                completion=replace(cfg.subject_attributes.completion, enabled=True),
            ),
        )
    else:
        case[2].write_text(case[2].read_text() + "\n")
    with pytest.raises(ValueError):
        api.initialize_shard(cfg, paths, git_commit="new")
    assert storage.read_run().git_commit == "first"


def test_enumeration_is_canonical_sorted_and_excludes_nonparts(case):
    _write_rows(case, [])
    (case[2].parent / "shard-00002.jsonl").write_text("")
    (case[1] / "shard-00001.jsonl").write_text("")
    api, _, _ = _start(case)
    assert [path.name for path in api.enumerate_shards(case[1])] == [
        "shard-00000.jsonl",
        "shard-00002.jsonl",
    ]


def test_destination_symlink_never_grants_hydration_access_to_frozen_source(case):
    row = _ready(case)
    _write_rows(case, [row])
    api, paths, storage = _start(case)
    source = case[1] / row["artifact_root"] / "run" / "clips" / "clip-0"
    before = (source / "clip.json").read_bytes()
    paths.staging_root.symlink_to(source, target_is_directory=True)
    result = _hydrate(case, api, paths, storage)
    assert result.corrupt == 1
    assert not (source / "clip-0").exists()
    assert (source / "clip.json").read_bytes() == before


def test_changed_source_manifest_after_hydration_does_not_overwrite_destination(case):
    row = _ready(case)
    _write_rows(case, [row])
    api, paths, storage = _start(case)
    _hydrate(case, api, paths, storage)
    before = storage.clip_path("clip-0").read_bytes()
    source = case[1] / row["artifact_root"] / "run" / "clips" / "clip-0/clip.json"
    source.write_text(source.read_text() + "\n")
    assert _hydrate(case, api, paths, storage).corrupt == 1
    assert storage.clip_path("clip-0").read_bytes() == before


def test_malformed_row_is_isolated_and_source_line_is_recorded(case):
    _write_rows(case, [_ready(case)])
    case[2].write_text("{bad\n" + case[2].read_text())
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.ready, result.corrupt) == (1, 1)
    failure = json.loads(paths.input_failures_path.read_text())
    assert failure["line_number"] == 1


def test_stage2_segment_pool_is_execution_only_for_post_mask(case):
    _write_rows(case, [_ready(case)])
    api, paths, storage = _start(case)
    cfg = case[0]
    parallel = replace(
        cfg,
        runtime=replace(
            cfg.runtime,
            stage_workers=replace(cfg.runtime.stage_workers, segment=2),
            gpu_workers=RuntimeGpuWorkersConfig(segment_pool=("4", "6")),
        ),
    )
    parallel.validate()
    restarted = api.initialize_shard(parallel, paths, git_commit="new-code")
    assert restarted.read_run() == storage.read_run()


def test_initializer_rejects_frozen_input_destination_before_any_write(
    case, monkeypatch
):
    _write_rows(case, [_ready(case)])
    api, _, _ = _start(case)
    # Even an input located under the writable filesystem is frozen by contract.
    monkeypatch.setattr(api.config_module, "ALLOWED_WRITABLE_ROOT", case[1].parent)
    paths = api.ShardPaths(
        case[2], case[1] / "new-run", case[1] / "new-export", case[1] / "new-state"
    )
    with pytest.raises(ValueError, match="frozen"):
        api.initialize_shard(case[0], paths, git_commit="first")
    assert not paths.run_root.exists()
    assert not paths.state_root.exists()


def test_ready_row_cannot_borrow_matching_identity_from_another_shard(case):
    row = _ready(case)
    source = case[1] / row["artifact_root"]
    other = case[1] / "artifacts/shard-99999/clip-0"
    other.parent.mkdir(parents=True)
    source.rename(other)
    row["artifact_root"] = "artifacts/shard-99999/clip-0"
    _write_rows(case, [row])
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.ready, result.corrupt) == (0, 1)
    assert not storage.clip_path("clip-0").exists()
    assert (
        "canonical shard"
        in json.loads(paths.input_failures_path.read_text())["failure_reason"]
    )


@pytest.mark.parametrize(
    "change", ["cpu", "qwen", "timeout", "stage_workers", "mode", "compile", "deferred"]
)
def test_execution_setting_remaps_preserve_shard_identity(case, change):
    _write_rows(case, [_ready(case)])
    api, paths, storage = _start(case)
    cfg = case[0]
    values = {
        "cpu": {"cpu_workers": 16},
        "qwen": {"qwen_max_inflight": 8},
        "timeout": {"worker_timeout_seconds": 7200},
        "stage_workers": {
            "stage_workers": replace(
                cfg.runtime.stage_workers, frames=7, pair=6, instruct=9
            )
        },
        "mode": {"mode": "streaming_v1"},
        "compile": {"sam3_compile_enabled": True},
        "deferred": {"subject_attributes_deferred": True},
    }
    remapped = replace(cfg, runtime=replace(cfg.runtime, **values[change]))
    restarted = api.initialize_shard(remapped, paths, git_commit="new-code")
    assert restarted.read_run() == storage.read_run()
    assert restarted.config.runtime == storage.config.runtime


def test_boolean_source_index_does_not_exclude_integer_neighbor(case):
    malformed = _ready(case)
    malformed["source_index"] = True
    good = _ready(case, uid="clip-1", index=1)
    _write_rows(case, [malformed, good])
    api, paths, storage = _start(case)
    result = _hydrate(case, api, paths, storage)
    assert (result.clip_uids, result.corrupt) == (("clip-1",), 1)
    assert not storage.clip_path("clip-0").exists()
    assert storage.read_clip("clip-1").source.source_index == 1
