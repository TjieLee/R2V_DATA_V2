#!/usr/bin/env python3
"""Static whole-shard Post-Mask cluster launcher; no Stage2 generation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND, load_config
from r2v_data_v2.v3.post_mask_production import (
    ShardPaths,
    _semantic_identity,
    enumerate_shards,
    prepare_shard_config,
)
from r2v_data_v2.v3.production_source import JEA_VIDEO_MOTION_ADAPTER

SOURCE_ROOT = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed")
EXPECTED_PRODUCTION_SHARDS = 384


def emit(event):
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_gpu_groups(value):
    groups, used = [], set()
    for item in value.split(","):
        if not re.fullmatch(r"[0-9]+:[0-9]+", item):
            raise ValueError("GPU groups must be numeric SAM:Boogu pairs")
        pair = tuple(str(int(gpu)) for gpu in item.split(":"))
        if pair[0] == pair[1] or any(gpu in used for gpu in pair):
            raise ValueError("GPU groups must not overlap")
        used.update(pair)
        groups.append(pair)
    return tuple(groups)


def assigned_shards(shards, rank, world_size, slot, local_group_count):
    if (
        world_size < 1
        or not 0 <= rank < world_size
        or not 0 <= slot < local_group_count
    ):
        raise ValueError("invalid rank/world size/local slot")
    return tuple(
        shards[rank * local_group_count + slot :: world_size * local_group_count]
    )


@contextmanager
def exclusive_lock(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_production_config(config):
    if not (
        config.remove.enabled
        and config.remove.backend == BOOGU_REMOVE_BACKEND
        and config.remove.inference_profile == "boogu_4step_v1"
    ):
        raise ValueError(
            "accepted production base requires remove.enabled=true, Boogu backend and boogu_4step_v1"
        )
    if config.subject_attribute_gme.enabled:
        raise ValueError("accepted production base must disable GME")
    for section in ("pair", "reference_edit", "reference_integrity", "instruction"):
        if not getattr(config, section).enabled:
            raise ValueError(
                f"accepted production base requires {section}.enabled=true"
            )
    config.validate()


@dataclass(frozen=True)
class Campaign:
    root: Path
    entity_mask_root: Path
    shards: tuple[Path, ...]
    runs_root: Path
    exports_root: Path
    source_yaml: Path
    source_jsonl: Path
    identity: dict

    def paths(self, shard):
        return ShardPaths.for_shard(
            self.root,
            shard,
            runs_root=self.runs_root,
            exports_root=self.exports_root / "shards",
        )

    @property
    def compact_marker(self):
        return self.root / "compacted.json"


def prepare_campaign(
    config,
    *,
    base_config_path,
    entity_mask_root,
    post_mask_root,
    source_jsonl,
    clips_root,
):
    validate_production_config(config)
    root = Path(post_mask_root).resolve()
    frozen = Path(entity_mask_root).resolve(strict=True)
    writable = config_module.ALLOWED_WRITABLE_ROOT.resolve()
    source = Path(source_jsonl).resolve(strict=True)
    if config.dataset_json.resolve() != source:
        raise ValueError("config.dataset_json must match campaign source_jsonl")
    clips = Path(clips_root).resolve(strict=True)
    dataset = config_module.ALLOWED_DATASET_ROOT.resolve()
    if (
        not root.is_relative_to(writable)
        or root == writable
        or root.is_relative_to(frozen)
        or frozen.is_relative_to(root)
    ):
        raise ValueError(
            "Post-Mask state root must be writable and disjoint from frozen input"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", root.name):
        raise ValueError("invalid campaign tag")
    if not (
        source.is_file()
        and source.suffix == ".jsonl"
        and source.is_relative_to(dataset)
        and clips.is_dir()
        and clips != dataset
        and clips.is_relative_to(dataset)
    ):
        raise ValueError("source JSONL and clips root must be public dataset inputs")
    shards = enumerate_shards(frozen)
    stage1 = {p.name for p in enumerate_shards(frozen.parent / "entity_annotations")}
    stage2 = {p.name for p in shards}
    if (
        len(stage1) != EXPECTED_PRODUCTION_SHARDS
        or len(stage2) != EXPECTED_PRODUCTION_SHARDS
        or stage1 != stage2
    ):
        raise ValueError(
            f"expected Stage1/Stage2 canonical shard closure {EXPECTED_PRODUCTION_SHARDS}/{EXPECTED_PRODUCTION_SHARDS}; "
            f"stage1={len(stage1)}, stage2={len(stage2)}, "
            f"missing={sorted(stage1 - stage2)[:5]}, extra={sorted(stage2 - stage1)[:5]}"
        )
    tag = root.name
    runs = writable / "r2v_v3_runs/production/jea_motion_v1" / tag
    exports = writable / "r2v_v3_exports/production/jea_motion_v1" / tag
    source_yaml = (
        writable / "r2v_v3_configs/production/jea_motion_v1" / tag / "source.yaml"
    )
    identities = {}
    for shard in shards:
        paths = ShardPaths.for_shard(
            root, shard, runs_root=runs, exports_root=exports / "shards"
        )
        identities[shard.stem] = _semantic_identity(
            prepare_shard_config(config, paths), paths
        )
    identity = {
        "entity_mask_root": str(frozen),
        "source_jsonl": str(source),
        "clips_root": str(clips),
        "runs_root": str(runs),
        "exports_root": str(exports),
        "shards": identities,
    }
    # The shared config-path lock also prevents two distinct state roots with the
    # same tag from independently claiming the same run/export destination.
    with exclusive_lock(source_yaml.with_suffix(".lock")):
        receipt = root / "campaign.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            # Retain legacy campaign bindings (and global markers) without
            # re-reading source bodies. Per-shard resume still checks bytes.
            semantic = {**saved, "shards": {}}
            for name, item in saved["shards"].items():
                if "shard_sha256" in item and not re.fullmatch(
                    r"[a-f0-9]{64}", str(item["shard_sha256"])
                ):
                    raise ValueError("Post-Mask campaign shard identity mismatch")
                semantic["shards"][name] = {
                    k: v for k, v in item.items() if k != "shard_sha256"
                }
            if semantic != identity:
                raise ValueError("Post-Mask campaign identity mismatch")
            identity = saved
        binding = {
            "campaign_sha256": digest(identity),
            "post_mask_root": str(root),
            "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
            "source_jsonl": str(source),
            "clips_root": str(clips),
        }
        if source_yaml.exists():
            import yaml

            descriptor = yaml.safe_load(source_yaml.read_text())
            if not isinstance(descriptor, dict) or any(
                descriptor.get(k) != v for k, v in binding.items()
            ):
                raise ValueError("Post-Mask source descriptor identity mismatch")
        else:
            write_json_atomic(
                source_yaml,
                {
                    **binding,
                    "base_config_path": str(Path(base_config_path).resolve()),
                    "base_config_sha256": file_digest(base_config_path),
                    "base_config_fingerprint": load_config(
                        base_config_path
                    ).fingerprint(),
                },
            )
        write_json_atomic(receipt, identity)
    return Campaign(root, frozen, shards, runs, exports, source_yaml, source, identity)


def completed_receipt(campaign, shard):
    """Small receipt checks only: never walk/hydrate/decode Stage2 on a barrier."""
    paths = campaign.paths(shard)
    marker = paths.state_root / "completed.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    identity = value.get("identity", {})
    shard_identity = identity.get("shard_identity", {})
    expected = campaign.identity["shards"][shard.stem]
    if (
        not isinstance(shard_identity, dict)
        or not re.fullmatch(r"[a-f0-9]{64}", str(shard_identity.get("shard_sha256")))
        or {k: v for k, v in shard_identity.items() if k != "shard_sha256"}
        != {k: v for k, v in expected.items() if k != "shard_sha256"}
        or ("shard_sha256" in expected and shard_identity != expected)
    ):
        raise ValueError("completed shard identity mismatch")
    if not isinstance(identity.get("clip_count"), int) or not re.fullmatch(
        r"[a-f0-9]{64}", str(identity.get("clip_sources_sha256"))
    ):
        raise ValueError("invalid completed shard source receipt")
    dataset = json.loads((paths.export_root / "dataset.json").read_text())
    run = json.loads((paths.run_root / "run.json").read_text())
    if (
        json.loads(paths.identity_path.read_text()) != identity["shard_identity"]
        or json.loads((paths.state_root / "export_identity.json").read_text())
        != identity
        or dataset.get("git_commit") != identity.get("git_commit")
        or run.get("git_commit") != identity.get("git_commit")
        or dataset.get("config_hash") != identity["shard_identity"]["config_hash"]
        or dataset.get("sample_count") != value.get("sample_count")
        or not (paths.export_root / "samples.jsonl").is_file()
    ):
        raise ValueError("completed shard export identity mismatch")
    return value


@contextmanager
def heartbeat(state, callback, seconds=45):
    stop = threading.Event()

    def tick():
        while not stop.wait(seconds):
            callback(
                {
                    **(state() if callable(state) else state),
                    "event": "post_mask_heartbeat",
                }
            )

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


@contextmanager
def _worker_model_resources(config, execution, root, event):
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    resources = PostMaskWorkerResources(config, execution, runtime_root=root)
    try:
        with resources:
            yield resources
    finally:
        event({"event": "post_mask_worker_resource_summary", **resources.summary()})


def run_worker(
    campaign,
    config,
    *,
    runtime,
    rank,
    world_size,
    slot,
    groups,
    git_commit,
    runner=None,
    event_callback=emit,
    heartbeat_seconds=45,
    clip_inflight=8,
    shard_inflight=2,
    scheduler_mode="wavefront_v2",
):
    sam_gpu, boogu_gpu = groups[slot]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != sam_gpu:
        raise ValueError("worker requires SAM GPU isolation before imports")
    from r2v_data_v2.v3.post_mask_runtime import ExecutionSettings, run_post_mask_shard
    from r2v_data_v2.v3.post_mask_wavefront import bounded_results

    runner = runner or run_post_mask_shard
    assigned = assigned_shards(campaign.shards, rank, world_size, slot, len(groups))
    node = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:24]
    execution = ExecutionSettings(
        runtime,
        sam_gpu,
        boogu_gpu,
        campaign.root / "qwen-gates" / node,
        clip_inflight,
        shard_inflight,
        scheduler_mode,
    )
    state = {
        "worker_id": rank * len(groups) + slot,
        "total_shards": len(campaign.shards),
        "assigned_shards": len(assigned),
        "completed_shards": 0,
        "ready": 0,
        "excluded": 0,
        "corrupt": 0,
        "current_shard": None,
        "current_stage": None,
    }
    state_lock = threading.RLock()
    progress = {}
    unresolved_shards = []
    failures = []

    def snapshot():
        with state_lock:
            return {
                **state,
                "active_shards": {
                    name: value["stage"]
                    for name, value in progress.items()
                    if value["active"]
                },
            }

    def update(shard, *, stage=None, counts=None, active=None):
        with state_lock:
            item = progress.setdefault(
                shard.stem,
                {
                    "stage": "waiting_for_shard_lock",
                    "active": True,
                    "ready": 0,
                    "excluded": 0,
                    "corrupt": 0,
                },
            )
            if stage is not None:
                item["stage"] = stage
            if active is not None:
                item["active"] = active
            if counts is not None:
                item.update(counts)
            for key in ("ready", "excluded", "corrupt"):
                state[key] = sum(value[key] for value in progress.values())
            state.update(current_shard=shard.stem, current_stage=item["stage"])

    def event(value):
        with state_lock:
            event_callback({**snapshot(), **value})

    event(
        {
            "event": "post_mask_worker_started",
            "sam_gpu": sam_gpu,
            "boogu_gpu": boogu_gpu,
        }
    )
    runtime_root = (
        campaign.root
        / "runtime"
        / node
        / f"worker-{rank * len(groups) + slot}-{uuid.uuid4().hex}"
    )
    with (
        _worker_model_resources(config, execution, runtime_root, event) as resources,
        heartbeat(snapshot, event_callback, heartbeat_seconds),
    ):

        def run_shard(shard):
            paths = campaign.paths(shard)
            update(shard)

            def shard_event(value):
                with state_lock:
                    counts = (
                        {key: value[key] for key in ("ready", "excluded", "corrupt")}
                        if value["event"] == "post_mask_hydrate_completed"
                        else None
                    )
                    update(shard, stage=value.get("stage"), counts=counts)
                    event(
                        {
                            **value,
                            "current_shard": shard.stem,
                            "current_stage": progress[shard.stem]["stage"],
                        }
                    )

            try:
                with resources.activity("shards"):
                    while not resources.stopping:
                        with ExitStack() as locks:
                            try:
                                locks.enter_context(
                                    exclusive_lock(
                                        paths.state_root / "shard.lock", blocking=False
                                    )
                                )
                            except BlockingIOError:
                                # A blocking flock in a pool thread cannot receive
                                # the main thread's SIGTERM/KeyboardInterrupt.
                                time.sleep(0.1)
                                continue
                            if resources.stopping:
                                return
                            return execute_shard(shard, paths, shard_event)
            except Exception as exc:  # noqa: BLE001 - isolate shard infrastructure failures
                with state_lock:
                    failures.append((shard.stem, exc))
                shard_event(
                    {
                        "event": "post_mask_shard_failed",
                        "shard": shard.stem,
                        "reason": str(exc),
                    }
                )
            finally:
                update(shard, active=False)

        def execute_shard(shard, paths, shard_event):
            receipt = completed_receipt(campaign, shard)
            if receipt is None:
                legacy_hash = campaign.identity["shards"][shard.stem].get(
                    "shard_sha256"
                )
                if legacy_hash is not None and file_digest(shard) != legacy_hash:
                    raise ValueError("Post-Mask legacy shard identity mismatch")
                update(shard, stage="hydrate")
                result = runner(
                    config,
                    entity_mask_root=campaign.entity_mask_root,
                    paths=paths,
                    git_commit=git_commit,
                    execution=execution,
                    event_callback=shard_event,
                    resources=resources,
                )
                update(
                    shard,
                    counts={
                        key: getattr(result, key)
                        for key in ("ready", "excluded", "corrupt")
                    },
                )
                if not result.completed:
                    with state_lock:
                        unresolved_shards.append(shard.stem)
                    shard_event(
                        {
                            "event": "post_mask_shard_incomplete",
                            "shard": shard.stem,
                            "retryable_clip_uids": list(result.retryable_clip_uids),
                        }
                    )
                    return
                if completed_receipt(campaign, shard) is None:
                    raise RuntimeError(
                        "worker returned without durable shard completion"
                    )
            else:
                counts = {"ready": receipt["identity"]["clip_count"]}
                for key, inventory in (
                    ("excluded", paths.exclusions_path),
                    ("corrupt", paths.input_failures_path),
                ):
                    with inventory.open() as handle:
                        counts[key] = sum(bool(line.strip()) for line in handle)
                update(shard, counts=counts)
                shard_event(
                    {
                        "event": "post_mask_shard_completed",
                        "shard": shard.stem,
                        "skipped": True,
                    }
                )
            with state_lock:
                state["completed_shards"] += 1

        for _ in bounded_results(
            run_shard,
            assigned,
            1
            if execution.scheduler_mode == "legacy_serial"
            else execution.shard_inflight,
            on_interrupt=resources.stop_accepting,
        ):
            pass
        with state_lock:
            state.update(current_shard=None, current_stage=None)
        if unresolved_shards or failures:
            event(
                {
                    "event": "post_mask_worker_incomplete",
                    "unresolved_shards": sorted(unresolved_shards),
                    "failed_shards": sorted(name for name, _ in failures),
                }
            )
            if failures:
                raise failures[0][1]
            raise RuntimeError(
                f"Post-Mask shards have retryable work: {', '.join(unresolved_shards)}"
            )
        event({"event": "post_mask_worker_completed"})


def _validate_compact(campaign, receipts):
    catalog_path = campaign.exports_root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entries = catalog.get("included_shards", [])
    expected = [p.stem for p in campaign.shards]
    if (
        [entry.get("shard_id") for entry in entries] != expected
        or catalog.get("source_jsonl") != str(campaign.source_jsonl)
        or catalog.get("total_samples")
        != sum(r["sample_count"] for r in receipts.values())
        or catalog.get("samples_jsonl_sha256")
        != file_digest(campaign.exports_root / "samples.jsonl")
    ):
        raise ValueError("compacted output does not match complete campaign")
    for entry in entries:
        receipt = receipts[entry["shard_id"]]
        if (
            entry.get("git_commit") != receipt["identity"]["git_commit"]
            or entry.get("config_hash")
            != receipt["identity"]["shard_identity"]["config_hash"]
            or entry.get("sample_count") != receipt["sample_count"]
        ):
            raise ValueError("compacted shard identity mismatch")
    if catalog["total_samples"] == 0:
        raise ValueError(
            "zero accepted Visual samples; no H3-ready production can be published"
        )
    return file_digest(catalog_path)


def _cleanup_compact_scratch(output):
    """Only after validated publication, under this campaign's compact lock."""
    owned = re.compile(
        r"\.(?:samples\.jsonl|catalog\.json|enriched_samples\.jsonl)\.tmp-[a-f0-9]{32}"
        r"|\.production-sample-ids-[a-z0-9_]{8}\.sqlite3(?:-journal|-wal|-shm)?"
    )
    for path in output.iterdir():
        if owned.fullmatch(path.name) and path.is_file() and not path.is_symlink():
            path.unlink()


def wait_for_global(
    campaign,
    *,
    rank,
    timeout_seconds=604800,
    poll_seconds=45,
    compactor=None,
    event_callback=emit,
):
    deadline = time.monotonic() + timeout_seconds
    while True:
        receipts = {}
        for shard in campaign.shards:
            receipt = completed_receipt(campaign, shard)
            if receipt is not None:
                receipts[shard.stem] = receipt
        event_callback(
            {
                "event": "post_mask_waiting_for_global_completion",
                "rank": rank,
                "total_shards": len(campaign.shards),
                "completed_shards": len(receipts),
            }
        )
        if len(receipts) == len(campaign.shards):
            binding = {
                "campaign_sha256": digest(campaign.identity),
                "shards_sha256": digest(receipts),
            }
            if campaign.compact_marker.exists():
                marker = json.loads(campaign.compact_marker.read_text())
                if any(marker.get(k) != v for k, v in binding.items()) or marker.get(
                    "catalog_sha256"
                ) != _validate_compact(campaign, receipts):
                    raise ValueError("global compact completion identity mismatch")
                event_callback(
                    {
                        "event": "post_mask_global_completed",
                        "rank": rank,
                        "total_shards": len(receipts),
                        "exports_root": str(campaign.exports_root),
                    }
                )
                return
            if rank == 0:
                try:
                    with exclusive_lock(campaign.root / "compact.lock", blocking=False):
                        if campaign.compact_marker.exists():
                            continue
                        event_callback(
                            {
                                "event": "post_mask_compaction_started",
                                "total_shards": len(receipts),
                            }
                        )
                        if compactor is None:
                            from tools.compact_v3_production_exports import (
                                compact_production_exports,
                            )

                            compactor = compact_production_exports
                        # Recover publish-before-marker without a second complete
                        # compaction. A missing/partial publication is restartable
                        # using the existing compactor's own atomic manifests.
                        try:
                            catalog_digest = _validate_compact(campaign, receipts)
                        except (OSError, ValueError, KeyError, TypeError):
                            with heartbeat(
                                {
                                    "current_stage": "compact",
                                    "total_shards": len(receipts),
                                },
                                event_callback,
                                poll_seconds,
                            ):
                                compactor(
                                    shards_root=campaign.exports_root / "shards",
                                    output_root=campaign.exports_root,
                                    source_jsonl=campaign.source_jsonl,
                                    source_yaml=campaign.source_yaml,
                                    runs_root=campaign.runs_root,
                                )
                            catalog_digest = _validate_compact(campaign, receipts)
                        _cleanup_compact_scratch(campaign.exports_root)
                        write_json_atomic(
                            campaign.compact_marker,
                            {**binding, "catalog_sha256": catalog_digest},
                        )
                        event_callback(
                            {
                                "event": "post_mask_compaction_completed",
                                "total_shards": len(receipts),
                            }
                        )
                    continue
                except BlockingIOError:
                    pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "waiting for Post-Mask global completion; preserve roots and resume"
            )
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def terminate_owned_children(children):
    # Each child starts its own session; its Boogu descendants inherit that
    # owned process group. Never inspect/kill unrelated GPU or serving processes.
    for child in children:
        if child.poll() is not None:
            continue
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for child in children:
        if child.poll() is not None:
            continue
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def supervise_children(children, *, poll_seconds=1):
    try:
        while True:
            codes = [child.poll() for child in children]
            failures = [code for code in codes if code not in (None, 0)]
            if all(code is not None for code in codes):
                break
            time.sleep(poll_seconds)
    except BaseException:
        terminate_owned_children(children)
        raise
    if failures:
        raise RuntimeError(f"Post-Mask workers failed with exit codes {codes}")


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _scheduler_mode(value):
    from r2v_data_v2.v3.post_mask_runtime import SCHEDULER_MODES

    if value not in SCHEDULER_MODES:
        raise argparse.ArgumentTypeError(f"must be one of {', '.join(SCHEDULER_MODES)}")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scheduler-mode",
        type=_scheduler_mode,
        default=os.environ.get("POST_MASK_SCHEDULER_MODE", "wavefront_v2"),
    )
    p.add_argument(
        "--base-config", type=Path, default=os.environ.get("POST_MASK_BASE_CONFIG")
    )
    p.add_argument("--entity-mask-root", type=Path, default=SOURCE_ROOT / "entity_mask")
    p.add_argument(
        "--source-jsonl", type=Path, default=SOURCE_ROOT / "shots_f03_motion.jsonl"
    )
    p.add_argument(
        "--clips-root", type=Path, default=SOURCE_ROOT / "clips_clean_cropped"
    )
    p.add_argument("--tag", default=os.environ.get("POST_MASK_TAG", "post-mask-v1"))
    p.add_argument(
        "--post-mask-root", type=Path, default=os.environ.get("POST_MASK_ROOT")
    )
    p.add_argument(
        "--gpu-groups", default=os.environ.get("POST_MASK_GPU_GROUPS", "4:5,6:7")
    )
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1"))
    )
    p.add_argument(
        "--qwen-max-inflight",
        type=int,
        default=os.environ.get("POST_MASK_QWEN_MAX_INFLIGHT"),
    )
    p.add_argument("--cpu-workers", type=int)
    p.add_argument(
        "--clip-inflight",
        type=_positive_int,
        default=os.environ.get("POST_MASK_CLIP_INFLIGHT", "8"),
    )
    p.add_argument("--worker-timeout-seconds", type=int)
    p.add_argument(
        "--shard-inflight",
        type=_positive_int,
        default=os.environ.get("POST_MASK_SHARD_INFLIGHT", "2"),
    )
    p.add_argument("--global-wait-timeout-seconds", type=float, default=604800)
    return p


def load_launch(args):
    if args.base_config is None:
        raise ValueError(
            "POST_MASK_BASE_CONFIG/--base-config must name the accepted production YAML"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise ValueError("invalid Post-Mask tag")
    groups = parse_gpu_groups(args.gpu_groups)
    assigned_shards((), args.rank, args.world_size, 0, len(groups))
    if args.global_wait_timeout_seconds <= 0:
        raise ValueError("global wait timeout must be positive")
    config = load_config(args.base_config)
    updates = {
        name: getattr(args, name)
        for name in ("qwen_max_inflight", "cpu_workers", "worker_timeout_seconds")
        if getattr(args, name) is not None
    }
    config = replace(config, runtime=replace(config.runtime, **updates))
    config.validate()
    root = (
        args.post_mask_root
        or config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_post_mask/jea_motion_v1"
        / args.tag
    )
    campaign = prepare_campaign(
        config,
        base_config_path=args.base_config,
        entity_mask_root=args.entity_mask_root,
        post_mask_root=root,
        source_jsonl=args.source_jsonl,
        clips_root=args.clips_root,
    )
    return config, campaign, groups


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def main(argv=None):
    args = parser().parse_args(argv)
    children = []
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        config, campaign, groups = load_launch(args)
        del config
        original_args = list(argv if argv is not None else sys.argv[1:])
        python = REPOSITORY_ROOT / ".venv/bin/python"
        for slot, (sam_gpu, boogu_gpu) in enumerate(groups):
            command = [
                str(python),
                str(REPOSITORY_ROOT / "tools/run_v3_post_mask_worker.py"),
                "--worker-slot",
                str(slot),
                "--sam-gpu",
                sam_gpu,
                "--boogu-gpu",
                boogu_gpu,
                *original_args,
            ]
            children.append(
                subprocess.Popen(
                    command,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": sam_gpu,
                        "PYTHONUNBUFFERED": "1",
                    },
                    start_new_session=True,
                )
            )
        try:
            supervise_children(children)
        finally:
            children.clear()  # supervisor already closed this owned process set
        wait_for_global(
            campaign, rank=args.rank, timeout_seconds=args.global_wait_timeout_seconds
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - CLI failure boundary
        emit({"event": "post_mask_failed", "rank": args.rank, "reason": str(exc)})
        return 2
    finally:
        if children:
            terminate_owned_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
