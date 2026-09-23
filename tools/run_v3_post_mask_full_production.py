"""Elastic, fixed-root Post-Mask Resource Epoch production entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from r2v_data_v2.v3 import config as config_module
from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND, V3Config, load_config
from r2v_data_v2.v3.post_mask_epoch_groups import build_groups
from r2v_data_v2.v3.post_mask_epoch_production import (
    production_group_completed,
    run_elastic_groups,
)
from r2v_data_v2.v3.post_mask_epoch_removal import run_removal_epoch
from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_json, file_lock
from r2v_data_v2.v3.post_mask_production import enumerate_shards
from r2v_data_v2.v3.production_source import JEA_VIDEO_MOTION_ADAPTER
from tools.compact_v3_production_exports import compact_production_exports
from tools.run_v3_post_mask_resource_epoch import build_campaign

BASE_CONFIG = REPO / "configs/v3_post_mask_resource_epoch_production.yaml"
ENTITY_MASK_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask"
)
CLIPS_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped"
)
QWEN_8B = Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct")


def _emit(event: str, **details: object) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def validate_formal_config(config: V3Config, root: Path) -> None:
    if config.export_root.resolve() != root / "shards":
        raise ValueError("formal config export_root differs from official root")
    if {service.model for _, service in config.qwen_services()} != {str(QWEN_8B)}:
        raise ValueError("all formal Post-Mask Qwen services must use Qwen3-VL-8B")
    if config.remove.backend != BOOGU_REMOVE_BACKEND:
        raise ValueError("formal Post-Mask removal must use Boogu")
    if not (config.reference_edit.enabled and config.reference_integrity.enabled):
        raise ValueError("formal Post-Mask reference stages must be enabled")
    if config.subject_attribute_gme.enabled:
        raise ValueError("formal Resource Epoch cannot enable GME")


def write_source_descriptor(config: V3Config, config_path: Path) -> Path:
    """Prepare the existing compactor's source contract after the DAG barrier."""
    path = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_configs/production/jea_motion_v1/in_pair_reference/source.yaml"
    )
    payload = {
        "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
        "source_jsonl": str(config.dataset_json),
        "clips_root": str(CLIPS_ROOT),
        "base_config_path": str(config_path.resolve()),
        "base_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_config_fingerprint": config.fingerprint(),
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("formal Post-Mask source descriptor differs")
    else:
        atomic_write_json(path, payload)
    return path


def compact_if_complete(
    root: Path, groups: tuple, config: V3Config, config_path: Path
) -> None:
    """Compact once, after every small group marker is durable."""
    state = root / "state"
    with file_lock(state / "compaction.lock") as held:
        if not held:
            raise RuntimeError("could not acquire production compaction lock")
        marker = state / "compaction_completed.json"
        if marker.is_file():
            value = json.loads(marker.read_text(encoding="utf-8"))
            if value != {"status": "completed", "group_count": len(groups)}:
                raise ValueError("formal Post-Mask compaction marker differs")
            return
        if not all(production_group_completed(state, group) for group in groups):
            raise RuntimeError("cannot compact before all groups complete")
        descriptor = write_source_descriptor(config, config_path)
        runs = (
            config_module.ALLOWED_WRITABLE_ROOT
            / "r2v_v3_runs/production/jea_motion_v1/in_pair_reference"
        )
        catalog = compact_production_exports(
            shards_root=root / "shards",
            output_root=root,
            source_jsonl=config.dataset_json,
            source_yaml=descriptor,
            runs_root=runs,
        )
        atomic_write_json(
            marker, {"status": "completed", "group_count": len(groups)}
        )
        _emit("post_mask_production_compacted", samples=catalog["total_samples"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--entity-mask-root", type=Path, default=ENTITY_MASK_ROOT)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = config_module.OFFICIAL_POST_MASK_EXPORT_ROOT.resolve()
    config = load_config(args.base_config)
    validate_formal_config(config, root)
    shards = enumerate_shards(args.entity_mask_root)
    if not shards:
        raise ValueError("formal Post-Mask source has no canonical shards")
    campaign = build_campaign(
        config, entity_mask_root=args.entity_mask_root,
        canonical_shard_count=len(shards),
    )
    groups = build_groups(
        [shard.stem for shard in shards], campaign=campaign,
        group_size=args.group_size,
    )
    _emit(
        "post_mask_production_started", rank=args.rank,
        world_size=args.world_size, group_count=len(groups),
        entity_mask_root=str(args.entity_mask_root), output_root=str(root),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise PermissionError(f"official Post-Mask output is not writable: {root}")
    os.environ.update(
        POST_MASK_BASE_CONFIG=str(args.base_config),
        POST_MASK_ENTITY_MASK_ROOT=str(args.entity_mask_root),
        POST_MASK_ROOT=str(root / "state"),
        POST_MASK_REPO=str(REPO),
        POST_MASK_CANONICAL_SHARD_COUNT=str(len(shards)),
        POST_MASK_FORMAL_PRODUCTION="1",
        POST_MASK_QWEN_MODEL_PATH=str(QWEN_8B),
        POST_MASK_EPOCH_TEMP_ROOT=str(
            config_module.ALLOWED_WRITABLE_ROOT
            / "r2v_v3_runs/production/jea_motion_v1/in_pair_reference/tmp/resource_epoch"
        ),
    )
    run_elastic_groups(
        root / "state", groups, run_removal_epoch,
        rank=args.rank, world_size=args.world_size, emit=_emit,
    )
    compact_if_complete(root, groups, config, args.base_config)
    _emit("post_mask_production_complete", group_count=len(groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
