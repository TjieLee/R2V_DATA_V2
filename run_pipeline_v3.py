from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage

STAGE_ORDER = (
    "manifest",
    "annotate",
    "frames",
    "segment",
    "rank",
    "background",
    "remove",
    "pair",
    "instruct",
    "export",
)
_IMPLEMENTED_STAGES = frozenset({"export"})


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def run_pipeline_v3(
    *,
    config_path: str | Path,
    stages: tuple[str, ...] = (),
    overwrite: bool = False,
    git_commit: str | None = None,
) -> dict[str, object]:
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown V3 pipeline stages: {unknown}")
    unavailable = [stage for stage in stages if stage not in _IMPLEMENTED_STAGES]
    if unavailable:
        raise NotImplementedError(
            "Commit 1 provides storage and export scaffolding only; "
            f"unimplemented stages requested: {unavailable}"
        )
    config = load_config(config_path)
    storage = RunStorage(config)
    run = storage.initialize(git_commit=git_commit or _git_commit())
    results: dict[str, object] = {
        "run": {
            "run_id": run.run_id,
            "run_root": str(storage.root),
            "config_hash": run.config_hash,
        }
    }
    if "export" in stages:
        dataset = DatasetExporter(config, storage).export(overwrite=overwrite)
        results["export"] = dataset.model_dump(mode="json")
    results["completed_stages"] = list(stages)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize V3 storage or run implemented V3 stages"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        default="",
        help=(
            "comma-separated V3 stages; Commit 1 implements only the export "
            "storage skeleton"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing final dataset during export",
    )
    args = parser.parse_args()
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    result = run_pipeline_v3(
        config_path=args.config,
        stages=stages,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
