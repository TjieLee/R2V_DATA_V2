"""Explicit Subject Attribute intermediate-binary GC for formal Post-Mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from r2v_data_v2.v3 import config as config_module
from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="delete eligible intermediates; default is read-only dry-run"
    )
    args = parser.parse_args(argv)
    root = config_module.OFFICIAL_POST_MASK_EXPORT_ROOT / "state" / "resource_epochs"
    summary = gc_state_root(root, apply=args.apply)
    print(json.dumps({"event": "post_mask_sa_binary_gc_completed", "apply": args.apply, **summary}, sort_keys=True))
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
