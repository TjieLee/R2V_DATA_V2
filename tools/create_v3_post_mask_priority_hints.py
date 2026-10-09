"""Generate advisory group priorities once before restarting all nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.v3.post_mask_epoch_priority import create_priority_hints
from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private advisory JSON output, outside shared state",
    )
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.state_root.resolve()):
        raise ValueError("priority hint output must be outside shared state")
    hints = create_priority_hints(args.state_root)
    atomic_write_json(args.output, hints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
