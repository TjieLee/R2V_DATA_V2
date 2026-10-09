#!/usr/bin/env python3
"""Export immutable voice donor reserve sidecars from a frozen training bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.voice_donor_reserve import export_voice_donor_reserve


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    result = export_voice_donor_reserve(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
