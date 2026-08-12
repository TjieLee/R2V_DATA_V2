from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.v3.entity_composition_audit import audit_entity_composition


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit V3 final entity-type composition without changing the run.",
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()
    summary = audit_entity_composition(
        run_root=args.run_root,
        output_root=args.output_root,
        contact_sheets=args.contact_sheets,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
