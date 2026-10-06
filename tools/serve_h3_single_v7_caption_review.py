"""Serve a read-only overlay of frozen single-v7 reconcile results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.single_v7_caption_review import (
    build_caption_cases,
    make_caption_server,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="View latest single-v7 RA2VA captions")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--override-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    cases = build_caption_cases(args.base_root, args.override_root)
    server = make_caption_server(host=args.host, port=args.port, cases=cases)
    print(f"Single-v7 captions: http://{args.host}:{server.server_port}/ ({len(cases)} clips)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
