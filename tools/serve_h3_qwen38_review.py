#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.qwen38_human_review import make_review_handler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a persisted Qwen3.8 H3 recaption human review",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if not 0 < arguments.port <= 65535:
        raise ValueError("review server port must be between 1 and 65535")
    handler = make_review_handler(arguments.output_root)
    server = HTTPServer((arguments.host, arguments.port), handler)
    print(
        f"Serving Qwen3.8 H3 review at http://{arguments.host}:{arguments.port}/",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
