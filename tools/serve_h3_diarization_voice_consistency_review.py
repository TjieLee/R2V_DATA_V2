#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.diarization_voice_consistency_review import (
    initialize_review,
    make_review_handler,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve persisted H3 diarization voice-consistency review",
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if not 0 < arguments.port <= 65535:
        raise ValueError("review server port must be between 1 and 65535")
    context, summary = initialize_review(arguments.audit_root)
    server = HTTPServer(
        (arguments.host, arguments.port),
        make_review_handler(context),
    )
    print(
        "Serving H3 diarization voice-consistency review at "
        f"http://{arguments.host}:{arguments.port}/review.html "
        f"({summary.reviewed}/{summary.total} reviewed)",
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
