#!/usr/bin/env python3
"""Review and export frozen R(A)2VA training tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.training_task_review import (
    TrainingTaskReviewStore,
    build_review_cases,
    export_reviewed_training_manifests,
    make_review_server,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review frozen R(A)2VA training tasks")
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("serve", "export"):
        action = actions.add_parser(name)
        action.add_argument("--ra2va-shadow-root", type=Path, required=True)
        action.add_argument("--review-root", type=Path, required=True)
        if name == "serve":
            action.add_argument("--host", default="127.0.0.1")
            action.add_argument("--port", type=int, default=8769)
        else:
            action.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> Path | None:
    arguments = _parser().parse_args(argv)
    source = arguments.ra2va_shadow_root.expanduser().resolve(strict=True)
    destinations = [arguments.review_root]
    if arguments.action == "export":
        destinations.append(arguments.output_root)
    if any(path.expanduser().resolve().is_relative_to(source) for path in destinations):
        raise ValueError("review and export roots must be outside the frozen shadow")
    cases = build_review_cases(arguments.ra2va_shadow_root)
    store = TrainingTaskReviewStore(arguments.review_root, cases)
    store.publish_derived()
    if arguments.action == "export":
        result = export_reviewed_training_manifests(
            cases=cases, store=store, output_root=arguments.output_root
        )
        print(f"Reviewed R(A)2VA training manifests: {result}")
        return result
    server = make_review_server(
        host=arguments.host, port=arguments.port, cases=cases, store=store
    )
    print(
        f"R(A)2VA training review: http://{arguments.host}:{server.server_port}/",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return None


if __name__ == "__main__":
    main()
