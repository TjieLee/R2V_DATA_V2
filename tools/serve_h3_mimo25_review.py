#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.mimo25_human_review import (
    MimoReviewStore,
    build_review_cases,
    make_review_server,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve MiMo-V2.5 AV shadow review")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--legacy-qwen38-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    mimo_root = paths.root / "mimo25_av_reconcile_v4"
    shadow_root = paths.root / "mimo25_h3_shadow_v4"
    cases, media = build_review_cases(
        mimo_root=mimo_root,
        shadow_root=shadow_root,
        legacy_qwen38_root=arguments.legacy_qwen38_root,
    )
    store = MimoReviewStore(shadow_root / "human_review", cases)
    store.publish_derived()
    server = make_review_server(
        host=arguments.host,
        port=arguments.port,
        cases=cases,
        media=media,
        store=store,
    )
    print(f"MiMo review: http://{arguments.host}:{arguments.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
