#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.qwen_speech_presentation_ab_review import (
    build_qwen_speech_presentation_ab_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the four-way Qwen speech-presentation A/B review",
    )
    parser.add_argument("--qwen35-old-root", type=Path, required=True)
    parser.add_argument("--qwen35-new-root", type=Path, required=True)
    parser.add_argument("--qwen38-old-root", type=Path, required=True)
    parser.add_argument("--qwen38-new-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = build_qwen_speech_presentation_ab_review(
        qwen35_old_root=arguments.qwen35_old_root,
        qwen35_new_root=arguments.qwen35_new_root,
        qwen38_old_root=arguments.qwen38_old_root,
        qwen38_new_root=arguments.qwen38_new_root,
        output_root=arguments.output_root,
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
