from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.text_usability import (
    plan_text_usability,
    publish_text_usability,
    text_usability_output_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen H3 ASR V2 text-display eligibility policy."
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.dry_run:
        if arguments.overwrite:
            raise ValueError("--dry-run cannot be combined with --overwrite")
        result = plan_text_usability(audio_run_root=arguments.audio_run_root)
    else:
        summary = publish_text_usability(
            audio_run_root=arguments.audio_run_root,
            overwrite=arguments.overwrite,
        )
        result = {
            "output_root": str(
                text_usability_output_root(
                    arguments.audio_run_root.expanduser().resolve(strict=True)
                )
            ),
            "source_asr_v2_inventory_fingerprint": (
                summary.source_asr_v2_inventory_fingerprint
            ),
            "segment_count": summary.segment_count,
            "trusted_text_count": summary.trusted_text_count,
            "hidden_text_count": summary.hidden_text_count,
            "policy_version": summary.policy_version,
            "language_probability_threshold": (summary.language_probability_threshold),
            "model_calls": 0,
            "gpu_calls": 0,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
