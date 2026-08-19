from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.asr_v2_text_calibration import (
    CALIBRATION_OUTPUT_DIRECTORY,
    analyze_asr_v2_text_usability,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze H3 ASR V2 transcript usability without model calls."
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = analyze_asr_v2_text_usability(
        audio_run_root=arguments.audio_run_root,
        qa_json=arguments.qa_json,
        overwrite=arguments.overwrite,
    )
    result = {
        "output_root": str(
            arguments.audio_run_root.expanduser().resolve()
            / CALIBRATION_OUTPUT_DIRECTORY
        ),
        "pilot_segment_count": summary.pilot_segment_count,
        "production_segment_count": summary.production_segment_count,
        "text_usability_policy_validated": False,
        "text_usability_gate_applied": False,
        "transcript_confidence_threshold_used": False,
        "model_calls": 0,
        "gpu_calls": 0,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
