"""Bernini-R 1.3B pilot using the unchanged Full Bernini timeline and prompt pipeline."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.bernini import plan_bernini_timeline
from r2v_data_v2.person_replacement.bernini_r_1p3b import (
    BerniniR1p3BAdapter,
    ManifestDescriptionsQwen,
)
from r2v_data_v2.person_replacement.pipeline import run_cases, select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
from r2v_data_v2.person_replacement.timeline import inspect_video_timeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--bernini-code-root", type=Path, default=os.environ.get("BERNINI_CODE_ROOT"))
    parser.add_argument("--bernini-r-config", type=Path, default=os.environ.get("BERNINI_R_1P3B_CONFIG"))
    parser.add_argument("--descriptions-from", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.bernini_r_config is None:
        parser.error("Provide --bernini-r-config or BERNINI_R_1P3B_CONFIG (local model directory)")
    if args.seed < 0 or args.limit < 1:
        parser.error("require non-negative --seed and positive --limit")
    if args.descriptions_from and args.limit != 1:
        parser.error("--descriptions-from requires --limit 1")
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root, limit=args.limit)
    reused = (ManifestDescriptionsQwen(args.descriptions_from, cases[0]["target_video_path"])
              if args.descriptions_from else None)
    if args.dry_run:
        for case in cases:
            source = inspect_video_timeline(Path(case["target_video_path"]))
            plan = plan_bernini_timeline(source)
            case["source_timeline"] = asdict(source)
            case["bernini_timeline"] = {"internal_frame_count": plan.internal_frame_count,
                                        "internal_fps": plan.internal_fps, "pad_frames": plan.pad_frames}
        print(json.dumps({"dry_run": True, "variant": "bernini-r-1.3b", "cases": cases}, indent=2))
        return 0
    results = run_cases(cases, qwen=reused if reused is not None else LocalQwen(args.qwen_model),
                        bernini=BerniniR1p3BAdapter(args.bernini_code_root, args.bernini_r_config),
                        seed=args.seed)
    print(json.dumps({"variant": "bernini-r-1.3b", "manifests": [str(path) for path in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
