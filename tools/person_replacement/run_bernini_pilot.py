"""Independent Bernini person replacement pilot; dry-run never loads a model."""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.bernini import BerniniAdapter
from r2v_data_v2.person_replacement.pipeline import run_cases, select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path,
                        default=Path("/mnt/workspace/liutao/X_human_data/collected_face_1.jsonl"))
    parser.add_argument("--clips-root", type=Path, required=True,
                        help="Explicit existing processed-clip root; never inferred from source_video_path")
    parser.add_argument("--output-root", type=Path, default=Path(
        "/mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_bernini_pilot"))
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--bernini-config", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Bernini-Diffusers"))
    parser.add_argument("--bernini-code-root", type=Path, default=os.environ.get("BERNINI_CODE_ROOT"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root, limit=args.limit)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "cases": cases}, ensure_ascii=False, indent=2))
        return 0
    results = run_cases(
        cases, qwen=LocalQwen(args.qwen_model),
        bernini=BerniniAdapter(args.bernini_code_root, args.bernini_config), seed=args.seed,
    )
    print(json.dumps({"manifests": [str(path) for path in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
