"""Full-timeline JoyAI pilot; models run only after an explicit non-dry invocation."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.joyai import JoyAIAdapter, run_joyai_cases
from r2v_data_v2.person_replacement.pipeline import select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
from r2v_data_v2.person_replacement.timeline import inspect_video_timeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path,
                        default=Path("/mnt/workspace/liutao/X_human_data/collected_face_1.jsonl"))
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(
        "/mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_joyai_pilot"))
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--joyai-code-root", type=Path, default=os.environ.get("JOYAI_CODE_ROOT"))
    parser.add_argument("--joyai-python", type=Path, default=os.environ.get("JOYAI_PYTHON"))
    parser.add_argument("--joyai-model-root", type=Path, default=os.environ.get(
        "JOYAI_MODEL_ROOT", "/mnt/workspace/public/pretrained/jdopensource/JoyAI-Video-Edit"))
    parser.add_argument("--joyai-text-encoder-ckpt", type=Path, default=os.environ.get(
        "JOYAI_TEXT_ENCODER_CKPT", "/mnt/workspace/public/pretrained/MiMo/MiMo-VL-7B-RL-2508"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root, limit=args.limit)
    if args.dry_run:
        report = [{**case, "timeline": asdict(inspect_video_timeline(Path(case["target_video_path"]))) }
                  for case in cases]
        print(json.dumps({"dry_run": True, "cases": report}, ensure_ascii=False, indent=2))
        return 0
    manifests = run_joyai_cases(
        cases, qwen=LocalQwen(args.qwen_model), seed=args.seed,
        adapter=JoyAIAdapter(args.joyai_code_root, args.joyai_python,
                             args.joyai_model_root, args.joyai_text_encoder_ckpt),
    )
    print(json.dumps({"manifests": [str(path) for path in manifests]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
