"""Qualitative paired text-vs-Boogu-frame0 H3-PDD two-person pilot; no production normalization."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pdd import PDDBackend
from r2v_data_v2.person_replacement.h3_ref2va import (
    CONTRACTS,
    match_aspect_ratio,
    output_size,
    plan_h3_timeline,
    require_local,
)
from r2v_data_v2.person_replacement.h3_two_person_pipeline import VARIANTS, run_cases
from r2v_data_v2.person_replacement.pipeline import select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
from r2v_data_v2.person_replacement.timeline import inspect_video_timeline
from r2v_data_v2.v3.reference_edit_boogu import (
    DEFAULT_BOOGU_CODE_ROOT,
    DEFAULT_BOOGU_MODEL_PATH,
    DEFAULT_BOOGU_PYTHON,
    BooguWorkerConfig,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path,
                        default=Path("/mnt/workspace/liutao/X_human_data/collected_face_2.jsonl"))
    parser.add_argument("--clips-root", type=Path, default=Path(
        "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    for name, env in (("h3-python","H3_PYTHON"), ("h3-model-root","H3_MODEL_ROOT"),
                      ("pdd-code-root","H3_PDD_CODE_ROOT"), ("pdd-lora","H3_PDD_LORA")):
        parser.add_argument(f"--{name}", type=Path, default=os.environ.get(env))
    parser.add_argument("--boogu-python", type=Path, default=DEFAULT_BOOGU_PYTHON)
    parser.add_argument("--boogu-code-root", type=Path, default=DEFAULT_BOOGU_CODE_ROOT)
    parser.add_argument("--boogu-model-root", type=Path, default=DEFAULT_BOOGU_MODEL_PATH)
    parser.add_argument("--variant", choices=["text","frame0","both"], default="both")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root,
                         limit=args.limit, allow_two_person=True)
    if args.dry_run:
        for case in cases:
            source = inspect_video_timeline(Path(case["target_video_path"]))
            case.update(timeline=asdict(plan_h3_timeline(source)), aspect_ratio=match_aspect_ratio(source))
            case["output_size"] = output_size(case["aspect_ratio"])
        print(json.dumps({"dry_run":True, "cases":cases, "contract":CONTRACTS["pdd"],
                          "variants":list(VARIANTS.values()) if args.variant == "both" else [VARIANTS[args.variant]],
                          "resources":{key:str(value) if value is not None else None
                                       for key,value in vars(args).items() if key.endswith(("root","python","model","lora"))},
                          "seed":args.seed}, ensure_ascii=False, indent=2))
        return 0
    boogu_config = None
    if args.variant != "text":
        boogu_config = BooguWorkerConfig(
            python_executable=require_local(args.boogu_python,"Boogu python"),
            code_root=require_local(args.boogu_code_root,"Boogu code",directory=True),
            model_path=require_local(args.boogu_model_root,"Boogu model",directory=True), seed=42,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    report = run_cases(cases, qwen=LocalQwen(args.qwen_model,max_new_tokens=1024),
                       backend=PDDBackend(args.h3_python,args.h3_model_root,args.pdd_code_root,args.pdd_lora),
                       boogu_config=boogu_config, variant=args.variant, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
