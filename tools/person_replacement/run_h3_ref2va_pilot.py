"""Independent H3 native-timeline candidate comparison; no production normalization."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_lightx2v import LightX2VBackend
from r2v_data_v2.person_replacement.h3_pdd import PDDBackend
from r2v_data_v2.person_replacement.h3_pipeline import read_descriptions, run_cases
from r2v_data_v2.person_replacement.h3_ref2va import (
    CONTRACTS,
    match_aspect_ratio,
    output_size,
    plan_h3_timeline,
)
from r2v_data_v2.person_replacement.pipeline import select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
from r2v_data_v2.person_replacement.timeline import inspect_video_timeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path,
        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    for option, env in [("h3-python","H3_PYTHON"), ("h3-model-root","H3_MODEL_ROOT"),
                        ("lightx2v-code-root","H3_TURBO_CODE_ROOT"), ("lightx2v-lora","H3_LIGHTX2V_LORA"),
                        ("pdd-code-root","H3_PDD_CODE_ROOT"), ("pdd-lora","H3_PDD_LORA")]:
        parser.add_argument(f"--{option}", type=Path, default=os.environ.get(env))
    parser.add_argument("--backend", choices=["lightx2v","pdd","both"], default="both")
    parser.add_argument("--descriptions-from", type=Path,
                        help="Reuse the same source/replacement descriptions from an existing Bernini manifest")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root, limit=args.limit)
    if args.descriptions_from:
        for case in cases:
            read_descriptions(args.descriptions_from, Path(case["target_video_path"]))
    if args.dry_run:
        for case in cases:
            source = inspect_video_timeline(Path(case["target_video_path"]))
            case["h3_timeline"] = asdict(plan_h3_timeline(source))
            case["aspect_ratio"] = match_aspect_ratio(source)
            case["output_size"] = output_size(case["aspect_ratio"])
        print(json.dumps({"dry_run":True, "cases":cases, "contracts":CONTRACTS,
                          "production_timeline_compatible":False}, ensure_ascii=False, indent=2))
        return 0
    backends = []
    if args.backend in {"lightx2v","both"}:
        fsdp = os.environ.get("H3_FSDP2", "0")
        if fsdp not in {"0","1"}:
            parser.error("H3_FSDP2 must be 0 or 1")
        backends.append(LightX2VBackend(args.h3_python, args.h3_model_root, args.lightx2v_code_root,
            args.lightx2v_lora, fsdp2=fsdp == "1", nproc=int(os.environ.get("H3_NPROC_PER_NODE","1"))))
    if args.backend in {"pdd","both"}:
        backends.append(PDDBackend(args.h3_python, args.h3_model_root, args.pdd_code_root, args.pdd_lora))
    report = run_cases(cases, qwen=LocalQwen(args.qwen_model), backends=backends, seed=args.seed,
                       descriptions_from=args.descriptions_from)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
