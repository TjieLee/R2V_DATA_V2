"""Restartable 2/4/8-GPU, text-only person replacement shard executor."""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pair_executor import (
    make_config,
    run_pair,
    validate_identity,
)
from r2v_data_v2.person_replacement.h3_pair_state import inventory
from r2v_data_v2.person_replacement.h3_pdd_distributed import parallel_layout
from r2v_data_v2.v3.reference_edit_boogu import (
    DEFAULT_BOOGU_CODE_ROOT,
    DEFAULT_BOOGU_MODEL_PATH,
    DEFAULT_BOOGU_PYTHON,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl",type=Path,
                        default=Path("/mnt/workspace/liutao/X_human_data/collected_face_2.jsonl"))
    parser.add_argument("--clips-root",type=Path,default=Path(
        "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped"))
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--qwen-model",type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"),
                        help="Legacy 8B writer; frame0 experiment only, text never loads it")
    parser.add_argument("--prompt-writer-python",type=Path,
                        default=Path("/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env/bin/python"),
                        help="Text V22 isolated prompt-writer runtime; frame0 never uses it")
    parser.add_argument("--prompt-writer-model",type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3.5-27B"),
                        help="Prompt-writer checkpoint identity; local path for local backend")
    parser.add_argument("--prompt-writer-backend",choices=("local","sglang","mimo"),default="local",
                        help="Text captioner backend: local Qwen3.5-27B, Qwen3.8 SGLang, or MiMo-V2.5 SGLang")
    parser.add_argument("--prompt-writer-base-url",default="http://127.0.0.1:8000/v1",
                        help="Qwen3.8 SGLang OpenAI-compatible endpoint")
    parser.add_argument("--prompt-writer-served-model",default="Qwen/Qwen3.8-Flash-Next",
                        help="Qwen3.8 SGLang served model name")
    parser.add_argument("--h3-python",type=Path,default=Path(
        "/mnt/workspace/litengjie/data/person_replacement_deps/minimax-h3-env/bin/python"))
    parser.add_argument("--h3-model-root",type=Path,default=Path("/mnt/workspace/public/pretrained/MiniMaxAI/MiniMax-H3"))
    parser.add_argument("--pdd-code-root",type=Path,
                        default=Path("/mnt/workspace/litengjie/data/vendor/MiniMax-H3-Acc-LoRAs"))
    parser.add_argument("--pdd-lora",type=Path,default=os.environ.get("H3_PDD_LORA"),
                        help="Existing exact server checkpoint; no guessed path or downloads")
    parser.add_argument("--variant",choices=("text","frame0"),default="text",
                        help="text keeps the text-only Ref2VA baseline; frame0 adds a Boogu-edited "
                             "first-frame <Picture 1> next to the source <Video 1>")
    # Server defaults. Never validated here; only frame0 runtime resolves them.
    parser.add_argument("--boogu-python",type=Path,default=DEFAULT_BOOGU_PYTHON)
    parser.add_argument("--boogu-code-root",type=Path,default=DEFAULT_BOOGU_CODE_ROOT)
    parser.add_argument("--boogu-model-root",type=Path,default=DEFAULT_BOOGU_MODEL_PATH)
    parser.add_argument("--pair-id",type=int,default=0)
    parser.add_argument("--pair-size",type=int,default=1000)
    parser.add_argument("--group-size",type=int,choices=(2,4,8),default=2)
    parser.add_argument("--ulysses-degree",type=int,choices=(1,2,4,8),default=1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max-prepare-attempts",type=int,default=2)
    parser.add_argument("--max-generate-attempts",type=int,default=2)
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args(argv)
    try:
        parallel_layout(args.group_size,args.ulysses_degree)
    except ValueError as exc:
        parser.error(str(exc))
    if args.seed < 0 or min(args.max_prepare_attempts,args.max_generate_attempts) < 1:
        parser.error("seed >= 0 and positive attempt budgets required")
    return args


def main(argv=None):
    args = arguments(argv)
    root,config = make_config(args)
    if args.status or args.dry_run:
        validate_identity(root,config,read_only=True)
        print(json.dumps({"read_only":True,"shard":str(root),"state":inventory(config["cases"],config["limits"]),
                          "cases":config["cases"],"identity":config["identity"],
                          "resources":config["identity_details"]["resources"]},ensure_ascii=False,indent=2))
        return 0
    result = run_pair(root,config,args)
    print(json.dumps({"event":"pair_scan_completed","pair_id":args.pair_id,**result},ensure_ascii=False))
    return 2 if result["exhausted_prepare"] or result["exhausted_generate"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
