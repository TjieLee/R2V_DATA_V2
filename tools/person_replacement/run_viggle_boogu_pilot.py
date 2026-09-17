"""Candidate-only official Viggle pilot with the repository's existing Boogu worker."""

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.pipeline import select_cases
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
from r2v_data_v2.person_replacement.timeline import inspect_video_timeline
from r2v_data_v2.person_replacement.viggle_boogu import (
    ViggleAdapter,
    has_audio,
    plan_viggle_timeline,
    read_descriptions,
    run_cases,
    select_viggle_canvas,
    validate_boogu,
)
from r2v_data_v2.v3.reference_edit_boogu import (
    DEFAULT_BOOGU_CODE_ROOT,
    DEFAULT_BOOGU_MODEL_PATH,
    DEFAULT_BOOGU_PYTHON,
    BooguWorkerConfig,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-jsonl", "clips-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path,
                        default=Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--descriptions-from", type=Path)
    for name, variable, default in (
        ("boogu-python","BOOGU_PYTHON",DEFAULT_BOOGU_PYTHON),
        ("boogu-code-root","BOOGU_CODE_ROOT",DEFAULT_BOOGU_CODE_ROOT),
        ("boogu-model-root","BOOGU_MODEL_ROOT",DEFAULT_BOOGU_MODEL_PATH),
        ("viggle-python","VIGGLE_PYTHON",None),
        ("viggle-root","VIGGLE_ROOT",None),
        ("h3-model-root","H3_MODEL_ROOT",None),
    ):
        parser.add_argument(f"--{name}", type=Path, default=os.environ.get(variable, default))
    parser.add_argument("--boogu-gpu", default="0")
    parser.add_argument("--viggle-gpu", default="1")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("This first pilot fixes both model seeds at 42")
    if args.limit < 1 or (args.descriptions_from is not None and args.limit != 1):
        parser.error("positive --limit required; --descriptions-from requires --limit 1")
    if any(value is None for value in (args.viggle_python, args.viggle_root, args.h3_model_root)):
        parser.error("Provide --viggle-python, --viggle-root and --h3-model-root or their environment variables")
    if any(not gpu.isdigit() for gpu in (args.boogu_gpu, args.viggle_gpu)):
        parser.error("Each backend requires one physical GPU index")
    config = BooguWorkerConfig(python_executable=args.boogu_python.expanduser().absolute(),
                              code_root=args.boogu_code_root.expanduser().resolve(),
                              model_path=args.boogu_model_root.expanduser().resolve(),
                              seed=42, cuda_visible_devices=args.boogu_gpu)
    validate_boogu(config)
    viggle = ViggleAdapter(args.viggle_python, args.viggle_root, args.h3_model_root, args.viggle_gpu)
    viggle.validate()
    cases = select_cases(args.input_jsonl, args.clips_root, args.output_root, limit=args.limit)
    sources, audio, prepared = [], [], []
    for case in cases:
        validate_boogu(replace(config, temporary_root=Path(case["output_dir"]) / "runtime/boogu"))
        source = Path(case["target_video_path"])
        timeline = inspect_video_timeline(source)
        canvas = select_viggle_canvas(timeline.width, timeline.height)
        plan = plan_viggle_timeline(timeline)
        if args.descriptions_from:
            read_descriptions(args.descriptions_from, source)
        sources.append(timeline)
        audio.append(has_audio(source))
        prepared.append({**case, "source_timeline":asdict(timeline), "canvas":asdict(canvas),
                         "timeline":asdict(plan), "source_has_audio":audio[-1]})
    if args.dry_run:
        print(json.dumps({"dry_run":True, "cases":prepared, "boogu_model_path":str(config.model_path),
            "boogu_python":str(config.python_executable), "viggle_root":str(viggle.root),
            "viggle_python":str(viggle.python), "h3_model_root":str(viggle.h3_root),
            "viggle_prompt_mode":"official_frozen_embedding", "custom_h3_prompt":False}, indent=2))
        return 0
    results = run_cases(cases, sources=sources, audio=audio,
                        qwen=None if args.descriptions_from else LocalQwen(args.qwen_model),
                        descriptions=args.descriptions_from, boogu_config=config, viggle=viggle)
    print(json.dumps({"manifests":[str(path) for path in results], "candidate_only":True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
