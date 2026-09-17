"""Thin independent four-pair launcher, not a scheduler or node-wide barrier."""

import argparse
import os
import sys
import uuid
from pathlib import Path

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pair_executor import run_children
from r2v_data_v2.person_replacement.pipeline import validate_output_root
from tools.person_replacement.run_h3_pdd_pair_executor import main as pair_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pair-start",type=int,default=0)
    parser.add_argument("--output-root",type=Path,required=True)
    args, remaining = parser.parse_known_args(argv)
    devices = args.gpus.split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or any(not x.strip() for x in devices) or args.pair_start < 0:
        parser.error("Exactly eight distinct GPUs and nonnegative pair-start required")
    if any(arg == "--pair-id" or arg.startswith("--pair-id=") for arg in remaining):
        parser.error("Use --pair-start, not --pair-id")
    root = validate_output_root(args.output_root)
    specs = []
    session = uuid.uuid4().hex
    for i in range(4):
        options = ["--output-root",str(root),"--pair-id",str(args.pair_start+i),*remaining]
        if "--status" in remaining or "--dry-run" in remaining:
            pair_main(options)  # strictly read-only, no log/lock/process creation
            continue
        specs.append({"command":[sys.executable,str(Path(__file__).with_name("run_h3_pdd_pair_executor.py")),*options],
                      "env":{**os.environ,"CUDA_VISIBLE_DEVICES":",".join(devices[2*i:2*i+2])},
                      "log":root/"node_logs"/session/f"pair-{args.pair_start+i}.log"})
    return 1 if specs and any(run_children(specs,stop_on_error=False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
