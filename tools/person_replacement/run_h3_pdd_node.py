"""Thin independent GPU-group launcher, not a scheduler or node-wide barrier."""

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
    parser.add_argument("--group-size",type=int,choices=(2,4,8),default=2)
    parser.add_argument("--output-root",type=Path,required=True)
    args, remaining = parser.parse_known_args(argv)
    devices = [device.strip() for device in args.gpus.split(",")]
    if len(devices) % args.group_size or len(set(devices)) != len(devices) or not all(devices) or args.pair_start < 0:
        parser.error("Distinct GPUs divisible by group-size and nonnegative pair-start required")
    if any(arg == "--pair-id" or arg.startswith("--pair-id=") for arg in remaining):
        parser.error("Use --pair-start, not --pair-id")
    root = validate_output_root(args.output_root)
    specs = []
    session = uuid.uuid4().hex
    for i in range(len(devices)//args.group_size):
        options = ["--output-root",str(root),"--pair-id",str(args.pair_start+i),
                   "--group-size",str(args.group_size),*remaining]
        if "--status" in remaining or "--dry-run" in remaining:
            pair_main(options)  # strictly read-only, no log/lock/process creation
            continue
        specs.append({"command":[sys.executable,str(Path(__file__).with_name("run_h3_pdd_pair_executor.py")),*options],
                      "env":{**os.environ,"CUDA_VISIBLE_DEVICES":",".join(devices[args.group_size*i:args.group_size*(i+1)])},
                      "log":root/"node_logs"/session/f"pair-{args.pair_start+i}.log"})
    return 1 if specs and any(run_children(specs,stop_on_error=False)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
