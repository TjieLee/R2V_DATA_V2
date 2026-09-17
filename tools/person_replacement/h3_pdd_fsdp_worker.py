"""Exactly one torchrun session/model per pair; only rank zero publishes."""

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pair_generation import generate_loop
from r2v_data_v2.person_replacement.h3_pair_state import read_json
from r2v_data_v2.person_replacement.h3_pdd_distributed import (
    DistributedChannel,
    PersistentPDD,
)
from r2v_data_v2.person_replacement.h3_ref2va import (
    PDD_CHECKPOINT,
    PDD_REVISION,
    require_local,
    validate_checkpoint,
    validate_code,
)
from tools.person_replacement.h3_pair_child import arm_parent_death


def main(argv=None):
    # torchrun creates separate rank process groups. Also bind each rank's
    # lifetime to its launcher, including when that launcher is SIGKILLed.
    if "R2V_PAIR_LAUNCHER_PID" in os.environ:
        arm_parent_death(int(os.environ["R2V_PAIR_LAUNCHER_PID"]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--stats",type=Path,required=True)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    code = Path(config["pdd_code_root"])
    validate_code(code,PDD_REVISION,("minimax_h3_pdd.py","predict_ref2v.py"))
    validate_checkpoint(require_local(config["pdd_lora"],"PDD weights"),PDD_CHECKPOINT)
    sys.path.insert(0,str(code))
    channel = DistributedChannel()
    try:
        stats = generate_loop(config,channel,lambda:PersistentPDD(config,channel),args.stats)
        return 75 if stats["restart_required"] else 0
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
