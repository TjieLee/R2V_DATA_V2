"""One GPU, one Boogu frame0 worker; runs only after all Qwen workers exit."""

import argparse
import sys
from pathlib import Path

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pair_frame0 import prepare_frame0_partition
from r2v_data_v2.person_replacement.h3_pair_state import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--worker",type=int,required=True)
    args = parser.parse_args(argv)
    prepare_frame0_partition(read_json(args.config),args.worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
