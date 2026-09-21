"""One GPU, one resident writer model, deterministic group partition preparation."""

import argparse
import sys
from pathlib import Path

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_pair_prepare import (
    prepare_partition,
    prepare_text_partition_v19,
)
from r2v_data_v2.person_replacement.h3_pair_state import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--worker",type=int,required=True)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    # text V19 uses the resident 27B writer; frame0 keeps the 8B + Boogu path.
    prepare = prepare_partition if config.get("variant") == "frame0" else prepare_text_partition_v19
    prepare(config,args.worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
