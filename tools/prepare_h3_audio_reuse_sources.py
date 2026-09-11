#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_reuse_prepared import prepare_audio_reuse_sources


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Prepare frozen stem reconcile inputs for model-free Audio reuse")
    for name in ("audio-production-root", "visual-production-root", "visual-runs-root", "stem-shadow-root",
                 "base-reconcile-root", "source-h3-root", "prepared-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--override-reconcile-root", type=Path)
    parser.add_argument("--binding-evidence-mode", choices=("legacy_lr_asd", "none"), default="legacy_lr_asd")
    parser.add_argument("--stem-diarization-root", type=Path)
    parser.add_argument("--stem-asr-root", type=Path)
    result = prepare_audio_reuse_sources(**vars(parser.parse_args(argv))).model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
