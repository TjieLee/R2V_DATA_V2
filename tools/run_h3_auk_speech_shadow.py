#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.auk_speech_shadow import (
    PersistentAukBackend,
    auk_configuration,
    build_auk_inventory,
    preflight,
    run_auk_speech_shadow,
)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Independent local AuK speech shadow pilot"
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--auk-python", type=Path, required=True)
    parser.add_argument("--auk-code-root", type=Path, required=True)
    parser.add_argument("--auk-checkpoint", type=Path, required=True)
    parser.add_argument("--auk-qwen-path", type=Path, required=True)
    parser.add_argument("--auk-device", default="cuda:0")
    parser.add_argument("--auk-dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    configuration = auk_configuration(
        python_path=args.auk_python,
        code_root=args.auk_code_root,
        checkpoint=args.auk_checkpoint,
        qwen_path=args.auk_qwen_path,
        device=args.auk_device,
        dtype=args.auk_dtype,
    )
    inventory = build_auk_inventory(
        audio_production_root=args.audio_production_root,
        shadow_run_id=args.shadow_run_id,
        case_manifest_path=args.case_manifest,
        configuration=configuration,
    )
    output = preflight(inventory, overwrite=args.overwrite)
    with PersistentAukBackend(configuration) as backend:
        summary = run_auk_speech_shadow(
            inventory=inventory,
            backend=backend,
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
        )
    result = {"output_root": str(output), "summary": summary.model_dump(mode="json")}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
