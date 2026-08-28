from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.association import FaceEntityAssociationPolicy
from r2v_data_v2.h3.laser_asd import (
    LaserASDRuntimeConfig,
    LaserASDSubprocessBackend,
)
from r2v_data_v2.h3.laser_pilot import run_h3_laser_audio_binding_pilot
from r2v_data_v2.h3.lr_asd import (
    SileroVADRuntimeConfig,
    SileroVADSubprocessBackend,
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded read-only LoCoNet+LASER H3 shadow pilot",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-id", action="append")
    parser.add_argument("--clip-id-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--laser-asd-code-root", type=Path)
    parser.add_argument("--laser-asd-python", type=Path)
    parser.add_argument("--laser-asd-model-path", type=Path)
    parser.add_argument("--laser-asd-config-path", type=Path)
    parser.add_argument("--laser-asd-landmark-model-path", type=Path)
    parser.add_argument("--laser-asd-s3fd-model-path", type=Path)
    parser.add_argument("--laser-asd-device")
    parser.add_argument("--laser-asd-cuda-visible-devices")
    parser.add_argument("--silero-python", type=Path)
    parser.add_argument("--silero-model-path", type=Path)
    parser.add_argument("--maximum-timestamp-delta-seconds", type=float, default=0.06)
    parser.add_argument("--minimum-face-bbox-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-matched-sampled-slots", type=int, default=2)
    parser.add_argument("--minimum-temporal-consistency", type=float, default=0.50)
    parser.add_argument("--minimum-top1-top2-margin", type=float, default=0.15)
    return parser


def _path(explicit: Path | None, environment_name: str) -> Path:
    if explicit is not None:
        return explicit
    value = os.environ.get(environment_name)
    if not value:
        raise ValueError(f"missing runtime input: {environment_name}")
    return Path(value)


def _value(explicit: str | None, environment_name: str) -> str:
    if explicit is not None:
        return explicit
    value = os.environ.get(environment_name)
    if not value:
        raise ValueError(f"missing runtime input: {environment_name}")
    return value


def _clip_ids(explicit: list[str] | None, path: Path | None) -> list[str] | None:
    values = list(explicit or [])
    if path is not None:
        values.extend(
            line.strip()
            for line in path.expanduser().resolve(strict=True).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not values:
        return None
    if any(
        _SAFE_COMPONENT.fullmatch(value) is None or value in {".", ".."}
        for value in values
    ):
        raise ValueError("clip IDs must be safe path components")
    return list(dict.fromkeys(values))


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    laser_python = _path(arguments.laser_asd_python, "LASER_ASD_PYTHON")
    summary = run_h3_laser_audio_binding_pilot(
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        laser_backend=LaserASDSubprocessBackend(
            LaserASDRuntimeConfig(
                code_root=_path(
                    arguments.laser_asd_code_root, "LASER_ASD_CODE_ROOT"
                ),
                python_path=laser_python,
                model_path=_path(
                    arguments.laser_asd_model_path, "LASER_ASD_MODEL_PATH"
                ),
                config_path=_path(
                    arguments.laser_asd_config_path, "LASER_ASD_CONFIG_PATH"
                ),
                landmark_model_path=_path(
                    arguments.laser_asd_landmark_model_path,
                    "LASER_ASD_LANDMARK_MODEL_PATH",
                ),
                s3fd_model_path=_path(
                    arguments.laser_asd_s3fd_model_path,
                    "LASER_ASD_S3FD_MODEL_PATH",
                ),
                device=_value(arguments.laser_asd_device, "LASER_ASD_DEVICE"),
                cuda_visible_devices=_value(
                    arguments.laser_asd_cuda_visible_devices,
                    "LASER_ASD_CUDA_VISIBLE_DEVICES",
                ),
            )
        ),
        speech_backend=SileroVADSubprocessBackend(
            SileroVADRuntimeConfig(
                python_path=(
                    arguments.silero_python
                    or Path(os.environ.get("SILERO_VAD_PYTHON", str(laser_python)))
                ),
                model_path=_path(
                    arguments.silero_model_path, "SILERO_VAD_MODEL_PATH"
                ),
            )
        ),
        clip_ids=_clip_ids(arguments.clip_id, arguments.clip_id_file),
        limit=arguments.limit,
        association_policy=FaceEntityAssociationPolicy(
            maximum_timestamp_delta_seconds=(
                arguments.maximum_timestamp_delta_seconds
            ),
            minimum_face_bbox_coverage=arguments.minimum_face_bbox_coverage,
            minimum_matched_sampled_slots=(
                arguments.minimum_matched_sampled_slots
            ),
            minimum_temporal_consistency=(
                arguments.minimum_temporal_consistency
            ),
            minimum_top1_top2_margin=arguments.minimum_top1_top2_margin,
        ),
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
