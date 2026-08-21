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
from r2v_data_v2.h3.lr_asd import (
    LRASDRuntimeConfig,
    LRASDSubprocessBackend,
    SileroVADRuntimeConfig,
    SileroVADSubprocessBackend,
)
from r2v_data_v2.h3.pilot import run_h3_audio_binding_pilot
from r2v_data_v2.h3.review import ExternalReviewMediaBackend

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded read-only LR-ASD H3 audio-binding pilot",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-id", action="append")
    parser.add_argument("--clip-id-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=_positive_integer, default=1)
    parser.add_argument("--lr-asd-code-root", type=Path)
    parser.add_argument("--lr-asd-python", type=Path)
    parser.add_argument("--lr-asd-model-path", type=Path)
    parser.add_argument("--silero-python", type=Path)
    parser.add_argument("--silero-model-path", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--maximum-timestamp-delta-seconds", type=float, default=0.06)
    parser.add_argument("--minimum-face-bbox-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-matched-sampled-slots", type=int, default=2)
    parser.add_argument("--minimum-temporal-consistency", type=float, default=0.50)
    parser.add_argument("--minimum-top1-top2-margin", type=float, default=0.15)
    return parser


def _read_clip_id_file(path: Path) -> list[str]:
    resolved = path.expanduser().resolve(strict=True)
    clip_ids = [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(
        _SAFE_COMPONENT.fullmatch(clip_uid) is None or clip_uid in {".", ".."}
        for clip_uid in clip_ids
    ):
        raise ValueError("clip ID file contains a malformed clip UID")
    return clip_ids


def _combined_clip_ids(
    explicit: list[str] | None,
    clip_id_file: Path | None,
) -> list[str] | None:
    values = list(explicit or [])
    if clip_id_file is not None:
        values.extend(_read_clip_id_file(clip_id_file))
    if not values:
        return None
    if any(
        _SAFE_COMPONENT.fullmatch(clip_uid) is None or clip_uid in {".", ".."}
        for clip_uid in values
    ):
        raise ValueError("clip IDs must be safe path components")
    return list(dict.fromkeys(values))


def _path_argument(
    explicit: Path | None,
    environment_name: str,
    *,
    fallback: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit
    value = os.environ.get(environment_name)
    if value:
        return Path(value)
    if fallback is not None:
        return fallback
    raise ValueError(f"missing runtime input: {environment_name}")


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    code_root = _path_argument(arguments.lr_asd_code_root, "LR_ASD_CODE_ROOT")
    lr_python = _path_argument(arguments.lr_asd_python, "LR_ASD_PYTHON")
    lr_model = _path_argument(arguments.lr_asd_model_path, "LR_ASD_MODEL_PATH")
    silero_python = _path_argument(
        arguments.silero_python,
        "SILERO_VAD_PYTHON",
        fallback=lr_python,
    )
    silero_model = _path_argument(
        arguments.silero_model_path,
        "SILERO_VAD_MODEL_PATH",
    )
    lr_backend = LRASDSubprocessBackend(
        LRASDRuntimeConfig(
            code_root=code_root,
            python_path=lr_python,
            model_path=lr_model,
        )
    )
    summary = run_h3_audio_binding_pilot(
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        lr_asd_backend=lr_backend,
        speech_backend=SileroVADSubprocessBackend(
            SileroVADRuntimeConfig(
                python_path=silero_python,
                model_path=silero_model,
            )
        ),
        review_media_backend=ExternalReviewMediaBackend(
            python_path=lr_python,
            ffmpeg_path=arguments.ffmpeg,
        ),
        clip_ids=_combined_clip_ids(arguments.clip_id, arguments.clip_id_file),
        limit=arguments.limit,
        workers=arguments.workers,
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
    quality_summary_path = (
        Path(summary.output_root) / "voice_reference_quality_summary.json"
    )
    result["voice_reference_quality"] = json.loads(
        quality_summary_path.read_text(encoding="utf-8")
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
