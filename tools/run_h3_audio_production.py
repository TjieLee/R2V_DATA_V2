#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.association import FaceEntityAssociationPolicy
from r2v_data_v2.h3.audio_backends import (
    FFmpegAudioMediaBackend,
    PersistentSubprocessEmbeddingBackend,
    fingerprint_local_model_path,
)
from r2v_data_v2.h3.audio_production import (
    atomic_replace_stage,
    build_h3_production_pairs,
    enumerate_h3_production_inventory,
    orchestrate_production_stages,
    parse_production_stages,
    production_paths,
    run_primary_voice_production_stage,
    run_production_embedding_stage,
    write_production_inventory,
)
from r2v_data_v2.h3.lr_asd import (
    LRASDRuntimeConfig,
    LRASDSubprocessBackend,
    SileroVADRuntimeConfig,
    SileroVADSubprocessBackend,
)
from r2v_data_v2.h3.pilot import run_h3_audio_binding_pilot
from r2v_data_v2.h3.review import ExternalReviewMediaBackend


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete fixed-root H3 Audio production workflow",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--workers", type=_positive_integer, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--lr-asd-code-root", type=Path)
    parser.add_argument("--lr-asd-python", type=Path)
    parser.add_argument("--lr-asd-model-path", type=Path)
    parser.add_argument("--silero-python", type=Path)
    parser.add_argument("--silero-model-path", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")

    parser.add_argument("--face-python")
    parser.add_argument("--face-model-root", type=Path)
    parser.add_argument("--face-model-name")
    parser.add_argument("--face-model-identifier")
    parser.add_argument("--speaker-python")
    parser.add_argument("--speaker-model-path", type=Path)
    parser.add_argument(
        "--speaker-model-identifier",
        default="speechbrain/spkrec-ecapa-voxceleb",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=300.0)
    return parser


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


def _text_argument(explicit: str | None, environment_name: str) -> str:
    value = explicit or os.environ.get(environment_name)
    if value is None or not value.strip():
        raise ValueError(f"missing runtime input: {environment_name}")
    return value


def _python_executable(value: str) -> str:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"embedding Python executable is missing: {value}")
    return str(path)


def _embedding_backends(
    arguments: argparse.Namespace,
    diagnostics_root: Path,
) -> tuple[PersistentSubprocessEmbeddingBackend, PersistentSubprocessEmbeddingBackend]:
    face_python = _text_argument(arguments.face_python, "FACE_EMBEDDING_PYTHON")
    face_model_root = _path_argument(arguments.face_model_root, "FACE_MODEL_ROOT")
    face_model_name = _text_argument(arguments.face_model_name, "FACE_MODEL_NAME")
    face_model_identifier = _text_argument(
        arguments.face_model_identifier,
        "FACE_MODEL_IDENTIFIER",
    )
    speaker_python = _text_argument(
        arguments.speaker_python,
        "SPEAKER_EMBEDDING_PYTHON",
    )
    speaker_model = _path_argument(
        arguments.speaker_model_path,
        "SPEAKER_MODEL_PATH",
    ).expanduser().resolve(strict=True)
    face_root = face_model_root.expanduser().resolve(strict=True)
    face_pack = (face_root / "models" / face_model_name).resolve(strict=True)
    face_pack.relative_to(face_root)
    face_fingerprint = fingerprint_local_model_path(face_pack)
    speaker_fingerprint = fingerprint_local_model_path(speaker_model)
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cuda_visible = arguments.cuda_visible_devices or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if cuda_visible is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible
    face_backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(face_python),
            str(REPOSITORY_ROOT / "tools" / "run_h3_face_embedding_worker.py"),
            "--model-root",
            str(face_root),
            "--model-name",
            face_model_name,
            "--model-identifier",
            face_model_identifier,
            "--model-fingerprint",
            face_fingerprint,
            "--device",
            arguments.device,
        ],
        model_identifier=face_model_identifier,
        checkpoint_sha256=face_fingerprint,
        timeout_seconds=arguments.embedding_timeout_seconds,
        diagnostics_root=diagnostics_root / "face",
        environment=environment,
    )
    speaker_backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(speaker_python),
            str(REPOSITORY_ROOT / "tools" / "run_h3_speaker_embedding_worker.py"),
            "--model-path",
            str(speaker_model),
            "--model-identifier",
            arguments.speaker_model_identifier,
            "--model-fingerprint",
            speaker_fingerprint,
            "--device",
            arguments.device,
        ],
        model_identifier=arguments.speaker_model_identifier,
        checkpoint_sha256=speaker_fingerprint,
        timeout_seconds=arguments.embedding_timeout_seconds,
        diagnostics_root=diagnostics_root / "speaker",
        environment=environment,
    )
    return face_backend, speaker_backend


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    stages = parse_production_stages(arguments.stages)
    inventory = enumerate_h3_production_inventory(arguments.run_root)
    paths = production_paths(arguments.audio_run_root)
    source = Path(inventory.source_run_root)
    if paths.root == source or source in paths.root.parents or paths.root in source.parents:
        raise ValueError("H3 production root must be separate from the Visual run")
    plan = {
        "production_root": str(paths.root),
        "requested_stages": list(stages),
        "scanned_clip_count": inventory.scanned_clip_count,
        "eligible_clip_count": inventory.eligible_clip_count,
        "eligible_occurrence_count": inventory.eligible_occurrence_count,
        "bounded_limit_applied": False,
        "parent_quota_applied": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan
    write_production_inventory(inventory, paths.inventory)

    def run_audio(overwrite: bool) -> object:
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
        return atomic_replace_stage(
            paths.audio,
            overwrite=overwrite,
            runner=lambda: run_h3_audio_binding_pilot(
                run_root=source,
                output_root=paths.audio,
                lr_asd_backend=LRASDSubprocessBackend(
                    LRASDRuntimeConfig(
                        code_root=code_root,
                        python_path=lr_python,
                        model_path=lr_model,
                    )
                ),
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
                clip_ids=inventory.eligible_clip_uids,
                limit=None,
                workers=arguments.workers,
                association_policy=FaceEntityAssociationPolicy(),
            ),
        )

    def run_primary(overwrite: bool) -> object:
        return run_primary_voice_production_stage(
            audio_root=paths.audio,
            output_root=paths.primary_voice,
            audio_backend=FFmpegAudioMediaBackend(
                ffmpeg=arguments.ffmpeg,
                ffprobe=arguments.ffprobe,
            ),
            overwrite=overwrite,
        )

    def run_embedding(overwrite: bool) -> object:
        face_backend, speaker_backend = _embedding_backends(
            arguments,
            paths.root / ".embedding_worker_diagnostics",
        )
        try:
            return run_production_embedding_stage(
                inventory=inventory,
                primary_voice_root=paths.primary_voice,
                output_root=paths.embedding,
                face_backend=face_backend,
                speaker_backend=speaker_backend,
                overwrite=overwrite,
            )
        finally:
            face_backend.close()
            speaker_backend.close()

    def run_pair(overwrite: bool) -> object:
        return build_h3_production_pairs(
            inventory=inventory,
            audio_root=paths.audio,
            primary_voice_root=paths.primary_voice,
            embedding_root=paths.embedding,
            output_root=paths.pairs,
            overwrite=overwrite,
        )

    status = orchestrate_production_stages(
        stages=stages,
        roots={
            "audio": paths.audio,
            "primary-voice": paths.primary_voice,
            "embedding": paths.embedding,
            "pair": paths.pairs,
        },
        runners={
            "audio": run_audio,
            "primary-voice": run_primary,
            "embedding": run_embedding,
            "pair": run_pair,
        },
        overwrite=arguments.overwrite,
    )
    result = {**plan, "stage_status": status}
    if (paths.pairs / "summary.json").is_file():
        result["summary"] = json.loads(
            (paths.pairs / "summary.json").read_text(encoding="utf-8")
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
