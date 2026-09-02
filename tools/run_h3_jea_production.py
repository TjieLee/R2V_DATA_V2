#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.audio_production import atomic_replace_stage
from r2v_data_v2.h3.binding_audit import run_speaker_binding_audit
from r2v_data_v2.h3.diarization_binding import run_diarization_binding_pilot
from r2v_data_v2.h3.jea_audio_production import (
    JEAOccurrenceEmbedding,
    build_jea_diarization_inventory,
    build_jea_pairs,
    jea_production_paths,
    run_jea_audio_stage,
    run_jea_embedding_stage,
    run_jea_primary_voice_stage,
)
from r2v_data_v2.h3.jea_diarization import publish_readable_diarization_metadata
from r2v_data_v2.h3.jea_final_renderer import render_jea_final_samples
from r2v_data_v2.h3.lr_asd import (
    LRASDRuntimeConfig,
    LRASDSubprocessBackend,
    SileroVADRuntimeConfig,
    SileroVADSubprocessBackend,
)
from r2v_data_v2.h3.pilot import resolve_lr_asd_parallelism
from r2v_data_v2.h3.qwen3_asr import (
    QWEN3_ASR_MODEL_IDENTIFIER,
    Qwen3ASRSummary,
)
from r2v_data_v2.h3.review import ExternalReviewMediaBackend
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory
from tools.run_h3_audio_production import _embedding_backends, _path_argument
from tools.run_h3_diarization_binding import _runtime_backend

_STAGES = (
    "audio",
    "primary-voice",
    "embedding",
    "pair",
    "diarization",
    "binding-audit",
    "qwen3-asr",
    "h3",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finish readable JEA Audio Production with Qwen3 ASR",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--audio-semantics-root", type=Path)
    parser.add_argument("--stages", default=",".join(_STAGES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr-asd-gpus")
    parser.add_argument("--lr-asd-workers-per-gpu", type=int)
    parser.add_argument("--review-media", choices=("all", "none"), default="all")
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


def _parse_stages(value: str) -> tuple[str, ...]:
    if value.strip() == "all":
        return _STAGES
    stages = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    unknown = set(stages) - set(_STAGES)
    if not stages or unknown:
        raise ValueError(f"invalid JEA production stages: {sorted(unknown)}")
    return tuple(stage for stage in _STAGES if stage in stages)


def _audio_parallelism(arguments: argparse.Namespace) -> tuple[int, tuple[str, ...], int, int]:
    raw_gpu_ids = arguments.lr_asd_gpus
    if raw_gpu_ids is None:
        raw_gpu_ids = os.environ.get("LR_ASD_GPUS")
    gpu_ids = (
        None
        if raw_gpu_ids is None
        else [part.strip() for part in raw_gpu_ids.split(",")]
    )
    workers_per_gpu = arguments.lr_asd_workers_per_gpu
    if workers_per_gpu is None:
        workers_per_gpu = int(os.environ.get("LR_ASD_WORKERS_PER_GPU", "4"))
    return resolve_lr_asd_parallelism(
        workers=arguments.workers,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        legacy_default_workers=4,
    )


def _read_occurrences(path: Path) -> list[JEAOccurrenceEmbedding]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            JEAOccurrenceEmbedding.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def _require_stage_artifact(stage: str, prerequisite: str, path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"JEA stage {stage} requires completed {prerequisite}: {path}"
        ) from exc


def _run_isolated_qwen3_asr(
    *,
    visual_production_root: str,
    visual_runs_root: str,
    audio_production_root: Path,
    ffmpeg: str,
    overwrite: bool,
) -> Qwen3ASRSummary:
    environment_root = os.environ.get("QWEN3_ASR_ENV")
    if not environment_root:
        raise ValueError("QWEN3_ASR_ENV is required for the active Qwen3 stage")
    python = Path(environment_root).expanduser() / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"QWEN3_ASR_ENV Python is missing: {python}")
    command = [
        str(python),
        str(REPOSITORY_ROOT / "tools" / "run_h3_qwen3_asr.py"),
        "--visual-production-root",
        visual_production_root,
        "--visual-runs-root",
        visual_runs_root,
        "--audio-production-root",
        str(audio_production_root),
        "--ffmpeg",
        ffmpeg,
    ]
    if overwrite:
        command.append("--overwrite")
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"isolated Qwen3 ASR subprocess failed ({result.returncode}): "
            f"{diagnostic}"
        )
    return Qwen3ASRSummary.model_validate_json(
        (audio_production_root / "asr" / "summary.json").read_text(
            encoding="utf-8"
        )
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    stages = _parse_stages(arguments.stages)
    visual = load_visual_production_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
    )
    paths = jea_production_paths(arguments.audio_production_root)
    effective_workers, gpu_ids, workers_per_gpu, gpu_slot_capacity = (
        _audio_parallelism(arguments)
    )
    plan: dict[str, object] = {
        "visual_production_root": visual.visual_production_root,
        "visual_runs_root": visual.visual_runs_root,
        "visual_input_schema": visual.visual_input_schema,
        "visual_input_mode": visual.visual_input_mode,
        "audio_production_root": str(paths.root),
        "canonical_sample_count": visual.canonical_sample_count,
        "eligible_subject_occurrence_count": visual.eligible_subject_occurrence_count,
        "media_collection_count": visual.media_collection_count,
        "media_collection_clip_counts": visual.media_collection_clip_counts,
        "shard_count": visual.shard_count,
        "selected_asr_model": QWEN3_ASR_MODEL_IDENTIFIER,
        "requested_stages": list(stages),
        "bounded_limit_applied": False,
        "quota_applied": False,
        "audio_effective_workers": effective_workers,
        "lr_asd_gpu_ids": list(gpu_ids),
        "lr_asd_workers_per_gpu": workers_per_gpu,
        "lr_asd_gpu_slot_capacity": gpu_slot_capacity,
        "review_media_mode": arguments.review_media,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan

    stage_results: dict[str, object] = {}
    media_backend = FFmpegAudioMediaBackend(
        ffmpeg=arguments.ffmpeg,
        ffprobe=arguments.ffprobe,
    )
    if "audio" in stages:
        lr_python = _path_argument(arguments.lr_asd_python, "LR_ASD_PYTHON")
        stage_results["audio"] = atomic_replace_stage(
            paths.audio,
            overwrite=arguments.overwrite,
            runner=lambda: run_jea_audio_stage(
                visual_inventory=visual,
                output_root=paths.audio,
                lr_asd_backend=LRASDSubprocessBackend(
                    LRASDRuntimeConfig(
                        code_root=_path_argument(
                            arguments.lr_asd_code_root, "LR_ASD_CODE_ROOT"
                        ),
                        python_path=lr_python,
                        model_path=_path_argument(
                            arguments.lr_asd_model_path, "LR_ASD_MODEL_PATH"
                        ),
                    )
                ),
                speech_backend=SileroVADSubprocessBackend(
                    SileroVADRuntimeConfig(
                        python_path=_path_argument(
                            arguments.silero_python,
                            "SILERO_VAD_PYTHON",
                            fallback=lr_python,
                        ),
                        model_path=_path_argument(
                            arguments.silero_model_path, "SILERO_VAD_MODEL_PATH"
                        ),
                    )
                ),
                review_media_backend=ExternalReviewMediaBackend(
                    python_path=lr_python,
                    ffmpeg_path=arguments.ffmpeg,
                ),
                audio_backend=media_backend,
                workers=effective_workers,
                lr_asd_gpu_ids=list(gpu_ids),
                lr_asd_workers_per_gpu=workers_per_gpu,
                review_media_mode=arguments.review_media,
            ),
        ).model_dump(mode="json")
    if "primary-voice" in stages:
        _require_stage_artifact("primary-voice", "audio", paths.audio / "summary.json")
        stage_results["primary-voice"] = run_jea_primary_voice_stage(
            visual_inventory=visual,
            audio_root=paths.audio.resolve(strict=True),
            output_root=paths.primary_voice,
            audio_backend=media_backend,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "embedding" in stages:
        _require_stage_artifact(
            "embedding",
            "primary-voice",
            paths.primary_voice / "primary_voice_references.jsonl",
        )
        face_backend, speaker_backend = _embedding_backends(
            arguments,
            paths.root / ".embedding_worker_diagnostics",
        )
        try:
            stage_results["embedding"] = run_jea_embedding_stage(
                visual_inventory=visual,
                primary_voice_root=paths.primary_voice.resolve(strict=True),
                output_root=paths.embedding,
                face_backend=face_backend,
                speaker_backend=speaker_backend,
                overwrite=arguments.overwrite,
            ).model_dump(mode="json")
        finally:
            face_backend.close()
            speaker_backend.close()
    if "pair" in stages:
        occurrences_path = _require_stage_artifact(
            "pair", "embedding", paths.embedding / "occurrences.jsonl"
        )
        stage_results["pair"] = build_jea_pairs(
            visual_inventory=visual,
            occurrences=_read_occurrences(occurrences_path),
            audio_root=paths.audio,
            output_root=paths.pairs,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "diarization" in stages:
        _require_stage_artifact(
            "diarization", "canonical audio", paths.audio / "canonical_clips.jsonl"
        )
        inventory = build_jea_diarization_inventory(
            visual_inventory=visual,
            audio_root=paths.audio.resolve(strict=True),
            ffprobe=arguments.ffprobe,
        )
        backend, diagnostics_root = _runtime_backend(output_root=paths.diarization)
        try:
            with backend:
                summary = run_diarization_binding_pilot(
                    inventory=inventory,
                    output_root=paths.diarization,
                    backend=backend,
                    overwrite=arguments.overwrite,
                )
            readable = publish_readable_diarization_metadata(
                visual_inventory=visual,
                diarization_root=paths.diarization,
            )
        finally:
            if diagnostics_root.exists():
                shutil.rmtree(diagnostics_root)
        stage_results["diarization"] = {
            "summary": summary.model_dump(mode="json"),
            "readable": readable.model_dump(mode="json"),
        }
    if "binding-audit" in stages:
        _require_stage_artifact(
            "binding-audit",
            "diarization",
            paths.diarization / "inventory.json",
        )
        stage_results["binding-audit"] = run_speaker_binding_audit(
            audio_production_root=paths.root,
            ffmpeg=arguments.ffmpeg,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "qwen3-asr" in stages:
        _require_stage_artifact(
            "qwen3-asr",
            "diarization",
            paths.diarization / "readable_segments.jsonl",
        )
        stage_results["qwen3-asr"] = _run_isolated_qwen3_asr(
            visual_production_root=visual.visual_production_root,
            visual_runs_root=visual.visual_runs_root,
            audio_production_root=paths.root,
            ffmpeg=arguments.ffmpeg,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "h3" in stages:
        for prerequisite, path in (
            ("canonical audio", paths.audio / "canonical_clips.jsonl"),
            ("diarization", paths.diarization / "inventory.json"),
            ("diarization", paths.diarization / "bound_segments.jsonl"),
            ("binding-audit", paths.root / "binding_audit_v1" / "segments.jsonl"),
            ("qwen3-asr", paths.asr / "segments.jsonl"),
        ):
            _require_stage_artifact("h3", prerequisite, path)
        semantics_root = arguments.audio_semantics_root
        default_semantics_root = paths.root / "audio_semantics_specialized_v1"
        if semantics_root is None and (
            default_semantics_root / "assembled/records.jsonl"
        ).is_file():
            semantics_root = default_semantics_root
        stage_results["h3"] = render_jea_final_samples(
            visual_inventory=visual,
            audio_root=paths.audio,
            pairs_root=paths.pairs if paths.pairs.exists() else None,
            diarization_root=paths.diarization,
            binding_audit_root=paths.root / "binding_audit_v1",
            qwen3_asr_root=paths.asr,
            output_root=paths.h3,
            primary_voice_root=paths.primary_voice,
            audio_semantics_root=semantics_root,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    result = {**plan, "stage_results": stage_results}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
