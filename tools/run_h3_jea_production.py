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
    parser.add_argument("--stages", default=",".join(_STAGES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
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
        if arguments.workers <= 0:
            raise ValueError("JEA audio workers must be positive")
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
                workers=arguments.workers,
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
            "diarization", "pair", paths.pairs / "in_pairs.jsonl"
        )
        inventory = build_jea_diarization_inventory(
            pairs_root=paths.pairs.resolve(strict=True)
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
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "h3" in stages:
        for prerequisite, path in (
            ("pair", paths.pairs / "in_pairs.jsonl"),
            ("diarization", paths.diarization / "bound_segments.jsonl"),
            ("qwen3-asr", paths.asr / "segments.jsonl"),
        ):
            _require_stage_artifact("h3", prerequisite, path)
        stage_results["h3"] = render_jea_final_samples(
            visual_inventory=visual,
            pairs_root=paths.pairs,
            diarization_root=paths.diarization,
            qwen3_asr_root=paths.asr,
            output_root=paths.h3,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    result = {**plan, "stage_results": stage_results}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
