from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import (
    PrecomputedAudioMediaBackend,
    PrecomputedEmbeddingBackend,
)
from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    TranscriptSegment,
    build_audio_clip_binding_dataset,
    load_clip_bindings,
    load_visual_clip_inputs,
)
from r2v_data_v2.h3.audio_export import (
    export_h3_audio_dataset,
    publish_audio_pair_dataset,
)
from r2v_data_v2.h3.audio_pairing import AudioPairingConfig
from r2v_data_v2.h3.audio_schemas import AudioStreamProvenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build R2V Audio/H3 V1 datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser("bind", help="Build canonical clip bindings")
    bind.add_argument("--run-root", type=Path, required=True)
    bind.add_argument("--visual-export-root", type=Path, required=True)
    bind.add_argument("--sidecar-root", type=Path, required=True)
    bind.add_argument("--precomputed-json", type=Path)
    bind.add_argument("--output-root", type=Path, required=True)
    bind.add_argument("--clip-id", action="append")
    bind.add_argument("--limit", type=int)
    bind.add_argument("--speech-merge-gap", type=float, default=0.08)
    bind.add_argument("--minimum-voice-duration", type=float, default=0.50)
    bind.add_argument("--overwrite", action="store_true")
    bind.add_argument("--dry-run", action="store_true")

    pair = subparsers.add_parser("pair", help="Build strict in/cross-pair samples")
    pair.add_argument("--audio-root", type=Path, required=True)
    pair.add_argument("--output-root", type=Path, required=True)
    pair.add_argument("--top-k", type=int, default=20)
    pair.add_argument("--face-threshold", type=float, default=0.70)
    pair.add_argument("--face-margin", type=float, default=0.04)
    pair.add_argument("--voice-threshold", type=float, default=0.75)
    pair.add_argument("--voice-margin", type=float, default=0.04)
    pair.add_argument("--text-threshold", type=float, default=0.30)
    pair.add_argument("--max-cross-pair-variants", type=int, default=1)
    pair.add_argument("--seed", type=int, default=0)
    pair.add_argument("--speech-open-tag", default="<d>")
    pair.add_argument("--speech-close-tag", default="</d>")
    pair.add_argument("--renderer-profile", default="h3_v1")
    pair.add_argument("--overwrite", action="store_true")
    pair.add_argument("--report-only", action="store_true")

    export = subparsers.add_parser("export-h3", help="Export r2v.h3.sample.1")
    export.add_argument("--audio-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    inspect = subparsers.add_parser("inspect", help="Inspect canonical audio dataset")
    inspect.add_argument("--audio-root", type=Path, required=True)
    return parser


def _load_vectors(mapping: dict[str, str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for occurrence_id, path in mapping.items():
        with Path(path).open("rb") as stream:
            vectors[occurrence_id] = np.load(stream, allow_pickle=False)
    return vectors


def _embedding_backend(
    payload: dict[str, object],
    key: str,
) -> PrecomputedEmbeddingBackend:
    record = payload[key]
    if not isinstance(record, dict):
        raise TypeError(f"precomputed {key} record must be an object")
    vectors = record.get("vectors")
    if not isinstance(vectors, dict):
        raise TypeError(f"precomputed {key}.vectors must be an object")
    return PrecomputedEmbeddingBackend(
        _load_vectors({str(name): str(path) for name, path in vectors.items()}),
        model_identifier=str(record["model_identifier"]),
        checkpoint_sha256=(
            str(record["checkpoint_sha256"])
            if record.get("checkpoint_sha256") is not None
            else None
        ),
    )


def _bind(arguments: argparse.Namespace) -> dict[str, object]:
    allowlist = set(arguments.clip_id) if arguments.clip_id else None
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("--limit must be positive")
    if arguments.dry_run:
        clips = load_visual_clip_inputs(
            run_root=arguments.run_root,
            visual_export_root=arguments.visual_export_root,
            clip_allowlist=allowlist,
            limit=arguments.limit,
        )
        return {"dry_run": True, "selected_clip_ids": [item.clip_uid for item in clips]}
    if arguments.precomputed_json is None:
        raise ValueError("--precomputed-json is required unless --dry-run is used")
    payload = json.loads(arguments.precomputed_json.read_text(encoding="utf-8"))
    audio = payload["audio"]
    media = PrecomputedAudioMediaBackend(
        full_audio_by_clip={
            str(key): Path(value) for key, value in audio["full_audio_by_clip"].items()
        },
        voice_audio_by_occurrence={
            str(key): Path(value)
            for key, value in audio["voice_audio_by_occurrence"].items()
        },
        stream_by_clip={
            str(key): AudioStreamProvenance.model_validate(value)
            for key, value in audio["stream_by_clip"].items()
        },
    )
    face = _embedding_backend(payload, "face_embeddings")
    voice = _embedding_backend(payload, "voice_embeddings")
    text = (
        _embedding_backend(payload, "text_embeddings")
        if "text_embeddings" in payload
        else None
    )
    transcript_by_clip = {
        str(clip_uid): [
            TranscriptSegment(
                start_time=float(segment["start_time"]),
                end_time=float(segment["end_time"]),
                text=str(segment["text"]),
                status=str(segment.get("status", "auto")),
            )
            for segment in segments
        ]
        for clip_uid, segments in payload.get("transcripts", {}).items()
    }
    outputs = build_audio_clip_binding_dataset(
        run_root=arguments.run_root,
        visual_export_root=arguments.visual_export_root,
        sidecar_root=arguments.sidecar_root,
        output_root=arguments.output_root,
        audio_backend=media,
        face_backend=face,
        speaker_backend=voice,
        text_backend=text,
        transcript_by_clip=transcript_by_clip,
        config=AudioBindingProductionConfig(
            speech_merge_gap_seconds=arguments.speech_merge_gap,
            minimum_voice_reference_duration_seconds=(
                arguments.minimum_voice_duration
            ),
        ),
        clip_allowlist=allowlist,
        limit=arguments.limit,
        overwrite=arguments.overwrite,
    )
    return {"clip_binding_count": len(outputs), "output_root": str(arguments.output_root)}


def _pair(arguments: argparse.Namespace) -> dict[str, object]:
    config = AudioPairingConfig(
        top_k=arguments.top_k,
        face_strict_threshold=arguments.face_threshold,
        face_margin_threshold=arguments.face_margin,
        voice_strict_threshold=arguments.voice_threshold,
        voice_margin_threshold=arguments.voice_margin,
        text_threshold=arguments.text_threshold,
        max_cross_pair_variants_per_target=arguments.max_cross_pair_variants,
        deterministic_seed=arguments.seed,
        speech_open_tag=arguments.speech_open_tag,
        speech_close_tag=arguments.speech_close_tag,
        renderer_profile=arguments.renderer_profile,
    )
    return publish_audio_pair_dataset(
        audio_binding_root=arguments.audio_root,
        output_root=arguments.output_root,
        config=config,
        overwrite=arguments.overwrite,
        report_only=arguments.report_only,
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.command == "bind":
        result = _bind(arguments)
    elif arguments.command == "pair":
        result = _pair(arguments)
    elif arguments.command == "export-h3":
        samples = export_h3_audio_dataset(
            audio_root=arguments.audio_root,
            output_root=arguments.output_root,
            overwrite=arguments.overwrite,
        )
        result = {"sample_count": len(samples), "output_root": str(arguments.output_root)}
    else:
        bindings = load_clip_bindings(arguments.audio_root / "clip_bindings.jsonl")
        pair_path = arguments.audio_root / "pair_samples.jsonl"
        pair_count = sum(1 for line in pair_path.read_text("utf-8").splitlines() if line)
        failures_path = arguments.audio_root / "failures.jsonl"
        result = {
            "clip_binding_count": len(bindings),
            "pair_sample_count": pair_count,
            "failed_clip_count": sum(
                1
                for line in failures_path.read_text("utf-8").splitlines()
                if line
            )
            if failures_path.is_file()
            else 0,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
