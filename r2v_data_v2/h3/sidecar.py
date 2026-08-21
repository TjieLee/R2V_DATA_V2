from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

from r2v_data_v2.h3.backends import (
    AudioBindingEvidenceBackend,
    VoiceReferenceBackend,
)
from r2v_data_v2.h3.fusion import AudioBindingPolicy, build_audio_binding_sidecar
from r2v_data_v2.h3.schemas import (
    AudioBindingRunSummary,
    AudioBindingSidecar,
    H3TaskSpecification,
)
from r2v_data_v2.v3.schemas import ClipRecord

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _model_json(value: object) -> str:
    return value.model_dump_json(  # type: ignore[attr-defined]
        indent=2,
        exclude_none=False,
    )


def _validate_roots(run_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = run_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("audio sidecar output must be separate from source run_root")
    if not (source / "run.json").is_file() or not (source / "clips").is_dir():
        raise ValueError("source run_root is not an initialized V3 run")
    return source, output


def _failed_sidecar(
    clip: ClipRecord,
    *,
    source_run_root: str,
    reason: str,
) -> AudioBindingSidecar:
    return AudioBindingSidecar(
        clip_uid=clip.clip_uid,
        source_run_root=source_run_root,
        source_video_path=clip.source.video_path,
        status="failed",
        reason=reason,
    )


def _summary(
    source_run_root: str,
    records: list[AudioBindingSidecar],
) -> AudioBindingRunSummary:
    status_counts = {
        status: sum(record.status == status for record in records)
        for status in ("ready", "ineligible", "failed")
    }
    binding_counts = {
        status: sum(
            binding.status == status
            for record in records
            for binding in record.bindings
        )
        for status in ("bound", "overlap", "offscreen", "ambiguous", "no_speech")
    }
    return AudioBindingRunSummary(
        source_run_root=source_run_root,
        clip_count=len(records),
        ready_count=status_counts["ready"],
        ineligible_count=status_counts["ineligible"],
        failed_count=status_counts["failed"],
        bound_interval_count=binding_counts["bound"],
        overlap_interval_count=binding_counts["overlap"],
        offscreen_interval_count=binding_counts["offscreen"],
        ambiguous_interval_count=binding_counts["ambiguous"],
        no_speech_interval_count=binding_counts["no_speech"],
        voice_reference_count=sum(len(record.voice_references) for record in records),
    )


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"audio sidecar output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    published = False
    try:
        temporary.replace(destination)
        published = True
    finally:
        if published:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)


def build_audio_binding_sidecar_run(
    *,
    run_root: Path,
    output_root: Path,
    backend: AudioBindingEvidenceBackend,
    voice_reference_backend: VoiceReferenceBackend | None = None,
    task: H3TaskSpecification | None = None,
    policy: AudioBindingPolicy | None = None,
    overwrite: bool = False,
) -> AudioBindingRunSummary:
    source, destination = _validate_roots(run_root, output_root)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"audio sidecar output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[AudioBindingSidecar] = []
    try:
        clip_paths = sorted((source / "clips").glob("*/clip.json"))
        for clip_path in clip_paths:
            clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
            if (
                _SAFE_COMPONENT.fullmatch(clip.clip_uid) is None
                or clip.clip_uid in {".", ".."}
                or clip_path.parent.name != clip.clip_uid
            ):
                raise ValueError("V3 clip_uid is not a safe matching path component")
            try:
                evidence = backend.collect(clip)
                record = build_audio_binding_sidecar(
                    clip,
                    evidence,
                    source_run_root=str(source),
                    voice_reference_backend=voice_reference_backend,
                    task=task,
                    policy=policy,
                )
            except Exception as exc:  # noqa: BLE001 - preserve neighboring cases
                record = _failed_sidecar(
                    clip,
                    source_run_root=str(source),
                    reason=f"evidence_or_fusion_failed:{type(exc).__name__}:{exc}",
                )
            records.append(record)
            destination_path = (
                temporary / "clips" / clip.clip_uid / "audio_binding.json"
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(_model_json(record) + "\n", encoding="utf-8")
        jsonl = temporary / "audio_bindings.jsonl"
        jsonl.write_text(
            "".join(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        summary = _summary(str(source), records)
        (temporary / "summary.json").write_text(
            _model_json(summary) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
