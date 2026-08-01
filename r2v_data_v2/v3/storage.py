from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.schemas import (
    AnnotationState,
    BackgroundReferenceState,
    ClipRecord,
    ClipSource,
    CoverageState,
    DatasetRecord,
    DatasetReference,
    DatasetSample,
    ExportState,
    FailureRecord,
    FinalizedBackgroundReference,
    FinalizedEntityReference,
    InstructionState,
    PairingState,
    ReferenceFinalizationState,
    ReferencesState,
    RunRecord,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_EXPORT_TOKEN = re.compile(r"<ref_(subject|object|group|bg)_(\d+)>")
_UTC = getattr(datetime_module, "UTC", timezone.utc)  # noqa: UP017 - Python 3.9 CI
_SECTION_INVALIDATIONS = {
    "annotation": (
        "coverage",
        "references",
        "pairing",
        "reference_finalization",
        "instruction",
        "export",
    ),
    "frames": (
        "coverage",
        "references",
        "pairing",
        "reference_finalization",
        "instruction",
        "export",
    ),
    "masks": (
        "coverage",
        "references",
        "pairing",
        "reference_finalization",
        "instruction",
        "export",
    ),
    "coverage": (
        "references",
        "pairing",
        "reference_finalization",
        "instruction",
        "export",
    ),
    "references": (
        "pairing",
        "reference_finalization",
        "instruction",
        "export",
    ),
    "pairing": ("reference_finalization", "instruction", "export"),
    "reference_finalization": ("export",),
    "instruction": ("export",),
}


def utc_now() -> str:
    return datetime.now(_UTC).isoformat()


def _safe_component(value: str, field_name: str) -> str:
    if _SAFE_COMPONENT.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path component")
    return value


def _cleared_section_value(section: str) -> object:
    if section == "references":
        return ReferencesState()
    if section == "export":
        return ExportState()
    return None


def _section_updates_after_change(section: str) -> dict[str, object]:
    return {
        downstream: _cleared_section_value(downstream)
        for downstream in _SECTION_INVALIDATIONS.get(section, ())
    }


def _model_dict(value: object) -> dict[str, object]:
    dumped = value.model_dump(mode="json")  # type: ignore[attr-defined]
    if not isinstance(dumped, dict):
        raise TypeError("schema serialization must produce a JSON object")
    return dumped


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class RunStorage:
    def __init__(self, config: V3Config) -> None:
        config.validate()
        self.config = config
        self.root = config.resolved_run_root

    @property
    def run_path(self) -> Path:
        return self.root / "run.json"

    @property
    def failures_path(self) -> Path:
        return self.root / "failures.jsonl"

    def initialize(
        self,
        *,
        git_commit: str,
        created_at: str | None = None,
    ) -> RunRecord:
        expected = RunRecord(
            run_id=_safe_component(self.root.name, "run_id"),
            created_at=created_at or utc_now(),
            git_commit=git_commit,
            config_hash=self.config.fingerprint(),
            model_identifiers=self.config.model_identifiers(),
            source_manifest_path=str(
                self.config.dataset_json.expanduser().resolve(strict=False)
            ),
            counts={},
        )
        if self.run_path.is_file():
            existing = self.read_run()
            identity_fields = (
                existing.run_id == expected.run_id,
                existing.git_commit == expected.git_commit,
                existing.config_hash == expected.config_hash,
                existing.model_identifiers == expected.model_identifiers,
                existing.source_manifest_path == expected.source_manifest_path,
            )
            if not all(identity_fields):
                raise ValueError(
                    "existing run.json does not match the requested V3 run"
                )
            return existing
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(
                "run_root is non-empty but has no matching run.json"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.run_path, _model_dict(expected))
        return expected

    def _require_initialized(self) -> None:
        if not self.run_path.is_file():
            raise FileNotFoundError("initialize V3 run storage before writing artifacts")

    def _require_clip(self, clip_uid: str) -> None:
        if not self.clip_path(clip_uid).is_file():
            raise FileNotFoundError(f"clip.json does not exist for {clip_uid}")

    def read_run(self) -> RunRecord:
        return RunRecord.model_validate_json(self.run_path.read_text(encoding="utf-8"))

    def update_run_counts(self, counts: dict[str, int]) -> RunRecord:
        if any(value < 0 for value in counts.values()):
            raise ValueError("run counts must be non-negative")
        current = self.read_run()
        updated = current.model_copy(update={"counts": dict(counts)})
        write_json_atomic(self.run_path, _model_dict(updated))
        return updated

    def update_stage_counts(
        self,
        stage: str,
        counts: dict[str, int],
    ) -> RunRecord:
        _safe_component(stage, "stage")
        current = self.read_run()
        merged = dict(current.counts)
        merged.update(
            {
                f"{stage}.{name}": value
                for name, value in counts.items()
            }
        )
        return self.update_run_counts(merged)

    def clip_dir(self, clip_uid: str) -> Path:
        return self.root / "clips" / _safe_component(clip_uid, "clip_uid")

    def clip_path(self, clip_uid: str) -> Path:
        return self.clip_dir(clip_uid) / "clip.json"

    def create_clip(self, *, clip_uid: str, source: ClipSource) -> ClipRecord:
        self._require_initialized()
        source_path = Path(source.video_path).expanduser()
        if not source_path.is_absolute():
            raise ValueError("source.video_path must be an absolute path")
        resolved_source_path = source_path.resolve(strict=False)
        dataset_root = v3_config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
        if dataset_root not in resolved_source_path.parents:
            raise ValueError(
                "source.video_path must be inside "
                "/mnt/workspace/public/dataset"
            )
        normalized_source = source.model_copy(
            update={"video_path": str(resolved_source_path)}
        )
        record = ClipRecord(clip_uid=clip_uid, source=normalized_source)
        path = self.clip_path(clip_uid)
        if path.is_file():
            existing = self.read_clip(clip_uid)
            if existing.clip_uid != clip_uid:
                raise ValueError(f"clip.json identity mismatch for {clip_uid}")
            if existing.source != normalized_source:
                raise ValueError(
                    f"existing clip source does not match for {clip_uid}"
                )
            return existing
        write_json_atomic(path, _model_dict(record))
        return record

    def read_clip(self, clip_uid: str) -> ClipRecord:
        return ClipRecord.model_validate_json(
            self.clip_path(clip_uid).read_text(encoding="utf-8")
        )

    def iter_clips(self) -> Iterator[ClipRecord]:
        clips_root = self.root / "clips"
        if not clips_root.is_dir():
            return
        for path in sorted(clips_root.glob("*/clip.json")):
            yield ClipRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _replace_section(
        self,
        clip_uid: str,
        section: str,
        value: object,
    ) -> ClipRecord:
        current = self.read_clip(clip_uid)
        if getattr(current, section) == value:
            return current
        updates = {
            section: value,
            **_section_updates_after_change(section),
        }
        updated = current.model_copy(update=updates)
        validated = ClipRecord.model_validate(updated.model_dump(mode="json"))
        self._invalidate_artifacts_after_change(clip_uid, section)
        write_json_atomic(self.clip_path(clip_uid), _model_dict(validated))
        return validated

    def _invalidate_artifacts_after_change(
        self,
        clip_uid: str,
        section: str,
    ) -> None:
        if section == "annotation":
            frames_dir = self.frames_dir(clip_uid)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
        if section in {"annotation", "frames"}:
            self.masks_path(clip_uid).unlink(missing_ok=True)
        if "reference_finalization" in _SECTION_INVALIDATIONS.get(section, ()):
            finalized = self.clip_dir(clip_uid) / "finalized"
            if finalized.exists():
                shutil.rmtree(finalized)

    def write_annotation(
        self,
        clip_uid: str,
        value: AnnotationState,
    ) -> ClipRecord:
        return self._replace_section(clip_uid, "annotation", value)

    def write_coverage(self, clip_uid: str, value: CoverageState) -> ClipRecord:
        return self._replace_section(clip_uid, "coverage", value)

    def clear_coverage(self, clip_uid: str) -> ClipRecord:
        current = self.read_clip(clip_uid)
        updates = {
            "coverage": None,
            **_section_updates_after_change("coverage"),
        }
        updated = current.model_copy(update=updates)
        validated = ClipRecord.model_validate(
            updated.model_dump(mode="json")
        )
        if validated != current:
            self._invalidate_artifacts_after_change(clip_uid, "coverage")
            write_json_atomic(
                self.clip_path(clip_uid),
                _model_dict(validated),
            )
        return validated

    def write_references(
        self,
        clip_uid: str,
        value: ReferencesState,
    ) -> ClipRecord:
        return self._replace_section(clip_uid, "references", value)

    def write_pairing(self, clip_uid: str, value: PairingState) -> ClipRecord:
        return self._replace_section(clip_uid, "pairing", value)

    def write_references_and_pairing(
        self,
        clip_uid: str,
        references: ReferencesState,
        pairing: PairingState,
    ) -> ClipRecord:
        current = self.read_clip(clip_uid)
        validated_references = ReferencesState.model_validate(
            references.model_dump(mode="json")
        )
        validated_pairing = PairingState.model_validate(
            pairing.model_dump(mode="json")
        )
        updated = current.model_copy(
            update={
                "references": validated_references,
                "pairing": validated_pairing,
                "reference_finalization": None,
                "instruction": None,
                "export": ExportState(),
            }
        )
        validated = ClipRecord.model_validate(
            updated.model_dump(mode="json")
        )
        if validated != current:
            self._invalidate_artifacts_after_change(clip_uid, "pairing")
            write_json_atomic(
                self.clip_path(clip_uid),
                _model_dict(validated),
            )
        return validated

    def write_reference_finalization(
        self,
        clip_uid: str,
        value: ReferenceFinalizationState,
    ) -> ClipRecord:
        return self._replace_section(
            clip_uid,
            "reference_finalization",
            value,
        )

    def write_instruction(
        self,
        clip_uid: str,
        value: InstructionState,
    ) -> ClipRecord:
        return self._replace_section(clip_uid, "instruction", value)

    def write_export(self, clip_uid: str, value: ExportState) -> ClipRecord:
        return self._replace_section(clip_uid, "export", value)

    def write_masks(
        self,
        clip_uid: str,
        value: TrackedMasksArtifact,
    ) -> Path:
        self._require_clip(clip_uid)
        if value.clip_uid != clip_uid:
            raise ValueError("mask artifact clip_uid does not match destination")
        destination = self.masks_path(clip_uid)
        if destination.is_file():
            existing = TrackedMasksArtifact.model_validate_json(
                destination.read_text(encoding="utf-8")
            )
            if existing == value:
                return destination
        current = self.read_clip(clip_uid)
        invalidated = current.model_copy(
            update=_section_updates_after_change("masks")
        )
        validated = ClipRecord.model_validate(invalidated.model_dump(mode="json"))
        if validated != current:
            self._invalidate_artifacts_after_change(clip_uid, "masks")
            write_json_atomic(self.clip_path(clip_uid), _model_dict(validated))
        write_json_atomic(destination, _model_dict(value))
        return destination

    def frames_dir(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        return self.clip_dir(clip_uid) / "frames"

    def frames_manifest_path(self, clip_uid: str) -> Path:
        return self.frames_dir(clip_uid) / "frames.json"

    def masks_path(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        return self.clip_dir(clip_uid) / "masks.rle.json"

    def read_frames(self, clip_uid: str) -> SampledFramesArtifact:
        return SampledFramesArtifact.model_validate_json(
            self.frames_manifest_path(clip_uid).read_text(encoding="utf-8")
        )

    def read_masks(self, clip_uid: str) -> TrackedMasksArtifact:
        return TrackedMasksArtifact.model_validate_json(
            self.masks_path(clip_uid).read_text(encoding="utf-8")
        )

    def prepare_frames_publication(self, clip_uid: str) -> None:
        self._require_clip(clip_uid)
        current = self.read_clip(clip_uid)
        invalidated = current.model_copy(
            update=_section_updates_after_change("frames")
        )
        validated = ClipRecord.model_validate(
            invalidated.model_dump(mode="json")
        )
        self._invalidate_artifacts_after_change(clip_uid, "frames")
        self.frames_manifest_path(clip_uid).unlink(missing_ok=True)
        if validated != current:
            write_json_atomic(
                self.clip_path(clip_uid),
                _model_dict(validated),
            )

    def prepare_masks_publication(self, clip_uid: str) -> None:
        self._require_clip(clip_uid)
        current = self.read_clip(clip_uid)
        invalidated = current.model_copy(
            update=_section_updates_after_change("masks")
        )
        validated = ClipRecord.model_validate(
            invalidated.model_dump(mode="json")
        )
        self._invalidate_artifacts_after_change(clip_uid, "masks")
        self.masks_path(clip_uid).unlink(missing_ok=True)
        if validated != current:
            write_json_atomic(
                self.clip_path(clip_uid),
                _model_dict(validated),
            )

    def frame_path(self, clip_uid: str, frame_slot: int) -> Path:
        self._require_clip(clip_uid)
        if not 0 <= frame_slot < self.config.frames.count:
            raise ValueError("frame_slot is outside configured frame range")
        destination = self.clip_dir(clip_uid) / "frames" / f"{frame_slot:02d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def selected_path(self, clip_uid: str, filename: str) -> Path:
        self._require_clip(clip_uid)
        safe_name = _safe_component(filename, "selected filename")
        destination = self.clip_dir(clip_uid) / "selected" / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def selected_entity_path(self, clip_uid: str, entity_id: str) -> Path:
        if re.fullmatch(r"e[1-9]\d*", entity_id) is None:
            raise ValueError("entity_id must match e[1-9]\\d*")
        return self.selected_path(clip_uid, f"{entity_id}.png")

    def pair_output_temporary_path(
        self,
        clip_uid: str,
        entity_id: str,
    ) -> Path:
        directory = self.selected_entity_path(clip_uid, entity_id).parent
        return directory / (
            f".tmp-pair-{entity_id}-{uuid.uuid4().hex}.png"
        )

    def pair_output_backup_path(
        self,
        clip_uid: str,
        entity_id: str,
    ) -> Path:
        directory = self.selected_entity_path(clip_uid, entity_id).parent
        return directory / (
            f".backup-pair-{entity_id}-{uuid.uuid4().hex}.png"
        )

    def finalized_dir(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        return self.clip_dir(clip_uid) / "finalized"

    def finalized_entity_path(self, clip_uid: str, entity_id: str) -> Path:
        if re.fullmatch(r"e[1-9]\d*", entity_id) is None:
            raise ValueError("entity_id must match e[1-9]\\d*")
        destination = self.finalized_dir(clip_uid) / f"{entity_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def finalize_output_temporary_path(
        self,
        clip_uid: str,
        entity_id: str,
    ) -> Path:
        directory = self.finalized_entity_path(clip_uid, entity_id).parent
        return directory / (
            f".tmp-reference-finalize-{entity_id}-{uuid.uuid4().hex}.png"
        )

    def finalize_output_backup_path(
        self,
        clip_uid: str,
        entity_id: str,
    ) -> Path:
        directory = self.finalized_entity_path(clip_uid, entity_id).parent
        return directory / (
            f".backup-reference-finalize-{entity_id}-{uuid.uuid4().hex}.png"
        )

    def pair_debug_dir(self, clip_uid: str, entity_id: str) -> Path:
        if not self.config.debug.save_diagnostics:
            raise RuntimeError("pair debug artifact saving is disabled")
        self.selected_entity_path(clip_uid, entity_id)
        destination = (
            self.clip_dir(clip_uid) / "debug" / "pair" / entity_id
        )
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def background_dir(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        return self.clip_dir(clip_uid) / "background"

    def prepare_background_publication(self, clip_uid: str) -> Path:
        destination = self.background_dir(clip_uid)
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def background_source_mask_path(
        self,
        clip_uid: str,
        sha256: str,
    ) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("background source mask sha256 must be lowercase hex")
        return (
            self.prepare_background_publication(clip_uid)
            / f"source_mask_{sha256}.png"
        )

    def background_generation_mask_path(
        self,
        clip_uid: str,
        sha256: str,
    ) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("background generation mask sha256 must be lowercase hex")
        return (
            self.prepare_background_publication(clip_uid)
            / f"generation_mask_{sha256}.png"
        )

    def selected_background_output_path(self, clip_uid: str) -> Path:
        return self.selected_path(clip_uid, "bg_removed.png")

    def remove_output_temporary_path(self, clip_uid: str) -> Path:
        directory = self.selected_background_output_path(clip_uid).parent
        return directory / f".tmp-remove-{uuid.uuid4().hex}.png"

    def remove_output_backup_path(self, clip_uid: str) -> Path:
        directory = self.selected_background_output_path(clip_uid).parent
        return directory / f".backup-bg_removed-{uuid.uuid4().hex}.png"

    def remove_debug_dir(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        if not (
            self.config.debug.save_diagnostics
            or self.config.remove.save_rejected_candidates
        ):
            raise RuntimeError("remove debug artifact saving is disabled")
        destination = self.clip_dir(clip_uid) / "debug" / "remove"
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def cleanup_remove_artifacts(
        self,
        clip_uid: str,
        *,
        keep_generation_mask: Path | None = None,
    ) -> None:
        background = self.background_dir(clip_uid)
        resolved_keep = (
            None
            if keep_generation_mask is None
            else keep_generation_mask.resolve(strict=False)
        )
        if resolved_keep is not None and (
            resolved_keep.parent != background.resolve(strict=False)
            or not resolved_keep.name.startswith("generation_mask_")
        ):
            raise ValueError(
                "kept generation mask must be inside background_dir"
            )
        if background.is_dir():
            for path in background.glob("generation_mask_*.png"):
                if path.resolve(strict=False) != resolved_keep:
                    path.unlink(missing_ok=True)
            for path in background.glob(".tmp-remove-*.png"):
                path.unlink(missing_ok=True)
        selected = self.clip_dir(clip_uid) / "selected"
        if selected.is_dir():
            for pattern in (
                ".tmp-remove-*.png",
                ".backup-bg_removed-*.png",
            ):
                for path in selected.glob(pattern):
                    path.unlink(missing_ok=True)

    def cleanup_pair_artifacts(self, clip_uid: str) -> None:
        selected = self.clip_dir(clip_uid) / "selected"
        if not selected.is_dir():
            return
        for pattern in (
            ".tmp-pair-*.png",
            ".backup-pair-*.png",
        ):
            for path in selected.glob(pattern):
                path.unlink(missing_ok=True)

    def cleanup_reference_finalize_artifacts(
        self,
        clip_uid: str,
        *,
        remove_published: bool = False,
    ) -> None:
        directory = self.finalized_dir(clip_uid)
        if not directory.is_dir():
            return
        if remove_published:
            shutil.rmtree(directory)
            return
        for pattern in (
            ".tmp-reference-finalize-*.png",
            ".backup-reference-finalize-*.png",
        ):
            for path in directory.glob(pattern):
                path.unlink(missing_ok=True)
        if not any(directory.iterdir()):
            directory.rmdir()

    def cleanup_background_artifacts(
        self,
        clip_uid: str,
        *,
        keep: Path | None = None,
    ) -> None:
        directory = self.background_dir(clip_uid)
        if not directory.is_dir():
            return
        resolved_keep = None if keep is None else keep.resolve(strict=False)
        if resolved_keep is not None and resolved_keep.parent != directory.resolve():
            raise ValueError(
                "kept background artifact must be inside background_dir"
            )
        for path in directory.glob("source_mask_*.png"):
            if path.resolve(strict=False) != resolved_keep:
                path.unlink(missing_ok=True)
        for path in directory.glob(".tmp-*.png"):
            path.unlink(missing_ok=True)
        if not any(directory.iterdir()):
            directory.rmdir()

    def debug_path(self, clip_uid: str, filename: str) -> Path:
        self._require_clip(clip_uid)
        if not self.config.debug.save_diagnostics:
            raise RuntimeError("debug artifact saving is disabled")
        safe_name = _safe_component(filename, "debug filename")
        destination = self.clip_dir(clip_uid) / "debug" / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def segment_debug_dir(self, clip_uid: str) -> Path:
        self._require_clip(clip_uid)
        if not (
            self.config.debug.save_diagnostics
            or self.config.sam3.save_debug_overlays
        ):
            raise RuntimeError("segment debug artifact saving is disabled")
        destination = self.clip_dir(clip_uid) / "debug" / "segment"
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def relative_artifact_path(self, path: Path) -> str:
        resolved = path.expanduser().resolve(strict=False)
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("run artifact must remain inside run_root") from exc

    def append_failure(
        self,
        *,
        stage: str,
        reason: str,
        clip_uid: str | None = None,
        details: dict[str, object] | None = None,
        created_at: str | None = None,
    ) -> FailureRecord:
        self._require_initialized()
        record = FailureRecord(
            clip_uid=clip_uid,
            stage=stage,
            reason=reason,
            created_at=created_at or utc_now(),
            details=details or {},
        )
        _append_jsonl(self.failures_path, _model_dict(record))
        return record


class DatasetExporter:
    def __init__(self, config: V3Config, storage: RunStorage) -> None:
        config.validate()
        if storage.root != config.resolved_run_root:
            raise ValueError("storage run_root does not match exporter configuration")
        self.config = config
        self.storage = storage
        self.destination = config.resolved_export_root

    def export(
        self,
        *,
        overwrite: bool = False,
        created_at: str | None = None,
    ) -> DatasetRecord:
        if self.destination.exists() and not overwrite:
            raise FileExistsError(
                f"dataset_root already exists: {self.destination}"
            )
        run = self.storage.read_run()
        temporary = self.destination.with_name(
            f".{self.destination.name}.tmp-{uuid.uuid4().hex}"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            references_root = temporary / "references"
            references_root.mkdir(parents=True)
            samples: list[DatasetSample] = []
            reference_count = 0
            for clip in self.storage.iter_clips():
                if not clip.export.accepted:
                    continue
                sample = self._export_clip(clip, temporary)
                samples.append(sample)
                reference_count += len(sample.references)
            sample_records = [_model_dict(sample) for sample in samples]
            _write_jsonl_atomic(temporary / "samples.jsonl", sample_records)
            dataset = DatasetRecord(
                dataset_version=_safe_component(
                    self.destination.name,
                    "dataset_version",
                ),
                created_at=created_at or utc_now(),
                git_commit=run.git_commit,
                config_hash=run.config_hash,
                annotation_model=Path(self.config.qwen.annotation.model).name,
                background_remove_backend=self.config.remove.backend,
                sample_count=len(samples),
                reference_count=reference_count,
            )
            write_json_atomic(temporary / "dataset.json", _model_dict(dataset))
            self._validate_export_tree(temporary, samples)
            self._publish(temporary, overwrite=overwrite)
            return dataset
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _export_clip(
        self,
        clip: ClipRecord,
        temporary: Path,
    ) -> DatasetSample:
        if (
            clip.annotation is None
            or clip.annotation.status != "ready"
            or clip.coverage is None
            or not clip.coverage.passed
            or clip.pairing is None
            or clip.pairing.status != "ready"
            or clip.instruction is None
            or clip.instruction.status != "ready"
        ):
            raise ValueError(
                f"accepted clip is missing ready V3 state: {clip.clip_uid}"
            )
        retained = set(clip.pairing.retained_entity_ids)
        if not retained.intersection(clip.coverage.qualifying_entity_ids):
            raise ValueError(
                f"accepted clip has no bound qualifying entity: {clip.clip_uid}"
            )
        references_by_id = {
            reference.entity_id: reference
            for reference in clip.references.entities
            if reference.status == "ready"
        }
        finalization = clip.reference_finalization
        if finalization is not None and finalization.status != "ready":
            raise ValueError(
                f"accepted clip finalization is not ready: {clip.clip_uid}"
            )
        finalized_by_id = {
            entity.entity_id: entity
            for entity in (
                finalization.entities if finalization is not None else []
            )
        }
        destination_dir = temporary / "references" / _safe_component(
            clip.clip_uid,
            "sample_id",
        )
        dataset_references: list[DatasetReference] = []
        for entity_id in clip.pairing.retained_entity_ids:
            reference = references_by_id.get(entity_id)
            if reference is None:
                raise ValueError(
                    f"retained entity has no ready reference: {entity_id}"
                )
            token = clip.pairing.tokens[entity_id]
            if finalization is None:
                relative_path = (
                    Path("references")
                    / clip.clip_uid
                    / self._filename_for_token(token)
                )
                self._copy_png(
                    self._resolve_run_artifact(reference.image_path),
                    temporary / relative_path,
                    background=False,
                )
                dataset_reference = DatasetReference(
                    token=token,
                    type="entity",
                    entity_id=entity_id,
                    scope=reference.reference_scope,
                    visible_region=reference.visible_region,
                    image_path=relative_path.as_posix(),
                    source_frame_index=reference.source_frame_index,
                    synthetic=False,
                )
            else:
                finalized = finalized_by_id.get(entity_id)
                if finalized is None:
                    raise ValueError(
                        f"retained entity has no finalized reference: {entity_id}"
                    )
                dataset_reference = self._export_finalized_entity(
                    clip_uid=clip.clip_uid,
                    source_frame_index=reference.source_frame_index,
                    metadata=finalized,
                    temporary=temporary,
                )
            dataset_references.append(dataset_reference)
        if clip.pairing.background_token is not None:
            background = clip.references.background
            if background is None or background.status not in {
                "clean_raw",
                "ready_removed",
            }:
                raise ValueError(
                    f"paired background is not ready: {clip.clip_uid}"
                )
            dataset_references.append(
                self._export_background(
                    clip_uid=clip.clip_uid,
                    token=clip.pairing.background_token,
                    background=background,
                    finalized=(
                        finalization.background
                        if finalization is not None
                        else None
                    ),
                    temporary=temporary,
                )
            )
        if not destination_dir.is_dir():
            raise ValueError(f"accepted clip exported no references: {clip.clip_uid}")
        return DatasetSample(
            sample_id=clip.clip_uid,
            target_video=clip.source.video_path,
            t2v_caption=clip.annotation.t2v_caption,
            r2v_instruction=clip.instruction.r2v_instruction,
            references=dataset_references,
            source={
                "parent_video_id": clip.source.parent_video_id,
                "clip_suffix": clip.source.clip_suffix,
            },
        )

    def _export_finalized_entity(
        self,
        *,
        clip_uid: str,
        source_frame_index: int | None,
        metadata: FinalizedEntityReference,
        temporary: Path,
    ) -> DatasetReference:
        if metadata.status != "ready":
            raise ValueError("export requires a ready finalized entity")
        if source_frame_index is None:
            raise ValueError("ready entity reference is missing source frame index")
        filename = self._filename_for_token(metadata.token)
        relative_path = Path("references") / clip_uid / filename
        source_relative_path = relative_path.with_name(
            f"{relative_path.stem}.source.png"
        )
        self._copy_exact_png(
            self._resolve_run_artifact(metadata.source_image_path),
            temporary / source_relative_path,
            expected_sha256=metadata.source_sha256,
        )
        self._copy_exact_png(
            self._resolve_run_artifact(metadata.normalized_image_path),
            temporary / relative_path,
            expected_sha256=metadata.normalized_sha256,
        )
        return DatasetReference(
            token=metadata.token,
            type="entity",
            entity_id=metadata.entity_id,
            scope=metadata.reference_scope,
            visible_region=metadata.visible_region,
            image_path=relative_path.as_posix(),
            source_frame_index=source_frame_index,
            synthetic=False,
            source_image_path=source_relative_path.as_posix(),
            width=metadata.normalized_width,
            height=metadata.normalized_height,
            source_width=metadata.source_width,
            source_height=metadata.source_height,
            content_bbox_xyxy=metadata.normalized_content_bbox_xyxy,
            source_content_bbox_xyxy=metadata.source_content_bbox_xyxy,
            scale_factor=metadata.scale_factor,
            foreground_ratio=metadata.normalized_foreground_ratio,
            source_foreground_ratio=metadata.source_foreground_ratio,
            quality_tier=metadata.quality_tier,
            quality_flags=list(metadata.quality_flags),
            normalization_profile=metadata.normalization_profile,
            sha256=metadata.normalized_sha256,
            source_sha256=metadata.source_sha256,
        )

    def _export_background(
        self,
        *,
        clip_uid: str,
        token: str,
        background: BackgroundReferenceState,
        finalized: FinalizedBackgroundReference | None,
        temporary: Path,
    ) -> DatasetReference:
        relative_path = (
            Path("references") / clip_uid / self._filename_for_token(token)
        )
        source = self._resolve_background_artifact(
            clip_uid,
            background.output_image_path,
        )
        if finalized is not None and (
            self.storage.relative_artifact_path(source) != finalized.image_path
            or _sha256_file(source) != finalized.sha256
        ):
            raise ValueError("background finalization metadata is stale")
        destination = temporary / relative_path
        self._copy_png(source, destination, background=True)
        exported_sha256 = _sha256_file(destination)
        return DatasetReference(
            token=token,
            type="background",
            entity_id=None,
            scope="scene",
            visible_region="whole",
            image_path=relative_path.as_posix(),
            source_frame_index=background.source_frame_index,
            synthetic=background.status == "ready_removed",
            width=finalized.width if finalized is not None else None,
            height=finalized.height if finalized is not None else None,
            sha256=exported_sha256 if finalized is not None else None,
        )

    def _resolve_background_artifact(
        self,
        clip_uid: str,
        value: str | None,
    ) -> Path:
        if value is None:
            raise ValueError("ready background is missing output_image_path")
        path = Path(value).expanduser()
        if not path.is_absolute() and path.parts[:1] == ("frames",):
            candidate = self.storage.clip_dir(clip_uid) / path
        else:
            candidate = path if path.is_absolute() else self.storage.root / path
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.storage.root)
        except ValueError as exc:
            raise ValueError("background image must remain inside run_root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"background image is missing: {resolved}")
        return resolved

    def _resolve_run_artifact(self, value: str | None) -> Path:
        if value is None:
            raise ValueError("ready reference is missing image_path")
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else self.storage.root / path
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.storage.root)
        except ValueError as exc:
            raise ValueError("reference image must remain inside run_root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"reference image is missing: {resolved}")
        return resolved

    @staticmethod
    def _filename_for_token(token: str) -> str:
        match = _EXPORT_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid reference token: {token}")
        kind = "background" if match.group(1) == "bg" else match.group(1)
        return f"{kind}_{match.group(2)}.png"

    @staticmethod
    def _copy_exact_png(
        source: Path,
        destination: Path,
        *,
        expected_sha256: str | None,
    ) -> None:
        if expected_sha256 is None:
            raise ValueError("finalized PNG is missing SHA256 metadata")
        if _sha256_file(source) != expected_sha256:
            raise ValueError("finalized PNG source SHA256 does not match metadata")
        with Image.open(source) as opened:
            opened.load()
            if opened.format != "PNG":
                raise ValueError("finalized entity artifact must be PNG")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _sha256_file(destination) != expected_sha256:
            raise ValueError("exact PNG export changed source bytes")

    @staticmethod
    def _copy_png(source: Path, destination: Path, *, background: bool) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            opened.load()
            if background:
                image = opened.convert("RGB")
            elif "A" in opened.getbands() or opened.mode == "RGB":
                image = opened.copy()
            elif "transparency" in opened.info:
                image = opened.convert("RGBA")
            else:
                image = opened.convert("RGB")
        image.save(destination, format="PNG")

    def _validate_export_tree(
        self,
        root: Path,
        samples: list[DatasetSample],
    ) -> None:
        if {path.name for path in root.iterdir()} != {
            "dataset.json",
            "samples.jsonl",
            "references",
        }:
            raise ValueError("dataset root contains unexpected top-level artifacts")
        expected = {
            path
            for sample in samples
            for reference in sample.references
            for path in (reference.image_path, reference.source_image_path)
            if path is not None
        }
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / "references").rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise ValueError("exported reference files do not match sample records")
        for relative_path in expected:
            resolved = (root / relative_path).resolve(strict=False)
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("exported reference path escapes dataset root") from exc
        for sample in samples:
            for reference in sample.references:
                if (
                    reference.sha256 is not None
                    and _sha256_file(root / reference.image_path)
                    != reference.sha256
                ):
                    raise ValueError("exported reference SHA256 is invalid")
                if reference.source_image_path is not None:
                    if reference.source_sha256 is None:
                        raise ValueError(
                            "source reference path requires source SHA256"
                        )
                    if (
                        _sha256_file(root / reference.source_image_path)
                        != reference.source_sha256
                    ):
                        raise ValueError("exported source reference SHA256 is invalid")

    def _publish(self, temporary: Path, *, overwrite: bool) -> None:
        if not self.destination.exists():
            temporary.replace(self.destination)
            return
        if not overwrite:
            raise FileExistsError(
                f"dataset_root already exists: {self.destination}"
            )
        backup = self.destination.with_name(
            f".{self.destination.name}.backup-{uuid.uuid4().hex}"
        )
        self.destination.replace(backup)
        published = False
        try:
            temporary.replace(self.destination)
            published = True
        finally:
            if published:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink(missing_ok=True)
            elif backup.exists() and not self.destination.exists():
                backup.replace(self.destination)
