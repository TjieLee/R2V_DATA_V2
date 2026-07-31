from __future__ import annotations

import datetime as datetime_module
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

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
    InstructionState,
    PairingState,
    ReferencesState,
    RunRecord,
    TrackedMasksArtifact,
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_EXPORT_TOKEN = re.compile(r"<ref_(subject|object|group|bg)_(\d+)>")
_UTC = getattr(datetime_module, "UTC", timezone.utc)  # noqa: UP017 - Python 3.9 CI


def utc_now() -> str:
    return datetime.now(_UTC).isoformat()


def _safe_component(value: str, field_name: str) -> str:
    if _SAFE_COMPONENT.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path component")
    return value


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

    def clip_dir(self, clip_uid: str) -> Path:
        return self.root / "clips" / _safe_component(clip_uid, "clip_uid")

    def clip_path(self, clip_uid: str) -> Path:
        return self.clip_dir(clip_uid) / "clip.json"

    def create_clip(self, *, clip_uid: str, source: ClipSource) -> ClipRecord:
        self._require_initialized()
        record = ClipRecord(clip_uid=clip_uid, source=source)
        path = self.clip_path(clip_uid)
        if path.is_file():
            existing = self.read_clip(clip_uid)
            if existing.clip_uid != clip_uid:
                raise ValueError(f"clip.json identity mismatch for {clip_uid}")
            if existing.source != source:
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
        updated = current.model_copy(update={section: value})
        validated = ClipRecord.model_validate(updated.model_dump(mode="json"))
        write_json_atomic(self.clip_path(clip_uid), _model_dict(validated))
        return validated

    def write_annotation(
        self,
        clip_uid: str,
        value: AnnotationState,
    ) -> ClipRecord:
        return self._replace_section(clip_uid, "annotation", value)

    def write_coverage(self, clip_uid: str, value: CoverageState) -> ClipRecord:
        return self._replace_section(clip_uid, "coverage", value)

    def write_references(
        self,
        clip_uid: str,
        value: ReferencesState,
    ) -> ClipRecord:
        return self._replace_section(clip_uid, "references", value)

    def write_pairing(self, clip_uid: str, value: PairingState) -> ClipRecord:
        return self._replace_section(clip_uid, "pairing", value)

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
        destination = self.clip_dir(clip_uid) / "masks.rle.json"
        write_json_atomic(destination, _model_dict(value))
        return destination

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

    def debug_path(self, clip_uid: str, filename: str) -> Path:
        self._require_clip(clip_uid)
        if not self.config.debug.save_diagnostics:
            raise RuntimeError("debug artifact saving is disabled")
        safe_name = _safe_component(filename, "debug filename")
        destination = self.clip_dir(clip_uid) / "debug" / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
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
            dataset_references.append(
                DatasetReference(
                    token=token,
                    type="entity",
                    entity_id=entity_id,
                    scope=reference.reference_scope,
                    visible_region=reference.visible_region,
                    image_path=relative_path.as_posix(),
                    source_frame_index=reference.source_frame_index,
                    synthetic=False,
                )
            )
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

    def _export_background(
        self,
        *,
        clip_uid: str,
        token: str,
        background: BackgroundReferenceState,
        temporary: Path,
    ) -> DatasetReference:
        relative_path = (
            Path("references") / clip_uid / self._filename_for_token(token)
        )
        self._copy_png(
            self._resolve_run_artifact(background.image_path),
            temporary / relative_path,
            background=True,
        )
        return DatasetReference(
            token=token,
            type="background",
            entity_id=None,
            scope="scene",
            visible_region="whole",
            image_path=relative_path.as_posix(),
            source_frame_index=background.source_frame_index,
            synthetic=background.status == "ready_removed",
        )

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
    def _copy_png(source: Path, destination: Path, *, background: bool) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            opened.load()
            if background:
                image = opened.convert("RGB")
            elif opened.mode in {"RGB", "RGBA"}:
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
            reference.image_path
            for sample in samples
            for reference in sample.references
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
