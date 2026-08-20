#!/usr/bin/env python3
"""Stream and atomically publish compact V3 production export manifests."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import yaml
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.production_export import (
    PRODUCTION_SAMPLE_SCHEMA_VERSION,
    ProductionReference,
    ProductionSample,
    ProductionSampleSource,
)
from r2v_data_v2.v3.production_source import JEA_VIDEO_MOTION_ADAPTER
from r2v_data_v2.v3.schemas import DatasetRecord, DatasetReference, DatasetSample
from r2v_data_v2.v3.subject_attributes import EnrichedSample

DEFAULT_SHARDS_ROOT = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_exports/production/"
    "jea_motion_v1/prod-v1/shards"
)
DEFAULT_SOURCE_JSONL = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/shots_f03_motion.jsonl"
)
DEFAULT_SOURCE_YAML = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_configs/production/"
    "jea_motion_v1/prod-v1/source.yaml"
)
_SHARD_ID = re.compile(r"^shard-(?P<start>\d{9})-(?P<end>\d{9})$")
_VISUAL_TOKEN = re.compile(r"^<ref_(subject|object|group)_\d+>$")
_ATTRIBUTE_LINK_FALLBACK_ERRNOS = {
    errno.EACCES,
    errno.EPERM,
    errno.EXDEV,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _below(path: Path, root: Path) -> bool:
    return root in path.parents


def _resolve_source_jsonl(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if (
        not resolved.is_file()
        or resolved.suffix.lower() != ".jsonl"
        or not _below(resolved, dataset_root)
    ):
        raise ValueError("source_jsonl must be a JSONL below public dataset")
    return resolved


def _load_source_descriptor(
    path: str | Path,
    *,
    source_jsonl: Path,
) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    config_root = (
        config_module.ALLOWED_WRITABLE_ROOT / "r2v_v3_configs" / "production"
    ).resolve(strict=False)
    if not resolved.is_file() or not _below(resolved, config_root):
        raise ValueError("source_yaml must be below production r2v_v3_configs")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("source.yaml must contain a YAML mapping")
    required_strings = (
        "source_adapter",
        "source_jsonl",
        "base_config_path",
        "base_config_sha256",
        "base_config_fingerprint",
        "clips_root",
    )
    if any(not isinstance(value.get(name), str) for name in required_strings):
        raise ValueError("source.yaml is missing immutable production identity")
    if value["source_jsonl"] != str(source_jsonl):
        raise ValueError("source.yaml source_jsonl does not match compaction input")
    if value["source_adapter"] != JEA_VIDEO_MOTION_ADAPTER:
        raise ValueError("source.yaml production adapter does not match")
    return value


def _resolve_clips_root(source_descriptor: dict[str, object]) -> Path:
    value = source_descriptor["clips_root"]
    assert isinstance(value, str)
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError("source.yaml clips_root must be absolute")
    resolved = raw.resolve(strict=True)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if not resolved.is_dir() or not _below(resolved, dataset_root):
        raise ValueError("source.yaml clips_root must be below public dataset")
    return resolved


def _safe_artifact(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference path must remain relative to its artifact root")
    resolved = (resolved_root / relative).resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"referenced artifact is missing or escaped: {relative_path}")
    return resolved


def _validate_roots(
    shards_root: str | Path,
    output_root: str | Path | None,
) -> tuple[Path, Path]:
    shards = Path(shards_root).expanduser().resolve(strict=True)
    if not shards.is_dir():
        raise ValueError("shards_root must be a directory")
    output = (
        Path(output_root).expanduser().resolve(strict=False)
        if output_root is not None
        else shards.parent
    )
    export_root = (
        config_module.ALLOWED_WRITABLE_ROOT / "r2v_v3_exports" / "production"
    ).resolve(strict=False)
    if export_root not in shards.parents or export_root not in output.parents:
        raise ValueError("production export paths must remain below r2v_v3_exports")
    return shards, output


def _validate_runs_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    run_base = (
        config_module.ALLOWED_WRITABLE_ROOT / "r2v_v3_runs" / "production"
    ).resolve(strict=False)
    if not resolved.is_dir() or not _below(resolved, run_base):
        raise ValueError("runs_root must remain below production V3 runs")
    return resolved


def _validate_attribute_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image_format = image.format
            image_mode = image.mode
            image.verify()
    except Exception as exc:  # noqa: BLE001 - normalize decoder failures
        raise ValueError(f"attribute reference is not a decodable image: {path}") from exc
    if image_format != "PNG" or image_mode != "RGBA":
        raise ValueError(f"attribute reference must be an RGBA PNG: {path}")


def _safe_component(value: str, field_name: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field_name} is not a safe path component")
    return value


def _canonical_reference_directory(*, target_video: str, clips_root: Path) -> Path:
    if any(component in {".", ".."} for component in target_video.split("/")):
        raise ValueError("target_video contains an unsafe dot path component")
    raw = Path(target_video).expanduser()
    if not raw.is_absolute():
        raise ValueError("target_video must be an absolute path")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("target_video must be an existing file")
    try:
        relative = resolved.relative_to(clips_root)
    except ValueError as exc:
        raise ValueError("target_video must remain below clips_root") from exc
    if resolved.suffix.lower() != ".mp4":
        raise ValueError("target_video must be a processed MP4 shot")
    return Path("references") / relative.with_suffix("")


def _canonical_visual_reference_path(
    *,
    directory: Path,
    reference: DatasetReference,
) -> Path:
    kind = _dataset_reference_kind(reference)
    if kind == "background":
        return directory / "background.png"
    assert reference.entity_id is not None
    entity_id = _safe_component(reference.entity_id, "entity_id")
    return directory / f"{kind}_{entity_id}.png"


def _materialize_reference(
    *,
    source: Path,
    output_root: Path,
    relative_destination: Path,
    require_rgba: bool,
    allow_copy_fallback: bool,
) -> str:
    if require_rgba:
        _validate_attribute_png(source)
    if relative_destination.is_absolute() or ".." in relative_destination.parts:
        raise ValueError("canonical reference path must remain relative")
    destination = output_root / relative_destination
    source_sha = _sha256_file(source)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_sha:
            raise FileExistsError(
                f"conflicting published reference: {destination}"
            )
        if require_rgba:
            _validate_attribute_png(destination)
        return destination.relative_to(output_root).as_posix()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            if (
                not allow_copy_fallback
                or exc.errno not in _ATTRIBUTE_LINK_FALLBACK_ERRNOS
            ):
                raise
            shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or _sha256_file(destination) != source_sha:
                raise FileExistsError(
                    f"conflicting published reference: {destination}"
                ) from None
        if require_rgba:
            _validate_attribute_png(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.relative_to(output_root).as_posix()


def _materialize_attribute_reference(
    *,
    source: Path,
    output_root: Path,
    directory: Path,
    owner_entity_id: str,
    attribute_id: str,
    attribute_type: str,
) -> str:
    filename = "attribute_{}_{}_{}.png".format(
        _safe_component(owner_entity_id, "owner_entity_id"),
        _safe_component(attribute_id, "attribute_id"),
        _safe_component(attribute_type, "attribute_type"),
    )
    return _materialize_reference(
        source=source,
        output_root=output_root,
        relative_destination=directory / filename,
        require_rgba=True,
        allow_copy_fallback=True,
    )


def _dataset_reference_kind(reference: DatasetReference) -> str:
    if reference.type == "background":
        return "background"
    match = _VISUAL_TOKEN.fullmatch(reference.token)
    if match is None:
        raise ValueError(f"cannot derive production kind from {reference.token}")
    return match.group(1)


def _production_visual_reference(
    *,
    reference: DatasetReference,
    image_index: int,
    kind: str,
    image_path: str,
) -> ProductionReference:
    return ProductionReference(
        image_id=f"image_{image_index}",
        image_index=image_index,
        kind=kind,
        entity_id=reference.entity_id,
        image_path=image_path,
        source_frame_index=reference.source_frame_index,
        scope=reference.scope,
        visible_region=reference.visible_region,
        synthetic=reference.synthetic,
    )


def _visual_only_sample(
    sample: DatasetSample,
    *,
    shard_id: str,
    reference_paths: list[str],
) -> ProductionSample:
    references = [
        _production_visual_reference(
            reference=reference,
            image_index=index,
            kind=_dataset_reference_kind(reference),
            image_path=reference_paths[index - 1],
        )
        for index, reference in enumerate(sample.references, 1)
    ]
    return ProductionSample(
        sample_id=sample.sample_id,
        clip_uid=sample.sample_id,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=sample.r2v_instruction,
        references=references,
        source=ProductionSampleSource(
            parent_video_id=sample.source.parent_video_id,
            clip_suffix=sample.source.clip_suffix,
            shard_id=shard_id,
        ),
    )


def _enriched_production_sample(
    sample: DatasetSample,
    enriched: EnrichedSample,
    *,
    shard_id: str,
    reference_paths: list[str],
    reference_directory: Path,
    run_root: Path,
    output_root: Path,
) -> ProductionSample:
    if enriched.clip_uid != sample.sample_id:
        raise ValueError(f"enriched clip_uid mismatch: {enriched.sample_id}")
    original_target = enriched.original_visual.get("target_video")
    if original_target is not None and original_target != sample.target_video:
        raise ValueError(f"enriched target_video mismatch: {enriched.sample_id}")
    original_source = enriched.original_visual.get("source")
    if original_source is not None and original_source != sample.source.model_dump():
        raise ValueError(f"enriched source provenance mismatch: {enriched.sample_id}")

    entity_references: dict[str, tuple[DatasetReference, str]] = {}
    background_reference: tuple[DatasetReference, str] | None = None
    for reference, path in zip(sample.references, reference_paths, strict=True):
        if reference.type == "background":
            if background_reference is not None:
                raise ValueError("visual sample contains duplicate background references")
            background_reference = (reference, path)
        else:
            assert reference.entity_id is not None
            if reference.entity_id in entity_references:
                raise ValueError(
                    f"visual sample contains duplicate entity_id: {reference.entity_id}"
                )
            entity_references[reference.entity_id] = (reference, path)

    attributes = {
        record.attribute_id: record for record in enriched.accepted_attributes
    }
    visual_entity_ids = set(entity_references)
    if len(attributes) != len(enriched.accepted_attributes):
        raise ValueError(f"duplicate accepted attribute_id: {enriched.sample_id}")
    production_references: list[ProductionReference] = []
    for enriched_reference in enriched.references:
        index = enriched_reference.image_index
        if enriched_reference.kind == "attribute":
            assert enriched_reference.attribute_id is not None
            assert enriched_reference.owner_entity_id is not None
            record = attributes.pop(enriched_reference.attribute_id, None)
            if record is None:
                raise ValueError(
                    f"missing accepted attribute: {enriched_reference.attribute_id}"
                )
            if (
                record.owner_entity_id != enriched_reference.owner_entity_id
                or record.owner_entity_id not in visual_entity_ids
                or record.image_path != enriched_reference.image_path
                or record.source_frame_index != enriched_reference.source_frame_index
            ):
                raise ValueError(
                    f"attribute provenance mismatch: {record.attribute_id}"
                )
            source_image = _safe_artifact(
                run_root / "subject_attributes",
                enriched_reference.image_path,
            )
            published_path = _materialize_attribute_reference(
                source=source_image,
                output_root=output_root,
                directory=reference_directory,
                owner_entity_id=record.owner_entity_id,
                attribute_id=record.attribute_id,
                attribute_type=record.attribute_type,
            )
            assert enriched_reference.source_frame_index is not None
            production_references.append(
                ProductionReference(
                    image_id=f"image_{index}",
                    image_index=index,
                    kind="attribute",
                    attribute_id=record.attribute_id,
                    owner_entity_id=record.owner_entity_id,
                    attribute_type=record.attribute_type,
                    image_path=published_path,
                    source_frame_index=enriched_reference.source_frame_index,
                    synthetic=False,
                )
            )
            continue

        _safe_artifact(run_root, enriched_reference.image_path)
        if enriched_reference.kind == "background":
            pair = background_reference
            background_reference = None
        else:
            assert enriched_reference.entity_id is not None
            pair = entity_references.pop(enriched_reference.entity_id, None)
        if pair is None:
            raise ValueError(
                f"enriched Visual reference has no shard export match: "
                f"{enriched_reference.image_id}"
            )
        visual_reference, visual_path = pair
        if _dataset_reference_kind(visual_reference) != enriched_reference.kind:
            raise ValueError(
                f"enriched Visual kind mismatch: {enriched_reference.image_id}"
            )
        if visual_reference.source_frame_index != enriched_reference.source_frame_index:
            raise ValueError(
                f"enriched Visual frame provenance mismatch: "
                f"{enriched_reference.image_id}"
            )
        production_references.append(
            _production_visual_reference(
                reference=visual_reference,
                image_index=index,
                kind=enriched_reference.kind,
                image_path=visual_path,
            )
        )

    if entity_references or background_reference is not None or attributes:
        raise ValueError(
            f"enriched references do not cover the visual sample: {sample.sample_id}"
        )
    return ProductionSample(
        sample_id=sample.sample_id,
        clip_uid=enriched.clip_uid,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=enriched.enriched_instruction,
        references=production_references,
        source=ProductionSampleSource(
            parent_video_id=sample.source.parent_video_id,
            clip_suffix=sample.source.clip_suffix,
            shard_id=shard_id,
        ),
    )


def _load_shard_enriched_samples(
    *,
    shard_id: str,
    runs_root: Path,
    database: sqlite3.Connection,
    audit_handle: TextIO | None,
) -> tuple[Path, dict[str, EnrichedSample]]:
    run_root = (runs_root / shard_id).resolve(strict=True)
    if runs_root not in run_root.parents or not run_root.is_dir():
        raise ValueError(f"production run shard escapes runs_root: {shard_id}")
    source = run_root / "subject_attributes" / "enriched_samples.jsonl"
    enriched_by_id: dict[str, EnrichedSample] = {}
    if not source.is_file():
        return run_root, enriched_by_id
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            enriched = EnrichedSample.model_validate_json(line)
            if Path(enriched.source_run_root).resolve(strict=False) != run_root:
                raise ValueError(
                    f"enriched source_run_root mismatch at {source}:{line_number}"
                )
            try:
                database.execute(
                    "INSERT INTO enriched_ids(sample_id) VALUES (?)",
                    (enriched.sample_id,),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"duplicate enriched sample_id: {enriched.sample_id}"
                ) from exc
            enriched_by_id[enriched.sample_id] = enriched
            if audit_handle is not None:
                audit_handle.write(_json_line(enriched.model_dump(mode="json")))
    return run_root, enriched_by_id


def compact_production_exports(
    *,
    shards_root: str | Path = DEFAULT_SHARDS_ROOT,
    output_root: str | Path | None = None,
    source_jsonl: str | Path = DEFAULT_SOURCE_JSONL,
    source_yaml: str | Path = DEFAULT_SOURCE_YAML,
    runs_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    shards, output = _validate_roots(shards_root, output_root)
    source = _resolve_source_jsonl(source_jsonl)
    source_descriptor = _load_source_descriptor(source_yaml, source_jsonl=source)
    clips_root = _resolve_clips_root(source_descriptor)
    resolved_runs = _validate_runs_root(runs_root) if runs_root is not None else None

    samples_destination = output / "samples.jsonl"
    catalog_destination = output / "catalog.json"
    enriched_destination = output / "enriched_samples.jsonl"
    samples_temporary = _temporary_path(samples_destination)
    catalog_temporary = _temporary_path(catalog_destination)
    enriched_temporary = (
        _temporary_path(enriched_destination) if resolved_runs is not None else None
    )
    database_fd, database_name = tempfile.mkstemp(
        prefix=".production-sample-ids-",
        suffix=".sqlite3",
        dir=output,
    )
    os.close(database_fd)
    database_path = Path(database_name)
    database = sqlite3.connect(database_path)
    database.execute("CREATE TABLE sample_ids(sample_id TEXT PRIMARY KEY)")
    database.execute("CREATE TABLE enriched_ids(sample_id TEXT PRIMARY KEY)")
    shard_catalog: list[dict[str, object]] = []
    total_samples = 0
    total_visual_references = 0
    total_attribute_references = 0
    total_enriched_samples = 0
    try:
        shard_directories = sorted(path for path in shards.iterdir() if path.is_dir())
        if not shard_directories:
            raise ValueError("no production shard exports were found")
        with samples_temporary.open("x", encoding="utf-8") as output_handle:
            audit_context = (
                enriched_temporary.open("x", encoding="utf-8")
                if enriched_temporary is not None
                else nullcontext(None)
            )
            with audit_context as audit_handle:
                for shard in shard_directories:
                    shard_id = shard.name
                    match = _SHARD_ID.fullmatch(shard_id)
                    if match is None:
                        raise ValueError(
                            f"invalid production shard directory: {shard_id}"
                        )
                    shard = shard.resolve(strict=True)
                    if shards not in shard.parents:
                        raise ValueError(
                            f"production shard escapes shards_root: {shard_id}"
                        )
                    dataset_path = shard / "dataset.json"
                    samples_path = shard / "samples.jsonl"
                    if not dataset_path.is_file() or not samples_path.is_file():
                        raise FileNotFoundError(
                            f"production shard export is incomplete: {shard_id}"
                        )
                    dataset = DatasetRecord.model_validate_json(
                        dataset_path.read_text(encoding="utf-8")
                    )
                    run_root: Path | None = None
                    enriched_by_id: dict[str, EnrichedSample] = {}
                    if resolved_runs is not None:
                        run_root, enriched_by_id = _load_shard_enriched_samples(
                            shard_id=shard_id,
                            runs_root=resolved_runs,
                            database=database,
                            audit_handle=audit_handle,
                        )
                        total_enriched_samples += len(enriched_by_id)

                    shard_samples = 0
                    shard_references = 0
                    with samples_path.open("r", encoding="utf-8") as input_handle:
                        for line in input_handle:
                            if not line.strip():
                                continue
                            sample = DatasetSample.model_validate_json(line)
                            try:
                                database.execute(
                                    "INSERT INTO sample_ids(sample_id) VALUES (?)",
                                    (sample.sample_id,),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(
                                    f"duplicate sample_id: {sample.sample_id}"
                                ) from exc
                            reference_directory = _canonical_reference_directory(
                                target_video=sample.target_video,
                                clips_root=clips_root,
                            )
                            reference_paths: list[str] = []
                            for reference in sample.references:
                                source_reference = _safe_artifact(
                                    shard, reference.image_path
                                )
                                reference_paths.append(
                                    _materialize_reference(
                                        source=source_reference,
                                        output_root=output,
                                        relative_destination=(
                                            _canonical_visual_reference_path(
                                                directory=reference_directory,
                                                reference=reference,
                                            )
                                        ),
                                        require_rgba=False,
                                        allow_copy_fallback=False,
                                    )
                                )
                            enriched = enriched_by_id.pop(sample.sample_id, None)
                            if enriched is None:
                                production = _visual_only_sample(
                                    sample,
                                    shard_id=shard_id,
                                    reference_paths=reference_paths,
                                )
                            else:
                                assert run_root is not None
                                production = _enriched_production_sample(
                                    sample,
                                    enriched,
                                    shard_id=shard_id,
                                    reference_paths=reference_paths,
                                    reference_directory=reference_directory,
                                    run_root=run_root,
                                    output_root=output,
                                )
                            for reference in production.references:
                                _safe_artifact(output, reference.image_path)
                            output_handle.write(
                                _json_line(production.model_dump(mode="json"))
                            )
                            shard_samples += 1
                            shard_references += len(sample.references)
                            total_visual_references += sum(
                                reference.kind != "attribute"
                                for reference in production.references
                            )
                            total_attribute_references += sum(
                                reference.kind == "attribute"
                                for reference in production.references
                            )
                    if enriched_by_id:
                        orphan = sorted(enriched_by_id)[0]
                        raise ValueError(
                            f"orphan enriched sample_id in {shard_id}: {orphan}"
                        )
                    if shard_samples != dataset.sample_count:
                        raise ValueError(
                            f"{shard_id} sample_count does not match samples.jsonl"
                        )
                    if shard_references != dataset.reference_count:
                        raise ValueError(
                            f"{shard_id} reference_count does not match samples.jsonl"
                        )
                    total_samples += shard_samples
                    shard_catalog.append(
                        {
                            "shard_id": shard_id,
                            "source_start_index": int(match.group("start")),
                            "source_end_index": int(match.group("end")),
                            "sample_count": shard_samples,
                            "visual_reference_count": shard_references,
                            "git_commit": dataset.git_commit,
                            "config_hash": dataset.config_hash,
                        }
                    )
                if audit_handle is not None:
                    audit_handle.flush()
                    os.fsync(audit_handle.fileno())
            output_handle.flush()
            os.fsync(output_handle.fileno())

        samples_sha256 = _sha256_file(samples_temporary)
        total_references = total_visual_references + total_attribute_references
        catalog: dict[str, object] = {
            "schema_version": "r2v.v3.production-catalog.1",
            "canonical_sample_schema_version": PRODUCTION_SAMPLE_SCHEMA_VERSION,
            "production_dataset_id": "jea_motion_v1",
            "production_dataset_version": "prod-v1",
            "source_jsonl": str(source),
            "production_adapter_version": source_descriptor["source_adapter"],
            "base_config_path": source_descriptor["base_config_path"],
            "base_config_sha256": source_descriptor["base_config_sha256"],
            "base_config_fingerprint": source_descriptor[
                "base_config_fingerprint"
            ],
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "samples_jsonl_sha256": samples_sha256,
            "included_shards": shard_catalog,
            "source_ranges": [
                {
                    "start_index": shard["source_start_index"],
                    "end_index": shard["source_end_index"],
                }
                for shard in shard_catalog
            ],
            "total_canonical_samples": total_samples,
            "total_visual_references": total_visual_references,
            "total_attribute_references": total_attribute_references,
            "total_samples": total_samples,
            "total_references": total_references,
            "total_enriched_samples": total_enriched_samples,
        }
        with catalog_temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        database.commit()

        os.replace(samples_temporary, samples_destination)
        if enriched_temporary is not None:
            os.replace(enriched_temporary, enriched_destination)
            enriched_temporary = None
        os.replace(catalog_temporary, catalog_destination)
        return catalog
    finally:
        database.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").unlink(missing_ok=True)
        samples_temporary.unlink(missing_ok=True)
        catalog_temporary.unlink(missing_ok=True)
        if enriched_temporary is not None:
            enriched_temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, default=DEFAULT_SHARDS_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--source-yaml", type=Path, default=DEFAULT_SOURCE_YAML)
    parser.add_argument("--runs-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    catalog = compact_production_exports(
        shards_root=args.shards_root,
        output_root=args.output_root,
        source_jsonl=args.source_jsonl,
        source_yaml=args.source_yaml,
        runs_root=args.runs_root,
    )
    print(json.dumps(catalog, ensure_ascii=False))


if __name__ == "__main__":
    main()
