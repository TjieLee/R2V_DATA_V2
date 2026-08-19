#!/usr/bin/env python3
"""Stream and atomically publish compact V3 production export manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.schemas import DatasetRecord, DatasetSample
from r2v_data_v2.v3.subject_attributes import EnrichedSample

DEFAULT_SHARDS_ROOT = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_exports/production/"
    "jea_motion_v1/prod-v1/shards"
)
DEFAULT_SOURCE_JSONL = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/shots_f03_motion.jsonl"
)
_SHARD_ID = re.compile(r"^shard-(?P<start>\d{9})-(?P<end>\d{9})$")


def _json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")


def _safe_artifact(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference path must remain relative to its shard root")
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


def _write_enriched_samples(
    *,
    shard_ids: list[str],
    runs_root: Path,
    destination: Path,
    database: sqlite3.Connection,
) -> tuple[Path | None, int]:
    sources: list[tuple[str, Path, Path]] = []
    for shard_id in shard_ids:
        run_root = (runs_root / shard_id).resolve(strict=True)
        if runs_root not in run_root.parents:
            raise ValueError(f"production run shard escapes runs_root: {shard_id}")
        source = run_root / "subject_attributes" / "enriched_samples.jsonl"
        if source.is_file():
            sources.append((shard_id, run_root, source))
    if not sources:
        return None, 0
    temporary = _temporary_path(destination)
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for _, run_root, source in sources:
                with source.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        sample = EnrichedSample.model_validate_json(line)
                        try:
                            database.execute(
                                "INSERT INTO enriched_ids(sample_id) VALUES (?)",
                                (sample.sample_id,),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                f"duplicate enriched sample_id: {sample.sample_id}"
                            ) from exc
                        if Path(sample.source_run_root).resolve(strict=False) != run_root:
                            raise ValueError(
                                f"enriched source_run_root mismatch at {source}:{line_number}"
                            )
                        for reference in sample.references:
                            root = (
                                run_root
                                if reference.origin == "visual_run"
                                else run_root / "subject_attributes"
                            )
                            _safe_artifact(root, reference.image_path)
                        output.write(_json_line(sample.model_dump(mode="json")))
                        count += 1
            output.flush()
            os.fsync(output.fileno())
        return temporary, count
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def compact_production_exports(
    *,
    shards_root: str | Path = DEFAULT_SHARDS_ROOT,
    output_root: str | Path | None = None,
    source_jsonl: str | Path = DEFAULT_SOURCE_JSONL,
    runs_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    shards, output = _validate_roots(shards_root, output_root)
    source = Path(source_jsonl).expanduser().resolve(strict=False)
    samples_destination = output / "samples.jsonl"
    catalog_destination = output / "catalog.json"
    enriched_destination = output / "enriched_samples.jsonl"
    samples_temporary = _temporary_path(samples_destination)
    catalog_temporary = _temporary_path(catalog_destination)
    enriched_temporary: Path | None = None
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
    total_references = 0
    shard_ids: list[str] = []
    try:
        shard_directories = sorted(path for path in shards.iterdir() if path.is_dir())
        if not shard_directories:
            raise ValueError("no production shard exports were found")
        with samples_temporary.open("x", encoding="utf-8") as output_handle:
            for shard in shard_directories:
                shard_id = shard.name
                match = _SHARD_ID.fullmatch(shard_id)
                if match is None:
                    raise ValueError(f"invalid production shard directory: {shard_id}")
                shard = shard.resolve(strict=True)
                if shards not in shard.parents:
                    raise ValueError(f"production shard escapes shards_root: {shard_id}")
                dataset_path = shard / "dataset.json"
                samples_path = shard / "samples.jsonl"
                if not dataset_path.is_file() or not samples_path.is_file():
                    raise FileNotFoundError(
                        f"production shard export is incomplete: {shard_id}"
                    )
                dataset = DatasetRecord.model_validate_json(
                    dataset_path.read_text(encoding="utf-8")
                )
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
                        record = sample.model_dump(mode="json")
                        references = record.get("references")
                        assert isinstance(references, list)
                        for reference in references:
                            assert isinstance(reference, dict)
                            image_path = str(reference["image_path"])
                            _safe_artifact(shard, image_path)
                            reference["image_path"] = (
                                Path("shards") / shard_id / image_path
                            ).as_posix()
                        output_handle.write(_json_line(record))
                        shard_samples += 1
                        shard_references += len(references)
                if shard_samples != dataset.sample_count:
                    raise ValueError(
                        f"{shard_id} sample_count does not match samples.jsonl"
                    )
                if shard_references != dataset.reference_count:
                    raise ValueError(
                        f"{shard_id} reference_count does not match samples.jsonl"
                    )
                shard_ids.append(shard_id)
                total_samples += shard_samples
                total_references += shard_references
                shard_catalog.append(
                    {
                        "shard_id": shard_id,
                        "source_start_index": int(match.group("start")),
                        "source_end_index": int(match.group("end")),
                        "sample_count": shard_samples,
                        "reference_count": shard_references,
                        "git_commit": dataset.git_commit,
                        "config_hash": dataset.config_hash,
                    }
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())

        enriched_count = 0
        if runs_root is not None:
            resolved_runs = Path(runs_root).expanduser().resolve(strict=True)
            run_base = (
                config_module.ALLOWED_WRITABLE_ROOT / "r2v_v3_runs" / "production"
            ).resolve(strict=False)
            if run_base not in resolved_runs.parents:
                raise ValueError("runs_root must remain below production V3 runs")
            enriched_temporary, enriched_count = _write_enriched_samples(
                shard_ids=shard_ids,
                runs_root=resolved_runs,
                destination=enriched_destination,
                database=database,
            )

        catalog: dict[str, object] = {
            "schema_version": "r2v.v3.production-catalog.1",
            "production_dataset_id": "jea_motion_v1",
            "production_dataset_version": "prod-v1",
            "source_jsonl": str(source),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "included_shards": shard_catalog,
            "total_samples": total_samples,
            "total_references": total_references,
            "total_enriched_samples": enriched_count,
        }
        with catalog_temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        database.commit()
        os.replace(catalog_temporary, catalog_destination)
        if enriched_temporary is not None:
            os.replace(enriched_temporary, enriched_destination)
            enriched_temporary = None
        os.replace(samples_temporary, samples_destination)
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
    parser.add_argument("--runs-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    catalog = compact_production_exports(
        shards_root=args.shards_root,
        output_root=args.output_root,
        source_jsonl=args.source_jsonl,
        runs_root=args.runs_root,
    )
    print(json.dumps(catalog, ensure_ascii=False))


if __name__ == "__main__":
    main()
