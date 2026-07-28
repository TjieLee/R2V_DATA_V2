from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_json_objects(paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"stage artifact must contain a JSON object: {path}")
        records.append(value)
    return records


def rebuild_jsonl(
    destination: Path,
    artifact_paths: Iterable[Path],
) -> int:
    records = _load_json_objects(artifact_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
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
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(records)


def reconcile_annotations(output_root: Path) -> int:
    return rebuild_jsonl(
        output_root / "manifests" / "annotations.jsonl",
        (output_root / "annotations").glob("*.json"),
    )


def reconcile_references(output_root: Path) -> int:
    return rebuild_jsonl(
        output_root / "manifests" / "references.jsonl",
        (output_root / "references").glob("*/*/metadata.json"),
    )


def reconcile_final_samples(output_root: Path) -> int:
    return rebuild_jsonl(
        output_root / "manifests" / "final_samples.jsonl",
        (output_root / "samples").glob("*.json"),
    )


def augmentation_artifact_paths(output_root: Path) -> list[Path]:
    return sorted((output_root / "references").glob("*/*/augmented/*.json"))


def reconcile_augmentations(output_root: Path) -> int:
    return rebuild_jsonl(
        output_root / "manifests" / "augmentations.jsonl",
        augmentation_artifact_paths(output_root),
    )
