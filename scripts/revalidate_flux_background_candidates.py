from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Protocol

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import PipelineConfig, load_config
from r2v_data_v2.inpainting import (
    INPAINTING_SOURCE_METADATA_VERSION,
    ProductionConsistencyValidator,
)
from r2v_data_v2.reconciliation import write_json_atomic


class CandidateValidator(Protocol):
    def __call__(
        self,
        *,
        original: Image.Image,
        repaired: Image.Image,
        repair_mask: Image.Image,
        reference: dict[str, object],
        mode: str,
        diagnostics_dir: Path | None = None,
    ) -> dict[str, object]: ...

    def close(self) -> None: ...


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(value)
    return records


def _load_reference(
    record: dict[str, object],
    *,
    original_path: Path,
) -> dict[str, object]:
    artifact_path = Path(str(record.get("artifact_path") or ""))
    reference: dict[str, object] = {}
    if artifact_path.is_file():
        value = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            reference.update(value)
    for field in ("clip_uid", "reference_id", "phrase", "canonical_label"):
        if field not in reference and record.get(field) is not None:
            reference[field] = record[field]
    reference["raw_canonical_path"] = str(original_path)
    return reference


def _write_jsonl_atomic(
    destination: Path,
    records: list[dict[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_revalidation(
    config: PipelineConfig,
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    validator: CandidateValidator | None = None,
) -> Path:
    source_path = run_dir / "candidates.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"candidate manifest does not exist: {source_path}")
    if output_dir is None:
        timestamp = (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + f"_{time.time_ns() % 1_000_000_000:09d}"
        )
        output_dir = run_dir / (
            f"revalidation_v{INPAINTING_SOURCE_METADATA_VERSION}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "candidates.jsonl"

    active_validator = validator or ProductionConsistencyValidator(config)
    owns_validator = validator is None
    results: list[dict[str, object]] = []
    try:
        for source_index, record in enumerate(_read_jsonl(source_path)):
            candidate_dir = Path(str(record.get("candidate_dir") or ""))
            original_path = candidate_dir / "original.png"
            candidate_path = Path(
                str(record.get("candidate_path") or candidate_dir / "candidate.png")
            )
            generation_mask_path = candidate_dir / "generation_mask.png"
            result: dict[str, object] = {
                "source_record_index": source_index,
                "clip_uid": record.get("clip_uid"),
                "reference_id": record.get("reference_id"),
                "candidate_dir": str(candidate_dir),
                "candidate_path": str(candidate_path),
                "source_candidate_sha256": (
                    _sha256_path(candidate_path)
                    if candidate_path.is_file()
                    else None
                ),
                "source_original_sha256": (
                    _sha256_path(original_path)
                    if original_path.is_file()
                    else None
                ),
                "source_generation_mask_sha256": (
                    _sha256_path(generation_mask_path)
                    if generation_mask_path.is_file()
                    else None
                ),
                "validator_metadata_version": (
                    INPAINTING_SOURCE_METADATA_VERSION
                ),
                "accepted": False,
                "rejection_reasons": [],
            }
            missing = [
                str(path)
                for path in (
                    original_path,
                    candidate_path,
                    generation_mask_path,
                )
                if not path.is_file()
            ]
            if missing:
                result.update(
                    {
                        "rejection_reasons": [
                            "revalidation_candidate_artifacts_missing"
                        ],
                        "missing_paths": missing,
                    }
                )
                results.append(result)
                continue
            try:
                reference = _load_reference(
                    record,
                    original_path=original_path,
                )
                validation = active_validator(
                    original=Image.open(original_path).convert("RGB"),
                    repaired=Image.open(candidate_path).convert("RGB"),
                    repair_mask=Image.open(generation_mask_path).convert("L"),
                    reference=reference,
                    mode="background_hole_fill",
                    diagnostics_dir=(
                        output_dir / f"candidate_{source_index:05d}"
                    ),
                )
                result.update(
                    {
                        "validator": validation,
                        "accepted": validation.get("accepted") is True,
                        "rejection_reasons": validation.get(
                            "rejection_reasons",
                            [],
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result.update(
                    {
                        "rejection_reasons": ["candidate_revalidation_failed"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            results.append(result)
    finally:
        if owns_validator:
            active_validator.close()

    _write_jsonl_atomic(output_path, results)
    write_json_atomic(
        output_dir / "summary.json",
        {
            "source_manifest_path": str(source_path),
            "source_candidate_count": len(results),
            "revalidated_count": sum(
                "validator" in result for result in results
            ),
            "accepted_count": sum(
                result.get("accepted") is True for result in results
            ),
            "validator_metadata_version": (
                INPAINTING_SOURCE_METADATA_VERSION
            ),
            "flux_inference_performed": False,
            "output_manifest_path": str(output_path),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate existing FLUX background candidates without running FLUX"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = run_revalidation(
        load_config(args.config),
        run_dir=args.run_dir,
        output_dir=args.output_dir,
    )
    print(destination)


if __name__ == "__main__":
    main()
