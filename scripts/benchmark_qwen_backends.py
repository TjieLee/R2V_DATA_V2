from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import QwenConfig, QwenVideoConfig
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.qwen_client import (
    QwenAnnotationClient,
    QwenAnnotationFailure,
)
from r2v_data_v2.schemas import AnnotationResult


@dataclass(frozen=True)
class BackendSpec:
    name: str
    base_url: str
    model: str


class AnnotationClient(Protocol):
    def annotate(
        self,
        *,
        video_path: Path,
        caption_raw: str,
        metadata: dict[str, object],
    ) -> tuple[AnnotationResult, list[str]]: ...


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_backend(
    backend: BackendSpec,
    records: list[dict[str, object]],
) -> dict[str, object]:
    successes = [record for record in records if record["status"] == "ok"]
    latencies = [float(record["latency_seconds"]) for record in records]
    issue_codes = Counter(
        str(code)
        for record in records
        for code in record.get("issue_codes", [])
    )
    warnings = [
        str(warning)
        for record in successes
        for warning in record.get("warnings", [])
    ]
    reference_counts = [
        int(record["reference_worthy_entity_count"]) for record in successes
    ]
    entity_counts = [int(record["entity_count"]) for record in successes]
    return {
        "backend": asdict(backend),
        "processed": len(successes),
        "failed": len(records) - len(successes),
        "average_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "issue_code_counts": dict(sorted(issue_codes.items())),
        "phrase_alignment_warning_count": sum(
            "phrase" in warning.casefold() for warning in warnings
        )
        + sum(
            count
            for code, count in issue_codes.items()
            if "phrase" in code.casefold()
        ),
        "structure_sanitize_warning_count": sum(
            "sanitize" in warning.casefold()
            or "structure" in warning.casefold()
            for warning in warnings
        ),
        "no_reference_entity": sum(
            count == 0
            and "background_reference_deferred"
            not in record.get("warnings", [])
            for count, record in zip(reference_counts, successes)
        ),
        "generic_entity_labels": sum(
            int(record["generic_entity_labels"]) for record in successes
        ),
        "reference_worthy_entity_count": sum(reference_counts),
        "average_entities_per_clip": (
            sum(entity_counts) / len(entity_counts) if entity_counts else 0.0
        ),
    }


def benchmark_backend(
    *,
    backend: BackendSpec,
    source_records: list[dict[str, object]],
    client: AnnotationClient,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for source in source_records:
        clip_uid = str(source["clip_uid"])
        started = time.perf_counter()
        try:
            annotation, warnings = client.annotate(
                video_path=Path(str(source["video_path"])),
                caption_raw=str(source.get("caption_raw", "")),
                metadata=(
                    source["metadata"]
                    if isinstance(source.get("metadata"), dict)
                    else {}
                ),
            )
            latency = time.perf_counter() - started
            generic_labels = sum(
                entity.canonical_label.casefold() in {"man", "woman", "person"}
                for entity in annotation.entities
            )
            results.append(
                {
                    "backend_name": backend.name,
                    "clip_uid": clip_uid,
                    "status": "ok",
                    "latency_seconds": latency,
                    "warnings": warnings,
                    "issue_codes": [],
                    "entity_count": len(annotation.entities),
                    "reference_worthy_entity_count": sum(
                        entity.reference_worthy for entity in annotation.entities
                    ),
                    "generic_entity_labels": generic_labels,
                    "annotation": annotation.model_dump(mode="json"),
                }
            )
        except QwenAnnotationFailure as exc:
            results.append(
                {
                    "backend_name": backend.name,
                    "clip_uid": clip_uid,
                    "status": "failed",
                    "latency_seconds": time.perf_counter() - started,
                    "warnings": [],
                    "issue_codes": [issue.code for issue in exc.issues],
                    "error": str(exc),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "backend_name": backend.name,
                    "clip_uid": clip_uid,
                    "status": "failed",
                    "latency_seconds": time.perf_counter() - started,
                    "warnings": [],
                    "issue_codes": ["benchmark_request_failed"],
                    "error": str(exc),
                }
            )
    return results


def _selected_source_records(
    source_manifest: Path,
    *,
    clip_uids: set[str],
    limit: int | None,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for record in iter_source_records(source_manifest):
        if clip_uids and str(record.get("clip_uid")) not in clip_uids:
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _write_results(
    *,
    output_dir: Path,
    run_id: str,
    records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_id}.json"
    records_path = output_dir / f"{run_id}.jsonl"
    if summary_path.exists() or records_path.exists():
        raise FileExistsError(f"benchmark run already exists: {run_id}")
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    try:
        with records_path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    except Exception:
        summary_path.unlink(missing_ok=True)
        raise
    return summary_path, records_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark explicit OpenAI-compatible Qwen video annotation backends"
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--backend",
        action="append",
        nargs=3,
        metavar=("NAME", "BASE_URL", "MODEL"),
        required=True,
        help="repeat for each served endpoint/model combination",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--clip-uid", action="append", default=[])
    parser.add_argument("--clip-uids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/qwen"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--do-sample-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--total-pixels", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--repair-retries", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    clip_uids = set(args.clip_uid)
    if args.clip_uids_file is not None:
        clip_uids.update(
            line.strip()
            for line in args.clip_uids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    source_records = _selected_source_records(
        args.source_manifest,
        clip_uids=clip_uids,
        limit=args.limit,
    )
    if not source_records:
        parser.error("no source records matched the requested selection")

    video = QwenVideoConfig(
        fps=args.fps,
        do_sample_frames=args.do_sample_frames,
        max_pixels=args.max_pixels,
        total_pixels=args.total_pixels,
    )
    all_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for name, base_url, model in args.backend:
        backend = BackendSpec(name=name, base_url=base_url, model=model)
        client = QwenAnnotationClient(
            QwenConfig(
                base_url=base_url,
                api_key=args.api_key,
                model=model,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                repair_retries=args.repair_retries,
                video=video,
            )
        )
        records = benchmark_backend(
            backend=backend,
            source_records=source_records,
            client=client,
        )
        all_records.extend(records)
        summaries.append(summarize_backend(backend, records))

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
        + "-"
        + uuid.uuid4().hex[:8]
    )
    summary = {
        "run_id": run_id,
        "source_manifest": str(args.source_manifest.resolve()),
        "selected_clip_uids": [
            str(record["clip_uid"]) for record in source_records
        ],
        "backends": summaries,
    }
    summary_path, records_path = _write_results(
        output_dir=args.output_dir,
        run_id=run_id,
        records=all_records,
        summary=summary,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "summary_path": str(summary_path),
                "records_path": str(records_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
