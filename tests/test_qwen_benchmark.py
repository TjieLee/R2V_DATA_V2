from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _benchmark_module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_qwen_backends.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_benchmark_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_contains_required_quality_and_latency_metrics() -> None:
    module = _benchmark_module()
    backend = module.BackendSpec("large", "http://127.0.0.1:8000/v1", "model-a")
    records = [
        {
            "status": "ok",
            "latency_seconds": 1.0,
            "warnings": ["phrase alignment warning", "structure sanitize warning"],
            "issue_codes": [],
            "reference_worthy_entity_count": 0,
            "entity_count": 1,
            "generic_entity_labels": 1,
        },
        {
            "status": "ok",
            "latency_seconds": 3.0,
            "warnings": ["background_reference_deferred"],
            "issue_codes": [],
            "reference_worthy_entity_count": 2,
            "entity_count": 3,
            "generic_entity_labels": 0,
        },
        {
            "status": "failed",
            "latency_seconds": 5.0,
            "warnings": [],
            "issue_codes": ["phrase_missing_from_caption"],
        },
    ]

    summary = module.summarize_backend(backend, records)

    assert summary["processed"] == 2
    assert summary["failed"] == 1
    assert summary["average_latency_seconds"] == 3.0
    assert summary["p50_latency_seconds"] == 3.0
    assert summary["p95_latency_seconds"] == pytest.approx(4.8)
    assert summary["issue_code_counts"] == {"phrase_missing_from_caption": 1}
    assert summary["phrase_alignment_warning_count"] == 2
    assert summary["structure_sanitize_warning_count"] == 1
    assert summary["no_reference_entity"] == 1
    assert summary["generic_entity_labels"] == 1
    assert summary["reference_worthy_entity_count"] == 2
    assert summary["average_entities_per_clip"] == 2.0


def test_benchmark_outputs_are_never_overwritten(tmp_path: Path) -> None:
    module = _benchmark_module()
    summary_path, records_path = module._write_results(
        output_dir=tmp_path,
        run_id="fixed-run",
        records=[{"status": "ok"}],
        summary={"run_id": "fixed-run"},
    )

    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "run_id": "fixed-run"
    }
    assert json.loads(records_path.read_text(encoding="utf-8")) == {
        "status": "ok"
    }
    with pytest.raises(FileExistsError):
        module._write_results(
            output_dir=tmp_path,
            run_id="fixed-run",
            records=[],
            summary={},
        )
