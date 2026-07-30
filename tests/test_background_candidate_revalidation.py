from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from r2v_data_v2.config import PipelineConfig
from scripts import revalidate_flux_background_candidates as revalidate


class _Validator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "accepted": True,
            "rejection_reasons": [],
        }

    def close(self) -> None:
        raise AssertionError("injected validators are not owned")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_revalidates_existing_candidates_without_regenerating_them(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "benchmark"
    run_dir.mkdir()
    artifact = tmp_path / "reference.json"
    artifact.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "reference_id": "bg1",
                "phrase": "open water",
                "category": "background",
            }
        ),
        encoding="utf-8",
    )
    source_records = []
    candidate_hashes = []
    for index in range(2):
        candidate_dir = run_dir / f"candidate_{index:05d}"
        candidate_dir.mkdir()
        original_path = candidate_dir / "original.png"
        candidate_path = candidate_dir / "candidate.png"
        mask_path = candidate_dir / "generation_mask.png"
        Image.new("RGB", (16, 12), (10, 20, 30)).save(original_path)
        Image.new("RGB", (16, 12), (40 + index, 50, 60)).save(
            candidate_path
        )
        Image.new("L", (16, 12), 255).save(mask_path)
        candidate_hashes.append(_sha256(candidate_path))
        source_records.append(
            {
                "clip_uid": "clip-1",
                "reference_id": "bg1",
                "artifact_path": str(artifact),
                "candidate_dir": str(candidate_dir),
                "candidate_path": str(candidate_path),
            }
        )
    (run_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in source_records),
        encoding="utf-8",
    )
    validator = _Validator()
    output_dir = tmp_path / "revalidation"

    result_dir = revalidate.run_revalidation(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=tmp_path / "production",
        ),
        run_dir=run_dir,
        output_dir=output_dir,
        validator=validator,
    )

    assert result_dir == output_dir
    assert len(validator.calls) == 2
    assert all(
        call["mode"] == "background_hole_fill"
        for call in validator.calls
    )
    assert [
        _sha256(run_dir / f"candidate_{index:05d}" / "candidate.png")
        for index in range(2)
    ] == candidate_hashes
    output_records = [
        json.loads(line)
        for line in (output_dir / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(output_records) == 2
    assert all(record["accepted"] is True for record in output_records)
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["revalidated_count"] == 2
    assert summary["flux_inference_performed"] is False
