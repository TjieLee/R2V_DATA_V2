from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.compare_v3_reference_embedding_audits import compare_embedding_audits


def _write_audit(
    root: Path,
    *,
    scores: dict[tuple[str, str], tuple[float, float, float]],
) -> None:
    root.mkdir()
    records = []
    cases = (
        ("clip-a", "e1", "subject", "candidate_1"),
        ("clip-b", "e2", "object", "candidate_2"),
    )
    for clip_uid, entity_id, reference_type, selected in cases:
        for index, score in enumerate(scores[(clip_uid, entity_id)], start=1):
            candidate_id = f"candidate_{index}"
            records.append(
                {
                    "clip_uid": clip_uid,
                    "entity_id": entity_id,
                    "reference_type": reference_type,
                    "phrase": f"{reference_type} phrase",
                    "artifact_scope": "candidate",
                    "candidate_id": candidate_id,
                    "is_current_selected": candidate_id == selected,
                    "embedding": {
                        "status": "succeeded",
                        "representativeness_score": score,
                    },
                }
            )
    (root / "audit.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "audit.summary.json").write_text(
        json.dumps(
            {
                "embedding_dimensions": [4],
                "runtime": {
                    "visual_encoder": {
                        "calls": 6,
                        "total_s": 0.6,
                        "mean_s": 0.1,
                        "estimated_seconds_per_three_candidate_entity": 0.3,
                    }
                },
                "selected_representativeness_rank": {
                    "selected_rank_1_rate": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )


def test_embedding_comparison_reports_backend_and_qwen_agreement(
    tmp_path: Path,
) -> None:
    dino = tmp_path / "dino"
    siglip = tmp_path / "siglip"
    _write_audit(
        dino,
        scores={
            ("clip-a", "e1"): (0.9, 0.5, 0.2),
            ("clip-b", "e2"): (0.1, 0.4, 0.8),
        },
    )
    _write_audit(
        siglip,
        scores={
            ("clip-a", "e1"): (0.5, 0.9, 0.2),
            ("clip-b", "e2"): (0.2, 0.3, 0.7),
        },
    )
    output = tmp_path / "embedding_comparison.json"

    result = compare_embedding_audits(dino, siglip, output_path=output)

    assert result["qwen_calls_added"] == 0
    assert result["entity_count"] == 2
    assert result["dinov2"]["agreement_with_qwen"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert result["siglip2"]["agreement_with_qwen"] == {
        "count": 0,
        "denominator": 2,
        "rate": 0.0,
    }
    assert result["backend_agreement"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert result["object_only"]["dinov2_siglip2_agreement"]["rate"] == 1.0
    assert len(result["disagreement_cases"]) == 2
    assert result["cases"][0]["dinov2_best_candidate_id"] == "candidate_1"
    assert result["cases"][0]["siglip2_best_candidate_id"] == "candidate_2"
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_embedding_comparison_requires_matching_cases_and_is_atomic(
    tmp_path: Path,
) -> None:
    dino = tmp_path / "dino"
    siglip = tmp_path / "siglip"
    scores = {
        ("clip-a", "e1"): (0.9, 0.5, 0.2),
        ("clip-b", "e2"): (0.1, 0.4, 0.8),
    }
    _write_audit(dino, scores=scores)
    _write_audit(siglip, scores=scores)
    lines = (siglip / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    (siglip / "audit.jsonl").write_text(
        "\n".join(lines[:3]) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "embedding_comparison.json"

    with pytest.raises(ValueError, match="same entity cases"):
        compare_embedding_audits(dino, siglip, output_path=output)

    assert not output.exists()
