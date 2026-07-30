from __future__ import annotations

import json
from pathlib import Path

from r2v_data_v2.entity_coverage import build_clip_entity_coverage


def _write_entity_state(
    output_root: Path,
    *,
    clip_uid: str,
    entity_id: str,
    visible_frame_count: int,
    strict_candidate_count: int,
    ready_canonical: bool,
) -> dict[str, object]:
    candidate_dir = output_root / "candidates" / clip_uid / entity_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    slots = {
        f"frame_{slot:02d}": {
            "mask_available": slot < visible_frame_count,
            "candidate_accepted": slot < strict_candidate_count,
        }
        for slot in range(10)
    }
    (candidate_dir / "mask_coverage.json").write_text(
        json.dumps({"slots": slots}),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_status.json").write_text(
        json.dumps(
            {
                "status": (
                    "ready" if strict_candidate_count else "no_valid_candidate"
                ),
                "candidate_count": strict_candidate_count,
            }
        ),
        encoding="utf-8",
    )
    if ready_canonical:
        reference_dir = output_root / "references" / clip_uid / entity_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = reference_dir / "canonical.jpg"
        canonical_path.write_bytes(b"canonical")
        (reference_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "rejected": False,
                    "canonical_path": str(canonical_path),
                }
            ),
            encoding="utf-8",
        )
    return {
        "entity_id": entity_id,
        "reference_worthy": True,
    }


def _coverage(
    output_root: Path,
    entities: list[dict[str, object]],
) -> dict[str, object]:
    return build_clip_entity_coverage(
        output_root=output_root,
        clip_uid="clip-1",
        entities=entities,
        sampled_frame_count=10,
        minimum_entity_visible_ratio=0.80,
    )


def test_any_visible_entity_with_ready_reference_passes_clip(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    entity_a = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="a",
        visible_frame_count=8,
        strict_candidate_count=1,
        ready_canonical=True,
    )
    entity_b = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="b",
        visible_frame_count=3,
        strict_candidate_count=1,
        ready_canonical=True,
    )

    result = _coverage(output_root, [entity_a, entity_b])

    assert result["required_visible_frames"] == 8
    assert result["qualifying_entity_ids"] == ["a"]
    assert result["entity_coverage_passed"] is True
    summaries = result["entity_visibility_summary"]
    assert summaries["a"]["visible_frame_count"] == 8
    assert summaries["a"]["visible_frame_ratio"] == 0.8
    assert summaries["a"]["temporal_coverage_passed"] is True
    assert summaries["b"]["visible_frame_count"] == 3
    assert summaries["b"]["temporal_coverage_passed"] is False


def test_all_entities_below_temporal_threshold_reject_clip(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    entities = [
        _write_entity_state(
            output_root,
            clip_uid="clip-1",
            entity_id=entity_id,
            visible_frame_count=visible_count,
            strict_candidate_count=1,
            ready_canonical=True,
        )
        for entity_id, visible_count in (("a", 7), ("b", 3))
    ]

    result = _coverage(output_root, entities)

    assert result["qualifying_entity_ids"] == []
    assert result["entity_coverage_passed"] is False


def test_temporally_visible_entity_without_ready_canonical_does_not_qualify(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    entity = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="a",
        visible_frame_count=9,
        strict_candidate_count=1,
        ready_canonical=False,
    )

    result = _coverage(output_root, [entity])
    summary = result["entity_visibility_summary"]["a"]

    assert summary["temporal_coverage_passed"] is True
    assert summary["strict_candidate_count"] == 1
    assert summary["has_ready_canonical_reference"] is False
    assert summary["qualifies"] is False
    assert result["entity_coverage_passed"] is False


def test_temporal_coverage_uses_tracked_masks_not_strict_candidates(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    entity = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="a",
        visible_frame_count=8,
        strict_candidate_count=2,
        ready_canonical=True,
    )

    result = _coverage(output_root, [entity])
    summary = result["entity_visibility_summary"]["a"]

    assert summary["visible_frame_count"] == 8
    assert summary["strict_candidate_count"] == 2
    assert summary["temporal_coverage_passed"] is True
    assert summary["qualifies"] is True
    assert result["entity_coverage_passed"] is True


def test_clip_coverage_uses_any_entity_never_all(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    non_qualifying = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="first",
        visible_frame_count=1,
        strict_candidate_count=0,
        ready_canonical=False,
    )
    qualifying = _write_entity_state(
        output_root,
        clip_uid="clip-1",
        entity_id="second",
        visible_frame_count=10,
        strict_candidate_count=1,
        ready_canonical=True,
    )

    result = _coverage(output_root, [non_qualifying, qualifying])

    assert result["qualifying_entity_ids"] == ["second"]
    assert result["entity_coverage_passed"] is True
