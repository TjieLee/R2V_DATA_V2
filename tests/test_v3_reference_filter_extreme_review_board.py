from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.build_v3_reference_filter_extreme_review_board as board_module
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files
from r2v_data_v2.v3.schemas import (
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from tools.build_v3_reference_filter_extreme_review_board import (
    build_extreme_review_board,
)

WIDTH = 80
HEIGHT = 60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_frame(slot: int, mask: np.ndarray, object_id: str) -> TrackedMaskFrame:
    rows, columns = np.nonzero(mask)
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        track_valid=True,
        confidence=0.9,
        backend_confidences=[0.9],
        backend_object_ids=[object_id],
        area_pixels=int(mask.sum()),
        area_ratio=float(mask.mean()),
        bbox_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        rle=encode_binary_mask(mask),
    )


def _write_run(run_root: Path) -> None:
    clip_dir = run_root / "clips" / "clip-a"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    frames = []
    tracked = {"e1": [], "e2": []}
    for slot in range(10):
        yy, xx = np.indices((HEIGHT, WIDTH))
        pixels = np.stack(
            (
                (xx * 5 + slot * 7) % 256,
                (yy * 9 + slot * 11) % 256,
                ((xx + yy) * 3 + slot * 13) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        path = frames_dir / f"{slot:02d}.jpg"
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=_sha256(path),
            )
        )
        for entity_id, start in (("e1", 10), ("e2", 45)):
            mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            mask[12:52, start : start + 24] = True
            tracked[entity_id].append(
                _tracked_frame(slot, mask, f"object-{entity_id}")
            )
    (frames_dir / "frames.json").write_text(
        SampledFramesArtifact(
            clip_uid="clip-a",
            width=WIDTH,
            height=HEIGHT,
            frames=frames,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (clip_dir / "masks.rle.json").write_text(
        TrackedMasksArtifact(
            clip_uid="clip-a",
            width=WIDTH,
            height=HEIGHT,
            entities={
                "e1": TrackedEntityMasks(
                    status="ready",
                    reference_type="subject",
                    grounding_prompt="the subject",
                    backend_object_ids=["object-e1"],
                    frames=tracked["e1"],
                ),
                "e2": TrackedEntityMasks(
                    status="ready",
                    reference_type="object",
                    grounding_prompt="the object",
                    backend_object_ids=["object-e2"],
                    frames=tracked["e2"],
                ),
            },
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


def _record(
    entity_id: str,
    candidate_id: str,
    slot: int,
    reference_type: str,
    *,
    face_detected: bool | None,
) -> dict[str, object]:
    pose: dict[str, object]
    if reference_type == "subject":
        pose = {
            "status": "succeeded",
            "face_detected": face_detected,
            "face_bbox_area_ratio": 0.02 if candidate_id == "candidate_2" else None,
            "yaw": 58.0 if candidate_id == "candidate_1" else 5.0,
            "pitch": 2.0,
            "roll": 1.0,
        }
    else:
        pose = {"status": "not_applicable"}
    return {
        "clip_uid": "clip-a",
        "entity_id": entity_id,
        "candidate_id": candidate_id,
        "reference_type": reference_type,
        "phrase": f"distinct {reference_type}",
        "artifact_scope": "candidate",
        "frame_slot": slot,
        "source_frame_index": slot * 10,
        "is_current_selected": candidate_id == "candidate_1",
        "crop_padding_ratio": 0.08,
        "technical_quality": {
            "status": "succeeded",
            "luma_mean": 12.0 + slot,
            "dark_fraction_32": 0.8 - slot * 0.1,
            "laplacian_variance": 4.0 + slot,
            "tenengrad_mean": 8.0 + slot,
            "rms_contrast": 10.0 + slot,
        },
        "subject_pose": pose,
        "embedding": {
            "status": "succeeded",
            "backend": "dinov2",
            "representativeness_score": 0.7,
        },
        "raw_metadata": {"preserved": True},
    }


@pytest.fixture
def review_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_run(run_root)
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    records = [
        _record("e1", "candidate_1", 0, "subject", face_detected=False),
        _record("e1", "candidate_2", 1, "subject", face_detected=True),
        _record("e2", "candidate_1", 2, "object", face_detected=None),
    ]
    (audit_root / "audit.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    def entry(entity_id: str, candidate_id: str) -> dict[str, object]:
        return {
            "clip_uid": "clip-a",
            "entity_id": entity_id,
            "candidate_id": candidate_id,
        }

    summary = {
        "review_lists": {
            "darkest_candidates": [entry("e1", "candidate_1")],
            "highest_dark_fraction_candidates": [entry("e2", "candidate_1")],
            "lowest_laplacian_candidates": [entry("e1", "candidate_2")],
            "lowest_tenengrad_candidates": [entry("e1", "candidate_2")],
            "lowest_contrast_candidates": [entry("e2", "candidate_1")],
            "no_face_candidates": [entry("e1", "candidate_1")],
            "smallest_face_candidates": [entry("e1", "candidate_2")],
            "largest_abs_yaw_candidates": [entry("e1", "candidate_1")],
        }
    }
    (audit_root / "audit.summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    review_root = tmp_path / "reviews"
    review_root.mkdir()
    monkeypatch.setattr(board_module, "ALLOWED_REVIEW_ROOT", review_root)
    return run_root, audit_root, review_root


def test_extreme_board_is_read_only_and_renders_metrics(
    review_inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, audit_root, review_root = review_inputs
    before = {
        root: snapshot_run_files(root) for root in (run_root, audit_root)
    }
    labels: list[str] = []
    original_text = board_module._text

    def capture_text(*args: object, **kwargs: object) -> None:
        labels.append(str(args[2]))
        original_text(*args, **kwargs)

    monkeypatch.setattr(board_module, "_text", capture_text)
    output = review_root / "all"

    result = build_extreme_review_board(
        run_root=run_root,
        audit_root=audit_root,
        output_root=output,
        mode="all",
    )

    assert result["case_count"] == 3
    assert result["qwen_calls_added"] == 0
    assert result["all_inputs_unchanged"] is True
    assert all(snapshot_run_files(root) == before[root] for root in before)
    cases = json.loads((output / "review_cases.json").read_text())
    assert len(cases) == 3
    assert cases[0]["audit_record"]["raw_metadata"] == {"preserved": True}
    assert len(list((output / "cases").glob("*.png"))) == 3
    with Image.open(output / cases[0]["board_path"]) as board:
        assert board.size == (1320, 760)
    assert any("luma_mean=" in label for label in labels)
    assert any("face_detected=" in label for label in labels)
    page = (output / "review_index.html").read_text(encoding="utf-8")
    assert "V3 Reference Filter Extreme Review" in page


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("darkest", {("e1", "candidate_1"), ("e2", "candidate_1")}),
        ("blur", {("e1", "candidate_2")}),
        ("no_face", {("e1", "candidate_1")}),
        ("small_face", {("e1", "candidate_2")}),
        ("extreme_pose", {("e1", "candidate_1")}),
    ],
)
def test_extreme_board_modes(
    review_inputs: tuple[Path, Path, Path],
    mode: str,
    expected: set[tuple[str, str]],
) -> None:
    run_root, audit_root, review_root = review_inputs
    output = review_root / mode

    build_extreme_review_board(
        run_root=run_root,
        audit_root=audit_root,
        output_root=output,
        mode=mode,
    )

    cases = json.loads((output / "review_cases.json").read_text())
    assert {(case["entity_id"], case["candidate_id"]) for case in cases} == expected


def test_extreme_board_source_has_no_model_or_production_calls() -> None:
    source = inspect.getsource(board_module)
    assert "OpenAI" not in source
    assert "QwenEntityReferenceJudge" not in source
    assert "chat.completions" not in source
    assert "run_reference_filter_audit" not in source
    assert "write_references" not in source
    assert "write_pairing" not in source
