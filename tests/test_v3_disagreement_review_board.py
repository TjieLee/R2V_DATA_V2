from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.build_v3_disagreement_review_board as board_module
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files
from r2v_data_v2.v3.schemas import (
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from tools.build_v3_disagreement_review_board import (
    build_disagreement_review_board,
)

WIDTH = 80
HEIGHT = 60
FOCUS_CLIP = "15409fe27a23cb0a16bdd459"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_frame(slot: int, mask: np.ndarray, entity_id: str) -> TrackedMaskFrame:
    rows, columns = np.nonzero(mask)
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        track_valid=True,
        confidence=0.9,
        backend_confidences=[0.9],
        backend_object_ids=[f"object-{entity_id}"],
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


def _write_clip(
    run_root: Path,
    *,
    clip_uid: str,
    entity_id: str,
    reference_type: str,
) -> None:
    clip_dir = run_root / "clips" / clip_uid
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    sampled_frames = []
    tracked_frames = []
    for slot in range(10):
        yy, xx = np.indices((HEIGHT, WIDTH))
        pixels = np.stack(
            (
                (xx * (slot + 2) + 13) % 256,
                (yy * 7 + slot * 19) % 256,
                ((xx + yy) * 5 + slot * 3) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        frame_path = frames_dir / f"{slot:02d}.jpg"
        Image.fromarray(pixels, mode="RGB").save(
            frame_path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        sampled_frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot + 1),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=_sha256(frame_path),
            )
        )
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        shift = slot % 3
        mask[12:50, 18 + shift : 58 + shift] = True
        tracked_frames.append(_tracked_frame(slot, mask, entity_id))
    (frames_dir / "frames.json").write_text(
        SampledFramesArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            frames=sampled_frames,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (clip_dir / "masks.rle.json").write_text(
        TrackedMasksArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            entities={
                entity_id: TrackedEntityMasks(
                    status="ready",
                    reference_type=reference_type,
                    grounding_prompt=f"the {reference_type} entity",
                    backend_object_ids=[f"object-{entity_id}"],
                    frames=tracked_frames,
                )
            },
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


CASES = (
    (FOCUS_CLIP, "e1", "subject", (0.1, 0.9, 0.5), (0.2, 0.8, 0.4)),
    ("clip-object", "e1", "object", (0.9, 0.5, 0.1), (0.4, 0.8, 0.2)),
    ("clip-rank3", "e1", "group", (0.1, 0.5, 0.9), (0.8, 0.4, 0.2)),
    ("clip-other", "e1", "subject", (0.8, 0.4, 0.2), (0.3, 0.9, 0.1)),
)


def _audit_record(
    *,
    clip_uid: str,
    entity_id: str,
    reference_type: str,
    candidate_index: int,
    score: float,
) -> dict[str, object]:
    candidate_id = f"candidate_{candidate_index}"
    return {
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "reference_type": reference_type,
        "phrase": f"distinct {reference_type} entity",
        "artifact_scope": "candidate",
        "candidate_id": candidate_id,
        "frame_slot": candidate_index - 1,
        "source_frame_index": (candidate_index - 1) * 10,
        "is_current_selected": candidate_id == "candidate_1",
        "embedding": {
            "status": "succeeded",
            "representativeness_score": score,
        },
        "production_baseline": {
            "selected_candidate_id": "candidate_1",
            "completeness": "complete",
            "reference_scope": "full",
            "viewpoint": "front" if reference_type == "subject" else "not_applicable",
            "truncation_severity": "none",
        },
        "raw_test_metadata": {
            "preserved": True,
            "candidate_index": candidate_index,
        },
    }


def _write_audits(tmp_path: Path) -> tuple[Path, Path]:
    dino_root = tmp_path / "dinov2-audit"
    siglip_root = tmp_path / "siglip2-audit"
    dino_root.mkdir()
    siglip_root.mkdir()
    dino_records = []
    siglip_records = []
    for clip_uid, entity_id, reference_type, dino_scores, siglip_scores in CASES:
        for index in range(1, 4):
            common = {
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "reference_type": reference_type,
                "candidate_index": index,
            }
            dino_records.append(
                _audit_record(**common, score=dino_scores[index - 1])
            )
            siglip_records.append(
                _audit_record(**common, score=siglip_scores[index - 1])
            )
    for root, records in (
        (dino_root, dino_records),
        (siglip_root, siglip_records),
    ):
        (root / "audit.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (root / "audit.summary.json").write_text("{}\n", encoding="utf-8")
    return dino_root, siglip_root


@pytest.fixture
def review_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "source-run"
    run_root.mkdir()
    for clip_uid, entity_id, reference_type, _, _ in CASES:
        _write_clip(
            run_root,
            clip_uid=clip_uid,
            entity_id=entity_id,
            reference_type=reference_type,
        )
    dino_root, siglip_root = _write_audits(tmp_path)
    return run_root, dino_root, siglip_root


def test_review_board_outputs_sorted_cases_html_and_annotations(
    tmp_path: Path,
    review_inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, dino_root, siglip_root = review_inputs
    before = {
        root: snapshot_run_files(root) for root in (run_root, dino_root, siglip_root)
    }
    drawn_labels: list[str] = []
    original_draw_text = board_module._draw_safe_text

    def record_drawn_text(*args: object, **kwargs: object) -> None:
        drawn_labels.append(str(args[2]))
        original_draw_text(*args, **kwargs)

    monkeypatch.setattr(board_module, "_draw_safe_text", record_drawn_text)
    output = tmp_path / "review-all"

    summary = build_disagreement_review_board(
        run_root=run_root,
        dinov2_audit_root=dino_root,
        siglip2_audit_root=siglip_root,
        output_root=output,
    )

    assert all(snapshot_run_files(root) == before[root] for root in before)
    assert summary["all_inputs_unchanged"] is True
    assert summary["source_run_unchanged"] is True
    assert summary["qwen_calls_added"] == 0
    assert summary["case_count"] == 4
    assert summary["dino_eq_siglip_ne_qwen_count"] == 1
    assert summary["object_count"] == 1
    assert summary["qwen_rank3_by_dino_count"] == 2
    assert (output / "review_cases.json").is_file()
    assert (output / "review_index.html").is_file()
    assert json.loads((output / "summary.json").read_text()) == summary
    cases = json.loads((output / "review_cases.json").read_text())
    assert [case["clip_uid"] for case in cases] == [
        FOCUS_CLIP,
        "clip-object",
        "clip-rank3",
        "clip-other",
    ]
    assert cases[0]["candidates"][0]["dinov2_audit_record"][
        "raw_test_metadata"
    ] == {"preserved": True, "candidate_index": 1}
    candidate = cases[0]["candidates"][0]
    assert candidate["candidate_id"] == "candidate_1"
    assert candidate["dinov2_representativeness_score"] == 0.1
    assert candidate["siglip2_representativeness_score"] == 0.2
    assert candidate["qwen_selected"] is True
    assert candidate["dinov2_top_1"] is False
    assert candidate["siglip2_top_1"] is False
    assert any("DINO representativeness:" in label for label in drawn_labels)
    assert any("Qwen selected?" in label for label in drawn_labels)
    board_path = output / cases[0]["board_path"]
    with Image.open(board_path) as board:
        board.load()
        assert board.size == (1340, 1520)
        pixels = np.asarray(board)
        assert np.unique(pixels.reshape(-1, 3), axis=0).shape[0] > 20
    page = (output / "review_index.html").read_text(encoding="utf-8")
    assert "V3 Disagreement Review Board" in page
    assert "DINO = SigLIP != Qwen" in page
    assert cases[0]["board_path"] in page
    assert FOCUS_CLIP in page


@pytest.mark.parametrize(
    ("mode", "expected_clip_uids"),
    [
        ("all", [FOCUS_CLIP, "clip-object", "clip-rank3", "clip-other"]),
        ("dino_eq_siglip_ne_qwen", [FOCUS_CLIP]),
        ("qwen_rank3_by_dino", [FOCUS_CLIP, "clip-rank3"]),
        ("object_only", ["clip-object"]),
        ("focus_cases", [FOCUS_CLIP]),
    ],
)
def test_review_board_mode_filtering(
    tmp_path: Path,
    review_inputs: tuple[Path, Path, Path],
    mode: str,
    expected_clip_uids: list[str],
) -> None:
    run_root, dino_root, siglip_root = review_inputs
    before = snapshot_run_files(run_root)
    output = tmp_path / f"review-{mode}"

    summary = build_disagreement_review_board(
        run_root=run_root,
        dinov2_audit_root=dino_root,
        siglip2_audit_root=siglip_root,
        output_root=output,
        mode=mode,
    )

    cases = json.loads((output / "review_cases.json").read_text())
    assert [case["clip_uid"] for case in cases] == expected_clip_uids
    assert summary["case_count"] == len(expected_clip_uids)
    assert len(list((output / "cases").glob("*.png"))) == len(expected_clip_uids)
    assert snapshot_run_files(run_root) == before


def test_review_board_source_contains_no_model_calls_or_source_writes() -> None:
    source = inspect.getsource(board_module)
    assert "OpenAI" not in source
    assert "QwenEntityReferenceJudge" not in source
    assert "chat.completions" not in source
    assert "write_references" not in source
    assert "write_pairing" not in source


def test_review_board_rejects_output_inside_an_input_root(
    review_inputs: tuple[Path, Path, Path],
) -> None:
    run_root, dino_root, siglip_root = review_inputs

    with pytest.raises(ValueError, match="separate from all input roots"):
        build_disagreement_review_board(
            run_root=run_root,
            dinov2_audit_root=dino_root,
            siglip2_audit_root=siglip_root,
            output_root=dino_root / "review",
        )
