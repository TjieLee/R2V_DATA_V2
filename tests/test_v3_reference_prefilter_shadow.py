from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tools.build_v3_prefilter_shadow_review_board as board_module
import tools.simulate_v3_reference_prefilter as simulator_module
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files
from r2v_data_v2.v3.schemas import (
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from tools.build_v3_prefilter_shadow_review_board import (
    build_prefilter_shadow_review_board,
)
from tools.simulate_v3_reference_prefilter import (
    _subject_near_silhouette,
    simulate_reference_prefilter,
)

WIDTH = 80
HEIGHT = 60

ENTITY_CASES = (
    ("clip-near", "e1", "subject"),
    ("clip-blur-one", "e1", "subject"),
    ("clip-blur-two", "e1", "subject"),
    ("clip-object", "e1", "object"),
    ("clip-group", "e1", "group"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _technical(
    *,
    luma: float,
    dark_fraction: float,
    laplacian: float,
    tenengrad: float,
) -> dict[str, object]:
    return {
        "status": "succeeded",
        "luma_mean": luma,
        "dark_fraction_32": dark_fraction,
        "laplacian_variance": laplacian,
        "tenengrad_mean": tenengrad,
        "rms_contrast": 12.0,
    }


def _record(
    *,
    clip_uid: str,
    entity_id: str,
    reference_type: str,
    candidate_index: int,
    technical: dict[str, object],
    selected_candidate_index: int,
) -> dict[str, object]:
    candidate_id = f"candidate_{candidate_index}"
    pose = (
        {
            "status": "succeeded",
            "face_detected": candidate_index != 2,
            "face_bbox_area_ratio": 0.04,
            "yaw": float(candidate_index * 4),
        }
        if reference_type == "subject"
        else {"status": "not_applicable"}
    )
    return {
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "candidate_id": candidate_id,
        "reference_type": reference_type,
        "phrase": f"distinct {reference_type}",
        "artifact_scope": "candidate",
        "frame_slot": candidate_index - 1,
        "source_frame_index": (candidate_index - 1) * 10,
        "crop_padding_ratio": 0.08,
        "is_current_selected": candidate_index == selected_candidate_index,
        "technical_quality": technical,
        "subject_pose": pose,
        "embedding": {
            "status": "succeeded",
            "representativeness_score": 0.5,
        },
    }


def _audit_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(1, 4):
        records.append(
            _record(
                clip_uid="clip-near",
                entity_id="e1",
                reference_type="subject",
                candidate_index=index,
                technical=_technical(
                    luma=10.0,
                    dark_fraction=0.97,
                    laplacian=4.0,
                    tenengrad=80.0,
                ),
                selected_candidate_index=3,
            )
        )
    for clip_uid, values, selected in (
        ("clip-blur-one", ((30, 45), (100, 100), (90, 90)), 1),
        ("clip-blur-two", ((20, 40), (30, 50), (100, 100)), 3),
    ):
        for index, (laplacian, tenengrad) in enumerate(values, start=1):
            records.append(
                _record(
                    clip_uid=clip_uid,
                    entity_id="e1",
                    reference_type="subject",
                    candidate_index=index,
                    technical=_technical(
                        luma=90.0,
                        dark_fraction=0.0,
                        laplacian=float(laplacian),
                        tenengrad=float(tenengrad),
                    ),
                    selected_candidate_index=selected,
                )
            )
    for clip_uid, reference_type in (
        ("clip-object", "object"),
        ("clip-group", "group"),
    ):
        for index, (laplacian, tenengrad) in enumerate(
            ((10, 10), (100, 100), (90, 90)),
            start=1,
        ):
            records.append(
                _record(
                    clip_uid=clip_uid,
                    entity_id="e1",
                    reference_type=reference_type,
                    candidate_index=index,
                    technical=_technical(
                        luma=8.0,
                        dark_fraction=0.99,
                        laplacian=float(laplacian),
                        tenengrad=float(tenengrad),
                    ),
                    selected_candidate_index=1,
                )
            )
    return records


def _write_audit(audit_root: Path) -> None:
    audit_root.mkdir(parents=True)
    (audit_root / "audit.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in _audit_records()),
        encoding="utf-8",
    )
    (audit_root / "audit.summary.json").write_text("{}\n", encoding="utf-8")


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
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=_sha256(frame_path),
            )
        )
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[12:50, 18:58] = True
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
                    grounding_prompt=f"the {reference_type}",
                    backend_object_ids=[f"object-{entity_id}"],
                    frames=tracked_frames,
                )
            },
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


@pytest.fixture
def shadow_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    audit_parent = tmp_path / "audits"
    audit_root = audit_parent / "source-audit"
    _write_audit(audit_root)
    run_root = tmp_path / "source-run"
    run_root.mkdir()
    for clip_uid, entity_id, reference_type in ENTITY_CASES:
        _write_clip(
            run_root,
            clip_uid=clip_uid,
            entity_id=entity_id,
            reference_type=reference_type,
        )
    review_parent = tmp_path / "reviews"
    review_parent.mkdir()
    monkeypatch.setattr(simulator_module, "ALLOWED_AUDIT_ROOT", audit_parent)
    monkeypatch.setattr(board_module, "ALLOWED_REVIEW_ROOT", review_parent)
    return run_root, audit_root, audit_parent, review_parent


@pytest.mark.parametrize(
    ("technical", "expected"),
    [
        (
            _technical(
                luma=15,
                dark_fraction=0.95,
                laplacian=5,
                tenengrad=100,
            ),
            True,
        ),
        (
            _technical(
                luma=16,
                dark_fraction=0.95,
                laplacian=5,
                tenengrad=100,
            ),
            False,
        ),
        (
            _technical(
                luma=15,
                dark_fraction=0.94,
                laplacian=5,
                tenengrad=100,
            ),
            False,
        ),
        (
            _technical(
                luma=10,
                dark_fraction=0.99,
                laplacian=50,
                tenengrad=1000,
            ),
            False,
        ),
        (
            _technical(
                luma=100,
                dark_fraction=0,
                laplacian=2,
                tenengrad=20,
            ),
            False,
        ),
    ],
)
def test_near_silhouette_requires_all_four_conditions(
    technical: dict[str, object],
    expected: bool,
) -> None:
    assert _subject_near_silhouette("subject", technical) is expected


def test_shadow_cli_defaults_to_all_rules_and_review_cases() -> None:
    simulation_arguments = simulator_module._parser().parse_args(
        ["--audit-root", "audit", "--output", "simulation.json"]
    )
    board_arguments = board_module._parser().parse_args(
        [
            "--run-root",
            "run",
            "--audit-root",
            "audit",
            "--simulation",
            "simulation.json",
            "--output-root",
            "review",
        ]
    )

    assert simulation_arguments.rule == "all"
    assert board_arguments.mode == "all"


def test_simulation_counts_shadow_states_and_evidence_without_mutating_audit(
    shadow_inputs: tuple[Path, Path, Path, Path],
) -> None:
    _, audit_root, audit_parent, _ = shadow_inputs
    before = snapshot_run_files(audit_root)
    output = audit_parent / "simulation.json"

    simulation = simulate_reference_prefilter(
        audit_root=audit_root,
        output=output,
    )

    assert snapshot_run_files(audit_root) == before
    assert json.loads(output.read_text()) == simulation
    candidates = simulation["candidates"]
    assert isinstance(candidates, list)
    object_and_group = [
        candidate
        for candidate in candidates
        if candidate["reference_type"] in {"object", "group"}
    ]
    assert all(candidate["relative_blur_flag"] is False for candidate in object_and_group)
    assert all(candidate["near_silhouette_flag"] is False for candidate in object_and_group)
    blur_candidate = next(
        candidate
        for candidate in candidates
        if candidate["clip_uid"] == "clip-blur-one"
        and candidate["candidate_id"] == "candidate_1"
    )
    assert blur_candidate["laplacian_ratio"] == pytest.approx(0.3)
    assert blur_candidate["tenengrad_ratio"] == pytest.approx(0.45)
    assert blur_candidate["relative_blur_flag"] is True
    assert blur_candidate["subject_pose_evidence"]["face_detected"] is True

    entities = {
        entity["clip_uid"]: entity for entity in simulation["entities"]
    }
    assert entities["clip-near"]["shadow_state"] == "all_candidates_flagged"
    assert entities["clip-near"]["estimated_qwen_call_skippable"] is True
    assert entities["clip-blur-one"]["shadow_state"] == "3_to_2"
    assert entities["clip-blur-one"]["potential_image_reduction"] == 2
    assert entities["clip-blur-two"]["shadow_state"] == "3_to_1"
    assert entities["clip-blur-two"]["potential_image_reduction"] == 4
    assert entities["clip-object"]["shadow_state"] == "unchanged"
    assert entities["clip-group"]["shadow_state"] == "unchanged"

    summary = simulation["summary"]
    assert summary["candidate_count"] == 15
    assert summary["near_silhouette_flag_count"] == 3
    assert summary["relative_blur_flag_count"] == 3
    assert summary["combined_flag_count"] == 6
    assert summary["entity_count"] == 5
    assert summary["entity_unchanged_count"] == 2
    assert summary["entity_3_to_2_count"] == 1
    assert summary["entity_3_to_1_count"] == 1
    assert summary["entity_all_flagged_count"] == 1
    assert summary["qwen_selected_flagged_count"] == 2
    assert summary["potential_input_images_before"] == 30
    assert summary["potential_input_images_after"] == 18
    assert summary["potential_input_images_reduced"] == 12
    assert summary["potential_qwen_calls_skipped"] == 1
    assert summary["by_reference_type"]["object"]["combined_flag_count"] == 0
    assert summary["by_reference_type"]["group"]["combined_flag_count"] == 0

    review_lists = simulation["review_lists"]
    assert len(review_lists["near_silhouette_cases"]) == 3
    assert len(review_lists["relative_blur_cases"]) == 3
    assert len(review_lists["qwen_selected_flagged_cases"]) == 2
    assert len(review_lists["all_candidates_flagged_cases"]) == 1
    assert review_lists["all_candidates_flagged_cases"][0]["candidates"][0][
        "technical_metrics"
    ]["luma_mean"] == 10.0


def test_rule_selection_disables_the_other_shadow_condition(
    shadow_inputs: tuple[Path, Path, Path, Path],
) -> None:
    _, audit_root, audit_parent, _ = shadow_inputs
    simulation = simulate_reference_prefilter(
        audit_root=audit_root,
        output=audit_parent / "near-only.json",
        rule="near_silhouette",
    )

    assert simulation["summary"]["near_silhouette_flag_count"] == 3
    assert simulation["summary"]["relative_blur_flag_count"] == 0
    assert all(
        candidate["relative_blur_flag"] is False
        for candidate in simulation["candidates"]
    )


@pytest.mark.parametrize(
    ("mode", "expected_count"),
    [
        ("near_silhouette", 3),
        ("relative_blur", 3),
        ("qwen_selected_flagged", 2),
        ("all_candidates_flagged", 3),
        ("all", 6),
    ],
)
def test_shadow_review_board_modes_and_read_only_inputs(
    shadow_inputs: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_count: int,
) -> None:
    run_root, audit_root, audit_parent, review_parent = shadow_inputs
    simulation_path = audit_parent / f"simulation-{mode}.json"
    simulate_reference_prefilter(audit_root=audit_root, output=simulation_path)
    before = {
        root: snapshot_run_files(root) for root in (run_root, audit_root)
    }
    simulation_sha = _sha256(simulation_path)
    labels: list[str] = []
    original_draw_text = board_module._draw_text

    def capture_text(*args: object, **kwargs: object) -> None:
        labels.append(str(args[2]))
        original_draw_text(*args, **kwargs)

    monkeypatch.setattr(board_module, "_draw_text", capture_text)
    output = review_parent / mode

    summary = build_prefilter_shadow_review_board(
        run_root=run_root,
        audit_root=audit_root,
        simulation=simulation_path,
        output_root=output,
        mode=mode,
    )

    assert summary["case_count"] == expected_count
    assert summary["qwen_calls_added"] == 0
    assert summary["all_inputs_unchanged"] is True
    assert all(snapshot_run_files(root) == before[root] for root in before)
    assert _sha256(simulation_path) == simulation_sha
    cases = json.loads((output / "review_cases.json").read_text())
    assert len(cases) == expected_count
    assert len(list((output / "cases").glob("*.png"))) == expected_count
    if cases:
        with Image.open(output / cases[0]["board_path"]) as board:
            assert board.size == (1320, 760)
    assert any("shadow rules:" in label for label in labels)
    assert any("lap_ratio=" in label for label in labels)
    forbidden = {"pass", "fail", "drop", "reject"}
    assert not any(label.strip().lower() in forbidden for label in labels)
    assert "V3 Prefilter Shadow Review" in (
        output / "review_index.html"
    ).read_text(encoding="utf-8")


def test_shadow_tools_contain_no_model_or_production_calls() -> None:
    source = inspect.getsource(simulator_module) + inspect.getsource(board_module)
    assert "OpenAI" not in source
    assert "QwenEntityReferenceJudge" not in source
    assert "chat.completions" not in source
    assert "run_reference_filter_audit" not in source
    assert "write_references" not in source
    assert "write_pairing" not in source
