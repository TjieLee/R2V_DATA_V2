from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.h3.pair_calibration import (
    PAIR_CALIBRATION_POLICY_VERSION,
    VisualCandidatePair,
    plan_h3_pair_calibration,
)
from r2v_data_v2.h3.pilot import _selected_clip_paths
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipRecord,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    PairingState,
    ReferencesState,
    RunRecord,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from tools.eval_h3_audio_binding_lr_asd import (
    _combined_clip_ids,
    _read_clip_id_file,
)
from tools.eval_h3_audio_binding_lr_asd import (
    _parser as audio_pilot_parser,
)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _initialize_run(root: Path) -> None:
    root.mkdir()
    (root / "clips").mkdir()
    record = RunRecord(
        run_id="fixture-run",
        created_at="2026-08-17T00:00:00+00:00",
        git_commit="a" * 40,
        config_hash="b" * 64,
        model_identifiers={},
        source_manifest_path="/fixture/source.json",
    )
    (root / "run.json").write_text(
        record.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _visibility() -> EntityVisibilitySummary:
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(10)),
        visible_frame_count=10,
        coverage_ratio=1.0,
        qualifies=True,
        per_frame_area_ratio=[0.25] * 10,
        per_frame_confidence=[0.95] * 10,
    )


def _tracked_frame(slot: int) -> TrackedMaskFrame:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 2:6] = 1
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        confidence=0.95,
        backend_confidences=[0.95],
        backend_object_ids=["subject-1"],
        area_pixels=int(mask.sum()),
        area_ratio=float(mask.mean()),
        bbox_xyxy=(2, 1, 6, 7),
        rle=encode_binary_mask(mask),
    )


def _write_clip(
    run_root: Path,
    *,
    clip_uid: str,
    parent_video_id: str,
    clip_suffix: str,
    reference_type: str = "subject",
    donor: tuple[str, str] | None = None,
    canonical: bool = True,
    ready: bool = True,
) -> ClipRecord:
    media = run_root.parent / "media"
    media.mkdir(exist_ok=True)
    video_path = media / f"{clip_uid}.mp4"
    video_path.write_bytes(f"video:{clip_uid}".encode())
    entity = AnnotationEntity(
        entity_id="e1",
        reference_type=reference_type,
        phrase=f"person in {clip_uid}",
        grounding_prompt=f"person in {clip_uid}",
    )
    source_clip_uid, source_entity_id = donor or (clip_uid, "e1")
    reference = EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="fixture canonical reference",
        image_path=f"clips/{clip_uid}/selected/e1.png",
        source_frame_index=5,
        source_clip_uid=source_clip_uid,
        source_entity_id=source_entity_id,
        image_quality="high",
        completeness="complete",
        synthetic=False,
    )
    clip = ClipRecord(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(video_path),
            parent_video_id=parent_video_id,
            clip_suffix=clip_suffix,
            source_index=0,
            caption_raw="fixture",
            metadata={},
        ),
        annotation=AnnotationState(
            status="ready",
            instruction_template="{{entity_1}} crosses the frame.",
            entities=[entity],
        ),
        coverage=(
            CoverageState(
                passed=True,
                qualifying_entity_ids=["e1"],
                required_visible_frames=7,
                entity_visibility_summary={"e1": _visibility()},
            )
            if ready
            else None
        ),
        references=ReferencesState(entities=[reference]) if ready else ReferencesState(),
        pairing=(
            PairingState(
                status="ready",
                retained_entity_ids=["e1"],
                tokens={"e1": f"<ref_{reference_type}_1>"},
            )
            if ready
            else None
        ),
    )
    clip_dir = run_root / "clips" / clip_uid
    clip_dir.mkdir()
    (clip_dir / "clip.json").write_text(
        clip.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir()
    frames = SampledFramesArtifact(
        clip_uid=clip_uid,
        width=8,
        height=8,
        frames=[
            SampledFrame(
                slot=slot,
                source_frame_index=slot,
                timestamp_seconds=float(slot + 1),
                image_path=f"frames/{slot:02d}.jpg",
                sha256="0" * 64,
            )
            for slot in range(10)
        ],
    )
    (frames_dir / "frames.json").write_text(
        frames.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    masks = TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=8,
        height=8,
        entities={
            "e1": TrackedEntityMasks(
                status="ready",
                reference_type=reference_type,
                grounding_prompt=entity.grounding_prompt,
                backend_object_ids=["subject-1"],
                frames=[_tracked_frame(slot) for slot in range(10)],
            )
        },
    )
    (clip_dir / "masks.rle.json").write_text(
        masks.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if canonical and ready:
        selected = clip_dir / "selected"
        selected.mkdir()
        Image.new("RGB", (8, 8), (20, 30, 40)).save(selected / "e1.png")
    return clip


def _write_seed(seed_root: Path, clip_ids: list[str]) -> None:
    seed_root.mkdir()
    (seed_root / "audio_bindings.jsonl").write_text(
        "".join(json.dumps({"clip_uid": clip_uid}) + "\n" for clip_uid in clip_ids),
        encoding="utf-8",
    )


def _pair_rows(output_root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_root / "visual_candidate_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


def test_plan_preserves_deduplicated_seeds_then_priority_a_endpoints(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _initialize_run(run)
    _write_clip(
        run,
        clip_uid="clip-a",
        parent_video_id="parent-a",
        clip_suffix="1",
    )
    _write_clip(
        run,
        clip_uid="clip-b",
        parent_video_id="parent-a",
        clip_suffix="2",
        donor=("clip-a", "e1"),
    )
    _write_clip(
        run,
        clip_uid="clip-c",
        parent_video_id="parent-b",
        clip_suffix="1",
    )
    _write_clip(
        run,
        clip_uid="clip-d",
        parent_video_id="parent-b",
        clip_suffix="2",
    )
    seed = tmp_path / "seed"
    _write_seed(seed, ["clip-c", "clip-c"])

    plan = plan_h3_pair_calibration(
        run_root=run,
        output_root=tmp_path / "plan",
        seed_audio_pilot_root=seed,
        max_clips=4,
        max_clips_per_parent=4,
    )

    assert [item.clip_uid for item in plan.selected_clips] == [
        "clip-c",
        "clip-a",
        "clip-b",
        "clip-d",
    ]
    assert plan.seed_clip_count == 1
    assert plan.priority_a_clip_count == 2
    assert plan.priority_b_clip_count == 1
    priority_a = [
        row
        for row in _pair_rows(tmp_path / "plan")
        if row["candidate_source"] == "existing_v3_cross_pair_provenance"
    ]
    assert len(priority_a) == 1
    assert priority_a[0]["same_person_label"] is None
    assert priority_a[0]["thresholds_calibrated"] is False
    assert priority_a[0]["same_parent"] is True


def test_priority_b_round_robin_enforces_parent_and_total_caps(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _initialize_run(run)
    for parent in ("parent-a", "parent-b"):
        for index in range(1, 4):
            _write_clip(
                run,
                clip_uid=f"{parent}-clip-{index}",
                parent_video_id=parent,
                clip_suffix=str(index),
            )

    plan = plan_h3_pair_calibration(
        run_root=run,
        output_root=tmp_path / "plan",
        max_clips=6,
        max_clips_per_parent=2,
    )

    assert [item.clip_uid for item in plan.selected_clips] == [
        "parent-a-clip-1",
        "parent-a-clip-2",
        "parent-b-clip-1",
        "parent-b-clip-2",
    ]
    assert plan.selected_clip_count == 4
    assert plan.candidate_parent_group_count == 2
    assert plan.selected_parent_group_count == 2
    assert plan.skip_reason_counts["priority_b_max_clips_per_parent_reached"] == 2
    assert all(row["same_person_label"] is None for row in _pair_rows(tmp_path / "plan"))


def test_priority_a_pair_is_not_partially_selected_when_budget_cannot_fit(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _initialize_run(run)
    _write_clip(
        run,
        clip_uid="clip-a",
        parent_video_id="parent-a",
        clip_suffix="1",
    )
    _write_clip(
        run,
        clip_uid="clip-b",
        parent_video_id="parent-a",
        clip_suffix="2",
        donor=("clip-a", "e1"),
    )

    plan = plan_h3_pair_calibration(
        run_root=run,
        output_root=tmp_path / "plan",
        max_clips=1,
        max_clips_per_parent=4,
    )

    assert plan.selected_clips == []
    assert plan.skip_reason_counts["priority_a_max_clips_reached"] == 1
    assert len(plan.skipped_priority_a_pairs) == 1
    assert _pair_rows(tmp_path / "plan") == []


def test_planner_excludes_invalid_inputs_is_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _initialize_run(run)
    _write_clip(
        run,
        clip_uid="good-a",
        parent_video_id="parent-good",
        clip_suffix="1",
    )
    _write_clip(
        run,
        clip_uid="good-b",
        parent_video_id="parent-good",
        clip_suffix="2",
    )
    _write_clip(
        run,
        clip_uid="object-only",
        parent_video_id="parent-object",
        clip_suffix="1",
        reference_type="object",
    )
    _write_clip(
        run,
        clip_uid="no-canonical",
        parent_video_id="parent-missing",
        clip_suffix="1",
        canonical=False,
    )
    _write_clip(
        run,
        clip_uid="unready",
        parent_video_id="parent-unready",
        clip_suffix="1",
        ready=False,
    )
    malformed = run / "clips" / "malformed"
    malformed.mkdir()
    (malformed / "clip.json").write_text("not json\n", encoding="utf-8")
    before = _tree_hashes(run)

    first = plan_h3_pair_calibration(
        run_root=run,
        output_root=tmp_path / "plan-1",
        max_clips=60,
        max_clips_per_parent=4,
    )
    second = plan_h3_pair_calibration(
        run_root=run,
        output_root=tmp_path / "plan-2",
        max_clips=60,
        max_clips_per_parent=4,
    )

    assert [item.clip_uid for item in first.selected_clips] == ["good-a", "good-b"]
    assert first == second
    assert (tmp_path / "plan-1" / "clip_ids.txt").read_bytes() == (
        tmp_path / "plan-2" / "clip_ids.txt"
    ).read_bytes()
    assert (tmp_path / "plan-1" / "visual_candidate_pairs.jsonl").read_bytes() == (
        tmp_path / "plan-2" / "visual_candidate_pairs.jsonl"
    ).read_bytes()
    assert first.skip_reason_counts["no_ready_human_subject"] == 1
    assert first.skip_reason_counts["coverage_not_passed"] == 1
    assert first.skip_reason_counts["invalid_visual_clip"] == 2
    assert first.thresholds_calibrated is False
    assert first.deterministic_selection_policy_version == (
        PAIR_CALIBRATION_POLICY_VERSION
    )
    assert _tree_hashes(run) == before


def test_clip_id_file_parsing_and_combination_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "clip_ids.txt"
    path.write_text(
        "# selected calibration clips\nclip-b\n\nclip-a\nclip-b\n",
        encoding="utf-8",
    )

    assert _read_clip_id_file(path) == ["clip-b", "clip-a", "clip-b"]
    assert _combined_clip_ids(["clip-c", "clip-b"], path) == [
        "clip-c",
        "clip-b",
        "clip-a",
    ]
    source = tmp_path / "run"
    for clip_uid in ("clip-a", "clip-b", "clip-c"):
        destination = source / "clips" / clip_uid / "clip.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")
    assert [
        item.parent.name
        for item in _selected_clip_paths(
            source,
            clip_ids=["clip-c", "clip-b", "clip-c", "clip-a"],
            limit=None,
        )
    ] == ["clip-c", "clip-b", "clip-a"]
    path.write_text("clip-a\n../escape\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _read_clip_id_file(path)


def test_audio_cli_workers_semantics_and_candidate_schema_remain_conservative(
    tmp_path: Path,
) -> None:
    required = ["--run-root", str(tmp_path), "--output-root", str(tmp_path / "out")]
    parser = audio_pilot_parser()
    assert parser.parse_args(required).workers == 1
    assert parser.parse_args([*required, "--workers", "4"]).workers == 4
    pair = VisualCandidatePair(
        left_clip_uid="clip-a",
        left_entity_id="e1",
        right_clip_uid="clip-b",
        right_entity_id="e1",
        same_parent=True,
        candidate_source="same_parent_multi_clip",
        provenance={"parent_video_id": "parent"},
    )
    payload = pair.model_dump(mode="json")
    assert payload["same_person_label"] is None
    assert payload["thresholds_calibrated"] is False
    assert not any("threshold" in key for key in payload if key != "thresholds_calibrated")
