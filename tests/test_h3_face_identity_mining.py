from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.h3 import (
    audio_binding,
    audio_export,
    audio_pairing,
    embedding_pilot,
    fusion,
    primary_voice,
)
from r2v_data_v2.h3.audio_backends import EmbeddingResult, FaceEmbeddingResult
from r2v_data_v2.h3.face_identity_mining import mine_face_identity_candidates
from r2v_data_v2.h3.face_identity_review import _review_rows, build_face_identity_review
from r2v_data_v2.h3.face_label_planning import (
    FaceLabelAudioPlan,
    plan_audio_from_face_labels,
)
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


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _initialize_run(root: Path) -> None:
    root.mkdir()
    (root / "clips").mkdir()
    (root / "run.json").write_text(
        RunRecord(
            run_id="face-mining-fixture",
            created_at="2026-08-17T00:00:00+00:00",
            git_commit="a" * 40,
            config_hash="b" * 64,
            model_identifiers={},
            source_manifest_path="/fixture/source.json",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_clip(
    run_root: Path,
    *,
    clip_uid: str,
    parent_video_id: str,
    clip_suffix: str,
) -> Path:
    media = run_root.parent / "media"
    media.mkdir(exist_ok=True)
    video = media / f"{clip_uid}.mp4"
    video.write_bytes(f"video:{clip_uid}".encode())
    entity = AnnotationEntity(
        entity_id="e1",
        reference_type="subject",
        phrase=f"person in {clip_uid}",
        grounding_prompt=f"person in {clip_uid}",
    )
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
        source_clip_uid=clip_uid,
        source_entity_id="e1",
        image_quality="high",
        completeness="complete",
        synthetic=False,
    )
    clip = ClipRecord(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(video),
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
        coverage=CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            required_visible_frames=7,
            entity_visibility_summary={
                "e1": EntityVisibilitySummary(
                    status="ready",
                    visible_frame_slots=list(range(10)),
                    visible_frame_count=10,
                    coverage_ratio=1.0,
                    qualifies=True,
                    per_frame_area_ratio=[0.2] * 10,
                    per_frame_confidence=[0.95] * 10,
                )
            },
        ),
        references=ReferencesState(entities=[reference]),
        pairing=PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
        ),
    )
    clip_dir = run_root / "clips" / clip_uid
    clip_dir.mkdir()
    (clip_dir / "clip.json").write_text(
        clip.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "frames.json").write_text(
        SampledFramesArtifact(
            clip_uid=clip_uid,
            width=8,
            height=8,
            frames=[
                SampledFrame(
                    slot=slot,
                    source_frame_index=slot,
                    timestamp_seconds=float(slot),
                    image_path=f"frames/{slot:02d}.jpg",
                    sha256="0" * 64,
                )
                for slot in range(10)
            ],
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 2:6] = 1
    tracked = [
        TrackedMaskFrame(
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
        for slot in range(10)
    ]
    (clip_dir / "masks.rle.json").write_text(
        TrackedMasksArtifact(
            clip_uid=clip_uid,
            width=8,
            height=8,
            entities={
                "e1": TrackedEntityMasks(
                    status="ready",
                    reference_type="subject",
                    grounding_prompt=entity.grounding_prompt,
                    backend_object_ids=["subject-1"],
                    frames=tracked,
                )
            },
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    selected = clip_dir / "selected"
    selected.mkdir()
    canonical = selected / "e1.png"
    Image.new("RGB", (12, 10), (30, 60, 90)).save(canonical)
    (frames_dir / "decoy.jpg").write_bytes(b"must-not-be-used")
    return canonical


class _FaceBackend:
    model_identifier = "test/buffalo_l"
    checkpoint_sha256 = "f" * 64

    def __init__(self, results: dict[str, np.ndarray | str]) -> None:
        self.results = results
        self.requests: list[tuple[str, Path]] = []

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        self.requests.append((entity_occurrence_id, image_path))
        value = self.results[entity_occurrence_id]
        if isinstance(value, str):
            if value == "raise":
                raise RuntimeError("fixture face runtime failure")
            return FaceEmbeddingResult(status="unavailable", reason=value)
        return FaceEmbeddingResult(
            status="available",
            embedding=EmbeddingResult(
                vector=value,
                model_identifier=self.model_identifier,
                checkpoint_sha256=self.checkpoint_sha256,
            ),
            face_crop=Image.new("RGB", (8, 8), (120, 80, 40)),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mining_fixture(tmp_path: Path) -> tuple[Path, Path, _FaceBackend]:
    run = tmp_path / "visual-run"
    _initialize_run(run)
    canonical = {
        clip_uid: _write_clip(
            run,
            clip_uid=clip_uid,
            parent_video_id=parent,
            clip_suffix=suffix,
        )
        for clip_uid, parent, suffix in (
            ("clip-a", "parent-a", "1"),
            ("clip-b", "parent-a", "2"),
            ("clip-c", "parent-a", "3"),
            ("clip-d", "parent-b", "1"),
        )
    }
    backend = _FaceBackend(
        {
            "clip-a/e1": np.array([3.0, 0.0]),
            "clip-b/e1": np.array([4.0, 3.0]),
            "clip-c/e1": np.array([0.0, 2.0]),
            "clip-d/e1": np.array([-1.0, 0.0]),
        }
    )
    output = tmp_path / "audio" / "pair_calibration" / "face_mining"
    before = _tree_hashes(run)
    summary = mine_face_identity_candidates(
        run_root=run,
        output_root=output,
        face_backend=backend,
        top_k=2,
    )
    assert summary.occurrence_count == 4
    assert _tree_hashes(run) == before
    assert [request[1] for request in backend.requests] == [
        canonical[clip_uid] for clip_uid in sorted(canonical)
    ]
    return run, output, backend


def test_face_mining_is_visual_only_exact_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    run, output, _ = _mining_fixture(tmp_path)

    assert not (run / "audio_bindings.jsonl").exists()
    vector = np.load(output / "embeddings/face/clip-a/e1.npy")
    assert vector.dtype == np.float32
    assert np.array_equal(vector, np.array([1.0, 0.0], dtype=np.float32))
    pairs = _read_jsonl(output / "face_pairs.jsonl")
    ab = next(
        item
        for item in pairs
        if item["left_occurrence_id"] == "clip-a/e1"
        and item["right_occurrence_id"] == "clip-b/e1"
    )
    assert ab["face_similarity"] == pytest.approx(0.8)
    candidates = _read_jsonl(output / "face_candidates.jsonl")
    same_parent = [item for item in candidates if item["candidate_pool"] == "same_parent"]
    assert len(same_parent) == 3
    assert all(item["thresholds_calibrated"] is False for item in candidates)
    assert all(item["same_person_label"] is None for item in candidates)
    assert all("accepted" not in item and "cross_pair_eligible" not in item for item in candidates)
    ab_candidate = next(
        item
        for item in same_parent
        if item["left_occurrence_id"] == "clip-a/e1"
        and item["right_occurrence_id"] == "clip-b/e1"
    )
    assert ab_candidate["left_to_right_rank"] == 1
    assert ab_candidate["right_to_left_rank"] == 1
    assert ab_candidate["mutual_top_k"] is True

    repeat_backend = _FaceBackend(
        {occurrence_id: value for occurrence_id, value in {
            "clip-a/e1": np.array([3.0, 0.0]),
            "clip-b/e1": np.array([4.0, 3.0]),
            "clip-c/e1": np.array([0.0, 2.0]),
            "clip-d/e1": np.array([-1.0, 0.0]),
        }.items()}
    )
    repeat = tmp_path / "repeat"
    mine_face_identity_candidates(
        run_root=run,
        output_root=repeat,
        face_backend=repeat_backend,
        top_k=2,
    )
    for relative in (
        "occurrences.jsonl",
        "face_pairs.jsonl",
        "face_candidates.jsonl",
        "summary.json",
        "embeddings/face/clip-a/e1.npy",
        "face_crops/clip-b/e1.png",
    ):
        assert (output / relative).read_bytes() == (repeat / relative).read_bytes()


def test_zero_and_multiple_faces_fail_closed_without_fallback(tmp_path: Path) -> None:
    run = tmp_path / "visual-run"
    _initialize_run(run)
    canonical_a = _write_clip(
        run,
        clip_uid="clip-a",
        parent_video_id="parent-a",
        clip_suffix="1",
    )
    canonical_b = _write_clip(
        run,
        clip_uid="clip-b",
        parent_video_id="parent-a",
        clip_suffix="2",
    )
    backend = _FaceBackend(
        {"clip-a/e1": "face_not_found", "clip-b/e1": "multiple_faces"}
    )
    output = tmp_path / "face_mining"

    summary = mine_face_identity_candidates(
        run_root=run,
        output_root=output,
        face_backend=backend,
    )

    assert backend.requests == [("clip-a/e1", canonical_a), ("clip-b/e1", canonical_b)]
    assert summary.face_embedding_available_count == 0
    assert summary.face_embedding_unavailable_count == 2
    rows = _read_jsonl(output / "occurrences.jsonl")
    assert [item["face"]["failure_reason"] for item in rows] == [
        "face_not_found_in_canonical_reference",
        "multiple_faces_in_canonical_reference",
    ]
    assert _read_jsonl(output / "face_pairs.jsonl") == []


def test_review_emits_each_unordered_pair_once_and_human_label_plan_is_exact(
    tmp_path: Path,
) -> None:
    _, mining, _ = _mining_fixture(tmp_path)
    rows = _review_rows(mining)
    assert len(rows) == 3
    assert len(
        {
            (item["left_occurrence_id"], item["right_occurrence_id"])
            for item in rows
        }
    ) == 3
    review = build_face_identity_review(mining)
    html = review.read_text(encoding="utf-8")
    assert "Are these two occurrences the same physical person?" in html
    assert "1 SAME" in html and "2 DIFFERENT" in html and "3 UNCERTAIN" in html
    assert "localStorage" in html and "Export JSONL" in html

    labels = tmp_path / "face_identity_labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "left_occurrence_id",
                        "right_occurrence_id",
                        "face_similarity",
                        "left_to_right_rank",
                        "right_to_left_rank",
                        "mutual_top_k",
                        "parent_video_id",
                    )
                }
                | {"same_person_label": label},
                sort_keys=True,
            )
            + "\n"
            for row, label in zip(rows, ("same", "different", "uncertain"), strict=True)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audio-plan"
    plan = plan_audio_from_face_labels(
        face_mining_root=mining,
        labels_path=labels,
        output_root=output,
    )

    first = rows[0]
    expected_clips = [
        str(first["left_occurrence_id"]).split("/", maxsplit=1)[0],
        str(first["right_occurrence_id"]).split("/", maxsplit=1)[0],
    ]
    assert plan.selected_clip_ids == expected_clips
    assert plan.confirmed_face_pair_count == 1
    assert plan.excluded_different_pair_count == 1
    assert plan.excluded_uncertain_pair_count == 1
    assert plan.parent_count_quota_applied is False
    assert (output / "clip_ids.txt").read_text().splitlines() == expected_clips
    confirmed = _read_jsonl(output / "confirmed_face_pairs.jsonl")
    assert len(confirmed) == 1
    assert confirmed[0]["same_person_label"] == "same"
    assert confirmed[0]["label_source"] == "human_face_identity_review"
    assert confirmed[0]["thresholds_calibrated"] is False
    assert "max_clips_per_parent" not in json.loads(
        (output / "plan.json").read_text()
    )

    limited = tmp_path / "limited-audio-plan"
    limited_plan = plan_audio_from_face_labels(
        face_mining_root=mining,
        labels_path=labels,
        output_root=limited,
        max_clips=1,
    )
    assert limited_plan.selected_clip_ids == []
    assert limited_plan.confirmed_face_pair_count == 0
    assert limited_plan.skipped_same_pair_due_to_global_limit_count == 1
    assert (limited / "clip_ids.txt").read_text() == ""


def test_production_h3_paths_have_no_parent_count_truncation_contract() -> None:
    for module in (
        audio_binding,
        audio_export,
        audio_pairing,
        embedding_pilot,
        fusion,
        primary_voice,
    ):
        assert "max_clips_per_parent" not in inspect.getsource(module)
    assert "max_clips_per_parent" not in FaceLabelAudioPlan.model_fields
