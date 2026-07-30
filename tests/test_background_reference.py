from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from r2v_data_v2.background_reference import build_background_references
from r2v_data_v2.config import BackgroundConfig, PipelineConfig
from r2v_data_v2.mask_utils import encode_mask
from r2v_data_v2.schemas import BackgroundJudgeResult


def _annotation(*, with_entity: bool) -> dict[str, object]:
    entities: list[dict[str, object]] = []
    if with_entity:
        entities.append(
            {
                "entity_id": "e1",
                "phrase": "a red bicycle",
                "grounding_prompt": "red bicycle",
                "canonical_label": "red bicycle",
                "category": "vehicle",
                "reference_worthy": True,
                "salience": "primary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "independent",
                "selection_reason": "primary subject",
                "ref_token": "<ref_subject_1>",
            }
        )
    return {
        "clip_uid": "clip-1",
        "caption": "A red bicycle is parked beside a brick wall.",
        "prompt_with_refs": (
            "<ref_subject_1> is parked beside a brick wall."
            if with_entity
            else "A quiet brick courtyard at noon."
        ),
        "entities": entities,
        "relations": [],
        "background": {
            "phrase": "a quiet brick courtyard at noon",
            "grounding_prompt": "quiet brick courtyard, daylight",
            "reference_worthy": True,
            "ref_token": None,
        },
    }


def _write_fixture(
    output_root: Path,
    *,
    foreground_ratio: float | None,
) -> None:
    manifest = output_root / "manifests" / "annotations.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(_annotation(with_entity=foreground_ratio is not None)) + "\n",
        encoding="utf-8",
    )
    frame_dir = output_root / "frames" / "clip-1"
    frame_dir.mkdir(parents=True)
    y, x = np.indices((100, 100))
    frame = np.stack(
        (
            70 + (x % 80),
            80 + (y % 80),
            90 + ((x + y) % 80),
        ),
        axis=-1,
    ).astype(np.uint8)
    for slot in range(10):
        assert cv2.imwrite(str(frame_dir / f"frame_{slot:02d}.jpg"), frame)
    (frame_dir / "frames.json").write_text(
        json.dumps({"sampled_indices": list(range(100, 110))}),
        encoding="utf-8",
    )
    if foreground_ratio is None:
        return
    height = round(100 * foreground_ratio)
    encoded: dict[str, object] = {}
    for slot in range(10):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:height, :] = True
        encoded[f"frame_{slot:02d}"] = encode_mask(mask)
    candidate_dir = output_root / "candidates" / "clip-1" / "e1"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "top_masks.rle.json").write_text(
        json.dumps(encoded),
        encoding="utf-8",
    )
    (candidate_dir / "tracked_masks.rle.json").write_text(
        json.dumps(encoded),
        encoding="utf-8",
    )
    (candidate_dir / "mask_coverage.json").write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "entity_id": "e1",
                "slots": {
                    f"frame_{slot:02d}": {
                        "tracked": True,
                        "mask_available": True,
                        "candidate_accepted": True,
                        "filtered_reasons": [],
                    }
                    for slot in range(10)
                },
            }
        ),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    background: BackgroundConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        background=background or BackgroundConfig(),
    )


def _metadata(output_root: Path) -> dict[str, object]:
    return json.loads(
        (
            output_root
            / "references"
            / "clip-1"
            / "bg1"
            / "reference_metadata.json"
        ).read_text(encoding="utf-8")
    )


def test_background_only_clip_produces_raw_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=None)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.processed == 1
    assert stats.raw_background_count == 1
    assert metadata["reference_type"] == "background"
    assert metadata["reference_id"] == "bg1"
    assert metadata["ref_token"] == "<ref_bg_1>"
    assert metadata["entity_id"] is None
    assert metadata["needs_inpainting"] is False
    assert metadata["inpainted"] is False
    assert metadata["status"] == "ready"
    assert Path(str(metadata["raw_canonical_path"])).is_file()
    assert Path(str(metadata["canonical_path"])).is_file()
    assert len(list((config.output_root / "references" / "clip-1").iterdir())) == 1


def test_low_foreground_ratio_keeps_raw_frame(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.04)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.raw_background_count == 1
    assert metadata["foreground_area_ratio"] == 0.04
    assert metadata["needs_inpainting"] is False


def test_medium_foreground_ratio_marks_candidate_for_inpainting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.needs_inpainting_count == 1
    assert metadata["foreground_area_ratio"] == 0.10
    assert metadata["needs_inpainting"] is True
    assert metadata["status"] == "pending_inpainting"


def test_non_reference_worthy_separable_entity_is_in_foreground_mask(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)
    manifest = config.output_root / "manifests" / "annotations.jsonl"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entities"][0]["reference_worthy"] = False
    payload["entities"][0]["ref_token"] = None
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.processed == 1
    assert metadata["foreground_area_ratio"] == 0.10
    assert metadata["needs_inpainting"] is True


def test_composite_candidate_requires_foreground_mask(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)
    manifest = config.output_root / "manifests" / "annotations.jsonl"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entities"][0]["reference_worthy"] = False
    payload["entities"][0]["ref_token"] = None
    payload["entities"][0]["separability"] = "composite_candidate"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.processed == 1
    assert metadata["foreground_area_ratio"] == 0.10
    assert metadata["needs_inpainting"] is True


def test_background_uses_tracked_mask_rejected_by_entity_size_gate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    accepted = json.loads(
        (candidate_dir / "top_masks.rle.json").read_text(encoding="utf-8")
    )
    tracked = dict(accepted)
    small_mask = np.zeros((100, 100), dtype=bool)
    small_mask[:2, :] = True
    tracked["frame_00"] = encode_mask(small_mask)
    accepted.pop("frame_00")
    (candidate_dir / "top_masks.rle.json").write_text(
        json.dumps(accepted),
        encoding="utf-8",
    )
    (candidate_dir / "tracked_masks.rle.json").write_text(
        json.dumps(tracked),
        encoding="utf-8",
    )
    (candidate_dir / "mask_coverage.json").write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "entity_id": "e1",
                "slots": {
                    f"frame_{slot:02d}": {
                        "tracked": True,
                        "mask_available": True,
                        "candidate_accepted": slot != 0,
                        "filtered_reasons": (
                            ["effective_short_side"] if slot == 0 else []
                        ),
                    }
                    for slot in range(10)
                },
            }
        ),
        encoding="utf-8",
    )

    stats = build_background_references(config)
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    frame_zero = next(
        candidate
        for candidate in ranking["candidates"]
        if candidate["frame_slot"] == 0
    )

    assert stats.processed == 1
    assert frame_zero["foreground_area_ratio"] == 0.02
    assert ranking["incomplete_mask_slots"] == []
    assert ranking["mask_coverage"]["frame_00"][0][
        "candidate_accepted"
    ] is False
    assert ranking["mask_coverage"]["frame_00"][0]["filtered_reasons"] == [
        "effective_short_side"
    ]


def test_clean_raw_candidate_preferred_over_higher_scoring_inpaint_candidate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    tracked = json.loads(
        (candidate_dir / "tracked_masks.rle.json").read_text(
            encoding="utf-8"
        )
    )
    clean_mask = np.zeros((100, 100), dtype=bool)
    clean_mask[:4, :] = True
    tracked["frame_00"] = encode_mask(clean_mask)
    (candidate_dir / "tracked_masks.rle.json").write_text(
        json.dumps(tracked),
        encoding="utf-8",
    )
    frame_dir = config.output_root / "frames" / "clip-1"
    flat = np.full((100, 100, 3), 110, dtype=np.uint8)
    checker = (
        (np.indices((100, 100)).sum(axis=0) % 2) * 180 + 30
    ).astype(np.uint8)
    textured = np.repeat(checker[..., None], 3, axis=2)
    assert cv2.imwrite(str(frame_dir / "frame_00.jpg"), flat)
    for slot in range(1, 10):
        assert cv2.imwrite(str(frame_dir / f"frame_{slot:02d}.jpg"), textured)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    selected_score = next(
        candidate["ranking_score"]
        for candidate in ranking["candidates"]
        if candidate["frame_slot"] == 0
    )
    highest_pending_score = max(
        candidate["ranking_score"]
        for candidate in ranking["candidates"]
        if candidate["frame_slot"] != 0
    )

    assert stats.raw_background_count == 1
    assert metadata["frame_slot"] == 0
    assert metadata["status"] == "ready"
    assert highest_pending_score > selected_score


def test_hole_ratio_over_limit_is_quality_rejection(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        background=BackgroundConfig(maximum_hole_area_ratio=0.08),
    )
    _write_fixture(config.output_root, foreground_ratio=0.10)

    stats = build_background_references(config)

    assert stats.no_valid_candidate == 1
    assert stats.failed == 0
    assert not (
        config.output_root
        / "references"
        / "clip-1"
        / "bg1"
        / "reference_metadata.json"
    ).exists()
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        "hole_area_ratio" in candidate["rejection_reasons"]
        for candidate in ranking["candidates"]
    )


def test_disabled_background_stage_is_noop(tmp_path: Path) -> None:
    config = _config(tmp_path, background=BackgroundConfig(enabled=False))

    stats = build_background_references(config)

    assert stats.processed == 0
    assert not config.output_root.exists()


def test_legacy_candidate_masks_are_disabled_by_default(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    (candidate_dir / "tracked_masks.rle.json").unlink()
    (candidate_dir / "mask_coverage.json").unlink()

    class _NeverDino:
        def encode(self, images: list[object]) -> np.ndarray:
            del images
            raise AssertionError("incomplete masks must skip semantic ranking")

    class _NeverSiglip:
        def score(self, *args: object) -> list[object]:
            del args
            raise AssertionError("incomplete masks must skip semantic ranking")

    stats = build_background_references(
        config,
        dino_embedder=_NeverDino(),  # type: ignore[arg-type]
        siglip_aligner=_NeverSiglip(),  # type: ignore[arg-type]
    )

    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert stats.no_valid_candidate == 1
    assert stats.failed == 0
    assert ranking["incomplete_mask_entities"] == ["e1"]
    assert ranking["incomplete_mask_slots"] == list(range(10))
    assert all(
        candidate["foreground_masks_incomplete"]
        for candidate in ranking["candidates"]
    )
    assert ranking["mask_coverage"]["frame_00"][0][
        "filtered_reasons"
    ] == ["legacy_candidate_masks_disabled"]
    assert all(
        "incomplete_foreground_masks" in candidate["rejection_reasons"]
        for candidate in ranking["candidates"]
    )


def test_enabled_legacy_masks_mark_every_missing_slot_incomplete(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        background=BackgroundConfig(allow_legacy_candidate_masks=True),
    )
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    (candidate_dir / "tracked_masks.rle.json").unlink()
    (candidate_dir / "mask_coverage.json").unlink()
    legacy = json.loads(
        (candidate_dir / "top_masks.rle.json").read_text(encoding="utf-8")
    )
    legacy.pop("frame_00")
    (candidate_dir / "top_masks.rle.json").write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )

    stats = build_background_references(config)
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    frame_zero = next(
        candidate
        for candidate in ranking["candidates"]
        if candidate["frame_slot"] == 0
    )

    assert stats.processed == 1
    assert ranking["incomplete_mask_slots"] == [0]
    assert frame_zero["foreground_masks_incomplete"] is True
    assert "incomplete_foreground_masks" in frame_zero["rejection_reasons"]
    assert ranking["mask_coverage"]["frame_00"][0][
        "filtered_reasons"
    ] == ["legacy_mask_slot_missing"]


def test_missing_entity_masks_require_qwen_review_when_enabled(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        background=BackgroundConfig(qwen_judge_enabled=True),
    )
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    (candidate_dir / "tracked_masks.rle.json").unlink()
    (candidate_dir / "mask_coverage.json").unlink()

    class _Judge:
        def __init__(self) -> None:
            self.calls = 0

        def review(
            self,
            *,
            background_phrase: str,
            contact_sheet: Path,
            frame_slots: list[int],
            incomplete_mask_entities: tuple[str, ...],
        ) -> BackgroundJudgeResult:
            self.calls += 1
            assert background_phrase
            assert contact_sheet.is_file()
            assert incomplete_mask_entities == ("e1",)
            return BackgroundJudgeResult.model_validate(
                {
                    "candidates": [
                        {
                            "frame_slot": slot,
                            "scene_completeness": 0.9,
                            "scene_recognizability": 0.9,
                            "foreground_distraction": 0.1,
                            "visual_quality": 0.9,
                            "reusable_as_background": True,
                            "rejection_reasons": [],
                        }
                        for slot in frame_slots
                    ],
                    "best_frame_slot": frame_slots[0],
                }
            )

    judge = _Judge()
    stats = build_background_references(
        config,
        judge=judge,  # type: ignore[arg-type]
    )

    assert judge.calls == 1
    assert stats.processed == 1
    assert _metadata(config.output_root)["status"] == "ready"


def test_qwen_hard_rejects_distraction_when_masks_are_incomplete(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        background=BackgroundConfig(qwen_judge_enabled=True),
    )
    _write_fixture(config.output_root, foreground_ratio=0.10)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    (candidate_dir / "tracked_masks.rle.json").unlink()
    (candidate_dir / "mask_coverage.json").unlink()

    class _Judge:
        def review(
            self,
            *,
            background_phrase: str,
            contact_sheet: Path,
            frame_slots: list[int],
            incomplete_mask_entities: tuple[str, ...],
        ) -> BackgroundJudgeResult:
            assert background_phrase
            assert contact_sheet.is_file()
            assert incomplete_mask_entities == ("e1",)
            return BackgroundJudgeResult.model_validate(
                {
                    "candidates": [
                        {
                            "frame_slot": slot,
                            "scene_completeness": 0.9,
                            "scene_recognizability": 0.9,
                            "foreground_distraction": 0.11,
                            "visual_quality": 0.9,
                            "reusable_as_background": True,
                            "rejection_reasons": [],
                        }
                        for slot in frame_slots
                    ],
                    "best_frame_slot": frame_slots[0],
                }
            )

    stats = build_background_references(
        config,
        judge=_Judge(),  # type: ignore[arg-type]
    )
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    reviewed = [
        candidate
        for candidate in ranking["candidates"]
        if candidate["visual_review"] is not None
    ]

    assert stats.no_valid_candidate == 1
    assert stats.processed == 0
    assert reviewed
    assert all(
        "unmasked_foreground_distraction"
        in candidate["rejection_reasons"]
        for candidate in reviewed
    )


def test_background_ranking_uses_dino_and_siglip_semantics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.04)

    class _Dino:
        def encode(self, images: list[object]) -> np.ndarray:
            assert len(images) == 10
            embeddings = np.eye(10, dtype=np.float32)
            embeddings[8] = embeddings[7]
            embeddings[9] = embeddings[7]
            return embeddings

    class _Siglip:
        def score(
            self,
            images: list[object],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            assert len(images) == 10
            assert target_text == "a quiet brick courtyard at noon"
            assert distractor_texts == []
            return [
                SimpleNamespace(
                    target_similarity=0.95 if index == 9 else 0.1
                )
                for index in range(10)
            ]

    stats = build_background_references(
        config,
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
    )

    metadata = _metadata(config.output_root)
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    selected = next(
        candidate
        for candidate in ranking["candidates"]
        if candidate["frame_slot"] == 9
    )
    assert stats.processed == 1
    assert metadata["frame_slot"] == 9
    assert selected["dino_representativeness"] == 1.0
    assert selected["siglip_alignment"] == 0.95


def test_background_overwrite_invalidates_stale_inpainting_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=None)
    first = build_background_references(config)
    reference_dir = config.output_root / "references" / "clip-1" / "bg1"
    stale_names = (
        "repair_mask.png",
        "canonical_repaired.png",
        "canonical_repaired_candidate.png",
        "inpainting_metadata.json",
    )
    for name in stale_names:
        (reference_dir / name).write_bytes(b"stale")

    second = build_background_references(config, overwrite=True)

    assert first.processed == 1
    assert second.processed == 1
    assert all(not (reference_dir / name).exists() for name in stale_names)
