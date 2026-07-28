from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.config import PipelineConfig, QwenConfig, RankingConfig
from r2v_data_v2.mask_utils import encode_mask
from r2v_data_v2.metrics import CandidateMetrics, classify_entity_overlap
from r2v_data_v2.ranking import (
    _candidate_sheet_label,
    _top_candidate_sheet,
    basic_hard_rejection_reasons,
    rank_candidates,
    rank_manifest_references,
    siglip_distractor_entities,
)
from r2v_data_v2.schemas import (
    AnnotationResult,
    CandidateJudgeResult,
    CandidateVisualReview,
)
from r2v_data_v2.semantic_alignment import AlignmentMetrics
from r2v_data_v2.visual_embedding import TemporalRepresentationMetrics


def _metrics(
    *,
    border_touch: bool,
    sharpness: float,
    exposure: float = 0.9,
    crop_subject_ratio: float = 0.7,
    other_overlap: float = 0.0,
) -> CandidateMetrics:
    return CandidateMetrics(
        effective_short_side=300,
        mask_area_ratio=0.2,
        border_touch=border_touch,
        laplacian_variance=sharpness,
        tenengrad_sharpness=sharpness,
        exposure_score=exposure,
        mask_area_continuity=0.9,
        maximum_other_mask_overlap=other_overlap,
        maximum_bbox_containment=0.0,
        crop_subject_ratio=crop_subject_ratio,
    )


def _review(slot: int, visual_quality: float = 0.8) -> CandidateVisualReview:
    return CandidateVisualReview(
        frame_slot=slot,
        completeness=0.9,
        recognizability=0.9,
        occlusion=0.1,
        mask_quality=0.9,
        visual_quality=visual_quality,
        identity_features_visible=True,
        rejection_reasons=[],
    )


def test_hard_gate_beats_weighted_score() -> None:
    config = RankingConfig()
    invalid = (
        0,
        10,
        0.99,
        _metrics(border_touch=True, sharpness=1000),
        _review(0, 1.0),
    )
    valid = (
        1,
        20,
        0.70,
        _metrics(border_touch=False, sharpness=10),
        _review(1, 0.5),
    )

    ranked = rank_candidates([invalid, valid], config=config)

    assert ranked[0].frame_slot == 1
    assert ranked[0].hard_rejection_reasons == ()
    assert "border_touch" in ranked[1].hard_rejection_reasons


def test_low_completeness_is_hard_failure() -> None:
    review = _review(0).model_copy(update={"completeness": 0.2})
    ranked = rank_candidates(
        [(0, 10, 0.9, _metrics(border_touch=False, sharpness=10), review)],
        config=RankingConfig(),
    )
    assert "incomplete" in ranked[0].hard_rejection_reasons


def test_entity_overlap_classification() -> None:
    attached = classify_entity_overlap(
        statistics={
            "mask_iou": 0.4,
            "left_contained_in_right": 0.85,
            "right_contained_in_left": 0.2,
            "bbox_iou": 0.5,
        },
        temporal_cooccurrence=0.9,
        relation="holding",
        child_has_independent_candidate=False,
    )
    important = classify_entity_overlap(
        statistics={
            "mask_iou": 0.4,
            "left_contained_in_right": 0.85,
            "right_contained_in_left": 0.2,
            "bbox_iou": 0.5,
        },
        temporal_cooccurrence=0.9,
        relation="holding",
        child_has_independent_candidate=True,
    )
    duplicate = classify_entity_overlap(
        statistics={
            "mask_iou": 0.9,
            "left_contained_in_right": 0.95,
            "right_contained_in_left": 0.95,
            "bbox_iou": 0.9,
        },
        temporal_cooccurrence=1.0,
        relation=None,
        child_has_independent_candidate=False,
    )

    assert attached == "attached_accessory"
    assert important == "important_independent_object"
    assert duplicate == "duplicate_entity"


def test_candidate_sheet_contains_three_panels_and_metric_label(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame = np.full((120, 180, 3), (30, 120, 220), dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), frame)
    mask = np.zeros((120, 180), dtype=bool)
    mask[25:100, 55:135] = True
    metrics = _metrics(border_touch=False, sharpness=123.4)
    record: dict[str, object] = {
        "frame_slot": 2,
        "sam_confidence": 0.91,
    }
    destination = tmp_path / "sheet.jpg"

    _top_candidate_sheet(
        frame_paths={2: frame_path},
        masks={2: mask},
        candidates=[(record, metrics)],
        destination=destination,
    )

    with Image.open(destination) as sheet:
        assert sheet.size == (1440, 362)
    label = _candidate_sheet_label(record, metrics)
    assert "frame_slot=2" in label
    assert "SAM=0.910" in label
    assert "effective_short_side=300" in label
    assert "mask_area_ratio=0.2000" in label
    assert "Tenengrad=123.4" in label
    assert "border_touch=false" in label


def test_qwen_best_frame_does_not_override_code_hard_gate() -> None:
    ranked = rank_candidates(
        [
            (
                0,
                10,
                0.99,
                _metrics(border_touch=True, sharpness=100),
                _review(0),
            ),
            (
                1,
                20,
                0.90,
                _metrics(border_touch=False, sharpness=80),
                _review(1),
            ),
        ],
        config=RankingConfig(),
    )

    qwen_diagnostic_best_frame_slot = 0
    assert qwen_diagnostic_best_frame_slot == 0
    assert ranked[0].frame_slot == 1
    assert ranked[0].hard_rejection_reasons == ()


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        (
            _metrics(border_touch=False, sharpness=10, exposure=0.1),
            "extreme_exposure",
        ),
        (
            _metrics(
                border_touch=False,
                sharpness=10,
                crop_subject_ratio=0.01,
            ),
            "subject_too_small_in_crop",
        ),
        (
            _metrics(
                border_touch=False,
                sharpness=10,
                crop_subject_ratio=0.99,
            ),
            "crop_too_tight",
        ),
        (
            _metrics(
                border_touch=False,
                sharpness=10,
                other_overlap=0.8,
            ),
            "other_subject_contamination",
        ),
    ],
)
def test_basic_model_free_hard_gates(
    metrics: CandidateMetrics,
    reason: str,
) -> None:
    assert reason in basic_hard_rejection_reasons(
        metrics=metrics,
        sam_confidence=0.9,
        config=RankingConfig(),
    )


def test_dino_outlier_cannot_be_rescued_by_visual_quality() -> None:
    candidates = [
        (
            0,
            10,
            0.99,
            _metrics(border_touch=False, sharpness=1000),
            _review(0, 1.0),
        ),
        (
            1,
            20,
            0.70,
            _metrics(border_touch=False, sharpness=10),
            _review(1, 0.5),
        ),
    ]
    dino_metrics = {
        slot: TemporalRepresentationMetrics(
            frame_slot=slot,
            dino_representativeness=1.0 if slot == 0 else 0.7,
            dino_nearest_similarity=0.9,
            dino_cluster_id=slot,
            dino_in_stable_cluster=slot == 1,
        )
        for slot in (0, 1)
    }

    ranked = rank_candidates(
        candidates,
        config=RankingConfig(
            dinov3_exclude_cluster_outliers=True,
            siglip2_enabled=False,
        ),
        dino_metrics_by_slot=dino_metrics,
        dino_outlier_slots={0},
    )

    assert ranked[0].frame_slot == 1
    assert "dino_temporal_outlier" in ranked[1].hard_rejection_reasons


def test_siglip_wrong_entity_cannot_be_rescued_by_sharpness() -> None:
    candidates = [
        (
            0,
            10,
            0.99,
            _metrics(border_touch=False, sharpness=1000),
            _review(0, 1.0),
        ),
        (
            1,
            20,
            0.70,
            _metrics(border_touch=False, sharpness=10),
            _review(1, 0.5),
        ),
    ]
    alignments = {
        0: AlignmentMetrics(0.1, -0.4, "distractor"),
        1: AlignmentMetrics(0.8, 0.4, "target"),
    }

    ranked = rank_candidates(
        candidates,
        config=RankingConfig(
            dinov3_enabled=False,
            siglip2_hard_reject_wrong_entity=True,
        ),
        alignment_metrics_by_slot=alignments,
        siglip_wrong_entity_slots={0},
    )

    assert ranked[0].frame_slot == 1
    assert "siglip_wrong_entity" in ranked[1].hard_rejection_reasons


def test_dino_and_siglip_filters_are_soft_by_default() -> None:
    candidate = (
        0,
        10,
        0.9,
        _metrics(border_touch=False, sharpness=10),
        _review(0),
    )
    alignment = AlignmentMetrics(0.1, -0.4, "distractor")
    config = RankingConfig(dinov3_enabled=True, siglip2_enabled=True)

    ranked = rank_candidates(
        [candidate],
        config=config,
        dino_outlier_slots={0},
        alignment_metrics_by_slot={0: alignment},
        siglip_wrong_entity_slots={0},
    )

    assert not config.dinov3_exclude_cluster_outliers
    assert not config.siglip2_hard_reject_wrong_entity
    assert "dino_temporal_outlier" not in ranked[0].hard_rejection_reasons
    assert "siglip_wrong_entity" not in ranked[0].hard_rejection_reasons


def test_disabled_model_weights_are_removed_and_renormalized() -> None:
    metrics = _metrics(border_touch=False, sharpness=10)
    review = _review(0)
    ranked = rank_candidates(
        [(0, 10, 0.8, metrics, review)],
        config=RankingConfig(dinov3_enabled=False, siglip2_enabled=False),
    )

    crop_score = 1.0 - abs(0.7 - 0.55) / 0.55
    expected = (
        0.18 * 0.9
        + 0.15 * 0.9
        + 0.08 * 0.9
        + 0.06 * 0.9
        + 0.06 * (0.6 * 0.5 + 0.4 * 0.9)
        + 0.05 * 1.0
        + 0.03 * crop_score
        + 0.02 * 0.8
    ) / 0.63
    assert ranked[0].ranking_score == pytest.approx(expected)


class _FakeDinoEmbedder:
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        assert len(images) == 3
        return np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )


class _FakeSiglipAligner:
    def __init__(self) -> None:
        self.target_text = ""
        self.distractor_texts: list[str] = []

    def score(
        self,
        images: list[Image.Image],
        target_text: str,
        distractor_texts: list[str],
    ) -> list[AlignmentMetrics]:
        self.target_text = target_text
        self.distractor_texts = distractor_texts
        return [
            AlignmentMetrics(
                0.8,
                0.2 if distractor_texts else None,
                target_text,
            )
            for _ in images
        ]


class _FakeCandidateJudge:
    def review(
        self,
        *,
        entity_id: str,
        entity_phrase: str,
        contact_sheet: Path,
        frame_slots: list[int],
    ) -> CandidateJudgeResult:
        del entity_phrase
        assert contact_sheet.is_file()
        return CandidateJudgeResult(
            entity_id=entity_id,
            candidates=[_review(slot) for slot in frame_slots],
            best_frame_slot=frame_slots[-1],
        )


def _write_ranking_fixture(output_root: Path) -> None:
    annotation = {
        "clip_uid": "clip_1",
        "video_path": str(output_root / "source.mp4"),
        "caption": "A gray-haired man holds a wine glass.",
        "prompt_with_refs": ("<ref_subject_1> A gray-haired man holds a wine glass."),
        "entities": [
            {
                "entity_id": "e1",
                "phrase": "A gray-haired man",
                "grounding_prompt": "gray-haired man wearing a dark suit",
                "canonical_label": "gray-haired man",
                "category": "person",
                "reference_worthy": True,
                "salience": "primary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "independent",
                "selection_reason": "primary subject",
                "ref_token": "<ref_subject_1>",
            },
            {
                "entity_id": "e2",
                "phrase": "a wine glass",
                "grounding_prompt": "small stemmed wine glass",
                "canonical_label": "wine glass",
                "category": "object",
                "reference_worthy": False,
                "salience": "secondary",
                "genericity": "generic",
                "name_evidence": "none",
                "separability": "attached_accessory",
                "selection_reason": "held accessory",
                "ref_token": None,
            },
        ],
        "relations": [
            {
                "subject_id": "e1",
                "predicate": "holding",
                "object_id": "e2",
            }
        ],
        "background": None,
    }
    manifest = output_root / "manifests" / "annotations.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(annotation) + "\n", encoding="utf-8")

    candidate_dir = output_root / "candidates" / "clip_1" / "e1"
    candidate_dir.mkdir(parents=True)
    frame_dir = output_root / "frames" / "clip_1"
    frame_dir.mkdir(parents=True)
    mask = np.zeros((240, 240), dtype=bool)
    mask[40:200, 55:185] = True
    records = []
    encoded_masks = {}
    for slot, confidence in enumerate((0.90, 0.95, 0.99)):
        frame = np.full((240, 240, 3), 80 + slot * 30, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"frame_{slot:02d}.jpg"), frame)
        key = f"mask_{slot}"
        records.append(
            {
                "frame_slot": slot,
                "source_frame_index": slot * 10,
                "sam_confidence": confidence,
                "mask_rle_key": key,
            }
        )
        encoded_masks[key] = encode_mask(mask)
    (candidate_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (candidate_dir / "top_masks.rle.json").write_text(
        json.dumps(encoded_masks),
        encoding="utf-8",
    )


def _distractor_entity(
    entity_id: str,
    *,
    canonical_label: str,
    grounding_prompt: str,
    reference_worthy: bool = True,
    salience: str = "secondary",
    separability: str = "independent",
) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "phrase": f"entity {entity_id}",
        "grounding_prompt": grounding_prompt,
        "canonical_label": canonical_label,
        "category": "object",
        "reference_worthy": reference_worthy,
        "salience": salience,
        "genericity": "descriptive",
        "name_evidence": "none",
        "separability": separability,
        "selection_reason": "test entity",
        "ref_token": None,
    }


def test_siglip_distractors_keep_only_distinct_independent_entities() -> None:
    annotation = AnnotationResult.model_validate(
        {
            "caption": "Test entities stand together.",
            "prompt_with_refs": "Test entities stand together.",
            "entities": [
                _distractor_entity(
                    "target",
                    canonical_label="gray-haired man",
                    grounding_prompt="gray-haired man wearing a dark suit",
                ),
                _distractor_entity(
                    "attached",
                    canonical_label="wine glass",
                    grounding_prompt="small stemmed wine glass",
                    separability="attached_accessory",
                ),
                _distractor_entity(
                    "same-label",
                    canonical_label="Gray-Haired Man",
                    grounding_prompt="man in another pose",
                ),
                _distractor_entity(
                    "same-prompt",
                    canonical_label="person",
                    grounding_prompt="  GRAY-HAIRED   MAN wearing a DARK suit ",
                ),
                _distractor_entity(
                    "incidental",
                    canonical_label="chair",
                    grounding_prompt="small chair in the distance",
                    reference_worthy=False,
                    salience="incidental",
                ),
                _distractor_entity(
                    "independent",
                    canonical_label="woman",
                    grounding_prompt="woman wearing a red coat",
                ),
                _distractor_entity(
                    "secondary-nonref",
                    canonical_label="car",
                    grounding_prompt="blue sedan",
                    reference_worthy=False,
                ),
            ],
            "relations": [],
            "background": None,
        }
    )

    distractors = siglip_distractor_entities(
        annotation,
        annotation.entities[0],
    )

    assert [entity.entity_id for entity in distractors] == [
        "independent",
        "secondary-nonref",
    ]


def test_stage_ranking_uses_grounding_prompt_and_saves_dino_embedding(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    _write_ranking_fixture(output_root)
    siglip = _FakeSiglipAligner()
    stats = rank_manifest_references(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
            qwen=QwenConfig(model="served-model-name"),
            ranking=RankingConfig(
                dinov3_enabled=True,
                dinov3_cluster_similarity_threshold=0.7,
                siglip2_enabled=True,
            ),
        ),
        judge=_FakeCandidateJudge(),
        dino_embedder=_FakeDinoEmbedder(),
        siglip_aligner=siglip,
    )

    reference_dir = output_root / "references" / "clip_1" / "e1"
    assert stats.processed == 1
    assert stats.candidate_count_before_models == 3
    assert stats.candidate_count_after_dino == 3
    assert stats.candidate_count_after_siglip == 3
    assert siglip.target_text == "gray-haired man wearing a dark suit"
    assert siglip.distractor_texts == []
    selected_embedding = np.load(
        reference_dir / "dinov3_embedding.npy",
        allow_pickle=False,
    )
    assert selected_embedding.dtype == np.float16
    metadata = json.loads(
        (
            output_root / "candidates" / "clip_1" / "e1" / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["candidates"][2]["hard_rejection_reasons"] == []
    assert not metadata["candidates"][2]["dino"]["dino_in_stable_cluster"]
    assert metadata["dino_medoid_slot"] == 0
    assert metadata["candidates"][0]["siglip"]["best_matching_entity_id"] == "e1"
    reference_metadata = json.loads(
        (reference_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert reference_metadata["dino_medoid_slot"] == 0
