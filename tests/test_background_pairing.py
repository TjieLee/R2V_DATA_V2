from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.config import PipelineConfig
from r2v_data_v2.pairing import build_cross_pair_index, build_pairs
from r2v_data_v2.reference_binding import assign_reference_tokens
from r2v_data_v2.schemas import QwenAnnotationResult


class _NeverCalledJudge:
    def judge(self, **kwargs: object) -> None:
        del kwargs
        raise AssertionError("background references must not reach cross-pair judge")


def _semantic_annotation(*, with_entity: bool) -> QwenAnnotationResult:
    entities: list[dict[str, object]] = []
    caption = "A red bicycle stands in a brick courtyard."
    if with_entity:
        entities.append(
            {
                "entity_id": "e1",
                "phrase": "A red bicycle",
                "grounding_prompt": "red bicycle",
                "canonical_label": "red bicycle",
                "category": "vehicle",
                "reference_worthy": True,
                "salience": "primary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "independent",
                "selection_reason": "primary subject",
            }
        )
    else:
        caption = "The scene shows a brick courtyard lit by afternoon sun."
    return QwenAnnotationResult.model_validate(
        {
            "caption": caption,
            "entities": entities,
            "relations": [],
            "background": {
                "phrase": "a brick courtyard",
                "grounding_prompt": "brick courtyard, afternoon light",
                "reference_worthy": False,
            },
        }
    )


def _write_image_and_mask(
    destination: Path,
    *,
    empty_mask: bool,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True)
    image_path = destination / "canonical.jpg"
    pixels = np.full((32, 48, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), pixels)
    mask_path = destination / "mask.png"
    mask = np.zeros((32, 48), dtype=np.uint8)
    if not empty_mask:
        mask[4:28, 8:40] = 255
    assert cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def _write_pair_fixture(
    tmp_path: Path,
    *,
    with_entity: bool,
    background_phrase: str = "a brick courtyard",
) -> tuple[PipelineConfig, Path]:
    output_root = tmp_path / "output"
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True)
    annotation = assign_reference_tokens(
        _semantic_annotation(with_entity=with_entity)
    )
    annotation_record = {
        **annotation.model_dump(mode="json"),
        "clip_uid": "clip-1",
        "video_path": "/read-only/video.mp4",
        "parent_video_id": "parent",
        "clip_suffix": "1_0",
        "warnings": ["background_reference_deferred"],
    }
    (manifests / "annotations.jsonl").write_text(
        json.dumps(annotation_record) + "\n",
        encoding="utf-8",
    )
    references: list[dict[str, object]] = []
    if with_entity:
        image, mask = _write_image_and_mask(
            output_root / "references" / "clip-1" / "e1",
            empty_mask=False,
        )
        entity = annotation.entities[0]
        references.append(
            {
                "clip_uid": "clip-1",
                "reference_id": "e1",
                "reference_type": "entity",
                "entity_id": "e1",
                "phrase": entity.phrase,
                "canonical_label": entity.canonical_label,
                "category": entity.category,
                "ref_token": entity.ref_token,
                "genericity": entity.genericity,
                "name_evidence": entity.name_evidence,
                "canonical_path": str(image),
                "mask_path": str(mask),
                "source_frame_index": 4,
                "ranking_score": 0.9,
            }
        )
    background_image, background_mask = _write_image_and_mask(
        output_root / "references" / "clip-1" / "bg1",
        empty_mask=True,
    )
    references.append(
        {
            "clip_uid": "clip-1",
            "reference_id": "bg1",
            "reference_type": "background",
            "entity_id": None,
            "phrase": background_phrase,
            "canonical_label": "background",
            "category": "background",
            "ref_token": "<ref_bg_1>",
            "canonical_path": str(background_image),
            "mask_path": str(background_mask),
            "source_frame_index": 5,
            "ranking_score": 0.8,
        }
    )
    (manifests / "references.jsonl").write_text(
        "".join(json.dumps(reference) + "\n" for reference in references),
        encoding="utf-8",
    )
    return (
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        manifests,
    )


def test_background_only_clip_produces_valid_sample(tmp_path: Path) -> None:
    config, manifests = _write_pair_fixture(tmp_path, with_entity=False)

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert stats.in_pair_count == 1
    assert sample["entities"] == []
    assert sample["references"][0]["reference_type"] == "background"
    assert sample["references"][0]["pair_type"] == "in_pair"
    assert "a brick courtyard <ref_bg_1>" in sample["prompt_with_refs"]


def test_entity_and_background_are_emitted_together(tmp_path: Path) -> None:
    config, manifests = _write_pair_fixture(tmp_path, with_entity=True)

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert {reference["reference_type"] for reference in sample["references"]} == {
        "entity",
        "background",
    }
    assert sample["background_reference"]["reference_id"] == "bg1"


def test_cross_pair_index_ignores_background() -> None:
    background = {
        "clip_uid": "clip-1",
        "parent_video_id": "parent",
        "clip_suffix": "1_0",
        "reference_type": "background",
        "category": "background",
        "canonical_label": "background",
    }

    assert build_cross_pair_index({"clip-1": [background]}) == {}


def test_unbindable_background_is_dropped_without_losing_entity(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=True,
        background_phrase="a beach at sunset",
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert [reference["reference_type"] for reference in sample["references"]] == [
        "entity"
    ]
    assert sample["background_reference"] is None
    assert any(
        warning.startswith("background_reference_dropped:")
        for warning in sample["warnings"]
    )
