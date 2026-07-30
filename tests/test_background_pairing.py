from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import run_pipeline as pipeline_runner
from r2v_data_v2.config import PairingConfig, PipelineConfig
from r2v_data_v2.pairing import (
    PairingStats,
    build_cross_pair_index,
    build_pairs,
)
from r2v_data_v2.reference_binding import assign_reference_tokens
from r2v_data_v2.schemas import QwenAnnotationResult


class _NeverCalledJudge:
    def judge(self, **kwargs: object) -> None:
        del kwargs
        raise AssertionError("background references must not reach cross-pair judge")


def _assert_token_set_invariant(sample: dict[str, object]) -> None:
    prompt_tokens = re.findall(
        r"<ref_(?:subject|object|group|bg)_\d+>",
        str(sample["prompt_with_refs"]),
    )
    references = sample["references"]
    assert isinstance(references, list)
    reference_tokens = [
        str(reference["ref_token"])
        for reference in references
        if isinstance(reference, dict)
    ]
    assert sorted(prompt_tokens) == sorted(reference_tokens)


def _semantic_annotation(
    *,
    with_entity: bool,
    caption: str | None = None,
) -> QwenAnnotationResult:
    entities: list[dict[str, object]] = []
    if caption is None:
        caption = (
            "A red bicycle stands in a brick courtyard."
            if with_entity
            else "The scene shows a brick courtyard lit by afternoon sun."
        )
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
    with_background: bool = True,
    require_entity_reference: bool = True,
    background_phrase: str = "a brick courtyard",
    caption: str | None = None,
) -> tuple[PipelineConfig, Path]:
    output_root = tmp_path / "output"
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True)
    annotation = assign_reference_tokens(
        _semantic_annotation(
            with_entity=with_entity,
            caption=caption,
        )
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
    if with_background:
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
            pairing=PairingConfig(
                require_entity_reference=require_entity_reference
            ),
        ),
        manifests,
    )


def test_background_only_clip_is_skipped_when_entity_reference_required(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(tmp_path, with_entity=False)

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    assert stats.processed == 0
    assert stats.failed == 0
    assert stats.skipped_background_only_reference == 1
    assert (manifests / "final_samples.jsonl").read_text(encoding="utf-8") == ""
    assert not (config.output_root / "samples" / "clip-1.json").exists()
    assert not (config.output_root / "logs" / "pairing_failed.jsonl").exists()
    diagnostic = json.loads(
        (
            config.output_root
            / "logs"
            / "background_only_sample_skipped.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic == {
        "clip_uid": "clip-1",
        "reference_ids": ["bg1"],
        "reason": "final_sample_requires_entity_reference",
    }
    references = [
        json.loads(line)
        for line in (manifests / "references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [reference["reference_id"] for reference in references] == ["bg1"]
    assert (
        config.output_root / "references" / "clip-1" / "bg1" / "canonical.jpg"
    ).is_file()


def test_background_only_clip_is_emitted_when_entity_requirement_disabled(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=False,
        require_entity_reference=False,
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert stats.in_pair_count == 1
    assert stats.skipped_background_only_reference == 0
    assert sample["entities"] == []
    assert sample["references"][0]["reference_type"] == "background"
    assert sample["references"][0]["pair_type"] == "in_pair"
    assert "a brick courtyard <ref_bg_1>" in sample["prompt_with_refs"]
    _assert_token_set_invariant(sample)


def test_entity_only_clip_produces_valid_sample(tmp_path: Path) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=True,
        with_background=False,
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert stats.skipped_background_only_reference == 0
    assert [reference["reference_type"] for reference in sample["references"]] == [
        "entity"
    ]
    assert sample["background_reference"] is None
    _assert_token_set_invariant(sample)


def test_background_phrase_is_resolved_before_binding(tmp_path: Path) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=False,
        require_entity_reference=False,
        background_phrase="THE   BRICK COURTYARD",
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    background = sample["references"][0]
    assert stats.processed == 1
    assert background["phrase"] == "brick courtyard"
    assert background["source_phrase"] == "THE   BRICK COURTYARD"
    assert "brick courtyard <ref_bg_1>" in sample["prompt_with_refs"]
    _assert_token_set_invariant(sample)


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
    _assert_token_set_invariant(sample)


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
    assert stats.failed == 0
    assert stats.skipped_no_bindable_reference == 0
    assert [reference["reference_type"] for reference in sample["references"]] == [
        "entity"
    ]
    assert sample["background_reference"] is None
    assert (
        "background_reference_dropped:background_phrase_unresolvable"
        in sample["warnings"]
    )
    assert "<ref_bg_1>" not in sample["prompt_with_refs"]
    diagnostics = [
        json.loads(line)
        for line in (
            config.output_root / "logs" / "background_binding_failed.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert diagnostics == [
        {
            "original_background_phrase": "a beach at sunset",
            "issue_codes": ["background_phrase_unresolvable"],
            "clip_uid": "clip-1",
            "caption": sample["caption"],
            "entity_reference_count": 1,
        }
    ]
    _assert_token_set_invariant(sample)


def test_unbindable_background_only_clip_is_skipped_without_failure(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=False,
        background_phrase="a beach at sunset",
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    assert stats.processed == 0
    assert stats.failed == 0
    assert stats.skipped_no_ready_reference == 0
    assert stats.skipped_no_bindable_reference == 1
    assert (manifests / "final_samples.jsonl").read_text(encoding="utf-8") == ""
    assert not (config.output_root / "samples" / "clip-1.json").exists()
    assert not (config.output_root / "logs" / "pairing_failed.jsonl").exists()
    diagnostic = json.loads(
        (
            config.output_root / "logs" / "background_binding_failed.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic == {
        "original_background_phrase": "a beach at sunset",
        "issue_codes": ["background_phrase_unresolvable"],
        "clip_uid": "clip-1",
        "caption": "The scene shows a brick courtyard lit by afternoon sun.",
        "entity_reference_count": 0,
    }


def test_repeated_background_phrase_is_not_arbitrarily_bound(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=False,
        require_entity_reference=False,
        background_phrase="brick courtyard",
        caption=(
            "A brick courtyard opens onto another brick courtyard in the distance."
        ),
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    assert stats.processed == 0
    assert stats.failed == 0
    assert stats.skipped_no_bindable_reference == 1
    assert (manifests / "final_samples.jsonl").read_text(encoding="utf-8") == ""
    diagnostic = json.loads(
        (
            config.output_root / "logs" / "background_binding_failed.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic["issue_codes"] == ["background_phrase_unresolvable"]


def test_reference_binding_error_uses_standard_background_failure_schema(
    tmp_path: Path,
) -> None:
    config, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=True,
        background_phrase="red bicycle",
    )

    stats = build_pairs(
        config,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    sample = json.loads(
        (manifests / "final_samples.jsonl").read_text(encoding="utf-8")
    )
    assert stats.processed == 1
    assert stats.failed == 0
    assert [reference["reference_type"] for reference in sample["references"]] == [
        "entity"
    ]
    assert (
        "background_reference_dropped:overlapping_phrase_spans"
        in sample["warnings"]
    )
    diagnostic = json.loads(
        (
            config.output_root / "logs" / "background_binding_failed.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic == {
        "original_background_phrase": "red bicycle",
        "issue_codes": ["overlapping_phrase_spans"],
        "clip_uid": "clip-1",
        "caption": sample["caption"],
        "entity_reference_count": 1,
    }
    _assert_token_set_invariant(sample)


def test_enabling_entity_requirement_removes_existing_background_only_sample(
    tmp_path: Path,
) -> None:
    permissive, manifests = _write_pair_fixture(
        tmp_path,
        with_entity=False,
        require_entity_reference=False,
    )
    first_stats = build_pairs(
        permissive,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )
    assert first_stats.processed == 1

    strict = PipelineConfig(
        dataset_json=permissive.dataset_json,
        output_root=permissive.output_root,
        pairing=PairingConfig(require_entity_reference=True),
    )
    strict_stats = build_pairs(
        strict,
        judge=_NeverCalledJudge(),  # type: ignore[arg-type]
    )

    assert strict_stats.processed == 0
    assert strict_stats.skipped_background_only_reference == 1
    assert not (strict.output_root / "samples" / "clip-1.json").exists()
    assert (manifests / "final_samples.jsonl").read_text(encoding="utf-8") == ""


def test_pipeline_summary_includes_background_only_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
    )
    monkeypatch.setattr(pipeline_runner, "load_config", lambda _: config)
    monkeypatch.setattr(
        pipeline_runner,
        "build_pairs",
        lambda _config, *, overwrite: PairingStats(
            skipped_background_only_reference=3
        ),
    )

    result = pipeline_runner.run_pipeline(
        config_path="unused.yaml",
        stages=("pair",),
    )

    assert result["pair"]["skipped_background_only_reference"] == 3
    assert result["summary"]["skipped_background_only_reference"] == 3
