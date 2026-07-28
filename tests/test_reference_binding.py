from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from r2v_data_v2.caption_validation import validate_annotation
from r2v_data_v2.reference_binding import (
    ReferenceBindingError,
    assign_reference_tokens,
    build_prompt_with_refs,
    rebuild_for_retained_entities,
    validate_final_reference_binding,
)
from r2v_data_v2.schemas import QwenAnnotationResult
from tests.test_caption_validation import _valid_payload


def _multi_entity_payload() -> dict[str, object]:
    payload = _valid_payload()
    payload["entities"] = [
        {
            "entity_id": "e1",
            "phrase": "A woman in a red raincoat",
            "grounding_prompt": "woman wearing a red raincoat",
            "canonical_label": "woman",
            "category": "person",
            "reference_worthy": True,
            "salience": "primary",
            "genericity": "descriptive",
            "name_evidence": "none",
            "separability": "independent",
            "selection_reason": "stable primary subject",
        },
        {
            "entity_id": "e2",
            "phrase": "a passing tram",
            "grounding_prompt": "passing tram",
            "canonical_label": "tram",
            "category": "vehicle",
            "reference_worthy": True,
            "salience": "secondary",
            "genericity": "descriptive",
            "name_evidence": "none",
            "separability": "independent",
            "selection_reason": "stable vehicle",
        },
        {
            "entity_id": "e3",
            "phrase": "glass storefronts",
            "grounding_prompt": "glass storefronts",
            "canonical_label": "storefronts",
            "category": "object",
            "reference_worthy": True,
            "salience": "secondary",
            "genericity": "descriptive",
            "name_evidence": "none",
            "separability": "composite_candidate",
            "selection_reason": "visual group",
        },
    ]
    return payload


def test_tokens_are_assigned_by_caption_order_and_kind() -> None:
    result = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_multi_entity_payload())
    )
    assert [entity.ref_token for entity in result.entities] == [
        "<ref_subject_1>",
        "<ref_object_1>",
        "<ref_group_1>",
    ]
    assert "A woman in a red raincoat <ref_subject_1> walks" in result.prompt_with_refs
    assert "a passing tram <ref_object_1>" in result.prompt_with_refs
    assert "glass storefronts <ref_group_1>" in result.prompt_with_refs


def test_same_token_kind_increments_in_caption_order() -> None:
    payload = _multi_entity_payload()
    second = payload["entities"][1]  # type: ignore[index]
    second["category"] = "person"  # type: ignore[index]
    result = assign_reference_tokens(QwenAnnotationResult.model_validate(payload))
    assert result.entities[0].ref_token == "<ref_subject_1>"
    assert result.entities[1].ref_token == "<ref_subject_2>"


@pytest.mark.parametrize(
    ("caption", "code"),
    [
        (
            (
                "A different subject crosses the plaza through cool reflected light "
                "while the camera tracks steadily beside the moving scene outside."
            ),
            "phrase_missing_from_caption",
        ),
        (
            (
                "A woman in a red raincoat walks slowly. A woman in a red raincoat "
                "turns toward the tram while the camera follows beside the wet plaza."
            ),
            "phrase_occurs_multiple_times",
        ),
    ],
)
def test_missing_or_repeated_phrase_fails(caption: str, code: str) -> None:
    payload = _valid_payload()
    payload["caption"] = caption
    with pytest.raises(ReferenceBindingError) as caught:
        assign_reference_tokens(QwenAnnotationResult.model_validate(payload))
    assert code in {issue.code for issue in caught.value.issues}


def test_overlapping_phrase_spans_fail() -> None:
    payload = _valid_payload()
    nested = deepcopy(payload["entities"][0])  # type: ignore[index]
    nested["entity_id"] = "e2"
    nested["phrase"] = "woman in a red raincoat"
    payload["entities"].append(nested)  # type: ignore[union-attr]
    annotation = QwenAnnotationResult.model_validate(payload)
    with pytest.raises(ReferenceBindingError) as caught:
        assign_reference_tokens(annotation)
    assert "overlapping_phrase_spans" in {
        issue.code
        for issue in validate_annotation(
            annotation,
            caption_raw="",
            metadata={},
        )
    }
    assert caught.value.issues


def test_removing_generated_tokens_restores_caption_exactly() -> None:
    result = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_valid_payload())
    )
    stripped = result.prompt_with_refs.replace(" <ref_subject_1>", "")
    assert stripped == result.caption


def test_filtered_entity_has_no_token() -> None:
    payload = _multi_entity_payload()
    payload["entities"][1]["reference_worthy"] = False  # type: ignore[index]
    result = assign_reference_tokens(QwenAnnotationResult.model_validate(payload))
    assert result.entities[1].ref_token is None
    assert "<ref_object_1>" not in result.prompt_with_refs


def test_background_is_deferred_without_token() -> None:
    payload = _valid_payload()
    payload["background"] = {
        "phrase": "wet stone plaza",
        "grounding_prompt": "empty wet stone plaza",
        "reference_worthy": True,
    }
    result = assign_reference_tokens(QwenAnnotationResult.model_validate(payload))
    assert result.background is not None
    assert not result.background.reference_worthy
    assert result.background.ref_token is None
    assert "<ref_bg_1>" not in result.prompt_with_refs


def test_prompt_builder_rejects_overlap_even_with_manual_tokens() -> None:
    result = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_valid_payload())
    )
    with pytest.raises(ReferenceBindingError):
        build_prompt_with_refs(
            result.caption,
            [
                result.entities[0],
                result.entities[0].model_copy(
                    update={
                        "entity_id": "e2",
                        "phrase": "woman in a red raincoat",
                        "ref_token": "<ref_subject_2>",
                    }
                ),
            ],
            None,
        )


def _final_sample(
    tmp_path: Path,
) -> tuple[dict[str, object], object]:
    annotation = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_valid_payload())
    )
    image = tmp_path / "reference.jpg"
    image.write_bytes(b"image")
    sample: dict[str, object] = {
        "prompt_with_refs": annotation.prompt_with_refs,
        "references": [
            {
                "entity_id": "e1",
                "phrase": annotation.entities[0].phrase,
                "ref_token": annotation.entities[0].ref_token,
                "image_path": str(image),
            }
        ],
        "background_reference": None,
    }
    return sample, annotation


def test_final_prompt_tokens_match_existing_reference(tmp_path: Path) -> None:
    sample, annotation = _final_sample(tmp_path)
    assert validate_final_reference_binding(sample, annotation) == []  # type: ignore[arg-type]


def test_final_binding_rejects_missing_or_zero_reference(tmp_path: Path) -> None:
    sample, annotation = _final_sample(tmp_path)
    sample["references"][0]["image_path"] = str(tmp_path / "missing.jpg")  # type: ignore[index]
    codes = {
        issue.code
        for issue in validate_final_reference_binding(
            sample,
            annotation,  # type: ignore[arg-type]
        )
    }
    assert "reference_image_missing" in codes

    sample["references"] = []
    codes = {
        issue.code
        for issue in validate_final_reference_binding(
            sample,
            annotation,  # type: ignore[arg-type]
        )
    }
    assert {"no_references", "final_prompt_token_mismatch"} <= codes


def test_filtered_entity_does_not_leave_dangling_token() -> None:
    annotation = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_multi_entity_payload())
    )
    retained = rebuild_for_retained_entities(annotation, {"e1"})
    assert retained.entities[0].ref_token == "<ref_subject_1>"
    assert all(entity.ref_token is None for entity in retained.entities[1:])
    assert "<ref_object_" not in retained.prompt_with_refs
    assert "<ref_group_" not in retained.prompt_with_refs
