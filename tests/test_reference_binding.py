from __future__ import annotations

from copy import deepcopy

import pytest

from r2v_data_v2.caption_validation import validate_annotation
from r2v_data_v2.reference_binding import (
    ReferenceBindingError,
    assign_reference_tokens,
    build_prompt_with_refs,
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
        issue.code for issue in validate_annotation(annotation)
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
