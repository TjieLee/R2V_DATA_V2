from __future__ import annotations

from copy import deepcopy

from prompts.qwen_annotation_prompt import ICL_EXAMPLES
from r2v_data_v2.caption_validation import validate_annotation
from r2v_data_v2.schemas import QwenAnnotationResult


def _codes(result: QwenAnnotationResult) -> set[str]:
    return {issue.code for issue in validate_annotation(result)}


def _valid_payload() -> dict[str, object]:
    caption = (
        "A woman in a red raincoat walks across a wet stone plaza and raises her "
        "left hand toward a passing tram. The camera tracks beside her at waist "
        "height while cool daylight reflects from the pavement and glass storefronts."
    )
    return {
        "caption": caption,
        "entities": [
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
            }
        ],
        "relations": [],
        "background": None,
    }


def test_valid_semantic_annotation() -> None:
    result = QwenAnnotationResult.model_validate(_valid_payload())
    assert validate_annotation(result) == []


def test_reference_phrase_must_be_unique_in_caption() -> None:
    payload = _valid_payload()
    payload["caption"] = f"{payload['caption']} A woman in a red raincoat turns."
    assert "phrase_occurs_multiple_times" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_reference_phrase_does_not_match_inside_longer_word() -> None:
    payload = _valid_payload()
    payload["entities"][0]["phrase"] = "rain"  # type: ignore[index]
    assert "phrase_missing_from_caption" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_repeated_caption_sentence_is_rejected() -> None:
    payload = _valid_payload()
    sentence = (
        "A woman in a red raincoat walks across a wet stone plaza and waves slowly."
    )
    payload["caption"] = f"{sentence} {sentence}"
    assert "caption_repeated_sentence" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_caption_over_200_words_is_rejected() -> None:
    payload = _valid_payload()
    payload["caption"] = " ".join(["visible"] * 201)
    assert "caption_over_200_words" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_named_identity_requires_explicit_evidence() -> None:
    payload = deepcopy(_valid_payload())
    entity = payload["entities"][0]  # type: ignore[index]
    entity["genericity"] = "named"  # type: ignore[index]
    entity["name_evidence"] = "none"  # type: ignore[index]
    assert "named_identity_without_evidence" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_attached_accessory_cannot_keep_independent_ref_token() -> None:
    payload = deepcopy(_valid_payload())
    entity = payload["entities"][0]  # type: ignore[index]
    entity["separability"] = "attached_accessory"  # type: ignore[index]
    assert "attached_accessory_reference" in _codes(
        QwenAnnotationResult.model_validate(payload)
    )


def test_four_icl_examples_have_complete_valid_json() -> None:
    assert len(ICL_EXAMPLES) == 4
    for example in ICL_EXAMPLES:
        result = QwenAnnotationResult.model_validate(example["output"])
        assert validate_annotation(result) == []
