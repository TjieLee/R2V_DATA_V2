from __future__ import annotations

from copy import deepcopy

import pytest

from prompts.qwen_annotation_prompt import ICL_EXAMPLES
from r2v_data_v2.caption_validation import validate_annotation
from r2v_data_v2.phrase_alignment import (
    resolve_background_caption_phrase,
    resolve_reference_caption_phrase,
)
from r2v_data_v2.schemas import QwenAnnotationResult


def _codes(
    result: QwenAnnotationResult,
    *,
    caption_raw: str = "",
    metadata: dict[str, object] | None = None,
    visible_text: list[str] | None = None,
) -> set[str]:
    return {
        issue.code
        for issue in validate_annotation(
            result,
            caption_raw=caption_raw,
            metadata=metadata or {},
            visible_text=visible_text,
        )
    }


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
    assert validate_annotation(result, caption_raw="", metadata={}) == []


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


def test_background_phrase_resolution_returns_exact_caption_text() -> None:
    caption = "A diver moves above rocky underwater terrain."
    assert (
        resolve_background_caption_phrase(caption, "rocky underwater terrain")
        == "rocky underwater terrain"
    )
    assert (
        resolve_background_caption_phrase(caption, "ROCKY   UNDERWATER TERRAIN")
        == "rocky underwater terrain"
    )
    assert (
        resolve_background_caption_phrase(caption, "the rocky underwater terrain")
        == "rocky underwater terrain"
    )


def test_phrase_resolution_preserves_conservative_final_word_repair() -> None:
    caption = "An enemy advances through the ruins."
    assert (
        resolve_background_caption_phrase(caption, "an enemy soldier")
        == "An enemy"
    )
    assert (
        resolve_reference_caption_phrase(caption, "an enemy soldier")
        == "An enemy"
    )


def test_background_phrase_resolution_rejects_multiple_occurrences() -> None:
    caption = "rocky terrain borders another stretch of rocky terrain."
    assert resolve_background_caption_phrase(caption, "rocky terrain") is None


def test_exact_background_phrase_takes_precedence_over_casefold_ambiguity() -> None:
    caption = "Ocean borders another ocean."
    assert resolve_background_caption_phrase(caption, "Ocean") == "Ocean"


@pytest.mark.parametrize(
    ("source_phrase", "caption", "expected"),
    [
        (
            "a vast expanse of deep blue ocean bordered by barren, rocky hills",
            (
                "The camera reveals a deep blue ocean shimmering under bright "
                "sunlight, bordered by arid, rocky hills."
            ),
            "deep blue ocean",
        ),
        (
            "a vast green grassland leading up to a towering mountain",
            (
                "The camera pans across a vast green grassland, revealing a "
                "towering mountain."
            ),
            "a vast green grassland",
        ),
        (
            "deep blue water with sunlight filtering through",
            (
                "A whale glides through deep blue water as sunlight filters "
                "from above."
            ),
            "deep blue water",
        ),
        (
            "a narrow underwater canyon with rugged coral walls",
            (
                "A diver moves through a narrow underwater canyon, flanked by "
                "rugged coral walls."
            ),
            "a narrow underwater canyon",
        ),
    ],
)
def test_background_phrase_recovers_unique_longest_shared_span(
    source_phrase: str,
    caption: str,
    expected: str,
) -> None:
    assert resolve_background_caption_phrase(caption, source_phrase) == expected


@pytest.mark.parametrize(
    ("source_phrase", "caption"),
    [
        (
            "a sandy beach lined with tall swaying palm trees",
            "Gentle waves reach a sandy shore as tall palm trees sway.",
        ),
        (
            "rough stone walls and a ceiling",
            "Mechanical ports are set into a rough stone wall.",
        ),
        (
            "a vast expanse of the ocean",
            "Ocean waves roll toward dark rocks.",
        ),
    ],
)
def test_background_phrase_longest_shared_span_fails_closed(
    source_phrase: str,
    caption: str,
) -> None:
    assert resolve_background_caption_phrase(caption, source_phrase) is None


def test_background_phrase_longest_shared_span_rejects_longest_tie() -> None:
    source_phrase = "deep blue ocean beside rugged coral walls"
    caption = "A deep blue ocean lies beyond rugged coral walls."
    assert resolve_background_caption_phrase(caption, source_phrase) is None


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
        source = example["input"]
        assert isinstance(source, dict)
        if result.background is not None:
            assert not result.background.reference_worthy
        assert (
            validate_annotation(
                result,
                caption_raw=str(source["draft_caption"]),
                metadata=source["metadata"],  # type: ignore[arg-type]
            )
            == []
        )


def _named_payload() -> dict[str, object]:
    payload = _valid_payload()
    payload["caption"] = str(payload["caption"]).replace(
        "A woman in a red raincoat",
        "Serena Williams",
    )
    entity = payload["entities"][0]  # type: ignore[index]
    entity["phrase"] = "Serena Williams"  # type: ignore[index]
    entity["canonical_label"] = "Serena Williams"  # type: ignore[index]
    entity["genericity"] = "named"  # type: ignore[index]
    return payload


def test_named_identity_present_in_draft_caption_passes() -> None:
    payload = _named_payload()
    payload["entities"][0]["name_evidence"] = "draft_caption"  # type: ignore[index]
    result = QwenAnnotationResult.model_validate(payload)
    assert "named_identity_without_evidence" not in _codes(
        result,
        caption_raw="Serena Williams walks across a plaza.",
    )


def test_named_identity_cannot_lie_about_draft_caption_evidence() -> None:
    payload = _named_payload()
    payload["entities"][0]["name_evidence"] = "draft_caption"  # type: ignore[index]
    result = QwenAnnotationResult.model_validate(payload)
    assert "named_identity_without_evidence" in _codes(
        result,
        caption_raw="A tennis player walks across a plaza.",
    )


def test_named_identity_in_allowed_metadata_field_passes() -> None:
    payload = _named_payload()
    payload["entities"][0]["name_evidence"] = "metadata"  # type: ignore[index]
    result = QwenAnnotationResult.model_validate(payload)
    assert "named_identity_without_evidence" not in _codes(
        result,
        metadata={"person_name": "Serena Williams"},
    )


def test_named_identity_ignores_unapproved_metadata_fields() -> None:
    payload = _named_payload()
    payload["entities"][0]["name_evidence"] = "metadata"  # type: ignore[index]
    result = QwenAnnotationResult.model_validate(payload)
    assert "named_identity_without_evidence" in _codes(
        result,
        metadata={"identity_source": "Serena Williams"},
    )


def test_visible_text_evidence_is_rejected_until_ocr_exists() -> None:
    payload = _named_payload()
    payload["entities"][0]["name_evidence"] = "visible_text"  # type: ignore[index]
    result = QwenAnnotationResult.model_validate(payload)
    assert "named_identity_without_evidence" in _codes(
        result,
        visible_text=["Serena Williams"],
    )
