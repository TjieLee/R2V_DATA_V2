from __future__ import annotations

import re
import string

from r2v_data_v2.schemas import AnnotationResult

_REF_TOKEN = re.compile(r"<ref_(?:subject|object|bg|group)_\d+>")
_GENERIC_LABELS = {"man", "woman", "child", "person"}


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _normalize_binding_text(value: str) -> str:
    return re.sub(r"\s+([,.;:!?])", r"\1", _normalize_whitespace(value))


def _normalize_sentence(value: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return _normalize_whitespace(value.lower().translate(table))


def _sentences(caption: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", caption.strip())
        if sentence.strip()
    ]


def _trigrams(value: str) -> set[tuple[str, str, str]]:
    words = _normalize_sentence(value).split()
    return set(zip(words, words[1:], words[2:]))


def _jaccard(left: set[object], right: set[object]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def validate_annotation(result: AnnotationResult) -> list[str]:
    errors: list[str] = []
    caption = result.caption.strip()
    if "\n" in caption:
        errors.append("caption must be one paragraph")
    word_count = len(caption.split())
    if word_count < 20:
        errors.append("caption must contain at least 20 English words")
    if word_count > 200:
        errors.append("caption exceeds the 200-word hard limit")

    sentences = _sentences(caption)
    normalized_sentences = [_normalize_sentence(sentence) for sentence in sentences]
    if len(normalized_sentences) != len(set(normalized_sentences)):
        errors.append("caption contains an exact repeated sentence")
    for index in range(len(sentences) - 1):
        left, right = sentences[index : index + 2]
        if _jaccard(_trigrams(left), _trigrams(right)) > 0.85:
            errors.append("caption contains highly similar adjacent sentences")
            break

    entity_ids = [entity.entity_id for entity in result.entities]
    if len(entity_ids) != len(set(entity_ids)):
        errors.append("entity_id values must be unique")
    phrases = [
        _normalize_whitespace(entity.phrase).lower() for entity in result.entities
    ]
    if len(phrases) != len(set(phrases)):
        errors.append("entity phrases must be unique")

    expected_tokens = [
        entity.ref_token for entity in result.entities if entity.ref_token is not None
    ]
    if result.background and result.background.ref_token:
        expected_tokens.append(result.background.ref_token)
    if len(expected_tokens) != len(set(expected_tokens)):
        errors.append("ref_token values must be unique")
    prompt_tokens = _REF_TOKEN.findall(result.prompt_with_refs)
    if len(prompt_tokens) != len(set(prompt_tokens)):
        errors.append("each ref token must occur exactly once")
    if sorted(prompt_tokens) != sorted(expected_tokens):
        errors.append("prompt ref tokens do not match selected entities")

    prompt_without_tokens = _REF_TOKEN.sub("", result.prompt_with_refs)
    if _normalize_binding_text(prompt_without_tokens) != _normalize_binding_text(
        caption
    ):
        errors.append("prompt_with_refs must equal caption after removing ref tokens")

    for entity in result.entities:
        if entity.reference_worthy != (entity.ref_token is not None):
            errors.append(f"{entity.entity_id} reference_worthy and ref_token disagree")
        if entity.ref_token:
            binding = f"{entity.phrase} {entity.ref_token}"
            if result.prompt_with_refs.count(binding) != 1:
                errors.append(
                    f"{entity.entity_id} token must immediately follow its phrase"
                )
        if entity.genericity == "named" and entity.name_evidence == "none":
            errors.append(f"{entity.entity_id} named identity has no explicit evidence")
        if entity.separability == "attached_accessory" and entity.reference_worthy:
            errors.append(
                f"{entity.entity_id} attached accessory must not have a ref token"
            )
        if (
            entity.separability == "composite_candidate"
            and entity.reference_worthy
            and (entity.ref_token or "").startswith("<ref_group_") is False
        ):
            errors.append(f"{entity.entity_id} composite reference must use ref_group")

    if result.background:
        background = result.background
        if background.reference_worthy != (background.ref_token is not None):
            errors.append("background reference_worthy and ref_token disagree")
        if background.ref_token:
            binding = f"{background.phrase} {background.ref_token}"
            if result.prompt_with_refs.count(binding) != 1:
                errors.append("background token must immediately follow its phrase")

    known_ids = set(entity_ids)
    for relation in result.relations:
        if relation.subject_id not in known_ids or relation.object_id not in known_ids:
            errors.append("relation refers to an unknown entity_id")

    non_background_references = sum(
        entity.reference_worthy for entity in result.entities
    )
    if non_background_references > 3:
        errors.append("at most three non-background references are allowed")
    return errors


def annotation_warnings(result: AnnotationResult) -> list[str]:
    warnings: list[str] = []
    word_count = len(result.caption.split())
    if not 40 <= word_count <= 180:
        warnings.append("caption is outside the recommended 40-180 word range")
    generic = sum(
        entity.canonical_label.lower() in _GENERIC_LABELS for entity in result.entities
    )
    if generic:
        warnings.append(f"generic_entity_label_count={generic}")
    return warnings
