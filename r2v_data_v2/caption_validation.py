from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass

from r2v_data_v2.schemas import QwenAnnotationResult

_ENTITY_ID = re.compile(r"e[1-9]\d*")
_GENERIC_LABELS = {"man", "woman", "child", "person"}
_ALLOWED_IDENTITY_METADATA_FIELDS = {
    "title",
    "caption",
    "text",
    "person_name",
    "entity_names",
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str | None
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _issue(code: str, field: str | None, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, field=field, message=message)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


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


def exact_phrase_spans(caption: str, phrase: str) -> list[tuple[int, int]]:
    words = phrase.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(word) for word in words)
    if words[0][0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if words[-1][-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return [match.span() for match in re.finditer(pattern, caption)]


def exact_phrase_occurrence_count(caption: str, phrase: str) -> int:
    return len(exact_phrase_spans(caption, phrase))


def _metadata_text(metadata: dict[str, object]) -> str:
    values: list[str] = []
    for key in _ALLOWED_IDENTITY_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return " ".join(values)


def _contains_identity(evidence: str, phrase: str, label: str) -> bool:
    normalized = _normalize_whitespace(evidence).casefold()
    for candidate in (phrase, label):
        identity = _normalize_whitespace(candidate).casefold()
        if not identity:
            continue
        pattern = re.escape(identity)
        if identity[0].isalnum():
            pattern = rf"(?<!\w){pattern}"
        if identity[-1].isalnum():
            pattern = rf"{pattern}(?!\w)"
        if re.search(pattern, normalized):
            return True
    return False


def validate_annotation(
    result: QwenAnnotationResult,
    *,
    caption_raw: str,
    metadata: dict[str, object],
    visible_text: list[str] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    caption = result.caption.strip()
    if not caption:
        issues.append(_issue("caption_empty", "caption", "caption must not be empty"))
    if "\n" in caption:
        issues.append(
            _issue(
                "caption_multiple_paragraphs",
                "caption",
                "caption must be one paragraph",
            )
        )
    word_count = len(caption.split())
    if word_count < 20:
        issues.append(
            _issue(
                "caption_too_short",
                "caption",
                "caption must contain at least 20 English words",
            )
        )
    if word_count > 200:
        issues.append(
            _issue(
                "caption_over_200_words",
                "caption",
                "caption exceeds the 200-word hard limit",
            )
        )

    sentences = _sentences(caption)
    normalized_sentences = [_normalize_sentence(sentence) for sentence in sentences]
    if len(normalized_sentences) != len(set(normalized_sentences)):
        issues.append(
            _issue(
                "caption_repeated_sentence",
                "caption",
                "caption contains an exact repeated sentence",
            )
        )
    for left_index, left in enumerate(sentences):
        for right_index in range(left_index + 1, len(sentences)):
            right = sentences[right_index]
            threshold = 0.85 if right_index == left_index + 1 else 0.90
            if _jaccard(_trigrams(left), _trigrams(right)) > threshold:
                issues.append(
                    _issue(
                        "caption_high_similarity",
                        "caption",
                        "caption contains highly similar sentences",
                    )
                )
                break
        if any(issue.code == "caption_high_similarity" for issue in issues):
            break

    entity_ids = [entity.entity_id for entity in result.entities]
    if len(entity_ids) != len(set(entity_ids)):
        issues.append(
            _issue(
                "duplicate_entity_id",
                "entities",
                "entity_id values must be unique",
            )
        )
    for index, entity in enumerate(result.entities):
        if _ENTITY_ID.fullmatch(entity.entity_id) is None:
            issues.append(
                _issue(
                    "invalid_entity_id",
                    f"entities[{index}].entity_id",
                    "entity_id must match e followed by a positive integer",
                )
            )
    phrases = [
        _normalize_whitespace(entity.phrase).casefold() for entity in result.entities
    ]
    if len(phrases) != len(set(phrases)):
        issues.append(
            _issue(
                "duplicate_phrase",
                "entities",
                "entity phrases must be unique",
            )
        )

    selected_spans: list[tuple[int, int, str]] = []
    for index, entity in enumerate(result.entities):
        if not entity.reference_worthy:
            continue
        field = f"entities[{index}].phrase"
        spans = exact_phrase_spans(caption, entity.phrase)
        if not spans:
            issues.append(
                _issue(
                    "phrase_missing_from_caption",
                    field,
                    "reference-worthy phrase must occur in caption",
                )
            )
        elif len(spans) > 1:
            issues.append(
                _issue(
                    "phrase_occurs_multiple_times",
                    field,
                    "reference-worthy phrase must occur exactly once",
                )
            )
        else:
            selected_spans.append((*spans[0], field))
    selected_spans.sort()
    for left, right in zip(selected_spans, selected_spans[1:]):
        if right[0] < left[1]:
            issues.append(
                _issue(
                    "overlapping_phrase_spans",
                    right[2],
                    "reference-worthy phrase spans must not overlap",
                )
            )

    for index, entity in enumerate(result.entities):
        field = f"entities[{index}]"
        if entity.genericity == "named":
            evidence_valid = False
            if entity.name_evidence == "draft_caption":
                evidence_valid = _contains_identity(
                    caption_raw,
                    entity.phrase,
                    entity.canonical_label,
                )
            elif entity.name_evidence == "metadata":
                evidence_valid = _contains_identity(
                    _metadata_text(metadata),
                    entity.phrase,
                    entity.canonical_label,
                )
            if not evidence_valid:
                issues.append(
                    _issue(
                        "named_identity_without_evidence",
                        f"{field}.name_evidence",
                        "named identity is not supported by the declared input evidence",
                    )
                )
        if entity.separability == "attached_accessory" and entity.reference_worthy:
            issues.append(
                _issue(
                    "attached_accessory_reference",
                    f"{field}.reference_worthy",
                    "attached accessory must not be reference-worthy",
                )
            )
    if result.background:
        background = result.background
        spans = exact_phrase_spans(caption, background.phrase)
        if background.reference_worthy and not spans:
            issues.append(
                _issue(
                    "background_phrase_missing",
                    "background.phrase",
                    "reference-worthy background phrase must occur in caption",
                )
            )
        elif background.reference_worthy and len(spans) > 1:
            issues.append(
                _issue(
                    "phrase_occurs_multiple_times",
                    "background.phrase",
                    "reference-worthy background phrase must occur exactly once",
                )
            )
    known_ids = set(entity_ids)
    for index, relation in enumerate(result.relations):
        if relation.subject_id not in known_ids or relation.object_id not in known_ids:
            issues.append(
                _issue(
                    "invalid_relation_entity",
                    f"relations[{index}]",
                    "relation refers to an unknown entity_id",
                )
            )

    non_background_references = sum(
        entity.reference_worthy for entity in result.entities
    )
    if non_background_references > 3:
        issues.append(
            _issue(
                "too_many_references",
                "entities",
                "at most three non-background references are allowed",
            )
        )
    return issues


def annotation_warnings(result: QwenAnnotationResult) -> list[str]:
    warnings: list[str] = []
    word_count = len(result.caption.split())
    if not 40 <= word_count <= 180:
        warnings.append("caption is outside the recommended 40-180 word range")
    generic = sum(
        entity.canonical_label.casefold() in _GENERIC_LABELS
        for entity in result.entities
    )
    if generic:
        warnings.append(f"generic_entity_label_count={generic}")
    return warnings
