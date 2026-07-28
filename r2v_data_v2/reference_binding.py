from __future__ import annotations

import re
from collections import defaultdict

from r2v_data_v2.caption_validation import (
    ValidationIssue,
    exact_phrase_spans,
)
from r2v_data_v2.schemas import (
    AnnotationEntity,
    AnnotationResult,
    BackgroundAnnotation,
    QwenAnnotationResult,
)

_REF_TOKEN_WITH_PREFIX = re.compile(r" <ref_(?:subject|object|bg|group)_\d+>")


class ReferenceBindingError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        super().__init__("reference token assignment or prompt construction failed")
        self.issues = issues


def _token_kind(entity: AnnotationEntity) -> str:
    if entity.separability == "composite_candidate":
        return "group"
    if entity.category in {"person", "animal", "character"}:
        return "subject"
    return "object"


def build_prompt_with_refs(
    caption: str,
    entities: list[AnnotationEntity],
    background: BackgroundAnnotation | None,
) -> str:
    insertions: list[tuple[int, int, str, str]] = []
    issues: list[ValidationIssue] = []
    for index, entity in enumerate(entities):
        if entity.ref_token is None:
            continue
        spans = exact_phrase_spans(caption, entity.phrase)
        field = f"entities[{index}].phrase"
        if not spans:
            issues.append(
                ValidationIssue(
                    "phrase_missing_from_caption",
                    field,
                    "reference phrase must occur in caption",
                )
            )
        elif len(spans) > 1:
            issues.append(
                ValidationIssue(
                    "phrase_occurs_multiple_times",
                    field,
                    "reference phrase must occur exactly once",
                )
            )
        else:
            insertions.append((*spans[0], entity.ref_token, field))
    if background and background.ref_token:
        spans = exact_phrase_spans(caption, background.phrase)
        if not spans:
            issues.append(
                ValidationIssue(
                    "background_phrase_missing",
                    "background.phrase",
                    "reference background phrase must occur in caption",
                )
            )
        elif len(spans) > 1:
            issues.append(
                ValidationIssue(
                    "phrase_occurs_multiple_times",
                    "background.phrase",
                    "reference background phrase must occur exactly once",
                )
            )
        else:
            insertions.append((*spans[0], background.ref_token, "background.phrase"))
    insertions.sort()
    for left, right in zip(insertions, insertions[1:]):
        if right[0] < left[1]:
            issues.append(
                ValidationIssue(
                    "overlapping_phrase_spans",
                    right[3],
                    "reference phrase spans must not overlap",
                )
            )
    if issues:
        raise ReferenceBindingError(issues)

    prompt = caption
    for _, end, token, _ in reversed(insertions):
        prompt = f"{prompt[:end]} {token}{prompt[end:]}"
    if _REF_TOKEN_WITH_PREFIX.sub("", prompt) != caption:
        raise ReferenceBindingError(
            [
                ValidationIssue(
                    "prompt_caption_mismatch",
                    "prompt_with_refs",
                    "removing generated tokens must restore the caption exactly",
                )
            ]
        )
    return prompt


def assign_reference_tokens(annotation: QwenAnnotationResult) -> AnnotationResult:
    selected: list[tuple[int, int]] = []
    issues: list[ValidationIssue] = []
    for index, entity in enumerate(annotation.entities):
        if not entity.reference_worthy:
            continue
        spans = exact_phrase_spans(annotation.caption, entity.phrase)
        field = f"entities[{index}].phrase"
        if len(spans) != 1:
            code = (
                "phrase_missing_from_caption"
                if not spans
                else "phrase_occurs_multiple_times"
            )
            issues.append(
                ValidationIssue(
                    code,
                    field,
                    "reference phrase must occur exactly once in caption",
                )
            )
            continue
        selected.append((spans[0][0], index))
    if len(selected) > 3:
        issues.append(
            ValidationIssue(
                "too_many_references",
                "entities",
                "at most three non-background references are allowed",
            )
        )
    if issues:
        raise ReferenceBindingError(issues)

    counters: dict[str, int] = defaultdict(int)
    tokens: dict[int, str] = {}
    for _, index in sorted(selected):
        semantic_entity = annotation.entities[index]
        entity = AnnotationEntity.model_validate(
            {**semantic_entity.model_dump(mode="json"), "ref_token": None}
        )
        kind = _token_kind(entity)
        counters[kind] += 1
        tokens[index] = f"<ref_{kind}_{counters[kind]}>"

    entities = [
        AnnotationEntity.model_validate(
            {
                **entity.model_dump(mode="json"),
                "ref_token": tokens.get(index),
            }
        )
        for index, entity in enumerate(annotation.entities)
    ]
    background = None
    if annotation.background is not None:
        background = BackgroundAnnotation.model_validate(
            {
                **annotation.background.model_dump(mode="json"),
                "reference_worthy": False,
                "ref_token": None,
            }
        )
    prompt = build_prompt_with_refs(annotation.caption, entities, background)
    return AnnotationResult(
        caption=annotation.caption,
        prompt_with_refs=prompt,
        entities=entities,
        relations=annotation.relations,
        background=background,
    )
