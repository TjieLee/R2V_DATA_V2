from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.caption_validation import exact_phrase_spans
from r2v_data_v2.schemas import (
    AnnotationEntity,
    AnnotationResult,
    BackgroundAnnotation,
    QwenAnnotationEntity,
    QwenAnnotationResult,
)
from r2v_data_v2.structured_output import ValidationIssue

_REF_TOKEN = re.compile(r"<ref_(?:subject|object|bg|group)_\d+>")
_ANY_REF_TOKEN = re.compile(r"<ref_[^>]+>")
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
    for index in range(len(insertions) - 1):
        left, right = insertions[index : index + 2]
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


def rebuild_for_retained_entities(
    annotation: AnnotationResult,
    retained_entity_ids: set[str],
) -> AnnotationResult:
    known_ids = {entity.entity_id for entity in annotation.entities}
    unknown = retained_entity_ids - known_ids
    if unknown:
        raise ReferenceBindingError(
            [
                ValidationIssue(
                    "unknown_retained_entity",
                    "references",
                    f"retained references contain unknown entity IDs: {sorted(unknown)}",
                )
            ]
        )
    semantic_entities = [
        QwenAnnotationEntity.model_validate(
            {
                **entity.model_dump(mode="json", exclude={"ref_token"}),
                "reference_worthy": entity.entity_id in retained_entity_ids,
            }
        )
        for entity in annotation.entities
    ]
    semantic = QwenAnnotationResult(
        caption=annotation.caption,
        entities=semantic_entities,
        relations=annotation.relations,
        background=(
            annotation.background.model_dump(mode="json", exclude={"ref_token"})
            if annotation.background
            else None
        ),
    )
    return assign_reference_tokens(semantic)


def validate_final_reference_binding(
    sample: dict[str, object],
    annotation: AnnotationResult | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    prompt = str(sample.get("prompt_with_refs", ""))
    prompt_tokens = _ANY_REF_TOKEN.findall(prompt)
    sample_entities = sample.get("entities")
    entities_by_id: dict[str, dict[str, object]] = {}
    if not isinstance(sample_entities, list):
        issues.append(
            ValidationIssue(
                "sample_entities_missing",
                "entities",
                "final sample must contain its complete entity list",
            )
        )
        sample_entities = []
    for index, entity in enumerate(sample_entities):
        if not isinstance(entity, dict) or not isinstance(entity.get("entity_id"), str):
            issues.append(
                ValidationIssue(
                    "invalid_sample_entity",
                    f"entities[{index}]",
                    "sample entity must be an object with entity_id",
                )
            )
            continue
        entity_id = str(entity["entity_id"])
        if entity_id in entities_by_id:
            issues.append(
                ValidationIssue(
                    "duplicate_sample_entity",
                    f"entities[{index}].entity_id",
                    "sample entity IDs must be unique",
                )
            )
        entities_by_id[entity_id] = entity

    relations = sample.get("relations", [])
    if not isinstance(relations, list):
        issues.append(
            ValidationIssue(
                "invalid_sample_relations",
                "relations",
                "sample relations must be a list",
            )
        )
        relations = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            issues.append(
                ValidationIssue(
                    "invalid_sample_relation",
                    f"relations[{index}]",
                    "sample relation must be a JSON object",
                )
            )
            continue
        if (
            relation.get("subject_id") not in entities_by_id
            or relation.get("object_id") not in entities_by_id
        ):
            issues.append(
                ValidationIssue(
                    "invalid_relation_entity",
                    f"relations[{index}]",
                    "relation subject_id and object_id must exist in sample.entities",
                )
            )

    references = sample.get("references")
    if not isinstance(references, list) or not references:
        issues.append(
            ValidationIssue(
                "no_references",
                "references",
                "final sample must contain at least one reference",
            )
        )
        references = []

    expected_tokens: list[str] = []
    annotation_by_id = (
        {entity.entity_id: entity for entity in annotation.entities}
        if annotation
        else {}
    )
    for index, reference in enumerate(references):
        field = f"references[{index}]"
        if not isinstance(reference, dict):
            issues.append(
                ValidationIssue(
                    "invalid_reference",
                    field,
                    "reference must be a JSON object",
                )
            )
            continue
        token = reference.get("ref_token")
        phrase = reference.get("phrase")
        if not isinstance(token, str) or _REF_TOKEN.fullmatch(token) is None:
            issues.append(
                ValidationIssue(
                    "invalid_reference_token",
                    f"{field}.ref_token",
                    "reference token is missing or invalid",
                )
            )
        else:
            expected_tokens.append(token)
            if not isinstance(phrase, str) or prompt.count(f"{phrase} {token}") != 1:
                issues.append(
                    ValidationIssue(
                        "reference_phrase_mismatch",
                        f"{field}.phrase",
                        "reference phrase and token must have one prompt binding",
                    )
                )
        image_path = reference.get("image_path")
        canonical_image = None
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            issues.append(
                ValidationIssue(
                    "reference_image_missing",
                    f"{field}.image_path",
                    "reference image must exist",
                )
            )
        else:
            canonical_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if canonical_image is None:
                issues.append(
                    ValidationIssue(
                        "reference_image_unreadable",
                        f"{field}.image_path",
                        "reference image must be readable",
                    )
                )
        mask_path = reference.get("mask_path")
        if not isinstance(mask_path, str) or not Path(mask_path).is_file():
            issues.append(
                ValidationIssue(
                    "reference_mask_missing",
                    f"{field}.mask_path",
                    "reference mask must exist",
                )
            )
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                issues.append(
                    ValidationIssue(
                        "reference_mask_unreadable",
                        f"{field}.mask_path",
                        "reference mask must be readable",
                    )
                )
            elif not np.any(mask):
                issues.append(
                    ValidationIssue(
                        "reference_mask_empty",
                        f"{field}.mask_path",
                        "reference mask must contain foreground pixels",
                    )
                )
            elif (
                canonical_image is not None and mask.shape != canonical_image.shape[:2]
            ):
                issues.append(
                    ValidationIssue(
                        "reference_mask_size_mismatch",
                        f"{field}.mask_path",
                        "reference mask dimensions must match canonical image",
                    )
                )
        entity_id = str(reference.get("entity_id", ""))
        sample_entity = entities_by_id.get(entity_id)
        if sample_entity is None or sample_entity.get("reference_worthy") is not True:
            issues.append(
                ValidationIssue(
                    "reference_entity_not_reference_worthy",
                    f"{field}.entity_id",
                    "reference must map to a reference-worthy sample entity",
                )
            )
        elif (
            sample_entity.get("ref_token") != token
            or sample_entity.get("phrase") != phrase
        ):
            issues.append(
                ValidationIssue(
                    "reference_entity_binding_mismatch",
                    field,
                    "reference phrase and token must match sample.entities",
                )
            )
        if annotation is not None:
            entity = annotation_by_id.get(entity_id)
            if (
                entity is None
                or entity.ref_token != token
                or entity.phrase != phrase
                or not entity.reference_worthy
            ):
                issues.append(
                    ValidationIssue(
                        "reference_annotation_mismatch",
                        field,
                        "reference binding does not match the retained annotation",
                    )
                )

    background = sample.get("background_reference")
    if background is not None:
        if not isinstance(background, dict):
            issues.append(
                ValidationIssue(
                    "invalid_background_reference",
                    "background_reference",
                    "background reference must be a JSON object or null",
                )
            )
        else:
            token = background.get("ref_token")
            image_path = background.get("image_path")
            if (
                not isinstance(token, str)
                or _REF_TOKEN.fullmatch(token) is None
                or not token.startswith("<ref_bg_")
            ):
                issues.append(
                    ValidationIssue(
                        "invalid_background_token",
                        "background_reference.ref_token",
                        "background reference token is missing or invalid",
                    )
                )
            else:
                expected_tokens.append(token)
            if not isinstance(image_path, str) or not Path(image_path).is_file():
                issues.append(
                    ValidationIssue(
                        "background_image_missing",
                        "background_reference.image_path",
                        "background reference image must exist",
                    )
                )

    if len(expected_tokens) != len(set(expected_tokens)):
        issues.append(
            ValidationIssue(
                "duplicate_reference_token",
                "references",
                "final reference tokens must be unique",
            )
        )
    if sorted(prompt_tokens) != sorted(expected_tokens):
        issues.append(
            ValidationIssue(
                "final_prompt_token_mismatch",
                "prompt_with_refs",
                "prompt tokens must exactly equal final reference tokens",
            )
        )
    return issues
