from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import BadRequestError, OpenAI

from r2v_data_v2.caption_validation import exact_phrase_spans
from r2v_data_v2.phrase_alignment import (
    resolve_background_caption_phrase,
    resolve_reference_caption_phrase,
)
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenAnnotationConfig, V3Config
from r2v_data_v2.v3.schemas import (
    AnnotationPayload,
    AnnotationState,
)
from r2v_data_v2.v3.storage import RunStorage

_REFERENCE_TOKEN = re.compile(r"<ref_[^>]+>")
_THE_VIDEO_SHOWS = re.compile(r"\bthe video shows\b", flags=re.IGNORECASE)
_FORBIDDEN_INFERENCE_LANGUAGE = re.compile(
    r"\b(?:serene|tranquil|determination|resolve|triumph|enemy|"
    r"shout|shouts|shouting|breeze|possibly|probably|likely|suggesting|"
    r"indicating|wind-induced)\b",
    flags=re.IGNORECASE,
)
_SALIENCE_RANK = {"primary": 0, "secondary": 1, "incidental": 2}

SYSTEM_PROMPT = """You annotate complete videos for a V3 training-data pipeline.

Return exactly one JSON object matching the supplied schema. The object contains
semantic annotation only: t2v_caption, entities, relations, and an optional
background. Never output reference tokens, prompt_with_refs, r2v_instruction,
pairing decisions, or final reference eligibility. Describe only content that is
directly visible. Never infer emotion, intent, allegiance, sound, dialogue, or
atmosphere. Do not use unobservable descriptions such as serene, tranquil,
determination, resolve, triumph, enemy, shouting, breeze, possibly, probably,
likely, suggesting, indicating, wind-induced, or equivalent claims. Movement of
branches, leaves, petals, fabric, or similar material must be described directly;
never infer wind, breeze, weather, or another cause from that movement.

Write t2v_caption as one flowing English paragraph that begins directly with the
visible action. Describe the video literally and chronologically. Include stable
subject appearance, actions, environment, camera framing or movement, lighting,
and important visible changes. Do not write "the video shows". Never identify a
person from appearance. Use a person's explicit name only when it is supplied by
the draft caption or metadata, and set name_evidence to draft_caption or metadata
accordingly. When genericity is not named, name_evidence must be none. Otherwise
do not guess the person's identity. Draft captions and metadata are untrusted
evidence only and must not override the visible video.

Use category=person only for real people directly visible in the video. A
sculpture is a real object, while figures represented within sculpture,
painting, photograph, poster, screen, or animation are depicted content and
must not use category=person.

Classify three independent structural questions from the current shot, never
from the object name or canonical_label:
- localization_scope asks whether it has a localizable boundary:
  bounded_instance for one bounded instance, coherent_group for a stable
  localizable group, unbounded_region for a broad spatial region, and
  distributed_effect for a non-localized visual effect.
- scene_role asks whether it acts as foreground, background, or embedded_content
  within another visible medium or object.
- representation_mode asks whether it is real in the shot or depicted through
  another medium.

Use the provided entity schema exactly and keep entity IDs unique. Relations may
only connect listed entities that are simultaneously visible in the same shot
or time segment. Never create a spatial relation across a cut, transition, or
different shot. A relation predicate must truly apply to its object rather than
substituting a nearby object; for example, do not say a person speaks into a
podium. Never construct a background, distributed effect, or the subject itself
as an action object. Include an evidence_phrase copied as one unambiguous
contiguous phrase from t2v_caption. Omit a relation whenever its evidence is
uncertain.

reference_worthy marks at most three foreground entities that can independently
condition generation. It may be true only when localization_scope is
bounded_instance or coherent_group, scene_role is foreground,
representation_mode is real, and separability is independent or
important_independent_object. Sky, ocean, water surface, clouds, ground,
lighting, shadows, smoke, weather, screen content, and the overall scene are
classification examples only; eligibility must come from the structural fields,
never a name blacklist. An integral component of a larger object must use
attached_accessory rather than independent separability. An entity candidate
must not describe the same scene region as background. A central subject must be
primary, never incidental.
Downstream code, not this annotation, makes final eligibility and pairing
decisions. Prefer each candidate phrase as one unique contiguous span copied
from t2v_caption. Return JSON only."""


@dataclass(frozen=True)
class AnnotationAttempt:
    annotation: AnnotationState
    raw_responses: tuple[str, ...]
    repair_attempts: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotationStats:
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0
    repaired: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class AnnotationFailure(StructuredOutputFailure):
    pass


class AnnotationClient(Protocol):
    def annotate(
        self,
        *,
        video_path: Path,
        caption_raw: str,
        metadata: dict[str, object],
    ) -> AnnotationAttempt: ...


def _video_processor_extra_body(
    config: QwenAnnotationConfig,
) -> dict[str, object]:
    return {
        "mm_processor_kwargs": {
            "fps": config.video.fps,
            "do_sample_frames": config.video.do_sample_frames,
        }
    }


def _text_fields(payload: AnnotationPayload) -> list[tuple[str, str]]:
    values = [("t2v_caption", payload.t2v_caption)]
    for index, entity in enumerate(payload.entities):
        values.extend(
            (
                (f"entities.{index}.phrase", entity.phrase),
                (f"entities.{index}.grounding_prompt", entity.grounding_prompt),
                (f"entities.{index}.canonical_label", entity.canonical_label),
                (f"entities.{index}.selection_reason", entity.selection_reason),
            )
        )
    for index, relation in enumerate(payload.relations):
        values.extend(
            (
                (f"relations.{index}.predicate", relation.predicate),
                (
                    f"relations.{index}.evidence_phrase",
                    relation.evidence_phrase,
                ),
            )
        )
    if payload.background is not None:
        values.extend(
            (
                ("background.phrase", payload.background.phrase),
                (
                    "background.grounding_prompt",
                    payload.background.grounding_prompt,
                ),
            )
        )
    return values


def _metadata_text(metadata: dict[str, object]) -> str:
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return " ".join(values).casefold()


def _contains_name(evidence: str, canonical_label: str) -> bool:
    name = " ".join(canonical_label.split()).casefold()
    if not name:
        return False
    pattern = re.escape(name)
    if name[0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if name[-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return re.search(pattern, evidence.casefold()) is not None


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _is_complete_phrase_within(phrase: str, text: str) -> bool:
    normalized_phrase = _normalized_phrase(phrase)
    normalized_text = _normalized_phrase(text)
    if not normalized_phrase or not normalized_text:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _append_forbidden_inference_issue(
    issues: list[ValidationIssue],
    *,
    field: str,
    value: str,
) -> None:
    match = _FORBIDDEN_INFERENCE_LANGUAGE.search(value)
    if match is None:
        return
    issues.append(
        ValidationIssue(
            code="forbidden_inference_language",
            field=field,
            message=(
                "annotation contains forbidden inference language: "
                f"{match.group(0)}"
            ),
        )
    )


def _validate_payload(
    payload: AnnotationPayload,
    *,
    caption_raw: str,
    metadata: dict[str, object],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not payload.t2v_caption.strip():
        issues.append(
            ValidationIssue(
                code="empty_t2v_caption",
                field="t2v_caption",
                message="t2v_caption must not be empty",
            )
        )
    if _THE_VIDEO_SHOWS.search(payload.t2v_caption):
        issues.append(
            ValidationIssue(
                code="forbidden_caption_intro",
                field="t2v_caption",
                message='t2v_caption must not say "the video shows"',
            )
        )
    _append_forbidden_inference_issue(
        issues,
        field="t2v_caption",
        value=payload.t2v_caption,
    )
    for field, value in _text_fields(payload):
        if _REFERENCE_TOKEN.search(value):
            issues.append(
                ValidationIssue(
                    code="reference_token_in_annotation",
                    field=field,
                    message="annotation semantic fields must not contain reference tokens",
                )
            )
    entity_ids = [entity.entity_id for entity in payload.entities]
    if len(entity_ids) != len(set(entity_ids)):
        issues.append(
            ValidationIssue(
                code="duplicate_entity_id",
                field="entities",
                message="annotation entity IDs must be unique",
            )
        )
    metadata_text = _metadata_text(metadata)
    for index, entity in enumerate(payload.entities):
        evidence = entity.name_evidence
        if entity.genericity != "named" and evidence != "none":
            issues.append(
                ValidationIssue(
                    code="unexpected_name_evidence",
                    field=f"entities.{index}.name_evidence",
                    message="non-named entities must use name_evidence=none",
                )
            )
        if entity.reference_worthy and entity.localization_scope not in {
            "bounded_instance",
            "coherent_group",
        }:
            issues.append(
                ValidationIssue(
                    code="invalid_reference_localization_scope",
                    field=f"entities.{index}.localization_scope",
                    message=(
                        "reference-worthy entities require bounded_instance "
                        "or coherent_group localization"
                    ),
                )
            )
        if entity.reference_worthy and entity.scene_role != "foreground":
            issues.append(
                ValidationIssue(
                    code="invalid_reference_scene_role",
                    field=f"entities.{index}.scene_role",
                    message="reference-worthy entities must be foreground",
                )
            )
        if entity.reference_worthy and entity.representation_mode != "real":
            issues.append(
                ValidationIssue(
                    code="invalid_reference_representation",
                    field=f"entities.{index}.representation_mode",
                    message="reference-worthy entities must be real",
                )
            )
        if entity.reference_worthy and entity.separability not in {
            "independent",
            "important_independent_object",
        }:
            issues.append(
                ValidationIssue(
                    code="invalid_reference_separability",
                    field=f"entities.{index}.separability",
                    message=(
                        "reference-worthy entities require independent "
                        "separability"
                    ),
                )
            )
        if entity.reference_worthy and entity.salience == "incidental":
            issues.append(
                ValidationIssue(
                    code="invalid_reference_salience",
                    field=f"entities.{index}.salience",
                    message="reference-worthy entities cannot be incidental",
                )
            )
        if (
            entity.category == "person"
            and entity.representation_mode == "depicted"
        ):
            issues.append(
                ValidationIssue(
                    code="depicted_person_category",
                    field=f"entities.{index}.category",
                    message=(
                        "depicted people must not use category=person"
                    ),
                )
            )
        if entity.reference_worthy and payload.background is not None:
            background_texts = (
                payload.background.phrase,
                payload.background.grounding_prompt,
            )
            if any(
                _is_complete_phrase_within(entity.phrase, background_text)
                for background_text in background_texts
            ):
                issues.append(
                    ValidationIssue(
                        code="reference_background_overlap",
                        field=f"entities.{index}.phrase",
                        message=(
                            "reference-worthy entity phrase overlaps the "
                            "background description"
                        ),
                    )
                )
        _append_forbidden_inference_issue(
            issues,
            field=f"entities.{index}.selection_reason",
            value=entity.selection_reason,
        )
        if (
            entity.category == "person"
            and entity.genericity == "named"
            and evidence not in {"draft_caption", "metadata"}
        ):
            issues.append(
                ValidationIssue(
                    code="missing_name_evidence",
                    field=f"entities.{index}.name_evidence",
                    message=(
                        "named people require explicit draft-caption or "
                        "metadata evidence"
                    ),
                )
            )
        if evidence == "draft_caption" and not _contains_name(
            caption_raw,
            entity.canonical_label,
        ):
            issues.append(
                ValidationIssue(
                    code="invalid_name_evidence",
                    field=f"entities.{index}.name_evidence",
                    message="canonical name is not present in the draft caption",
                )
            )
        if evidence == "metadata" and not _contains_name(
            metadata_text,
            entity.canonical_label,
        ):
            issues.append(
                ValidationIssue(
                    code="invalid_name_evidence",
                    field=f"entities.{index}.name_evidence",
                    message="canonical name is not present in metadata",
                )
            )
    for index, relation in enumerate(payload.relations):
        if relation.subject_id == relation.object_id:
            issues.append(
                ValidationIssue(
                    code="self_relation",
                    field=f"relations.{index}",
                    message="relation subject and object must be different",
                )
            )
        if len(
            exact_phrase_spans(
                payload.t2v_caption,
                relation.evidence_phrase,
            )
        ) != 1:
            issues.append(
                ValidationIssue(
                    code="invalid_relation_evidence",
                    field=f"relations.{index}.evidence_phrase",
                    message=(
                        "relation evidence must be one unique contiguous "
                        "caption phrase"
                    ),
                )
            )
        _append_forbidden_inference_issue(
            issues,
            field=f"relations.{index}.predicate",
            value=relation.predicate,
        )
    return issues


def _align_reference_phrases(
    payload: AnnotationPayload,
) -> tuple[AnnotationPayload, list[str]]:
    caption = " ".join(payload.t2v_caption.split())
    warnings: list[str] = []
    entities = []
    for entity in payload.entities:
        if not entity.reference_worthy:
            entities.append(entity)
            continue
        if len(exact_phrase_spans(caption, entity.phrase)) == 1:
            entities.append(entity)
            continue
        aligned = resolve_reference_caption_phrase(caption, entity.phrase)
        if (
            aligned is not None
            and len(exact_phrase_spans(caption, aligned)) == 1
        ):
            entities.append(entity.model_copy(update={"phrase": aligned}))
            warnings.append(f"aligned_reference_phrase:{entity.entity_id}")
            continue
        entities.append(entity.model_copy(update={"reference_worthy": False}))
        warnings.append(f"demoted_unalignable_reference:{entity.entity_id}")

    known_ids = {entity.entity_id for entity in entities}
    relations = []
    for index, relation in enumerate(payload.relations):
        if (
            relation.subject_id not in known_ids
            or relation.object_id not in known_ids
        ):
            warnings.append(f"dropped_invalid_relation:{index}")
            continue
        relations.append(relation)

    selected_indexes = {
        index
        for index, entity in sorted(
            enumerate(entities),
            key=lambda item: (_SALIENCE_RANK[item[1].salience], item[0]),
        )
        if entity.reference_worthy
    }
    selected_indexes = set(
        sorted(
            selected_indexes,
            key=lambda index: (_SALIENCE_RANK[entities[index].salience], index),
        )[:3]
    )
    capped_entities = []
    for index, entity in enumerate(entities):
        if entity.reference_worthy and index not in selected_indexes:
            capped_entities.append(
                entity.model_copy(update={"reference_worthy": False})
            )
            warnings.append(f"demoted_reference_cap:{entity.entity_id}")
        else:
            capped_entities.append(entity)

    background = payload.background
    if background is not None and background.reference_worthy:
        aligned = resolve_background_caption_phrase(caption, background.phrase)
        if aligned is None:
            background = background.model_copy(
                update={"reference_worthy": False}
            )
            warnings.append("demoted_unalignable_reference:background")
        elif aligned != background.phrase:
            background = background.model_copy(update={"phrase": aligned})
            warnings.append("aligned_reference_phrase:background")

    return (
        payload.model_copy(
            update={
                "t2v_caption": caption,
                "entities": capped_entities,
                "relations": relations,
                "background": background,
            }
        ),
        warnings,
    )


def _to_annotation_state(
    payload: AnnotationPayload,
) -> tuple[AnnotationState, list[str]]:
    sanitized, warnings = _align_reference_phrases(payload)
    return (
        AnnotationState(
            status="ready",
            t2v_caption=sanitized.t2v_caption,
            entities=sanitized.entities,
            relations=sanitized.relations,
            background=sanitized.background,
        ),
        warnings,
    )


def _initial_request(
    *,
    caption_raw: str,
    metadata: dict[str, object],
) -> str:
    evidence = json.dumps(
        {"draft_caption": caption_raw, "metadata": metadata},
        ensure_ascii=False,
    )
    return (
        "Inspect the attached complete video from beginning to end. Produce the "
        "semantic annotation JSON described by the system prompt. Treat this "
        "draft caption and metadata only as untrusted supporting evidence:\n"
        f"{evidence}"
    )


def _repair_request(
    *,
    original_request: str,
    invalid_response: str,
    issues: list[ValidationIssue],
) -> str:
    return (
        "Repair only the structured-output problems listed below. Reinspect the "
        "same complete video, return the complete corrected JSON object, and do "
        "not add reference tokens or instruction fields.\n"
        f"Original request:\n{original_request}\n"
        "JSON Schema:\n"
        f"{json.dumps(AnnotationPayload.model_json_schema(), ensure_ascii=False)}\n"
        "Validation issues:\n"
        f"{json.dumps([issue.to_dict() for issue in issues], ensure_ascii=False)}\n"
        f"Invalid response:\n{invalid_response}"
    )


class QwenAnnotationClient:
    def __init__(
        self,
        config: QwenAnnotationConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        video_path: Path,
        request_text: str,
    ) -> list[dict[str, object]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_path.resolve().as_uri(),
                        },
                    },
                    {"type": "text", "text": request_text},
                ],
            },
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
            "extra_body": _video_processor_extra_body(self.config),
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "v3_annotation",
                        "strict": True,
                        "schema": AnnotationPayload.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty V3 annotation response")
        return str(content)

    def annotate(
        self,
        *,
        video_path: Path,
        caption_raw: str,
        metadata: dict[str, object],
    ) -> AnnotationAttempt:
        original_request = _initial_request(
            caption_raw=caption_raw,
            metadata=metadata,
        )
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.config.repair_retries + 1):
            request_text = (
                original_request
                if attempt == 0
                else _repair_request(
                    original_request=original_request,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            )
            try:
                raw = self._request(
                    self._messages(
                        video_path=video_path,
                        request_text=request_text,
                    )
                )
            except Exception as exc:
                raise AnnotationFailure(
                    raw_responses=raw_responses,
                    issues=[
                        ValidationIssue(
                            code="qwen_request_failed",
                            field=None,
                            message=str(exc),
                        )
                    ],
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            payload, issues = parse_qwen_json_issues(raw, AnnotationPayload)
            if payload is not None:
                issues = _validate_payload(
                    payload,
                    caption_raw=caption_raw,
                    metadata=metadata,
                )
            if payload is not None and not issues:
                annotation, warnings = _to_annotation_state(payload)
                return AnnotationAttempt(
                    annotation=annotation,
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                    warnings=tuple(warnings),
                )
        raise AnnotationFailure(raw_responses=raw_responses, issues=issues)


def _write_debug_response(
    storage: RunStorage,
    *,
    clip_uid: str,
    raw_responses: tuple[str, ...] | list[str],
    issues: list[ValidationIssue] | None = None,
    warnings: tuple[str, ...] = (),
) -> None:
    if not storage.config.debug.save_diagnostics:
        return
    destination = storage.debug_path(clip_uid, "annotation_raw.json")
    write_json_atomic(
        destination,
        {
            "raw_responses": list(raw_responses),
            "issues": [
                issue.to_dict()
                for issue in (issues or [])
            ],
            "warnings": list(warnings),
        },
    )


def annotate_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    client: AnnotationClient | None = None,
) -> AnnotationStats:
    clips = list(storage.iter_clips())
    if not clips:
        raise FileNotFoundError(
            "annotate stage requires manifest stage to create clip.json records first"
        )
    qwen = client or QwenAnnotationClient(config.qwen.annotation)
    processed = skipped_existing = failed = repaired = 0

    for clip in clips:
        if (
            clip.annotation is not None
            and clip.annotation.status == "ready"
            and not overwrite
        ):
            skipped_existing += 1
            continue
        video_path = Path(clip.source.video_path)
        try:
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"source video does not exist: {video_path}"
                )
            attempt = qwen.annotate(
                video_path=video_path,
                caption_raw=clip.source.caption_raw,
                metadata=clip.source.metadata,
            )
            storage.write_annotation(clip.clip_uid, attempt.annotation)
            _write_debug_response(
                storage,
                clip_uid=clip.clip_uid,
                raw_responses=attempt.raw_responses,
                warnings=attempt.warnings,
            )
            processed += 1
            repaired += int(attempt.repair_attempts > 0)
        except AnnotationFailure as exc:
            reason = (
                exc.issues[0].code
                if exc.issues
                else "structured_output_failed"
            )
            storage.write_annotation(
                clip.clip_uid,
                AnnotationState(status="failed", reason=reason),
            )
            _write_debug_response(
                storage,
                clip_uid=clip.clip_uid,
                raw_responses=exc.raw_responses,
                issues=exc.issues,
            )
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="annotate",
                reason=reason,
                details={
                    "attempt_count": exc.attempt_count,
                    "issues": [
                        issue.to_dict()
                        for issue in exc.issues
                    ],
                },
            )
            failed += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
            reason = str(exc)
            storage.write_annotation(
                clip.clip_uid,
                AnnotationState(status="failed", reason=reason),
            )
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="annotate",
                reason=reason,
            )
            failed += 1

    stats = AnnotationStats(
        processed=processed,
        skipped_existing=skipped_existing,
        failed=failed,
        repaired=repaired,
    )
    storage.update_stage_counts("annotate", stats.to_dict())
    return stats
