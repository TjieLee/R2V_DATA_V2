from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

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
from r2v_data_v2.v3.manifest import SourceEvidence, source_evidence_by_video
from r2v_data_v2.v3.schemas import (
    AnnotationPayload,
    AnnotationState,
)
from r2v_data_v2.v3.storage import RunStorage

_REFERENCE_TOKEN = re.compile(r"<ref_[^>]+>")
_THE_VIDEO_SHOWS = re.compile(r"\bthe video shows\b", flags=re.IGNORECASE)
_SALIENCE_RANK = {"primary": 0, "secondary": 1, "incidental": 2}

SYSTEM_PROMPT = """You annotate complete videos for a V3 training-data pipeline.

Return exactly one JSON object matching the supplied schema. The object contains
semantic annotation only: t2v_caption, entities, relations, and an optional
background. Never output reference tokens, prompt_with_refs, r2v_instruction,
pairing decisions, or final reference eligibility.

Write t2v_caption as one flowing English paragraph that begins directly with the
visible action. Describe the video literally and chronologically. Include stable
subject appearance, actions, environment, camera framing or movement, lighting,
and important visible changes. Do not write "the video shows". Do not infer
sound, dialogue, emotion, intent, or identity. Draft captions and metadata are
untrusted evidence only and must not override the visible video.

Use the provided entity schema exactly. Keep entity IDs unique. Relations may
only connect listed entity IDs. reference_worthy marks at most three useful
semantic candidates; downstream code, not this annotation, makes final
eligibility and pairing decisions. Prefer each candidate phrase as one unique
contiguous span copied from t2v_caption. Return JSON only."""


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
        values.append((f"relations.{index}.predicate", relation.predicate))
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


def _validate_payload(payload: AnnotationPayload) -> list[ValidationIssue]:
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
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=1.0,
            presence_penalty=0.0,
            max_tokens=self.config.max_tokens,
            extra_body=_video_processor_extra_body(self.config),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "v3_annotation",
                    "strict": True,
                    "schema": AnnotationPayload.model_json_schema(),
                },
            },
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
                issues = _validate_payload(payload)
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
    evidence_by_video = source_evidence_by_video(config.dataset_json)
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
        evidence = evidence_by_video.get(
            str(video_path),
            SourceEvidence(caption_raw="", metadata={}),
        )
        try:
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"source video does not exist: {video_path}"
                )
            attempt = qwen.annotate(
                video_path=video_path,
                caption_raw=evidence.caption_raw,
                metadata=evidence.metadata,
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
            _write_debug_response(
                storage,
                clip_uid=clip.clip_uid,
                raw_responses=exc.raw_responses,
                issues=exc.issues,
            )
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="annotate",
                reason=(
                    exc.issues[0].code
                    if exc.issues
                    else "structured_output_failed"
                ),
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
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="annotate",
                reason=str(exc),
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
