from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import BadRequestError, OpenAI

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
)
from r2v_data_v2.v3.config import QwenAnnotationConfig, V3Config
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    RawAnnotationPayload,
)
from r2v_data_v2.v3.storage import RunStorage

_REFERENCE_TOKEN = re.compile(r"<ref_[^>]+>", flags=re.IGNORECASE)
_JSON_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",
    flags=re.IGNORECASE,
)
_PHRASE_EDGE_PUNCTUATION = (
    " \t\r\n.,;:!?\"'`()[]{}"
    "\u2018\u2019\u201c\u201d"
    "\uff0c\u3002\uff1b\uff1a\uff01\uff1f"
)
_ENTITY_FIELDS = frozenset(
    {"reference_type", "phrase", "grounding_prompt"}
)
_BACKGROUND_FIELDS = frozenset({"phrase", "grounding_prompt"})
_REFERENCE_TYPES = frozenset({"subject", "object", "group"})

SYSTEM_PROMPT = """You annotate a complete video for a V3 training-data pipeline.

Return exactly one JSON object matching the supplied minimal schema. Output only
t2v_caption, entities, and background. Do not output relations, entity_id,
category, salience, genericity, name_evidence, localization_scope, scene_role,
representation_mode, visual_scope, separability, selection_reason, reference
tokens, instructions, or any additional ontology fields.

Write t2v_caption as one complete English paragraph that begins directly with
visible content and describes actions and shot changes in chronological order.
Include visible subject appearance, action, scene, composition, camera behavior,
and lighting. Describe only directly visible content. Do not infer identity,
emotion, intent, sound, dialogue, or event causes. Do not write "the video
shows". Do not include <ref_...> tokens or image-number instruction labels.

Return at most three entities. Select only stable, discrete foreground reference
candidates that SAM3 can localize and track and that could be reused as an
independent reference image. Fewer than three is preferred when evidence is
weak. Do not select environmental regions such as sky, ocean, water surface,
ground, roads, or room space; distributed or transient content such as clouds,
smoke, flame, lighting, shadows, reflections, or weather; content depicted in a
screen, photo, poster, painting, sculpture, or animation; small attached
accessories without independent reference value; or tiny, brief, blurred, or
untrackable objects. These examples guide selection only and are not an object
name ontology.

For each entity output only reference_type, phrase, and grounding_prompt.
reference_type must be subject, object, or group. Use subject for one person,
animal, or character whose identity or appearance should be retained; object for
one independently referenceable product, vehicle, prop, device, piece of
furniture, or other object; and group for multiple subjects or objects whose
stable composition should be retained together. phrase briefly identifies the
candidate for binding and review. grounding_prompt describes visible appearance
and only the positional detail needed to distinguish it for SAM3. Both fields
must be non-empty and must not contain reference tokens.

background is optional. When reliable, describe the overall environment after
the principal foreground subjects are removed, using only phrase and
grounding_prompt. Do not repeat the main foreground subject. Otherwise return
null. Return JSON only."""


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


def _strip_json_fence(raw: str) -> str:
    stripped = raw.strip()
    if not stripped.startswith("```"):
        return stripped
    match = _JSON_FENCE.fullmatch(stripped)
    if match is None:
        raise ValueError("response must be one complete JSON object")
    return match.group(1).strip()


def _parse_raw_payload(
    raw: str,
) -> tuple[dict[str, object] | None, list[ValidationIssue]]:
    try:
        value = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError as exc:
        return None, [
            ValidationIssue(
                code="invalid_json",
                field=None,
                message=f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            )
        ]
    except ValueError as exc:
        return None, [
            ValidationIssue(
                code="invalid_json_object",
                field=None,
                message=str(exc),
            )
        ]
    if not isinstance(value, dict):
        return None, [
            ValidationIssue(
                code="invalid_json_object",
                field=None,
                message="Qwen response must be one JSON object",
            )
        ]
    return value, []


def _clean_text(value: object, *, trim_phrase_punctuation: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if trim_phrase_punctuation:
        cleaned = cleaned.strip(_PHRASE_EDGE_PUNCTUATION)
    return cleaned or None


def _normalized_phrase(value: str) -> str:
    return " ".join(value.casefold().split()).strip(_PHRASE_EDGE_PUNCTUATION)


def sanitize_entity_candidates(
    raw_entities: object,
) -> tuple[list[AnnotationEntity], tuple[str, ...]]:
    if not isinstance(raw_entities, list):
        return [], ("dropped_invalid_entities_collection",)

    accepted: list[tuple[str, str, str]] = []
    seen_phrases: set[str] = set()
    warnings: list[str] = []
    for index, candidate in enumerate(raw_entities):
        if not isinstance(candidate, dict):
            warnings.append(f"dropped_entity_not_object:{index}")
            continue
        if set(candidate) != _ENTITY_FIELDS:
            warnings.append(f"dropped_entity_fields:{index}")
            continue
        reference_type = _clean_text(candidate.get("reference_type"))
        phrase = _clean_text(
            candidate.get("phrase"),
            trim_phrase_punctuation=True,
        )
        grounding_prompt = _clean_text(candidate.get("grounding_prompt"))
        normalized_type = reference_type.casefold() if reference_type else ""
        if normalized_type not in _REFERENCE_TYPES:
            warnings.append(f"dropped_entity_reference_type:{index}")
            continue
        if phrase is None:
            warnings.append(f"dropped_entity_phrase:{index}")
            continue
        if grounding_prompt is None:
            warnings.append(f"dropped_entity_grounding_prompt:{index}")
            continue
        if _REFERENCE_TOKEN.search(phrase) or _REFERENCE_TOKEN.search(
            grounding_prompt
        ):
            warnings.append(f"dropped_entity_reference_token:{index}")
            continue
        phrase_key = _normalized_phrase(phrase)
        if not phrase_key:
            warnings.append(f"dropped_entity_phrase:{index}")
            continue
        if phrase_key in seen_phrases:
            warnings.append(f"dropped_duplicate_entity_phrase:{index}")
            continue
        seen_phrases.add(phrase_key)
        accepted.append((normalized_type, phrase, grounding_prompt))
        if len(accepted) == 3:
            if index + 1 < len(raw_entities):
                warnings.append("truncated_entity_candidates:3")
            break

    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=reference_type,
            phrase=phrase,
            grounding_prompt=grounding_prompt,
        )
        for index, (reference_type, phrase, grounding_prompt) in enumerate(
            accepted,
            start=1,
        )
    ]
    return entities, tuple(warnings)


def sanitize_background(
    raw_background: object,
) -> tuple[BackgroundAnnotation | None, tuple[str, ...]]:
    if raw_background is None:
        return None, ()
    if not isinstance(raw_background, dict):
        return None, ("dropped_invalid_background",)
    if set(raw_background) != _BACKGROUND_FIELDS:
        return None, ("dropped_invalid_background_fields",)
    phrase = _clean_text(
        raw_background.get("phrase"),
        trim_phrase_punctuation=True,
    )
    grounding_prompt = _clean_text(raw_background.get("grounding_prompt"))
    if (
        phrase is None
        or grounding_prompt is None
        or _REFERENCE_TOKEN.search(phrase)
        or _REFERENCE_TOKEN.search(grounding_prompt)
    ):
        return None, ("dropped_invalid_background",)
    return (
        BackgroundAnnotation(
            phrase=phrase,
            grounding_prompt=grounding_prompt,
        ),
        (),
    )


def sanitize_annotation_payload(
    raw_payload: dict[str, object],
) -> tuple[AnnotationState | None, list[ValidationIssue], tuple[str, ...]]:
    caption = _clean_text(raw_payload.get("t2v_caption"))
    issues: list[ValidationIssue] = []
    if caption is None:
        issues.append(
            ValidationIssue(
                code="empty_t2v_caption",
                field="t2v_caption",
                message="t2v_caption must be a non-empty string",
            )
        )
    elif _REFERENCE_TOKEN.search(caption):
        issues.append(
            ValidationIssue(
                code="reference_token_in_annotation",
                field="t2v_caption",
                message="t2v_caption must not contain reference tokens",
            )
        )
    if issues:
        return None, issues, ()

    entities, entity_warnings = sanitize_entity_candidates(
        raw_payload.get("entities", [])
    )
    background, background_warnings = sanitize_background(
        raw_payload.get("background")
    )
    return (
        AnnotationState(
            status="ready",
            t2v_caption=caption,
            entities=entities,
            background=background,
        ),
        [],
        (*entity_warnings, *background_warnings),
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
        "Inspect the attached complete video from beginning to end and return "
        "the minimal annotation JSON described by the system prompt. Treat this "
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
        "Repair only the top-level structured-output problems listed below. "
        "Reinspect the same complete video and return the complete minimal JSON "
        "object. Do not add relations, entity IDs, ontology fields, reference "
        "tokens, or instruction fields.\n"
        f"Original request:\n{original_request}\n"
        "JSON Schema:\n"
        f"{json.dumps(RawAnnotationPayload.model_json_schema(), ensure_ascii=False)}\n"
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
                        "schema": RawAnnotationPayload.model_json_schema(),
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
            raw_payload, issues = _parse_raw_payload(raw)
            if raw_payload is None:
                continue
            annotation, issues, warnings = sanitize_annotation_payload(raw_payload)
            if annotation is not None and not issues:
                return AnnotationAttempt(
                    annotation=annotation,
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                    warnings=warnings,
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
            "issues": [issue.to_dict() for issue in (issues or [])],
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
                    "issues": [issue.to_dict() for issue in exc.issues],
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
