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
from r2v_data_v2.v3.profiling import (
    get_model_profile_context,
    model_profile_context,
    profiled_openai_call,
)
from r2v_data_v2.v3.schemas import (
    MAX_ANNOTATION_ENTITIES,
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    RawAnnotationPayload,
    plain_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage

_REFERENCE_TOKEN = re.compile(r"<ref_[^>]+>", flags=re.IGNORECASE)
_ENTITY_MARKER = re.compile(r"\{\{entity_([1-9]\d*)\}\}")
_BACKGROUND_MARKER = re.compile(r"\{\{background\}\}")
_BRACED_MARKER = re.compile(r"\{\{([^{}]*)\}\}")
_ANNOTATION_IMAGE_MARKER = re.compile(
    r"\{\{image_[^{}]*\}\}|<Image\s+\d+>",
    flags=re.IGNORECASE,
)
_UNSUPPORTED_CAPTION_INFERENCE = re.compile(
    r"\b(?:breeze|wind-induced|suggesting|indicating|possibly|probably|likely|"
    r"enemy|determination|resolve|triumph)\b|"
    r"\b(?:wind\s+causes|caused\s+by\s+wind)\b",
    flags=re.IGNORECASE,
)
_ENGLISH_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_TRANSIENT_GROUNDING_ACTION = re.compile(
    r"\b(?:gesturing|speaking|talking|turning|walking|running|raising|"
    r"holding\s+up)\b|\bmoving\s+(?:his|her|his/her)\s+hand\b",
    flags=re.IGNORECASE,
)
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
_ENTITY_COUNT_WORDS = ("zero", "one", "two", "three", "four", "five")
_ENTITY_LIMIT_WORD = _ENTITY_COUNT_WORDS[MAX_ANNOTATION_ENTITIES]

SYSTEM_PROMPT = f"""You annotate a complete video for a V3 training-data pipeline.

Return exactly one JSON object matching the supplied minimal schema. Output only
instruction_template, entities, and background. Do not output t2v_caption. Do
not output relations, entity_id,
category, salience, genericity, name_evidence, localization_scope, scene_role,
representation_mode, visual_scope, separability, selection_reason, reference
tokens, image_N placeholders, rendered <Image N> labels, or any additional
ontology fields.

Write instruction_template as one coherent English paragraph that begins
directly with visible content and describes the target video naturally. It is
not an imperative request: do not write "Use the reference image", "Generate",
"Create", or a references section. Describe actions and shot changes in
chronological order. Include visible subject appearance, action, scene,
composition, camera behavior, and lighting without repetition. Prefer 60 to 110
English content words and never exceed 120; internal markers do not count.
Describe only directly visible content. Do not infer identity,
weather, emotion, allegiance, intent, mental state, sound, dialogue, or event
causes. Describe visible motion directly without assigning an unseen cause.
Write "branches sway slightly" instead of claiming that wind causes movement.
Do not use hedging or causal inference wording such as breeze, wind-induced,
suggesting, indicating, possibly, probably, or likely. Do not identify a person
as an enemy, ally, criminal, victim, officer, or another role unless that role
is explicitly supported by source metadata. For statues and depicted figures,
describe visible facial geometry and pose without inferring determination,
resolve, triumph, fear, or effort. Do not write "the video shows". Do not include
<ref_...> tokens or image-number instruction labels.

Return at most {_ENTITY_LIMIT_WORD} entities. Select only stable, discrete
foreground reference candidates that SAM3 can localize and track and that could
be reused as an independent reference image. Fewer than {_ENTITY_LIMIT_WORD} is
preferred when evidence is weak. Do not select environmental regions such as
sky, ocean, water surface,
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
candidate for binding and review in 3 to 10 words and must not exceed 12 words.
phrase is a stable concise label and need not occur verbatim in
instruction_template. It must be a stable noun phrase rather than an action and sufficiently distinguish
the target: prefer "man in a light gray military uniform" over
"military officer speaking at podium". grounding_prompt describes stable visible
appearance and location in 6 to 18 words and must not exceed 24 words. Do not
include transient actions or enumerate every clothing detail. A stable seated
or standing pose is allowed only when needed to distinguish the target for
SAM3. grounding_prompt need not occur in instruction_template. Both fields must be
non-empty and must not contain reference tokens.

Insert exactly one internal marker for every entity in the same array order:
entities[0] uses {{{{entity_1}}}}, entities[1] uses {{{{entity_2}}}}, and so on
through at most {{{{entity_5}}}}. Place each marker immediately after that
entity's first clear mention and before following punctuation. A marker must be
preceded by exactly one ordinary ASCII space, as in "a woman {{{{entity_1}}}},"
or "a boat {{{{entity_2}}}}." Do not put a marker at the paragraph start or
after another marker. The marker is an internal binding, not visible prose.

background is optional. When reliable, describe the overall environment after
the principal foreground subjects are removed, using only phrase and
grounding_prompt. Return background only when one stable environment persists
through most of the clip. When the video contains a major scene transition
between different environments, return background=null. Do not repeat the main
foreground subject. Neither background text field must occur verbatim in the
instruction_template. When background is non-null, place {{{{background}}}}
exactly once after a clear environment mention, using the same ASCII-space and
pre-punctuation placement rule. When background is null, do not include that
marker. Return JSON only."""


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


def _clean_instruction_template(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalized_phrase(value: str) -> str:
    return " ".join(value.casefold().split()).strip(_PHRASE_EDGE_PUNCTUATION)


def _english_word_count(value: str) -> int:
    return len(_ENGLISH_WORD.findall(value))


def _entity_text_issues(raw_entities: object) -> list[ValidationIssue]:
    if not isinstance(raw_entities, list):
        return []
    issues: list[ValidationIssue] = []
    for index, candidate in enumerate(raw_entities):
        if not isinstance(candidate, dict):
            continue
        phrase = _clean_text(candidate.get("phrase"))
        if phrase is not None and _english_word_count(phrase) > 12:
            issues.append(
                ValidationIssue(
                    code="entity_phrase_too_long",
                    field=f"entities.{index}.phrase",
                    message="entity phrase must not exceed 12 English words",
                )
            )
        grounding_prompt = _clean_text(candidate.get("grounding_prompt"))
        if grounding_prompt is None:
            continue
        if _english_word_count(grounding_prompt) > 24:
            issues.append(
                ValidationIssue(
                    code="grounding_prompt_too_long",
                    field=f"entities.{index}.grounding_prompt",
                    message="grounding prompt must not exceed 24 English words",
                )
            )
        action_match = _TRANSIENT_GROUNDING_ACTION.search(grounding_prompt)
        if action_match is not None:
            issues.append(
                ValidationIssue(
                    code="transient_action_in_grounding_prompt",
                    field=f"entities.{index}.grounding_prompt",
                    message=(
                        "grounding prompt contains a transient action: "
                        f"{action_match.group(0)}"
                    ),
                )
            )
    return issues


def _marker_position_issue(
    *,
    template: str,
    start: int,
    end: int,
    marker: str,
) -> ValidationIssue | None:
    prefix = template[:start]
    suffix = template[end:]
    if (
        start == 0
        or prefix[-1:] != " "
        or prefix[-2:-1] == " "
        or prefix.rstrip().endswith("}}")
        or suffix
        and not (suffix[0].isspace() or suffix[0] in _PHRASE_EDGE_PUNCTUATION)
    ):
        return ValidationIssue(
            code="invalid_marker_position",
            field="instruction_template",
            message=(
                f"{marker} must follow a visible mention after one ASCII space"
            ),
        )
    return None


def _annotation_marker_issues(
    *,
    template: str,
    raw_entities: object,
    raw_background: object,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    entity_count = len(raw_entities) if isinstance(raw_entities, list) else 0
    entity_matches = list(_ENTITY_MARKER.finditer(template))
    entity_indexes = [int(match.group(1)) for match in entity_matches]

    for expected_index in range(1, entity_count + 1):
        count = entity_indexes.count(expected_index)
        if count == 0:
            issues.append(
                ValidationIssue(
                    code="missing_entity_marker",
                    field="instruction_template",
                    message=f"instruction_template is missing {{{{entity_{expected_index}}}}}",
                )
            )
        elif count > 1:
            issues.append(
                ValidationIssue(
                    code="duplicate_entity_marker",
                    field="instruction_template",
                    message=f"{{{{entity_{expected_index}}}}} must appear exactly once",
                )
            )
    for index in sorted(set(entity_indexes)):
        if index > entity_count:
            issues.append(
                ValidationIssue(
                    code="unexpected_entity_marker",
                    field="instruction_template",
                    message=f"unexpected entity marker: {{{{entity_{index}}}}}",
                )
            )

    background_count = len(_BACKGROUND_MARKER.findall(template))
    if raw_background is not None and background_count == 0:
        issues.append(
            ValidationIssue(
                code="background_marker_missing",
                field="instruction_template",
                message="non-null background requires {{background}} exactly once",
            )
        )
    elif raw_background is not None and background_count > 1:
        issues.append(
            ValidationIssue(
                code="duplicate_background_marker",
                field="instruction_template",
                message="{{background}} must appear exactly once",
            )
        )
    elif raw_background is None and background_count:
        issues.append(
            ValidationIssue(
                code="unexpected_background_marker",
                field="instruction_template",
                message="null background forbids {{background}}",
            )
        )

    recognized_spans = {
        match.span() for match in [*entity_matches, *_BACKGROUND_MARKER.finditer(template)]
    }
    for match in _BRACED_MARKER.finditer(template):
        if match.span() in recognized_spans:
            continue
        marker = match.group(0)
        code = (
            "invalid_annotation_image_marker"
            if match.group(1).casefold().startswith("image_")
            else "invalid_entity_marker"
        )
        issues.append(
            ValidationIssue(
                code=code,
                field="instruction_template",
                message=f"invalid annotation marker: {marker}",
            )
        )
    remainder = _BRACED_MARKER.sub("", template)
    if "{{" in remainder or "}}" in remainder:
        issues.append(
            ValidationIssue(
                code="invalid_entity_marker",
                field="instruction_template",
                message="instruction_template contains malformed marker braces",
            )
        )
    if _ANNOTATION_IMAGE_MARKER.search(template):
        issues.append(
            ValidationIssue(
                code="invalid_annotation_image_marker",
                field="instruction_template",
                message="annotation must not contain image_N or <Image N>",
            )
        )
    if _REFERENCE_TOKEN.search(template):
        issues.append(
            ValidationIssue(
                code="invalid_annotation_reference_token",
                field="instruction_template",
                message="annotation must not contain <ref_...> tokens",
            )
        )

    for match in [*entity_matches, *_BACKGROUND_MARKER.finditer(template)]:
        position_issue = _marker_position_issue(
            template=template,
            start=match.start(),
            end=match.end(),
            marker=match.group(0),
        )
        if position_issue is not None:
            issues.append(position_issue)
    return issues


def _sanitize_entity_candidates_with_indices(
    raw_entities: object,
) -> tuple[list[AnnotationEntity], tuple[str, ...], tuple[int, ...]]:
    if not isinstance(raw_entities, list):
        return [], ("dropped_invalid_entities_collection",), ()

    accepted: list[tuple[int, str, str, str]] = []
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
        if len(accepted) == MAX_ANNOTATION_ENTITIES:
            warnings.append(
                f"truncated_entity_candidates:{MAX_ANNOTATION_ENTITIES}"
            )
            break
        accepted.append((index + 1, normalized_type, phrase, grounding_prompt))

    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=reference_type,
            phrase=phrase,
            grounding_prompt=grounding_prompt,
        )
        for index, (
            _source_index,
            reference_type,
            phrase,
            grounding_prompt,
        ) in enumerate(
            accepted,
            start=1,
        )
    ]
    return (
        entities,
        tuple(warnings),
        tuple(source_index for source_index, *_ in accepted),
    )


def sanitize_entity_candidates(
    raw_entities: object,
) -> tuple[list[AnnotationEntity], tuple[str, ...]]:
    entities, warnings, _ = _sanitize_entity_candidates_with_indices(
        raw_entities
    )
    return entities, warnings


def _rewrite_markers_after_sanitization(
    template: str,
    *,
    accepted_source_indexes: tuple[int, ...],
    keep_background: bool,
) -> str:
    final_indexes = {
        source_index: final_index
        for final_index, source_index in enumerate(
            accepted_source_indexes,
            start=1,
        )
    }

    def replace_entity(match: re.Match[str]) -> str:
        source_index = int(match.group(1))
        final_index = final_indexes.get(source_index)
        return "" if final_index is None else f" {{{{entity_{final_index}}}}}"

    rewritten = re.sub(
        r" \{\{entity_([1-9]\d*)\}\}",
        replace_entity,
        template,
    )
    if not keep_background:
        rewritten = rewritten.replace(" {{background}}", "")
    return rewritten


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
    template = _clean_instruction_template(
        raw_payload.get("instruction_template")
    )
    issues: list[ValidationIssue] = []
    if template is None:
        issues.append(
            ValidationIssue(
                code="empty_instruction_template",
                field="instruction_template",
                message="instruction_template must be a non-empty string",
            )
        )
    else:
        plain_text = plain_instruction_text(template)
        if _english_word_count(plain_text) > 120:
            issues.append(
                ValidationIssue(
                    code="caption_too_long",
                    field="instruction_template",
                    message=(
                        "instruction_template must not exceed 120 English "
                        "content words"
                    ),
                )
            )
        inference_match = _UNSUPPORTED_CAPTION_INFERENCE.search(plain_text)
        if inference_match is not None:
            issues.append(
                ValidationIssue(
                    code="unsupported_caption_inference",
                    field="instruction_template",
                    message=(
                        "instruction_template contains unsupported inference "
                        "language: "
                        f"{inference_match.group(0)}"
                    ),
                )
            )
        issues.extend(
            _annotation_marker_issues(
                template=template,
                raw_entities=raw_payload.get("entities", []),
                raw_background=raw_payload.get("background"),
            )
        )
    issues.extend(_entity_text_issues(raw_payload.get("entities", [])))
    if issues:
        return None, issues, ()

    entities, entity_warnings, accepted_source_indexes = (
        _sanitize_entity_candidates_with_indices(
        raw_payload.get("entities", [])
        )
    )
    background, background_warnings = sanitize_background(
        raw_payload.get("background")
    )
    template = _rewrite_markers_after_sanitization(
        template,
        accepted_source_indexes=accepted_source_indexes,
        keep_background=background is not None,
    )
    return (
        AnnotationState(
            status="ready",
            instruction_template=template,
            t2v_caption="",
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
        "tokens, image placeholders, or fields outside the schema.\n"
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
        profile_context = get_model_profile_context()
        retry_index = profile_context.retry_index
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
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_annotation",
                            "strict": True,
                            "schema": RawAnnotationPayload.model_json_schema(),
                        },
                    },
                ),
                component="qwen_annotation",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={"response_format": "json_schema"},
            )
        except BadRequestError:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                ),
                component="qwen_annotation",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={"response_format": "json_object"},
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
                with model_profile_context(retry_index=attempt):
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
