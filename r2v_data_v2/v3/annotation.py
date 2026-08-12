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
from r2v_data_v2.v3.config import (
    DEFAULT_MAX_ANNOTATION_ENTITIES,
    DENSE_MAX_ANNOTATION_ENTITIES,
    QwenAnnotationConfig,
    V3Config,
)
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
    render_annotation_plain_text,
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
_ENTITY_COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
)
_ENTITY_LIMIT_WORD = _ENTITY_COUNT_WORDS[DEFAULT_MAX_ANNOTATION_ENTITIES]
_DENSE_GENERIC_OBJECT_HEAD = re.compile(
    r"^(?:(?:a|an|the)\s+)?"
    r"(?:(?:unknown|unidentified|floating|black|dark|large|small|strange|awkward)\s+)*"
    r"(?:object|thing|item)s?"
    r"(?:\s+(?:with|without|near|beside|in|on|at|of)\b.*)?$",
    flags=re.IGNORECASE,
)
_PREFERRED_ENTITY_PHRASE_WORD_LIMIT = 12
_HARD_ENTITY_PHRASE_WORD_LIMIT = 18
_PREFERRED_INSTRUCTION_WORD_LIMIT = 120
_TARGET_INSTRUCTION_WORD_LIMIT = 180
_HARD_INSTRUCTION_WORD_LIMIT = 220

SYSTEM_PROMPT = f"""You annotate a complete video for a V3 training-data pipeline.

Return exactly one JSON object matching the supplied minimal schema. Output only
entities, background, and instruction_template, in that generation order. Do
not output t2v_caption. Do
not output relations, entity_id,
category, salience, genericity, name_evidence, localization_scope, scene_role,
representation_mode, visual_scope, separability, selection_reason, reference
tokens, image_N placeholders, rendered <Image N> labels, or any additional
ontology fields.

STEP 1: Select the reference entity proposals.
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
candidate for binding, review, and deterministic placeholder substitution in 3
to 10 words. Keep phrase at or below 12 words as the target and never exceed the
absolute maximum of 18 words. phrase is a stable, natural English noun phrase
rather than an action. Include an article or determiner when
natural, such as "a bald monk in a light brown robe", "an ornate wooden altar",
or "the three scuba divers". Do not add trailing punctuation, internal markers,
or transient actions. The phrase need not occur verbatim in instruction_template
because its placeholder represents the phrase itself. It must sufficiently
distinguish the target: prefer "a man in a light gray military uniform" over
"military officer speaking at podium". grounding_prompt describes stable visible
appearance and location in 6 to 18 words and must not exceed 24 words. Do not
include transient actions or enumerate every clothing detail. A stable seated
or standing pose is allowed only when needed to distinguish the target for
SAM3. grounding_prompt need not occur in instruction_template. Both fields must be
non-empty and must not contain reference tokens.

Do not output an entity proposal unless you can place its corresponding
placeholder exactly once in instruction_template. Use the placeholder at the
first clear natural mention of that entity. Later mentions must use pronouns or
ordinary natural-language references such as "the man" or "the diver". Never
repeat the same entity placeholder later in the paragraph.

STEP 2: Decide whether one stable background reference is useful.
background is optional. When reliable, describe the overall environment after
the principal foreground subjects are removed, using only phrase and
grounding_prompt. Return background only when one stable environment persists
through most of the clip. When the video contains a major scene transition
between different environments, return background=null. Do not repeat the main
foreground subject. Neither background text field must occur verbatim in the
instruction_template. Do not output a non-null background unless you can place
{{{{background}}}} exactly once in instruction_template. Place it at the first
natural environment mention and never repeat it later in the paragraph.

STEP 3: After the entity and background proposals are fixed, write
instruction_template as one coherent English paragraph that begins directly
with visible content and describes the target video naturally. It is not an
imperative request: do not write "Use the reference image", "Generate",
"Create", or a references section. Describe actions and shot changes in
chronological order. Include visible subject appearance, action, scene,
composition, camera behavior, and lighting without repetition. Prefer roughly
60 to 120 English content words. Longer descriptions are allowed when needed for
complete chronology, multiple entities, or shot changes. Target no more than
180 English content words after every placeholder is replaced by its phrase.
The absolute validation ceiling is 220 words, not a recommended length.
Describe only directly visible content. Do not infer identity, weather,
emotion, allegiance, intent,
mental state, sound, dialogue, or event causes. Describe visible motion directly
without assigning an unseen cause. Write "branches sway slightly" instead of
claiming that wind causes movement. Do not use hedging or causal inference
wording such as breeze, wind-induced, suggesting, indicating, possibly,
probably, or likely. Do not identify a person as an enemy, ally, criminal,
victim, officer, or another role unless that role is explicitly supported by
source metadata. For statues and depicted figures, describe visible facial
geometry and pose without inferring determination, resolve, triumph, fear, or
effort. Do not write "the video shows". Do not include <ref_...> tokens or
image-number instruction labels.

Insert exactly one internal placeholder for every entity in the same array
order: entities[0] MUST use {{{{entity_1}}}} exactly once, entities[1] MUST use
{{{{entity_2}}}} exactly once, and so on through at most
{{{{entity_5}}}}. Every listed entity must have its placeholder. Each
{{{{entity_N}}}} placeholder represents that entity's complete phrase and must
replace the noun phrase itself. Write "{{{{entity_1}}}} kneels" rather than "a
monk {{{{entity_1}}}} kneels". Treat the placeholder as though the phrase were
already written there: when phrase is "a white boat", write
"{{{{entity_1}}}} moves slowly", never "A {{{{entity_1}}}} moves slowly". A
placeholder may appear at the paragraph start or beside punctuation, but must
not be embedded inside another word. Every placeholder must appear exactly
once; duplicate entity or background placeholders are forbidden. If background
is non-null,
{{{{background}}}} MUST appear exactly once and represents the complete
background phrase in the same way. When background is null, do not include that
placeholder. Return JSON only."""

COMPOSITION_BALANCED_ENTITY_SELECTION_PROMPT = """After selecting the clearest
primary reference candidate, perform one brief complementary scan. When both a
stable independently referenceable subject and a stable independently
referenceable non-human object are clearly visible and trackable, prefer at
least one candidate of each type before extra candidates of the same type. This
is a preference, not a quota; never invent a candidate to satisfy it. A useful
object may be a distinct garment, tool, container, vehicle, piece of furniture,
device, food item, or other stable physical object with independent reference
value. Never classify a person, animal, body part, depicted person, photograph,
poster, screen content, or a whole person-plus-object composite as an object."""

_DEFAULT_STEP_1_SELECTION = f"""STEP 1: Select the reference entity proposals.
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
name ontology."""

REFERENCE_DENSE_STEP_1_SELECTION_PROMPT = """STEP 1: Select the reference entity proposals.
Return at most eight stable, visually distinct foreground entities that SAM3 can
localize and track and that can be independently useful as reference images.
Do not stop after finding one strong primary entity. After selecting the clearest
entity, inspect the video once for additional clearly visible, stable, trackable
entities with independent control value. A larger or more salient person does
not automatically exclude smaller useful objects.

Use subject for exactly one person, animal, or character.
Animals are subjects, never objects. Body parts are not independent objects.
Use object only for one
concrete, discrete, physical foreground object with independent reference or
control value. Amorphous materials, liquids, sauces, smoke, shadows, lighting,
and similar non-discrete content are not objects. Static environmental structure
such as buildings, room architecture, bridges, trees, landscape elements, and
similar scene structure should normally remain in the scene or background
description. Content inside a screen, painting, photograph, poster,
visualization, diagram, animation, or other depicted representation must not
become a physical object reference merely because it is visually distinct.

Worn or attached objects may be selected when they are visually distinct,
stable across the clip, specifically nameable, and independently useful. These
may include a hat, glasses, bag, jacket, shirt, distinctive footwear, tool,
weapon, handheld device, or similar stable physical object. This is not a quota:
never invent entities, and return fewer entities when evidence is weak. Do not
split a subject into arbitrary tiny details such as buttons, shoelaces, fingers,
ears, tiny decorations, indistinct fragments, or anything that cannot be
reliably localized.

Discrete foods and food containers may remain valid objects when they have a
recognizable bounded form, such as a lobster, fish, bowl, or frying pan.
A plate of meatballs, loaf of bread, or pot of food can also remain valid.
Do not promote amorphous sauce, steam, oil puddles, or similar substances into
object references.

An object phrase must name a concrete recognizable physical entity. Do not use
generic phrases such as object, thing, item, unknown object, awkward object,
unidentified object, large black object, or a large black awkward object. Do not
select environmental regions, distributed or transient effects, depicted
content, brief or blurred content, or untrackable entities. These examples guide
selection only and are not an object ontology."""

_DEFAULT_PLACEHOLDER_LIMIT = """and so on through at most
{{entity_5}}."""
_DENSE_PLACEHOLDER_LIMIT = """and so on through at most
{{entity_8}}."""


def _reference_dense_system_prompt() -> str:
    if SYSTEM_PROMPT.count(_DEFAULT_STEP_1_SELECTION) != 1:
        raise RuntimeError("default annotation STEP 1 prompt boundary changed")
    prompt = SYSTEM_PROMPT.replace(
        _DEFAULT_STEP_1_SELECTION,
        REFERENCE_DENSE_STEP_1_SELECTION_PROMPT,
        1,
    )
    if prompt.count(_DEFAULT_PLACEHOLDER_LIMIT) != 1:
        raise RuntimeError("default annotation placeholder limit changed")
    return prompt.replace(
        _DEFAULT_PLACEHOLDER_LIMIT,
        _DENSE_PLACEHOLDER_LIMIT,
        1,
    )


def annotation_system_prompt(config: QwenAnnotationConfig) -> str:
    if config.entity_selection_mode == "default":
        return SYSTEM_PROMPT
    if config.entity_selection_mode == "composition_balanced_v1":
        return f"{SYSTEM_PROMPT}\n\n{COMPOSITION_BALANCED_ENTITY_SELECTION_PROMPT}"
    if config.entity_selection_mode == "reference_dense_v1":
        return _reference_dense_system_prompt()
    raise ValueError(
        f"unsupported annotation entity selection mode: "
        f"{config.entity_selection_mode}"
    )


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
        if (
            phrase is not None
            and _english_word_count(phrase)
            > _HARD_ENTITY_PHRASE_WORD_LIMIT
        ):
            issues.append(
                ValidationIssue(
                    code="entity_phrase_too_long",
                    field=f"entities.{index}.phrase",
                    message="entity phrase must not exceed 18 English words",
                )
            )
    return issues


def _safe_entity_fallback_phrases(
    raw_entities: object,
) -> dict[int, str]:
    if not isinstance(raw_entities, list):
        return {}
    phrases: dict[int, str] = {}
    for source_index, candidate in enumerate(raw_entities, start=1):
        if not isinstance(candidate, dict):
            continue
        phrase = _clean_text(
            candidate.get("phrase"),
            trim_phrase_punctuation=True,
        )
        if (
            phrase is None
            or _REFERENCE_TOKEN.search(phrase)
            or "{{" in phrase
            or "}}" in phrase
        ):
            continue
        phrases[source_index] = phrase
    return phrases


def _safe_background_fallback_phrase(raw_background: object) -> str | None:
    if not isinstance(raw_background, dict):
        return None
    phrase = _clean_text(
        raw_background.get("phrase"),
        trim_phrase_punctuation=True,
    )
    if (
        phrase is None
        or _REFERENCE_TOKEN.search(phrase)
        or "{{" in phrase
        or "}}" in phrase
    ):
        return None
    return phrase


def _placeholder_is_embedded(
    *,
    template: str,
    start: int,
    end: int,
) -> bool:
    before = template[start - 1] if start else ""
    after = template[end] if end < len(template) else ""
    return bool(
        before and (before.isalnum() or before == "_")
        or after and (after.isalnum() or after == "_")
    )


def _annotation_marker_hard_issues(
    *,
    template: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    entity_matches = list(_ENTITY_MARKER.finditer(template))
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
    return issues


@dataclass(frozen=True)
class _MarkerEligibility:
    entity_source_indexes: tuple[int, ...]
    background_eligible: bool
    warnings: tuple[str, ...]


def _inspect_recognizable_markers(
    *,
    template: str,
    raw_entities: object,
    raw_background: object,
) -> _MarkerEligibility:
    entity_count = len(raw_entities) if isinstance(raw_entities, list) else 0
    matches_by_index: dict[int, list[re.Match[str]]] = {}
    for match in _ENTITY_MARKER.finditer(template):
        matches_by_index.setdefault(int(match.group(1)), []).append(match)

    eligible_indexes: list[int] = []
    warnings: list[str] = []
    for source_index in range(1, entity_count + 1):
        matches = matches_by_index.get(source_index, [])
        if not matches:
            warnings.append(f"dropped_entity_missing_marker:{source_index}")
            continue
        if any(
            _placeholder_is_embedded(
                template=template,
                start=match.start(),
                end=match.end(),
            )
            for match in matches
        ):
            warnings.append(
                f"dropped_entity_embedded_placeholder:{source_index}"
            )
            continue
        eligible_indexes.append(source_index)

    for source_index in sorted(matches_by_index):
        if source_index > entity_count:
            warnings.append(
                f"removed_unexpected_entity_marker:{source_index}"
            )

    background_matches = list(_BACKGROUND_MARKER.finditer(template))
    background_eligible = False
    if raw_background is None:
        if background_matches:
            warnings.append("removed_unexpected_background_marker")
    elif not background_matches:
        warnings.append("dropped_background_missing_marker")
    elif any(
        _placeholder_is_embedded(
            template=template,
            start=match.start(),
            end=match.end(),
        )
        for match in background_matches
    ):
        warnings.append("dropped_background_embedded_placeholder")
    else:
        background_eligible = True

    return _MarkerEligibility(
        entity_source_indexes=tuple(eligible_indexes),
        background_eligible=background_eligible,
        warnings=tuple(warnings),
    )


def _sanitize_entity_candidates_with_indices(
    raw_entities: object,
    *,
    max_entities: int = DEFAULT_MAX_ANNOTATION_ENTITIES,
    reject_generic_object_phrases: bool = False,
) -> tuple[list[AnnotationEntity], tuple[str, ...], tuple[int, ...]]:
    if not 1 <= max_entities <= MAX_ANNOTATION_ENTITIES:
        raise ValueError("max_entities is outside the annotation schema capacity")
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
        source_index = index + 1
        if _english_word_count(grounding_prompt) > 24:
            warnings.append(
                f"dropped_entity_grounding_prompt_too_long:{source_index}"
            )
            continue
        if _TRANSIENT_GROUNDING_ACTION.search(grounding_prompt) is not None:
            warnings.append(
                f"dropped_entity_transient_grounding_action:{source_index}"
            )
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
        if (
            reject_generic_object_phrases
            and normalized_type == "object"
            and _DENSE_GENERIC_OBJECT_HEAD.fullmatch(phrase_key) is not None
        ):
            warnings.append(f"dropped_generic_object_phrase:{index}")
            continue
        phrase_word_count = _english_word_count(phrase)
        if phrase_word_count > _PREFERRED_ENTITY_PHRASE_WORD_LIMIT:
            warnings.append(
                "entity_phrase_over_preferred_length:"
                f"{source_index}:{phrase_word_count}"
            )
        seen_phrases.add(phrase_key)
        if len(accepted) == max_entities:
            warnings.append(
                f"truncated_entity_candidates:{max_entities}"
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
    *,
    max_entities: int = DEFAULT_MAX_ANNOTATION_ENTITIES,
    reject_generic_object_phrases: bool = False,
) -> tuple[list[AnnotationEntity], tuple[str, ...]]:
    entities, warnings, _ = _sanitize_entity_candidates_with_indices(
        raw_entities,
        max_entities=max_entities,
        reject_generic_object_phrases=reject_generic_object_phrases,
    )
    return entities, warnings


def _rewrite_placeholders_after_sanitization(
    template: str,
    *,
    accepted_source_indexes: tuple[int, ...],
    fallback_phrases: dict[int, str],
    keep_background: bool,
    background_fallback_phrase: str | None,
) -> str:
    final_indexes = {
        source_index: final_index
        for final_index, source_index in enumerate(
            accepted_source_indexes,
            start=1,
        )
    }

    replacements: list[tuple[int, int, str]] = []
    retained_placeholder_indexes: set[int] = set()
    for match in _ENTITY_MARKER.finditer(template):
        source_index = int(match.group(1))
        final_index = final_indexes.get(source_index)
        if (
            final_index is not None
            and source_index not in retained_placeholder_indexes
        ):
            replacement = f"{{{{entity_{final_index}}}}}"
            retained_placeholder_indexes.add(source_index)
        else:
            replacement = fallback_phrases.get(source_index, "")
        replacements.append((match.start(), match.end(), replacement))
    retained_background_placeholder = False
    for match in _BACKGROUND_MARKER.finditer(template):
        if keep_background and not retained_background_placeholder:
            replacement = "{{background}}"
            retained_background_placeholder = True
        else:
            replacement = background_fallback_phrase or ""
        replacements.append((match.start(), match.end(), replacement))

    rewritten = template
    for start, end, replacement in sorted(replacements, reverse=True):
        if _placeholder_is_embedded(
            template=rewritten,
            start=start,
            end=end,
        ):
            replacement = f" {replacement} "
        rewritten = f"{rewritten[:start]}{replacement}{rewritten[end:]}"
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    rewritten = re.sub(r" +([,.;:!?])", r"\1", rewritten)
    return rewritten.strip()


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
    if _english_word_count(grounding_prompt) > 24:
        return None, ("dropped_background_grounding_prompt_too_long",)
    if _TRANSIENT_GROUNDING_ACTION.search(grounding_prompt) is not None:
        return None, ("dropped_background_transient_grounding_action",)
    return (
        BackgroundAnnotation(
            phrase=phrase,
            grounding_prompt=grounding_prompt,
        ),
        (),
    )


def sanitize_annotation_payload(
    raw_payload: dict[str, object],
    *,
    max_entities: int = DEFAULT_MAX_ANNOTATION_ENTITIES,
    reject_generic_object_phrases: bool = False,
) -> tuple[AnnotationState | None, list[ValidationIssue], tuple[str, ...]]:
    template = _clean_instruction_template(
        raw_payload.get("instruction_template")
    )
    issues: list[ValidationIssue] = []
    warnings: list[str] = []
    if template is None:
        issues.append(
            ValidationIssue(
                code="empty_instruction_template",
                field="instruction_template",
                message="instruction_template must be a non-empty string",
            )
        )
    else:
        issues.extend(
            _annotation_marker_hard_issues(
                template=template,
            )
        )
    issues.extend(_entity_text_issues(raw_payload.get("entities", [])))
    if issues:
        return None, issues, ()

    marker_eligibility = _inspect_recognizable_markers(
        template=template,
        raw_entities=raw_payload.get("entities", []),
        raw_background=raw_payload.get("background"),
    )
    fallback_phrases = _safe_entity_fallback_phrases(
        raw_payload.get("entities", [])
    )
    entities, entity_warnings, accepted_source_indexes = (
        _sanitize_entity_candidates_with_indices(
            raw_payload.get("entities", []),
            max_entities=max_entities,
            reject_generic_object_phrases=reject_generic_object_phrases,
        )
    )
    marker_eligible_indexes = set(marker_eligibility.entity_source_indexes)
    accepted_pairs = [
        (source_index, entity)
        for source_index, entity in zip(
            accepted_source_indexes,
            entities,
            strict=True,
        )
        if source_index in marker_eligible_indexes
    ]
    accepted_source_indexes = tuple(
        source_index for source_index, _entity in accepted_pairs
    )
    entities = [
        entity.model_copy(update={"entity_id": f"e{index}"})
        for index, (_source_index, entity) in enumerate(
            accepted_pairs,
            start=1,
        )
    ]
    background, background_warnings = sanitize_background(
        raw_payload.get("background")
    )
    if not marker_eligibility.background_eligible:
        background = None
    template = _rewrite_placeholders_after_sanitization(
        template,
        accepted_source_indexes=accepted_source_indexes,
        fallback_phrases=fallback_phrases,
        keep_background=background is not None,
        background_fallback_phrase=_safe_background_fallback_phrase(
            raw_payload.get("background")
        ),
    )
    if not template:
        return None, [
            ValidationIssue(
                code="empty_instruction_template",
                field="instruction_template",
                message="instruction_template is empty after sanitization",
            )
        ], ()
    annotation = AnnotationState(
        status="ready",
        instruction_template=template,
        t2v_caption="",
        entities=entities,
        background=background,
    )
    plain_text = render_annotation_plain_text(
        annotation.instruction_template,
        annotation.entities,
        annotation.background,
    )
    word_count = _english_word_count(plain_text)
    if word_count > _HARD_INSTRUCTION_WORD_LIMIT:
        issues.append(
            ValidationIssue(
                code="instruction_template_too_long",
                field="instruction_template",
                message=(
                    "rendered instruction_template must not exceed 220 "
                    "English content words"
                ),
            )
        )
    elif word_count > _TARGET_INSTRUCTION_WORD_LIMIT:
        warnings.append(
            f"instruction_template_over_target_length:{word_count}"
        )
    elif word_count > _PREFERRED_INSTRUCTION_WORD_LIMIT:
        warnings.append(
            f"instruction_template_over_preferred_length:{word_count}"
        )
    inference_match = _UNSUPPORTED_CAPTION_INFERENCE.search(plain_text)
    if inference_match is not None:
        issues.append(
            ValidationIssue(
                code="unsupported_caption_inference",
                field="instruction_template",
                message=(
                    "rendered instruction_template contains unsupported "
                    f"inference language: {inference_match.group(0)}"
                ),
            )
        )
    if issues:
        return None, issues, ()
    return (
        annotation,
        [],
        (
            *warnings,
            *marker_eligibility.warnings,
            *entity_warnings,
            *background_warnings,
        ),
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
            {"role": "system", "content": annotation_system_prompt(self.config)},
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
            dense_mode = self.config.entity_selection_mode == "reference_dense_v1"
            annotation, issues, warnings = sanitize_annotation_payload(
                raw_payload,
                max_entities=(
                    DENSE_MAX_ANNOTATION_ENTITIES
                    if dense_mode
                    else DEFAULT_MAX_ANNOTATION_ENTITIES
                ),
                reject_generic_object_phrases=dense_mode,
            )
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
