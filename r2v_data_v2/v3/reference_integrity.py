from __future__ import annotations

import base64
import hashlib
import io
import math
import re
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageFilter

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import QwenServiceConfig, V3Config
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.profiling import profiled_openai_call
from r2v_data_v2.v3.schemas import (
    EntityReferenceState,
    PairingState,
    ReferenceEditState,
    ReferenceIntegrityEntityState,
    ReferenceIntegrityReview,
    ReferenceIntegrityState,
    ReferencesState,
    ReferenceTopologyDiagnostics,
    SourceBboxFallbackReview,
)
from r2v_data_v2.v3.storage import RunStorage

SYSTEM_PROMPT = """You are the final integrity reviewer for one entity reference.
Compare the highlighted target in the source context with the final reference.
First judge whether the annotated entity itself obeys V3 reference semantics.
Set reference_entity_semantically_valid to false when a living animal or
creature is labeled as an object, including a dog, a living clam, or a fish,
crab, lobster, spider, or turtle depicted as a creature. Clearly cooked or
prepared culinary food may remain a valid object, such as a cooked whole fish
in a wok or a cooked lobster dish; do not reject it solely because its noun
originates from an animal. Amorphous sauce, liquid, smoke, steam, fog, light, or
similar unbounded material is not an object by itself. A static scene structure
such as a cathedral, building, bridge, or tree is not an object reference.
Depicted, screen, painting, poster, photograph, animation, or visualization
content is not a physical reference entity when the annotation denotes only the
represented content. A physical carrier such as the actual screen, framed
painting, or poster may still be a valid object when that carrier is the entity.
The final reference must denote the same complete entity as the annotation
phrase. Do not reinterpret the target as a convenient sub-entity. When a
container-and-contents relationship is part of the annotated entity identity,
the reference must preserve that defining relationship; recognizable contents
alone are insufficient. Identity-defining nouns and structure must remain,
although modifiers describing only a transient state need not all remain
visually obvious. For example, "a clay pot of stew" must reject when only stew
remains, and "a bowl of noodles" must reject when only noodles remain. For the
object "a camera", the camera itself must remain. For the subject "a man in a
white t-shirt", an unrelated held bowl or chopsticks may disappear if the
stable subject identity remains. A held or transient non-target object may
disappear only when its removal leaves a visually plausible target/reference
surface. The fact that the removed item is transient, non-identity-defining, or
unrelated to the target does not excuse an artificial removal artifact. Reject a
white or transparent circular silhouette in a hand; a bottle-, cup-, fruit-,
spoon-, or tool-shaped blank cavity; an irregular white blob near a hand or
mouth; a solid-color placeholder where the source contained another object; a
conspicuous erased-object outline; a blank patch cutting through clothing,
skin, body surface, or another identity/reference-bearing region; or any
obviously artificial missing or reconstructed surface caused by deleting the
non-target item. Set no_severe_reference_artifact to false for these artifacts.
Also set no_unnatural_holes_or_surface_loss to false when the blank region is an
unnatural missing or replaced surface. The semantic and identity fields should
still describe the otherwise-correct target accurately.
REJECT an orange removed from a person's hand leaving a white circular
silhouette; a spoon or food removed near a child's mouth leaving an irregular
white blob; a bottle removed from a person's torso or hand leaving a
bottle-shaped white cavity; or a held item replaced by an unnatural white sphere
or placeholder. The orange, bottle, spoon, food, or other held item itself does
not have to remain; the failure is the artificial residue left by its removal.
Do not reject merely because a transient item disappeared. Clean removal may be
accepted when the hand, clothing, body, and background surfaces are visually
plausible and the target remains structurally and semantically correct. This
includes a cleanly removed bowl, disappearing chopsticks with no hole or
placeholder, or an unrelated nearby object that disappears cleanly. Naturally
white clothing or objects, white backgrounds, highlights, teeth, sclera, and
paper are not artifacts merely because they are white. Judge unnatural
source-to-reference alteration, not raw white-pixel presence.
For a subject, require enough stable identity-bearing appearance for the stated
scope, but do not require a visible face in every valid back view, masked person,
character, or animal. For a human subject, when the highlighted source context
visibly contains the person's head or head region, the final reference must
preserve a recognizable head region together with stable person appearance. A
visible face is not required. Accept a person viewed from behind with the head
present, a helmeted or masked person with the head present, and a side-profile
person with the head and upper body present. Reject a chef reference containing
only coat and arms when the source shows the head, a person reference cropped
completely below the neck, or a clothing-only fragment labeled as a subject.
Set preserves_primary_identity_region to false for these human head-region
failures. For a non-human subject or object, evaluate that field using its
existing identity-bearing region without imposing human anatomy.
Reject a torso, arms, legs, or clothing fragment when the source contains
substantially more useful identity evidence and the final image loses it. Apply
this human head-region rule only to human subjects; do not introduce a face
requirement or new rejection behavior for non-human subjects or objects. For an
object, require its recognizable structural core. Reject major missing surfaces,
large unnatural holes, disconnected remnants, identity-changing completion
artifacts, severe destructive truncation, or an unrelated dominant entity.
Set no_severe_reference_artifact to false for a conspicuous artificial defect
that makes the reference unusable even if identity remains recognizable. This
includes a large white or transparent erased-object-shaped cavity through the
entity, a severe blank patch or edit scar, a large unnatural missing surface,
or an obvious generation artifact that would teach incorrect appearance. For
example, reject a red-top woman whose removed held bottle leaves a large white
bottle-shaped hole through her reference. The bottle need not remain for her
identity; the artificial cavity itself is the failure.
Legitimate source-matching cutouts, handles, wheels, scissors, brackets, frames,
and truss structures are not defects merely because they contain holes.
Make verdict accept if and only if every boolean is true. Return exactly one
compact JSON object and follow the supplied schema key order. Reason must be one
concise sentence. Emit reason before verdict, make verdict the final key, and
close the JSON object immediately after verdict. Do not emit trailing whitespace,
markdown, or explanation."""

SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT = """You are reviewing a conservative raw-source
bbox alternative for an entity reference. Image 1 is source context with the target
highlighted. Image 2 is the current reference. Image 3 is a raw RGB bbox crop from
the exact selected source frame and target-mask bbox. Decide whether Image 3 is a
substantially better independent conditioning reference without changing the target
identity. A small held bottle, cup, fruit, tool, hand, or limited local background
may remain when the annotated target stays clearly dominant.
Reject another person or major body from another person, another large animal,
vehicle, or product, a large unrelated foreground object, dominating scene content
or text, a target that is too small, any severe artifact, or any ambiguity about the
target. Do not decide from an object-name blacklist alone; judge visual dominance
and conditioning ambiguity. Set verdict to accept if and only if every strict
boolean is true. Return exactly one compact JSON object and follow the supplied
schema key order. Reason must be one concise sentence. Emit reason before verdict,
make verdict the final key, and close the JSON object immediately after verdict.
Do not emit trailing whitespace, markdown, or explanation."""

SourceBboxFallbackTrigger = Literal[
    "artifact_review_reject",
    "topology_alpha_hole_upgrade",
]

_OBJECT_CREATURE_TERMS = (
    "animal",
    "bear",
    "bird",
    "cat",
    "clam",
    "cow",
    "crab",
    "creature",
    "deer",
    "dog",
    "dolphin",
    "elephant",
    "fish",
    "frog",
    "horse",
    "insect",
    "lion",
    "lizard",
    "lobster",
    "monkey",
    "octopus",
    "rabbit",
    "seal",
    "shark",
    "snake",
    "spider",
    "squid",
    "tiger",
    "turtle",
    "whale",
)
_CULINARY_TERMS = (
    "baked",
    "boiled",
    "cooked",
    "dish",
    "food",
    "fried",
    "grilled",
    "meal",
    "prepared",
    "roasted",
    "stewed",
)
_AMORPHOUS_OBJECT_TERMS = (
    "fire",
    "flame",
    "fog",
    "light",
    "liquid",
    "mist",
    "oil",
    "sauce",
    "shadow",
    "smoke",
    "steam",
    "water",
)
_SCENE_STRUCTURE_OBJECT_TERMS = (
    "building",
    "bridge",
    "cathedral",
    "church",
    "forest",
    "house",
    "mountain",
    "tree",
)
_REPRESENTED_CONTENT_TERMS = (
    "animated",
    "animation",
    "depiction",
    "depicted",
    "diagram",
    "illustration",
    "mural",
    "painting",
    "photo",
    "photograph",
    "poster",
    "screen",
    "screen image",
    "video display",
    "visualization",
)
_SEMANTIC_HEAD_BOUNDARIES = frozenset(
    {
        "against",
        "at",
        "behind",
        "beside",
        "by",
        "carrying",
        "containing",
        "emitting",
        "filled",
        "from",
        "holding",
        "in",
        "inside",
        "lying",
        "near",
        "of",
        "on",
        "outside",
        "over",
        "running",
        "sitting",
        "standing",
        "under",
        "walking",
        "wearing",
        "with",
        "without",
    }
)
_PHYSICAL_REPLICA_TERMS = (
    "figurine",
    "miniature",
    "model",
    "plush",
    "plushie",
    "replica",
    "sculpture",
    "statue",
    "stuffed",
    "toy",
)


def _contains_integrity_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def reference_semantic_risk_reason(
    *,
    reference_type: str,
    phrase: str,
    grounding_prompt: str,
) -> str | None:
    if reference_type != "object":
        return None
    text = " ".join(f"{phrase} {grounding_prompt}".casefold().split())
    creature = any(_contains_integrity_term(text, term) for term in _OBJECT_CREATURE_TERMS)
    culinary = any(_contains_integrity_term(text, term) for term in _CULINARY_TERMS)
    if creature and not culinary:
        return "object_creature_semantic_risk"
    if any(_contains_integrity_term(text, term) for term in _AMORPHOUS_OBJECT_TERMS):
        return "amorphous_object_semantic_risk"
    if any(
        _contains_integrity_term(text, term)
        for term in _SCENE_STRUCTURE_OBJECT_TERMS
    ):
        return "scene_structure_object_semantic_risk"
    if any(
        _contains_integrity_term(text, term) for term in _REPRESENTED_CONTENT_TERMS
    ):
        return "represented_content_semantic_risk"
    return None


def _semantic_entity_head(phrase: str) -> str | None:
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", phrase.casefold())
    head_tokens: list[str] = []
    for token in tokens:
        if token in _SEMANTIC_HEAD_BOUNDARIES:
            break
        head_tokens.append(token)
    return head_tokens[-1] if head_tokens else None


def _semantic_head_matches(head: str, terms: tuple[str, ...]) -> bool:
    if head in terms:
        return True
    singular_candidates = []
    if head.endswith("s"):
        singular_candidates.append(head[:-1])
    if head.endswith("es"):
        singular_candidates.append(head[:-2])
    if head.endswith("ies"):
        singular_candidates.append(f"{head[:-3]}y")
    return any(candidate in terms for candidate in singular_candidates)


def reference_semantic_hard_reject_reason(
    *,
    reference_type: str,
    phrase: str,
) -> str | None:
    """Return only high-confidence object taxonomy violations.

    The broad semantic-risk router intentionally considers grounding context.
    This hard gate instead uses the entity phrase's leading noun phrase so that
    context such as "of water" or "emitting light" cannot cause rejection.
    """

    if reference_type != "object":
        return None
    text = " ".join(phrase.casefold().split())
    head = _semantic_entity_head(text)
    if head is None:
        return None
    culinary = any(_contains_integrity_term(text, term) for term in _CULINARY_TERMS)
    physical_replica = any(
        _contains_integrity_term(text, term) for term in _PHYSICAL_REPLICA_TERMS
    )
    if (
        _semantic_head_matches(head, _OBJECT_CREATURE_TERMS)
        and not culinary
        and not physical_replica
    ):
        return "semantic_policy:living_creature_object"
    if (
        _semantic_head_matches(head, _AMORPHOUS_OBJECT_TERMS)
        and not physical_replica
    ):
        return "semantic_policy:amorphous_object"
    if (
        _semantic_head_matches(head, _SCENE_STRUCTURE_OBJECT_TERMS)
        and not physical_replica
    ):
        return "semantic_policy:scene_structure_object"
    return None


@dataclass(frozen=True)
class ReferenceIntegrityReviewAttempt:
    review: ReferenceIntegrityReview
    raw_response: str
    raw_responses: tuple[str, ...] = ()
    finish_reasons: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_responses:
            object.__setattr__(self, "raw_responses", (self.raw_response,))


class ReferenceIntegrityJudgeFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        raw_responses: tuple[str, ...] = (),
        finish_reasons: tuple[str | None, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses or (
            (raw_response,) if raw_response is not None else ()
        )
        self.raw_response = self.raw_responses[-1] if self.raw_responses else None
        self.finish_reasons = finish_reasons


class ReferenceIntegrityJudge(Protocol):
    def review(
        self,
        *,
        source_context: Image.Image,
        final_reference: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        synthetic: bool,
    ) -> ReferenceIntegrityReviewAttempt: ...


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


class QwenReferenceIntegrityJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(
        self,
        *,
        messages: list[dict[str, object]],
        metadata: dict[str, object],
        operation: str,
        retry_index: int,
    ) -> object:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            return profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_reference_integrity_review",
                            "strict": True,
                            "schema": ReferenceIntegrityReview.model_json_schema(),
                        },
                    },
                ),
                component="qwen_reference_integrity",
                operation=operation,
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={**metadata, "response_format": "json_schema"},
            )
        except BadRequestError:
            return profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                ),
                component="qwen_reference_integrity",
                operation=operation,
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={**metadata, "response_format": "json_object"},
            )

    @staticmethod
    def _response_content(response: object) -> tuple[str, str | None]:
        try:
            choice = response.choices[0]  # type: ignore[attr-defined]
            content = choice.message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError("Qwen integrity response has no message content") from exc
        finish_reason = getattr(choice, "finish_reason", None)
        return (str(content) if content is not None else ""), (
            str(finish_reason) if finish_reason is not None else None
        )

    def review(
        self,
        *,
        source_context: Image.Image,
        final_reference: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        synthetic: bool,
    ) -> ReferenceIntegrityReviewAttempt:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Reference type: {reference_type}\n"
                            f"Phrase: {phrase}\n"
                            f"Grounding prompt: {grounding_prompt}\n"
                            f"Scope: {reference_scope}\n"
                            f"Synthetic: {str(synthetic).lower()}\n"
                            "Image 1 is source context with the target highlighted. "
                            "Image 2 is the final reference."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(source_context)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(final_reference)},
                    },
                ],
            },
        ]
        metadata = {
            "reference_type": reference_type,
            "reference_scope": reference_scope,
            "synthetic": synthetic,
            "image_count": 2,
            "mode": "targeted_qwen_v1",
        }
        raw_responses: list[str] = []
        finish_reasons: list[str | None] = []
        validation_error: Exception | None = None
        active_messages = messages
        review: ReferenceIntegrityReview | None = None
        for retry_index, operation in ((0, "initial"), (1, "repair")):
            if retry_index:
                assert validation_error is not None
                active_messages = [
                    *messages,
                    {"role": "assistant", "content": raw_responses[-1]},
                    {
                        "role": "user",
                        "content": (
                            "Repair the preceding invalid response. Return only one "
                            "compact strict JSON object matching the supplied "
                            "ReferenceIntegrityReview schema. Preserve the original "
                            "semantic context and image evidence. Follow the supplied "
                            "schema key order. Reason must be one concise sentence. "
                            "Emit reason before verdict, make verdict the final key, "
                            "and close the JSON object immediately after verdict. Do "
                            "not emit trailing whitespace. Do not return markdown, "
                            "explanation, chain-of-thought, or extra fields.\n"
                            f"Validation error: {validation_error}"
                        ),
                    },
                ]
            try:
                response = self._request(
                    messages=active_messages,
                    metadata=metadata,
                    operation=operation,
                    retry_index=retry_index,
                )
                raw, finish_reason = self._response_content(response)
            except Exception as exc:
                raise ReferenceIntegrityJudgeFailure(
                    str(exc),
                    raw_responses=tuple(raw_responses),
                    finish_reasons=tuple(finish_reasons),
                ) from exc
            raw_responses.append(raw)
            finish_reasons.append(finish_reason)
            try:
                review = ReferenceIntegrityReview.model_validate_json(raw)
                break
            except ValueError as exc:
                validation_error = exc
        if review is None:
            assert validation_error is not None
            raise ReferenceIntegrityJudgeFailure(
                str(validation_error),
                raw_responses=tuple(raw_responses),
                finish_reasons=tuple(finish_reasons),
            ) from validation_error
        return ReferenceIntegrityReviewAttempt(
            review=review,
            raw_response=raw_responses[-1],
            raw_responses=tuple(raw_responses),
            finish_reasons=tuple(finish_reasons),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class SourceBboxFallbackReviewAttempt:
    review: SourceBboxFallbackReview
    raw_response: str
    finish_reason: str | None = None


class SourceBboxFallbackJudgeFailure(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class SourceBboxFallbackJudge(Protocol):
    def review(
        self,
        *,
        source_context: Image.Image,
        failed_reference: Image.Image,
        source_bbox_candidate: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        trigger: SourceBboxFallbackTrigger = "artifact_review_reject",
    ) -> SourceBboxFallbackReviewAttempt: ...


class QwenSourceBboxFallbackJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def review(
        self,
        *,
        source_context: Image.Image,
        failed_reference: Image.Image,
        source_bbox_candidate: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        trigger: SourceBboxFallbackTrigger = "artifact_review_reject",
    ) -> SourceBboxFallbackReviewAttempt:
        if trigger == "topology_alpha_hole_upgrade":
            trigger_context = (
                "Image 2 is the current Qwen-accepted reference, but deterministic "
                "alpha topology found a meaningful enclosed transparent hole. "
                "Accept Image 3 only if the raw source bbox is clearly preferable."
            )
        else:
            trigger_context = "Image 2 is the current failed reference."
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Reference type: {reference_type}\n"
                            f"Phrase: {phrase}\n"
                            f"Grounding prompt: {grounding_prompt}\n"
                            f"Scope: {reference_scope}\n"
                            f"Trigger: {trigger}\n"
                            f"{trigger_context}\n"
                            "Review Image 3 as the proposed source_bbox_fallback_v1."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(source_context)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(failed_reference)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(source_bbox_candidate)},
                    },
                ],
            },
        ]
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        metadata = {
            "reference_type": reference_type,
            "reference_scope": reference_scope,
            "image_count": 3,
            "mode": "source_bbox_fallback_v1",
            "trigger": trigger,
        }
        raw: str | None = None
        try:
            try:
                response = profiled_openai_call(
                    lambda: self.client.chat.completions.create(
                        **parameters,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "v3_source_bbox_fallback_review",
                                "strict": True,
                                "schema": SourceBboxFallbackReview.model_json_schema(),
                            },
                        },
                    ),
                    component="qwen_source_bbox_fallback",
                    operation="initial",
                    retry_index=0,
                    model=self.config.model,
                    messages=messages,
                    metadata={**metadata, "response_format": "json_schema"},
                )
            except BadRequestError:
                response = profiled_openai_call(
                    lambda: self.client.chat.completions.create(
                        **parameters,
                        response_format={"type": "json_object"},
                    ),
                    component="qwen_source_bbox_fallback",
                    operation="initial",
                    retry_index=0,
                    model=self.config.model,
                    messages=messages,
                    metadata={**metadata, "response_format": "json_object"},
                )
            raw, finish_reason = QwenReferenceIntegrityJudge._response_content(response)
            review = SourceBboxFallbackReview.model_validate_json(raw)
        except Exception as exc:
            raise SourceBboxFallbackJudgeFailure(
                str(exc), raw_response=raw
            ) from exc
        return SourceBboxFallbackReviewAttempt(
            review=review,
            raw_response=raw,
            finish_reason=finish_reason,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _component_areas(mask: np.ndarray) -> tuple[list[int], list[bool]]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary)
    areas: list[int] = []
    touches_border: list[bool] = []
    for y, x in np.argwhere(binary):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        area = 0
        border = False
        while queue:
            cy, cx = queue.popleft()
            area += 1
            border = border or cy in {0, height - 1} or cx in {0, width - 1}
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        areas.append(area)
        touches_border.append(border)
    return areas, touches_border


def reference_topology_diagnostics(image: Image.Image) -> ReferenceTopologyDiagnostics:
    if "A" not in image.getbands():
        return ReferenceTopologyDiagnostics(
            alpha_available=False,
            significant_component_count=0,
            largest_component_ratio=0.0,
            second_component_ratio=0.0,
            bbox_fill_ratio=0.0,
            border_contact=False,
            enclosed_transparent_hole_count=0,
            largest_enclosed_hole_area=0,
            enclosed_hole_bbox_ratio=0.0,
            suspicious=False,
        )
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
    foreground_area = int(alpha.sum())
    if foreground_area == 0:
        return ReferenceTopologyDiagnostics(
            alpha_available=True,
            significant_component_count=0,
            largest_component_ratio=0.0,
            second_component_ratio=0.0,
            bbox_fill_ratio=0.0,
            border_contact=False,
            enclosed_transparent_hole_count=0,
            largest_enclosed_hole_area=0,
            enclosed_hole_bbox_ratio=0.0,
            suspicious=True,
            suspicion_reasons=["empty_alpha"],
        )
    ys, xs = np.nonzero(alpha)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = (x2 - x1) * (y2 - y1)
    component_areas, _ = _component_areas(alpha)
    significant_minimum = max(16, round(foreground_area * 0.01))
    significant = sorted(
        (area for area in component_areas if area >= significant_minimum),
        reverse=True,
    )
    hole_areas, hole_borders = _component_areas(~alpha[y1:y2, x1:x2])
    enclosed = sorted(
        (
            area
            for area, border in zip(hole_areas, hole_borders)
            if not border
        ),
        reverse=True,
    )
    reasons: list[str] = []
    largest_ratio = significant[0] / foreground_area if significant else 0.0
    second_ratio = significant[1] / foreground_area if len(significant) > 1 else 0.0
    hole_ratio = enclosed[0] / bbox_area if enclosed else 0.0
    bbox_fill = foreground_area / bbox_area
    if len(significant) > 3 or largest_ratio < 0.80:
        reasons.append("fragmented_alpha_topology")
    if hole_ratio >= 0.05:
        reasons.append("large_enclosed_alpha_hole")
    if bbox_fill < 0.12:
        reasons.append("low_alpha_bbox_fill")
    border_contact = bool(
        alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any()
    )
    return ReferenceTopologyDiagnostics(
        alpha_available=True,
        significant_component_count=len(significant),
        largest_component_ratio=largest_ratio,
        second_component_ratio=second_ratio,
        bbox_fill_ratio=bbox_fill,
        border_contact=border_contact,
        enclosed_transparent_hole_count=len(enclosed),
        largest_enclosed_hole_area=enclosed[0] if enclosed else 0,
        enclosed_hole_bbox_ratio=hole_ratio,
        suspicious=bool(reasons),
        suspicion_reasons=reasons,
    )


def _resolve_run_artifact(storage: RunStorage, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference artifact path must be run-relative")
    path = (storage.root / relative).resolve(strict=False)
    try:
        path.relative_to(storage.root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("reference artifact must remain inside run_root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"reference artifact is missing: {path}")
    return path


@dataclass(frozen=True)
class _SourceEvidence:
    source_clip_uid: str
    source_entity_id: str
    frame_slot: int
    source_frame_index: int
    frame_path: Path
    image: Image.Image
    mask: np.ndarray


def _source_evidence(
    storage: RunStorage,
    *,
    clip_uid: str,
    reference: EntityReferenceState,
) -> _SourceEvidence:
    source_clip_uid = reference.source_clip_uid or clip_uid
    source_entity_id = reference.source_entity_id or reference.entity_id
    frames = storage.read_frames(source_clip_uid)
    frame = next(
        (
            item
            for item in frames.frames
            if item.source_frame_index == reference.source_frame_index
        ),
        None,
    )
    if frame is None:
        raise ValueError("reference source frame is absent from sampled frames")
    frame_path = storage.clip_dir(source_clip_uid) / frame.image_path
    with Image.open(frame_path) as opened:
        source = opened.convert("RGB")
        source.load()
    masks = storage.read_masks(source_clip_uid)
    track = masks.entities.get(source_entity_id)
    if track is None:
        raise ValueError("reference source entity mask is missing")
    mask_frame = next((item for item in track.frames if item.slot == frame.slot), None)
    if mask_frame is None or not mask_frame.present:
        raise ValueError("reference source mask is absent at selected frame")
    mask = decode_binary_mask(mask_frame.rle)
    if mask.shape != (source.height, source.width) or not mask.any():
        raise ValueError("reference source mask does not match its source frame")
    return _SourceEvidence(
        source_clip_uid=source_clip_uid,
        source_entity_id=source_entity_id,
        frame_slot=frame.slot,
        source_frame_index=frame.source_frame_index,
        frame_path=frame_path,
        image=source,
        mask=mask,
    )


def _source_context(evidence: _SourceEvidence) -> Image.Image:
    mask = evidence.mask
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    ring = np.asarray(mask_image.filter(ImageFilter.MaxFilter(7))) > 0
    ring &= ~mask
    pixels = np.asarray(evidence.image).copy()
    pixels[ring] = (255, 0, 0)
    return Image.fromarray(pixels)


def _source_bbox_candidate(
    evidence: _SourceEvidence,
    *,
    crop_padding_ratio: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(evidence.mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = math.ceil(max(x2 - x1, y2 - y1) * crop_padding_ratio)
    bbox = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(evidence.image.width, x2 + padding),
        min(evidence.image.height, y2 + padding),
    )
    return evidence.image.crop(bbox).convert("RGB"), bbox


def _write_context_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _rejected_reference(
    reference: EntityReferenceState,
    reason: str,
) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=reference.entity_id,
        status="rejected",
        reference_scope="reject",
        visible_region=reference.visible_region,
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason=reason,
        image_quality=reference.image_quality,
        completeness=reference.completeness,
        viewpoint=reference.viewpoint,
        independent_reference_value=False,
        requires_substantial_invention=reference.requires_substantial_invention,
        primary_identity_region_visible=False,
        major_structure_visible=False,
        truncation_severity=reference.truncation_severity,
        discrete_foreground_instance=reference.discrete_foreground_instance,
        mask_matches_target=reference.mask_matches_target,
        completion_needed_for_reference_use=reference.completion_needed_for_reference_use,
        detached_target_fragments_present=reference.detached_target_fragments_present,
    )


def _artifact_only_bbox_eligible(review: ReferenceIntegrityReview) -> bool:
    semantic_and_structure_passed = all(
        (
            review.matches_target,
            review.reference_entity_semantically_valid,
            review.preserves_annotated_entity_semantics,
            review.preserves_primary_identity_region,
            review.recognizable_as_named_entity,
            review.structurally_complete_for_scope,
            review.no_major_missing_regions,
            review.no_unrelated_entity_dominance,
        )
    )
    artifact_failed = (
        not review.no_severe_reference_artifact
        or not review.no_unnatural_holes_or_surface_loss
    )
    return (
        review.verdict == "reject"
        and semantic_and_structure_passed
        and artifact_failed
    )


def _topology_bbox_upgrade_eligible(
    *,
    review: ReferenceIntegrityReview,
    reference_type: str,
    reference: EntityReferenceState,
    clip_uid: str,
    diagnostics: ReferenceTopologyDiagnostics,
) -> bool:
    return all(
        (
            review.verdict == "accept",
            reference_type == "subject",
            reference.reference_scope == "local",
            not reference.synthetic,
            _is_self_sourced_reference(clip_uid=clip_uid, reference=reference),
            not reference.source_bbox_fallback,
            diagnostics.alpha_available,
            diagnostics.enclosed_transparent_hole_count >= 1,
            diagnostics.largest_enclosed_hole_area >= 1024,
            diagnostics.enclosed_hole_bbox_ratio >= 0.01,
        )
    )


def _is_self_sourced_reference(
    *,
    clip_uid: str,
    reference: EntityReferenceState,
) -> bool:
    return reference.source_clip_uid in {None, clip_uid} and (
        reference.source_entity_id in {None, reference.entity_id}
    )


def _source_bbox_reference(
    reference: EntityReferenceState,
    *,
    clip_uid: str,
    image_path: str,
    bbox_xyxy: tuple[int, int, int, int],
    metadata_path: str,
) -> EntityReferenceState:
    return EntityReferenceState.model_validate(
        reference.model_copy(
            update={
                "image_path": image_path,
                "source_clip_uid": clip_uid,
                "source_entity_id": reference.entity_id,
                "synthetic": False,
                "generation_metadata_path": None,
                "generation_source_sha256": None,
                "generation_output_sha256": None,
                "source_bbox_fallback": True,
                "source_bbox_xyxy": bbox_xyxy,
                "source_bbox_metadata_path": metadata_path,
            }
        ).model_dump(mode="json")
    )


def _reference_edit_after_source_bbox(
    state: ReferenceEditState | None,
    *,
    entity_id: str,
    output_image_path: str,
    metadata_path: str,
) -> ReferenceEditState | None:
    if state is None:
        return None
    updated = []
    found = False
    for item in state.entities:
        if item.entity_id != entity_id:
            updated.append(item)
            continue
        if item.status == "rejected":
            raise ValueError("rejected reference edit cannot publish bbox fallback")
        found = True
        updated.append(
            item.model_copy(
                update={
                    "status": "fallback",
                    "output_image_path": output_image_path,
                    "background_fallback": "none",
                    "fallback_policy": "source_bbox_fallback",
                    "source_bbox_fallback_metadata_path": metadata_path,
                    "reason": "source_bbox_fallback_v1",
                }
            )
        )
    if not found:
        raise ValueError("source bbox fallback requires matching reference edit state")
    return ReferenceEditState.model_validate(
        state.model_copy(update={"entities": updated}).model_dump(mode="json")
    )


@dataclass(frozen=True)
class _SourceBboxEvaluation:
    candidate_relative: str
    metadata_relative: str
    bbox_xyxy: tuple[int, int, int, int]
    attempt: SourceBboxFallbackReviewAttempt | None = None
    judge_failure: SourceBboxFallbackJudgeFailure | None = None


def _evaluate_source_bbox(
    *,
    storage: RunStorage,
    clip_uid: str,
    entity_id: str,
    source_evidence: _SourceEvidence,
    source_context: Image.Image,
    current_reference: Image.Image,
    current_reference_path: Path,
    reference: EntityReferenceState,
    reference_type: str,
    phrase: str,
    grounding_prompt: str,
    original_review: ReferenceIntegrityReview,
    diagnostics: ReferenceTopologyDiagnostics,
    crop_padding_ratio: float,
    trigger: SourceBboxFallbackTrigger,
    judge: SourceBboxFallbackJudge,
) -> _SourceBboxEvaluation:
    candidate, bbox_xyxy = _source_bbox_candidate(
        source_evidence,
        crop_padding_ratio=crop_padding_ratio,
    )
    candidate_path = storage.selected_path(
        clip_uid,
        f"source_bbox_fallback_{entity_id}.png",
    )
    metadata_path = storage.selected_path(
        clip_uid,
        f"source_bbox_fallback_{entity_id}.json",
    )
    _write_context_atomic(candidate_path, candidate)
    candidate_relative = storage.relative_artifact_path(candidate_path)
    metadata_relative = storage.relative_artifact_path(metadata_path)
    current_reference_sha256 = hashlib.sha256(
        current_reference_path.read_bytes()
    ).hexdigest()
    metadata: dict[str, object] = {
        "mode": "source_bbox_fallback_v1",
        "trigger": f"{trigger}_v1",
        "synthetic": False,
        "raw_source_pixels_only": True,
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "source_clip_uid": source_evidence.source_clip_uid,
        "source_entity_id": source_evidence.source_entity_id,
        "source_frame_slot": source_evidence.frame_slot,
        "source_frame_index": source_evidence.source_frame_index,
        "source_frame_path": storage.relative_artifact_path(
            source_evidence.frame_path
        ),
        "source_frame_sha256": hashlib.sha256(
            source_evidence.frame_path.read_bytes()
        ).hexdigest(),
        "source_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(source_evidence.mask.astype(np.uint8)).tobytes()
        ).hexdigest(),
        "bbox_xyxy": list(bbox_xyxy),
        "crop_padding_ratio": crop_padding_ratio,
        "candidate_path": candidate_relative,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "original_reference_path": reference.image_path,
        "original_reference_sha256": current_reference_sha256,
        "original_integrity_review": original_review.model_dump(mode="json"),
    }
    if trigger == "artifact_review_reject":
        metadata.update(
            {
                "failed_reference_path": reference.image_path,
                "failed_reference_sha256": current_reference_sha256,
            }
        )
    else:
        metadata["topology_evidence"] = {
            "alpha_available": diagnostics.alpha_available,
            "enclosed_transparent_hole_count": (
                diagnostics.enclosed_transparent_hole_count
            ),
            "largest_enclosed_hole_area": diagnostics.largest_enclosed_hole_area,
            "enclosed_hole_bbox_ratio": diagnostics.enclosed_hole_bbox_ratio,
        }
    try:
        attempt = judge.review(
            source_context=source_context,
            failed_reference=current_reference,
            source_bbox_candidate=candidate,
            reference_type=reference_type,
            phrase=phrase,
            grounding_prompt=grounding_prompt,
            reference_scope=reference.reference_scope,
            trigger=trigger,
        )
    except SourceBboxFallbackJudgeFailure as exc:
        metadata.update(
            {
                "status": "judge_failed",
                "reason": str(exc),
                "raw_response": exc.raw_response,
            }
        )
        write_json_atomic(metadata_path, metadata)
        return _SourceBboxEvaluation(
            candidate_relative=candidate_relative,
            metadata_relative=metadata_relative,
            bbox_xyxy=bbox_xyxy,
            judge_failure=exc,
        )
    metadata.update(
        {
            "status": "accepted" if attempt.review.verdict == "accept" else "rejected",
            "review": attempt.review.model_dump(mode="json"),
            "raw_response": attempt.raw_response,
            "finish_reason": attempt.finish_reason,
        }
    )
    write_json_atomic(metadata_path, metadata)
    return _SourceBboxEvaluation(
        candidate_relative=candidate_relative,
        metadata_relative=metadata_relative,
        bbox_xyxy=bbox_xyxy,
        attempt=attempt,
    )


def _tokens_for_retained(
    retained: list[str],
    reference_types: dict[str, str],
) -> dict[str, str]:
    counters = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity_id in retained:
        reference_type = reference_types[entity_id]
        counters[reference_type] += 1
        tokens[entity_id] = f"<ref_{reference_type}_{counters[reference_type]}>"
    return tokens


@dataclass(frozen=True)
class ReferenceIntegrityStats:
    processed: int = 0
    skipped_disabled: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    entities_reviewed: int = 0
    entities_skipped_review: int = 0
    entities_accepted: int = 0
    entities_rejected: int = 0
    semantic_policy_rejected: int = 0
    judge_failed: int = 0
    topology_suspicious: int = 0
    source_bbox_fallback_attempted: int = 0
    source_bbox_fallback_accepted: int = 0
    source_bbox_fallback_rejected: int = 0
    source_bbox_fallback_judge_failed: int = 0
    source_bbox_topology_upgrade_attempted: int = 0
    source_bbox_topology_upgrade_accepted: int = 0
    source_bbox_topology_upgrade_kept_original: int = 0
    source_bbox_topology_upgrade_judge_failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def reference_integrity_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    judge: ReferenceIntegrityJudge | None = None,
    bbox_fallback_judge: SourceBboxFallbackJudge | None = None,
) -> ReferenceIntegrityStats:
    counters = {field: 0 for field in ReferenceIntegrityStats.__dataclass_fields__}
    if not config.reference_integrity.enabled:
        counters["skipped_disabled"] = sum(1 for _ in storage.iter_clips())
        stats = ReferenceIntegrityStats(**counters)
        storage.update_stage_counts("reference_integrity", stats.to_dict())
        return stats
    owned_judge: QwenReferenceIntegrityJudge | None = None
    owned_bbox_fallback_judge: QwenSourceBboxFallbackJudge | None = None
    active_judge = judge
    active_bbox_fallback_judge = bbox_fallback_judge
    if active_judge is None:
        service = config.qwen.reference_integrity_judge
        if service is None:
            raise ValueError("reference integrity Qwen judge is not configured")
        owned_judge = QwenReferenceIntegrityJudge(service)
        active_judge = owned_judge
    try:
        for clip in storage.iter_clips():
            if clip.pairing is None or clip.pairing.status != "ready":
                counters["skipped_not_ready"] += 1
                continue
            if config.reference_edit.enabled and (
                clip.reference_edit is None or clip.reference_edit.status != "ready"
            ):
                counters["skipped_not_ready"] += 1
                continue
            if (
                clip.reference_integrity is not None
                and clip.reference_integrity.status == "ready"
                and not overwrite
            ):
                counters["skipped_existing"] += 1
                continue
            counters["processed"] += 1
            try:
                references_by_id = {
                    item.entity_id: item for item in clip.references.entities
                }
                entities_by_id = {
                    item.entity_id: item
                    for item in (
                        clip.annotation.entities if clip.annotation is not None else []
                    )
                }
                results: list[ReferenceIntegrityEntityState] = []
                rejected_ids: set[str] = set()
                fallback_references: dict[str, EntityReferenceState] = {}
                updated_reference_edit = clip.reference_edit
                for entity_id in clip.pairing.retained_entity_ids:
                    reference = references_by_id[entity_id]
                    entity = entities_by_id[entity_id]
                    assert reference.image_path is not None
                    final_path = _resolve_run_artifact(storage, reference.image_path)
                    with Image.open(final_path) as opened:
                        final_image = opened.copy()
                        final_image.load()
                    diagnostics = reference_topology_diagnostics(final_image)
                    counters["topology_suspicious"] += int(diagnostics.suspicious)
                    semantic_risk = reference_semantic_risk_reason(
                        reference_type=entity.reference_type,
                        phrase=entity.phrase,
                        grounding_prompt=entity.grounding_prompt,
                    )
                    semantic_policy_reason = reference_semantic_hard_reject_reason(
                        reference_type=entity.reference_type,
                        phrase=entity.phrase,
                    )
                    if semantic_policy_reason is not None:
                        rejected_ids.add(entity_id)
                        counters["entities_rejected"] += 1
                        counters["semantic_policy_rejected"] += 1
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="rejected",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                diagnostics=diagnostics,
                                reviewed=False,
                                semantic_policy_reason=semantic_policy_reason,
                                reason=semantic_policy_reason,
                            )
                        )
                        continue
                    requires_review = (
                        reference.synthetic
                        or reference.reference_scope == "local"
                        or reference.source_bbox_fallback
                        or diagnostics.suspicious
                        or semantic_risk is not None
                    )
                    if not requires_review:
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="skipped",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                diagnostics=diagnostics,
                                reviewed=False,
                                reason="clean_real_full_reference",
                            )
                        )
                        counters["entities_skipped_review"] += 1
                        counters["entities_accepted"] += 1
                        continue
                    source_evidence = _source_evidence(
                        storage, clip_uid=clip.clip_uid, reference=reference
                    )
                    context = _source_context(source_evidence)
                    context_path = storage.selected_path(
                        clip.clip_uid, f"integrity_context_{entity_id}.png"
                    )
                    _write_context_atomic(context_path, context)
                    context_relative = storage.relative_artifact_path(context_path)
                    counters["entities_reviewed"] += 1
                    try:
                        attempt = active_judge.review(
                            source_context=context,
                            final_reference=final_image,
                            reference_type=entity.reference_type,
                            phrase=entity.phrase,
                            grounding_prompt=entity.grounding_prompt,
                            reference_scope=reference.reference_scope,
                            synthetic=reference.synthetic,
                        )
                    except ReferenceIntegrityJudgeFailure as exc:
                        if config.debug.save_diagnostics:
                            debug_path = storage.debug_path(
                                clip.clip_uid,
                                f"reference_integrity_{entity_id}.json",
                            )
                            write_json_atomic(
                                debug_path,
                                {
                                    "judge_failed": True,
                                    "raw_responses": list(exc.raw_responses),
                                    "finish_reasons": list(exc.finish_reasons),
                                    "reason": str(exc),
                                },
                            )
                        rejected_ids.add(entity_id)
                        counters["judge_failed"] += 1
                        counters["entities_rejected"] += 1
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="rejected",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                source_context_path=context_relative,
                                diagnostics=diagnostics,
                                reviewed=True,
                                judge_failed=True,
                                reason=f"integrity_judge_failed:{exc}",
                            )
                        )
                        continue
                    if config.debug.save_diagnostics:
                        debug_path = storage.debug_path(
                            clip.clip_uid, f"reference_integrity_{entity_id}.json"
                        )
                        write_json_atomic(
                            debug_path,
                            {
                                "review": attempt.review.model_dump(mode="json"),
                                "raw_response": attempt.raw_response,
                                "raw_responses": list(attempt.raw_responses),
                                "finish_reasons": list(attempt.finish_reasons),
                            },
                        )
                    topology_upgrade_eligible = _topology_bbox_upgrade_eligible(
                        review=attempt.review,
                        reference_type=entity.reference_type,
                        reference=reference,
                        clip_uid=clip.clip_uid,
                        diagnostics=diagnostics,
                    )
                    artifact_fallback_eligible = (
                        _artifact_only_bbox_eligible(attempt.review)
                        and not reference.source_bbox_fallback
                        and _is_self_sourced_reference(
                            clip_uid=clip.clip_uid,
                            reference=reference,
                        )
                    )
                    if not (
                        topology_upgrade_eligible or artifact_fallback_eligible
                    ):
                        accepted = attempt.review.verdict == "accept"
                        if accepted:
                            counters["entities_accepted"] += 1
                        else:
                            rejected_ids.add(entity_id)
                            counters["entities_rejected"] += 1
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="accepted" if accepted else "rejected",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                source_context_path=context_relative,
                                diagnostics=diagnostics,
                                reviewed=True,
                                review=attempt.review,
                                reason=attempt.review.reason,
                            )
                        )
                        continue

                    trigger: SourceBboxFallbackTrigger = (
                        "topology_alpha_hole_upgrade"
                        if topology_upgrade_eligible
                        else "artifact_review_reject"
                    )
                    if topology_upgrade_eligible:
                        counters["source_bbox_topology_upgrade_attempted"] += 1
                    else:
                        counters["source_bbox_fallback_attempted"] += 1
                    if active_bbox_fallback_judge is None:
                        service = config.qwen.reference_integrity_judge
                        if service is None:
                            raise ValueError(
                                "reference integrity Qwen judge is not configured"
                            )
                        owned_bbox_fallback_judge = QwenSourceBboxFallbackJudge(
                            service
                        )
                        active_bbox_fallback_judge = owned_bbox_fallback_judge
                    bbox_evaluation = _evaluate_source_bbox(
                        storage=storage,
                        clip_uid=clip.clip_uid,
                        entity_id=entity_id,
                        source_evidence=source_evidence,
                        source_context=context,
                        current_reference=final_image,
                        current_reference_path=final_path,
                        reference=reference,
                        reference_type=entity.reference_type,
                        phrase=entity.phrase,
                        grounding_prompt=entity.grounding_prompt,
                        original_review=attempt.review,
                        diagnostics=diagnostics,
                        crop_padding_ratio=config.pair.crop_padding_ratio,
                        trigger=trigger,
                        judge=active_bbox_fallback_judge,
                    )
                    if bbox_evaluation.judge_failure is not None:
                        exc = bbox_evaluation.judge_failure
                        if topology_upgrade_eligible:
                            counters[
                                "source_bbox_topology_upgrade_judge_failed"
                            ] += 1
                            counters[
                                "source_bbox_topology_upgrade_kept_original"
                            ] += 1
                            counters["entities_accepted"] += 1
                            results.append(
                                ReferenceIntegrityEntityState(
                                    entity_id=entity_id,
                                    status="accepted",
                                    input_reference=reference,
                                    final_reference_path=reference.image_path,
                                    source_context_path=context_relative,
                                    diagnostics=diagnostics,
                                    reviewed=True,
                                    review=attempt.review,
                                    source_bbox_fallback_trigger=trigger,
                                    source_bbox_fallback_candidate_path=(
                                        bbox_evaluation.candidate_relative
                                    ),
                                    source_bbox_fallback_metadata_path=(
                                        bbox_evaluation.metadata_relative
                                    ),
                                    source_bbox_xyxy=bbox_evaluation.bbox_xyxy,
                                    source_bbox_fallback_judge_failed=True,
                                    reason=(
                                        "topology_bbox_upgrade_judge_failed_"
                                        f"kept_original:{exc}"
                                    ),
                                )
                            )
                            continue
                        counters["source_bbox_fallback_judge_failed"] += 1
                        counters["source_bbox_fallback_rejected"] += 1
                        counters["entities_rejected"] += 1
                        rejected_ids.add(entity_id)
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="rejected",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                source_context_path=context_relative,
                                diagnostics=diagnostics,
                                reviewed=True,
                                review=attempt.review,
                                reason=f"source_bbox_fallback_judge_failed:{exc}",
                            )
                        )
                        continue
                    fallback_attempt = bbox_evaluation.attempt
                    assert fallback_attempt is not None
                    fallback_accepted = fallback_attempt.review.verdict == "accept"
                    if fallback_accepted:
                        fallback_reference = _source_bbox_reference(
                            reference,
                            clip_uid=clip.clip_uid,
                            image_path=bbox_evaluation.candidate_relative,
                            bbox_xyxy=bbox_evaluation.bbox_xyxy,
                            metadata_path=bbox_evaluation.metadata_relative,
                        )
                        fallback_references[entity_id] = fallback_reference
                        updated_reference_edit = _reference_edit_after_source_bbox(
                            updated_reference_edit,
                            entity_id=entity_id,
                            output_image_path=bbox_evaluation.candidate_relative,
                            metadata_path=bbox_evaluation.metadata_relative,
                        )
                        if topology_upgrade_eligible:
                            counters["source_bbox_topology_upgrade_accepted"] += 1
                        else:
                            counters["source_bbox_fallback_accepted"] += 1
                        counters["entities_accepted"] += 1
                    elif topology_upgrade_eligible:
                        counters["source_bbox_topology_upgrade_kept_original"] += 1
                        counters["entities_accepted"] += 1
                    else:
                        rejected_ids.add(entity_id)
                        counters["source_bbox_fallback_rejected"] += 1
                        counters["entities_rejected"] += 1
                    results.append(
                        ReferenceIntegrityEntityState(
                            entity_id=entity_id,
                            status=(
                                "accepted"
                                if fallback_accepted or topology_upgrade_eligible
                                else "rejected"
                            ),
                            input_reference=reference,
                            final_reference_path=(
                                bbox_evaluation.candidate_relative
                                if fallback_accepted
                                else reference.image_path
                            ),
                            source_context_path=context_relative,
                            diagnostics=diagnostics,
                            reviewed=True,
                            review=attempt.review,
                            source_bbox_fallback_trigger=trigger,
                            source_bbox_fallback_candidate_path=(
                                bbox_evaluation.candidate_relative
                            ),
                            source_bbox_fallback_metadata_path=(
                                bbox_evaluation.metadata_relative
                            ),
                            source_bbox_xyxy=bbox_evaluation.bbox_xyxy,
                            source_bbox_fallback_review=fallback_attempt.review,
                            reason=fallback_attempt.review.reason,
                        )
                    )
                final_references = [
                    (
                        fallback_references[item.entity_id]
                        if item.entity_id in fallback_references
                        else _rejected_reference(
                            item, "reference_integrity_rejected"
                        )
                        if item.entity_id in rejected_ids
                        else item
                    )
                    for item in clip.references.entities
                ]
                retained = [
                    entity_id
                    for entity_id in clip.pairing.retained_entity_ids
                    if entity_id not in rejected_ids
                ]
                qualifying = set(
                    clip.coverage.qualifying_entity_ids
                    if clip.coverage is not None
                    else []
                )
                if not set(retained).intersection(qualifying):
                    pairing = PairingState(
                        status="rejected",
                        reason="no_qualifying_reference_after_integrity",
                    )
                else:
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(
                            retained,
                            {
                                item.entity_id: item.reference_type
                                for item in entities_by_id.values()
                            },
                        ),
                        background_token=clip.pairing.background_token,
                    )
                storage.write_reference_integrity_result(
                    clip.clip_uid,
                    ReferencesState(
                        entities=final_references,
                        background=clip.references.background,
                    ),
                    pairing,
                    ReferenceIntegrityState(status="ready", entities=results),
                    reference_edit=updated_reference_edit,
                )
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clip artifacts
                storage.write_reference_integrity_failure(clip.clip_uid, str(exc))
                storage.append_failure(
                    stage="reference_integrity",
                    clip_uid=clip.clip_uid,
                    reason=str(exc),
                    details={"exception_type": type(exc).__name__},
                )
                counters["failed"] += 1
    finally:
        if owned_judge is not None:
            owned_judge.close()
        if owned_bbox_fallback_judge is not None:
            owned_bbox_fallback_judge.close()
    stats = ReferenceIntegrityStats(**counters)
    storage.update_stage_counts("reference_integrity", stats.to_dict())
    return stats
