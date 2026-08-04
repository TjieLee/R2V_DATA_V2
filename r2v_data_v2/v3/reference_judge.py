from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image

from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    build_structured_repair_prompt,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    RawEntityReferenceDecision,
)

if TYPE_CHECKING:
    from r2v_data_v2.v3.pair import EntityReferenceCandidate

SYSTEM_PROMPT = """You select the best reference image for one known entity.

Candidate IDs and candidate order are immutable. Judge only visible evidence and
never infer a body, structure, accessory, or identity feature outside the
images. A full reference may still have minor edge contact or small peripheral
parts outside the frame; do not downgrade only because of border contact,
natural holes, or multiple natural connected components. Local means a
coherent, reusable, identity-bearing but incomplete visual region; it may still
be recognizable as the whole known entity when enough identity and structure
remain visible. Reject when identity is not visible, fragmentation or occlusion
is severe, or segmentation is wrong. The context image shows scene placement;
the isolated crop shows the
exact proposed reference content.

Do not output tokens or crop coordinates. Return one strict JSON object only
with selected_candidate_id, reference_scope, visible_region,
whole_entity_recognizable, identity_features_visible, and scope_reason."""


@dataclass(frozen=True)
class EntityReferenceDecisionAttempt:
    decision: RawEntityReferenceDecision
    raw_responses: tuple[str, ...]
    repair_attempts: int


class EntityReferenceJudgeFailure(StructuredOutputFailure):
    pass


class EntityReferenceJudge(Protocol):
    def decide(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> EntityReferenceDecisionAttempt: ...


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_entity_reference_request_payload(
    entity: AnnotationEntity,
    candidates: list[EntityReferenceCandidate],
) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "phrase": entity.phrase,
        "grounding_prompt": entity.grounding_prompt,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "frame_slot": candidate.frame_slot,
                "source_frame_index": candidate.source_frame_index,
                "area_ratio": candidate.area_ratio,
                "bbox_fill_ratio": candidate.bbox_fill_ratio,
                "border_contact_count": candidate.border_contact_count,
            }
            for candidate in candidates
        ],
    }


def validate_entity_reference_decision(
    decision: RawEntityReferenceDecision,
    *,
    candidate_ids: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    selected = decision.selected_candidate_id
    if selected is not None and selected not in candidate_ids:
        issues.append(
            ValidationIssue(
                code="unknown_candidate_id",
                field="selected_candidate_id",
                message=f"unknown candidate ID: {selected}",
            )
        )
    if decision.reference_scope == "full":
        if decision.visible_region != "whole":
            issues.append(
                ValidationIssue(
                    code="full_requires_whole",
                    field="visible_region",
                    message="full reference must use visible_region=whole",
                )
            )
        if not decision.whole_entity_recognizable:
            issues.append(
                ValidationIssue(
                    code="full_requires_recognizable",
                    field="whole_entity_recognizable",
                    message="full reference must recognize the whole entity",
                )
            )
        if not decision.identity_features_visible:
            issues.append(
                ValidationIssue(
                    code="full_requires_identity",
                    field="identity_features_visible",
                    message="full reference requires visible identity features",
                )
            )
    elif decision.reference_scope == "local":
        if decision.visible_region == "whole":
            issues.append(
                ValidationIssue(
                    code="local_must_be_non_whole",
                    field="visible_region",
                    message="local reference cannot use visible_region=whole",
                )
            )
        if not decision.identity_features_visible:
            issues.append(
                ValidationIssue(
                    code="local_requires_identity",
                    field="identity_features_visible",
                    message="local reference requires visible identity features",
                )
            )
    return issues


class QwenEntityReferenceJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        repair_retries: int = 1,
        crop_padding_ratio: float = 0.08,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.repair_retries = repair_retries
        self.crop_padding_ratio = crop_padding_ratio
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
        request_text: str,
    ) -> list[dict[str, object]]:
        from r2v_data_v2.v3.pair import (
            build_candidate_context_image,
            build_reference_crop,
        )

        content: list[dict[str, object]] = [
            {"type": "text", "text": request_text},
        ]
        for candidate in candidates:
            source = source_images.get(candidate.image_path)
            if source is None:
                raise KeyError(
                    f"missing source image for candidate: {candidate.image_path}"
                )
            context = build_candidate_context_image(source, candidate.mask)
            crop, _ = build_reference_crop(
                source,
                candidate.mask,
                crop_padding_ratio=self.crop_padding_ratio,
            )
            isolated = Image.new("RGB", crop.size, (255, 255, 255))
            isolated.paste(crop, mask=crop.getchannel("A"))
            for label, image in (
                (
                    f"Candidate {candidate.candidate_id} context",
                    context,
                ),
                (
                    f"Candidate {candidate.candidate_id} isolated crop",
                    isolated,
                ),
            ):
                content.append({"type": "text", "text": label})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(image)},
                    }
                )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "v3_entity_reference_decision",
                        "strict": True,
                        "schema": RawEntityReferenceDecision.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        result = response.choices[0].message.content
        if not result:
            raise RuntimeError("Qwen returned an empty entity reference decision")
        return str(result)

    def decide(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> EntityReferenceDecisionAttempt:
        if not candidates:
            raise ValueError("entity reference judge requires candidates")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if candidate_ids != [
            f"candidate_{index}" for index in range(1, len(candidates) + 1)
        ]:
            raise ValueError(
                "entity reference candidates must be contiguous and ordered"
            )
        payload = build_entity_reference_request_payload(entity, candidates)
        original_request = (
            "Select one immutable candidate or reject all candidates. "
            "Return strict JSON for this payload:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.repair_retries + 1):
            request_text = original_request
            if attempt:
                request_text = build_structured_repair_prompt(
                    original_request=original_request,
                    invalid_response=raw_responses[-1],
                    validation_issues=issues,
                    json_schema=(RawEntityReferenceDecision.model_json_schema()),
                )
            try:
                raw = self._request(
                    self._messages(
                        entity=entity,
                        candidates=candidates,
                        source_images=source_images,
                        request_text=request_text,
                    )
                )
            except Exception as exc:
                raise EntityReferenceJudgeFailure(
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
            decision, issues = parse_qwen_json_issues(
                raw,
                RawEntityReferenceDecision,
            )
            if decision is not None:
                issues = validate_entity_reference_decision(
                    decision,
                    candidate_ids=set(candidate_ids),
                )
            if decision is not None and not issues:
                return EntityReferenceDecisionAttempt(
                    decision=decision,
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                )
        raise EntityReferenceJudgeFailure(
            raw_responses=raw_responses,
            issues=issues,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
