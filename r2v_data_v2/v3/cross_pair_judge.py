from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image

from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    build_structured_repair_prompt,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import (
    get_model_profile_context,
    model_profile_context,
    profiled_openai_call,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    RawCrossPairDecision,
)

CrossPairTargetEvidenceMode = Literal["masked_candidate", "sampled_frames"]


SYSTEM_PROMPT = """You decide whether a donor reference shows the same physical entity as a target entity.

Judge people, objects, and groups only from the supplied visual evidence.
First confirm that the target phrase's entity is actually visible in the target
visual evidence. If you cannot find the target entity, reject it.
Matching category does not prove matching identity; the same class is not identity.
Similar clothing is insufficient to prove a person's identity. Generic trees,
fish, vehicles, furniture, and other category-level appearances are insufficient
to prove the same physical entity. Reject whenever target or donor identity is
uncertain.
Reject when the donor is not a faithful usable reference for the target.
Never accept from text alone, and never infer an identity relationship that is not visible.
Do not infer identity from a shared parent video, neighboring clips, or textual
semantics. Sampled-frame context-only evidence must still be judged visually.
Do not invent unseen identity features.

Return one strict JSON object only with verdict, target_entity_visible,
same_physical_entity, identity_features_match, reference_is_usable, and reason.
Accept if and only if all four boolean checks are true."""


@dataclass(frozen=True)
class CrossPairDecisionAttempt:
    decision: RawCrossPairDecision
    raw_responses: tuple[str, ...]
    repair_attempts: int


class CrossPairJudgeFailure(StructuredOutputFailure):
    pass


class CrossPairJudge(Protocol):
    def decide(
        self,
        *,
        target_clip_uid: str,
        target_entity: AnnotationEntity,
        target_evidence_mode: CrossPairTargetEvidenceMode,
        target_context_image: Image.Image,
        target_entity_crop: Image.Image | None,
        donor_clip_uid: str,
        donor_entity: AnnotationEntity,
        donor_reference_image: Image.Image,
    ) -> CrossPairDecisionAttempt: ...


def build_cross_pair_request_payload(
    *,
    target_clip_uid: str,
    target_entity: AnnotationEntity,
    target_evidence_mode: CrossPairTargetEvidenceMode,
    donor_clip_uid: str,
    donor_entity: AnnotationEntity,
) -> dict[str, object]:
    return {
        "target": {
            "clip_uid": target_clip_uid,
            "entity_id": target_entity.entity_id,
            "reference_type": target_entity.reference_type,
            "phrase": target_entity.phrase,
            "grounding_prompt": target_entity.grounding_prompt,
            "evidence_mode": target_evidence_mode,
        },
        "donor": {
            "clip_uid": donor_clip_uid,
            "entity_id": donor_entity.entity_id,
            "phrase": donor_entity.phrase,
            "grounding_prompt": donor_entity.grounding_prompt,
        },
    }


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _white_background_rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("cross-pair evidence must be PIL images")
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    result = Image.new("RGB", rgba.size, (255, 255, 255))
    result.paste(rgba, mask=rgba.getchannel("A"))
    return result


class QwenCrossPairJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        repair_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.repair_retries = repair_retries
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        target_evidence_mode: CrossPairTargetEvidenceMode,
        target_context_image: Image.Image,
        target_entity_crop: Image.Image | None,
        donor_reference_image: Image.Image,
        request_text: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {"type": "text", "text": request_text},
        ]
        if target_evidence_mode == "masked_candidate":
            if target_entity_crop is None:
                raise ValueError(
                    "masked_candidate evidence requires a target entity crop"
                )
            target_images = (
                ("Target context frame", target_context_image),
                ("Target entity crop", target_entity_crop),
            )
        elif target_evidence_mode == "sampled_frames":
            if target_entity_crop is not None:
                raise ValueError(
                    "sampled_frames evidence must not include a target entity crop"
                )
            target_images = (
                ("Target sampled-frame contact sheet", target_context_image),
            )
        else:
            raise ValueError(
                f"unsupported target evidence mode: {target_evidence_mode}"
            )
        for label, image in (
            *target_images,
            ("Donor source-faithful reference", donor_reference_image),
        ):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _png_data_url(_white_background_rgb(image))
                    },
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        profile_context = get_model_profile_context()
        retry_index = profile_context.retry_index
        target_evidence_mode = str(
            profile_context.metadata.get("target_evidence_mode", "unknown")
        )
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_cross_pair_decision",
                            "strict": True,
                            "schema": RawCrossPairDecision.model_json_schema(),
                        },
                    },
                ),
                component="qwen_cross_pair_judge",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    "target_evidence_mode": target_evidence_mode,
                    "response_format": "json_schema",
                },
            )
        except BadRequestError:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                ),
                component="qwen_cross_pair_judge",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    "target_evidence_mode": target_evidence_mode,
                    "response_format": "json_object",
                },
            )
        result = response.choices[0].message.content
        if not result:
            raise RuntimeError("Qwen returned an empty cross-pair decision")
        return str(result)

    def decide(
        self,
        *,
        target_clip_uid: str,
        target_entity: AnnotationEntity,
        target_evidence_mode: CrossPairTargetEvidenceMode,
        target_context_image: Image.Image,
        target_entity_crop: Image.Image | None,
        donor_clip_uid: str,
        donor_entity: AnnotationEntity,
        donor_reference_image: Image.Image,
    ) -> CrossPairDecisionAttempt:
        payload = build_cross_pair_request_payload(
            target_clip_uid=target_clip_uid,
            target_entity=target_entity,
            target_evidence_mode=target_evidence_mode,
            donor_clip_uid=donor_clip_uid,
            donor_entity=donor_entity,
        )
        original_request = (
            "Decide whether the donor is a real usable reference for the exact "
            "target entity. Return strict JSON for this immutable payload:\n"
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
                    json_schema=RawCrossPairDecision.model_json_schema(),
                )
            try:
                with model_profile_context(
                    retry_index=attempt,
                    metadata={"target_evidence_mode": target_evidence_mode},
                ):
                    raw = self._request(
                        self._messages(
                            target_evidence_mode=target_evidence_mode,
                            target_context_image=target_context_image,
                            target_entity_crop=target_entity_crop,
                            donor_reference_image=donor_reference_image,
                            request_text=request_text,
                        )
                    )
            except Exception as exc:
                raise CrossPairJudgeFailure(
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
                RawCrossPairDecision,
            )
            if decision is not None and not issues:
                return CrossPairDecisionAttempt(
                    decision=decision,
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                )
        raise CrossPairJudgeFailure(
            raw_responses=raw_responses,
            issues=issues,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
