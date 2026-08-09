from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import profiled_openai_call

if TYPE_CHECKING:
    from r2v_data_v2.v3.storage import RunStorage


SYSTEM_PROMPT = """You judge whether one isolated entity image is useful as a generation reference.

Accept only if the visible content is a coherent, independently reusable
representation of the described entity.

For a person, an identity-bearing region such as the face or head must be
visible. Reject torso-only, clothing-only, back-only, extremely narrow,
severely truncated, fragmented, or detached body-part references.

For objects or groups, the primary recognizable structure must remain visible.

Do not require the entire physical entity when a natural local view is otherwise
coherent and recognizable.

Judge visible facts only. Return strict JSON."""


class ScaleCollapseFallbackReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    identity_or_primary_region_visible: StrictBool
    coherent_structure: StrictBool
    independent_reference_value: StrictBool
    not_severely_fragmented: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("scale-collapse fallback reason must not be empty")
        return stripped

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> ScaleCollapseFallbackReview:
        accepted = all(
            (
                self.identity_or_primary_region_visible,
                self.coherent_structure,
                self.independent_reference_value,
                self.not_severely_fragmented,
            )
        )
        if (self.verdict == "accept") != accepted:
            raise ValueError("scale-collapse fallback verdict must match all flags")
        return self


@dataclass(frozen=True)
class ScaleCollapseFallbackReviewAttempt:
    review: ScaleCollapseFallbackReview
    raw_responses: tuple[str, ...]

    @property
    def raw_response(self) -> str:
        return self.raw_responses[-1]


class ScaleCollapseFallbackJudgeFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_responses: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


class ScaleCollapseFallbackJudge(Protocol):
    def review(
        self,
        *,
        image: Image.Image,
        reference_type: str,
        entity_phrase: str,
    ) -> ScaleCollapseFallbackReviewAttempt: ...


def load_source_reference_image(
    storage: RunStorage,
    image_path: str,
) -> Image.Image:
    root = storage.root.resolve(strict=True)
    artifact = Path(image_path).expanduser()
    if ".." in artifact.parts:
        raise ValueError("source reference path cannot contain path traversal")
    candidate = artifact if artifact.is_absolute() else root / artifact
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("source reference image must remain inside run_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"source reference image is missing: {resolved}")
    with Image.open(resolved) as opened:
        image = opened.copy()
        image.load()
    return image


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenScaleCollapseFallbackJudge:
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
        messages: list[dict[str, object]],
        *,
        reference_type: str,
        retry_index: int,
    ) -> str:
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
            "image_count": 1,
            "mode": "qwen_v1",
        }
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_scale_collapse_fallback_review",
                            "strict": True,
                            "schema": ScaleCollapseFallbackReview.model_json_schema(),
                        },
                    },
                ),
                component="qwen_scale_collapse_fallback_guard",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
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
                component="qwen_scale_collapse_fallback_guard",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={**metadata, "response_format": "json_object"},
            )
        raw = response.choices[0].message.content
        if not raw:
            raise RuntimeError("Qwen returned an empty scale-collapse review")
        return str(raw)

    def review(
        self,
        *,
        image: Image.Image,
        reference_type: str,
        entity_phrase: str,
    ) -> ScaleCollapseFallbackReviewAttempt:
        if reference_type not in {"subject", "object", "group"}:
            raise ValueError("scale-collapse reference_type is invalid")
        phrase = entity_phrase.strip()
        if not phrase:
            raise ValueError("scale-collapse entity phrase must not be empty")
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Reference type: {reference_type}\n"
                    f"Entity phrase: {phrase}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": _png_data_url(image)},
            },
        ]
        original_messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        raw_responses: list[str] = []
        validation_error: Exception | None = None
        for retry_index in range(2):
            messages = original_messages
            if retry_index:
                messages = [
                    *original_messages,
                    {"role": "assistant", "content": raw_responses[-1]},
                    {
                        "role": "user",
                        "content": (
                            "Repair the JSON to match the schema exactly. "
                            f"Validation error: {validation_error}"
                        ),
                    },
                ]
            try:
                raw = self._request(
                    messages,
                    reference_type=reference_type,
                    retry_index=retry_index,
                )
            except Exception as exc:
                raise ScaleCollapseFallbackJudgeFailure(
                    f"scale-collapse fallback request failed: {exc}",
                    raw_responses=tuple(raw_responses),
                ) from exc
            raw_responses.append(raw)
            try:
                review = ScaleCollapseFallbackReview.model_validate_json(raw)
            except (TypeError, ValueError) as exc:
                validation_error = exc
                continue
            return ScaleCollapseFallbackReviewAttempt(
                review=review,
                raw_responses=tuple(raw_responses),
            )
        raise ScaleCollapseFallbackJudgeFailure(
            f"invalid scale-collapse fallback review: {validation_error}",
            raw_responses=tuple(raw_responses),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
