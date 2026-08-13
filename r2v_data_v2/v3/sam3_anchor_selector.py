from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import profiled_openai_call
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    MultiInstanceAnchorDecision,
)
from r2v_data_v2.v3.schemas import Sam3AnchorSelectionReview

SYSTEM_PROMPT = """You select a unique SAM3 anchor for one annotated entity.
The image contains numbered candidate masks from one source frame. Use the full
annotation phrase, grounding prompt, and reference type to identify the one mask
that corresponds to the annotated entity. Select exactly one candidate only
when its identity is unambiguous. Do not select by mask score, mask size, or
candidate order. Never combine candidates. Return reject when no candidate is
the target and uncertain when more than one candidate remains plausible.
Return strict JSON only."""

_COLORS = (
    (255, 40, 40),
    (40, 180, 255),
    (255, 210, 40),
    (80, 220, 100),
    (220, 80, 255),
    (255, 130, 30),
)


class QwenAnchorSelectionFailure(RuntimeError):
    pass


def render_numbered_anchor_candidates(
    image: Image.Image,
    candidates: tuple[BackendMaskObservation, ...],
) -> Image.Image:
    if len(candidates) < 2:
        raise ValueError("numbered anchor review requires multiple candidates")
    base = image.convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for candidate_id, observation in enumerate(candidates, start=1):
        mask = np.asarray(observation.mask, dtype=bool)
        if mask.shape != (height, width) or not mask.any():
            raise ValueError("anchor candidate mask does not match the source frame")
        color = _COLORS[(candidate_id - 1) % len(_COLORS)]
        alpha = Image.fromarray(mask.astype(np.uint8) * 90, mode="L")
        tint = Image.new("RGBA", base.size, (*color, 0))
        tint.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, tint)
        draw = ImageDraw.Draw(overlay)
        ys, xs = np.nonzero(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(*color, 255), width=3)
        label = str(candidate_id)
        label_box = draw.textbbox((x1, y1), label, stroke_width=2)
        draw.rectangle(label_box, fill=(0, 0, 0, 220))
        draw.text(
            (x1, y1),
            label,
            fill=(*color, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
    return Image.alpha_composite(base, overlay).convert("RGB")


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenSam3AnchorSelector:
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

    def select(
        self,
        *,
        frame_path: Path,
        candidates: tuple[BackendMaskObservation, ...],
        entity_phrase: str,
        grounding_prompt: str,
        reference_type: str,
    ) -> MultiInstanceAnchorDecision:
        with Image.open(frame_path) as opened:
            source = opened.convert("RGB")
            source.load()
        evidence = render_numbered_anchor_candidates(source, candidates)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Reference type: {reference_type}\n"
                            f"Annotation phrase: {entity_phrase}\n"
                            f"Grounding prompt: {grounding_prompt}\n"
                            f"Candidate IDs: 1 through {len(candidates)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(evidence)},
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
            "candidate_count": len(candidates),
            "image_count": 1,
            "mode": "qwen_anchor_select_v1",
        }
        try:
            try:
                response = profiled_openai_call(
                    lambda: self.client.chat.completions.create(
                        **parameters,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "v3_sam3_anchor_selection",
                                "strict": True,
                                "schema": (
                                    Sam3AnchorSelectionReview.model_json_schema()
                                ),
                            },
                        },
                    ),
                    component="qwen_sam3_anchor_select",
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
                    component="qwen_sam3_anchor_select",
                    operation="initial",
                    retry_index=0,
                    model=self.config.model,
                    messages=messages,
                    metadata={**metadata, "response_format": "json_object"},
                )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Qwen returned an empty SAM3 anchor selection")
            review = Sam3AnchorSelectionReview.model_validate_json(str(content))
        except Exception as exc:
            raise QwenAnchorSelectionFailure(str(exc)) from exc
        return MultiInstanceAnchorDecision(
            verdict=review.verdict,
            candidate_id=review.selected_candidate_id,
            reason=review.reason,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
