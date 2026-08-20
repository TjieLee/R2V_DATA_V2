from __future__ import annotations

import base64
import io
from typing import Any, Literal, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import profiled_openai_call
from r2v_data_v2.v3.schemas import BackgroundRemovalReview

SYSTEM_PROMPT = """You are the semantic quality guard for background removal.

You receive exactly four images in order:
1. the original source image;
2. the locally composited candidate background;
3. the original foreground source mask;
4. the generation mask used for the repaired region.

Judge only visible facts. Do not accept merely because an edit looks generally
plausible. Reject if any original foreground remains, is reconstructed, or is
replaced by another salient entity. Reject any new person, animal, vehicle,
product, text, sign, or other salient object. Reject changes outside the
generation mask. Reject ghosts, double exposure, seams, color or texture breaks,
lighting discontinuities, and visible artifacts. The repaired region must
contain only a continuous extension of the surrounding background.

Return one strict JSON object containing only:
verdict, foreground_absent, foreground_not_reconstructed,
no_new_salient_entity, background_only_in_repaired_region,
background_continuity_ok, no_visible_artifacts, reason.
verdict is accept if and only if every boolean is true. Return JSON only."""

FULL_FRAME_SYSTEM_PROMPT = """You are the semantic quality guard for full-frame background removal.

You receive exactly four images in order:
1. the original source image;
2. the resized full-frame candidate background;
3. the original foreground source mask;
4. the generation mask identifying the intended repaired region.

Judge only visible facts. The candidate is a full-frame image edit, so minor
registration, scale, resampling, interpolation, translation, and photometric differences are
allowed outside the generation mask. Do not reject those minor global changes
by themselves. Reject if any original foreground remains, is reconstructed, or
is duplicated. Reject any new person, animal, vehicle, product, text, sign, or
other salient entity. Reject disappearance or material alteration of unrelated
salient entities that were supposed to remain. Reject material unrelated scene
or layout changes, severe background hallucination, severe artifacts, ghosts,
double exposure, obvious seams, broken structure, duplicated geometry, color
or texture breaks, and lighting discontinuities.
background_only_in_repaired_region means that the substantive content edit is
limited to removing and reconstructing the requested foreground area; any
differences elsewhere are only minor registration, resampling, or photometric
differences and do not materially alter scene content. It does not require
pixel-exact identity outside the generation mask. The result must remain a
coherent background-only version of the source scene.

Return one strict JSON object containing only:
verdict, foreground_absent, foreground_not_reconstructed,
no_new_salient_entity, background_only_in_repaired_region,
background_continuity_ok, no_visible_artifacts, reason.
verdict is accept if and only if every boolean is true. Return JSON only."""

RemovalCandidateMode = Literal["masked_local", "full_frame"]


class BackgroundRemovalJudge(Protocol):
    def review(
        self,
        *,
        source_image: Image.Image,
        candidate_image: Image.Image,
        source_mask: Image.Image,
        generation_mask: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        candidate_mode: RemovalCandidateMode = "masked_local",
    ) -> BackgroundRemovalReview: ...


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenBackgroundRemovalJudge:
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

    def _messages(
        self,
        *,
        source_image: Image.Image,
        candidate_image: Image.Image,
        source_mask: Image.Image,
        generation_mask: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        candidate_mode: RemovalCandidateMode = "masked_local",
    ) -> list[dict[str, object]]:
        if candidate_mode not in {"masked_local", "full_frame"}:
            raise ValueError(f"unsupported removal candidate mode: {candidate_mode}")
        semantics = (
            "Foreground phrases: "
            + "; ".join(removal_phrases)
            + "\nExpected background: "
            + background_phrase
        )
        content: list[dict[str, object]] = [
            {"type": "text", "text": semantics},
        ]
        candidate_label = (
            "Image 2: locally composited candidate"
            if candidate_mode == "masked_local"
            else "Image 2: resized full-frame candidate"
        )
        for label, image in (
            ("Image 1: original source", source_image.convert("RGB")),
            (candidate_label, candidate_image.convert("RGB")),
            ("Image 3: foreground source mask", source_mask.convert("L")),
            ("Image 4: generation mask", generation_mask.convert("L")),
        ):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(image)},
                }
            )
        return [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    if candidate_mode == "masked_local"
                    else FULL_FRAME_SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": content},
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
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
                            "name": "v3_background_removal_review",
                            "strict": True,
                            "schema": BackgroundRemovalReview.model_json_schema(),
                        },
                    },
                ),
                component="qwen_background_remove_judge",
                operation="initial",
                retry_index=0,
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
                component="qwen_background_remove_judge",
                operation="initial",
                retry_index=0,
                model=self.config.model,
                messages=messages,
                metadata={"response_format": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(
                "Qwen returned an empty background removal review"
            )
        return str(content)

    def review(
        self,
        *,
        source_image: Image.Image,
        candidate_image: Image.Image,
        source_mask: Image.Image,
        generation_mask: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        candidate_mode: RemovalCandidateMode = "masked_local",
    ) -> BackgroundRemovalReview:
        if not removal_phrases or not all(
            isinstance(phrase, str) and phrase.strip()
            for phrase in removal_phrases
        ):
            raise ValueError("background removal judge requires removal phrases")
        if not background_phrase.strip():
            raise ValueError("background removal judge requires a background phrase")
        raw = self._request(
            self._messages(
                source_image=source_image,
                candidate_image=candidate_image,
                source_mask=source_mask,
                generation_mask=generation_mask,
                removal_phrases=removal_phrases,
                background_phrase=background_phrase,
                candidate_mode=candidate_mode,
            )
        )
        return BackgroundRemovalReview.model_validate_json(raw)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
