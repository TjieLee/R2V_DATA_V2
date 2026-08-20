from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import profiled_openai_call
from r2v_data_v2.v3.schemas import BackgroundReferenceState, FinalBackgroundReview

if TYPE_CHECKING:
    from r2v_data_v2.v3.storage import RunStorage


SYSTEM_PROMPT = """You judge whether an image is a reusable background reference.

You receive the full candidate, four spatial tiles, and the expected background
description.

Accept only if the scene matches the expected background, no unexpected
foreground subject is visible, the image contains useful background information,
and there are no obvious artifacts.

A foreground subject is a discrete person, animal, vehicle, equipment, product,
or similar instance that is not naturally part of the described environment.
Small, partial, distant, dark, or edge-of-frame subjects still count.

Do not reject plausible environmental elements merely because the description
does not enumerate them. Reject fades, nearly black frames, severe obstruction,
or wrong scenes.

Reject clearly visible non-diegetic screen-space overlays, including subtitles,
opening or ending credits, watermarks, channel or platform logos, captions, UI
overlays, and other text or graphics composited over the video frame. These are
not reusable scene content. Treat them as no_obvious_artifacts=false and
therefore verdict=reject.

Do not reject text physically present in the depicted environment, such as
writing painted on a wall, plaques, shop signs, books, or banners, unless it
dominates the image or otherwise makes the background unusable.

Judge visible facts only. Return strict JSON only."""

TILE_NAMES = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
)


@dataclass(frozen=True)
class FinalBackgroundReviewAttempt:
    review: FinalBackgroundReview
    raw_response: str


class FinalBackgroundJudgeFailure(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class FinalBackgroundJudge(Protocol):
    def review(
        self,
        *,
        image: Image.Image,
        background_phrase: str,
        background_grounding_prompt: str,
        background_status: str,
    ) -> FinalBackgroundReviewAttempt: ...


def deterministic_background_tiles(image: Image.Image) -> list[Image.Image]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        raise ValueError("background guard image must be at least 2x2 pixels")
    split_x = width // 2
    split_y = height // 2
    boxes = (
        (0, 0, split_x, split_y),
        (split_x, 0, width, split_y),
        (0, split_y, split_x, height),
        (split_x, split_y, width, height),
    )
    return [rgb.crop(box) for box in boxes]


def load_final_background_image(
    storage: RunStorage,
    *,
    clip_uid: str,
    background: BackgroundReferenceState,
) -> Image.Image:
    if background.status not in {"clean_raw", "ready_removed"}:
        raise ValueError("final background guard requires a ready background")
    if background.output_image_path is None:
        raise ValueError("ready background is missing output_image_path")
    artifact = Path(background.output_image_path).expanduser()
    if ".." in artifact.parts:
        raise ValueError("background output_image_path cannot contain path traversal")
    root = storage.root.resolve(strict=True)
    if not artifact.is_absolute() and artifact.parts[:1] == ("frames",):
        candidate = storage.clip_dir(clip_uid) / artifact
    else:
        candidate = artifact if artifact.is_absolute() else root / artifact
    path = candidate.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("background output image must remain inside run_root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"background output image is missing: {path}")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenFinalBackgroundJudge:
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
        image: Image.Image,
        background_phrase: str,
        background_grounding_prompt: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Expected background phrase: {background_phrase}\n"
                    "Expected background grounding prompt: "
                    f"{background_grounding_prompt}"
                ),
            }
        ]
        evidence = (("full", image.convert("RGB")),) + tuple(
            zip(TILE_NAMES, deterministic_background_tiles(image), strict=True)
        )
        for label, evidence_image in evidence:
            content.append({"type": "text", "text": f"Image: {label}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(evidence_image)},
                }
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _request(
        self,
        messages: list[dict[str, object]],
        *,
        background_status: str,
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
            "background_status": background_status,
            "image_count": 5,
            "tile_count": 4,
            "mode": "qwen_v1",
        }
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_final_background_review",
                            "strict": True,
                            "schema": FinalBackgroundReview.model_json_schema(),
                        },
                    },
                ),
                component="qwen_background_final_guard",
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
                component="qwen_background_final_guard",
                operation="initial",
                retry_index=0,
                model=self.config.model,
                messages=messages,
                metadata={**metadata, "response_format": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty final background review")
        return str(content)

    def review(
        self,
        *,
        image: Image.Image,
        background_phrase: str,
        background_grounding_prompt: str,
        background_status: str,
    ) -> FinalBackgroundReviewAttempt:
        if not background_phrase.strip() or not background_grounding_prompt.strip():
            raise FinalBackgroundJudgeFailure(
                "final background guard requires annotation background semantics"
            )
        raw: str | None = None
        try:
            raw = self._request(
                self._messages(
                    image=image,
                    background_phrase=background_phrase,
                    background_grounding_prompt=background_grounding_prompt,
                ),
                background_status=background_status,
            )
            review = FinalBackgroundReview.model_validate_json(raw)
        except FinalBackgroundJudgeFailure:
            raise
        except Exception as exc:
            raise FinalBackgroundJudgeFailure(
                str(exc),
                raw_response=raw,
            ) from exc
        return FinalBackgroundReviewAttempt(review=review, raw_response=raw)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
