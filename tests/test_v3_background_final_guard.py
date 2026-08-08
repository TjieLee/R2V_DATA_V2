from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.v3.background_final_guard import (
    SYSTEM_PROMPT,
    TILE_NAMES,
    QwenFinalBackgroundJudge,
    deterministic_background_tiles,
    load_final_background_image,
)
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.schemas import FinalBackgroundReview


def _review(*, accepted: bool = True) -> FinalBackgroundReview:
    return FinalBackgroundReview(
        verdict="accept" if accepted else "reject",
        background_matches_description=True,
        no_unexpected_foreground_subject=accepted,
        usable_background_information=True,
        no_obvious_artifacts=True,
        reason="usable scene" if accepted else "unexpected foreground",
    )


class _Completions:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=90,
                completion_tokens=20,
                total_tokens=110,
            ),
        )


def _judge(completions: _Completions) -> QwenFinalBackgroundJudge:
    return QwenFinalBackgroundJudge(
        QwenServiceConfig(model="local-qwen"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        ),
    )


def _data_url_image(value: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert value.startswith(prefix)
    with Image.open(io.BytesIO(base64.b64decode(value[len(prefix) :]))) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _background(*, status: str, output_image_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        output_image_path=output_image_path,
    )


def _storage(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        clip_dir=lambda clip_uid: root / "clips" / clip_uid,
    )


def test_load_final_background_resolves_clean_raw_frames_from_clip_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    frame_path = root / "clips" / "clip-1" / "frames" / "07.jpg"
    frame_path.parent.mkdir(parents=True)
    Image.new("RGB", (7, 5), (20, 40, 60)).save(frame_path, format="JPEG")

    image = load_final_background_image(
        _storage(root),
        clip_uid="clip-1",
        background=_background(
            status="clean_raw",
            output_image_path="frames/07.jpg",
        ),
    )

    assert image.size == (7, 5)
    assert image.mode == "RGB"


def test_load_final_background_resolves_ready_removed_from_run_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    output_path = root / "clips" / "clip-1" / "selected" / "bg_removed.png"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (9, 6), (30, 50, 70)).save(output_path, format="PNG")

    image = load_final_background_image(
        _storage(root),
        clip_uid="clip-1",
        background=_background(
            status="ready_removed",
            output_image_path="clips/clip-1/selected/bg_removed.png",
        ),
    )

    assert image.size == (9, 6)
    assert image.mode == "RGB"


def test_load_final_background_fails_when_artifact_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="background output image is missing"):
        load_final_background_image(
            _storage(root),
            clip_uid="clip-1",
            background=_background(
                status="clean_raw",
                output_image_path="frames/07.jpg",
            ),
        )


def test_load_final_background_rejects_traversal_and_outside_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (4, 4), "white").save(outside, format="PNG")

    with pytest.raises(ValueError, match="path traversal"):
        load_final_background_image(
            _storage(root),
            clip_uid="clip-1",
            background=_background(
                status="clean_raw",
                output_image_path="frames/../../../outside.png",
            ),
        )
    with pytest.raises(ValueError, match="remain inside run_root"):
        load_final_background_image(
            _storage(root),
            clip_uid="clip-1",
            background=_background(
                status="ready_removed",
                output_image_path=str(outside),
            ),
        )


def test_final_background_prompt_stays_short_and_generic() -> None:
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", SYSTEM_PROMPT)
    assert len(words) <= 130
    assert len(words) <= 160
    lowered = SYSTEM_PROMPT.casefold()
    assert "forbidden foreground entities" not in lowered
    assert "forbidden list" not in lowered


@pytest.mark.parametrize(
    "changes",
    [
        {"verdict": "reject"},
        {"no_obvious_artifacts": False},
        {"reason": " "},
        {"extra": True},
    ],
)
def test_final_background_review_rejects_inconsistent_output(
    changes: dict[str, object],
) -> None:
    payload = {**_review().model_dump(), **changes}
    with pytest.raises(ValidationError):
        FinalBackgroundReview.model_validate(payload)


def test_deterministic_tiles_cover_the_full_image_in_fixed_order() -> None:
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    image = Image.fromarray(pixels, mode="RGB")

    tiles = deterministic_background_tiles(image)

    assert TILE_NAMES == (
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    )
    assert [tile.size for tile in tiles] == [(2, 1), (3, 1), (2, 2), (3, 2)]
    reconstructed = np.concatenate(
        (
            np.concatenate((np.asarray(tiles[0]), np.asarray(tiles[1])), axis=1),
            np.concatenate((np.asarray(tiles[2]), np.asarray(tiles[3])), axis=1),
        ),
        axis=0,
    )
    assert np.array_equal(reconstructed, pixels)


def test_qwen_evidence_contains_exactly_full_image_and_four_tiles() -> None:
    completions = _Completions(_review().model_dump_json())
    judge = _judge(completions)
    image = Image.new("RGB", (8, 6), (10, 20, 30))

    attempt = judge.review(
        image=image,
        background_phrase="a quiet stone courtyard",
        background_grounding_prompt="the courtyard and its enclosing walls",
        background_status="clean_raw",
    )

    assert attempt.review.verdict == "accept"
    assert len(completions.calls) == 1
    call = completions.calls[0]
    messages = call["messages"]
    assert isinstance(messages, list)
    user_content = messages[1]["content"]
    images = [item for item in user_content if item["type"] == "image_url"]
    labels = [
        item["text"]
        for item in user_content
        if item["type"] == "text" and str(item["text"]).startswith("Image:")
    ]
    assert labels == [
        "Image: full",
        "Image: upper_left",
        "Image: upper_right",
        "Image: lower_left",
        "Image: lower_right",
    ]
    assert len(images) == 5
    decoded = [
        _data_url_image(item["image_url"]["url"])
        for item in images
    ]
    assert [item.size for item in decoded] == [
        (8, 6),
        (4, 3),
        (4, 3),
        (4, 3),
        (4, 3),
    ]
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "a quiet stone courtyard" in serialized
    assert "the courtyard and its enclosing walls" in serialized
    for forbidden in (
        "annotation entity phrase list",
        "forbidden entity list",
        "source mask",
        "generation mask",
    ):
        assert forbidden not in serialized.casefold()


def test_qwen_final_guard_profiles_five_images_and_component_metadata(
    tmp_path: Path,
) -> None:
    completions = _Completions(_review().model_dump_json())
    judge = _judge(completions)
    profiler = V3Profiler(tmp_path / "profile", git_commit="test")

    with active_profiler(profiler):
        judge.review(
            image=Image.new("RGB", (8, 6), (30, 40, 50)),
            background_phrase="an open plaza",
            background_grounding_prompt="the open paved plaza",
            background_status="ready_removed",
        )

    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    event = events[0]
    assert event["component"] == "qwen_background_final_guard"
    assert event["input_image_count"] == 5
    assert event["metadata"]["background_status"] == "ready_removed"
    assert event["metadata"]["image_count"] == 5
    assert event["metadata"]["tile_count"] == 4
    assert event["metadata"]["mode"] == "qwen_v1"
