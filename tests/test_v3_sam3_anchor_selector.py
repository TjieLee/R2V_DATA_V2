from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.sam3_anchor_selector import (
    QwenSam3AnchorSelector,
    render_numbered_anchor_candidates,
)
from r2v_data_v2.v3.sam3_backend import BackendMaskObservation


class FakeCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload),
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.completions = FakeCompletions(payload)
        self.chat = SimpleNamespace(completions=self.completions)


def _candidates() -> tuple[BackendMaskObservation, ...]:
    left = np.zeros((12, 16), dtype=bool)
    left[2:10, 1:6] = True
    right = np.zeros((12, 16), dtype=bool)
    right[2:10, 10:15] = True
    return (
        BackendMaskObservation(5, left, 0.99, "left"),
        BackendMaskObservation(5, right, 0.51, "right"),
    )


def test_numbered_anchor_evidence_preserves_source_size_and_draws_masks() -> None:
    source = Image.new("RGB", (16, 12), (90, 100, 110))

    rendered = render_numbered_anchor_candidates(source, _candidates())

    assert rendered.size == source.size
    assert rendered.mode == "RGB"
    assert rendered.tobytes() != source.tobytes()


def test_qwen_anchor_selector_sends_semantics_and_returns_exact_candidate(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 12), (90, 100, 110)).save(frame_path)
    client = FakeClient(
        {
            "verdict": "select",
            "selected_candidate_id": 2,
            "reason": "the second numbered mask is the annotated woman",
        }
    )
    selector = QwenSam3AnchorSelector(
        QwenServiceConfig(model="local-qwen"),
        client=client,
    )

    decision = selector.select(
        frame_path=frame_path,
        candidates=_candidates(),
        entity_phrase="the woman in the blue jacket",
        grounding_prompt="woman standing on the right",
        reference_type="subject",
    )

    assert decision.verdict == "select"
    assert decision.candidate_id == 2
    assert len(client.completions.calls) == 1
    request = client.completions.calls[0]
    assert request["temperature"] == 0.0
    assert request["top_p"] == 1.0
    messages = request["messages"]
    user_text = messages[1]["content"][0]["text"]
    assert "Annotation phrase: the woman in the blue jacket" in user_text
    assert "Grounding prompt: woman standing on the right" in user_text
    assert "Reference type: subject" in user_text
    response_schema = request["response_format"]["json_schema"]["schema"]
    assert "selected_candidate_id" in response_schema["properties"]


def test_qwen_anchor_selector_records_profile_component(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 12), (90, 100, 110)).save(frame_path)
    selector = QwenSam3AnchorSelector(
        QwenServiceConfig(model="local-qwen"),
        client=FakeClient(
            {
                "verdict": "select",
                "selected_candidate_id": 1,
                "reason": "candidate 1 is the annotated target",
            }
        ),
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="test")

    with active_profiler(profiler):
        selector.select(
            frame_path=frame_path,
            candidates=_candidates(),
            entity_phrase="the woman in the blue jacket",
            grounding_prompt="woman standing on the left",
            reference_type="subject",
        )

    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["component"] == "qwen_sam3_anchor_select"
    assert events[0]["input_image_count"] == 1
    assert events[0]["metadata"]["candidate_count"] == 2
    assert events[0]["metadata"]["reference_type"] == "subject"
