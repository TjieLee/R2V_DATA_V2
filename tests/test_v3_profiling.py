from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import run_pipeline_v3 as pipeline_module
from r2v_data_v2.v3.profiling import (
    QwenConcurrencyGate,
    V3Profiler,
    _print_report,
    active_profiler,
    profile_model_call,
    profile_stage,
    profiled_openai_call,
    qwen_concurrency_gate,
)


class _Clock:
    def __init__(self) -> None:
        self.value = -1.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def _events(profiler: V3Profiler) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_stage_and_model_call_events_are_aggregated(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc123", clock=_Clock())

    with active_profiler(profiler):
        with profile_stage("segment") as stage:
            stage.set_counters({"processed": 2})
        for retry_index in (0, 1):
            with profile_model_call(
                component="qwen_candidate_judge",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model="qwen",
                input_text_chars=12,
                input_image_count=2,
                metadata={"candidate_count": 1},
            ) as observation:
                if retry_index == 0:
                    observation.prompt_tokens = 10
                    observation.completion_tokens = 4

    summary = profiler.write_summary()

    assert summary["stages"]["segment"] == {
        "calls": 1,
        "total_seconds": 1.0,
        "successes": 1,
    }
    component = summary["components"]["qwen_candidate_judge"]
    assert component["calls"] == 2
    assert component["initial_calls"] == 1
    assert component["repair_calls"] == 1
    assert component["total_seconds"] == 2.0
    assert component["mean_seconds"] == 1.0
    assert component["max_seconds"] == 1.0
    assert component["prompt_tokens_total"] == 10
    assert component["completion_tokens_total"] == 4


def test_stage_exception_is_recorded_and_reraised(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc123")

    with (
        pytest.raises(RuntimeError, match="stage failed"),
        active_profiler(profiler),
        profile_stage("rank"),
    ):
        raise RuntimeError("stage failed")

    event = _events(profiler)[0]
    assert event["stage"] == "rank"
    assert event["success"] is False
    assert event["error_type"] == "RuntimeError"


def test_openai_usage_and_payload_shape_are_observed_without_content(
    tmp_path: Path,
) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc123")
    messages = [
        {"role": "system", "content": "private system instructions"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "short request"},
                {"type": "image_url", "image_url": {"url": "opaque"}},
            ],
        },
    ]
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(
            prompt_tokens=19,
            completion_tokens=3,
            total_tokens=22,
        ),
    )

    with active_profiler(profiler):
        returned = profiled_openai_call(
            lambda: response,
            component="qwen_instruction",
            operation="initial",
            retry_index=0,
            model="qwen",
            messages=messages,
        )

    assert returned is response
    event = _events(profiler)[0]
    assert event["input_image_count"] == 1
    assert event["input_text_chars"] == len(
        "private system instructionsshort request"
    )
    assert event["output_text_chars"] == 2
    assert event["prompt_tokens"] == 19
    assert event["completion_tokens"] == 3
    assert event["total_tokens"] == 22
    serialized = profiler.events_path.read_text(encoding="utf-8")
    assert "private system instructions" not in serialized
    assert "data:image" not in serialized
    assert "base64" not in serialized


def test_missing_openai_usage_is_recorded_as_null(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc123")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
    )

    with active_profiler(profiler):
        profiled_openai_call(
            lambda: response,
            component="qwen_annotation",
            operation="initial",
            retry_index=0,
            model="qwen",
            messages=[],
        )

    event = _events(profiler)[0]
    assert event["prompt_tokens"] is None
    assert event["completion_tokens"] is None
    assert event["total_tokens"] is None


def test_qwen_gate_slot_is_recorded_in_model_profile_metadata(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc123")
    gate = QwenConcurrencyGate(1, lock_directory=tmp_path / "qwen")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
    )

    with active_profiler(profiler), qwen_concurrency_gate(gate):
        profiled_openai_call(
            lambda: response,
            component="qwen_annotation",
            operation="initial",
            retry_index=0,
            model="qwen",
            messages=[],
        )

    event = _events(profiler)[0]
    assert event["metadata"]["qwen_slot"] == 0


def test_profiler_aggregates_qwen_gate_wait_inflight_and_slot_usage(
    tmp_path: Path,
) -> None:
    profiler = V3Profiler(
        tmp_path / "run",
        git_commit="abc123",
        qwen_max_inflight=6,
    )
    with active_profiler(profiler):
        for wait, inflight, slot in (
            (0.0, 1, 0),
            (0.25, 4, 3),
            (0.5, 3, 3),
            (0.75, 5, 5),
        ):
            with profile_model_call(
                component="qwen_annotation",
                operation="initial",
                retry_index=0,
                metadata={
                    "queue_wait_seconds": wait,
                    "qwen_inflight": inflight,
                    "qwen_slot": slot,
                },
            ):
                pass

    summary = profiler.write_summary()

    assert summary["qwen_calls"] == 4
    assert summary["qwen_gate_wait_seconds_total"] == 1.5
    assert summary["qwen_gate_wait_seconds_mean"] == 0.375
    assert summary["qwen_gate_wait_seconds_max"] == 0.75
    assert summary["qwen_max_local_inflight_observed"] == 5
    assert summary["qwen_slot_usage"] == {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 2,
        "4": 0,
        "5": 1,
    }
    assert summary["components"]["qwen_annotation"]["calls"] == 4


class _Storage:
    roots: ClassVar[list[Path]] = []

    def __init__(self, config: object) -> None:
        del config
        self.root = self.roots.pop(0)

    def initialize(self, *, git_commit: str) -> SimpleNamespace:
        return SimpleNamespace(
            run_id="profile-test",
            git_commit=git_commit,
            config_hash="config-hash",
        )


def test_pipeline_profile_flag_is_opt_in_and_result_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    _Storage.roots = [run_root, run_root]
    monkeypatch.setattr(pipeline_module, "load_config", lambda path: object())
    monkeypatch.setattr(pipeline_module, "RunStorage", _Storage)

    without_profile = pipeline_module.run_pipeline_v3(
        config_path=tmp_path / "config.yaml",
        stages=(),
        git_commit="abc123",
    )
    assert not (run_root / "profiling").exists()
    with_profile = pipeline_module.run_pipeline_v3(
        config_path=tmp_path / "config.yaml",
        stages=(),
        git_commit="abc123",
        profile=True,
    )

    assert without_profile == with_profile
    assert (run_root / "profiling" / "events.jsonl").is_file()
    assert (run_root / "profiling" / "summary.json").is_file()


def test_human_readable_report_prints_stage_and_component(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_report(
        {
            "stages": {"segment": {"total_seconds": 12.5}},
            "components": {
                "qwen_candidate_judge": {
                    "calls": 2,
                    "repair_calls": 1,
                    "initial_calls": 1,
                    "total_seconds": 8.0,
                    "mean_seconds": 4.0,
                    "max_seconds": 5.0,
                    "input_images_total": 4,
                    "input_text_chars_total": 20,
                }
            },
        }
    )

    output = capsys.readouterr().out
    assert "===== STAGES =====" in output
    assert "segment" in output
    assert "qwen_candidate_judge" in output
    assert "candidate judge repair rate: 100.00%" in output
