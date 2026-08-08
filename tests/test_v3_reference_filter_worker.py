from __future__ import annotations

import io
import json
import sys

import pytest
from PIL import Image

from tools import run_v3_reference_filter_worker as worker


class ClosableScorer:
    def __init__(self, *, fail_first_inspection: bool = False) -> None:
        self.inspect_calls = 0
        self.close_calls = 0
        self.fail_first_inspection = fail_first_inspection

    def inspect(self, image: Image.Image) -> dict[str, object]:
        assert image.mode == "RGB"
        self.inspect_calls += 1
        if self.fail_first_inspection and self.inspect_calls == 1:
            raise RuntimeError("temporary inspection failure")
        return {"face_detected": False}

    def close(self) -> None:
        self.close_calls += 1


class ScorerWithoutClose:
    def inspect(self, image: Image.Image) -> dict[str, object]:
        assert image.mode == "RGB"
        return {"face_detected": False}


def _arguments() -> list[str]:
    return [
        "--kind",
        "subject_pose",
        "--backend",
        "fake",
        "--code-root",
        "adapter",
        "--model-path",
        "models",
    ]


def _image_hex() -> str:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue().hex()


def _run_worker(
    monkeypatch: pytest.MonkeyPatch,
    scorer: object,
    requests: list[dict[str, object] | str],
) -> list[dict[str, object]]:
    stdin = io.StringIO(
        "".join(
            value + "\n"
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":")) + "\n"
            for value in requests
        )
    )
    stdout = io.StringIO()
    monkeypatch.setattr(worker, "_load_adapter", lambda _arguments: scorer)
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert worker.main(_arguments()) == 0

    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_worker_closes_scorer_once_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ClosableScorer()

    responses = _run_worker(
        monkeypatch,
        scorer,
        [{"request_id": "shutdown", "shutdown": True}],
    )

    assert responses == []
    assert scorer.close_calls == 1


def test_worker_closes_scorer_once_on_stdin_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ClosableScorer()

    responses = _run_worker(monkeypatch, scorer, [])

    assert responses == []
    assert scorer.close_calls == 1


def test_worker_keeps_scorer_open_after_request_error_and_closes_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = ClosableScorer(fail_first_inspection=True)

    responses = _run_worker(
        monkeypatch,
        scorer,
        [
            {"request_id": "bad", "image_png_hex": _image_hex()},
            {"request_id": "good", "image_png_hex": _image_hex()},
            {"request_id": "shutdown", "shutdown": True},
        ],
    )

    assert [response["status"] for response in responses] == ["failed", "ok"]
    assert "temporary inspection failure" in str(responses[0]["error"])
    assert scorer.inspect_calls == 2
    assert scorer.close_calls == 1


def test_worker_accepts_scorer_without_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _run_worker(
        monkeypatch,
        ScorerWithoutClose(),
        [
            {"request_id": "good", "image_png_hex": _image_hex()},
            {"request_id": "shutdown", "shutdown": True},
        ],
    )

    assert [response["status"] for response in responses] == ["ok"]
