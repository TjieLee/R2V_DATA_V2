from __future__ import annotations

import json
import math
import os
import selectors
import subprocess
import tempfile
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ContextManager, TextIO

from PIL import Image

from r2v_data_v2.v3.config import (
    SUBJECT_ATTRIBUTE_GME_BACKEND,
    SubjectAttributeGmeConfig,
)

TEXT_TO_IMAGE_INSTRUCTION = "Find an image that best matches the given description."

_ATTRIBUTE_COMPONENT_DESCRIPTION = {
    "face": "a face with recognizable facial structure and useful appearance details",
    "hair": "a coherent hairstyle with recognizable structure and useful appearance details",
    "headwear": "headwear with recognizable item structure and useful appearance details",
    "glasses": "glasses with recognizable item structure and useful appearance details",
    "upper_clothing": (
        "upper-body clothing with recognizable garment structure and useful "
        "appearance details"
    ),
    "lower_clothing": (
        "lower-body clothing with recognizable garment structure and useful "
        "appearance details"
    ),
    "dress_or_skirt": (
        "a dress or skirt with recognizable garment structure and useful "
        "appearance details"
    ),
    "shoes": "shoes with recognizable item structure and useful appearance details",
    "bag": "a bag with recognizable item structure and useful appearance details",
    "accessory": (
        "a wearable accessory with recognizable component structure and useful "
        "appearance details"
    ),
}

_GENERAL_NEGATIVE_DESCRIPTIONS = {
    "owner_body": (
        "An arbitrary crop dominated by a person's body or silhouette rather than "
        "a distinct wearable or appearance attribute."
    ),
    "background": (
        "Background or unrelated scene content rather than the intended attribute."
    ),
    "generic_fragment": (
        "An unrecognizable fragment, narrow strip, isolated edge, or disconnected "
        "partial piece with insufficient component structure."
    ),
}

_TYPE_NEGATIVE_DESCRIPTIONS = {
    "hair": (
        "Thin hair strands, fringe, contour, or a small hair edge without a "
        "coherent hairstyle region."
    ),
    "face": (
        "A small isolated facial patch without enough facial structure to function "
        "as a face appearance reference."
    ),
    "upper_clothing": (
        "An isolated sleeve, cuff, shoulder, hem, trouser edge, or narrow garment "
        "fragment without coherent garment structure."
    ),
    "lower_clothing": (
        "An isolated sleeve, cuff, shoulder, hem, trouser edge, or narrow garment "
        "fragment without coherent garment structure."
    ),
    "dress_or_skirt": (
        "An isolated sleeve, cuff, shoulder, hem, trouser edge, or narrow garment "
        "fragment without coherent garment structure."
    ),
}


def build_positive_description(*, phrase: str, attribute_type: str) -> str:
    normalized_phrase = phrase.strip()
    if not normalized_phrase:
        raise ValueError("GME positive description requires an attribute phrase")
    component = _ATTRIBUTE_COMPONENT_DESCRIPTION.get(attribute_type)
    if component is None:
        raise ValueError(f"unsupported GME attribute type: {attribute_type}")
    description = (
        f"A clear isolated view of {normalized_phrase}. "
        f"The main visible component is {component}."
    )
    _require_english_query(description)
    return description


def build_negative_descriptions(attribute_type: str) -> dict[str, str]:
    if attribute_type not in _ATTRIBUTE_COMPONENT_DESCRIPTION:
        raise ValueError(f"unsupported GME attribute type: {attribute_type}")
    descriptions = dict(_GENERAL_NEGATIVE_DESCRIPTIONS)
    specific = _TYPE_NEGATIVE_DESCRIPTIONS.get(attribute_type)
    if specific is not None:
        descriptions[f"{attribute_type}_fragment"] = specific
    for description in descriptions.values():
        _require_english_query(description)
    return descriptions


def _require_english_query(value: str) -> None:
    if not value.strip() or not value.isascii():
        raise ValueError("GME semantic queries must be non-empty English text")


@dataclass(frozen=True)
class GmeRelativeMarginResult:
    positive_score: float
    negative_scores: dict[str, float]
    max_negative_score: float
    margin: float
    passed: bool
    reason: str


def relative_margin_result(
    *,
    positive_score: float,
    negative_scores: dict[str, float],
    min_margin: float,
) -> GmeRelativeMarginResult:
    values = [positive_score, min_margin, *negative_scores.values()]
    if not negative_scores:
        raise ValueError("GME scoring requires at least one negative score")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("GME scores and margin threshold must be finite numbers")
    positive = float(positive_score)
    negatives = {key: float(value) for key, value in negative_scores.items()}
    maximum = max(negatives.values())
    margin = positive - maximum
    passed = margin >= float(min_margin)
    return GmeRelativeMarginResult(
        positive_score=positive,
        negative_scores=negatives,
        max_negative_score=maximum,
        margin=margin,
        passed=passed,
        reason="gme_relative_margin_pass" if passed else "gme_semantic_quality_reject",
    )


@dataclass(frozen=True)
class GmeAttributeWorkerConfig:
    python_executable: Path
    model_path: Path
    model_name: str
    cuda_visible_devices: str
    timeout_seconds: int
    temporary_root: Path
    stderr_log_path: Path
    device: str = "cuda:0"
    worker_script: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "run_v3_gme_attribute_worker.py"
        )
    )

    def validate(self, *, allowed_root: Path) -> None:
        root = allowed_root.expanduser().resolve(strict=False)
        for name, path in (
            ("python_executable", self.python_executable),
            ("model_path", self.model_path),
            ("temporary_root", self.temporary_root),
            ("stderr_log_path", self.stderr_log_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"GME {name} must be an absolute pathlib.Path")
            resolved = path.expanduser().resolve(strict=False)
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"GME {name} must remain inside allowed_root")
        if self.device != "cuda:0":
            raise ValueError("GME worker device must be cuda:0")
        if not self.cuda_visible_devices.isdigit():
            raise ValueError("GME worker requires one physical CUDA index")
        if not self.model_name.strip():
            raise ValueError("GME model_name must not be empty")
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds < 1
        ):
            raise ValueError("GME timeout_seconds must be a positive integer")


class PersistentGmeAttributeScreener:
    """One offline GME model process shared by all attribute screening calls."""

    def __init__(
        self,
        config: SubjectAttributeGmeConfig,
        worker_config: GmeAttributeWorkerConfig,
        *,
        allowed_root: Path,
        inference_lock: threading.Lock | None = None,
    ) -> None:
        if not config.enabled or config.backend != SUBJECT_ATTRIBUTE_GME_BACKEND:
            raise ValueError("persistent GME screener requires enabled GME config")
        worker_config.validate(allowed_root=allowed_root)
        self.config = config
        self.worker_config = worker_config
        self._process: subprocess.Popen[str] | None = None
        self._stderr: TextIO | None = None
        self._request_lock = threading.Lock()
        self._inference_lock = inference_lock
        self.starts = 0

    @classmethod
    def from_runtime(
        cls,
        config: SubjectAttributeGmeConfig,
        *,
        physical_gpu: str,
        run_root: Path,
        allowed_root: Path,
        inference_lock: threading.Lock | None = None,
    ) -> PersistentGmeAttributeScreener:
        root = run_root.expanduser().resolve(strict=False)
        return cls(
            config,
            GmeAttributeWorkerConfig(
                python_executable=config.python_executable.expanduser().resolve(),
                model_path=config.model_path.expanduser().resolve(),
                model_name=config.model_name,
                cuda_visible_devices=physical_gpu,
                timeout_seconds=config.timeout_seconds,
                temporary_root=root / "subject_attributes" / "gme_tmp",
                stderr_log_path=root / "logs" / "subject_attributes_gme.stderr.log",
            ),
            allowed_root=allowed_root,
            inference_lock=inference_lock,
        )

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("GME worker is already started")
        worker = self.worker_config
        worker.temporary_root.mkdir(parents=True, exist_ok=True)
        worker.stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = worker.stderr_log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "CUDA_VISIBLE_DEVICES": worker.cuda_visible_devices,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            str(worker.python_executable),
            str(worker.worker_script),
            "--serve",
            "--model-path",
            str(worker.model_path),
            "--model-name",
            worker.model_name,
            "--device",
            worker.device,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
            )
            ready = self._read_response()
            if ready != {"schema_version": 1, "status": "ok", "type": "ready"}:
                raise RuntimeError(f"invalid GME worker startup response: {ready}")
            self.starts += 1
        except Exception:
            self._terminate()
            raise

    def screen(
        self,
        *,
        crop: Image.Image,
        phrase: str,
        attribute_type: str,
    ) -> GmeRelativeMarginResult:
        if self._process is None:
            raise RuntimeError("GME worker must be started before screening")
        positive = build_positive_description(
            phrase=phrase,
            attribute_type=attribute_type,
        )
        negatives = build_negative_descriptions(attribute_type)
        with tempfile.TemporaryDirectory(
            prefix="r2v-gme-",
            dir=self.worker_config.temporary_root,
        ) as temporary_name:
            image_path = Path(temporary_name) / "attribute_crop.png"
            crop.convert("RGBA").save(image_path, format="PNG")
            request_id = uuid.uuid4().hex
            request = {
                "schema_version": 1,
                "type": "score",
                "request_id": request_id,
                "input_image_path": str(image_path),
                "instruction": TEXT_TO_IMAGE_INSTRUCTION,
                "positive_text": positive,
                "negative_texts": negatives,
            }
            lock_context: ContextManager[object] = (
                self._inference_lock
                if self._inference_lock is not None
                else nullcontext()
            )
            with self._request_lock, lock_context:
                self._write_request(request)
                response = self._read_response()
        if response.get("request_id") != request_id:
            self._terminate()
            raise RuntimeError("GME worker response request_id mismatch")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("reason") or "GME scoring failed"))
        raw_negative_scores = response.get("negative_scores")
        if not isinstance(raw_negative_scores, dict):
            raise TypeError("GME worker omitted negative scores")
        return relative_margin_result(
            positive_score=response.get("positive_score"),  # type: ignore[arg-type]
            negative_scores=raw_negative_scores,  # type: ignore[arg-type]
            min_margin=self.config.min_margin,
        )

    def _write_request(self, request: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("GME worker request pipe is unavailable")
        process.stdin.write(json.dumps(request, ensure_ascii=True, sort_keys=True) + "\n")
        process.stdin.flush()

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("GME worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.worker_config.timeout_seconds):
                raise TimeoutError(
                    "GME worker timed out after "
                    f"{self.worker_config.timeout_seconds}s"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError(
                "GME worker exited without a response: "
                f"returncode={process.poll()}"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise TypeError("GME worker response must be a JSON object")
        return response

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        request_id = uuid.uuid4().hex
        try:
            with self._request_lock:
                self._write_request(
                    {
                        "schema_version": 1,
                        "type": "shutdown",
                        "request_id": request_id,
                    }
                )
                response = self._read_response()
            if response != {
                "schema_version": 1,
                "type": "shutdown",
                "request_id": request_id,
                "status": "ok",
            }:
                raise RuntimeError(f"invalid GME worker shutdown response: {response}")
            process.wait(timeout=10)
        finally:
            self._terminate()

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


class UnavailableGmeAttributeScreener:
    """Fail-open adapter that preserves per-candidate startup failure metrics."""

    def __init__(
        self,
        config: SubjectAttributeGmeConfig,
        failure: BaseException,
    ) -> None:
        self.config = config
        self._reason = f"{type(failure).__name__}:{failure}"

    def screen(
        self,
        *,
        crop: Image.Image,
        phrase: str,
        attribute_type: str,
    ) -> GmeRelativeMarginResult:
        del crop, phrase, attribute_type
        raise RuntimeError(f"GME worker startup failed: {self._reason}")


__all__ = [
    "GmeAttributeWorkerConfig",
    "GmeRelativeMarginResult",
    "PersistentGmeAttributeScreener",
    "TEXT_TO_IMAGE_INSTRUCTION",
    "UnavailableGmeAttributeScreener",
    "build_negative_descriptions",
    "build_positive_description",
    "relative_margin_result",
]
