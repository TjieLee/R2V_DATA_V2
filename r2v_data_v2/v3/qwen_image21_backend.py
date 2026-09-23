"""Parent-side client for a persistent local Qwen-Image-2.1 GPU worker."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, TextIO

from PIL import Image


@dataclass(frozen=True)
class QwenImage21WorkerConfig:
    python_executable: Path
    model_path: Path
    prompt_enhancer_i2i_path: Path
    device: str = "cuda:0"
    seed: int = 0
    num_inference_steps: int = 40
    timeout_seconds: int = 3600
    cuda_visible_devices: str = "0"
    temporary_root: Path | None = None
    worker_script: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "run_v3_qwen_image21_worker.py"
        )
    )

    def validate(self) -> None:
        for name in (
            "python_executable",
            "model_path",
            "prompt_enhancer_i2i_path",
            "worker_script",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
        if self.temporary_root is not None and (
            not isinstance(self.temporary_root, Path)
            or not self.temporary_root.is_absolute()
        ):
            raise ValueError("temporary_root must be an absolute pathlib.Path")
        if not self.device.strip() or not self.cuda_visible_devices.strip():
            raise ValueError("device and cuda_visible_devices must be non-empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("num_inference_steps", "timeout_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class QwenImage21EditOutput:
    png_bytes: bytes
    original_instruction: str
    rewritten_instruction: str | None
    effective_instruction: str
    worker_metadata: dict[str, Any] = field(default_factory=dict)
    returned_size: tuple[int, int] = (0, 0)


class QwenImage21SubprocessBackend:
    """One JSONL process serves sequential edits until close()."""

    def __init__(self, config: QwenImage21WorkerConfig) -> None:
        config.validate()
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle: TextIO | None = None
        self._stderr_path: Path | None = None

    @property
    def started(self) -> bool:
        return self._process is not None

    def start(self, *, stderr_log_path: Path) -> None:
        if self.started:
            raise RuntimeError("Qwen-Image-2.1 worker is already started")
        stderr_path = stderr_log_path.expanduser().resolve(strict=False)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        config = self.config
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["HF_DATASETS_OFFLINE"] = "1"
        command = [
            str(config.python_executable),
            str(config.worker_script),
            "--serve",
            "--model-path",
            str(config.model_path),
            "--prompt-enhancer-i2i-path",
            str(config.prompt_enhancer_i2i_path),
            "--device",
            config.device,
            "--seed",
            str(config.seed),
            "--num-inference-steps",
            str(config.num_inference_steps),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=config.worker_script.parent.parent,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                bufsize=1,
            )
        except Exception:
            stderr_handle.close()
            raise
        self._process = process
        self._stderr_handle = stderr_handle
        self._stderr_path = stderr_path
        try:
            ready = self._read_response()
            if ready != {"schema_version": 1, "type": "ready", "status": "ok"}:
                raise RuntimeError(f"invalid Qwen-Image-2.1 startup response: {ready}")
        except Exception:
            self._terminate()
            raise

    def edit(
        self,
        *,
        source_rgb: Image.Image,
        instruction: str,
        width: int,
        height: int,
        thinking_enabled: bool,
        instruction_rewrite_enabled: bool | None = None,
        seed: int | None = None,
    ) -> QwenImage21EditOutput:
        if source_rgb.mode != "RGB":
            raise ValueError("Qwen-Image-2.1 source image must be RGB")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if type(thinking_enabled) is not bool:
            raise TypeError("thinking_enabled must be a boolean")
        if (
            instruction_rewrite_enabled is not None
            and type(instruction_rewrite_enabled) is not bool
        ):
            raise TypeError("instruction_rewrite_enabled must be a boolean")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or value <= 0 or value % 16:
                raise ValueError(f"{name} must be a positive multiple of 16")
        if not self.started:
            raise RuntimeError("Qwen-Image-2.1 worker must be started before editing")
        temporary_root = self.config.temporary_root
        if temporary_root is not None:
            temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="r2v-qwen-image21-", dir=temporary_root
        ) as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "source_input_rgb.png"
            output_path = temporary / "candidate.png"
            source_rgb.save(input_path, format="PNG")
            request_id = uuid.uuid4().hex
            request = {
                "schema_version": 1,
                "type": "edit",
                "request_id": request_id,
                "input_image_path": str(input_path),
                "output_image_path": str(output_path),
                "instruction": instruction.strip(),
                "thinking_enabled": thinking_enabled,
                "instruction_rewrite_enabled": bool(instruction_rewrite_enabled),
                "width": width,
                "height": height,
                "seed": self.config.seed if seed is None else seed,
            }
            try:
                self._write_request(request)
                response = self._read_response()
            except Exception:
                self._terminate()
                raise
            if (
                response.get("request_id") != request_id
                or response.get("type") != "response"
            ):
                self._terminate()
                raise RuntimeError("Qwen-Image-2.1 worker response identity mismatch")
            if response.get("status") == "error":
                raise RuntimeError(
                    f"Qwen-Image-2.1 worker rejected request: {response.get('reason', 'generation failed')}"
                )
            if response.get("status") != "ok":
                self._terminate()
                raise RuntimeError("Qwen-Image-2.1 worker returned invalid status")
            if response.get("original_instruction") != instruction.strip():
                raise RuntimeError("Qwen-Image-2.1 worker changed original instruction")
            rewritten = response.get("rewritten_instruction")
            if rewritten is not None and (
                not isinstance(rewritten, str) or not rewritten.strip()
            ):
                raise TypeError("rewritten_instruction must be non-empty text or null")
            effective = response.get("effective_instruction")
            if not isinstance(effective, str) or not effective.strip():
                raise ValueError("Qwen-Image-2.1 worker returned empty instruction")
            if response.get("returned_size") != [width, height]:
                raise RuntimeError(
                    "Qwen-Image-2.1 worker returned unexpected dimensions"
                )
            if not output_path.is_file():
                raise RuntimeError("Qwen-Image-2.1 worker did not publish output")
            png_bytes = output_path.read_bytes()
            with Image.open(BytesIO(png_bytes)) as image:
                image.load()
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (width, height)
                ):
                    raise ValueError(
                        "Qwen-Image-2.1 worker output must be requested RGB PNG"
                    )
            return QwenImage21EditOutput(
                png_bytes=png_bytes,
                original_instruction=instruction.strip(),
                rewritten_instruction=rewritten,
                effective_instruction=effective.strip(),
                worker_metadata={
                    key: value
                    for key, value in response.items()
                    if key
                    not in {
                        "original_instruction",
                        "rewritten_instruction",
                        "effective_instruction",
                    }
                },
                returned_size=(width, height),
            )

    def _write_request(self, request: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("Qwen-Image-2.1 worker stdin is unavailable")
        process.stdin.write(
            json.dumps(request, ensure_ascii=True, sort_keys=True) + "\n"
        )
        process.stdin.flush()

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Qwen-Image-2.1 worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.config.timeout_seconds):
                raise TimeoutError(
                    f"Qwen-Image-2.1 worker timed out after {self.config.timeout_seconds}s"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError(
                "Qwen-Image-2.1 worker exited without response: "
                f"returncode={process.poll()}, stderr_log={self._stderr_path}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen-Image-2.1 worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise TypeError("Qwen-Image-2.1 worker response must be a JSON object")
        return response

    def close(self) -> None:
        if not self.started:
            return
        request_id = uuid.uuid4().hex
        try:
            self._write_request(
                {"schema_version": 1, "type": "shutdown", "request_id": request_id}
            )
            response = self._read_response()
            if response != {
                "schema_version": 1,
                "type": "shutdown",
                "request_id": request_id,
                "status": "ok",
            }:
                raise RuntimeError(
                    f"invalid Qwen-Image-2.1 shutdown response: {response}"
                )
            assert self._process is not None
            self._process.wait(timeout=self.config.timeout_seconds)
            if self._process.returncode != 0:
                raise RuntimeError(
                    f"Qwen-Image-2.1 worker exited with code {self._process.returncode}"
                )
        except Exception:
            self._terminate()
            raise
        finally:
            self._close_pipes()

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._close_pipes()

    def _close_pipes(self) -> None:
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        self._process = None
        self._stderr_handle = None
