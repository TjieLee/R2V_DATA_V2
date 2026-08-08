from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from PIL import Image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local lightweight V3 reference audit adapter",
    )
    parser.add_argument(
        "--kind",
        choices=("quality", "embedding", "subject_pose"),
        required=True,
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    return parser


def _load_adapter(arguments: argparse.Namespace) -> object:
    code_root = arguments.code_root.expanduser().resolve(strict=True)
    model_path = arguments.model_path.expanduser().resolve(strict=True)
    if not code_root.is_dir():
        raise ValueError("reference filter code root must be a directory")
    sys.path.insert(0, str(code_root))
    import r2v_reference_filter_adapter as adapter_module

    loader = getattr(adapter_module, "load_scorer", None)
    if not callable(loader):
        raise TypeError("adapter must define load_scorer(kind, backend, model_path)")
    scorer = loader(
        kind=arguments.kind,
        backend=arguments.backend,
        model_path=model_path,
        local_files_only=True,
    )
    eval_method = getattr(scorer, "eval", None)
    if callable(eval_method):
        eval_method()
    return scorer


def _inference_context() -> AbstractContextManager[Any]:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _mapping_result(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("reference filter adapter result must be an object")
    return dict(value)


def _call_scorer(scorer: object, kind: str, image: Image.Image) -> dict[str, object]:
    method_name = {
        "quality": "score",
        "embedding": "embed",
        "subject_pose": "inspect",
    }[kind]
    method = getattr(scorer, method_name, None)
    if not callable(method):
        raise TypeError(f"reference filter adapter must define {method_name}()")
    started = time.monotonic()
    with _inference_context():
        result = _mapping_result(method(image))
    result.setdefault("runtime_seconds", time.monotonic() - started)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    scorer = _load_adapter(arguments)
    try:
        for line in sys.stdin:
            request_id: object = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("worker request must be an object")
                request_id = request.get("request_id")
                if request.get("shutdown") is True:
                    return 0
                encoded = request.get("image_png_hex")
                if not isinstance(encoded, str):
                    raise TypeError("worker request image_png_hex must be a string")
                with Image.open(io.BytesIO(bytes.fromhex(encoded))) as opened:
                    opened.load()
                    image = opened.convert("RGB")
                response = {
                    "request_id": request_id,
                    "status": "ok",
                    "result": _call_scorer(scorer, arguments.kind, image),
                }
            except Exception as exc:  # noqa: BLE001 - isolate one audit request
                response = {
                    "request_id": request_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        return 0
    finally:
        close_method = getattr(scorer, "close", None)
        if callable(close_method):
            close_method()


if __name__ == "__main__":
    raise SystemExit(main())
