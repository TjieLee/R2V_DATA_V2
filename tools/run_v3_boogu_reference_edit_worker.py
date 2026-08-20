"""Persistent Linux CUDA JSONL worker for V3 Boogu reference editing."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_edit_request(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("worker request must be a JSON object")
    required = {
        "schema_version",
        "type",
        "request_id",
        "input_image_path",
        "output_image_path",
        "instruction",
        "thinking_enabled",
        "width",
        "height",
    }
    allowed = required | {"instruction_rewrite_enabled", "seed"}
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing or unknown:
        raise ValueError(
            f"invalid worker request keys: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != 1 or payload["type"] != "edit":
        raise ValueError("unsupported worker request")
    _nonempty(payload["request_id"], "request_id")
    for name in ("input_image_path", "output_image_path"):
        value = payload[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    _nonempty(payload["instruction"], "instruction")
    if not isinstance(payload["thinking_enabled"], bool):
        raise TypeError("thinking_enabled must be a boolean")
    for name in ("width", "height"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value % 16:
            raise ValueError(f"{name} must be aligned to 16 pixels")
    request_seed = payload.get("seed")
    if request_seed is not None and (
        not isinstance(request_seed, int)
        or isinstance(request_seed, bool)
        or request_seed < 0
    ):
        raise ValueError("seed must be a non-negative integer when supplied")
    rewrite_enabled = payload.get("instruction_rewrite_enabled")
    if rewrite_enabled is not None and not isinstance(rewrite_enabled, bool):
        raise TypeError("instruction_rewrite_enabled must be a boolean")
    return payload


def _first_instruction(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0].strip() or None
    return None


def _load_rewrite_result(
    path: Path,
    original_instruction: str,
) -> tuple[str | None, str]:
    if not path.is_file():
        raise RuntimeError(
            "Boogu thinking was enabled but no rewrite metadata was saved"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Boogu rewrite metadata must be a JSON object")
    original = _first_instruction(payload.get("ori_instruction"))
    if original != original_instruction:
        raise RuntimeError("Boogu rewrite metadata changed the original instruction")
    rewritten = _first_instruction(payload.get("rewritten_instruction"))
    if rewritten is None:
        raise RuntimeError("Boogu thinking produced no rewritten instruction")
    return rewritten, rewritten


def _load_pipeline(
    *, code_root: Path, model_path: Path, device: str
) -> tuple[Any, Any]:
    resolved_code_root = code_root.expanduser().resolve(strict=True)
    resolved_model_path = model_path.expanduser().resolve(strict=True)
    sys.path.insert(0, str(resolved_code_root))
    os.environ["device"] = device
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from boogu.pipelines.boogu.pipeline_boogu_turbo import (
            BooguImageTurboPipeline,
        )

        pipeline = BooguImageTurboPipeline.from_pretrained(
            str(resolved_model_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        pipeline.to(device)
        pipeline.devices_manager(
            instant_rewriter_device=device,
            user_set_pipe_device=device,
            user_set_rewriter_device=device,
            execution_device=device,
            unload_rewriter_level="keep",
        )
    return pipeline, torch


def _run_loaded_request(
    payload: dict[str, Any],
    *,
    pipeline: Any,
    torch_module: Any,
    device: str,
    seed: int,
    model_name: str,
    model_revision: str,
) -> dict[str, Any]:
    request = _validate_edit_request(payload)
    input_path = Path(request["input_image_path"]).resolve(strict=True)
    output_path = Path(request["output_image_path"]).resolve(strict=False)
    width = int(request["width"])
    height = int(request["height"])
    instruction = str(request["instruction"]).strip()
    thinking_enabled = bool(request["thinking_enabled"])
    instruction_rewrite_enabled = bool(
        request.get("instruction_rewrite_enabled", thinking_enabled)
    )
    effective_seed = int(request.get("seed", seed))
    with Image.open(input_path) as loaded:
        loaded.load()
        if loaded.mode != "RGB":
            raise ValueError(f"Boogu worker input must be RGB, got {loaded.mode}")
        source_rgb = loaded.copy()

    rewrite_path = output_path.parent / (
        f"rewritten_instruction_{request['request_id']}.json"
    )
    with contextlib.redirect_stdout(sys.stderr):
        result = pipeline(
            instruction=[instruction],
            input_images=[[source_rgb]],
            input_image_paths=None,
            negative_instruction="",
            width=width,
            height=height,
            device=device,
            rewriter_device=device,
            unload_rewriter_level="keep",
            enable_inner_devices_manager=False,
            align_res=False,
            max_input_image_pixels=2048 * 2048,
            max_input_image_side_length=2048 * 2,
            num_inference_steps=4,
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.0,
            generator=torch_module.Generator(device).manual_seed(effective_seed),
            use_rewrite_text_instruction=instruction_rewrite_enabled,
            merge_original_and_rewritten_instructions=True,
            save_rewritten_instruction=instruction_rewrite_enabled,
            save_rewritten_instruction_path=(
                str(rewrite_path) if instruction_rewrite_enabled else None
            ),
            output_type="pil",
        )
    images = getattr(result, "images", None)
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("Boogu worker expected exactly one generated image")
    candidate = images[0]
    if not isinstance(candidate, Image.Image):
        raise TypeError("Boogu worker output must be a PIL image")
    if candidate.mode != "RGB":
        raise ValueError(f"Boogu worker output must be RGB, got {candidate.mode}")
    if candidate.size != (width, height):
        raise ValueError(
            "Boogu worker output dimensions do not match request: "
            f"requested_size={(width, height)}, returned_size={candidate.size}"
        )

    rewritten: str | None = None
    effective = instruction
    if instruction_rewrite_enabled:
        rewritten, effective = _load_rewrite_result(rewrite_path, instruction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(output_path, format="PNG")
    rewrite_path.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "type": "response",
        "request_id": request["request_id"],
        "status": "ok",
        "model_name": model_name,
        "model_revision": model_revision,
        "original_instruction": instruction,
        "rewritten_instruction": rewritten,
        "effective_instruction": effective,
        "thinking_enabled": thinking_enabled,
        "instruction_rewrite_enabled": instruction_rewrite_enabled,
        "requested_size": [width, height],
        "returned_size": [candidate.width, candidate.height],
        "seed": effective_seed,
        "num_inference_steps": 4,
    }


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility helper for unit tests; the CLI uses the persistent server."""

    required = {
        "code_root",
        "model_path",
        "model_name",
        "model_revision",
        "device",
        "seed",
        "input_image_path",
        "output_image_path",
        "instruction",
        "thinking_enabled",
        "width",
        "height",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"legacy worker request missing keys: {missing}")
    pipeline, torch_module = _load_pipeline(
        code_root=Path(payload["code_root"]),
        model_path=Path(payload["model_path"]),
        device=str(payload["device"]),
    )
    request = {
        "schema_version": 1,
        "type": "edit",
        "request_id": "legacy_request",
        "input_image_path": payload["input_image_path"],
        "output_image_path": payload["output_image_path"],
        "instruction": payload["instruction"],
        "thinking_enabled": payload["thinking_enabled"],
        "width": payload["width"],
        "height": payload["height"],
    }
    response = _run_loaded_request(
        request,
        pipeline=pipeline,
        torch_module=torch_module,
        device=str(payload["device"]),
        seed=int(payload["seed"]),
        model_name=str(payload["model_name"]),
        model_revision=str(payload["model_revision"]),
    )
    response.pop("type")
    response.pop("request_id")
    response.pop("status")
    result_path = payload.get("result_path")
    if isinstance(result_path, str):
        Path(result_path).write_text(
            json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return response


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    sys.stdout.flush()


def serve(args: argparse.Namespace) -> int:
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    pipeline, torch_module = _load_pipeline(
        code_root=args.code_root,
        model_path=args.model_path,
        device=args.device,
    )
    _write_response({"schema_version": 1, "type": "ready", "status": "ok"})
    for line in sys.stdin:
        request_id: str | None = None
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                raw_request_id = payload.get("request_id")
                if isinstance(raw_request_id, str):
                    request_id = raw_request_id
                if payload.get("type") == "shutdown":
                    _write_response(
                        {
                            "schema_version": 1,
                            "type": "shutdown",
                            "request_id": _nonempty(request_id, "request_id"),
                            "status": "ok",
                        }
                    )
                    return 0
            response = _run_loaded_request(
                payload,
                pipeline=pipeline,
                torch_module=torch_module,
                device=args.device,
                seed=args.seed,
                model_name=args.model_name,
                model_revision=args.model_revision,
            )
        except Exception as exc:  # noqa: BLE001 - process boundary response
            response = {
                "schema_version": 1,
                "type": "response",
                "request_id": request_id,
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        _write_response(response)
    return 0


def main() -> int:
    args = _parser().parse_args()
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
