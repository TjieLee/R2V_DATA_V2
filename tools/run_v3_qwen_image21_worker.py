"""Persistent local Qwen-Image-2.1 image-edit worker (one process per GPU)."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from PIL import Image


@dataclass(frozen=True)
class LoadedModels:
    pipeline: Any
    prompt_enhancer: Any
    processor: Any
    system_prompt: str
    torch_module: Any


def load_models(
    *, model_path: Path, prompt_enhancer_i2i_path: Path, device: str
) -> LoadedModels:
    """Load both local models once, before emitting the ready response."""
    system_prompt = (
        (prompt_enhancer_i2i_path / "system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if not system_prompt:
        raise ValueError("PE-I2I system_prompt.txt must not be empty")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from diffusers import QwenImage21Pipeline
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(prompt_enhancer_i2i_path), local_files_only=True
        )
        prompt_enhancer = (
            AutoModelForImageTextToText.from_pretrained(
                str(prompt_enhancer_i2i_path),
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        pipeline = QwenImage21Pipeline.from_pretrained(
            str(model_path), dtype=torch.bfloat16, local_files_only=True
        ).to(device)
    return LoadedModels(pipeline, prompt_enhancer, processor, system_prompt, torch)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_request(payload: object) -> dict[str, Any]:
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
    missing = sorted(required - payload.keys())
    unknown = sorted(payload.keys() - allowed)
    if missing or unknown:
        raise ValueError(
            f"invalid worker request keys: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != 1 or payload["type"] != "edit":
        raise ValueError("unsupported worker request")
    _nonempty(payload["request_id"], "request_id")
    _nonempty(payload["instruction"], "instruction")
    for name in ("input_image_path", "output_image_path"):
        value = payload[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if type(payload["thinking_enabled"]) is not bool:
        raise TypeError("thinking_enabled must be a boolean")
    rewrite = payload.get("instruction_rewrite_enabled", False)
    if type(rewrite) is not bool:
        raise TypeError("instruction_rewrite_enabled must be a boolean")
    for name in ("width", "height"):
        if _positive_int(payload[name], name) % 16:
            raise ValueError(f"{name} must be aligned to 16 pixels")
    seed = payload.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    return payload


def _rewrite_instruction(
    *, runtime: LoadedModels, source_rgb: Image.Image, instruction: str, seed: int
) -> tuple[str, dict[str, str]]:
    if source_rgb.width * source_rgb.height > 1024 * 1024:
        scale = ((1024 * 1024) / (source_rgb.width * source_rgb.height)) ** 0.5
        pe_image = source_rgb.resize(
            (
                max(1, int(source_rgb.width * scale)),
                max(1, int(source_rgb.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        pe_image = source_rgb
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": runtime.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pe_image},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    inputs = runtime.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    ).to(runtime.prompt_enhancer.device)
    if "mm_token_type_ids" not in inputs and hasattr(
        runtime.processor, "create_mm_token_type_ids"
    ):
        inputs["mm_token_type_ids"] = runtime.processor.create_mm_token_type_ids(
            inputs["input_ids"]
        )
    runtime.torch_module.manual_seed(seed)
    with runtime.torch_module.no_grad(), contextlib.redirect_stdout(sys.stderr):
        tokens = runtime.prompt_enhancer.generate(
            **inputs,
            max_new_tokens=24000,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            pad_token_id=runtime.processor.tokenizer.eos_token_id,
        )
    prompt_length = len(inputs["input_ids"][0])
    decoded = runtime.processor.tokenizer.decode(
        tokens[0][prompt_length:], skip_special_tokens=True
    )
    return parse_pe_answer(decoded)


def parse_pe_answer(decoded: str) -> tuple[str, dict[str, str]]:
    """Extract the final well-formed PE edit JSON after its thinking block."""
    if "</think>" in decoded:
        answer = decoded.partition("</think>")[2]
    elif "<think>" in decoded:
        raise ValueError("PE-I2I thinking block did not finish")
    else:
        answer = decoded
    decoder = json.JSONDecoder()
    for position in reversed([i for i, char in enumerate(answer) if char == "{"]):
        try:
            parsed, _ = decoder.raw_decode(answer[position:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        rewritten = parsed.get("rewritten_prompt") or parsed.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return rewritten.strip(), {
            "pe_wh_ratio": str(parsed.get("wh_ratio") or "").strip(),
            "pe_ratio_follow": str(parsed.get("ratio_follow") or "").strip(),
        }
    raise ValueError("PE-I2I output has no valid rewritten_prompt JSON")


def run_loaded_request(
    payload: object,
    *,
    runtime: LoadedModels,
    device: str,
    default_seed: int,
    num_inference_steps: int,
) -> dict[str, Any]:
    request = _validate_request(payload)
    input_path = Path(request["input_image_path"])
    output_path = Path(request["output_image_path"])
    instruction = _nonempty(request["instruction"], "instruction")
    width = request["width"]
    height = request["height"]
    seed = request.get("seed", default_seed)
    with Image.open(input_path) as loaded:
        loaded.load()
        if loaded.mode != "RGB":
            raise ValueError("Qwen-Image-2.1 input must be RGB")
        source_rgb = loaded.copy()
    rewritten: str | None = None
    pe_metadata: dict[str, str] = {}
    if request.get("instruction_rewrite_enabled", False):
        rewritten, pe_metadata = _rewrite_instruction(
            runtime=runtime,
            source_rgb=source_rgb,
            instruction=instruction,
            seed=seed,
        )
    effective = rewritten or instruction
    with runtime.torch_module.no_grad(), contextlib.redirect_stdout(sys.stderr):
        result = runtime.pipeline(
            prompt=effective,
            image=source_rgb,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=runtime.torch_module.Generator(device).manual_seed(seed),
        )
    images = getattr(result, "images", None)
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("Qwen-Image-2.1 must return one image")
    candidate = images[0]
    if not isinstance(candidate, Image.Image):
        raise TypeError("Qwen-Image-2.1 output must be a PIL image")
    if candidate.size != (width, height):
        raise ValueError("Qwen-Image-2.1 output dimensions changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.convert("RGB").save(output_path, format="PNG")
    return {
        "schema_version": 1,
        "type": "response",
        "request_id": request["request_id"],
        "status": "ok",
        "backend": "qwen_image_2_1",
        "original_instruction": instruction,
        "rewritten_instruction": rewritten,
        "effective_instruction": effective,
        "thinking_enabled": request["thinking_enabled"],
        "instruction_rewrite_enabled": request.get(
            "instruction_rewrite_enabled", False
        ),
        "requested_size": [width, height],
        "returned_size": [width, height],
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        **pe_metadata,
    }


def _write_response(payload: dict[str, Any], outgoing: TextIO) -> None:
    outgoing.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    outgoing.flush()


def serve(
    args: argparse.Namespace,
    *,
    incoming: TextIO | None = None,
    outgoing: TextIO | None = None,
) -> int:
    incoming = incoming if incoming is not None else sys.stdin
    outgoing = outgoing if outgoing is not None else sys.stdout
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    _positive_int(args.num_inference_steps, "num_inference_steps")
    runtime = load_models(
        model_path=args.model_path,
        prompt_enhancer_i2i_path=args.prompt_enhancer_i2i_path,
        device=args.device,
    )
    _write_response({"schema_version": 1, "type": "ready", "status": "ok"}, outgoing)
    for line in incoming:
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
                        },
                        outgoing,
                    )
                    return 0
            response = run_loaded_request(
                payload,
                runtime=runtime,
                device=args.device,
                default_seed=args.seed,
                num_inference_steps=args.num_inference_steps,
            )
        except Exception as exc:  # noqa: BLE001 - report per-request failures over the process boundary
            response = {
                "schema_version": 1,
                "type": "response",
                "request_id": request_id,
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        _write_response(response, outgoing)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-enhancer-i2i-path", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    return serve(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
