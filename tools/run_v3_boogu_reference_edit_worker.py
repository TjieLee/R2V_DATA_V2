"""Linux CUDA worker for one Boogu reference-edit request.

This entrypoint intentionally imports torch and Boogu only after validating the
request. The parent process invokes this file with the dedicated Boogu Python;
no shell or environment activation is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    return parser


def _load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("worker request must be a JSON object")
    required = {
        "schema_version",
        "code_root",
        "model_path",
        "model_name",
        "model_revision",
        "device",
        "seed",
        "input_image_path",
        "output_image_path",
        "result_path",
        "instruction",
        "thinking_enabled",
        "width",
        "height",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        raise ValueError(
            f"invalid worker request keys: missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported worker request schema_version")
    for name in ("code_root", "model_path", "input_image_path", "output_image_path", "result_path"):
        value = payload[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    for name in ("model_name", "model_revision", "device", "instruction"):
        value = payload[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(payload["thinking_enabled"], bool):
        raise TypeError("thinking_enabled must be a boolean")
    for name in ("width", "height"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value % 16:
            raise ValueError(f"{name} must be aligned to 16 pixels")
    seed = payload["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return payload


def _first_instruction(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0].strip() or None
    return None


def _load_rewrite_result(path: Path, original_instruction: str) -> tuple[str | None, str]:
    if not path.is_file():
        raise RuntimeError("Boogu thinking was enabled but no rewrite metadata was saved")
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


def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    code_root = Path(payload["code_root"]).resolve(strict=True)
    model_path = Path(payload["model_path"]).resolve(strict=True)
    input_path = Path(payload["input_image_path"]).resolve(strict=True)
    output_path = Path(payload["output_image_path"]).resolve(strict=False)
    result_path = Path(payload["result_path"]).resolve(strict=False)
    device = str(payload["device"])
    width = int(payload["width"])
    height = int(payload["height"])
    instruction = str(payload["instruction"]).strip()
    thinking_enabled = bool(payload["thinking_enabled"])

    sys.path.insert(0, str(code_root))
    os.environ["device"] = device

    import torch
    from boogu.pipelines.boogu.pipeline_boogu_turbo import (
        BooguImageTurboPipeline,
    )

    pipeline = BooguImageTurboPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    pipeline.to(device)
    with Image.open(input_path) as loaded:
        loaded.load()
        if loaded.mode != "RGB":
            raise ValueError(f"Boogu worker input must be RGB, got {loaded.mode}")
        source_rgb = loaded.copy()

    rewrite_path = output_path.parent / "rewritten_instruction.json"
    result = pipeline(
        instruction=[instruction],
        input_images=[[source_rgb]],
        input_image_paths=None,
        negative_instruction="",
        width=width,
        height=height,
        align_res=True,
        max_input_image_pixels=2048 * 2048,
        max_input_image_side_length=2048 * 2,
        num_inference_steps=4,
        text_guidance_scale=1.0,
        image_guidance_scale=1.0,
        empty_instruction_guidance_scale=0.0,
        use_dmd_student_inference=True,
        dmd_conditioning_sigma=0.0,
        generator=torch.Generator(device).manual_seed(int(payload["seed"])),
        use_rewrite_text_instruction=thinking_enabled,
        merge_original_and_rewritten_instructions=True,
        save_rewritten_instruction=thinking_enabled,
        save_rewritten_instruction_path=(
            str(rewrite_path) if thinking_enabled else None
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
    if thinking_enabled:
        rewritten, effective = _load_rewrite_result(rewrite_path, instruction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(output_path, format="PNG")
    response = {
        "schema_version": 1,
        "model_name": payload["model_name"],
        "model_revision": payload["model_revision"],
        "original_instruction": instruction,
        "rewritten_instruction": rewritten,
        "effective_instruction": effective,
        "thinking_enabled": thinking_enabled,
        "requested_size": [width, height],
        "returned_size": [candidate.width, candidate.height],
        "seed": payload["seed"],
        "num_inference_steps": 4,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return response


def main() -> int:
    args = _parser().parse_args()
    request = _load_request(args.request)
    run_request(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
