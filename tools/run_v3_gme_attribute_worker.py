#!/usr/bin/env python3
"""Persistent offline GME worker for V3 subject-attribute crop screening."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", required=True)
    return parser


def _nonempty_english(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or not value.isascii():
        raise ValueError(f"{field_name} must be non-empty English text")
    return value.strip()


def _validate_score_request(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("GME worker request must be a JSON object")
    required = {
        "schema_version",
        "type",
        "request_id",
        "input_image_path",
        "instruction",
        "positive_text",
        "negative_texts",
    }
    if set(payload) != required:
        raise ValueError("GME score request must contain exactly the expected fields")
    if payload["schema_version"] != 1 or payload["type"] != "score":
        raise ValueError("unsupported GME worker request")
    _nonempty_english(payload["request_id"], "request_id")
    image_value = payload["input_image_path"]
    if not isinstance(image_value, str) or not Path(image_value).is_absolute():
        raise ValueError("input_image_path must be absolute")
    if not Path(image_value).resolve(strict=True).is_file():
        raise ValueError("input_image_path must be an existing file")
    _nonempty_english(payload["instruction"], "instruction")
    _nonempty_english(payload["positive_text"], "positive_text")
    negatives = payload["negative_texts"]
    if not isinstance(negatives, dict) or not negatives:
        raise ValueError("negative_texts must be a non-empty object")
    for name, text in negatives.items():
        _nonempty_english(name, "negative_text key")
        _nonempty_english(text, "negative_text value")
    return payload


def _load_model(model_path: Path, *, device: str) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(model_path.expanduser().resolve(strict=True)),
        device=device,
        trust_remote_code=True,
        local_files_only=True,
    )


def _as_embedding_matrix(value: object, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite embedding matrix")
    return array


def _score_request(payload: dict[str, Any], *, model: Any) -> dict[str, object]:
    request = _validate_score_request(payload)
    negative_texts = dict(request["negative_texts"])
    text_items = [request["positive_text"], *negative_texts.values()]
    instruction = str(request["instruction"])
    text_embeddings = _as_embedding_matrix(
        model.encode(
            [
                {"text": text, "prompt": instruction}
                for text in text_items
            ],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        field_name="text embeddings",
    )
    image_embeddings = _as_embedding_matrix(
        model.encode(
            [{"image": request["input_image_path"]}],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        field_name="image embeddings",
    )
    if text_embeddings.shape[0] != len(text_items) or image_embeddings.shape[0] != 1:
        raise ValueError("GME worker returned an unexpected embedding count")
    if text_embeddings.shape[1] != image_embeddings.shape[1]:
        raise ValueError("GME text and image embedding dimensions do not match")
    scores = text_embeddings @ image_embeddings[0]
    if not np.isfinite(scores).all():
        raise ValueError("GME similarity scores must be finite")
    negative_scores = {
        name: float(score)
        for name, score in zip(negative_texts, scores[1:], strict=True)
    }
    positive_score = float(scores[0])
    if not math.isfinite(positive_score) or not all(
        math.isfinite(value) for value in negative_scores.values()
    ):
        raise ValueError("GME similarity scores must be finite")
    return {
        "schema_version": 1,
        "type": "response",
        "request_id": request["request_id"],
        "status": "ok",
        "positive_score": positive_score,
        "negative_scores": negative_scores,
    }


def _write_response(output: TextIO, payload: dict[str, object]) -> None:
    output.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    output.flush()


def serve(
    args: argparse.Namespace,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    model_loader: Callable[..., Any] = _load_model,
) -> int:
    if args.device != "cuda:0":
        raise ValueError("GME worker must use cuda:0 inside its isolated process")
    if args.model_name != "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct":
        raise ValueError("GME worker must use the configured 2B model")
    model = model_loader(args.model_path, device=args.device)
    source = input_stream or sys.stdin
    destination = output_stream or sys.stdout
    _write_response(
        destination,
        {"schema_version": 1, "type": "ready", "status": "ok"},
    )
    for line in source:
        request_id: str | None = None
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                raw_request_id = payload.get("request_id")
                if isinstance(raw_request_id, str):
                    request_id = raw_request_id
                if payload.get("type") == "shutdown":
                    _write_response(
                        destination,
                        {
                            "schema_version": 1,
                            "type": "shutdown",
                            "request_id": request_id,
                            "status": "ok",
                        },
                    )
                    return 0
            response = _score_request(payload, model=model)
        except Exception as exc:  # noqa: BLE001 - process boundary response
            response = {
                "schema_version": 1,
                "type": "response",
                "request_id": request_id,
                "status": "error",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        _write_response(destination, response)
    return 0


def main() -> int:
    return serve(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
