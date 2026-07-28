from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.visual_embedding import l2_normalize_embeddings


@dataclass(frozen=True)
class AlignmentMetrics:
    target_similarity: float
    alignment_margin: float | None
    best_matching_text: str


def alignment_metrics_from_embeddings(
    *,
    image_embeddings: np.ndarray,
    target_embedding: np.ndarray,
    distractor_embeddings: np.ndarray,
    target_text: str,
    distractor_texts: list[str],
) -> list[AlignmentMetrics]:
    images = l2_normalize_embeddings(image_embeddings)
    target = l2_normalize_embeddings(np.asarray(target_embedding).reshape(1, -1))
    if distractor_texts:
        distractors = l2_normalize_embeddings(distractor_embeddings)
        if len(distractors) != len(distractor_texts):
            raise ValueError(
                "distractor embeddings and distractor texts must have equal lengths"
            )
        text_embeddings = np.concatenate([target, distractors], axis=0)
    else:
        text_embeddings = target
    if images.shape[1] != text_embeddings.shape[1]:
        raise ValueError("image and text embedding dimensions must match")

    similarities = images @ text_embeddings.T
    labels = [target_text, *distractor_texts]
    results: list[AlignmentMetrics] = []
    for row in similarities:
        target_similarity = float(row[0])
        alignment_margin = (
            target_similarity - float(np.max(row[1:])) if distractor_texts else None
        )
        results.append(
            AlignmentMetrics(
                target_similarity=target_similarity,
                alignment_margin=alignment_margin,
                best_matching_text=labels[int(np.argmax(row))],
            )
        )
    return results


def is_siglip_wrong_entity(
    metrics: AlignmentMetrics,
    *,
    target_text: str,
    minimum_margin: float = 0.03,
) -> bool:
    return (
        metrics.alignment_margin is not None
        and metrics.best_matching_text != target_text
        and metrics.alignment_margin < 0.0
        and abs(metrics.alignment_margin) >= minimum_margin
    )


class Siglip2Aligner:
    def __init__(self, model_path: Path, batch_size: int) -> None:
        self.model_path = model_path.expanduser()
        self.batch_size = batch_size
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._device = ""
        self._closed = False
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"local SigLIP 2 model directory does not exist: {self.model_path}; "
                "download it explicitly with scripts/download_optional_models.py"
            )
        if batch_size < 1:
            raise ValueError("SigLIP 2 batch_size must be positive")
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()

    def _device_inputs(self, inputs: Any) -> dict[str, Any]:
        return {key: value.to(self._device) for key, value in inputs.items()}

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self._processor(
                    text=texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                features = self._model.get_text_features(**self._device_inputs(inputs))
                batches.append(features.detach().float().cpu().numpy())
        return l2_normalize_embeddings(np.concatenate(batches, axis=0))

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                inputs = self._processor(
                    images=images[start : start + self.batch_size],
                    return_tensors="pt",
                )
                features = self._model.get_image_features(**self._device_inputs(inputs))
                batches.append(features.detach().float().cpu().numpy())
        return l2_normalize_embeddings(np.concatenate(batches, axis=0))

    def score(
        self,
        images: list[Image.Image],
        target_text: str,
        distractor_texts: list[str],
    ) -> list[AlignmentMetrics]:
        if self._closed:
            raise RuntimeError("Siglip2Aligner is closed")
        if not images:
            return []
        texts = [target_text, *distractor_texts]
        text_embeddings = self._encode_texts(texts)
        image_embeddings = self._encode_images(images)
        return alignment_metrics_from_embeddings(
            image_embeddings=image_embeddings,
            target_embedding=text_embeddings[0],
            distractor_embeddings=text_embeddings[1:],
            target_text=target_text,
            distractor_texts=distractor_texts,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._processor = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
