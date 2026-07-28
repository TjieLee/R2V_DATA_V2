from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.config import RankingConfig


@dataclass(frozen=True)
class TemporalRepresentationMetrics:
    frame_slot: int
    dino_representativeness: float
    dino_nearest_similarity: float
    dino_cluster_id: int
    dino_in_stable_cluster: bool


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must have nonzero L2 norm")
    return np.asarray(values / norms, dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = l2_normalize_embeddings(embeddings)
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


def _connected_components(
    similarity: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    count = similarity.shape[0]
    unseen = set(range(count))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        pending = [root]
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            neighbors = [
                index
                for index in sorted(unseen)
                if similarity[current, index] >= threshold
            ]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                pending.append(neighbor)
        components.append(sorted(component))
    return components


def temporal_representation_metrics(
    *,
    frame_slots: list[int],
    embeddings: np.ndarray,
    sam_confidences: list[float],
    base_quality_scores: list[float],
    threshold: float,
) -> tuple[list[TemporalRepresentationMetrics], int | None, str | None]:
    count = len(frame_slots)
    if not (
        count == len(sam_confidences) == len(base_quality_scores) == len(embeddings)
    ):
        raise ValueError("temporal metric inputs must have equal lengths")
    if count == 0:
        return [], None, "no_stable_dino_cluster"

    similarity = cosine_similarity_matrix(embeddings)
    components = _connected_components(similarity, threshold)
    component_by_index: dict[int, int] = {}
    for cluster_id, component in enumerate(components):
        for index in component:
            component_by_index[index] = cluster_id

    stable_component = max(
        components,
        key=lambda component: (
            len(component),
            float(np.mean([sam_confidences[index] for index in component])),
            float(np.mean([base_quality_scores[index] for index in component])),
            -min(component),
        ),
    )
    has_stable_cluster = len(stable_component) >= 2
    stable_indexes = set(stable_component) if has_stable_cluster else set()
    metrics: list[TemporalRepresentationMetrics] = []
    for index, frame_slot in enumerate(frame_slots):
        if count == 1:
            nearest_similarity = 0.0
        else:
            nearest_similarity = float(np.max(np.delete(similarity[index], index)))
        if index in stable_indexes:
            peers = [peer for peer in stable_component if peer != index]
            representativeness = (
                float(np.mean(similarity[index, peers])) if peers else 0.5
            )
        elif not has_stable_cluster:
            representativeness = 0.5
        else:
            representativeness = 0.0
        metrics.append(
            TemporalRepresentationMetrics(
                frame_slot=frame_slot,
                dino_representativeness=representativeness,
                dino_nearest_similarity=nearest_similarity,
                dino_cluster_id=component_by_index[index],
                dino_in_stable_cluster=index in stable_indexes,
            )
        )

    warning = None if has_stable_cluster else "no_stable_dino_cluster"
    medoid_slot = None
    if has_stable_cluster:
        stable_metrics = [metric for metric in metrics if metric.dino_in_stable_cluster]
        medoid_slot = min(
            stable_metrics,
            key=lambda metric: (
                -metric.dino_representativeness,
                metric.frame_slot,
            ),
        ).frame_slot
    return metrics, medoid_slot, warning


def save_dinov3_embeddings(
    path: Path,
    *,
    frame_slots: list[int],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    normalized = l2_normalize_embeddings(embeddings)
    if len(frame_slots) != len(normalized):
        raise ValueError("frame_slots and embeddings must have equal lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            frame_slots=np.asarray(frame_slots, dtype=np.int32),
            embeddings=normalized.astype(np.float16),
            model_name=np.asarray(model_name),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_dinov3_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["frame_slots"], dtype=np.int32),
            np.asarray(payload["embeddings"], dtype=np.float16),
            str(payload["model_name"].item()),
        )


def save_selected_dinov3_embedding(path: Path, embedding: np.ndarray) -> None:
    normalized = l2_normalize_embeddings(np.asarray(embedding).reshape(1, -1))[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, normalized.astype(np.float16))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DinoV3Embedder:
    def __init__(self, config: RankingConfig) -> None:
        self.config = config
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._layout = ""
        self._device = ""
        self._closed = False
        self._load()

    def _load(self) -> None:
        model_path = self.config.dinov3_model_path
        hf_path = model_path if model_path is not None and model_path.is_dir() else None
        is_huggingface_layout = hf_path is not None and all(
            (hf_path / filename).is_file()
            for filename in ("config.json", "preprocessor_config.json")
        )
        is_torch_hub_layout = (
            (self.config.dinov3_repo_dir / "hubconf.py").is_file()
            and model_path is not None
            and model_path.is_file()
        )
        if not is_huggingface_layout and not is_torch_hub_layout:
            raise FileNotFoundError(
                "unable to resolve a local DINOv3 model; "
                f"repo_dir={self.config.dinov3_repo_dir}; "
                f"model_path={model_path}; "
                f"model_name={self.config.dinov3_model_name}; "
                "expected either model_path/{config.json,preprocessor_config.json} "
                "or repo_dir/hubconf.py plus a checkpoint file at model_path"
            )

        import torch

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if is_huggingface_layout:
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(
                hf_path,
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(
                hf_path,
                local_files_only=True,
            ).to(self._device)
            self._layout = "huggingface"
        else:
            self._model = torch.hub.load(
                repo_or_dir=str(self.config.dinov3_repo_dir),
                model=self.config.dinov3_model_name,
                source="local",
                weights=str(model_path),
            ).to(self._device)
            self._layout = "torch_hub"
        self._model.eval()

    def _hub_tensor(self, images: list[Image.Image]) -> Any:
        arrays = []
        for image in images:
            resized = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            values = np.asarray(resized, dtype=np.float32) / 255.0
            values = (
                values - np.array([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.array(
                [0.229, 0.224, 0.225],
                dtype=np.float32,
            )
            arrays.append(np.transpose(values, (2, 0, 1)))
        return self._torch.from_numpy(np.stack(arrays)).to(self._device)

    def _features_from_output(self, output: Any) -> Any:
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                return output["x_norm_clstoken"]
            for key in ("pooler_output", "last_hidden_state"):
                if key in output:
                    value = output[key]
                    return value[:, 0] if key == "last_hidden_state" else value
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        if getattr(output, "last_hidden_state", None) is not None:
            return output.last_hidden_state[:, 0]
        if hasattr(output, "ndim"):
            return output[:, 0] if output.ndim == 3 else output
        raise RuntimeError("DINOv3 model did not return a global feature")

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("DinoV3Embedder is closed")
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(images), self.config.dinov3_batch_size):
                batch_images = images[start : start + self.config.dinov3_batch_size]
                if self._layout == "huggingface":
                    inputs = self._processor(images=batch_images, return_tensors="pt")
                    inputs = {
                        key: value.to(self._device) for key, value in inputs.items()
                    }
                    output = self._model(**inputs)
                else:
                    tensor = self._hub_tensor(batch_images)
                    output = (
                        self._model.forward_features(tensor)
                        if hasattr(self._model, "forward_features")
                        else self._model(tensor)
                    )
                features = self._features_from_output(output)
                batches.append(features.detach().float().cpu().numpy())
        return l2_normalize_embeddings(np.concatenate(batches, axis=0))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._processor = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
