from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.config import RankingConfig
from r2v_data_v2.visual_embedding import (
    DinoV3Embedder,
    cosine_similarity_matrix,
    l2_normalize_embeddings,
    load_dinov3_embeddings,
    save_dinov3_embeddings,
    save_selected_dinov3_embedding,
    temporal_representation_metrics,
)


def test_embedding_normalization_and_similarity() -> None:
    embeddings = l2_normalize_embeddings(
        np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    )
    similarity = cosine_similarity_matrix(embeddings)

    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0, 1.0])
    assert similarity == pytest.approx(embeddings @ embeddings.T)


def test_stable_cluster_outlier_and_medoid() -> None:
    embeddings = l2_normalize_embeddings(
        np.asarray(
            [
                [1.0, 0.00],
                [1.0, 0.05],
                [1.0, -0.05],
                [0.0, 1.00],
            ],
            dtype=np.float32,
        )
    )
    metrics, medoid_slot, warning = temporal_representation_metrics(
        frame_slots=[0, 1, 2, 3],
        embeddings=embeddings,
        sam_confidences=[0.8, 0.9, 0.8, 1.0],
        base_quality_scores=[0.8, 0.8, 0.8, 1.0],
        threshold=0.70,
    )

    assert warning is None
    assert medoid_slot == 0
    assert {item.frame_slot for item in metrics if item.dino_in_stable_cluster} == {
        0,
        1,
        2,
    }
    outlier = next(item for item in metrics if item.frame_slot == 3)
    assert not outlier.dino_in_stable_cluster
    assert outlier.dino_representativeness == 0.0


def test_no_stable_cluster_keeps_all_candidates() -> None:
    metrics, medoid_slot, warning = temporal_representation_metrics(
        frame_slots=[0, 1],
        embeddings=np.eye(2, dtype=np.float32),
        sam_confidences=[0.8, 0.9],
        base_quality_scores=[0.8, 0.9],
        threshold=0.70,
    )

    assert warning == "no_stable_dino_cluster"
    assert medoid_slot is None
    assert all(not item.dino_in_stable_cluster for item in metrics)
    assert all(item.dino_representativeness == 0.5 for item in metrics)


def test_medoid_cleanup_removes_single_link_chain_member() -> None:
    angles = np.deg2rad([0.0, 40.0, 80.0, 120.0])
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    metrics, medoid_slot, warning = temporal_representation_metrics(
        frame_slots=[0, 1, 2, 3],
        embeddings=embeddings,
        sam_confidences=[0.9, 0.9, 0.9, 0.9],
        base_quality_scores=[0.8, 0.8, 0.8, 0.8],
        threshold=0.70,
    )

    assert warning is None
    assert medoid_slot == 1
    assert {item.frame_slot for item in metrics if item.dino_in_stable_cluster} == {
        0,
        1,
        2,
    }
    chain_tail = next(item for item in metrics if item.frame_slot == 3)
    assert not chain_tail.dino_in_stable_cluster
    assert chain_tail.dino_representativeness == 0.0


def test_embedding_npz_and_selected_embedding_are_float16(tmp_path: Path) -> None:
    path = tmp_path / "dinov3_embeddings.npz"
    embeddings = np.asarray([[3.0, 4.0], [4.0, 3.0]], dtype=np.float32)
    save_dinov3_embeddings(
        path,
        frame_slots=[2, 5],
        embeddings=embeddings,
        model_name="dinov3_vits16",
    )

    slots, loaded, model_name = load_dinov3_embeddings(path)
    assert slots.tolist() == [2, 5]
    assert loaded.dtype == np.float16
    assert model_name == "dinov3_vits16"

    selected_path = tmp_path / "dinov3_embedding.npy"
    save_selected_dinov3_embedding(selected_path, embeddings[0])
    selected = np.load(selected_path, allow_pickle=False)
    assert selected.dtype == np.float16
    assert selected == pytest.approx(np.asarray([0.6, 0.8]), abs=1e-3)


def test_missing_local_dinov3_layout_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="expected either"):
        DinoV3Embedder(
            RankingConfig(
                dinov3_repo_dir=tmp_path / "repo",
                dinov3_model_path=tmp_path / "missing.pth",
            )
        )
