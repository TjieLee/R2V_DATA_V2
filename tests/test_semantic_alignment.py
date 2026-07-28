from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.semantic_alignment import (
    Siglip2Aligner,
    alignment_metrics_from_embeddings,
    is_siglip_wrong_entity,
)


def _metrics(
    image: np.ndarray,
    *,
    with_distractor: bool = True,
):
    return alignment_metrics_from_embeddings(
        image_embeddings=image.reshape(1, -1),
        target_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        distractor_embeddings=(
            np.asarray([[0.0, 1.0]], dtype=np.float32)
            if with_distractor
            else np.empty((0, 2), dtype=np.float32)
        ),
        target_text="gray-haired man",
        distractor_texts=["wine glass"] if with_distractor else [],
    )[0]


def test_target_prompt_wins_alignment() -> None:
    metrics = _metrics(np.asarray([1.0, 0.1], dtype=np.float32))

    assert metrics.best_matching_text == "gray-haired man"
    assert metrics.alignment_margin is not None
    assert metrics.alignment_margin > 0
    assert not is_siglip_wrong_entity(metrics, target_text="gray-haired man")


def test_clear_distractor_match_is_hard_wrong_entity() -> None:
    metrics = _metrics(np.asarray([0.1, 1.0], dtype=np.float32))

    assert metrics.best_matching_text == "wine glass"
    assert metrics.alignment_margin is not None
    assert metrics.alignment_margin <= -0.03
    assert is_siglip_wrong_entity(metrics, target_text="gray-haired man")


def test_single_entity_has_no_wrong_entity_hard_reject() -> None:
    metrics = _metrics(
        np.asarray([-1.0, 0.0], dtype=np.float32),
        with_distractor=False,
    )

    assert metrics.alignment_margin is None
    assert not is_siglip_wrong_entity(metrics, target_text="gray-haired man")


def test_missing_local_siglip2_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download it explicitly"):
        Siglip2Aligner(tmp_path / "missing", batch_size=8)
