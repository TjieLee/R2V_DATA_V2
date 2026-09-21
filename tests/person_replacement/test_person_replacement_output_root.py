from pathlib import Path

import pytest

from r2v_data_v2.person_replacement.pipeline import (
    PERSON_REPLACEMENT_PRODUCTION_ROOT,
    validate_output_root,
)


def test_formal_person_replacement_production_root_is_the_only_public_dataset_exception():
    root = Path(
        "/mnt/workspace/public/dataset/jea-video/"
        "moive-183t-0808_processed/multi_person_replace"
    )
    assert PERSON_REPLACEMENT_PRODUCTION_ROOT == root
    assert validate_output_root(root) == root
    assert validate_output_root(root / "shard-000000") == root / "shard-000000"


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed",
        "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped",
        "/mnt/workspace/public/pretrained/MiniMaxAI/MiniMax-H3",
        "/mnt/workspace/liutao/X_human_data/something",
    ],
)
def test_other_source_trees_remain_protected(path):
    with pytest.raises(ValueError, match="read-only source tree"):
        validate_output_root(Path(path))
