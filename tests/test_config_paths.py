from __future__ import annotations

from pathlib import Path

import pytest

import r2v_data_v2.config as config_module
from r2v_data_v2.config import PipelineConfig, QwenConfig


def test_output_and_dataset_must_stay_inside_allowed_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed_output = tmp_path / "workspace" / "data"
    allowed_dataset = tmp_path / "public" / "dataset"
    monkeypatch.setattr(config_module, "ALLOWED_OUTPUT_ROOT", allowed_output.resolve())
    monkeypatch.setattr(
        config_module,
        "ALLOWED_DATASET_ROOT",
        allowed_dataset.resolve(),
    )
    valid = PipelineConfig(
        dataset_json=allowed_dataset / "source.jsonl",
        output_root=allowed_output / "run",
        qwen=QwenConfig(model="served-model-name"),
    )
    assert valid.ensure_output_root() == (allowed_output / "run").resolve()

    outside_output = PipelineConfig(
        dataset_json=allowed_dataset / "source.jsonl",
        output_root=tmp_path / "outside-output",
        qwen=QwenConfig(model="served-model-name"),
    )
    with pytest.raises(ValueError, match="output_root must be inside"):
        outside_output.ensure_output_root()

    outside_dataset = PipelineConfig(
        dataset_json=tmp_path / "outside-source.jsonl",
        output_root=allowed_output / "run",
        qwen=QwenConfig(model="served-model-name"),
    )
    with pytest.raises(ValueError, match="dataset_json must be inside"):
        outside_dataset.ensure_output_root()


def test_absolute_local_qwen_model_must_use_pretrained_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrained = tmp_path / "public" / "pretrained"
    monkeypatch.setattr(
        config_module,
        "ALLOWED_PRETRAINED_ROOT",
        pretrained.resolve(),
    )
    common = {
        "dataset_json": tmp_path / "source.jsonl",
        "output_root": tmp_path / "output",
    }
    PipelineConfig(
        **common,
        qwen=QwenConfig(model=str(pretrained / "Qwen" / "model")),
    ).validate_paths()
    PipelineConfig(
        **common,
        qwen=QwenConfig(model="served-model-name"),
    ).validate_paths()

    with pytest.raises(ValueError, match="local qwen.model must be inside"):
        PipelineConfig(
            **common,
            qwen=QwenConfig(model=str(tmp_path / "private-model")),
        ).validate_paths()
