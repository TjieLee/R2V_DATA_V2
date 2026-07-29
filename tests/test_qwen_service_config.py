from __future__ import annotations

from pathlib import Path

import pytest

from r2v_data_v2.config import (
    PipelineConfig,
    QwenConfig,
    QwenServicesConfig,
    config_to_dict,
    load_config,
)


def _write_config(tmp_path: Path, qwen_yaml: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "qwen:\n"
        f"{qwen_yaml}",
        encoding="utf-8",
    )
    return path


def test_flat_qwen_config_maps_to_all_services(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            "  base_url: http://127.0.0.1:8000/v1\n"
            "  model: shared-model\n"
            "  repair_retries: 2\n"
            "  video:\n"
            "    fps: 3.0\n",
        )
    )

    assert isinstance(config.qwen, QwenServicesConfig)
    assert config.qwen.annotation.model == "shared-model"
    assert config.qwen.annotation.video.fps == 3.0
    assert config.qwen.candidate_judge.model == "shared-model"
    assert config.qwen.background_judge.model == "shared-model"
    assert config.qwen.cross_pair_judge.model == "shared-model"
    assert config.qwen.model == "shared-model"
    assert config.qwen.repair_retries == 2


def test_nested_qwen_services_use_independent_models_and_endpoints(
    tmp_path: Path,
) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            "  annotation:\n"
            "    base_url: http://127.0.0.1:8000/v1\n"
            "    model: annotation-model\n"
            "    video:\n"
            "      fps: 2.5\n"
            "  candidate_judge:\n"
            "    base_url: http://127.0.0.1:8001/v1\n"
            "    model: candidate-model\n"
            "  background_judge:\n"
            "    model: background-model\n"
            "  cross_pair_judge:\n"
            "    model: cross-pair-model\n",
        )
    )

    assert isinstance(config.qwen, QwenServicesConfig)
    assert config.qwen.annotation.model == "annotation-model"
    assert config.qwen.annotation.video.fps == 2.5
    assert config.qwen.candidate_judge.base_url == "http://127.0.0.1:8001/v1"
    assert config.qwen.candidate_judge.model == "candidate-model"
    assert config.qwen.background_judge.model == "background-model"
    assert config.qwen.cross_pair_judge.model == "cross-pair-model"
    serialized = config_to_dict(config)["qwen"]
    assert "video" not in serialized["candidate_judge"]


def test_python_flat_qwen_config_remains_compatible(tmp_path: Path) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        qwen=QwenConfig(model="legacy-python-model"),
    )

    assert isinstance(config.qwen, QwenServicesConfig)
    assert config.qwen.model == "legacy-python-model"
    assert config.qwen.candidate_judge.model == "legacy-python-model"


def test_image_judge_rejects_video_configuration(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "  annotation:\n"
        "    model: annotation-model\n"
        "  candidate_judge:\n"
        "    model: candidate-model\n"
        "    video:\n"
        "      fps: 1.0\n",
    )

    with pytest.raises(ValueError, match="image-only"):
        load_config(path)
