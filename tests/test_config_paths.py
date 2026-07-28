from __future__ import annotations

from pathlib import Path

import pytest

import r2v_data_v2.config as config_module
from r2v_data_v2.config import (
    PipelineConfig,
    QwenConfig,
    RankingConfig,
    load_config,
)


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
        "ranking": RankingConfig(dinov3_repo_dir=pretrained / "dinov3"),
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


def test_dinov3_paths_must_use_allowed_model_roots(tmp_path: Path) -> None:
    common = {
        "dataset_json": tmp_path / "source.jsonl",
        "output_root": tmp_path / "output",
        "qwen": QwenConfig(model="served-model-name"),
    }
    with pytest.raises(ValueError, match="ranking.dinov3_model_path must be inside"):
        PipelineConfig(
            **common,
            ranking=RankingConfig(
                dinov3_enabled=True,
                dinov3_repo_dir=config_module.ALLOWED_PRETRAINED_ROOT / "dinov3",
                dinov3_model_path=tmp_path / "unapproved" / "model.pth",
            ),
        ).validate_paths()


def test_siglip2_path_must_use_allowed_model_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ranking.siglip2_model_path must be inside"):
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=tmp_path / "output",
            qwen=QwenConfig(model="served-model-name"),
            ranking=RankingConfig(
                siglip2_enabled=True,
                siglip2_model_path=tmp_path / "unapproved" / "siglip2",
            ),
        ).validate_paths()


def _write_visual_model_config(
    tmp_path: Path,
    *,
    ranking_yaml: str,
) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "qwen:\n"
        "  model: served-model-name\n"
        "ranking:\n"
        f"{ranking_yaml}",
        encoding="utf-8",
    )
    return path


def test_enabled_dinov3_requires_explicit_model_path(tmp_path: Path) -> None:
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml="  dinov3_enabled: true\n  dinov3_model_path: null\n",
    )

    with pytest.raises(ValueError, match="dinov3_model_path is required"):
        load_config(config_path)


def test_enabled_dinov3_rejects_missing_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    missing = model_root / "missing.pth"
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(f"  dinov3_enabled: true\n  dinov3_model_path: {missing}\n"),
    )

    with pytest.raises(FileNotFoundError, match="model path does not exist"):
        load_config(config_path)


def test_enabled_dinov3_accepts_complete_local_hf_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    model_path = model_root / "dinov3-hf"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(f"  dinov3_enabled: true\n  dinov3_model_path: {model_path}\n"),
    )

    assert load_config(config_path).ranking.dinov3_model_path == model_path


def test_enabled_dinov3_accepts_local_torch_hub_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    repo = model_root / "dinov3"
    repo.mkdir(parents=True)
    (repo / "hubconf.py").write_text("", encoding="utf-8")
    checkpoint = model_root / "dinov3_vits16.pth"
    checkpoint.write_bytes(b"checkpoint")
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(
            "  dinov3_enabled: true\n"
            f"  dinov3_repo_dir: {repo}\n"
            f"  dinov3_model_path: {checkpoint}\n"
        ),
    )

    loaded = load_config(config_path)
    assert loaded.ranking.dinov3_model_path == checkpoint
    assert loaded.ranking.dinov3_repo_dir == repo


def test_enabled_siglip2_rejects_missing_local_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    missing = model_root / "missing-siglip2"
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(f"  siglip2_enabled: true\n  siglip2_model_path: {missing}\n"),
    )

    with pytest.raises(FileNotFoundError, match="model directory does not exist"):
        load_config(config_path)
