from __future__ import annotations

from pathlib import Path

import pytest

import r2v_data_v2.config as config_module
from r2v_data_v2.config import (
    DinoEvaluatorConfig,
    InpaintingConfig,
    PipelineConfig,
    QwenConfig,
    RankingConfig,
    RankingEvaluatorsConfig,
    SiglipEvaluatorConfig,
    load_config,
)


def test_inpainting_max_sequence_length_above_512_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PipelineConfig, "validate_paths", lambda self: None)
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        qwen=QwenConfig(model="served-model"),
        inpainting=InpaintingConfig(max_sequence_length=513),
    )

    with pytest.raises(
        ValueError,
        match="max_sequence_length must be between 1 and 512",
    ):
        config_module._validate_config(config)


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
                evaluators=RankingEvaluatorsConfig(
                    dinov3=DinoEvaluatorConfig(enabled=True)
                ),
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
                evaluators=RankingEvaluatorsConfig(
                    siglip2=SiglipEvaluatorConfig(enabled=True)
                ),
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
        ranking_yaml=(
            "  evaluators:\n"
            "    dinov3:\n"
            "      enabled: true\n"
            "  dinov3_model_path: null\n"
        ),
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
        ranking_yaml=(
            "  evaluators:\n"
            "    dinov3:\n"
            "      enabled: true\n"
            f"  dinov3_model_path: {missing}\n"
        ),
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
        ranking_yaml=(
            "  evaluators:\n"
            "    dinov3:\n"
            "      enabled: true\n"
            f"  dinov3_model_path: {model_path}\n"
        ),
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
            "  evaluators:\n"
            "    dinov3:\n"
            "      enabled: true\n"
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
        ranking_yaml=(
            "  evaluators:\n"
            "    siglip2:\n"
            "      enabled: true\n"
            f"  siglip2_model_path: {missing}\n"
        ),
    )

    with pytest.raises(FileNotFoundError, match="model directory does not exist"):
        load_config(config_path)


def test_unknown_ranking_metric_is_rejected_during_config_load(
    tmp_path: Path,
) -> None:
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml="  final_weights:\n    unknown_metric: 1.0\n",
    )

    with pytest.raises(ValueError, match="unknown metrics"):
        load_config(config_path)


def test_production_background_inpainting_requires_repair_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    flux_path = model_root / "flux-fill"
    flux_path.mkdir(parents=True)
    config_path = tmp_path / "inpainting.yaml"
    config_path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "qwen:\n"
        "  annotation:\n"
        "    model: annotation-model\n"
        "ranking:\n"
        f"  dinov3_repo_dir: {model_root / 'dinov3'}\n"
        f"  siglip2_model_path: {model_root / 'siglip2'}\n"
        "  evaluators:\n"
        "    dinov3:\n"
        "      enabled: true\n"
        "    siglip2:\n"
        "      enabled: true\n"
        "inpainting:\n"
        "  enabled: true\n"
        "  backend: flux1_fill\n"
        f"  model_path: {flux_path}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"background_hole_fill requires qwen\.repair_judge",
    ):
        load_config(config_path)

    config_path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "qwen:\n"
        "  annotation:\n"
        "    model: annotation-model\n"
        "  repair_judge:\n"
        "    model: consistency-model\n"
        "inpainting:\n"
        "  enabled: true\n"
        "  backend: flux1_fill\n"
        f"  model_path: {flux_path}\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)
    assert loaded.qwen.repair_judge is not None


def test_production_entity_only_config_allows_dino_siglip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", model_root)
    flux_path = model_root / "flux-fill"
    flux_path.mkdir(parents=True)
    dino_path = model_root / "dinov3-model"
    dino_path.mkdir()
    (dino_path / "config.json").write_text("{}", encoding="utf-8")
    (dino_path / "preprocessor_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (model_root / "siglip2").mkdir()
    config_path = tmp_path / "entity-inpainting.yaml"
    config_path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "qwen:\n"
        "  annotation:\n"
        "    model: annotation-model\n"
        "ranking:\n"
        f"  dinov3_repo_dir: {model_root / 'dinov3'}\n"
        f"  dinov3_model_path: {dino_path}\n"
        f"  siglip2_model_path: {model_root / 'siglip2'}\n"
        "  evaluators:\n"
        "    dinov3:\n"
        "      enabled: true\n"
        "    siglip2:\n"
        "      enabled: true\n"
        "inpainting:\n"
        "  enabled: true\n"
        "  backend: flux1_fill\n"
        f"  model_path: {flux_path}\n"
        "  background:\n"
        "    enabled: false\n"
        "  entity:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.qwen.repair_judge is None
    assert loaded.inpainting.background.enabled is False
    assert loaded.inpainting.entity.enabled is True


def test_zero_enabled_ranking_weights_are_rejected(tmp_path: Path) -> None:
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml="  final_weights:\n    sam_confidence: 0.0\n",
    )

    with pytest.raises(ValueError, match="at least one enabled positive weight"):
        load_config(config_path)


def test_invalid_normalization_policy_is_rejected(tmp_path: Path) -> None:
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(
            "  normalization:\n"
            "    sharpness:\n"
            "      method: zscore\n"
        ),
    )

    with pytest.raises(ValueError, match="method must be one of"):
        load_config(config_path)


def test_fixed_range_normalization_requires_ordered_bounds(tmp_path: Path) -> None:
    config_path = _write_visual_model_config(
        tmp_path,
        ranking_yaml=(
            "  normalization:\n"
            "    dino_representativeness:\n"
            "      method: fixed_range\n"
            "      minimum: 0.9\n"
            "      maximum: 0.6\n"
        ),
    )

    with pytest.raises(ValueError, match="minimum < maximum"):
        load_config(config_path)
