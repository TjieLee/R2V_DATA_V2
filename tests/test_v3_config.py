from __future__ import annotations

from pathlib import Path

import pytest

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import load_config


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_judge: bool,
    background_remove_judge: bool,
    pair_enabled: bool = True,
    remove_enabled: bool = True,
) -> Path:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for directory in (writable, dataset_root, pretrained, user_models):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)

    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    model = pretrained / "Qwen" / "judge"
    qwen_lines = [
        "qwen:",
        "  annotation:",
        f"    model: {model}",
        "  instruction_writer:",
        f"    model: {model}",
    ]
    if candidate_judge:
        qwen_lines.extend(
            [
                "  candidate_judge:",
                f"    model: {model}",
            ]
        )
    if background_remove_judge:
        qwen_lines.extend(
            [
                "  background_remove_judge:",
                f"    model: {model}",
            ]
        )

    path = tmp_path / "v3.yaml"
    path.write_text(
        "\n".join(
            [
                f"dataset_json: {dataset_json}",
                f"run_root: {writable / 'runs' / 'production-gates'}",
                f"export_root: {writable / 'datasets' / 'production-gates'}",
                "source:",
                "  limit: 1",
                *qwen_lines,
                "pair:",
                f"  enabled: {str(pair_enabled).lower()}",
                "remove:",
                f"  enabled: {str(remove_enabled).lower()}",
                f"  base_model_path: {pretrained / 'Qwen' / 'edit'}",
                f"  adapter_path: {user_models / 'object-remover'}",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_requires_background_remove_judge_when_remove_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "qwen.background_remove_judge is required when "
            "remove.enabled is true"
        ),
    ):
        load_config(path)


def test_load_config_requires_candidate_judge_when_pair_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=False,
        background_remove_judge=True,
    )

    with pytest.raises(
        ValueError,
        match="qwen.candidate_judge is required when pair.enabled is true",
    ):
        load_config(path)


def test_disabled_pair_does_not_require_candidate_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=False,
        background_remove_judge=True,
        pair_enabled=False,
    )

    assert load_config(path).qwen.candidate_judge is None


def test_disabled_remove_does_not_require_background_remove_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=False,
        remove_enabled=False,
    )

    assert load_config(path).qwen.background_remove_judge is None


def test_candidate_judge_does_not_substitute_for_background_remove_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=False,
    )

    with pytest.raises(ValueError, match="qwen.background_remove_judge"):
        load_config(path)


def test_instruction_writer_does_not_substitute_for_candidate_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=False,
        background_remove_judge=True,
    )

    with pytest.raises(ValueError, match="qwen.candidate_judge"):
        load_config(path)


def test_explicit_judges_may_share_one_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
    )

    config = load_config(path)

    assert config.qwen.candidate_judge is not None
    assert config.qwen.background_remove_judge is not None
    assert (
        config.qwen.candidate_judge.base_url
        == config.qwen.background_remove_judge.base_url
    )
    assert (
        config.qwen.candidate_judge.model
        == config.qwen.background_remove_judge.model
    )