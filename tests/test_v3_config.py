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
    cross_pair_judge: bool = False,
    same_parent_fallback_enabled: bool = False,
    same_parent_max_donor_references: object = 8,
    synthetic_completion_enabled: object = False,
    reference_edit_enabled: bool = False,
    reference_edit_judge: bool = False,
    reference_edit_target_area: int = 1024 * 1024,
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
    if cross_pair_judge:
        qwen_lines.extend(
            [
                "  cross_pair_judge:",
                f"    model: {model}",
            ]
        )
    if reference_edit_judge:
        qwen_lines.extend(
            [
                "  reference_edit_judge:",
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
                "reference_scope:",
                (
                    "  allow_synthetic_completion: "
                    f"{str(synthetic_completion_enabled).lower()}"
                ),
                "pair:",
                f"  enabled: {str(pair_enabled).lower()}",
                (
                    "  same_parent_fallback_enabled: "
                    f"{str(same_parent_fallback_enabled).lower()}"
                ),
                (
                    "  same_parent_max_donor_references: "
                    f"{str(same_parent_max_donor_references).lower()}"
                ),
                "remove:",
                f"  enabled: {str(remove_enabled).lower()}",
                f"  base_model_path: {pretrained / 'Qwen' / 'edit'}",
                f"  adapter_path: {user_models / 'object-remover'}",
                "reference_edit:",
                f"  enabled: {str(reference_edit_enabled).lower()}",
                f"  python_executable: {writable / 'venvs' / 'boogu' / 'python'}",
                f"  code_root: {writable / 'vendor' / 'Boogu-Image'}",
                f"  model_path: {writable / 'models' / 'Boogu-Image'}",
                f"  target_area: {reference_edit_target_area}",
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
        cross_pair_judge=True,
    )

    config = load_config(path)

    assert config.qwen.candidate_judge is not None
    assert config.qwen.background_remove_judge is not None
    assert config.qwen.cross_pair_judge is not None
    assert (
        config.qwen.candidate_judge.base_url
        == config.qwen.background_remove_judge.base_url
    )
    assert (
        config.qwen.candidate_judge.model
        == config.qwen.background_remove_judge.model
        == config.qwen.cross_pair_judge.model
    )
    assert (
        config.qwen.candidate_judge.base_url
        == config.qwen.background_remove_judge.base_url
        == config.qwen.cross_pair_judge.base_url
    )


def test_same_parent_fallback_requires_explicit_cross_pair_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        same_parent_fallback_enabled=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "qwen.cross_pair_judge is required when "
            "pair.same_parent_fallback_enabled is true"
        ),
    ):
        load_config(path)


def test_explicit_cross_pair_judge_enables_same_parent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        cross_pair_judge=True,
        same_parent_fallback_enabled=True,
    )

    config = load_config(path)

    assert config.pair.same_parent_fallback_enabled is True
    assert config.pair.same_parent_max_donor_references == 8
    assert config.qwen.cross_pair_judge is not None


@pytest.mark.parametrize("value", [True, 0, 65])
def test_same_parent_donor_limit_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        cross_pair_judge=True,
        same_parent_fallback_enabled=True,
        same_parent_max_donor_references=value,
    )

    with pytest.raises(
        ValueError,
        match="same_parent_max_donor_references",
    ):
        load_config(path)


def test_same_parent_settings_participate_in_config_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        cross_pair_judge=True,
        same_parent_fallback_enabled=True,
        same_parent_max_donor_references=4,
    )
    first = load_config(first_path)
    second_path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        cross_pair_judge=True,
        same_parent_fallback_enabled=True,
        same_parent_max_donor_references=5,
    )
    second = load_config(second_path)

    assert first.fingerprint() != second.fingerprint()


def test_same_parent_fallback_flag_is_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        cross_pair_judge=True,
    )
    contents = path.read_text(encoding="utf-8").replace(
        "same_parent_fallback_enabled: false",
        "same_parent_fallback_enabled: 1",
    )
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(TypeError, match="same_parent_fallback_enabled"):
        load_config(path)

def test_synthetic_completion_fallback_is_explicit_and_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = load_config(
        _write_config(
            tmp_path,
            monkeypatch,
            candidate_judge=True,
            background_remove_judge=True,
        )
    )
    enabled = load_config(
        _write_config(
            tmp_path,
            monkeypatch,
            candidate_judge=True,
            background_remove_judge=True,
            synthetic_completion_enabled=True,
        )
    )

    assert disabled.reference_scope.allow_synthetic_completion is False
    assert enabled.reference_scope.allow_synthetic_completion is True
    assert disabled.fingerprint() != enabled.fingerprint()


def test_synthetic_completion_requires_pair_and_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled_pair = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=False,
        background_remove_judge=True,
        pair_enabled=False,
        synthetic_completion_enabled=True,
    )
    with pytest.raises(ValueError, match="requires pair.enabled"):
        load_config(disabled_pair)

    non_boolean = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        synthetic_completion_enabled=1,
    )
    with pytest.raises(TypeError, match="allow_synthetic_completion"):
        load_config(non_boolean)


def test_reference_edit_requires_dedicated_qwen_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        candidate_judge=True,
        background_remove_judge=True,
        reference_edit_enabled=True,
    )

    with pytest.raises(ValueError, match="qwen.reference_edit_judge"):
        load_config(path)


def test_reference_edit_config_participates_in_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = load_config(
        _write_config(
            tmp_path,
            monkeypatch,
            candidate_judge=True,
            background_remove_judge=True,
            reference_edit_enabled=True,
            reference_edit_judge=True,
            reference_edit_target_area=1024 * 1024,
        )
    )
    second = load_config(
        _write_config(
            tmp_path,
            monkeypatch,
            candidate_judge=True,
            background_remove_judge=True,
            reference_edit_enabled=True,
            reference_edit_judge=True,
            reference_edit_target_area=900_000,
        )
    )

    assert first.reference_edit.enabled is True
    assert first.qwen.reference_edit_judge is not None
    assert first.fingerprint() != second.fingerprint()
