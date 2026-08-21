from __future__ import annotations

from pathlib import Path

import pytest

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.qwen3_asr import (
    QWEN3_ASR_MODEL_IDENTIFIER,
    Qwen3ASRConfiguration,
)
from tools.run_h3_jea_production import _parse_stages


def test_direct_audio_production_root_has_all_seven_stages(tmp_path: Path) -> None:
    paths = jea_production_paths(tmp_path / "prod-v1")
    assert [
        path.name
        for path in (
            paths.audio,
            paths.primary_voice,
            paths.embedding,
            paths.pairs,
            paths.diarization,
            paths.asr,
            paths.h3,
        )
    ] == [
        "audio",
        "primary_voice",
        "embedding",
        "pairs",
        "diarization",
        "asr",
        "h3",
    ]
    assert paths.root == (tmp_path / "prod-v1").resolve()


def test_production_stage_parser_is_direct_and_ordered() -> None:
    assert _parse_stages("h3,pair,qwen3-asr,pair") == (
        "pair",
        "qwen3-asr",
        "h3",
    )
    with pytest.raises(ValueError):
        _parse_stages("whisper")


def test_qwen3_environment_defaults_select_only_qwen3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", "/local/qwen3-asr")
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)
    monkeypatch.delenv("QWEN3_ASR_DTYPE", raising=False)
    configuration = Qwen3ASRConfiguration.from_environment()
    assert configuration.model_identifier == QWEN3_ASR_MODEL_IDENTIFIER
    assert configuration.device == "cuda:0"
    assert configuration.dtype == "bfloat16"
    assert configuration.local_files_only
    assert configuration.max_new_tokens == 256
