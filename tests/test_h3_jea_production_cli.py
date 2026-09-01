from __future__ import annotations

from pathlib import Path

import pytest

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.qwen3_asr import (
    QWEN3_ASR_MODEL_IDENTIFIER,
    Qwen3ASRConfiguration,
)
from tools.run_h3_jea_production import _audio_parallelism, _parse_stages, _parser


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
    assert _parse_stages("all") == (
        "audio",
        "primary-voice",
        "embedding",
        "pair",
        "diarization",
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


def test_jea_lr_asd_gpu_parallelism_uses_capacity_without_legacy_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LR_ASD_GPUS", raising=False)
    monkeypatch.delenv("LR_ASD_WORKERS_PER_GPU", raising=False)
    required = [
        "--visual-production-root",
        "/visual",
        "--visual-runs-root",
        "/runs",
        "--audio-production-root",
        "/audio",
    ]
    legacy = _parser().parse_args(required)
    assert _audio_parallelism(legacy) == (4, (), 4, 0)

    multi_gpu = _parser().parse_args(
        [
            *required,
            "--lr-asd-gpus",
            "0,1,2,3,4,5,6,7",
            "--lr-asd-workers-per-gpu",
            "4",
        ]
    )
    assert _audio_parallelism(multi_gpu) == (
        32,
        tuple(str(index) for index in range(8)),
        4,
        32,
    )

    capped = _parser().parse_args(
        [*required, "--workers", "12", "--lr-asd-gpus", "0,1,2,3"]
    )
    assert _audio_parallelism(capped)[0] == 12


def test_jea_lr_asd_gpu_parallelism_supports_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LR_ASD_GPUS", "2,5")
    monkeypatch.setenv("LR_ASD_WORKERS_PER_GPU", "3")
    arguments = _parser().parse_args(
        [
            "--visual-production-root",
            "/visual",
            "--visual-runs-root",
            "/runs",
            "--audio-production-root",
            "/audio",
        ]
    )
    assert _audio_parallelism(arguments) == (6, ("2", "5"), 3, 6)
