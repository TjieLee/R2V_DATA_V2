"""Regression checks for cluster-portable SAM defaults in the full launcher."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/run_h3_t2va_full_production.sh"


def test_full_launcher_sets_shared_sam_defaults():
    text = LAUNCHER.read_text()
    for expected in (
        'SAM_DEPS_ROOT="${SAM_DEPS_ROOT:-/mnt/workspace/litengjie/data/audio_deps}"',
        'export SAM_AUDIO_CODE_ROOT="${SAM_AUDIO_CODE_ROOT:-$SAM_DEPS_ROOT/sam-audio-src}"',
        'export SAM_AUDIO_MODEL_PATH="${SAM_AUDIO_MODEL_PATH:-/mnt/workspace/public/pretrained/Facebook/sam-audio-large-tv}"',
        'export SAM_AUDIO_MODEL_NAME="${SAM_AUDIO_MODEL_NAME:-facebook/sam-audio-large-tv}"',
        'export SAM_AUDIO_T5_BASE_PATH="${SAM_AUDIO_T5_BASE_PATH:-/mnt/workspace/public/pretrained/google/t5-base}"',
        'export SAM_AUDIO_RUNTIME_PYTHONPATH="${SAM_AUDIO_RUNTIME_PYTHONPATH:-$SAM_AUDIO_CODE_ROOT:$PERCEPTION_MODELS_ROOT:$DACVAE_ROOT:$SAM_AUDIO_PYDEPS}"',
    ):
        assert expected in text


def test_full_launcher_preflights_sam_dependencies_before_mimo():
    text = LAUNCHER.read_text()
    preflight = text.index('command -v setsid')
    mimo_start = text.index('setsid "${serve[@]}"')
    assert preflight < mimo_start
    for variable in (
        "SAM_AUDIO_CODE_ROOT",
        "SAM_AUDIO_MODEL_PATH",
        "SAM_AUDIO_T5_BASE_PATH",
        "PERCEPTION_MODELS_ROOT",
        "DACVAE_ROOT",
        "SAM_AUDIO_PYDEPS",
    ):
        marker = f'Missing {variable}:'
        assert marker in text
        assert text.index(marker) < mimo_start
