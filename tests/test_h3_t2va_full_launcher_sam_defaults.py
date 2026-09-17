"""Regression checks for cluster-portable SAM defaults in the full launcher."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/run_h3_t2va_full_production.sh"


def test_full_launcher_sets_shared_sam_defaults():
    text = LAUNCHER.read_text()
    for expected in (
        'SAM_DEPS_ROOT="${SAM_DEPS_ROOT:-/mnt/workspace/litengjie/data/audio_deps}"',
        'export SAM_AUDIO_CODE_ROOT="${SAM_AUDIO_CODE_ROOT:-$SAM_DEPS_ROOT/sam-audio-src}"',
        'PERCEPTION_MODELS_ROOT="${PERCEPTION_MODELS_ROOT:-$SAM_DEPS_ROOT/perception-models-src}"',
        'DACVAE_ROOT="${DACVAE_ROOT:-$SAM_DEPS_ROOT/dacvae-src}"',
        'SAM_AUDIO_PYDEPS="${SAM_AUDIO_PYDEPS:-$SAM_DEPS_ROOT/sam-audio-pydeps}"',
        'export SAM_AUDIO_MODEL_PATH="${SAM_AUDIO_MODEL_PATH:-/mnt/workspace/public/pretrained/Facebook/sam-audio-large-tv}"',
        'export SAM_AUDIO_MODEL_NAME="${SAM_AUDIO_MODEL_NAME:-facebook/sam-audio-large-tv}"',
        'export SAM_AUDIO_T5_BASE_PATH="${SAM_AUDIO_T5_BASE_PATH:-/mnt/workspace/public/pretrained/google/t5-base}"',
        'export SAM_AUDIO_RUNTIME_PYTHONPATH="${SAM_AUDIO_RUNTIME_PYTHONPATH:-$SAM_AUDIO_CODE_ROOT:$PERCEPTION_MODELS_ROOT:$DACVAE_ROOT:$SAM_AUDIO_PYDEPS}"',
    ):
        assert expected in text


def test_full_launcher_keeps_sam_runtime_path_before_pythonpath_clear():
    text = LAUNCHER.read_text()
    runtime_default = text.index("export SAM_AUDIO_RUNTIME_PYTHONPATH=")
    pythonpath_clear = text.index("unset PYTHONPATH")
    assert runtime_default < pythonpath_clear
