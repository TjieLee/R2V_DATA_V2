# Server Audio Pilot Runbook

This file records the current server-side paths and setup for the Audio/H3 pilot.
It is operational documentation only; it does not change any Visual V3 or Audio
schema contract.

## Current shared paths

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export V3_RUN=/mnt/workspace/litengjie/data/r2v_v3_runs/e2e1000-s0-samfix-20260814-101818

export AUDIO_DEPS=/mnt/workspace/litengjie/data/audio_deps
export AUDIO_ENV=$AUDIO_DEPS/lr-asd-venv
export UV_PYTHON_INSTALL_DIR=$AUDIO_DEPS/uv-python
```

The Audio environment is intentionally separate from the repository `.venv`.
Keep both the managed Python and the Audio venv under `/mnt/workspace` so they
remain usable after changing compute nodes, assuming the workspace mount is
shared.

## Create the Audio environment with uv

```bash
mkdir -p "$AUDIO_DEPS"

uv python install 3.10
uv venv --python 3.10 "$AUDIO_ENV"

uv pip install \
  --python "$AUDIO_ENV/bin/python" \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  --python "$AUDIO_ENV/bin/python" \
  "numpy==1.26.4" \
  "scipy<2" \
  "scikit-learn<2" \
  "pandas<3" \
  tqdm \
  "scenedetect<0.7" \
  "opencv-python-headless<5" \
  python_speech_features \
  soundfile \
  gdown \
  "silero-vad==6.2.1"
```

Always pass `--python "$AUDIO_ENV/bin/python"` to `uv pip` from inside the R2V
repository so dependencies are not accidentally installed into the repository
`.venv`.

## LR-ASD checkout and weights

```bash
cd "$AUDIO_DEPS"

if [ ! -d LR-ASD/.git ]; then
  git clone https://github.com/Junhua-Liao/LR-ASD.git
fi

cd LR-ASD
git fetch origin
git checkout 1b6dcd2d8fc2895683de6508ec6294ec47d388ca

mkdir -p model/faceDetector/s3fd

"$AUDIO_ENV/bin/python" -m gdown \
  1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt \
  -O model/faceDetector/s3fd/sfd_face.pth
```

Recent `gdown` versions take the Google Drive ID as the positional `url_or_id`
argument; do not use the removed `--id` flag.

The LR-ASD repository already contains:

```text
weight/pretrain_AVA.model
```

Expected S3FD path:

```text
/mnt/workspace/litengjie/data/audio_deps/LR-ASD/model/faceDetector/s3fd/sfd_face.pth
```

Using `python -m gdown` avoids depending on a shell-level `gdown` executable.
If `gdown` is missing, install only into the Audio venv:

```bash
uv pip install --python "$AUDIO_ENV/bin/python" gdown
```

## Silero VAD model path

The installed `silero-vad` package contains `silero_vad.jit`. Resolve it instead
of assuming a hard-coded site-packages layout:

```bash
export SILERO_VAD_MODEL_PATH=$(
"$AUDIO_ENV/bin/python" - <<'PY'
from importlib.resources import files
print(files("silero_vad.data").joinpath("silero_vad.jit"))
PY
)
```

## Runtime environment variables

After changing nodes or starting a new shell, restore:

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export V3_RUN=/mnt/workspace/litengjie/data/r2v_v3_runs/e2e1000-s0-samfix-20260814-101818

export AUDIO_DEPS=/mnt/workspace/litengjie/data/audio_deps
export AUDIO_ENV=$AUDIO_DEPS/lr-asd-venv
export UV_PYTHON_INSTALL_DIR=$AUDIO_DEPS/uv-python

export LR_ASD_CODE_ROOT=$AUDIO_DEPS/LR-ASD
export LR_ASD_PYTHON=$AUDIO_ENV/bin/python
export LR_ASD_MODEL_PATH=$LR_ASD_CODE_ROOT/weight/pretrain_AVA.model

export SILERO_VAD_PYTHON=$AUDIO_ENV/bin/python
export SILERO_VAD_MODEL_PATH=$(
"$AUDIO_ENV/bin/python" - <<'PY'
from importlib.resources import files
print(files("silero_vad.data").joinpath("silero_vad.jit"))
PY
)
```

Use a process-local GPU mapping only after checking `nvidia-smi`, for example:

```bash
export CUDA_VISIBLE_DEVICES=7
```

Do not make this a global server setting. LR-ASD sees the selected physical GPU
as process-local `cuda:0`.

## Preflight checks

```bash
"$AUDIO_ENV/bin/python" --version
command -v ffmpeg
ffmpeg -version | head -n 1

test -f "$LR_ASD_CODE_ROOT/Columbia_test.py"
test -f "$LR_ASD_MODEL_PATH"
test -f "$LR_ASD_CODE_ROOT/model/faceDetector/s3fd/sfd_face.pth"
test -f "$SILERO_VAD_MODEL_PATH"

"$AUDIO_ENV/bin/python" - <<'PY'
import torch
from silero_vad import get_speech_timestamps, read_audio
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
print("silero imports OK")
PY
```

## Select a Visual-eligible smoke clip

Do not use `--limit 1` for the first smoke test. The pilot's bounded limit is
applied to the sorted raw clip paths before Visual eligibility checks, so the
first path may legitimately have failed a Visual gate such as coverage.

Select an explicit clip that satisfies the same Visual prerequisites as the
pilot loader:

```bash
cd "$REPO"
export R2V_PYTHON=$REPO/.venv/bin/python

export SMOKE_CLIP=$(
"$R2V_PYTHON" - <<'PY'
import os
from pathlib import Path
from r2v_data_v2.v3.schemas import ClipRecord, SampledFramesArtifact, TrackedMasksArtifact

root = Path(os.environ["V3_RUN"])
for clip_path in sorted((root / "clips").glob("*/clip.json")):
    try:
        clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
        if clip.annotation is None or clip.annotation.status != "ready":
            continue
        if clip.coverage is None or not clip.coverage.passed:
            continue
        if clip.pairing is None or clip.pairing.status != "ready":
            continue
        frames = SampledFramesArtifact.model_validate_json(
            (clip_path.parent / "frames" / "frames.json").read_text(encoding="utf-8")
        )
        masks = TrackedMasksArtifact.model_validate_json(
            (clip_path.parent / "masks.rle.json").read_text(encoding="utf-8")
        )
        if frames.clip_uid != clip.clip_uid or masks.clip_uid != clip.clip_uid:
            continue
        if not any(entity.status == "ready" for entity in masks.entities.values()):
            continue
        if not Path(clip.source.video_path).expanduser().is_file():
            continue
        print(clip.clip_uid)
        break
    except Exception:
        continue
PY
)

echo "SMOKE_CLIP=$SMOKE_CLIP"
test -n "$SMOKE_CLIP"
```

## Pilot command

Run the pilot from the R2V repository while keeping all output outside the
frozen Visual run:

```bash
cd "$REPO"

export AUDIO_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs
mkdir -p "$AUDIO_RUN_ROOT"

export PILOT_OUT=$AUDIO_RUN_ROOT/lr-asd-smoke-${SMOKE_CLIP}-$(date +%Y%m%d-%H%M%S)

"$R2V_PYTHON" tools/eval_h3_audio_binding_lr_asd.py \
  --run-root "$V3_RUN" \
  --output-root "$PILOT_OUT" \
  --clip-id "$SMOKE_CLIP"
```

The R2V driver should use the repository Python environment. LR-ASD and Silero
are invoked through the explicit `LR_ASD_PYTHON` and `SILERO_VAD_PYTHON`
subprocess paths above.
