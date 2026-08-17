# Server Audio Pilot Runbook

This is the operational runbook for the server-side Audio/H3 pilot. Keep the
layout small and stable. Do not create a timestamped directory for every smoke
attempt.

## Canonical directory layout

```text
/mnt/workspace/litengjie/data/
├── R2V_DATA_V2/                         # repository
├── audio_deps/                          # Audio-only dependencies
│   ├── LR-ASD/                          # pinned vendor checkout
│   ├── lr-asd-venv/                     # Python 3.10 uv venv
│   ├── uv-python/                       # uv managed Python
│   └── audio-python-wrapper             # temporary venv symlink workaround
└── r2v_audio_runs/
    ├── smoke/                           # one reusable single-clip smoke output
    ├── pilot20/                         # one reusable 20-clip validation output
    └── production/                      # create only when batch production starts
```

LR-ASD itself creates `pyavi`, `pyframes`, `pywork`, and `pycrop` under its
per-clip runtime directory. Those are internal vendor scratch artifacts, not
additional top-level datasets.

Do not move the existing uv venv after creation; installed scripts may contain
absolute interpreter paths.

## Restore environment in a new shell

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export V3_RUN=/mnt/workspace/litengjie/data/r2v_v3_runs/e2e1000-s0-samfix-20260814-101818

export AUDIO_DEPS=/mnt/workspace/litengjie/data/audio_deps
export AUDIO_ENV=$AUDIO_DEPS/lr-asd-venv
export UV_PYTHON_INSTALL_DIR=$AUDIO_DEPS/uv-python

export LR_ASD_CODE_ROOT=$AUDIO_DEPS/LR-ASD
export LR_ASD_MODEL_PATH=$LR_ASD_CODE_ROOT/weight/pretrain_AVA.model

export AUDIO_PY_WRAPPER=$AUDIO_DEPS/audio-python-wrapper
export LR_ASD_PYTHON=$AUDIO_PY_WRAPPER
export SILERO_VAD_PYTHON=$AUDIO_PY_WRAPPER

export SILERO_VAD_MODEL_PATH=$(
"$AUDIO_ENV/bin/python" - <<'PY'
from importlib.resources import files
print(files("silero_vad.data").joinpath("silero_vad.jit"))
PY
)

export R2V_PYTHON=$REPO/.venv/bin/python
export AUDIO_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs

# Known Visual-eligible smoke clip from the current 1000-clip run.
export SMOKE_CLIP=00e8c6125ed42d4aaed9fbde
```

Select a physical GPU only after checking `nvidia-smi`, for example:

```bash
export CUDA_VISIBLE_DEVICES=7
```

LR-ASD then sees the selected physical GPU as process-local `cuda:0`.

## Audio environment

Use Python 3.10 and keep Audio dependencies separate from the repository `.venv`.
Always pass the interpreter explicitly to `uv pip`.

```bash
uv pip install \
  --python "$AUDIO_ENV/bin/python" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  --python "$AUDIO_ENV/bin/python" \
  "numpy==1.23.5" \
  "scipy<2" "scikit-learn<2" "pandas<3" \
  tqdm "scenedetect<0.7" "opencv-python-headless<5" \
  python_speech_features soundfile gdown "silero-vad==6.2.1"
```

`numpy==1.23.5` is intentional. The pinned LR-ASD S3FD code still uses removed
NumPy aliases such as `np.int`; NumPy 1.26.4 fails during face detection.

## LR-ASD checkout and weights

Pinned checkout:

```text
Junhua-Liao/LR-ASD
1b6dcd2d8fc2895683de6508ec6294ec47d388ca
```

Required files:

```text
$LR_ASD_CODE_ROOT/weight/pretrain_AVA.model
$LR_ASD_CODE_ROOT/model/faceDetector/s3fd/sfd_face.pth
```

If the S3FD weight must be downloaded again:

```bash
cd "$LR_ASD_CODE_ROOT"
mkdir -p model/faceDetector/s3fd
"$AUDIO_ENV/bin/python" -m gdown \
  1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt \
  -O model/faceDetector/s3fd/sfd_face.pth
```

Recent `gdown` uses the Drive ID as a positional argument; do not use `--id`.

## Temporary uv interpreter wrapper

The current Audio runtime resolves configured Python executable paths before
launching LR-ASD and Silero. Resolving a uv venv `bin/python` symlink can bypass
the venv site-packages. Until that runtime bug is removed, use one regular
wrapper file:

```bash
cat > "$AUDIO_PY_WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$AUDIO_ENV/bin/python" "\$@"
EOF
chmod 755 "$AUDIO_PY_WRAPPER"
```

Verify:

```bash
"$LR_ASD_PYTHON" -c 'import numpy,tqdm,torch; print(numpy.__version__, tqdm.__file__, torch.__version__)'
```

## Reusable smoke output

Smoke tests always reuse exactly this directory:

```bash
export SMOKE_OUT=$AUDIO_RUN_ROOT/smoke
```

Before rerunning, delete only the previous smoke result and its abandoned atomic
temporary siblings:

```bash
rm -rf "$SMOKE_OUT"
find "$AUDIO_RUN_ROOT" -maxdepth 1 -type d -name '.smoke.tmp-*' -exec rm -rf {} +
```

Then run:

```bash
cd "$REPO"

"$R2V_PYTHON" tools/eval_h3_audio_binding_lr_asd.py \
  --run-root "$V3_RUN" \
  --output-root "$SMOKE_OUT" \
  --clip-id "$SMOKE_CLIP"
```

Inspect only the stable paths:

```bash
cat "$SMOKE_OUT/summary.json"
cat "$SMOKE_OUT/failures.jsonl" 2>/dev/null || true
cat "$SMOKE_OUT/runtime/$SMOKE_CLIP/lr_asd/lr_asd.stderr.log" 2>/dev/null || true
```

## 20-clip validation

Use one stable output root as well:

```bash
export PILOT20_OUT=$AUDIO_RUN_ROOT/pilot20
rm -rf "$PILOT20_OUT"

cd "$REPO"
"$R2V_PYTHON" tools/eval_h3_audio_binding_lr_asd.py \
  --run-root "$V3_RUN" \
  --output-root "$PILOT20_OUT" \
  --limit 20
```

Only preserve/archive a run when it is actually useful for comparison or an
accepted production checkpoint. Routine failed smoke attempts are disposable.

Verified server note: official LR-ASD track boxes may extend slightly outside
the model-video bounds because the vendor crop pads the frame. The R2V bridge
clips only the published artifact coordinates to the model-video bounds; it
continues to use the raw vendor box for detection-confidence matching.
`face_entity_association_failures` in the pilot summary counts unmatched
face-track associations, not failed clips.

## Cleanup of old timestamped smoke attempts

Preview first:

```bash
find "$AUDIO_RUN_ROOT" -maxdepth 1 -type d \
  \( -name 'lr-asd-smoke-*' -o -name '.lr-asd-smoke-*.tmp-*' \) \
  -print
```

After checking the list, remove those legacy smoke directories:

```bash
find "$AUDIO_RUN_ROOT" -maxdepth 1 -type d \
  \( -name 'lr-asd-smoke-*' -o -name '.lr-asd-smoke-*.tmp-*' \) \
  -exec rm -rf {} +
```

Do not delete `pilot20`, `production`, the frozen Visual run, or `audio_deps`.
