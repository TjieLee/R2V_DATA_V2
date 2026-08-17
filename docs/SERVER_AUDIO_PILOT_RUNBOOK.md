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
│   ├── embedding-venv/                  # isolated face/speaker inference env
│   ├── embedding_models/                # manually staged local model files
│   └── uv-python/                       # uv managed Python
└── r2v_audio_runs/
    ├── smoke/                           # one reusable single-clip smoke output
    ├── pilot20/                         # one reusable 20-clip validation output
    ├── pair_calibration/                # fixed pair-calibration workspace
    │   ├── plan/                        # deterministic Visual clip plan
    │   ├── audio/                       # LR-ASD Audio binding outputs
    │   ├── primary_voice/               # frozen V1 primary voice references
    │   └── embedding/                   # face/speaker retrieval diagnostics
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
export EMBEDDING_ENV=$AUDIO_DEPS/embedding-venv
export EMBEDDING_MODELS=$AUDIO_DEPS/embedding_models
export UV_PYTHON_INSTALL_DIR=$AUDIO_DEPS/uv-python

export LR_ASD_CODE_ROOT=$AUDIO_DEPS/LR-ASD
export LR_ASD_MODEL_PATH=$LR_ASD_CODE_ROOT/weight/pretrain_AVA.model

export LR_ASD_PYTHON=$AUDIO_ENV/bin/python
export SILERO_VAD_PYTHON=$AUDIO_ENV/bin/python

export SILERO_VAD_MODEL_PATH=$(
"$AUDIO_ENV/bin/python" - <<'PY'
from importlib.resources import files
print(files("silero_vad.data").joinpath("silero_vad.jit"))
PY
)

export R2V_PYTHON=$REPO/.venv/bin/python
export AUDIO_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs

export FACE_EMBEDDING_PYTHON=$EMBEDDING_ENV/bin/python
export FACE_MODEL_ROOT=$EMBEDDING_MODELS/face
export FACE_MODEL_NAME=YOUR_LOCAL_ARCFACE_PACK_NAME
export FACE_MODEL_IDENTIFIER=insightface/$FACE_MODEL_NAME

export SPEAKER_EMBEDDING_PYTHON=$EMBEDDING_ENV/bin/python
export SPEAKER_MODEL_PATH=$EMBEDDING_MODELS/speaker/spkrec-ecapa-voxceleb
export SPEAKER_MODEL_IDENTIFIER=speechbrain/spkrec-ecapa-voxceleb

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

## Embedding environment and local models

Keep face and speaker inference out of the constrained LR-ASD environment. The
following creates the separate environment and installs adapter dependencies;
select the Torch/torchaudio build that is already verified for the server's
CUDA stack rather than replacing the node's working GPU runtime implicitly.

```bash
uv venv --python 3.12 "$EMBEDDING_ENV"

uv pip install --python "$EMBEDDING_ENV/bin/python" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --python "$EMBEDDING_ENV/bin/python" \
  insightface "onnxruntime-gpu==1.26.0" opencv-python-headless \
  speechbrain soundfile Pillow numpy "pydantic>=2,<3"
```

Record the resolved package versions after installation:

```bash
uv pip freeze --python "$EMBEDDING_ENV/bin/python" \
  > "$AUDIO_DEPS/embedding-venv.freeze.txt"
```

Model files are staged manually. Neither adapter downloads a checkpoint or
accepts an implicit cache-only model ID. The expected local layouts are:

```text
$FACE_MODEL_ROOT/models/$FACE_MODEL_NAME/*.onnx
$SPEAKER_MODEL_PATH/hyperparams.yaml
$SPEAKER_MODEL_PATH/...local SpeechBrain checkpoint files...
```

`FACE_MODEL_NAME` is deliberately deployment-configured; the repository does
not bundle or select a private ArcFace model pack. The speaker adapter supports
the locally staged `speechbrain/spkrec-ecapa-voxceleb` snapshot. Both workers
verify a deterministic model-directory fingerprint before importing or loading
the model. Set `CUDA_VISIBLE_DEVICES` in the parent shell; a selected physical
GPU is exposed to each worker as process-local `cuda:0`.

`onnxruntime-gpu==1.26.0` is intentional for the validated
Torch 2.5.1+cu121 environment: it is a CUDA 12/cuDNN 9 build and provides
`onnxruntime.preload_dlls()`. PyPI `onnxruntime-gpu` 1.27 and newer default to
CUDA 13 and must not be substituted in this environment. The face worker
preloads CUDA/cuDNN libraries before constructing any InsightFace session,
requires `CUDAExecutionProvider` when `cuda:N` is requested, and fails closed
if a created session falls back to CPU.

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

## Audio interpreter launch

The Audio runtime validates the configured Python executable without resolving
the uv venv symlink before subprocess launch. Use `$AUDIO_ENV/bin/python`
directly for LR-ASD, Silero VAD, and review rendering; the old
`audio-python-wrapper` workaround is no longer needed.

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
  --clip-id "$SMOKE_CLIP" \
  --workers 1
```

Inspect only the stable paths:

```bash
cat "$SMOKE_OUT/summary.json"
cat "$SMOKE_OUT/failures.jsonl" 2>/dev/null || true
cat "$SMOKE_OUT/runtime/$SMOKE_CLIP/lr_asd/lr_asd.stderr.log" 2>/dev/null || true
```

## 20-clip validation

Use one stable output root:

```bash
export PILOT20_OUT=$AUDIO_RUN_ROOT/pilot20
```

For validation and worker benchmarks, use an explicit fixed `--clip-id` set.
Do not use `--limit 20` when comparing runs, because that can select a different
raw clip set than the manually reviewed Visual-eligible pilot set.

The verified pilot20 reported:

```text
clips_attempted=20
clips_succeeded=20
clips_failed=0
asd_runtime_failures=0
```

Manual review of the generated binding videos found no persistent
speaker/entity misbinding. Keep association thresholds conservative; the pilot
summary field `face_entity_association_failures` counts non-matched face-track
associations, not failed clips.

## Verified worker benchmark

On the same fixed 20-clip set and the same H200 node, all worker counts produced
matching binding semantics and deterministic outputs.

```text
workers  wall_seconds  speedup_vs_w1
1        399           1.00x
2        223           1.79x
4        155           2.57x
6        146           2.73x
```

Use `--workers 4` as the current production default. Six workers saved only
9 seconds versus four workers (about 5.8% wall-clock reduction) while increasing
CPU, filesystem, and GPU concurrency. Re-benchmark only if the node type,
storage, LR-ASD implementation, or workload distribution changes materially.

## Voice-reference quality diagnostics

The pilot writes calibration-only diagnostics without changing binding status,
voice-reference eligibility, or any Audio/Visual threshold:

```text
$PILOT20_OUT/clips/<clip_uid>/voice_reference_quality.json
$PILOT20_OUT/review/<clip_uid>/voice_reference_quality.json
$PILOT20_OUT/voice_reference_quality.jsonl
$PILOT20_OUT/voice_reference_quality_summary.json
```

Inspect the bounded turn-level report after a pilot:

```bash
cat "$PILOT20_OUT/voice_reference_quality_summary.json"
head -n 5 "$PILOT20_OUT/voice_reference_quality.jsonl"
```

These measurements use coalesced bound speech turns and the existing LR-ASD
16 kHz mono PCM audio. The diagnostics-v2 artifacts retain their historical
calibration provenance with `thresholds_calibrated=false`; the formal policy
below consumes those measurements without changing frame-level binding status.

Local background diagnostics use only `no_speech` bindings within two seconds
before or after each candidate turn. They exclude `offscreen`, `ambiguous`, and
`overlap` intervals. The reported local noise level is the median RMS of 20 ms
windows, and remains unavailable unless at least 0.20 seconds of explicit
`no_speech` context exists.

Recompute these diagnostics from an existing pilot without running LR-ASD,
S3FD, Silero, or review rendering:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/recompute_h3_voice_quality.py \
  --pilot-root "$PILOT20_OUT"

cat "$PILOT20_OUT/voice_reference_quality_summary.json"
```

The calibrated V1 primary-voice policy requires duration at least 1.0 seconds,
minimum association confidence at least 0.85, mean LR-ASD score at least 0.50,
LR-ASD p10 at least 0.20, RMS dBFS at least -40, clipping ratio at most 0.0001,
available local no-speech context, and estimated SNR at least 10 dB. All gates
must pass. Missing noise context fails closed; it is never estimated from
offscreen or ambiguous speech.

Export primary voice references from an existing pilot without running LR-ASD,
S3FD, Silero, review rendering, or any embedding model:

```bash
PRIMARY_VOICE_OUT="${PILOT20_OUT}-primary-voice-v1"

cd "$REPO"
"$R2V_PYTHON" tools/export_h3_primary_voice_references.py \
  --pilot-root "$PILOT20_OUT" \
  --output-root "$PRIMARY_VOICE_OUT"

cat "$PRIMARY_VOICE_OUT/summary.json"
head -n 5 "$PRIMARY_VOICE_OUT/primary_voice_references.jsonl"
```

The exporter cuts the exact selected sample interval from the existing 16 kHz
mono PCM LR-ASD audio and losslessly encodes it as mono FLAC. It performs no
padding, normalization, enhancement, denoising, speaker embedding, or pairing.

## Face and speaker embedding calibration pilot

Run this only after the formal primary-voice export exists. The pilot consumes
only occurrences whose `primary_voice_reference` is non-null. It uses each
occurrence's one canonical Visual reference and exact selected 16 kHz mono
FLAC; it does not rerun video/frame search, LR-ASD, S3FD, Silero, calibration,
or primary-voice selection.

```bash
export PRIMARY_VOICE_OUT="${PILOT20_OUT}-primary-voice-v1"
export EMBEDDING_OUT=$AUDIO_RUN_ROOT/embedding_pilot20

cd "$REPO"
"$R2V_PYTHON" tools/eval_h3_embedding_pilot.py \
  --audio-pilot-root "$PILOT20_OUT" \
  --primary-voice-root "$PRIMARY_VOICE_OUT" \
  --output-root "$EMBEDDING_OUT" \
  --face-python "$FACE_EMBEDDING_PYTHON" \
  --face-model-root "$FACE_MODEL_ROOT" \
  --face-model-name "$FACE_MODEL_NAME" \
  --face-model-identifier "$FACE_MODEL_IDENTIFIER" \
  --speaker-python "$SPEAKER_EMBEDDING_PYTHON" \
  --speaker-model-path "$SPEAKER_MODEL_PATH" \
  --speaker-model-identifier "$SPEAKER_MODEL_IDENTIFIER" \
  --device cuda:0 \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --top-k 5
```

Each persistent worker loads its model once and serves JSONL requests. Normal
logs go to fixed sibling diagnostics under
`$AUDIO_RUN_ROOT/.embedding_pilot20.worker_diagnostics`; model stdout remains
machine-readable. Face detection fails closed on zero or multiple reliable
faces in the canonical reference and never searches another frame. Speaker
inference performs no additional trim, VAD, denoise, enhancement,
normalization, padding, or utterance averaging. R2V stores both modalities as
L2-normalized float32 `.npy` files.

Inspect the bounded calibration evidence:

```bash
cat "$EMBEDDING_OUT/summary.json"

"$R2V_PYTHON" tools/inspect_h3_embedding_pilot.py \
  --embedding-root "$EMBEDDING_OUT" \
  --top 30
```

`thresholds_calibrated=false` is mandatory. Similarities, directional ranks,
and the face/voice top-K intersection are retrieval diagnostics only; they do
not assert same-person, same-voice, cross-pair eligibility, or final acceptance.

## Fixed 60-clip pair-calibration expansion

This workflow expands the manually reviewed pilot into a bounded calibration
set with more likely repeated human occurrences. It does not rerun Visual V3,
assign same-person labels, calibrate thresholds, or publish cross-pairs. The
planner reads the frozen Visual run and seed Audio pilot without modifying
either one. It preserves eligible seed clips first, then complete endpoint pairs
from accepted V3 cross-reference provenance, then fills the remaining budget by
deterministic round-robin across same-parent multi-clip groups. Same parent and
existing V3 donor provenance are candidate priors only, never identity truth.
The first same-parent round selects enough clips to form a two-clip comparison
unit; later rounds add at most one more clip per parent until the configured cap.

Use one fixed workspace and no timestamped calibration directories:

```bash
export PAIR_CALIBRATION_ROOT=$AUDIO_RUN_ROOT/pair_calibration
export PAIR_PLAN_ROOT=$PAIR_CALIBRATION_ROOT/plan
export PAIR_AUDIO_ROOT=$PAIR_CALIBRATION_ROOT/audio
export PAIR_PRIMARY_VOICE_ROOT=$PAIR_CALIBRATION_ROOT/primary_voice
export PAIR_EMBEDDING_ROOT=$PAIR_CALIBRATION_ROOT/embedding
```

Build the deterministic 60-clip plan. The seed path deliberately points to the
existing reviewed pilot; only still-valid clips are retained:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/plan_h3_pair_calibration.py \
  --run-root "$V3_RUN" \
  --seed-audio-pilot-root "$AUDIO_RUN_ROOT/voice_quality_pilot20" \
  --output-root "$PAIR_PLAN_ROOT" \
  --max-clips 60 \
  --max-clips-per-parent 4

cat "$PAIR_PLAN_ROOT/plan.json"
wc -l "$PAIR_PLAN_ROOT/clip_ids.txt"
```

Run the unchanged Audio binding pipeline over exactly the planned clip IDs.
The file order is deterministic, blank lines and full-line comments are ignored,
and duplicate IDs are removed before the existing worker path runs:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/eval_h3_audio_binding_lr_asd.py \
  --run-root "$V3_RUN" \
  --output-root "$PAIR_AUDIO_ROOT" \
  --clip-id-file "$PAIR_PLAN_ROOT/clip_ids.txt" \
  --workers 4

cat "$PAIR_AUDIO_ROOT/summary.json"
cat "$PAIR_AUDIO_ROOT/failures.jsonl" 2>/dev/null || true
```

Apply the frozen `voice_reference_quality_v1` policy and export only accepted
primary voice turns. This step reuses Audio artifacts and does not rerun LR-ASD,
S3FD, or Silero:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/export_h3_primary_voice_references.py \
  --pilot-root "$PAIR_AUDIO_ROOT" \
  --output-root "$PAIR_PRIMARY_VOICE_ROOT"

cat "$PAIR_PRIMARY_VOICE_ROOT/summary.json"
```

Run the validated persistent InsightFace `buffalo_l` and SpeechBrain ECAPA
workers. These outputs remain calibration-only retrieval diagnostics:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/eval_h3_embedding_pilot.py \
  --audio-pilot-root "$PAIR_AUDIO_ROOT" \
  --primary-voice-root "$PAIR_PRIMARY_VOICE_ROOT" \
  --output-root "$PAIR_EMBEDDING_ROOT" \
  --face-python "$FACE_EMBEDDING_PYTHON" \
  --face-model-root "$FACE_MODEL_ROOT" \
  --face-model-name buffalo_l \
  --face-model-identifier insightface/buffalo_l \
  --speaker-python "$SPEAKER_EMBEDDING_PYTHON" \
  --speaker-model-path "$SPEAKER_MODEL_PATH" \
  --speaker-model-identifier speechbrain/spkrec-ecapa-voxceleb \
  --device cuda:0 \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --top-k 5
```

Inspect the summary plus the top 100 face, voice, and joint candidates with the
existing bounded inspection tool:

```bash
cat "$PAIR_EMBEDDING_ROOT/summary.json"

"$R2V_PYTHON" tools/inspect_h3_embedding_pilot.py \
  --embedding-root "$PAIR_EMBEDDING_ROOT" \
  --top 100
```

`plan.json` and every `visual_candidate_pairs.jsonl` row retain
`thresholds_calibrated=false`; every candidate pair has
`same_person_label=null`. Human review remains mandatory before any later
threshold or cross-pair policy work.

Routine benchmark outputs are disposable after the results are recorded.

Verified server note: official LR-ASD track boxes may extend slightly outside
the model-video bounds because the vendor crop pads the frame. The R2V bridge
clips only the published artifact coordinates to the model-video bounds; it
continues to use the raw vendor box for detection-confidence matching.

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
