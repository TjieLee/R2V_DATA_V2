# Server Audio Pilot Runbook

This is the operational runbook for the server-side Audio/H3 pilot. Keep the
layout small and stable. Do not create a timestamped directory for every smoke
attempt.

## Canonical dots3 semantic sequence

Current milestone boundaries:

- Audio binding, primary voice, embeddings, and PairPolicy V1 are **COMPLETE / FROZEN**.
- dots3 semantic augmentation through vLLM is the active **PILOT / PRODUCTION** workflow.
- The H3 exporter is **DEMO / NOT FINAL**.
- Visual subject attributes are a separate workstream and are not consumed here.

The old Qwen/DashScope semantic commands are obsolete and are not a production
fallback. Use this single A-K sequence.

### A. Create the dedicated dots3 environment on the 8-H200 node

```bash
export DOTS3_VLLM_ENV=/mnt/workspace/litengjie/data/audio_deps/dots3-vllm-env
uv venv --python 3.12 --seed "$DOTS3_VLLM_ENV"
source "$DOTS3_VLLM_ENV/bin/activate"
```

### B. Install the validated pinned vLLM runtime

The validated environment uses vLLM `main` commit
`e0e5a7fb2808504ba86c94f7b379e38496002fd0`, observed as
`0.27.2rc1.dev191+ge0e5a7fb2`. Current PyPI vLLM is not a substitute for this
validated revision. Dots3Note native video preprocessing also requires the
audio optional dependencies PyAV and soundfile:

```bash
export VLLM_COMMIT=e0e5a7fb2808504ba86c94f7b379e38496002fd0
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url "https://wheels.vllm.ai/$VLLM_COMMIT"
uv pip install av soundfile

python -c 'import vllm; print(vllm.__version__)'
```

### C. Start dots3 on all eight H200 GPUs

The staged checkpoint is
`/mnt/workspace/public/pretrained/dots3-note-prev`. Its inspected config reports
`Dots3NoteForCausalLM`, model type `dots3_note`, `bfloat16`, and unquantized
weights. The following is the validated serve shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /mnt/workspace/public/pretrained/dots3-note-prev \
  --served-model-name dots3-note-prev \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --allowed-local-media-path /mnt/workspace
```

### D. Verify the OpenAI-compatible endpoint

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/models | python -m json.tool
```

### E. Configure the R2V producer

The validated endpoint and shared-file media contract are:

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export R2V_PYTHON="$REPO/.venv/bin/python"
export AUDIO_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs
export DOTS3_BASE_URL='http://6.167.57.88:8000/v1'
export DOTS3_API_KEY='EMPTY'
export DOTS3_MODEL='dots3-note-prev'
export DOTS3_CHECKPOINT_ID='/mnt/workspace/public/pretrained/dots3-note-prev'
export DOTS3_MEDIA_MODE=file
export DOTS3_MEDIA_ROOT=/mnt/workspace
unset DOTS3_MEDIA_BASE_URL
```

### F. Keep the native-video transport on the shared filesystem

No HTTP media server is needed. In particular, do not start a port 8767 media
server for this workflow. The semantic request contains text plus the native
target video only; Dots3Note reads the embedded audio through native video
preprocessing. The separately extracted canonical full-audio file remains
path/hash provenance and is verified by the producer, but it is not sent as an
`audio_url` item. No media is copied, transcoded, resized, or Base64-encoded.

### G. Build the semantic inventory without inference

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

### H. Run the fixed 20-target pilot

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

### I. Serve and review the pilot

```bash
cd "$AUDIO_RUN_ROOT/semantic_pilot20"
"$R2V_PYTHON" -m http.server 8766 --bind 127.0.0.1
```

Open `http://127.0.0.1:8766/review.html` through the SSH port forward.

### J. Run all unique production targets

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production
```

This uses every target in production `in_pairs.jsonl` (75 in the frozen pair
set), with no bounded sampler or parent quota.

### K. Inspect the concise production summary

```bash
python -m json.tool \
  "$AUDIO_RUN_ROOT/production/semantic/summary.json"
```

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
    │   ├── face_mining/                 # Visual-only face retrieval + review
    │   ├── face_audio_plan/              # HUMAN SAME pair Audio endpoints
    │   ├── pair_policy_review/           # HUMAN hard-negative review/report
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

## Visual-first face identity mining

Run Visual-only face mining before expanding expensive LR-ASD calibration. It
enumerates every eligible retained subject occurrence in the frozen V3 run,
opens only its canonical Visual reference, and applies the existing strict
single-face InsightFace worker. It does not require Audio binding, speech, or a
primary voice reference. Zero-face, multi-face, and runtime failures remain
fail closed. Retrieval scores and ranks are diagnostics only; no identity
threshold or automatic SAME label is produced.

Use the fixed output directory:

```bash
export PAIR_CALIBRATION_ROOT=$AUDIO_RUN_ROOT/pair_calibration
export FACE_MINING_ROOT=$PAIR_CALIBRATION_ROOT/face_mining
export FACE_AUDIO_PLAN_ROOT=$PAIR_CALIBRATION_ROOT/face_audio_plan

cd "$REPO"
"$R2V_PYTHON" tools/mine_h3_face_identity_candidates.py \
  --run-root "$V3_RUN" \
  --output-root "$FACE_MINING_ROOT" \
  --face-python "$FACE_EMBEDDING_PYTHON" \
  --face-model-root "$FACE_MODEL_ROOT" \
  --face-model-name buffalo_l \
  --face-model-identifier insightface/buffalo_l \
  --device cuda:0 \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --top-k 5

cat "$FACE_MINING_ROOT/summary.json"
```

Build and open the fixed HUMAN review page. Forward port 8765 over SSH when the
server has no desktop browser:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/build_h3_face_identity_review.py \
  --face-mining-root "$FACE_MINING_ROOT"

"$R2V_PYTHON" -m http.server 8765 --directory "$FACE_MINING_ROOT"
# Open http://127.0.0.1:8765/review.html through the SSH port forward.
```

The page asks only whether the two occurrences are the same physical person.
Keys `1`, `2`, and `3` assign SAME, DIFFERENT, and UNCERTAIN; `j`/`k` or arrow
keys navigate. Labels stay in browser localStorage until **Export JSONL** is
used. Put that exported file at an explicit writable path, then build the Audio
endpoint plan:

```bash
export FACE_LABELS=$FACE_MINING_ROOT/face_identity_labels.jsonl

cd "$REPO"
"$R2V_PYTHON" tools/plan_h3_audio_from_face_labels.py \
  --face-mining-root "$FACE_MINING_ROOT" \
  --labels "$FACE_LABELS" \
  --output-root "$FACE_AUDIO_PLAN_ROOT"

cat "$FACE_AUDIO_PLAN_ROOT/plan.json"
cat "$FACE_AUDIO_PLAN_ROOT/clip_ids.txt"
```

Only HUMAN SAME pairs contribute endpoints. Both endpoints are retained as a
unit; DIFFERENT and UNCERTAIN are excluded. The resulting `clip_ids.txt` feeds
the unchanged LR-ASD command through `--clip-id-file`. Human SAME labels remain
calibration evidence and never become automatic production identity truth.

Initial manually reviewed calibration evidence is retained for provenance: an
8-occurrence pilot yielded 28 unordered pairs, with SAME=1, DIFFERENT=27, and
UNCERTAIN=0. The one confirmed positive was
`17270abd38d44def5a88391d/e1` versus `9d1d740ad31c0deacb7287fd/e1`, with face
cosine 0.728202 and voice cosine 0.390468. These identifiers and values are not
hard-coded, do not define a threshold, and do not make thresholds calibrated.

## PairPolicy hard-negative review

Do not freeze production cross-pair acceptance from positives alone. The current
complete calibration evidence contains 11 HUMAN-confirmed positive pairs: one
from the initial pilot and ten face-first pairs with valid voice on both
endpoints. Face cosine spans 0.729658 to 0.960353 with median 0.878478. Voice
cosine spans 0.239183 to 0.627407 with median 0.453455. In particular,
`2acb9056.../e1` versus `ca86f70e.../e1` has face cosine 0.931867 and voice
cosine 0.239183. Speaker cosine therefore remains supporting or contradiction
evidence; do not invent a high speaker threshold that silently removes this
confirmed positive.

Build a fixed top-50 UNKNOWN hard-negative review from the embedding run where
both modalities are available. The confirmed file may contain direct HUMAN SAME
edges from multiple reviewed calibration rounds. Direct SAME pairs and pairs
implied by their calibration-only connected components are excluded from the
review queue, but implied pairs are not published as production identity truth:

```bash
export PAIR_POLICY_EMBEDDING_ROOT=$PAIR_CALIBRATION_ROOT/positive_embedding
export CONFIRMED_FACE_PAIRS=$PAIR_CALIBRATION_ROOT/confirmed_face_pairs.jsonl
export PAIR_POLICY_REVIEW_ROOT=$PAIR_CALIBRATION_ROOT/pair_policy_review

cd "$REPO"
"$R2V_PYTHON" tools/build_h3_pair_policy_review.py \
  --embedding-root "$PAIR_POLICY_EMBEDDING_ROOT" \
  --confirmed-face-pairs "$CONFIRMED_FACE_PAIRS" \
  --face-mining-root "$FACE_MINING_ROOT" \
  --output-root "$PAIR_POLICY_REVIEW_ROOT" \
  --top 50

cat "$PAIR_POLICY_REVIEW_ROOT/summary.json"
```

Serve the shared calibration root so the review page can reach sibling face
crops and primary voice FLAC files. Forward port 8765 over SSH, then open the
shown URL:

```bash
"$R2V_PYTHON" -m http.server 8765 --directory "$PAIR_CALIBRATION_ROOT"
# Open http://127.0.0.1:8765/pair_policy_review/review.html
```

The identity question is whether the two entity occurrences are the same
physical person. Audio is supporting evidence only. Use `1`, `2`, `3` for SAME,
DIFFERENT, UNCERTAIN and `j`/`k` or arrows to navigate. Export the HUMAN labels
as `pair_policy_review_labels.jsonl`, place them at an explicit writable path,
then generate the calibration report:

```bash
export PAIR_POLICY_LABELS=$PAIR_POLICY_REVIEW_ROOT/pair_policy_review_labels.jsonl

cd "$REPO"
"$R2V_PYTHON" tools/report_h3_pair_policy_calibration.py \
  --embedding-root "$PAIR_POLICY_EMBEDDING_ROOT" \
  --confirmed-face-pairs "$CONFIRMED_FACE_PAIRS" \
  --hard-negative-labels "$PAIR_POLICY_LABELS" \
  --face-mining-root "$FACE_MINING_ROOT" \
  --output-root "$PAIR_POLICY_REVIEW_ROOT"

cat "$PAIR_POLICY_REVIEW_ROOT/pair_policy_calibration_report.json"
```

The report contains raw distributions, rank/margin diagnostics, and boundary
cases. Optional threshold simulation runs only when explicit `--simulate-*`
arguments are supplied. It reports TP/FP/FN/TN, precision, and recall but never
searches for a best threshold or writes production configuration. Both review
and report retain `thresholds_calibrated=false` until a human explicitly freezes
a later precision-oriented policy.

This entire workflow is calibration-only. It introduces no parent quota,
production clustering, donor selection, pair publication, or threshold
acceptance. Every production-eligible occurrence remains retained; uncertainty
means no pair rather than dataset truncation.

## Frozen H3 PairPolicy V1 and accepted-pair pilot

HUMAN calibration freezes production `h3_pair_policy_v1` at face cosine
`>= 0.72` and voice cosine `>= 0.20`. The voice threshold is a low
contradiction floor; it cannot rescue a face score below `0.72`. Rank, margin,
and identity text remain diagnostics only. Production candidate evaluation is
exact across every eligible cross-clip occurrence, with no top-K intersection,
parent quota, clustering, or in/cross ratio gate. Donor selection uses the
highest passing face cosine and then `entity_occurrence_id` as the deterministic
tie-break. Every valid in-pair remains published when no strict cross-pair is
available.

Build the fixed real-artifact pilot without rerunning Visual, LR-ASD, primary
voice selection, InsightFace, or ECAPA:

```bash
export PAIR_AUDIO_ROOT=$PAIR_CALIBRATION_ROOT/audio
export PAIR_EMBEDDING_ROOT=$PAIR_CALIBRATION_ROOT/embedding
export PAIRING_PILOT_ROOT=$PAIR_CALIBRATION_ROOT/pairing_pilot

cd "$REPO"
"$R2V_PYTHON" tools/build_h3_pairing_pilot.py \
  --audio-pilot-root "$PAIR_AUDIO_ROOT" \
  --embedding-root "$PAIR_EMBEDDING_ROOT" \
  --output-root "$PAIRING_PILOT_ROOT"

cat "$PAIRING_PILOT_ROOT/summary.json"
```

The tool reads existing Audio sidecars, primary voice assets, face crops, and
face/speaker embeddings. It validates source hashes, uses no HUMAN calibration
labels as production identity truth, and writes only the fixed pilot output.
`review.html` contains every accepted target/donor mapping.

Serve the shared calibration directory so the page can load sibling embedding
and primary-voice assets, then review every accepted mapping with `1`, `2`, or
`3` for CORRECT, WRONG, or UNCERTAIN:

```bash
"$R2V_PYTHON" -m http.server 8765 --directory "$PAIR_CALIBRATION_ROOT"
# Open http://127.0.0.1:8765/pairing_pilot/review.html
```

Export the browser labels and place them at the explicit path below. Reporting
never changes PairPolicy thresholds:

```bash
export ACCEPTED_PAIR_LABELS=$PAIRING_PILOT_ROOT/accepted_pair_review_labels.jsonl
export ACCEPTED_PAIR_REPORT=$PAIRING_PILOT_ROOT/accepted_pair_review_report.json

cd "$REPO"
"$R2V_PYTHON" tools/report_h3_accepted_pair_review.py \
  --pairing-pilot-root "$PAIRING_PILOT_ROOT" \
  --labels "$ACCEPTED_PAIR_LABELS" \
  --output "$ACCEPTED_PAIR_REPORT"

cat "$ACCEPTED_PAIR_REPORT"
```

## Frozen 1000-clip H3 production workflow

The HUMAN-reviewed accepted-pair pilot freezes `h3_pair_policy_v1` at face
cosine `>= 0.72` and voice cosine `>= 0.20`. Production uses no rank, margin,
text, parent-count, source-count, or in/cross-ratio gate. It never consumes
HUMAN labels or any `pair_calibration/` artifact. Every Visual-eligible subject
occurrence is retained in the production inventory; missing Audio, primary
voice, face, speaker, or donor evidence is recorded as unavailable rather than
turning into dataset sampling.

Published in/cross rows are target-clip samples, while
`pair_evidence.jsonl` remains directional occurrence-level evidence. A
cross-pair keeps the target video, target full audio, target Audio sidecar, and
every target picture. Only each subject's primary voice reference is replaced
by its selected donor primary voice. Multi-speaker targets require a complete,
one-to-one legal assignment maximizing total face cosine; an incomplete
assignment publishes no cross-pair and leaves the in-pair intact.

All outputs use one stable root with no timestamp suffix. First inspect the
complete metadata-only input set:

```bash
export H3_PRODUCTION_ROOT=$AUDIO_RUN_ROOT/production

cd "$REPO"
"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

The dry run writes nothing and exposes no `--limit` or parent quota. Run each
expensive stage explicitly so completed outputs are reused rather than silently
regenerated:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --stages audio \
  --workers 4

"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --stages primary-voice

"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --stages embedding \
  --face-python "$FACE_EMBEDDING_PYTHON" \
  --face-model-root "$FACE_MODEL_ROOT" \
  --face-model-name buffalo_l \
  --face-model-identifier insightface/buffalo_l \
  --speaker-python "$SPEAKER_EMBEDDING_PYTHON" \
  --speaker-model-path "$SPEAKER_MODEL_PATH" \
  --speaker-model-identifier speechbrain/spkrec-ecapa-voxceleb \
  --device cuda:0 \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"

"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --stages pair
```

After this publication-contract fix is deployed, overwrite **only** the existing
production Pair stage. This command reuses Audio, primary voice, face, and
speaker artifacts and atomically replaces only `production/pairs`:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_audio_production.py \
  --run-root "$V3_RUN" \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --stages pair \
  --overwrite
```

The fixed artifacts are:

```text
$AUDIO_RUN_ROOT/production/
  source_inventory.json
  audio/
  primary_voice/
  embedding/
  pairs/
    in_pairs.jsonl       # clip-level target samples
    cross_pairs.jsonl    # clip-level target samples
    pair_evidence.jsonl  # directional occurrence evidence
    summary.json
    review.html
    review_media/
```

Inspect the final aggregate and deterministic pair rows with:

```bash
cat "$H3_PRODUCTION_ROOT/pairs/summary.json"
wc -l \
  "$H3_PRODUCTION_ROOT/pairs/in_pairs.jsonl" \
  "$H3_PRODUCTION_ROOT/pairs/cross_pairs.jsonl" \
  "$H3_PRODUCTION_ROOT/pairs/pair_evidence.jsonl"
```

Serve the production root so every copied target/donor face and voice asset is
available to the review page:

```bash
cd "$H3_PRODUCTION_ROOT"
"$R2V_PYTHON" -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/pairs/review.html` through the SSH port forward.
The page includes every selected subject mapping, supports 1/2/3 and
j/k/arrow navigation, and exports JSONL. These labels validate the accepted
assets only; they never change PairPolicy thresholds or production identity.

Except for an explicit pair-only contract migration such as the command above,
`--overwrite` is debug-only. Overwriting an upstream stage fails unless every
already-completed downstream stage is requested in the same invocation, which
prevents stale primary-voice, embedding, or pair artifacts.

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
Eligible seed clips are retained before expansion and are not subject to the
parent cap; they still count toward the global maximum, and eligible seeds over
that maximum fail deterministically instead of being silently truncated.

`max_clips_per_parent` is calibration-sampling-only. It applies only to this
planner's newly mined Priority A/B expansion. It must never appear in production
H3 eligibility, embedding, retrieval, identity, donor, or pair-construction
configuration or behavior. Production retains every eligible occurrence; source
and parent provenance may be a retrieval prior but never a quota. Strict identity
gates decide pair usability, and uncertainty yields no pair rather than dataset
truncation. Calibration selection helpers must not be reused by production
candidate generation unless every sampling limit has been removed.

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
