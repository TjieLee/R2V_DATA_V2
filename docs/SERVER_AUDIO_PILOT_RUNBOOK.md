# Active JEA production path

For full canonical JEA production, readable media-collection grouping, all
seven sequential stage commands, the isolated `qwen-asr==0.0.6` environment,
and Qwen3/final-renderer operation, use `docs/H3_QWEN3_ASR.md`. Audio,
primary-voice, embedding, pair, DiariZen, and H3 use `R2V_PYTHON`; only
`qwen3-asr` uses `QWEN3_ASR_ENV/bin/python`. The Whisper commands below are
legacy pilot procedures, not the active JEA production ASR path.

# Server Audio Pilot Runbook

This is the operational runbook for the server-side Audio/H3 pilot. Keep the
layout small and stable. Do not create a timestamped directory for every smoke
attempt.

## Canonical Whisper ASR Sequence

Current milestone boundaries:

- Visual, Audio binding, primary voice, embeddings, and PairPolicy V1 are **COMPLETE / FROZEN**.
- Whisper-large-v3 ASR V1 baseline is **COMPLETE / FROZEN**; raw transcripts
  are not yet final H3 transcript truth.
- dots3 transcript generation is **BLOCKED AFTER FAILED HUMAN QA**.
- The existing `semantic_pilot20` is diagnostic evidence only; do not delete it
  and do not run complete semantic production.
- DiariZen sparse-anchor mapping policy V1 is **COMPLETE / FROZEN** after the
  repaired 20/20 pilot and 22/22 correct human cluster reviews. The active step
  has completed formal production over all 75 unique in-pair targets.
- Formal DiariZen production is **COMPLETE / FROZEN**.
- Formal ASR V2 raw-transcript production is **COMPLETE / FROZEN** over all 179
  segments.
- ASR V2 segmentation calibration is **COMPLETE / FROZEN** after all 50 pilot
  segments received human QA. Raw transcripts remain observations, not final
  H3 text truth.
- TextUsabilityPolicy V1 is **COMPLETE / FROZEN** at the exact
  `language_probability >= 0.65` display threshold.
- Deterministic final H3 renderer V1 is **IMPLEMENTED / PENDING SERVER RUN**.
  Speech enhancement and unresolved-only MLLM remain future optional work.

### One-Time Setup

```bash
export ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/asr-venv
export ASR_HF_MODEL=/mnt/workspace/public/pretrained/LongCat-Video-Avatar-1.5/whisper-large-v3
export ASR_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/asr_models/whisper-large-v3-ct2

uv venv --python 3.12 --seed "$ASR_ENV"
uv pip install --python "$ASR_ENV/bin/python" \
  faster-whisper transformers "pydantic>=2,<3" Pillow \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12

"$ASR_ENV/bin/ct2-transformers-converter" \
  --model "$ASR_HF_MODEL" \
  --output_dir "$ASR_MODEL_PATH" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

Keep this environment separate from system Python, LR-ASD, embeddings, and
dots3/vLLM. The complete 32-decoder-layer large-v3 checkpoint above was selected
for the accuracy baseline; the available four-decoder-layer turbo checkpoint
was not used.

The first real pilot failed safely for all 82 calls with
`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`. After
installing the NVIDIA runtime wheels above, this was the validated library setup
and smoke check:

```bash
ASR_SITE=$(
"$ASR_ENV/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
export LD_LIBRARY_PATH="$ASR_SITE/nvidia/cublas/lib:$ASR_SITE/nvidia/cudnn/lib:$ASR_SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$ASR_ENV/bin/python" - <<'PY'
import ctypes
ctypes.CDLL("libcublas.so.12")
ctypes.CDLL("libcudnn.so.9")
print("CUDA libs OK")
PY
```

### Normal Pilot And Production Run

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export AUDIO_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs
export ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/asr-venv
export ASR_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/asr_models/whisper-large-v3-ct2
export ASR_MODEL=large-v3
export ASR_DEVICE=cuda:0
export ASR_COMPUTE_TYPE=float16

ASR_SITE=$(
"$ASR_ENV/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
export LD_LIBRARY_PATH="$ASR_SITE/nvidia/cublas/lib:$ASR_SITE/nvidia/cudnn/lib:$ASR_SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO"
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

Confirm `parent_quota_applied=false`, `donor_media_used=false`, the selected
target count, turn count, and output root `$AUDIO_RUN_ROOT/asr_pilot20`.

Run the fixed pilot only when inference regeneration is intended:

```bash
cd "$REPO"
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

Regenerate the static QA page without loading Whisper, CUDA, or changing
`inventory.json`, `turns.jsonl`, or `summary.json`:

```bash
cd "$REPO"
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --regenerate-review

cd "$AUDIO_RUN_ROOT/asr_pilot20"
python -m http.server 8766 --bind 127.0.0.1
```

Review exact turn crops, then use **Export QA JSON**. Labels persist in browser
localStorage; export includes `r2v.h3.asr_human_qa.1`, the inventory
fingerprint, deterministic turn order, and explicit unlabeled count.

Formal production remains gated on ASR/segmentation policy decisions. Once that
gate is explicitly cleared, use:

```bash
cd "$REPO"
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production

python -m json.tool "$AUDIO_RUN_ROOT/production/asr/summary.json"
```

Production includes every authoritative turn from every unique target in-pair.
It has no `limit`, parent quota, calibration sampler, or cross-pair duplication.

### Validated Pilot Milestone

- Source inventory: 75 unique production target clips.
- Pilot: 20 clips, 82 turns, 82 backend calls.
- Runtime after CUDA library repair: 81 transcribed, one uncertain, zero failed.
- Human QA: 59 `CORRECT`, 15 `WRONG`, 8 `UNCERTAIN`; all 82 labeled.
- Explicit `CORRECT` rate among `CORRECT` plus `WRONG`: 59/74 (79.7%).

The QA set includes Cantonese, Hong Kong-accented Mandarin, isolated phonetic
substitutions, proper-name/near-homophone errors, and likely short-context or
segmentation limitations. Dedicated Whisper-large-v3 remains the baseline and
is more trustworthy than dots3-generated dialogue, but raw ASR is not final H3
truth. DiariZen speaker segmentation/continuity and constrained contextual
resolution remain outside frozen ASR V1.

## DiariZen-Assisted Speaker Binding Pilot

This calibration stage reuses the exact ordered target list and fingerprint in
`$AUDIO_RUN_ROOT/asr_pilot20/inventory.json`. It reads canonical full audio and
raw frozen AudioEntityBinding evidence from production in-pairs, creates one
DiariZen call per unique target clip, and never reads cross-pairs as jobs.
Outputs use only `$AUDIO_RUN_ROOT/diarization_pilot20`.

Dry-run before staging or loading the model:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

Expected runtime variables after the separate environment and local cache have
been staged:

```bash
export DIARIZEN_PYTHON=/mnt/workspace/litengjie/data/audio_deps/diarizen-venv/bin/python
export DIARIZEN_CODE_ROOT=/mnt/workspace/litengjie/data/audio_deps/DiariZen
export DIARIZEN_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/diarizen-model-cache
export DIARIZEN_MODEL_IDENTIFIER=BUT-FIT/diarizen-wavlm-large-s80-md-v2
export DIARIZEN_DEVICE=cuda:0
export DIARIZEN_TIMEOUT_SECONDS=900
```

Real pilot after an operator has validated the dedicated environment:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite

cd "$AUDIO_RUN_ROOT/diarization_pilot20"
"$R2V_PYTHON" -m http.server 8767 --bind 127.0.0.1
```

### Accepted Calibration And Production

The validated runtime loaded
`BUT-FIT/diarizen-wavlm-large-s80-md-v2` through the official pipeline and made
20 backend calls. The first attempt produced 11 ready clips, nine failed clips,
and no empty clips. All nine failures were
`DiarizationBackendFailure: diarization segment exceeds canonical source audio`.
The successful subset contained 27 raw segments and 12 clusters: 11 mapped, one
ambiguous, zero unbound, and zero conflict. Its 2.54-second median DiariZen
segment and 1.08-second legacy median turn are partial 11-ready-clip evidence,
not final pilot-quality results.

The bridge now intersects a reported terminal segment with authoritative
canonical sample extent, clamps only its effective end to EOF, and preserves
reported times plus exact overrun diagnostics. This is not an arbitrary time
tolerance. The repaired rerun completed all 20 clips with 50 raw segments and
22 clusters: 21 mapped, one ambiguous abstention, zero unbound, and zero
conflict. Human QA marked all 22 cluster decisions `CORRECT`; none were wrong,
uncertain, or unlabeled.

Mapped speaker time was 114.4155 seconds, comprising 89.62 direct-anchor seconds
and 24.7955 within-cluster identity-propagated seconds. Nine accepted EOF
intersections totaled 0.2245 seconds, with median 0.0205 and maximum 0.0525
seconds / 840 samples. This validates the existing threshold-free sparse-anchor
policy; it does not introduce a coverage, share, or margin gate.

Formal production dry-run:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --dry-run
```

In `summary.json`, `identity_propagated_speaker_seconds` means mapped speaker
duration whose identity comes from within-cluster speaker continuity rather
than direct LR-ASD/Visual evidence. It includes unanchored portions of a segment
that also contains a direct anchor. Boundary counts and exact overrun
distribution remain part of production diagnostics.

Review the overlap-preserving pilot lanes and retain exported cluster QA as
calibration evidence only. Formal production uses all 75 targets and writes to
the fixed `$AUDIO_RUN_ROOT/production/diarization` root:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production
```

Add `--overwrite` only for an intentional complete rerun. Production has no
clip limit or parent quota, reads no cross-pair jobs, and does not rerun ASR.
The model candidate and staging procedure are recorded in
`docs/H3_DIARIZEN_SPEAKER_BINDING.md`. Official source is MIT licensed, while
released model weights are CC BY-NC 4.0 for research/non-commercial use.

### ASR V2 DiariZen-segment production

Formal DiariZen production completed 75/75 clips with 179 raw segments and 81
clusters: 79 candidate-mapped, one ambiguous, one unbound, and zero conflict.
Mapped speaker time was 461.828 seconds: 371.345 direct-anchor and 90.483
identity-propagated. Forty EOF-adjusted segments had median positive overrun
0.0305 seconds and maximum 0.0805 seconds / 1288 samples. Do not claim all 81
production clusters were human-reviewed; the accepted 22/22 QA belongs to the
calibration pilot.

ASR V2 uses the same complete Whisper-large-v3 CT2 float16 backend and decoder
settings as V1. Only segmentation changes. It reads exact effective production
DiariZen sample intervals, retains overlapping speakers and unresolved identity,
and never reruns DiariZen or modifies ASR V1.

The accepted same-clip/same-model pilot contained 20 clips and 50 segments: 48
transcribed, two backend-uncertain, and zero failed. Human QA labeled all 50:
41 `CORRECT`, three `WRONG`, six `UNCERTAIN`, and zero unlabeled. Backend
uncertain `2` and human-QA uncertain `6` are different concepts. The ASR V1
reference contained 82 shorter units with QA 59/15/8. Because the units differ,
do not report this as a paired per-turn accuracy delta. Production raw ASR
remains immutable and does not filter its own records. The separate
text-usability sidecar applies the frozen display-only threshold.

```bash
export ASR_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/asr_models/whisper-large-v3-ct2
export ASR_MODEL=large-v3
export ASR_DEVICE=cuda:0
export ASR_COMPUTE_TYPE=float16

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --dry-run

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production

# Explicit full replacement only when intentionally requested:
"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --overwrite

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --regenerate-review
```

Serve the static production review:

```bash
cd "$AUDIO_RUN_ROOT/production/asr_v2"
python -m http.server 8768 --bind 127.0.0.1
```

### ASR V2 transcript-usability calibration

Raw ASR V2 production remains read-only. This analyzer joins the complete
50-row pilot human QA to the frozen pilot, evaluates simple diagnostic rules,
and applies shortlisted rules to production as coverage-only shadows. It does
not run Whisper, DiariZen, a GPU, or a network model, and it does not freeze a
text policy. In particular, `language_probability` is not transcript
correctness probability.

```bash
export QA_JSON=/path/to/browser-exported/asr_v2_pilot20_human_qa.json

"$R2V_PYTHON" tools/analyze_h3_asr_v2_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --qa-json "$QA_JSON" \
  --overwrite

cd "$AUDIO_RUN_ROOT"
python -m http.server 8768 --bind 127.0.0.1
# Open asr_v2_text_calibration/report.html
```

The fixed output is `$AUDIO_RUN_ROOT/asr_v2_text_calibration`. Every shortlisted
rule is `CALIBRATION CANDIDATE ONLY`; production shadow results are coverage,
not precision or accuracy. See `docs/H3_ASR_V2_TEXT_USABILITY.md`.

### Frozen ASR V2 text-usability policy

The accepted V1 rule trusts display text only for `transcribed`, non-empty raw
ASR records with `language_probability >= 0.65`. The exact threshold is a
human-approved eligibility threshold, not transcript correctness probability.
No identity, voice, pair, duration, or other decoder diagnostic is a text gate.

Inspect the complete production plan without an ASR, DiariZen, or GPU runtime:

```bash
"$R2V_PYTHON" tools/run_h3_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

Publish the fixed derived sidecar:

```bash
"$R2V_PYTHON" tools/run_h3_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT"

python -m json.tool \
  "$AUDIO_RUN_ROOT/production/text_usability/summary.json"
```

The command reads `$AUDIO_RUN_ROOT/production/asr_v2` only and writes
`$AUDIO_RUN_ROOT/production/text_usability`. It does not overwrite the retained
calibration report. Use `--overwrite` only for an intentional atomic replacement
of the derived sidecar. Final `<d>` rendering is a separate model-free stage.

### Deterministic final H3 renderer

First inspect the complete plan. This command reads frozen production
artifacts, hashes source media, and writes nothing:

```bash
"$R2V_PYTHON" tools/run_h3_final_renderer.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

Publish every in-pair and cross-pair row to the fixed derived root:

```bash
"$R2V_PYTHON" tools/run_h3_final_renderer.py \
  --audio-run-root "$AUDIO_RUN_ROOT"

python -m json.tool "$AUDIO_RUN_ROOT/production/h3/summary.json"
wc -l \
  "$AUDIO_RUN_ROOT/production/h3/samples.jsonl" \
  "$AUDIO_RUN_ROOT/production/h3/failures.jsonl"
```

No Whisper, DiariZen, MLLM, GPU, or network service is needed. The renderer
reuses the exact accepted Visual V3 instruction, frozen pair order, ASR V2
timeline, and authoritative text-usability sidecar. Detected language is
metadata only and is never inserted as prompt conditioning. Cross-pairs replace
only the subject voice asset; donor transcript and donor timeline are never
read. Use `--overwrite` only to atomically replace `production/h3` after an
intentional derived-output rebuild. See `docs/H3_FINAL_RENDERER.md`.

## Blocked dots3 diagnostic sequence

Current milestone boundaries:

- Audio binding, primary voice, embeddings, and PairPolicy V1 are **COMPLETE / FROZEN**.
- dots3 native-video semantic augmentation is **BLOCKED AFTER FAILED HUMAN QA**.
- The H3 exporter is **DEMO / NOT FINAL**.
- Visual subject attributes are a separate workstream and are not consumed here.

The old Qwen/DashScope semantic commands are obsolete and are not a production
fallback. The following records the validated dots3 runtime only. Do not run
complete semantic production until a future trusted-ASR semantic contract is
implemented and reviewed.

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

The current validated deployment endpoint is
`http://6.167.57.88:8000/v1`. This is current server deployment configuration,
not a stable API address, and may change later. Python code must never hardcode
this IP; every producer continues to read `DOTS3_BASE_URL` from the environment.

From another machine with network access to the deployment, verify:

```bash
curl --fail --silent --show-error \
  http://6.167.57.88:8000/v1/models | python -m json.tool
```

When operating directly on the same host as vLLM, `127.0.0.1` is an optional
same-host health-check alternative only:

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

### J. Complete dots3 semantic production is blocked

Do not run `tools/run_h3_omni_semantic.py --mode production`. The native-video
20-clip pilot produced severe hallucinated dialogue and remains diagnostic only.

### K. Inspect existing diagnostic output only

```bash
python -m json.tool "$AUDIO_RUN_ROOT/semantic_pilot20/summary.json"
```

## Canonical directory layout

```text
/mnt/workspace/litengjie/data/
├── R2V_DATA_V2/                         # repository
├── audio_deps/                          # Audio-only dependencies
│   ├── LR-ASD/                          # pinned vendor checkout
│   ├── lr-asd-venv/                     # Python 3.10 uv venv
│   ├── embedding-venv/                  # isolated face/speaker inference env
│   ├── asr-venv/                        # isolated Whisper/faster-whisper env
│   ├── asr_models/                      # writable local CT2 Whisper models
│   ├── embedding_models/                # manually staged local model files
│   └── uv-python/                       # uv managed Python
└── r2v_audio_runs/
    ├── smoke/                           # one reusable single-clip smoke output
    ├── pilot20/                         # one reusable 20-clip validation output
    ├── asr_pilot20/                     # fixed Whisper ASR pilot output
    ├── pair_calibration/                # fixed pair-calibration workspace
    │   ├── plan/                        # deterministic Visual clip plan
    │   ├── face_mining/                 # Visual-only face retrieval + review
    │   ├── face_audio_plan/              # HUMAN SAME pair Audio endpoints
    │   ├── pair_policy_review/           # HUMAN hard-negative review/report
    │   ├── audio/                       # LR-ASD Audio binding outputs
    │   ├── primary_voice/               # frozen V1 primary voice references
    │   └── embedding/                   # face/speaker retrieval diagnostics
    └── production/                      # fixed formal production outputs
        ├── pairs/                       # frozen in/cross pair source artifacts
        └── asr/                         # all authoritative turn transcripts
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

## Target Audio Caption MLLM Pilot V1

This is a fixed 20-clip human-review pilot, not the blocked dialogue-producing
semantic workflow and not a 75-clip production run. It reuses the exact ordered
clip IDs in `$AUDIO_RUN_ROOT/asr_v2_pilot20` and reads the frozen production
DiariZen, ASR V2, TextUsability, and Audio artifacts without modifying them.

The validated Dots3 runtime above is fully reusable: checkpoint
`/mnt/workspace/public/pretrained/dots3-note-prev`, served name
`dots3-note-prev`, pinned vLLM commit
`e0e5a7fb2808504ba86c94f7b379e38496002fd0`, shared `file://` media under
`/mnt/workspace`, and endpoint `http://6.167.57.88:8000/v1`. The request is text
plus native target video only. The video's embedded audio is analyzed; the
canonical extracted full-audio path/hash remains verified provenance and is not
sent separately.

After starting vLLM with the already documented command and restoring the
`DOTS3_*` environment variables, first verify the frozen inventory:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

Run the pilot once:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT"
```

Inspect the summary and serve the static review bundle:

```bash
cat "$AUDIO_RUN_ROOT/target_audio_caption_pilot20/summary.json"
cd "$AUDIO_RUN_ROOT/target_audio_caption_pilot20"
"$R2V_PYTHON" -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/report.html`. Exported QA includes the frozen
inventory fingerprint and `CORRECT` / `WRONG` / `UNCERTAIN` decisions plus
optional hallucination, ambience, delivery, dialogue-leakage, and other flags.
Do not run a 75-clip caption job or integrate the draft into final H3 before the
20-clip human review is accepted.

The existing `target_audio_caption_pilot20` completed 20/20 runtime calls. Its
ASR-pilot clips are retained as a clean/negative set for checking abstention and
hallucination. Formal QA was intentionally skipped because the set lacks useful
positive background cases. It cannot establish ambience, music, or
non-speech-event recall or approve Target Audio Caption production.

### Model-free background-audio scout

Build a separate navigation report over all 75 frozen production targets. This
command reads `production/audio` and `production/diarization`, calls no model,
applies no threshold or parent quota, and does not modify the clean pilot:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/scout_h3_background_audio.py \
  --audio-run-root "$AUDIO_RUN_ROOT"
```

Inspect the fixed output and serve its static review page:

```bash
cat "$AUDIO_RUN_ROOT/background_audio_scout/summary.json"
cd "$AUDIO_RUN_ROOT/background_audio_scout"
"$R2V_PYTHON" -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/report.html`, listen to the clips, and manually label
every obvious positive as `background-rich`. The current 75-target dataset has
fewer than 20 such clips; this is dataset-distribution-specific, not a global
assumption. The positive pilot is therefore dynamically sized and non-empty.
Never fabricate positives to meet a quota. The exported file is
`background_audio_pilot_selection.json`. Non-speech RMS/ratio/duration are
sorting aids only; they do not establish that music or another background event
is audible. Prompt V3 asks for one freeform nullable background-audio prompt,
including faint or speech-masked accompaniment when audible, without a
replacement taxonomy. Run it over exactly the same manual selection in its
separate fixed output root:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --clip-selection-json /path/to/background_audio_pilot_selection.json

cat "$AUDIO_RUN_ROOT/target_audio_caption_background_pilot_v3/summary.json"
```

This command does not read the `asr_v2_pilot20` clip selection and does not
overwrite `$AUDIO_RUN_ROOT/target_audio_caption_pilot20`, the V1 A/B baseline at
`$AUDIO_RUN_ROOT/target_audio_caption_background_pilot`, or the V2 baseline at
`$AUDIO_RUN_ROOT/target_audio_caption_background_pilot_v2`.

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
## JEA target-audio-caption A/B

The current JEA semantic-caption sidecar is independent from the historical
ASR-V2/TextUsability pilot. Validate and run Dots3 and Qwen3-Omni separately:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend dots3 \
  --dry-run

"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend qwen3-omni \
  --dry-run
```

The Dots3 runtime reads only `DOTS3_*`; the Qwen3-Omni runtime reads only
`QWEN3_OMNI_*`. Dots3 transports native target video with embedded audio.
Qwen3-Omni-30B-A3B-Instruct transports both whole target video and canonical
full audio and requests text output only. Qwen3-Omni-Captioner is not part of
this A/B because it does not accept the required text prompt.
Neither receives Qwen3-ASR transcript text, entity IDs, donor media, reference
images, or primary voice. Outputs remain separate under
`audio_caption/dots3/` and `audio_caption/qwen3_omni/`; never use `--overwrite`
until the exact backend directory has been reviewed.

## Canonical clip-level Audio universe

Visual Production `samples.jsonl` is the sole clip-level admission source. All
validated rows must appear in `audio/canonical_clips.jsonl` and receive canonical
full audio, even when they have no subject reference, primary voice, pair,
DiariZen speaker, or Qwen3-ASR segment. Those subject/speech stages are optional
enrichment only. Subject Audio binding continues to process only the
subject-reference subset and no empty subject, speaker, voice, or transcript is
fabricated for other clips.

For an existing production root, publish/backfill the canonical Audio manifest
without any AI model call:

```bash
"$R2V_PYTHON" tools/backfill_h3_canonical_audio.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

The specialized Audio semantics inventory is built from that manifest. Readable
DiariZen speaker evidence is an optional subset overlay and Qwen3-ASR is not an
inventory dependency. Future clip-level Visual/Audio rendering joins exactly on
`clip_uid`.
