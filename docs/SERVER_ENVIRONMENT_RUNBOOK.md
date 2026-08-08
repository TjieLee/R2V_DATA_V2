# R2V V3 Server Environment Runbook

Last updated: 2026-08-08

This file is the single source of truth for the Linux server environment used by
the V3 pipeline. Update it whenever a runtime, source checkout, model path,
service endpoint, GPU assignment, launch command, or isolation rule changes.
Do not reconstruct these values from chat history.

`CONFIRMED` means the value is backed by tracked repository evidence or supplied
server output. `UNVERIFIED` means it must be checked on the server before use.
Never replace `UNVERIFIED` with a guess.

## 1. Runtime layers

```text
PRODUCTION
  main R2V .venv
    - pipeline Python
    - SAM3 source through PYTHONPATH
    - Qwen OpenAI client
  external service
    - vLLM serving Qwen3-VL
  separate persistent worker
    - Boogu Image runtime

AUDIT ONLY
  main R2V .venv
    - DINOv2 image embeddings
    - SigLIP2 image embeddings
  separate reference-filter venv
    - SCRFD through ONNX Runtime
    - MediaPipe Face Landmarker
```

Never install SAM3 into the pose venv, run the production pipeline from the pose
venv, or import the Boogu runtime into the main R2V process.

## 2. Environment inventory

| Component | Purpose | Code/runtime | Model/checkpoint | Endpoint | Scope | Status and notes |
| --- | --- | --- | --- | --- | --- | --- |
| R2V main pipeline | V3 orchestration | `/mnt/workspace/litengjie/data/R2V_DATA_V2`; `.venv/bin/python` | Configured local models | Qwen client | Production | Paths CONFIRMED; versions UNVERIFIED |
| Qwen3-VL / vLLM | Structured VLM requests | `/usr/bin/python3 /usr/local/bin/vllm`; resolved Python `/usr/bin/python3.12` | `/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct` | `http://127.0.0.1:8000/v1` | Production service | Launch command and runtime versions CONFIRMED; TP workers used H200 GPUs 0-3 at validation time |
| SAM3 | Tracking and review | `/mnt/workspace/litengjie/data/vendor/sam3`, imported through `PYTHONPATH` | `/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt` | In-process | Production | Paths CONFIRMED; checkout revision UNVERIFIED |
| Boogu Image | Reference editing | `/mnt/workspace/litengjie/data/vendor/Boogu-Image`; `/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python` | `/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708` | Persistent JSONL worker | Production | Source revision `25f8f888298224a94e5ec2abafb98abea9031a0d`; model revision `hotfix-1k-20260708` |
| DINOv2-large | Representativeness evidence | Main R2V `.venv`; `tools/reference_filter_adapters/transformers_embedding` | `/mnt/workspace/public/pretrained/facebook/dinov2-large` | External audit worker | Audit only | Image encoder only; not a production selector |
| SigLIP2 | Representativeness evidence | Main R2V `.venv`; same adapter | `/mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex` | External audit worker | Audit only | Image encoder only; text encoder is not used by this audit |
| SCRFD | Face detection | Pose venv; `tools/reference_filter_adapters/subject_pose` | `/mnt/workspace/public/pretrained/face_analysis/models/scrfd_10g_bnkps.onnx` | External audit worker | Audit only | Supplied server probe confirms dynamic input and 9 outputs at `640x640` |
| MediaPipe Face Landmarker | Head pose | Pose venv; same adapter | `/mnt/workspace/public/pretrained/face_analysis/models/face_landmarker_v2_with_blendshapes.task` | External audit worker | Audit only | Model and API creation CONFIRMED; explicit `close()` remains mandatory |
| Pose venv | Isolated SCRFD/MediaPipe runtime | `/mnt/workspace/litengjie/data/venvs/r2v-reference-filter/bin/python` | Face-analysis model root above | None | Audit only | Versions and clean `pip check` CONFIRMED; must always run with `env -u PYTHONPATH` |
| OneAlign | No active use | Runtime UNVERIFIED | `/mnt/workspace/public/pretrained/q-future/one-align` | None | Not active | Roughly 16 GB mPLUG-Owl2-family model; unsuitable for the cheap prefilter goal |

The public dataset and pretrained roots are read-only. Never create caches,
temporary files, lock files, or modified artifacts below either root.

## 3. Start a fresh production shell

Do not enable `set -e`, `set -u`, or `set -o pipefail` in an interactive
terminal. A failed inspection command can otherwise close the shell.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/activate

set +e
set +u
set +o pipefail

export R2V_REPO_ROOT="/mnt/workspace/litengjie/data/R2V_DATA_V2"
export MAIN_PYTHON="$R2V_REPO_ROOT/.venv/bin/python"
export BOOGU_CODE_ROOT="/mnt/workspace/litengjie/data/vendor/Boogu-Image"
export BOOGU_PYTHON="/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python"
export BOOGU_MODEL_PATH="/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708"
export SAM3_CODE_ROOT="/mnt/workspace/litengjie/data/vendor/sam3"
export SAM3_MODEL_PATH="/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt"
export QWEN_MODEL_PATH="/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export PYTHONPATH="$SAM3_CODE_ROOT:$R2V_REPO_ROOT:${PYTHONPATH:-}"

export HF_HOME="/mnt/workspace/litengjie/data/cache/huggingface"
export TORCH_HOME="/mnt/workspace/litengjie/data/cache/torch"
export XDG_CACHE_HOME="/mnt/workspace/litengjie/data/cache/xdg"
export TMPDIR="/mnt/workspace/litengjie/data/tmp"
```

Do not globally export `CUDA_VISIBLE_DEVICES` in this shell. Assign a service GPU
only on its launch command, or use the explicit device fields in the active
configuration. During Boogu integration, physical GPU `6` was exposed inside
the worker as `cuda:0`; this is historical evidence, not authority to override
the current `reference_edit.cuda_visible_devices` value.

A machine-local `server_env.sh` may contain these exports. It is not tracked and
must never contain credentials in this document. Validate before sourcing:

```bash
bash -n /mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh
source /mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh
```

## 4. Fixed-path and version preflight

Run these checks after every reconnect and before a smoke, pilot, or production
run. They are read-only with respect to public model paths.

```bash
printf 'repo=%s\npython=%s\nhead=%s\nbranch=%s\n' \
  "$R2V_REPO_ROOT" "$MAIN_PYTHON" \
  "$(git rev-parse HEAD)" "$(git branch --show-current)"

test -x "$MAIN_PYTHON" || echo "ERROR: missing main Python"
test -d "$SAM3_CODE_ROOT" || echo "ERROR: missing SAM3 source"
test -f "$SAM3_MODEL_PATH" || echo "ERROR: missing SAM3 checkpoint"
test -x "$BOOGU_PYTHON" || echo "ERROR: missing Boogu Python"
test -d "$BOOGU_CODE_ROOT" || echo "ERROR: missing Boogu source"
test -d "$BOOGU_MODEL_PATH" || echo "ERROR: missing Boogu model"
test -d "$QWEN_MODEL_PATH" || echo "ERROR: missing Qwen model"

"$MAIN_PYTHON" - <<'PY'
import sys
import numpy
import PIL
import torch
import transformers

print("Python", sys.version)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("Pillow", PIL.__version__)
try:
    import cv2
except ImportError:
    print("opencv", "NOT INSTALLED")
else:
    print("opencv", cv2.__version__)
PY

"$MAIN_PYTHON" - <<'PY'
from pathlib import Path
import sam3
from sam3.model_builder import build_sam3_video_predictor

package = Path(sam3.__file__).resolve()
expected = Path("/mnt/workspace/litengjie/data/vendor/sam3").resolve()
print("sam3 package", package)
if expected not in package.parents:
    raise SystemExit(f"Unexpected SAM3 package: {package}")
print("SAM3 import PASS", build_sam3_video_predictor)
PY

git -C "$SAM3_CODE_ROOT" rev-parse HEAD
git -C "$BOOGU_CODE_ROOT" rev-parse HEAD

env -u PYTHONPATH "$BOOGU_PYTHON" - <<'PY'
import sys
import torch
print("Python", sys.version)
print("torch", torch.__version__)
PY
```

Record the output in a server validation note when versions or revisions change.
The exact main, SAM3, and Boogu package versions were not available to the Mac
checkout during the 2026-08-08 update and remain `UNVERIFIED` here.

## 5. Qwen3-VL and vLLM

Confirmed client values:

```text
model path: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
base URL:   http://127.0.0.1:8000/v1
API key:    EMPTY
```

The tracked repository confirms that full-video service startup must include:

```text
--media-io-kwargs '{"video":{"num_frames":-1}}'
--allowed-local-media-path /mnt/workspace/public/dataset
```

Confirmed vLLM runtime:

```text
command entry point: /usr/bin/python3 /usr/local/bin/vllm
resolved executable: /usr/bin/python3.12
Python:              3.12.3
torch:               2.11.0+cu130
vllm:                0.24.0
served model name:   /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
tensor parallel:     4
max model length:    32768
max sequences:       1
GPU memory fraction: 0.90
execution mode:      enforce eager
```

The following options were not explicitly set in the confirmed launch command:

```text
dtype:                not explicitly set in confirmed launch command
trust_remote_code:    not explicitly set in confirmed launch command
CUDA_VISIBLE_DEVICES: not explicitly set in confirmed launch command
```

At validation time, the four active tensor-parallel workers mapped to GPU
indices `0,1,2,3`; all four devices were NVIDIA H200 GPUs. This observed mapping
does not imply an unrecorded `CUDA_VISIBLE_DEVICES` setting.

Exact confirmed cold-start command:

```bash
/usr/bin/python3 /usr/local/bin/vllm serve \
  /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --allowed-local-media-path /mnt/workspace/public/dataset \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

The repository's 8B README example is not the production command and must not
be reused as one.

When the service is running, identify it without broad process termination:

```bash
ps -ef | grep '[v]llm'
ps -ef | grep '[a]pi_server'

# Set this only after reading the process table.
VLLM_PID="UNVERIFIED_PID"
test -r "/proc/$VLLM_PID/cmdline" || echo "ERROR: invalid vLLM PID"
readlink -f "/proc/$VLLM_PID/exe"
tr '\0' ' ' < "/proc/$VLLM_PID/cmdline"
printf '\n'
```

Use `/proc/<PID>/environ` only when necessary and never copy credentials or
tokens into tracked documentation. Reconfirm the executable and versions after
any service environment change:

```bash
VLLM_EXE="$(readlink -f "/proc/$VLLM_PID/exe")"
"$VLLM_EXE" - <<'PY'
import sys
import torch
import vllm
print("Python", sys.version)
print("vllm", vllm.__version__)
print("torch", torch.__version__)
PY
```

Health check:

```bash
curl -fsS --noproxy '*' http://127.0.0.1:8000/v1/models
```

Before launch, inspect GPU occupancy with `nvidia-smi`. Do not add or infer
`CUDA_VISIBLE_DEVICES`, dtype, or `trust-remote-code` unless the deployed command
is deliberately changed and revalidated.

For shutdown, first print and inspect `/proc/$VLLM_PID/cmdline`, then terminate
that specific PID with `kill -TERM "$VLLM_PID"`. Never use `pkill -f python`, a
broad process pattern, or `kill -9` as the first action.

## 6. Pose audit environment

```bash
export POSE_VENV="/mnt/workspace/litengjie/data/venvs/r2v-reference-filter"
export POSE_PYTHON="$POSE_VENV/bin/python"
export POSE_ADAPTER_ROOT="$R2V_REPO_ROOT/tools/reference_filter_adapters/subject_pose"
export POSE_MODEL_ROOT="/mnt/workspace/public/pretrained/face_analysis/models"
export SCRFD_MODEL="$POSE_MODEL_ROOT/scrfd_10g_bnkps.onnx"
export FACE_LANDMARKER_MODEL="$POSE_MODEL_ROOT/face_landmarker_v2_with_blendshapes.task"

test -x "$POSE_PYTHON" || echo "ERROR: missing pose Python"
test -f "$SCRFD_MODEL" || echo "ERROR: missing SCRFD model"
test -f "$FACE_LANDMARKER_MODEL" || echo "ERROR: missing Face Landmarker model"
```

Every pose command must remove the production shell's `PYTHONPATH`:

```bash
env -u PYTHONPATH "$POSE_PYTHON" -m pip check
env -u PYTHONPATH "$POSE_PYTHON" -m pip show sam3
```

Confirmed pose runtime:

```text
Python:      3.12.3
numpy:       2.5.1
Pillow:      12.3.0
opencv/cv2:  5.0.0
onnxruntime: 1.28.0
mediapipe:   1.0.0
pip check:   No broken requirements found.
```

The expected `sam3` result is `Package(s) not found`. Do not install SAM3 here.
This pose runtime must use NumPy 2.x because the current
`opencv-contrib-python 5.0.0.93` requirement is `numpy>=2`; do not apply SAM3's
NumPy constraint to this isolated environment.

Capture real versions rather than guessing:

```bash
env -u PYTHONPATH "$POSE_PYTHON" - <<'PY'
import sys
import numpy
import PIL
import cv2
import onnxruntime
import mediapipe

print("Python", sys.version)
print("numpy", numpy.__version__)
print("Pillow", PIL.__version__)
print("opencv", cv2.__version__)
print("onnxruntime", onnxruntime.__version__)
print("mediapipe", mediapipe.__version__)
print("providers", onnxruntime.get_available_providers())
print("BaseOptions", mediapipe.tasks.BaseOptions)
print("FaceLandmarkerOptions", mediapipe.tasks.vision.FaceLandmarkerOptions)
print("FaceLandmarker.create_from_options", mediapipe.tasks.vision.FaceLandmarker.create_from_options)
print("Image", mediapipe.Image)
print("ImageFormat.SRGB", mediapipe.ImageFormat.SRGB)
PY
```

Real local model preflight:

```bash
env -u PYTHONPATH "$POSE_PYTHON" - \
  "$POSE_ADAPTER_ROOT" "$POSE_MODEL_ROOT" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
import r2v_reference_filter_adapter as adapter

scorer = adapter.load_scorer(
    kind="subject_pose",
    backend="scrfd_mediapipe",
    model_path=Path(sys.argv[2]),
    local_files_only=True,
)
try:
    print("scrfd_model_input_shape", scorer.detector.model_input_shape)
    print("scrfd_dynamic_spatial", scorer.detector.dynamic_spatial)
    print("scrfd_inference_size", scorer.detector.input_width, scorer.detector.input_height)
    print("POSE MODEL LOAD PASS")
finally:
    scorer.close()
del scorer
import gc
gc.collect()
PY
```

MediaPipe Task lifecycle is explicit. `FaceLandmarker.close()` must run before
interpreter teardown; do not rely on `FaceLandmarker.__del__`. The external
reference-filter worker calls an adapter scorer's optional `close()` method from
a single `finally` block on shutdown, stdin EOF, or worker exit. DINOv2 and
SigLIP2 adapters do not need to implement `close()`.

Server evidence confirms that the SCRFD ONNX file and MediaPipe FaceLandmarker
both load successfully. A `(1,3,640,640)` SCRFD probe returns 9 tensors with rows
`12800`, `3200`, and `800` for score, bbox, and 5-landmark heads. The adapter uses
fixed `640x640` for dynamic H/W. Package versions and dependency consistency are
confirmed above. Real candidate detection rate and per-image pose runtime remain
`UNVERIFIED` until the candidate audit is completed.

## 7. Cold-start order

1. Start a fresh production shell using section 3.
2. Verify branch, commit, all fixed paths, environment versions, and SAM3 import.
3. Inspect or start the confirmed vLLM service, then wait for `/v1/models`.
4. Verify the Boogu source, Python, model, revision, and configured GPU.
5. Check the selected config and run-root identity.
6. Inspect stale pipeline and Boogu worker processes.
7. Run only the requested V3 stages.

```bash
ps -ef | grep '[r]un_pipeline_v3.py'
ps -ef | grep '[r]un_v3_boogu_reference_edit_worker.py'
```

If a stale process is found, inspect its full command and terminate only its
specific PID with `kill -TERM`. Do not use broad `pkill` commands.

## 8. Pipeline launch and profiling

Set explicit machine-local paths after validating them:

```bash
export CONFIG="/mnt/workspace/litengjie/data/r2v_v3_configs/UNVERIFIED.yaml"
test -f "$CONFIG" || echo "ERROR: invalid config"
```

The CLI runs only explicitly requested stages. A standard full sequence is:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,instruct,export
```

Run a bounded stage subset when validating a downstream change:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate
```

Add `--overwrite` only when regeneration is explicitly intended and the target
run root has been checked. Add observational profiling with `--profile`:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages reference_edit \
  --overwrite \
  --profile
```

Profiling writes `profiling/events.jsonl` and `profiling/summary.json` under the
configured run root. It does not change environment setup.

### Conservative reference prefilter rollout

The production reference prefilter is versioned and disabled by default:

```yaml
pair:
  reference_prefilter_mode: "off"  # or conservative_v1
```

`conservative_v1` applies only the frozen `subject_near_silhouette_v1` and
`subject_relative_blur_v2` rules before the Qwen candidate judge. It uses the
main R2V virtual environment and deterministic foreground-only cheap CV. It
does not load another model or start another worker. SCRFD, MediaPipe, face
evidence, and pose evidence remain `AUDIT ONLY` and are never production
prefilter inputs. Object and group references are not filtered by either rule.

Validate the opt-in mode with a fresh full20 run on the server. Never overwrite
the known baseline `full20-samfirst-bg-20260808-021450`. The following creates a
sibling config with independent run and export roots while preserving all other
frozen production settings:

```bash
export PREFILTER_BASE_CONFIG="/mnt/workspace/litengjie/data/r2v_v3_configs/full20-samfirst-bg-20260808-021450.yaml"
export PREFILTER_BASE_RUN="/mnt/workspace/litengjie/data/r2v_v3_runs/full20-samfirst-bg-20260808-021450"
export PREFILTER_STAMP="$(date +%Y%m%d-%H%M%S)"
export PREFILTER_CONFIG="/mnt/workspace/litengjie/data/r2v_v3_configs/full20-prefilter-${PREFILTER_STAMP}.yaml"
export PREFILTER_RUN="/mnt/workspace/litengjie/data/r2v_v3_runs/full20-prefilter-${PREFILTER_STAMP}"
export PREFILTER_EXPORT="/mnt/workspace/litengjie/data/r2v_v3_exports/full20-prefilter-${PREFILTER_STAMP}"

test -f "$PREFILTER_BASE_CONFIG" || echo "ERROR: missing baseline config"
test -d "$PREFILTER_BASE_RUN" || echo "ERROR: missing baseline run"
test ! -e "$PREFILTER_RUN" || echo "ERROR: target run already exists"
test ! -e "$PREFILTER_EXPORT" || echo "ERROR: target export already exists"

"$MAIN_PYTHON" - \
  "$PREFILTER_BASE_CONFIG" "$PREFILTER_CONFIG" \
  "$PREFILTER_RUN" "$PREFILTER_EXPORT" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination, run_root, export_root = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["run_root"] = str(run_root)
config["export_root"] = str(export_root)
config.setdefault("pair", {})["reference_prefilter_mode"] = "conservative_v1"
destination.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$PREFILTER_CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,instruct,export \
  --profile
```

Require the fresh run to finish through export. Compare its pair/reference-edit
states, selected references, export count, candidate-judge call count, input
image count, and total candidate-judge duration against `PREFILTER_BASE_RUN`.
For retained subsets, profiling must report the post-filter candidate count and
exactly two evidence images per retained candidate. Explain every difference;
do not infer server results from local tests.

### Final background binding guard

The final background guard is versioned and disabled by default:

```yaml
pair:
  background_final_guard_mode: "off"  # or qwen_v1
```

`qwen_v1` uses Qwen3-VL with exactly five images: the final full background and
four deterministic `upper_left`, `upper_right`, `lower_left`, and `lower_right`
tiles. The request contains only the annotation background phrase and grounding
prompt. It contains no forbidden-entity list, entity phrases, masks, source
comparison, embedding evidence, face evidence, or pose evidence.

The guard runs only after entity pairing is otherwise ready. An explicit reject,
request failure, timeout, malformed structured output, or unexpected guard error
removes only `<ref_bg_1>` from the pairing. It does not reject the entity sample
and does not rewrite the `clean_raw` or `ready_removed` background state. Profile
events use component `qwen_background_final_guard` with `image_count=5` and
`tile_count=4`.

The RC production configuration must explicitly contain the following values;
do not infer or silently change device allocation:

```yaml
qwen:
  annotation:
    repair_retries: 2
  background_final_judge:
    model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

pair:
  repair_retries: 2
  reference_prefilter_mode: conservative_v1
  background_final_guard_mode: qwen_v1

remove:
  device: cuda:4

reference_edit:
  cuda_visible_devices: "6"
```

The confirmed vLLM service occupies physical GPUs `0,1,2,3`; Boogu uses physical
GPU `6`. Keep the existing remover LoRA and its GPU `4` settings unchanged.

Replay the guard against the existing RC2 bound backgrounds before a new large
production run. Set `RC2_CONFIG` to the exact config associated with that run;
do not edit the source run or reuse it as an output directory:

```bash
export RC2_RUN="/mnt/workspace/litengjie/data/r2v_v3_runs/rc2-100-20260808-233059"
export RC2_CONFIG="/path/to/the/exact/rc2-config.yaml"
export RC2_STAMP="$(date +%Y%m%d-%H%M%S)"
export RC2_REPLAY="/mnt/workspace/litengjie/data/r2v_v3_audits/background-final-rc2-${RC2_STAMP}.jsonl"
export RC2_SNAPSHOT="/mnt/workspace/litengjie/data/r2v_v3_audits/background-final-rc2-${RC2_STAMP}-source.sha256"

test -d "$RC2_RUN" || echo "ERROR: missing RC2 run"
test -f "$RC2_CONFIG" || echo "ERROR: missing exact RC2 config"
find "$RC2_RUN" -type f -print0 | sort -z | xargs -0 sha256sum > "$RC2_SNAPSHOT.before"

"$MAIN_PYTHON" tools/replay_v3_background_final_guard.py \
  --config "$RC2_CONFIG" \
  --run-root "$RC2_RUN" \
  --base-url "$QWEN_BASE_URL" \
  --model "$QWEN_MODEL_PATH" \
  --output "$RC2_REPLAY"

find "$RC2_RUN" -type f -print0 | sort -z | xargs -0 sha256sum > "$RC2_SNAPSHOT.after"
diff -u "$RC2_SNAPSHOT.before" "$RC2_SNAPSHOT.after"
```

Review all records and `<output>.summary.json`. The replay calls Qwen only for
currently bound, ready backgrounds and writes only to the requested output and
its sibling summary. After replay sanity review, use three to five targeted fresh
samples for routing validation; do not immediately rerun a fresh 100-sample job.

Before `reference_edit`, verify Qwen health, SAM3 import, Boogu paths and GPU,
and absence of a stale worker. The stage starts one persistent Boogu JSONL worker
and reuses it for eligible entities. The worker's production settings remain:

```text
align_res=False
device=cuda:0
rewriter_device=cuda:0
enable_inner_devices_manager=False
unload_rewriter_level=keep
```

## 9. Run identity and reconnect recovery

`run.json` binds `run_id`, Git commit, config hash, model identifiers, and source
manifest. Do not reuse a run root with a different identity. Prefer a new run
root; never edit `run.json` casually.

Shell variables do not survive reconnects. Recover a known pilot only from its
selection file:

```bash
export PILOT_ROOT="$({
  find /mnt/workspace/litengjie/data/r2v_v3_runs \
    -mindepth 2 -maxdepth 2 -name pilot_selection.json -print
} | xargs -r ls -1t | head -1 | xargs -r dirname)"

export PILOT_CONFIG="$("$MAIN_PYTHON" - "$PILOT_ROOT/pilot_selection.json" <<'PY'
from pathlib import Path
import json
import sys
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["config_path"])
PY
)"

printf 'PILOT_ROOT=%s\nPILOT_CONFIG=%s\n' "$PILOT_ROOT" "$PILOT_CONFIG"
test -d "$PILOT_ROOT" || echo "ERROR: missing pilot root"
test -f "$PILOT_CONFIG" || echo "ERROR: missing pilot config"
```

An empty config variable resolves to the repository directory and causes
`IsADirectoryError`.

## 10. Optional audit and diagnostics

Everything in this section is `AUDIT ONLY`. It must not modify the source run,
select candidates, set thresholds, add Qwen calls, rerun pipeline stages, or
publish references.

```bash
export SOURCE_RUN="/mnt/workspace/litengjie/data/r2v_v3_runs/full20-samfirst-bg-20260808-021450"
export SOURCE_CONFIG="/mnt/workspace/litengjie/data/r2v_v3_configs/full20-samfirst-bg-20260808-021450.yaml"
export AUDIT_ROOT="/mnt/workspace/litengjie/data/r2v_v3_audits"
export REVIEW_ROOT="/mnt/workspace/litengjie/data/r2v_v3_reviews"
export EMBEDDING_ADAPTER_ROOT="$R2V_REPO_ROOT/tools/reference_filter_adapters/transformers_embedding"

test -d "$SOURCE_RUN" || echo "ERROR: missing source run"
test -f "$SOURCE_CONFIG" || echo "ERROR: missing source config"
```

Snapshot the source tree before and after an audit. Store snapshots outside the
source run and compare both paths and SHA-256 values:

```bash
STAMP="$(date +%Y%m%d-%H%M%S)"
SNAPSHOT_DIR="$AUDIT_ROOT/source-snapshots-$STAMP"
mkdir -p "$SNAPSHOT_DIR"
find "$SOURCE_RUN" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SNAPSHOT_DIR/before.sha256"

# Run one or more audit commands below.

find "$SOURCE_RUN" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$SNAPSHOT_DIR/after.sha256"
diff -u "$SNAPSHOT_DIR/before.sha256" "$SNAPSHOT_DIR/after.sha256"
```

### DINOv2 candidate audit

```bash
export DINO_MODEL="/mnt/workspace/public/pretrained/facebook/dinov2-large"
export DINO_AUDIT="$AUDIT_ROOT/dinov2-$STAMP"

"$MAIN_PYTHON" tools/audit_v3_reference_filters.py \
  --config "$SOURCE_CONFIG" \
  --run-root "$SOURCE_RUN" \
  --output-root "$DINO_AUDIT" \
  --artifact-scope candidates \
  --embedding-backend dinov2 \
  --embedding-python "$MAIN_PYTHON" \
  --embedding-code-root "$EMBEDDING_ADAPTER_ROOT" \
  --embedding-model-path "$DINO_MODEL"
```

### SigLIP2 candidate audit

```bash
export SIGLIP_MODEL="/mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex"
export SIGLIP_AUDIT="$AUDIT_ROOT/siglip2-$STAMP"

"$MAIN_PYTHON" tools/audit_v3_reference_filters.py \
  --config "$SOURCE_CONFIG" \
  --run-root "$SOURCE_RUN" \
  --output-root "$SIGLIP_AUDIT" \
  --artifact-scope candidates \
  --embedding-backend siglip2 \
  --embedding-python "$MAIN_PYTHON" \
  --embedding-code-root "$EMBEDDING_ADAPTER_ROOT" \
  --embedding-model-path "$SIGLIP_MODEL"
```

### Cheap CV plus SCRFD/MediaPipe audit

```bash
export POSE_AUDIT="$AUDIT_ROOT/pose-cheap-cv-$STAMP"

"$MAIN_PYTHON" tools/audit_v3_reference_filters.py \
  --config "$SOURCE_CONFIG" \
  --run-root "$SOURCE_RUN" \
  --output-root "$POSE_AUDIT" \
  --artifact-scope candidates \
  --technical-metrics cheap_cv \
  --subject-pose-backend scrfd_mediapipe \
  --subject-pose-python "$POSE_PYTHON" \
  --subject-pose-code-root "$POSE_ADAPTER_ROOT" \
  --subject-pose-model-path "$POSE_MODEL_ROOT"
```

The external audit worker removes inherited `PYTHONPATH` and sets
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`.
The parent production shell and all non-reference-filter processes are unchanged.

Inspect summary counts and runtime for subject candidates, face detections,
SCRFD success, Face Landmarker success, yaw distribution, and seconds per image.
Observe, but do not hardcode, the diver case
`7c75e9a21e0d29abbba6863c e1` and blue-shirted-man case
`6f46fce06e330bb20aa19f8e e1`.

### Review boards

```bash
"$MAIN_PYTHON" tools/build_v3_disagreement_review_board.py \
  --run-root "$SOURCE_RUN" \
  --dinov2-audit-root "$DINO_AUDIT" \
  --siglip2-audit-root "$SIGLIP_AUDIT" \
  --output-root "$REVIEW_ROOT/disagreement-$STAMP" \
  --mode all

"$MAIN_PYTHON" tools/build_v3_reference_filter_extreme_review_board.py \
  --run-root "$SOURCE_RUN" \
  --audit-root "$POSE_AUDIT" \
  --output-root "$REVIEW_ROOT/extremes-$STAMP" \
  --mode all
```

The extreme board must render darkest, blur, no-face, small-face, and
extreme-pose cases. It supplies human evidence only; this task sets no gate.

### Conservative prefilter shadow simulation

This simulation reads an existing candidate audit and estimates evidence changes
without altering candidates, Qwen calls, routing, references, or exports. The
`subject_near_silhouette_v1` condition is frozen at its original experimental
values. `subject_relative_blur_v1` is deprecated after a clear high-detail
candidate satisfied its relative-only thresholds. The current
`subject_relative_blur_v2` condition requires both relative blur evidence and
absolute values of at most 50 Laplacian variance and 1500 mean Tenengrad. The
shadow tools remain audit-only; the same two frozen rules are also available in
production only through the explicit `conservative_v1` mode documented above.

```bash
export SHADOW_SIMULATION="$AUDIT_ROOT/prefilter-shadow-$STAMP.json"
export SHADOW_REVIEW="$REVIEW_ROOT/prefilter-shadow-$STAMP"

"$MAIN_PYTHON" tools/simulate_v3_reference_prefilter.py \
  --audit-root "$POSE_AUDIT" \
  --output "$SHADOW_SIMULATION" \
  --rule all

"$MAIN_PYTHON" tools/build_v3_prefilter_shadow_review_board.py \
  --run-root "$SOURCE_RUN" \
  --audit-root "$POSE_AUDIT" \
  --simulation "$SHADOW_SIMULATION" \
  --output-root "$SHADOW_REVIEW" \
  --mode all
```

Available simulation rule modes are `near_silhouette`, `relative_blur_v2`, and
`all`. Review modes are `near_silhouette`, `relative_blur_v2`,
`qwen_selected_flagged`, `all_candidates_flagged`, and `all`. An
`all_candidates_flagged` state estimates one potentially skippable Qwen call but
does not skip the call or reject the entity. Pose, DINOv2, and SigLIP2 evidence
never triggers either shadow condition. Review every shadow case and validate a
larger audit set before discussing production integration.

### Large-run prefilter shadow validation

Run this only on the server. Start with the known candidate and inspect it before
selecting it; do not assume an older V3 artifact layout is compatible:

```bash
export LARGE_RUN="/mnt/workspace/litengjie/data/r2v_v3_runs/production500-s0-20260804-030702-d5f42ae5"
test -f "$LARGE_RUN/run.json" || echo "ERROR: missing large-run run.json"
find "$LARGE_RUN/clips" -mindepth 2 -maxdepth 3 -type d -name pair -print \
  | head -20
```

Set `SOURCE_RUN` to the selected compatible run and `SOURCE_CONFIG` to its exact
original config. If the preferred run is incompatible, inspect only immediate
children below `/mnt/workspace/litengjie/data/r2v_v3_runs`; do not search outside
that root. Require completed pair/candidate artifacts and substantially more
candidates than the full20 audit.

Use the source snapshot procedure above, then run only cheap CV and optional
SCRFD/MediaPipe evidence. Do not enable DINOv2, SigLIP2, Qwen, pair, or
reference-edit stages:

```bash
export LARGE_AUDIT="$AUDIT_ROOT/prefilter-large-cheap-cv-$STAMP"
export LARGE_SIMULATION="$AUDIT_ROOT/prefilter-large-shadow-v2-$STAMP.json"

"$MAIN_PYTHON" tools/audit_v3_reference_filters.py \
  --config "$SOURCE_CONFIG" \
  --run-root "$SOURCE_RUN" \
  --output-root "$LARGE_AUDIT" \
  --artifact-scope candidates \
  --technical-metrics cheap_cv \
  --subject-pose-backend scrfd_mediapipe \
  --subject-pose-python "$POSE_PYTHON" \
  --subject-pose-code-root "$POSE_ADAPTER_ROOT" \
  --subject-pose-model-path "$POSE_MODEL_ROOT"

"$MAIN_PYTHON" tools/simulate_v3_reference_prefilter.py \
  --audit-root "$LARGE_AUDIT" \
  --output "$LARGE_SIMULATION" \
  --rule all
```

Generate only flagged-case boards. Separate mode outputs keep each review set
small; when a set is large, review at most the first 20-30 deterministic boards
while retaining the complete case list in the simulation JSON:

```bash
for MODE in near_silhouette relative_blur_v2 qwen_selected_flagged all_candidates_flagged; do
  "$MAIN_PYTHON" tools/build_v3_prefilter_shadow_review_board.py \
    --run-root "$SOURCE_RUN" \
    --audit-root "$LARGE_AUDIT" \
    --simulation "$LARGE_SIMULATION" \
    --output-root "$REVIEW_ROOT/prefilter-large-${MODE}-$STAMP" \
    --mode "$MODE" \
    --max-cases 30
done
```

Before interpreting results, require both
`summary.by_reference_type.object.combined_flag_count` and
`summary.by_reference_type.group.combined_flag_count` to equal zero. Compare the
before/after source snapshots exactly. These checks still do not authorize a
production filter.

## 11. Known technical-only evidence

The existing 51-candidate cheap-CV audit had broad distributions:

```text
luma_mean:          min 9.06, p50 40.72, p90 83.56, max 119.72
dark_fraction_32:   min 0, p50 0.278, p90 0.912, max 0.970
laplacian_variance: min 3.51, p50 47.20, p90 381.37, max 761.79
tenengrad_mean:     min 58.34, p50 1430.56, p90 5419.57, max 12501.47
```

`thresholds_applied=false`. These observations do not justify a production gate.

## 12. Failure signatures

### `SCRFD input must have static NCHW spatial dimensions`

The old audit adapter rejected dynamic ONNX H/W. Current code accepts symbolic
or `None` spatial dimensions and runs deterministic `640x640` inference.

### Pose venv reports a SAM3 or NumPy conflict

The production `PYTHONPATH` leaked into the pose venv. Re-run with
`env -u PYTHONPATH`; confirm `pip show sam3` reports no package. Keep NumPy 2.x
for the current OpenCV contrib requirement.

### `ModuleNotFoundError: No module named 'sam3'` in production

The production shell lacks `$SAM3_CODE_ROOT` in `PYTHONPATH`. The checkpoint is
not the Python source package.

### Boogu output size or mixed-device failure

Keep `align_res=False`; use `pipeline.devices_manager(...)`; pass matching
`device` and `rewriter_device` on each worker request.

### `existing run.json does not match the requested V3 run`

The commit, config hash, model identifiers, source manifest, or run ID differs.
Use a new run root rather than weakening identity validation.

## 13. Maintenance checklist

Before giving or executing server commands:

1. Read this file and inspect the current Git branch and commit.
2. Print every path variable and run bounded `test -e`, `test -f`, `test -d`, or
   `test -x` checks on the server.
3. Capture only critical package versions; do not dump full `pip freeze`.
4. Verify vLLM from the real process and health endpoint, never from the 8B
   README example.
5. Verify SAM3 import origin and Boogu's separate runtime.
6. Use `env -u PYTHONPATH` for every pose-v-env command.
7. Keep audit outputs below `/mnt/workspace/litengjie/data/r2v_v3_audits` and
   review outputs below `/mnt/workspace/litengjie/data/r2v_v3_reviews`.
8. Hash the source run before and after read-only audits.
9. Do not add thresholds from review-board observations alone.
10. Replace `UNVERIFIED` only with actual server output and update this document
    in the same change.
