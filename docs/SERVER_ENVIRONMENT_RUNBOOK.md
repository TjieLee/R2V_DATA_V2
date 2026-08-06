# R2V V3 Server Environment Runbook

Last updated: 2026-08-06

This file is the single source of truth for starting and validating the Linux server environment used by the V3 pipeline. Any change to a model path, virtual environment, source checkout, service endpoint, GPU assignment, run-root convention, or startup command must update this file in the same change.

## 1. Fixed paths

```text
Repository:
/mnt/workspace/litengjie/data/R2V_DATA_V2

Main Python environment:
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv

Boogu source checkout:
/mnt/workspace/litengjie/data/vendor/Boogu-Image

Boogu Python environment:
/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python

Boogu model:
/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708

Boogu source revision:
25f8f888298224a94e5ec2abafb98abea9031a0d

Boogu model revision:
hotfix-1k-20260708

SAM3 checkpoint:
/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

Qwen model:
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

Qwen OpenAI-compatible endpoint:
http://127.0.0.1:8000/v1

Production source run used during Boogu integration:
/mnt/workspace/litengjie/data/r2v_v3_runs/production500-s0-20260804-030702-d5f42ae5

Boogu smoke run:
/mnt/workspace/litengjie/data/r2v_v3_runs/boogu-smoke-20260805-212506
```

The SAM3 checkpoint is not the Python package. The directory containing `sam3/model_builder.py` must also be present in `PYTHONPATH`. Do not assume the checkpoint directory provides importable Python code.

## 2. Start a new interactive shell

Do not use `set -e`, `set -u`, or `set -o pipefail` in an interactive terminal. A failed `test`, `grep`, or pipeline can close the shell or terminal session.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/activate

set +e
set +u
set +o pipefail

export R2V_REPO_ROOT="/mnt/workspace/litengjie/data/R2V_DATA_V2"
export BOOGU_CODE_ROOT="/mnt/workspace/litengjie/data/vendor/Boogu-Image"
export BOOGU_PYTHON="/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python"
export BOOGU_MODEL_PATH="/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708"
export SAM3_MODEL_PATH="/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt"
export QWEN_MODEL_PATH="/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

## 3. Resolve and export the SAM3 source path

First try an already configured value. Otherwise locate `sam3/model_builder.py` and export its package parent.

```bash
if [ -z "${SAM3_CODE_ROOT:-}" ]; then
  SAM3_MODEL_BUILDER="$(
    find \
      /mnt/workspace/litengjie/data \
      /mnt/workspace/public \
      -maxdepth 9 \
      -type f \
      -path '*/sam3/model_builder.py' \
      -print \
      2>/dev/null \
    | head -n 1
  )"

  if [ -n "$SAM3_MODEL_BUILDER" ]; then
    export SAM3_CODE_ROOT="$(dirname "$(dirname "$SAM3_MODEL_BUILDER")")"
  fi
fi

if [ -z "${SAM3_CODE_ROOT:-}" ] || [ ! -f "$SAM3_CODE_ROOT/sam3/model_builder.py" ]; then
  echo "ERROR: SAM3 source checkout is not configured"
else
  export PYTHONPATH="$SAM3_CODE_ROOT:$R2V_REPO_ROOT:${PYTHONPATH:-}"
  echo "SAM3_CODE_ROOT=$SAM3_CODE_ROOT"
fi
```

After the actual server path is confirmed, replace this discovery step with the fixed path in `server_env.sh` and update this document.

## 4. Required preflight checks

Run these before every smoke, pilot, or production run.

```bash
printf 'repo=%s\n' "$R2V_REPO_ROOT"
printf 'python=%s\n' "$(which python)"
printf 'head=%s\n' "$(git rev-parse HEAD)"
printf 'branch=%s\n' "$(git branch --show-current)"
printf 'sam3_code_root=%s\n' "${SAM3_CODE_ROOT:-<unset>}"

python - <<'PY'
import importlib.util
import sys

print("python:", sys.executable)
print("sam3:", importlib.util.find_spec("sam3"))
print("r2v_data_v2:", importlib.util.find_spec("r2v_data_v2"))
PY

python - <<'PY'
from sam3.model_builder import build_sam3_video_predictor
print("SAM3 import OK:", build_sam3_video_predictor)
PY

test -x "$BOOGU_PYTHON" || echo "ERROR: missing Boogu Python"
test -d "$BOOGU_CODE_ROOT" || echo "ERROR: missing Boogu source"
test -d "$BOOGU_MODEL_PATH" || echo "ERROR: missing Boogu model"
test -f "$SAM3_MODEL_PATH" || echo "ERROR: missing SAM3 checkpoint"

curl -fsS --noproxy '*' \
  "$QWEN_BASE_URL/models" \
  -o /tmp/qwen-models.json \
  && python - <<'PY'
import json
print([item.get("id") for item in json.load(open("/tmp/qwen-models.json", encoding="utf-8")).get("data", [])])
PY

ps -ef \
  | grep -E 'run_pipeline_v3.py|run_v3_boogu_reference_edit_worker' \
  | grep -v grep \
  || true
```

A run must not start if `from sam3.model_builder import ...` fails. SAM failure is fail-closed and can otherwise turn all Qwen-accepted candidates into `keep_source` fallbacks.

## 5. GPU rules

Do not set a global `CUDA_VISIBLE_DEVICES` before launching `run_pipeline_v3.py`.

Boogu uses the physical GPU configured in `reference_edit.cuda_visible_devices`. During the integration runs this was physical GPU `6`. Inside the worker that device is exposed as `cuda:0`.

The Boogu worker must use:

```text
align_res=False
device=cuda:0
rewriter_device=cuda:0
enable_inner_devices_manager=False
unload_rewriter_level=keep
```

The main process may use separate device settings for SAM3 and the object-removal stage. Global GPU remapping can break those assignments.

## 6. Qwen service rules

The local Qwen service normally uses:

```text
base_url: http://127.0.0.1:8000/v1
api_key: EMPTY
model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
temperature: 0
max_tokens: 1024
timeout_seconds: 3600
```

Always bypass proxies for localhost:

```bash
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

## 7. Recover shell variables after reconnecting

Shell variables do not survive terminal reconnects. Never assume `PILOT_ROOT`, `PILOT_CONFIG`, `STAMP`, or `PILOT_LOG` is still set.

Recover the most recent pilot explicitly:

```bash
export PILOT_ROOT="$(
python - <<'PY'
from pathlib import Path

root = Path('/mnt/workspace/litengjie/data/r2v_v3_runs')
manifests = list(root.glob('*/pilot_selection.json'))
if not manifests:
    raise SystemExit('No pilot_selection.json found')
latest = max(manifests, key=lambda path: path.stat().st_mtime)
print(latest.parent)
PY
)"

export PILOT_CONFIG="$(
python - "$PILOT_ROOT/pilot_selection.json" <<'PY'
from pathlib import Path
import json
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(manifest['config_path'])
PY
)"

printf 'PILOT_ROOT=%s\n' "$PILOT_ROOT"
printf 'PILOT_CONFIG=%s\n' "$PILOT_CONFIG"

if [ ! -d "$PILOT_ROOT" ]; then
  echo "ERROR: PILOT_ROOT does not exist"
fi
if [ ! -f "$PILOT_CONFIG" ]; then
  echo "ERROR: PILOT_CONFIG does not exist"
fi
```

An empty `--config "$PILOT_CONFIG"` resolves to the current directory and produces `IsADirectoryError`. Print and validate every path variable before starting a run.

## 8. Run identity rules

`run.json` identity is strict. It binds:

- `run_id`
- `git_commit`
- `config_hash`
- `model_identifiers`
- `source_manifest_path`

Do not reuse a run root with a different config. Prefer a new run root for a new pilot or production run.

Only update `run.json.git_commit` for a controlled overwrite test when all other identity fields exactly match. Keep a backup of the old `run.json` before changing it.

## 9. Launch reference-edit

```bash
STAMP="$(date +%Y%m%d-%H%M%S)"
export PILOT_LOG="/mnt/workspace/litengjie/data/logs/$(basename "$PILOT_ROOT")-$STAMP.log"

NO_PROXY="127.0.0.1,localhost" \
no_proxy="127.0.0.1,localhost" \
PYTHONPATH="$SAM3_CODE_ROOT:$R2V_REPO_ROOT:${PYTHONPATH:-}" \
python run_pipeline_v3.py \
  --config "$PILOT_CONFIG" \
  --stages reference_edit \
  --overwrite \
  2>&1 | tee "$PILOT_LOG"
```

Do not start until all of these are true:

```text
PILOT_ROOT is a directory
PILOT_CONFIG is a file
SAM3 imports successfully
Qwen /v1/models responds
No stale pipeline or worker process remains
Current git commit matches run.json, or a new run root is used
```

## 10. Known failure signatures

### `ModuleNotFoundError: No module named 'sam3'`

Cause: SAM3 source checkout is missing from `PYTHONPATH`. The checkpoint path alone is insufficient.

### `IsADirectoryError: .../R2V_DATA_V2`

Cause: the config variable is empty, so `Path("")` resolves to the current repository directory.

### Terminal closes immediately

Cause: `set -e`, `set -u`, or `pipefail` was enabled in an interactive shell and a command returned non-zero.

### `existing run.json does not match the requested V3 run`

Cause: commit, config hash, model identifiers, source manifest, or run ID differs. Do not bypass this without checking every identity field.

### Boogu output size mismatch

The worker must use `align_res=False` and explicit 16-pixel-aligned output dimensions.

### `cuda:0 and cpu`

The instruction rewriter must be explicitly placed on the same device through `pipeline.devices_manager(...)`, and every call must pass matching `device` and `rewriter_device`.

## 11. Maintenance rule

Before giving or executing server commands:

1. Read this file.
2. Print all path variables.
3. Run the import and service preflight checks.
4. Use a new run root unless an overwrite test is explicitly intended.
5. Update this document in the same commit whenever an environment path or startup rule changes.
