# R2V V3 Server Environment Runbook

Last updated: 2026-08-06

This file is the single source of truth for the Linux server environment used by the V3 pipeline. Update it whenever a virtual environment, source checkout, model path, service endpoint, GPU assignment, startup command, preflight check, or run-root rule changes. Do not reconstruct these values from chat history.

## 1. Confirmed fixed paths

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

SAM3 source checkout used by V3:
/mnt/workspace/litengjie/data/vendor/sam3

Alternate SAM3 checkout not used by default:
/mnt/workspace/litengjie/data/third_party/sam3

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

The SAM3 checkpoint is not the Python package. `/mnt/workspace/litengjie/data/vendor/sam3` must be present in `PYTHONPATH` so that `sam3/model_builder.py` can be imported. Do not run a broad `find` over `/mnt/workspace/public`; the source path is fixed above.

## 2. Start a new interactive shell

Do not enable `set -e`, `set -u`, or `set -o pipefail` in an interactive terminal. A failed `test`, `grep`, or pipeline can close the shell.

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
export SAM3_CODE_ROOT="/mnt/workspace/litengjie/data/vendor/sam3"
export SAM3_MODEL_PATH="/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt"
export QWEN_MODEL_PATH="/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export PYTHONPATH="$SAM3_CODE_ROOT:$R2V_REPO_ROOT:${PYTHONPATH:-}"
```

A machine-local `server_env.sh` may contain the same exports. Before sourcing it, always validate it:

```bash
bash -n /mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh
source /mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh
```

When editing shell files interactively, prefer Python or a text editor. Do not paste nested heredocs unless the terminating token is alone on its own line.

## 3. Required preflight checks

Run these before every smoke, pilot, or production run.

```bash
printf 'repo=%s\n' "$R2V_REPO_ROOT"
printf 'python=%s\n' "$(which python)"
printf 'head=%s\n' "$(git rev-parse HEAD)"
printf 'branch=%s\n' "$(git branch --show-current)"
printf 'sam3_code_root=%s\n' "$SAM3_CODE_ROOT"
printf 'pythonpath=%s\n' "$PYTHONPATH"

python - <<'PY'
from pathlib import Path
import sam3
from sam3.model_builder import build_sam3_video_predictor

package = Path(sam3.__file__).resolve()
expected = Path('/mnt/workspace/litengjie/data/vendor/sam3').resolve()
print('sam3 package:', package)
if expected not in package.parents:
    raise SystemExit(f'Unexpected SAM3 package: {package}')
print('SAM3 import OK:', build_sam3_video_predictor)
PY

test -x "$BOOGU_PYTHON" || echo "ERROR: missing Boogu Python"
test -d "$BOOGU_CODE_ROOT" || echo "ERROR: missing Boogu source"
test -d "$BOOGU_MODEL_PATH" || echo "ERROR: missing Boogu model"
test -f "$SAM3_MODEL_PATH" || echo "ERROR: missing SAM3 checkpoint"

curl -fsS --noproxy '*' "$QWEN_BASE_URL/models" -o /tmp/qwen-models.json
python - <<'PY'
import json
print([item.get('id') for item in json.load(open('/tmp/qwen-models.json', encoding='utf-8')).get('data', [])])
PY

ps -ef \
  | grep -E 'run_pipeline_v3.py|run_v3_boogu_reference_edit_worker' \
  | grep -v grep \
  || true
```

Do not start a run if the SAM3 import or Qwen service check fails. SAM review is fail-closed; a missing SAM3 package can convert every otherwise acceptable candidate into a `keep_source` fallback.

## 4. GPU rules

Do not set global `CUDA_VISIBLE_DEVICES` before launching `run_pipeline_v3.py`.

Boogu uses the physical GPU configured by `reference_edit.cuda_visible_devices`. During integration this was physical GPU `6`, exposed inside the worker as `cuda:0`.

The Boogu worker must use:

```text
align_res=False
device=cuda:0
rewriter_device=cuda:0
enable_inner_devices_manager=False
unload_rewriter_level=keep
```

The main process may use separate device settings for SAM3 and object removal. Global remapping can break those assignments.

## 5. Qwen service rules

```text
base_url: http://127.0.0.1:8000/v1
api_key: EMPTY
model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
temperature: 0
max_tokens: 1024
timeout_seconds: 3600
```

Always bypass proxies for localhost.

## 6. Recover pilot variables after reconnecting

Shell variables do not survive reconnects. Never assume `PILOT_ROOT`, `PILOT_CONFIG`, `STAMP`, or `PILOT_LOG` is still set.

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

[ -d "$PILOT_ROOT" ] || echo "ERROR: PILOT_ROOT does not exist"
[ -f "$PILOT_CONFIG" ] || echo "ERROR: PILOT_CONFIG does not exist"
```

An empty `--config "$PILOT_CONFIG"` resolves to the current directory and causes `IsADirectoryError`.

## 7. Run identity rules

`run.json` identity binds:

- `run_id`
- `git_commit`
- `config_hash`
- `model_identifiers`
- `source_manifest_path`

Do not reuse a run root with a different config. Prefer a new run root. Only update `run.json.git_commit` for a controlled overwrite test when every other identity field matches, and keep a backup first.

## 8. Launch reference-edit

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

Do not start until the pilot paths are valid, SAM3 imports, Qwen responds, no stale worker remains, and the current commit matches `run.json` or a new run root is used.

## 9. Known failure signatures

### `ModuleNotFoundError: No module named 'sam3'`

`/mnt/workspace/litengjie/data/vendor/sam3` is missing from `PYTHONPATH`. The checkpoint alone is insufficient.

### `IsADirectoryError: .../R2V_DATA_V2`

The config variable is empty, so `Path("")` resolves to the repository directory.

### Terminal closes immediately

`set -e`, `set -u`, or `pipefail` was enabled in an interactive shell and a command returned non-zero.

### `existing run.json does not match the requested V3 run`

Commit, config hash, model identifiers, source manifest, or run ID differs. Check every identity field before changing anything.

### Boogu output size mismatch

The worker must use `align_res=False` and explicit 16-pixel-aligned dimensions.

### `cuda:0 and cpu`

Use `pipeline.devices_manager(...)`; pass matching `device` and `rewriter_device` on every call.

### `generated local reference must preserve local completeness`

Legacy local references with missing quality/completeness must normalize accepted synthetic local output to `image_quality=acceptable` and `completeness=local_usable` without promoting scope.

## 10. Maintenance rule

Before giving or executing server commands:

1. Read this file.
2. Print and validate all path variables.
3. Run SAM3, Qwen, model-path, and stale-process preflight checks.
4. Use a new run root unless an overwrite test is explicitly intended.
5. Update this document in the same change whenever an environment path or startup rule changes.
6. Never infer a server path solely from previous chat context.
