# H3 Qwen3-Omni Serving Runbook

This runbook records the validated server deployment used by the JEA target-audio-caption pipeline for `Qwen3-Omni-30B-A3B-Instruct` text-only semantic inference. Keep this runtime isolated from the validated Dots3 vLLM environment.

## Active paths

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export R2V_PYTHON="$REPO/.venv/bin/python"

export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/e2e200-random-seed20260821-20260821-143232

export QWEN3_OMNI_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen3-omni-venv
export QWEN3_OMNI_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct
export QWEN3_OMNI_THINKER_VIEW=/mnt/workspace/litengjie/data/audio_deps/Qwen3-Omni-30B-A3B-Instruct-thinker-only
export QWEN3_OMNI_TP4_CONFIG=/mnt/workspace/litengjie/data/audio_deps/qwen3-omni-thinker-tp4.yaml
```

Do not install or upgrade Qwen3-Omni dependencies inside
`/mnt/workspace/litengjie/data/audio_deps/dots3-vllm-env`. Dots3 uses a separate validated runtime and may be serving on another node.

## Why a thinker-only serving view is required

The full Instruct checkpoint advertises audio output and therefore resolves to the full Qwen3-Omni pipeline:

```text
Stage 0: Thinker
Stage 1: Talker
Stage 2: Code2Wav
```

The H3 caption producer requests `modalities=["text"]` and needs only multimodal understanding plus text output. The installed vLLM-Omni resolver selects its internal thinker-only Qwen3-Omni pipeline when the Hugging Face config has:

```json
"enable_audio_output": false
```

Do not edit the shared public checkpoint. Instead use a local serving view in which every model artifact remains a symlink to the public checkpoint except `config.json`.

## One-time thinker-only serving view

First inspect the original config:

```bash
"$QWEN3_OMNI_ENV/bin/python" - <<'PY'
import json
from pathlib import Path

p = Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct/config.json")
cfg = json.loads(p.read_text())
print("enable_audio_output =", cfg.get("enable_audio_output"))
print("model_type =", cfg.get("model_type"))
PY
```

Create the serving view once:

```bash
test ! -e "$QWEN3_OMNI_THINKER_VIEW" || {
  echo "ERROR: thinker-only serving view already exists: $QWEN3_OMNI_THINKER_VIEW"
  exit 1
}

mkdir -p "$QWEN3_OMNI_THINKER_VIEW"

for src in "$QWEN3_OMNI_MODEL_PATH"/*; do
  name=$(basename "$src")
  [ "$name" = "config.json" ] && continue
  ln -s "$src" "$QWEN3_OMNI_THINKER_VIEW/$name"
done

export QWEN3_OMNI_MODEL_PATH QWEN3_OMNI_THINKER_VIEW

"$QWEN3_OMNI_ENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["QWEN3_OMNI_MODEL_PATH"]) / "config.json"
dst = Path(os.environ["QWEN3_OMNI_THINKER_VIEW"]) / "config.json"

cfg = json.loads(src.read_text())
cfg["enable_audio_output"] = False
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("serving-view enable_audio_output:", cfg["enable_audio_output"])
print("written:", dst)
PY
```

The view does not duplicate the model weights.

## TP4 deploy configuration

The validated four-GPU evaluation topology uses physical GPUs 4-7. Under
`CUDA_VISIBLE_DEVICES=4,5,6,7`, they become process-local devices `0,1,2,3`.

Create the deployment file:

```bash
cat > "$QWEN3_OMNI_TP4_CONFIG" <<'YAML'
distributed_executor_backend: mp
enable_prefix_caching: false
async_chunk: false

stages:
  - stage_id: 0
    devices: "0,1,2,3"
    tensor_parallel_size: 4
    gpu_memory_utilization: 0.9
    max_num_seqs: 16
    max_num_batched_tokens: 32768
    trust_remote_code: true
    default_sampling_params:
      temperature: 0.0
      max_tokens: 2048

platforms:
  cuda:
    stages:
      - stage_id: 0
        compilation_config:
          custom_ops: ["+rotary_embedding"]
YAML

cat "$QWEN3_OMNI_TP4_CONFIG"
```

Do not add `pipeline: qwen3_omni_moe_thinker_only` to this YAML. The currently validated installed runtime does not expose that internal resolved pipeline as an `OMNI_PIPELINES` registry key. The serving view's `enable_audio_output=false` lets the official Qwen3-Omni resolver choose thinker-only topology instead.

## Resolver preflight before using GPUs

Always validate the local installed runtime before starting vLLM:

```bash
export QWEN3_OMNI_THINKER_VIEW QWEN3_OMNI_TP4_CONFIG

"$QWEN3_OMNI_ENV/bin/python" - <<'PY'
import os
from vllm_omni.config.config_factory import StageConfigFactory

model = os.environ["QWEN3_OMNI_THINKER_VIEW"]
deploy = os.environ["QWEN3_OMNI_TP4_CONFIG"]

hf = StageConfigFactory.get_hf_config(model=model, trust_remote_code=True)
print("HF enable_audio_output:", getattr(hf, "enable_audio_output", None))

pipe = StageConfigFactory.get_pipeline_config(
    model=model,
    trust_remote_code=True,
    deploy_config_path=deploy,
)

print("PIPELINE:", pipe.model_type)
print("STAGES:", [(s.stage_id, s.model_stage) for s in pipe.stages])

assert pipe.model_type == "qwen3_omni_moe_thinker_only"
assert [s.stage_id for s in pipe.stages] == [0]
print("THINKER-ONLY RESOLUTION OK")
PY
```

Do not start inference unless this ends with:

```text
THINKER-ONLY RESOLUTION OK
```

## Start Qwen3-Omni Thinker TP4

First make sure no stale Qwen3-Omni process owns port 8091 or GPUs 4-7:

```bash
ss -ltnp | grep ':8091' || echo "8091 free"
ps -eo pid,ppid,cmd | grep -E 'StageEngineCoreProc|Qwen3-Omni-30B-A3B' | grep -v grep
nvidia-smi -i 4,5,6,7
```

Do not use a broad `pkill vllm` on a shared node because unrelated vLLM services may be running.

Start the text-only Thinker service:

```bash
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'

CUDA_VISIBLE_DEVICES=4,5,6,7 \
"$QWEN3_OMNI_ENV/bin/vllm" serve \
  "$QWEN3_OMNI_THINKER_VIEW" \
  --omni \
  --deploy-config "$QWEN3_OMNI_TP4_CONFIG" \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8091 \
  --allowed-local-media-path /mnt/workspace
```

This command intentionally has no `--stage-id`, `--omni-master-address`, or
`--omni-master-port`: the resolved pipeline itself contains only Stage 0.

`--allowed-local-media-path /mnt/workspace` is mandatory for the current H3 producer because it sends local `file://` video and audio URLs. Omitting this flag causes HTTP 400 with `Cannot load local files without --allowed-local-media-path`.

## Health checks

Four physical GPUs should participate in the Stage 0 TP4 worker:

```bash
nvidia-smi -i 4,5,6,7
ps -eo pid,ppid,etime,cmd | grep -E 'StageEngineCoreProc|Qwen3-Omni' | grep -v grep
```

There should be no Stage 1 or Stage 2 process in the thinker-only deployment.

Verify the API and served model name:

```bash
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:8091/v1/models | python -m json.tool
```

Text-only smoke request:

```bash
curl --noproxy '*' -sS \
  http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Reply with exactly OK"}],
    "modalities": ["text"],
    "temperature": 0,
    "max_tokens": 8
  }' | python -m json.tool
```

## Producer environment

Use the public checkpoint path as provenance even though the server loads through the thinker-only symlink view:

```bash
export QWEN3_OMNI_BASE_URL='http://127.0.0.1:8091/v1'
export QWEN3_OMNI_API_KEY='EMPTY'
export QWEN3_OMNI_MODEL='Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_MEDIA_MODE=file
export QWEN3_OMNI_MEDIA_ROOT=/mnt/workspace

export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
```

By default the producer sends canonical full audio only to Qwen3-Omni and
requests text only. Add `--include-video` to send the whole target video plus
canonical full audio for an explicit A/B run. It does not send Qwen3-ASR
transcript text, entity IDs, reference images, voice references, or donor media.
The target video path and hash remain validated canonical provenance in both
modes. After the existing four-field semantic pass, the producer makes one
additional lightweight, sequential whole-audio request per clip for a high-recall
non-vocal description. It uses the same audio-only default or video-plus-audio
experiment transport, requests text only, and does not send transcript, identity,
speaker clusters, or the main semantic response. Dots3 does not run this pass.

This default is evidence-driven: the real 35-clip video-plus-audio run produced
15/35 whitespace-only generation failures, and reducing producer concurrency
from 4 to 1 did not change that count. Audio-only recovered schema-valid output
for 13 of the 15 failed clips.

## Dry-run, inference, and review

Model-free validation:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend qwen3-omni \
  --max-concurrency 4 \
  --dry-run
```

Current four-GPU evaluation run:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend qwen3-omni \
  --max-concurrency 4 \
  --overwrite
```

The command above uses the default audio-only transport. Add `--include-video`
only when intentionally reproducing the video-plus-audio experiment.

`--max-concurrency` is producer-side clip concurrency, not tensor parallelism.
The current evaluation starts at `4`; benchmark `1`, `2`, `4`, and `8` before
freezing a large-scale production value. Each clip's primary request,
structured-output repair, same-schema semantic fallback, and independent
whole-audio description pass remain sequential.

Review the atomically published output:

```bash
export QWEN_OUT="$AUDIO_PRODUCTION_ROOT/audio_caption/qwen3_omni"
cd "$QWEN_OUT"
"$R2V_PYTHON" -m http.server 8769 --bind 0.0.0.0
```

Open `/review.html` through the notebook or SSH port proxy.

Inspect summary and failures:

```bash
python -m json.tool "$QWEN_OUT/summary.json"

jq -c '
  select(.status == "failed")
  | {target_clip_uid, clip_display_path, failure}
' "$QWEN_OUT/records.jsonl"
```

## Known failure mode: whitespace-only generation

A small number of observed Qwen3-Omni calls returned a long string consisting only of whitespace/newline characters. The producer classifies `response.strip() == ""` as `qwen3_omni_vllm_empty_response`. For Qwen only, a primary `h3_target_audio_semantics_v2` whitespace-only failure invokes exactly one `h3_target_audio_semantics_v2_recheck` fallback, including when whitespace occurs on the primary structured repair after an earlier schema-invalid response. Primary and fallback use the same H3-aligned schema: `overall_soundscape`, `non_diegetic_music`, `temporal_audio_events`, and `speaker_delivery`. Partial-null valid output remains accepted, and ordinary HTTP, timeout, model, or media failures do not trigger fallback. Completion finish reason, optional token usage, and whitespace counts are retained for diagnosis; validation issues that caused a repair are also retained if the repair later fails. This is distinct from a schema-valid JSON response whose semantic fields are explicitly all null or empty.

Do not interpret an individual null semantic field in an otherwise ready record
as a transport or token-limit error. Complete all-null means null soundscape,
null non-diegetic music, no temporal events, and null delivery for every supplied
speaker. Preserve raw diagnostics for failed and repaired cases.

The independent `h3_overall_audio_description_v1` pass has its own whitespace
policy: a whitespace-only primary result or a schema-valid null gets exactly one
`h3_overall_audio_description_v1_recheck`. Schema-invalid output gets one
structured repair. Infrastructure failures do not trigger recheck, and failure of
this subpass does not invalidate already-ready main semantics.

## Dots3 endpoint used for the current A/B

Dots3 is served independently on the validated eight-GPU node. The current recorded endpoint is deployment-specific and must remain an environment variable rather than a code constant:

```bash
export DOTS3_BASE_URL='http://6.167.57.88:8000/v1'
export DOTS3_API_KEY='EMPTY'
export DOTS3_MODEL='dots3-note-prev'
export DOTS3_CHECKPOINT_ID='/mnt/workspace/public/pretrained/dots3-note-prev'
export DOTS3_MEDIA_MODE=file
export DOTS3_MEDIA_ROOT=/mnt/workspace
```

See `docs/SERVER_AUDIO_PILOT_RUNBOOK.md` for the validated Dots3 eight-H200 launch command.
