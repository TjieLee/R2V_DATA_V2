# H3 Qwen3-Omni Serving Runbook

This runbook records the current specialized-audio deployment used by
`tools/run_h3_specialized_audio_semantics.py` on the JEA production node.
The specialized pipeline now uses **two Qwen3-Omni services at the same time**:

- Qwen3-Omni-30B-A3B-Instruct, thinker-only, for Local speaker/event semantics.
- Qwen3-Omni-30B-A3B-Captioner for canonical whole-audio captioning and Global
  recaption rescue.

The existing Qwen3-VL-32B-Instruct service remains separate on GPU 0-3.
Do not use broad `pkill vllm` commands on this shared node.

## Current validated pilot topology

The current 35-clip specialized-audio pilot has been run with this topology:

```text
physical GPU 0-3  Qwen3-VL-32B-Instruct     http://127.0.0.1:8000/v1
physical GPU 4-5  Qwen3-Omni-Instruct TP2  http://127.0.0.1:8091/v1
physical GPU 6-7  Qwen3-Omni-Captioner TP2 http://127.0.0.1:8092/v1
```

This split is validated operationally for the current pilot. It is not a claim
that TP2 is the final large-scale production topology. Historical Instruct TP4
on physical GPU 4-7 remains available as a fallback evaluation topology.

## Active paths

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export R2V_PYTHON="$REPO/.venv/bin/python"

export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/e2e200-random-seed20260821-20260821-143232

export QWEN3_OMNI_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen3-omni-venv
export QWEN3_OMNI_LOG_DIR=/mnt/workspace/litengjie/data/audio_deps/logs

# Instruct checkpoint and thinker-only serving view.
export QWEN3_OMNI_INSTRUCT_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct
export QWEN3_OMNI_THINKER_VIEW=/mnt/workspace/litengjie/data/audio_deps/Qwen3-Omni-30B-A3B-Instruct-thinker-only
export QWEN3_OMNI_INSTRUCT_TP2_CONFIG=/mnt/workspace/litengjie/data/audio_deps/qwen3-omni-instruct-thinker-tp2.yaml

# Historical TP4 fallback config.
export QWEN3_OMNI_TP4_CONFIG=/mnt/workspace/litengjie/data/audio_deps/qwen3-omni-thinker-tp4.yaml

# Captioner checkpoint.
export QWEN3_OMNI_CAPTIONER_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Captioner

# Existing Qwen3-VL checkpoint.
export QWEN3_VL_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

mkdir -p "$QWEN3_OMNI_LOG_DIR"
```

Keep this environment separate from the old Dots3 vLLM environment.

## Instruct thinker-only serving view

The Instruct producer needs multimodal understanding plus text output only. The
shared public checkpoint must not be edited. The serving view at
`$QWEN3_OMNI_THINKER_VIEW` is expected to contain symlinks to the public model
artifacts and a local `config.json` with:

```json
"enable_audio_output": false
```

That causes the installed vLLM-Omni resolver to use the thinker-only Stage 0
pipeline rather than Talker/Code2Wav stages.

## Instruct TP2 deployment file

The current TP2 Instruct service uses physical GPU 4-5. Under
`CUDA_VISIBLE_DEVICES=4,5`, they become local devices `0,1`.

Create or verify the deploy file:

```bash
cat > "$QWEN3_OMNI_INSTRUCT_TP2_CONFIG" <<'YAML'
distributed_executor_backend: mp
enable_prefix_caching: false
async_chunk: false

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.90
    max_num_seqs: 8
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

cat "$QWEN3_OMNI_INSTRUCT_TP2_CONFIG"
```

Optional resolver preflight:

```bash
export QWEN3_OMNI_THINKER_VIEW QWEN3_OMNI_INSTRUCT_TP2_CONFIG

"$QWEN3_OMNI_ENV/bin/python" - <<'PY'
import os
from vllm_omni.config.config_factory import StageConfigFactory

model = os.environ["QWEN3_OMNI_THINKER_VIEW"]
deploy = os.environ["QWEN3_OMNI_INSTRUCT_TP2_CONFIG"]

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

## Start Qwen3-Omni Instruct TP2 on GPU 4-5 / port 8091

Check ownership first:

```bash
ss -ltnp | grep ':8091' || echo "8091 free"
nvidia-smi -i 4,5
```

Start:

```bash
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'

nohup env CUDA_VISIBLE_DEVICES=4,5 \
  "$QWEN3_OMNI_ENV/bin/vllm" serve \
  "$QWEN3_OMNI_THINKER_VIEW" \
  --omni \
  --deploy-config "$QWEN3_OMNI_INSTRUCT_TP2_CONFIG" \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 \
  --port 8091 \
  --allowed-local-media-path /mnt/workspace \
  > "$QWEN3_OMNI_LOG_DIR/qwen3-omni-instruct-tp2.log" 2>&1 &

echo $!
```

`--allowed-local-media-path /mnt/workspace` is required because Local inference
uses canonical `file://` audio URLs.

## Start Qwen3-Omni Captioner TP2 on GPU 6-7 / port 8092

The Captioner uses the public Captioner checkpoint directly. The current
validated launch uses the same Qwen3-Omni environment and legacy vLLM engine
selection for this checkpoint.

Check ownership first:

```bash
ss -ltnp | grep ':8092' || echo "8092 free"
nvidia-smi -i 6,7
```

Start:

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=6,7 \
  VLLM_USE_V1=0 \
  "$QWEN3_OMNI_ENV/bin/vllm" serve \
  "$QWEN3_OMNI_CAPTIONER_MODEL_PATH" \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Captioner \
  --host 127.0.0.1 \
  --port 8092 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --allowed-local-media-path /mnt/workspace \
  > "$QWEN3_OMNI_LOG_DIR/qwen3-omni-captioner-tp2.log" 2>&1 &

echo $!
```

The producer uses Captioner sampling:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
max_tokens  = 16384
```

`top_k` is sent through the OpenAI-compatible `extra_body` transport.

## Existing Qwen3-VL service on GPU 0-3 / port 8000

The specialized producer expects the already validated Qwen3-VL service at:

```text
model:    /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
endpoint: http://127.0.0.1:8000/v1
GPUs:     physical 0-3
layout:   TP1 x DP4
```

Do not restart or disturb this service while the current production node is in
use unless explicitly required.

## Health checks for all three endpoints

```bash
echo '=== Qwen3-VL ==='
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models | python -m json.tool

echo '=== Qwen3-Omni Instruct ==='
curl --noproxy '*' -fsS http://127.0.0.1:8091/v1/models | python -m json.tool

echo '=== Qwen3-Omni Captioner ==='
curl --noproxy '*' -fsS http://127.0.0.1:8092/v1/models | python -m json.tool

nvidia-smi -i 0,1,2,3,4,5,6,7
```

Expected served names:

```text
8091 -> Qwen/Qwen3-Omni-30B-A3B-Instruct
8092 -> Qwen/Qwen3-Omni-30B-A3B-Captioner
```

## Specialized producer environment

```bash
export QWEN3_OMNI_CAPTIONER_BASE_URL='http://127.0.0.1:8092/v1'
export QWEN3_OMNI_CAPTIONER_API_KEY='EMPTY'
export QWEN3_OMNI_CAPTIONER_MODEL='Qwen/Qwen3-Omni-30B-A3B-Captioner'
export QWEN3_OMNI_CAPTIONER_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Captioner'
export QWEN3_OMNI_CAPTIONER_MEDIA_MODE=file
export QWEN3_OMNI_CAPTIONER_MEDIA_ROOT=/mnt/workspace

export QWEN3_VL_AUDIO_SEMANTICS_BASE_URL='http://127.0.0.1:8000/v1'
export QWEN3_VL_AUDIO_SEMANTICS_API_KEY='EMPTY'
export QWEN3_VL_AUDIO_SEMANTICS_MODEL='/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct'
export QWEN3_VL_AUDIO_SEMANTICS_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct'

export QWEN3_OMNI_LOCAL_BASE_URL='http://127.0.0.1:8091/v1'
export QWEN3_OMNI_LOCAL_API_KEY='EMPTY'
export QWEN3_OMNI_LOCAL_MODEL='Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_LOCAL_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_LOCAL_MEDIA_MODE=file
export QWEN3_OMNI_LOCAL_MEDIA_ROOT=/mnt/workspace

export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
```

## Which services are required by each phase

The Global phase now has a one-time **recaption rescue**. Therefore it may call
Captioner again when Global semantics are missing. Service requirements are:

```text
--phase captioner         requires 8092 Captioner
--phase global-semantics  requires 8000 Qwen3-VL + 8092 Captioner
--phase local-semantics   requires 8091 Instruct
--phase assemble          requires no model service
--phase pipeline          requires 8000 + 8091 + 8092
```

This is why the current specialized deployment keeps both Qwen3-Omni services
online simultaneously.

## Current full-pipeline command

```bash
cd "$REPO"

"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase pipeline \
  --captioner-max-inflight 1 \
  --global-vl-max-inflight 4 \
  --local-instruct-max-inflight 1
```

Use `--overwrite` only when intentionally regenerating all compatible stages.
For a Global-only policy change, prefer `--phase global-semantics --overwrite`
and then `--phase assemble --overwrite` so the stochastic primary Captioner and
Local artifacts are not needlessly regenerated.

## Safe stop / restart

Do not use `pkill vllm`. Stop only the intended service:

```bash
fuser -v 8091/tcp
fuser -v 8092/tcp

# Only when intentionally stopping that service:
fuser -k 8091/tcp
fuser -k 8092/tcp
```

Port 8000 belongs to the existing Qwen3-VL service and should normally be left
untouched.

## Known runtime behavior

Qwen3-Omni-Instruct can occasionally return whitespace-only text, Markdown JSON,
`Assistant:` prefixes, or prompt echo. The Local adapter keeps strict semantic
validation but performs deterministic envelope normalization, array ordering,
and conservative `speaker_delivery` salvage so an independent event-format
failure does not discard valid speaker semantics.

The Captioner is stochastic by design. A single caption can miss an audible fact
or make a false negative statement such as `There are no musical elements`.
Global extraction therefore has one best-effort recaption rescue when the primary
Global record fails or when `overall_audio_description` or
`non_diegetic_music` is missing. The rescue never overwrites a non-null primary
Global field; it only fills fields that were missing. Rescue raw caption,
Qwen3-VL output, diagnostics, and exact call counts are preserved in the Global
raw artifact.

## Historical Instruct TP4 fallback

The historical Instruct-only topology remains available for controlled A/B or
fallback testing:

```text
config: /mnt/workspace/litengjie/data/audio_deps/qwen3-omni-thinker-tp4.yaml
GPUs:   physical 4-7
port:   8091
```

Example launch:

```bash
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

TP4 occupies GPU 4-7 and therefore cannot coexist with the current Captioner
TP2 service on GPU 6-7. Use it only when intentionally switching deployment
modes.
