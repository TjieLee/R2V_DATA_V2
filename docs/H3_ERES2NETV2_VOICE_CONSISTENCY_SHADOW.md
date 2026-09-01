# H3 ERes2NetV2 Voice-Consistency Shadow

This evaluator is a read-only comparison of the existing SpeechBrain ECAPA
score and an official SpeakerLab ERes2NetV2 score against current human
`SAME` / `DIFFERENT` annotations. It does not modify DiariZen, bindings,
primary voices, H3 samples, the existing ECAPA audit, or any production
threshold.

## Inputs And Outputs

The evaluator reads:

- `diarization_voice_consistency_audit_v1/records.jsonl`
- `diarization_voice_consistency_audit_v1/human_review/annotations.jsonl`
- the hash-locked segment audio and target primary voice paths in those records

Stale annotations are excluded using the existing review fingerprint contract.
`UNCERTAIN` is counted but excluded from binary metrics and model inference.

The default output is:

```text
<audio-production-root>/diarization_voice_consistency_audit_v1/
  eres2netv2_shadow_v1/
    records.jsonl
    errors.jsonl
    comparison.csv
    summary.json
```

All reported best thresholds are diagnostic sweep results only. The summary
always publishes `production_threshold_applied=false`, `binding_modified=false`,
and `production_artifacts_modified=false`.

## One-Time Server Setup

Use an environment and source checkout separate from the R2V repository and
all existing Audio/ASR/model environments. The model download is a one-time
explicit setup action; evaluator inference itself is offline.

```bash
export ERES_ROOT=/mnt/workspace/litengjie/data/audio_deps/eres2netv2
export ERES2NETV2_ENV="$ERES_ROOT/venv"
export SPEAKERLAB_CODE_ROOT="$ERES_ROOT/3D-Speaker"

uv venv --python 3.10 "$ERES2NETV2_ENV"
git clone https://github.com/modelscope/3D-Speaker.git "$SPEAKERLAB_CODE_ROOT"
git -C "$SPEAKERLAB_CODE_ROOT" rev-parse HEAD

uv pip install --python "$ERES2NETV2_ENV/bin/python" \
  -r "$SPEAKERLAB_CODE_ROOT/requirements.txt"
```

Download the exact official ModelScope revision during setup:

```bash
"$ERES2NETV2_ENV/bin/python" - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

path = snapshot_download(
    "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    revision="v1.0.1",
)
print(path)
PY
```

Record the printed local directory and verify it contains:

```text
pretrained_eres2netv2.ckpt
```

Then set:

```bash
export ERES2NETV2_PYTHON="$ERES2NETV2_ENV/bin/python"
export ERES2NETV2_MODEL_PATH=/absolute/path/printed/by/snapshot_download
```

The worker uses SpeakerLab's official ERes2NetV2 class, 80-bin mean-normalized
FBank preprocessing, and strict checkpoint loading. It never calls ModelScope
or another network downloader at inference time.

## Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0

"$R2V_PYTHON" tools/evaluate_h3_eres2netv2_voice_consistency.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --speakerlab-code-root "$SPEAKERLAB_CODE_ROOT" \
  --eres2netv2-python "$ERES2NETV2_PYTHON" \
  --eres2netv2-model-path "$ERES2NETV2_MODEL_PATH" \
  --model-identifier iic/speech_eres2netv2_sv_zh-cn_16k-common \
  --device cuda:0
```

Use `--overwrite` only to atomically replace this shadow output. It never
overwrites the source audit or human annotations.
