# H3 Omni Semantic Augmentation

This milestone adds read-only semantic augmentation after the frozen
`r2v.h3.production_pairs.2` pair stage. It does not render final H3 samples and
does not modify Visual V3, Audio binding, primary voice, embeddings, PairPolicy,
or pair artifacts.

## Source Of Truth

The producer reads only:

```text
$AUDIO_RUN_ROOT/production/pairs/in_pairs.jsonl
```

It creates exactly one semantic job per unique `target_clip_uid`. Cross-pairs
reuse the target in-pair semantics and never cause another Omni request. Donor
video, face, voice, and identity evidence are not model inputs.

For each target, `target_audio_binding_path` resolves the canonical Audio
sidecar. The existing deterministic binding coalescer reconstructs speech turns
from the frozen frame-level bindings; only bound turns are supplied to Omni.
Their turn IDs, entity IDs, occurrence IDs, and timestamps are authoritative.
Model output that adds, removes, reorders, or changes those fields is invalid.

## Qwen Omni Runtime

Configure secrets and endpoint selection only through the environment or CLI:

```bash
export DASHSCOPE_API_KEY='<server-secret>'
export QWEN_OMNI_BASE_URL='<OpenAI-compatible-endpoint>/v1'
export QWEN_OMNI_MODEL='qwen3.5-omni-plus'
```

`QWEN_OMNI_MODEL` defaults to `qwen3.5-omni-plus`. The optional
`QWEN_OMNI_MAX_BASE64_BYTES` defaults to `10000000`, matching the documented
requirement that the encoded Base64 video string be smaller than 10 MB. The
preflight uses the exact Base64 expansion before reading or encoding the video.
An oversized clip becomes a deterministic failed semantic record; the source is
never resized, transcoded, or changed.

Requests send the original target video with its audio as a Base64 data URL.
The OpenAI-compatible call is always streaming and requests `modalities=["text"]`
only. Streamed text chunks are concatenated before strict Pydantic validation.
Malformed output receives exactly one JSON repair request, then fails closed.
Raw model text is retained under the semantic output's `raw/` directory.

## Dry-Run Inventory

Dry-run requires no API key and makes no Omni request:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

python tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

The output reports the source target count, selected target count, forced
multi-subject count, inventory fingerprint, fixed destination, and confirms
that no parent quota or donor media is used.

## Fixed Pilot20

The fixed pilot includes every multi-subject target, then fills to 20 by sorted
target clip UID. It always writes or reuses:

```text
$AUDIO_RUN_ROOT/semantic_pilot20/
  inventory.json
  records.jsonl
  summary.json
  raw/
  media/
  review.html
```

Run it with:

```bash
python tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

To review with source video and audio controls:

```bash
cd "$AUDIO_RUN_ROOT/semantic_pilot20"
python -m http.server 8766 --bind 127.0.0.1
```

Open `http://127.0.0.1:8766/review.html` through the server's SSH port forward.
The `CORRECT`, `WRONG`, and `UNCERTAIN` labels are stored in browser local
storage and are QA only; they are never identity truth.

## Formal Semantic Production

Formal production has no `limit`, parent quota, or calibration sampler. It
covers every target in `in_pairs.jsonl` and writes the fixed directory:

```bash
python tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production
```

```text
$AUDIO_RUN_ROOT/production/semantic/
  inventory.json
  records.jsonl
  summary.json
  raw/
  media/
  review.html
```

Without `--overwrite`, a complete output is reused only when its inventory
fingerprint and model identifier match. A semantic failure produces a failed
record with null transcript text and no fabricated summary; it never removes or
rewrites a valid in-pair or cross-pair. Final H3 rendering remains a separate,
future milestone.
