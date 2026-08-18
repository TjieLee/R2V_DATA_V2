# H3 dots3 Semantic Augmentation

This milestone adds read-only target-clip semantics after the frozen
`r2v.h3.production_pairs.2` pair stage. The R2V producer is an OpenAI-compatible
client of a separately managed vLLM server running
`dots-studio/dots3-note-prev-fp8`. It never loads the model, starts vLLM, or
allocates model GPUs.

Status boundaries:

- Audio binding, primary voice, embeddings, and PairPolicy V1: **COMPLETE / FROZEN**.
- dots3 semantic augmentation: **PILOT / PRODUCTION WORKFLOW**.
- H3 exporter: **DEMO / NOT FINAL**.
- Visual subject attributes: **SEPARATE WORKSTREAM / NOT CONSUMED**.

## Source Of Truth

The producer reads only:

```text
$AUDIO_RUN_ROOT/production/pairs/in_pairs.jsonl
```

It creates exactly one semantic job per unique `target_clip_uid`. Cross-pairs
reuse that target record and never cause another model call. Donor video, donor
face/voice, target primary voice, embeddings, and PairPolicy evidence are never
sent to dots3.

Every request contains three content items in one user message:

1. text with frozen bound-turn metadata and the strict JSON schema;
2. `video_url` for `target_video_path`;
3. `audio_url` for `target_full_audio_path`.

The model may return only `turn_id`, transcript `status`, `text`, and `language`
for each speech turn. The producer copies `entity_id`, `entity_occurrence_id`,
`start_time`, and `end_time` from authoritative input. Unknown, missing, or
reordered turn IDs fail validation. Uncertain or inaudible speech uses null text;
there is no heuristic transcript fallback.

## dots3 vLLM Client

Runtime configuration belongs on the R2V producer node:

```bash
export DOTS3_BASE_URL='http://<H200_NODE_PRIVATE_IP>:8000/v1'
export DOTS3_API_KEY='EMPTY'
export DOTS3_MODEL='dots3-note-prev'
export DOTS3_CHECKPOINT_ID='dots-studio/dots3-note-prev-fp8'
```

The client uses ordinary non-streaming Chat Completions with
`enable_thinking=false`, video and audio URL content items, temperature zero,
and text output. The first malformed structured response receives exactly one
repair request with the malformed text and validation issues. API errors,
timeouts, empty output, or a second invalid response produce a failed semantic
record. No Qwen or other backend fallback exists. Raw model text is retained in
the semantic output's `raw/` directory for diagnostics.

Durable provenance records `backend=vllm`, the served name
`dots3-note-prev`, the checkpoint ID `dots-studio/dots3-note-prev-fp8`, the
endpoint, media transport, prompt version, and a configuration fingerprint. API
keys are never persisted.

## Media Transport

Only two root-confined transports are supported. Neither mode copies,
transcodes, resizes, or Base64-encodes media.

### HTTP mode (canonical cross-node path)

```bash
export DOTS3_MEDIA_MODE=http
export DOTS3_MEDIA_ROOT=/mnt/workspace/litengjie/data
export DOTS3_MEDIA_BASE_URL='http://<DATA_NODE_PRIVATE_IP>:8767/'

cd "$DOTS3_MEDIA_ROOT"
python -m http.server 8767 --bind 0.0.0.0
```

For example, `$DOTS3_MEDIA_ROOT/foo/bar.mp4` maps deterministically to
`$DOTS3_MEDIA_BASE_URL/foo/bar.mp4`. The media server must be restricted to the
private production network.

### File mode (shared filesystem only)

```bash
export DOTS3_MEDIA_MODE=file
export DOTS3_MEDIA_ROOT=/mnt/workspace/litengjie/data
unset DOTS3_MEDIA_BASE_URL
```

File mode is valid only when both nodes see the exact same absolute paths. Start
vLLM with the trusted root added as:

```text
--allowed-local-media-path /mnt/workspace/litengjie/data
```

Both modes resolve symlinks and reject missing files, path traversal, or any
resolved path outside `DOTS3_MEDIA_ROOT`.

## Fixed Outputs

Dry-run builds the deterministic inventory and makes zero inference calls:

```bash
python tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

The fixed pilot includes every multi-subject target, then fills to 20 by sorted
target clip UID:

```bash
python tools/run_h3_omni_semantic.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

It writes `$AUDIO_RUN_ROOT/semantic_pilot20/`. Serve its review page with:

```bash
cd "$AUDIO_RUN_ROOT/semantic_pilot20"
python -m http.server 8766 --bind 127.0.0.1
```

Formal production has no `limit`, parent quota, or calibration sampler. It
covers every unique target in `in_pairs.jsonl` (75 in the frozen production
pair set) and writes the fixed directory:

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

Without `--overwrite`, complete output is reused only when inventory,
checkpoint, and served-model identity match. Failed semantics never remove or
rewrite valid in-pairs or cross-pairs. Final H3 rendering remains a separate
future milestone.
