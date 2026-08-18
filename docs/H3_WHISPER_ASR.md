# H3 Whisper-large-v3 ASR

This stage produces authoritative transcripts for the frozen Audio/entity-bound
speech turns. It is data production only and is independent of the diagnostic
dots3 semantic experiment.

## Frozen Input Contract

The sole inventory source is:

```text
$AUDIO_RUN_ROOT/production/pairs/in_pairs.jsonl
```

For every unique target clip, the producer loads `target_full_audio_path` and
`target_audio_binding_path`, then reuses the same canonical speech-turn
coalescing used by the semantic inventory. Cross-pairs are never read, so they
cannot cause extra ASR jobs.

Code, not Whisper, owns:

- `target_clip_uid`;
- `turn_id`;
- `entity_id` and `entity_occurrence_id`;
- authoritative `start_time` and `end_time`;
- exact source start/end sample indexes.

Each turn also carries model-independent `segment_provenance`: the boundary
source, source segment ID, optional anonymous speaker-cluster ID, and the source
of the entity binding. V1 records `frozen_audio_binding_turns_v1` boundaries
and `lr_asd_visual_entity_binding_v1` identity. These are provenance values,
not Whisper inputs or hard-coded backend policy.

Whisper receives only the exact turn waveform. It never receives target video,
donor media, primary voice references, embeddings, PairPolicy evidence, or
identity fields. It cannot add, remove, merge, split, retime, or re-identify
turns.

## ASR Contract

The production backend is faster-whisper with Whisper-large-v3. Every turn is
decoded independently with:

```text
task=transcribe
condition_on_previous_text=false
vad_filter=false
word_timestamps=false
local_files_only=true
```

`translate` is never exposed as a CLI option. Decoder timestamps are diagnostic
implementation details and never replace frozen Audio timing. A non-empty
transcript is `transcribed`; an empty successful result is `uncertain` with null
text; inference failure is `failed` with null text and an explicit reason. No
transcript fallback or confidence gate is used in V1.

The current canonical full audio is uncompressed PCM16 WAV. The producer crops
exact source samples before any inference preprocessing. It downmixes and/or
resamples only the in-memory turn waveform when needed for 16 kHz mono Whisper
input. It applies no padding, VAD resegmentation, denoising, enhancement, or
normalization and never overwrites canonical audio.

## Schemas And Outputs

Schema versions:

```text
r2v.h3.asr_turn.1
r2v.h3.asr_inventory.1
r2v.h3.asr_summary.1
```

Fixed output roots:

```text
$AUDIO_RUN_ROOT/asr_pilot20/
$AUDIO_RUN_ROOT/production/asr/
```

Each contains:

```text
inventory.json
turns.jsonl
summary.json
review.html
review_media/
```

The inventory and backend configuration are fingerprinted. Existing output is
reused only when both fingerprints match; otherwise the operator must use
`--overwrite`. Publication is atomic, and pair artifacts remain read-only.

The pilot selects the same deterministic 20 targets as the semantic pilot:
every multi-subject target first, then sorted target clip UID. It includes all
turns belonging to those targets. Production includes every target and every
authoritative turn, with no parent quota or arbitrary limit.

## Dedicated Runtime

Use the isolated environment:

```bash
export ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/asr-venv
uv venv --python 3.12 --seed "$ASR_ENV"
uv pip install --python "$ASR_ENV/bin/python" \
  faster-whisper "pydantic>=2,<3" Pillow
```

Recent faster-whisper GPU builds require a compatible CUDA 12/cuBLAS and cuDNN
9 runtime. Validate the server combination before formal production. Stage a
local CTranslate2 Whisper-large-v3 model directory; do not rely on an automatic
weight download during production.

Runtime variables:

```bash
export ASR_MODEL_PATH=/mnt/workspace/public/pretrained/<local-whisper-large-v3-ct2>
export ASR_MODEL=large-v3
# Optional for an identifier; local paths are fingerprinted automatically:
# export ASR_MODEL_FINGERPRINT=<64-character-sha256>
export ASR_DEVICE=cuda:0
export ASR_COMPUTE_TYPE=float16
```

`ASR_MODEL_PATH` takes precedence over `ASR_MODEL`. The model loader uses
`local_files_only=true` and does not silently fall back from a requested CUDA
device to CPU. When `ASR_MODEL_PATH` is set, the tool derives a deterministic
content fingerprint from the local checkpoint and verifies an explicitly set
`ASR_MODEL_FINGERPRINT` against it. For a model identifier, the optional
fingerprint is included directly in strict output-reuse checks.

## Commands

Inventory-only dry run; this does not import or load faster-whisper:

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

Fixed pilot:

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

After human QA accepts the pilot, formal production is:

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production
```

Review `review.html` for hallucinated words, translation, repeated text across
unrelated turns, language errors, missed speech, and incorrect empty output.
Review labels are QA only and never become identity truth.

## Stage Boundary

The existing dots3 `semantic_pilot20` is retained as diagnostic evidence, but
its dialogue hallucinations block dots3 transcript generation and complete
semantic production. A later stage may give trusted Whisper transcripts to
dots3 for multimodal semantic annotation; that is not part of this task.

Future diarization direction, documentation only:

```text
DiariZen
  -> audio-domain speaker segments and improved boundaries
LR-ASD + Visual
  -> visible speaker to entity ID
temporal overlap join
  -> aggregate speaker-cluster overlap with visible entity anchors
confident cluster-to-entity mapping
  -> propagate that entity identity to every segment in the cluster,
     including occluded or offscreen segments with no Visual overlap
```

The Whisper backend accepts only a waveform and sample rate. A future inventory
builder may therefore replace the temporary LR-ASD-derived turn baseline with
mapped DiariZen speaker segments while preserving the same inference and turn
record publication path. This ASR V1 does not change the frozen production
turns and does not implement DiariZen, overlap reconciliation, speech
enhancement, dots3 annotation, or the final structured H3 renderer.
