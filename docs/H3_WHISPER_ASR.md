# H3 Whisper-large-v3 ASR

This stage produces dedicated ASR outputs for the frozen Audio/entity-bound
speech turns. It is data production only and is independent of the diagnostic
dots3 semantic experiment. Whisper-large-v3 is the retained ASR baseline, but
raw pilot transcripts are not yet declared final H3 transcript ground truth.

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
r2v.h3.asr_human_qa.1
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

The static review page persists `CORRECT`, `WRONG`, and `UNCERTAIN` labels in
browser localStorage. Its **Export QA JSON** action downloads a deterministic
QA-only JSON file containing the inventory fingerprint and explicit unlabeled
count. Human labels never become identity or transcript truth.

## One-Time Server Setup

The validated baseline uses the complete Whisper-large-v3 checkpoint, not the
4-decoder-layer turbo checkpoint:

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

The inspected source config reports `_name_or_path=openai/whisper-large-v3`,
`d_model=1280`, 32 encoder layers, 32 decoder layers, 128 mel bins, and
`torch_dtype=float16`. The available
`/mnt/workspace/public/pretrained/openai/whisper-large-v3-turbo` has only four
decoder layers and was not used for this accuracy baseline.

The first server attempt failed closed for all 82 calls with
`RuntimeError: Library libcublas.so.12 is not found or cannot be loaded`. After
installing the CUDA runtime wheels above, derive their library roots from the
environment's actual `purelib`:

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

## Normal Pilot And Production Run

Restore only runtime variables in each new shell:

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
```

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

Regenerate the updated review and any missing exact-turn review WAV without
loading Whisper or touching inference artifacts:

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --regenerate-review

cd "$AUDIO_RUN_ROOT/asr_pilot20"
python -m http.server 8766 --bind 127.0.0.1
```

Formal production remains gated on ASR/segmentation policy decisions. When that
gate is explicitly cleared, the fixed production command is:

```bash
cd "$REPO"
"$ASR_ENV/bin/python" tools/run_h3_asr_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production
```

Review `review.html` for hallucinated words, translation, repeated text across
unrelated turns, language errors, missed speech, and incorrect empty output.
Review labels are QA only and never become identity truth.

## Validated Pilot Milestone

The fixed pilot selected 20 of 75 production target clips and all 82 frozen
bound turns. Its inventory fingerprint was:

```text
ead8ce8aad5dc587517c4d38e74962152fbae96721fe1f92797b832d648c6a75
```

After the CUDA runtime fix, runtime results were 81 transcribed, one uncertain,
zero failed, and 82 backend calls. All 82 turns were manually labeled:

- `CORRECT`: 59 (72.0% of all turns);
- `WRONG`: 15 (18.3%);
- `UNCERTAIN`: 8 (9.8%);
- explicitly correct among `CORRECT` plus `WRONG`: 59/74 (79.7%).

QA found Cantonese and Hong Kong-accented Mandarin cases, short-word phonetic
substitutions, near-homophone/proper-name character errors, and likely
short-turn segmentation/context limitations. Dedicated Whisper ASR is clearly
more trustworthy than dots3-generated dialogue and remains the baseline, but
these raw transcripts are not final production truth. Contextual spelling or
proper-name correction may be evaluated later only under constraints that
cannot invent dialogue.

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
ASR backend
  -> waveform to original-language transcript
unresolved-only constrained MLLM resolver
  -> entity to nullable subject
final H3 renderer
  -> subject-bound speaker plus <d>transcript</d>
```

The Whisper backend accepts only a waveform and sample rate. A future inventory
builder may therefore replace the temporary LR-ASD-derived turn baseline with
mapped DiariZen speaker segments while preserving the same inference and turn
record publication path. This ASR V1 does not change the frozen production
turns and does not implement DiariZen, overlap reconciliation, speech
enhancement, dots3 annotation, or the final structured H3 renderer.
