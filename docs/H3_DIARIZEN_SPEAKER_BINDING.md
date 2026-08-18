# H3 DiariZen-Assisted Speaker Binding Pilot V1

## Status

This is an implemented **calibration pilot** with a fake-backend-tested runtime
boundary. The real DiariZen environment, CUDA runtime, model cache, throughput,
and mapping quality are **UNVALIDATED** until the fixed server pilot is run.
Production execution is intentionally blocked by
`production_blocked_pending_diarization_binding_calibration`.

Completed and frozen inputs remain unchanged:

- Visual V3 and SAM3 entity evidence;
- LR-ASD Audio/entity binding V1;
- primary voice V1;
- face/speaker embeddings and PairPolicy V1;
- production in/cross pairs;
- Whisper-large-v3 ASR V1 pilot records.

The stage writes only to `$AUDIO_RUN_ROOT/diarization_pilot20`.

## Data Flow

```text
canonical full audio
  -> DiariZen clip-local anonymous speaker segments (overlap preserved)
  -> raw frozen LR-ASD + Visual bound intervals as sparse identity anchors
  -> sample-domain atomic overlap accounting
  -> speaker cluster -> candidate entity mapping
  -> entity propagation to every segment in a mapped cluster
  -> static human review and QA export
```

DiariZen owns speaker timeline and within-clip continuity. LR-ASD plus Visual V3
provides sparse identity anchors. A later MLLM may inspect unresolved cases only;
it does not replace deterministic binding.

The current mapping policy has no numeric support or coverage threshold:

- no usable entity support: `unbound`;
- support for exactly one entity: `candidate_mapped`;
- support for more than one entity: `ambiguous`;
- the same entity mapped to temporally overlapping clusters: `conflict`.

Visual-anchor coverage is diagnostic only. A sparse anchor can bind the whole
cluster, including offscreen or occluded segments with zero direct Visual
overlap. Overlapped anchor spans with more than one active speaker cluster are
contested and support no cluster.

## Versioned Artifacts

- `r2v.h3.diarization_inventory.1`
- `r2v.h3.diarization_segment.1`
- `r2v.h3.diarization_cluster_binding.1`
- `r2v.h3.diarization_bound_segment.1`
- `r2v.h3.diarization_clip_result.1`
- `r2v.h3.diarization_summary.1`
- `r2v.h3.diarization_human_qa.1`

`speaker_cluster_id` is always clip-local. The stage performs no cross-clip or
transitive speaker clustering. Raw segments use integer half-open source sample
ranges and retain overlapping speakers.

The pilot inventory reuses the exact ordered 20 targets in
`$AUDIO_RUN_ROOT/asr_pilot20/inventory.json`. It validates that the source is
the frozen 20-of-75 pilot and that the current `production/pairs/in_pairs.jsonl`
still has the same SHA-256. Cross-pairs never create diarization jobs.

## Commands

Inventory-only pilot dry run; this imports no DiariZen module and starts no
worker:

```bash
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run
```

Real fixed pilot after the separate runtime has been staged and validated:

```bash
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite
```

Future complete inventory proof, with no model or worker:

```bash
"$R2V_PYTHON" tools/run_h3_diarization_binding.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --dry-run
```

A non-dry-run production command fails closed pending human calibration.

## Runtime Contract

Required or supported environment variables:

```bash
export DIARIZEN_PYTHON=/mnt/workspace/litengjie/data/audio_deps/diarizen-venv/bin/python
export DIARIZEN_CODE_ROOT=/mnt/workspace/litengjie/data/audio_deps/DiariZen
export DIARIZEN_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/diarizen-model-cache
export DIARIZEN_MODEL_IDENTIFIER=BUT-FIT/diarizen-wavlm-large-s80-md-v2
export DIARIZEN_DEVICE=cuda:0
export DIARIZEN_TIMEOUT_SECONDS=900
```

`DIARIZEN_MODEL_PATH` is a dedicated, fully staged local Hugging Face cache.
The official current `DiariZenPipeline.from_pretrained` resolves both the named
DiariZen repository and `pyannote/wespeaker-voxceleb-resnet34-LM` from that
cache. Formal inference sets offline mode and cannot download missing files.
The persistent JSONL worker loads one pipeline per run and processes one request
per selected target clip. Worker diagnostics go to stderr, never protocol
stdout. A requested CUDA device never falls back to CPU.

The official pipeline reads the supplied audio path with `torchaudio`, selects
the first channel, and preserves its time coordinate. R2V passes the immutable
canonical Audio V1 PCM16 WAV directly, verifies its hash before inference, and
records `official_torchaudio_first_channel_passthrough_v1`. It performs no
denoising, enhancement, interpolation, or timestamp shift.

## Proposed One-Time Server Staging (Unvalidated)

The following is a proposed starting point based on the current official
repository, not a validated server recipe. Inspect the checked-out revision and
dependency resolver output before accepting it:

```bash
export DIARIZEN_ENV=/mnt/workspace/litengjie/data/audio_deps/diarizen-venv
export DIARIZEN_CODE_ROOT=/mnt/workspace/litengjie/data/audio_deps/DiariZen
export DIARIZEN_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/diarizen-model-cache

git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git \
  "$DIARIZEN_CODE_ROOT"
uv venv --python 3.10 --seed "$DIARIZEN_ENV"
uv pip install --python "$DIARIZEN_ENV/bin/python" \
  torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$DIARIZEN_ENV/bin/python" \
  -r "$DIARIZEN_CODE_ROOT/requirements.txt"
uv pip install --python "$DIARIZEN_ENV/bin/python" \
  -e "$DIARIZEN_CODE_ROOT"
uv pip install --python "$DIARIZEN_ENV/bin/python" \
  -e "$DIARIZEN_CODE_ROOT/pyannote-audio"

"$DIARIZEN_ENV/bin/hf" download \
  BUT-FIT/diarizen-wavlm-large-s80-md-v2 \
  --cache-dir "$DIARIZEN_MODEL_PATH"
"$DIARIZEN_ENV/bin/hf" download \
  pyannote/wespeaker-voxceleb-resnet34-LM \
  --cache-dir "$DIARIZEN_MODEL_PATH"
```

The explicit downloads are a separate, operator-approved staging action. They
must not occur during a formal pilot run.

## License And Future Boundaries

The official DiariZen source is MIT licensed. Released pretrained model weights,
including the current candidate
`BUT-FIT/diarizen-wavlm-large-s80-md-v2`, are CC BY-NC 4.0 and restricted to
non-commercial/research use. Confirm authorization before staging or using the
weights.

This commit does not feed DiariZen segments to Whisper. A future ASR V2 can use
`speaker_cluster_id` plus nullable `entity_id` without changing the frozen
waveform-to-text backend. Transcript usability and voice-reference usability
remain independent. Subject IDs, `<d>` rendering, enhancement, and unresolved
speaker MLLM handling remain future work.
