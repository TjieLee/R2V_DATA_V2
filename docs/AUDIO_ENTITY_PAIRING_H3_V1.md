# Audio Entity Pairing and H3 Export V1

## Scope

This track is a producer-only extension beside the frozen Visual V3 pipeline. It
does not add a stage to `run_pipeline_v3.py`, modify `ClipRecord`, change Visual
reference selection, or implement a model reader, training loss, or inference
runtime. Visual runs and Visual exports are immutable inputs. Audio bindings,
pair reports, and H3 samples are published to separate roots.

All face, speaker, transcript, and media backends are injectable. The base
installation supports precomputed evidence and local FFmpeg media extraction;
server models may be connected through explicit subprocess adapters. No adapter
downloads weights. The numeric thresholds in V1 are scaffold defaults and have
not been calibrated for production.

## Data Flow

### Phase A: clip-local binding

1. Read a completed Visual V3 export and the matching read-only Audio Binding
   V1 sidecar.
2. Materialize one full-audio target from the explicitly selected source audio
   stream. Preserve stream index, codec, sample rate, channels, duration, and
   time base as provenance.
3. Coalesce adjacent bound 25 FPS frame bindings when their entity ID and
   face-track ID match and their gap is within policy. Overlap, offscreen,
   ambiguous, no-speech, and identity changes are hard turn boundaries. Voice
   eligibility is recomputed after merge from the full turn duration plus every
   frame's audio-quality and synchronization evidence; a 40 ms frame is not
   required to be independently voice-reference eligible.
4. Select at most one primary voice-reference turn per bound subject using
   confidence descending, duration descending, start time ascending, and turn
   ID ascending.
5. Copy exactly the final published Visual reference for an occurrence. The
   bytes, synthetic flag, source frame, donor provenance, token, scope, and
   visible region are preserved. No sampled frame is searched or decoded.
6. Produce at most one canonical face crop/embedding and one speaker embedding
   per occurrence. Embeddings are L2-normalized `float32` `.npy` assets; JSONL
   stores model ID, checkpoint hash when supplied, dimension, path, and asset
   hash.

The canonical result is `r2v.audio.clip_binding.1`. Frame-level evidence remains
available under `diagnostics/`; `SpeechTurn` is the machine-readable source of
truth. A missing transcript remains `text=null` and never creates invented
dialogue.

### Phase B: corpus-level pairing

Face and voice candidate retrieval use independent top-K searches. The default
fallback is blockwise NumPy and never materializes an `N x N` similarity matrix;
FAISS is an optional adapter. All vectors are normalized, and score ties use the
stable occurrence order.

An accepted same-person edge requires the configured face threshold, mutual
face top-K, face margin, and no identity-text conflict. Missing or generic text
uses the stricter face-only threshold. High text similarity cannot rescue a low
face score. An accepted same-voice edge independently requires its voice
threshold, mutual voice top-K, and voice margin. Every threshold and renderer
option is included in the producer fingerprint with
`thresholds_calibrated=false`.

The pair builder creates pairwise edges only. It never forms a transitive global
person or voice cluster. In particular, `A~B` and `B~C` do not imply `A~C`.

## In-pair and Cross-pair Contracts

An in-pair sample uses the target clip's full audio, immutable picture, bound
speech turns, and that same occurrence's primary voice reference. A clip emits
at most one in-pair sample.

A strict cross-pair keeps the target clip's video, full audio, picture, speech
turns, and transcript. Only the voice-reference asset may come from another
occurrence. Both same-person and same-voice edges must be accepted. Same-clip
references, duplicate source video bytes, identical source spans, face-only or
voice-only matches, and candidate-grade edges are rejected.

For a multi-speaker target, every speaking subject must have face evidence and a
strict legal cross reference. Missing any subject blocks the cross-pair but does
not remove the in-pair. A bounded deterministic search chooses the complete
one-to-one donor assignment with maximum combined score. V1 allows zero or one
cross variant; it does not enumerate a Cartesian product.

## Draft H3 Rendering

`r2v.audio.pair_sample.1` records explicit subject-to-picture and
subject-to-voice bindings. The deterministic renderer begins with existing
Visual V3 text, explicitly states the `<Subject N>` / `<Picture N>` / `<Audio N>`
mapping, and appends only transcript text from bound target speech turns.
Its default speech delimiters are configurable `<d>` and `</d>`. Tags must be
paired and may not enclose empty invented dialogue.

The rendered text is explicitly marked:

```text
annotation_status = draft
is_final_annotation = false
```

It also stores renderer profile/version and an input SHA-256. It is an
intermediate H3 annotation, not a claim of final caption quality, and it does
not call Gemini or any other caption model.

`r2v.h3.sample.1` exports contiguous `picture_1..N`, `subject_1..N`, and
`audio_1..N` IDs. `audio_1` is always target full audio; following audio assets
are voice references in subject order. Every H3 media path is export-relative.

## Artifact Trees

Canonical Audio output:

```text
<audio_root>/
  dataset.json
  clip_bindings.jsonl
  pair_samples.jsonl
  pair_report.json
  failures.jsonl
  full_audio/<clip_uid>.<flac|wav>
  visual_references/<clip_uid>/<entity_id>.png
  voice_refs/<clip_uid>/<entity_id>/voice_ref_1.<flac|wav>
  face_crops/<clip_uid>/<entity_id>.png
  embeddings/face/<clip_uid>/<entity_id>.npy
  embeddings/voice/<clip_uid>/<entity_id>.npy
  embeddings/text/<clip_uid>/<entity_id>.npy
  diagnostics/<clip_uid>/frame_bindings.json
```

H3 output:

```text
<h3_root>/
  dataset.json
  samples.jsonl
  pair_report.json
  videos/<sample-key>.<source-extension>
  pictures/<sample-key>/picture_N.png
  audio/<sample-key>/audio_N.<flac|wav>
```

Publication uses a temporary sibling directory and an atomic rename. Overwrite
must be explicit. Existing outputs are restored if publication fails. Source
roots and output roots may not contain one another. Asset paths are traversal
checked, copied assets are SHA-256 verified, and the H3 tree must exactly match
the paths declared in `samples.jsonl`.

## CLI

The existing H3 pilot tools remain unchanged. The new producer entry point is:

```bash
python tools/audio_data.py bind \
  --run-root /path/to/read-only-v3-run \
  --visual-export-root /path/to/read-only-v3-export \
  --sidecar-root /path/to/read-only-audio-sidecar \
  --precomputed-json /path/to/backend-inputs.json \
  --output-root /path/to/audio-bindings \
  --limit 5 --dry-run

python tools/audio_data.py pair \
  --audio-root /path/to/audio-bindings \
  --output-root /path/to/audio-pairs \
  --top-k 20 \
  --face-threshold 0.70 --face-margin 0.04 \
  --voice-threshold 0.75 --voice-margin 0.04 \
  --max-cross-pair-variants 1

python tools/audio_data.py export-h3 \
  --audio-root /path/to/audio-pairs \
  --output-root /path/to/h3-dataset

python tools/audio_data.py inspect --audio-root /path/to/audio-pairs
```

`bind` supports a repeated `--clip-id`, `--limit`, speech merge gap, minimum
voice duration, dry run, and explicit overwrite. Its precomputed manifest maps
clip/occurrence IDs to full audio, voice audio, face/speaker/text `.npy`
fixtures, optional transcript segments, and exact audio-stream provenance.
`pair --report-only` performs no publication.

The LR-ASD pilot publishes each successful sidecar both in its diagnostic
`review/<clip_uid>/audio_binding.json` bundle and at canonical
`clips/<clip_uid>/audio_binding.json`, with a root `audio_bindings.jsonl`. The
canonical tree can therefore be passed directly to `audio_data.py bind`.
Non-ready sidecars and binding failures remain per-clip records, while
`pair_report.json` accounts for selected, ready, ineligible, and failed clips.
These binding counts are preserved through pair publication and H3 export.

## Server Adapter Boundary

Future server work should supply verified ArcFace-compatible face detection and
embedding, speaker embedding, and optional ASR through the backend protocols or
the bounded subprocess JSON contracts. The subprocess response must echo its
request ID; embeddings must declare `float32`, dimension, and
`normalized=true`; stdout/stderr are diagnostic files; nonzero exit, timeout,
invalid JSON, or metadata mismatch fails that clip closed. Phase A isolates
clip failures and records them without modifying the Visual run.

Real model identifiers, checkpoint paths, GPU allocation, threshold quality,
and ASR quality remain unverified and are intentionally not specified here.
