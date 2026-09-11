# Independent T2VA Shadow

The authoritative population is the ordered valid rows of JEA
`shots_f03_motion.jsonl`, not an Audio production manifest. The whole-video
`videos.jsonl` precedes shot/temporal processing; each processed shot's
`video_path` is the T2VA target. Selection reuses the Video branch's
`JeaVideoMotionAdapter` and path-based `parse_clip_identity`, without running
Visual annotation, SAM3, entity/reference selection or any pair pipeline.
Clips without Pictures/Subjects remain eligible.

Canonical Audio and finalized **resolved-speech DiariZen + Qwen3-ASR** are
downstream caches only. A selected shot may reuse them only with matching clip
ID, target path/hash and compatible timeline, followed by full resolved lineage
validation. Missing Audio remains selected with `audio_preprocessing_required`;
failed upstream speech remains a skip, never an empty-speech fallback.

```text
JEA shots -> deterministic selection -> original processed shot videos
  -> canonical Audio -> SAM music_first + AuK speech -> resolved_stems
  -> DiariZen -> Qwen3-ASR -> T2VA AV semantics -> frozen R2VA audio finalizer
```

Each normally completed eligible clip makes two MiMo requests. The first uses
the **original target MP4 and its embedded audio**, plus segment ID, acoustic
cluster, start/end, language and has_transcript, never the exact ASR words.
The second imports the frozen R2VA audio-finalize prompt/schema/validation and
uses original AV plus the exact resolved SAM music and SFX stems. AuK speech is
not sent to this finalizer. Named resolved lineage and file hashes are validated
read-only; no alternate SAM route is used. There is no repair, retry, polish or
fallback. A failed first call prevents the second; actual calls are counted.

## Core and Authority

`H3NoReferenceAVCore` .4 stores speaker assignments, the typed integrated sequence,
authoritative speech facts/target duration, and the two sound fields. The pure
renderer produces only, in order:

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

MiMo assigns descriptive, stable, contiguous Sx identities by first vocal
appearance. The same acoustic source cluster cannot split into multiple Sx;
different clusters may share one Sx. These are not production entity bindings.
Assignments cover all supplied segments; only transcribed segments receive
dialogue. MiMo returns `integrated_sequence` with discriminated `prose{text}` and
`speech{segment_id, lead_in}` parts. It owns source/presentation/delivery prose and
placement, but never dialogue words, translations, transliterations or `<d>`/Sx
serialization. There must be exactly one speech slot per transcribed segment in
chronological order; missing, duplicate, unknown or non-transcribed slots fail.

The renderer copies `[Language] EXACT_ASR_TEXT` from immutable input facts and
inserts the assignment's `(Sx)` immediately before the generated `<d>` block.
Voice-over wording remains outside dialogue. It never fills missing slots or
reassigns speakers. Final dialogue count, byte-exact payloads, order and speaker
markers are still validated. No translation inference or text-repair model is
used; speech slots have no model-owned transcript field. Copied ASR in model prose
and pipeline syntax are rejected rather than cleaned. Invalid output fails that
clip and retains the original response. Conditioning labels and Ref2VA headers
are rejected. The first shot has no timestamp; later sequential shot markers
require increasing in-range cut timestamps, matching the local official
`VIDEO_PROMPT_WRITING_GUIDE_base_en.md` T2VA conventions.

The first T2VA system prompt is a no-reference adaptation of the frozen visual
and speech instructions, not their concatenation. It retains visual detail,
observational restraint, original-AV authority and stable speaker placement.
It removes reference/retention/entity-binding instructions and final sound fields.
The separate audio call uses frozen `audio_finalize_v6` unchanged. Semantic sound
classification remains a model judgment, not a keyword-based validator.

The current contract uses T2VA prompt v4/backend .4 and core .4. Model-visible
speech facts contain only segment/cluster, times, language and `has_transcript`;
exact ASR words stay inside the job/core and are never sent in text conditioning.
The explicit ordered `dialogue_segment_ids` list controls speech slots, while
assignments still cover every supplied segment. Bare Sx tokens, as well as
parenthesized markers, are forbidden in model-owned text. Only the renderer
emits final Sx/dialogue serialization. A narrow cleanup removes redundant Sx only
from a speech lead-in when it matches that segment's validated assignment.
Conflicting markers and markers elsewhere still fail. Raw output and correction
counts are retained. Prose parts are bounded at 12,000 characters and lead-ins
at 512; substantial exact normalized sentence/block loops fail without rewriting.

This is not fingerprint-compatible with v3 or the prior model-written dialogue
string. A prior-contract run ID is rejected even with `--overwrite`: use a new
run ID; frozen prior outputs are not rewritten. Inventory .3 binds resolved media;
raw .2 separates both attempts; record/summary .2 account for up to two calls.
The three-section final format is unchanged. QA .3 shows segment/cluster, Sx and
presentation, model lead-in, authoritative ASR and rendered speech side by side,
plus original video, four Audio sources and both calls' raw responses/diagnostics.

TA2VA is intentionally not implemented. A future renderer can reuse the stored
core and add separately validated Audio contracts without another AV inference.
Existing Ref2VA/RA2VA, Audio reuse, Visual and frame-conditioned products are
unchanged.

## Run

### Select and prepare Audio without Visual

Use a NEW caller-owned Audio workspace, not an existing production root.
The CPU-only bootstrap reuses `FFmpegAudioMediaBackend.materialize_full_audio`
to extract 32 kHz stereo FLAC directly from selected shot videos. It does not
transcode an existing analysis file or run a model. It atomically publishes
`selection.json`, `case_manifest.json`, `audio/canonical_clips.jsonl`,
`audio/full_audio/` and a no-reference `diarization/inventory.json`.
Existing output roots are rejected; no production artifacts are overwritten.

```bash
export SHOT_MANIFEST=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
export JEA_CLIPS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
export JEA_SOURCE_VIDEOS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808
export T2VA_AUDIO_ROOT=/mnt/workspace/litengjie/data/t2va_audio/unseen20-v1

"$R2V_PYTHON" tools/prepare_h3_t2va_audio.py \
  --shot-manifest "$SHOT_MANIFEST" \
  --clips-root "$JEA_CLIPS_ROOT" --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT" \
  --sample-size 20 --sample-seed 20260911 \
  --output-root "$T2VA_AUDIO_ROOT" --dry-run
```

Remove `--dry-run` for CPU extraction only when ready. Alternatively select by
`--case-manifest`. Relative media paths use the explicit adapter roots; omitted
roots default to the shot manifest's directory (use the explicit roots above
for the server dataset). Invalid rows are recorded with source indices; duplicate
clip identities fail closed. No reference eligibility field is consulted.

Then use the existing named Audio commands in `H3_AUDIO_SERVER_RUNBOOK.md` with
`AUDIO_PRODUCTION_ROOT="$T2VA_AUDIO_ROOT"`,
`CASE_MANIFEST="$T2VA_AUDIO_ROOT/case_manifest.json"`, one new shadow run ID,
and `--sam-route music_first`: SAM -> AuK -> resolve -> DiariZen -> Qwen3-ASR.
Do not run the Visual/MiMo Ref2VA/finalizer steps for T2VA.
The Qwen3-ASR stem CLI needs no `--visual-production-root` for this explicit
JEA-shot lineage; legacy Visual-rooted runs still require that provenance.
Model stages remain separate explicit operator actions, not bootstrap side effects.

### T2VA from finalized speech

Use the main R2V Python without a SAM/AuK runtime overlay. The named source run
must have `resolved_stems_v1/`, `diarization/` and `asr/` finalized for a clip
to receive a MiMo call. A dry run can inspect selection before Audio exists.
First inspect a model-free dry run:

```bash
"$R2V_PYTHON" tools/run_h3_t2va_shadow.py \
  --shot-manifest "$SHOT_MANIFEST" \
  --clips-root "$JEA_CLIPS_ROOT" --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --audio-shadow-run-id "$AUDIO_SHADOW_RUN_ID" \
  --t2va-run-id t2va-random20-v4 \
  --case-manifest "$AUDIO_PRODUCTION_ROOT/case_manifest.json" \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang --model mimo-v2.5 \
  --media-root /mnt/workspace \
  --max-completion-tokens 32768 \
  --dry-run
```

Remove `--dry-run` only for an explicitly approved real run. This coding task
does not perform one. Temperature is fixed at zero, thinking disabled, video
sampling 4 fps, and OpenAI SDK retries disabled. SGLang uses strict JSON-schema
output and `use_audio_in_video=true`; Xiaomi remains available with JSON-object
output and disabled thinking. API credentials use `MIMO_API_KEY` and are never
persisted. `--media-mode http --media-base-url URL` is optional; Base64 is default.

Alternatively use `--case-manifest PATH` with ordered `clip_uids` (the existing
MiMo case manifest format is also accepted), or omit selection arguments for all
valid JEA shots. Random size and seed must be supplied together. Selection order
is persisted and independent of upstream map ordering.

Random JSONL selection uses a lazy seeded permutation of nonblank row indices,
not a parsed population. Only drawn rows are decoded/adapted; invalid rows and
duplicate clip identities consume a draw and are replaced until N valid unique
clips are selected. Exhaustion fails closed. Selection .2 records this indexed
policy; .1 remains readable. Seeds are reproducible within this policy, but do
not reproduce the previous full-population `random.sample` selection.

Both selection CLIs accept `--shot-index-root PATH`, a writable cache outside
the source directory (default: the system temporary directory's
`r2v-t2va-shot-index`). The initial streaming scan hashes the manifest and writes
8-byte offsets, without JSON parsing. Warm selection seeks directly to drawn
rows. Cache identity includes resolved path, device/inode, size, mtime and ctime;
changes rebuild the index, and changes during selection fail closed. Offset
checksums detect damaged caches. Cached manifest hashes rely on these filesystem
identity fields, so use a local filesystem that reports file changes faithfully.
No index is written beside the public source manifest. Selected rows retain
their row/clip/video provenance. Random `valid_row_count` is null because the
unvisited population has not been validated; all/case selection retains its
existing exhaustive lookup and exact valid-row count.

Inventory .3 records source shot manifest path/hash, adapter roots, source row
index/hash, source video/shot identity and selected video path/hash. The no-Visual
DiariZen/ASR inventories use .5; legacy .4 inputs remain readable and retain their
existing behavior. No TA2VA is implemented.

Outputs are isolated under:

```text
<audio-production-root>/t2va_shadow_v1/runs/<t2va-run-id>/
  inventory.json
  records.jsonl
  raw/<clip_uid>.json
  core/<clip_uid>.json
  prompts/<clip_uid>.txt
  summary.json
```

Existing output requires explicit `--overwrite` and validated T2VA ownership.
The full replacement is built in a temporary sibling directory before atomic
publication. Source artifacts are never modified. Source and configuration hashes
bind each request; raw responses, finish reasons, usage and failure reasons remain
auditable. `core/` and `prompts/` exist only for ready records.

## Static QA

Serve the common root containing both the selected JEA videos and resolved Audio,
not only the QA directory. This is a manual local review command, not an inference:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory /mnt/workspace
```

```bash
"$R2V_PYTHON" tools/build_h3_t2va_qa.py \
  --t2va-root "$AUDIO_PRODUCTION_ROOT/t2va_shadow_v1/runs/t2va-random20-v4" \
  --media-root /mnt/workspace \
  --media-base-url http://127.0.0.1:8765
```

Open the emitted `qa/review.html` through that HTTP server using its path relative
to /mnt/workspace (or locally with the HTTP media URLs). Without the two media
options, URLs remain relative filesystem paths. All exposed media must remain
under the supplied media root, including resolved symlink targets. No media is
copied, symlinked, cropped or transcoded.

Before exposing media, QA validates the named resolved inventory, canonical
target identity, resolved record fingerprints and all four Audio hashes.
Players show original/full target audio, AuK speech, SAM music and SAM SFX.
Each speech row offers original-audio and speech-stem segment playback by seeking
the existing full-timeline file and pausing at the available segment end. No
segment WAV is created. Changing clips stops segment playback.

The dedicated R2VA-style audio-finalizer section shows final soundscape/music,
with its response and diagnostics separate from T2VA semantic raw output.
Compare those judgments against original audio and the two SAM stems; compare
speaker/presentation and exact dialogue against original video and AuK speech.
The page has no external dependencies and does not alter annotations or integrate
with existing H3 QA.

Manually inspect a fresh small unseen canary for full-shot visual coverage,
speaker transitions, exact speech timing/text, soundscape/music partition and
absence of reference syntax before scaling. Local synthetic tests prove transport
shape and deterministic contracts, not model accuracy or server performance.
