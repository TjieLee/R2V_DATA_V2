# Independent T2VA Shadow

This additive product reads the JEA canonical clip inventory and the selected
named run's finalized **resolved-speech DiariZen + Qwen3-ASR** lineage. It does
not require a Visual export, reference graph, or a previous MiMo annotation.
Missing/failed upstream speech is a recorded skip, never an empty-speech fallback.

Each eligible clip makes one MiMo AV request with the **original target MP4 and
its embedded audio**, plus a compact table of segment ID, acoustic cluster,
start/end, exact ASR language and text. No additional media is attached. Generic
lineage validation hashes upstream files read-only; stems are not decoded or
provided as model evidence. There is no repair, retry, polish or fallback call.

## Core and Authority

`H3NoReferenceAVCore` stores speaker assignments and the three no-reference
semantic fields. The pure renderer produces only, in order:

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

MiMo assigns descriptive, stable, contiguous Sx identities by first vocal
appearance. The same acoustic source cluster cannot split into multiple Sx;
different clusters may share one Sx. These are not production entity bindings.
Assignments cover all supplied segments; only transcribed segments receive
dialogue. Exact language/text, chronological dialogue inventory and each dialogue's
speaker lead-in are checked without rewriting output. Invalid output fails that
clip and retains the original response. Conditioning labels and Ref2VA headers
are rejected. The first shot has no timestamp; later sequential shot markers
require increasing in-range cut timestamps, matching the local official
`VIDEO_PROMPT_WRITING_GUIDE_base_en.md` T2VA conventions.

The T2VA system prompt is a no-reference adaptation of the frozen current
`visual_only_v5`, `speech_assembly_v48`, `speaker_profile_v2` and
`audio_finalize_v6` prompts, not their concatenation. It retains visual detail,
observational restraint, original-AV authority, exact ASR, stable speaker placement
and physical-sound/score separation. It removes all staged-turn, reference,
retention and entity-binding instructions. The canonical absent-soundscape
sentence is reused; absent audience-only music is `N/A`. Semantic sound
classification remains a model judgment, not a keyword-based validator.

TA2VA is intentionally not implemented. A future renderer can reuse the stored
core and add separately validated Audio contracts without another AV inference.
Existing Ref2VA/RA2VA, Audio reuse, Visual and frame-conditioned products are
unchanged.

## Run

Use the main R2V Python without a SAM/AuK runtime overlay. The named source run
must already have `resolved_stems_v1/`, `diarization/` and `asr/` finalized.
First inspect a model-free dry run:

```bash
"$R2V_PYTHON" tools/run_h3_t2va_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --audio-shadow-run-id "$AUDIO_SHADOW_RUN_ID" \
  --t2va-run-id t2va-random20-v1 \
  --sample-size 20 --sample-seed 20260911 \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang --model mimo-v2.5 \
  --media-root "$JEA_MEDIA_ROOT" \
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
canonical clips. Random size and seed must be supplied together. Selection order
is persisted and independent of upstream map ordering.

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

```bash
"$R2V_PYTHON" tools/build_h3_t2va_qa.py \
  --t2va-root "$AUDIO_PRODUCTION_ROOT/t2va_shadow_v1/runs/t2va-random20-v1"
```

Open the emitted `qa/review.html` alongside accessible original media. Default
video URLs are relative filesystem paths; no media is copied. For an existing
HTTP media host, provide both `--media-root "$JEA_MEDIA_ROOT"` and
`--media-base-url "$JEA_MEDIA_BASE_URL"`; paths must remain under that media root.
The page has no external dependencies and displays the original video, rendered
prompt, factual speech table with assigned Sx/presentation, warnings and raw
response. It does not alter annotations or integrate with existing H3 QA.

Manually inspect a fresh small unseen canary for full-shot visual coverage,
speaker transitions, exact speech timing/text, soundscape/music partition and
absence of reference syntax before scaling. Local synthetic tests prove transport
shape and deterministic contracts, not model accuracy or server performance.
