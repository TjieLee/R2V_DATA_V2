# Frozen reconcile to R(A)2VA training bundle

This CPU-only entry point consumes `source_contract.json`, `records.jsonl`, and
`summary.json` from a completed reference-only reconcile. It performs no model
calls and does not rebuild Visual, DiariZen, ASR, or reference selection.

The reconcile's parent is the named run. Its `resolved_stems_v1` inventory points
to the canonical Audio manifest, which identifies the Audio production root.
Canonical sample metadata comes from `<audio-production-root>/h3/samples.jsonl`.
Only the matching canonical sample IDs are used; in-pair/cross-pair rows are not
imported. Preparation checks frozen reference selection and the Subject graph
with the existing projection adapter, but saves the complete original canonical
`visual_references`. The H3 materializer projects those references once for the
final product. Voice bindings are cleared, and speech comes exactly from the
frozen job.

If canonical H3 metadata is absent, the command reports the missing manifest.
Supply `--source-h3-root PATH` only when the actual matching canonical manifest
exists elsewhere. Sample identity and original Visual reference metadata cannot
be invented from reconcile alone. No Visual directory search is performed.

Prepared bundles use `r2v.h3.audio_reuse_frozen_inventory.1`, since reconcile
does not contain the old full-Visual inventory provenance. Existing prepared
`MimoInventory` versions remain readable and their preparation path is unchanged.
No prompt, caption schema, materializer policy, or inference version changes.

## Server smoke, then fixed20

Use a NEW output directory for every invocation. There is no overwrite or media
copy mode. The existing unverified-stem pilot opt-in remains explicit.

```bash
SHADOW=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415/sam_audio_stem_shadow_v1/runs/random20-voicefirst-v1
C="$SHADOW/mimo_v26_ra2va_single_v10_ab_action20"
OUT="$SHADOW/ra2va_action20_training_bundle_v1"

# Pick three existing ready records: first visible, first unbound/offscreen,
# and first positive-music record. Duplicate choices are harmless.
readarray -t SMOKE_UIDS < <(.venv/bin/python - "$C" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in (Path(sys.argv[1]) / "records.jsonl").read_text().splitlines() if line.strip()]
rows = [r for r in rows if r["status"] == "ready"]
def bindings(r):
    return {g["binding_status"] for g in r["annotation"]["av_grounding"]["segment_groundings"]}
choices = [next((r for r in rows if "visible_entity" in bindings(r)), rows[0]),
           next((r for r in rows if bindings(r) & {"offscreen", "no_reliable_entity", "uncertain"}), rows[0]),
           next((r for r in rows if (r["annotation"]["h3_semantics"]["non_diegetic_music"] or "").strip().lower() not in {"", "n/a", "unknown"}), rows[-1])]
print("\n".join(dict.fromkeys(r["clip_uid"] for r in choices)))
PY
)
SMOKE_ARGS=()
for uid in "${SMOKE_UIDS[@]}"; do SMOKE_ARGS+=(--clip-uid "$uid"); done
.venv/bin/python tools/materialize_h3_ra2va_training_bundle.py \
  --reconcile-root "$C" --output-root "${OUT}-smoke" \
  --allow-unverified "${SMOKE_ARGS[@]}"

# After inspecting the smoke, materialize all ready fixed20 clips.
.venv/bin/python tools/materialize_h3_ra2va_training_bundle.py \
  --reconcile-root "$C" --output-root "$OUT" --allow-unverified
```

The output owns `audio_reuse_assets_v1`, `audio_reuse_prepared_v1`,
`h3_audio_reuse_products_v1`, `h3_frame_conditioned_products_v1`, and a summary.
Existing stage writers provide atomic publication. Frames are decoded from the
current target video; full Audio remains the original canonical FLAC. Speech
uses existing ownership exclusions and timeline placement. Music publication
uses the existing annotation policy, not mere stem existence. Unavailable
speech variants are omitted, not filled with music-only conditioning.

The bundle summary reports per-task counts, product failures, unavailable speech
variants, zero model calls, `production_artifacts_modified=false`, and
`human_review_required=false`. There is no manual QA stage, pending-review state,
or human approval prerequisite for production export. Warnings are non-blocking
diagnostics; unavailable identity-specific Audio variants are omitted automatically
while the other valid products continue. Actual annotation/ownership contradictions
retain their existing failures rather than entering a human review queue.

### Repeat the failed v1 smoke without overwriting it

Reuse exactly its three selected clips and the same frozen action20 reconcile,
but publish to a new directory. This does not rerun any model.

```bash
# SHADOW and C are the same paths defined above.
readarray -t SMOKE_UIDS < <(.venv/bin/python - "$SHADOW/ra2va_action20_training_bundle_v1-smoke/audio_reuse_prepared_v1/inventory.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    print("\n".join(job["clip_uid"] for job in json.load(handle)["jobs"]))
PY
)
SMOKE_ARGS=()
for uid in "${SMOKE_UIDS[@]}"; do SMOKE_ARGS+=(--clip-uid "$uid"); done
.venv/bin/python tools/materialize_h3_ra2va_training_bundle.py \
  --reconcile-root "$C" \
  --output-root "$SHADOW/ra2va_action20_training_bundle_v2-smoke" \
  --allow-unverified "${SMOKE_ARGS[@]}"
```

Existing H3 failures retain their actual `failure_reason`; no fallback caption
or reference graph is synthesized.

## Separate diagnosis: speaker waveform exclusions

The compact-to-legacy adapter in `mimo25_single_backend.py` deliberately emits
`vocal_composition="uncertain"` as a compatibility placeholder. Compact2 does
not supply verified single-speaker composition evidence. This means **isolated
ownership is unverified**, not that multiple speakers were positively detected.
`speaker_ownership_reasons()` nevertheless excludes it as `non_single_speaker`,
so unchanged compact outputs can systematically yield no speaker reuse assets.
This patch does not convert that placeholder into `single_speaker` or change
the frozen annotation/ownership policy.

`cross_speaker_overlap` is a separate sample-interval check: intersecting ranges
are excluded when their groups differ or either range is already unsafe. It
can therefore also occur for the same group when its composition is uncertain;
the code alone does not prove multiple speakers were heard. Cross-group or unsafe
overlaps remain blocked, as do confirmed multiple-speaker compositions. Resolving missing
ownership evidence is a separate task, not part of the reference-projection fix.

## Automatic production export

```bash
.venv/bin/python tools/export_h3_training_manifests.py \
  --ra2va-shadow-root "$OUT" --output-root "${OUT}-training"
```

The existing exporter consumes ready products directly across the 12 R(A)2VA
task categories. It does not require PASS/ISSUE/SKIP records or read a review
store, including for historical bundles whose summary said
`human_review_required=true`. Optional pilot viewers are separate tools, never
part of the production path. Export rows have exactly `video`, `images`,
`audios`, `caption`; `videos.jsonl` has only `video` and `tasks`.

## Lightweight frozen Bundle export

Frozen Bundle consumers use the already-published jobs, products, captions,
Audio references, and Subject metadata. Frame projection does not reload stems
or annotations, reconstruct H3 products, compare rendered captions, or rehash
input media. It computes each new PNG digest once for the existing metadata.
Each clip's first/last frame pair is extracted once and reused across Audio
conditioning variants. Last-frame extraction uses the previously server-tested
`-vf reverse -frames:v 1`, without resizing, autorotation, or FPS conversion.
The tail-seek/update strategy introduced in `64a467e` was withdrawn after a real
server run stalled in FFmpeg. Each first/last subprocess has a 30-second timeout;
Python stops and reaps the child on timeout, then reports the clip UID, frame
role, video path, elapsed time, and exact command. Flushed start/done logs show
each frame's progress and elapsed time. A timeout fails the export rather than
retrying, substituting a frame, or leaving an indefinitely running subprocess.

The frozen Audio asset writer reads source PCM for the actual timeline copy,
writes lossless FLAC, and records its digest once. It does not hash input media
or decode the generated FLAC again for a PCM comparison. Audio product generation
does not repeat source/asset integrity scans. Existing speaker ownership,
overlap, identity-publication restrictions, and task selection stay unchanged.
The legacy standalone Audio reuse loader keeps its existing behavior. No new
trust/skip/unsafe CLI switch or human review gate is needed.

Run all three CPU commands in fresh directories; no MiMo endpoint is contacted:

```bash
set -e
: "${SHADOW:?Set SHADOW to the existing Random20 named-run directory}"
EFFECTIVE="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_effective20_v85"
STAMP=$(date +%Y%m%d-%H%M%S)
BUNDLE="$SHADOW/ra2va_two_step_effective20_v85_bundle_light_${STAMP}"
TRAIN="$SHADOW/ra2va_two_step_effective20_v85_jsonl_light_${STAMP}"

.venv/bin/python tools/materialize_h3_ra2va_training_bundle.py \
  --reconcile-root "$EFFECTIVE" --output-root "$BUNDLE" --allow-unverified
.venv/bin/python tools/export_h3_training_manifests.py \
  --ra2va-shadow-root "$BUNDLE" --output-root "$TRAIN"
.venv/bin/python tools/export_h3_voice_donor_reserve.py --bundle-root "$BUNDLE"
```

The commands print elapsed seconds for Prepared, Audio Assets, Audio Products,
Frame Projection, JSONL Export, and Donor Reserve. The first four timings also
appear in Bundle `summary.json` under `stage_elapsed_seconds`. Compare Frame
Projection against the measured server baseline of **39.39 seconds**; local
synthetic CPU tests do not establish the optimized effective20 server time.

Inspect counts without media scans or a production validation gate:

```bash
.venv/bin/python - "$BUNDLE" "$TRAIN" <<'PY'
import json, sys
from pathlib import Path
bundle, train = map(Path, sys.argv[1:])
summary = json.loads((bundle / "summary.json").read_text())
print("Products ready/failed:", summary["product_ready_count"], summary["product_failed_count"])
print("Stage seconds:", summary["stage_elapsed_seconds"])
counts = {p.stem: sum(bool(line.strip()) for line in p.read_text().splitlines())
          for p in sorted(train.glob("*.jsonl")) if p.stem.startswith(("r2va_", "ra2va_"))}
print("Task counts:", counts, "total:", sum(counts.values()))
donors = json.loads((bundle / "voice_donor_reserve.json").read_text())["donors"]
print("Donors/segments:", len(donors), sum(d["segment_count"] for d in donors))
for line in (bundle / "h3_audio_reuse_products_v1/records.jsonl").read_text().splitlines():
    product = json.loads(line)
    if product["status"] == "failed":
        print("Failure:", product["clip_uid"], product["conditioning_variant"], product["failure_reason"])
PY
```

Expected from the existing effective20: 45 ready / 0 failed products, 180 rows
(four Visual tasks with 17 each, four Full Audio tasks with 17 each, four
Speech+BGM tasks with 11 each), 8 donors / 14 segments. These are the supplied
server baseline counts, not a claim that a local synthetic fixture replayed
the server dataset. Do not replace actual failures or overwrite older bundles.
