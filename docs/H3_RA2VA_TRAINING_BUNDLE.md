# Frozen reconcile to R(A)2VA review bundle

This CPU-only entry point consumes `source_contract.json`, `records.jsonl`, and
`summary.json` from a completed reference-only reconcile. It performs no model
calls and does not rebuild Visual, DiariZen, ASR, or reference selection.

The reconcile's parent is the named run. Its `resolved_stems_v1` inventory points
to the canonical Audio manifest, which identifies the Audio production root.
Canonical sample metadata comes from `<audio-production-root>/h3/samples.jsonl`.
Only the matching canonical sample IDs are used; in-pair/cross-pair rows are not
imported. Frozen job references are projected with the existing adapter, voice
bindings are cleared, and speech comes exactly from the frozen job.

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
variants, zero model calls, and `production_artifacts_modified=false`.

## Human review and PASS-only export

```bash
.venv/bin/python tools/serve_h3_training_task_review.py serve \
  --ra2va-shadow-root "$OUT" --review-root "${OUT}-review" \
  --host 127.0.0.1 --port 8769

# Only after recording human PASS / ISSUE / SKIP decisions:
.venv/bin/python tools/serve_h3_training_task_review.py export \
  --ra2va-shadow-root "$OUT" --review-root "${OUT}-review" \
  --output-root "${OUT}-pass-only"
```

Review uses the existing 12 R(A)2VA task categories. Ready is not PASS. Known
semantic issues such as the `469d` table/speaker binding are left for human
ISSUE review, not repaired here. Export rows have exactly `video`, `images`,
`audios`, `caption`; `videos.jsonl` has only `video` and `tasks`.
