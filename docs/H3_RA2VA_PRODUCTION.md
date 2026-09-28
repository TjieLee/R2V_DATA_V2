# RA2VA MiMo production

This runner consumes frozen Visual production, canonical target Audio, resolved
stems, no-LR-ASD raw DiariZen, and Qwen3-ASR. It never regenerates those stages.
The default is the existing V2.5 `multi` backend. V2.6 `single` is opt-in and
has not been validated with real inference by this code-only change.

Provide the actual frozen roots for the run. `SHADOW_RUN_ID` must name a run
whose `resolved_stems_v1`, `diarization_no_lrasd_v1`, and `asr_no_lrasd_v1`
are complete. The selected case manifest must be an ordered subset of that
stem inventory. Omit `--case-manifest` to select the reference-eligible Visual
clips in upstream stem order. The runner does not independently hash source
videos or canonical full-audio files during selection; MiMo still reads the
selected media for inference. Existing frozen path/hash metadata is retained.

```bash
python tools/run_h3_ra2va_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" \
  --case-manifest /tmp/mimo-refgraph-random20.json \
  --production-root "$RA2VA_PRODUCTION_ROOT" \
  --media-root /mnt/workspace \
  --mimo-model mimo-v2.6-flash-rl --mimo-call-mode single \
  --mimo-gpu-groups '0,1,2,3;4,5,6,7' --mimo-ports 8094,8095 \
  --request-workers 2 --allow-unverified --dry-run
```

Remove `--dry-run` only after reviewing the input paths and fixed-case
selection. The runbook's historical `random20-refgraph-v1` V2.5 output is not
an input substitute for a completed **resolved/no-LR-ASD** run. Use a separate
caller-owned production root; do not overwrite that historical output.

For four nodes, give every node the same roots, seed and shard range, changing
only `--rank` from 0 through 3:

```bash
python tools/run_h3_ra2va_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" \
  --production-root "$RA2VA_PRODUCTION_ROOT" \
  --media-root /mnt/workspace \
  --mimo-model mimo-v2.6-flash-rl --mimo-call-mode single \
  --mimo-gpu-groups '0,1,2,3;4,5,6,7' --mimo-ports 8094,8095 \
  --request-workers 2 --shard-start 0 --shard-end 99 \
  --shard-seed 20260917 --rank "$RANK" --world-size 4
```

The two MiMo servers start only while a shard has pending MiMo work. Each
endpoint has one active request; whichever becomes free first serves the next
clip. Per-shard `shard.lock` prevents concurrent writers. Each ready clip is
published under `shards/<shard>/artifacts/<clip_uid>/reconcile/` before its
status is appended to `state.jsonl.partial`. Restarting the same command
reuses ready artifacts, retries failed clips once in that invocation, and
leaves skipped upstream clips terminal. Connection/startup failures stop the
shard without marking unstarted clips failed. No retry, repair, polish or
multi-call fallback is added to single mode.

To keep the existing V2.5 multi-call comparison, omit all V2.6 options and
run with `--mimo-model mimo-v2.5 --mimo-call-mode multi`; its single local
endpoint remains on port 8092.

When ready annotations should enter the existing Audio-reuse/R2VA projection
tools, publish one immutable view (no MiMo call):

```bash
python tools/build_h3_ra2va_reconcile_view.py \
  --production-root "$RA2VA_PRODUCTION_ROOT" \
  --output-root "$RA2VA_READY_VIEW_ROOT" \
  --shards 0,1,2
```

The view contains the same ready `MimoStemReconcileRecord` annotations for
both RA2VA Audio-reuse preparation and later R2VA frame-conditioned products.
Point the existing `prepare_h3_audio_reuse_sources.py` at it as
`--base-reconcile-root` with `--binding-evidence-mode none`; then use the existing materialization and frame
projection tools. No separate R2VA MiMo pass is needed. A new view path is
required for a later, larger snapshot; existing views are immutable.
