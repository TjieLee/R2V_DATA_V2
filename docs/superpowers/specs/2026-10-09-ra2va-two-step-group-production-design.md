# RA2VA Two-step Group Production

Date: 2026-10-09
Branch: `feature/h3-audio-jea-qwen3-v1`
Inspected baseline: `40c34d6c9992af33f92ad06e5ff56037fb9658d1`

## Goal and Authorization

All participating nodes process the same current group and dynamically claim
its clip tasks. Preserve the accepted Two-step semantics and publish the
existing 12 four-field training tasks plus a complete group donor reserve.
Implement and validate incrementally, not as an immediate full-group launch.

Confirmed input:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask/group-*/samples.jsonl
```

The originally proposed `post_mask/litengjie/` directory was absent during
read-only SSH inspection. The user confirmed the proposed actual input and
authorized this output root:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
```

This authorization covers only this output subtree, not adjacent public data.
Use `R2VA/<run-id>/` for one shared run; separate CPU pilots, GPU pilots, and
production. Resume reuses the exact run directory. Never overwrite another run
or historical output. The repository's output-path policy must gain this narrow
exception when runtime code is implemented. SSH remains read-only in the
current development session; this design does not authorize remote writes or
model execution by Codex.

## Verified Source Contract

Read `ProductionSample` from `v3/production_export.py`, not private V3 sidecars.
The first 20 real rows of `group-000000` passed the existing schema; their 39
reference PNG paths and 20 target video paths existed. This was a bounded path
inspection and CPU schema check, not an Audio/GPU production run.

Each relative `image_path` is rooted at its published group directory. Existing
`references/visual/<shard>` directory symlinks lead to official shard references.
Consume those aliases without copying PNGs or rejecting them merely because
their resolved target is outside the group directory. Never write to either
the aliases or their targets.

Preserve reference order, `image_id`, entity ownership, attribute ownership,
and source-frame metadata. Image N maps to Picture N. Subject construction
follows the existing entity/attribute/background ordering, never Subject-number
speaker inference. Do not invoke randomized reference selection or rerun V3.
Unsupported source inventories remain explicit terminal source failures rather
than being silently truncated or repaired.

## Chosen Approach and Reuse

Use a Two-step-specific file-backed coordinator and thin stage adapters.
Reusing the old shard runner wholesale is unsuitable: it assigns shards to
nodes, holds a shard lock, and retries failed reconcile clips. A claim-directory
lease with heartbeat expiry would require stale-owner fencing and more recovery
machinery; prefer existing advisory file-lock primitives, subject to a real
two-node shared-filesystem test before GPU rollout.

Reuse:

- T2VA phase order, GPU masks, worker backend factories, and owned-process cleanup.
- Stage-local SAM/AuK/DiariZen/ASR models, resident across tasks within a stage.
- MiMo V2.6 TP4/Marlin server commands, child environment, endpoint pool, and cleanup.
- Two-step Visual v5 and Joint v2 request construction, normalization, validation,
  ASR authority, and identity-publication restrictions.
- Existing reconcile record types, lightweight Audio/H3 materializers, training
  manifest exporter, and voice donor exporter.

Do not reuse the old hash-validating worker receipt/resume path or the Visual
inventory loader that requires `clip.json` and rejects the published symlinks.
Keep Multi/Single and existing T2VA entry points unchanged.

## Module Boundaries

New modules are specific to this pipeline, not a generic scheduler framework:

- `ra2va_group_source.py`: discover published groups and import ordered V3 rows
  into an output-owned task inventory; adapt reference ownership and paths.
- `ra2va_group_production.py`: current group/stage, dynamic claims, terminal
  publication, recovery, stage barriers, progress, and basic timing counters.
- `ra2va_group_workers.py`: isolated stage workers using existing model factories
  and a Two-step-only durable turn adapter; no model implementation duplication.
- `tools/run_h3_ra2va_group_production.py`: explicit source/output roots,
  node identity, GPU groups, bounded pilot inventory, run/status entry points.

Add only a thin export adapter if needed to combine existing per-clip products
into group JSONL and donor indexes. Do not broaden the legacy exporters' defaults.

## Shared Scheduling and Recovery

One small run control record identifies the current group and stage. A short
control lock serializes group selection and stage advancement, never clip
execution. Select already-started incomplete groups first, then numeric group
order. Poll only group directory names and newly published `samples.jsonl`
files to discover later groups. Do not assume the initial listing is final.

Import a group's JSONL once in source order into an output-owned inventory with
row offsets. Restart consumes that inventory and stage metadata, not all source
media. A bounded pilot freezes only its selected first 10-20 rows and cannot
subsequently masquerade as completion of the entire source group.

For each stage, keep an atomic dispatch record containing the next inventory
offset, terminal counters, and the bounded set of active claims. Under a short
dispatch lock, a worker acquires the clip-stage lock, records its claim, and
advances the cursor. It holds only that clip-stage lock while doing work.
There is no rank/world-size partition, shuffle, or whole-group ownership.

Publish a complete per-clip stage result atomically before removing the active
claim and incrementing terminal counters together in the dispatch record. If a
process dies between those writes, recovery consumes the existing result and
accounts for it once. If no result exists and the clip-stage lock is available,
the interrupted claim can be reassigned. Do not reclaim a live lock by age.
Normal `ready`, `failed`, and upstream `skipped` results are terminal and are
not retried on restart. An infrastructure failure stops the affected local
worker session; it does not invent a semantic failure for unexecuted clips.

When the inventory is exhausted and every clip is terminal, active nodes drain
and release that stage's models before acknowledging the barrier. Only then
advance the stage. Dead-node membership is determined by released session
locks, not a fixed world size. A late joiner reads the current stage rather than
starting an earlier one. After exports finish, atomically mark the group complete
and select the next group. A final group exporter may own its publication lock;
this never grants exclusive ownership of the group's clip work.

## Stages and Durable MiMo Turns

```text
canonical -> SAM -> AuK -> resolve -> DiariZen -> ASR
          -> MiMo (Visual, Joint) -> products/frames -> JSONL/donor publication
```

CPU and GPU workers claim tasks from the same stage inventory. GPU backends
load once per local stage session; physical GPU masks and child `cuda:0` remain
as in T2VA. MiMo starts only for its stage, using TP4 replicas and existing HTTP
media serving. The next stage must not inherit an earlier model's GPU residency.

Persist each real Visual/Joint raw and its real completion diagnostic immediately
after the response, before parsing or starting another turn. On recovery, parse
and reuse saved Visual output and request only a missing, not-yet-issued Joint.
If both responses exist, finish publication without another request. Carry the
original HTTP/token audit and distinguish reused calls from calls made during
the current worker session; never fabricate a new HTTP completion.

A successful clip has one real Visual and one real Joint request. There is an
unavoidable interruption window after sending a request but before durably
saving its response. Record request-start intent; if that intent has no saved
response after owner loss, publish `interrupted_request_unresolved` as Failed
instead of automatically issuing a possible third call. Other unfinished tasks
remain reclaimable. No retry, polish, profile call, or Multi-call fallback.

Keep Joint v2, Visual v5, temperature, ASR, gN/Sx, binding, multiple-speaker
exclusion, and identity-unconfirmed publication rules unchanged.

## Products and Lightweight Operation

Materialize missing clip products through existing lightweight functions, then
aggregate group manifests in frozen source order. Do not rerender completed H3,
regenerate completed frames, or decode published audio for verification.
Keep existing FFmpeg timeouts and actual-operation error reporting.

Do not add media hash scans, checkpoint/model/config hashes, seed shuffle,
integrity frameworks, or a human QA gate. Existing schema-required metadata is
compatibility data, not a reason to introduce source-media verification or
resume scans. Newly generated artifact metadata is written once, not rechecked
by reading the artifact back. Resolve required legacy metadata fields in the
group adapter without changing frozen consumers or pretending that unchecked
source content was verified.

Emit the unchanged `video/images/audios/caption` task format. Aggregate
`voice_donor_reserve.json` per group, preserving every eligible speech segment
under `(clip_uid, entity_id, speaker_group)`. Reuse existing donor eligibility;
never merge clip-local entity IDs across clips or publish restricted identities.

Progress reports group/stage, pending/active/ready/failed/skipped counts,
completed clips per second, stage elapsed seconds, actual/reused MiMo calls,
correction counts, and failure reasons. Diagnostics are automatic audit data,
not a review/PASS dependency.

## Incremental Acceptance

1. CPU-only input/queue milestone: use the 20 inspected real V3 rows as a small
   fixture with fake stage workers. Test ordering, ownership, multiple workers,
   interrupted claims, terminal failures, restart, and discovery of a new group.
2. Two-node CPU milestone: first test actual shared-filesystem lock exclusion and
   crash release in a new authorized run directory, then demonstrate both nodes
   claiming the same group with no duplicate terminal publication. Test worker
   interruption, node loss, model-release acknowledgments, and restart.
3. GPU milestone: attach existing stage backends and run only 10-20 clips.
   Check durable Visual/Joint recovery, exactly two calls for successful clips,
   frozen semantics, automatic identity restrictions, all available task types,
   and multi-segment donor reserve.
4. Scale milestone: run 100-200 clips in another new directory and report real
   stage throughput, memory/lifecycle behavior, and recovery results. Do not
   begin full groups automatically after this pilot.

Focused tests must retain Multi/Single and existing Bundle regressions. Inspect
the actual diff, run Ruff/compile/diff checks, and provide runnable commands only
for implemented milestones. Local fake workers do not establish real two-node
or GPU throughput, and schema/path checks do not establish model quality.
