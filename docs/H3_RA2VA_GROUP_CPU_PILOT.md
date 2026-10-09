# RA2VA Shared-group CPU Pilot

This entry point is fake-only. It tests scheduling, ownership adaptation, and
restart behavior; it does not run Audio/MiMo, materialize H3, or export training
JSONL. No upstream media is opened, copied, or hashed.

## Source Publication and Identity

The supported producer is `tools/export_completed_postmask_group.py`. It writes
and fsyncs a temporary JSONL, then atomically exposes the whole group directory
with its reference aliases. The consumer requires that published JSONL plus
`in_pair_reference/state/resource_epochs/<group>/completed.json` with matching
group ID, `status=completed`, `export_completed=true`, and a sample count.
It ignores hidden staging directories and incomplete markers. Append-in-place
producers are not supported without an explicit final publication signal.

References retain their source order and ownership. Relative image paths are
joined to the published group directory; official shard symlinks are not copied,
rewritten, or rejected for pointing outside that directory. The CPU GroupTask
does not fill legacy H3 media-hash fields or select/drop references.

## Launch and Resume

From a checkout containing these local commits, a bounded server CPU pilot can
be launched by the operator with:

```bash
python tools/run_h3_ra2va_group_production.py run \
  --fake \
  --source-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask \
  --output-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA \
  --run-id cpu-group20-01 \
  --limit 20 --workers 4 --max-groups 1
```

Resume repeats this exact command and run ID. More launchers may use the same
run/settings and dynamically claim the same group. `max_groups` is persisted
and cumulative across every launcher and restart, not a per-process allowance.
A conflicting source root, limit, or max_groups is rejected. Changing worker
count is allowed. Once the global group allowance is complete, another launch
performs no clip work. For discovery tests, explicitly choose a larger finite
max_groups; the runner waits for newly published groups until that allowance
is reached or the operator interrupts it.

```bash
python tools/run_h3_ra2va_group_production.py status \
  --output-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA \
  --run-id cpu-group20-01
```

The CLI requires `--fake` and a limit of 1-200. It has no full-group mode.
`PILOT_COMPLETE` only completes the selected CPU inventory, never the upstream
group. It cannot serve as a real reconcile or training Bundle completion marker.

## Locking and Recovery

- Claim/recover/advance acquire control, then dispatch; any clip/session flock
  probes are nonblocking. Processing retains only its own clip-stage lock.
- Publish retains that owned clip lock, acquires control then dispatch, and
  atomically publishes the result before committing counters. No operation waits
  for another worker's clip lock while holding dispatch. Stable lock files are not
  unlinked, and a live lock is never reclaimed by age.
- Cursor and active claims are one atomic JSON update. Recovery first consumes
  already-published results, then reclaims unlocked unpublished claims. It does
  not repeat terminal ready/failed/skipped work or scan completed media.
- Stage generation binds claims, results, sessions, and release acknowledgments.
  A draining stage refuses new membership and waits for live Fake Worker release;
  dead sessions are identified by released flock, not heartbeat expiry.
- Fake infrastructure errors stop the local launcher. SIGINT/SIGTERM closes its
  children with bounded cleanup. Resume uses the same run ID; it does not create
  an automatic retry loop for business failures.

## Local Evidence: 2026-10-09

Read-only SSH retrieved only the completed marker and first 20 real JSON rows
of group-000000. The formal group has 39,585 rows; only 20 were frozen locally.
The fixture has 39 Pictures: 26 subject, 8 attribute, and 5 background. Absolute
remote video/reference-alias paths were preserved without downloading media.

Four local workers completed eight fake stages, yielding 160 unique stage/clip
results, all ready, zero model calls, and one PILOT_COMPLETE. Workers published
40/41/40/39 results; the measured invocation took 1.180 seconds.

| Fake stage | Wall seconds | Tasks/second |
| --- | ---: | ---: |
| canonical | 0.747 | 26.8 |
| SAM | 0.047 | 427.5 |
| AuK | 0.042 | 478.5 |
| resolve | 0.047 | 426.2 |
| DiariZen | 0.047 | 421.2 |
| ASR | 0.042 | 474.8 |
| MiMo | 0.041 | 488.5 |
| export | 0.041 | 492.1 |

These are local metadata/Fake Worker rates, not Audio/GPU production throughput.
The first stage includes process startup. Completed stage times are persisted
and do not grow when status is read later.

Two separate real-input runs killed an event-controlled child:

- After claim: recovered pending=19/active=1/ready=0; the first clip was reclaimed
  once, then all 160 results completed. Resume invocation: 1.016 seconds.
- After result publication, before count commit: recovery produced
  pending=19/active=0/ready=1, preserving that result without re-execution;
  all 160 results completed. Resume invocation: 1.055 seconds.

Local evidence is under `/tmp/r2va-group-cpu-20261009-local/`, including
`pilot.log`, `reports.json`, and the three isolated run directories. Resume the
completed baseline locally without source access:

```bash
python tools/run_h3_ra2va_group_production.py run --fake \
  --source-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask \
  --output-root /tmp/r2va-group-cpu-20261009-local/runs \
  --run-id baseline --limit 20 --workers 4 --max-groups 1
```

## Boundaries

No server writes, GPU calls, real Audio/H3 products, or full group were run.
Two local launchers were tested, but actual two-node shared-filesystem flock
behavior remains the next milestone. A worker that stays alive while holding
its lock is not forcibly reclaimed. A crash before a result is published may
repeat unpublished fake work; this phase does not assert external exactly-once
model requests. Durable real Visual/Joint checkpoints remain deferred.
