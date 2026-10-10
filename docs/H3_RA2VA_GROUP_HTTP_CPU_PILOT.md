# RA2VA HTTP Group CPU Pilot

Status: Tasks 1-3 implemented and locally tested; Task 4 real two-node HTTP CPU
acceptance passed on 2026-10-10 using the committed `cd7875c` snapshot.
No GPU, model calls, real Audio/H3 products, full group, or automatic failover.

## Ownership and Recovery

Only the designated Coordinator updates shared run metadata. Startup order is
verified local process guard, exclusive TCP bind, then persistent state recovery.
The guard must be on a confirmed local mount, not JDUFS. HTTP and maintenance use
the same process RLock; inherited metadata locks are same-host details only.
Workers never acquire scheduling file locks or write shared control state.
The resolved guard path and bind host/port are persisted with the designated
hostname. Changing them on resume is rejected before recovery/state mutation.

Each claim has generation/task/session/token. Accepted receipts are immutable;
resending confirms them without rewriting/counting, even after generation change.
Previously unaccepted stale tokens cannot publish. Persistent leases use Coordinator
Unix epoch time; Worker deadlines use monotonic request-send time. Heartbeat and
watchdog are independent of Fake execution. Short disconnects retain work.

After watchdog expiry the Fake context closes before stopped-execution reporting
and rejoining. A lost result reply may then only confirm an accepted receipt,
never publish an unaccepted result. Truncated replies retain the same session and
claim through reconnect. The HTTP launcher uses a shared byte stop flag, not a
killable inter-process Event mutex, so killed-child cleanup stays bounded.
Lease revocation is not proof of physical process death; future
GPU zombie cleanup is unproven. Failed/skipped business results are never retried.
Resume uses the original run-id on Node A; no automatic failover or run migration.

## Commands

Use an existing Python 3.12 environment. Set `R2V_GROUP_COORDINATOR_TOKEN` privately
in each process environment, never in committed commands or logs. Assign RUN_ID
once, then reuse it. Local tests may use temporary published V3 fixtures/output.
For a new authorized remote Pilot, verify A_PRIVATE_IP, PORT and LOCAL_GUARD:

```bash
SOURCE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask
OUTPUT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
# Assign a new cpu-http-group20-<timestamp> RUN_ID and retain its value.
python tools/run_h3_ra2va_group_production.py coordinator \
  --source-root "$SOURCE" --output-root "$OUTPUT" --run-id "$RUN_ID" \
  --limit 20 --max-groups 1 --host "$A_PRIVATE_IP" --port "$PORT" \
  --local-lock-path "$LOCAL_GUARD"
```

Run on each node (multiple local terminals may instead test loopback):

```bash
python tools/run_h3_ra2va_group_production.py worker --fake \
  --coordinator-url "http://$A_PRIVATE_IP:$PORT" --node-id "$NODE_ID" \
  --workers 1 --fake-delay-seconds 1
```

Resume Coordinator with the identical command, Node A, guard and run-id. Restart
stopped Workers with their command. Never start a Node B Coordinator takeover.

Read stable per-stage elapsed time, throughput and terminal counts:

```bash
python tools/run_h3_ra2va_group_production.py status \
  --output-root "$OUTPUT" --run-id "$RUN_ID"
```

Authenticated `GET /v1/status` returns current group/generation/phase, memberships,
global settings and progress. The service stays up after PILOT_COMPLETE for status
and receipt confirmation; stop it explicitly when finished.

All-success expected CPU result: 160 unique results (20 x 8), zero model calls,
one PILOT_COMPLETE, no full COMPLETE or second inventory. Interrupted attempts
are separate from accepted results. Logs include node/session, generation,
claim/reclaim/publication/release/revocation, elapsed time and throughput.

## Real Two-node Evidence (2026-10-10)

Run: `cpu-http-group20-20261010T023035`, below the authorized R2VA output root.
The shared `code/` directory is a Git archive of the tested `cd7875c` commit, used
by both nodes. Neither existing server checkout was changed (both stayed at
`65f4dad`); no push was performed. Node A was `nb-94ayoy238c-0`, Node B was
`nb-6numx1uhso-0`. Node B's direct private HTTP request returned 200. Node A's
`findmnt -T /tmp` returned `/ overlay overlay`; its process guard was not on
JDUFS. A second Coordinator launch was rejected by that local guard.

The published group-000000 declares 39,585 source rows. Only its first twenty
ordered V3 rows were frozen; reference/media files were not scanned or decoded.
Initial Fake execution delay was 2s, then 0.5s after the controlled interruption.

- 160 unique terminal results: 160 ready, 0 failed, 0 skipped, 0 active/pending.
- Node A published 89 results; Node B published 71. Both held different clips
  concurrently in group-000000, generation 1 (ordinals 0 and 1).
- Zero model calls. One group inventory and one `PILOT_COMPLETE`, no full
  `COMPLETE`. Cumulative `max_groups=1` survived every restart.
- Total harness elapsed: 139.88s, including deliberate crashes and lease waiting.

| Stage (all Fake) | Accepted | Elapsed (s) | Tasks/s |
| --- | ---: | ---: | ---: |
| canonical | 20 | 73.795 | 0.271 |
| sam | 20 | 10.596 | 1.888 |
| auk | 20 | 5.324 | 3.757 |
| resolve | 20 | 5.277 | 3.790 |
| diarizen | 20 | 5.264 | 3.799 |
| asr | 20 | 5.280 | 3.788 |
| mimo | 20 | 5.276 | 3.791 |
| export | 20 | 5.270 | 3.795 |

These numbers measure bounded Fake scheduling, not Audio/MiMo throughput. The
canonical stage includes the real 60s lease expiry; sam includes the second
Coordinator crash/recovery.

### Recovery Proof

1. SIGKILL Node B's owned child after a claim and before publication, then SIGKILL
   Node A Coordinator. Same-host/run-id restart retained that unexpired claim and
   token. Node A reclaimed it only after the persisted lease expired, 58.03s after
   the kill snapshot, with a different token. The old token returned HTTP 409.
2. A live heartbeat session held generation 1 in draining with all twenty results
   ready. Releasing it allowed generation 2; the dead revoked Worker did not
   permanently block the barrier.
3. A test-only wrapper stopped the Coordinator after atomic result publication
   and before accounting. SIGKILL/restart recovered the unchanged result and
   counted it once. The repeated accepted receipt returned 200. No production
   fault-control endpoint, source modification or new flag was added.
4. Across generation change, an accepted old receipt returned 200 without
   rewriting; an unaccepted old claim and old heartbeat returned 409.
5. Completed-run restart preserved every terminal result byte. Both resumed
   Worker launchers reported `executed=0`, `accepted=0`, `model_call_count=0`.
   Terminal failed/skipped immutability was verified in the local controlled
   regressions, not injected into this all-success remote inventory.

The first setup attempt failed before shared-state initialization because the
test wrapper had not transferred after an SCP timeout. SSH-streamed transfer
succeeded; the error log was retained. This was not a clip/business failure.
Final checks found no owned Coordinator/Worker PIDs and no listening test port.
No existing output or source data was overwritten or deleted.

### Evidence and Resume

Evidence is retained under:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/cpu-http-group20-20261010T023035/evidence/
```

It includes `acceptance-summary.json`, the completed `timeline.jsonl`, per-node
Worker and Coordinator logs, `duplicate-coordinator-guard.log`, `focused.log`,
and the one-off `pilot.py` / `coordinator_gate.py` test harness. The harness is
historical evidence, not a command to rerun against this completed directory.
Private endpoint details remain in that evidence, not repository configuration.

To inspect the completed run without launching work:

```bash
OUTPUT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
RUN_ID=cpu-http-group20-20261010T023035
CODE="$OUTPUT/$RUN_ID/code"
PY=/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
"$PY" "$CODE/tools/run_h3_ra2va_group_production.py" status \
  --output-root "$OUTPUT" --run-id "$RUN_ID"
```

Authorized resume uses the Coordinator/Worker commands above with this CODE/PY,
original Node A private bind, port 18970 and
`LOCAL_GUARD=/tmp/cpu-http-group20-20261010T023035.coordinator.lock`.
Set the same private token in each participating process environment. Never
resume this HTTP Fake run as a native/GPU run. The completed service is stopped.

Six focused Group suites passed again: 80 passed in 22.86s (26 multiprocessing
resource-tracker warnings during intentional process kills). No runtime source
changed for this remote milestone. Actual network-partition/watchdog expiry and
GPU process termination were not exercised remotely; their CPU watchdog tests
remain local. No real Audio/H3 training products were generated by this Fake run.
Real Audio/MiMo device isolation, model release and quality acceptance remain
separate work; this result does not authorize full-group production.

## Local Evidence (2026-10-10)

Six Group suites: 80 passed after the independent review fixes. Event-controlled
SIGKILL covered claim publication, lost claim reply, result-before-accounting,
generation transition, and Worker
death. Unexpired work stayed owned; revoked tokens could not publish. Receipt
reposts did not rewrite files/counts, and draining waited for live release.
Heartbeat/delayed-response/watchdog and local/HTTP isolation regressions passed.
Request-intent/checkpoint tests are Fake only, not a real model adapter.

The cached first 20 real group-000000 tasks were reused read-only in a fresh local
run, through the actual CLI, loopback HTTP, and four spawned Fake Workers. Each
task retained its original V3/reference metadata; no media was read. This is
one Mac with two node labels, NOT two physical nodes.

| Stage | Accepted | Elapsed (s) | Tasks/s |
| --- | ---: | ---: | ---: |
| canonical | 20 | 1.474 | 13.57 |
| sam | 20 | 0.252 | 79.50 |
| auk | 20 | 0.236 | 84.88 |
| resolve | 20 | 0.230 | 86.79 |
| diarizen | 20 | 0.238 | 84.03 |
| asr | 20 | 0.236 | 84.57 |
| mimo (Fake) | 20 | 0.267 | 75.00 |
| export (Fake) | 20 | 0.254 | 78.79 |

Total including CLI startup/restart: 4.618s; node labels completed 79 and 81 tasks.
All 160 results remained byte-identical after Coordinator restart with the same
run-id/guard/port. One PILOT_COMPLETE, no full COMPLETE, zero model calls. Fake
throughput is not Audio/MiMo capacity or a cross-node filesystem result.

Protected regression: 260 passed, 2 skipped, 4 failed. The four failures reproduce
unchanged on baseline a47741d in an isolated archive:

- test_turn2_is_minimal_speech_integration_without_visual_priming
- test_visual_prompt_and_schema_remain_frozen
- test_cache_is_diagnostic_only_and_four_stage_provenance_is_fingerprinted[0]
- test_cache_is_diagnostic_only_and_four_stage_provenance_is_fingerprinted[73]

They assert older Multi-call prompt/backend versions. Those files were not changed.
Ruff, py_compile, and diff-check passed for the changed Python paths. Local tests
used an existing Python 3.12 environment and read-only cached soundfile/FFmpeg
packages; the checkout's Python 3.9 environment cannot import current typing.Self.

Full repository pytest after review fixes: 6450 passed, 144 skipped, 251 failed
(280.05s). An isolated
unchanged a47741d run returned 6401 passed, 144 skipped, 253 failed (297.47s).
All 251 current failed node IDs also failed on the baseline; there are no new
failed node IDs. The existing `test_no_model_imports` fails in both full runs
because earlier tests already import the Two-step module; it passes in the
focused Group run. The full suite is not green. No unrelated source/fixture
repairs were made to hide these failures.
