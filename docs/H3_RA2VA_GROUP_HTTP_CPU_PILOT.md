# RA2VA HTTP Group CPU Pilot

Status: Tasks 1-3 implemented and locally tested; real two-node acceptance is
pending SSH recovery.
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
For a later authorized remote Pilot, verify A_PRIVATE_IP, PORT and LOCAL_GUARD:

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

## Pending Real-node Proof

Local loopback does not prove actual two-node behavior. After access restoration,
use a fresh run to prove both hostnames complete concurrent distinct clips, Worker
kill/reclaim, stale rejection, Coordinator restart with unexpired work, draining
barrier, and completed-run resume. No remote connections or writes were made for
the local implementation milestone.

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
