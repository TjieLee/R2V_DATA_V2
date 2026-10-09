# RA2VA HTTP Group CPU Pilot

Status: local CPU implementation; real two-node acceptance is pending SSH recovery.
No GPU, model calls, real Audio/H3 products, full group, or automatic failover.

## Ownership and Recovery

Only the designated Coordinator updates shared run metadata. Startup order is
verified local process guard, exclusive TCP bind, then persistent state recovery.
The guard must be on a confirmed local mount, not JDUFS. HTTP and maintenance use
the same process RLock; inherited metadata locks are same-host details only.
Workers never acquire scheduling file locks or write shared control state.

Each claim has generation/task/session/token. Accepted receipts are immutable;
resending confirms them without rewriting/counting, even after generation change.
Previously unaccepted stale tokens cannot publish. Persistent leases use Coordinator
Unix epoch time; Worker deadlines use monotonic request-send time. Heartbeat and
watchdog are independent of Fake execution. Short disconnects retain work.

After watchdog expiry the Fake context closes before stopped-execution reporting
and rejoining. Lease revocation is not proof of physical process death; future
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
