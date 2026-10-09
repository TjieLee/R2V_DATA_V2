# RA2VA Group HTTP Coordinator: CPU Two-node Pilot

Date: 2026-10-09
Code baseline: `f40a6d5`
Status: written design for review; HTTP implementation and acceptance pending.

## Scope and Verified Constraint

Use one designated Coordinator and HTTP Workers on both nodes. All Workers
share the current group and dynamically claim clips from its ordered inventory.
This replaces only the multi-node control plane in the earlier Group design.
Keep the accepted local Fake Worker mode, V3 adapter, eight stages, generation
barriers, terminal results, resume ordering, and cumulative group budget.

The real two-node preflight found that the current JDUFS mount excludes flock
contenders on the same node but not across nodes. Both nodes read the same seven
probe records. Evidence remains in the isolated
`R2VA/cpu-two-node-20261009-211614` run; no Group tasks ran there. Do not reuse
that directory or substitute another unverified distributed file lock.

This milestone is CPU-only: first 20 rows of the published `group-000000`,
eight Fake stages, no media reads, GPU, MiMo, real products, or full group.
Do not change T2VA, Multi/Single, Two-step prompts, ASR/Speaker semantics, or
Audio/H3 export behavior. No hashes, shuffle, human approval gate, or database.

## Ownership and Deployment

- Node A is the sole designated Coordinator host; it may also run HTTP Workers.
- Only that Coordinator writes control, dispatch, membership, claims, terminal
  results, counts, generation, and PILOT_COMPLETE under `R2VA/<new-run-id>`.
- Workers receive GroupTask and upstream status over HTTP. They never call the
  shared-file coordinator directly or mutate its scheduling files.
- A stdlib ThreadingHTTPServer and one process-local mutex serialize state
  transactions. No model work or network wait happens while holding the mutex.
- A same-host process guard on a confirmed local filesystem and exclusive TCP
  bind prevent duplicate local Coordinators. The persisted designated host is
  checked before state mutation. These are not distributed leader election.
- Resume starts on that same host, run-id, and endpoint. Never start a standby
  Coordinator on Node B. Automatic failover is explicitly unsupported.
- Persist HTTP transport mode at run creation. Local `run --fake` must refuse
  an HTTP-owned run; the HTTP service must refuse a local-mode run. Keep prior
  run directories unchanged rather than implicitly migrating their ownership.

Bind a verified node-to-node interface, not a public endpoint. Use one shared
Bearer token supplied through an environment variable, not committed or printed.
Verify direct HTTP reachability before the test. SSH/browser access to a notebook
does not prove that Node B can reach the Coordinator interface. Do not change
firewalls, notebook proxies, or security settings as part of this milestone.

## API and Worker Client

All mutations are POSTs. Replies include current group, stage, generation, and
phase. A task ID is `(group_id, generation, stage, ordinal)` within the run;
clip_uid remains the frozen source identity. Session IDs and claim tokens are
unique opaque identifiers, not content hashes.

| Endpoint | Contract |
| --- | --- |
| GET `/v1/status` | Read progress, current phase, global budget, node participation. |
| POST `/v1/workers/connect` | Register a worker instance or reconcile its existing session/claim after reconnect. Return existing work or receipt before offering new work. |
| POST `/v1/workers/heartbeat` | Renew only an unexpired current-generation session; report running/stopped execution. Never revive a revoked claim. |
| POST `/v1/tasks/claim` | Return one GroupTask, upstream status, generation, task ID, token, and lease duration. One outstanding claim per session. |
| POST `/v1/tasks/event` | Persist execution/request-start/response-saved milestones for the current token. Request-start acknowledgment must precede any future model request. |
| POST `/v1/tasks/result` | Publish ready/failed/skipped for the matching current token; result first, counts second. |
| POST `/v1/workers/release` | Acknowledge stage-resource closure for that session and generation. |

Connect is idempotent for the same Worker instance. Claim replies lost in transit
do not consume another task: the same session gets its existing claim. A result
repost with the same accepted token, status, and payload
acknowledges the existing receipt without rewriting or recounting it. A different
payload, token, owner, task, or generation is rejected with an explicit conflict.
An old-generation result may be queried as a receipt but cannot be published.

The Coordinator alone advances stages during its maintenance tick. Workers have
no advance or count-update API. Completed run status causes clients to exit;
new groups are discovered only while the persisted global budget permits them.
`max_groups=1` remains a whole-run limit across clients and restarts.

## Reusing Existing Persistence

Keep existing inventory JSONL/offsets and per-generation stage directories.
Extend current membership with node, worker instance, session, lease deadline,
and release/expired state. Extend each active entry with claim token and execution
milestones. Publish those fields with cursor/active in the same atomic update;
do not issue a task before its token is durable. Terminal receipts retain task,
generation, session/node, token, elapsed time, status, and Fake payload.

Reuse GroupCoordinator's result-before-count ordering, bounded active-result
reconciliation, stage transitions, and Pilot completion rules. Add only narrow
hooks for atomic claim metadata, terminal receipt metadata, and restoring owned
active/session handles on the designated host. Do not fork the entire scheduler.
Shared-file flock may remain an internal same-host implementation detail, never
the authority for a remote Worker's liveness or permission to execute.

Lock order is process mutex -> control -> dispatch. Any inherited clip/session
flock acquisition stays nonblocking. Restore or release owned handles only within
serialized Coordinator operations. Never wait for a Worker, HTTP response, or
another clip lock while holding dispatch. Workers hold no scheduling file locks.

On startup, reconcile already-published active results first. Restore unexpired
membership and claim ownership, including a draining stage, before allowing
claim or advance. Losing Coordinator file descriptors on restart must not make
still-running remote work immediately reclaimable. No completed media scan or
all-result rerender is needed.

## Leases, Interruption, and Recovery

Normal timing: heartbeat every 5 seconds, request timeout 3 seconds, Worker
watchdog 30 seconds, bounded execution cleanup 5 seconds, Coordinator lease
60 seconds. Persist deadlines for Coordinator restart; Worker watchdogs use a
local monotonic deadline anchored to the renewal request's send time. A delayed
reply cannot extend local execution beyond its authorized window. Short network
outages retain the claim and do not cause reassignment.

The Fake execution loop is interruptible and checks stop/lease state. A Worker
must stop its own execution before reconnecting after watchdog expiry, and must
close stage resources before release acknowledgment. Lease expiry revokes
authority; it is not proof that a remote process physically died. Late tokens
cannot publish even if an old process resumes. This phase does not claim a
network lease proves future GPU process termination.

| Interrupted state | Coordinator action |
| --- | --- |
| Terminal result exists, counts incomplete | Account that result once; never execute again. |
| Session unexpired, no terminal result | Keep ownership through short disconnection or Coordinator restart. |
| Worker reports execution stopped and resources closed | Revoke unfinished claim; allow safe Fake/pre-request recovery. |
| Session expires without a result or model-request intent | Revoke token, close owned handles, reclaim unfinished Fake/pre-request task. |
| Request-start intent exists without a durable saved response | Preserve `interrupted_request_unresolved`; never blindly issue the request again. |
| Response checkpoint already saved | Reuse that response, not another model request. Real model adapter remains a later milestone. |

Only interrupted unpublished work can execute again. Terminal business failed
and skipped are not retries. There is one accepted terminal result per task;
this is not an external exactly-once execution guarantee across a crash.
The CPU tests simulate request intent/response checkpoints without calling a
model or importing GPU backends.

Draining waits for every valid session to finish its claim and acknowledge
resource closure. Expired sessions are explicitly revoked, not falsely marked
released, and cannot block healthy CPU Workers forever. A returning expired
Worker must finish local cleanup before joining the current generation with a
new session. Old generation heartbeats/releases/results never mutate the new
stage. Actual GPU cleanup and durable Visual/Joint integration are out of scope.

## Module and CLI Boundaries

- `ra2va_group_source.py`: unchanged published V3 identity/ownership adapter.
- `ra2va_group_production.py`: localized persistence/restore hooks only; preserve
  existing local mode and its tests.
- `ra2va_group_http.py`: single-writer HTTP service, leases, token fencing,
  membership, recovery tick, and result receipts.
- `ra2va_group_http_worker.py`: stdlib HTTP client and CPU Fake lifecycle,
  reconnect reconciliation, watchdog, and release acknowledgment.
- Existing Group CLI: add explicit `coordinator` and `worker --fake` commands;
  retain local `run --fake` and status behavior. No full-group/GPU option.
- Focused HTTP tests and a small CPU runbook: no changes to protected pipelines.

The implementation plan will split persistence/API, Worker lifecycle, crash and
barrier tests, and bounded real-node acceptance into separately tested commits.
Keep commits local; pushing requires separate user confirmation.

## Acceptance and Evidence

First run local event-controlled tests for lost claim/result replies, concurrent
clients, Worker death, Coordinator death before/after result publication,
unexpired recovery, lease expiry, stale tokens/generation, draining release,
terminal immutability, dynamic groups, and cumulative max_groups. Retain existing
source/local scheduler regressions and protected pipeline regressions.

Then create a new authorized 20-row CPU run and use two actual nodes:

1. Gate two Workers until both hold different clips of the same generation and
   group; release them and retain successful results from each node.
2. Kill a claimed Node B Worker before publication; Node A completes the safely
   recovered task after expiry. Retain the old token and demonstrate rejection.
3. Kill the Coordinator with an unexpired running claim, then restart it on Node A
   with the same run-id. Confirm the task is not immediately reassigned.
4. Exercise result-written/count-not-committed recovery and a live release barrier;
   verify a revoked dead session cannot block advancement indefinitely.
5. Resume after completion: no terminal work repeats, no second group starts,
   one PILOT_COMPLETE exists, and no full-group COMPLETE is created.

Use event gates and bounded Fake delays, not just two launcher exit codes.
Logs record timestamps, node, session, task, generation, claim/reclaim/publish,
release/revoke, elapsed time, and stage pending/active/terminal throughput.
Report accepted unique results and interrupted attempts separately. Preserve
existing evidence and report real connectivity failures rather than substituting
local tests. No real two-node HTTP or GPU success is claimed by this document.
