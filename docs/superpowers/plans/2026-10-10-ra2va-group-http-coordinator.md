# RA2VA Group HTTP Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native Tasks 1-3 were approved and are implemented locally. Task 4 remains deferred.

**Goal:** Replace cross-node file-lock coordination with one persistent HTTP Coordinator and verify CPU Fake scheduling/recovery locally, then on two real nodes with a 20-row Pilot.

**Architecture:** The designated Coordinator alone owns global state and reuses the existing GroupCoordinator through narrow persistence/restore hooks. HTTP Workers dynamically claim the same current group, maintain independent heartbeats, stop local Fake execution on watchdog expiry, and submit fenced terminal receipts. Existing local Fake mode remains isolated and compatible.

**Tech Stack:** Python 3.12, stdlib http.server/urllib/threading/multiprocessing, existing atomic_json and GroupTask, same-host fcntl process guard, pytest, Ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-09-ra2va-group-http-coordinator-design.md`, approved on 2026-10-10, plus the user's implementation clarifications. Starting HEAD: `a47741d`; unchanged runtime baseline: `f40a6d5`.

## Global Constraints

- Branch `feature/h3-audio-jea-qwen3-v1`; keep local commits, never push automatically.
- Tasks 1-3 are authorized for local CPU implementation; no server access during this milestone.
- Source is read-only `/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask/group-*/samples.jsonl`; relative reference paths remain rooted at the published group, including official symlinks.
- Real output is a new run under `/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/<run-id>/`; resume uses that exact run-id.
- Use existing V3 adapter/GroupTask, ordered inventory, eight stages (`canonical`, `sam`, `auk`, `resolve`, `diarizen`, `asr`, `mimo`, `export`), generation and cumulative max_groups.
- One designated Node A Coordinator, no standby/election/automatic Node B takeover. Workers never modify shared scheduling files.
- JDUFS stores persistent data, not distributed lock authority. Coordinator mutual exclusion uses a verified local filesystem and unique TCP bind.
- Heartbeat 5s, HTTP timeout 3s, Worker watchdog 30s, cleanup bound 5s, Coordinator lease 60s. Persist Coordinator Unix-epoch deadlines; Worker deadlines use local monotonic time.
- Lease expiry revokes submission authority, not proof of remote process death. A stopped/expired Fake Worker must close local execution before rejoining. Future GPU termination remains unproven.
- Terminal ready/failed/skipped is immutable. Reconcile existing results before reclaiming interrupted work; no business-failure retry or extra model request.
- No GPU/MiMo, full group, media reads/hash scans, seed shuffle, static shard assignment, QA gate, Redis/database, or protected T2VA/Multi/Single/Two-step/Audio/H3 changes.
- SSH currently fails with `proxy read: Broken pipe`. Do not retry or change network configuration during Tasks 1-3; Task 4 waits for restored access and verified direct node-to-node HTTP.

## Review Focus

- Lost connect/claim reply: reconnect returns the same durable session/claim, never another cursor increment (Tasks 1-2).
- Accepted result reply lost across a generation advance: acknowledge the existing receipt without mutation; reject any previously unaccepted stale result (Task 3).
- Delayed renewal acknowledgment during a blocked task: independent heartbeat and watchdog cannot extend permission from reply-receive time (Task 2).
- Coordinator dies while a stage is draining: restore unexpired sessions/claims before any claim or advance; never mistake lost local descriptors for dead remote Workers (Task 3).
- Wrong transport, host, guard path or busy TCP port: fail before shared state mutation; no implicit run migration or Coordinator failover (Tasks 1 and 4).

## Files and Shared Interfaces

- Modify `r2v_data_v2/h3/ra2va_group_production.py`: constructor transport/owner isolation; optional claim/member/receipt metadata; restore owned handles. Retain claim cursor, result accounting, selection and barrier algorithms.
- Create `r2v_data_v2/h3/ra2va_group_http.py`: HTTP service, serialized state transactions, leases/token fencing, receipts, maintenance/recovery and same-host process guard.
- Create `r2v_data_v2/h3/ra2va_group_http_worker.py`: HTTP client, independent heartbeat/watchdog, interruptible Fake execution, reconnect and resource-release lifecycle.
- Modify `tools/run_h3_ra2va_group_production.py`: explicit `coordinator` and `worker --fake` subcommands. Keep local `run --fake` and file status compatible.
- Create `tests/test_h3_ra2va_group_http.py`, `tests/test_h3_ra2va_group_http_worker.py`, `tests/test_h3_ra2va_group_http_recovery.py`, and `tests/h3_ra2va_group_http_helpers.py` for loopback/event-controlled fixtures only.
- Create `docs/H3_RA2VA_GROUP_HTTP_CPU_PILOT.md`; link it from the existing CPU runbook once executable commands exist.
- Keep `ra2va_group_source.py`, existing FakeWorker semantics, original fixtures, prompts, validators, materializers and donor export unchanged.

Wire identity: `task_id={group_id, stage, generation, ordinal}`; every task reply includes GroupTask, session_id, claim_token, upstream_status, and lease_seconds. Metadata remains ordinary JSON; opaque UUID tokens are claim identities, not data hashes.

HTTP API: GET `/v1/status`; POST `/v1/workers/connect`, `/heartbeat`, `/release` under `/v1/workers/`; POST `/v1/tasks/claim`, `/event`, `/result`. All endpoints use an environment-supplied Bearer token; never persist or log it.

Connect body identifies node_id and worker_instance, with previous_session_id when reconnecting and resources_closed when joining a new session. Claim/heartbeat/release include session_id and generation. Event/result include session_id, task_id and claim_token. Result adds status, payload and elapsed_seconds. After local watchdog expiry, receipt_only=true permits acknowledgment only, never a new publication. Release is accepted only after local resources are closed; it cannot silently abandon a running task.

Active metadata is nested under `http` in dispatch entries; member metadata under `http` in members; terminal receipt metadata under `http` in results. Core worker_id is the session_id. Nested metadata cannot override frozen task identity, generation, terminal status or counts.

### Task 1: Coordinator Persistence and Minimal HTTP API

**Files:** production hooks, new HTTP module, HTTP tests/helpers, Coordinator CLI subcommand.

**Interfaces:**
- Extend `GroupCoordinator(..., max_groups=1, transport="local", coordinator_host=None, coordinator_identity=None)`. Existing records without transport mean local; HTTP requires and persists designated host plus resolved local guard path/bind host/port. Opposite-mode or changed-identity resume is rejected without rewriting existing files.
- Extend `claim(..., claim_metadata: dict | None=None)`, `join(..., member_metadata: dict | None=None)`, and `publish(..., receipt_metadata: dict | None=None)` with keyword-only metadata. Default local calls retain current behavior.
- Add `restore_claim(snapshot: StageSnapshot, ordinal: int, worker_id: str) -> TaskClaim` and `restore_session(snapshot: StageSnapshot, worker_id: str) -> WorkerSession`: restore matching persisted ownership, not a new claim/join; allow current draining membership restoration, no cursor/count change.
- `HttpGroupCoordinator(core: GroupCoordinator, *, wall_clock: Callable[[], float]=time.time)` exposes `dispatch(method: str, path: str, body: dict | None=None) -> tuple[int, dict]`, `tick() -> None`, `close() -> None`. One RLock serializes its state operations; tests may inject a wall clock.
- `coordinator_process_guard(local_lock_path: Path)` is a context manager owning a nonblocking same-host flock; reject resolved paths under the shared `/mnt/workspace` tree. Task 4 must additionally prove the chosen parent is a real local mount.
- `build_http_server(host: str, port: int, *, service_factory: Callable[[], HttpGroupCoordinator], bearer_token: str) -> ThreadingHTTPServer` binds before calling the factory. CLI acquires the local guard first, binds second, then creates/restores shared state and starts serving.

- [x] **Step 1: Add failing ownership/API tests.** Use published-group fixtures and two loopback HTTP clients. Pin these assertions:
  ```python
  assert a_claim["task_id"]["generation"] == b_claim["task_id"]["generation"]
  assert a_claim["task_id"]["ordinal"] != b_claim["task_id"]["ordinal"]
  assert a_claim["claim_token"] != b_claim["claim_token"]
  assert reconnect_claim == a_claim  # lost reply, same session, no extra claim
  assert dispatch_state["cursor"] == 2
  ```
  Add `test_transport_modes_do_not_migrate_runs`, `test_owner_host_mismatch_does_not_write`, `test_local_guard_and_busy_bind_reject_before_factory`, `test_auth_failure_does_not_mutate_state`, and `test_claim_token_and_cursor_publish_atomically`. Compare relevant file bytes on rejection; event-control a crash before dispatch replacement and assert unchanged cursor.
- [x] **Step 2: Run RED.** `python -m pytest -q tests/test_h3_ra2va_group_http.py`. Confirm missing new interfaces/assertions fail; do not count unrelated import/environment failures as a useful RED.
- [x] **Step 3: Implement only the defined hooks and service.** Persist token with the original cursor/active atomic update. Use Coordinator-owned session/clip handles, not Worker file locks. Lock order is RLock -> control -> dispatch -> nonblocking handle probe; no network wait/work under those locks. Implement bounded active-result reconciliation, basic current-token terminal publication and all seven routes. Keep an expiry/restore tick before dispatching work; Task 3 hardens fault boundaries rather than shipping an unfenced intermediate service.
- [x] **Step 4: Add Coordinator CLI.** Require source/output/run identity, bounded limit, max_groups, host/port, and explicit local_lock_path; read `R2V_GROUP_COORDINATOR_TOKEN` from the environment. Default budget stays 1. Never expose automatic takeover or full-group mode. Cleanly close server and owned handles on interruption.
- [x] **Step 5: Run GREEN and local compatibility.** Run the new HTTP tests plus `tests/test_h3_ra2va_group_source.py`, `tests/test_h3_ra2va_group_production.py`, and `tests/test_h3_ra2va_group_fake.py`. Existing local bytes/results and same-run completion behavior remain compatible; no HTTP-owned run can be launched through local mode.
- [x] **Step 6: Review diff and commit locally.** Stage only this Task's paths; commit `feat(h3): add single-writer group HTTP coordinator`. No push or server launch.

### Task 2: HTTP Worker, Heartbeat, Watchdog and Reconnect

**Files:** new Worker module/tests, HTTP test helpers, Worker CLI subcommand, new CPU runbook.

**Interfaces:**
- Consume Task 1's wire/API and existing GroupTask/FakeWorker; Worker code receives no GroupCoordinator or shared-state paths.
- `GroupHttpClient(base_url: str, bearer_token: str, *, timeout_seconds: float=3)`; `call(method: str, path: str, body: dict | None=None) -> dict`. One network operation per call; caller reconciles uncertain outcomes rather than blindly obtaining new work. HTTP 409 is an explicit conflict, not a retry request.
- `WorkerHeartbeat(client: GroupHttpClient, *, monotonic: Callable[[], float]=time.monotonic)` provides `start(session: dict) -> None`, `stop() -> None`, and `execution_allowed() -> bool`. A heartbeat thread owns renewal; an independent watchdog signals execution stop even when a renewal response is delayed. Worker state is protected only by a short local mutex, not held during HTTP or task execution.
- `run_http_fake_worker(client: GroupHttpClient, *, node_id: str, worker_instance: str, stop_event: object, fake_delay_seconds: float=0) -> dict` owns one session/task at a time and returns invocation counters.
- `run_http_fake_workers(*, coordinator_url: str, node_id: str, workers: int, fake_delay_seconds: float=0) -> dict` uses spawned local processes and bounded owned-child cleanup. Credentials come from the environment, not arguments/logs.

- [x] **Step 1: Add RED Worker tests.** `test_long_task_keeps_independent_heartbeats` blocks Fake execution on an Event and observes at least two renewals without finishing the task. `test_delayed_ack_uses_request_send_monotonic_time` delays a renewal reply past the watchdog; assert `execution_allowed() is False` despite that late reply. Use injected clocks/events or test-local timing overrides, not 30-second sleeps.
- [x] **Step 2: Pin reconnect and stop assertions.** Add `test_lost_connect_and_claim_replies_keep_original_claim`, `test_short_network_outage_does_not_reexecute`, `test_result_reply_loss_reconciles_receipt`, `test_watchdog_closes_fake_context_before_rejoin`, `test_release_follows_resource_close`, and `test_idle_worker_heartbeats_while_other_task_runs`. Assert one execution of accepted tasks, unchanged token through short outage, stop/close before new session, and no direct shared metadata writes from Worker processes. Run `python -m pytest -q tests/test_h3_ra2va_group_http_worker.py` for RED.
- [x] **Step 3: Implement client and lifecycle.** Renewal deadlines are anchored to send-time monotonic timestamps, never receive time. Drain closes Fake context, then acknowledges release. Watchdog expiry cancels interruptible delay/execution and fences old submissions; reconcile receipt/current claim before any new claim. Fake delay and test task gates must observe stop events; do not promise to terminate arbitrary blocked Python/GPU work. Preserve failed/skipped terminal upstream propagation. No model imports, request retry/fallback, or GPU cleanup claim.
- [x] **Step 4: Add `worker --fake` CLI and runbook.** Options: coordinator_url, node_id, workers (default 1), fake_delay_seconds (default 0, bounded 0-2s). No output-root or scheduling-state writer option for Workers. Include local launch/resume/status commands, timings, lost-connection behavior and the pending remote milestone; keep old local CLI unchanged.
- [x] **Step 5: Run GREEN.** New HTTP/Worker suites plus existing three Group suites; prove 20x8 unique terminal Fake results, zero model calls, one PILOT_COMPLETE, and no full COMPLETE. Simulate two node labels only as a local test; do not call it two-node acceptance.
- [x] **Step 6: Review diff and commit locally.** Commit `feat(h3): add group HTTP fake workers and lease watchdog`. No push or server connection attempts.

### Task 3: Fencing, Receipts, Crash Recovery and Generation Barriers

**Files:** HTTP service/Worker only where tests expose gaps, dedicated recovery tests/helpers; minimal core hooks only if necessary.

**Interfaces:** Consume Tasks 1-2 unchanged. `tick()` reconciles persisted results, restores or expires current membership/claims, and alone advances stages. Persist epoch `lease_deadline` in member metadata, execution milestones in active metadata, and original accepted token in the receipt. No extra scheduler or receipt database.

- [x] **Step 1: Add terminal/fencing RED tests.** `test_accepted_receipt_reack_after_generation_advance` repeats the identical accepted token/status/payload after the stage changes; assert HTTP 200, original result bytes unchanged, counts unchanged, no new-generation mutation. `test_unaccepted_old_generation_result_is_rejected` asserts 409 and unchanged state. Also test changed payload/status/token, expired token, and duplicate concurrent result posts. Receipt lookup precedes current-generation publication checks only for an exact already-accepted result; it is acknowledgment, not a new publication.
- [x] **Step 2: Add event-controlled crash RED tests.** Spawn the Coordinator, gate it before claim replacement, after durable claim/before reply, after result rename/before accounting, and during generation publication. Kill it, restart with the same local guard/endpoint/run-id, and assert no cursor loss, receipt rewrite, double count, or terminal reexecution. Kill a Worker before result publication; preserve the unexpired claim, then expire/reclaim it with a new token. Never use a fixed sleep as proof of reaching a crash boundary.
- [x] **Step 3: Add lease/barrier RED tests.** `test_restart_restores_unexpired_claim_without_reassignment`, `test_restart_restores_draining_members`, `test_barrier_waits_for_valid_worker_release`, `test_expired_session_is_revoked_not_marked_released`, and `test_old_generation_heartbeat_release_and_event_cannot_mutate_current`. Assert stage stays draining until valid release; expired dead CPU membership eventually stops blocking; late client must stop/close before fresh joining. Add `test_dynamic_groups_and_global_max_groups_survive_restart`, `test_terminal_failed_skipped_never_reexecute`, and `test_completed_run_never_opens_second_inventory` with cumulative budget 1.
- [x] **Step 4: Add simulated request-intent RED tests.** With labeled Fake checkpoints only: request-start without saved response becomes terminal `interrupted_request_unresolved`, never another request; saved response permits checkpoint reuse without a new request. Inspect only the active claim's declared checkpoint, not media trees. These tests do not implement real Visual/Joint/GPU adapters. Run `python -m pytest -q tests/test_h3_ra2va_group_http_recovery.py` for RED.
- [x] **Step 5: Implement minimal recovery fixes.** On startup account durable results before restoring unexpired session/clip handles, including draining stages, before serving claim/tick. Persist revocation before allowing reassignment; reject late original tokens. No reset of generation, lease, cursor or global budget on restart. Coordinator-owned handle close is not evidence of remote cleanup. Reconnect can query old accepted receipts but cannot revive an expired session.
- [x] **Step 6: Run GREEN and protected regressions.** Run all six Group suites plus `tests/test_h3_ra2va_production.py`, `tests/test_v3_post_mask_completed_group_export.py`, `tests/test_h3_mimo26_two_step.py`, `tests/test_h3_mimo25_single_compact.py`, `tests/test_h3_mimo25_two_turn.py`, and `tests/test_h3_t2va_full_production.py`. For baseline-only failures/environment gaps, reproduce on `a47741d` in an isolated comparison without changing source or hiding failures. No unrelated fixture repair.
- [x] **Step 7: Verify and commit locally.** Ruff the changed Python modules/tests; py_compile the changed runtime/CLI modules; `git diff --check`; inspect the full diff against `a47741d`. Assert protected paths are unchanged and preexisting `docs/visualizations/` untouched. Run one final Native independent review, address only scoped findings, then commit `test(h3): harden HTTP group fencing and crash recovery`. Report Tasks 1-3 actual results while Task 4 remains pending.

### Task 4: Real Two-node 20-row CPU Fake Pilot

**Files:** CPU runbook and measured evidence summary only, unless actual defects require a separate scoped fix/test. Remote private rows, tokens and full logs remain outside Git in the new run's evidence directory.

**Interfaces:** Use the already-tested Coordinator/Worker CLI from Tasks 1-3. Nodes share only source/output storage; both Workers communicate with Node A HTTP. No new production fault-control endpoints or distributed lock primitives.

- [ ] **Step 1: Wait for restored access.** This Task is deferred while SSH is broken. Once the user reports restored connectivity, perform one bounded read-only preflight: hostnames, Node A local `/tmp` or `/run` mount using `findmnt -T`, and available private address/port. Direct HTTP is checked after service startup in Step 2. Stop/report failure rather than loop, expose a public service, or modify network settings. Browser access alone is insufficient evidence.
- [ ] **Step 2: Freeze and launch in a new run.** Use `--limit 20 --max-groups 1`, the full source root below, and new `R2VA/cpu-http-group20-<timestamp>`; never reuse old lock-test/Pilot directories. Snapshot tested code under that new run if needed; do not edit server checkouts. Store token only in process environment. Place the process guard on the verified local mount and use the same guard/bind on resume. After starting the Coordinator, call its authenticated `/v1/status` directly from Node B; failure stops owned test processes and leaves this milestone pending. Assign RUN_ID once, record it, and reuse it unchanged on every resume. A_PRIVATE_IP, PORT and LOCAL_GUARD come from the preflight, not guessed values.
  Planned CLI shape (not executable until Tasks 1-3 are implemented):
  ```bash
  SOURCE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask
  OUTPUT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
  python tools/run_h3_ra2va_group_production.py coordinator \
    --source-root "$SOURCE" --output-root "$OUTPUT" --run-id "$RUN_ID" \
    --limit 20 --max-groups 1 --host "$A_PRIVATE_IP" --port "$PORT" \
    --local-lock-path "$LOCAL_GUARD"
  python tools/run_h3_ra2va_group_production.py worker --fake \
    --coordinator-url "http://$A_PRIVATE_IP:$PORT" --node-id "$NODE_ID" \
    --workers 1 --fake-delay-seconds 1
  ```
- [ ] **Step 3: Prove concurrent participation.** Start one actual Worker per node with bounded delay. Observe both active on different clips in the same group/generation before either stage drains; capture overlapping execution timestamps and successful terminal receipts from both hostnames. If launch latency prevents overlap, use test-harness Events/gates, not extra production scheduling controls. Two successful launcher exits alone do not pass.
- [ ] **Step 4: Prove Worker interruption and stale rejection.** Kill only the owned Node B Fake Worker after a confirmed claim and before publication. Node A must preserve that claim until permission expires, then recover unpublished work with a new token. Submit the retained old token through the normal API and record rejection without mutation. Restart Node B only after its old local execution has stopped; retain both interrupted-attempt and accepted-result records.
- [ ] **Step 5: Prove Coordinator restart and barrier.** Kill the owned Coordinator with an unexpired claim, restart only on Node A with the original run-id/guard/endpoint, and show no immediate reassignment. Exercise one result-before-count failure and one live release hold using test-harness synchronization, not shipping crash flags. Verify a dead revoked member eventually stops blocking, while a live unreleased member still blocks. Never kill unrelated server processes.
- [ ] **Step 6: Complete and resume without work.** For an all-success Fake inventory, verify 160 unique clip-stage terminal results (20x8), pending=active=0, zero model calls, actual completions from both nodes, one PILOT_COMPLETE and no full COMPLETE. Verify failed/skipped immutability in the controlled local cases from Task 3; never relabel a real terminal result. Restart completed run and confirm no result bytes change, no task executes, no second group inventory is created, and max_groups remains 1. Interrupted attempts are reported separately from accepted terminal counts.
- [ ] **Step 7: Report actual evidence and commit docs locally.** Preserve timestamps, node/worker/session, generation, task/token, interruption/reclaim, receipt/accounting, release/revoke, per-stage elapsed/throughput and owned-process cleanup outcomes in the new run. Publish a concise measured summary and exact launch/resume/status commands in the runbook. Mark any missing remote proof explicitly; never substitute local simulations or claim GPU readiness. Commit `docs(h3): record two-node HTTP CPU pilot evidence` only after real acceptance; no automatic push or GPU/full-group launch.

## Verification Commands and Handoff

Use an existing Python 3.12 environment; no dependency installation. During implementation, run new suites separately for RED/GREEN, then the named regression files above. These commands are implementation-time verification, not tests performed by this planning turn:

```bash
python -m pytest -q tests/test_h3_ra2va_group_http.py \
  tests/test_h3_ra2va_group_http_worker.py tests/test_h3_ra2va_group_http_recovery.py
python -m pytest -q tests/test_h3_ra2va_group_source.py \
  tests/test_h3_ra2va_group_production.py tests/test_h3_ra2va_group_fake.py \
  tests/test_h3_ra2va_production.py tests/test_v3_post_mask_completed_group_export.py \
  tests/test_h3_mimo26_two_step.py tests/test_h3_mimo25_single_compact.py \
  tests/test_h3_mimo25_two_turn.py tests/test_h3_t2va_full_production.py
python -m ruff check r2v_data_v2/h3/ra2va_group_production.py \
  r2v_data_v2/h3/ra2va_group_http.py r2v_data_v2/h3/ra2va_group_http_worker.py \
  tools/run_h3_ra2va_group_production.py tests/test_h3_ra2va_group_http.py \
  tests/test_h3_ra2va_group_http_worker.py tests/test_h3_ra2va_group_http_recovery.py \
  tests/h3_ra2va_group_http_helpers.py
python -m py_compile r2v_data_v2/h3/ra2va_group_production.py \
  r2v_data_v2/h3/ra2va_group_http.py r2v_data_v2/h3/ra2va_group_http_worker.py \
  tools/run_h3_ra2va_group_production.py
```

Review with:

```bash
git diff --check
git diff --stat a47741d
git diff a47741d -- r2v_data_v2/h3/ra2va_group_production.py
git status --short
```

Tasks 1-3 are implemented locally with TDD and separate commits. Task 4 remains pending until real access is restored. Local verification and its known baseline failures are recorded in docs/H3_RA2VA_GROUP_HTTP_CPU_PILOT.md. Passing CPU scheduling tests never authorizes automatic GPU integration, production rollout, or push.
