# RA2VA Group CPU Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement and test shared-group CPU Fake Worker scheduling and durable recovery using the first 20 real V3 production rows, without models or media processing.

**Architecture:** A source adapter imports only published V3 rows into an output-owned inventory. A file-backed coordinator uses short control/dispatch locks, nonblocking per-clip flock claims, atomic results, and generation-bound stage barriers. An explicit fake-only CLI exercises eight stages without producing real Audio/H3 or training artifacts.

**Tech Stack:** Python 3.12, existing Pydantic ProductionSample, multiprocessing, fcntl.flock, existing atomic_json, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-09-ra2va-two-step-group-production-design.md`, approved by the user; this plan implements only its first CPU milestone and the user's four subsequent clarifications.

## Global Constraints

- Branch: `feature/h3-audio-jea-qwen3-v1`; remote baseline verified as `40c34d6c9992af33f92ad06e5ff56037fb9658d1`; approved design commit: `5197fb9`.
- Input: `/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask/group-*/samples.jsonl`.
- Future shared output: `/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/<run-id>/`; resume preserves the run ID and inventory.
- This phase uses local temporary outputs and read-only SSH input access. Do not write the server, run GPU/MiMo, or launch a full group.
- Preserve source order, image_id, Image-to-Picture numbering, entity and attribute ownership, and existing official reference symlinks.
- No media reads/scans for validation, media hashes, fake hash fields, seed shuffle, rank/world_size partition, manual QA, or changes to Multi/Single/T2VA.
- Fake stages: `canonical`, `sam`, `auk`, `resolve`, `diarizen`, `asr`, `mimo`, `export`. Fake success is not a real reconcile or training export.
- Terminal ready/failed/skipped is immutable; restart retries only interrupted, unpublished work. A failed/skipped upstream clip becomes skipped in later fake stages.
- Runtime GPU stages, two-turn checkpoints, training Bundle integration, and real two-node locking are explicitly deferred.

## Review Focus

- Upstream staging directory or incomplete Export marker: never import, even when a JSONL is visible.
- Worker killed during claim or result accounting: recover the same task exactly once without skipping cursor positions.
- Previous-generation worker joins/publishes/acknowledges late: cannot change the new stage or advance its barrier.
- Restart after source disappears: consume the frozen inventory and existing results without rereading source media.
- Pilot and dynamic group discovery: new groups are discoverable, but a limited inventory never marks the source group fully complete.

## Decisions Before Implementation

### Source Publication

The current `tools/export_completed_postmask_group.py:convert_group()` verifies formal group/shard Export markers, writes the complete JSONL and reference aliases under a hidden temporary directory, flushes/fsyncs the JSONL, then atomically renames the directory to `group-NNNNNN`. It never appends to or overwrites a published group. The directory rename is the final publication event for this known producer, not arbitrary directory existence.

Discovery accepts only exact `group-[0-9]{6}` names with `samples.jsonl` and the corresponding `source_root.parent/state/resource_epochs/<group>/completed.json` declaring matching `group_id`, `status=completed`, `export_completed=true`, and a nonnegative sample_count. It does not repeat the producer's per-shard validation. Hidden staging directories, missing/incomplete markers, and missing JSONL are not candidates. A future append-in-place producer requires an explicit final publication signal before it can be supported; a size-stability heuristic is not a substitute.

Read-only SSH verified the real `group-000000` marker declares 39,585 samples. CPU pilot import reads only its first 20 nonblank rows, not all 39,585 and not any PNG/video bytes. Missing or invalid JSON rows abort inventory publication with a source error; no partial inventory becomes claimable.

### Claim Transaction and Recovery

Lock order is control -> dispatch -> nonblocking clip-stage lock. Processing holds only its clip-stage lock; it never holds control/dispatch locks during work. Clip lock files remain at stable paths and are never unlinked/replaced, avoiding multiple lock inodes for one task.

Under the short dispatch lock, acquire the clip-stage lock with LOCK_EX|LOCK_NB, then atomically replace one dispatch JSON containing cursor and active claims together. A crash before replacement leaves the cursor unchanged; a crash after it leaves a recoverable active claim. Recover active entries before taking new cursor work. A busy clip lock is never waited on or reclaimed by elapsed time.

An atomic per-clip result is the publication point. Under dispatch lock, remove the active entry and increment terminal counts in one atomic update. Recovery first consumes a matching existing result and accounts for it once; only an unpublished entry whose clip lock is available may execute again. This includes crashes after result publication but before count commit. Results for other generations cannot satisfy the current task. Recovery inspects bounded active metadata/results, not all completed media or result directories.

### Generation and Barrier

Run control holds a monotonically increasing generation, group_id, stage, and phase (`running` or `draining`). Resume keeps the same generation; advancing to the next stage/group increments it. Claim/result/session/ack records carry the same generation. All mutations compare it under control/dispatch locking and reject stale participants without altering current state.

Each worker session owns an independent session flock. After every task is terminal and active claims are accounted, control enters draining and refuses new worker membership. Live members close their Fake Worker context before recording a generation-bound release acknowledgment. The stage advances only when all live members acknowledged; a released session lock proves a dead/exited member no longer holds resources. An idle live worker that has not acknowledged still blocks advancement. No heartbeat timeout or fixed world size is introduced.

Initialize the next generation's empty dispatch state before atomically advancing control; control is the authority if a crash leaves an unreferenced dispatch directory. Repeating initialization is idempotent and never clears existing results. Publish PILOT_COMPLETE before advancing to another group, and recover that boundary without restarting the finished fake export stage.

### Adapter Boundary and Pilot Completion

GroupTask is a dedicated CPU/source identity type, not MimoClipJob or a legacy H3 contract. It stores source ordinal/sample/clip identity, original ProductionSample, target path, references with absolute alias paths and Picture labels, and Subject ownership metadata without media hashes.

Use the existing pure `entity_reference_order()` semantics: entity Subjects in first reference appearance, followed by attribute Subjects in reference order and one background Subject when present. Preserve all source fields. Join image_path to the published group directory without resolving it for containment checks or stat'ing every image. Tests may read fixture symlinks; production import does not open PNGs.

Phase-one CLI requires a bounded `--limit` (1-200) and `--fake`; it cannot run a full group. Persist `mode=pilot`, selected_count, declared_source_count, source root/group, and limit in inventory metadata. Even when selected_count equals a tiny fixture's total, fake runs publish only PILOT_COMPLETE, never COMPLETE. A new group gets a separate pilot inventory; it does not extend an existing group's frozen inventory.

## Files and Interfaces

- Create `r2v_data_v2/h3/ra2va_group_source.py`: publication discovery, pure V3 identity adapter, atomic bounded inventory import, indexed task reads.
- Create `r2v_data_v2/h3/ra2va_group_production.py`: stage control/generations, claims, atomic terminal publication/recovery, membership/barriers, compact progress.
- Create `r2v_data_v2/h3/ra2va_group_fake.py`: minimal context-managed Fake Worker and local multiprocessing launcher; no model imports.
- Create `tools/run_h3_ra2va_group_production.py`: `run`/`status`, fake-only pilot options, source/output/run identity, worker count and discovery polling.
- Create `tests/test_h3_ra2va_group_source.py`, `tests/test_h3_ra2va_group_production.py`, `tests/test_h3_ra2va_group_fake.py`.
- Create `docs/H3_RA2VA_GROUP_CPU_PILOT.md`: only implemented CPU launch/resume/status commands and honest evidence boundaries.
- Do not modify legacy inventory, shared validator, prompts, hashes, Audio/H3 materializers, or donor exporter.

### Task 1: Published V3 Inventory and Identity Adapter

**Files:** source module and its focused tests.

**Interfaces:**
- `PublishedGroup(group_id: str, directory: Path, samples_path: Path, declared_count: int)`.
- `GroupTask(ordinal: int, clip_uid: str, sample_id: str, sample: dict, target_video_path: str, pictures: list[dict], subjects: list[dict])`.
- `discover_published_groups(source_root: Path) -> list[PublishedGroup]`: numeric order and formal publication conditions above.
- `adapt_production_sample(sample: ProductionSample, group_directory: Path, ordinal: int) -> GroupTask`: pure path/ownership mapping, no file verification.
- `freeze_pilot_inventory(group: PublishedGroup, destination: Path, limit: int) -> dict`: atomic directory publication of metadata, ordered tasks JSONL, and row offsets; reopen existing inventory without source reads.
- `read_inventory_task(inventory_root: Path, ordinal: int) -> GroupTask`: seek one frozen row using the saved offset.

- [x] Write tests `test_only_formally_published_groups_are_discovered`, `test_staging_and_incomplete_marker_are_not_imported`, and `test_inventory_preserves_order_and_ownership`: assert exact source order, image_1 -> Picture 1, entity ownership, attributes, and background mapping.
- [x] Add `test_official_reference_symlink_is_read_only` with a reference symlink outside the group and unchanged source bytes; `test_resume_uses_inventory_without_source` removes the fixture source after import; `test_pilot_never_claims_full_group` compares selected_count=20 and declared_count=39585.
- [x] Run `pytest -q tests/test_h3_ra2va_group_source.py`; verify missing new interfaces cause failure before implementation.
- [x] Implement only the interfaces above using ProductionSample and the pure entity-order helper; publish inventory once and reject duplicate clip identities during import, not during resume.
- [x] Rerun the source tests and commit the source adapter/tests only.

### Task 2: Atomic Claims and Terminal Recovery

**Files:** production module and focused production tests.

**Interfaces:**
- `StageSnapshot(group_id: str, stage: str, generation: int, phase: str)`; `TaskClaim(snapshot: StageSnapshot, task: GroupTask, worker_id: str, lock_handle: object)` owns the clip lock until close/context exit.
- `GroupCoordinator(run_root: Path, source_root: Path, limit: int)`; persist run settings and reject a conflicting resume limit/source rather than rebuilding inventory.
- `select_current() -> StageSnapshot | None`: recover started groups first, otherwise discover the next numbered unprocessed published group.
- `claim(snapshot: StageSnapshot, worker_id: str) -> TaskClaim | None`: nonblocking clip lock and one cursor/active update.
- `publish(claim: TaskClaim, status: str, payload: dict, elapsed_seconds: float) -> None`: only ready/failed/skipped, atomic result before accounting, no terminal overwrite.
- `recover(snapshot: StageSnapshot) -> None`; `progress(snapshot: StageSnapshot) -> dict`: active-entry repair and pending/active/ready/failed/skipped plus elapsed/throughput.

- [x] Write `test_multiprocess_claim_and_publish_once` with four spawned processes and 20 tasks; assert 20 unique terminal results, terminal sum=20, pending=active=0, and no duplicate successful publication. Spawn prevents accidental inherited flock descriptors in the test harness.
- [x] Write event-synchronized crash tests, not sleep races: `test_crash_before_dispatch_replace`, `test_crash_after_claim`, `test_crash_after_result_before_accounting`, and `test_interrupted_task_is_reclaimed`. Kill the child at each point and assert recovery loses no cursor task or counter and executes no published result again.
- [x] Add `test_terminal_failed_and_skipped_are_not_retried` and `test_live_clip_lock_is_not_reclaimed`; run the production tests to confirm they fail before implementation.
- [x] Implement claim/recover/publish with existing atomic_json and stable independent flock files, following the transaction and lock order above; keep optional fault synchronization in tests, not production CLI switches.
- [x] Rerun focused source/production tests; commit the coordinator claim/recovery deliverable.

### Task 3: Generation-bound Membership and Stage Barriers

**Files:** production module and its focused tests.

**Interfaces:**
- `join(snapshot: StageSnapshot, worker_id: str) -> WorkerSession`, holding a generation-bound session flock; WorkerSession is a context manager and releases its lock on exit.
- `acknowledge_release(session: WorkerSession) -> None`: permitted only after the caller closes its Fake Worker/backend context.
- `advance(snapshot: StageSnapshot) -> bool`: enter draining after terminal accounting, then advance only after live releases; increment generation and initialize next stage atomically through control.
- `finish_pilot(snapshot: StageSnapshot) -> None`: after fake export barrier, atomic PILOT_COMPLETE and selection of another discovered group; never full COMPLETE.

- [x] Add `test_barrier_waits_for_live_worker_release`, `test_dead_session_does_not_block_barrier`, `test_late_join_is_rejected_while_draining`, and `test_stale_generation_cannot_claim_publish_or_ack`; assert no advancement before release and no stale mutation after advancement.
- [x] Add `test_started_group_wins_over_lower_new_group`, `test_new_group_discovered_while_running`, and `test_stage_restart_keeps_generation`; use marker plus atomic group publication fixtures from Task 1.
- [x] Add `test_crash_during_generation_advance` and `test_resume_after_pilot_completion_before_next_group`: the authoritative generation is resumed, terminal results remain intact, and no finished stage is repeated.
- [x] Run the production tests for a red baseline, implement the interfaces, then rerun Tasks 1-3 tests.
- [x] Commit the stage-generation/barrier deliverable without altering existing production entry points.

### Task 4: Fake-only CLI, Real-input Pilot, and Evidence

**Files:** fake module, CLI, fake tests, CPU pilot runbook.

**Interfaces:**
- `FakeWorker(stage: str)` context manager; `process(task: GroupTask) -> dict`: deterministic source identity payload and zero model calls. A test-only outcome policy may produce terminal failed/skipped; fake output is explicitly labeled and cannot be read as a real reconcile.
- `run_fake_worker(coordinator: GroupCoordinator, worker_id: str, stop_event: object) -> None`: join generation, claim/process/publish dynamically, close worker context, acknowledge barrier, and follow subsequent stages.
- `run_fake_pilot(*, source_root: Path, run_root: Path, workers: int, limit: int, max_groups: int, poll_seconds: float) -> dict`: launch local processes without static partitions; default max_groups=1 (persisted global cumulative run budget), handle SIGINT/SIGTERM with child cleanup and keep durable unfinished state.
- CLI: `run --fake --source-root PATH --output-root PATH --run-id NAME --limit 20 --workers 4 --max-groups 1`; `status --output-root PATH --run-id NAME`. Optional bounded discovery uses max_groups>1. Resume repeats the exact run identity/settings.

- [x] Test all eight fake stages, automatic upstream skip propagation, no model imports/calls, run-ID resume, no full COMPLETE, worker interruption, and two local launchers sharing one run directory. The latter is not a real two-node filesystem test.
- [x] Run the fake tests before implementation; implement the thin runner/CLI. Use the existing narrow output policy or a group-specific check so only the authorized R2VA public subtree is writable; never broaden global public-dataset permission. Local test temporary roots remain allowed.
- [x] Retrieve only the marker and first 20 real group-000000 rows over read-only SSH into a fresh local scratch source snapshot. For offline import, explicitly construct PublishedGroup with the real remote group directory and the local snapshot samples_path, then freeze that inventory before starting the local coordinator. This preserves actual video/reference-alias paths without requiring the remote filesystem locally. Exercise fresh CLI discovery separately with local published-group fixtures. Do not download media or commit private rows; record 20 clips, source order, and ownership/Picture counts.
- [x] Run four local Fake Workers across all eight stages, then a separate event-controlled interrupted run and resume; retain local logs showing claims, recoveries, generations, result-first accounting, stage release, counts, elapsed seconds, and throughput. Expect 160 fake ready task results for the all-success 20x8 case, zero model calls, and only PILOT_COMPLETE. Report measured results, not this expectation as evidence.
- [x] Run `pytest -q tests/test_h3_ra2va_group_source.py tests/test_h3_ra2va_group_production.py tests/test_h3_ra2va_group_fake.py tests/test_v3_post_mask_completed_group_export.py tests/test_h3_ra2va_production.py`; report unrelated environmental failures separately.
- [x] Run Ruff on new Python modules/tests, py_compile on new runtime/CLI modules, and `git diff --check`; inspect the actual tracked diff and confirm preexisting `docs/visualizations/` remains untouched.
- [x] Document exact implemented launch/resume/status commands and measured local pilot/recovery results. Explicitly state no real Audio/H3 export, two-node lock test, GPU quality test, or full group was performed. Commit the tested deliverable; normal push only when authorized, never force-push.

## Plan Handoff

Recommended execution method: **Native**, implementing these tightly coupled source/coordinator/worker interfaces in this session with focused TDD and final review. The written plan requires user review before product-code implementation; the already-approved design does not need another design review.
