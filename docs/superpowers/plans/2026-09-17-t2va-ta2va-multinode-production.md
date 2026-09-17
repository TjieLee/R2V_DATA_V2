# T2VA / TA2VA Multinode Production Implementation Plan

> **For agentic workers:** Use executing-plans and test-driven-development task by task.

**Goal:** Publish frozen target-side products incrementally from fixed 10,000-row
JEA shards, with independent stage resume and immutable training snapshots.

**Architecture:** New orchestration only. Immutable source shards feed existing
T2VA inventory/semantic APIs and a production-local TA2VA adapter calling the
existing ownership, PCM, profile and renderer functions. A shard lock and
append-only journal serialize publication; complete per-stage directories are
the recovery authority if a process dies before its journal append.

**Tech stack:** Python 3.12, pathlib/json, fcntl, ThreadPoolExecutor, existing
Pydantic Audio/H3 contracts, Bash and the accepted SGLang command.

**Spec:** User's 2026-09-17 production requirements in this task.

## Global Constraints

- Starting HEAD: 7505cce5cdf2eca7153b6163c8175f15d24d1b0f.
- Never edit frozen t2va_shadow.py, t2va_mimo_backend.py, ta2va_shadow.py or
  their existing runners. Never edit Visual.
- Frozen versions: T2VA prompt v7/backend .7/core .6; TA2VA .2/profile v2;
  audio finalizer v6. No new semantic gates, prompts or retries.
- Default output is the explicitly authorized public dataset T2VA directory.
  Require writable preflight; no fallback root.
- No Visual input flags. Audio preprocessing is an existing downstream cache;
  missing upstream data is skipped using the existing upstream_failure policy.
- Local implementation/CPU validation only in this environment. Real server
  smoke is a separate acceptance step, not a simulated success.
- Do not touch untracked docs/visualizations/.

## Files and Interfaces

- t2va_production.py: index/schedule, journal/lock/atomic publication, stage
  adapters, incremental exports and snapshot assembly.
- run_h3_t2va_production.py: source/config CLI and shard invocation.
- build_h3_t2va_snapshot.py: immutable snapshot CLI.
- scripts/run_h3_t2va_production.sh: owned MiMo lifecycle and dry-run.
- test_h3_t2va_production.py and test_h3_t2va_snapshot.py: CPU fixtures.
- H3_T2VA_PRODUCTION.md: operation, resume, snapshot and smoke instructions.

## Task 1: Fixed Source Shards

- [ ] Add failing tests for shard 59 bounds, deterministic shuffle/rank union,
  explicit schedule, direct byte seeking and source identity changes.
- [ ] Implement shard_name(id), shard_order(...), build_source_index(source, root)
  and materialize_shard(index, id, root). Index raw JSONL rows, including malformed
  rows, so eligibility cannot move shard boundaries. Store only one offset/10k.
- [ ] Publish index and raw shard manifests with temporary file + rename under a
  short index lock. Hash source while building; verify stat identity on reuse.
  Never rebuild an existing production population silently on source change.
- [ ] Run new tests and commit Task 1.

## Task 2: Stage Resume

- [ ] Red tests: A ready/B fails/C ready; TA-only retry; interrupt after three;
  artifact rename before state append; locked shard; endpoint outage.
- [ ] Implement process_shard(rows, processor, ...) with a bounded worker pool,
  serialized journal writer, one attempt per stage per invocation, no persisted
  running state. Processor writes under a stage temporary directory and returns
  a result; publication writes a completion envelope before rename.
- [ ] On resume load last newline-terminated state, recover complete stage
  envelopes, and repair incomplete append tails before further appends.
- [ ] EndpointUnavailable stops new scheduling; active work may finish and
  publish. KeyboardInterrupt never becomes a sample failure.
- [ ] Run tests and commit Task 2.

## Task 3: Frozen T2VA Adapter

- [ ] Add shadow-vs-production test with the same deterministic fake backend.
- [ ] Implement T2VA stage using request_fingerprint, annotate,
  parse_t2va_semantic, frozen audio draft parser, validate_t2va_draft and
  render_t2va_prompt. Keep raw diagnostics and validated core.
- [ ] Use existing build_t2va_inventory on owned shard manifests; keep original
  source index/hash in the production manifest. Isolate inventory errors to
  individual source rows if bulk preparation fails.
- [ ] Bind resume identity to job/backend/source provenance; no old shadow-root
  redirection or monkeypatching production globals.
- [ ] Run focused equivalence tests and commit Task 3.

## Task 4: Frozen TA2VA Adapter

- [ ] Add old-vs-new variant/prompt/PCM tests and TA-only retry coverage.
- [ ] Load matching resolved stems/raw segments read-only; validate frozen T2VA
  facts against raw/ASR data. Call existing probe/_check_stem/sample_ranges/
  select_speakers/speech_track/profile/render_product and TA2VAProduct schema.
- [ ] Keep full audio eligible with no speech; retain exclusion provenance.
  Write asset paths for their final atomic directory, not temporary paths.
- [ ] Treat profile failure as TA failure, retain T2VA, do not rerun successful
  T2VA. Never introduce alternative eligibility.
- [ ] Run reuse/equivalence regressions and commit Task 4.

## Task 5: Incremental Exports

- [ ] Red tests assert immediate publication and exactly four training keys.
- [ ] Append each ready stage's task rows with flush/fsync. Production manifest
  maps video to clip_uid for dedup without adding internal keys to training rows.
  Replay on recovery may append identical rows; snapshots deduplicate.
- [ ] Only terminal ready/skipped shards receive COMPLETE and atomic export
  rename; failed stages remain resumable and snapshot-visible.
- [ ] Run tests and commit Task 5.

## Task 6: Immutable Snapshots

- [ ] Red tests: 327 rows in incomplete shard, truncated tail, duplicate
  equality/conflict, later TA completion, immutable existing snapshot.
- [ ] Merge complete lines from final/partial task exports in stable shard and
  source order; key by clip_uid/task using manifest identity. Reject conflicts.
  Regenerate videos.jsonl tasks, write summary, atomically publish directory,
  then atomically update plain-text LATEST. No database.
- [ ] Add snapshot CLI, run tests and commit Task 6.

## Task 7: Launcher and Documentation

- [ ] Add CLI/dry-run tests and shell syntax/lifecycle tests with fake binaries.
- [ ] Implement CLI inputs, default eight workers, progress, writable preflight,
  no Visual flags. Bash sources existing environments, unsets PYTHONPATH,
  starts exact accepted MiMo command, waits for readiness, logs per hostname,
  kills only owned PID on exit/signals. Dry-run starts no model.
- [ ] Document fresh roots, fixed shards, upstream prerequisites, retries,
  snapshots, default skipped behavior and missing target-side T2V contract.
- [ ] Run focused/new/related tests, Ruff, Python 3.12 compile and diff-check.
  Commit Task 7.

## Task 8: Server Acceptance (Not Executable on This Mac)

- [ ] On an authorized server, use frozen random20 upstream with 2-5 eligible
  source rows; compare core/prompt/variants with frozen reference products.
- [ ] Confirm no drift, then run a bounded small source fixture shard. Do not
  start the full 3.8M population.
- [ ] Report real outcomes only from real logs. Current task explicitly
  continues locally without accessing server paths.

## Final Verification

Run pytest for both new files plus existing T2VA/TA2VA/reuse regressions.
Run Ruff, Python 3.12 py_compile, bash -n and git diff --check.
Provide starting-head...HEAD full diff and stat, commit list, new/modified files,
and explicitly verify zero frozen semantic/Visual modifications.
