# H3 Production Performance V2 Implementation Plan

**Goal:** Keep upstream models resident across fixed 2,000-row shards while
overlapping bounded CPU preparation, without changing frozen semantics.

**Architecture:** A node-owned persistent executor uses private request files
and readiness/completion receipts, never model stdout. Production adapters bind
per-shard context separately from model configuration. Canonical preparation
runs in an owned subprocess with bounded threads and one-shard lookahead.

**Constraints:** No frozen model, prompt, validator, renderer, Visual, or LR-ASD
changes. MiMo remains launcher-owned with request concurrency one. Training rows
remain exactly video/images/audios/caption. Old 10k roots are not migrated.

## Ordered TDD Commits

- [x] 1. Change SHARD_SIZE to 2,000 in t2va_production.py. Test bounds 0/1/59,
  seek boundaries, stable rank schedules, and immutable rejection of 10k roots.
- [x] 2. Add t2va_full_worker_pool.py and focused subprocess tests. Reuse durable
  receipt helpers from the unchanged ephemeral executor. Test three requests,
  load/close once, physical GPU isolation, startup failure, crash/resume, cleanup.
- [x] 3. Bind request configuration in production SAM adapter; connect four
  resident pools through FullPipeline/supervisor. Reject model identity changes.
  Run all stem/speech adapter regressions.
- [x] 4. Parallelize bootstrap_audio with bounded ThreadPoolExecutor; preserve
  ordered canonical inventory and per-clip frozen preparation/hash validation.
  Add canonical-workers CLI/env default 16 and concurrency/failure tests.
- [x] 5. Introduce owned canonical subprocess lookahead, canonical.lock and
  per-shard prepared state. Test overlap, GPU barriers, one lookahead, cancellation.
- [x] 6. Add ASR worker depth-one waveform preparation only. Test next-job
  overlap, serial transcribe, and delayed attribution of loader errors.
- [x] 7. Update production documentation and run integrated local regressions,
  Ruff, Python 3.12 compilation, shell syntax, and diff checks. Report server
  raw5/100/2k/two-shard persistence validation as pending, not executed.

Each step starts with failing tests, implements only orchestration/adapters,
inspects its diff, and commits independently. Existing ephemeral workers remain
available. No real model inference is part of local validation.

## Validation Outcome

- Combined production/T2VA/TA2VA/resolved/AuK regression: 467 passed, 3 skipped
  (optional local Playwright).
- Ruff, Python 3.12 compilation, shell syntax, and diff checks passed.
- Review added writer-lock exclusion and refresh of durable canonical counts
  after invocation ownership; both have regression tests.
- Frozen semantic modules, Visual, server_env.sh, and the existing ephemeral
  executor have no changes.
- Real server raw5/100/2k/two-shard PID acceptance remains pending.
