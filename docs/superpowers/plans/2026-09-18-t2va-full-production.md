# Full Target-Side Production

Starting branch: feature/h3-audio-jea-qwen3-v1.
Starting HEAD: d915d6b65fc7eb782a5ca2e31d0fb7848b0c8b75 (latest Visual advances retained).
The supplied Task 9 specification is the architecture: one node owns one fixed
10k shard at a time; MiMo persists across all stage barriers and shards.

## Implementation Sequence

- [x] 9.1 Preserve proxies but bypass localhost after server_env.sh; shell test.
- [x] 9.2 Seek existing source shard index, bootstrap canonical audio through
  prepare_t2va_audio, resume successful clips and isolate media failures.
- [x] 9.3 Spawn isolated per-GPU workers, durable per-job receipts, barriers,
  model-once lifecycle, interruption cleanup and worker-crash handling.
- [x] 9.4 SAM full-inventory adapter, music_first, private worker outputs and
  frozen record construction; one canonical ordered publication.
- [x] 9.5 AuK cached generation, frozen canonicalization and fixed stem resolve.
- [x] 9.6 DiariZen cached backend results, frozen full-shard publication.
- [x] 9.7 Qwen3-ASR cached segment results, frozen crop/publication semantics.
- [x] 9.8 Full supervisor, persistent MiMo launcher, existing downstream API.
- [x] 9.9 CPU regressions, docs, raw-video smoke and resume commands.
- [ ] Server acceptance: fresh raw-five, interrupted ASR resume, bounded 100,
  then one real 10k shard. Not run on the development Mac.

## Boundaries

No frozen SAM/AuK/DiariZen/ASR/T2VA/TA2VA semantic files or Visual files change.
No server-local environment file is edited. Child CUDA_VISIBLE_DEVICES maps
physical GPU to local cuda:0; parent MiMo environment is unchanged. No real
model calls are made on the development Mac.

The frozen target-side fixture uses a JEA bootstrap DiarizationInventory with
no reference or binding sidecar and the default stem loader, not the RA2VA
binding_evidence_mode=none builder which requires Visual provenance. Reuse
that verified target-side path exactly.

## Verification

Tests first for each adapter. Full-shard inventories are canonical, never
constructed by joining fingerprints of subset inventories. Ready inference
receipts survive stage re-publication; failed jobs get one attempt per invocation.
Infrastructure death stops the stage after other workers finish, leaving
unattempted jobs pending. Canonical publication runs only after a worker barrier.

Run new tests and existing production/snapshot/Audio regressions, Ruff, Python
3.12 compilation, shell syntax, diff-check. Real raw-five and 100-clip server
smokes are deferred to the server operator and must not be claimed as executed.

## Local Verification Result

Python 3.12, CPU FFmpeg, fake model backends only:

- New full-production tests plus existing production, snapshot, T2VA, TA2VA,
  AuK and resolved-stem regressions: 363 passed, 2 skipped (optional Playwright).
- Existing SAM shadow suite: 104 passed, 15 failed. A clean archive of starting
  HEAD reproduces exactly the same 15 failures: stale MiMo version/output-root
  assertions and old StemAwareOpenAIMimo25Backend interfaces. No frozen test or
  implementation was changed to mask them.
- Ruff, Python 3.12 py_compile, bash -n and git diff --check passed.
- Full shell --dry-run printed the accepted serving flags and worker defaults;
  it did not source server_env.sh, write production data, or start a model.
- Fake end-to-end runs cover one/all SAM failures, recovery, immutable ready
  downstream artifacts, zero repeated ready inference and incremental snapshots.
- A 10k-job CPU subprocess test covers all eight GPU partitions, stable order,
  child-only CUDA mapping and one load/close per worker.

Frozen semantic modules and Visual implementation changes: zero. The only
pre-existing code edit is the two-line standalone launcher proxy bypass, with
its existing shell test updated. Existing t2va_production.py is unchanged.

Resume assumes pinned dependency runtimes. Model/config/media hashes are checked;
an in-place interpreter or package upgrade is not a supported cache migration.
