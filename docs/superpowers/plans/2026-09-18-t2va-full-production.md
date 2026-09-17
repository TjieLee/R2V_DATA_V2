# Full Target-Side Production

Starting branch: feature/h3-audio-jea-qwen3-v1. Starting HEAD: d915d6b.
The supplied Task 9 specification is the architecture: one node owns one fixed
10k shard at a time; MiMo persists across all stage barriers and shards.

## Implementation Sequence

- [x] 9.1 Preserve proxies but bypass localhost after server_env.sh; shell test.
- [x] 9.2 Seek existing source shard index, bootstrap canonical audio through
  prepare_t2va_audio, resume successful clips and isolate media failures.
- [ ] 9.3 Spawn isolated per-GPU workers, durable per-job receipts, barriers,
  model-once lifecycle, interruption cleanup and worker-crash handling.
- [ ] 9.4 SAM full-inventory adapter, music_first, private worker outputs and
  frozen record construction; one canonical ordered publication.
- [ ] 9.5 AuK cached generation, frozen canonicalization and fixed stem resolve.
- [ ] 9.6 DiariZen cached backend results, frozen full-shard publication.
- [ ] 9.7 Qwen3-ASR cached segment results, frozen crop/publication semantics.
- [ ] 9.8 Full supervisor, persistent MiMo launcher, existing downstream API.
- [ ] 9.9 CPU regressions, docs, raw-video smoke and resume commands.

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
