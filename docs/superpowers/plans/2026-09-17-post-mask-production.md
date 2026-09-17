# Post-Mask Visual production implementation plan

> Execute with subagent-driven-development, test-first, in the isolated Post-Mask worktree.

## Authority and architecture

The user-approved specification is the complete task attachment at
`/Users/litengjie.3/.codex/attachments/6dc462c2-68ac-4290-8b90-8528b9699325/pasted-text.txt`.
The shipped operator contract will be `docs/V3_POST_MASK_PRODUCTION.md`.
Baseline: `49c191c3a64a511f794ca0511d8b8371f1d32a88` on
`feature/h3-audio-jea-qwen3-v1`.

Use a thin, separate orchestration layer. A frozen Stage2 shard hydrates to one
normal writable V3 RunStorage, then runs existing downstream APIs by phase.
Physical placement is execution state, never durable semantic identity.
Whole-shard assignment and shared flock protect writable destinations.

## Global Constraints

- Stage order: hydrate -> remove -> pair -> reference_edit -> reference_integrity -> instruct -> subject_attributes -> export -> compact.
- Never regenerate Stage2 input or invoke its production/recovery launchers. Corrupt ready input is excluded model-free per clip.
- Do not modify frozen frames.py, segment.py, sam3_backend.py, sam3_anchor_selector.py, rank.py, background.py, mask_codec.py, Subject Attribute policy, or r2v_data_v2/h3/*.
- No real GPU/model/production execution; tests use temporary fixtures and fakes.
- Frozen entity_mask input is strictly read-only. Mutable hydrated state must not share writable inodes with input.
- New server writes stay under /mnt/workspace/litengjie/data. No overwrite, reset, clean, rebase or force push.
- Existing V3 config/model/source identity validation remains strict. Preserve historical run git_commit on restart.
- Reuse DatasetExporter and existing production compactor; public schema remains r2v.v3.production_sample.1.
- No distributed queue/work stealing, broad canonical audit, per-JPEG hashing, or new public schema.

### Task 1: Model-free hydration and stable shard identity

Files: new `r2v_data_v2/v3/post_mask_production.py`, `tests/test_v3_post_mask_production.py`.
Read current RunStorage, ClipRecord, frames/masks manifests, Stage2 row/path layout and tests first. Do not import Stage2 production/recovery modules in the new runtime path.

Provide small documented interfaces for Task 2/3: canonical shard enumeration; a paths object giving stable run/export/state paths per shard; semantic config preparation/initialization; `hydrate_shard(...)` returning eligible clip IDs and ready/excluded/corrupt counts; atomic JSON helper/provenance records where needed. Names may follow repository conventions; report exact signatures.

Implement ready_* eligibility and deterministic exclusion/input-failure JSONL preserving source_index, clip_uid, status/reason, canonical shard/artifact root. Parse minimum Stage2 row projection without a competing public schema. Validate safe paths, target identity, required parseable frames/masks/clip/background manifests and referenced file existence; no JPEG hash/PIL audit. Preserve source, annotation, frames, masks, coverage/background exactly, including run-relative paths after hydration. Ensure downstream readable candidate/reference assets are included. Corrupt clips are isolated; duplicate/unsafe row identity cannot collide with another destination.

Initialize normal RunStorage with a stable Post-Mask semantic config (run/export per shard, no physical topology/device values). Preserve source model settings and Visual policy. Existing run uses historical git_commit but strict config/model/source checks. Additional subject-attribute identity can be bound by a lightweight config sidecar because normal fingerprint excludes it. Use copy for mutable state; link immutable bytes only if downstream cannot mutate them. Atomic per-clip staging/provenance makes complete hydration idempotent and partial hydration safely rebuildable without overwriting completed downstream work.

Tests first: ready hydrate facts/assets; each non-ready class excluded; missing/corrupt files model-free isolation; source bytes unchanged after destination state mutation; safe-path rejection; idempotence and partial hydration; physical remap unchanged identity; historical commit restart; config/source mismatch fail closed. Run new focused tests and Ruff. Commit only Task 1 code/tests and this plan if still uncommitted. No push.

### Task 2: Existing downstream phase adapter and resumable shard execution

Files: new `r2v_data_v2/v3/post_mask_runtime.py`, extend production module only for orchestrator integration as needed, new `tests/test_v3_post_mask_runtime.py`.
Consume Task 1 APIs. Read run_pipeline_v3 stage dispatch, persistent Boogu/SAM adapters, stage durable skip conditions, DatasetExporter, compactor and canary config normalization.

Implement real stage adapter with lazy constructors, no stub production path. Per shard remove owns one persistent Boogu backend then closes; pair bounded Qwen; reference_edit persistent Boogu on boogu_gpu and unchanged SAM reviewer on sam_gpu; integrity/instruct bounded Qwen; attrs persistent single-frame SAM on sam_gpu and optional persistent Boogu completion on boogu_gpu. Close on exceptions too, no per-request model reload. Existing algorithms/prompts/thresholds unchanged. Use process isolation/device override local to Post-Mask; not normal V3 identity changes. Shared per-node Qwen gate capacity across local group processes (node identity only in execution lock path).

Run only missing durable phase work with overwrite=False; clip-scoped calls must retain original full shard visibility where pairing requires it. Isolate clip failures, keep diagnostics, do not publish a complete shard while retryable work remains. Terminal quality rejection can be excluded with existing semantics. Hydration errors never enter models. Subject attributes use existing owner artifacts and run-local sidecar semantics. DatasetExporter export once, resume an already published export with provenance/identity checks. Completed marker only after export succeeds.

Clean only owned proven scratch/staging without deleting generated outputs/resume references; inspect current Boogu consumers first. Emit stage/hydrate/clip failure/shard events and counters, with optional event callback for launcher. Avoid large parallelism or checkpoint frameworks.

Tests first with fakes for every phase: phase order; each completed phase skipped; clip failure isolation; persistent backend once/phase and close; correct sam/boogu physical placement; no Stage2 generation/import; restart export idempotence; cleanup preserves resume/final assets; real DatasetExporter fixture through existing compactor and existing H3 compacted production reader. Run focused tests, existing V3 runtime and H3 reader tests, Ruff. Commit Task 2 only. No push.

### Task 3: Cluster launcher, worker, one-click script and runbook

Files: new tools/run_v3_post_mask_visual.py, tools/run_v3_post_mask_worker.py, scripts/run_v3_post_mask_visual_cluster.sh, tests/test_v3_post_mask_cluster.py, docs/V3_POST_MASK_PRODUCTION.md. Extend new Post-Mask modules only as needed.

Use all sorted parts/shard-*.jsonl. Deterministic balanced whole-shard assignment with worker_id=rank*local_group_count+slot, count=world_size*local_group_count. Parse default groups 4:5,6:7 (SAM:Boogu), validate non-overlap and rank/world. Shard roots unchanged across topology restarts; shared-storage per-shard flock held for mutation, completed shards skipped. Worker subprocess uses repo Python and main GPU isolation; configured Boogu subprocess interpreter unchanged. Reuse Task 2 shard runner.

Require explicit accepted base config override if no repository-authoritative current production YAML exists; document old canary server default as a convention, not silently configs/default.yaml. Ensure accepted production remove backend/profile and disabled GME conventions without changing prompts/gates; make any fixed normalization explicit and tested. Expose POST_MASK_BASE_CONFIG, POST_MASK_ROOT/tag, POST_MASK_GPU_GROUPS, POST_MASK_QWEN_MAX_INFLIGHT. Durable semantic identity excludes physical GPU/rank/world and per-node gate directories.

One-click bash cd /mnt/workspace/litengjie/data/R2V_DATA_V2; .venv/bin/python; PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3:${PYTHONPATH:-}; OMP_NUM_THREADS=1; injected RANK/WORLD_SIZE default 0/1. No `set -o pipefail`, no `exit 1`, no dreamidv. Full sequence automatic. Rank0 waits for every expected marker then calls existing compactor with proper source descriptor/run roots. Other ranks wait for durable compact marker before success. Failure paths return nonzero, terminate only own children, keep progress. Compact only once under shared lock, bind marker to campaign/shard identities, never treat partial result as success.

Events: post_mask_worker_started, post_mask_shard_started, post_mask_hydrate_completed, post_mask_stage_completed, post_mask_clip_failed, post_mask_shard_completed, post_mask_worker_completed, post_mask_waiting_for_global_completion, post_mask_compaction_started, post_mask_compaction_completed, post_mask_global_completed. Progress/heartbeat ~30-60 seconds, no per-row heartbeat spam. Include totals/assigned/completed/ready/excluded/corrupt/current shard/stage.

Tests first: 384 assignment covers/disjoint; topology change stable paths; lock prevents concurrent writes; rank0 only compacts all complete; nonzero child failure/no false global complete; non-rank0 waits; script env/no forbidden shell fragments; fake end-to-end orchestration and H3 compatibility. Document roots, exact one-click command and overrides, no GPU run claimed, failure/resume, retention table, source exclusion, final event.

Run all new focused tests, affected existing V3 runtime and H3 reader tests, broad V3 if feasible, full suite (report environment failures distinctly), changed-file Ruff, compileall, diff-check. Commit task. Final review whole diff against baseline, verify frozen/H3 files untouched. Fetch/verify no unexpected target divergence; normal push to origin feature/h3-audio-jea-qwen3-v1 and verify remote HEAD. No force.
