# RA2VA Group Production Status

Updated: 2026-10-10. Branch: `feature/h3-audio-jea-qwen3-v1`.

## Completed

- V3 published group input, ordered Image/Picture/Subject ownership adaptation, bounded prefix freezing without traversing the complete source JSONL.
- Unique HTTP Coordinator, dynamic claims from the same current group, eight stage barriers, generation fencing, cumulative max_groups, terminal receipts and same-run resume. No cross-node JDUFS flock dependency or static rank/shard allocation.
- Real two-node CPU test: 160 unique terminal results, Node A 89 / Node B 71, controlled Worker/Coordinator interruptions, stale-token rejection and completed-run zero-work resume.
- Native Audio + MiMo integration: persistent per-stage GPU workers, AuK speech and SAM music/SFX, DiariZen, Qwen3-ASR, original Visual v5 + Joint v2, TP4 MiMo replicas, 12 training tasks and multi-segment donor export.
- Real single-node 10-clip GPU Pilot, exactly 20 historical MiMo calls. Saved responses can be replayed without new requests; unresolved request intents are not blindly retried.
- Single Bash entry point starts the Node A Coordinator and local/SSH Native Workers, forwards each node's existing GPU/TP4 JSON and stops only owned processes. Same command/run-id resumes existing work.

## Production Root

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/<run-id>/
```

Resume retains the same run-id. Source is the read-only published
`in_pair_reference/post_mask/group-*/samples.jsonl`. Existing experiments are not overwritten.

## Yield, Not Perfect Evidence Chains

MiMo AV evidence is a model judgment, not independent ground truth. Two-step now
treats sparse-sampling `uncertain` and `not_assessable` alike when a visible entity
and temporal alignment exist without explicit source/competing-speaker conflict.
`no_visible_lip_motion` alone does not discard the whole clip in this case.

Unconfirmed identity remains excluded automatically from identity-specific reuse
and donors. Caption speech leads become neutral voices without rewriting visual
actions, ASR, gN/Sx, AV decisions or raw. Common speech verbs, colons, adverbs and
same-speaker continuations no longer cause formatting-only rejection. Ambiguous
Summary attribution is projected from existing visual context plus the acoustic
inventory. These are deterministic corrections, not another model/QA stage.

Shared validator, Multi/Single, T2VA, frozen prompts, overlap/multiple-speaker
exclusions and exact ASR requirements are unchanged. No new hash/seed scans,
polish, fallback, model retry or manual PASS gate was introduced.

## Real Offline Result

CPU replay/export at code commit `18f40e5`, using only the original 10-clip Pilot:

| Metric | Original | Previous repair | Current |
| --- | ---: | ---: | ---: |
| Reconcile Ready / Failed | 8 / 2 | 8 / 2 | 9 / 1 |
| Usable videos | 6 | 8 | 9 |
| Products Ready / Failed | 15 / 4 | 19 / 0 | 22 / 0 |
| Training tasks | 60 | 76 | 88 |
| Donors / segments | 2 / 6 | 2 / 6 | 3 / 10 |
| New model requests | - | 0 | 0 |

Current CPU elapsed: 18.980s; 20 saved responses, 20 historical calls.

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/ra2va-pilot10-publication-v3-20261010T055451Z/group-000000
```

The four Visual tasks each have 9 rows, four Full Audio tasks each 9, four
Speech+BGM tasks each 4: 36 + 36 + 16 = 88.

`d89769` recovered with all 6 exact Chinese dialogues. Its only Caption change
is `<Subject 1> (S1) says,` to `A voice (S1) says,`; `and adds` inherits g1/S1.
All visual action/camera clauses and the g2/S2 reporting sequence are unchanged.
g1/segments 1-2 remain identity excluded. Existing eligible g2/e2/S2 contributes
four donor segments; this is not admission of the unconfirmed g1 voice.

`6107ee` remains Failed: frozen segments are empty but Joint invents
segment_0001. Reordering cannot repair a genuinely extra segment.

Read-only acceptance confirmed all ten raw strings/inference provenance/source
paths, ordered source contract and Audio/AV/Visual facts unchanged. Previous
eight Ready annotations and all 76 training rows remain semantically unchanged
(apart from generated output roots). All 88 task rows have exactly
video/images/audios/caption; all 61 unique referenced files exist. All donor
segment paths/counts are valid, and d89769/g1 is absent from reuse/donors.

## One Bash Launch

Run on Node A in `/mnt/workspace/litengjie/data/R2V_DATA_V2`. The same shared
checkout/version and existing local model environments must be available on each
node. Set `R2V_GROUP_COORDINATOR_TOKEN` privately in the environment. Each node's
JSON names its GPUs, TP4 replicas and existing local HTTP media service (8766
serving `/mnt/workspace`); unrelated media services are never stopped.

```bash
bash scripts/run_h3_ra2va_group_production.sh --run-id ra2va-cluster-pilot10 --limit 10 --max-groups 1 --host 6.167.51.186 --node local docs/examples/ra2va_group_native_node.json 4 --node root@6.167.57.88 docs/examples/ra2va_group_native_node.json 4
```

Use a new run-id for new work; reuse it on resume. `--dry-run` prints the plan
without SSH, GPU work, output writes or token disclosure. Eight GPUs per node
use an existing node-local JSON with gpu_ids 0-7, TP4 groups 0-3 and 4-7, ports
8094/8095 and worker count 8; no new scheduler is needed.

This example selects the published group's first ten rows; it was not executed.
A later new-clip Pilot needs its bounded input window selected before launch.

## Verification

- Final related regression: 739 passed, 32 skipped in 111.91s, including Group,
  Two-step, Single Compact, Audio/H3, donor and T2VA lifecycle consumers.
- New/related Caption publication and materializer tests: 134 passed.
- Native process / one-command launcher tests: 39 passed; wrapper-only final
  run: 26 passed, using local process doubles, not real SSH/GPU execution.
- Ruff, py_compile, bash -n and git diff --check passed.
- One earlier broad run hit a Native detached-child exit timing assertion;
  its isolated rerun and the final combined run passed without modifying the
  existing Native process code. Real two-node GPU teardown remains unverified.
- Static independent review found no concrete policy correctness issue.

## Remaining Acceptance

The one-command wrapper has local subprocess tests, not a real two-node GPU run.
Real HTTP two-node CPU scheduling and real single-node GPU inference are separate
completed milestones. Next is only 6-10 new clips on both nodes, not a full group
or performance expansion. Current CLI remains explicitly bounded by --limit and
publishes PILOT_COMPLETE, not full-group COMPLETE. No whole-group production
acceptance or 90% population-wide yield is claimed from this ten-clip sample.
