# V3 Subject Attributes State

Last updated: 2026-08-19

This file records the current validated state for subject-bound Visual attribute
references. It is a sidecar extension to the frozen Visual V3 path. Do not
reconstruct these values from chat history.

## Development Identity

```text
branch: feature/v3-subject-attributes-v1
validated code baseline: c1c056675dac9cdce5e585cd3f934c4bb573fc96
frozen Visual branch: feature/v3-runtime-integrity-v1
frozen Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
```

The subject-attribute branch must not change the frozen Visual sample/export
semantics or thresholds. Attribute outputs are sidecars and enriched samples;
the frozen Visual export remains `r2v.v3.dataset.1` / `r2v.v3.sample.1`.

Documentation-only commits may move repository HEAD after the validated code
baseline. Record both when freezing or reproducing a run.

## Attribute Contract

Supported attribute types:

```text
face
hair
headwear
glasses
upper_clothing
lower_clothing
dress_or_skirt
shoes
bag
accessory
```

Every attribute is bound to one subject through `owner_entity_id`.

The production policy is precision-first and fail-closed:

- only eligible retained human subjects with a ready main reference are
  considered;
- discovery makes at most one Qwen call per eligible owner;
- at most three attributes are discovered per owner;
- each attribute probes only the owner's existing Top3 candidate frames;
- attribute SAM3 is single-frame prompt segmentation only;
- there is no 10-frame attribute tracking, no 7/10 attribute coverage rule, and
  no temporal propagation;
- prefer a source frame different from the owner's final main-reference frame;
  same-frame publication is only a fallback;
- wrong-owner rejection uses deterministic geometry and other subjects' usable
  SAM masks when available;
- all geometry-passing crops are reviewed once in a batched Qwen review per
  owner/context group;
- publication requires every review flag to pass: attribute match, owner
  binding, recognizability, characteristic appearance visibility, and usability
  as an attribute condition;
- accepted crops are raw source RGB plus the accepted mask, exported as RGBA
  transparency;
- no background fill, Object Remover, Boogu completion, or other generative
  completion is used for attribute references.

Current deterministic size/shape gates include the type-specific hair and
headwear minimum long-side requirements plus the clothing area/aspect and
near-duplicate-mask guards. These gates must not be loosened merely to recover
attribute yield.

Zero accepted attributes for an owner is valid. Missing attribute references are
preferred over low-quality or wrong-owner references.

## Sidecar Outputs

Streaming attribute artifacts live below:

```text
<run_root>/subject_attributes/
  owners/<clip_uid>/<entity_id>.json
  references/<clip_uid>/*.png
  samples/<clip_uid>.json
  attributes.jsonl
  enriched_samples.jsonl
  summary.json
```

The durable owner/sample artifacts are reconciled atomically into the JSONL
sidecars. The frozen Visual `ClipRecord` and frozen Visual export are not mutated
by attribute enrichment.

Enriched instructions append owner-aware attribute conditions only for accepted
references, for example:

```text
<Image 1>, with hairstyle shown in <Image 2>, wearing clothing shown in <Image 3> ...
```

Rejected attributes are omitted.

## Integrated Streaming Runtime

Current full streaming stage order:

```text
manifest
-> annotate
-> frames
-> segment
-> rank
-> background
-> remove
-> pair
-> reference_edit
-> reference_integrity
-> instruct
-> subject_attributes
-> export
```

`subject_attributes` is supported only in `runtime.mode: streaming_v1` for the
integrated full-pipeline path.

Attribute Qwen requests reuse the existing global `QwenConcurrencyGate`; there
is no private attribute Qwen executor/semaphore. The current gate uses
acquire-any-slot behavior across the shared file-lock slots to avoid waiting on a
busy preselected slot while another global Qwen slot is free.

Main temporal SAM3 and attribute SAM3 use the same checkpoint and segmentation
semantics but may run in separate persistent processes:

```yaml
runtime:
  gpu_workers:
    segment: "5"
    subject_attributes_segment: "7"
```

With `subject_attributes_segment` configured, GPU5 receives main temporal
`run_clip` work and GPU7 receives attribute `attribute_probe` work. Without the
optional setting, attribute probes fall back to the main segment worker exactly
as before.

The subject-attribute stage worker count and optional dedicated attribute SAM GPU
assignment are sidecar runtime-capacity settings and are excluded from the
frozen Visual config fingerprint/model identifiers. The Git commit still remains
part of run identity, so code changes require a fresh run root.

## Current Production Runtime Baseline

Validated runtime configuration:

```text
Qwen model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
Qwen endpoint: http://127.0.0.1:8000/v1
Qwen dtype: BF16
Qwen tensor parallel: 1
Qwen data parallel: 4
Qwen max model length: 32768
runtime.qwen_max_inflight: 4
runtime.stage_workers.subject_attributes: 2
```

Validated GPU allocation:

```text
GPU 0-3  Qwen3-VL-32B-Instruct, TP1 x DP4
GPU 4    Object Remover
GPU 5    main temporal SAM3
GPU 6    Boogu
GPU 7    dedicated subject-attribute single-frame SAM3
```

For SAM3 worker isolation, the parent pipeline must not globally remap CUDA.
The worker exposes its assigned physical GPU and uses worker-local `cuda`; e.g.
physical GPU5 or GPU7 appears as local CUDA device 0 inside that worker.

## Performance Evidence

### Integrated TP4 baseline

Run:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-b0790e-20260819-211823
```

Observed on 10 clips:

```text
full profile/wall: about 889 s
Qwen aggregate model service: about 1601 s
Qwen queue wait: about 1547 s
attribute accepted references: 17
```

This exposed Qwen serving as the dominant shared-capacity bottleneck.

### TP1 x DP4 benchmark

Run:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-dp4-889b45-20260819-221452
```

Observed on the same selected population:

```text
full profile: 325.7 s
shell real time: 327.3 s
attribute accepted references: 19
attribute Qwen blocking time: 72.9 s
attribute SAM3 blocking time: 266.8 s
```

Compared with the TP4 integrated run, end-to-end wall time improved by about
2.7x. TP1 x DP4 is therefore the production Qwen serving baseline. Do not return
to TP4 merely because one tensor-parallel rank shows lower instantaneous GPU
utilization.

### Dedicated GPU7 attribute SAM benchmark

Run:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-dp4-sam7-c1c056-20260819-223752
```

Observed:

```text
shell real time: 368.9 s
eligible human owners: 12
SAM3 attempts: 35
attribute accepted references: 18
attribute Qwen blocking time: 85.0 s
attribute SAM3 blocking time: 79.5 s
failures: 0
```

The dedicated GPU7 worker reduced attribute SAM blocking substantially compared
with the shared-GPU5 run, but the 10-clip end-to-end wall time did not improve
because the two full-pipeline executions followed different model-dependent
paths and workloads. Keep the dedicated worker for resource isolation; do not
claim a separate end-to-end throughput gain from this small canary.

## Current Decision

Stop further performance tuning for now.

Production baseline:

```text
code: c1c056675dac9cdce5e585cd3f934c4bb573fc96
Qwen: BF16, TP1 x DP4, max_model_len=32768
runtime.qwen_max_inflight: 4
GPU5: main temporal SAM3
GPU7: dedicated attribute SAM3
subject_attributes workers: 2
```

Do not introduce FP8, TP2 x DP2, additional attribute concurrency, relaxed
quality gates, or scheduler changes without a new explicit benchmark/review.
Future performance work should be driven by larger production evidence rather
than 10-clip timing noise.

## Known Non-Blocking Resume Edge Cases

Two edge cases remain worth tracking before an eventual final freeze:

1. If upstream work is rerun so a previously eligible clip becomes
   export-ineligible, old durable attribute sidecars can remain unless attribute
   overwrite/reconciliation explicitly invalidates them.
2. An owner exception that occurs outside durable owner-artifact creation may be
   counted by the current invocation/runtime failure path but underrepresented
   when `summary.json` is later rebuilt solely from durable owner artifacts.

These are unusual resume/reconciliation cases, not failures observed in the
fresh production canaries above. Do not weaken publication gates as a workaround.

## Validation at Current Baseline

Reported validation for `c1c0566`:

```text
focused tests: 77 passed
full pytest: 1729 passed, 1 warning
git diff --check: PASS
working tree after push: clean
```

Ruff was not installed in the project `.venv`, so no Ruff result is claimed for
this baseline.
