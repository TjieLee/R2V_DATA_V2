# V3 Runtime and Reference Integrity V1

## Scope

This design is based on parent commit
`098a02f3b79625d7cb15accd6dcd4ba949280388` and is developed on
`feature/v3-runtime-integrity-v1`. It keeps the established V3 coverage,
candidate ranking, fragmentation, background guard, Boogu, and cross-pair
quality policies unchanged.

No completed run or export tree is migrated by this change. Existing configs
remain on `runtime.mode: staged_legacy` unless streaming is explicitly enabled.

## Density Evidence

The `density120` review contained 117 ready annotations, 145 annotation
entities, 51 accepted samples, and 63 final entity references. Its final
density was 1.235 entity references per sample: 40 samples had one reference,
10 had two, and one had three. The object funnel retained 35 of 38
coverage-qualified objects; the subject funnel retained 28 of 30
coverage-qualified subjects.

This evidence supports retaining `reference_dense_v1`, `phrase_retry_v1`, and
`type_aware_v1`. The new work removes semantic pollution and structurally
unusable final references instead of loosening downstream gates.

## Semantic Cleanup

Dense annotation defines a subject as one visible person, animal, or character.
An object is one concrete, discrete physical foreground object with independent
conditioning value. Animals are never objects. Body parts, amorphous materials,
liquids, sauces, smoke, shadows, lighting, scene structure, vegetation, and
depicted or screen content are not promoted to physical object references.

Discrete food, vehicles, tools, bags, clothing, and other worn or attached
controllable objects remain eligible. A conservative syntax check drops vague
object heads such as `object`, `thing`, and `item`, including modified phrases
such as `a large black object with cutouts`. It does not guess a replacement
category and does not contain an animal or product ontology.

## Progressive SAM3 Anchors

`sam3.anchor_search_mode: progressive_v1` probes the existing fast order
`5, 2, 7, 0, 9`, stopping at the first unique usable target. Only when that
order yields no unique target does it probe `4, 6, 3, 8, 1`. An ambiguous early
probe does not prevent a later unique subject or object anchor. A multi-object
group remains `unverified_multi_object_group`; group masks are not unioned.

The segment summary records fast hits, fallback attempts, fallback hits,
all-frame misses, and total probe calls. `legacy` remains the default.

## Reference Integrity Stage

The opt-in stage order is:

```text
pair -> reference_edit -> reference_integrity -> instruct -> export
```

`targeted_qwen_v1` calculates alpha topology evidence including component
ratios, bbox fill, border contact, and enclosed transparent holes. Topology is
only a suspicion signal. Legitimate handles, brackets, wheels, scissors,
frames, and other source-matching cutouts are never rejected by topology alone.

Qwen review is required for every synthetic reference, every local reference,
and every topology-suspicious reference. It receives the final reference plus
the sampled source frame with the source target highlighted. A clean, real,
full, non-suspicious reference skips the extra call.

The review checks target identity, recognizability for the declared scope,
major structure, missing surfaces, unnatural holes, unrelated dominance, and
independent reference usability. Subject review does not require a face, but it
rejects a crop that loses essentially all identity-bearing evidence present in
the source. Judge failure rejects that entity fail-closed.

An integrity rejection removes only that entity, rebuilds tokens from the
remaining ready references, and invalidates instruction/export. A clip becomes
non-exportable only when no qualifying entity reference remains.

## Object-Remover Contract

The production profile is:

```yaml
remove:
  inference_profile: object_remover_4step_v1
  num_inference_steps: 4
```

It requires the existing Qwen-Image-Edit-2511 Object-Remover LoRA. Base-only
generation is not allowed. The backend verifies the active adapter and emits
startup diagnostics for model, adapter, weight file, steps, dtype, configured
device, and visible CUDA count before first inference.

Historical production configs, including `prod5000`, stored the unintended
40-step value. Historical runtime comparisons must therefore not be treated as
4-step performance. Forty steps remain possible only under
`inference_profile: experimental_override`.

A candidate round containing any backend, CUDA, or runtime failure remains
`pending_remove` with diagnostic attempts and can be retried by running only
the remove stage. Permanent rejection occurs only when every candidate in the
completed round was generated and quality-rejected. Overwrite of an existing
ready removal preserves the previous publication when a new attempt has an
infrastructure failure.

## Streaming Runtime

`runtime.mode: streaming_v1` executes a bounded per-clip DAG. Each clip owns one
writer, while different clips may occupy different stages. CPU stage executors
are constrained by both their stage worker count and `runtime.cpu_workers`.
Global export begins only after all clip tasks finish.

All Qwen calls share one budget, including calls made by isolated workers. A
process-local semaphore and run-local file slots enforce
`runtime.qwen_max_inflight` across threads and worker processes.

SAM3, remove, and reference-edit each have one long-lived JSONL worker process.
The process sets `CUDA_VISIBLE_DEVICES` before model imports and uses local
`cuda` or `cuda:0`, avoiding physical-versus-visible ordinal ambiguity. One
request failure returns a failed response without terminating neighboring
requests. Invalid JSON, timeout, process exit, and request-ID mismatch fail
closed.

Recommended server mapping:

```yaml
runtime:
  mode: streaming_v1
  qwen_max_inflight: 2
  cpu_workers: 8
  gpu_workers:
    remove: "4"
    segment: "5"
    reference_edit: "6"
```

The Qwen service remains on physical GPUs 0-3. Server `max-num-seqs` must be a
deployment variable and must not be raised without a GPU-memory pilot.

Streaming fails configuration validation when same-parent fallback is enabled,
because donor availability would otherwise depend on task completion order.

## Profiling

Streaming events record queue wait, service time, stage inflight count, clip
end-to-end time, per-stage throughput, Qwen inflight observations, and GPU
worker busy/idle estimates. Existing model-call retry data, SAM anchor counters,
and integrity counters remain available. Summary identifies the lowest measured
stage throughput as the current bottleneck.

## Known Limitations

- The first streaming release supports independent clips only; same-parent
  fallback requires staged execution.
- Heavy model worker counts are fixed at one. Throughput comes from cross-stage
  overlap, not duplicate model copies.
- GPU/model compatibility and throughput still require server smoke tests.
- Existing runs whose config hash predates the new schema are not rewritten.

## Server Smoke Commands

These commands are documentation only and were not run by Codex. Use a fresh
20-clip run root in
`/mnt/workspace/litengjie/data/R2V_DATA_V2/configs/runtime-integrity-20.local.yaml`.
That local file must set `source.limit: 20`, progressive anchors, the 4-step
remover profile, integrity enabled, streaming enabled, and GPU mappings 4/5/6.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate
export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
CONFIG=/mnt/workspace/litengjie/data/R2V_DATA_V2/configs/runtime-integrity-20.local.yaml
```

Progressive SAM3 smoke on the fresh run:

```bash
python run_pipeline_v3.py --config "$CONFIG" --stages manifest,annotate,frames,segment --profile
```

Four-step remover smoke after rank/background evidence exists:

```bash
python run_pipeline_v3.py --config "$CONFIG" --stages rank,background,remove --profile
```

Integrity smoke after pairing/reference edit:

```bash
python run_pipeline_v3.py --config "$CONFIG" --stages pair,reference_edit,reference_integrity --profile
```

Full streaming 20-clip pilot with a separate fresh run root:

```bash
python run_pipeline_v3.py --config "$CONFIG" --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,export --profile
```

Do not run these commands against a completed production or pilot run root.
