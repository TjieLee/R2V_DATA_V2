# V3 Runtime Integrity State

Last updated: 2026-08-13

This file is the current handoff state for `feature/v3-runtime-integrity-v1`.
It records the validated code baseline, real server evidence, and the minimum
remaining work before the visual V3 path is frozen. Do not reconstruct this
state from chat history.

## Development Identity

```text
branch: feature/v3-runtime-integrity-v1
current validated code baseline: 32a9e0e17598b6bd2d7912b6fafdb08d81187285
```

Recent integrity lineage:

```text
42769a31 Harden final V3 reference integrity
  -> e575af6 Retry malformed V3 integrity reviews
  -> b30942e Add conservative V3 source bbox fallback
  -> 32a9e0e Enforce deterministic V3 reference semantics
```

Documentation-only commits may move repository HEAD after the code baseline.
When operating the server, record both the code baseline and the docs HEAD.

## Frozen Pipeline Contracts

The current visual V3 contract keeps all of the following unchanged:

- dense annotation with explicit subject/object semantic boundaries;
- fixed ten-frame sampling;
- SAM3 progressive anchor search and rescue modes;
- 7/10 temporal coverage;
- pair candidate Top3 ranking and current geometry thresholds;
- `pair.max_candidates_per_entity: 3`;
- `pair.crop_padding_ratio: 0.08`;
- conservative reference prefilter;
- final Qwen background guard;
- Qwen scale-collapse fallback guard;
- same-parent cross-pair disabled;
- mandatory Object-Remover LoRA at four steps;
- staged legacy remains the safe default runtime.

The stage order is:

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
-> export
```

## SAM3 Rescue Evidence

The fresh 120-clip A/B run is:

```text
run:    /mnt/workspace/litengjie/data/r2v_v3_runs/sam3-rescue120-20260813-110929
config: /mnt/workspace/litengjie/data/r2v_v3_configs/sam3-rescue120-20260813-110929.yaml
config hash: f5346dab94d7e0ecb7c2c8d0ede38df47f423de78323c9321005915e64bb0ff9
```

Frozen rescue modes:

```yaml
sam3:
  anchor_search_mode: progressive_v1
  object_rescue_mode: phrase_retry_v1
  not_found_rescue_mode: entity_phrase_retry_v1
  multi_instance_rescue_mode: qwen_anchor_select_v1
```

Observed A/B result:

```text
old ready -> new ready: 87 -> 122
ready recall:            53.7% -> 75.3%
net recovered:           +35
old ready -> new non-ready regressions: 0
subject ready:           52 -> 82
object ready:            34 -> 39
coverage-qualified:      80 -> 103
ready->ready median mask IoU < 0.80: 0
```

Phrase retry produced the largest gain. Multi-instance selector attempted 95
cases and selected 11 anchors. Identity switches remain an audit signal, not a
reason to loosen coverage or tracking thresholds. The known blue-woman
zero-candidate grounding miss is intentionally not being tuned further.

## Reference Integrity Contracts

Final integrity has three independent mechanisms.

### 1. Deterministic semantic policy

Commit `32a9e0e` hard-rejects only high-confidence `object` taxonomy violations
before any Qwen integrity call. It uses the entity phrase head rather than broad
grounding context.

Hard-reject classes:

```text
semantic_policy:amorphous_object
semantic_policy:scene_structure_object
semantic_policy:living_creature_object
```

The gate is deliberately conservative. Culinary/prepared animal food and clear
physical carriers/replicas such as toys, models, figurines, bottles, shells, or
replicas remain eligible. Represented/screen/painting content remains on the
Qwen semantic-risk review path.

### 2. Qwen final integrity review

Synthetic, local, topology-suspicious, represented-content-risk, and otherwise
routed references are checked against highlighted source evidence. A clean real
full reference may skip the Qwen call when no risk route applies.

Commit `e575af6` adds exactly one structured-output repair retry. A valid first
response makes one call. A malformed/truncated/schema-invalid response gets one
repair call with the same image evidence and context. Fail closed occurs only
after both structured responses fail. Raw responses and finish reasons are
retained for diagnostics.

### 3. Conservative source bbox fallback

Commit `b30942e` adds `source_bbox_fallback_v1` only for artifact-local integrity
failures where semantic, target identity, primary identity region, and major
structure all passed.

The fallback:

- uses the exact already-selected source frame;
- crops raw RGB using the tracked target mask bbox plus normal crop padding;
- does not apply the mask;
- does not inpaint or generate pixels;
- is disabled for cross-pair, semantic failures, identity failures, wrong target,
  major structural failures, and integrity judge failures;
- requires a separate strict Qwen bbox review before publication;
- publishes `source_bbox_fallback=true` and `synthetic=false` with explicit
  source/bbox provenance.

## Real Targeted Integrity Evidence

### Bbox / structured-output targeted replay

```text
run_id: integrity-bbox-targeted-20260813-174737
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/integrity-bbox-targeted-20260813-174737
config_hash: 268fe598b026c9a974b034cb6ed58b3d7f5f3ff8ca55b000c35b6709027f477c
```

Observed summary before the deterministic semantic hard gate was added:

```text
processed: 11
failed: 0
entities_reviewed: 16
entities_skipped_review: 2
entities_accepted: 18
entities_rejected: 0
judge_failed: 0
topology_suspicious: 1
source_bbox_fallback_attempted: 1
source_bbox_fallback_accepted: 1
source_bbox_fallback_rejected: 0
source_bbox_fallback_judge_failed: 0
```

Important real cases:

- `e9f509acaea60f5699d91287 e2`, woman in a red top with glasses:
  the generated/cutout reference was rejected for a severe white bottle-shaped
  artifact; the raw source bbox was accepted and published as real pixels with
  `synthetic=false`.
- `f71655c87e485f9f5998b1f2 e2`, green laser pointer:
  the previous malformed-JSON failure did not recur as a `judge_failed` result;
  the normal integrity path completed successfully and bbox fallback was not
  invoked.
- robe-person examples were accepted by normal integrity and therefore did not
  trigger bbox fallback. Do not widen fallback merely to force those cases onto
  the bbox path.

The same replay demonstrated why a deterministic semantic hard gate was needed:
Qwen still accepted sauce, cathedral, dog-as-object, and giant-clam-as-object
despite semantic-risk routing.

### Deterministic semantic final-5 replay — PASS

Real server replay after commit `32a9e0e`:

```text
run_id: semantic-final5-20260813-182752
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/semantic-final5-20260813-182752
config_hash: 717840669bb12167214350c71a4eaf22d74edd929ae0502b739bde872c5dd512
```

Observed stage summary:

```text
processed: 5
skipped_disabled: 0
skipped_existing: 0
skipped_not_ready: 0
failed: 0
entities_reviewed: 3
entities_skipped_review: 0
entities_accepted: 3
entities_rejected: 4
semantic_policy_rejected: 4
judge_failed: 0
topology_suspicious: 0
source_bbox_fallback_attempted: 0
source_bbox_fallback_accepted: 0
source_bbox_fallback_rejected: 0
source_bbox_fallback_judge_failed: 0
```

All five freeze-gate cases matched expectations:

```text
0f32c6b7fa9934c159a03ff7 e2  thick golden-brown sauce
  rejected, reviewed=false
  semantic_policy:amorphous_object
  no source context, Qwen review, or bbox fallback

34da1ad5a39a0389de87568b e1  large domed cathedral
  rejected, reviewed=false
  semantic_policy:scene_structure_object
  no source context, Qwen review, or bbox fallback

4e892f7740e1557b495a64da e3  light-colored dog typed object
  rejected, reviewed=false
  semantic_policy:living_creature_object
  no source context, Qwen review, or bbox fallback

b527c92f98b7f27f1d301f7c e1  giant clam typed object
  rejected, reviewed=false
  semantic_policy:living_creature_object
  no source context, Qwen review, or bbox fallback

82f312a07328785e228802d3 e1  cooked red lobster
  not semantic-policy rejected
  normal Qwen integrity review accepted
  reference_entity_semantically_valid=true
```

This closes the deterministic semantic policy validation. Do not tune the hard
gate further without new concrete false-positive or false-negative evidence.

## Local Validation for Current Code Baseline

Reported validation for `32a9e0e17598b6bd2d7912b6fafdb08d81187285`:

```text
reference-integrity targeted tests: 66 passed
storage/schema tests:              63 passed, 1 warning
full pytest:                       1651 passed, 1 warning
Ruff:                              PASS
git diff --check:                  PASS
working tree:                      clean
```

No real Qwen, SAM3, Boogu, CUDA, GPU, or server data job was run by the local
implementation task. Real server evidence is recorded separately above.

## Remaining Freeze Check

The five-case semantic replay has passed. The only remaining visual-V3 freeze
step is one final 120-clip `reference_integrity` replay on the fixed development
population, followed by a contact-sheet/audit review.

For this final replay:

- reuse already-materialized annotation, frames, SAM3 masks, coverage, pair,
  removal, and reference-edit evidence;
- do not rerun SAM3, annotation, remover, pair, or Boogu;
- create a fresh run root with a current matching `run.json`;
- preserve old integrity rejects monotonically when preparing replay artifacts;
- clear only `reference_integrity`, instruction, and export states that need to
  be recomputed;
- require `ClipRecord.model_validate()` before model calls;
- keep `debug.save_diagnostics=true`;
- run only `--stages reference_integrity --profile`;
- review counters, all rejects, all bbox fallbacks, all judge repairs/failures,
  and a final contact sheet before freezing.

Freeze requires no new infrastructure failure or broad regression. A small
number of intended semantic/artifact rejects is expected and is not itself a
failure.

After the final 120 replay and contact-sheet review pass, freeze the visual V3
code/config and return to the audio/H3 branch.

## Historical Runtime Note

Historical configs including `prod5000` may contain the accidental 40-step
remover setting. They are evidence of the old runtime, not the production
Object-Remover contract. The current production profile is always the mandatory
Object-Remover LoRA with four inference steps.
