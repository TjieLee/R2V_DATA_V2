# V3 Runtime Integrity State

Last updated: 2026-08-13

This file is the current handoff state for `feature/v3-runtime-integrity-v1`.
It records the validated code baseline, real server evidence, and the minimum
remaining work before the visual V3 path is frozen. Do not reconstruct this
state from chat history.

## Development Identity

```text
branch: feature/v3-runtime-integrity-v1
current validated code baseline: b7e46fc748bbedb245d883c0fdf055b5aa90a988
```

Recent final-integrity lineage:

```text
42769a31 Harden final V3 reference integrity
  -> e575af6 Retry malformed V3 integrity reviews
  -> b30942e Add conservative V3 source bbox fallback
  -> 32a9e0e Enforce deterministic V3 reference semantics
  -> 7752dca Prevent V3 integrity schema whitespace stalls
  -> b7e46fc Tighten V3 transient removal artifact review
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

Final integrity has four relevant contracts.

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

### 2. Qwen final integrity review and bounded repair

Synthetic, local, topology-suspicious, represented-content-risk, and otherwise
routed references are checked against highlighted source evidence. A clean real
full reference may skip the Qwen call when no risk route applies.

Transient held objects are not identity requirements by themselves, but their
removal must leave a visually plausible target/reference surface. A white or
transparent silhouette, blank cavity, irregular blob, solid-color placeholder,
or erased-object outline left by removing a held fruit, spoon, cup, bottle, tool,
or other occluder is an artifact failure even when the target identity remains
correct. Such artifact-only failures set the artifact/surface-loss booleans false
while preserving truthful semantic and identity fields, which allows the existing
source-bbox eligibility policy to operate without any routing change. Clean
transient-object removal remains acceptable; naturally white content is not an
artifact merely because it is white.

Commit `e575af6` adds exactly one structured-output repair retry. A valid first
response makes one call. A malformed/truncated/schema-invalid response gets one
repair call with the same image evidence and context. Fail closed occurs only
after both structured responses fail. Raw responses and finish reasons are
retained for diagnostics.

### 3. Structured-output termination order

Commit `7752dca` fixes a real vLLM/Qwen `json_schema` whitespace stall. The two
review schemas now put `reason` before `verdict`, with `verdict` as the final
schema property. Prompts require a short reason, verdict last, immediate object
closure, and no trailing whitespace.

Production `reference_integrity_judge.max_tokens` remains `1024`. Raising it to
2048 was diagnostic only and did not solve the stall.

### 4. Conservative source bbox fallback

Commit `b30942e` adds `source_bbox_fallback_v1` for artifact-local integrity
failures where semantic, target identity, primary identity region, and major
structure all passed. Code baseline `3cfb11f` adds a separate optional
`topology_alpha_hole_upgrade` trigger for a Qwen-accepted reference.

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

The topology trigger is limited to a normal-Qwen-accepted `subject` with
`reference_scope=local`, `synthetic=false`, self-sourced provenance, no prior
bbox fallback, alpha available, at least one enclosed transparent hole, largest
hole area at least `1024`, and enclosed-hole bbox ratio at least `0.01`. It does
not alter the broad `0.05` topology suspicion threshold. Bbox accept publishes
the real source bbox; bbox reject or judge failure keeps the original accepted
reference. The artifact-review-reject path remains fail-closed.

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

- `e9f509acaea60f5699d91287 e2`, woman in a red top with glasses: the edited
  reference was rejected for a severe white bottle-shaped artifact; the raw
  source bbox was accepted and published as real pixels with `synthetic=false`.
- `f71655c87e485f9f5998b1f2 e2`, green laser pointer: normal integrity completed
  successfully and bbox fallback was not invoked.

### Deterministic semantic final-5 replay — PASS

```text
run_id: semantic-final5-20260813-182752
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/semantic-final5-20260813-182752
config_hash: 717840669bb12167214350c71a4eaf22d74edd929ae0502b739bde872c5dd512
```

Observed summary:

```text
processed: 5
failed: 0
entities_reviewed: 3
entities_accepted: 3
entities_rejected: 4
semantic_policy_rejected: 4
judge_failed: 0
source_bbox_fallback_attempted: 0
```

All four negative examples were deterministic rejects with no Qwen or bbox call:

```text
0f32c6b7fa9934c159a03ff7 e2 -> semantic_policy:amorphous_object
34da1ad5a39a0389de87568b e1 -> semantic_policy:scene_structure_object
4e892f7740e1557b495a64da e3 -> semantic_policy:living_creature_object
b527c92f98b7f27f1d301f7c e1 -> semantic_policy:living_creature_object
```

`82f312a07328785e228802d3 e1`, cooked red lobster, remained eligible and was
accepted by normal integrity review.

### Schema-order two-case probe — PASS

Real server probe after `7752dca`:

```text
run_id: integrity-schema-order2-20260813-201628
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/integrity-schema-order2-20260813-201628
config_hash: 6430982581c1771181ec736f498c9d7326a020e9845958684be27f38ff2ebd33
reference_integrity_judge.max_tokens: 1024
```

The two historical whitespace-stall subjects were:

```text
0395f7806515a3efb1c9c6c4 e3  a young man with dark hair in a blue suit jacket
3588dd1d1b1c3dd21788b423 e2  a boy in a red sweater with a necklace
```

After the schema-order fix both were accepted on the first request:

```text
finish_reason:       stop
raw_response_count:  1
trailing_whitespace: 0
judge_failed:        false
```

Observed Qwen integrity call times across the two clips were approximately
`8.641s`, `7.898s`, `7.875s`, and `7.468s`. The previous pathological cases had
consumed roughly 85 seconds per 2048-token attempt and still ended in `length`.
This validates the ordering/termination fix rather than a larger token budget.

## Final 120 Integrity Freeze Replay — INFRA PASS

Final real server replay:

```text
run_id: integrity-final-freeze120-20260813-202412
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/integrity-final-freeze120-20260813-202412
config_hash: 810fdb46cc7b03deaf411d84a13c2c285abde2f584dc42de4d05174505242c47
code baseline: 7752dca272388a61d7619ea5ef69a2618ba7a446
```

Observed stage summary:

```text
processed: 63
skipped_not_ready: 57
failed: 0
entities_reviewed: 72
entities_skipped_review: 10
entities_accepted: 82
entities_rejected: 4
semantic_policy_rejected: 4
judge_failed: 0
topology_suspicious: 2
source_bbox_fallback_attempted: 1
source_bbox_fallback_accepted: 1
source_bbox_fallback_rejected: 0
source_bbox_fallback_judge_failed: 0
runtime: about 10m11s
```

Structured-output audit:

```text
repair_count: 0
length_count: 0
judge_failed_count: 0
```

The four final rejects are exactly the intended deterministic semantic rejects:

```text
0f32c6b7fa9934c159a03ff7 e2 -> semantic_policy:amorphous_object
34da1ad5a39a0389de87568b e1 -> semantic_policy:scene_structure_object
4e892f7740e1557b495a64da e3 -> semantic_policy:living_creature_object
b527c92f98b7f27f1d301f7c e1 -> semantic_policy:living_creature_object
```

The only bbox fallback is the intended red-top-woman artifact case:

```text
e9f509acaea60f5699d91287 e2
  source_bbox_fallback verdict: accept
  final representation: real source RGB bbox
```

This closes the final integrity infrastructure gate. Do not rerun Qwen, SAM3,
Boogu, removal, pair, or annotation for freeze validation unless a visual audit
shows a concrete new problem.

### Population-aware visual review — ONE TARGET REMAINS

The final120 infrastructure run passed. The initial contact sheet overstated the
final-population problem because it included every raw
`references.entities.status=ready` entry, including references from pair-rejected
clips. Final contact sheets must instead use retained and published final
references.

- contact-sheet label `7d3c89d8bb... e2`, the young-boy white-blob case, was
  pair-rejected and is not part of the final integrity or training population;
- `eace10dad52d7534c50dae01 e2`, the man in a white silk robe, now correctly
  follows normal artifact reject -> bbox accept and publishes real source pixels
  with `synthetic=false`;
- `eace10dad52d7534c50dae01 e1`, the man in a red robe, is the only remaining
  final visual-freeze issue. Normal Qwen accepted it, while alpha diagnostics
  show one enclosed transparent hole of about `2362` pixels with ratio about
  `0.0131`.

Code baseline `3cfb11f` makes that last class eligible for an optional strict bbox
comparison without lowering the broad topology threshold or converting topology
into an automatic reject.

## Local Validation for Current Code Baseline

Reported local validation for `3cfb11f`:

```text
reference-integrity targeted tests: 82 passed
storage/schema tests:              63 passed, 1 warning
full pytest:                       1667 passed, 1 warning
Ruff:                              PASS
git diff --check:                  PASS
working tree:                      clean
```

## Remaining Freeze Check

The semantic five-case replay and final120 infrastructure checks have passed.
After correcting the contact-sheet population, the only remaining visual-V3
check is one targeted integrity replay for `eace10dad52d7534c50dae01 e1` on code
baseline `3cfb11f`, using already-materialized evidence and rerunning no upstream
stage. Review the normal integrity verdict, topology trigger, strict bbox
decision, and final published reference. Do not rerun the full 120 population.

If this one targeted outcome is visually correct with no infrastructure
failure, record the replay evidence and freeze the visual V3 code/config before
returning to the audio/H3 branch.

## Historical Runtime Note

Historical configs including `prod5000` may contain the accidental 40-step
remover setting. They are evidence of the old runtime, not the production
Object-Remover contract. The current production profile is always the mandatory
Object-Remover LoRA with four inference steps.
