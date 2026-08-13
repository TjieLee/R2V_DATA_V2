# V3 Runtime and Reference Integrity V1

Last updated: 2026-08-13

This document is the operational/design specification for
`feature/v3-runtime-integrity-v1`. It records the current visual V3 contracts
and the commands needed to create, resume, or replay a server run. Current
experiment state and freeze evidence are in `V3_RUNTIME_INTEGRITY_STATE.md`.
Machine paths, services, and GPU rules are in `SERVER_ENVIRONMENT_RUNBOOK.md`.

## 1. Current Code Baseline

```text
branch: feature/v3-runtime-integrity-v1
validated code baseline: b7e46fc748bbedb245d883c0fdf055b5aa90a988
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

A later documentation-only commit may move HEAD without changing the validated
code baseline. Always record both values in a production/freeze note.

## 2. Frozen Stage Order

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

`rank.py` is temporal coverage, not candidate Top3 ranking. Candidate extraction
and Top3 ordering after SAM masks live in `pair.py`.

Do not loosen coverage or candidate thresholds as part of integrity work.

## 3. Source Selection

The validated fixed-selection mode is:

```yaml
source:
  selection_mode: fixed_selection_v1
```

The 120-case development selection used for the current rescue/integrity work is:

```text
/mnt/workspace/litengjie/data/r2v_v3_selections/density120-fixed.json
```

Fixed selection is loaded in O(K), validates identity/path/duplicates, and
preserves manifest order. Do not replace it with `start_index/limit` when an
exact A/B population is required.

## 4. Annotation Semantics

`reference_dense_v1` may emit up to eight stable foreground references.

Subject:

- one visible person, animal, or character;
- animals and other living creatures belong here, not under `object`.

Object:

- one concrete, discrete physical foreground object with independent reference
  value;
- cooked/prepared culinary food may be a valid object;
- body parts, amorphous sauces/liquids/smoke/light, scene structure, vegetation,
  and represented/screen content are not ordinary physical object references.

Annotation semantics are intentionally conservative. Final integrity now has an
additional deterministic hard gate for high-confidence violations; it does not
replace annotation cleanup.

## 5. SAM3 Rescue

Frozen modes:

```yaml
sam3:
  anchor_search_mode: progressive_v1
  object_rescue_mode: phrase_retry_v1
  not_found_rescue_mode: entity_phrase_retry_v1
  multi_instance_rescue_mode: qwen_anchor_select_v1
```

The flow is:

1. probe the progressive anchor slots;
2. use the grounding prompt first;
3. on object not-found, retry the entity phrase where configured;
4. for ambiguous multi-instance anchors, show numbered candidates to Qwen and
   allow only a strict selected target;
5. keep existing track/coverage thresholds unchanged.

The current 120 A/B improved ready clips from 87 to 122 with zero old-ready to
new-non-ready regressions. This rescue path is frozen unless a final audit shows
new concrete regressions.

## 6. Coverage and Candidate Top3

Coverage remains 7/10.

After SAM, `pair.py` builds entity reference candidates only from frames that
are present, track-valid, non-empty, and geometrically sane. Severe tiny or
fragmented candidates are removed before sorting.

Current Top3 sort priority is:

```python
(
    candidate.border_contact_count,
    -candidate.area_ratio,
    -candidate.sharpness_score,
    candidate.normalized_center_distance,
    _SLOT_PRIORITY_INDEX[candidate.frame_slot],
)
```

with slot priority:

```text
5, 4, 6, 3, 7, 2, 8, 1, 9, 0
```

and current limit:

```yaml
pair:
  max_candidates_per_entity: 3
  crop_padding_ratio: 0.08
  reference_prefilter_mode: conservative_v1
```

Do not move this logic into `rank.py`.

## 7. Background Removal and Reference Edit

Production remover profile:

```yaml
remove:
  inference_profile: object_remover_4step_v1
  num_inference_steps: 4
  device: cuda:4
```

The Qwen-Image-Edit-2511 Object-Remover LoRA is mandatory. Do not use old
40-step settings, base-only generation, Lightning/Rapid substitute weights, or
historical experimental defaults as production guidance.

Reference editing uses the validated Boogu runtime. The current completion and
background prompts are already frozen; integrity work must not broaden them.

## 8. Final Reference Integrity

### 8.1 Deterministic semantic hard gate

Commit `32a9e0e` adds a high-precision policy classifier before Qwen review.
Only `reference_type == object` is eligible for deterministic semantic reject.
The classifier uses the entity phrase head and does not use broad grounding
context as the rejection trigger.

Current policy reasons:

```text
semantic_policy:amorphous_object
semantic_policy:scene_structure_object
semantic_policy:living_creature_object
```

Examples expected to hard reject:

```text
a thick golden-brown sauce
a large domed cathedral
a light-colored dog with long fur   # if typed object
a giant clam with a blue-spotted mantle  # if typed object
```

Examples deliberately not hard rejected solely by this classifier:

```text
a cooked red lobster on a wooden cutting board
a cooked whole fish in a wok
a bottle of water
an oil bottle
a green laser pointer emitting bright green light
a dog toy
a clam shell
a cathedral model
a tree branch
```

Represented content remains a semantic-risk route to Qwen because a physical
carrier such as a screen, framed painting, or poster requires visual context.

A deterministic policy reject:

- does not call the Qwen integrity judge;
- does not call the source-bbox fallback judge;
- publishes an explicit `semantic_policy_reason`;
- removes the entity using the same downstream pairing/reference invalidation
  semantics as other integrity rejection;
- remains valid under normal `ClipRecord` validation.

### 8.2 Qwen integrity review

Review evidence is the final reference plus highlighted source context.
Synthetic, local, topology-suspicious, source-bbox, represented-content-risk,
and other risk-routed references require review. Clean real full references may
skip it when no risk applies.

The hard review booleans include:

```text
matches_target
reference_entity_semantically_valid
preserves_annotated_entity_semantics
preserves_primary_identity_region
recognizable_as_named_entity
structurally_complete_for_scope
no_major_missing_regions
no_unnatural_holes_or_surface_loss
no_unrelated_entity_dominance
no_severe_reference_artifact
usable_as_independent_reference
```

The verdict must match all booleans.

A transient held or occluding item is not automatically part of target identity,
but its removal must leave a plausible target/reference surface. Reject an
otherwise-correct person reference when removal leaves a white/transparent
silhouette, object-shaped cavity, irregular mouth/hand blob, solid-color
placeholder, erased outline, or artificial missing/reconstructed surface. Set
`no_severe_reference_artifact=false` and also set
`no_unnatural_holes_or_surface_loss=false` when the residue is an unnatural
missing or replaced surface. Keep the semantic/identity booleans truthful so an
artifact-only rejection can enter the existing bbox fallback policy naturally.

Do not require a transient bowl, orange, bottle, spoon, food item, tool, or
chopsticks to remain when removal is visually clean. Do not infer artifacts from
white pixels alone: naturally white clothing/objects/background, highlights,
teeth, sclera, and paper remain valid source-matching content.

### 8.3 Structured-output repair

`QwenReferenceIntegrityJudge` performs one normal request. If JSON is truncated,
invalid, or schema-invalid, it performs exactly one repair request using the
same image evidence/context plus the invalid response and validation error.

Profiling distinguishes:

```text
operation=initial retry_index=0
operation=repair  retry_index=1
```

A valid first response adds no extra call. Both raw responses and finish reasons
are retained on repair/final failure. Do not globally increase Qwen max tokens
as a substitute for this bounded repair.

### 8.4 Source bbox fallback

`source_bbox_fallback_v1` is a narrow recall rescue for artifact-only failures.
It is not a generic second reference selector.

Eligibility requires semantic, target, identity, and major structure checks to
pass. The failed reference must primarily have a localized edit/cutout artifact,
for example an unnatural erased-object-shaped cavity.

Never use bbox fallback for:

- deterministic semantic-policy rejection;
- Qwen integrity judge failure;
- wrong target;
- semantic mismatch;
- lost primary identity region;
- major structural loss;
- severe fragmentation;
- cross-pair in V1.

The proposed fallback uses the exact selected source frame and target tracked
mask bbox with the normal crop padding, then crops raw RGB only. It does not
apply the mask, erase an occluder, inpaint, or generate pixels.

A dedicated three-image Qwen review compares highlighted source context, the
failed reference, and the proposed raw bbox crop. All strict fallback booleans
must pass. On acceptance the final reference is real source pixels with:

```text
source_bbox_fallback=true
synthetic=false
```

and explicit source clip/entity/frame/bbox/hash metadata. If a prior Boogu edit
was the published reference, the edit state is atomically transitioned to the
source-bbox fallback policy while the original generated evidence remains for
diagnostics.

## 9. Server Environment

Canonical paths:

```text
repo:          /mnt/workspace/litengjie/data/R2V_DATA_V2
python:        /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
SAM3 code:     /mnt/workspace/litengjie/data/vendor/sam3
SAM3 ckpt:     /mnt/workspace/public/pretrained/facebook/sam3/sam3.pt
Qwen model:    /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
Qwen endpoint: http://127.0.0.1:8000/v1
Boogu code:    /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu python:  /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model:   /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
Remover LoRA:  /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover/Qwen-Image-Edit-2511-Object-Remover.safetensors
```

Preferred physical GPU allocation:

```text
GPU 0-3: Qwen3-VL vLLM TP4
GPU 4:   Object remover
GPU 5:   SAM3
GPU 6:   Boogu
GPU 7:   spare
```

Do not globally export `CUDA_VISIBLE_DEVICES` in a general production shell.
For an isolated staged SAM3-only invocation, exposing physical GPU 5 and using
worker-local `cuda` is valid. Do not wrap the full pipeline or the remove stage
with `CUDA_VISIBLE_DEVICES=4`; staged removal uses the configured physical
`cuda:4` device.

See `SERVER_ENVIRONMENT_RUNBOOK.md` for exact Qwen startup, shell initialization,
preflight, monitoring, and process-safety rules.

## 10. Fresh Server Preflight

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

set +e
set +u
set +o pipefail

export MAIN_PYTHON=/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
export SAM3_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/sam3
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
export PYTHONPATH="$SAM3_CODE_ROOT:/mnt/workspace/litengjie/data/R2V_DATA_V2${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp

git fetch origin
git switch feature/v3-runtime-integrity-v1
git pull --ff-only

git rev-parse HEAD
git status --short
curl -fsS --noproxy '*' "$QWEN_BASE_URL/models" | head
```

Before a code-sensitive replay, compare HEAD with the recorded validated code
baseline. A docs-only HEAD after that baseline is allowed when the code tree is
unchanged.

## 11. Fresh Full Pipeline Run

Use a fresh timestamped run root and exact config. Never reuse a non-empty run
root with a different `run.json` identity.

```bash
CONFIG=/mnt/workspace/litengjie/data/r2v_v3_configs/<exact-config>.yaml
LOG=/mnt/workspace/litengjie/data/r2v_v3_logs/<exact-run>.log

mkdir -p /mnt/workspace/litengjie/data/r2v_v3_logs

nohup "$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,export \
  --profile \
  > "$LOG" 2>&1 &

echo $!
```

Do not omit `reference_integrity` from the final production/freeze stage list.

## 12. Stage-Only Replay Rules

Run only the stage that needs new evidence. Do not rerun expensive upstream work
merely to satisfy a downstream code change.

Typical commands:

```bash
# SAM3 rescue A/B after annotation/frames are already materialized
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages segment,rank \
  --profile

# Final-integrity-only replay after references/reference_edit are prepared
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages reference_integrity \
  --profile
```

If the run uses old `ClipRecord` JSON that predates newly required integrity
fields, do not blindly parse the whole historical clip under the newest schema
while extracting evidence. For read-only extraction, parse raw JSON and validate
only stable sections such as `AnnotationState`, `SampledFramesArtifact`, and
`TrackedMasksArtifact`.

For an actual replay run, create a fresh run root with a matching current
`run.json`. Preserve previous integrity rejects monotonically when appropriate,
remove stale reference-edit entries that point at no-longer-ready references,
clear only downstream states that are being recomputed, and require
`ClipRecord.model_validate()` before model calls. Never delete `run.json` from a
non-empty run root to bypass identity checks.

## 13. Current Five-Case Semantic Replay

Before the final 120 replay, validate only these existing cases against code
baseline `32a9e0e`:

```text
0f32c6b7fa9934c159a03ff7  sauce       -> deterministic reject
34da1ad5a39a0389de87568b  cathedral   -> deterministic reject
4e892f7740e1557b495a64da  dog object  -> deterministic reject
b527c92f98b7f27f1d301f7c giant clam  -> deterministic reject
82f312a07328785e228802d3  cooked lobster -> no deterministic reject
```

Expected counters for the four negative cases:

```text
semantic_policy_rejected = 4
Qwen integrity calls      = 0
source bbox calls         = 0
```

The cooked lobster must continue onto the normal integrity path.

Use a fresh targeted replay root derived from already-materialized source
artifacts; do not rerun annotation, SAM3, removal, pair, or Boogu for this check.

### 13.1 Final120 infrastructure and contact-sheet result

The final fixed-population integrity replay completed without infrastructure
failure:

```text
run_id: integrity-final-freeze120-20260813-202412
config_hash: 810fdb46cc7b03deaf411d84a13c2c285abde2f584dc42de4d05174505242c47

processed: 63
failed: 0
entities_reviewed: 72
entities_skipped_review: 10
entities_accepted: 82
entities_rejected: 4
semantic_policy_rejected: 4
judge_failed: 0
source_bbox_fallback_attempted: 1
source_bbox_fallback_accepted: 1
repair_count: 0
length_count: 0
```

Infrastructure therefore passed, including the structured-output termination
fix. The contact-sheet visual freeze did not pass: the young-boy reference with
contact-sheet label `7d3c89d8bb... e2` retained an irregular white mouth-region
blob, while `eace10dad52d7534c50dae01 e1` and `e2` retained white circular or
spherical placeholders after held-item removal.

After the prompt hardening in `b7e46fc`, run only a targeted replay of those
three references. Reuse existing evidence and do not rerun annotation, SAM3,
coverage, pair, remover, or Boogu. Another full 120 replay is not the current
next step.

## 14. Monitoring

```bash
pgrep -af 'run_pipeline_v3.py'
nvitop
nvidia-smi

tail -n 100 "$LOG"
tail -f "$LOG"
```

For a selected run root:

```bash
RUN=/mnt/workspace/litengjie/data/r2v_v3_runs/<exact-run>

find "$RUN/clips" -name masks.rle.json | wc -l
find "$RUN/clips" -name clip.json | wc -l

test -f "$RUN/profiling/events.jsonl" && \
  tail -n 20 "$RUN/profiling/events.jsonl"
```

SAM3 can run in the main pipeline process; absence of a separately named SAM3
process is not itself a failure.

## 15. Freeze Procedure

The visual V3 path is ready to freeze when all of the following are true:

1. five-case semantic replay matches all expected hard-gate outcomes;
2. no semantic policy reject calls Qwen or bbox fallback;
3. final 120 integrity replay has no infrastructure failure requiring a code
   change;
4. source bbox rescue remains narrow and does not revive semantic/identity
   failures;
5. contact-sheet/audit review finds no systematic regression;
6. exact freeze code commit, docs commit, config hash, fixed selection, run root,
   and export/audit paths are recorded in `V3_RUNTIME_INTEGRITY_STATE.md`.

After freeze, do not continue tuning visual V3 from isolated anecdotes. Resume
the H3/audio branch as a separate workstream.
