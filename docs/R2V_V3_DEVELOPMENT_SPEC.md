# R2V V3 Development Specification

Status: **approved original development specification (historical baseline)**
Audience: Codex and repository contributors  
Repository: `TjieLee/R2V_DATA_V2`  
Implementation target: a V3 pipeline developed alongside the existing V2 pipeline

> **Current V3 notice (2026-08-23):** This is the original approved V3
> development specification. Validated production work superseded some initial
> implementation choices, so this file must not be used alone as current
> operational truth. Current state is defined by
> `V3_RUNTIME_INTEGRITY_STATE.md`, `V3_SUBJECT_ATTRIBUTES_STATE.md`,
> `V3_PRODUCTION_SHARDS.md`, `SERVER_ENVIRONMENT_RUNBOOK.md`,
> `ANNOTATION_ENTITY_PRODUCTION.md`, and
> `VISUAL_ATTRIBUTE_DEVELOPMENT_HANDOFF.md`. In particular, original
> Qwen-Image-Edit background-removal text and original stage order are
> historical initial-design material wherever they conflict with those
> current-state documents.

Read `AGENTS.md` first. Use this specification for original design intent and
archaeology; where it conflicts with the current-state documents above, the
current-state documents take precedence.

---

## 1. Objective

Build a cleaner, quality-first R2V data construction pipeline with four deliberate changes:

1. use a stronger video annotation model for production captions and entity analysis;
2. separate the ordinary T2V caption from a reference-aware generation instruction;
3. replace FLUX background inpainting with an object-removal backend based on Qwen Image Edit;
4. classify entity references as full, local, or rejected, with an optional generated fallback only after real-reference selection fails.

The exported training dataset must be compact. It must not repeat the V2 pattern of many stage manifests, per-sample JSON files, duplicated canonical files, raw/repaired aliases, and diagnostics mixed into the final dataset.

---

## 2. Non-negotiable decisions

### 2.1 Production annotation model

The V3 production annotation model is:

```text
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

Rules:

- Use the dense 32B Instruct model as the default annotation model.
- Do not use the A3B model as the production default.
- Keep the existing 8B model only as a speed baseline, test endpoint, or fallback explicitly selected in configuration.
- Other Qwen 3.5/3.6 models may be benchmarked later, but they are not part of the first V3 implementation.
- Model paths under `/mnt/workspace/public/pretrained/**` are strictly read-only.
- Do not auto-download models or modify the server GPU software stack.

Only the annotation service is mandated to use 32B. Candidate judges, removal validators, and repair-format retries remain independently configurable and must not be silently switched to 32B.

### 2.2 Background generation policy

FLUX background inpainting is no longer the V3 production path.

- Keep the V2 FLUX implementation intact as a legacy baseline.
- Do not delete old FLUX code or historical pilot artifacts.
- V3 uses an object-removal stage named `remove`, not `inpaint`.
- A background containing any tracked foreground mask must never be exported as a clean raw background.
- Removal failure is fail-closed. Never fall back to a contaminated raw image.

The first removal backend to evaluate is:

```text
base model:
/mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511

LoRA adapter:
/mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover
```

The remove stage uses Qwen-Image-Edit-2511 with the required Object-Remover LoRA. The adapter is mandatory: a missing path, unsupported
format, load failure, inactive adapter, or unexpected active adapter fails
closed before inference. The pipeline never runs the base model alone, selects
a similarly named adapter, or downloads either component.

### 2.3 Entity completion policy

V3 does not synthesize missing entity parts by default. When the explicit
`allow_synthetic_completion` fallback is enabled, it may repair only an already
selected, identity-visible local reference after real self and same-parent donor
options are exhausted.

- Do not densely resample videos to search for more reference frames.
- Outside the explicit gated fallback, do not generate legs, clothing, object
  parts, or other unseen structure to create a supposedly complete reference;
  generated content is never treated as a real source reference.
- Preserve a sufficiently recognizable whole object as a full reference even if a peripheral part is clipped or occluded.
- Convert a semantically useful but incomplete entity into an honest local reference.
- Reject unusably fragmented references.

`full`, `local`, and `reject` are the only entity-reference quality
classifications. V3 does not add a second normalization-specific quality tier.

### 2.4 Caption policy

Every accepted sample has two distinct text fields:

- `t2v_caption`: literal, chronological, complete video description;
- `r2v_instruction`: generation-oriented English instruction using rendered
  image labels such as `<Image 1>`.

The R2V instruction must not be produced by mechanically inserting internal
pairing tokens into the T2V caption. Pairing tokens remain internal metadata and
never appear in the final instruction.

---

## 3. Scope and non-goals

### 3.1 In scope

- V3 schemas and configuration;
- stronger video annotation;
- full/local/reject entity reference classification;
- Qwen Image Edit object removal for backgrounds;
- post-pair R2V instruction generation;
- a compact run layout and compact exported dataset layout;
- unit tests and a small server pilot;
- migration-free coexistence with V2.

### 3.2 Out of scope for the first V3 implementation

- changing the source dataset;
- changing the fixed ten-frame SAM/ranking policy;
- dense temporal resampling;
- standalone synthetic-completion publication or manual-approval workflows;
- generated viewpoints or entity augmentation;
- cross-parent reference pairing;
- training-code changes in `R2V-Next`;
- automatic model downloads;
- a general plugin framework, database, workflow engine, or complex state machine;
- migration or rewriting of existing V2 pilot outputs.

---

## 4. Development isolation

V3 must be implemented alongside V2 until the V3 pilot passes.

Preferred new entry point:

```text
run_pipeline_v3.py
```

Preferred package location:

```text
r2v_data_v2/v3/
```

The V2 entry point, V2 schemas, and current V2 output roots must remain usable.

Codex should create focused V3 modules rather than rewriting every V2 module in place. Reuse stable pure utilities for video reading, SAM3 invocation, mask encoding, image I/O, and metrics where practical. Shared-module edits must be minimal and covered by V2 regression tests.

---

## 5. V3 stage order

The authoritative V3 stage order is:

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
-> instruct
-> export
```

There is no generic `reference_finalize` stage. Pairing publishes canonical
source-faithful references and routing decisions. The explicit
`reference_edit` stage may replace an accepted entity's published path with a
validated native Boogu `final_reference_1k.png` before instruction and export.

### 5.1 Stage responsibilities

#### `manifest`

Create source clip records only. Do not copy source videos.

#### `annotate`

Use full-video input with the 32B annotation model. Produce:

- `t2v_caption`;
- up to five provisional entity candidates;
- optional background semantics.

Do not generate reference tokens or `r2v_instruction` here.

#### `frames`

Sample exactly ten unique chronological JPEG frames from the complete source
video for SAM3 and ranking. Selection is deterministic over all decodable
frames, includes the first and last decoded frame, and records the source frame
index, real timestamp, relative image path, and SHA-256 in `frames/frames.json`.
This remains independent from Qwen video sampling. A video with fewer than ten
distinct decodable frames fails this stage for that clip.

#### `segment`

Run each annotation entity independently through the lazily loaded SAM3
text-grounded video predictor and store all ten slots in one strongly typed
clip-level `masks.rle.json`. After selecting an anchor, run forward and backward
propagation in independent sessions, reapplying the same prompt at the anchor in
each session. SAM3 object IDs are session-local: validate the two anchor masks
with IoU before remapping the backward track to the forward canonical ID. Merge
anchor, forward, and backward masks deterministically without allowing a lower
priority duplicate to overwrite an earlier result. The published anchor mask
comes exclusively from `add_prompt`; forward propagation owns only frames after
the anchor, and backward propagation owns only frames before it. Propagation
responses at the anchor are ignored because SAM3 may re-estimate a slightly
different boundary for the same frame. The independent forward and backward
`add_prompt` anchor masks are still checked for identity consistency at IoU
`>= 0.95`. A failed entity is recorded without blocking other entities; a clip
with zero entities publishes an empty, ready mask artifact without loading
SAM3. Single-subject, single-object, and current first-pass group tracking
retain exactly one identity rather than unioning unrelated detections.
Multi-object groups remain unverified and are rejected.

Conservative recall rescue is opt-in. With
`sam3.not_found_rescue_mode: entity_phrase_retry_v1`, a not-found subject or
object receives at most one retry using its annotation `phrase`; normalized
duplicate prompts and groups do not trigger another inference. Existing
`object_rescue_mode: phrase_retry_v1` remains compatible for object-only retry
and collision handling. With
`sam3.multi_instance_rescue_mode: qwen_anchor_select_v1`, each ambiguous
subject/object probe is rendered with numbered masks and sent to the configured
candidate-judge VLM. Only one explicit valid candidate ID may become the anchor;
reject, uncertainty, an invalid candidate ID, or judge failure makes only that
ambiguous probe unusable and anchor probing continues. If all configured probes
are exhausted after any ambiguity without a usable anchor, the entity fails
closed as `ambiguous_multi_object_instance`. Unique anchors make no VLM call,
groups are never rescued, masks are never selected by SAM score alone, and
candidates are never unioned.

If one propagation session changes object ID, the mismatched observation and
all later slots owned by that direction are invalidated. Earlier verified masks,
the `add_prompt` anchor, and the independent opposite direction remain eligible
for publication. The segment stage never remaps the changed ID or accepts later
observations; the unchanged 7/10 coverage stage decides whether the resulting
partial track is temporally sufficient. Segment counts record phrase retries,
multi-instance selection, identity switches, and ready versus insufficient
partial-track salvage.

After all annotation entities have been tracked, non-group ready tracks are
compared across entity IDs. A pair is a duplicate only with at least three
common present-valid frames, median mask IoU at least `0.85`, and at least 75%
of common frames at IoU `>= 0.80`. The track with more present-valid frames is
retained, followed by higher median published object confidence and then earlier
annotation order. The loser remains in artifact order with `status=failed` and
`reason=duplicate_cross_entity_track:<winner_entity_id>`. Masks are never
unioned by this gate, and group tracks are excluded.

SAM3 `out_probs` values are published object-score diagnostics propagated with
the track. They are not independently estimated per-frame tracking confidence,
and temporal visibility must not threshold on them. Visibility continues to
depend only on a ready entity, a present non-empty mask, and `track_valid`.

#### `rank`

The current implementation computes temporal coverage only from
`masks.rle.json`. It does not rerun SAM3, select canonical frames, classify
full/local/reject scope, or publish references. The later `pair` stage owns
candidate selection and Qwen owns the semantic full/local/reject decision.

#### `background`

This stage performs only deterministic source selection and construction of the
exact union foreground mask. It does not call a generation model, rerun SAM3,
dilate a mask, or create a generation mask. For every valid slot, ready entity
masks are combined with logical OR. The source frame has the smallest union
area, with the fixed center-first slot order used as the deterministic
tie-break.

- zero union area: `clean_raw`, directly reusing the sampled JPEG without a copy;
- union area ratio in `(0, max_pending_remove_area_ratio]`: `pending_remove`
  with an exact, binary source mask under `clip/background/`;
- union area ratio above `max_pending_remove_area_ratio`: reject the background
  with `foreground_mask_too_large`;
- any non-ready entity track: reject only the background reference with
  `incomplete_foreground_tracking`, while preserving the clip and entity refs.

The raw foreground threshold is fixed at zero. The pending-removal maximum
defaults to `0.50`. Dilation, generation masks, and image editing belong to the
later `remove` stage, which is fail-closed and has no raw fallback.

#### `remove`

Remove foreground entities from pending backgrounds using the configured Qwen Image Edit backend. Validate and publish only accepted background-only results.

#### `pair`

Select deterministic in-pair entity candidates from the ten sampled frames and
validated tracked masks, then ask the configured Qwen candidate judge for the
semantic full/local/reject decision. Code owns mask integrity, geometry, crop
publication, final retained filtering, optional ready-background binding, and
deterministic per-type tokens. Every ready entity reference is retained in
annotation order, and at least one retained entity must pass temporal coverage.
The stage does not resize references or perform cross-parent pairing. When the
optional same-parent fallback is enabled, its second phase may replace a rejected
or local target reference from an exact `parent_video_id` match after a dedicated
visual judge accepts the same physical entity. When `reference_edit.enabled` is
true, pair does not invoke the legacy localized Qwen completion fallback.
Legacy source and artifacts remain compatible for old runs. Pairing records
`image_quality`, viewpoint, independent reference value, substantial-invention
risk, and the completeness route consumed by the later stage. Rear-only subject
views, non-independent environment fragments, and candidates requiring major
invented identity or structure are rejected before publication. Real references
remain variable-size, source-faithful RGBA mask-bbox crops.
Candidate sharpness is the variance of a discrete Laplacian on the original RGB
frame inside a two-pixel-eroded entity mask, with the original mask used when
erosion leaves too few pixels. The deterministic shortlist order is border
contact, descending area ratio, descending sharpness, center distance, then the
fixed slot priority. Bbox fill remains diagnostic metadata only. Before the
shortlist is formed, candidates whose alpha/mask content bounding box is below
the configured area threshold or whose longest side is below the configured
long-side threshold are removed. If every candidate is tiny, the entity is
rejected with `tiny_reference_candidates` before the semantic judge runs.
For each non-tiny candidate, NumPy connected-component diagnostics record the
significant component count and the largest two component area ratios. A
component is significant at `max(16 pixels, 2% of foreground area)`. Non-group
candidates are filtered before the shortlist when the largest component is
below `0.70`, the second-largest exceeds `0.20`, or more than three components
are significant. If this removes every non-tiny candidate, the reason is
`fragmented_reference_candidates`. Groups retain the diagnostics but bypass
this hard fragmentation gate.

For subjects only, a non-severe deterministic signal marks the gray zone where
the main component remains dominant but a non-trivial secondary component may
be a detached part of the same target. The signal is true when significant
component count is at least two, largest-component ratio is at least `0.70`,
and second-largest-component ratio is from `0.05` through `0.20`, inclusive.
It is included in the existing Qwen request and never rejects a candidate by
itself. Object and group candidates do not use this deterministic signal.

The existing single Qwen candidate judgment also reports whether the primary
identity region and major structure are visible, whether the crop is a discrete
foreground instance, whether the mask matches the target, and whether
truncation is `none`, `minor`, or `major`. It separately reports
`completion_needed_for_reference_use`, which means generative completion is
strictly necessary before the visible crop can serve as a training reference.
Code rejects any non-reject result with a failed evidence boolean or major
truncation. `complete` requires no truncation and no completion. `local_usable`
allows none or minor truncation but must not need completion or contain detached
target fragments. A selected subject carrying the deterministic detached-part
signal cannot publish as `complete` or `local_usable`; the existing structured
repair retry asks Qwen to correct the route. `repairable` requires minor
truncation and completion necessity, and may explicitly report
`detached_target_fragments_present=true` when identity and major structure are
preserved. These optional state fields are persisted for audit while legacy
clip JSON without them remains readable.

#### `reference_edit`

Run after pairing and before instruction generation. `complete` references run
background generation only. `repairable` references run completion, Qwen/SAM
completion review, then background generation from the accepted completion
candidate. `local_usable` references run background generation but remain local
references; their visible-region and whole-entity fields are not promoted.
Severely incomplete or fragmented references do not enter the stage. One
persistent JSONL worker loads Boogu once for all eligible entities and both
repairable operations reuse it. Accepted native RGB output is published as
`reference_edit/<entity_id>/final_reference_1k.png`. If repairable completion
passes but background generation fails, the completion candidate is the
explicit fallback. Qwen and SAM3 are independent production guards. SAM3 masks
are review-only, and no mask paste-back or foreground pixel restoration is
permitted. Source alpha-bbox geometry rejects tiny inputs before a Boogu
request. Candidate SAM masks must retain at least the configured normalized
bbox-area ratio and stay within the configured normalized center distance. A
background is an optional enhancement: it publishes only when the existing
Qwen review says that identity, scale, layout, background coherence, and
reference usefulness pass and the candidate is preferable to the source.
Tiny ready references still count toward `reference_edit.entities_eligible`,
then take the deterministic `keep_source` fallback without initializing Boogu,
Qwen review, or SAM3 runtime. Runtime initialization occurs lazily at the first
non-tiny entity that actually needs generation and happens at most once per
stage. Entity counters are committed only with the successful clip result.
If a `repairable` completion is rejected or fails, its source remains immutable
on disk but the entity is rejected and removed from retained pairing IDs and
tokens. `keep_source` does not override this publication gate. If completion is
accepted and only the later background operation fails, the accepted completion
candidate remains the explicit fallback. Complete and local-usable background
failures continue to keep their source references.

#### `instruct`

Generate `r2v_instruction` only after the final reference set is known.

#### `export`

Write the compact final dataset atomically. No other stage writes final training records.

---

## 6. Model service configuration

V3 keeps independently configurable services.

Recommended configuration shape:

```yaml
qwen:
  annotation:
    base_url: http://127.0.0.1:8000/v1
    model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
    temperature: 0.0
    max_tokens: 4096
    timeout_seconds: 3600
    video:
      input_mode: full_video
      fps: 2.0
      do_sample_frames: false

  instruction_writer:
    base_url: http://127.0.0.1:8000/v1
    model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
    temperature: 0.0
    max_tokens: 1024
    timeout_seconds: 3600

  candidate_judge:
    # independently configurable

  background_remove_judge:
    # independently configurable

  cross_pair_judge:
    # required explicitly only when same-parent fallback is enabled

  reference_edit_judge:
    # required explicitly only when reference_edit is enabled
```

`qwen.cross_pair_judge` never falls back to `candidate_judge` or
`instruction_writer`. Multiple explicitly configured services may point to the
same vLLM endpoint.

`full_video` means that the pipeline submits the complete local video file.
vLLM's media I/O layer decodes and samples the video at the configured 2 FPS
before passing the resulting frames to the Transformers processor.
`do_sample_frames` must remain `false` to prevent the Transformers processor
from sampling those already sampled frames a second time. The vLLM service
requires these media options:

```text
--media-io-kwargs '{"video":{"num_frames":-1}}'
--allowed-local-media-path /mnt/workspace/public/dataset
```

Choose `tensor-parallel-size` at deployment time based on the GPUs actually
assigned to the service; V3 does not prescribe a fixed value.

The instruction writer may use a text-only request built from the validated annotation and final bindings. It does not need to resubmit the video unless a later benchmark proves that video input materially improves instruction quality.

---

## 7. Annotation schema V3

The annotation stage returns semantic content only.

Qwen's raw response contains no entity IDs:

```json
{
  "t2v_caption": "...",
  "entities": [
    {
      "reference_type": "subject",
      "phrase": "...",
      "grounding_prompt": "..."
    }
  ],
  "background": {
    "phrase": "...",
    "grounding_prompt": "..."
  }
}
```

`reference_type` is limited to `subject`, `object`, or `group`. After local
candidate validation, phrase deduplication, and truncation to five candidates,
code assigns contiguous IDs from `e1` through at most `e5`. Invalid candidates
are dropped without discarding a valid caption. An invalid background is
normalized to `null`. A valid caption with zero retained entities is still a
ready annotation; later stages own reference eligibility and clip rejection.

The persisted entity schema contains only `entity_id`, `reference_type`,
`phrase`, and `grounding_prompt`. Relations and the previous category,
salience, genericity, evidence, visual-scope, separability, and selection-reason
ontology are not part of V3 annotation schema version 2.

### 7.1 `t2v_caption` requirements

- one flowing English paragraph;
- 60 to 110 English words recommended, with a hard maximum of 120;
- literal and chronological;
- complete visible semantics;
- actions, camera behavior, environment, lighting, and stable appearance details;
- no reference tokens;
- no generation command wording solely to make it look instructional;
- describe visible motion without assigning an unseen cause;
- no unsupported identity, weather cause, allegiance, intent, mental state,
  emotion, sound, or dialogue;
- no hedging or causal inference wording such as `breeze`, `wind-induced`,
  `suggesting`, `indicating`, `possibly`, `probably`, or `likely`;
- no unsupported role labels such as enemy, ally, criminal, victim, or officer;
- describe statue and depicted-figure geometry and pose without inferring
  determination, resolve, triumph, fear, or effort.

Code applies a deliberately small, case-insensitive word-boundary check to
`t2v_caption` only. It rejects the listed inference words plus `wind causes`
and `caused by wind` with `unsupported_caption_inference`, which enters the
existing repair lifecycle. This is not an entity ontology or object-name
blacklist. Directly visible wording such as `branches sway slightly`, `sunny`,
`overcast`, `cloudy`, `speaks`, and `talking` remains valid.
Captions above the hard maximum produce `caption_too_long`; code never silently
truncates them.

### 7.2 Candidate sanitation

Entity candidates are processed in model order. Code strips and normalizes
candidate text, drops invalid candidates, deduplicates normalized phrases while
keeping the first occurrence, retains at most five, and assigns entity IDs
after all filtering. Phrase text is not required to match one exact contiguous
caption span. Annotation remains separate from final reference eligibility.
Entity phrases should normally be stable noun phrases rather than actions.
Phrases should contain 3 to 10 English words and have a hard maximum of 12.
Grounding prompts should contain 6 to 18 words and have a hard maximum of 24;
they describe stable appearance and location rather than transient action or an
inventory of every clothing detail. `seated` and `standing` remain valid stable
disambiguation. The conservative transient-action check covers `gesturing`,
`speaking`, `talking`, `turning`, `walking`, `running`, `raising`, `holding up`,
and `moving his/her hand`. Violations produce `entity_phrase_too_long`,
`grounding_prompt_too_long`, or `transient_action_in_grounding_prompt` and enter
the existing repair lifecycle without truncation or local candidate dropping.

### 7.3 Background stability

Return background semantics only when one stable environment persists through
most of the clip. A major transition between different environments requires
`background=null`. Invalid background output still degrades locally to `null`
without failing an otherwise valid caption.

---

## 8. Reference scope V3

Every selected entity candidate receives:

```json
{
  "reference_scope": "full|local|reject",
  "visible_region": "whole|head_shoulders|upper_body|lower_body|front|rear|side|central|custom",
  "whole_entity_recognizable": true,
  "identity_features_visible": true,
  "viewpoint": "front|three_quarter|side|rear|not_applicable",
  "independent_reference_value": true,
  "requires_substantial_invention": false,
  "scope_reason": "..."
}
```

Subjects use a directional viewpoint; objects and groups use `not_applicable`.
Rear-only subjects reject. A side subject is usable only when a face or another
explicit identity feature is visible. Clothing or a visible back alone is not
identity evidence. A ready reference must have independent reference value and
must not require substantial invented content. The three persisted judge fields
are optional only so legacy `clip.json` records created before this contract
remain loadable; newly judged ready and rejected states record them, cross-pair
references inherit them, and reference editing does not alter them.

### 8.1 `full`

Use `full` when the overall entity remains recognizable and its characteristic appearance is sufficiently represented.

`full` does not require every peripheral component to be visible. Examples:

- a picnic table remains `full` when one peripheral seat is partly clipped but the complete table structure, material, color, and major seats remain recognizable;
- a vehicle may remain `full` when a small rear portion leaves the frame but the whole vehicle identity and structure are clear.

Do not downgrade solely because:

- the mask has natural holes;
- thin structures form multiple connected components;
- a peripheral part is occluded;
- the entity touches the image border.

### 8.2 `local`

Use `local` when a coherent, semantically useful region is available but the whole entity is not reliably represented.

Examples:

- upper body of a person whose lower body is heavily occluded;
- head and shoulders with stable identity and clothing details;
- a recognizable front section of a vehicle;
- the central product body when accessories are hidden.

For a local reference:

- crop the exact selected tracked-mask bbox plus deterministic padding;
- preserve natural holes, thin structures, and disconnected mask components;
- store the visible region explicitly;
- never describe the result as a full-body or complete-object reference.

The pair stage performs only the deterministic severe-fragmentation gate above;
it does not attempt semantic component pruning. Qwen still judges fragmentation
and target mismatch from context, isolated crop, and component diagnostics.

Completeness routing is intentionally conservative. `local_usable` includes
natural local framing: a coherent head-and-shoulders, upper-body, waist-up,
side, three-quarter, or similarly identity-bearing region can be independently
reusable even when the rest of the physical entity is outside the camera frame.
Partial body visibility alone is not a defect, and minor truncation does not by
itself imply completion. This natural framing requires a coherent visible region
without non-trivial pieces of the same target detached elsewhere in the mask.
`repairable` is limited to a minor local low-risk omission or detached target
part, such as a clipped hand or foot terminal, short limb edge, detached hand or
trouser piece, or small object part, when the source still preserves identity
and major structure and repair does not require guessing them. Missing identity,
the head, most of a body or garment, torso-only/back-only fragments, or any
repair requiring major invention is `severely_incomplete` and rejects. Bad
masks, environment fragments, and non-target content are `fragmented` and
reject.

### 8.3 `reject`

Reject when:

- identity features are not visible;
- remaining components are too fragmented to form a useful local region;
- the mask is dominated by occluders or segmentation errors;
- the selected crop cannot be understood without guessing missing structure.

### 8.4 Decision ownership

Qwen provides semantic scope judgment. Code validates geometry and existing hard gates.

Connected-component count alone is diagnostic only because many valid objects
naturally have separated or thin parts. The bounded subject-only signal combines
count with largest and second-largest component ratios, and only prevents direct
`complete` or `local_usable` publication; it does not hard reject or rewrite the
semantic decision.

### 8.5 Optional generated fallback

V3 must not add a second entity-repair implementation. The optional fallback
reuses the existing localized Qwen completion backend and SAM3 backend, and it
runs only after real self and same-parent donor selection. Real references
export with `synthetic=false`; accepted generated fallbacks export with
`synthetic=true`. No manual approval state or standalone publication workflow is
part of production.

---

## 9. Background object removal

### 9.1 Background source lifecycle

Background state is one of:

```text
none
clean_raw
pending_remove
ready_removed
rejected
```

Rules:

- Union foreground mask empty: `clean_raw`.
- Union foreground mask non-empty: `pending_remove`.
- `raw_foreground_area_ratio` is not used to permit visible foreground leakage.
- Removal accepted: `ready_removed`.
- Removal rejected or failed: `rejected`.
- `rejected` backgrounds never enter final samples.

For V3, the effective raw threshold is always zero. The default V2 value must not leak into V3 behavior.

### 9.2 Removal backend interface

Implement a backend protocol similar to:

```python
class BackgroundRemovalBackend(Protocol):
    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image: ...
```

The first backend identifier is:

```text
qwen_image_edit_2511_object_remover
```

The backend is Qwen-Image-Edit-2511 with the required Object-Remover LoRA and
loads lazily only after the `remove` stage encounters work. It loads the local
base pipeline with `local_files_only=True`, loads the verified adapter as
`object_remover`, activates it, queries active adapters, and allows inference
only when `object_remover` is active. A single adapter file is loaded from its
parent with an explicit `weight_name`; an ambiguous directory requires
`remove.adapter_weight_name`. There is no base-only, FLUX, CPU, or Hub
fallback.

Each source mask is deterministically dilated by
`remove.generation_mask_dilation_pixels` to form a distinct generation mask.
The generation mask must contain the source mask, remain within
`remove.max_generation_mask_area_ratio`, and is stored as a content-addressed
single-channel PNG. Whole-image model output is always locally composited so
pixels outside the generation mask remain exactly equal to source pixels.

At most two configured seeds are tried in order. The first candidate passing
hard local checks and the configured Qwen background-removal judge wins.
Malformed judge output fails that candidate; no configured or injected judge
fails the clip explicitly. If no candidate is accepted, the background becomes
`rejected` with all attempts recorded. There is no raw fallback.

### 9.3 Prompt contract

The prompt must name the actual foreground entities when reliable semantic labels exist and must explicitly prohibit replacement objects.

Required intent:

```text
Remove the specified foreground subjects from the image.
Replace the removed regions only with a seamless continuation of the surrounding background.
Preserve geometry, perspective, depth, texture, lighting, color, shadows, and camera characteristics.
Do not add a person, animal, vehicle, product, text, sign, or any other salient object.
```

### 9.4 Mask use and pixel preservation

The backend API may be native mask-conditioned editing or whole-image editing, depending on verified compatibility with the installed model and adapter.

Regardless of backend mode, the published candidate must be locally composited:

```text
outside generation mask = exact original pixels
inside generation mask  = edited pixels
```

The generation mask may be a validated dilation of the source foreground mask. Its derivation and hash belong in clip working metadata.

### 9.5 Removal acceptance

Accept only when all conditions pass:

- original foreground is absent;
- original foreground was not reconstructed;
- no new salient entity appears in the removal region;
- the edited region contains background scenery only;
- background continuity passes;
- no visible seam, ghosting, double exposure, artificial blob, or major texture/color discontinuity;
- pixels outside the effective generation mask are identical to the source image.
- the Qwen judge accepts all six strict review booleans.

No condition may be bypassed to increase recall. There is no raw fallback.

### 9.6 Initial benchmark gate

Before enabling the backend in a production pilot:

- run it on the same 20 non-empty background masks used in the FLUX pilot;
- generate at most two candidates per reference initially;
- compare against the FLUX baseline of one accepted sample out of twenty;
- manually inspect every accepted candidate;
- require zero known false positives;
- record rejection reasons and runtime.

Do not integrate the backend into full-data execution before this benchmark is reviewed.

---

## 10. Pairing and clip coverage

V3 preserves the corrected clip-level coverage semantics:

- sample ten frames;
- a clip passes when at least one annotated entity reaches
  `coverage.required_visible_frames`;
- the default is seven of ten frames, and the integer threshold may be changed
  from 1 through 10 without rerunning SAM3;
- once a clip passes, other shorter-lived entities may remain if they have ready references;
- the final sample must bind at least one qualifying entity;
- an entity without a final reference remains in natural language but has no token;
- token assignment and final reference filtering happen together.

Pairing has three deterministic priority phases and runs after `remove`. Phase A
is the existing in-pair candidate selection and full/local/reject decision.
Phase B is disabled by default. When `pair.same_parent_fallback_enabled` is true,
it considers rejected targets and lower-priority local self references. A ready
full real self reference is immutable; an accepted same-parent full real donor
may replace a local self reference.

The legacy Phase C localized-completion fallback remains available only to runs
that do not enable `reference_edit`. Production runs with `reference_edit`
enabled stop after real self and same-parent donor selection, then defer any
generative work to the explicit stage.

The resulting priority is: full real self, then same-parent full real donor,
then the configured post-pair reference edit, then the original local/no
reference outcome. Generated references cannot enter the donor index. An
accepted Boogu sidecar records source, candidate, model, review, and hash
provenance. Instruction and export consume the final path in `clip.json`, which
is either the source-faithful selected reference or accepted
`final_reference_1k.png`.

One clip may retain at most five entity references, one per annotation entity.
An optional background is separate from that limit, so a final sample may
contain five entity references plus one background reference, for six total.
`pair.max_candidates_per_entity: 3` remains the per-entity candidate-frame
shortlist sent to the visual judge; it is not the per-video reference limit.

A donor must use the exact same `source.parent_video_id`, have a different
`clip_uid`, ready annotation and pairing, and a retained full, identity-visible
reference of the same type. Local/background references and invalid or missing
PNGs are ineligible. Donors are ordered naturally by `clip_suffix`, then by
`clip_uid` and donor `entity_id`, and capped by
`pair.same_parent_max_donor_references`. Cross-parent and fuzzy parent matching
are forbidden.

The dedicated visual judge receives one target context frame, one target crop,
and the donor source-faithful RGBA reference shown on white. It accepts only the
same physical entity with matching identity features and a usable donor. A
rejection advances to the next donor; uncertainty fails closed. On acceptance,
the donor PNG is copied byte-for-byte to the target's canonical `selected/eN.png`
without resize or re-encoding. Target tokens and instruction semantics remain
target-owned. `source_clip_uid` and `source_entity_id` record the donor, while
legacy references may omit both fields. Publication is transactional and rolls
back target files when the authoritative clip write fails.

---

## 11. R2V instruction generation

### 11.1 Timing

Generate `r2v_instruction` after pairing. At this point the exact final references and tokens are known.

### 11.2 Instruction input

Code assigns one deterministic English image binding per final reference. Entity
references retain pairing order and an accepted background is last:

```json
{
  "t2v_caption": "...",
  "bindings": [
    {
      "image_id": "image_1",
      "image_index": 1,
      "reference_type": "subject",
      "entity_id": "e1",
      "phrase": "...",
      "grounding_prompt": "..."
    },
    {
      "image_id": "image_2",
      "image_index": 2,
      "reference_type": "background",
      "entity_id": null,
      "phrase": "...",
      "grounding_prompt": "..."
    }
  ],
  "source_transcript": null
}
```

All JSON field names, enum values, identifiers, and placeholders are English.
The model cannot change binding IDs, indexes, order, types, or entity IDs.
`source_transcript` is supplied only when source metadata contains explicit
transcript or dialogue text.

### 11.3 Structured output and rendering

The instruction writer returns:

```json
{
  "instruction_body_template": "... {{image_1}} ...",
  "reference_legend": [
    {
      "image_id": "image_1",
      "description": "..."
    }
  ]
}
```

`instruction_body_template` and every legend description are English. The body
is normally 80 to 160 English words with a hard maximum of 180. It preserves
the core action, needed spatial relationships, camera, composition, lighting,
and chronology without copying the caption or restating full reference
appearance. It may use the same image placeholder multiple times and must use
every final binding at least once. Legend descriptions contain stable visual
appearance only, are normally 8 to 20 words, and have a hard maximum of 24.
Without `source_transcript`, the body cannot invent quoted dialogue.

Code performs the only presentation-layer conversion:

```text
{{image_1}} -> <Image 1>
{{image_2}} -> <Image 2>
```

It then appends the legend in binding order:

```text
<rendered instruction body>

<Image 1>: <description>
<Image 2>: <description>
```

Rendered image labels are never schema identifiers, enum values, binding IDs,
or raw model placeholders. Internal `<ref_...>` pairing tokens remain separate
and do not appear in the structured instruction output or rendered instruction.

### 11.4 Instruction validation

Validate raw structured output before deterministic English rendering:

- reject a body above 180 English words with `instruction_body_too_long`;
- reject a legend description above 24 words with
  `legend_description_too_long`;
- send either issue through the existing instruction repair lifecycle without
  truncating model output;

- the body template is non-empty and uses only exact `{{image_N}}` placeholders;
- every binding appears at least once; repeated placeholders are allowed;
- no unknown placeholder or `<ref_...>` token appears;
- raw output contains no rendered `<Image N>`, plain `Image N`, or Chinese `图N`
  label;
- the body and every legend description contain no CJK characters;
- legend count, IDs, and order exactly match bindings;
- every legend description is non-empty;
- without a source transcript, quoted dialogue is forbidden;
- the body is not a verbatim copy of `t2v_caption`.

Validation failures enter one structured repair attempt. Exhausted repair writes
`InstructionState(status="failed")` without changing the ready annotation.

---

## 12. Clean storage architecture

V3 separates ephemeral run artifacts from the final training dataset.

Two roots are required:

```yaml
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/<run_id>
dataset_root: /mnt/workspace/litengjie/data/r2v_v3_datasets/<dataset_version>
```

Both must remain under the writable user root.

### 12.1 Run root

The run root contains resumable computation state and diagnostics. It is not training data.

Required layout:

```text
<run_root>/
├── run.json
├── failures.jsonl
└── clips/
    └── <clip_uid>/
        ├── clip.json
        ├── background/
        │   ├── source_mask_<sha256>.png
        │   └── generation_mask_<sha256>.png
        ├── frames/
        │   ├── 00.jpg
        │   ├── 01.jpg
        │   ├── ... 09.jpg
        │   └── frames.json
        ├── masks.rle.json
        ├── selected/
        │   ├── e1.png
        │   ├── e2.png
        │   └── bg_removed.png
        └── debug/                 # created only when debug saving is enabled
            ├── segment/           # per-slot overlays and entity contact sheets
            ├── pair/              # optional requests, responses, and contact sheets
            └── remove/
                ├── candidate_seed_0.png
                └── review_seed_0.json

```

Rules:

- One `clip.json` is the authoritative metadata record for the clip.
- Do not create separate `annotations.json`, `ranking_metadata.json`, `reference_metadata.json`, `inpainting_metadata.json`, per-stage sample JSON, and repeated JSONL records for the same clip.
- Do not copy the ten sampled frames into additional candidate directories.
- Publish sampled JPEGs atomically and publish `frames.json` last. Its image
  paths are relative to the clip directory.
- Store all tracked masks in one `masks.rle.json` per clip.
- `masks.rle.json` stores every ordered slot, including absent masks, and uses
  validated two-dimensional binary run-length encoding at the sampled frame
  dimensions.
- Store only selected entity images in `selected/`. Each ready entity uses the
  fixed `selected/eN.png` path and is an RGBA PNG cropped without resize. Alpha
  is the exact binary tracked-mask crop; opaque RGB equals the sampled source,
  and transparent RGB is white.
- Entity reference dimensions remain the variable mask-bbox crop dimensions.
  Do not upscale, resize, place them on a 1024-by-1024 canvas, or create a
  second normalized entity artifact.
- Pair publication validates temporary PNGs, backs up only entity PNGs, updates
  references and pairing in one atomic clip write, and restores the old files
  and state on failure. It never deletes `bg_removed.png` or other selected files.
- Same-parent publication copies accepted donor bytes to the target canonical
  path with a temporary file plus atomic rename, verifies byte equality before
  and after publication, and never modifies the donor artifact.
- New in-pair references record self/self source provenance. Same-parent
  references record donor clip/entity provenance. Legacy artifacts with neither
  provenance field remain readable; exactly one field is invalid.
- `background/source_mask_<sha256>.png` is the exact, single-channel 0/255
  union mask used by `pending_remove` and oversized audit states.
- Background source masks are content-addressed and atomically published; stale
  hashes are removed only after `clip.json` successfully references the new state.
- Pending source masks never belong in `selected/`.
- The `remove` stage owns deterministic mask dilation and stores only the
  accepted content-addressed `generation_mask_<sha256>.png`.
- Successful publication is transactional: output replacement, generation
  mask publication, and `clip.json` update roll back together on failure.
- Store only an accepted removed background in `selected/bg_removed.png`.
- A clean raw background points to its selected sampled frame and does not require a duplicate image in `selected/`.
- Rejected removal candidates and reviews are written under `debug/remove/`
  only when `debug.save_diagnostics` or `remove.save_rejected_candidates` is true.
- Default production runs must not create `debug/`.

### 12.2 `clip.json`

`clip.json` consolidates stage state:

```json
{
  "schema_version": "r2v.v3.clip.2",
  "clip_uid": "...",
  "source": {
    "video_path": "...",
    "parent_video_id": "...",
    "clip_suffix": "..."
  },
  "annotation": {
    "status": "ready|failed",
    "t2v_caption": "...",
    "entities": [],
    "background": null
  },
  "coverage": {
    "passed": true,
    "qualifying_entity_ids": ["e1"],
    "required_visible_frames": 7,
    "entity_visibility_summary": {
      "e1": {
        "status": "ready",
        "visible_frame_slots": [0, 1, 2, 3, 4, 5, 6],
        "visible_frame_count": 7,
        "coverage_ratio": 0.7,
        "qualifies": true,
        "per_frame_area_ratio": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        "per_frame_confidence": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, null, null, null]
      }
    }
  },
  "references": {
    "entities": [],
    "background": null
  },
  "pairing": {
    "status": "ready|rejected",
    "retained_entity_ids": ["e1"],
    "tokens": {"e1": "<ref_subject_1>"}
  },
  "reference_edit": {
    "status": "ready|failed",
    "entities": []
  },
  "instruction": {
    "status": "ready|failed",
    "instruction_body_template": "... {{image_1}} ...",
    "reference_legend": [
      {
        "image_id": "image_1",
        "description": "..."
      }
    ],
    "r2v_instruction": "..."
  },
  "export": {
    "accepted": true,
    "reason": null
  }
}
```

`per_frame_confidence` is a compatibility field name in the current schema. For
SAM3 masks it stores the per-frame published object-score diagnostic described
above, not an independent tracking-confidence estimate.

This is a simple consolidated record, not a generic workflow engine. Each stage updates only its owned section using atomic file replacement.

`r2v.v3.clip.2` is intentionally incompatible with annotation smoke runs
written using `r2v.v3.clip.1`. Start a new run root instead of migrating or
overwriting an older run.

### 12.3 Run-level files

`run.json` contains:

- run ID;
- creation time;
- Git commit;
- config hash;
- model identifiers;
- source manifest path;
- counts by stage/status.

`failures.jsonl` contains one structured failure record per failed clip or stage. Do not create a separate log file for every failure category.

Human-readable process logs may still be produced with shell `tee` outside the run root.

---

## 13. Final dataset layout

The final training dataset contains only three elements:

```text
<dataset_root>/
├── dataset.json
├── samples.jsonl
└── references/
    └── <sample_id>/
        ├── subject_1.png
        ├── object_1.png
        ├── group_1.png
        └── background_1.png
```

Only files referenced by an accepted sample may exist under `references/`.
Each retained entity and background is exported exactly once. Do not create
separate source and normalized copies such as `*.source.png` plus `*.png`.

Do not export:

- sampled frames;
- masks;
- duplicate source/normalized reference variants;
- rejected images;
- candidate images;
- contact sheets;
- heatmaps;
- embeddings;
- Qwen raw responses;
- warnings;
- ranking scores;
- per-sample JSON files;
- per-stage manifests;
- duplicated `canonical.jpg`, `canonical_raw.jpg`, or `canonical_repaired.png` aliases.

### 13.1 `dataset.json`

```json
{
  "schema_version": "r2v.v3.dataset.1",
  "dataset_version": "...",
  "created_at": "...",
  "git_commit": "...",
  "config_hash": "...",
  "annotation_model": "Qwen3-VL-32B-Instruct",
  "background_remove_backend": "qwen_image_edit_2511_object_remover",
  "sample_count": 0,
  "reference_count": 0
}
```

Do not place private server endpoints, API keys, or absolute writable work paths in `dataset.json`.

### 13.2 `samples.jsonl`

Each line is one compact training record:

```json
{
  "schema_version": "r2v.v3.sample.1",
  "sample_id": "<clip_uid>",
  "target_video": "/mnt/workspace/public/dataset/.../clip.mp4",
  "t2v_caption": "...",
  "r2v_instruction": "Use <Image 2> as the background while <Image 1> walks forward.\\n\\n<Image 1>: ...\\n<Image 2>: ...",
  "references": [
    {
      "token": "<ref_subject_1>",
      "type": "entity",
      "entity_id": "e1",
      "scope": "local",
      "visible_region": "upper_body",
      "image_path": "references/<sample_id>/subject_1.png",
      "source_frame_index": 128,
      "source_clip_uid": "<donor_or_self_clip_uid>",
      "source_entity_id": "<donor_or_self_entity_id>",
      "synthetic": false
    },
    {
      "token": "<ref_bg_1>",
      "type": "background",
      "entity_id": null,
      "scope": "scene",
      "visible_region": "whole",
      "image_path": "references/<sample_id>/background_1.png",
      "source_frame_index": 128,
      "synthetic": true
    }
  ],
  "source": {
    "parent_video_id": "...",
    "clip_suffix": "..."
  }
}
```

Keep the final schema minimal. Entity scores, judge reasons, masks, and removal
metadata remain in `clip.json` unless a downstream training consumer explicitly
requires them.

### 13.3 Reference file format

- Export references as lossless PNG.
- Export each retained entity as the single final variable-size RGBA PNG produced
  by pairing. Real references are source-faithful; an accepted generated fallback
  carries `synthetic=true` and complete run-side provenance.
- Preserve the exact crop geometry, source RGB under the binary mask, white RGB
  where alpha is zero, and binary alpha; do not resize or normalize entities.
- Export backgrounds as RGB PNG.
- The exporter must not silently alter crop geometry, colors, or alpha semantics.
- Reference paths in `samples.jsonl` are relative to `dataset_root`.

### 13.4 Atomic export

Export into a temporary sibling directory, validate it, then atomically rename it to `dataset_root`.

If `dataset_root` already exists, do not delete or overwrite it without an explicit `--overwrite` flag and a clear printed target path.

---

## 14. Configuration V3

Recommended top-level shape:

```yaml
dataset_json: /mnt/workspace/public/dataset/jea-video/zicai_5th_moive/train_zicai_5th_moive.json
run_root: /mnt/workspace/litengjie/data/r2v_v3_runs/pilot40
export_root: /mnt/workspace/litengjie/data/r2v_v3_datasets/pilot40-v1

source:
  start_index: 0
  limit: 5
  allow_full_run: false

frames:
  count: 10

sam3:
  backend: sam3
  model_path: /mnt/workspace/litengjie/data/models/sam3/checkpoint.pt
  device: cuda
  save_debug_overlays: false
  object_rescue_mode: off
  not_found_rescue_mode: off
  multi_instance_rescue_mode: off
  anchor_search_mode: legacy

coverage:
  required_visible_frames: 7

reference_scope:
  enabled: true
  allow_local: true
  allow_synthetic_completion: false

pair:
  enabled: true
  max_candidates_per_entity: 3
  crop_padding_ratio: 0.08
  repair_retries: 1
  same_parent_fallback_enabled: false
  same_parent_max_donor_references: 8

background:
  enabled: true
  raw_foreground_area_ratio: 0.0
  max_pending_remove_area_ratio: 0.50

remove:
  enabled: true
  backend: qwen_image_edit_2511_object_remover
  base_model_path: /mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511
  adapter_path: /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover
  candidate_seeds: [0, 17]
  fallback_to_raw: false
  preserve_unmasked_pixels: true
  device: cuda
  dtype: bfloat16
  num_inference_steps: 40
  true_cfg_scale: 4.0
  guidance_scale: 1.0
  negative_prompt: " "
  generation_mask_dilation_pixels: 16
  max_generation_mask_area_ratio: 0.65
  adapter_weight_name: null
  save_rejected_candidates: false

reference_edit:
  enabled: false
  backend: boogu_image_0_1_edit_turbo
  python_executable: /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
  code_root: /mnt/workspace/litengjie/data/vendor/Boogu-Image
  model_path: /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
  model_revision: hotfix-1k-20260708
  cuda_visible_devices: "0"
  target_area: 1048576
  alignment: 16
  timeout_seconds: 3600
  add_background_to_complete: true
  fallback_policy: keep_source
  scale_collapse_fallback_guard_mode: "off"
  sam_max_area_growth_ratio: 3.0
  sam_max_significant_components: 4
  min_source_content_area_pixels: 16384
  min_source_content_long_side_pixels: 128
  min_candidate_scale_ratio: 0.60
  max_candidate_center_shift: 0.20

instruction:
  enabled: true
  repair_retries: 1

debug:
  save_diagnostics: false
```

The SAM3 checkpoint path above is an example within the writable model root.
Verify the installed SAM3 package and checkpoint path on the server before the
first real run. The adapter imports and loads the model only when `segment`
needs it and passes only arguments exposed by the installed builder API.

Validation must reject:

- writable output roots outside `/mnt/workspace/litengjie/data/**`;
- model downloads into public paths;
- `coverage.required_visible_frames` outside 1 through `frames.count`;
- non-boolean `reference_scope` switches, or synthetic completion enabled while
  `pair.enabled` is false;
- non-boolean `pair.enabled`, candidate limits outside 1 through 10,
  non-finite/non-float crop padding outside 0 through 0.5, or negative/non-integer
  pair repair retries;
- `remove.fallback_to_raw: true`;
- remove candidate seed lists other than one or two unique non-negative integers;
- unsupported remove dtypes, non-finite guidance values, or invalid mask limits;
- an empty `remove.adapter_weight_name`; adapter existence is checked lazily by
  the real backend, where a missing required adapter fails closed.
- enabled `reference_edit` without a dedicated Qwen judge, local worker paths,
  revision `hotfix-1k-20260708`, 16-pixel alignment, positive timeout and target
  area, or an explicit `keep_source|reject_entity` fallback policy;
- non-positive or non-integer reference-edit source geometry thresholds,
  `min_candidate_scale_ratio` outside `(0, 1]`, or
  `max_candidate_center_shift` outside `[0, sqrt(2)]`;
- nonzero `background.raw_foreground_area_ratio` for V3;
- an empty `source.limit` unless `source.allow_full_run` is `true`;
- a non-positive or non-integer `source.limit`;
- a negative `source.start_index`;
- `frames.count` other than ten unless the user explicitly starts a new experiment.

---

## 15. Implementation modules

Preferred V3 modules:

```text
r2v_data_v2/v3/
├── config.py
├── schemas.py
├── storage.py
├── annotation.py
├── pair.py
├── reference_judge.py
|-- cross_pair_judge.py
├── reference_edit.py
├── reference_edit_boogu.py
├── background.py
├── remove.py
├── qwen_image_edit_backend.py
├── removal_judge.py
├── instruction.py
├── export.py
└── pipeline.py
```

Reuse existing V2 utilities when stable. Do not duplicate large implementations simply to place them under `v3/`.

Potential shared utilities:

- video frame sampling;
- SAM3 backend wrapper;
- RLE mask encoding/decoding;
- atomic JSON writes;
- image compositing;
- structured-output request helpers;
- DINOv3/SigLIP2 metrics.

Keep V3 schemas separate from V2 schemas until the pilot passes.

---

## 16. Required tests

Local tests must run without real large models.

Run:

```bash
python -m pytest -q
python -m ruff check .
```

### 16.1 Storage tests

- one clip produces one `clip.json`;
- all tracked masks are stored in one `masks.rle.json`;
- no per-stage JSONL manifests are created in the run root;
- no `samples/*.json` directory is created;
- final dataset contains only `dataset.json`, `samples.jsonl`, and referenced files under `references/`;
- debug artifacts are absent by default;
- rejected candidates are not exported;
- export paths are relative and resolve correctly;
- atomic export does not destroy an existing dataset without `--overwrite`.

### 16.2 Caption tests

- `t2v_caption` contains no tokens;
- `r2v_instruction` differs from `t2v_caption`;
- every final `image_N` binding appears in the raw body template;
- repeated image placeholders are allowed;
- no unknown placeholder or internal `<ref_...>` token appears;
- legend IDs exactly match final binding order;
- raw body and legend descriptions are English and contain no CJK characters;
- raw output rejects `<Image N>`, plain `Image N`, and Chinese `图N` labels;
- deterministic rendering introduces `<Image N>` labels and an English legend;
- quoted dialogue requires an explicit source transcript.

### 16.3 Reference-scope tests

Include fixtures for:

1. a mostly complete picnic table with one clipped peripheral seat -> `full`;
2. a person with a coherent upper body and non-trivial detached lower fragments
   -> `repairable`, `upper_body`, completion required;
3. a mask with only tiny disconnected fragments and no identity features -> `reject`;
4. a thin valid object with multiple natural components -> not rejected solely for component count.

### 16.4 Background tests

- empty union mask -> `clean_raw`;
- a non-empty union mask at or below the configured maximum -> `pending_remove`;
- an oversized union mask -> `rejected`;
- equal-area candidates use the fixed center-first ordering;
- non-ready entity tracking rejects the background reference without rejecting
  the clip;
- rejected removal -> no background export;
- accepted removal -> `ready_removed`;
- outside-mask pixels remain identical;
- removal result with a new salient object is rejected;
- removal result with reconstructed foreground is rejected;
- no raw fallback path exists in V3.

### 16.5 Pairing and export tests

- clip-level coverage defaults to 7/10, is configurable, and uses ANY
  semantics;
- shorter-lived ready references remain after another entity qualifies the clip;
- same-parent fallback is off by default and requires an explicit dedicated judge;
- donor eligibility, natural ordering, configured cap, reject-continue, and
  all-reject behavior are deterministic;
- accepted donor PNG bytes are unchanged, publication rolls back on clip-write
  failure, and corrupt existing cross-pair artifacts fail validation;
- full real self references are never replaced; a full real donor may replace a
  lower-priority local self reference, while cross-parent donors are never
  considered;
- generated fallback is disabled by default, never becomes a donor, runs only
  after real donor fallback, and preserves the local source on every failure;
- accepted generated fallback uses existing Qwen/SAM3/gates/ranking components,
  publishes binary-alpha RGBA transactionally, and validates its provenance on
  idempotent reruns;
- legacy references without provenance remain readable, while new provenance is
  exported with the dataset reference;
- every accepted sample binds at least one qualifying target entity;
- final reference order and instruction image bindings match exactly;
- only accepted references are copied to the final dataset;
- final reference count matches files on disk.

---

## 17. Server pilot plan

Do not run full data immediately.

The first annotation smoke run must use `source.limit: 5`. Set
`source.allow_full_run: true` only for an explicitly authorized full production
run. For the first real SAM3 smoke, enable segment overlays and inspect every
per-slot overlay and contact sheet manually; mask counts alone are not
sufficient validation.
run. Do not default `allow_full_run` to `true` for convenience.

### 17.1 Annotation A/B

Use 30-40 stratified clips covering:

- stable single subjects;
- multiple interacting subjects;
- short-lived entities;
- occluded people;
- fragmented objects;
- small targets;
- camera motion;
- documentary underwater scenes;
- human scenes;
- products and vehicles.

Compare 8B and 32B with identical:

- video input;
- fps;
- prompt;
- schema;
- temperature;
- source metadata.

Record:

- first-pass valid JSON rate;
- repair rate;
- caption completeness;
- chronology;
- entity precision and recall;
- reference-worthy precision;
- relation correctness;
- phrase alignment rate;
- full/local/reject judgment quality;
- manual preference.

The production configuration remains 32B unless the pilot exposes a blocking runtime or quality regression.

### 17.2 Removal benchmark

Use the existing 20 non-empty background masks.

Record:

- generated candidate count;
- accepted count;
- known false positives;
- foreground remains/reconstructed;
- new-object errors;
- continuity errors;
- runtime and peak memory.

Do not relax validation merely to exceed the FLUX acceptance count.

### 17.3 End-to-end pilot

After annotation and removal benchmarks pass, run a 40-80 clip V3 pilot into new roots. Do not point V3 at `r2v_data_v2_pilot80_explore`.

Final acceptance requires:

- zero pipeline failures caused by storage lifecycle errors;
- zero dangling instruction image bindings;
- zero internal pairing tokens in rendered instructions;
- zero exported rejected references;
- zero contaminated raw backgrounds;
- manual approval of every exported removed background;
- manual review of a stratified set of full and local entity references;
- final dataset layout exactly matches Section 13.

---

## 18. Development sequence

Codex must not implement all changes as one unreviewable patch.

Recommended commits:

### Commit 1: V3 storage and schemas

- add V3 config and schemas;
- add run/dataset storage abstraction;
- add `clip.json` lifecycle;
- add compact exporter tests;
- no model execution yet.

### Commit 2: 32B annotation and dual-text schema

- add V3 annotation client;
- produce `t2v_caption`;
- keep tokens out of annotation;
- add instruction schema placeholders;
- add mock tests.

### Commit 3: reference scope

- add full/local/reject fields;
- update candidate review prompt and schema;
- implement coherent local crop/mask cleanup;
- add picnic-table/person-fragment fixtures.

### Commit 4: object-removal backend benchmark path

- add backend protocol;
- add Qwen Image Edit base+LoRA loader;
- add local compositing and validators;
- add benchmark command/config;
- keep production integration disabled until benchmark review.

### Commit 5: pairing, instruction writer, and export

- deterministic final tokens;
- post-pair instruction generation;
- compact final samples;
- atomic export;
- end-to-end mocked tests.

### Commit 6: pilot fixes only

- address actual server pilot issues;
- do not introduce unrelated refactors.

---

## 19. Definition of done

V3 is complete only when all statements are true:

- V2 remains functional and its tests pass.
- V3 uses Qwen3-VL-32B-Instruct for production annotation.
- V3 has separate `t2v_caption` and `r2v_instruction`.
- The instruction is generated after final references are known.
- New instructions use English raw text with `{{image_N}}` placeholders and
  deterministic `<Image N>` rendered labels.
- Entity references are explicitly full, local, or rejected.
- Unedited entity references remain variable-size source-faithful RGBA crops.
  Accepted Boogu references remain native approximately-one-megapixel RGB PNGs;
  both are exported once without a second normalized copy.
- The legacy synthetic completion fallback remains disabled by default.
  Production Boogu generation occurs only in the explicit `reference_edit`
  stage when it is enabled and passes both Qwen and SAM3 review.
- Non-empty background masks always require removal.
- FLUX is not the V3 production backend.
- Rejected removal outputs never fall back to raw.
- The Qwen Image Edit object-remover benchmark is recorded and manually reviewed.
- The final dataset contains no intermediate or diagnostic files.
- The final dataset has no per-sample JSON directory.
- `samples.jsonl` references only files present under the final `references/` tree.
- All local tests and lint checks pass.
- A server pilot satisfies Section 17.3.

---

## 20. Codex start checklist

Before coding:

1. Read `AGENTS.md` and this document.
2. Inspect current `main`, working tree, and recent commits.
3. Do not reset or overwrite newer user changes.
4. Create a feature branch for V3 implementation.
5. Keep the existing V2 pipeline untouched except for small shared utility extraction.
6. Implement Commit 1 only, run tests, and present the diff before proceeding to model integration.
7. Never run large models, download weights, or overwrite persistent server outputs without an explicit user instruction.
