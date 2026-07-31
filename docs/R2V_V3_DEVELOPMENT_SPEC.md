# R2V V3 Development Specification

Status: **approved development specification**  
Audience: Codex and repository contributors  
Repository: `TjieLee/R2V_DATA_V2`  
Implementation target: a V3 pipeline developed alongside the existing V2 pipeline

This document is normative. Read `AGENTS.md` first, then follow this specification for V3 work. Where this document explicitly changes an older V2 behavior, this document takes precedence for V3 only.

---

## 1. Objective

Build a cleaner, quality-first R2V data construction pipeline with four deliberate changes:

1. use a stronger video annotation model for production captions and entity analysis;
2. separate the ordinary T2V caption from a reference-aware generation instruction;
3. replace FLUX background inpainting with an object-removal backend based on Qwen Image Edit;
4. classify entity references as full, local, or rejected instead of attempting synthetic entity completion.

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

The LoRA is an optional user-provisioned asset. The pipeline must not download it automatically.

### 2.3 Entity completion policy

V3 does not synthesize missing entity parts.

- Do not densely resample videos to search for more reference frames.
- Do not generate legs, clothing, object parts, or other unseen structure to create a supposedly complete reference.
- Preserve a sufficiently recognizable whole object as a full reference even if a peripheral part is clipped or occluded.
- Convert a semantically useful but incomplete entity into an honest local reference.
- Reject unusably fragmented references.

### 2.4 Caption policy

Every accepted sample has two distinct text fields:

- `t2v_caption`: literal, chronological, complete video description;
- `r2v_instruction`: generation-oriented Chinese instruction using rendered
  image labels such as `图1`.

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
- synthetic entity completion;
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
-> instruct
-> export
```

### 5.1 Stage responsibilities

#### `manifest`

Create source clip records only. Do not copy source videos.

#### `annotate`

Use full-video input with the 32B annotation model. Produce:

- `t2v_caption`;
- up to three provisional entity candidates;
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
clip-level `masks.rle.json`. Preserve backend confidence and object/track IDs.
A failed entity is recorded without blocking other entities; a clip with zero
entities publishes an empty, ready mask artifact without loading SAM3.
Single-subject and single-object masks must retain one tracked identity rather
than unioning unrelated detections. A group may union multiple tracks only when
the backend verifies that they came from the group prompt.

#### `rank`

The current implementation computes temporal coverage only from
`masks.rle.json`. It does not rerun SAM3, select canonical frames, classify
full/local/reject scope, or publish references. Later ranking work may add
those separate reference-quality decisions.

#### `background`

Select a background source frame and construct the union foreground mask.

- empty mask: clean raw background candidate;
- non-empty mask: pending object removal;
- excessively large or invalid mask: reject background candidate.

#### `remove`

Remove foreground entities from pending backgrounds using the configured Qwen Image Edit backend. Validate and publish only accepted background-only results.

#### `pair`

Choose the final retained references and assign deterministic tokens. The
clip-level coverage gate uses ANY-entity semantics and defaults to 7/10, with
the integer threshold configurable independently from SAM3.

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
    # independently configurable
```

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
candidate validation, phrase deduplication, and truncation to three candidates,
code assigns contiguous IDs `e1`, `e2`, and `e3`. Invalid candidates are
dropped without discarding a valid caption. An invalid background is normalized
to `null`. A valid caption with zero retained entities is still a ready
annotation; later stages own reference eligibility and clip rejection.

The persisted entity schema contains only `entity_id`, `reference_type`,
`phrase`, and `grounding_prompt`. Relations and the previous category,
salience, genericity, evidence, visual-scope, separability, and selection-reason
ontology are not part of V3 annotation schema version 2.

### 7.1 `t2v_caption` requirements

- one flowing English paragraph;
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

### 7.2 Candidate sanitation

Entity candidates are processed in model order. Code strips and normalizes
candidate text, drops invalid candidates, deduplicates normalized phrases while
keeping the first occurrence, retains at most three, and assigns entity IDs
after all filtering. Phrase text is not required to match one exact contiguous
caption span. Annotation remains separate from final reference eligibility.
Entity phrases should normally be stable noun phrases rather than actions.
Grounding prompts may add location or current pose only when needed to
distinguish a SAM3 target.

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
  "scope_reason": "..."
}
```

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

- crop only the coherent selected region;
- remove disconnected irrelevant fragments from the reference mask;
- store the visible region explicitly;
- never describe the result as a full-body or complete-object reference.

### 8.3 `reject`

Reject when:

- identity features are not visible;
- remaining components are too fragmented to form a useful local region;
- the mask is dominated by occluders or segmentation errors;
- the selected crop cannot be understood without guessing missing structure.

### 8.4 Decision ownership

Qwen provides semantic scope judgment. Code validates geometry and existing hard gates.

Connected-component count is diagnostic only. It must not independently decide full/local/reject because many valid objects naturally have separated or thin parts.

### 8.5 No synthetic entity repair

V3 must not add an entity-repair backend. Fields in old V2 inpainting schemas may remain for compatibility, but V3 export must set:

```json
"synthetic": false
```

for all entity references.

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
        mask: Image.Image,
        removal_phrases: list[str],
        prompt: str,
        seed: int,
    ) -> Image.Image: ...
```

The first backend identifier is:

```text
qwen_image_edit_2511_object_remover
```

The backend must load lazily only in the `remove` stage.

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
- SAM3 or another configured guard does not redetect a removed entity in the repaired region;
- pixels outside the effective generation mask are identical to the source image.

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

Same-parent cross-pair remains optional and conservative. A failed cross-pair judgment falls back to in-pair when configured.

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

`instruction_body_template` is Chinese, preserves chronology, action,
environment, camera, composition, and lighting, and may use the same image
placeholder multiple times. It must use every final binding at least once.
Without `source_transcript`, it cannot invent quoted dialogue.

Code performs the only presentation-layer conversion:

```text
{{image_1}} -> 图1
{{image_2}} -> 图2
```

It then appends the legend in binding order:

```text
<rendered instruction body>

图1：<description>
图2：<description>
```

Chinese image labels are never schema identifiers, enum values, binding IDs, or
raw model placeholders. Internal `<ref_...>` pairing tokens remain separate and
do not appear in the structured instruction output or rendered instruction.

### 11.4 Instruction validation

Validate raw structured output before Chinese rendering:

- the body template is non-empty and uses only exact `{{image_N}}` placeholders;
- every binding appears at least once; repeated placeholders are allowed;
- no unknown placeholder or `<ref_...>` token appears;
- raw output contains no direct Chinese image label matching `图` plus a number;
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
            └── segment/           # per-slot overlays and entity contact sheets
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
- Store only selected entity images in `selected/`.
- Store only an accepted removed background in `selected/bg_removed.png`.
- A clean raw background points to its selected sampled frame and does not require a duplicate image in `selected/`.
- Rejected removal candidates and contact sheets are written only under `debug/` when `debug.save_diagnostics: true`.
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

Do not export:

- sampled frames;
- masks;
- raw source reference images;
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
  "r2v_instruction": "以图2作为整体背景，图1向前行走。\\n\\n图1：...\\n图2：...",
  "references": [
    {
      "token": "<ref_subject_1>",
      "type": "entity",
      "entity_id": "e1",
      "scope": "local",
      "visible_region": "upper_body",
      "image_path": "references/<sample_id>/subject_1.png",
      "source_frame_index": 128,
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
- Preserve alpha for entity cutouts when the selected entity artifact is mask-backed.
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

coverage:
  required_visible_frames: 7

reference_scope:
  enabled: true
  allow_local: true
  allow_synthetic_completion: false

background:
  enabled: true
  raw_foreground_area_ratio: 0.0

remove:
  enabled: true
  backend: qwen_image_edit_2511_object_remover
  base_model_path: /mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511
  adapter_path: /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover
  candidate_seeds: [0, 17]
  fallback_to_raw: false
  preserve_unmasked_pixels: true

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
- `allow_synthetic_completion: true` in the initial V3 implementation;
- `remove.fallback_to_raw: true`;
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
├── reference_scope.py
├── background.py
├── object_removal.py
├── pairing.py
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
- Chinese `图N` labels are introduced only by deterministic rendering;
- quoted dialogue requires an explicit source transcript.

### 16.3 Reference-scope tests

Include fixtures for:

1. a mostly complete picnic table with one clipped peripheral seat -> `full`;
2. a person with coherent upper body and disconnected lower fragments -> `local`, `upper_body`;
3. a mask with only tiny disconnected fragments and no identity features -> `reject`;
4. a thin valid object with multiple natural components -> not rejected solely for component count.

### 16.4 Background tests

- empty union mask -> `clean_raw`;
- any non-empty union mask -> `pending_remove`;
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
- every accepted sample binds at least one qualifying entity;
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
- Entity references are explicitly full, local, or rejected.
- No synthetic entity completion exists.
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
