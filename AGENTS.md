# R2V_DATA_V2 Development Memory

This file is the repository-level handoff and operating contract for Codex, ChatGPT, and other coding agents. Read it before inspecting, editing, testing, or proposing server commands.

> **Branch-specific V3 override — `feature/v3-subject-attributes-v1`:** The
> current Visual/Subject Attribute code freeze is
> `51fef9d44bb1372b4afad5fed9795d5c3d46bda7`. Read
> `docs/V3_SUBJECT_ATTRIBUTES_STATE.md` and
> `docs/VISUAL_ATTRIBUTE_DEVELOPMENT_HANDOFF.md` before changing V3
> references. The older V2/main flow below is legacy context and does not
> override current V3 state documents. Annotation production remains frozen;
> Audio/H3 is a separate branch. Do not infer current V3 removal or completion
> behavior from the old V2 FLUX or Qwen Image Edit text in this file.

## 1. Repository identity

- Current repository: `TjieLee/R2V_DATA_V2`
- Current development branch: `main`
- Server checkout: `/mnt/workspace/litengjie/data/R2V_DATA_V2`
- Server virtual environment: `/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv`
- Implementation snapshot immediately before this document was added: `83f48b768d2d8325ce84628fce6222a50f15989b` (`Fix clip-level entity coverage pairing`, 2026-07-30).
- Always inspect the actual `HEAD`, working tree, and recent commits before editing. The snapshot above is context, not a reason to reset or overwrite newer work.

Do not confuse this repository with the older `TjieLee/R2V_data_pipeline` or the training repository `R2V-Next`. They are outside the modification scope unless the user explicitly names them.

## 2. Development and execution model

The normal workflow is split across machines:

1. Code is inspected and edited through Git/GitHub, commonly from the user's Mac with Codex.
2. Real data/model execution happens on the company server.
3. The coding agent must provide tests and minimal reproducible server commands, but must not claim that a server run succeeded unless actual output was supplied.
4. The user runs server validation and returns logs or artifacts when another repair cycle is needed.

The intended server runtime normally has only two user-managed processes:

1. A vLLM process serving the configured Qwen model or endpoint.
2. The R2V pipeline process calling that endpoint.

SAM3, DINOv3, SigLIP2, and FLUX are loaded by the pipeline when enabled. They do not require independent service terminals.

Machine-specific configuration belongs in `configs/server.local.yaml`. Files matching `*.local.yaml` are ignored by Git. Do not overwrite, recreate, normalize, or commit a server-local configuration unless the user explicitly requests that exact action.

## 3. Filesystem access contract

### Writable root

The only generally writable server root is:

```text
/mnt/workspace/litengjie/data/
```

Repository files, environments, caches, temporary files, downloaded optional assets, outputs, benchmarks, and generated diagnostics must remain below this root.

Known writable locations include:

```text
/mnt/workspace/litengjie/data/R2V_DATA_V2
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv
/mnt/workspace/litengjie/data/vendor/sam3
/mnt/workspace/litengjie/data/models
/mnt/workspace/litengjie/data/cache
/mnt/workspace/litengjie/data/tmp
/mnt/workspace/litengjie/data/r2v_data_v2
/mnt/workspace/litengjie/data/r2v_data_v2_pilot1
/mnt/workspace/litengjie/data/r2v_data_v2_pilot80_explore
```

Existing pilot outputs are valuable state. Do not delete, rename, bulk-rewrite, or run `--overwrite` against them merely to simplify debugging.

### Read-only inputs

These inputs are strictly read-only:

```text
/mnt/workspace/public/dataset/
/mnt/workspace/public/pretrained/
```

The fixed source manifest currently used by the project is:

```text
/mnt/workspace/public/dataset/jea-video/zicai_5th_moive/train_zicai_5th_moive.json
```

The corresponding source videos are under the public dataset tree, including:

```text
/mnt/workspace/public/dataset/jea-video/zicai_5th_moive/videos_clips/data/
```

For every read-only path, agents must not:

- modify file contents or metadata;
- create files, caches, lock files, indexes, symlinks, or temporary files;
- rename, move, delete, truncate, or replace anything;
- download model files into the path;
- point package/model caches at the path.

Do not touch `/mnt/workspace/liutao/**` or any other user's workspace. A path being readable does not make it writable or in scope.

Recommended writable cache variables are:

```bash
export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp
```

When SAM3 import resolution is needed, the known local source root is:

```bash
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
```

## 4. Dependency and network boundaries

- Runtime is local-files-only unless the user explicitly authorizes a download step.
- Never silently download model weights.
- Never upgrade, replace, or reinstall the server's Torch, torchvision, torchaudio, CUDA, vLLM, flash-attn, or related GPU stack.
- Do not let SAM3, FLUX, Transformers, or another dependency replace the existing GPU environment.
- `requirements.txt` is intentionally lightweight. Optional vision and inpainting dependencies must remain optional.
- Base imports and unit tests must still work when FLUX, SAM3, DINOv3, SigLIP2, or a live Qwen endpoint is absent.
- Large models must be loaded lazily only in stages that need them.
- New costly behavior must be controlled by explicit YAML options and default safely.

## 5. Project goal and deliberate non-goals

The repository implements a lightweight, sequential, fully automated pipeline that converts existing video clips into reference-conditioned video training samples and writes `final_samples.jsonl`.

The canonical flow is:

```text
source JSON/JSONL
-> Qwen full-video caption, entities, and explicit reference bindings
-> independent fixed ten-frame sampling
-> SAM3 text-prompted tracking and candidate masks
-> hard gates and visual ranking
-> entity and optional background canonical references
-> optional local FLUX.1 Fill repair
-> entity in-pair/cross-pair and background in-pair binding
-> final_samples.jsonl
```

The lightweight MVP deliberately does not contain a runtime factory, plugin system, Gold Judge, watermark workflow, evidence-chain framework, state machine, or complex resume manager. Do not reintroduce those systems without an explicit new requirement.

Q-Align is not part of the current pilot. Do not add it casually to the reference-ranking path.

## 6. Hard semantic invariants

These rules are established project decisions. Do not change them incidentally while fixing another issue.

### Video and frame handling

- `frames.count` is fixed at **10** for this MVP. Do not revert to eight frames or expand it to 24 frames.
- Qwen annotation receives the complete source video through a local `file://` URI.
- Qwen's internal video sampling is independent from the ten JPEG frames used by SAM3 and ranking.
- The intended Qwen video-processing rate is `fps: 2.0` unless the user explicitly changes the experiment.
- Prompts and loops referring to sampled frames must consistently say/use ten chronological frames. Do not leave hard-coded `range(8)` logic.

### Annotation and entity binding

- Qwen produces an LTX-style single-paragraph caption, structured entities, relations, and explicit phrases that can receive reference tokens.
- Reference tokens must bind to retained caption phrases deterministically.
- The code, not Qwen, owns final filtering, normalization, and ranking decisions.

### SAM3 and temporal coverage

- SAM3 tracking masks across all sampled slots are distinct from strict entity-reference candidate masks.
- Background removal may consume tracking masks; entity ranking consumes only masks that pass candidate gates.
- Temporal visibility and reference-image quality are separate decisions.
- The clip-level entity-coverage gate uses ANY semantics: one qualifying reference-worthy entity is sufficient.
- A shorter-lived entity may still remain as a ready reference after another entity makes the clip pass coverage.
- A final sample must still contain at least one ready entity reference that intersects the clip's qualifying entity set.

### Ranking

- DINOv3 is used for representativeness/coarse retrieval, not as an exact-instance identity oracle.
- SigLIP2 is optional alignment evidence.
- Qwen visual judgments cover completeness, recognizability, mask quality, visual quality, occlusion, and canonical-view preference.
- Hard gates use raw measurements or raw judgments, never candidate-relative normalized scores.
- Canonical view is a preferred tier, not a front-view hard gate.
- Border contact is a soft completeness signal by default.
- `qwen_suggested_best_frame_slot` is diagnostic only and never overrides hard gates or the code-owned weighted ordering.

### Background references

- Background references use `<ref_bg_1>` and remain in-pair.
- A background phrase must have exactly one valid caption binding.
- An unbindable background is dropped without dropping otherwise valid entity references.
- Final samples require an entity reference by default; background-only artifacts may remain on disk but must not enter `final_samples.jsonl`.
- Background masks must cover every separable visible foreground entity.
- A clean raw background is preferred over a candidate that requires repair.

### Inpainting

- FLUX.1 Fill is optional and disabled by default.
- Inpainting must preserve pixels outside the effective repair mask exactly when configured to do so.
- Entity and background validation are different tasks: entity repair checks identity/completeness; background repair checks foreground removal and coherent scene continuation.
- A background repair that fails validation is rejected. It must not silently fall back to a contaminated raw background.
- Qwen prompt generation and repair review fail closed unless a generic prompt mode was explicitly selected.
- Repair metadata must remain bound to source image, source mask, frame index, semantics, prompts, model settings, and effective masks.
- Production artifacts such as raw canonical images, masks, RGBA foregrounds, neutral backgrounds, and embeddings must remain mutually aligned after repair or fallback.

### Pairing

- Pairing can use in-pair entity references and same-parent cross-pair references.
- Cross-pair candidates must come from the same `parent_video_id` and a different complete numeric `clip_suffix`.
- DINOv3 provides coarse retrieval only; Qwen dual-image review decides exact-instance compatibility.
- Uncertain, conflicting, near-duplicate, or low-confidence cross-pair decisions fall back to in-pair.
- Prompt tokens and the final reference list must be rebuilt together after filtering; they may never disagree.

## 7. Pipeline stage contract

The authoritative stage order in `run_pipeline.py` is:

```text
manifest -> qwen -> frames -> sam -> rank -> background -> inpaint -> pair -> augment
```

Stages are ordinary Python calls, not subprocesses. Existing durable artifacts are normally skipped. `--overwrite` is explicit and potentially destructive to derived state.

Durable stage state includes per-clip JSON/metadata artifacts as well as reconstructed JSONL manifests. A process interruption may leave a completed artifact that is not yet present in a JSONL file; rerunning the same stage should reconcile it rather than blindly rebuilding unrelated upstream work.

One bad sample should be logged and should not stop neighboring samples.

When debugging a downstream stage:

1. Inspect its existing per-clip artifacts and JSONL records first.
2. Re-run only the smallest relevant stage.
3. Do not re-run SAM3, ranking, background generation, or FLUX merely to test a pairing-only fix.
4. Do not use `--overwrite` unless the task explicitly requires regeneration and the affected output root has been checked.
5. Put benchmarks in a separate output directory outside the production pipeline output.

Current operational note: `/mnt/workspace/litengjie/data/r2v_data_v2_pilot80_explore` has been used for pair/background verification. For a pairing-only validation against this root, do not rerun SAM, ranking, background generation, or inpainting.

## 8. Code read and modification scope

Agents may read:

- the entire `TjieLee/R2V_DATA_V2` repository and Git history;
- tests, README, configuration schema/defaults, and generated logs/artifacts supplied by the user;
- the public dataset and pretrained roots strictly as read-only inputs when operating on the server;
- the configured local SAM3 source strictly as required for integration inspection.

Agents may modify:

- only files inside the current `R2V_DATA_V2` checkout;
- only the minimum source, tests, documentation, and example/default configuration needed for the current task;
- writable runtime outputs only when the task explicitly calls for execution or regeneration.

Agents must not:

- edit old repositories or training code as a side effect;
- broaden a narrowly scoped fix into an architecture rewrite;
- modify unrelated modules for style cleanup;
- overwrite `configs/server.local.yaml`;
- commit secrets, tokens, machine-local endpoint details, or private output manifests;
- remove tests to make a patch pass;
- change fixed ten-frame semantics, public data paths, or read-only policy without an explicit user decision;
- force-push `main`.

For pairing/background-binding work, inspect these files first and keep changes localized when possible:

```text
r2v_data_v2/phrase_alignment.py
r2v_data_v2/qwen_client.py
r2v_data_v2/background_reference.py
r2v_data_v2/inpainting.py
r2v_data_v2/pairing.py
run_pipeline.py
tests/
```

This list is a preferred locality, not permission to edit every listed file. A task-specific allowlist is stricter and overrides this general guidance.

## 9. Validation contract

Local validation must not require real Qwen, SAM3, DINOv3, SigLIP2, or FLUX models.

Run:

```bash
python -m pytest -q
python -m ruff check .
```

Tests must use `tmp_path`, fixtures, stubs, or mocks. They must not write into public data/model roots or depend on the user's persistent pilot outputs.

A typical server shell starts with:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate
export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
```

A small end-to-end invocation is:

```bash
python run_pipeline.py \
  --config configs/server.local.yaml \
  --limit 20 \
  --stages manifest,qwen,frames,sam,rank,background,pair
```

Do not copy this command mechanically for a downstream-only repair. Select the minimal stage set and use the exact output root/config involved in the reported failure.

Every server-validation request should state:

- repository commit;
- configuration file;
- output root;
- selected stages;
- whether `--overwrite` is used;
- expected files or counters;
- commands to inspect failure logs without deleting state.

## 10. Current implementation state

Recent work on `main` has already implemented or hardened:

- canonical entity and background reference production;
- raw/pending/repaired/rejected reference lifecycle;
- FLUX background hole-fill masks, prompts, multi-seed artifacts, and semantic review;
- immutable raw artifact restoration and stale-artifact checks;
- exact background phrase binding and pairing diagnostics;
- final-sample entity-reference requirements;
- separation of temporal entity coverage from reference-image quality;
- clip-level ANY entity-coverage semantics while retaining other ready references.

An earlier failure mode was that background phrases were not unique exact caption substrings, causing backgrounds to be dropped and some samples to end with `no_references`. Do not assume that historical diagnosis still applies to the current `HEAD`. Reproduce a failure from current artifacts and tests before changing phrase alignment or pairing again.

## 11. Required first steps for a new coding context

Before proposing or applying a patch:

1. Read this file, `README.md`, `configs/default.yaml`, and the relevant tests.
2. Run `git status`, `git branch --show-current`, and `git log -10 --oneline` in the actual checkout when available.
3. Identify the exact failing stage, clip IDs, output root, and durable artifacts.
4. Determine whether the issue reproduces on current `HEAD` without destroying cached upstream work.
5. State the intended file modification scope.
6. Add or update regression tests before requesting a server rerun.
7. Run local tests and lint where the environment permits.
8. Provide the smallest server command that validates the fix.

Do not guess from an old handoff when current code, tests, logs, or artifacts are available.