# Visual V3 Reference Development Handoff

Last updated: 2026-08-26

This is the entry point for future Visual/reference-image work. Read it before
older design documents or chat history, then read:

1. `V3_SUBJECT_ATTRIBUTES_STATE.md`
2. `SERVER_ENVIRONMENT_RUNBOOK.md`
3. `V3_VISUAL_AUDIO_INTEGRATION.md`

Verify the live remote branch and HEAD before developing or running anything.

## Frozen identity

```text
repository: TjieLee/R2V_DATA_V2
Visual/reference branch: feature/v3-subject-attributes-v1
final Visual/reference code freeze: d7f3d6b99e5da02bd8ef275ab53cd47cd649cfa0

frozen original Visual branch: feature/v3-runtime-integrity-v1
frozen original Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core original Visual algorithm baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Docs-only commits may advance branch HEAD without changing the code freeze.
Annotation production is frozen. Audio/H3 remains a separate development line.

## Final Subject/Object contract

Fresh reference selection is:

```text
complete -> canonical alpha
local_usable -> canonical alpha
repairable -> candidate 1 Boogu completion
  -> comparative Qwen accepts: completion 1
  -> rejects or is not better:
       alternate source candidate 2, when available
       -> Boogu completion 2
       -> three-image comparative Qwen accepts: completion 2
       -> rejects or is not better: canonical alpha
```

The selected completion then enters reference integrity. If an accepted
completion fails integrity, canonical alpha is checked; bbox is considered only
after canonical alpha fails. Completion must never fall directly to bbox.

Fresh `add_entity_background` calls are zero. The generated-background slot is
retained only for schema compatibility and is:

```json
{
  "image_path": null,
  "status": "unavailable",
  "reviewed": false,
  "review_status": "not_applicable",
  "reason": "entity_background_disabled_by_policy",
  "synthetic": true
}
```

The exact Subject/Object generation prompt is:

```text
Complete the missing or broken parts of the same target entity: "{entity_phrase}".
Preserve its identity, appearance, colors, materials, proportions, and style.
Do not add another entity or unrelated content.
Remove broken fragments and keep uncertain completion simple and consistent with visible evidence.
```

A modest real improvement is sufficient. Equivalent output returns to canonical
alpha. Same physical entity, identity, and semantics are hard gates.
Translation, recentering, moderate scale changes, crop, and layout changes are
allowed. Source-relative area, scale, and center movement are diagnostics only
for completion. Warped, duplicate, wrong-instance, tiny, or extreme-corner
outputs reject.

## Final Attribute contract

The ten supported types, Top3 owner candidate reuse, maximum two source
candidates, single-frame-only SAM3 probing, exact completion prompt and mapping,
six hard raw review flags, evidence routing, and publication modes are frozen in
`V3_SUBJECT_ATTRIBUTES_STATE.md`.

Key routing points:

- at most three attributes per eligible human owner;
- `structure_complete` and `completion_recommended` are diagnostics only;
- insufficient source evidence never authorizes Boogu hallucination;
- eligible completion compares raw alpha, source RGB bbox identity evidence,
  and Boogu candidate;
- same person or same physical item/component is a hard gate;
- equivalent completion returns to alpha;
- bbox is last resort only after no accepted completion and no accepted alpha;
- fresh Attribute generated-background calls are zero.

## Production runtime

```text
repo: /mnt/workspace/litengjie/data/R2V_DATA_V2
python: /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
SAM3 code: /mnt/workspace/litengjie/data/vendor/sam3
SAM3 checkpoint: /mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

Qwen model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
Qwen endpoint: http://127.0.0.1:8000/v1
Qwen: BF16, TP1 x DP4, GPUs 0-3, max model length 49152
runtime.qwen_max_inflight: 4

Boogu code: /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu python: /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model: /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708

GPU 4: Boogu background removal
GPU 5 + 7: SAM3 pool
GPU 6: Boogu reference_edit and Attribute completion
GME: disabled
```

Use `SERVER_ENVIRONMENT_RUNBOOK.md` for the exact environment and fresh-run
commands. Do not precreate the run or export root.

## Final validation status

The latest fixed-100 real-model run is
`e2e100-verify-324a29a-20260825-234558`, using commit
`324a29aebcf4b573cab59332d337ec9d10ad9deb`. It recorded zero fresh background
attempts for both reference edit and Attributes. Detailed counts are in
`V3_SUBJECT_ATTRIBUTES_STATE.md`.

The later validator correction after `622be6807490d57aebbda76b6dcf102beded7aff`
and candidate-2 provenance correction at the final freeze were unit-tested but
did not rerun the fixed-100 GPU/model canary. At the final freeze, local evidence
was 368 focused tests and 1,991 full tests passing with one warning, plus clean
diff-check and compileall results.

## Freeze boundary

Do not casually reopen prompts, thresholds, selection, model topology, schema,
Annotation production, or Audio/H3. New production evidence and an explicit
unfreeze decision are required. Audio developers must follow
`V3_VISUAL_AUDIO_INTEGRATION.md` and prefer the compacted production contract.
