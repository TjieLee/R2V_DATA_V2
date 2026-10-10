# Group Pilot10 Offline Caption Recovery

Baseline: `fa7fb5b11feec228fea474ad676ebdc7fa71f53d`.

Source group (read-only):
```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/ra2va-native-group0-pilot10-20261010T034522/groups/group-000000
```

## Four Real Failures

| Clip | Frozen / model facts | Root cause | This patch |
| --- | --- | --- | --- |
| ab27449eaf98b06401161810 | 1/1 exact Chinese dialogue, g1/e1; Visual articulation not_assessable; Joint av_temporal_alignment + no_visible_lip_motion | Materializer identity neutralizer accepts only standalone says/asks clauses, not a visual posture clause ending in saying | Split the explicit speech lead from unchanged visual posture; neutralize Summary attribution; keep g1 identity excluded |
| 83871dd0401c1f1ce9986ec2 | 1/1 exact Chinese dialogue, g1/e1; Visual not_assessable; Joint includes both visible_lip_motion and no_visible_lip_motion | Embedded mouth-movement/speech attribution is outside the narrow neutralizer | Preserve mouth movement as a visual observation, separate neutral voice; retain all original contradictory evidence and identity restriction |
| 6107ee1e8a646cccb1583b3f | Frozen segments=[]; Visual invents segment_0001; Joint creates segment_0001/g1 Audio decision and AV grounding | _normalize_joint_draft requires IDs to equal frozen inventory; this is an extra invented segment, not an ordering issue | Remains Failed; no fabricated ASR/decisions and no deletion of acoustic facts to force acceptance |
| d897696ed13b1ea1b2104ea2 | 6/6 Chinese dialogues; g1/e1 for first 2, g2/e2 for last 4; first Visual articulation=uncertain; Joint no_visible_lip_motion + av_temporal_alignment | Shared validator rejects the negative-code binding; existing Two-step publication exception requires not_assessable, not uncertain | Remains Failed; do not reinterpret uncertain or change the frozen binding policy in this caption-only repair |

The d89769 failure does **not** prove a wrong speaker identity. It is unresolved under the current policy. Its Caption also inherits g1 through `and adds`, and its Summary assigns the report to an armored runner while the later reporting dialogues are g2/e2. Safely publishing an identity-independent version needs a separately specified treatment of that multi-speaker narrative, not a blind evidence-code bypass.

For 6107ee, the complete **expected** ID set is empty. Both generated lists contain an extra ID, so sorting or identical-row deduplication cannot make the inventories correct. No inventory normalization change is justified by this case.

## Code Boundaries

- `speaker_ownership.py::neutralize_two_step_caption_identity`: two narrow embedded lead forms plus their Summary forms; exact ordered frozen dialogue comparison before any neutralization.
- `mimo26_two_step_backend.py::_normalize_joint_draft`: unchanged inventory / ASR / speaker normalization.
- `mimo25_backend.py::_visible_binding_is_permitted` and `validate_annotation`: unchanged.
- `mimo25_h3_materializer.py::_prepare_materialization_context`: same consumer and publication restrictions; forwards frozen reference_subjects into the Two-step-only neutralizer.
- Group postprocessing audit: `ra2va_group_durable_two_step_v2`. Historical backend .85 and inference provenance stay unchanged.
- No prompt, gN/Sx, binding, audio ownership, donor eligibility, Multi/Single, T2VA, or model-call change.

## Full Caption Comparison

### ab27449eaf98b06401161810

Before:
```text
A medium shot opens on <Subject 1>, the bald man in the light grey traditional tunic, standing in a dimly lit interior with hanging paper lanterns and a cool blue night scene visible through the open structure behind him. He stands beside a man in ornate dark armour and a helmet, with vertical Chinese text overlaid on the left side of the frame. <Subject 1> (S1) faces slightly to the right with his mouth open as if speaking, his brow furrowed, saying, <d>[Chinese] 大人亲手举荐为官，若此事落实，恐怕大人您也难辞其咎。</d> The camera then shifts focus and pans slightly right as a second bald man in a dark high-collared tunic with a moustache moves into the foreground, becoming the dominant figure while <Subject 1> and the armoured man fall out of focus in the background. The foreground man holds a steady, serious gaze forward with his lips pressed together, while the background figures remain visible but blurred behind him.
```

After:
```text
A medium shot opens on <Subject 1>, the bald man in the light grey traditional tunic, standing in a dimly lit interior with hanging paper lanterns and a cool blue night scene visible through the open structure behind him. He stands beside a man in ornate dark armour and a helmet, with vertical Chinese text overlaid on the left side of the frame. <Subject 1> faces slightly to the right with his mouth open as if speaking, his brow furrowed. A voice (S1) says, <d>[Chinese] 大人亲手举荐为官，若此事落实，恐怕大人您也难辞其咎。</d> The camera then shifts focus and pans slightly right as a second bald man in a dark high-collared tunic with a moustache moves into the foreground, becoming the dominant figure while <Subject 1> and the armoured man fall out of focus in the background. The foreground man holds a steady, serious gaze forward with his lips pressed together, while the background figures remain visible but blurred behind him.
```

### 83871dd0401c1f1ce9986ec2

Before:
```text
A medium close-up shows <Subject 1>, a man in a dark, wet, hooded robe with a geometric pattern, standing in heavy rain against a dark blue background defined by <Subject 3>. The camera remains static as rain streaks fall continuously across the frame. <Subject 2>, the man's face, is visible through the veil of rain, looking slightly upward and to the side with a serious expression. His mouth moves as he speaks (S1), <d>[Chinese] 你们所运的货物，乃是历城县令的贿银。</d>, and water droplets cling to the fabric of his hood and shoulders. The lighting is dim and cool, emphasizing the wet texture of his clothing and the density of the rainfall.
```

After:
```text
A medium close-up shows <Subject 1>, a man in a dark, wet, hooded robe with a geometric pattern, standing in heavy rain against a dark blue background defined by <Subject 3>. The camera remains static as rain streaks fall continuously across the frame. <Subject 2>, the man's face, is visible through the veil of rain, looking slightly upward and to the side with a serious expression. His mouth moves. A voice (S1) says, <d>[Chinese] 你们所运的货物，乃是历城县令的贿银。</d>, and water droplets cling to the fabric of his hood and shoulders. The lighting is dim and cool, emphasizing the wet texture of his clothing and the density of the rainfall.
```

## Offline Server Command

Run from the existing server checkout after pulling the repair. No endpoint or GPU is needed. The command reads only the already-frozen 10-row inventory and saved stage results/completions; it never traverses the original full group samples.jsonl.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git pull --ff-only origin feature/h3-audio-jea-qwen3-v1
ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
.venv/bin/python tools/replay_h3_ra2va_group_pilot.py \\
  --source-group "$ROOT/ra2va-native-group0-pilot10-20261010T034522/groups/group-000000" \\
  --output-root "$ROOT/ra2va-pilot10-caption-recovery-$(date +%Y%m%dT%H%M%S)/group-000000"
```

Outputs: `offline_replay.json`, per-clip reconcile/audit and H3 products, aggregate `bundle/source_contract.json`, `bundle/records.jsonl`, twelve `bundle/training/*.jsonl` task files and `bundle/voice_donor_reserve.json`.

## Next GPU Pilot (Not Started)

After this CPU export result is accepted: select only 6-10 **new** frozen clips from the next bounded input window, use a new run-id on the existing two-node HTTP Coordinator, exercise worker/model release and one controlled interruption, retain exactly two MiMo requests per successful clip. Measure usable video/task yield separately from inference Ready count. Do not begin a whole group or a 100-200 clip run.

## Validation

- Original focused regression: 504 passed, 7 skipped (Group, Two-step, Single Compact, Audio reuse/materializer/donor, T2VA speech/server).
- Final caption/replay/materializer/Two-step focused run: 208 passed, including real recorded drafts, unsupported identity prose, unknown Subject, Visual uncertainty, and exact ASR negative cases.
- Ruff, py_compile and diff-check passed.
- Additional whole-repository run: 6584 passed, 250 failed, 144 skipped. All 250 failing node IDs reproduced on isolated baseline fa7fb5b (234 + 16 cases); no attempt to repair historical tests or unrelated modules. Representative failures include old backend/prompt provenance assertions in test_h3_mimo25_two_turn.py, historical fixtures in test_h3_mimo25_av_shadow.py, and unrelated V3/Person Replacement contracts.
