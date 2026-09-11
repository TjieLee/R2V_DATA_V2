# Current Audio/H3 End-to-End Pipeline

This top-level README is the current high-level source of truth for the Audio/H3 path after Visual/reference extraction. The root `README.md` still contains substantial legacy V2/MVP material; do not infer the current Audio/H3 product path from those legacy sections.

Detailed operational documentation lives in:

- `docs/H3_AUDIO_SERVER_RUNBOOK.md`
- `docs/H3_AUDIO_REUSE_ASSETS.md`
- `docs/V3_VISUAL_AUDIO_INTEGRATION.md`

## Pipeline overview

```text
Visual branch
│
├─ final reference Pictures
├─ entity/background/attribute graph
└─ target videos
        │
        ▼
JEA Audio base production
│
├─ canonical audio
├─ primary voice
├─ face/speaker embedding
├─ in-pair / cross-pair
├─ diarization / binding audit / ASR
└─ frozen h3/samples.jsonl
        │
        ▼
Named Audio run
│
├─ SAM Audio (music_first)
│    ├─ speech stem (QA/debug only)
│    ├─ music stem
│    └─ SFX
│
├─ AuK Base → speech
│
├─ resolved_stems_v1 (speech=AuK, music/SFX=SAM)
│
├─ DiariZen (resolved AuK speech)
│
├─ Qwen3-ASR (resolved AuK speech)
│
├─ MiMo
│    ├─ Turn 1: visual
│    ├─ Turn 2: speech + AV binding
│    ├─ Turn 3: speaker profile
│    └─ Turn 4: soundscape/music
│
└─ finalized annotation
        │
        ▼
Audio reuse finalizer   [0 model calls]
│
├─ Sx_gN.flac speaker tracks
├─ music.flac
├─ prepared frozen snapshot
└─ final H3 products
     ├─ visual_only
     ├─ target_speech_reuse
     │    ├─ speaker_speech_reuse
     │    └─ music_reuse (if finalized music is positive)
     ├─ full_audio_reuse
     └─ cross_voice (if a valid frozen donor reuse asset is available)
        │
        ▼
rendered_h3_prompt
        │
        ▼
Frame-conditioning projection [0 model calls]
        │
        ▼
QA review page
```

## Current product contract

The final Audio/H3 products are built around these conditioning assets:

- **Visual references**: selected `<Picture N>` / `<Subject N>` graph from the frozen Visual output. Audio does not redo Visual ownership or Subject numbering.
- **`speaker_speech_reuse`**: target-speaker speech copied from the resolved AuK speech stem back onto the original target timeline. Each output track has the same decoded frame count and zero offset as the canonical target audio; non-owned regions are digital zero. This is timeline-preserving reuse, not tight concatenation.
- **`music_reuse`**: target-timeline music reuse from the resolved SAM `music_first` music stem. Stem existence alone is not sufficient to publish it; finalized MiMo `non_diegetic_music` semantics decide whether the Audio slot is used.
- **`full_audio_reuse`**: unchanged canonical full target audio as a separate `fully_copy` variant.
- **`cross_voice`**: for a frozen cross-pair, the target Subject/Sx uses the frozen donor speaker reuse track as a voice/timbre reference. No new identity matching is performed in the reuse path. A donor outside the prepared/reuse inventory is omitted with an explicit warning rather than substituted.

Legacy short target-voice references and `music_reference` remain separate compatibility/debug capabilities; they are not the primary final in-pair Audio reuse product path described here.

## Stage ownership

### 1. Visual output

Audio consumes the frozen Visual/reference graph: target video, selected Pictures, entity/background/attribute ownership, and final Subject numbering. These are immutable downstream inputs.

### 2. JEA Audio base production

The production base creates the durable Audio/H3 source graph. The existing high-level stage order is:

```text
audio
→ primary-voice
→ embedding
→ pair
→ diarization
→ binding-audit
→ qwen3-asr
→ h3
```

The resulting `AUDIO_PRODUCTION_ROOT/h3/samples.jsonl` contains canonical/in-pair/cross-pair samples, Visual references, and frozen cross-donor mappings used by the later named run.

### 3. Named Audio run

The current named-run generation path is:

```text
SAM music_first (music/SFX) + AuK Base (speech)
→ resolved_stems_v1
→ resolved-speech DiariZen
→ resolved-speech Qwen3-ASR
→ MiMo reconcile
→ Audio reuse finalizer
→ frame-conditioning projection
→ QA
```

Named runs require the fixed SAM `music_first` source, with no both-route run.
SAM speech is QA/debug only; unavailable AuK speech never falls back to SAM.
Generic `StemRecord` lineage follows the resolved source through every downstream
stage. Historical/default SAM-only runs remain readable for compatibility only.
Independent T2VA consumes finalized resolved speech/ASR facts; it is not changed
by this source-policy cleanup.

MiMo ownership is intentionally separated:

- **Turn 1** sees target video + frozen reference Pictures and owns the visual draft.
- **Turn 2** sees original target AV + Turn-1 text + Subject mapping + ASR/diarization evidence and owns speech/AV grounding, Sx assignment, summary, caption, and dialogue presentation.
- **Turn 3** sees only finalized speaker targets and exact speech-stem snippets and owns speaker voice profiles.
- **Turn 4** sees original target AV + music/SFX stems and owns `overall_soundscape` and `non_diegetic_music`.

LR-ASD/source clusters are supporting evidence only. They are not prerequisites for a valid visible-speaker binding.

### 4. Audio reuse finalizer

The named-run finalizer is model-free and wires the existing stages together:

```text
build_audio_reuse_assets
→ prepare_audio_reuse_sources
→ materialize_audio_reuse_products
```

The high-level CLI is `tools/finalize_h3_audio_reuse_shadow.py`. It publishes run-local shadow outputs only and keeps `model_call_count=0` and `production_artifacts_modified=false`.

The final prompt products live under:

```text
<shadow-run>/h3_audio_reuse_products_v1/
  records.jsonl
  summary.json
```

For each ready product, `rendered_h3_prompt` is the final six-section H3 prompt.

### 5. QA

QA is read-only review. It should discover current published products and display target video, Pictures, Audio reuse tracks, Subject graph, rendered prompt, and warnings. QA is not the source of product semantics and should not manufacture speaker-speech reuse assets.

## Important frozen semantics

- Audio/speech timeline facts are independent from entity binding.
- `multiple_speakers` is a hard identity-publication exclusion; do not bind or publish identity-specific voice products from confirmed multi-speaker speech.
- LR-ASD is supporting evidence only, never a binding prerequisite.
- Speaker speech reuse eligibility is waveform-ownership based and does not require a visible Subject identity. Offscreen/unbound speakers may still produce reusable speaker tracks when ownership is safe.
- `acoustic_refinement_unresolved` is a materialization warning and does not by itself exclude `speaker_speech_reuse`.
- Music stem existence is not semantic evidence that music is present; finalized MiMo music semantics control publication.
- Cross-pair donor mapping is frozen upstream; the reuse materializer does not rematch identities.
- Reuse assets/products are additive, model-free, hash/fingerprint checked, and must not overwrite frozen production inputs.

## User-facing operating model

Keep the normal workflow simple:

```text
1. Produce/freeze the JEA Audio base when needed.
2. Run the named Audio generation path through the Audio reuse finalizer.
3. Build the QA review page.
```

Internal `assets → prepared → materializer` stages are implementation details of the finalizer and should not require routine manual execution.

When server-validated commands or versions change, update this overview together with `docs/H3_AUDIO_SERVER_RUNBOOK.md` and `docs/H3_AUDIO_REUSE_ASSETS.md` rather than documenting a divergent pipeline elsewhere.
