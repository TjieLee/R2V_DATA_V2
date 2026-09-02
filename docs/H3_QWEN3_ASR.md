# JEA Audio Production with Qwen3 ASR

This is the active JEA H3 Audio production path. Legacy Whisper pilots remain in
the repository for history, but the commands below do not select them.

## Fixed roots

```bash
export VISUAL_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_v3_exports/production/jea_motion_v1/prod-v1
export VISUAL_RUNS_ROOT=/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/prod-v1
export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/prod-v1
```

This Audio integration is synchronized with Visual head
`71276f976ed178242abe4db3e66e3fecce357832`. Ordinary V3 input supports both
legacy enriched sidecars and the latest variant-aware sidecars. H3 consumes
only the Visual-selected default attribute image at
`SubjectAttributeRecord.image_path`; choosing among alpha, bbox, generated
background, and accepted-base variants remains owned by Visual.

Only `$VISUAL_PRODUCTION_ROOT/samples.jsonl` selects canonical Visual samples.
Every validated row is retained in `VisualProductionInventory.canonical_clips`
and defines the clip-level Audio universe. The existing `.clips` collection is
the subject-reference subset used by LR-ASD, primary voice, embedding, and pair
stages. A `no_subject_reference` count therefore means "not eligible for the
subject pipeline", not "invalid or removed from Audio".
The loader detects one of two supported layouts from its first non-empty row:

- `r2v.v3.production_sample.1` is compacted production. In this mode
  `VISUAL_RUNS_ROOT` is the parent runs root and each clip record is
  `$VISUAL_RUNS_ROOT/<shard_id>/clips/<clip_uid>/clip.json`.
- `r2v.v3.sample.1` is an ordinary completed single-run export. In this mode
  `VISUAL_PRODUCTION_ROOT` is that export root, `VISUAL_RUNS_ROOT` is the exact
  matching V3 run root, and each clip record is
  `$VISUAL_RUNS_ROOT/clips/<sample_id>/clip.json`.

The argument name `--visual-production-root` remains stable for compatibility.
The Audio path does not scan shard exports or every clip directory. In ordinary
export mode, an optional `subject_attributes/enriched_samples.jsonl` is loaded
once: matching clips preserve its enriched instruction and ordered attribute
references, while clips without a matching record retain the original export
instruction and references. Audio validates this sidecar through an H3-owned,
downstream-stable projection of sample identity, run/source provenance,
instruction bindings, selected image paths, ownership, frame indexes, and
variant selection. Visual-internal `review` and `completion_review` QA
checklists are deliberately not parsed, so checklist evolution cannot invalidate
otherwise compatible frozen runs. Missing or inconsistent downstream fields
still fail closed. Visual artifacts stay read-only.

The same boundary applies to each selected `clip.json`. Audio/H3 parses an
H3-owned projection of clip identity, source video and readable source paths,
annotation entities, coverage admission, pairing retention, and ready reference
image paths. Visual-only sections such as `reference_edit`, subject-attribute
state, review checklists, and diagnostics are ignored rather than validated
against the current Visual implementation. Required projected fields, duplicate
or unknown entity bindings, missing ready references, clip identity, and target
video provenance still fail closed. Pilot and sidecar/fusion paths use this same
projection; Visual frames and tracked-mask artifacts retain their canonical V3
schemas.

For compacted production each canonical row reads exactly:

```text
$VISUAL_RUNS_ROOT/<shard_id>/clips/<clip_uid>/clip.json
```

The processed `sample.target_video` MP4 must equal `clip.source.video_path`.
Original source MKV files are never target media.

Readable paths are derived from `clip.source.metadata`:

```python
from pathlib import PurePosixPath

video_path = PurePosixPath(source_relative_video_path)
media_collection_relpath = PurePosixPath(*video_path.parts[:2]).as_posix()
media_collection_name = video_path.parts[1]
episode_name = PurePosixPath(source_relative_source_video_path).stem
clip_name = video_path.stem
clip_display_path = video_path.with_suffix("").as_posix()
```

Unicode and spaces are preserved. Absolute paths, backslashes, and `..` are
rejected. Cross-pair donors must have the exact same full
`media_collection_relpath` and a different `clip_uid`; a matching basename or
legacy `parent_video_id` is insufficient.

## Output layout

Stages are direct children of `$AUDIO_PRODUCTION_ROOT`:

```text
audio/
primary_voice/
embedding/
pairs/
diarization/
asr/
h3/
```

Clip artifacts use the readable path:

```text
audio/clips/<clip_display_path>/audio_binding.json
audio/full_audio/<clip_display_path>.flac
audio/canonical_clips.jsonl
audio/canonical_clips_summary.json
primary_voice/<clip_display_path>/<entity_id>.flac
```

Fresh Audio production materializes one lossless canonical artifact for every
`canonical_clip` while running subject binding only for `.clips`.
`audio/full_audio/` is always 32 kHz stereo FLAC decoded directly from the
original processed target video audio stream. No persistent 16 kHz analysis
copy or second H3-audio tree is published.

The canonical manifest publishes one `r2v.h3.canonical_audio_clip.2` row per
Visual canonical clip in Visual order. `target_full_audio_path` and
`target_full_audio_sha256` mean the single 32 kHz stereo canonical artifact.
The row records its sample rate, channel count, duration, and original-video
source policy.
No-subject rows have null binding path/hash instead of a fabricated empty
sidecar. Its summary requires the Visual and Audio canonical counts to match.

Canonical Audio rows, DiariZen rows, Qwen rows, and final samples carry `clip_uid`,
`clip_display_path`, `media_collection_relpath`, `media_collection_name`,
`episode_name`, `clip_name`, and `shard_id`. Pair review pages sort readable
paths first.

The active resolved-path output contracts are `r2v.h3.final_sample.5` and
`r2v.h3.final_summary.6`; the summary declares
`final_sample_schema_version=r2v.h3.final_sample.5`. Final H3 samples publish
media paths with explicit ownership semantics:

- `target_video` is the directly readable processed target-video path.
- `target_full_audio_path` is the directly readable 32 kHz stereo H3-native
  target audio used by AudioVAE/training.
- `visual_references[].image_path` is the original Visual-relative provenance
  path and intentionally retains its source representation.
- `visual_references[].image_artifact_path` is the directly readable absolute
  reference-image path copied from the normalized Visual inventory.
- `subject_voices[].voice_reference_path` is the directly readable 32 kHz
  stereo H3 conditioning reference.

Primary-voice quality selection keeps its existing diagnostics, then the
persisted reference is cropped directly from canonical 32 kHz stereo audio
using authoritative start/end times and `round(time_seconds * 32000)`.
Cross-pair variants use the donor occurrence's canonical crop under the same
policy. ECAPA converts that reference to a 16 kHz mono runtime waveform only
inside its worker.

DiariZen and Qwen3-ASR both receive temporary 16 kHz mono model inputs, while
persisted segment `source_start_sample` and `source_end_sample` values remain in
the canonical 32 kHz source domain. Their implementations are intentionally
identified separately: DiariZen uses
`h3_diarizen_torchaudio_kaiser_32k_stereo_to_16k_mono_v1`, while Qwen3-ASR uses
`h3_qwen3_asr_ffmpeg_32k_stereo_to_16k_mono_v1`. These provenance values do not
claim bit-identical preprocessing across model runtimes.

Visual reference assets are not copied into H3. A training reader must read
`image_artifact_path` directly and must not reconstruct Visual export/run-root
resolution rules from `image_path`; references in one sample may originate from
different validated Visual ownership roots.

PairPolicy V1 remains unchanged: face similarity is at least `0.72`, voice
similarity is at least `0.20`, there are no rank, margin, or text gates,
multi-subject mappings are one-to-one, and there is at most one cross sample
per target clip.

## Isolated Qwen3 ASR environment

Do not add `qwen-asr` to the repository's general requirements or change the
main Torch/CUDA environment.

```bash
export QWEN3_ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen3-asr-venv
export QWEN3_ASR_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-ASR-1.7B
export QWEN3_ASR_DEVICE=cuda:0
export QWEN3_ASR_DTYPE=bfloat16
export QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE=1

uv venv --python 3.12 --seed "$QWEN3_ASR_ENV"
uv pip install --python "$QWEN3_ASR_ENV/bin/python" \
  "qwen-asr==0.0.6" "pydantic>=2,<3" numpy pillow
```

`qwen-asr` pins its own Python dependencies, but the server must provide a
CUDA-compatible PyTorch runtime in this isolated environment. Do not infer a
Torch or CUDA version; verify the installed runtime before production:

```bash
"$QWEN3_ASR_ENV/bin/python" - <<'PY'
import torch
import soundfile
from qwen_asr import Qwen3ASRModel
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available()
PY
```

Runtime is local-files-only. The backend loads
`Qwen/Qwen3-ASR-1.7B` once per process with `max_new_tokens=256` and calls the
official API with the exact DiariZen segment waveform and source sample rate:

```python
result = model.transcribe(
    audio=(waveform, sample_rate_hz),
    context="",
    language=None,
    return_time_stamps=False,
)[0]
```

The published text is `result.text` without translation or correction and the
language is `result.language`. No confidence field or Whisper
`language_probability` is published, and the former `0.65` gate does not apply.
Empty and failed rows remain auditable in `asr/segments.jsonl` but are omitted
from final speech segments. There is no fallback ASR, VAD resegmentation,
denoising, enhancement, timestamp aligner, visual prompt, identity prompt, or
network access. Dots3 is paused and is not used by this path.

The isolated ASR process reads `diarization/readable_segments.jsonl` as its
complete readable identity and sample-range input. It validates those rows
against `readable_summary.json`, `raw_segments.jsonl`, and
`bound_segments.jsonl` without loading Visual V3 samples or importing Visual
model dependencies. The Visual production root remains inventory provenance
only. The isolated environment does not require `openai`, SAM, or general V3
production dependencies.

## Commands

Dry-run enumerates the detected `visual_input_schema` and `visual_input_mode`,
canonical count, subject occurrence count, exact media collections,
per-collection clip counts, shard count, output root, and selected Qwen model.
It has no limit or quota option.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

# Required for isolated embedding workers launched from embedding-venv.
# The worker scripts import r2v_data_v2 directly, so the repository root must
# remain importable in child processes even though their Python executable is
# outside the repository virtualenv.
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --dry-run
```

Before running the embedding stage, verify the isolated face/speaker workers can
import the repository package from the configured embedding Python:

```bash
"$FACE_EMBEDDING_PYTHON" - <<'PY'
import r2v_data_v2
print("r2v_data_v2:", r2v_data_v2.__file__)
PY
```

If this import fails with `ModuleNotFoundError: No module named 'r2v_data_v2'`,
restore the `PYTHONPATH` export above before retrying embedding. A failed
embedding run may still publish an all-zero summary because per-occurrence model
failures are isolated; inspect `$AUDIO_PRODUCTION_ROOT/.embedding_worker_diagnostics/face/worker.stderr.log`
before running `pair` when `face_embedding_available_count` unexpectedly equals
zero.

Run each stage explicitly from the normal repository environment. DiariZen is
rooted in `audio/canonical_clips.jsonl` plus the canonical Visual inventory;
primary voice and pair availability do not select its targets. Audio bindings
are optional identity evidence, so a missing or non-ready sidecar produces
unbound DiariZen segments rather than dropping the clip. For the
`qwen3-asr` stage, the JEA orchestrator launches exactly one child process with
`$QWEN3_ASR_ENV/bin/python`; that child loads Qwen once and processes all
readable DiariZen segments. Do not launch the all-stage orchestrator with the
isolated Qwen Python:

```bash
export R2V_PYTHON=/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages audio --workers 4

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages primary-voice

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages embedding

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages pair

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages diarization

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages binding-audit

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages qwen3-asr

"$R2V_PYTHON" tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --stages h3
```

The active order is `audio -> primary-voice -> embedding -> pair ->
diarization -> binding-audit -> qwen3-asr -> h3`. The binding audit remains
the existing model-free sidecar rather than a new production inference stage;
the final renderer requires its canonical segment evidence. Every stage reads
only its canonical upstream artifacts. Existing stage directories are not
silently reused; inspect them and pass `--overwrite` only for an intentional
replacement. Audio semantics is never auto-discovered. Omit
`--audio-semantics-root` to publish explicit missing semantics coverage, or
pass only a freshly generated root whose canonical-audio hashes match the
current 32 kHz manifest; an old 16 kHz-derived semantics root fails closed.

For a minimal ASR-only rerun:

```bash
"$QWEN3_ASR_ENV/bin/python" tools/run_h3_qwen3_asr.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

Do not use `--overwrite` unless the exact target stage directory has been
checked and intentional replacement is required.

For an existing Audio root, model-free canonical full-audio backfill is:

```bash
"$R2V_PYTHON" tools/backfill_h3_canonical_audio.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

It does not rerun Audio binding or modify primary voice, embedding, pairs,
diarization, ASR, specialized semantics, or H3 output.
`--audio-production-root` names the whole production root; the backfill uses the
official JEA layout and publishes only under its `audio/` child, including
`audio/canonical_clips.jsonl` and `audio/full_audio/`.

Specialized clip-level audio semantics reads `audio/canonical_clips.jsonl` and
does not use Qwen3-ASR as an admission dependency. It publishes one assembled
record per Visual canonical clip. The final renderer validates exact semantics
coverage when that root is supplied and projects the record into the
backend-neutral `full_clip_audio_semantics` field; raw caption text is not
published. The exact Visual/Audio clip-level join key is `clip_uid`.

Qwen3-ASR inventory and summary contract `.3` report
`source_target_clip_count`, `clips_with_diarization_segments`, and
`clips_without_diarization_segments`. No-speech canonical clips receive no fake
ASR row and still produce canonical Final H3 samples with an empty
`speech_segments` list.

## Model-free Qwen3 ASR human review

After `qwen3-asr` completes, generate the independent QA sidecar without
loading Qwen or modifying `asr/`, `pairs/`, `diarization/`, or `h3/`:

```bash
"$R2V_PYTHON" tools/run_h3_qwen3_asr_review.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

The fixed output is `$AUDIO_PRODUCTION_ROOT/asr_review/`. It contains a static
`review.html`, a deterministic `manifest.json`, and browser-compatible H.264/AAC
review proxies under `media/`. Proxies retain the complete video and audio
timeline and are QA-only derivatives; canonical target videos and Qwen rows are
never changed. The page displays transcribed, empty, failed, and unbound rows,
supports exact-segment playback, persists labels/notes in browser localStorage,
and exports a deterministic QA JSON sidecar.

Serve the static directory from the server when reviewing remotely:

```bash
python -m http.server 8768 \
  --directory "$AUDIO_PRODUCTION_ROOT/asr_review"
```
