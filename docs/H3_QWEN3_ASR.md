# JEA Audio Production with Qwen3 ASR

This is the active JEA H3 Audio production path. Legacy Whisper pilots remain in
the repository for history, but the commands below do not select them.

## Fixed roots

```bash
export VISUAL_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_v3_exports/production/jea_motion_v1/prod-v1
export VISUAL_RUNS_ROOT=/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/prod-v1
export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/prod-v1
```

Only `$VISUAL_PRODUCTION_ROOT/samples.jsonl` selects canonical Visual samples.
The Audio path does not scan shard exports or every clip directory. For each
canonical row it reads exactly:

```text
$VISUAL_RUNS_ROOT/<shard_id>/clips/<clip_uid>/clip.json
```

The processed `sample.target_video` MP4 must equal `clip.source.video_path`.
Original source MKV files are never target media.

Readable paths are derived from `clip.source.metadata`:

```python
media_collection_relpath = Path(source_relative_source_video_path).parent.as_posix()
media_collection_name = Path(media_collection_relpath).name
episode_name = Path(source_relative_source_video_path).stem
clip_name = Path(source_relative_video_path).stem
clip_display_path = Path(source_relative_video_path).with_suffix("").as_posix()
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
primary_voice/<clip_display_path>/<entity_id>.flac
```

Pair rows, DiariZen rows, Qwen rows, and final samples carry `clip_uid`,
`clip_display_path`, `media_collection_relpath`, `media_collection_name`,
`episode_name`, `clip_name`, and `shard_id`. Pair review pages sort readable
paths first.

PairPolicy V1 remains unchanged: face similarity is at least `0.72`, voice
similarity is at least `0.20`, there are no rank, margin, or text gates,
multi-subject mappings are one-to-one, and there is at most one cross sample
per target clip.

## Isolated Qwen3 ASR environment

Do not add `qwen-asr` to the repository's general requirements or change the
main Torch/CUDA environment.

```bash
export QWEN3_ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen3-asr-venv
export QWEN3_ASR_MODEL_PATH=<local-Qwen3-ASR-1.7B-directory>
export QWEN3_ASR_DEVICE=cuda:0
export QWEN3_ASR_DTYPE=bfloat16
export QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE=1

uv venv --python 3.12 --seed "$QWEN3_ASR_ENV"
uv pip install --python "$QWEN3_ASR_ENV/bin/python" \
  "qwen-asr==0.0.6" "pydantic>=2,<3" numpy pillow
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

## Commands

Dry-run enumerates the canonical count, subject occurrence count, exact media
collections, per-collection clip counts, shard count, output root, and selected
Qwen model. It has no limit or quota option.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

python tools/run_h3_jea_production.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --dry-run
```

Run each stage explicitly. The first six commands use the normal repository
environment. Only Qwen3 ASR uses the isolated Qwen Python:

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

"$QWEN3_ASR_ENV/bin/python" tools/run_h3_jea_production.py \
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
diarization -> qwen3-asr -> h3`. Every stage reads only its canonical upstream
artifacts. Existing stage directories are not silently reused; inspect them and
pass `--overwrite` only for an intentional replacement.

For a minimal ASR-only rerun:

```bash
"$QWEN3_ASR_ENV/bin/python" tools/run_h3_qwen3_asr.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

Do not use `--overwrite` unless the exact target stage directory has been
checked and intentional replacement is required.
