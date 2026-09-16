# H3 training manifest export

`tools/export_h3_training_manifests.py` flattens the frozen R(A)2VA / T(A)2VA outputs into dataloader-facing JSONL files.

## Status

The exporter and tests are implemented. **Do not run the formal dataset export yet.** Production export is intentionally deferred until the current higher-priority work is finished. The exporter is read-only with respect to all frozen inference artifacts.

## Loader row format

Every task JSONL uses exactly four fields:

```json
{"video":"/path/to/video.mp4","images":["/path/to/p1.png"],"audios":["/path/to/a1.flac"],"caption":"final caption"}
```

- `video`: target video path.
- `images`: ordered conditioning-image paths. Use `[]` when the task has no image input.
- `audios`: ordered conditioning-audio paths. Use `[]` when the task has no audio input.
- `caption`: final frozen caption/prompt consumed by training.

The export intentionally contains **no hashes, fingerprints, stage names, model diagnostics, or internal provenance**. Media remain external files and are referenced only by path.

## Output files

A completed export directory contains:

```text
videos.jsonl

r2va_reference.jsonl
r2va_first_frame.jsonl
r2va_last_frame.jsonl
r2va_first_last_frame.jsonl

ra2va_reference_full_audio.jsonl
ra2va_reference_speech_bgm.jsonl
ra2va_first_frame_full_audio.jsonl
ra2va_first_frame_speech_bgm.jsonl
ra2va_last_frame_full_audio.jsonl
ra2va_last_frame_speech_bgm.jsonl
ra2va_first_last_frame_full_audio.jsonl
ra2va_first_last_frame_speech_bgm.jsonl

t2va.jsonl
ta2va_full_audio.jsonl
ta2va_speech_bgm.jsonl
```

### Visual modes

- `reference`: frozen selected Ref2V Pictures.
- `first_frame`: extracted first frame.
- `last_frame`: extracted last frame.
- `first_last_frame`: extracted first and last frames, in that order.

### Audio modes

- `full_audio`: canonical complete target-audio reuse.
- `speech_bgm`: target speaker-speech reuse plus optional BGM/music reuse. If no BGM is published for a sample, it remains a `speech_bgm` sample and `audios` contains only the speech track(s).

Cross-voice/reference-only Audio variants are intentionally excluded from this training export.

## `videos.jsonl`

`videos.jsonl` is the summary index. It has one row per target video and lists every fine-grained task currently available for that video:

```json
{"video":"/path/to/video.mp4","tasks":["r2va_reference","ra2va_reference_full_audio","ta2va_speech_bgm"]}
```

Use this file for inventory inspection and dataset-mixture planning. Use the individual task JSONL files for actual sampling/training.

## Dataloader usage

A loader does not need to understand the internal H3 directory structure. It only reads the selected task file:

```python
import json

with open("ra2va_reference_speech_bgm.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

sample = rows[0]
video_path = sample["video"]
image_paths = sample["images"]
audio_paths = sample["audios"]
caption = sample["caption"]
```

Different training mixtures can be built by sampling the task JSONL files at different ratios. No regeneration of captions or media is required when only the data mixture changes.

## Export inputs

The exporter accepts the already-frozen roots:

- `--ra2va-shadow-root`: reference-side R(A)2VA shadow root. It reads the frozen reference-conditioned Audio-reuse products and, when present, `h3_frame_conditioned_products_v1/records.jsonl` for first/last-frame variants.
- `--t2va-root`: frozen T2VA run root.
- `--ta2va-root`: frozen TA2VA run root.
- `--output-root`: new export directory. It must not already exist.

The three source roots are optional so individual families can be exported independently. Files for omitted families are still created as empty JSONL files.

## Command for later production use

When formal export is resumed:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate
source server_env.sh
export R2V_PYTHON="$(command -v python)"

"$R2V_PYTHON" tools/export_h3_training_manifests.py \
  --ra2va-shadow-root "$RA2VA_SHADOW_ROOT" \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-root "$TA2VA_ROOT" \
  --output-root "$EXPORT_ROOT"
```

Use a fresh `EXPORT_ROOT`. Do not write the export inside any frozen source run.

After running, the basic inspection is:

```bash
wc -l "$EXPORT_ROOT"/*.jsonl
head -n 2 "$EXPORT_ROOT/videos.jsonl"
head -n 1 "$EXPORT_ROOT/r2va_reference.jsonl"
head -n 1 "$EXPORT_ROOT/ra2va_reference_speech_bgm.jsonl"
head -n 1 "$EXPORT_ROOT/ta2va_full_audio.jsonl"
```

The exporter does not change any frozen caption, reference selection, speaker binding, Audio selection, media asset, or inference artifact.
