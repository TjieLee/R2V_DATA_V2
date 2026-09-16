# H3 training manifest export

Use `tools/export_h3_training_manifests.py` to flatten frozen H3 products into one summary JSONL plus one JSONL per training task. The exporter is read-only with respect to frozen R(A)2VA / T(A)2VA artifacts.

Each task row has exactly:

```json
{"video":"/path/to/video.mp4","images":["/path/to/p1.png"],"audios":["/path/to/a1.flac"],"caption":"final caption"}
```

Missing modalities use `[]`. Exported rows intentionally contain no hashes, fingerprints, stage names, or internal provenance.

Output task files:

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

`videos.jsonl` contains one row per target video and lists the exact available fine-grained task names:

```json
{"video":"/path/to/video.mp4","tasks":["r2va_reference","ra2va_reference_full_audio","t2va"]}
```

Visual modes:

- `reference`: frozen selected reference Pictures;
- `first_frame`: extracted first frame;
- `last_frame`: extracted last frame;
- `first_last_frame`: both extracted first and last frames.

Audio modes:

- `full_audio`: canonical full target-audio reuse;
- `speech_bgm`: target speaker-speech reuse plus optional BGM/music reuse. If the finalized product contains no music reference, the task remains `speech_bgm` and `audios` contains only the speech track(s).

Cross-voice/reference-only Audio variants are not exported by this training manifest layer.

Run:

```bash
"$R2V_PYTHON" tools/export_h3_training_manifests.py \
  --ra2va-shadow-root "$RA2VA_SHADOW_ROOT" \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-root "$TA2VA_ROOT" \
  --output-root "$EXPORT_ROOT"
```

For R(A)2VA, the exporter reads both the reference-conditioned final products and, when present, `h3_frame_conditioned_products_v1/records.jsonl` under the same shadow root. Arguments are optional except `--output-root`; omitted families produce empty task JSONL files. The output root must not already exist.

The exporter does not change any frozen caption, reference selection, speaker binding, Audio selection, or inference artifact.
