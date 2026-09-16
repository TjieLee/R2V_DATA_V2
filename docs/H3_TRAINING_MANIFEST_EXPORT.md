# H3 training manifest export

Use `tools/export_h3_training_manifests.py` to flatten the frozen H3 products into one summary JSONL plus one JSONL per training task. The exporter is read-only with respect to all frozen R(A)2VA / T(A)2VA artifacts.

Output:

```text
<output-root>/
  videos.jsonl
  r2va.jsonl
  ra2va.jsonl
  t2va.jsonl
  ta2va.jsonl
```

`videos.jsonl` contains only:

```json
{"video":"/path/to/video.mp4","tasks":["r2va","ra2va","t2va","ta2va"]}
```

Every task file uses exactly this loader-facing row shape:

```json
{"video":"/path/to/video.mp4","images":["/path/to/p1.png"],"audios":["/path/to/a1.flac"],"caption":"final caption"}
```

Missing modalities use empty arrays. Exported rows intentionally contain no hashes, fingerprints, stage names, or internal provenance. Media payloads remain external files referenced only by path.

Run:

```bash
"$R2V_PYTHON" tools/export_h3_training_manifests.py \
  --ra2va-shadow-root "$RA2VA_SHADOW_ROOT" \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-root "$TA2VA_ROOT" \
  --output-root "$EXPORT_ROOT"
```

Arguments are optional except `--output-root`; omitted task families produce empty JSONL files. The output root must not already exist.

Task mapping:

- `r2va.jsonl`: ready reference-conditioned `visual_only` products; reference Picture paths are taken from the frozen prepared MiMo inventory.
- `ra2va.jsonl`: ready reference + Audio products; Audio paths come from the final published Audio contracts.
- `t2va.jsonl`: ready T2VA samples; video comes from the frozen T2VA inventory and caption comes from the published per-clip prompt.
- `ta2va.jsonl`: ready TA2VA products; video is resolved through the frozen source T2VA inventory and Audio paths come from the TA2VA product row.

The exporter does not change any frozen caption, reference selection, speaker binding, Audio selection, or inference artifact.
