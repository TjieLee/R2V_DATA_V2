# H3 Speaker-Binding Audit V1

`binding_audit_v1` is a model-free, read-only diagnostic sidecar for the frozen
LR-ASD to DiariZen sparse-anchor mapping. It does not alter bindings, clusters,
thresholds, ASR records, or final H3 samples.

The audit validates and reads the existing production Audio and DiariZen
artifacts, then publishes:

- `binding_audit_v1/summary.json`
- `binding_audit_v1/clusters.jsonl`
- `binding_audit_v1/segments.jsonl`
- `binding_audit_v1/review_manifest.jsonl`

For each raw segment and cluster it measures overlap with every LR-ASD frame
where exactly one backend-native active visible face exists. Both accepted
face-to-entity associations and unmatched face tracks remain visible in the
evidence. Structural contradiction flags are factual observations only; direct
support duration, ratio, and anchor-length fields are review metrics, never new
acceptance thresholds.

Run the audit after the production Audio and DiariZen stages:

```bash
python tools/audit_h3_speaker_bindings.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

`AUDIO_PRODUCTION_ROOT` is the active JEA root whose direct children include
`audio/`, `diarization/`, and `h3/`; the tool does not add a second
`production/` path component. The command makes zero model calls. Existing
output is immutable by default; use `--overwrite` only to atomically replace
this audit sidecar. Source production artifacts are never modified. Summary
provenance includes aggregate hashes of every consumed Audio binding sidecar
and LR-ASD native artifact.
