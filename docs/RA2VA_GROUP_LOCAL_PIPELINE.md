# RA2VA Group Local Integration

Local CPU integration only. This does not accept real two-node locking/network
behavior, GPU resource cleanup or model quality. No server data was modified.
HTTP Workers remain fake-only until their separate real-node acceptance.

## Modules

- `ra2va_group_pipeline.py`: native V3 Picture/Subject/job metadata and the eight
  stage adapters; one model context per Worker/stage, no legacy Visual inventory.
- `ra2va_group_runtime.py`: wiring to existing SAM, isolated AuK, DiariZen,
  Qwen3-ASR and a supplied MiMo endpoint. These factories are not a GPU launcher.
- `ra2va_group_mimo.py`: persist request intent and full response before parsing;
  saved Visual resumes Joint only; response-less intent is unresolved, not retried.
  Inference provenance remains that of the original request; missing turns cannot
  silently mix changed request settings with an earlier Visual response.
- `ra2va_group_export.py`: existing Audio eligibility, H3 rendering, exact frame
  extraction, twelve-task export and every eligible donor segment. Native typed
  contracts express unavailable legacy provenance as null, never invented hashes.
- `ra2va_group_local.py`: bounded local CPU Worker integration using the existing
  scheduler/barriers. Its `cpu_pipeline_pilot` run cannot resume a Fake/HTTP run.
- `ra2va_group_cpu_fixtures.py`: explicit recorded model outputs, no network.

Legacy consumer hooks default to their original models/contracts. Multi/Single,
T2VA, prompts and shared Speaker/ASR validation are unchanged. Native Group AuK
requests use a distinct operation without a source-media digest scan; legacy
AuK requests retain their original behavior. SAM uses the existing music-first
prompts and preserves unverified state. Final Music conditioning follows the
annotation, not the mere presence of a separated stem.

## CPU Fixture Contract

`--cpu-fixtures` is JSON with `media_root` and a `clips` dictionary keyed by UID.
Each clip supplies `canonical_audio_path`, `diarization` (start/end/speaker_label),
`asr` (ordered text/language), `visual` (MimoVisualDraft) and `joint`
(MimoJointAVAudioDraft). Input remains a formally published V3 source group.
Fixture references may be read-only symlinks. No fixture path contacts MiMo.

```bash
python tools/run_h3_ra2va_group_cpu_pipeline.py \
  --source-root "$CPU_SOURCE" --output-root "$CPU_OUTPUT" \
  --run-id local-integration-20 --cpu-fixtures "$CPU_FIXTURES" \
  --workers 2 --limit 20 --max-groups 1 --ffmpeg "$FFMPEG"
```

Resume with the identical command and run-id. Terminal stage results are not
executed again. An unresolved MiMo request is terminal, not an implicit retry.
Audio stages persist their returned paths before scheduler publication. If the
process dies before that receipt, a fresh isolated attempt cannot overwrite an
incomplete file. Unpublished interrupted attempts remain inside their own run.
The CLI deliberately exposes no real-GPU mode. Never point synthetic fixtures
at an existing production run.

## Local Evidence (2026-10-10)

The retained test Pilot uses twenty synthetic two-second videos/audio inputs,
two CPU processes, existing real media/H3/export consumers, and fixture model
responses. It is not the server effective20 acceptance dataset.

- 160 unique clip-stage terminal results, all ready, two Workers with 80 each.
- 40 simulated Visual/Joint requests, zero real model requests.
- 60 ready / 0 failed products; 240 training tasks, twenty in each of 12 classes.
- 20 donors / 40 segments, exact source-PCM crop comparisons in tests.
- One `PILOT_COMPLETE`; no full-group `COMPLETE` and no second group.
- Resume preserves result/media bytes and invokes no FFmpeg/model operation.
- Semantic ASR mismatch stays failed; downstream export skips it, without retry.
- Saved Visual/interrupted Joint, lost responses and invalid structured raw have
  separate recovery tests. Original raw and usage are retained.
- A native isolated AuK worker using a fake engine also passes interruption after
  raw output and recovery, without colliding with that interrupted output.

Stage wall times for this small synthetic workload (not a GPU benchmark):

| Stage | Seconds |
| --- | ---: |
| Canonical | 0.15 |
| SAM fixture + actual canonicalization | 0.57 |
| AuK fixture + actual canonicalization | 0.22 |
| Resolve | 0.03 |
| DiariZen fixture | 0.03 |
| ASR fixture + actual audio slicing | 0.38 |
| MiMo fixtures + real validation | 0.14 |
| H3, frames, JSONL, donors | 0.47 |

Local retained evidence: `/tmp/ra2va-native-cpu-20261010-final/` and
`/tmp/ra2va-native-cpu-20261010-final.log`. The cached real group-000000 inventory
still contains its original twenty ordered V3 tasks; no server media was read.
The independent CLI run completed in 3.12 seconds; same-run resume took 0.87
seconds with an intentionally nonexistent FFmpeg executable. CLI logs are
`/tmp/ra2va-native-cpu-cli-20261010.log` and
`/tmp/ra2va-native-cpu-cli-resume-20261010.log`.
The final fresh CLI run completed in 2.95 seconds with the same output counts;
its log is `/tmp/ra2va-native-cpu-cli-final-20261010.log`.

## Verification

- Related regression: 568 passed, 39 skipped; final native-core check: 22 passed.
- Ruff, py_compile and staged diff-check passed.
- Broad suite is not green: 6469 passed, 144 skipped, 251 failed. Of those,
  250 failure nodes match the earlier baseline. The additional unchanged T2VA
  detached-child cleanup test also failed in an isolated checkout run, while
  two isolated baseline archive runs passed. Its cause remains unconfirmed;
  no T2VA code or test was changed to mask it.
- Logs: `/tmp/ra2va-group-local-acceptance.log`,
  `/tmp/ra2va-group-local-commit-check.log`,
  `/tmp/ra2va-group-local-full-final.log` and
  `/tmp/ra2va-group-local-baseline-pool*.log`.

## Remaining Acceptance

Real two-node HTTP claims/recovery and real Audio/MiMo GPU execution are still
pending. No automatic failover or GPU watchdog-cleanup guarantee is claimed.
Before real models, validate process/device isolation, release acknowledgments,
the single TP4 endpoint lifecycle and interrupted-request handling on the server.
Do not run a full group based on these CPU results.
