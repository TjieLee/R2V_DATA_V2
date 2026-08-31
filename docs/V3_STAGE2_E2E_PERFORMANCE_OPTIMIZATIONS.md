# V3 Stage2 / E2E SAM3 performance optimizations

This note records two production-relevant optimizations validated on the JEA Visual V3 Stage2 pipeline on 2026-08-28. They are intended to be reusable when Stage2 is embedded back into an end-to-end Visual V3 run.

The frozen Visual algorithm is not changed. In particular, do not modify `frames.py`, `segment.py`, `sam3_backend.py`, `sam3_anchor_selector.py`, `rank.py`, or `background.py` to apply these optimizations.

## 1. Disable expensive debug image generation in production

For throughput-oriented runs that do not need debug images, use:

```yaml
sam3:
  save_debug_overlays: false

debug:
  save_diagnostics: false
```

The frozen `segment.py` writes SAM3 overlay/contact-sheet images when either `debug.save_diagnostics` or `sam3.save_debug_overlays` is true. Therefore setting only `save_debug_overlays: false` is not sufficient when `save_diagnostics: true`.

Measured 80-row / 8-GPU canary:

```text
save_diagnostics=true,  reuse=off : real 5m11.471s
save_diagnostics=false, reuse=off : real 4m59.722s
wall-time reduction               : 11.749s (~3.8%)
```

This optimization removes diagnostic image generation only; SAM masks remain published in `masks.rle.json`. Final entity reference PNGs are created later by the Pair stage, not by SAM segmentation.

Changing `debug.save_diagnostics` changes the base-config identity/fingerprint. Do not switch this value inside an existing Stage2 output root. Start a fresh output root when changing the production config.

For future E2E runs, prefer `save_diagnostics: false` unless the run is specifically collecting debug artifacts.

## 2. Reuse one physical SAM3 session per clip

Implementation commit:

```text
ec2da87a2a4753052192147332ffe2061fbb2896
```

Standalone Stage2 flag:

```bash
--sam3-session-reuse-mode clip_reset_v1
```

The optimization is implemented outside the frozen SAM3 backend through a transparent predictor proxy:

```text
one clip / one frames_dir
    -> one physical start_session
    -> frozen logical probe/track call
    -> reset_session at every frozen logical close
    -> next frozen logical probe/track reuses the same physical session
    -> real close_session only when the clip changes or the worker closes
```

Every frozen logical session boundary is preserved. Prompt contents, probe order, retries, Qwen anchor selection, forward/backward propagation, masks, thresholds, and published schemas remain unchanged.

### Real GPU timing result

Baseline without debug diagnostics:

```text
reuse=off          : real 4m59.722s
clip_reset_v1      : real 3m50.727s
wall-time reduction: 68.995s (~23.0%)
```

Compared with the original diagnostics-enabled baseline, the combined effect of both optimizations was:

```text
5m11.471s -> 3m50.727s
wall-time reduction: ~25.9%
```

SAM3 request accounting changed from repeated physical session creation to one physical session per SAM clip:

```text
logical start_session calls : 665
physical start_session calls: 54
reused start_session calls  : 611
physical reset_session calls: 665
physical close_session calls: 54
```

Request timing:

```text
baseline start_session total : 461.286s
reuse start_session total    : 38.646s
reuse reset_session total    : 0.107s
baseline close_session total : 87.370s
reuse close_session total    : 9.749s
baseline SAM track total     : 750.110s
reuse SAM track total        : 245.491s
```

The logical algorithm counters remained unchanged in the measured run:

```text
anchor_probe_calls        : 537 -> 537
anchor_fallback_attempted : 47  -> 47
anchor_fallback_hits      : 2   -> 2
anchor_all_frames_not_found: 45 -> 45
propagate calls           : 128 -> 128
SAM clips                 : 54  -> 54
```

### Real-output semantic check

The baseline and `clip_reset_v1` runs produced:

```text
mask files compared: 54
different masks    : 0
```

All 54 published `masks.rle.json` artifacts were exact JSON-equal in the real GPU A/B. This is stronger evidence than the fake-predictor unit tests and confirms mask-level semantic equivalence for this canary.

A final canonical `parts/` diff should still be run when promoting a new production root or when changing surrounding orchestration/configuration.

### E2E reuse guidance

If future E2E code uses the standalone Stage2 runner, pass:

```bash
--sam3-session-reuse-mode clip_reset_v1
```

`--sam3-request-timing` is diagnostic only and should normally be disabled for production.

If a future E2E runner invokes the frozen Visual stages directly instead of the standalone Stage2 runner, reuse the same predictor-proxy strategy rather than editing the frozen SAM3 backend:

```text
persistent SAM3 backend per GPU worker
+ one physical SAM3 session per clip/frames_dir
+ reset_session between every frozen logical session
+ real close_session on clip switch / backend shutdown
```

Do not reuse an un-reset session across probes or propagation directions.

`clip_reset_v1` has its own runtime identity marker. Do not mix `off` and `clip_reset_v1` work inside the same Stage2 output root.

## 3. Vectorized binary-mask RLE codec

The V3 binary-mask codec now finds run boundaries with a NumPy change-point
scan and decodes runs with `numpy.repeat`. It no longer executes a Python loop
for every mask pixel. The existing contract is unchanged: masks are flattened
in C / row-major order, counts begin with the zero run, and serialized
`MaskRle` JSON remains exactly identical to the scalar implementation.

Correctness coverage includes exhaustive enumeration of every binary mask up
to 2x3 and 3x2, fixed-seed randomized masks through 720p at foreground ratios
from 0% to 100%, and realistic checkerboard, stripe, rectangle, fragmented,
sparse, and dense patterns. Each vectorized encoding is compared with a
test-only copy of the old scalar algorithm before round-trip decoding.

Run the local developer benchmark without a GPU or model dependency:

```bash
.venv/bin/python tools/benchmark_mask_codec.py
```

It reports scalar and vectorized encode time, vectorized decode time, encode
speedup, throughput, and exact equality. Its timing is diagnostic only; CI does
not assert a machine-dependent speedup.

### Real server A/B result

The server A/B used the same persisted 80 rows, the same config,
`--chunk-rows 1`, and `--sam3-session-reuse-mode clip_reset_v1`. Debug
diagnostics were disabled. SAM3 request timing was also disabled for the
vectorized-RLE run.

Scalar-RLE session-reuse baseline:

```text
output: /mnt/workspace/litengjie/data/r2v_v3_stage2_canary/chunk1-nodiag-sessionreuse-80-qwen-20260828
real: 3m50.727s (230.727s)
```

Vectorized-RLE run:

```text
real: 1m43.285s (103.285s)
user: 15m28.505s
sys : 1m26.107s
```

This reduced wall time by 127.442 seconds, approximately 55.2%, and delivered
approximately 2.23x throughput relative to the scalar-RLE session-reuse
baseline.

Across the complete optimization sequence, the original diagnostics-enabled
baseline improved from 5m11.471s to 1m43.285s: approximately 66.8% less wall
time and approximately 3.02x throughput.

Correctness comparison covered all 54 produced `masks.rle.json` files:

```text
compared:  54
different: 0
```

The serialized RLE JSON was exactly identical. Canonical `parts/` equality is
still a separate promotion check and is not claimed here until its diff result
is confirmed.

The vectorized codec can be reused by future end-to-end Visual V3 execution
without changing mask semantics or configuration identity. Do not restore the
Python per-pixel RLE implementation.

## Recommended production combination

For a throughput-oriented Stage2 or E2E run after the above checks:

```yaml
sam3:
  save_debug_overlays: false

debug:
  save_diagnostics: false
```

and, where the runner supports it:

```bash
--sam3-session-reuse-mode clip_reset_v1
```

These optimizations are orthogonal: the first removes debug-image CPU work; the second removes repeated SAM3 physical session initialization while preserving frozen logical session boundaries.
