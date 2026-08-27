# H3 LoCoNet + LASER ASD Shadow Pilot

## Purpose And Boundary

This backend is an additive, read-only shadow pilot for empirical comparison
with the frozen LR-ASD Audio Binding path. Production remains LR-ASD by default.
The LASER pilot stops after Audio Binding and manual review. It does not feed
voice-quality calibration, primary-voice selection, embeddings, PairPolicy,
DiariZen, ASR, specialized semantics, or final H3 rendering.

The audited source is
[`plnguyen2908/LASER_ASD`](https://github.com/plnguyen2908/LASER_ASD), MIT
licensed, pinned to commit
`3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9`. The repository is external and is
not vendored into R2V_DATA_V2.

## Why The Official Raw Demo Is Not Invoked Directly

At the pinned commit, `LoCoNet/demoLoCoNet_landmark.py` exposes
`--pretrainModel`, but its inference function calls `model.loadParameters('')`.
It also creates an all-zero landmark tensor and calls `forward_evaluation` with
landmarks disabled. For more than one context candidate it uses
`random.shuffle`.

The R2V bridge therefore reuses the upstream 25 FPS S3FD and shot-aware IoU
preprocessing while explicitly:

- loading and checking the requested LoCoNet + LASER checkpoint;
- running MediaPipe FaceLandmarker with the exact upstream 82-index lip list;
- representing unavailable landmark coordinates as `(-1, -1)`;
- enabling landmark input in `forward_evaluation`;
- ordering tracks deterministically and selecting context tracks by stable
  first/last order.

These changes make the published LASER model path runnable and reproducible;
they do not claim support for TalkNCE + LASER or TalkNet + LASER.

## External Runtime

Use a dedicated environment and an external checkout at the pinned commit.
No runtime input is downloaded automatically and no LASER dependency belongs
in the main R2V requirements.

```bash
export LASER_ASD_CODE_ROOT=/path/to/LASER_ASD
export LASER_ASD_PYTHON=/path/to/laser-env/bin/python
export LASER_ASD_MODEL_PATH=/path/to/loconet_laser.model
export LASER_ASD_CONFIG_PATH="$LASER_ASD_CODE_ROOT/LoCoNet/configs/multi.yaml"
export LASER_ASD_LANDMARK_MODEL_PATH=/path/to/face_landmarker_v2_with_blendshapes.task

export SILERO_VAD_PYTHON=/path/to/audio-env/bin/python
export SILERO_VAD_MODEL_PATH=/path/to/silero_vad.jit
```

The code root, Python, checkpoint, config, and FaceLandmarker asset must all be
local files. The checkout commit is verified before inference. Checkpoint,
config, and landmark-asset SHA-256 values are recorded in the strict
`r2v.h3.laser_asd_native.1` artifact.

## Bounded Pilot

Run only an explicit clip list or bounded pilot limit. The output must be
outside the Visual source run.

```bash
python tools/eval_h3_audio_binding_laser.py \
  --run-root /path/to/completed-v3-run \
  --output-root /path/to/laser-shadow-pilot \
  --clip-id clip_uid_1 \
  --clip-id clip_uid_2
```

Each successful clip publishes:

```text
review/<clip_uid>/
  source.mp4
  visualization.mp4
  timeline.json
  audio_binding.json
  laser_asd_native.json
  face_entity_association.json
```

The pilot reuses the existing Visual mask association, Silero speech presence,
AudioBindingPolicy, and deterministic fusion. Its score is recorded honestly as
`laser_loconet_native_score`; it is not called a probability. Native active
state follows the upstream visualization rule `score >= 0`.

## Model-Free A/B Review

Build a static review from already completed LR-ASD and LASER roots:

```bash
python tools/build_h3_asd_backend_comparison_review.py \
  --lr-asd-root /path/to/lr-asd-pilot \
  --laser-root /path/to/laser-shadow-pilot \
  --output-root /path/to/asd-comparison-review
```

The utility copies review media into a separate output and never changes either
source root. It intentionally computes no automatic accuracy metric. Review
clear visible-speaker misses, wrong-person activation, multi-person scenes,
offscreen/background speech, and alternating speakers manually.

## Voice-Quality Limitation

`voice_reference_quality_v1` thresholds were calibrated on LR-ASD native
scores. LASER scores are not interchangeable with those values. This pilot
therefore publishes no voice-quality report and cannot produce primary voice
references until a separate, evidence-backed calibration is approved.
