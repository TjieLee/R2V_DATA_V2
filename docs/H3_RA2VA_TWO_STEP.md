# MiMo V2.6 Two-step pilot

This is an explicit experiment, not a new default. `--call-mode multi` remains
unchanged; Single Compact2/3 and T2VA/TA2VA are untouched.

## Contract

1. The original Multi Visual Turn 1 request is preserved: full muted target
   video, frozen Pictures, Visual v5, official ICL v4, `MimoVisualDraft`, and its
   existing canonicalization. CPU tests compare the complete request payload.
2. One original-AV request also receives the complete Visual draft, frozen
   DiariZen/Qwen3-ASR job, and resolved speech/music/SFX audio. Its thin
   `MimoJointAVAudioDraft` contains `MimoSpeechAVAssemblyDraft`, a list of the
   existing nullable `MimoSpeakerVoiceProfile`, and `MimoAudioFinalizeDraft`.
3. The joint prompt specializes the frozen speech/profile/finalizer prompts:
   stage-specific return/defer/snippet instructions become joint field
   ownership. Caption stays Visual prose plus speech presentation and exact
   dialogue; non-dialogue audio stays in the two final audio fields.
4. Profile targets are derived from the returned Speech/AV assembly with the
   existing `_speaker_profile_targets()`. Missing, extra, duplicate or reordered
   profile groups fail before common normalization. Null is allowed when acoustic
   traits cannot be reliably described. No-transcript clips have no profiles,
   Sx or dialogue; no-segment clips keep empty Audio/AV inventories.
5. The final annotation remains `.20`, using the same Multi assembly, common
   normalization, `validate_annotation()`, exact-ASR checks and materializer.
   Original multiple-speaker/overlap ownership exclusions remain in force.

Successful clips have exactly two requests. There is no snippet extraction,
separate profile/finalizer call, marker polish, retry or Multi fallback. Failed
clips retain only the actual attempted stages and raw responses.

Only experimental provenance changes: backend `.80`, joint prompt
`h3_mimo26_ra2va_two_step_joint_v1`. Visual v5, component profile v2/finalizer v6,
official ICL v4, authority v18, annotation `.20` and materializer v30 stay frozen.
The diagnostic `target_video_joint_av_audio` identifies the joint request.
Its raw is stored in `speech_av_raw_response`; independent profile/finalizer raw
fields remain null. Existing provenance fields are retained without new media
hashing, scans, seed gates or integrity infrastructure.

## Runner opt-in

The existing stem reconcile runner accepts:

```text
--call-mode two_step --model mimo-v2.6-flash-rl
```

Without an explicit output root this mode uses its own
`mimo_v26_ra2va_two_step_v1` shadow directory. Single-only compact/ICL flags are
rejected. Multi defaults and old 3/4/5-call record validation stay unchanged.
For the pilot below, use frozen jobs directly so reference sampling is not
repeated by the inventory builder.

## Five-clip frozen HTTP pilot

No GPU/model run was performed during development. Use the existing single
deterministic V2.6 TP4/Marlin endpoint on `http://127.0.0.1:8094/v1`.
Do not start a pool or rerun SAM/AuK/DiariZen/ASR.

Set `FROZEN` to the existing successful random20 reconcile whose
`source_contract.json` is authoritative. The frozen jobs and resolved stems
must be from the same named run. This example uses the accepted no-LR-ASD
random20 lineage documented in the Audio runbook; an existing action20 contract
can be supplied instead, with its own matching parent run.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export MIMO_API_KEY="${MIMO_API_KEY:-local-no-key}"
export MIMO_MEDIA_BASE_URL=http://127.0.0.1:8766/
AUDIO_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415
SHADOW="$AUDIO_ROOT/sam_audio_stem_shadow_v1/runs/ra2va-no-lrasd-random20-v1"
FROZEN="$SHADOW/mimo_reconcile_no_lrasd_http_final_v4"

# Reuse the verified HTTP server rooted at /mnt/workspace. If not running,
# start this in a separate terminal in the same namespace as MiMo:
# .venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory /mnt/workspace

# Start with Two-step only: five clips, at most ten model requests.
# Use a fresh output name; there is deliberately no --overwrite.
MODE=two_step
OUT="$SHADOW/mimo_v26_ra2va_two_step_joint_v1_pilot5"
.venv/bin/python - "$FROZEN" "$OUT" "$MODE" <<'PY'
import json, os, sys, time
from pathlib import Path
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend, run_mimo25_stem_reconcile_shadow
from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source

source, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
contract = json.loads((source / "source_contract.json").read_text())
jobs = [MimoClipJob.model_validate(j) for j in contract["jobs"][:5]]
_, stems, _ = load_stem_source(source.parent / "resolved_stems_v1")
config = MimoBackendConfig(
    api_key=os.environ["MIMO_API_KEY"], transport="sglang", model="mimo-v2.6-flash-rl",
    base_url="http://127.0.0.1:8094/v1", temperature=0, thinking="disabled",
    max_completion_tokens=32768, icl="official_ref2va_v1",
    media_resolver=MimoMediaResolver(mode="http", media_root=Path("/mnt/workspace"),
                                    media_base_url=os.environ["MIMO_MEDIA_BASE_URL"]),
)
cls = TwoStepOpenAIMimo26Backend if mode == "two_step" else StemAwareOpenAIMimo25Backend
backend = cls(config, stem_records_by_clip={s.clip_uid: s for s in stems})
elapsed = {}

class TimedBackend:
    provenance = backend.provenance

    def reconcile(self, job, **kwargs):
        started = time.perf_counter()
        try:
            return backend.reconcile(job, **kwargs)
        finally:
            elapsed[job.clip_uid] = time.perf_counter() - started

summary = run_mimo25_stem_reconcile_shadow(
    jobs=jobs, stem_records=stems, backend=TimedBackend(), output_root=output,
    source_clip_uids=[j.clip_uid for j in jobs], route="resolved", allow_unverified=True,
    binding_evidence_mode=contract["binding_evidence_mode"],
)
metrics = []
for line in (output / "records.jsonl").read_text().splitlines():
    record = json.loads(line)
    metrics.append({
        "clip_uid": record["clip_uid"], "status": record["status"],
        "elapsed_seconds": round(elapsed[record["clip_uid"]], 3),
        "model_call_count": record["model_call_count"],
        "failure_reason": record["failure_reason"], "failure_issues": record["failure_issues"],
        "warnings": [w for d in record["diagnostics"] for w in d["warnings"]],
        "qa_warnings": (record["annotation"] or {}).get("warnings", []),
    })
(output / "pilot_metrics.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metrics))
print(summary.model_dump_json(indent=2))
for metric in metrics:
    print(json.dumps(metric, ensure_ascii=False))
PY
```

If there is no matching V2.6 Multi result for those same five frozen jobs, run
the exact Python block again with `MODE=multi` and
`OUT="$SHADOW/mimo_v26_ra2va_multi_joint_ab_pilot5"`. That exercises the existing
unchanged Multi backend; do not compare an unrelated population/model as if the
call layout were the sole experimental variable. For three clips, change the
single `[:5]` slice to `[:3]` consistently in both runs.

Raw responses, stage diagnostics and structural failures are in `records.jsonl`.
The joint JSON is `speech_av_raw_response`, not a fabricated Turn 3/4 result.
`pilot_metrics.jsonl` adds elapsed time to the existing calls/failures/warnings.

## Compare quality, not just readiness

The existing read-only caption viewer can display projected annotations:

```bash
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$SHADOW/mimo_v26_ra2va_multi_joint_ab_pilot5" --port 8770
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$SHADOW/mimo_v26_ra2va_two_step_joint_v1_pilot5" --port 8771
```

Its collapsed raw panel shows the first response (Visual for both staged modes);
inspect `speech_av_raw_response` in records for the joint JSON. Profile fields
can be compared without materialization:

```bash
jq '{clip_uid,status,model_call_count,failure_reason,
     profiles:.annotation.audio_observation.speaker_voice_profiles,
     speakers:.annotation.av_grounding.segment_groundings,
     caption:.annotation.h3_semantics.shot1_caption,
     soundscape:.annotation.h3_semantics.overall_soundscape,
     music:.annotation.h3_semantics.non_diegetic_music}' "$OUT/records.jsonl"
```

Check scene/action/state progression, exact ASR/Sx order, entity/spatial speaker
judgments, supported acoustic traits and positive/absent audio layers. The main
quality risk is replacing targeted speaker snippets with full-stem/window
evidence. Report any profile degradation/null increase honestly; do not hide it
with a third request or promote Two-step on CPU readiness alone. No training
export, new reference assets, full random20 or production run is part of this
pilot.
