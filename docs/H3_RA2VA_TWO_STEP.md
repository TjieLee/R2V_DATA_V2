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
   existing `_speaker_profile_targets()`. The same unique required groups may be
   reordered by group key without changing traits. Profiles for known
   non-transcribed-only groups are excluded from publication, whether null or
   non-null; their Audio decisions and original raw remain intact. This does
   not imply absence of vocal activity. Missing/duplicate/unknown groups fail.
   Null is allowed when acoustic
   traits cannot be reliably described. No-transcript clips have no profiles,
   Sx or dialogue; no-segment clips keep empty Audio/AV inventories.
5. The final annotation remains `.20`, using the same Multi assembly, common
   normalization, `validate_annotation()`, exact-ASR checks and materializer.
   Original multiple-speaker/overlap ownership exclusions remain in force.

Successful clips have exactly two requests. There is no snippet extraction,
separate profile/finalizer call, marker polish, retry or Multi fallback. Failed
clips retain only the actual attempted stages and raw responses.

The joint authoritative input includes ordered frozen segment IDs and complete
`<d>[Language] exact text</d>` blocks. Only identical Audio segment rows are
deduplicated; conflicts fail before evidence canonicalization. Parsed Audio/AV
inventories must exactly match the frozen order, including no-transcript clips.
Missing dialogue language markers are restored only when the entire dialogue
count/order/text is exact. Original joint raw is never changed. Correction counts
are saved in diagnostic warnings as `deterministic_correction_count:<code>=<n>`.
Speaker marker warnings remain visible. Same-entity/group contradictions fail
with the existing validator issues and original annotation, before shared group
merging; no Sx guessing or group merge is used to raise readiness.

Only experimental provenance changes: backend `.83`, joint prompt
`h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity`. Version v2 adds visual
performance fidelity, source responsibilities and distinct-speaker narration
guidance only. It preserves meaningful Turn 1 actions and ordering rather than
targeting caption length. There is no new caption validator or automatic prose
rewrite. Backend `.82` changes only non-transcribed-only profile publication;
the v2 prompt stays byte-for-byte unchanged. The original `.80/.81` records
remain readable with their historical provenance.
Backend `.83` adds only the scoped pre-parse format normalization documented
below; `.82` records also remain readable. It changes no model request.
Visual v5, component profile v2/finalizer v6,
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
FROZEN="${FROZEN:-$SHADOW/mimo_reconcile_no_lrasd_http_final_v4}"

# Reuse the verified HTTP server rooted at /mnt/workspace. If not running,
# start this in a separate terminal in the same namespace as MiMo:
# .venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory /mnt/workspace

# Start with Two-step only: five clips, at most ten model requests.
# Use a fresh output name; there is deliberately no --overwrite.
MODE="${MODE:-two_step}"
OUT="${OUT:-$SHADOW/mimo_v26_ra2va_two_step_joint_v2_caption_fidelity_pilot5}"
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
    base_url="http://127.0.0.1:8094/v1", temperature=0.0, thinking="disabled",
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

### Retest the identical frozen five

After the inventory-format fix, use the first pilot's own frozen source contract,
not a rebuilt random20 inventory. Set the following variables, then run the
complete pilot block above; its defaults preserve these explicit values:

```bash
FROZEN="$SHADOW/mimo_v26_ra2va_two_step_joint_v1_pilot5"
MODE=two_step
OUT="$SHADOW/mimo_v26_ra2va_two_step_joint_v1_inventory_fix_pilot5"
```

This retains those exact five jobs, their order/Pictures/ASR and matching resolved
stems. The new output is separate and there is no overwrite. If the original
pilot used another directory name, set `FROZEN` to that actual pilot directory.
Use the existing endpoint and HTTP media server; do not rerun upstream models.
Inspect missing Sx and conflicting gN even when a format correction succeeds:

```bash
jq '{clip_uid,status,model_call_count,failure_reason,failure_issues,
     warnings:[.diagnostics[].warnings[]],
     caption:.annotation.h3_semantics.shot1_caption,
     speakers:.annotation.av_grounding.segment_groundings}' "$OUT/records.jsonl"
```

### Caption-fidelity v2 pilot

Keep the same first pilot's frozen five jobs and matching resolved stems. Set
these variables, then run the complete Python block above with the current code:

```bash
FROZEN="$SHADOW/mimo_v26_ra2va_two_step_joint_v1_pilot5"
MODE=two_step
OUT="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_caption_fidelity_pilot5"
```

Use the same V2.6 endpoint on port 8094 and verified HTTP media server on 8766.
The new output does not overwrite either v1 experiment. Keep the existing
per-clip timings, two raw responses, call counts, failures and QA warnings.
With the same caption viewer, compare the format-fixed v1 run (port 8770) with
v2 (port 8771):

```bash
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$SHADOW/mimo_v26_ra2va_two_step_joint_v1_inventory_fix_pilot5" --port 8770
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$SHADOW/mimo_v26_ra2va_two_step_joint_v2_caption_fidelity_pilot5" --port 8771
```

Inspect `shot1_visual_description` in the parsed `visual_raw_response` alongside the final
caption for retained intermediate actions, gaze/gesture changes and ordering.
Check speaker transitions, exact ASR, Subject/reference relationships and audio
field separation. The 195457 male/female speaker split remains correct; its
`She continues (S2)` issue is a model-narration quality check, not a deterministic
correction. CPU fake-model tests verify the request and unchanged output
handling, not that MiMo has learned the new narration.

## Compare quality, not just readiness

Manual QA corrections from the five-clip pilot:

- `1954570387e32d57a8726383`: one male and one female spoken line are two
  distinct speakers. Preserve both original gN groups and S1/S2; only reorder
  profiles from g1,g2 to the transcribed appearance order g2,g1, keeping each
  group's acoustic description. **QA ISSUE:** `She continues (S2)` incorrectly
  implies speaker continuity. Keep this caption/raw unchanged for review; do
  not merge speakers or automatically replace the prose or binding.
- `2b92925573f18e0508c3189f`: the two original `segment_0001` Audio rows are
  field-for-field identical. Deduplicate the parsed inventory, preserve both
  rows in the original joint raw, and record
  `joint_audio_segment_duplicate_removed=1`. Any differing duplicate remains
  a failure.

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

## Frozen random20 pilot after the profile fix

User-supplied v2 frozen5 result: 4 ready / 1 failed, 10 calls. The remaining
25755 failure had g1 on transcribed segments 1/3 and g2 on empty segments 2/4,
with a non-null g2 profile. Backend `.82` excludes only that extra profile; it
does not drop the observed g2 vocal activity, alter binding or change Caption.
The null exclusion keeps `joint_non_transcribed_null_profile_removed`; non-null
exclusion records `joint_non_transcribed_non_null_profile_excluded` plus a
group-specific QA warning explaining that traits remain in
`speech_av_raw_response`, not an identity-related published voice profile.
Local CPU tests reproduce the four-segment inventory with fake responses;
the server's real frozen5 raw was not available locally for replay.

Reuse the current V2.6 TP4 endpoint on 8094 and HTTP media server on 8766 rooted
at `/mnt/workspace`. Do not start a new model pool or rerun upstream producers.
This block reads **all** jobs from the action20 contract, without sampling or
reference reconstruction. Set `SHADOW` to that same named run if its actual
path differs. The output is fresh; no overwrite is used.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export MIMO_API_KEY="${MIMO_API_KEY:-local-no-key}"
AUDIO_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415
SHADOW="${SHADOW:-$AUDIO_ROOT/sam_audio_stem_shadow_v1/runs/ra2va-no-lrasd-random20-v1}"
FROZEN="$SHADOW/mimo_v26_ra2va_single_v10_ab_action20"
OUT="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_random20"
.venv/bin/python - "$FROZEN" "$OUT" <<'PY'
import json, os, sys, time
from pathlib import Path
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source

source, output = map(Path, sys.argv[1:])
contract = json.loads((source / "source_contract.json").read_text())
jobs = [MimoClipJob.model_validate(j) for j in contract["jobs"]]
_, stems, _ = load_stem_source(source.parent / "resolved_stems_v1")
config = MimoBackendConfig(
    api_key=os.environ["MIMO_API_KEY"], transport="sglang", model="mimo-v2.6-flash-rl",
    base_url="http://127.0.0.1:8094/v1", temperature=0.0, thinking="disabled",
    max_completion_tokens=32768, icl="official_ref2va_v1",
    media_resolver=MimoMediaResolver(mode="http", media_root=Path("/mnt/workspace"),
                                    media_base_url="http://127.0.0.1:8766/"),
)
backend = TwoStepOpenAIMimo26Backend(config, stem_records_by_clip={s.clip_uid: s for s in stems})
elapsed, corrections = {}, {}

class TimedBackend:
    provenance = backend.provenance

    def reconcile(self, job, **kwargs):
        started = time.perf_counter()
        try:
            result = backend.reconcile(job, **kwargs)
            corrections[job.clip_uid] = result.deterministic_correction_counts
            return result
        finally:
            elapsed[job.clip_uid] = round(time.perf_counter() - started, 3)

summary = run_mimo25_stem_reconcile_shadow(
    jobs=jobs, stem_records=stems, backend=TimedBackend(), output_root=output,
    source_clip_uids=[j.clip_uid for j in jobs], route="resolved", allow_unverified=True,
    binding_evidence_mode=contract["binding_evidence_mode"],
)
metrics = []
for line in (output / "records.jsonl").read_text().splitlines():
    r = json.loads(line)
    warnings = [w for d in r["diagnostics"] for w in d["warnings"]]
    counts = corrections.get(r["clip_uid"], {})
    for warning in warnings:
        if warning.startswith("deterministic_correction_count:"):
            code, _, count = warning.removeprefix("deterministic_correction_count:").partition("=")
            counts.setdefault(code, int(count))
    metrics.append({
        "clip_uid": r["clip_uid"], "status": r["status"],
        "elapsed_seconds": elapsed.get(r["clip_uid"]), "model_call_count": r["model_call_count"],
        "failure_reason": r["failure_reason"], "failure_issues": r["failure_issues"],
        "correction_counts": counts, "warnings": warnings,
        "qa_warnings": (r["annotation"] or {}).get("warnings", []),
    })
(output / "pilot_metrics.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metrics))
print(summary.model_dump_json(indent=2))
for metric in metrics:
    print(json.dumps(metric, ensure_ascii=False))
PY
```

Normal ready clips use two requests, so twenty normal completions use forty.
Failures are retained, not repaired; no profile/polish/retry/fallback call is
added. `records.jsonl` stores both real raw responses, complete annotation and
diagnostics; `pilot_metrics.jsonl` adds timing/correction counts. The same
records/source contract feed the existing read-only viewer.

Inspect results and the profile-exclusion audit:

```bash
jq '{clip_count,ready_count,failed_count,model_call_count}' "$OUT/summary.json"
jq '{clip_uid,status,elapsed_seconds,model_call_count,correction_counts,failure_reason,warnings,qa_warnings}' \
  "$OUT/pilot_metrics.jsonl"
jq 'select(.clip_uid == "25755cbc2545bad6928c736c") |
    {status,failure_reason,profiles:.annotation.audio_observation.speaker_voice_profiles,
     audio:.annotation.audio_observation.segment_decisions,diagnostics,speech_av_raw_response}' \
  "$OUT/records.jsonl"
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$OUT" --host 127.0.0.1 --port 8771
```

Open `http://127.0.0.1:8771/`. Compare Turn 1 actions with Caption, source
transitions (including the correct 195457 male/female split), exact ASR and
soundscape/music boundaries. No training export or materialization is required.

## Random20 format normalization and cached CPU replay

User-supplied `.82` server result: 11/20 ready, 9 failed, 38 requests. Backend
`.83` preserves Joint v2 and Visual v5, all raw strings, schemas and shared
validators. Its two scoped pre-parse changes are:

- Remove exactly `{"code":"possible_asr_conflict","segment_id":null}` only
  when the frozen segment inventory is empty. Keep other warnings unchanged.
  Record `unscoped_possible_asr_conflict_no_segments` in diagnostics and its
  deterministic correction count. This does not assert absence of vocals.
- Remove supported Picture source phrases only inside Subject definitions,
  and only when every cited Picture belongs to that frozen Subject's sources.
  Preserve appearance detail, labels, caption and retention. Unknown/mismatched
  sources, unsupported wording and empty cleaned definitions still fail.
  Record `visual_subject_picture_provenance_removed`.

The four response payloads supplied by the user are extracted into a focused
test fixture without media paths or fingerprints. Cached-response CPU results:

| Clip | CPU result |
| --- | --- |
| `aed328a5a4eb2ff45d0d4d66` | Complete backend validation passes; unscoped warning excluded and audited. |
| `02750d804022e9e8ba5f2964` | Warning fixed, but full validation still fails: the unchanged Visual prose contains `(S1)` through `(S6)`, rejected by `MimoVisualBlock`. No Sx/prose cleanup is added. |
| `86ca3660f631858590d00852` | Visual schema passes after three definition corrections; no Joint raw exists, so not a ready clip. |
| `aa28e97595dd82cbdc96457e` | Visual schema passes after one face-definition correction; no Joint raw exists, so not a ready clip. |

The other five reported Speaker/binding/ASR/composition failures remain outside
this fix. Do not infer a new Random20 ready count from partial replays.

### Read original server records without models or media reads

This command uses the original frozen jobs and raw strings, runs the actual
Two-step backend with cached completions, and never calls an endpoint. The
replay-only resolver supplies placeholder URLs without opening media. Token
usage/finish reasons are copied from saved diagnostics, not invented. Nothing
is written to the old output directory. No stems, source media or hashes are
scanned; no request is counted as a real model call.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
OLD="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_random20"
.venv/bin/python - "$OLD" <<'PY'
import json, sys
from pathlib import Path
from types import SimpleNamespace
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoBackendFailure, MimoVisualDraft
from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend, _canonicalize_two_step_visual_payload
from r2v_data_v2.structured_output import parse_structured_json_issues

root = Path(sys.argv[1])
jobs = [MimoClipJob.model_validate(j) for j in json.loads((root / "source_contract.json").read_text())["jobs"]]
records = {r["clip_uid"]: r for r in map(json.loads, (root / "records.jsonl").read_text().splitlines())}
selected = {"02750d804022e9e8ba5f2964", "aed328a5a4eb2ff45d0d4d66",
            "86ca3660f631858590d00852", "aa28e97595dd82cbdc96457e"}

class ReplayMedia:
    mode, media_root, media_base_url = "http", Path.cwd(), "http://cached.invalid/"
    def resolve(self, path):
        return self.media_base_url + path.name

for job in jobs:
    if job.clip_uid not in selected:
        continue
    row = records[job.clip_uid]
    report = {"clip_uid": job.clip_uid, "actual_model_calls": 0}
    if row["speech_av_raw_response"] is None:
        canonical, counts = _canonicalize_two_step_visual_payload(row["visual_raw_response"], job)
        visual, issues = parse_structured_json_issues(canonical, MimoVisualDraft)
        report.update(status="visual_schema_passed_joint_missing" if visual is not None and not issues else "visual_failed",
                      correction_counts=counts, issues=[i.to_dict() for i in issues])
    else:
        cached = iter(zip((row["visual_raw_response"], row["speech_av_raw_response"]), row["diagnostics"], strict=True))
        def create(**request):
            raw, diagnostic = next(cached)
            usage = diagnostic["usage"]
            return {"choices": [{"message": {"content": raw}, "finish_reason": diagnostic["finish_reason"]}],
                    "usage": {**usage, "prompt_tokens_details": {k: usage[k] for k in
                              ("image_tokens", "video_tokens", "audio_tokens", "cached_tokens")}}}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = MimoBackendConfig(api_key="cpu-replay", transport="sglang", model="mimo-v2.6-flash-rl",
                                   base_url="http://cached.invalid/v1", media_resolver=ReplayMedia())
        backend = TwoStepOpenAIMimo26Backend(config, stem_records_by_clip={}, client=client)
        labels = {r.picture_label for r in job.reference_images} | {s.subject_label for s in job.reference_subjects}
        try:
            result = backend.reconcile(job, segment_ids=[s.segment_id for s in job.segments],
                transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
                allowed_entity_ids=set(backend.build_compact_task_contract(job)["allowed_speaker_bindable_entity_ids"]),
                allowed_reference_labels=labels,
                auxiliary_audio_paths={kind: Path(kind + ".wav") for kind in ("speech", "music", "sfx")})
            report.update(status="cached_backend_validation_ready", correction_counts=result.deterministic_correction_counts)
        except MimoBackendFailure as exc:
            result = exc
            report.update(status="failed", failure_reason=exc.reason, issues=[i.to_dict() for i in exc.issues])
        report["qa_warnings"] = [w for d in result.diagnostics for w in d.warnings]
    print(json.dumps(report, ensure_ascii=False))
PY
```

### Only two Visual failures need new joint responses

After confirming the CPU report, reuse the existing TP4 endpoint on 8094 and
HTTP media server on 8766. The following runs just `86ca` and `aa28`, four
requests if both complete normally. It performs the normal unchanged Visual
request followed by Joint; it does not generate a missing Joint result during
CPU replay or overwrite Random20. If another hard failure appears, keep it.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export MIMO_API_KEY="${MIMO_API_KEY:-local-no-key}"
OLD="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_random20"
OUT="$SHADOW/mimo_v26_ra2va_two_step_joint_v2_format2_recheck"
.venv/bin/python - "$OLD" "$OUT" <<'PY'
import json, os, sys, time
from pathlib import Path
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source
source, output = map(Path, sys.argv[1:])
contract = json.loads((source / "source_contract.json").read_text())
selected = {"86ca3660f631858590d00852", "aa28e97595dd82cbdc96457e"}
jobs = [MimoClipJob.model_validate(j) for j in contract["jobs"] if j["clip_uid"] in selected]
_, stems, _ = load_stem_source(source.parent / "resolved_stems_v1")
config = MimoBackendConfig(api_key=os.environ["MIMO_API_KEY"], transport="sglang", model="mimo-v2.6-flash-rl",
    base_url="http://127.0.0.1:8094/v1", temperature=0.0, thinking="disabled", max_completion_tokens=32768,
    media_resolver=MimoMediaResolver(mode="http", media_root=Path("/mnt/workspace"), media_base_url="http://127.0.0.1:8766/"))
backend = TwoStepOpenAIMimo26Backend(config, stem_records_by_clip={s.clip_uid: s for s in stems})
class TimedBackend:
    provenance = backend.provenance
    def reconcile(self, job, **kwargs):
        started = time.perf_counter()
        try:
            return backend.reconcile(job, **kwargs)
        finally:
            print(job.clip_uid, "elapsed_seconds=", round(time.perf_counter() - started, 3), flush=True)
summary = run_mimo25_stem_reconcile_shadow(jobs=jobs, stem_records=stems, backend=TimedBackend(), output_root=output,
    source_clip_uids=[j.clip_uid for j in jobs], route="resolved", allow_unverified=True,
    binding_evidence_mode=contract["binding_evidence_mode"])
print(summary.model_dump_json(indent=2))
PY
jq '{clip_count,ready_count,failed_count,model_call_count}' "$OUT/summary.json"
jq '{clip_uid,status,model_call_count,failure_reason,failure_issues,diagnostics}' "$OUT/records.jsonl"
.venv/bin/python tools/serve_h3_single_v7_caption_review.py \
  --base-root "$OLD" --override-root "$OUT" --host 127.0.0.1 --port 8771
```

The viewer overlays the two new records for review only; the original Random20
directory remains unchanged. Correction counts and QA warnings are in
`diagnostics`; both real raw responses and annotation remain in `records.jsonl`.
