# Single Compact3: shared Audio/H3 contract

This opt-in backend fixes missing Single-call evidence, not downstream product
definitions. Historical Compact2 and Multi defaults remain unchanged.

## Interface audit

```text
Multi visual / speech AV / acoustic profile / audio finalize calls
  -> MimoAVAnnotationDraft .20 -> existing Audio/H3/frame/training consumers

Single original AV + frozen Pictures + resolved speech/music/SFX (one call)
  -> Compact3 -> deterministic adapter -> MimoAVAnnotationDraft .20
  -> exactly the same consumers
```

| Field | Multi authority | Historical Compact2 | Compact3 authority / consumer |
| --- | --- | --- | --- |
| Visual blocks | Visual AV observation | Model prose | Model prose; H3 description unchanged |
| Exact-segment visibility | Visual Turn 1 | Derived only from bound speaker | Independent model `visible_subject_labels`, mapped through frozen Subjects; common AV validator |
| Detailed face/mouth observations | Visual Turn 1 when available | Empty | Empty means unsupplied, never observed; no synthesized lip evidence |
| gN | Speech AV acoustic grouping | Model gN | Model gN, reused by ownership, profiles and Sx projection |
| Resolution | Speech AV | Adapter: grouped means resolved | Actual model resolved/refinement/uncertain, same discriminated audio branches |
| Vocal composition | Actual segment audio | Adapter: uncertain | Actual model composition; shared `speaker_ownership_reasons()` |
| Secondary vocal activity | Actual segment audio | Adapter: absent | Actual model activity/relation/kind; shared ownership and overlap exclusions |
| Audio evidence / confidence | Speech AV | Adapter placeholders | Actual model values, unchanged in annotation |
| Binding | Speech AV + frozen entity graph | Model Subject mapped to entity | Model Subject mapped to frozen entity; never inferred from grouping |
| Presentation / AV evidence | Speech AV | Adapter-derived presentation + insufficient evidence | Actual model presentation/context/alignment evidence; common AV validator |
| AV confidence | Speech AV | Low placeholder | Model confidence; no deterministic upgrade |
| Delivery | Segment audio | Model delivery | Model delivery, existing ASR segment inventory |
| Voice profiles | Acoustic snippet call | Model acoustic prose | Model acoustic prose from speech evidence; no appearance-to-voice inference |
| ASR words / language / order | Frozen job | Frozen required dialogue | Same exact input facts and hard dialogue protection |
| Sx | First distinct transcribed source | Deterministic projection | Same projection; non-transcribed-only groups consume no Sx |
| Pictures / Subjects / face promotion | Frozen reference selection | Frozen graph | Same frozen graph; no resampling in downstream preparation |
| Soundscape / music | Audio finalize original AV + stems | Model fields | Same fields from original AV + stem evidence, no new sound rules |

Compact2 uses its narrow inventory/dialogue validator and does **not** run full
`validate_annotation()`. Compact3 also runs that existing full validator after
projection. Exact visibility, AV presentation evidence, group/entity identity,
reference inventory and no-LR-ASD checks remain hard. Only the existing Multi
`_REVIEW_ONLY_ISSUES` are warnings; no validator or severity is changed.

Audio composition branches reuse the existing Pydantic audio models. gN alone
never certifies an isolated waveform. Uncertain ownership remains excluded;
confirmed overlap/sequential multiple-speaker composition remains excluded and
retains the existing H3 `multi_speaker_segment_requires_turn_refinement`
failure. Offscreen/unbound isolated voices can remain entity-free speech assets.
No repair, retry, polish, LR-ASD, extra profile call or invented observation is
added. Actual evidence remains a model judgment, requiring real AV review.

New opt-in versions: backend `.79`, prompt
`h3_mimo26_ra2va_single_v11_compact3_audio_av`, compact schema `.3`.
Final annotation `.20`, official ICL v4, authority policy v18, materializers,
frame products and training row formats are unchanged. The synthetic action_v2
caption/summary/retention/profile content is unchanged; only new segment fields
and the compact schema version differ.

Two CLI entry points expose `--single-contract compact3` alongside their
existing Single call-mode flag. Compact3 defaults to its action_v2 semantic ICL.
Without this option, Compact2 retains baseline/dense_v1/action_v2 behavior.

## Local validation

The Single/raw-stem/reuse/preparation/H3/frame/export/production CPU regressions
passed: 501 passed, 20 skipped (FFmpeg unavailable), four known old named-run
fixtures deselected because they lack `resolved_stems_v1`. The wider historical
MiMo suites were also compared against a clean copy of starting HEAD
`bc314ca30ff8084a8ededd245e262fba0b3b3ae1`: baseline 858 passed / 207 failed /
20 skipped, patch 891 passed / 207 failed / 20 skipped. The failure-name sets
are identical; these suites are not claimed green. Ruff, Python 3.12
`py_compile`, and `git diff --check` passed. No model or production data was run.

## Separate existing downstream limitation

`select_reuse_audio()` currently adds acoustic profile text to cross-voice
contracts, but target `speaker_speech_reuse` contracts leave it null. The H3
materializer applies its profile projection before injecting reuse contracts.
Thus a valid profile survives in annotation/prepared records but is not currently
rendered as `featuring ...` for target reuse in **either** backend. This patch
does not change that common behavior or introduce a Single-specific workaround.

## Small server A/B using identical frozen jobs

No real inference was run during development. Run only 3-5 clips first. Reuse
an existing reliable Multi output only if its model, frozen jobs and upstream
stems match; otherwise use the commands below. These call existing backend and
publisher APIs directly, so reference selection is not repeated between A/B.
The same raw-DiariZen/Qwen3-ASR facts and face promotions are retained.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export MIMO_API_KEY="${MIMO_API_KEY:-local-no-key}"
SHADOW=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415/sam_audio_stem_shadow_v1/runs/random20-voicefirst-v1
FROZEN="$SHADOW/mimo_v26_ra2va_single_v10_ab_action20"

# Start the existing accepted deterministic V2.6 TP4/Marlin endpoint on 8094.
# Both paths read the very same five frozen jobs, original canonical metadata,
# and resolved AuK/SAM stems. Choose new output names if these already exist.
for MODE in multi single; do
  .venv/bin/python - "$FROZEN" "$SHADOW/mimo_v26_compact3_ab_${MODE}5" "$MODE" <<'PY'
import json, os, sys, time
from pathlib import Path
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_single_backend import SingleCallOpenAIMimo25Backend
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend, run_mimo25_stem_reconcile_shadow
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source

source, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
jobs = [MimoClipJob.model_validate(j) for j in json.loads((source / "source_contract.json").read_text())["jobs"][:5]]
_, stems, _ = load_stem_source(source.parent / "resolved_stems_v1")
config = MimoBackendConfig(
    api_key=os.environ["MIMO_API_KEY"], transport="sglang", model="mimo-v2.6-flash-rl",
    base_url="http://127.0.0.1:8094/v1", temperature=0.0, thinking="disabled",
    max_completion_tokens=32768, icl="official_ref2va_v1",
    media_resolver=MimoMediaResolver(mode="base64", media_root=Path("/mnt/workspace")),
)
cls = SingleCallOpenAIMimo25Backend if mode == "single" else StemAwareOpenAIMimo25Backend
backend = cls(config, stem_records_by_clip={s.clip_uid: s for s in stems},
              **({"single_contract": "compact3"} if mode == "single" else {}))
started = time.perf_counter()
summary = run_mimo25_stem_reconcile_shadow(
    jobs=jobs, stem_records=stems, backend=backend, output_root=output,
    source_clip_uids=[j.clip_uid for j in jobs], route="resolved",
    allow_unverified=True, binding_evidence_mode="none",
)
print(summary.model_dump_json(indent=2))
print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
PY
done

# Identical CPU-only consumers, new output roots; failures retain real reasons.
for MODE in multi single; do
  .venv/bin/python tools/materialize_h3_ra2va_training_bundle.py \
    --reconcile-root "$SHADOW/mimo_v26_compact3_ab_${MODE}5" \
    --output-root "$SHADOW/ra2va_compact3_ab_${MODE}5_bundle" --allow-unverified
done
```

If matching canonical H3 metadata is stored elsewhere, supply its existing
`--source-h3-root`; do not fabricate missing metadata. This is the same frozen
bundle path for both backends. Original canonical Pictures are retained in
prepared samples and the existing materializer projects them exactly once.

Review actual per-clip `model_call_count` and diagnostics, elapsed time,
ready/failed reasons, ownership exclusions, speech/music/full-audio availability,
H3 failures, frame 3N coverage and the twelve existing task counts. A normal
Single request is exactly one; Multi normally uses four with profile targets,
three without, plus its existing eligible polish call if triggered. Caption
length/ready count alone is not success. Check scene/action/time progression,
speaker/Subject referents (especially the 469d table), exact ASR/Sx, and whether
any positive evidence was hallucinated. Human PASS/ISSUE review still applies;
do not promote Compact3 to the default on CPU test results alone.
