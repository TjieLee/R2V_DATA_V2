# RA2VA Group Native Worker Pilot

## Scope

Explicit `worker --native` connects the existing eight Group stages to HTTP
claims. `coordinator --execution-mode native` is a separate persisted run mode;
old Fake runs cannot be resumed as Native. No T2VA, Multi/Single, Visual v5,
Joint v2, ASR/Speaker or training task semantics are changed.

The local CPU fixtures exercise real media conversion, Audio/H3 consumers,
12 four-field manifests and all eligible donor segments. Their model responses
are simulated, not evidence of GPU/model quality or actual server throughput.
Real GPU execution requires approval. Do not launch a full group.

## Interfaces And Recovery

- Claims contain read-only `upstream_results` paths for all preceding stages and
  a unique `artifact_root` below `stages/<generation>-<stage>/attempts/<ordinal>/<token>`.
  Only Coordinator publishes terminal receipts and group-level bundle indexes.
- Each Native worker owns one persistent inference subprocess per stage with
  physical `CUDA_VISIBLE_DEVICES`; backends use `cuda:0`. Heartbeats stay in the
  parent. Watchdog/stop closes the inference process group and observed detached
  children before release or exit. Infrastructure failures stop the launcher;
  restart with the original run-id. Business failures remain terminal.
- One launcher owns the node-local MiMo replicas. Disjoint four-GPU groups use
  the frozen T2VA TP4+Marlin command. Local workers share those endpoints with
  one in-flight clip per replica. Models start once per active stage context,
  not per clip, and stop before the final local release acknowledgment.
  Do not run multiple Native launchers over the same node GPU set.
- The MiMo child persists request intent, then waits for a durable
  `/v1/tasks/event` acknowledgment before calling MiMo. Visual and Joint events
  are bound to group/stage/generation/ordinal/clip/turn/claim token.
  Lost HTTP acknowledgments repeat the event or terminal receipt, not inference.
- Saved Visual resumes only Joint; both saved responses replay with zero new
  calls. Reclaimed claims read specifically authorized old response paths and
  write new files only to their new attempt. Raw/usage and per-turn inference
  provenance retain their actual source. A request with no saved response stays
  `interrupted_request_unresolved`, never an automatic retry or fallback.
  Its call count is the persisted request-intent audit, not proof that a lost
  remote request finished. Successful clips still have exactly two requests.
- Coordinator restart preserves unexpired claims. Lease expiration fences
  commits; it is not proof that remote GPU work stopped. Watchdog provides
  cooperative cleanup. Hard-killed launchers/hosts and GPU driver hangs still
  require operator process cleanup before restarting. No automatic failover.

## Approved Single-Node GPU Pilot Commands

The commands below limit execution to a new Pilot run. After explicit GPU
approval, the first 10 rows were used for the single-node run recorded below.
Sync reviewed code to `/mnt/workspace/litengjie/data/R2V_DATA_V2` on Node A
using `git pull --ff-only`; Git pushes are performed locally, never on servers.
Select unused GPUs and ports before starting; an occupied SGLang port is
rejected, not adopted.

Run this setup once in Node A Terminal 1. In Terminals 2 and 3, use the same
environment values, especially the exact printed RUN and Coordinator token.
Do not regenerate RUN in another terminal or on resume.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
export PY="$PWD/.venv/bin/python"
export SOURCE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask
export OUTPUT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
export RUN=ra2va-native-group0-pilot-$(date +%Y%m%dT%H%M%S)
printf 'RUN=%s\n' "$RUN"
export R2VA_RUN_ROOT="$OUTPUT/$RUN"
export HOSTNAME="$(hostname)"
read -rs -p 'Coordinator token: ' R2V_GROUP_COORDINATOR_TOKEN
export R2V_GROUP_COORDINATOR_TOKEN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Terminal 1, Coordinator only on Node A; `/tmp` must be real local storage:

```bash
"$PY" tools/run_h3_ra2va_group_production.py coordinator \
  --execution-mode native --source-root "$SOURCE" --output-root "$OUTPUT" \
  --run-id "$RUN" --limit 20 --max-groups 1 \
  --host 127.0.0.1 --port 8780 --local-lock-path "/tmp/$RUN.coordinator.lock"
```

`--limit 10` is also supported, but must stay unchanged on resume. Inventory
freezing reads only the requested prefix of the published group; it does not
traverse the full samples.jsonl or inspect all source media.

Terminal 2, media service on the **same node as SGLang**:

```bash
"$PY" -m http.server 8766 --bind 127.0.0.1 --directory /mnt/workspace
```

Reuse an existing HTTP media service only if it already serves this root.
Each node must have its own explicit reachable media address; another node's
`127.0.0.1` is not a shared endpoint.

Terminal 3, one node launcher, four ordinary GPU workers and one shared TP4 replica:

```bash
DEPS=/mnt/workspace/litengjie/data/audio_deps
export PYTHONPATH="$PWD:$DEPS/sam-audio-src:$DEPS/perception-models-src:$DEPS/dacvae-src:$DEPS/sam-audio-pydeps${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/run_h3_ra2va_group_production.py worker --native \
  --coordinator-url http://127.0.0.1:8780 --node-id "$(hostname)" --workers 4 \
  --configuration docs/examples/ra2va_group_native_node.json
```

The example uses GPUs 0,1,2,3 and port 8094. Adjust the node-local JSON for available
resources, not frozen prompts or sampling rules. Two TP4 replicas can use
`gpu_groups=["0,1,2,3","4,5,6,7"]`, `ports=[8094,8095]` and eight distinct
ordinary `gpu_ids`. The main Python must have the same SAM dependencies used by
the existing T2VA runner; isolated AuK/DiariZen/Qwen-ASR interpreters are reused.
The Group SGLang process removes inherited `PYTHONPATH` so SAM's compiled
dependencies cannot shadow packages in the dedicated MiMo virtualenv.

## Progress, Results And Stop

```bash
"$PY" tools/run_h3_ra2va_group_production.py status --output-root "$OUTPUT" --run-id "$RUN"
curl --noproxy '*' -s -H "Authorization: Bearer $R2V_GROUP_COORDINATOR_TOKEN" \
  http://127.0.0.1:8780/v1/status | "$PY" -m json.tool
```

Per-stage pending/active/ready/failed/skipped, elapsed time and throughput are
persisted. Worker logs include model load/release timings; receipts contain
per-clip execution time and actual response diagnostics/correction counts.

Accepted group products are indexed under:

```text
$R2VA_RUN_ROOT/groups/group-000000/bundle/
  source_contract.json
  records.jsonl
  summary.json
  training/*.jsonl
  voice_donor_reserve.json
```

The group has `PILOT_COMPLETE` only, never full-group `COMPLETE`. Report actual
MiMo Ready/Failed/calls, product failures, all 12 task counts and donor/segment
counts; do not substitute synthetic expected counts for real acceptance.

Stop with SIGTERM/Ctrl+C on the Native launcher first; it terminates its owned
stage/model subprocesses before returning. Then stop media and Coordinator.
Do not broadly `pkill` other workloads. Resume the same Coordinator command
and worker command with the original RUN, token, port, local guard and limits.
Completed terminal tasks are not re-executed. An unresolved MiMo request remains
Failed rather than being retried. No history is deleted or overwritten.

## Local CPU Validation

- Task 1: 39 focused tests passed, including unchanged Fake HTTP execution.
- Task 2: 27 focused tests passed for isolated stage processes and shared TP4
  lifecycle. Final lifecycle regressions also cover exited group leaders,
  detached children, busy ports and event acknowledgments arriving after expiry.
- Final Group suite after server integration fixes: 127 passed in 59.14 seconds.
  The earlier 20-clip HTTP Native
  fixture case took 10.42 seconds, with 160 unique terminal results, 60 products,
  240 four-field tasks, 20 donors and 40 segments. It made zero real model calls;
  40 MiMo requests were simulated. These counts are not GPU Pilot expectations.
- Related Two-step, Single, Multi-call, T2VA lifecycle and Audio/H3 consumers:
  511 passed, 25 skipped, 4 failed. All four failures were reproduced on the
  original `6ee2743` baseline: old Turn 2 prompt length, Visual v4 and backend
  `.61` assertions disagree with already-existing v5/`.66`. No protected code
  or historical test was changed to hide these failures.
- After the two server integration fixes, the focused Two-step, Single,
  Audio/H3, donor and T2VA lifecycle regression subset passed: 360 passed,
  7 skipped in 10.18 seconds.
- Ruff, Python compilation and `git diff --check` passed. Independent review
  found no remaining blocking issue after the process/event lifecycle fixes.
- Initial read-only SSH confirmed runtime paths and read only the first source
  row of group-000000. Subsequent explicitly approved GPU execution is recorded
  separately below; no full source JSONL traversal or full-group run was used.

## Single-Node Server Pilot, 2026-10-10

Node A: `r2v_data` / `nb-94ayoy238c-0`, four ordinary workers on GPUs 0-3,
one shared MiMo V2.6 TP4+Marlin replica on port 8094. Existing HTTP media service
8766 serves `/mnt/workspace` and is not owned or stopped by this run.

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/
  ra2va-native-group0-pilot10-20261010T034522/
```

The frozen inventory is only the first 10 rows of published group-000000,
`limit=10`, `max_groups=1`. The source's publication marker reports 39,585 rows;
the Pilot does not traverse them or mark the full source group COMPLETE.

Two runtime integration errors were found before any MiMo request:

- ASR context lacked its existing required `ffmpeg` configuration. Fixed only
  in the Group adapter, with a regression exercising the existing ASR context.
- SGLang inherited SAM's `PYTHONPATH`, loading an incompatible SciPy against
  MiMo's NumPy. Fixed by isolating the Group SGLang environment, with a regression.

The same run-id resumed at the missing stage after each locally pushed fix and
server fast-forward pull. Earlier terminal results were reused, not recomputed.
Original `worker.log`, `worker-resume1.log`, `worker-resume2.log` and corresponding
Coordinator logs retain the actual failures and restart evidence.

Runtime code on the server: `3c68ee7881057c11c6a3a4b4049521c86cca9db5`.
All eight stages completed, with 80 clip-stage terminal receipts and no pending
or active claims. Only `PILOT_COMPLETE` exists, not full-group `COMPLETE`, and
only group-000000 was started. The following are actual server results:

| Stage | Ready | Failed | Skipped | Persisted wall seconds |
| --- | ---: | ---: | ---: | ---: |
| canonical | 10 | 0 | 0 | 5.53 |
| SAM | 10 | 0 | 0 | 100.57 |
| AuK | 10 | 0 | 0 | 133.09 |
| resolve | 10 | 0 | 0 | 2.63 |
| DiariZen | 10 | 0 | 0 | 38.22 |
| ASR | 10 | 0 | 0 | 235.58 |
| MiMo Two-step | 8 | 2 | 0 | 785.21 |
| export | 6 | 2 | 2 | 7.96 |

ASR/MiMo persisted wall times include the startup errors, local fixes, pulls
and restart delays, so are not steady-state throughput measurements. The final
MiMo/export invocation took 556.58 seconds, including 94.87 seconds of SGLang
cold startup. The 20 saved real requests span 449.55 seconds (448.81 seconds
summed request durations). Every clip has exactly two requests and two saved
responses, zero replayed requests and no model retry/polish/fallback.
The SGLang HTTP log has 21 completion POSTs: its built-in startup warmup at
12:00:21 and the 20 audited clip requests ending at 12:07:47. The startup source
uses a generated warmup image and at most 8 output tokens; it is not a clip
request, retry or additional pipeline turn.
SAM receipts retain `separation_state=unverified` with ranking disabled; they
were not relabeled as verified separation success.

Actual terminal business failures, left unchanged:

- `d897696ed13b1ea1b2104ea2`: `visible_entity_binding_not_permitted` on
  segment_0001. Visual articulation is `uncertain`; joint evidence contains
  `av_temporal_alignment` and `no_visible_lip_motion`. The frozen existing
  binding policy rejects it; this integration does not alter that policy.
- `6107ee1e8a646cccb1583b3f`: `segment_inventory_mismatch` in joint audio
  decisions. Both real responses are saved; export is skipped.
- `ab27449eaf98b06401161810` and `83871dd0401c1f1ce9986ec2`: MiMo Ready, but
  existing H3 publication rejects visual-only/full-audio products with
  `two_step_identity_caption_publication_restricted`. Their unconfirmed
  identity attribution cannot be safely neutralized without changing visual
  prose. No unsafe identity product was published and no caption was rewritten.

Bundle: **15 Ready / 4 Failed products**, **60 training tasks**, **2 donors /
6 segments**. Four Visual task types each have 6 rows, four Full Audio types
each have 6, and four Speech+BGM types each have 3. All 60 task rows use exactly
`video/images/audios/caption`; the separate `videos.jsonl` is the 6-video index,
not another training task. Donors are 42038d7/e1/g1/S1 (2 segments) and
9fc15fd/e1/g1/S1 (4 segments); these clip-local e1 identities are not merged.
All six generated segment paths exist.

The launcher and Coordinator exited, ports 8094/8780 closed, and all eight GPUs
returned to 0 MiB. The pre-existing 8766 media server remains running. Server
tracked files are clean; unrelated pre-existing untracked files are untouched.

This validates single-node real stage execution, startup-failure resume and
bounded resource cleanup, not full production quality or a real mid-request
GPU interruption. Two-node GPU execution, hard-killed-host cleanup and a larger
production run remain untested. The four product failures remain visible in
the existing summaries; no manual QA/PASS gate or additional checks were added.
