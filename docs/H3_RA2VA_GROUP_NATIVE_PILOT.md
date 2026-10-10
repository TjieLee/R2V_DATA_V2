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

The commands below are **prepared, not executed**. Sync the reviewed code to
Node A first; this change is locally committed and has not been pushed.
Paths were checked by read-only SSH, not by loading weights. Select unused GPUs
and ports before starting; an occupied SGLang port is rejected, not adopted.

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

## Remaining Validation

Real SAM/AuK/DiariZen/ASR/SGLang loads, GPU memory release, node-local media
fetching and real end-to-end quality are untested in this change. First run
10-20 clips on one node after approval, then the same HTTP plane on two GPU nodes.
No GPU models, production group, or remote outputs were started by Codex.

## Local CPU Validation

- Task 1: 39 focused tests passed, including unchanged Fake HTTP execution.
- Task 2: 27 focused tests passed for isolated stage processes and shared TP4
  lifecycle. Final lifecycle regressions also cover exited group leaders,
  detached children, busy ports and event acknowledgments arriving after expiry.
- Final Group suite: 125 passed in 57.78 seconds. The 20-clip HTTP Native
  fixture case took 10.42 seconds, with 160 unique terminal results, 60 products,
  240 four-field tasks, 20 donors and 40 segments. It made zero real model calls;
  40 MiMo requests were simulated. These counts are not GPU Pilot expectations.
- Related Two-step, Single, Multi-call, T2VA lifecycle and Audio/H3 consumers:
  511 passed, 25 skipped, 4 failed. All four failures were reproduced on the
  original `6ee2743` baseline: old Turn 2 prompt length, Visual v4 and backend
  `.61` assertions disagree with already-existing v5/`.66`. No protected code
  or historical test was changed to hide these failures.
- Ruff, Python compilation and `git diff --check` passed. Independent review
  found no remaining blocking issue after the process/event lifecycle fixes.
- Read-only SSH confirmed runtime paths and read only the first source row of
  group-000000. No full source JSONL traversal, model load or server write ran.
