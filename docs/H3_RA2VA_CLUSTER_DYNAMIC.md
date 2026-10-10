# RA2VA Per-Node One-Command Launch

Each node runs the same Bash script. Choose exactly one Coordinator node;
other nodes discover its URL through the shared run directory and initiate HTTP
connections. No Worker IP list, SSH fan-out, rank sharding, or automatic election.
The existing central SSH launcher remains available and unchanged.

## Launch

On the designated Coordinator node (also runs local Workers):

```bash
R2VA_ROLE=coordinator bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

On every other node:

```bash
bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

Default run: `ra2va-group-pilot20-v1`, 20 clips, cumulative `max_groups=1`.
This remains a bounded Pilot, not full-group production. Restart the same commands
on the same designated Coordinator node to resume. Use an identical
`R2VA_RUN_ID` on every node for a new experiment. Workers may start first and wait
up to 300 seconds (`R2VA_WAIT_SECONDS`) for Coordinator readiness.
Worker-only commands exit after completion. The Coordinator command stays in the
foreground until Ctrl+C, including after local Workers finish, so delayed or
reconnecting nodes can still obtain the terminal state. SIGINT, SIGTERM and
SIGHUP use the same owned-process cleanup.

## Full-Group Production

Full-group scope is explicit and uses a different default run-id. No partial
source selection or Pilot completion marker is reused. On the designated node:

```bash
R2VA_ROLE=coordinator R2VA_FULL_GROUP=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

On every other node:

```bash
R2VA_FULL_GROUP=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

Default production run-id: `ra2va-group-production-v1`. Use the same run-id and
scope on restart. Full scope freezes all rows of each published group once,
in source order, without reading media or computing hashes. `R2VA_MAX_GROUPS=0`
(the full-scope default) processes newly published groups continuously;
`R2VA_MAX_GROUPS=1` stops after one complete group. This budget remains cumulative
across restarts. All nodes still share one current group and stage; there is no
rank/shard assignment. Only a full group writes `COMPLETE`; bounded runs continue
to write only `PILOT_COMPLETE`. `--dry-run` remains available on either command.

`MAX_CLIP_DURATION_SECONDS=20` is the default, matching the supplied T2VA setup.
Canonical probes video duration once and skips longer clips before audio
extraction or model inference. It does not truncate videos. Skipped clips retain
terminal state and are not retried on resume.

SAM, AuK and DiariZen use persistent per-GPU stage workers and per-clip inference,
as in T2VA. ASR now actually calls the batch API: each clip's ordered segments are
processed in batches of at most `ASR_BATCH_SIZE=8`, decoding its Speech stem once.
This does not change sample windows, segment ownership or transcript order.
Unlike T2VA's shard-wide segment queue, these batches do not cross clip receipt
boundaries. Models stay loaded across clips within a stage and are released at
the barrier before the next stage, not reloaded for every clip. MiMo remains two
TP4 replicas per eight-GPU node, Visual + Joint exactly two requests per completed
clip, with no additional model repair calls.

Full-group scope has CPU source/HTTP/barrier/resume regression coverage. The
16-clip real GPU Pilot below validates the existing model path, not a full-group
throughput or long-running stability measurement. Do not interpret a dry run as
starting production.

For an independent 8-clip Pilot after the original first 10 rows, use identical
`R2VA_START_ROW=10 R2VA_LIMIT=8` on both nodes. The zero-based start counts
nonempty source rows. Only the prefix through the selected range is read;
start row and limit are frozen in run control and inventory. Resume must use
the same values, and never silently selects another range.

Output is exclusively:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/<run-id>/
```

Defaults mirror the supplied T2VA node setup: `GPU_IDS=0,1,2,3,4,5,6,7`,
`MIMO_GPU_GROUPS='0,1,2,3;4,5,6,7'`, `MIMO_PORTS=8094,8095`,
`MIMO_MEM_FRACTION_STATIC=0.65`, `ASR_BATCH_SIZE=8`. `REQUEST_WORKERS`
defaults to the number of configured GPU IDs (8). These are runtime resource
settings, not task partitioning. MiMo stays V2.6 Two-step with existing prompts
and inference parameters; SGLang starts only in the MiMo stage.

Use `R2VA_CONFIGURATION=/absolute/node.json` for an existing custom Native
configuration; this preserves its GPU and backend settings instead of applying
the environment defaults. `R2VA_REPO_ROOT` can explicitly select a checkout.
Both nodes must have the same installed revision and compatible environments.

## Discovery And Ownership

The Coordinator selects its default-route local address without a network probe.
`R2VA_COORDINATOR_HOST` is an optional override for multi-interface hosts, not a
required IP field. The normal Worker connection establishes actual reachability.
The TCP port defaults to 8780 (`R2VA_COORDINATOR_PORT`).

Only after acquiring `/tmp/<run-id>.coordinator.lock`, binding TCP and restoring
the existing run does Coordinator atomically publish `coordinator_endpoint.json`.
That file contains only the URL, never the bearer credential. Before attaching,
Workers authenticate and confirm that `/v1/status` names the requested run root;
a stale URL serving a different run is rejected. Every subsequent Worker
connect/reconnect also carries that run root and is rejected before joining or
loading resources if the endpoint now serves another run.

Deployment provisions the same credential once on each authorized node at
`/root/.config/r2va/coordinator.token` (directory 0700, file 0600), using the
authenticated SSH channel rather than shared storage. The normal Bash command
needs no credential argument. `R2VA_TOKEN_FILE` can select another node-local
file; it must be outside the configured HTTP media root. A missing/private-mode
error stops startup before any Worker or model is launched. The existing
`R2V_GROUP_COORDINATOR_TOKEN` environment override remains supported.

The credential is reused on resume, never printed or passed on the command
line. Discovery, run logs and shared output contain no credential. Do not put
the token in `/mnt/workspace`: even a 0600 file there can be read by a
same-account media server. New nodes require the same one-time private credential
provisioning; they do not obtain secrets from the public discovery file.
Workers never write discovery, cursor, claims, generation or terminal state.
This launcher assumes trusted nodes and private-network HTTP, not hostile
co-located users or automatic Coordinator election.

There is no automatic Coordinator failover. A Worker with missing/unreachable
discovery waits and reports a bounded startup error; it never promotes itself.
The existing HTTP heartbeat, watchdog, claim fencing and receipt semantics are
unchanged. Stop/restart Coordinator on the original designated node only.

For default local media URL `http://127.0.0.1:8766/`, use an existing HTTP service
or start the same `python -m http.server --directory /mnt/workspace` service used
by the successful Pilot. Only a service started by this invocation is stopped
on exit. Existing media services are not adopted or terminated. Native Workers
close their owned model resources before the local Coordinator stops.
The launcher retains the latest observed child processes while its Worker is
alive, so an abnormal Worker exit can also clean up detached model backends.

## CPU And Dry Run

```bash
R2VA_ROLE=coordinator bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh --dry-run
bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh --dry-run
```

`--dry-run` does not write run state, contact SSH or load models. `--fake` runs
the existing eight CPU Fake stages and does not produce training data. Use a
separate run-id for Fake tests; Native/Fake state is not interchangeable.

Install the entry after updating the correct checkout, never a T2VA checkout:

```bash
install -m 755 /mnt/workspace/litengjie/data/R2V_DATA_V2/scripts/run_h3_ra2va_cluster_dynamic.sh /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

Local regression proves two independent launch processes can discover one run,
both execute tasks, publish 160 unique terminal rows, and resume with zero task
reexecution. This is not evidence of a new real two-node GPU Pilot.

## 2026-10-10 Deployment Preflight

Both `nb-94ayoy238c-0` (Coordinator) and `nb-6numx1uhso-0` updated using
`git pull --ff-only` to `b821367cb41eaad91c4ef59cce7c90e74e73078d` and
installed the Bash entry. `/root` and `/tmp` are node-local `overlay` storage.
The identical SSH-provisioned credentials use directory 0700 and file 0600.

New Native run, with no Workers or models started:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA/ra2va-dualnode-pilot8-r11-18-20261010T072147Z
```

Only source rows 11-18 were selected (`start_row=10`, `limit=8`), with no
clip overlap against `ra2va-native-group0-pilot10-20261010T034522`.
Node B discovered `http://6.167.51.186:8780` without an IP argument:
authenticated status 200, absent/wrong credential 401, wrong-run connect 409.
Discovery contains only `url`; the credential is absent from shared run files.

Coordinator stop/restart on Node A preserved the inventory, settings,
generation 1, canonical stage, eight pending clips and zero active/terminal
results. Resume status was available from Node B in 1.062 seconds. The owned
Coordinator was then stopped with no remaining run processes. Evidence is in
`cpu-preflight.json`, `cpu-preflight-resume.json`, `cpu-preflight-cleanup.json`
and the two Coordinator logs under that run. No GPU Pilot, inference or new
training output was produced: the review found startup issues, so this run
stopped at the requested preflight boundary after fixing them.

The frozen Native run is ready for a separately authorized GPU start/resume.
On Node A:

```bash
R2VA_ROLE=coordinator R2VA_RUN_ID=ra2va-dualnode-pilot8-r11-18-20261010T072147Z R2VA_START_ROW=10 R2VA_LIMIT=8 R2VA_MAX_GROUPS=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

On Node B:

```bash
R2VA_RUN_ID=ra2va-dualnode-pilot8-r11-18-20261010T072147Z R2VA_START_ROW=10 R2VA_LIMIT=8 R2VA_MAX_GROUPS=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

Local verification: 356 related Group/Two-step/Single/T2VA tests passed;
the unchanged T2VA Worker-pool suite passed all 15 tests separately, while its
process-cleanup test was timing-sensitive in combined runs. An older
Multi-call suite produced the same 143 passes and 18 failures on this revision
and baseline `a457702`. Ruff, `py_compile`, Bash syntax and diff checks passed.

## 2026-10-10 Real Pilot16

Run `ra2va-dualnode-pilot16-r19-34-20261010T084641Z` selected source rows 19-34,
without overlap with the earlier 18 clips. Both nodes used revision
`023942cb6cc8f761752ddeca528604bc5c9bdae2`. There were 128 unique terminal
clip-stage results: Node A executed 52 and Node B 76. MiMo processed eight clips
on each node, 32 real requests total. MiMo: 15 Ready / 1 Failed. Effective training
videos: 13/16. Products: 37 Ready / 5 Failed. Training: 148 rows across 12 tasks
(four Visual tasks x 13, four Full Audio tasks x 13, four Speech+BGM tasks x 11).
Donors: 11 speakers / 26 segments. Only `PILOT_COMPLETE` was written. Both nodes
finished with zero GPU memory use and no owned model processes.

Wall time was 847.36 seconds, including startup and barriers:

| Stage | Total seconds | Slowest worker initialization seconds |
| --- | ---: | ---: |
| Canonical | 6.98 | 2.18 |
| SAM | 148.88 | 147.26 |
| AuK | 152.66 | 151.16 |
| Resolve | 4.50 | 2.19 |
| DiariZen | 37.60 | 35.99 |
| ASR | 28.76 | 21.89 |
| MiMo | 457.05 | 190.07 |
| Export | 10.92 | 2.31 |

Each node started its two TP4 servers sequentially; the two nodes loaded in
parallel. Summing the slowest initialization per stage gives 553.04 seconds
(65.27%), but some of that overlaps another worker's useful processing. It is
not pure startup wall time, and the remaining time includes queueing and cleanup,
not only inference. A small Pilot therefore does not establish steady-state
production throughput. Source evidence remains in the original run's node
launch logs, `native_stage_loaded` events and stage `dispatch.json` timestamps.
