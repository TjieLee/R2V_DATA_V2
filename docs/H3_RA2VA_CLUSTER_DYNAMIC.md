# RA2VA Per-Node One-Command Launch

Each node runs the same Bash command. The cluster's global `RANK=0` selects the
Coordinator node; other ranks discover its URL through the shared run directory
and initiate HTTP connections. RANK selects roles only, never clips or shards.
No Worker IP list, SSH fan-out, rank sharding, or automatic election.
The existing central SSH launcher remains available and unchanged.

## Launch

Submit this exact command on every node, with no environment prefix:

```bash
bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

The script defaults to full-group production and `R2VA_ROLE=auto`. The platform
must supply a global, nonnegative `RANK`, with exactly one rank 0. If `WORLD_SIZE`
is supplied, rank must be smaller than it. Missing/invalid rank stops before run
state, Workers or models are created; it never defaults every node to rank 0.
Do not use per-GPU `LOCAL_RANK` in place of the global node rank. Explicit
`R2VA_ROLE=coordinator|worker` remains available for existing manual deployments.

Default run: `ra2va-group-production-v1`. Restart the same command with the same
rank-0 host to resume; there is no automatic Coordinator failover. Use an
identical `R2VA_RUN_ID` on every node for a new run. Workers may start first and wait
up to 300 seconds (`R2VA_WAIT_SECONDS`) for Coordinator readiness.
Worker-only commands exit after completion. The Coordinator command stays in the
foreground until Ctrl+C, including after local Workers finish, so delayed or
reconnecting nodes can still obtain the terminal state. SIGINT, SIGTERM and
SIGHUP use the same owned-process cleanup.

## Production Scope And Resources

Full-group scope is set inside the entry script. No partial source selection or
Pilot completion marker is reused. No `R2VA_FULL_GROUP=1` or role prefix is needed
in the submitted command.

Default production run-id: `ra2va-group-production-v1`. Use the same run-id and
scope on restart. Full scope freezes all rows of each published group once,
in source order, without reading media or computing hashes. `R2VA_MAX_GROUPS=0`
(the full-scope default) processes newly published groups continuously;
`R2VA_MAX_GROUPS=1` stops after one complete group. This budget remains cumulative
across restarts. All nodes still share one current group and stage; there is no
rank/shard assignment. Only all completed production parts write the group-level
`COMPLETE`; bounded runs continue to write only `PILOT_COMPLETE`. `--dry-run`
remains available on the same command.

### Early Training Delivery

New full-production runs default to four ordered parts per source group. All
nodes work on the same current part through canonical, SAM, AuK, resolve,
DiariZen, ASR, MiMo and export. The next part starts only after training JSONL
and donor reserve publication and the existing Worker release barrier. Models
still stay loaded across clips within each stage, not across part boundaries.
This trades four sets of stage/model startups for earlier training data.

The full inventory is frozen once. Part `i` uses zero-based original ordinals
`floor(N*i/4)` through `floor(N*(i+1)/4)-1`, without shuffling, renumbering clips,
or re-reading source media. Empty parts in groups smaller than four clips are
omitted. Source groups containing 35,000-40,000 clips yield about 8,750-10,000
clips per part; duration-skipped and failed clips retain their terminal status.

```text
R2VA/<run-id>/groups/group-000000/
  inventory/                     # one full frozen source inventory
  stages/                        # unique generation for every part/stage
  parts/part-000/
    bundle/training/             # existing 12 task JSONL files + index
    bundle/voice_donor_reserve.json
    bundle/source_contract.json
    bundle/records.jsonl
    bundle/summary.json
    COMPLETE                     # this part, not the whole source group
  parts/part-001/                 # independent publication; no rewrite of part-000
  parts/part-002/
  parts/part-003/
  COMPLETE                       # only after the last part completes
```

Train from each published part's `bundle/training/` as soon as that part's
`COMPLETE` exists. Parts contain disjoint clips; combine their task files later
without deduplication or re-materialization. The donor index references the
existing per-clip audio assets and includes every eligible speech segment.
There is no extra group-wide training merge or model request.

Resume uses the original run-id, part boundaries, generation and terminal
receipts. `max_groups` still counts source groups, not parts, so a budget of one
finishes all four parts but never starts the next group. Status includes
`part_index` (zero-based), `part_count`, ordinal bounds and `completed_parts`.
Logs emit `part_complete` with the delivered bundle path.

Historical unsplit full runs and bounded Pilots remain unsplit on restart:
the entry reads the saved part setting instead of silently migrating them.
`R2VA_GROUP_PARTS=1` is available for legacy deployments; new normal submissions
need no prefix or additional argument. An explicit changed setting for an
existing run is rejected. Multi/Single, prompts, ASR, Speaker rules and training
row formats are unchanged. CPU tests cover part boundaries, lost control writes,
claim fencing, release barriers and actual per-part audio/H3/JSONL exports;
this is not a new GPU throughput measurement.

CPU regression on 2026-10-10: 12 focused part tests passed. Two local HTTP Fake
Workers completed 20 clips across 32 generations with 160 unique terminal
clip-stage results. The real CPU materialization fixture completed 8 synthetic
clips in four parts: 24 ready products, 0 failed products, 96 four-field task
rows (8 per task), 8 donors and 16 segments. The first part was published before
the second started, and its files stayed byte-for-byte unchanged afterward.
Real model calls: 0; simulated Visual/Joint responses: 16. These local tests are
not a new real two-node or GPU Pilot.

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

Manual bounded tests must explicitly set `R2VA_FULL_GROUP=0`; the old 20-row
default and cumulative `max_groups=1` then remain available under
`ra2va-group-pilot20-v1`. This override is for tests, not the cluster submission.
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

Fresh cluster containers no longer need `/root/.config/r2va/coordinator.token`.
By default, the designated Coordinator creates one random per-run credential at
`<run-root>/.ra2va-private/coordinator.token`, only after acquiring its local lock
and binding TCP. The directory is 0700, the atomically published file is 0600,
and resume reuses it without rotation. Workers wait for that file read-only;
they never initialize it or promote themselves to Coordinator. The normal
Bash command needs no credential argument or environment injection.

The launcher starts a loopback-only media server that rejects `.ra2va-private`
paths, including URL-encoded paths and resolved symlinks, and disables directory
listings. It checks this policy before shared-credential initialization. An
existing unrestricted media server is not reused or stopped: startup explains
that it must be stopped by its owner or a free media port configured. Do not
independently serve the shared tree with an unrestricted file server. Unix
permissions alone do not protect credentials from a same-account HTTP server.

`R2VA_TOKEN_FILE` still supports an explicitly provisioned private node-local
file outside the HTTP media root. `R2V_GROUP_COORDINATOR_TOKEN` remains supported
for platforms that can inject the same secret on all nodes. Explicit overrides
take precedence and do not create a shared token. Credentials are never printed,
passed on the command line or copied into discovery, logs, bundles or JSONL.
The private per-run file is the only shared credential location.
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
bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh --dry-run
```

`--dry-run` reads the platform's RANK but does not write run state, contact SSH or
load models. Outside a cluster job, supply RANK explicitly for a manual dry run
or use the existing explicit role override. `--fake` runs
the existing eight CPU Fake stages and does not produce training data. Set
`R2VA_FULL_GROUP=0` and a separate run-id for bounded Fake tests;
Native/Fake state is not interchangeable.

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
R2VA_FULL_GROUP=0 R2VA_ROLE=coordinator R2VA_RUN_ID=ra2va-dualnode-pilot8-r11-18-20261010T072147Z R2VA_START_ROW=10 R2VA_LIMIT=8 R2VA_MAX_GROUPS=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
```

On Node B:

```bash
R2VA_FULL_GROUP=0 R2VA_ROLE=worker R2VA_RUN_ID=ra2va-dualnode-pilot8-r11-18-20261010T072147Z R2VA_START_ROW=10 R2VA_LIMIT=8 R2VA_MAX_GROUPS=1 bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_ra2va_cluster_dynamic.sh
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
