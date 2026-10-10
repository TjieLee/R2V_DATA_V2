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
