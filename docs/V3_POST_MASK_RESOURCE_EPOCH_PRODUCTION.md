# Post-Mask Resource Epoch Production

On each 8-GPU node, from the repository checkout:

```bash
bash scripts/run_v3_post_mask_full_production.sh
```

For a path/config review without loading models or requiring server dependencies:

```bash
bash scripts/run_v3_post_mask_full_production.sh --dry-run
```

The launcher accepts externally assigned `RANK` and `WORLD_SIZE` (default
`0/1`). They rotate scan order only. Every node sees every canonical group and
claims unfinished work via a nonblocking shared `flock`. A node that encounters
busy groups waits and scans again. A crash releases the lock; a later launch can
resume with a different node count. Canonical per-shard locks remain in force.

The frozen Stage2 input is
`/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask`.
The direct DatasetExporter destination and aggregate production output are under
`/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference`.
Stable group/shard state and locks live in `in_pair_reference/state`; private
intermediate RunStorage and model scratch remain under
`/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/in_pair_reference`.
No timestamped output root or copy-back step is used.

Each shard gets a small atomic `completed.json` only after the existing
Subject Attributes terminal barrier and DatasetExporter publication. The group
marker follows only after every shard export succeeds. Restart skips completed
shards/groups by reading those markers, without rehashing export trees, receipts,
or media. Missing markers enter the existing durable ModelJob receipt replay.
Once all groups finish, one node holds the compaction lock and uses the existing
V3 compactor to publish `samples.jsonl` and `catalog.json` at the official root.
The compactor's normal publication checks are not used as a completion/resume
gate. A plain compaction marker makes later launches skip that work as well.

The committed config pins Boogu removal and all Qwen services to
`Qwen3-VL-8B-Instruct`; the launcher pins the managed physical Qwen model after
loading `server_env.sh`. Existing Resource Epoch resource sessions own 8 Boogu,
8 SAM and TP1/DP8 Qwen GPU slots in sequence, not concurrently. No Audio/H3
stage is launched here.

Server-local checkpoint availability, shared-filesystem `flock`, write access
to the public output root, and real model quality still require an on-server
small smoke before full production. This code change runs no real models.
