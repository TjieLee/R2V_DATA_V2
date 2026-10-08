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


## One completed group to final-schema JSONL (2026-10-08)

Formal Global Compaction publishes the official `in_pair_reference/samples.jsonl`
and `catalog.json` only after **all 48 groups** have completed. This barrier
is **not** required for consuming an already completed group. A shard's SA
terminal count reaching 100% does not imply Export is complete: require
`state/resource_epochs/<group>/completed.json` with `export_completed=true`
and the corresponding eight `state/shards/<shard>/completed.json` markers.

**Standalone opt-in converter:** `tools/export_completed_postmask_group.py`.
It reads just the specified completed group's shard `samples.jsonl` files
and **the same shard's already exported `enriched_samples.jsonl`**, when
present, then emits one JSONL of `r2v.v3.production_sample.1` records.
The completed shard export already rewrites Visual and Attribute image paths
and publishes the accepted final Attribute PNGs under its own `references/`.
No private SA `enriched_samples.jsonl` is required; this matters because
DatasetExporter can also source its enrichment from private
`subject_attributes/samples/*.json` when the aggregate is missing.
No model calls, scheduler change, production-state changes, or global compaction.
No SHA/hash/checksum, media decoding, whole-tree scans, or source image copies.

From a checkout that already contains the helper:

```bash
WT=/mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod
"$WT/.venv/bin/python" tools/export_completed_postmask_group.py \
  --repo "$WT" --group group-000000
```

**Never switch/update the shared production worktree while jobs run.** If the
helper exists only on its GitHub topic branch, fetch that branch in the
ordinary source checkout and extract the tool into a private file without
changing the production checkout:

```bash
SRC=/mnt/workspace/litengjie/data/R2V_DATA_V2
WT=/mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod
HELPER=/mnt/workspace/litengjie/data/export_completed_postmask_group.py

git -C "$SRC" fetch origin feature/postmask-completed-group-export-20261008
git -C "$SRC" show FETCH_HEAD:tools/export_completed_postmask_group.py > "$HELPER"
"$WT/.venv/bin/python" "$HELPER" --repo "$WT" --group group-000000
```

The default output is a **private** directory:

```text
/mnt/workspace/litengjie/data/r2v_v3_group_exports/post_mask/group-000000/
  samples.jsonl
  references/
    visual/<shard>/...          # links to official shard reference directories,
                                 # including accepted SA attribute PNGs
```

The output uses the final **schema and semantics**: `sample_id`,
`clip_uid`, `target_video`, `t2v_caption`, `r2v_instruction`,
`source.shard_id`, `image_id`, `image_index`, `kind`, and final
Subject Attributes references. A visually-only record preserves the original
instruction. For enriched records the instruction becomes
`enriched_instruction` and accepted attribute images are appended in the
documented order. The original `target_video` and `t2v_caption` are retained.

**Paths are not byte-identical to global Compaction.** In the standalone
group JSONL, each `references[*].image_path` is relative to the
`group-000000/` output root; the eventual global Compaction rewrites the
references into its own canonical placement. The group converter uses
**directory symlinks**, so consumers must resolve each image path against the
group output root and keep official shard **exported** reference images
(including final SA PNGs) accessible. Do not GC those inputs while consumers depend
on these symlinks. The helper does not reference private SA PNGs.
This one-shot command does not publish a global
`catalog.json`, a portable image bundle, or a flattened global
`enriched_samples.jsonl`.

The command publishes the group output directory by **atomic rename** only
after the JSONL and symlinks have been prepared. It refuses to overwrite an
existing result, checks formal completion markers and per-shard/group sample
counts, and fails on duplicate/orphan/mismatched enriched records rather than
silently dropping attributes. It only reads the named group's manifest and
enriched files, not the rest of the campaign; it is manual, not a streaming
publisher.

**2026-10-08 server execution result (operator-reported):** The corrected
standalone converter completed on the production server with the pinned
production repo imported via `--repo "$WT"`, without switching the active
production worktree. It printed:

```text
shard-000000000-000009999: 4846 final-schema samples
shard-000010000-000019999: 5732 final-schema samples
shard-000020000-000029999: 5931 final-schema samples
shard-000030000-000039999: 5788 final-schema samples
shard-000040000-000049999: 3599 final-schema samples
shard-000050000-000059999: 4417 final-schema samples
shard-000060000-000069999: 4419 final-schema samples
shard-000070000-000079999: 4853 final-schema samples
DONE: group-000000 samples=39585, enriched=27837
```

Output: `/mnt/workspace/litengjie/data/r2v_v3_group_exports/post_mask/group-000000/samples.jsonl`.
All eight per-shard counts add up to the completed group's `sample_count=39585`.
Of those, `27837` had a published enriched sidecar record; `11748` used
the original visual-only instruction. The 48,128 historical SA terminal/eligible
count is a different denominator (not the accepted Export sample count).

This confirms that the patched helper completed and atomically published a
full real-world Group 0 conversion. It does **not** independently establish
that every symlink target can be opened, verify every JSON record, or prove
parity with the not-yet-run global compactor. Preserve shard Export images
and downstream-relative image path resolution. The unchanged formal global
Compaction will still run after all groups complete.

**Important: independent Global Compaction gap discovered 2026-10-08.**
The current full compactor's `_load_shard_enriched_samples` reads only the
private `<runs>/<shard>/subject_attributes/enriched_samples.jsonl`.
If that private aggregate is absent but the shard's already published
`enriched_samples.jsonl` exists, the current full compactor can silently
produce a visual-only record for that shard. This does not change or invalidate
the finished shard Export; it is a separate final-Compaction input-selection
bug, **not fixed by the standalone group helper**. Before the 48-group global
compaction, investigate and fix/test that loader without changing the active
production worktree. Do not treat a visual-only global result as acceptable
merely because the private aggregate is missing. The group helper always
prefers the shard's finished exported sidecar and avoids this dependency.
