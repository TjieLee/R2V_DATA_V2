# V3 JEA production shards

This adapter prepares the growing JEA motion JSONL for the frozen Visual V3
pipeline. It does not change the generic source parser, copy source videos, or
change any model, threshold, prompt, schema, or scheduling behavior.

## Production roots

The source is:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
```

The authoritative path-discovery result is the generated file:

```text
/mnt/workspace/litengjie/data/r2v_v3_configs/production/jea_motion_v1/prod-v1/source.yaml
```

`source.yaml` is created only after at least 100 non-empty source records have
resolved to existing files below the explicitly supplied `clips_root`. It
records the confirmed `clips_root`, `source_videos_root`, source JSONL, base
config, adapter version, and shard size. It is immutable on later resumes.

The development Mac used for this patch does not mount `/mnt/workspace`, so no
repository document should guess whether the confirmed root is the processed
directory itself or `clips_clean_cropped`. On the production server, try the
two explicit candidates below. A failed probe creates no `source.yaml`:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

BASE=/mnt/workspace/litengjie/data/r2v_v3_configs/e2e1000-s0-samfix-20260814-101818.yaml
SOURCE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
SOURCE_VIDEOS=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed

python tools/prepare_v3_production_shards.py \
  --source-jsonl "$SOURCE" \
  --base-config "$BASE" \
  --clips-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed \
  --source-videos-root "$SOURCE_VIDEOS" \
  --path-probe-records 100
```

If and only if that probe reports that fewer than 100 paths resolve, rerun with:

```bash
  --clips-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
```

After the first successful run, inspect the immutable result rather than
inferring the root again:

```bash
sed -n '1,80p' \
  /mnt/workspace/litengjie/data/r2v_v3_configs/production/jea_motion_v1/prod-v1/source.yaml
```

## Shard preparation and resume

The default shard size is 1000 source records. Only complete shards are sealed
by default. Use `--seal-tail` when the current incomplete EOF tail should become
an immutable shard. The cursor records byte offsets and the SHA-256 of the last
committed line. Resume verifies that committed boundary and seeks directly to
`next_byte_offset`; it does not scan from line zero.

Malformed records are written without raw payloads to:

```text
state/source_errors.jsonl
```

Each generated shard YAML inherits all model, runtime, Visual, and attribute
settings from the explicit base config. Only `dataset_json`, `run_root`,
`export_root`, and the fixed-selection source fields are replaced.

To seal the current tail, repeat the confirmed command with:

```bash
  --seal-tail
```

Run each generated YAML independently with the existing V3 pipeline command;
the preparation layer does not modify `run_pipeline_v3` scheduling.

## Compact shard exports

After shard exports are complete:

```bash
python tools/compact_v3_production_exports.py \
  --shards-root /mnt/workspace/litengjie/data/r2v_v3_exports/production/jea_motion_v1/prod-v1/shards \
  --source-jsonl "$SOURCE"
```

This streams shard samples, validates every `dataset.json`, uses a temporary
SQLite uniqueness index, verifies reference files, and publishes
`samples.jsonl` plus `catalog.json` atomically per file. Reference paths are
rewritten to `shards/<shard-id>/references/...`; neither references nor videos
are copied.

To also validate and merge subject-attribute enriched samples, add:

```bash
  --runs-root /mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/prod-v1
```
