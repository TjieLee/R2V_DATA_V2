# V3 JEA production shards

This production adapter prepares the growing JEA motion JSONL for the frozen
Visual V3 pipeline. It preserves the generic JSON/JSONL parser and
`fixed_selection_v1` behavior. It never copies source videos or changes model,
prompt, threshold, existing Visual schema, or pipeline scheduling behavior.

## Frozen production identity

The production inputs are:

```text
source JSONL:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl

clips_root:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped

source_videos_root:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808
```

`record.video_path` is the processed shot used as the V3 Visual frame/model
input. It must be an existing MP4 below `clips_root`. In contrast,
`record.source_video_path` is original/full-video provenance only. It must be a
safe existing regular file below `source_videos_root`, but its container and
extension are unrestricted. A provenance path such as
`01/丁宝桢/01 4K.mkv` is valid and must never be substituted for `video_path` as
the Visual input.

The source dataset may grow indefinitely. A `prod-v1` state root nevertheless
has one immutable identity recorded in:

```text
/mnt/workspace/litengjie/data/r2v_v3_configs/production/jea_motion_v1/prod-v1/source.yaml
```

That file freezes the source JSONL, adapter version, both roots, shard size,
base-config path, exact base YAML SHA-256, and loaded V3 config fingerprint.
Every resume recomputes and checks the base-config identities and the other
immutable fields. A changed value requires a new production version/root; do
not mutate `prod-v1`.

## Quick functional canary

For an isolated 20-clip functional check, run:

```bash
.venv/bin/python tools/run_v3_canary.py \
  --count 20 \
  --exclude-source-name 丁宝桢
```

The helper scans in source order for the first 20 consecutive valid shots from
one non-excluded source video, prints the selected video and true global source
range before model execution, runs the full profiled V3 pipeline, and compacts
the successful result. It automatically creates a timestamped ASCII-safe
`canary-e2e20-jea-...` config, run, and export root. The final export contains
`canary_summary.json`, `samples.jsonl`, `catalog.json`, and the readable
`references/` tree; the pipeline stream is also saved as `canary.log`.

This helper never reads or advances the formal `prod-v1` cursor. Formal
production continues to use `tools/prepare_v3_production_shards.py`; the canary
and formal preparation tools must not share mutable cursor state.

## Formal production shard workflow

Start with a non-mutating path probe. It verifies at least 100 existing clip
MP4s and a bounded sample of unique, safe existing full-source provenance
files, independent of their filename extensions. It prints one JSON result and
creates no `source.yaml`, cursor, selection, shard config, or other production
state.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

BASE=/mnt/workspace/litengjie/data/r2v_v3_configs/e2e1000-s0-samfix-20260814-101818.yaml
SOURCE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
CLIPS=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
SOURCE_VIDEOS=/mnt/workspace/public/dataset/jea-video/moive-183t-0808
STATE=/mnt/workspace/litengjie/data/r2v_v3_configs/production/jea_motion_v1/prod-v1

python tools/prepare_v3_production_shards.py \
  --source-jsonl "$SOURCE" \
  --base-config "$BASE" \
  --clips-root "$CLIPS" \
  --source-videos-root "$SOURCE_VIDEOS" \
  --state-root "$STATE" \
  --path-probe-records 100 \
  --probe-only
```

Inspect the JSON probe result. Then generate exactly one immutable shard:

```bash
python tools/prepare_v3_production_shards.py \
  --source-jsonl "$SOURCE" \
  --base-config "$BASE" \
  --clips-root "$CLIPS" \
  --source-videos-root "$SOURCE_VIDEOS" \
  --state-root "$STATE" \
  --path-probe-records 100 \
  --max-shards 1

sed -n '1,120p' "$STATE/source.yaml"
find "$STATE/shards" -maxdepth 1 -type f -name '*.yaml' -print | sort
```

Run that one generated shard with the ordinary V3 command and no overwrite:

```bash
SHARD_CONFIG="$STATE/shards/shard-000000000-000000999.yaml"
.venv/bin/python run_pipeline_v3.py \
  --config "$SHARD_CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export
```

Compact only the completed shard export and inspect the canonical interface:

```bash
EXPORT=/mnt/workspace/litengjie/data/r2v_v3_exports/production/jea_motion_v1/prod-v1
RUNS=/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/prod-v1

.venv/bin/python tools/compact_v3_production_exports.py \
  --shards-root "$EXPORT/shards" \
  --source-jsonl "$SOURCE" \
  --source-yaml "$STATE/source.yaml" \
  --runs-root "$RUNS"

sed -n '1,3p' "$EXPORT/samples.jsonl"
sed -n '1,160p' "$EXPORT/catalog.json"
```

Only after the probe, one-shard run, compaction, and canonical sample inspection
succeed should normal production resume without `--max-shards`.

## 2026-08-20 new-data functional canary

The server canary at commit
`06051245ea4f58ca8b1df5aa117fab918f211533` covered source indexes 0-9:

```text
input clips:                  10
Visual exports:               3
Visual references:            4
eligible human owners:        2
accepted attribute refs:      2
enriched samples:             2
canonical samples:            3
canonical total references:   6
failed_tasks =                []
copied videos:                0
```

The Qwen TP1 x DP4 BF16 service used `max-model-len=49152`; this eliminated the
previous 32768 context-length infrastructure failure. This bounded canary is
functional evidence only. Its 3 exports from 10 inputs are not a production
yield estimate.

## Validated Boogu Background Removal

Boogu background removal is the validated intended production backend. Canary
tooling explicitly overrides stale base configs with:

```yaml
remove:
  enabled: true
  backend: boogu_image_0_1_edit_turbo
  inference_profile: boogu_4step_v1
runtime:
  gpu_workers:
    remove: "4"
```

The worker uses the pinned Boogu runtime/model and physical GPU4. Qwen remains
the semantic judge where configured; Qwen-Image-Edit is not the active
background image generator. The separate Boogu reference-edit worker remains
on physical GPU6. This does not change `prod-v1` shard/cursor semantics.

## Shards and cursor semantics

The default shard size is 1000 source records. Complete shards are sealed by
default; `--seal-tail` explicitly seals the current incomplete EOF tail.
`--max-shards N` limits only the number of new shards sealed by one invocation
and leaves the cursor exactly after the last sealed shard without consuming the
following records.

Selections and shard configs are immutable. The cursor stores the exact next
unread byte offset and the SHA-256 plus start offset of the last committed
non-empty source line. Resume validates that boundary line, tolerates already
consumed blank lines after it, rejects truncation before the cursor, and seeks
directly to the cursor. This is boundary-line and truncation protection; it is
not a cryptographic verification of every historical byte in the entire source
prefix.

Malformed source rows are isolated without raw payloads in:

```text
state/source_errors.jsonl
```

After `source.yaml` exists and its identity is validated, normal row parsing
keeps `source_video_path` as provenance without a per-row source-video `stat`.
Each selected record retains both unambiguous relative paths:

- `source_relative_video_path`: `video_path` relative to `clips_root`;
- `source_relative_source_video_path`: `source_video_path` relative to
  `source_videos_root`.

## Canonical production export

Per-shard V3 `DatasetSample` exports remain unchanged and are internal/audit
artifacts. The production-facing downstream interface is one file with one
schema on every line:

```text
/mnt/workspace/litengjie/data/r2v_v3_exports/production/jea_motion_v1/prod-v1/samples.jsonl
```

The compactor streams one bounded shard at a time and uses SQLite for global
sample and enriched-sample identity checks. A sample with a valid enriched
sidecar uses the enriched instruction and ordered visual-plus-attribute
references. A sample without attributes is normalized into the same
`r2v.v3.production_sample.1` schema and remains valid.

Every canonical reference for one processed shot is published into one directly
browsable directory. The directory is the shot's `video_path` relative to the
immutable `clips_root` in `source.yaml`, with only the final `.mp4` suffix
removed. Unicode and spaces are preserved. For example:

```text
video:
  01/丁宝桢/01 4K/v_b597a0641cf76c22e880_00041.mp4

references:
  references/01/丁宝桢/01 4K/v_b597a0641cf76c22e880_00041/
    subject_e1.png
    object_e2.png
    group_e3.png
    background.png
    attribute_e1_a1_hair.png
```

The canonical directory and filenames never contain the opaque `clip_uid`.
`clip_uid` remains the stable machine identity in `ProductionSample`, and
shard/internal run directories may continue to use it. The canonical filesystem
intentionally uses human-readable video identity so reviewers never need to
browse UID directories. Visual names are `subject_<entity_id>.png`,
`object_<entity_id>.png`,
`group_<entity_id>.png`, and `background.png`. Visual PNGs are hard-linked from
their immutable shard exports, so the reviewable tree does not duplicate their
bytes or modify the shard artifacts.

Canonical attribute PNGs keep the filename
`attribute_<owner_entity_id>_<attribute_id>_<attribute_type>.png` in that same
directory. Raw or legacy-`None` selections must be RGBA and publish with
`synthetic=false`. Completed selections must be RGB, require an accepted
`completion_review`, and publish with `synthetic=true`.

The compactor validates image mode from `final_selection`, does not convert the
image, hardlinks when possible with the existing safe copy fallback, and
validates the destination again. Publication therefore preserves the original
bytes. An existing destination is reused only when its SHA-256 matches; any
path collision with different bytes fails closed. These references never
depend on a working `run_root` after publication. Target videos remain the
processed public-dataset shot MP4 paths and are never copied. Original/full
`source_video_path` values remain provenance and are never substituted as
Visual inputs.

The optional top-level `enriched_samples.jsonl` remains an audit artifact when
`--runs-root` is supplied. Per-shard sample JSONLs and enriched sidecars are not
the downstream join interface.

### Final Subject Attribute Production State

The GME experiment is not part of the production Subject Attribute path and
remains disabled. The final discovery, frame-local SAM3, deterministic
geometry, raw review, optional Boogu completion, and direct publication flow
is authoritative in `V3_SUBJECT_ATTRIBUTES_STATE.md`.

### Validated Reference-Image Evidence

The fixed random-200 E2E exercised the final direct completion/publication
design and exposed duplicate discovery plus atomic-state issues. The fresh
targeted-10 regression after `51fef9d...` cleared both issues: 10/10
`clip.json` files, zero orphan/temp leftovers, zero runtime failures, and zero
duplicate-discovery failures. See `V3_SUBJECT_ATTRIBUTES_STATE.md` for exact
metrics and the explicit non-yield interpretation.

`samples.jsonl` is fully validated, fsynced, and atomically replaced. The
catalog records its exact SHA-256, canonical schema version, counts, source
ranges, shard commits/config hashes, adapter, and base-config identity. The
catalog is replaced last and acts as the publication commit marker.
