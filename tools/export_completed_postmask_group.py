#!/usr/bin/env python3
"""Export ONE completed Post-Mask group as a final-schema JSONL, without media copies.

The JSON records use r2v.v3.production_sample.1 and the same R2V/SA
conversion semantics as the frozen Post-Mask compactor. Reference image paths
are relative to the published group directory. Its ``references`` subtree
contains only a small number of directory symlinks to the already published
shard Export images (both Visual and Subject Attributes). The corresponding
exported ``enriched_samples.jsonl`` is the source for SA enrichment; no private
SA aggregate is required.

Usage (run from any directory on the shared production server):
  /mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod/.venv/bin/python \
      export_completed_postmask_group.py --group group-000000

No source artifacts, production state, models or worktree contents are changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_DEFAULT = Path('/mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod')
ROOT_DEFAULT = Path('/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference')
RUNS_DEFAULT = Path('/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/in_pair_reference')
OUTPUT_BASE_DEFAULT = ROOT_DEFAULT / 'post_mask'
GROUP_NAME = re.compile(r'group-[0-9]{6}')
SHARD_NAME = re.compile(r'shard-[0-9]{9}-[0-9]{9}')


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'not a JSON object: {path}')
    return payload


def _relative_path(path: str) -> str:
    """Reject escaping image names without stat'ing every image on JPFS."""
    if not isinstance(path, str) or not path:
        raise ValueError(f'invalid image path: {path!r}')
    raw = Path(path)
    if raw.is_absolute() or '..' in raw.parts:
        raise ValueError(f'image path escapes its reference root: {path!r}')
    return raw.as_posix()


def _visual_path(shard: str, image_path: str) -> str:
    relative = _relative_path(image_path)
    prefix = 'references/'
    if not relative.startswith(prefix):
        raise ValueError(f'unexpected shard reference path: {image_path!r}')
    return f'references/visual/{shard}/{relative[len(prefix):]}'


def _load_enriched(shard_root: Path, run_root: Path, models: object) -> dict[str, object]:
    """Load the already exported SA sidecar (including exported image paths)."""
    path = shard_root / 'enriched_samples.jsonl'
    if not path.is_file():
        return {}
    enriched = {}
    with path.open('r', encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = models.EnrichedSample.model_validate_json(line)
            if Path(value.source_run_root).resolve(strict=False) != run_root.resolve(strict=False):
                raise ValueError(f'{path}:{line_no}: source_run_root mismatch')
            if value.sample_id in enriched:
                raise ValueError(f'{path}:{line_no}: repeated sample_id: {value.sample_id}')
            enriched[value.sample_id] = value
    return enriched


def _convert_enriched(sample: object, enriched: object, shard: str, models: object) -> object:
    """Mirror the compactor's semantic mapping; change image placement only."""
    if enriched.clip_uid != sample.sample_id:
        raise ValueError(f'enriched clip_uid mismatch: {enriched.sample_id}')
    old_video = enriched.original_visual.get('target_video')
    if old_video is not None and old_video != sample.target_video:
        raise ValueError(f'enriched target_video mismatch: {enriched.sample_id}')
    old_source = enriched.original_visual.get('source')
    if old_source is not None and old_source != sample.source.model_dump():
        raise ValueError(f'enriched source mismatch: {enriched.sample_id}')

    visual_entities = {}
    background = None
    for ref in sample.references:
        if ref.type == 'background':
            if background is not None:
                raise ValueError('duplicate background reference')
            background = ref
        else:
            if ref.entity_id in visual_entities:
                raise ValueError('duplicate Visual entity reference')
            visual_entities[ref.entity_id] = ref
    owner_ids = set(visual_entities)
    attributes = {v.attribute_id: v for v in enriched.accepted_attributes}
    if len(attributes) != len(enriched.accepted_attributes):
        raise ValueError(f'duplicate SA attribute: {enriched.sample_id}')

    references = []
    for ref in enriched.references:
        index = ref.image_index
        if ref.kind == 'attribute':
            record = attributes.pop(ref.attribute_id, None)
            if record is None:
                raise ValueError(f'missing accepted attribute: {ref.attribute_id}')
            if (
                record.owner_entity_id != ref.owner_entity_id
                or record.owner_entity_id not in owner_ids
                or record.image_path != ref.image_path
                or record.source_frame_index != ref.source_frame_index
            ):
                raise ValueError(f'attribute provenance mismatch: {record.attribute_id}')
            if ref.source_frame_index is None:
                raise ValueError('attribute source_frame_index missing')
            references.append(models.ProductionReference(
                image_id=f'image_{index}',
                image_index=index,
                kind='attribute',
                attribute_id=record.attribute_id,
                owner_entity_id=record.owner_entity_id,
                attribute_type=record.attribute_type,
                image_path=_visual_path(shard, ref.image_path),
                source_frame_index=ref.source_frame_index,
                synthetic=(
                    record.default_variant == 'generated_background'
                    or (
                        record.default_variant in {None, 'accepted_base'}
                        and record.final_selection == 'completed'
                    )
                ),
            ))
            continue

        if ref.kind == 'background':
            original = background
            background = None
        else:
            original = visual_entities.pop(ref.entity_id, None)
        if original is None:
            raise ValueError(f'unmatched Visual reference: {ref.image_id}')
        if models.dataset_reference_kind(original) != ref.kind:
            raise ValueError(f'Visual kind mismatch: {ref.image_id}')
        if original.source_frame_index != ref.source_frame_index:
            raise ValueError(f'Visual source frame mismatch: {ref.image_id}')
        references.append(models.production_visual_reference(
            reference=original,
            image_index=index,
            kind=ref.kind,
            image_path=_visual_path(shard, original.image_path),
        ))

    if visual_entities or background is not None or attributes:
        raise ValueError(f'enriched references incomplete: {enriched.sample_id}')
    return models.ProductionSample(
        sample_id=sample.sample_id,
        clip_uid=enriched.clip_uid,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=enriched.enriched_instruction,
        references=references,
        source=models.ProductionSampleSource(
            parent_video_id=sample.source.parent_video_id,
            clip_suffix=sample.source.clip_suffix,
            shard_id=shard,
        ),
    )


def _convert_visual(sample: object, shard: str, models: object) -> object:
    paths = [_visual_path(shard, ref.image_path) for ref in sample.references]
    return models.visual_only_sample(sample, shard_id=shard, reference_paths=paths)


def _models():
    from types import SimpleNamespace
    from r2v_data_v2.v3.schemas import DatasetSample
    from r2v_data_v2.v3.subject_attributes import EnrichedSample
    from r2v_data_v2.v3.production_export import (
        ProductionReference, ProductionSample, ProductionSampleSource,
    )
    from tools.compact_v3_production_exports import (
        _dataset_reference_kind, _production_visual_reference, _visual_only_sample,
    )
    return SimpleNamespace(
        DatasetSample=DatasetSample,
        EnrichedSample=EnrichedSample,
        ProductionReference=ProductionReference,
        ProductionSample=ProductionSample,
        ProductionSampleSource=ProductionSampleSource,
        dataset_reference_kind=_dataset_reference_kind,
        production_visual_reference=_production_visual_reference,
        visual_only_sample=_visual_only_sample,
    )


def _link_shard_references(staging: Path, root: Path, shard: str) -> None:
    visual = root / 'shards' / shard / 'references'
    if not visual.is_dir():
        raise FileNotFoundError(f'missing published shard references: {visual}')
    (staging / 'references' / 'visual' / shard).symlink_to(visual, target_is_directory=True)


def convert_group(group: str, root: Path, runs: Path, output_dir: Path, models: object) -> int:
    if not GROUP_NAME.fullmatch(group):
        raise ValueError(f'invalid group: {group}')
    group_marker = root / 'state' / 'resource_epochs' / group / 'completed.json'
    record = _json(group_marker)
    if record.get('status') != 'completed' or record.get('export_completed') is not True or record.get('group_id') != group:
        raise ValueError(f'group has not completed its formal Export: {group_marker}')
    shards = record.get('canonical_shards')
    if not isinstance(shards, list) or not shards or len(set(shards)) != len(shards):
        raise ValueError('group marker has invalid canonical_shards')
    if any(not isinstance(s, str) or not SHARD_NAME.fullmatch(s) for s in shards):
        raise ValueError('group marker has invalid shard name')
    if not isinstance(record.get('sample_count'), int) or record['sample_count'] < 0:
        raise ValueError('group marker has invalid sample_count')
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'output already exists; will not overwrite: {output_dir}')

    expected_by_shard = {}
    for shard in shards:
        marker = _json(root / 'state' / 'shards' / shard / 'completed.json')
        if (
            marker.get('status') != 'completed'
            or marker.get('export_completed') is not True
            or marker.get('shard_id') != shard
            or not isinstance(marker.get('sample_count'), int)
            or marker['sample_count'] < 0
        ):
            raise ValueError(f'shard Export marker is invalid: {shard}')
        expected_by_shard[shard] = marker['sample_count']
    if sum(expected_by_shard.values()) != record['sample_count']:
        raise ValueError('group and shard sample counts disagree')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.tmp-', dir=output_dir.parent))
    total = 0
    enriched_count = 0
    seen_sample_ids: set[str] = set()
    try:
        (temp_dir / 'references' / 'visual').mkdir(parents=True)
        with (temp_dir / 'samples.jsonl').open('w', encoding='utf-8') as dest:
            for shard in shards:
                _link_shard_references(temp_dir, root, shard)
                shard_root = root / 'shards' / shard
                run_root = runs / shard
                source = shard_root / 'samples.jsonl'
                enriched = _load_enriched(shard_root, run_root, models)
                shard_count = 0
                with source.open('r', encoding='utf-8') as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        sample = models.DatasetSample.model_validate_json(line)
                        if sample.sample_id in seen_sample_ids:
                            raise ValueError(f'duplicate sample_id across group: {sample.sample_id}')
                        seen_sample_ids.add(sample.sample_id)
                        extra = enriched.pop(sample.sample_id, None)
                        final = (_convert_enriched(sample, extra, shard, models)
                                 if extra is not None else _convert_visual(sample, shard, models))
                        dest.write(json.dumps(final.model_dump(mode='json'), ensure_ascii=False, separators=(',', ':')) + '\n')
                        shard_count += 1
                        enriched_count += int(extra is not None)
                if enriched:
                    raise ValueError(f'orphan enriched sample_id in {shard}: {next(iter(enriched))}')
                if shard_count != expected_by_shard[shard]:
                    raise ValueError(f'{shard}: expected {expected_by_shard[shard]} samples, got {shard_count}')
                total += shard_count
                print(f'{shard}: {shard_count} final-schema samples', flush=True)
            dest.flush()
            os.fsync(dest.fileno())
        if total != record['sample_count']:
            raise ValueError(f'group count mismatch: {total} != {record["sample_count"]}')
        # Atomically expose both the JSONL and its directory aliases.
        temp_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(f'DONE: {group} samples={total}, enriched={enriched_count}')
    print(f'JSONL: {output_dir / "samples.jsonl"}')
    print(f'IMAGE_ROOT: {output_dir}')
    print('Note: images are symlinked, not copied; keep source shard Export images available.')
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', default='group-000000')
    parser.add_argument('--repo', type=Path, default=REPO_DEFAULT)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--runs', type=Path, default=RUNS_DEFAULT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_BASE_DEFAULT / args.group
    sys.path.insert(0, str(args.repo))
    print(f'REPO: {args.repo}')
    print(f'OUTPUT: {output}')
    convert_group(args.group, args.root, args.runs, output, _models())


if __name__ == '__main__':
    main()