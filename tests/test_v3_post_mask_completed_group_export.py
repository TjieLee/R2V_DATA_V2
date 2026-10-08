"""Isolated group exporter regression tests, without models or JPFS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import export_completed_postmask_group as exporter


class Record:
    """Minimal in-memory model adapter; actual schema is loaded by the CLI."""

    def __init__(self, **fields):
        self.__dict__.update({
            k: (v if k == "original_visual" else wrap(v))
            for k, v in fields.items()
        })

    @classmethod
    def model_validate_json(cls, line):
        return cls(**json.loads(line))

    def model_dump(self, mode=None):
        return unwrap(self)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def wrap(value):
    if isinstance(value, dict):
        return Record(**value)
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value


def unwrap(value):
    if isinstance(value, Record):
        return {key: unwrap(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    return value


def visual_only(sample, shard_id, reference_paths):
    return Record(
        schema_version="r2v.v3.production_sample.1",
        sample_id=sample.sample_id,
        clip_uid=sample.sample_id,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=sample.r2v_instruction,
        source=dict(**sample.source.model_dump(), shard_id=shard_id),
        references=[
            dict(
                image_id=f"image_{i}", image_index=i, kind="subject",
                image_path=path, source_frame_index=ref.source_frame_index,
                entity_id=ref.entity_id, synthetic=False,
            )
            for i, (ref, path) in enumerate(zip(sample.references, reference_paths), 1)
        ],
    )


class ProductionSample(Record):
    def __init__(self, **fields):
        super().__init__(schema_version="r2v.v3.production_sample.1", **fields)


class Models:
    DatasetSample = Record
    EnrichedSample = Record
    ProductionReference = Record
    ProductionSample = ProductionSample
    ProductionSampleSource = Record
    visual_only_sample = staticmethod(visual_only)
    dataset_reference_kind = staticmethod(lambda ref: "subject")
    production_visual_reference = staticmethod(
        lambda reference, image_index, kind, image_path: Record(
            image_id=f"image_{image_index}", image_index=image_index,
            kind=kind, image_path=image_path,
            entity_id=reference.entity_id,
            source_frame_index=reference.source_frame_index,
            synthetic=False,
        )
    )


@pytest.fixture
def sample_group(tmp_path: Path):
    root, runs, out = (tmp_path / part for part in ("official", "private", "output/group-000000"))
    shard_a = "shard-000000000-000009999"
    shard_b = "shard-000010000-000019999"
    marker = root / "state/resource_epochs/group-000000/completed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(
        status="completed", export_completed=True, group_id="group-000000",
        canonical_shards=[shard_a, shard_b], sample_count=2,
    )))
    for shard, uid in ((shard_a, "clip-a"), (shard_b, "clip-b")):
        state = root / "state/shards" / shard
        state.mkdir(parents=True)
        (state / "completed.json").write_text(json.dumps(dict(
            status="completed", export_completed=True, shard_id=shard, sample_count=1,
        )))
        photo = root / "shards" / shard / "references" / uid / "subject_1.png"
        photo.parent.mkdir(parents=True)
        photo.write_bytes(b"image")
        (root / "shards" / shard / "samples.jsonl").write_text(json.dumps(dict(
            sample_id=uid, target_video=f"/videos/{uid}.mp4", t2v_caption="caption",
            r2v_instruction="original <Image 1>",
            source=dict(parent_video_id="parent", clip_suffix="0"),
            references=[dict(
                type="entity", token="<ref_subject_1>", entity_id="e1",
                image_path=f"references/{uid}/subject_1.png", source_frame_index=1,
            )],
        )) + "\n")
    attr = runs / shard_b / "subject_attributes"
    (attr / "final").mkdir(parents=True)
    (attr / "final/attr.png").write_bytes(b"attribute")
    (root / "shards" / shard_b / "enriched_samples.jsonl").write_text("{}\n")
    enrich = dict(
        sample_id="clip-b", clip_uid="clip-b", source_run_root=str(runs / shard_b),
        original_visual=dict(target_video="/videos/clip-b.mp4", source=dict(parent_video_id="parent", clip_suffix="0")),
        enriched_instruction="updated <Image 1> <Image 2>",
        references=[
            dict(image_index=1, kind="subject", image_id="image_1", entity_id="e1", source_frame_index=1),
            dict(image_index=2, kind="attribute", image_id="image_2", attribute_id="a1", owner_entity_id="e1", image_path="final/attr.png", source_frame_index=2),
        ],
        accepted_attributes=[dict(attribute_id="a1", owner_entity_id="e1", attribute_type="clothing", image_path="final/attr.png", source_frame_index=2, default_variant=None, final_selection="raw")],
    )
    (attr / "enriched_samples.jsonl").write_text(json.dumps(enrich) + "\n")
    return root, runs, out, marker, shard_b


def test_complete_group_maps_to_final_schema_and_images(sample_group):
    root, runs, out, _, shard_b = sample_group
    assert exporter.convert_group("group-000000", root, runs, out, Models()) == 2
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert rows[0]["r2v_instruction"] == "original <Image 1>"
    assert rows[1]["r2v_instruction"] == "updated <Image 1> <Image 2>"
    assert rows[1]["schema_version"] == "r2v.v3.production_sample.1"
    assert rows[1]["source"]["shard_id"] == shard_b
    assert [r["kind"] for r in rows[1]["references"]] == ["subject", "attribute"]
    assert all((out / ref["image_path"]).is_file() for row in rows for ref in row["references"])
    assert (out / "references/attributes" / shard_b).is_symlink()
    with pytest.raises(FileExistsError):
        exporter.convert_group("group-000000", root, runs, out, Models())


def test_unfinished_marker_fails_closed(sample_group):
    root, runs, out, marker, _ = sample_group
    value = json.loads(marker.read_text())
    value["export_completed"] = False
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="formal Export"):
        exporter.convert_group("group-000000", root, runs, out, Models())
    assert not out.exists()


def test_attribute_source_missing_fails_closed(sample_group):
    root, runs, out, _, shard_b = sample_group
    (runs / shard_b / "subject_attributes/enriched_samples.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="enriched input missing"):
        exporter.convert_group("group-000000", root, runs, out, Models())
    assert not out.exists()
    assert not list(out.parent.glob(".group-000000.tmp-*"))


def test_escape_references_rejected():
    with pytest.raises(ValueError, match="escapes"):
        exporter._attribute_path("shard-000000000-000009999", "../escape.png")
