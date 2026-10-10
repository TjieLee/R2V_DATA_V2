from __future__ import annotations

import json
from pathlib import Path

import pytest


def sample(uid="clip-a"):
    return {
        "schema_version": "r2v.v3.production_sample.1", "sample_id": uid,
        "clip_uid": uid, "target_video": f"/videos/{uid}.mp4", "t2v_caption": "visual",
        "r2v_instruction": "Person <Image 1>, shirt <Image 2>, room <Image 3>",
        "references": [
            {"image_id": "image_1", "image_index": 1, "kind": "subject", "entity_id": "e7",
                 "image_path": "references/visual/shard/subject.png", "source_frame_index": 3,
                 "scope": "full", "synthetic": False},
            {"image_id": "image_2", "image_index": 2, "kind": "attribute", "attribute_id": "a2",
                 "owner_entity_id": "e7", "attribute_type": "upper_clothing",
                 "image_path": "references/shirt.png", "source_frame_index": 4, "synthetic": False},
            {"image_id": "image_3", "image_index": 3, "kind": "background", "scope": "scene",
                 "image_path": "references/background.png", "source_frame_index": 5, "synthetic": False},
        ],
        "source": {"parent_video_id": "parent", "clip_suffix": "01", "shard_id": "shard"},
    }


def publish_group(source, name="group-000000", count=20, declared=None):
    directory = source / name
    directory.mkdir(parents=True)
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(sample(f"clip-{i}")) + "\n" for i in range(count))
    )
    marker = source.parent / "state/resource_epochs" / name / "completed.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "group_id": name, "status": "completed", "export_completed": True,
        "sample_count": count if declared is None else declared,
    }))
    return directory, marker


def api():
    from r2v_data_v2.h3 import ra2va_group_source
    return ra2va_group_source


def test_only_formally_published_groups_are_discovered(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source, "group-000018")
    publish_group(source, "group-000000")
    assert [g.group_id for g in api().discover_published_groups(source)] == [
        "group-000000", "group-000018",
    ]


def test_staging_and_incomplete_marker_are_not_imported(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source, ".group-000000.tmp")
    _, marker = publish_group(source, "group-000001")
    marker.write_text('{"status":"running"}')
    group, _ = publish_group(source, "group-000002")
    (group / "samples.jsonl").unlink()
    assert api().discover_published_groups(source) == []


def test_inventory_preserves_order_and_ownership(tmp_path):
    from r2v_data_v2.v3.production_export import ProductionSample
    source = tmp_path / "post_mask"
    directory, _ = publish_group(source)
    g = api().discover_published_groups(source)[0]
    root = tmp_path / "inventory"
    api().freeze_pilot_inventory(g, root, 20)
    tasks = [api().read_inventory_task(root, i) for i in range(20)]
    assert [t.clip_uid for t in tasks] == [f"clip-{i}" for i in range(20)]
    task = api().adapt_production_sample(ProductionSample.model_validate(sample()), directory, 0)
    assert task.sample == ProductionSample.model_validate(sample()).model_dump(mode="json")
    assert [p["picture_label"] for p in task.pictures] == [f"<Picture {i}>" for i in range(1, 4)]
    assert [p["image_id"] for p in task.pictures] == [f"image_{i}" for i in range(1, 4)]
    assert task.subjects[0]["entity_id"] == "e7"
    assert task.subjects[1]["owner_entity_id"] == "e7"
    assert task.subjects[1]["source_picture_labels"] == ["<Picture 2>"]
    assert task.subjects[2]["kind"] == "background"
    assert not any("sha" in key or "hash" in key for p in task.pictures for key in p)


def test_official_reference_symlink_is_read_only(tmp_path):
    source = tmp_path / "post_mask"
    directory, _ = publish_group(source, count=1)
    official = tmp_path / "official"
    official.mkdir()
    picture = official / "subject.png"
    picture.write_bytes(b"original")
    alias = directory / "references/visual/shard"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(official, target_is_directory=True)
    root = tmp_path / "inventory"
    api().freeze_pilot_inventory(api().discover_published_groups(source)[0], root, 20)
    task = api().read_inventory_task(root, 0)
    assert Path(task.pictures[0]["image_path"]) == alias / "subject.png"
    assert Path(task.pictures[0]["image_path"]).read_bytes() == b"original"
    assert alias.is_symlink() and picture.read_bytes() == b"original"


def test_resume_uses_inventory_without_source(tmp_path):
    source = tmp_path / "post_mask"
    directory, _ = publish_group(source, count=1)
    g = api().discover_published_groups(source)[0]
    root = tmp_path / "inventory"
    before = api().freeze_pilot_inventory(g, root, 20)
    (directory / "samples.jsonl").unlink()
    assert api().freeze_pilot_inventory(g, root, 20) == before
    assert api().read_inventory_task(root, 0).clip_uid == "clip-0"


def test_pilot_never_claims_full_group(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source, declared=39585)
    root = tmp_path / "inventory"
    meta = api().freeze_pilot_inventory(api().discover_published_groups(source)[0], root, 20)
    assert meta["mode"] == "pilot"
    assert meta["selected_count"] == 20 and meta["declared_source_count"] == 39585
    assert not (root / "COMPLETE").exists()


def test_invalid_and_duplicate_rows_never_publish_inventory(tmp_path):
    source = tmp_path / "post_mask"
    directory, _ = publish_group(source, count=2)
    g = api().discover_published_groups(source)[0]
    (directory / "samples.jsonl").write_text(json.dumps(sample()) + "\n" + json.dumps(sample()) + "\n")
    root = tmp_path / "inventory"
    with pytest.raises(ValueError, match="duplicate"):
        api().freeze_pilot_inventory(g, root, 20)
    assert not root.exists()
    (directory / "samples.jsonl").write_text("not JSON\n")
    with pytest.raises(ValueError):
        api().freeze_pilot_inventory(g, root, 20)
    assert not root.exists()


def test_start_row_freezes_only_the_requested_range_and_stops_reading(tmp_path):
    source = tmp_path / "post_mask"
    directory, _ = publish_group(source, count=18, declared=100)
    with (directory / "samples.jsonl").open("a") as stream:
        stream.write("must not read or parse the rest of the group\n")
    group = api().discover_published_groups(source)[0]
    root = tmp_path / "inventory"
    metadata = api().freeze_pilot_inventory(group, root, 8, start_row=10)
    assert metadata["start_row"] == 10 and metadata["selected_count"] == 8
    assert [api().read_inventory_task(root, i).clip_uid for i in range(8)] == [f"clip-{i}" for i in range(10, 18)]
    assert [api().read_inventory_task(root, i).ordinal for i in range(8)] == list(range(8))
    (directory / "samples.jsonl").unlink()
    assert api().freeze_pilot_inventory(group, root, 8, start_row=10) == metadata
    with pytest.raises(ValueError, match="resume inventory"):
        api().freeze_pilot_inventory(group, root, 8, start_row=0)


def test_start_row_is_persisted_in_coordinator_resume_settings(tmp_path):
    from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator

    source = tmp_path / "post_mask"
    publish_group(source, count=20)
    root = tmp_path / "run"
    coordinator = GroupCoordinator(root, source, 8, start_row=10)
    assert coordinator.select_current().group_id == "group-000000"
    resumed = GroupCoordinator(root, source, 8, start_row=10)
    assert resumed.select_current() == coordinator.select_current()
    with pytest.raises(ValueError, match="resume settings"):
        GroupCoordinator(root, source, 8, start_row=0)
