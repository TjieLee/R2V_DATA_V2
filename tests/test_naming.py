from __future__ import annotations

import json
from pathlib import Path

from r2v_data_v2.config import load_config
from r2v_data_v2.manifest import (
    build_manifest,
    iter_source_records,
    parse_source_record,
)
from r2v_data_v2.naming import clip_uid, parse_clip_identity


def test_preserves_complete_numeric_suffix() -> None:
    simple = parse_clip_identity("/dataset/abc_77.mp4")
    nested = parse_clip_identity("/dataset/abc_11_0.mp4")
    named = parse_clip_identity("/dataset/my_video_2024_11_0.mp4")

    assert (simple.parent_video_id, simple.clip_suffix, simple.clip_order) == (
        "abc",
        "77",
        (77,),
    )
    assert (nested.parent_video_id, nested.clip_suffix, nested.clip_order) == (
        "abc",
        "11_0",
        (11, 0),
    )
    assert named.parent_video_id == "my_video"
    assert named.clip_suffix == "2024_11_0"
    assert named.clip_order == (2024, 11, 0)


def test_clip_uid_uses_normalized_absolute_path(tmp_path: Path) -> None:
    video = tmp_path / "clips" / "sample_1.mp4"
    equivalent = video.parent / ".." / "clips" / video.name
    assert clip_uid(video) == clip_uid(equivalent)
    assert len(clip_uid(video)) == 24


def test_shared_source_record_parser_preserves_v2_fields(tmp_path: Path) -> None:
    video = tmp_path / "scene_1_0.mp4"

    parsed = parse_source_record(
        {
            "file_path": str(video),
            "text": "A draft caption.",
            "title": "Visible title evidence",
            "ignored": {"nested": "value"},
        }
    )

    assert parsed.video_path == video.resolve()
    assert parsed.caption_raw == "A draft caption."
    assert parsed.metadata == {
        "title": "Visible title evidence",
        "text": "A draft caption.",
    }


def test_streaming_manifest_build_skips_missing_and_existing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir()
    video = dataset / "scene_11_0.mp4"
    video.write_bytes(b"fake")
    source = dataset / "source.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {"file_path": str(video), "text": "caption"},
                    {"file_path": str(dataset / "missing_2.mp4"), "text": "missing"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"dataset_json: {source}\noutput_root: {output}\n",
        encoding="utf-8",
    )
    config = load_config(config_path)

    first = build_manifest(config)
    second = build_manifest(config)
    records = tuple(iter_source_records(output / "manifests" / "source.jsonl"))

    assert first.processed == 1
    assert first.missing_videos == 1
    assert second.skipped_existing == 1
    assert len(records) == 1
    assert records[0]["clip_suffix"] == "11_0"
    assert (output / "logs" / "missing_videos.jsonl").is_file()
