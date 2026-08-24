from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.manifest import parse_source_record
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.production_export import ProductionSample
from r2v_data_v2.v3.production_source import (
    JeaVideoMotionAdapter,
    parse_jea_video_motion_v1,
)
from r2v_data_v2.v3.schemas import DatasetRecord, DatasetSample
from r2v_data_v2.v3.storage import RunStorage
from r2v_data_v2.v3.subject_attributes import (
    EnrichedSample,
    OwnershipGeometry,
    SubjectAttributeCompletionReview,
    SubjectAttributeRecord,
    SubjectAttributeReview,
)
from tools.compact_v3_production_exports import compact_production_exports
from tools.prepare_v3_production_shards import (
    main as prepare_main,
    prepare_production_shards,
    probe_production_setup,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    models = writable / "models"
    for path in (writable, dataset, pretrained, models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", models)
    return writable, dataset, pretrained


def _record(
    index: int,
    *,
    parent: str = "movie-a",
    video_path: str | None = None,
    source_video_path: str | None = None,
) -> dict[str, object]:
    return {
        "video_path": video_path or f"{parent}_{index}.mp4",
        "source_video_id": parent,
        "source_video_path": source_video_path or f"{parent}.mp4",
        "shot_index": index,
        "start_frame": index * 10,
        "end_frame": index * 10 + 9,
        "num_frames": 10,
        "start_time": float(index),
        "end_time": float(index + 1),
        "duration": 1.0,
        "aesthetic_score_siglip": 0.8,
        "quality_score_qalign": 0.7,
        "watermark_crop": None,
        "motion_score": 0.6,
        "caption": f"caption {index}",
    }


def _production_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> dict[str, Path]:
    writable, dataset, pretrained = _roots(tmp_path, monkeypatch)
    source_dir = dataset / "jea"
    clips = source_dir / "clips_clean_cropped"
    videos = source_dir / "videos"
    clips.mkdir(parents=True)
    videos.mkdir(parents=True)
    records = [_record(index) for index in range(count)]
    for record in records:
        (clips / str(record["video_path"])).write_bytes(b"clip")
        (videos / str(record["source_video_path"])).write_bytes(b"source-video")
    source = source_dir / "shots_f03_motion.jsonl"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    base_config = writable / "r2v_v3_configs" / "base.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(
        yaml.safe_dump(
            {
                "dataset_json": str(source),
                "run_root": str(writable / "runs" / "base"),
                "export_root": str(writable / "exports" / "base"),
                "source": {"limit": 1},
                "qwen": {
                    "annotation": {"model": str(pretrained / "Qwen" / "judge")},
                    "instruction_writer": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                    "candidate_judge": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                    "background_remove_judge": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                },
                "remove": {
                    "base_model_path": str(pretrained / "Qwen" / "edit"),
                    "adapter_path": str(writable / "models" / "remover"),
                },
                "coverage": {"required_visible_frames": 7},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "writable": writable,
        "dataset": dataset,
        "source": source,
        "clips": clips,
        "videos": videos,
        "base_config": base_config,
        "state_root": writable
        / "r2v_v3_configs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1",
    }


def test_production_adapter_uses_clip_path_independent_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    selected = parse_jea_video_motion_v1(
        _record(0),
        source_index=12,
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
    )

    assert selected["video_path"] == str(fixture["clips"] / "movie-a_0.mp4")
    assert selected["video_path"] != str(fixture["videos"] / "movie-a.mp4")
    assert selected["caption_raw"] == "caption 0"
    assert selected["metadata"]["source_adapter"] == "jea_video_motion_v1"
    assert selected["metadata"]["source_relative_video_path"] == "movie-a_0.mp4"
    assert selected["metadata"]["source_relative_source_video_path"] == (
        "movie-a.mp4"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"source_video_id": "other"}, "source_video_id.*filename parent"),
        ({"shot_index": 9}, "shot_index.*filename suffix"),
        ({"video_path": "../movie-a_0.mp4"}, "escapes"),
    ),
)
def test_production_adapter_rejects_identity_and_relative_path_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    message: str,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    raw = {**_record(0), **change}

    with pytest.raises(ValueError, match=message):
        parse_jea_video_motion_v1(
            raw,
            source_index=0,
            clips_root=fixture["clips"],
            source_videos_root=fixture["videos"],
        )


def test_production_adapter_rejects_absolute_and_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    outside = fixture["dataset"] / "outside_0.mp4"
    outside.write_bytes(b"outside")
    adapter = JeaVideoMotionAdapter.create(
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
    )

    with pytest.raises(ValueError, match="escapes"):
        adapter.parse(_record(0, video_path=str(outside)), source_index=0)

    link = fixture["clips"] / "movie-a_9.mp4"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        adapter.parse(_record(9), source_index=0)


def test_probe_only_validates_both_roots_without_creating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    result = probe_production_setup(
        source_jsonl=fixture["source"],
        base_config=fixture["base_config"],
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
        path_probe_records=2,
    )
    assert result["clip_paths_verified"] == 2
    assert result["unique_source_video_paths_verified"] == 1
    assert not fixture["state_root"].exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_v3_production_shards.py",
            "--source-jsonl",
            str(fixture["source"]),
            "--base-config",
            str(fixture["base_config"]),
            "--clips-root",
            str(fixture["clips"]),
            "--source-videos-root",
            str(fixture["videos"]),
            "--state-root",
            str(fixture["state_root"]),
            "--path-probe-records",
            "2",
            "--probe-only",
        ],
    )
    prepare_main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["probe_only"] is True
    assert not fixture["state_root"].exists()


def test_source_video_probe_failure_creates_no_source_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    (fixture["videos"] / "movie-a.mp4").unlink()

    with pytest.raises(ValueError, match="existing source video"):
        prepare_production_shards(
            source_jsonl=fixture["source"],
            base_config=fixture["base_config"],
            clips_root=fixture["clips"],
            source_videos_root=fixture["videos"],
            state_root=fixture["state_root"],
            shard_size=1,
            path_probe_records=1,
        )

    assert not (fixture["state_root"] / "source.yaml").exists()


@pytest.mark.parametrize(
    "source_video_path",
    (
        "01/丁宝桢/01 4K.mkv",
        "originals/movie-a.mp4",
        "originals/movie-a.mov",
        "originals/movie-a.customvideo",
    ),
)
def test_source_video_probe_is_extension_agnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_video_path: str,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    provenance = fixture["videos"] / source_video_path
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_bytes(b"source-video")
    record_source_path = (
        str(provenance) if provenance.suffix == ".mkv" else source_video_path
    )
    fixture["source"].write_text(
        json.dumps(_record(0, source_video_path=record_source_path)) + "\n",
        encoding="utf-8",
    )

    result = probe_production_setup(
        source_jsonl=fixture["source"],
        base_config=fixture["base_config"],
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
        path_probe_records=1,
    )

    assert result["clip_paths_verified"] == 1
    assert result["unique_source_video_paths_verified"] == 1


@pytest.mark.parametrize(
    "failure_mode",
    (
        "missing",
        "directory",
        "relative_escape",
        "absolute_escape",
        "symlink_escape",
    ),
)
def test_source_video_probe_rejects_missing_and_escaped_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    outside = fixture["dataset"] / "outside.customvideo"
    outside.write_bytes(b"outside")
    if failure_mode == "missing":
        source_video_path = "missing.customvideo"
    elif failure_mode == "directory":
        directory = fixture["videos"] / "not-a-file.customvideo"
        directory.mkdir()
        source_video_path = directory.name
    elif failure_mode == "relative_escape":
        source_video_path = "../outside.customvideo"
    elif failure_mode == "absolute_escape":
        source_video_path = str(outside)
    else:
        link = fixture["videos"] / "linked.customvideo"
        link.symlink_to(outside)
        source_video_path = link.name
    fixture["source"].write_text(
        json.dumps(_record(0, source_video_path=source_video_path)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing source video"):
        probe_production_setup(
            source_jsonl=fixture["source"],
            base_config=fixture["base_config"],
            clips_root=fixture["clips"],
            source_videos_root=fixture["videos"],
            path_probe_records=1,
        )


def test_processed_clip_probe_still_requires_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    processed = fixture["clips"] / "movie-a_0.mov"
    processed.write_bytes(b"processed-shot")
    fixture["source"].write_text(
        json.dumps(_record(0, video_path=processed.name)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing source video"):
        probe_production_setup(
            source_jsonl=fixture["source"],
            base_config=fixture["base_config"],
            clips_root=fixture["clips"],
            source_videos_root=fixture["videos"],
            path_probe_records=1,
        )


def test_preparer_max_shards_stops_at_exact_cursor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=5)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
        "max_shards": 1,
    }

    first = prepare_production_shards(**arguments)
    cursor_path = fixture["state_root"] / "state" / "cursor.json"
    first_cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    first_two_bytes = sum(
        len(line)
        for line in fixture["source"].read_bytes().splitlines(keepends=True)[:2]
    )
    second = prepare_production_shards(**arguments)

    assert [path.name for path in first] == [
        "shard-000000000-000000001.yaml"
    ]
    assert first_cursor["next_source_index"] == 2
    assert first_cursor["next_byte_offset"] == first_two_bytes
    assert [path.name for path in second] == [
        "shard-000000002-000000003.yaml"
    ]


def test_legacy_source_parser_and_config_remain_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    raw = {
        "video_path": str(fixture["clips"] / "movie-a_0.mp4"),
        "source_video_path": str(fixture["videos"] / "wrong.mp4"),
        "caption": "legacy",
    }
    before = load_config(fixture["base_config"])
    base_bytes = fixture["base_config"].read_bytes()

    parsed = parse_source_record(raw)
    prepare_production_shards(
        source_jsonl=fixture["source"],
        base_config=fixture["base_config"],
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
        state_root=fixture["state_root"],
        shard_size=2,
        path_probe_records=1,
    )

    assert parsed.video_path == fixture["clips"] / "movie-a_0.mp4"
    assert parsed.caption_raw == "legacy"
    assert fixture["base_config"].read_bytes() == base_bytes
    assert load_config(fixture["base_config"]).fingerprint() == before.fingerprint()


def test_preparer_generates_current_fixed_selection_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)

    generated = prepare_production_shards(
        source_jsonl=fixture["source"],
        base_config=fixture["base_config"],
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
        state_root=fixture["state_root"],
        shard_size=2,
        path_probe_records=1,
    )

    assert len(generated) == 1
    config = load_config(generated[0])
    assert config.source.selection_mode == "fixed_selection_v1"
    assert config.source.start_index == 0
    assert config.source.limit is None
    assert config.source.allow_full_run is False
    assert config.coverage.required_visible_frames == 7
    storage = RunStorage(config)
    storage.initialize(git_commit="production-test")
    assert build_manifest(config, storage).processed == 2
    assert not list(fixture["state_root"].rglob("*.mp4"))


def test_preparer_resumes_from_cursor_after_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
    }
    prepare_production_shards(**arguments)
    cursor_path = fixture["state_root"] / "state" / "cursor.json"
    first_cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    appended = [_record(2), _record(3)]
    for record in appended:
        (fixture["clips"] / str(record["video_path"])).write_bytes(b"clip")
    with fixture["source"].open("a", encoding="utf-8") as handle:
        for record in appended:
            handle.write(json.dumps(record) + "\n")

    generated = prepare_production_shards(**arguments)
    second_cursor = json.loads(cursor_path.read_text(encoding="utf-8"))

    assert first_cursor["next_source_index"] == 2
    assert second_cursor["next_source_index"] == 4
    assert generated[0].name == "shard-000000002-000000003.yaml"
    records = [
        json.loads(line)
        for line in (
            fixture["state_root"]
            / "selections"
            / "shard-000000002-000000003.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["source_index"] for record in records] == [2, 3]


def test_preparer_freezes_base_config_bytes_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
    }
    prepare_production_shards(**arguments)
    source_yaml = fixture["state_root"] / "source.yaml"
    descriptor = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    original_base = fixture["base_config"].read_bytes()
    assert descriptor["base_config_sha256"] == hashlib.sha256(
        original_base
    ).hexdigest()
    assert descriptor["base_config_fingerprint"] == load_config(
        fixture["base_config"]
    ).fingerprint()

    fixture["base_config"].write_bytes(original_base + b"\n# byte mutation\n")
    with pytest.raises(ValueError, match="base_config_sha256"):
        prepare_production_shards(**arguments)

    fixture["base_config"].write_bytes(original_base)
    descriptor["base_config_fingerprint"] = "changed"
    source_yaml.write_text(
        yaml.safe_dump(descriptor, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_config_fingerprint"):
        prepare_production_shards(**arguments)


def test_preparer_resumes_after_committed_trailing_blank_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=1)
    with fixture["source"].open("ab") as handle:
        handle.write(b"\n  \n")
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
        "seal_tail": True,
    }
    prepare_production_shards(**arguments)
    cursor_path = fixture["state_root"] / "state" / "cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert cursor["next_byte_offset"] == fixture["source"].stat().st_size

    appended = _record(1)
    (fixture["clips"] / str(appended["video_path"])).write_bytes(b"clip")
    with fixture["source"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(appended) + "\n")
    generated = prepare_production_shards(**arguments)

    assert [path.name for path in generated] == [
        "shard-000000001-000000001.yaml"
    ]
    selection = (
        fixture["state_root"]
        / "selections"
        / "shard-000000001-000000001.jsonl"
    )
    selected = json.loads(selection.read_text(encoding="utf-8"))
    assert selected["source_index"] == 1


def test_preparer_leaves_partial_thousandth_record_unconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=999)
    final_record = _record(999)
    (fixture["clips"] / str(final_record["video_path"])).write_bytes(b"clip")
    complete_line = (json.dumps(final_record) + "\n").encode("utf-8")
    split = len(complete_line) // 2
    with fixture["source"].open("ab") as handle:
        handle.write(complete_line[:split])
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 1000,
        "path_probe_records": 1,
    }

    assert prepare_production_shards(**arguments) == []
    cursor_path = fixture["state_root"] / "state" / "cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert cursor["next_source_index"] == 0
    assert cursor["next_line_number"] == 1
    assert cursor["next_byte_offset"] == 0
    assert not list((fixture["state_root"] / "shards").glob("*.yaml"))

    with fixture["source"].open("ab") as handle:
        handle.write(complete_line[split:])
    generated = prepare_production_shards(**arguments)

    assert [path.name for path in generated] == [
        "shard-000000000-000000999.yaml"
    ]
    committed = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert committed["next_source_index"] == 1000
    assert committed["next_line_number"] == 1001
    assert committed["next_byte_offset"] == fixture["source"].stat().st_size


def test_preparer_seal_tail_stops_before_partial_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    complete_offset = fixture["source"].stat().st_size
    final_record = _record(2)
    (fixture["clips"] / str(final_record["video_path"])).write_bytes(b"clip")
    complete_line = (json.dumps(final_record) + "\n").encode("utf-8")
    split = len(complete_line) // 2
    with fixture["source"].open("ab") as handle:
        handle.write(complete_line[:split])
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 3,
        "path_probe_records": 1,
        "seal_tail": True,
    }

    generated = prepare_production_shards(**arguments)
    cursor_path = fixture["state_root"] / "state" / "cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))

    assert [path.name for path in generated] == [
        "shard-000000000-000000001.yaml"
    ]
    assert cursor["next_source_index"] == 2
    assert cursor["next_line_number"] == 3
    assert cursor["next_byte_offset"] == complete_offset

    with fixture["source"].open("ab") as handle:
        handle.write(complete_line[split:])
    resumed = prepare_production_shards(**arguments)
    assert [path.name for path in resumed] == [
        "shard-000000002-000000002.yaml"
    ]


def test_preparer_rejects_source_truncation_and_committed_line_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
    }
    original = fixture["source"].read_bytes()
    prepare_production_shards(**arguments)
    fixture["source"].write_bytes(original[:10])
    with pytest.raises(ValueError, match="truncated"):
        prepare_production_shards(**arguments)

    fixture["source"].write_bytes(original.replace(b"caption 1", b"caption X"))
    with pytest.raises(ValueError, match="committed prefix changed"):
        prepare_production_shards(**arguments)


def test_preparer_seals_complete_shard_and_optional_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=3)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
    }

    first = prepare_production_shards(**arguments)
    second = prepare_production_shards(**arguments)
    tail = prepare_production_shards(**arguments, seal_tail=True)

    assert [path.name for path in first] == ["shard-000000000-000000001.yaml"]
    assert second == []
    assert [path.name for path in tail] == ["shard-000000002-000000002.yaml"]


def test_preparer_refuses_changed_immutable_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    arguments = {
        "source_jsonl": fixture["source"],
        "base_config": fixture["base_config"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
        "state_root": fixture["state_root"],
        "shard_size": 2,
        "path_probe_records": 1,
    }
    prepare_production_shards(**arguments)
    selection = (
        fixture["state_root"]
        / "selections"
        / "shard-000000000-000000001.jsonl"
    )
    selection.write_text("changed\n", encoding="utf-8")
    (fixture["state_root"] / "state" / "cursor.json").unlink()

    with pytest.raises(FileExistsError, match="immutable production artifact"):
        prepare_production_shards(**arguments)


def test_preparer_isolates_malformed_record_without_raw_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _production_fixture(tmp_path, monkeypatch, count=2)
    malformed = {
        **_record(2),
        "source_video_id": "wrong-parent",
        "large_private_field": "x" * 100_000,
    }
    (fixture["clips"] / "movie-a_2.mp4").write_bytes(b"clip")
    with fixture["source"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(malformed) + "\n")

    prepare_production_shards(
        source_jsonl=fixture["source"],
        base_config=fixture["base_config"],
        clips_root=fixture["clips"],
        source_videos_root=fixture["videos"],
        state_root=fixture["state_root"],
        shard_size=3,
        path_probe_records=1,
    )

    error_text = (
        fixture["state_root"] / "state" / "source_errors.jsonl"
    ).read_text(encoding="utf-8")
    error = json.loads(error_text)
    assert error["line_number"] == 3
    assert error["source_index"] == 2
    assert error["video_path"] == "movie-a_2.mp4"
    assert "large_private_field" not in error_text
    selection = (
        fixture["state_root"]
        / "selections"
        / "shard-000000000-000000002.jsonl"
    )
    assert len(selection.read_text(encoding="utf-8").splitlines()) == 2


def _sample_target_path(sample_id: str) -> Path:
    return config_module.ALLOWED_DATASET_ROOT / "clips" / "review" / f"{sample_id}.mp4"


def _sample(sample_id: str, image_path: str, *, target_video: Path) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        target_video=str(target_video),
        t2v_caption="A person walks.",
        r2v_instruction="Use <image 1> to show the person walking.",
        references=[
            {
                "token": "<ref_subject_1>",
                "type": "entity",
                "entity_id": "e1",
                "scope": "full",
                "visible_region": "whole",
                "image_path": image_path,
                "source_frame_index": 0,
                "source_clip_uid": None,
                "source_entity_id": None,
                "synthetic": False,
            }
        ],
        source={"parent_video_id": "parent", "clip_suffix": "0"},
    )


def _write_export_shard(
    shards_root: Path,
    shard_id: str,
    *,
    sample_id: str,
    include_reference: bool = True,
    target_relative: Path | None = None,
    reference_color: tuple[int, int, int] = (10, 20, 30),
) -> Path:
    shard = shards_root / shard_id
    reference = shard / "references" / sample_id / "subject.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    if include_reference:
        Image.new("RGB", (8, 8), reference_color).save(reference)
    target = (
        config_module.ALLOWED_DATASET_ROOT
        / "clips"
        / (target_relative or Path("review") / f"{sample_id}.mp4")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"processed-shot")
    sample = _sample(
        sample_id,
        f"references/{sample_id}/subject.png",
        target_video=target,
    )
    (shard / "samples.jsonl").write_text(
        sample.model_dump_json() + "\n",
        encoding="utf-8",
    )
    dataset = DatasetRecord(
        dataset_version=shard_id,
        created_at="2026-08-19T00:00:00+00:00",
        git_commit="abc123",
        config_hash="config-hash",
        annotation_model="qwen",
        background_remove_backend="backend",
        sample_count=1,
        reference_count=1,
    )
    (shard / "dataset.json").write_text(
        dataset.model_dump_json(),
        encoding="utf-8",
    )
    return shard


def _write_reviewable_export_shard(
    shards_root: Path,
    shard_id: str,
    *,
    sample_id: str,
    target_relative: Path,
) -> tuple[Path, list[Path]]:
    shard = shards_root / shard_id
    source_paths: list[Path] = []
    references: list[dict[str, object]] = []
    specifications = (
        ("<ref_subject_1>", "entity", "e1", "full", "whole", "subject"),
        ("<ref_object_1>", "entity", "e2", "full", "whole", "object"),
        ("<ref_group_1>", "entity", "e3", "full", "whole", "group"),
        ("<ref_bg_1>", "background", None, "scene", "central", "background"),
    )
    for index, (token, reference_type, entity_id, scope, region, name) in enumerate(
        specifications,
        1,
    ):
        source_path = shard / "references" / sample_id / f"{name}.png"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index, index + 10, index + 20)).save(source_path)
        source_paths.append(source_path)
        references.append(
            {
                "token": token,
                "type": reference_type,
                "entity_id": entity_id,
                "scope": scope,
                "visible_region": region,
                "image_path": source_path.relative_to(shard).as_posix(),
                "source_frame_index": index - 1,
                "source_clip_uid": None,
                "source_entity_id": None,
                "synthetic": False,
            }
        )
    target = config_module.ALLOWED_DATASET_ROOT / "clips" / target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"processed-shot")
    sample = DatasetSample(
        sample_id=sample_id,
        target_video=str(target),
        t2v_caption="People and an object appear in a scene.",
        r2v_instruction=(
            "Use <image 1>, <image 2>, <image 3>, and <image 4> as references."
        ),
        references=references,
        source={"parent_video_id": "parent", "clip_suffix": "0"},
    )
    (shard / "samples.jsonl").write_text(
        sample.model_dump_json() + "\n",
        encoding="utf-8",
    )
    dataset = DatasetRecord(
        dataset_version=shard_id,
        created_at="2026-08-19T00:00:00+00:00",
        git_commit="abc123",
        config_hash="config-hash",
        annotation_model="qwen",
        background_remove_backend="backend",
        sample_count=1,
        reference_count=len(references),
    )
    (shard / "dataset.json").write_text(
        dataset.model_dump_json(),
        encoding="utf-8",
    )
    return target, source_paths


def _compaction_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    writable, dataset, _ = _roots(tmp_path, monkeypatch)
    output = (
        writable
        / "r2v_v3_exports"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    shards = output / "shards"
    shards.mkdir(parents=True)
    (dataset / "clips").mkdir()
    source = dataset / "jea" / "shots_f03_motion.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    source_yaml = (
        writable
        / "r2v_v3_configs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
        / "source.yaml"
    )
    source_yaml.parent.mkdir(parents=True)
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "schema_version": "r2v.v3.production-source.1",
                "source_adapter": "jea_video_motion_v1",
                "source_jsonl": str(source),
                "base_config_path": str(writable / "r2v_v3_configs" / "base.yaml"),
                "base_config_sha256": "base-sha256",
                "base_config_fingerprint": "base-fingerprint",
                "clips_root": str(dataset / "clips"),
                "source_videos_root": str(dataset / "videos"),
                "shard_size": 1000,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return shards, output, source, source_yaml


def test_compactor_rewrites_reference_paths_without_copying_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="sample-a",
    )

    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        created_at="2026-08-19T00:00:00+00:00",
    )

    record = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
    assert record["schema_version"] == "r2v.v3.production_sample.1"
    assert record["clip_uid"] == "sample-a"
    assert record["references"][0]["kind"] == "subject"
    assert record["references"][0]["attribute_id"] is None
    assert record["references"][0]["image_path"] == (
        "references/review/sample-a/subject_e1.png"
    )
    assert catalog["total_samples"] == catalog["total_references"] == 1
    assert catalog["total_visual_references"] == 1
    assert catalog["total_attribute_references"] == 0
    assert catalog["samples_jsonl_sha256"] == hashlib.sha256(
        (output / "samples.jsonl").read_bytes()
    ).hexdigest()
    assert not list(output.rglob("*.mp4"))

    source_reference = (
        shards
        / "shard-000000000-000000999"
        / "references"
        / "sample-a"
        / "subject.png"
    )
    published_reference = output / record["references"][0]["image_path"]
    assert source_reference.stat().st_ino == published_reference.stat().st_ino
    published_inode = published_reference.stat().st_ino
    compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        created_at="2026-08-19T00:00:00+00:00",
    )
    assert published_reference.stat().st_ino == published_inode


def test_compactor_publishes_human_reviewable_visual_reference_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    sample_id = "42038f3a-long-opaque-clip-uid"
    target_relative = Path("01") / "丁宝桢" / "01 4K" / "v_b..._00041.mp4"
    target, source_references = _write_reviewable_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id=sample_id,
        target_relative=target_relative,
    )
    source_hashes = [_file_sha256(path) for path in source_references]

    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        created_at="2026-08-20T00:00:00+00:00",
    )

    production = ProductionSample.model_validate_json(
        (output / "samples.jsonl").read_text(encoding="utf-8")
    )
    expected_directory = "references/01/丁宝桢/01 4K/v_b..._00041"
    assert [reference.image_path for reference in production.references] == [
        f"{expected_directory}/subject_e1.png",
        f"{expected_directory}/object_e2.png",
        f"{expected_directory}/group_e3.png",
        f"{expected_directory}/background.png",
    ]
    background_reference = production.references[-1]
    assert background_reference.kind == "background"
    assert background_reference.entity_id is None
    assert background_reference.scope == "scene"
    assert all(
        sample_id not in reference.image_path for reference in production.references
    )
    for source_reference, reference in zip(
        source_references,
        production.references,
        strict=True,
    ):
        destination = output / reference.image_path
        assert destination.resolve(strict=True).is_file()
        assert source_reference.stat().st_ino == destination.stat().st_ino
    assert [_file_sha256(path) for path in source_references] == source_hashes
    assert production.target_video == str(target)
    assert not list(output.rglob("*.mp4"))
    assert catalog["total_samples"] == 1
    assert catalog["total_visual_references"] == 4
    assert catalog["total_attribute_references"] == 0
    assert catalog["total_references"] == 4
    assert catalog["samples_jsonl_sha256"] == _file_sha256(
        output / "samples.jsonl"
    )


def test_compactor_rejects_canonical_reference_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    target_relative = Path("shared") / "shot.mp4"
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="sample-a",
        target_relative=target_relative,
        reference_color=(1, 2, 3),
    )
    _write_export_shard(
        shards,
        "shard-000001000-000001999",
        sample_id="sample-b",
        target_relative=target_relative,
        reference_color=(4, 5, 6),
    )

    with pytest.raises(FileExistsError, match="conflicting published reference"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
        )


@pytest.mark.parametrize("unsafe_target", ("outside", "dot_escape"))
def test_compactor_rejects_target_video_outside_or_escaping_clips_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_target: str,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard = _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="sample-a",
    )
    clips_root = config_module.ALLOWED_DATASET_ROOT / "clips"
    if unsafe_target == "outside":
        target = config_module.ALLOWED_DATASET_ROOT / "outside.mp4"
    else:
        target = clips_root / "review" / ".." / "escaped.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"processed-shot")
    samples_path = shard / "samples.jsonl"
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    payload["target_video"] = str(target)
    samples_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    expected = "below clips_root" if unsafe_target == "outside" else "unsafe dot"
    with pytest.raises(ValueError, match=expected):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
        )


def test_compactor_detects_duplicate_sample_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    for shard_id in (
        "shard-000000000-000000999",
        "shard-000001000-000001999",
    ):
        _write_export_shard(shards, shard_id, sample_id="duplicate")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
        )


def test_compactor_rejects_missing_reference_and_preserves_published_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="missing",
        include_reference=False,
    )
    published = output / "samples.jsonl"
    published.write_bytes(b"previous\n")

    with pytest.raises((FileNotFoundError, ValueError)):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
        )

    assert published.read_bytes() == b"previous\n"


def test_compactor_optionally_merges_valid_enriched_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    writable = config_module.ALLOWED_WRITABLE_ROOT
    runs_root = (
        writable
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    run_root = runs_root / shard_id
    visual = run_root / "clips" / "sample-a" / "frames" / "00.jpg"
    visual.parent.mkdir(parents=True)
    visual.write_bytes(b"jpg")
    enriched = EnrichedSample(
        sample_id="sample-a",
        clip_uid="sample-a",
        source_run_root=str(run_root),
        original_visual={},
        original_instruction="original",
        enriched_instruction="Use <Image 1>.",
        references=[
            {
                "image_id": "image_1",
                "image_index": 1,
                "kind": "subject",
                "origin": "visual_run",
                "entity_id": "e1",
                "image_path": "clips/sample-a/frames/00.jpg",
                "source_frame_index": 0,
            }
        ],
        accepted_attributes=[],
    )
    enriched_path = run_root / "subject_attributes" / "enriched_samples.jsonl"
    enriched_path.parent.mkdir(parents=True)
    enriched_path.write_text(enriched.model_dump_json() + "\n", encoding="utf-8")

    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        runs_root=runs_root,
    )

    published = EnrichedSample.model_validate_json(
        (output / "enriched_samples.jsonl").read_text(encoding="utf-8")
    )
    production = ProductionSample.model_validate_json(
        (output / "samples.jsonl").read_text(encoding="utf-8")
    )
    assert published.sample_id == "sample-a"
    assert production.r2v_instruction == "Use <Image 1>."
    assert production.references[0].image_path == (
        "references/review/sample-a/subject_e1.png"
    )
    assert catalog["total_enriched_samples"] == 1


def _accepted_attribute(
    image_path: str,
    *,
    final_selection: str | None = "raw",
) -> SubjectAttributeRecord:
    completed = final_selection == "completed"
    return SubjectAttributeRecord(
        attribute_id="a1",
        owner_entity_id="e1",
        attribute_type="hair",
        phrase="long dark hair",
        grounding_prompt="the person's long dark hair",
        status="accepted",
        image_path=image_path,
        source_frame_index=1,
        source_frame_slot=1,
        owner_candidate_id="candidate_1",
        same_frame_as_owner_reference=False,
        sam3_prompt="long dark hair",
        ownership_geometry=OwnershipGeometry(
            passed=True,
            reason="passed",
            owner_overlap_ratio=1.0,
            maximum_other_owner_overlap_ratio=0.0,
            attribute_to_owner_area_ratio=0.2,
            near_owner_region=True,
            attribute_area_pixels=400,
            attribute_long_side_pixels=200,
            significant_component_count=1,
            largest_component_ratio=1.0,
            second_largest_component_ratio=0.0,
        ),
        review=SubjectAttributeReview(
            attribute_id="a1",
            matches_attribute=True,
            owner_binding_correct=True,
            recognizable=True,
            characteristic_appearance_visible=True,
            usable_as_attribute_condition=True,
            structure_complete=not completed,
            completion_recommended=completed,
            reason=("repair recommended" if completed else "accepted"),
        ),
        completion_review=(
            SubjectAttributeCompletionReview(
                verdict="accept",
                same_physical_entity=True,
                identity_preserved=True,
                original_visible_attributes_preserved=True,
                exactly_one_entity=True,
                missing_parts_plausibly_completed=True,
                no_duplicate_entity=True,
                no_unrelated_entity=True,
                no_severe_structure_artifact=True,
                style_coherent=True,
                resolution_usable=True,
                reference_usable=True,
                certain=True,
                reason="accepted",
            )
            if completed
            else None
        ),
        final_selection=final_selection,
        completion_attempted=completed,
        completion_outcome=("selected_completed" if completed else "not_attempted"),
        reason="accepted",
    )


def _write_enriched_with_attribute(
    *,
    runs_root: Path,
    shard_id: str,
    sample_id: str,
    attribute_exists: bool = True,
    final_selection: str | None = "raw",
    image_mode: str | None = None,
    instruction: str = "Use <Image 1> with <Image 2>.",
) -> Path:
    run_root = runs_root / shard_id
    visual = run_root / "clips" / sample_id / "frames" / "00.jpg"
    visual.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(visual)
    attribute_path = f"references/{sample_id}/a1.png"
    attribute = run_root / "subject_attributes" / attribute_path
    attribute.parent.mkdir(parents=True)
    if attribute_exists:
        mode = image_mode or ("RGB" if final_selection == "completed" else "RGBA")
        color = (20, 30, 40) if mode == "RGB" else (20, 30, 40, 180)
        Image.new(mode, (16, 8), color).save(attribute)
    accepted = _accepted_attribute(
        attribute_path,
        final_selection=final_selection,
    )
    enriched = EnrichedSample(
        sample_id=sample_id,
        clip_uid=sample_id,
        source_run_root=str(run_root),
        original_visual={
            "target_video": str(_sample_target_path(sample_id)),
            "source": {"parent_video_id": "parent", "clip_suffix": "0"},
        },
        original_instruction="Use <image 1>.",
        enriched_instruction=instruction,
        references=[
            {
                "image_id": "image_1",
                "image_index": 1,
                "kind": "subject",
                "origin": "visual_run",
                "entity_id": "e1",
                "image_path": f"clips/{sample_id}/frames/00.jpg",
                "source_frame_index": 0,
            },
            {
                "image_id": "image_2",
                "image_index": 2,
                "kind": "attribute",
                "origin": "attribute_enrichment",
                "attribute_id": "a1",
                "owner_entity_id": "e1",
                "attribute_type": accepted.attribute_type,
                "image_path": attribute_path,
                "source_frame_index": 1,
            },
        ],
        accepted_attributes=[accepted],
    )
    destination = run_root / "subject_attributes" / "enriched_samples.jsonl"
    destination.write_text(enriched.model_dump_json() + "\n", encoding="utf-8")
    return destination


@pytest.mark.parametrize("final_selection", ["raw", None])
def test_compactor_publishes_canonical_attribute_with_owner_and_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_selection: str | None,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
        final_selection=final_selection,
    )

    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        runs_root=runs_root,
    )
    production = ProductionSample.model_validate_json(
        (output / "samples.jsonl").read_text(encoding="utf-8")
    )

    assert [reference.image_index for reference in production.references] == [1, 2]
    attribute = production.references[1]
    assert attribute.kind == "attribute"
    assert attribute.entity_id is None
    assert attribute.attribute_id == "a1"
    assert attribute.owner_entity_id == "e1"
    assert attribute.attribute_type == "hair"
    assert attribute.synthetic is False
    assert attribute.image_path == (
        "references/review/sample-a/attribute_e1_a1_hair.png"
    )
    assert (output / attribute.image_path).is_file()
    source_attribute = (
        runs_root
        / shard_id
        / "subject_attributes"
        / "references"
        / "sample-a"
        / "a1.png"
    )
    published_attribute = output / attribute.image_path
    assert source_attribute.stat().st_ino == published_attribute.stat().st_ino
    assert _file_sha256(source_attribute) == _file_sha256(published_attribute)
    with Image.open(published_attribute) as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
    assert str(runs_root) not in (output / "samples.jsonl").read_text(
        encoding="utf-8"
    )
    assert len(list(output.rglob("subject.png"))) == 1
    assert not list(output.rglob("*.mp4"))
    assert catalog["total_visual_references"] == 1
    assert catalog["total_attribute_references"] == 1

    published_inode = published_attribute.stat().st_ino
    compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        runs_root=runs_root,
    )
    assert (output / attribute.image_path).is_file()
    assert published_attribute.stat().st_ino == published_inode


def test_compactor_publishes_completed_rgb_attribute_as_synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
        final_selection="completed",
    )
    source_attribute = (
        runs_root
        / shard_id
        / "subject_attributes"
        / "references"
        / "sample-a"
        / "a1.png"
    )
    source_sha = _file_sha256(source_attribute)

    compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        runs_root=runs_root,
    )
    production = ProductionSample.model_validate_json(
        (output / "samples.jsonl").read_text(encoding="utf-8")
    )

    attribute = production.references[1]
    assert attribute.kind == "attribute"
    assert attribute.attribute_id == "a1"
    assert attribute.owner_entity_id == "e1"
    assert attribute.attribute_type == "hair"
    assert attribute.source_frame_index == 1
    assert attribute.synthetic is True
    published_attribute = output / attribute.image_path
    assert _file_sha256(published_attribute) == source_sha
    with Image.open(published_attribute) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"


@pytest.mark.parametrize(
    ("final_selection", "image_mode", "expected_mode"),
    [
        ("raw", "RGB", "RGBA"),
        ("completed", "RGBA", "RGB"),
    ],
)
def test_compactor_rejects_attribute_mode_mismatched_with_final_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_selection: str,
    image_mode: str,
    expected_mode: str,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
        final_selection=final_selection,
        image_mode=image_mode,
    )

    with pytest.raises(ValueError, match=rf"must be an {expected_mode} PNG"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
            runs_root=runs_root,
        )


def test_compactor_rejects_orphan_enriched_and_missing_attribute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    enriched_path = _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="orphan",
    )
    with pytest.raises(ValueError, match="orphan enriched sample_id"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
            runs_root=runs_root,
        )

    enriched_path.unlink()
    _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
        attribute_exists=False,
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
            runs_root=runs_root,
        )


def test_compactor_rejects_duplicate_enriched_sample_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    enriched_path = _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
    )
    line = enriched_path.read_text(encoding="utf-8")
    enriched_path.write_text(line + line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate enriched sample_id"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=source,
            source_yaml=source_yaml,
            runs_root=runs_root,
        )


def test_production_sample_accepts_out_of_order_and_repeated_image_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    shard_id = "shard-000000000-000000999"
    _write_export_shard(shards, shard_id, sample_id="sample-a")
    runs_root = (
        config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_runs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    _write_enriched_with_attribute(
        runs_root=runs_root,
        shard_id=shard_id,
        sample_id="sample-a",
        instruction="Use <Image 2> with <Image 1>.",
    )
    compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
        runs_root=runs_root,
    )
    production = ProductionSample.model_validate_json(
        (output / "samples.jsonl").read_text(encoding="utf-8")
    )
    assert production.r2v_instruction == "Use <Image 2> with <Image 1>."

    payload = production.model_dump(mode="json")
    payload["r2v_instruction"] = "Use <Image 2>, <Image 1>, then <Image 2>."
    repeated = ProductionSample.model_validate(payload)
    assert repeated.r2v_instruction.count("<Image 2>") == 2


@pytest.mark.parametrize(
    "instruction",
    (
        "Use only <Image 1>.",
        "Use <Image 1>, <Image 2>, and unknown <Image 3>.",
    ),
)
def test_production_sample_rejects_missing_or_unknown_image_label(
    instruction: str,
) -> None:
    payload = {
        "sample_id": "sample-a",
        "clip_uid": "sample-a",
        "target_video": "/source/sample-a.mp4",
        "t2v_caption": "A person walks.",
        "r2v_instruction": instruction,
        "references": [
            {
                "image_id": "image_1",
                "image_index": 1,
                "kind": "subject",
                "entity_id": "e1",
                "image_path": "shards/shard-a/references/e1.png",
                "source_frame_index": 0,
                "scope": "full",
                "visible_region": "whole",
                "synthetic": False,
            },
            {
                "image_id": "image_2",
                "image_index": 2,
                "kind": "object",
                "entity_id": "e2",
                "image_path": "shards/shard-a/references/e2.png",
                "source_frame_index": 1,
                "scope": "full",
                "visible_region": "whole",
                "synthetic": False,
            },
        ],
        "source": {
            "parent_video_id": "parent",
            "clip_suffix": "0",
            "shard_id": "shard-a",
        },
    }
    with pytest.raises(ValueError, match="image labels must match references"):
        ProductionSample.model_validate(payload)


def test_compactor_rejects_source_jsonl_outside_public_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _, _, source_yaml = _compaction_roots(tmp_path, monkeypatch)

    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="below public dataset"):
        compact_production_exports(
            shards_root=shards,
            source_jsonl=outside,
            source_yaml=source_yaml,
        )


def test_compactor_publishes_samples_before_catalog_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output, source, source_yaml = _compaction_roots(tmp_path, monkeypatch)
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="sample-a",
    )
    replacements: list[str] = []
    original_replace = os.replace

    def observe_replace(source_path: str | Path, destination: str | Path) -> None:
        replacements.append(Path(destination).name)
        original_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", observe_replace)
    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=source,
        source_yaml=source_yaml,
    )

    assert replacements[-2:] == ["samples.jsonl", "catalog.json"]
    assert catalog["samples_jsonl_sha256"] == hashlib.sha256(
        (output / "samples.jsonl").read_bytes()
    ).hexdigest()
