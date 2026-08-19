from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.manifest import parse_source_record
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.production_source import (
    JeaVideoMotionAdapter,
    parse_jea_video_motion_v1,
)
from r2v_data_v2.v3.schemas import DatasetRecord, DatasetSample
from r2v_data_v2.v3.storage import RunStorage
from r2v_data_v2.v3.subject_attributes import EnrichedSample
from tools.compact_v3_production_exports import compact_production_exports
from tools.prepare_v3_production_shards import prepare_production_shards


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
) -> dict[str, object]:
    return {
        "video_path": video_path or f"{parent}_{index}.mp4",
        "source_video_id": parent,
        "source_video_path": f"{parent}.mp4",
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


def _sample(sample_id: str, image_path: str) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        target_video=f"/source/{sample_id}.mp4",
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
) -> Path:
    shard = shards_root / shard_id
    reference = shard / "references" / sample_id / "subject.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    if include_reference:
        reference.write_bytes(b"png")
    sample = _sample(sample_id, f"references/{sample_id}/subject.png")
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


def _compaction_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    writable, _, _ = _roots(tmp_path, monkeypatch)
    output = (
        writable
        / "r2v_v3_exports"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
    )
    shards = output / "shards"
    shards.mkdir(parents=True)
    return shards, output


def test_compactor_rewrites_reference_paths_without_copying_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output = _compaction_roots(tmp_path, monkeypatch)
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="sample-a",
    )

    catalog = compact_production_exports(
        shards_root=shards,
        source_jsonl=tmp_path / "source.jsonl",
        created_at="2026-08-19T00:00:00+00:00",
    )

    record = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
    assert record["references"][0]["image_path"] == (
        "shards/shard-000000000-000000999/references/sample-a/subject.png"
    )
    assert catalog["total_samples"] == catalog["total_references"] == 1
    assert not list(output.rglob("*.mp4"))


def test_compactor_detects_duplicate_sample_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, _ = _compaction_roots(tmp_path, monkeypatch)
    for shard_id in (
        "shard-000000000-000000999",
        "shard-000001000-000001999",
    ):
        _write_export_shard(shards, shard_id, sample_id="duplicate")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        compact_production_exports(shards_root=shards)


def test_compactor_rejects_missing_reference_and_preserves_published_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output = _compaction_roots(tmp_path, monkeypatch)
    _write_export_shard(
        shards,
        "shard-000000000-000000999",
        sample_id="missing",
        include_reference=False,
    )
    published = output / "samples.jsonl"
    published.write_bytes(b"previous\n")

    with pytest.raises((FileNotFoundError, ValueError)):
        compact_production_exports(shards_root=shards)

    assert published.read_bytes() == b"previous\n"


def test_compactor_optionally_merges_valid_enriched_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards, output = _compaction_roots(tmp_path, monkeypatch)
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
        runs_root=runs_root,
    )

    published = EnrichedSample.model_validate_json(
        (output / "enriched_samples.jsonl").read_text(encoding="utf-8")
    )
    assert published.sample_id == "sample-a"
    assert catalog["total_enriched_samples"] == 1
