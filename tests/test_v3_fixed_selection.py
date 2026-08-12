from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.manifest as manifest_module
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.v3.config import (
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.manifest import (
    build_manifest,
    export_fixed_selection_manifest,
)
from r2v_data_v2.v3.storage import RunStorage


def _roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    return writable, dataset_root, pretrained, user_models


def _record(
    video_path: Path,
    *,
    source_index: int,
    caption: str,
) -> dict[str, object]:
    identity = parse_clip_identity(video_path)
    return {
        "source_index": source_index,
        "clip_uid": identity.clip_uid,
        "parent_video_id": identity.parent_video_id,
        "video_path": str(video_path.resolve()),
        "caption_raw": caption,
        "metadata": {"caption": caption, "source": "fixed-test"},
        "clip_suffix": identity.clip_suffix,
    }


def _selection_records(dataset_root: Path, count: int = 8) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    source_indices = [41, 7, 29, 3, 88, 14, 62, 5]
    for index in range(count):
        video = dataset_root / "videos" / f"parent-{index}_{index}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{index}".encode())
        records.append(
            _record(
                video,
                source_index=source_indices[index],
                caption=f"caption {index}",
            )
        )
    return records


def _write_selection(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "r2v.v3.fixed-selection.1",
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection_manifest: Path,
    run_name: str,
    limit: int | None = None,
) -> V3Config:
    writable, dataset_root, pretrained, user_models = _roots(tmp_path, monkeypatch)
    dataset_json = dataset_root / "large-source.json"
    dataset_json.write_text("not parsed in fixed mode", encoding="utf-8")
    model = str(pretrained / "Qwen" / "judge")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / run_name,
        source=SourceConfig(
            limit=limit,
            selection_mode="fixed_selection_v1",
            selection_manifest=selection_manifest,
        ),
        qwen=QwenServicesConfig(
            candidate_judge=QwenServiceConfig(model=model),
            background_remove_judge=QwenServiceConfig(model=model),
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "object-remover",
        ),
    )
    config.validate()
    return config


def test_fixed_five_selection_is_ordered_and_never_scans_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    records = _selection_records(dataset_root)
    selection = tmp_path / "fixed.json"
    _write_selection(selection, records)
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="fixed-five",
        limit=5,
    )
    original_iter = manifest_module.iter_source_records
    yielded = 0

    def guarded_iter(path: str | Path) -> Iterator[dict[str, Any]]:
        nonlocal yielded
        resolved = Path(path).expanduser().resolve()
        if resolved == config.dataset_json.resolve():
            raise AssertionError("fixed selection scanned dataset_json")
        for raw in original_iter(path):
            yielded += 1
            if yielded > 5:
                raise AssertionError("fixed selection read beyond source.limit")
            yield raw

    monkeypatch.setattr(manifest_module, "iter_source_records", guarded_iter)
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    stats = build_manifest(config, storage)

    provenance = json.loads(
        (storage.root / "source_selection.json").read_text(encoding="utf-8")
    )
    expected_uids = [str(item["clip_uid"]) for item in records[:5]]
    assert stats.processed == 5
    assert yielded == 5
    assert [item["clip_uid"] for item in provenance["selected"]] == expected_uids
    assert provenance["selection_mode"] == "fixed_selection_v1"
    assert provenance["selection_manifest"] == str(selection.resolve())
    assert provenance["requested_limit"] == 5
    assert provenance["source_records_scanned"] == 5
    assert provenance["filesystem_checks"] == 5
    assert [
        storage.read_clip(clip_uid).source.source_index for clip_uid in expected_uids
    ] == [int(item["source_index"]) for item in records[:5]]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("clip_uid", "wrong", "clip_uid does not match"),
        ("parent_video_id", "wrong", "parent_video_id does not match"),
        ("clip_suffix", "999", "clip_suffix does not match"),
    ),
)
def test_fixed_selection_rejects_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    record = _selection_records(dataset_root, count=1)[0]
    record[field] = value
    selection = tmp_path / f"bad-{field}.json"
    _write_selection(selection, [record])
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name=f"bad-{field}",
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    with pytest.raises(ValueError, match=message):
        build_manifest(config, storage)

    assert not (storage.root / "source_selection.json").exists()


def test_fixed_selection_rejects_duplicate_clip_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    record = _selection_records(dataset_root, count=1)[0]
    selection = tmp_path / "duplicate.json"
    _write_selection(selection, [record, dict(record)])
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="duplicate",
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    with pytest.raises(ValueError, match="duplicate clip_uid"):
        build_manifest(config, storage)


def test_fixed_selection_rejects_missing_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    missing = dataset_root / "videos" / "missing-parent_0.mp4"
    record = _record(missing, source_index=9, caption="missing")
    selection = tmp_path / "missing.json"
    _write_selection(selection, [record])
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="missing",
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    with pytest.raises(FileNotFoundError, match="source video does not exist"):
        build_manifest(config, storage)


def test_fixed_selection_rejects_video_outside_public_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _roots(tmp_path, monkeypatch)
    outside = tmp_path / "outside-parent_0.mp4"
    outside.write_bytes(b"outside")
    record = _record(outside, source_index=4, caption="outside")
    selection = tmp_path / "outside.json"
    _write_selection(selection, [record])
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="outside",
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    with pytest.raises(ValueError, match="inside public dataset"):
        build_manifest(config, storage)


def test_fixed_selection_supports_jsonl_without_dataset_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    records = _selection_records(dataset_root, count=2)
    selection = tmp_path / "fixed.jsonl"
    selection.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="jsonl",
    )
    original_iter = manifest_module.iter_source_records

    def guarded_iter(path: str | Path) -> Iterator[dict[str, Any]]:
        if Path(path).expanduser().resolve() == config.dataset_json.resolve():
            raise AssertionError("fixed selection scanned dataset_json")
        yield from original_iter(path)

    monkeypatch.setattr(manifest_module, "iter_source_records", guarded_iter)
    storage = RunStorage(config)
    storage.initialize(git_commit="fixed-selection-test")

    stats = build_manifest(config, storage)

    assert stats.processed == 2


def test_fixed_selection_config_contract_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    records = _selection_records(dataset_root, count=1)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_selection(first, records)
    _write_selection(second, records)
    config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=first,
        run_name="config-contract",
    )

    assert config.source.limit is None
    assert config.source.allow_full_run is False
    assert replace(
        config,
        source=replace(config.source, selection_manifest=second),
    ).fingerprint() != config.fingerprint()
    with pytest.raises(ValueError, match="selection_manifest is required"):
        replace(
            config,
            source=replace(config.source, selection_manifest=None),
        ).validate()
    with pytest.raises(ValueError, match="start_index must be 0"):
        replace(
            config,
            source=replace(config.source, start_index=1),
        ).validate()
    with pytest.raises(ValueError, match="only valid for fixed_selection_v1"):
        replace(
            config,
            source=replace(config.source, selection_mode="sequential", limit=1),
        ).validate()

    yaml_path = tmp_path / "fixed-selection.yaml"
    assert config.qwen.candidate_judge is not None
    assert config.qwen.background_remove_judge is not None
    yaml_path.write_text(
        "\n".join(
            (
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  selection_mode: fixed_selection_v1",
                f"  selection_manifest: {first}",
                "qwen:",
                "  annotation:",
                f"    model: {config.qwen.annotation.model}",
                "  instruction_writer:",
                f"    model: {config.qwen.instruction_writer.model}",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
            )
        ),
        encoding="utf-8",
    )

    loaded = load_config(yaml_path)

    assert loaded.source.selection_mode == "fixed_selection_v1"
    assert loaded.source.selection_manifest == first


def test_exported_fixed_selection_recreates_exact_clip_sources_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_root, _, _ = _roots(tmp_path, monkeypatch)
    records = _selection_records(dataset_root, count=3)
    selection = tmp_path / "original-selection.json"
    _write_selection(selection, records)
    original_config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=selection,
        run_name="original-run",
    )
    original_storage = RunStorage(original_config)
    original_storage.initialize(git_commit="fixed-selection-test")
    build_manifest(original_config, original_storage)
    source_snapshot = {
        path.relative_to(original_storage.root): path.read_bytes()
        for path in original_storage.root.rglob("*")
        if path.is_file()
    }
    exported = tmp_path / "exports" / "reusable-selection.json"

    result = export_fixed_selection_manifest(original_storage.root, exported)

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert result == exported.resolve()
    assert payload["schema_version"] == "r2v.v3.fixed-selection.1"
    assert payload["records"] == records
    assert {
        path.relative_to(original_storage.root): path.read_bytes()
        for path in original_storage.root.rglob("*")
        if path.is_file()
    } == source_snapshot

    replay_config = _config(
        tmp_path,
        monkeypatch,
        selection_manifest=exported,
        run_name="replayed-run",
    )
    replay_storage = RunStorage(replay_config)
    replay_storage.initialize(git_commit="fixed-selection-test")
    build_manifest(replay_config, replay_storage)
    replay_provenance = json.loads(
        (replay_storage.root / "source_selection.json").read_text(encoding="utf-8")
    )
    expected_uids = [str(item["clip_uid"]) for item in records]
    assert [item["clip_uid"] for item in replay_provenance["selected"]] == expected_uids
    for clip_uid in expected_uids:
        assert replay_storage.read_clip(clip_uid).source == original_storage.read_clip(
            clip_uid
        ).source
