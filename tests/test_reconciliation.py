from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from r2v_data_v2.config import PipelineConfig
from r2v_data_v2.qwen_client import annotate_manifest
from r2v_data_v2.reconciliation import (
    reconcile_augmentations,
    reconcile_final_samples,
    reconcile_references,
    write_json_atomic,
)


class _NeverCalledClient:
    def annotate(self, **kwargs: object) -> None:
        del kwargs
        raise AssertionError("existing annotation artifact should be reused")


def test_write_json_atomic_uses_unique_temporary_paths_for_concurrent_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "clip.json"
    payloads = [
        {"writer": 1, "values": list(range(100))},
        {"writer": 2, "values": list(range(100, 200))},
    ]
    replace_barrier = threading.Barrier(2)
    replace_paths: list[Path] = []
    replace_paths_lock = threading.Lock()
    replace_operation_lock = threading.Lock()
    original_replace = Path.replace

    def synchronized_replace(source: Path, target: Path) -> Path:
        if target == destination:
            with replace_paths_lock:
                replace_paths.append(source)
            replace_barrier.wait(timeout=5)
            with replace_operation_lock:
                return original_replace(source, target)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_json_atomic, destination, payload)
            for payload in payloads
        ]
        for future in futures:
            future.result(timeout=10)

    assert len(replace_paths) == 2
    assert len(set(replace_paths)) == 2
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8")) in payloads
    assert all(not temporary.exists() for temporary in replace_paths)


def test_annotation_artifact_recovers_missing_jsonl(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    source = output_root / "manifests" / "source.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "video_path": "/read-only/video.mp4",
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "clip_order": [1, 0],
                "caption_raw": "draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = output_root / "annotations" / "clip-1.json"
    write_json_atomic(
        artifact,
        {
            "clip_uid": "clip-1",
            "caption": "already complete",
            "entities": [],
        },
    )

    stats = annotate_manifest(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        client=_NeverCalledClient(),  # type: ignore[arg-type]
    )

    recovered = json.loads(
        (output_root / "manifests" / "annotations.jsonl").read_text(encoding="utf-8")
    )
    assert stats.skipped_existing == 1
    assert recovered["clip_uid"] == "clip-1"


def test_reference_final_and_augmentation_indexes_rebuild_from_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    write_json_atomic(
        output_root / "references" / "clip-1" / "e1" / "metadata.json",
        {"clip_uid": "clip-1", "entity_id": "e1"},
    )
    write_json_atomic(
        output_root / "samples" / "clip-1.json",
        {"clip_uid": "clip-1", "references": [{"entity_id": "e1"}]},
    )
    write_json_atomic(
        (
            output_root
            / "references"
            / "clip-1"
            / "e1"
            / "augmented"
            / "viewpoint_00.json"
        ),
        {
            "clip_uid": "clip-1",
            "entity_id": "e1",
            "variant_type": "viewpoint",
            "variant_index": 0,
        },
    )

    assert reconcile_references(output_root) == 1
    assert reconcile_final_samples(output_root) == 1
    assert reconcile_augmentations(output_root) == 1
    assert (
        json.loads(
            (output_root / "manifests" / "references.jsonl").read_text(encoding="utf-8")
        )["entity_id"]
        == "e1"
    )
    assert (
        json.loads(
            (output_root / "manifests" / "final_samples.jsonl").read_text(
                encoding="utf-8"
            )
        )["clip_uid"]
        == "clip-1"
    )
    assert (
        json.loads(
            (output_root / "manifests" / "augmentations.jsonl").read_text(
                encoding="utf-8"
            )
        )["variant_type"]
        == "viewpoint"
    )
