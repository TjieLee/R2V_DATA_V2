from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import r2v_data_v2.v3.annotation as annotation_module
import r2v_data_v2.v3.config as config_module
import tools.run_v3_annotation_batch as batch
from r2v_data_v2.structured_output import ValidationIssue
from r2v_data_v2.v3.annotation import AnnotationAttempt, AnnotationFailure
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
)


def _record(index: int) -> dict[str, object]:
    return {
        "video_path": f"movie-a_{index}.mp4",
        "source_video_id": "movie-a",
        "source_video_path": "movie-a.mov",
        "shot_index": index,
        "start_frame": index * 10,
        "end_frame": index * 10 + 9,
        "num_frames": 10,
        "start_time": float(index),
        "end_time": float(index + 1),
        "duration": 1.0,
        "caption": f"caption {index}",
    }


def _ready_annotation() -> AnnotationState:
    return AnnotationState(
        status="ready",
        instruction_template="{{entity_1}} stands in {{background}}.",
        entities=[
            AnnotationEntity(
                entity_id="e1",
                reference_type="subject",
                phrase="a person in a blue coat",
                grounding_prompt="the person in a blue coat near the center",
            )
        ],
        background=BackgroundAnnotation(
            phrase="a bright room",
            grounding_prompt="the bright indoor room behind the person",
        ),
    )


class _FakeClient:
    def __init__(self, *, fail_names: set[str] | None = None) -> None:
        self.fail_names = fail_names or set()
        self.calls: list[Path] = []
        self.active = 0
        self.maximum_active = 0

    def annotate(self, *, video_path, caption_raw, metadata):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            assert video_path.is_file()
            assert caption_raw.startswith("caption")
            assert metadata["source_relative_video_path"] == video_path.name
            self.calls.append(video_path)
            if video_path.name in self.fail_names:
                raise RuntimeError("synthetic annotation failure")
            return AnnotationAttempt(
                annotation=_ready_annotation(),
                raw_responses=("{}",),
                repair_attempts=0,
                warnings=(),
            )
        finally:
            self.active -= 1


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
    shard_size: int = 2,
) -> tuple[batch.AnnotationBatchSettings, Path]:
    dataset_root = tmp_path / "public" / "dataset"
    source_root = dataset_root / "jea"
    clips_root = source_root / "clips_clean_cropped"
    source_videos_root = source_root / "source_videos"
    output_root = source_root / "entity_annotations"
    clips_root.mkdir(parents=True)
    source_videos_root.mkdir(parents=True)
    (source_videos_root / "movie-a.mov").write_bytes(b"source")
    records = [_record(index) for index in range(count)]
    for record in records:
        (clips_root / str(record["video_path"])).write_bytes(b"processed")
    source_jsonl = source_root / "shots.jsonl"
    source_jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    settings = batch.AnnotationBatchSettings(
        source_jsonl=source_jsonl,
        clips_root=clips_root,
        source_videos_root=source_videos_root,
        output_root=output_root,
        shard_size=shard_size,
        annotation_model="/models/Qwen3-VL-32B-Instruct",
        api_key="SECRET-MUST-NOT-BE-WRITTEN",
        temperature=0.0,
        max_tokens=4096,
        timeout_seconds=3600,
        repair_retries=1,
        entity_selection_mode="default",
        fps=2.0,
        workers=(
            batch.AnnotationBatchWorker(
                name="qwen_0",
                base_url="http://127.0.0.1:8001/v1",
                shard_start=0,
                shard_end=99,
            ),
            batch.AnnotationBatchWorker(
                name="qwen_1",
                base_url="http://127.0.0.1:8002/v1",
                shard_start=100,
                shard_end=199,
            ),
        ),
    )
    return settings, source_jsonl


def _part_path(settings: batch.AnnotationBatchSettings, shard_id: int) -> Path:
    start, end = batch.shard_bounds(shard_id, settings.shard_size)
    return (
        settings.output_root
        / "parts"
        / f"shard-{start:09d}-{end:09d}.jsonl"
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _valid_output_row(source_index: int, settings: batch.AnnotationBatchSettings):
    record = _record(source_index)
    return {
        "source_index": source_index,
        "clip_uid": f"movie-a_{source_index}",
        "video_path": str(settings.clips_root / str(record["video_path"])),
        "source_video_id": "movie-a",
        "source_video_path": "movie-a.mov",
        "shot_index": source_index,
        "status": "ready",
        "entities": [
            entity.model_dump(mode="json")
            for entity in _ready_annotation().entities
        ],
        "background": _ready_annotation().background.model_dump(mode="json"),
        "instruction_template": _ready_annotation().instruction_template,
        "reason": None,
        "repair_attempts": 0,
        "warnings": [],
    }


def test_default_shard_math_is_fixed_at_ten_thousand() -> None:
    assert batch.shard_bounds(0) == (0, 9_999)
    assert batch.shard_bounds(1) == (10_000, 19_999)
    assert batch.shard_bounds(379) == (3_790_000, 3_799_999)


def test_worker_selection_and_cli_overrides() -> None:
    settings = SimpleNamespace(
        workers=(
            batch.AnnotationBatchWorker("qwen_0", "http://old/v1", 0, 49),
        )
    )
    selected = batch.select_worker(
        settings,
        "qwen_0",
        base_url="http://new/v1",
        shard_start=7,
        shard_end=9,
    )
    assert selected == batch.AnnotationBatchWorker(
        "qwen_0", "http://new/v1", 7, 9
    )


def test_yaml_worker_selection_and_main_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "batch.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "annotation": {"model": "/model"},
                "workers": [
                    {
                        "name": "qwen_0",
                        "base_url": "http://old/v1",
                        "shard_start": 0,
                        "shard_end": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def fake_run(settings, worker):
        observed["settings"] = settings
        observed["worker"] = worker
        return {"rows": 0}

    monkeypatch.setattr(batch, "run_annotation_batch", fake_run)
    result = batch.main(
        [
            "--config",
            str(config),
            "--worker",
            "qwen_0",
            "--base-url",
            "http://new/v1",
            "--shard-start",
            "4",
            "--shard-end",
            "6",
        ]
    )
    assert result == {"rows": 0}
    assert observed["worker"] == batch.AnnotationBatchWorker(
        "qwen_0", "http://new/v1", 4, 6
    )


def test_final_partial_shard_and_sequential_processed_clip_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=5, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_end=2)
    client = _FakeClient()

    result = batch.run_annotation_batch(settings, worker, client=client)

    assert [shard["rows"] for shard in result["shards"]] == [2, 2, 1]
    assert _part_path(settings, 2).is_file()
    assert client.maximum_active == 1
    assert [path.parent for path in client.calls] == [settings.clips_root] * 5
    assert all("source_videos" not in str(path) for path in client.calls)


def replace_worker(
    worker: batch.AnnotationBatchWorker,
    **updates: object,
) -> batch.AnnotationBatchWorker:
    return batch.AnnotationBatchWorker(
        name=str(updates.get("name", worker.name)),
        base_url=str(updates.get("base_url", worker.base_url)),
        shard_start=int(updates.get("shard_start", worker.shard_start)),
        shard_end=int(updates.get("shard_end", worker.shard_end)),
    )


def test_shard_range_selection_processes_only_owned_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=6, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_start=1, shard_end=1)
    client = _FakeClient()

    result = batch.run_annotation_batch(settings, worker, client=client)

    assert [shard["shard_id"] for shard in result["shards"]] == [1]
    assert [path.name for path in client.calls] == ["movie-a_2.mp4", "movie-a_3.mp4"]
    assert not _part_path(settings, 0).exists()
    assert not _part_path(settings, 2).exists()


def test_success_and_failure_rows_preserve_annotation_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=2, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_end=0)
    client = _FakeClient(fail_names={"movie-a_1.mp4"})

    batch.run_annotation_batch(settings, worker, client=client)
    rows = _rows(_part_path(settings, 0))

    assert [row["source_index"] for row in rows] == [0, 1]
    assert rows[0]["status"] == "ready"
    assert rows[0]["entities"] == [
        entity.model_dump(mode="json") for entity in _ready_annotation().entities
    ]
    assert rows[0]["background"] == _ready_annotation().background.model_dump(
        mode="json"
    )
    assert rows[0]["instruction_template"] == _ready_annotation().instruction_template
    assert rows[1]["status"] == "failed"
    assert rows[1]["entities"] == []
    assert rows[1]["background"] is None
    assert rows[1]["instruction_template"] == ""
    assert rows[1]["clip_uid"] is not None
    assert rows[1]["video_path"] == str(settings.clips_root / "movie-a_1.mp4")
    assert "synthetic annotation failure" in str(rows[1]["reason"])
    failures = _rows(settings.output_root / "failures" / _part_path(settings, 0).name)
    assert failures == [rows[1]]


def test_annotation_failure_uses_existing_stage_reason_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=1, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_end=0)

    class FailedClient:
        def annotate(self, **_kwargs):
            raise AnnotationFailure(
                raw_responses=["bad"],
                issues=[
                    ValidationIssue(
                        code="schema_validation",
                        field="entities",
                        message="invalid",
                    )
                ],
            )

    batch.run_annotation_batch(settings, worker, client=FailedClient())
    assert _rows(_part_path(settings, 0))[0]["reason"] == "schema_validation"


def test_invalid_source_row_is_never_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, source = _fixture(tmp_path, monkeypatch, count=2, shard_size=3)
    source.write_text(
        json.dumps(_record(0)) + "\n{not-json}\n" + json.dumps(_record(1)) + "\n",
        encoding="utf-8",
    )
    worker = replace_worker(settings.workers[0], shard_end=0)
    client = _FakeClient()

    batch.run_annotation_batch(settings, worker, client=client)
    rows = _rows(_part_path(settings, 0))

    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert [row["status"] for row in rows] == ["ready", "failed", "ready"]
    assert len(client.calls) == 2


def test_partial_restart_skips_completed_prefix_and_recovers_partial_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=3, shard_size=3)
    worker = replace_worker(settings.workers[0], shard_end=0)
    partial = _part_path(settings, 0).with_name(
        f"{_part_path(settings, 0).name}.partial"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(
        batch._json_line(_valid_output_row(0, settings)) + b'{"source_index":1'
    )
    monkeypatch.setenv("GIT_COMMIT", "unrelated-new-commit")
    client = _FakeClient()

    batch.run_annotation_batch(settings, worker, client=client)
    rows = _rows(_part_path(settings, 0))

    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert [path.name for path in client.calls] == ["movie-a_1.mp4", "movie-a_2.mp4"]


def test_base_url_change_does_not_invalidate_partial_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=2, shard_size=2)
    worker = replace_worker(
        settings.workers[0],
        base_url="http://127.0.0.1:8999/v1",
        shard_end=0,
    )
    partial = _part_path(settings, 0).with_name(
        f"{_part_path(settings, 0).name}.partial"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(batch._json_line(_valid_output_row(0, settings)))
    client = _FakeClient()

    batch.run_annotation_batch(settings, worker, client=client)

    assert [path.name for path in client.calls] == ["movie-a_1.mp4"]
    metadata = json.loads(
        (
            settings.output_root
            / "parts"
            / "shard-000000000-000000001.meta.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["base_url"] == "http://127.0.0.1:8999/v1"
    assert "api_key" not in metadata
    assert "SECRET-MUST-NOT-BE-WRITTEN" not in json.dumps(metadata)


def test_completed_valid_shard_is_skipped_without_model_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=2, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_end=0)
    batch.run_annotation_batch(settings, worker, client=_FakeClient())
    original = _part_path(settings, 0).read_bytes()

    forbidden = _FakeClient()
    result = batch.run_annotation_batch(settings, worker, client=forbidden)

    assert result["skipped_shards"] == 1
    assert forbidden.calls == []
    assert _part_path(settings, 0).read_bytes() == original


def test_overlapping_worker_cannot_acquire_locked_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _ = _fixture(tmp_path, monkeypatch, count=1, shard_size=2)
    worker = replace_worker(settings.workers[0], shard_end=0)
    lock_path = (
        settings.output_root
        / "locks"
        / "shard-000000000-000000001.lock"
    )
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(batch.ShardLockedError, match="already locked"):
                batch.run_annotation_batch(settings, worker, client=_FakeClient())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_inspect_mode_counts_entities(tmp_path: Path, capsys) -> None:
    path = tmp_path / "part.jsonl"
    ready = _valid_output_row(
        0,
        SimpleNamespace(clips_root=tmp_path),
    )
    ready["entities"].extend(
        [
            {
                "entity_id": "e2",
                "reference_type": "object",
                "phrase": "a red box",
                "grounding_prompt": "the red box beside the person",
            },
            {
                "entity_id": "e3",
                "reference_type": "group",
                "phrase": "three distant birds",
                "grounding_prompt": "the three birds in the upper sky",
            },
        ]
    )
    failed = batch._failure_row(batch.SourceRow(1, _record(1)), "failed")
    path.write_bytes(batch._json_line(ready) + batch._json_line(failed))

    result = batch.main(["--inspect", str(path)])

    assert result["rows"] == 2
    assert result["ready"] == result["failed"] == 1
    assert result["total_entities"] == 3
    assert result["subject"] == result["object"] == result["group"] == 1
    assert json.loads(capsys.readouterr().out)["rows"] == 2


def test_existing_annotation_client_and_sanitizer_are_reused() -> None:
    assert batch.QwenAnnotationClient is annotation_module.QwenAnnotationClient
    assert batch.ANNOTATION_SANITIZER is annotation_module.sanitize_annotation_payload
    source = Path(batch.__file__).read_text(encoding="utf-8")
    assert "RunStorage" not in source
    assert "ThreadPoolExecutor" not in source
