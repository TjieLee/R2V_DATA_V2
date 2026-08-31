from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_v3_entity_mask_auto as auto_tool
from tools import run_v3_entity_mask_worker as worker_tool


def _shards(count: int) -> list[Path]:
    return [
        Path(f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl")
        for index in range(count)
    ]


def _global_assignments(
    num_shards: int,
    *,
    world_size: int,
    gpu_count: int,
) -> list[auto_tool.StaticShardAssignment]:
    shards = _shards(num_shards)
    gpus = [str(index) for index in range(gpu_count)]
    return [
        assignment
        for rank in range(world_size)
        for assignment in auto_tool.build_local_assignments(
            shards,
            rank=rank,
            world_size=world_size,
            gpus=gpus,
        )
    ]


@pytest.mark.parametrize(
    ("world_size", "gpu_count", "expected_counts"),
    [
        (11, 8, {5: 32, 4: 56}),
        (1, 8, {48: 8}),
        (6, 8, {8: 48}),
    ],
)
def test_balanced_static_assignment_production_topologies(
    world_size: int,
    gpu_count: int,
    expected_counts: dict[int, int],
) -> None:
    assignments = _global_assignments(
        384,
        world_size=world_size,
        gpu_count=gpu_count,
    )

    observed = {
        count: sum(assignment.shard_count == count for assignment in assignments)
        for count in {assignment.shard_count for assignment in assignments}
    }
    assert observed == expected_counts
    assigned_indexes = [
        index
        for assignment in assignments
        for index in range(
            assignment.shard_start_position,
            assignment.shard_end_position,
        )
    ]
    assert assigned_indexes == list(range(384))
    assert len(assigned_indexes) == len(set(assigned_indexes))


def test_more_workers_than_shards_assigns_empty_ranges() -> None:
    assignments = _global_assignments(10, world_size=2, gpu_count=8)

    assert [assignment.shard_count for assignment in assignments] == [
        *([1] * 10),
        *([0] * 6),
    ]


def test_rank_ranges_are_disjoint_and_restart_deterministic() -> None:
    shards = _shards(101)
    gpus = [str(index) for index in range(8)]
    rank_two_first = auto_tool.build_local_assignments(
        shards,
        rank=2,
        world_size=6,
        gpus=gpus,
    )
    rank_two_second = auto_tool.build_local_assignments(
        shards,
        rank=2,
        world_size=6,
        gpus=gpus,
    )
    rank_three = auto_tool.build_local_assignments(
        shards,
        rank=3,
        world_size=6,
        gpus=gpus,
    )

    assert rank_two_first == rank_two_second
    rank_two_indexes = {
        index
        for assignment in rank_two_first
        for index in range(
            assignment.shard_start_position,
            assignment.shard_end_position,
        )
    }
    rank_three_indexes = {
        index
        for assignment in rank_three
        for index in range(
            assignment.shard_start_position,
            assignment.shard_end_position,
        )
    }
    assert rank_two_indexes.isdisjoint(rank_three_indexes)


def test_static_assignment_rejects_invalid_rank_and_gpu_list() -> None:
    shards = _shards(10)
    with pytest.raises(ValueError, match="RANK"):
        auto_tool.build_local_assignments(
            shards,
            rank=2,
            world_size=2,
            gpus=["0"],
        )
    with pytest.raises(ValueError, match="GPU"):
        auto_tool.build_local_assignments(
            shards,
            rank=0,
            world_size=1,
            gpus=[],
        )


def test_zero_shards_produces_empty_assignments() -> None:
    assignments = auto_tool.build_local_assignments(
        [],
        rank=0,
        world_size=1,
        gpus=["0", "1"],
    )

    assert [assignment.shard_count for assignment in assignments] == [0, 0]


def test_rank_zero_initializes_and_other_rank_only_validates_startup_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "entity-mask"
    expected = {"schema_version": 1, "strategy": "test"}
    calls: list[str] = []
    monkeypatch.setattr(auto_tool, "_safe_output_root", lambda path: path)
    monkeypatch.setattr(
        auto_tool,
        "ensure_execution_identity",
        lambda *args, **kwargs: calls.append("ensure_execution"),
    )
    monkeypatch.setattr(
        auto_tool,
        "ensure_sam3_session_reuse_identity",
        lambda *args, **kwargs: calls.append("ensure_session"),
    )
    monkeypatch.setattr(
        auto_tool,
        "validate_execution_identity",
        lambda *args, **kwargs: calls.append("validate_execution"),
    )
    monkeypatch.setattr(
        auto_tool,
        "validate_sam3_session_reuse_identity",
        lambda *args, **kwargs: calls.append("validate_session"),
    )

    auto_tool.coordinate_startup_identity(
        output_root=output,
        rank=0,
        expected_topology=expected,
    )
    auto_tool.coordinate_startup_identity(
        output_root=output,
        rank=1,
        expected_topology=expected,
        timeout_seconds=0,
    )

    assert calls == [
        "ensure_execution",
        "ensure_session",
        "validate_execution",
        "validate_session",
    ]


def test_static_topology_rejects_heterogeneous_gpu_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "entity-mask"
    path = output / "_internal" / "entity-mask-static-assignment.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":1,"local_gpu_count":8}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local_gpu_count"):
        auto_tool.validate_static_topology(
            output,
            {"schema_version": 1, "local_gpu_count": 7},
        )


def test_static_worker_reuses_one_backend_across_assigned_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = _shards(4)
    base_config = tmp_path / "production.yaml"
    base_config.write_text("config: test\n", encoding="utf-8")
    output = tmp_path / "entity-mask"
    identity = SimpleNamespace(config=object())
    backend = object()
    factory_calls = 0
    close_calls = 0
    process_calls: list[dict[str, object]] = []

    def factory(config: object) -> object:
        nonlocal factory_calls
        assert config is identity.config
        factory_calls += 1
        return backend

    def close(value: object) -> None:
        nonlocal close_calls
        assert value is backend
        close_calls += 1

    def process(path: Path, **kwargs: object) -> dict[str, object]:
        process_calls.append({"path": path, **kwargs})
        return {"skipped": False}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(worker_tool, "load_config_identity", lambda path: identity)
    monkeypatch.setattr(
        worker_tool,
        "validate_entity_mask_production_config",
        lambda config: None,
    )
    monkeypatch.setattr(worker_tool, "enumerate_annotation_shards", lambda root: shards)
    monkeypatch.setattr(worker_tool, "validate_static_topology", lambda *args: None)
    monkeypatch.setattr(worker_tool, "validate_execution_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker_tool,
        "validate_sam3_session_reuse_identity",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(worker_tool, "default_backend_factory", factory)
    monkeypatch.setattr(worker_tool, "enable_sam3_session_reuse", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_tool, "process_shard", process)
    monkeypatch.setattr(worker_tool, "close_backend", close)

    result = worker_tool.main(
        [
            "--input-root",
            str(tmp_path / "input"),
            "--base-config",
            str(base_config),
            "--output-root",
            str(output),
            "--rank",
            "0",
            "--world-size",
            "1",
            "--local-gpu-count",
            "2",
            "--local-slot",
            "0",
            "--gpu",
            "5",
            "--global-worker-id",
            "0",
            "--shard-start-position",
            "0",
            "--shard-end-position",
            "2",
        ]
    )

    assert factory_calls == 1
    assert close_calls == 1
    assert result["processed_shards"] == 2
    assert [call["path"] for call in process_calls] == shards[:2]
    assert all(call["backend"] is backend for call in process_calls)
    assert all(call["acquire_lock"] is False for call in process_calls)
    assert all(
        call["execution_identity_prevalidated"] is True
        for call in process_calls
    )
