from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_v3_entity_mask_auto as auto_tool
from tools import run_v3_entity_mask_recovery as recovery_tool
from tools import run_v3_entity_mask_worker as worker_tool


def _shards(count: int) -> list[Path]:
    return [
        Path(f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl")
        for index in range(count)
    ]


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds


class FakeWorkerProcess:
    def __init__(
        self,
        returncode: int | None,
        *,
        exit_on_terminate: bool = True,
    ) -> None:
        self.returncode = returncode
        self.exit_on_terminate = exit_on_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


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
    ("codes", "expected"),
    [
        ([0, 0, 0], 0),
        ([0, 2, 0], 2),
        ([0, -9, 0], 137),
        ([-15, 0], 143),
        ([0, -9, 2], 137),
    ],
)
def test_normalized_worker_exit_code(
    codes: list[int],
    expected: int,
) -> None:
    assert auto_tool.normalized_worker_exit_code(codes) == expected


def test_worker_cleanup_terminates_and_reaps_active_children() -> None:
    processes = [FakeWorkerProcess(None), FakeWorkerProcess(None)]

    result = auto_tool.terminate_worker_processes(processes)

    assert result["terminate_requested"] == 2
    assert result["killed"] == 0
    assert result["reaped"] == 2
    assert [process.terminate_calls for process in processes] == [1, 1]
    assert [process.kill_calls for process in processes] == [0, 0]


def test_worker_cleanup_kills_after_one_shared_grace_deadline() -> None:
    clock = FakeClock()
    processes = [
        FakeWorkerProcess(None, exit_on_terminate=False),
        FakeWorkerProcess(None, exit_on_terminate=False),
    ]

    result = auto_tool.terminate_worker_processes(
        processes,
        grace_seconds=0.2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert clock.current == pytest.approx(0.2)
    assert result["terminate_requested"] == 2
    assert result["killed"] == 2
    assert result["reaped"] == 2
    assert [process.kill_calls for process in processes] == [1, 1]


def test_worker_cleanup_only_reaps_already_exited_child() -> None:
    process = FakeWorkerProcess(0)

    result = auto_tool.terminate_worker_processes([process])

    assert result["already_exited"] == 1
    assert result["reaped"] == 1
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


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


def test_nonzero_rank_retries_delayed_execution_identity_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    attempts = 0
    sessions = 0
    monkeypatch.setattr(auto_tool, "_safe_output_root", lambda path: path)
    monkeypatch.setattr(auto_tool, "validate_static_topology", lambda *args: None)

    def validate_execution(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("execution marker not visible")

    def validate_session(*args: object, **kwargs: object) -> None:
        nonlocal sessions
        sessions += 1

    monkeypatch.setattr(auto_tool, "validate_execution_identity", validate_execution)
    monkeypatch.setattr(
        auto_tool,
        "validate_sam3_session_reuse_identity",
        validate_session,
    )

    auto_tool.coordinate_startup_identity(
        output_root=tmp_path,
        rank=1,
        expected_topology={"schema_version": 1},
        timeout_seconds=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert attempts == 2
    assert sessions == 1
    assert clock.sleeps == [1.0]


def test_nonzero_rank_retries_delayed_session_identity_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    executions = 0
    attempts = 0
    monkeypatch.setattr(auto_tool, "_safe_output_root", lambda path: path)
    monkeypatch.setattr(auto_tool, "validate_static_topology", lambda *args: None)

    def validate_execution(*args: object, **kwargs: object) -> None:
        nonlocal executions
        executions += 1

    def validate_session(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("session marker not visible")

    monkeypatch.setattr(auto_tool, "validate_execution_identity", validate_execution)
    monkeypatch.setattr(
        auto_tool,
        "validate_sam3_session_reuse_identity",
        validate_session,
    )

    auto_tool.coordinate_startup_identity(
        output_root=tmp_path,
        rank=1,
        expected_topology={"schema_version": 1},
        timeout_seconds=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert executions == 2
    assert attempts == 2
    assert clock.sleeps == [1.0]


def test_nonzero_rank_startup_identity_visibility_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(auto_tool, "_safe_output_root", lambda path: path)
    monkeypatch.setattr(
        auto_tool,
        "validate_static_topology",
        lambda *args: (_ for _ in ()).throw(FileNotFoundError("not visible")),
    )

    with pytest.raises(
        TimeoutError,
        match="complete Entity Mask startup identity",
    ):
        auto_tool.coordinate_startup_identity(
            output_root=tmp_path,
            rank=1,
            expected_topology={"schema_version": 1},
            timeout_seconds=2,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.sleeps == [1.0, 1.0]


@pytest.mark.parametrize("failure_stage", ["topology", "execution"])
def test_nonzero_rank_startup_identity_mismatch_fails_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(auto_tool, "_safe_output_root", lambda path: path)

    def validate_topology(*args: object) -> None:
        if failure_stage == "topology":
            raise ValueError("topology mismatch")

    def validate_execution(*args: object, **kwargs: object) -> None:
        if failure_stage == "execution":
            raise ValueError("execution mismatch")

    monkeypatch.setattr(auto_tool, "validate_static_topology", validate_topology)
    monkeypatch.setattr(auto_tool, "validate_execution_identity", validate_execution)
    monkeypatch.setattr(
        auto_tool,
        "validate_sam3_session_reuse_identity",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="mismatch"):
        auto_tool.coordinate_startup_identity(
            output_root=tmp_path,
            rank=1,
            expected_topology={"schema_version": 1},
            timeout_seconds=120,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.sleeps == []


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


@pytest.mark.parametrize("recovery", [False, True])
def test_static_worker_reuses_one_backend_across_assigned_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery: bool,
) -> None:
    shards = _shards(80 if recovery else 4)
    start, end = (18, 20) if recovery else (0, 2)
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
    monkeypatch.setattr(worker_tool, "_safe_output_root", lambda root: root)
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
            "1" if recovery else "0",
            "--world-size",
            "5" if recovery else "1",
            "--local-gpu-count",
            "8" if recovery else "2",
            "--local-slot",
            "1" if recovery else "0",
            "--gpu",
            "5",
            "--global-worker-id",
            "9" if recovery else "0",
            "--shard-start-position",
            str(start),
            "--shard-end-position",
            str(end),
            *(["--recover-incomplete-artifacts"] if recovery else []),
        ]
    )

    assert factory_calls == 1
    assert close_calls == 1
    assert result["processed_shards"] == 2
    assert [call["path"] for call in process_calls] == shards[start:end]
    assert all(call["recover_incomplete_artifacts"] is recovery for call in process_calls)
    assert all(call["backend"] is backend for call in process_calls)
    assert all(call["acquire_lock"] is False for call in process_calls)
    assert all(call["static_owner"] is True for call in process_calls)
    assert all(
        call["execution_identity_prevalidated"] is True
        for call in process_calls
    )


def _recovery_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_root=tmp_path / "input",
        base_config=tmp_path / "production.yaml",
        output_root=tmp_path / "output",
        log_root=tmp_path / "recovery-logs",
    )


def _recovery_worker(worker_id: int) -> dict[str, object]:
    start, end = auto_tool.balanced_shard_bounds(384, 40, worker_id)
    return {
        "global_worker_id": worker_id,
        "rank": worker_id // 8,
        "local_slot": worker_id % 8,
        "shard_start_position": start,
        "shard_end_position": end,
        "unfinished_canonical_shards": [f"shard-{start}"],
    }


def test_recovery_physical_gpu_remap_keeps_logical_shard_range(tmp_path: Path) -> None:
    worker = _recovery_worker(19)
    command = recovery_tool.worker_command(_recovery_args(tmp_path), worker, "7")
    args = worker_tool._parser().parse_args(command[2:])
    assert (args.rank, args.local_slot, args.global_worker_id) == (2, 3, 19)
    assert (args.world_size, args.local_gpu_count) == (5, 8)
    assert args.gpu == "7"
    assert (
        args.shard_start_position,
        args.shard_end_position,
    ) == auto_tool.balanced_shard_bounds(384, 40, 19)
    assert args.recover_incomplete_artifacts
    assert "--overwrite" not in command


def test_recovery_supervisor_reuses_gpu_and_separate_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands, environments, processes = [], [], []

    def launch(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        process = FakeWorkerProcess(0)
        processes.append(process)
        return process

    monkeypatch.setattr(recovery_tool.subprocess, "Popen", launch)
    args = _recovery_args(tmp_path)
    codes = recovery_tool.run_workers(
        args, [_recovery_worker(19), _recovery_worker(22)], ["6"]
    )
    assert codes == [0, 0]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == ["6", "6"]
    assert [
        command[command.index("--global-worker-id") + 1] for command in commands
    ] == ["19", "22"]
    assert len(list(args.log_root.glob("recovery-worker-*.log"))) == 2
    assert not args.output_root.exists()
    assert all(process.wait_calls == [None] for process in processes)


def test_recovery_supervisor_exception_reaps_only_its_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FakeWorkerProcess(None)
    calls = 0

    def launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("cannot spawn")
        return first

    monkeypatch.setattr(recovery_tool.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="cannot spawn"):
        recovery_tool.run_workers(
            _recovery_args(tmp_path),
            [_recovery_worker(0), _recovery_worker(1)],
            ["6", "7"],
        )
    assert first.terminate_calls == 1
    assert first.wait_calls


def test_recovery_logical_worker_duplicate_fails_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _recovery_args(tmp_path)
    command = recovery_tool.worker_command(args, _recovery_worker(19), "7")
    monkeypatch.setattr(worker_tool, "_safe_output_root", lambda root: root)
    monkeypatch.setattr(
        worker_tool, "_run", lambda args: pytest.fail("duplicate worker entered run")
    )
    lock = args.output_root / "_internal" / "recovery-workers" / "worker-19.lock"
    with worker_tool.shard_lock(lock), pytest.raises(RuntimeError):
        worker_tool.main(command[2:])
    # The lock is released after exit, so retry can safely start this same owner.
    monkeypatch.setattr(worker_tool, "_run", lambda args: {"success": True})
    assert worker_tool.main(command[2:]) == {"success": True}


@pytest.mark.parametrize("selection", ["19,19", "-1", "40"])
def test_recovery_rejects_invalid_worker_selection(selection: str) -> None:
    with pytest.raises(ValueError):
        recovery_tool.selected_workers(selection, {"workers": [_recovery_worker(19)]})


def test_recovery_dry_run_is_read_only_and_keeps_frozen_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _recovery_args(tmp_path)
    args.input_root.mkdir()
    paths = [args.input_root / path.name for path in _shards(80)]
    monkeypatch.setattr(recovery_tool, "_safe_output_root", lambda root: root)
    monkeypatch.setattr(
        recovery_tool, "enumerate_annotation_shards", lambda root: paths
    )
    monkeypatch.setattr(
        recovery_tool, "_validate_live_config_identity", lambda identity: None
    )
    monkeypatch.setattr(
        auto_tool, "validate_entity_mask_production_config", lambda config: None
    )
    monkeypatch.setattr(
        recovery_tool, "validate_execution_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        recovery_tool,
        "validate_sam3_session_reuse_identity",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        recovery_tool,
        "load_config_identity",
        lambda path: SimpleNamespace(config=object()),
    )
    monkeypatch.setattr(
        recovery_tool,
        "run_workers",
        lambda *args: pytest.fail("dry-run spawned workers"),
    )
    marker = auto_tool._static_topology_path(args.output_root)
    marker.parent.mkdir(parents=True)
    topology = auto_tool.expected_static_topology(
        input_root=args.input_root, shard_paths=paths, world_size=5, local_gpu_count=8
    )
    marker.write_text(json.dumps(topology))
    before = marker.read_bytes()
    result = recovery_tool.main(
        [
            "--input-root",
            str(args.input_root),
            "--output-root",
            str(args.output_root),
            "--base-config",
            str(args.base_config),
            "--dry-run",
            "--logical-workers",
            "19",
        ]
    )
    assert result["unfinished_logical_workers"] == list(range(40))
    assert result["selected_logical_workers"] == [19]
    assert result["workers"][19]["shard_start_position"] == 38
    assert result["topology"]["global_worker_count"] == 40
    assert marker.read_bytes() == before
    assert list(args.output_root.rglob("*.json")) == [marker]
    assert not args.log_root.exists()
    topology["world_size"] = 4
    marker.write_text(json.dumps(topology))
    with pytest.raises(ValueError, match="topology mismatch"):
        recovery_tool.main(
            [
                "--input-root",
                str(args.input_root),
                "--output-root",
                str(args.output_root),
                "--base-config",
                str(args.base_config),
                "--dry-run",
            ]
        )
