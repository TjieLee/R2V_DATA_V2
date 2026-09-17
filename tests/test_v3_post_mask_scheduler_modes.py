import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from tests.test_v3_post_mask_cluster import _completed, campaign
from tests.test_v3_post_mask_cluster import api as cluster_api
from tests.test_v3_post_mask_production import _write_rows
from tests.test_v3_post_mask_production import case as _case_fixture
from tests.test_v3_post_mask_runtime import (
    _accepted_factory,
    _all_phases_case,
    _ready,
    _setup,
)

MODES = ("legacy_serial", "wavefront_v2", "parallel_review_v21")
case = _case_fixture


@pytest.mark.parametrize("mode", MODES)
def test_retryable_suffix_and_terminal_publication_equivalent(case, mode):
    from r2v_data_v2.v3.post_mask_production import ShardPaths

    case = _all_phases_case(case)
    _write_rows(case, [_ready(case, "bad", 0), _ready(case, "good", 1)])
    api, paths, execution = _setup(case)
    results = []
    for index, selected in enumerate(("legacy_serial", mode)):
        paths = ShardPaths.for_shard(
            case[0].run_root.parent / ("comparison-" + str(index)), case[2]
        )
        settings = replace(execution, scheduler_mode=selected)
        args = {
            "entity_mask_root": case[1],
            "paths": paths,
            "git_commit": "test",
            "execution": settings,
        }
        first = api.run_post_mask_shard(
            case[0],
            **args,
            adapter_factory=_accepted_factory(
                [], terminal_edit_uid="bad", retry_stage="instruct"
            ),
        )
        assert not first.completed
        log = []
        last = api.run_post_mask_shard(
            case[0],
            **args,
            adapter_factory=_accepted_factory(log, terminal_edit_uid="bad"),
        )
        assert last.completed and last.sample_count == 1
        assert log == ["instruct", "subject_attributes"]
        results.append(
            [
                json.loads(line)
                for line in (paths.export_root / "samples.jsonl")
                .read_text()
                .splitlines()
            ]
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("mode", MODES)
def test_attribute_retryable_diagnostics_do_not_change_policy(case, mode):
    case = _all_phases_case(case)
    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)

    @contextmanager
    def factory(stage, storage, settings):
        with _accepted_factory([])(stage, storage, settings) as inner:

            class Phase:
                def run(self, uid=None):
                    if stage == "subject_attributes":
                        return {"failures": 1, "retryable_pending": 1}
                    return inner.run(uid)

            yield Phase()

    events = []
    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=replace(execution, scheduler_mode=mode),
        adapter_factory=factory,
        event_callback=events.append,
    )
    assert not result.completed and result.retryable_clip_uids == ("clip-0",)
    failure = next(e for e in events if e["event"] == "post_mask_clip_failed")
    assert failure["stage"] == "subject_attributes"
    assert failure["failures"] == failure["retryable_pending"] == 1
    assert failure["retryable"] is True


def test_scheduler_mode_cli_and_env(monkeypatch):
    monkeypatch.delenv("POST_MASK_SCHEDULER_MODE", raising=False)
    assert cluster_api().parser().parse_args([]).scheduler_mode == "wavefront_v2"
    for mode in MODES:
        monkeypatch.setenv("POST_MASK_SCHEDULER_MODE", mode)
        assert cluster_api().parser().parse_args([]).scheduler_mode == mode
        assert (
            cluster_api().parser().parse_args(["--scheduler-mode", mode]).scheduler_mode
            == mode
        )
    monkeypatch.setenv("POST_MASK_SCHEDULER_MODE", "typo")
    with pytest.raises(SystemExit):
        cluster_api().parser().parse_args([])
    with pytest.raises(SystemExit):
        cluster_api().parser().parse_args(["--scheduler-mode", "typo"])


def test_serial_has_whole_stage_barriers_and_resume_mode_is_identity_neutral(case):
    case = _all_phases_case(case)
    _write_rows(
        case,
        [
            _ready(case, "a", 0, "pending_remove"),
            _ready(case, "b", 1, "pending_remove"),
        ],
    )
    api, paths, execution = _setup(case)
    execution = replace(execution, scheduler_mode="legacy_serial")
    order = []

    @contextmanager
    def factory(stage, storage, settings):
        with _accepted_factory([])(stage, storage, settings) as inner:

            class Phase:
                def run(self, uid=None):
                    order.append((stage, uid))
                    return inner.run(uid)

            yield Phase()

    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=factory,
    )
    assert result.completed and result.sample_count == 2
    assert [stage for stage, _ in order] == [
        "remove",
        "remove",
        "pair",
        "reference_edit",
        "reference_edit",
        "reference_integrity",
        "reference_integrity",
        "instruct",
        "instruct",
        "subject_attributes",
        "subject_attributes",
    ]
    assert order[:5] == [
        ("remove", "a"),
        ("remove", "b"),
        ("pair", None),
        ("reference_edit", "a"),
        ("reference_edit", "b"),
    ]
    source_identity = (paths.run_root / "run.json").read_bytes()
    export_bytes = (paths.export_root / "samples.jsonl").read_bytes()
    for mode in MODES[1:]:
        resumed = api.run_post_mask_shard(
            case[0],
            entity_mask_root=case[1],
            paths=paths,
            git_commit="other",
            execution=replace(execution, scheduler_mode=mode),
            adapter_factory=lambda *a: pytest.fail("completed model stage rerun"),
        )
        assert resumed.completed
        assert (paths.run_root / "run.json").read_bytes() == source_identity
        assert (paths.export_root / "samples.jsonl").read_bytes() == export_bytes


def test_serial_worker_processes_assigned_shards_without_overlap(case, monkeypatch):
    from threading import Lock

    from r2v_data_v2.v3.post_mask_runtime import ShardResult

    _write_rows(case, [])
    for i in range(1, 3):
        case[2].with_name(f"shard-{i * 100:09d}-{i * 100 + 99:09d}.jsonl").write_text(
            ""
        )
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    lock, seen, events = Lock(), [], []

    def runner(config, *, paths, execution, **kw):
        assert lock.acquire(blocking=False), "serial shards overlapped"
        try:
            assert execution.scheduler_mode == "legacy_serial"
            seen.append(paths.shard_path)
            _completed(c, paths.shard_path, count=0)
            return ShardResult(True, 0, 0, 0, 0, (), {})
        finally:
            lock.release()

    cluster_api().run_worker(
        c,
        case[0],
        runtime=case[0].runtime,
        rank=0,
        world_size=1,
        slot=0,
        groups=(("4", "5"),),
        git_commit="test",
        runner=runner,
        scheduler_mode="legacy_serial",
        shard_inflight=8,
        event_callback=events.append,
    )
    assert seen == list(c.shards)
    assert events[-1]["max_active_shards_observed"] == 1
