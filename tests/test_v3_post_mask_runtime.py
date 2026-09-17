from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from r2v_data_v2.v3.post_mask_production import ShardPaths
from r2v_data_v2.v3.schemas import PairingState
from tests.test_v3_post_mask_production import _ready as _frozen_ready
from tests.test_v3_post_mask_production import _write_rows
from tests.test_v3_post_mask_production import case as _case_fixture

case = _case_fixture


def _ready(case, uid="clip-0", index=0, background="none"):
    from PIL import Image

    row = _frozen_ready(case, uid, index, background)
    for image in (
        case[1] / row["artifact_root"] / "run" / "clips" / uid / "frames"
    ).glob("*.jpg"):
        Image.new("RGB", (5, 4), (40, 50, 60)).save(image)
    return row


def _api():
    return importlib.import_module("r2v_data_v2.v3.post_mask_runtime")


def _setup(case):
    config, _root, shard = case
    paths = ShardPaths.for_shard(config.run_root.parent / "campaign", shard)
    api = _api()
    execution = api.ExecutionSettings(
        runtime=config.runtime,
        sam_gpu="4",
        boogu_gpu="5",
        qwen_lock_directory=paths.state_root.parent.parent / "qwen-node-0",
    )
    return api, paths, execution


def _fake_factory(log, *, fail=None):
    @contextmanager
    def factory(stage, storage, execution):
        log.append(("open", stage))

        class Phase:
            def run(self, clip_uid=None):
                clips = (
                    list(storage.iter_clips())
                    if clip_uid is None
                    else [storage.read_clip(clip_uid)]
                )
                log.append((stage, tuple(c.clip_uid for c in clips)))
                for clip in clips:
                    if fail == (stage, clip.clip_uid):
                        raise RuntimeError("injected transient failure")
                    if stage == "pair":
                        storage.write_pairing(
                            clip.clip_uid,
                            PairingState(status="rejected", reason="quality"),
                        )
                return {"processed": len(clips)}

        try:
            yield Phase()
        finally:
            log.append(("close", stage))

    return factory


def _run(case, log, *, fail=None, events=None):
    api, paths, execution = _setup(case)
    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="new",
        execution=execution,
        adapter_factory=_fake_factory(log, fail=fail),
        event_callback=events.append if events is not None else None,
    )
    return result, paths


@pytest.mark.parametrize("stage", ["remove", "reference_edit", "subject_attributes"])
def test_heavy_stage_clip_tasks_overlap(case, monkeypatch, stage):
    from threading import Barrier
    from types import SimpleNamespace

    config, root, shard = _all_phases_case(case)
    case = config, root, shard
    _write_rows(case, [_ready(case, "one", 0), _ready(case, "two", 1)])
    api, paths, execution = _setup(case)
    barrier = Barrier(2, timeout=2)
    entered = []
    overlapped = []
    monkeypatch.setattr(
        api, "phase_needed", lambda storage, clip, current: current == stage
    )

    @contextmanager
    def factory(current, storage, execution):
        def run(uid):
            entered.append(uid)
            barrier.wait()
            overlapped.append(uid)
            return {}

        yield SimpleNamespace(run=run)

    result = api.run_post_mask_shard(
        config,
        entity_mask_root=root,
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=factory,
    )
    assert not result.completed
    assert sorted(entered) == ["one", "two"]
    assert result.stage_counts[stage].get("clip_failures") == 2
    assert sorted(overlapped) == ["one", "two"]


@pytest.mark.parametrize("value", [0, -1, True])
@pytest.mark.parametrize("field", ["clip_inflight", "shard_inflight"])
def test_clip_inflight_rejects_invalid_execution_value(case, value, field):
    _, _, execution = _setup(case)
    with pytest.raises(ValueError, match=field):
        replace(execution, **{field: value})


@pytest.mark.parametrize("stage", ["remove", "reference_edit", "subject_attributes"])
def test_clip_b_boogu_overlaps_clip_a_qwen(case, monkeypatch, stage):
    from threading import Event
    from types import SimpleNamespace

    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources
    from r2v_data_v2.v3.profiling import profiled_openai_call

    a_qwen, b_boogu = Event(), Event()
    calls = []

    class Backend:
        started = False

        def __init__(self, config):
            pass

        def start(self, **kw):
            self.started = True

        def edit(self, **kw):
            calls.append(kw["uid"])

        def close(self):
            self.started = False

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Backend)
    config, root, shard = _all_phases_case(case)
    _write_rows((config, root, shard), [_ready(case, "a", 0), _ready(case, "b", 1)])
    api, paths, execution = _setup((config, root, shard))
    monkeypatch.setattr(
        api, "phase_needed", lambda storage, clip, current: current == stage
    )
    with PostMaskWorkerResources(
        config, execution, runtime_root=paths.state_root / "runtime"
    ) as resources:

        @contextmanager
        def factory(current, storage, settings):
            def run(uid):
                if uid == "b":
                    assert a_qwen.wait(3)
                    resources.edit(uid=uid)
                    b_boogu.set()
                else:
                    resources.edit(uid=uid)

                    def qwen():
                        a_qwen.set()
                        assert b_boogu.wait(3)

                    profiled_openai_call(
                        qwen,
                        component="test",
                        operation="test",
                        retry_index=0,
                        model="fake",
                        messages=[],
                    )
                return {}

            yield SimpleNamespace(run=run)

        result = api.run_post_mask_shard(
            config,
            entity_mask_root=root,
            paths=paths,
            git_commit="test",
            execution=execution,
            adapter_factory=factory,
            resources=resources,
        )
        assert not result.completed and b_boogu.is_set()
        assert calls == ["a", "b"]
        assert resources.summary()["boogu_start_count"] == 1


def test_heavy_threads_qwen_gate_allows_two_actual_requests(case, monkeypatch):
    from threading import Barrier
    from types import SimpleNamespace

    from r2v_data_v2.v3.profiling import _QWEN_CONCURRENCY_GATE, profiled_openai_call

    _write_rows(case, [_ready(case, "a", 0), _ready(case, "b", 1)])
    api, paths, execution = _setup(case)
    execution = replace(
        execution, runtime=replace(execution.runtime, qwen_max_inflight=2)
    )
    monkeypatch.setattr(
        api, "phase_needed", lambda storage, clip, stage: stage == "remove"
    )
    barrier, completed = Barrier(2, timeout=3), []

    @contextmanager
    def factory(stage, storage, settings):
        def run(uid):
            assert _QWEN_CONCURRENCY_GATE.get().maximum == 2

            def request():
                barrier.wait()
                completed.append(uid)

            profiled_openai_call(
                request,
                component="test",
                operation="test",
                retry_index=0,
                model="fake",
                messages=[],
            )
            return {}

        yield SimpleNamespace(run=run)

    api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=factory,
    )
    assert sorted(completed) == ["a", "b"]


@pytest.mark.parametrize("waiting_stage", ["reference_integrity", "subject_attributes"])
def test_post_pair_clip_stages_overlap_without_breaking_dependencies(
    case, waiting_stage, monkeypatch
):
    from threading import Event

    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources
    from r2v_data_v2.v3.profiling import profiled_openai_call

    case = _all_phases_case(case)
    _write_rows(case, [_ready(case, "a", 0), _ready(case, "b", 1)])
    api, paths, execution = _setup(case)
    a_waiting, b_generation = Event(), Event()
    calls, events = [], []
    generation_calls = []

    class Backend:
        started = False

        def __init__(self, config):
            pass

        def start(self, **kw):
            self.started = True

        def close(self):
            self.started = False

        def edit(self, *, uid):
            generation_calls.append(uid)
            if uid == "b":
                assert a_waiting.is_set()
                b_generation.set()

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Backend)
    with PostMaskWorkerResources(
        case[0], execution, runtime_root=paths.state_root / "runtime"
    ) as resources:

        @contextmanager
        def factory(stage, storage, settings):
            with _accepted_factory([])(stage, storage, settings) as inner:

                class Phase:
                    def run(self, uid=None):
                        if stage == "reference_edit" and uid == "b":
                            assert a_waiting.wait(3), (
                                "whole reference_edit stage blocked early clip"
                            )
                        if stage == "reference_edit":
                            resources.edit(uid=uid)
                        calls.append((stage, uid))
                        if stage == waiting_stage and uid == "a":

                            def qwen():
                                a_waiting.set()
                                assert b_generation.wait(3)

                            profiled_openai_call(
                                qwen,
                                component="test",
                                operation="test",
                                retry_index=0,
                                model="fake",
                                messages=[],
                            )
                        return inner.run(uid)

                yield Phase()

        result = api.run_post_mask_shard(
            case[0],
            entity_mask_root=case[1],
            paths=paths,
            git_commit="test",
            execution=execution,
            resources=resources,
            adapter_factory=factory,
            event_callback=events.append,
        )
        assert result.completed and result.sample_count == 2
        assert resources.summary()["max_active_clip_pipelines_observed"] == 2
        assert resources.summary()["boogu_start_count"] == 1
        assert resources.summary()["boogu_max_inflight_observed"] == 1
        assert generation_calls == ["a", "b"]
    for uid in ("a", "b"):
        assert [stage for stage, current in calls if current == uid] == [
            "reference_edit",
            "reference_integrity",
            "instruct",
            "subject_attributes",
        ]
    assert calls[0] == ("pair", None)
    assert calls.index((waiting_stage, "a")) < calls.index(("reference_edit", "b"))
    for stage in api.PHASES:
        assert (
            sum(
                e["event"] == "post_mask_stage_completed" and e["stage"] == stage
                for e in events
            )
            == 1
        )


def test_real_shard_runner_overlaps_pair_qwen_and_other_shard_remove(case, monkeypatch):
    from threading import Barrier, Event

    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.profiling import _QWEN_CONCURRENCY_GATE, profiled_openai_call
    from tests.test_v3_post_mask_cluster import accepted, campaign
    from tests.test_v3_post_mask_cluster import api as cluster_api

    case = _all_phases_case(case)
    config = accepted(case[0])
    config = replace(config, runtime=replace(config.runtime, qwen_max_inflight=2))
    case = config, case[1], case[2]
    second = case[2].with_name("shard-000000100-000000199.jsonl")
    _write_rows(case, [_ready(case, "a", 0, "pending_remove")])
    second_case = config, case[1], second
    _write_rows(second_case, [_ready(second_case, "b", 100, "pending_remove")])
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    a_pair, b_remove = Event(), Event()
    both_pairs = Barrier(2, timeout=3)
    pairs, events = [], []

    class Backend:
        started = False

        def __init__(self, config):
            pass

        def start(self, **kw):
            self.started = True

        def close(self):
            self.started = False

        def edit(self, *, uid):
            if uid == "b":
                assert a_pair.is_set()
                b_remove.set()

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Backend)
    api = _api()

    def runner(config, *, paths, resources, **kwargs):
        @contextmanager
        def factory(stage, storage, settings):
            with _accepted_factory([])(stage, storage, settings) as inner:

                class Phase:
                    def run(self, uid=None):
                        if stage == "remove":
                            if uid == "b":
                                assert a_pair.wait(3)
                            resources.edit(uid=uid)
                        if stage == "pair":
                            assert uid is None
                            pairs.append(
                                (
                                    paths.shard_path,
                                    tuple(c.clip_uid for c in storage.iter_clips()),
                                )
                            )

                            def qwen():
                                assert (
                                    _QWEN_CONCURRENCY_GATE.get() is resources.qwen_gate
                                )
                                if paths.shard_path == case[2]:
                                    a_pair.set()
                                    assert b_remove.wait(3)
                                both_pairs.wait()

                            profiled_openai_call(
                                qwen,
                                component="pair",
                                operation="test",
                                retry_index=0,
                                model="fake",
                                messages=[],
                            )
                        return inner.run(uid)

                yield Phase()

        return api.run_post_mask_shard(
            config, paths=paths, resources=resources, adapter_factory=factory, **kwargs
        )

    cluster_api().run_worker(
        c,
        config,
        runtime=config.runtime,
        rank=0,
        world_size=1,
        slot=0,
        groups=(("4", "5"),),
        git_commit="test",
        runner=runner,
        event_callback=events.append,
    )
    assert sorted(pairs) == [(case[2], ("a",)), (second, ("b",))]
    assert all(
        cluster_api().completed_receipt(c, shard) is not None for shard in c.shards
    )
    assert events[-1]["qwen_max_inflight_observed"] == 2
    assert events[-1]["boogu_start_count"] == 1
    assert events[-1]["max_active_shards_observed"] == 2


def test_ordinary_clip_infrastructure_failure_does_not_stop_worker_resources(case):
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources
    from r2v_data_v2.v3.post_mask_wavefront import bounded_results

    _, paths, execution = _setup(case)
    with PostMaskWorkerResources(
        case[0], execution, runtime_root=paths.state_root / "runtime"
    ) as resources:

        def broken_artifact(uid):
            raise ValueError("invalid run artifact")

        with pytest.raises(ValueError, match="invalid run artifact"):
            list(
                bounded_results(
                    broken_artifact, ["a"], 2, on_interrupt=resources.stop_accepting
                )
            )
        assert not resources.stopping
        assert resources.sam_call(lambda: "next shard") == "next shard"


def test_clip_inflight_change_reuses_durable_shard(case):
    _write_rows(case, [_ready(case)])
    result, paths = _run(case, [])
    assert result.completed
    original = paths.identity_path.read_bytes()
    export = (paths.export_root / "samples.jsonl").read_bytes()
    api, _, execution = _setup(case)
    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="later",
        execution=replace(execution, clip_inflight=2, shard_inflight=3),
        adapter_factory=lambda *a: pytest.fail("durable stage recomputed"),
    )
    assert result.completed and paths.identity_path.read_bytes() == original
    assert (paths.export_root / "samples.jsonl").read_bytes() == export


def test_shared_shard_write_lock_serializes_clip_storage_mutations(case, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    from r2v_data_v2.v3.post_mask_production import initialize_shard
    from r2v_data_v2.v3.runtime import ClipScopedStorage

    api, paths, _ = _setup(case)
    _write_rows(case, [])
    storage = initialize_shard(case[0], paths, git_commit="test")
    shared = api._ShardStorage(storage, ("a", "b"))
    a = ClipScopedStorage(shared, "a", shared_write_lock=shared.write_lock)
    b = ClipScopedStorage(shared, "b", shared_write_lock=shared.write_lock)
    inside, second_attempt = Event(), Event()
    guard, active, peak = Lock(), 0, 0
    original = storage.append_failure

    def append(**kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        if kwargs["clip_uid"] == "a":
            inside.set()
            assert second_attempt.wait(3)
        result = original(**kwargs)
        with guard:
            active -= 1
        return result

    monkeypatch.setattr(storage, "append_failure", append)

    def second():
        assert inside.wait(3)
        second_attempt.set()
        b.append_failure(stage="remove", clip_uid="b", reason="test")

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(a.append_failure, stage="remove", clip_uid="a", reason="test")
        two = pool.submit(second)
        one.result()
        two.result()
    assert peak == 1
    assert len(storage.failures_path.read_text().splitlines()) == 2


def test_quality_rejection_exports_once_and_restart_loads_publication(case):
    _write_rows(case, [_ready(case)])
    log, events = [], []
    result, paths = _run(case, log, events=events)
    assert result.completed and result.sample_count == 0
    assert paths.export_root.joinpath("dataset.json").is_file()
    assert paths.state_root.joinpath("completed.json").is_file()
    assert [e[1] for e in log if e[0] == "open"] == ["pair"]
    before = paths.export_root.joinpath("dataset.json").read_bytes()
    paths.state_root.joinpath("completed.json").unlink()  # crash after publish
    log.clear()
    second, _ = _run(case, log)
    assert second.completed and not log
    assert paths.export_root.joinpath("dataset.json").read_bytes() == before
    assert events[-1]["event"] == "post_mask_shard_completed"


def test_pair_waits_for_all_concurrent_remove_tasks(case):
    from threading import Barrier

    case = _all_phases_case(case)
    _write_rows(
        case,
        [
            _ready(case, "a", 0, "pending_remove"),
            _ready(case, "b", 1, "pending_remove"),
        ],
    )
    api, paths, execution = _setup(case)
    barrier = Barrier(2, timeout=2)
    removed, paired = [], []

    @contextmanager
    def factory(stage, storage, execution):
        with _accepted_factory([])(stage, storage, execution) as inner:

            class Phase:
                def run(self, uid=None):
                    if stage == "remove":
                        barrier.wait()
                        result = inner.run(uid)
                        removed.append(uid)
                        return result
                    if stage == "pair":
                        assert sorted(removed) == ["a", "b"]
                        assert uid is None
                        paired.append(tuple(c.clip_uid for c in storage.iter_clips()))
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
    assert result.completed
    assert paired == [("a", "b")]


def test_boogu_crash_keeps_clip_retryable_and_other_clip_continues(case, monkeypatch):
    from threading import Event

    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    case = _all_phases_case(case)
    _write_rows(
        case,
        [
            _ready(case, "a", 0, "pending_remove"),
            _ready(case, "b", 1, "pending_remove"),
        ],
    )
    api, paths, execution = _setup(case)
    failed = Event()
    calls, downstream = [], []

    class Backend:
        started = False

        def __init__(self, config):
            pass

        def start(self, **kw):
            self.started = True

        def close(self):
            self.started = False

        def edit(self, *, uid):
            calls.append(uid)
            if uid == "a":
                self.started = False
                failed.set()
                raise RuntimeError("Boogu transport died")

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Backend)
    with PostMaskWorkerResources(
        case[0], execution, runtime_root=paths.state_root / "runtime"
    ) as resources:

        @contextmanager
        def factory(stage, storage, execution):
            with _accepted_factory([])(stage, storage, execution) as inner:

                class Phase:
                    def run(self, uid=None):
                        if stage == "remove":
                            if uid == "b":
                                assert failed.wait(2)
                            resources.edit(uid=uid)
                        else:
                            downstream.extend(
                                (stage, c.clip_uid)
                                for c in storage.iter_clips()
                                if uid is None or c.clip_uid == uid
                            )
                        return inner.run(uid)

                yield Phase()

        result = api.run_post_mask_shard(
            case[0],
            entity_mask_root=case[1],
            paths=paths,
            git_commit="test",
            execution=execution,
            adapter_factory=factory,
            resources=resources,
        )
        assert resources.summary()["boogu_restart_count"] == 1
        assert resources.summary()["boogu_call_count"] == 2
    assert calls == ["a", "b"]
    assert not result.completed and result.retryable_clip_uids == ("a",)
    assert all(uid == "b" for _, uid in downstream)
    assert ("subject_attributes", "b") in downstream


def test_clip_exception_blocks_export_but_not_neighbour(case):
    _write_rows(
        case,
        [
            _ready(case, "bad", 0, "pending_remove"),
            _ready(case, "good", 1, "pending_remove"),
        ],
    )
    log, events = [], []
    result, paths = _run(case, log, fail=("remove", "bad"), events=events)
    assert not result.completed
    assert ("remove", ("good",)) in log
    assert ("close", "remove") in log
    assert not paths.export_root.exists()
    assert not paths.state_root.joinpath("completed.json").exists()
    assert any(
        e["event"] == "post_mask_clip_failed" and e["clip_uid"] == "bad" for e in events
    )


@pytest.mark.parametrize("failure_mode", ["exception", "pending"])
def test_pair_excludes_predecessor_pending_clip_and_restores_it_on_resume(
    case, failure_mode
):
    from r2v_data_v2.v3.storage import RunStorage

    _write_rows(
        case,
        [_ready(case, "bad", 0, "pending_remove"), _ready(case, "good", 1)],
    )
    api, paths, execution = _setup(case)
    blocked = True
    pairing_targets, donor_views = [], []

    @contextmanager
    def factory(stage, storage, settings):
        with _accepted_factory([])(stage, storage, settings) as ordinary:

            class Phase:
                def run(self, uid=None):
                    if stage == "remove" and uid == "bad" and blocked:
                        if failure_mode == "exception":
                            raise RuntimeError("remove failed before publication")
                        return {"retryable_pending": 1}
                    if stage == "pair":
                        pairing_targets.append(
                            tuple(
                                clip.clip_uid
                                for clip in storage.iter_clips()
                                if clip.pairing is None
                            )
                        )
                    result = ordinary.run(uid)
                    if stage == "pair":
                        # The existing pair fallback builds its donor index from
                        # this same phase storage after publishing first-pass refs.
                        donor_views.append(
                            tuple(
                                clip.clip_uid
                                for clip in storage.iter_clips()
                                if clip.pairing is not None
                                and clip.pairing.status == "ready"
                            )
                        )
                    return result

            yield Phase()

    args = {
        "entity_mask_root": case[1],
        "paths": paths,
        "git_commit": "test",
        "execution": execution,
        "adapter_factory": factory,
    }
    first = api.run_post_mask_shard(case[0], **args)
    assert not first.completed and first.retryable_clip_uids == ("bad",)
    assert pairing_targets == [("good",)]
    assert donor_views == [("good",)]
    from r2v_data_v2.v3.post_mask_production import prepare_shard_config

    stored = RunStorage(prepare_shard_config(case[0], paths))
    assert stored.read_clip("bad").pairing is None
    assert stored.read_clip("good").instruction.status == "ready"

    blocked = False
    second = api.run_post_mask_shard(case[0], **args)
    assert second.completed and second.sample_count == 2
    assert pairing_targets == [("good",), ("bad",)]
    assert donor_views == [("good",), ("bad", "good")]


def test_hydration_corruption_never_enters_phases(case):
    row = _ready(case)
    _write_rows(case, [dict(row, artifact_root="missing")])
    log = []
    result, _ = _run(case, log)
    assert result.completed and result.corrupt == 1 and not log


def test_existing_publication_rejects_changed_dataset_identity(case):
    _write_rows(case, [_ready(case)])
    _, paths = _run(case, [])
    file = paths.export_root / "dataset.json"
    value = json.loads(file.read_text())
    value["config_hash"] = "wrong"
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="export|publication|identity"):
        _run(case, [])


def test_execution_runtime_not_semantic_placeholders(case):
    api, paths, execution = _setup(case)
    runtime = replace(
        case[0].runtime, cpu_workers=3, qwen_max_inflight=7, worker_timeout_seconds=91
    )
    execution = replace(execution, runtime=runtime)
    assert execution.workers_for("instruct") == 2
    assert execution.workers_for("remove") == 8
    assert execution.qwen_gate().maximum == 7
    assert (
        api.ExecutionSettings(
            runtime=runtime,
            sam_gpu="4",
            boogu_gpu="5",
            qwen_lock_directory=paths.state_root / "gate",
        ).runtime.worker_timeout_seconds
        == 91
    )


def test_pair_view_retains_done_donors(case):
    _write_rows(case, [_ready(case, "first", 0), _ready(case, "second", 1)])
    _api_module, paths, _ = _setup(case)
    from r2v_data_v2.v3.post_mask_production import hydrate_shard, initialize_shard

    storage = initialize_shard(case[0], paths, git_commit="old")
    hydrate_shard(storage, entity_mask_root=case[1], shard_path=case[2], paths=paths)
    storage.write_pairing("first", PairingState(status="rejected", reason="quality"))
    log = []
    result, _ = _run(case, log)
    assert result.completed
    assert ("pair", ("first", "second")) in log


def test_owned_scratch_cleanup_removes_reference_edit_requests_only(case):
    api, paths, _ = _setup(case)
    scratch = paths.run_root / ".post_mask_scratch" / "reference_edit"
    scratch.mkdir(parents=True)
    (scratch / ".owner.json").write_text(
        json.dumps({"run_root": str(paths.run_root), "stage": "reference_edit"})
    )
    boogu = scratch / "r2v-boogu-dead"
    boogu.mkdir()
    (boogu / "request.png").write_bytes(b"scratch")
    sam = scratch / "sam3-review-dead"
    sam.mkdir()
    for slot in range(10):
        (sam / f"{slot:02d}.jpg").write_bytes(b"scratch")
    unrelated = scratch / "debug-capture"
    unrelated.mkdir()
    (unrelated / "keep.jpg").write_bytes(b"debug")
    note = scratch / "keep.log"
    note.write_text("keep")
    kept = paths.run_root / "subject_attributes" / "completion_candidates" / "keep.png"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"resume")
    api.cleanup_phase_scratch(paths.run_root, "reference_edit")
    assert not boogu.exists() and not sam.exists()
    assert unrelated.joinpath("keep.jpg").read_bytes() == b"debug"
    assert note.read_text() == "keep" and kept.read_bytes() == b"resume"


def test_unowned_scratch_is_never_removed(case):
    api, paths, _ = _setup(case)
    scratch = paths.run_root / ".post_mask_scratch" / "reference_edit"
    scratch.mkdir(parents=True)
    boogu = scratch / "r2v-boogu-dead"
    boogu.mkdir()
    sam = scratch / "sam3-review-dead"
    sam.mkdir()
    with pytest.raises(ValueError, match="owned|owner"):
        api.cleanup_phase_scratch(paths.run_root, "reference_edit")
    assert boogu.exists() and sam.exists()


def test_owned_sam_review_scratch_symlink_fails_closed(case):
    api, paths, _ = _setup(case)
    scratch = paths.run_root / ".post_mask_scratch" / "reference_edit"
    scratch.mkdir(parents=True)
    (scratch / ".owner.json").write_text(
        json.dumps({"run_root": str(paths.run_root), "stage": "reference_edit"})
    )
    target = paths.run_root / "unowned-sam-review"
    target.mkdir()
    (target / "keep.jpg").write_bytes(b"unowned")
    (scratch / "sam3-review-redirected").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        api.cleanup_phase_scratch(paths.run_root, "reference_edit")
    assert target.joinpath("keep.jpg").read_bytes() == b"unowned"


def _accepted_factory(log, *, terminal_edit_uid=None, retry_stage=None):
    from PIL import Image

    from r2v_data_v2.v3.schemas import (
        BackgroundReferenceState,
        EntityReferenceState,
        ReferenceIntegrityState,
        ReferencesState,
    )
    from tests.test_v3_storage import _instruction_state, _not_required_reference_edit

    @contextmanager
    def factory(stage, storage, execution):
        log.append(stage)

        class Phase:
            def run(self, clip_uid=None):
                clips = (
                    list(storage.iter_clips())
                    if clip_uid is None
                    else [storage.read_clip(clip_uid)]
                )
                for clip in clips:
                    uid = clip.clip_uid
                    if stage == retry_stage:
                        raise RuntimeError("interrupted before durable result")
                    if stage == "remove":
                        storage.write_references(
                            uid,
                            ReferencesState(
                                background=BackgroundReferenceState(
                                    status="rejected", reason="quality"
                                )
                            ),
                        )
                    elif stage == "pair" and clip.pairing is None:
                        image = storage.selected_path(uid, "e1.png")
                        Image.new("RGBA", (10, 10), (30, 40, 50, 255)).save(image)
                        ref = EntityReferenceState(
                            entity_id="e1",
                            status="ready",
                            reference_scope="full",
                            visible_region="whole",
                            whole_entity_recognizable=True,
                            identity_features_visible=True,
                            scope_reason="complete",
                            image_path=storage.relative_artifact_path(image),
                            source_frame_index=0,
                        )
                        storage.write_references(
                            uid,
                            ReferencesState(
                                entities=[ref], background=clip.references.background
                            ),
                        )
                        storage.write_pairing(
                            uid,
                            PairingState(
                                status="ready",
                                retained_entity_ids=["e1"],
                                tokens={"e1": "<ref_subject_1>"},
                            ),
                        )
                    elif stage == "reference_edit":
                        if uid == terminal_edit_uid:
                            storage.write_reference_edit_failure(
                                uid, "durable fail closed"
                            )
                        else:
                            storage.write_reference_edit_result(
                                uid,
                                clip.references,
                                clip.pairing,
                                _not_required_reference_edit(clip),
                            )
                    elif stage == "reference_integrity":
                        storage.write_reference_integrity_result(
                            uid,
                            clip.references,
                            clip.pairing,
                            ReferenceIntegrityState(status="ready", entities=[]),
                        )
                    elif stage == "instruct":
                        storage.write_instruction(uid, _instruction_state())
                    elif stage == "subject_attributes":
                        # Real sidecar algorithm sees no usable owner candidate in
                        # this deliberately undecodable Stage2 fixture; no model.
                        adapter = _api().DownstreamPhaseAdapter(
                            stage, storage, execution
                        )
                        adapter.kwargs = {
                            "discovery_client": object(),
                            "review_client": object(),
                            "segmentation_backend": object(),
                        }
                        return adapter.run(uid)
                return {"processed": len(clips)}

        yield Phase()

    return factory


def _all_phases_case(case):
    config, root, shard = case
    config = replace(
        config,
        reference_edit=replace(
            config.reference_edit,
            enabled=True,
            python_executable=config.run_root.parent / "python",
            code_root=config.run_root.parent / "boogu",
            model_path=config.run_root.parent.parent / "models/boogu",
        ),
        reference_integrity=replace(config.reference_integrity, enabled=True),
        qwen=replace(
            config.qwen,
            reference_edit_judge=config.qwen.candidate_judge,
            reference_integrity_judge=config.qwen.candidate_judge,
        ),
    )
    return config, root, shard


def test_all_phases_order_and_each_durable_phase_skipped_after_interruption(case):
    case = _all_phases_case(case)
    _write_rows(case, [_ready(case, background="pending_remove")])
    api, paths, execution = _setup(case)
    log = []
    args = {
        "entity_mask_root": case[1],
        "paths": paths,
        "git_commit": "test",
        "execution": execution,
    }
    result = api.run_post_mask_shard(
        case[0], **args, adapter_factory=_accepted_factory(log, retry_stage="instruct")
    )
    assert not result.completed
    assert log == [
        "remove",
        "pair",
        "reference_edit",
        "reference_integrity",
        "instruct",
    ]
    log.clear()
    result = api.run_post_mask_shard(
        case[0], **args, adapter_factory=_accepted_factory(log)
    )
    assert result.completed and result.sample_count == 1
    assert log == ["instruct", "subject_attributes"]


def test_durable_failed_clip_is_excluded_not_permanent_shard_blocker(case):
    case = _all_phases_case(case)
    _write_rows(case, [_ready(case, "bad", 0), _ready(case, "good", 1)])
    api, paths, execution = _setup(case)
    events = []
    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        event_callback=events.append,
        adapter_factory=_accepted_factory([], terminal_edit_uid="bad"),
    )
    assert result.completed and result.sample_count == 1
    samples = [
        json.loads(line)
        for line in (paths.export_root / "samples.jsonl").read_text().splitlines()
    ]
    assert [s["sample_id"] for s in samples] == ["good"]
    assert any(
        e["event"] == "post_mask_clip_failed"
        and e["clip_uid"] == "bad"
        and e["retryable"] is False
        for e in events
    )


def test_real_export_compactor_and_h3_reader(case):
    import yaml

    from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory
    from r2v_data_v2.v3.production_source import JEA_VIDEO_MOTION_ADAPTER
    from tools.compact_v3_production_exports import compact_production_exports

    row = _ready(case)
    frozen_clip = case[1] / row["artifact_root"] / "run/clips/clip-0/clip.json"
    value = json.loads(frozen_clip.read_text())
    value["source"]["metadata"] = {
        "source_relative_video_path": "movies/show/clip-0.mp4",
        "source_relative_source_video_path": "movies/show/parent.mp4",
    }
    frozen_clip.write_text(json.dumps(value))
    _write_rows(case, [row])
    api, paths, execution = _setup(case)
    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=_accepted_factory([]),
    )
    assert result.completed and result.sample_count == 1
    writable = case[0].run_root.parent.parent
    source = writable / "r2v_v3_configs/production/campaign/source.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
                "source_jsonl": str(case[0].dataset_json),
                "base_config_path": "/fixture.yaml",
                "base_config_sha256": "fixture",
                "base_config_fingerprint": case[0].fingerprint(),
                "clips_root": str(case[0].dataset_json.parent / "videos"),
            }
        )
    )
    compact_production_exports(
        shards_root=paths.export_root.parent,
        output_root=paths.export_root.parent.parent,
        source_jsonl=case[0].dataset_json,
        source_yaml=source,
        runs_root=paths.run_root.parent,
    )
    inventory = load_visual_production_inventory(
        visual_production_root=paths.export_root.parent.parent,
        visual_runs_root=paths.run_root.parent,
    )
    assert inventory.visual_input_mode == "compacted_production"
    assert (
        len(inventory.clips) == 1 and inventory.clips[0].identity.clip_uid == "clip-0"
    )


def test_abandoned_export_staging_only_owned_shard_is_cleaned(case):
    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)
    # A killed exporter had written its identity intent before staging output.
    from r2v_data_v2.v3.post_mask_production import hydrate_shard, initialize_shard

    storage = initialize_shard(case[0], paths, git_commit="test")
    hydrate_shard(storage, entity_mask_root=case[1], shard_path=case[2], paths=paths)
    identity = api._export_identity(storage, paths)
    (paths.state_root / "export_identity.json").write_text(json.dumps(identity))
    own = paths.export_root.with_name(f".{paths.export_root.name}.tmp-" + "a" * 32)
    other = paths.export_root.with_name(".shard-000000100-000000199.tmp-" + "b" * 32)
    own.mkdir(parents=True)
    other.mkdir()
    api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=_fake_factory([]),
    )
    assert not own.exists() and other.exists()


@pytest.mark.parametrize("stage", ["remove", "reference_edit", "subject_attributes"])
@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("scheduler_mode", ["legacy_serial", "wavefront_v2", "parallel_review_v21"])
def test_real_adapter_persistent_resources_placement_and_exception_close(
    case, monkeypatch, stage, fail_second, scheduler_mode
):
    from types import SimpleNamespace

    from r2v_data_v2.v3 import reference_edit_boogu, sam3_backend, subject_attributes
    from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND
    from r2v_data_v2.v3.post_mask_production import hydrate_shard, initialize_shard

    config, root, shard = _all_phases_case(case)
    config = replace(
        config,
        remove=replace(
            config.remove,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
        ),
        subject_attributes=replace(
            config.subject_attributes,
            completion=replace(config.subject_attributes.completion, enabled=True),
        ),
    )
    case = config, root, shard
    _write_rows(case, [_ready(case, "one", 0), _ready(case, "two", 1)])
    api, paths, execution = _setup(case)
    execution = replace(
        execution, runtime=replace(execution.runtime, worker_timeout_seconds=73),
        scheduler_mode=scheduler_mode,
    )
    storage = initialize_shard(config, paths, git_commit="test")
    hydrate_shard(storage, entity_mask_root=root, shard_path=shard, paths=paths)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    resources, calls = [], []
    scratch = storage.root / ".post_mask_scratch" / stage
    restart_sam = scratch / "sam3-review-restart"
    unrelated_scratch = scratch / "debug-capture"
    if stage == "reference_edit":
        scratch.mkdir(parents=True)
        (scratch / ".owner.json").write_text(
            json.dumps({"run_root": str(storage.root), "stage": stage})
        )
        restart_sam.mkdir()
        (restart_sam / "00.jpg").write_bytes(b"abandoned")
        unrelated_scratch.mkdir()
        (unrelated_scratch / "keep.jpg").write_bytes(b"debug")

    class Resource:
        def __init__(self, config, **kwargs):
            self.config, self.kwargs, self.starts, self.closes = config, kwargs, 0, 0
            self.started = False
            resources.append(self)

        def start(self, **kwargs):
            self.starts += 1
            self.started = True

        def edit(self, **kwargs):
            return None

        def close(self):
            self.closes += 1
            self.started = False

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Resource)
    monkeypatch.setattr(reference_edit_boogu, "QwenBooguReferenceEditJudge", Resource)
    monkeypatch.setattr(sam3_backend, "Sam3SegmentationBackend", Resource)
    # The lightweight attribute adapter borrows, and must not own/close, SAM.
    monkeypatch.setattr(
        subject_attributes,
        "Sam3AttributeFrameSegmenter",
        lambda config, *, backend: backend,
    )
    monkeypatch.setattr(subject_attributes, "QwenSubjectAttributeClient", Resource)
    monkeypatch.setattr(
        subject_attributes, "QwenSubjectAttributeCompletionJudge", Resource
    )
    removal_judge = importlib.import_module("r2v_data_v2.v3.removal_judge")
    monkeypatch.setattr(removal_judge, "QwenBackgroundRemovalJudge", Resource)

    def algorithm(config, scoped=None, **kwargs):
        assert config is storage.config
        assert kwargs["overwrite"] is False
        calls.append(kwargs)
        if stage == "remove":
            kwargs["backend"].backend.edit()
        elif stage == "reference_edit":
            kwargs["backend"].edit()
        else:
            kwargs["completion_backend"].backend.edit()
        if stage == "reference_edit":
            assert not restart_sam.exists()
            assert kwargs.get("review_execution", "sequential") == (
                "parallel_independent" if scheduler_mode == "parallel_review_v21"
                else "sequential"
            )
            assert callable(kwargs.get("review_observer")) == (
                scheduler_mode == "parallel_review_v21"
            )
            assert kwargs["manage_backend_lifecycle"] is False
            assert (
                kwargs["sam_reviewer"].max_area_growth_ratio
                == config.reference_edit.sam_max_area_growth_ratio
            )
            final_sam = scratch / "sam3-review-final"
            final_sam.mkdir(exist_ok=True)
            (final_sam / "00.jpg").write_bytes(b"abandoned")
        if len(calls) == 2 and fail_second:
            raise RuntimeError("injected algorithm failure")
        return SimpleNamespace(
            to_dict=lambda: {"processed": 1}, to_counts=lambda: {"failures": 0}
        )

    module_name, function_name = {
        "remove": ("remove", "remove_backgrounds"),
        "reference_edit": ("reference_edit", "reference_edit_clips"),
        "subject_attributes": ("subject_attributes", "process_subject_attribute_clip"),
    }[stage]
    monkeypatch.setattr(
        importlib.import_module(f"r2v_data_v2.v3.{module_name}"),
        function_name,
        algorithm,
    )
    try:
        with api.DownstreamPhaseAdapter(stage, storage, execution) as adapter:
            adapter.run("one")
            adapter.run("two")
    except RuntimeError as exc:
        assert fail_second and str(exc) == "injected algorithm failure"
    assert len(calls) == 2
    boogu = [r for r in resources if hasattr(r.config, "cuda_visible_devices")]
    assert len(boogu) == 1 and boogu[0].starts == 1
    assert boogu[0].config.cuda_visible_devices == "5"
    assert boogu[0].config.timeout_seconds == 73
    assert all(r.closes == 1 for r in resources)
    sam = [r for r in resources if r.config is storage.config.sam3]
    if stage != "remove":
        assert len(sam) == 1 and sam[0].config.device == "cuda"
    if stage == "reference_edit":
        assert not restart_sam.exists() and not (scratch / "sam3-review-final").exists()
        assert unrelated_scratch.joinpath("keep.jpg").read_bytes() == b"debug"
    assert storage.config.runtime.worker_timeout_seconds == 3600


def test_adapter_closes_boogu_when_start_raises(case, monkeypatch):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND
    from r2v_data_v2.v3.post_mask_production import initialize_shard

    config, root, shard = _all_phases_case(case)
    config = replace(
        config,
        remove=replace(
            config.remove,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
        ),
    )
    case = config, root, shard
    _write_rows(case, [])
    api, paths, execution = _setup(case)
    storage = initialize_shard(config, paths, git_commit="test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    closed = []

    class Backend:
        def __init__(self, config):
            self.config = config
            self.started = False

        def start(self, **kwargs):
            raise RuntimeError("worker startup failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Backend)
    with (
        pytest.raises(RuntimeError, match="startup"),
        api.DownstreamPhaseAdapter("remove", storage, execution) as adapter,
    ):
        adapter._boogu().edit()
    assert closed == [True]


def test_attribute_completion_exact_policy_bytes_and_run_local_output(case):
    from io import BytesIO
    from types import SimpleNamespace

    from PIL import Image

    from r2v_data_v2.v3.post_mask_production import initialize_shard

    api, paths, _ = _setup(case)
    _write_rows(case, [])
    storage = initialize_shard(case[0], paths, git_commit="test")
    source = storage.root / "subject_attributes/references/input.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (96, 64), (50, 60, 70)).save(source)
    stream = BytesIO()
    Image.new("RGB", (32, 32)).save(stream, format="PNG")
    calls = []

    class Backend:
        def edit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(png_bytes=stream.getvalue())

    adapter = api._AttributeCompletion(Backend(), storage)
    output = storage.root / "subject_attributes/completion_candidates/a/out.png"
    result = adapter.attribute_completion(
        source_path=source, output_path=output, instruction="Keep exact prompt", seed=17
    )
    assert output.read_bytes() == stream.getvalue()
    assert calls[0]["instruction"] == "Keep exact prompt" and calls[0]["seed"] == 17
    assert (
        calls[0]["thinking_enabled"] is False
        and calls[0]["instruction_rewrite_enabled"] is False
    )
    assert (
        result["thinking_enabled"] is False
        and result["instruction_rewrite_enabled"] is False
    )
    with pytest.raises(ValueError):
        adapter.attribute_completion(
            source_path=source,
            output_path=storage.root / "outside.png",
            instruction="same",
            seed=0,
        )
    assert len(calls) == 1


def test_instruction_adapter_preserves_existing_deterministic_policy(case):
    from r2v_data_v2.v3.post_mask_production import hydrate_shard, initialize_shard

    row = _ready(case)
    clip_path = case[1] / row["artifact_root"] / "run/clips/clip-0/clip.json"
    clip = json.loads(clip_path.read_text())
    clip["annotation"]["t2v_caption"] = ""
    clip["annotation"]["instruction_template"] = "{{entity_1}} walks."
    clip_path.write_text(json.dumps(clip))
    _write_rows(case, [row])
    api, paths, execution = _setup(case)
    storage = initialize_shard(case[0], paths, git_commit="test")
    hydration = hydrate_shard(
        storage, entity_mask_root=case[1], shard_path=case[2], paths=paths
    )
    assert hydration.ready == 1
    with _accepted_factory([])("pair", storage, execution) as pair:
        pair.run()
    with api.DownstreamPhaseAdapter("instruct", storage, execution) as adapter:
        counts = adapter.run("clip-0")
    assert counts["processed"] == 1
    assert storage.read_clip("clip-0").instruction.status == "ready"


def test_qwen_gate_propagates_into_worker_threads(case):
    import threading
    import time

    from r2v_data_v2.v3.profiling import profiled_openai_call

    _write_rows(case, [_ready(case, "a", 0), _ready(case, "b", 1)])
    api, paths, execution = _setup(case)
    execution = replace(
        execution, runtime=replace(execution.runtime, qwen_max_inflight=1)
    )
    lock = threading.Lock()
    active, peak, threads = 0, 0, set()

    @contextmanager
    def factory(stage, storage, settings):
        with _accepted_factory([])(stage, storage, settings) as ordinary:

            class Phase:
                def run(self, uid=None):
                    if stage == "instruct":

                        def request():
                            nonlocal active, peak
                            with lock:
                                active += 1
                                peak = max(peak, active)
                                threads.add(threading.get_ident())
                            time.sleep(0.03)
                            with lock:
                                active -= 1

                        profiled_openai_call(
                            request,
                            component="qwen_test",
                            operation="test",
                            retry_index=0,
                            model="fake",
                            messages=[],
                        )
                    return ordinary.run(uid)

            yield Phase()

    result = api.run_post_mask_shard(
        case[0],
        entity_mask_root=case[1],
        paths=paths,
        git_commit="test",
        execution=execution,
        adapter_factory=factory,
    )
    assert result.completed and peak == 1 and len(threads) == 2


def test_changed_published_instruction_fails_provenance_check(case):
    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)
    kwargs = {
        "entity_mask_root": case[1],
        "paths": paths,
        "git_commit": "test",
        "execution": execution,
        "adapter_factory": _accepted_factory([]),
    }
    api.run_post_mask_shard(case[0], **kwargs)
    samples = paths.export_root / "samples.jsonl"
    value = json.loads(samples.read_text())
    value["t2v_caption"] = "A completely different source scene."
    samples.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="provenance|source"):
        api.run_post_mask_shard(case[0], **kwargs)


def test_downstream_module_import_never_loads_upstream_stage_or_launcher():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import r2v_data_v2.v3.post_mask_runtime; assert not any('pre_qwen' in n or n.endswith('.segment') or n.endswith('.annotation') or n.endswith('run_pipeline_v3') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_completed_publication_resume_never_initializes_real_adapter(case, monkeypatch):
    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)
    args = {
        "entity_mask_root": case[1],
        "paths": paths,
        "git_commit": "test",
        "execution": execution,
    }
    api.run_post_mask_shard(case[0], **args, adapter_factory=_accepted_factory([]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # No factory injection on resume: any real GPU phase would reject missing
    # isolation before it could load a model.
    result = api.run_post_mask_shard(case[0], **args)
    assert result.completed and result.stage_counts == {}


def test_completed_marker_binds_source_digest_without_copying_source_payload(case):
    row = _ready(case)
    clip_path = case[1] / row["artifact_root"] / "run/clips/clip-0/clip.json"
    value = json.loads(clip_path.read_text())
    value["source"]["metadata"]["large_transcript"] = "large-source-transcript " * 10000
    clip_path.write_text(json.dumps(value))
    _write_rows(case, [row])
    _, paths = _run(case, [])
    marker = (paths.state_root / "completed.json").read_text()
    identity = json.loads(marker)["identity"]
    assert identity["clip_count"] == 1
    assert len(identity["clip_sources_sha256"]) == 64
    assert "large-source-transcript" not in marker and len(marker) < 4096


def test_export_failure_has_no_marker_and_completed_attributes_skip_on_retry(
    case, monkeypatch
):
    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)
    args = {
        "entity_mask_root": case[1],
        "paths": paths,
        "git_commit": "test",
        "execution": execution,
    }
    original = api.DatasetExporter.export

    def fail_export(self, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(api.DatasetExporter, "export", fail_export)
    with pytest.raises(OSError, match="publication"):
        api.run_post_mask_shard(case[0], **args, adapter_factory=_accepted_factory([]))
    assert not (paths.state_root / "completed.json").exists()
    monkeypatch.setattr(api.DatasetExporter, "export", original)
    events = []
    result = api.run_post_mask_shard(case[0], **args, event_callback=events.append)
    assert (
        result.completed and result.stage_counts["subject_attributes"]["scheduled"] == 0
    )
    assert any(
        e["event"] == "post_mask_stage_completed" and e["stage"] == "export"
        for e in events
    )


def test_execution_rejects_gate_directory_outside_writable_root(case):
    _api_module, _, execution = _setup(case)
    with pytest.raises(ValueError, match="writable"):
        replace(execution, qwen_lock_directory=case[1] / "outside-gate")


@pytest.mark.parametrize("durable", [False, True])
def test_attribute_failed_owner_durable_policy_versus_retryable_gap(
    case, monkeypatch, durable
):
    from types import SimpleNamespace

    from r2v_data_v2.v3 import subject_attributes
    from r2v_data_v2.v3.post_mask_production import hydrate_shard, initialize_shard

    _write_rows(case, [_ready(case)])
    api, paths, execution = _setup(case)
    storage = initialize_shard(case[0], paths, git_commit="test")
    hydrate_shard(storage, entity_mask_root=case[1], shard_path=case[2], paths=paths)
    with _accepted_factory([])("pair", storage, execution) as pair:
        pair.run()
    with _accepted_factory([])("instruct", storage, execution) as instruct:
        instruct.run("clip-0")
    root = storage.root / "subject_attributes"
    artifact = subject_attributes.OwnerEnrichmentArtifact(
        sample_id="clip-0",
        owner_entity_id="e1",
        owner_phrase="a woman",
        owner_grounding_prompt="a woman",
        attribute_id_start=1,
        owner_is_human=None,
        records=[],
        metrics=subject_attributes.OwnerEnrichmentMetrics(
            discovery_calls=1, failures=1
        ),
        failure_reason="discovery_failed:TimeoutError",
    )
    if durable:
        file = root / "owners/clip-0/e1.json"
        file.parent.mkdir(parents=True)
        file.write_text(artifact.model_dump_json())
    monkeypatch.setattr(
        subject_attributes,
        "process_subject_attribute_clip",
        lambda *args, **kwargs: SimpleNamespace(to_counts=lambda: {"failures": 1}),
    )
    adapter = api.DownstreamPhaseAdapter("subject_attributes", storage, execution)
    adapter.kwargs = {
        "discovery_client": object(),
        "review_client": object(),
        "segmentation_backend": object(),
    }
    counts = adapter.run("clip-0")
    assert counts["retryable_pending"] == (0 if durable else 1)
    assert (
        api.phase_needed(storage, storage.read_clip("clip-0"), "subject_attributes")
        is not durable
    )
