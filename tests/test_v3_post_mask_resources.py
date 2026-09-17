from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from r2v_data_v2.v3.config import RuntimeConfig, V3Config
from r2v_data_v2.v3.post_mask_runtime import ExecutionSettings


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExecutionSettings:
    from r2v_data_v2.v3 import config as config_module

    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", tmp_path)
    return ExecutionSettings(
        runtime=replace(
            RuntimeConfig(),
            qwen_max_inflight=7,
            worker_timeout_seconds=73,
        ),
        sam_gpu="4",
        boogu_gpu="5",
        qwen_lock_directory=tmp_path / "qwen-locks",
        clip_inflight=6,
    )


def _config(tmp_path: Path) -> V3Config:
    config = V3Config(
        dataset_json=tmp_path / "dataset.json",
        run_root=tmp_path / "run",
        export_root=tmp_path / "export",
    )
    return replace(
        config,
        reference_edit=replace(
            config.reference_edit,
            python_executable=tmp_path / "boogu-python",
            code_root=tmp_path / "boogu-code",
            model_path=tmp_path / "boogu-model",
            model_revision="test-revision",
            timeout_seconds=97,
        ),
    )


class _HealthyBackend:
    instances: ClassVar[list[_HealthyBackend]] = []

    def __init__(self, config):
        self.config = config
        self.started = False
        self.start_paths: list[Path] = []
        self.calls: list[dict[str, object]] = []
        self.close_count = 0
        type(self).instances.append(self)

    def start(self, *, stderr_log_path: Path) -> None:
        self.started = True
        self.start_paths.append(stderr_log_path)

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs

    def close(self) -> None:
        self.close_count += 1
        self.started = False


def test_lazy_boogu_reuses_exact_worker_configuration_and_closes_once(
    tmp_path, monkeypatch
):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    _HealthyBackend.instances = []
    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", _HealthyBackend)
    config = _config(tmp_path)
    execution = _settings(tmp_path, monkeypatch)
    runtime_root = tmp_path / "campaign-state" / "worker-0"

    resources = PostMaskWorkerResources(config, execution, runtime_root=runtime_root)
    assert resources.summary() == {
        "clip_inflight": 6,
        "boogu_start_count": 0,
        "boogu_restart_count": 0,
        "boogu_call_count": 0,
        "boogu_queue_wait_seconds": 0.0,
        "boogu_service_seconds": 0.0,
        "boogu_max_inflight_observed": 0,
        "sam_call_count": 0,
        "sam_queue_wait_seconds": 0.0,
        "sam_service_seconds": 0.0,
        "sam_max_inflight_observed": 0,
        "qwen_max_inflight_configured": 7,
    }
    assert not runtime_root.exists() and not _HealthyBackend.instances

    sent = [object() for _ in range(5)]
    with resources:
        # These represent remove, reference edit, subject attributes, and a
        # later shard. The owner deliberately has no stage/shard boundary.
        for index, marker in enumerate(sent):
            assert resources.edit(marker=marker, index=index) == {
                "marker": marker,
                "index": index,
            }

    assert len(_HealthyBackend.instances) == 1
    backend = _HealthyBackend.instances[0]
    assert backend.calls == [
        {"marker": marker, "index": index} for index, marker in enumerate(sent)
    ]
    assert backend.close_count == 1
    assert backend.start_paths == [runtime_root / "boogu.log"]
    assert backend.config.python_executable == config.reference_edit.python_executable
    assert backend.config.code_root == config.reference_edit.code_root
    assert backend.config.model_path == config.reference_edit.model_path
    assert backend.config.model_revision == "test-revision"
    assert backend.config.device == "cuda:0"
    assert backend.config.cuda_visible_devices == "5"
    assert backend.config.timeout_seconds == 73
    assert backend.config.temporary_root == runtime_root / "boogu-scratch"
    resources.close()
    assert backend.close_count == 1
    summary = resources.summary()
    assert summary["boogu_start_count"] == 1
    assert summary["boogu_restart_count"] == 0
    assert summary["boogu_call_count"] == 5
    assert summary["boogu_max_inflight_observed"] == 1


def test_boogu_serializes_actual_edits_but_not_following_qwen_wait(
    tmp_path, monkeypatch
):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    first_edit_entered = threading.Event()
    release_first_edit = threading.Event()
    first_waiting_in_qwen = threading.Event()
    release_qwen = threading.Event()
    second_edit_entered = threading.Event()
    state_lock = threading.Lock()
    active = 0
    peak = 0

    class BlockingBackend(_HealthyBackend):
        instances: ClassVar[list[_HealthyBackend]] = []

        def edit(self, **kwargs):
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            try:
                if kwargs["clip"] == "a":
                    first_edit_entered.set()
                    assert release_first_edit.wait(2)
                else:
                    second_edit_entered.set()
                return kwargs["clip"]
            finally:
                with state_lock:
                    active -= 1

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", BlockingBackend)
    resources = PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=tmp_path / "runtime",
    )

    def clip_a():
        assert resources.edit(clip="a") == "a"
        first_waiting_in_qwen.set()
        assert release_qwen.wait(2)

    def clip_b():
        assert resources.edit(clip="b") == "b"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(clip_a)
        assert first_edit_entered.wait(2)
        second = pool.submit(clip_b)
        assert not second_edit_entered.is_set()
        release_first_edit.set()
        assert first_waiting_in_qwen.wait(2)
        assert second_edit_entered.wait(2)
        # Clip B used Boogu while clip A remained in a Qwen-like wait.
        assert not first.done()
        release_qwen.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert peak == 1
    summary = resources.summary()
    assert summary["boogu_call_count"] == 2
    assert summary["boogu_max_inflight_observed"] == 1
    assert summary["boogu_queue_wait_seconds"] >= 0.0
    assert summary["boogu_service_seconds"] >= 0.0
    resources.close()


def test_crashed_boogu_request_is_not_retried_and_next_call_restarts(
    tmp_path, monkeypatch
):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    class CrashThenRecover(_HealthyBackend):
        instances: ClassVar[list[_HealthyBackend]] = []

        def edit(self, **kwargs):
            self.calls.append(kwargs)
            if len(type(self).instances) == 1:
                self.started = False
                raise RuntimeError("transport died")
            return "recovered"

    monkeypatch.setattr(
        reference_edit_boogu, "BooguSubprocessBackend", CrashThenRecover
    )
    resources = PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=tmp_path / "runtime",
    )

    with pytest.raises(RuntimeError, match="transport died"):
        resources.edit(request="first")
    assert len(CrashThenRecover.instances) == 1
    assert CrashThenRecover.instances[0].calls == [{"request": "first"}]
    assert resources.edit(request="later") == "recovered"

    assert len(CrashThenRecover.instances) == 2
    assert CrashThenRecover.instances[1].calls == [{"request": "later"}]
    summary = resources.summary()
    assert summary["boogu_start_count"] == 2
    assert summary["boogu_restart_count"] == 1
    assert summary["boogu_call_count"] == 2
    resources.close()
    assert CrashThenRecover.instances[0].close_count == 0
    assert CrashThenRecover.instances[1].close_count == 1


def test_healthy_boogu_rejection_does_not_restart_backend(tmp_path, monkeypatch):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    class RejectOnce(_HealthyBackend):
        instances: ClassVar[list[_HealthyBackend]] = []

        def edit(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("model rejected request")
            return "accepted"

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", RejectOnce)
    resources = PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=tmp_path / "runtime",
    )
    with pytest.raises(RuntimeError, match="model rejected"):
        resources.edit(request=1)
    assert resources.edit(request=2) == "accepted"
    assert len(RejectOnce.instances) == 1
    assert resources.summary()["boogu_restart_count"] == 0
    resources.close()


def test_sam_call_serializes_only_the_actual_function_and_releases_on_error(
    tmp_path, monkeypatch
):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources
    from r2v_data_v2.v3.post_mask_runtime import _SerializedSam

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", _HealthyBackend)

    resources = PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=tmp_path / "runtime",
    )
    first_entered = threading.Event()
    first_release = threading.Event()
    second_entered = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def inference(label, *, fail=False):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if label == "first":
                first_entered.set()
                assert first_release.wait(2)
            else:
                second_entered.set()
            if fail:
                raise RuntimeError("SAM failed")
            return label
        finally:
            with lock:
                active -= 1

    sampler = _SerializedSam(
        SimpleNamespace(track=inference, segment_frame=inference), resources
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(sampler.track, label="first", fail=True)
        assert first_entered.wait(2)
        second = pool.submit(sampler.segment_frame, label="second")
        assert not second_entered.is_set()
        # A different clip's Boogu edit progresses while actual SAM is blocked.
        assert resources.edit(request="other clip") == {"request": "other clip"}
        first_release.set()
        with pytest.raises(RuntimeError, match="SAM failed"):
            first.result(timeout=2)
        assert second.result(timeout=2) == "second"

    assert peak == 1
    summary = resources.summary()
    assert summary["sam_call_count"] == 2
    assert summary["sam_max_inflight_observed"] == 1
    assert summary["sam_queue_wait_seconds"] >= 0.0
    assert summary["sam_service_seconds"] >= 0.0
    resources.close()


def test_interrupted_lazy_start_closes_unpublished_backend(tmp_path, monkeypatch):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    class Interrupted(_HealthyBackend):
        instances: ClassVar[list] = []

        def start(self, **kwargs):
            super().start(**kwargs)
            raise KeyboardInterrupt("startup interrupted")

    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", Interrupted)
    with (
        pytest.raises(KeyboardInterrupt),
        PostMaskWorkerResources(
            _config(tmp_path),
            _settings(tmp_path, monkeypatch),
            runtime_root=tmp_path / "runtime",
        ) as resources,
    ):
        resources.edit(request=1)
    assert len(Interrupted.instances) == 1
    assert Interrupted.instances[0].close_count == 1


def test_stop_accepting_prevents_resource_waiters_and_restart(tmp_path, monkeypatch):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    _HealthyBackend.instances = []
    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", _HealthyBackend)
    with PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=tmp_path / "runtime",
    ) as resources:
        resources.edit(request=1)
        waiting = threading.Event()

        def next_request():
            waiting.set()
            resources.edit(request=2)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with resources._boogu_lock:
                future = pool.submit(next_request)
                assert waiting.wait(2)
                resources.stop_accepting()
            with pytest.raises(RuntimeError, match="closed"):
                future.result(timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            resources.sam_call(lambda: pytest.fail("new SAM after stop"))
    assert len(_HealthyBackend.instances) == 1
    assert _HealthyBackend.instances[0].calls == [{"request": 1}]
    assert _HealthyBackend.instances[0].close_count == 1


def test_close_before_first_use_does_not_create_resources(tmp_path, monkeypatch):
    from r2v_data_v2.v3 import reference_edit_boogu
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    _HealthyBackend.instances = []
    monkeypatch.setattr(reference_edit_boogu, "BooguSubprocessBackend", _HealthyBackend)
    runtime_root = tmp_path / "runtime"
    resources = PostMaskWorkerResources(
        _config(tmp_path),
        _settings(tmp_path, monkeypatch),
        runtime_root=runtime_root,
    )
    resources.close()
    assert not runtime_root.exists()
    assert not _HealthyBackend.instances
    with pytest.raises(RuntimeError, match="closed"):
        resources.edit(unused=SimpleNamespace())
