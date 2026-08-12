from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import (
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    RuntimeConfig,
    RuntimeGpuWorkersConfig,
    RuntimeStageWorkersConfig,
    SourceConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.profiling import (
    QwenConcurrencyGate,
    V3Profiler,
    profiled_openai_call,
    qwen_concurrency_gate,
)
from r2v_data_v2.v3.runtime import (
    PersistentStageProcess,
    StageWorkerConfig,
    StreamingDAGScheduler,
    StreamingStage,
    streaming_model_stage_enabled,
)
from r2v_data_v2.v3.storage import _append_jsonl


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> V3Config:
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
    source = dataset_root / "source.jsonl"
    source.write_text("", encoding="utf-8")
    qwen_model = str(pretrained / "Qwen" / "judge")
    config = V3Config(
        dataset_json=source,
        run_root=writable / "runs" / "runtime",
        export_root=writable / "datasets" / "runtime",
        source=SourceConfig(limit=3),
        qwen=QwenServicesConfig(
            candidate_judge=QwenServiceConfig(model=qwen_model),
            background_remove_judge=QwenServiceConfig(model=qwen_model),
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "object-remover",
        ),
    )
    config.validate()
    return config


def test_runtime_defaults_preserve_legacy_staged_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    assert config.runtime.mode == "staged_legacy"
    assert config.runtime.qwen_max_inflight == 2
    assert config.runtime.stage_workers.segment == 1
    assert config.runtime.stage_workers.remove == 1
    assert config.runtime.stage_workers.reference_edit == 1


def test_disabled_streaming_model_stage_does_not_require_worker_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    assert streaming_model_stage_enabled(config, "remove") is True
    assert streaming_model_stage_enabled(config, "reference_edit") is False


def test_runtime_selection_enters_config_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        runtime=replace(config.runtime, cpu_workers=config.runtime.cpu_workers + 1),
    )

    assert config.fingerprint() != changed.fingerprint()


def test_streaming_rejects_same_parent_cross_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="does not support"):
        replace(
            config,
            runtime=RuntimeConfig(mode="streaming_v1"),
            pair=replace(config.pair, same_parent_fallback_enabled=True),
            qwen=replace(
                config.qwen,
                cross_pair_judge=QwenServiceConfig(
                    model=config.qwen.candidate_judge.model
                ),
            ),
        ).validate()


@pytest.mark.parametrize("stage", ("segment", "remove", "reference_edit"))
def test_gpu_model_worker_count_must_remain_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    workers = replace(RuntimeStageWorkersConfig(), **{stage: 2})

    with pytest.raises(ValueError, match="duplicate GPU model copies"):
        replace(config, runtime=RuntimeConfig(stage_workers=workers)).validate()


def test_runtime_yaml_loads_nested_worker_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    yaml_path = tmp_path / "runtime.yaml"
    yaml_path.write_text(
        "\n".join(
            (
                f"dataset_json: {config.dataset_json}",
                f"run_root: {config.run_root}",
                f"export_root: {config.export_root}",
                "source:",
                "  limit: 3",
                "qwen:",
                "  candidate_judge:",
                f"    model: {config.qwen.candidate_judge.model}",
                "  background_remove_judge:",
                f"    model: {config.qwen.background_remove_judge.model}",
                "remove:",
                f"  base_model_path: {config.remove.base_model_path}",
                f"  adapter_path: {config.remove.adapter_path}",
                "runtime:",
                "  mode: streaming_v1",
                "  qwen_max_inflight: 2",
                "  cpu_workers: 6",
                "  stage_workers:",
                "    frames: 3",
                "  gpu_workers:",
                "    segment: '5'",
                "    remove: '4'",
                "    reference_edit: '6'",
            )
        ),
        encoding="utf-8",
    )

    loaded = load_config(yaml_path)

    assert loaded.runtime.mode == "streaming_v1"
    assert loaded.runtime.cpu_workers == 6
    assert loaded.runtime.stage_workers.frames == 3
    assert loaded.runtime.gpu_workers == RuntimeGpuWorkersConfig(
        segment="5", remove="4", reference_edit="6"
    )


def test_dag_overlaps_different_clips_but_never_same_clip_writers() -> None:
    active_by_clip: dict[str, int] = {}
    maximum_by_clip: dict[str, int] = {}
    state_lock = threading.Lock()
    slow_first_started = threading.Event()
    second_stage_seen = threading.Event()
    ordering: dict[str, list[str]] = {"a": [], "b": []}

    def enter(clip_uid: str, stage: str) -> None:
        with state_lock:
            active_by_clip[clip_uid] = active_by_clip.get(clip_uid, 0) + 1
            maximum_by_clip[clip_uid] = max(
                maximum_by_clip.get(clip_uid, 0), active_by_clip[clip_uid]
            )
            ordering[clip_uid].append(stage)

    def leave(clip_uid: str) -> None:
        with state_lock:
            active_by_clip[clip_uid] -= 1

    def first(clip_uid: str) -> dict[str, int]:
        enter(clip_uid, "first")
        try:
            if clip_uid == "b":
                slow_first_started.set()
                assert second_stage_seen.wait(2)
            return {"processed": 1}
        finally:
            leave(clip_uid)

    def second(clip_uid: str) -> dict[str, int]:
        enter(clip_uid, "second")
        try:
            if clip_uid == "a":
                assert slow_first_started.wait(2)
                second_stage_seen.set()
            return {"processed": 1}
        finally:
            leave(clip_uid)

    result = StreamingDAGScheduler(
        [
            StreamingStage("first", 2, first),
            StreamingStage("second", 2, second),
        ],
        cpu_workers=4,
    ).run(["b", "a"])

    assert maximum_by_clip == {"a": 1, "b": 1}
    assert ordering == {"a": ["first", "second"], "b": ["first", "second"]}
    assert result.completion_order == ["a", "b"]
    assert result.stage_counts["first"]["processed"] == 2


def test_global_qwen_budget_is_shared_by_concurrent_stage_tasks(
    tmp_path: Path,
) -> None:
    current = 0
    maximum = 0
    lock = threading.Lock()

    def request() -> object:
        nonlocal current, maximum
        with lock:
            current += 1
            maximum = max(maximum, current)
        try:
            time.sleep(0.02)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
            )
        finally:
            with lock:
                current -= 1

    def handler(_: str) -> dict[str, int]:
        profiled_openai_call(
            request,
            component="qwen_fake",
            operation="initial",
            retry_index=0,
            model="fake",
            messages=[{"role": "user", "content": "test"}],
        )
        return {"processed": 1}

    gate = QwenConcurrencyGate(2, lock_directory=tmp_path / "qwen")
    with qwen_concurrency_gate(gate):
        StreamingDAGScheduler(
            [StreamingStage("judge", 6, handler)],
            cpu_workers=6,
        ).run([f"clip-{index}" for index in range(6)])

    assert maximum == 2


def test_worker_exception_isolated_and_resume_can_skip_durable_work() -> None:
    durable: set[tuple[str, str]] = set()
    fail_once = True
    final_calls: list[str] = []

    def first(clip_uid: str) -> dict[str, int]:
        if (clip_uid, "first") in durable:
            return {"skipped_existing": 1}
        durable.add((clip_uid, "first"))
        return {"processed": 1}

    def second(clip_uid: str) -> dict[str, int]:
        nonlocal fail_once
        if clip_uid == "b" and fail_once:
            fail_once = False
            raise RuntimeError("interrupted")
        durable.add((clip_uid, "second"))
        return {"processed": 1}

    def final(clip_uid: str) -> dict[str, int]:
        final_calls.append(clip_uid)
        durable.add((clip_uid, "final"))
        return {"processed": 1}

    stages = [
        StreamingStage("first", 2, first),
        StreamingStage("second", 2, second),
        StreamingStage("final", 2, final),
    ]
    initial = StreamingDAGScheduler(stages, cpu_workers=2).run(["a", "b", "c"])
    assert sorted(final_calls) == ["a", "c"]
    resumed = StreamingDAGScheduler(stages, cpu_workers=2).run(["a", "b", "c"])

    assert initial.failed_tasks == [
        {
            "clip_uid": "b",
            "stage": "second",
            "error_type": "RuntimeError",
            "reason": "interrupted",
        }
    ]
    assert resumed.failed_tasks == []
    assert resumed.stage_counts["first"]["skipped_existing"] == 3
    assert ("b", "second") in durable
    assert ("b", "final") in durable


def test_profiler_shared_runtime_writes_are_valid_jsonl(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "run", git_commit="abc")

    result = StreamingDAGScheduler(
        [StreamingStage("frames", 4, lambda _: {"processed": 1})],
        cpu_workers=4,
        profiler=profiler,
    ).run([f"clip-{index}" for index in range(12)])
    summary = profiler.write_summary()

    assert result.failed_tasks == []
    assert summary["runtime"]["stages"]["frames"]["successes"] == 12
    assert summary["runtime"]["critical_bottleneck"] == "frames"
    for line in profiler.events_path.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)


def test_shared_failure_jsonl_writer_is_concurrency_safe(tmp_path: Path) -> None:
    destination = tmp_path / "failures.jsonl"

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_append_jsonl, destination, {"case": index})
            for index in range(80)
        ]
        for future in futures:
            future.result()

    values = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(value["case"] for value in values) == list(range(80))


def test_persistent_stage_process_reuses_one_worker(tmp_path: Path) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        """import json, sys
print(json.dumps({'schema_version': 1, 'status': 'ok', 'type': 'ready'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['type'] == 'shutdown':
        print(json.dumps({'schema_version': 1, 'type': 'shutdown', 'request_id': request['request_id'], 'status': 'ok'}), flush=True)
        break
    print(json.dumps({'schema_version': 1, 'type': 'response', 'request_id': request['request_id'], 'status': 'ok', 'counts': {'processed': 1}}), flush=True)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    worker = PersistentStageProcess(
        StageWorkerConfig(
            stage="segment",
            config_path=config_path,
            cuda_visible_devices="5",
            overwrite=False,
            timeout_seconds=5,
            profile=False,
            stderr_log_path=tmp_path / "worker.stderr.log",
            worker_script=script,
        )
    )

    worker.start()
    first = worker.request("clip-a")
    second = worker.request("clip-b")
    worker.close()

    assert first == second == {"processed": 1}
    assert worker.starts == 1


def test_single_worker_request_error_does_not_kill_persistent_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "recovering_worker.py"
    script.write_text(
        """import json, sys
print(json.dumps({'schema_version': 1, 'status': 'ok', 'type': 'ready'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['type'] == 'shutdown':
        print(json.dumps({'schema_version': 1, 'type': 'shutdown', 'request_id': request['request_id'], 'status': 'ok'}), flush=True)
        break
    if request['clip_uid'] == 'bad':
        response = {'schema_version': 1, 'type': 'response', 'request_id': request['request_id'], 'status': 'error', 'reason': 'generation failed'}
    else:
        response = {'schema_version': 1, 'type': 'response', 'request_id': request['request_id'], 'status': 'ok', 'counts': {'processed': 1}}
    print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    worker = PersistentStageProcess(
        StageWorkerConfig(
            stage="remove",
            config_path=config_path,
            cuda_visible_devices="4",
            overwrite=False,
            timeout_seconds=5,
            profile=False,
            stderr_log_path=tmp_path / "worker.stderr.log",
            worker_script=script,
        )
    )

    worker.start()
    with pytest.raises(RuntimeError, match="generation failed"):
        worker.request("bad")
    recovered = worker.request("good")
    worker.close()

    assert recovered == {"processed": 1}
    assert worker.starts == 1
