from __future__ import annotations

import fcntl
import inspect
import io
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
import run_pipeline_v3 as pipeline_module
from r2v_data_v2.v3.config import (
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    RuntimeConfig,
    RuntimeGpuWorkersConfig,
    RuntimeStageWorkersConfig,
    SourceConfig,
    SubjectAttributeCompletionConfig,
    SubjectAttributeGmeConfig,
    SubjectAttributesConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.profiling import (
    QwenConcurrencyGate,
    V3Profiler,
    profiled_openai_call,
    qwen_concurrency_gate,
)
from r2v_data_v2.v3 import subject_attributes
from r2v_data_v2.v3.runtime import (
    PersistentStageProcess,
    PersistentStageProcessPool,
    StageWorkerConfig,
    StreamingDAGScheduler,
    StreamingStage,
    streaming_model_stage_enabled,
    runtime_worker_config,
    runtime_segment_pool_worker_configs,
)
from r2v_data_v2.v3.storage import _append_jsonl
from r2v_data_v2.v3.subject_attributes import QwenSubjectAttributeClient
from run_pipeline_v3 import STAGE_ORDER
from tools.run_v3_streaming_stage_worker import _StageRuntime, _local_device_config


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
    assert config.runtime.sam3_compile_enabled is False
    assert config.runtime.stage_workers.segment == 1
    assert config.runtime.stage_workers.remove == 1
    assert config.runtime.stage_workers.reference_edit == 1
    assert config.runtime.stage_workers.subject_attributes == 2
    assert config.runtime.gpu_workers.subject_attributes_segment is None
    assert config.runtime.gpu_workers.segment_pool == ()
    assert config.runtime.subject_attributes_deferred is False
    assert STAGE_ORDER.index("instruct") < STAGE_ORDER.index("subject_attributes")
    assert STAGE_ORDER.index("subject_attributes") < STAGE_ORDER.index("export")


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


def test_sam3_compile_is_runtime_only_and_not_visual_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        runtime=replace(config.runtime, sam3_compile_enabled=True),
    )
    changed.validate()

    assert config.fingerprint() == changed.fingerprint()
    assert config.model_identifiers() == changed.model_identifiers()


def test_segment_pool_gpu_assignment_and_deferred_mode_are_runtime_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    first = replace(
        config,
        runtime=replace(
            config.runtime,
            subject_attributes_deferred=True,
            stage_workers=replace(
                config.runtime.stage_workers,
                segment=2,
            ),
            gpu_workers=replace(
                config.runtime.gpu_workers,
                segment_pool=("5", "7"),
            ),
        ),
    )
    second = replace(
        first,
        runtime=replace(
            first.runtime,
            subject_attributes_deferred=False,
            gpu_workers=replace(first.runtime.gpu_workers, segment_pool=("5", "8")),
        ),
    )
    first.validate()
    second.validate()

    assert first.runtime.sam3_compile_enabled is False
    assert first.fingerprint() == second.fingerprint()
    assert first.model_identifiers() == second.model_identifiers()


def test_sam3_compile_requires_strict_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    invalid = replace(
        config,
        runtime=replace(
            config.runtime,
            sam3_compile_enabled="true",  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(TypeError, match="sam3_compile_enabled must be a boolean"):
        invalid.validate()


def test_dedicated_attribute_worker_forces_sam3_eager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    compiled = replace(
        config,
        runtime=replace(
            config.runtime,
            sam3_compile_enabled=True,
            gpu_workers=replace(
                config.runtime.gpu_workers,
                subject_attributes_segment="7",
            ),
        ),
    )

    main = _local_device_config(compiled, "segment")
    attribute = _local_device_config(
        compiled,
        "segment",
        attribute_probe_only=True,
    )

    assert main.runtime.sam3_compile_enabled is True
    assert attribute.runtime.sam3_compile_enabled is False
    assert main.sam3.device == attribute.sam3.device == "cuda"
    config_path = tmp_path / "worker.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")
    worker = runtime_worker_config(
        compiled,
        config_path=config_path,
        stage="segment",
        gpu_assignment="subject_attributes_segment",
        overwrite=False,
        profile=False,
    )
    assert worker.attribute_probe_only is True


def test_compiled_main_worker_lazily_uses_separate_eager_attribute_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import r2v_data_v2.v3.segment as segment_module

    config = _config(tmp_path, monkeypatch)
    compiled = replace(
        config,
        runtime=replace(config.runtime, sam3_compile_enabled=True),
    )
    frame_path = tmp_path / "clip-a" / "frames" / "01.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"frame")

    class Backend:
        def close(self) -> None:
            pass

    created: list[object | None] = []

    class AttributeSegmenter:
        def __init__(self, _config: object, *, backend: object | None = None) -> None:
            created.append(backend)

        def segment_frame(self, **_kwargs: object) -> tuple[np.ndarray, ...]:
            return (np.ones((2, 2), dtype=bool),)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        segment_module,
        "build_sam3_segment_backend",
        lambda _config: Backend(),
    )
    monkeypatch.setattr(
        subject_attributes,
        "Sam3AttributeFrameSegmenter",
        AttributeSegmenter,
    )

    class Storage:
        def read_frames(self, _clip_uid: str) -> SimpleNamespace:
            return SimpleNamespace(
                frames=[
                    SimpleNamespace(
                        slot=1,
                        source_frame_index=9,
                        image_path=Path("frames/01.jpg"),
                    )
                ]
            )

        def clip_dir(self, _clip_uid: str) -> Path:
            return tmp_path / "clip-a"

    runtime = _StageRuntime(
        compiled,
        Storage(),  # type: ignore[arg-type]
        stage="segment",
        overwrite=False,
    )

    assert created == []
    masks = runtime.attribute_probe(
        clip_uid="clip-a",
        frame_slot=1,
        source_frame_index=9,
        grounding_prompt="hair",
    )
    runtime.close()

    assert created == [None]
    assert masks == [{"size": [2, 2], "counts": [0, 4]}]


def test_sidecar_worker_count_does_not_change_visual_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    workers = replace(config.runtime.stage_workers, subject_attributes=3)
    changed = replace(config, runtime=replace(config.runtime, stage_workers=workers))

    assert config.fingerprint() == changed.fingerprint()
    assert config.model_identifiers() == changed.model_identifiers()


def test_sidecar_gpu_assignment_does_not_change_visual_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    gpu_workers = replace(
        config.runtime.gpu_workers,
        subject_attributes_segment="7",
    )
    changed = replace(config, runtime=replace(config.runtime, gpu_workers=gpu_workers))

    assert config.fingerprint() == changed.fingerprint()
    assert config.model_identifiers() == changed.model_identifiers()


def test_disabled_gme_config_remains_fingerprint_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        subject_attribute_gme=replace(
            config.subject_attribute_gme,
            min_margin=0.25,
            model_path=Path("/unused/disabled/model"),
        ),
        runtime=replace(
            config.runtime,
            gpu_workers=replace(
                config.runtime.gpu_workers,
                subject_attributes_gme="7",
            ),
        ),
    )

    assert config.fingerprint() == changed.fingerprint()
    assert config.model_identifiers() == changed.model_identifiers()


def test_enabled_gme_paths_and_semantics_enter_attribute_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    writable = config.resolved_run_root.parents[1]
    executable = writable / "venvs" / "gme" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model_path = writable / "models" / "gme-Qwen2-VL-2B-Instruct"
    model_path.mkdir(parents=True)
    enabled = replace(
        config,
        subject_attribute_gme=SubjectAttributeGmeConfig(
            enabled=True,
            python_executable=executable,
            model_path=model_path,
        ),
        runtime=replace(
            config.runtime,
            gpu_workers=replace(
                config.runtime.gpu_workers,
                subject_attributes_gme="7",
            ),
        ),
    )
    enabled.validate()

    identifiers = enabled.model_identifiers()
    assert enabled.fingerprint() != config.fingerprint()
    assert identifiers["subject_attribute_gme.model"] == (
        "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    )
    assert identifiers["subject_attribute_gme.screen_mode"] == "relative_margin_v1"
    assert identifiers["subject_attribute_gme.min_margin"] == "0.0"
    assert "subject_attributes_gme" not in identifiers["runtime.gpu_workers"]


def test_subject_attributes_do_not_validate_or_require_gme_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    invalid = replace(
        config,
        subject_attribute_gme=SubjectAttributeGmeConfig(
            enabled=True,
            python_executable=tmp_path / "missing-python",
            model_path=tmp_path / "missing-model",
        ),
        runtime=replace(
            config.runtime,
            gpu_workers=replace(
                config.runtime.gpu_workers,
                subject_attributes_gme="7",
            ),
        ),
    )

    invalid.validate()


def test_attribute_completion_sidecar_config_does_not_change_visual_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    enabled = replace(
        config,
        subject_attributes=SubjectAttributesConfig(
            completion=SubjectAttributeCompletionConfig(enabled=True)
        ),
        runtime=replace(
            config.runtime,
            gpu_workers=replace(
                config.runtime.gpu_workers,
                subject_attributes_completion="6",
            ),
        ),
    )

    assert enabled.fingerprint() == config.fingerprint()
    assert enabled.model_identifiers() == config.model_identifiers()


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
                "    subject_attributes: 3",
                "  gpu_workers:",
                "    segment: '5'",
                "    subject_attributes_segment: '7'",
                "    remove: '4'",
                "    reference_edit: '6'",
            )
        ),
        encoding="utf-8",
    )

    loaded = load_config(yaml_path)

    assert loaded.runtime.mode == "streaming_v1"
    assert loaded.runtime.sam3_compile_enabled is False
    assert loaded.runtime.cpu_workers == 6
    assert loaded.runtime.stage_workers.frames == 3
    assert loaded.runtime.stage_workers.subject_attributes == 3
    assert loaded.runtime.gpu_workers == RuntimeGpuWorkersConfig(
        segment="5",
        subject_attributes_segment="7",
        subject_attributes_gme=None,
        remove="4",
        reference_edit="6",
    )


def _exercise_streaming_segment_worker_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dedicated_gpu: str | None,
    segment_pool: tuple[str, ...] = (),
    fail_scheduler: bool = False,
) -> list[object]:
    config = _config(tmp_path, monkeypatch)
    gpu_workers = replace(
        config.runtime.gpu_workers,
        segment="5",
        segment_pool=segment_pool,
        subject_attributes_segment=dedicated_gpu,
    )
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            mode="streaming_v1",
            stage_workers=replace(
                config.runtime.stage_workers,
                segment=max(1, len(segment_pool)),
            ),
            gpu_workers=gpu_workers,
        ),
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")

    class FakeStorage:
        root = config.resolved_run_root

        def iter_clips(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(clip_uid="clip-a")]

        def update_stage_counts(
            self,
            _stage: str,
            _counts: dict[str, int],
        ) -> None:
            return None

        def append_failure(self, **_kwargs: object) -> None:
            return None

    class FakeProcess:
        instances: list[FakeProcess] = []

        def __init__(self, worker_config: StageWorkerConfig) -> None:
            self.config = worker_config
            self.starts = 0
            self.closes = 0
            self.run_clip_requests: list[str] = []
            self.attribute_probe_requests: list[str] = []
            self.instances.append(self)

        def start(self) -> None:
            self.starts += 1

        def close(self) -> None:
            self.closes += 1

        def terminate(self) -> None:
            return None

        def request(self, clip_uid: str) -> dict[str, int]:
            self.run_clip_requests.append(clip_uid)
            return {"processed": 1}

        def attribute_probe(self, **kwargs: object) -> tuple[np.ndarray, ...]:
            self.attribute_probe_requests.append(str(kwargs["clip_uid"]))
            return ()

    def fake_stage_handler(
        *,
        stage: str,
        process: FakeProcess | None,
        attribute_segmentation_backend: object | None,
        **_kwargs: object,
    ) -> object:
        if stage == "segment":
            assert process is not None
            return process.request
        assert stage == "subject_attributes"
        assert attribute_segmentation_backend is not None

        def probe(clip_uid: str) -> dict[str, int]:
            attribute_segmentation_backend._client.attribute_probe(
                clip_uid=clip_uid,
                frame_slot=1,
                source_frame_index=10,
                grounding_prompt="the red jacket",
            )
            return {"processed": 1}

        return probe

    class FakeScheduler:
        def __init__(self, stages: list[StreamingStage], **_kwargs: object) -> None:
            self.stages = stages

        def run(self, clip_uids: list[str]) -> SimpleNamespace:
            counts = {
                stage.name: stage.handler(clip_uids[0]) for stage in self.stages
            }
            if fail_scheduler:
                raise RuntimeError("scheduler failed")
            return SimpleNamespace(
                stage_counts=counts,
                failed_tasks=[],
                to_dict=lambda: {},
            )

    monkeypatch.setattr(pipeline_module, "PersistentStageProcess", FakeProcess)
    monkeypatch.setattr(pipeline_module, "StreamingDAGScheduler", FakeScheduler)
    monkeypatch.setattr(pipeline_module, "_streaming_stage_handler", fake_stage_handler)
    monkeypatch.setattr(
        pipeline_module,
        "reconcile_subject_attribute_outputs",
        lambda **_kwargs: {},
    )

    def call() -> dict[str, object]:
        return pipeline_module._run_streaming_pipeline(
            config_path=config_path,
            config=config,
            storage=FakeStorage(),
            ordered_stages=("segment", "subject_attributes"),
            overwrite=False,
            profiler=None,
            results={},
            annotation_client=None,
            frame_decoder=None,
            segmentation_backend=None,
            instruction_client=None,
            background_removal_backend=None,
            background_removal_judge=None,
            entity_reference_judge=None,
            cross_pair_judge=None,
            background_final_judge=None,
            reference_completion_backend=None,
            reference_completion_judge=None,
            reference_edit_backend=None,
            reference_edit_judge=None,
            reference_edit_sam_reviewer=None,
            reference_integrity_judge=None,
            subject_attribute_discovery_client=object(),
            subject_attribute_review_client=object(),
            attribute_segmentation_backend=None,
            attribute_gme_screener=None,
            profile=False,
        )
    if fail_scheduler:
        with pytest.raises(RuntimeError, match="scheduler failed"):
            call()
    else:
        call()
    return list(FakeProcess.instances)


def test_dedicated_attribute_segment_worker_routes_probes_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = _exercise_streaming_segment_worker_routing(
        tmp_path,
        monkeypatch,
        dedicated_gpu="7",
    )

    assert len(processes) == 2
    main = next(
        process for process in processes if process.config.cuda_visible_devices == "5"
    )
    dedicated = next(
        process for process in processes if process.config.cuda_visible_devices == "7"
    )
    assert main.config.stage == dedicated.config.stage == "segment"
    assert main.run_clip_requests == ["clip-a"]
    assert main.attribute_probe_requests == []
    assert dedicated.run_clip_requests == []
    assert dedicated.attribute_probe_requests == ["clip-a"]
    assert dedicated.starts == 1
    assert all(process.closes == 1 for process in processes)


def test_attribute_segment_worker_falls_back_to_main_segment_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = _exercise_streaming_segment_worker_routing(
        tmp_path,
        monkeypatch,
        dedicated_gpu=None,
    )

    assert len(processes) == 1
    assert processes[0].run_clip_requests == ["clip-a"]
    assert processes[0].attribute_probe_requests == ["clip-a"]
    assert processes[0].starts == processes[0].closes == 1


def test_dual_segment_pool_routes_inline_attributes_without_dedicated_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = _exercise_streaming_segment_worker_routing(
        tmp_path,
        monkeypatch,
        dedicated_gpu="7",
        segment_pool=("5", "7"),
    )

    assert len(processes) == 2
    assert [process.config.cuda_visible_devices for process in processes] == [
        "5",
        "7",
    ]
    assert all(process.config.attribute_probe_only is False for process in processes)
    assert sum(len(process.run_clip_requests) for process in processes) == 1
    assert sum(len(process.attribute_probe_requests) for process in processes) == 1
    assert all(process.starts == process.closes == 1 for process in processes)


def test_dedicated_attribute_segment_worker_closes_on_pipeline_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = _exercise_streaming_segment_worker_routing(
        tmp_path,
        monkeypatch,
        dedicated_gpu="7",
        fail_scheduler=True,
    )

    assert len(processes) == 2
    assert all(process.closes == 1 for process in processes)


def test_streaming_does_not_start_gme_worker_even_for_legacy_enabled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = replace(
        config,
        subject_attribute_gme=replace(
            config.subject_attribute_gme,
            enabled=True,
        ),
        runtime=replace(
            config.runtime,
            mode="streaming_v1",
            gpu_workers=replace(
                config.runtime.gpu_workers,
                segment="5",
                subject_attributes_segment="7",
                subject_attributes_gme="7",
            ),
        ),
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")

    class _Storage:
        root = config.resolved_run_root

        def iter_clips(self):
            return [SimpleNamespace(clip_uid="clip-a")]

        def update_stage_counts(self, *_args):
            return None

        def append_failure(self, **_kwargs):
            return None

    class _Process:
        def __init__(self, worker_config):
            self.config = worker_config

        def start(self):
            return None

        def close(self):
            return None

        def attribute_probe(self, **_kwargs):
            return ()

    observed: dict[str, object] = {}

    def handler(
        *, attribute_segmentation_backend, attribute_gme_screener, **_kwargs
    ):
        observed["segmenter"] = attribute_segmentation_backend
        observed["gme"] = attribute_gme_screener
        return lambda _clip_uid: {"processed": 1}

    class _Scheduler:
        def __init__(self, stages, **_kwargs):
            self.stages = stages

        def run(self, clip_uids):
            counts = {
                stage.name: stage.handler(clip_uids[0]) for stage in self.stages
            }
            return SimpleNamespace(
                stage_counts=counts,
                failed_tasks=[],
                to_dict=lambda: {},
            )

    monkeypatch.setattr(pipeline_module, "PersistentStageProcess", _Process)
    monkeypatch.setattr(pipeline_module, "_streaming_stage_handler", handler)
    monkeypatch.setattr(pipeline_module, "StreamingDAGScheduler", _Scheduler)
    monkeypatch.setattr(
        pipeline_module,
        "reconcile_subject_attribute_outputs",
        lambda **_kwargs: {},
    )

    pipeline_module._run_streaming_pipeline(
        config_path=config_path,
        config=config,
        storage=_Storage(),
        ordered_stages=("subject_attributes",),
        overwrite=False,
        profiler=None,
        results={},
        annotation_client=None,
        frame_decoder=None,
        segmentation_backend=None,
        instruction_client=None,
        background_removal_backend=None,
        background_removal_judge=None,
        entity_reference_judge=None,
        cross_pair_judge=None,
        background_final_judge=None,
        reference_completion_backend=None,
        reference_completion_judge=None,
        reference_edit_backend=None,
        reference_edit_judge=None,
        reference_edit_sam_reviewer=None,
        reference_integrity_judge=None,
        subject_attribute_discovery_client=object(),
        subject_attribute_review_client=object(),
        attribute_segmentation_backend=None,
        attribute_gme_screener=None,
        profile=False,
    )

    assert observed["gme"] is None
    segmenter = observed["segmenter"]
    assert segmenter._inference_lock is None


def test_deferred_subject_attributes_start_no_attribute_or_gme_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = replace(
        config,
        subject_attribute_gme=replace(config.subject_attribute_gme, enabled=True),
        runtime=replace(
            config.runtime,
            mode="streaming_v1",
            subject_attributes_deferred=True,
        ),
    )
    config.validate()
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")

    class Storage:
        root = config.resolved_run_root

        def iter_clips(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(clip_uid="clip-a")]

    monkeypatch.setattr(
        pipeline_module,
        "PersistentStageProcess",
        lambda *_args, **_kwargs: pytest.fail("attribute SAM3 worker started"),
    )
    result = pipeline_module._run_streaming_pipeline(
        config_path=config_path,
        config=config,
        storage=Storage(),  # type: ignore[arg-type]
        ordered_stages=("subject_attributes",),
        overwrite=False,
        profiler=None,
        results={},
        annotation_client=None,
        frame_decoder=None,
        segmentation_backend=None,
        instruction_client=None,
        background_removal_backend=None,
        background_removal_judge=None,
        entity_reference_judge=None,
        cross_pair_judge=None,
        background_final_judge=None,
        reference_completion_backend=None,
        reference_completion_judge=None,
        reference_edit_backend=None,
        reference_edit_judge=None,
        reference_edit_sam_reviewer=None,
        reference_integrity_judge=None,
        subject_attribute_discovery_client=None,
        subject_attribute_review_client=None,
        attribute_segmentation_backend=None,
        attribute_gme_screener=None,
        profile=False,
    )

    assert result["subject_attributes"] == {"deferred": True}
    assert result["deferred_stages"] == ["subject_attributes"]
    assert result["completed_stages"] == []


def test_subject_attributes_without_completion_do_not_start_boogu_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            mode="streaming_v1",
            gpu_workers=replace(
                config.runtime.gpu_workers,
                reference_edit="6",
            ),
        ),
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")

    class Storage:
        root = config.resolved_run_root

        def iter_clips(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(clip_uid="clip-a")]

        def update_stage_counts(self, *_args: object) -> None:
            return None

        def append_failure(self, **_kwargs: object) -> None:
            return None

    class Scheduler:
        def __init__(self, stages, **_kwargs):
            self.stages = stages

        def run(self, clip_uids):
            return SimpleNamespace(
                stage_counts={stage.name: {"processed": 1} for stage in self.stages},
                failed_tasks=[],
                to_dict=lambda: {},
            )

    monkeypatch.setattr(
        pipeline_module,
        "PersistentStageProcess",
        lambda *_args, **_kwargs: pytest.fail("Boogu worker started"),
    )
    monkeypatch.setattr(pipeline_module, "StreamingDAGScheduler", Scheduler)
    monkeypatch.setattr(
        pipeline_module,
        "_streaming_stage_handler",
        lambda **_kwargs: (lambda _clip_uid: {"processed": 1}),
    )
    monkeypatch.setattr(
        pipeline_module,
        "reconcile_subject_attribute_outputs",
        lambda **_kwargs: {},
    )

    pipeline_module._run_streaming_pipeline(
        config_path=config_path,
        config=config,
        storage=Storage(),  # type: ignore[arg-type]
        ordered_stages=("subject_attributes",),
        overwrite=False,
        profiler=None,
        results={},
        annotation_client=None,
        frame_decoder=None,
        segmentation_backend=None,
        instruction_client=None,
        background_removal_backend=None,
        background_removal_judge=None,
        entity_reference_judge=None,
        cross_pair_judge=None,
        background_final_judge=None,
        reference_completion_backend=None,
        reference_completion_judge=None,
        reference_edit_backend=None,
        reference_edit_judge=None,
        reference_edit_sam_reviewer=None,
        reference_integrity_judge=None,
        subject_attribute_discovery_client=object(),
        subject_attribute_review_client=object(),
        attribute_segmentation_backend=object(),
        attribute_gme_screener=None,
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


def test_subject_attributes_stage_overlaps_upstream_work_for_other_clips() -> None:
    slow_instruct_started = threading.Event()
    attribute_started = threading.Event()
    ordering: dict[str, list[str]] = {"a": [], "b": []}

    def instruct(clip_uid: str) -> dict[str, int]:
        ordering[clip_uid].append("instruct")
        if clip_uid == "b":
            slow_instruct_started.set()
            assert attribute_started.wait(2)
        return {"processed": 1}

    def attributes(clip_uid: str) -> dict[str, int]:
        ordering[clip_uid].append("subject_attributes")
        if clip_uid == "a":
            assert slow_instruct_started.wait(2)
            attribute_started.set()
        return {"processed": 1}

    result = StreamingDAGScheduler(
        [
            StreamingStage("instruct", 2, instruct),
            StreamingStage("subject_attributes", 2, attributes),
        ],
        cpu_workers=4,
    ).run(["b", "a"])

    assert result.failed_tasks == []
    assert ordering == {
        "a": ["instruct", "subject_attributes"],
        "b": ["instruct", "subject_attributes"],
    }


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


def test_qwen_gate_acquires_free_slot_instead_of_waiting_on_busy_preferred_slot(
    tmp_path: Path,
) -> None:
    lock_directory = tmp_path / "qwen"
    gate = QwenConcurrencyGate(2, lock_directory=lock_directory)
    gate._ticket = -os.getpid() % gate.maximum
    slot_zero = (lock_directory / "slot-0.lock").open("a+")
    fcntl.flock(slot_zero.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    acquired = threading.Event()

    def acquire() -> int:
        with gate.acquire() as observation:
            acquired.set()
            return observation.qwen_slot

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(acquire)
            assert acquired.wait(0.5)
            assert future.result() == 1
    finally:
        fcntl.flock(slot_zero.fileno(), fcntl.LOCK_UN)
        slot_zero.close()


def test_independent_qwen_gates_share_global_slot_capacity(tmp_path: Path) -> None:
    lock_directory = tmp_path / "qwen"
    gates = [
        QwenConcurrencyGate(2, lock_directory=lock_directory),
        QwenConcurrencyGate(2, lock_directory=lock_directory),
    ]
    current = 0
    maximum = 0
    state_lock = threading.Lock()

    def acquire(index: int) -> None:
        nonlocal current, maximum
        with gates[index % len(gates)].acquire():
            with state_lock:
                current += 1
                maximum = max(maximum, current)
            try:
                time.sleep(0.02)
            finally:
                with state_lock:
                    current -= 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(acquire, index) for index in range(12)]
        for future in futures:
            future.result()

    assert maximum == 2


def test_qwen_gate_maximum_one_and_exception_release_slot(tmp_path: Path) -> None:
    gate = QwenConcurrencyGate(1, lock_directory=tmp_path / "qwen")

    with pytest.raises(RuntimeError, match="request failed"):
        with gate.acquire() as observation:
            assert observation.qwen_slot == 0
            raise RuntimeError("request failed")

    with gate.acquire() as observation:
        assert observation.qwen_slot == 0


def test_subject_attribute_qwen_calls_use_existing_global_gate(tmp_path: Path) -> None:
    current = 0
    maximum = 0
    lock = threading.Lock()

    def request(**_kwargs: object) -> object:
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

    openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=request),
        )
    )
    client = QwenSubjectAttributeClient(
        QwenServiceConfig(model="fake"),
        client=openai_client,
    )

    def handler(_: str) -> dict[str, int]:
        client._request(
            component="qwen_subject_attribute_discovery",
            system_prompt="test",
            content=[{"type": "text", "text": "test"}],
        )
        return {"processed": 1}

    gate = QwenConcurrencyGate(2, lock_directory=tmp_path / "qwen")
    with qwen_concurrency_gate(gate):
        StreamingDAGScheduler(
            [StreamingStage("subject_attributes", 6, handler)],
            cpu_workers=6,
        ).run([f"clip-{index}" for index in range(6)])

    assert maximum == 2
    source = inspect.getsource(subject_attributes)
    assert "ThreadPoolExecutor" not in source
    assert "Semaphore(" not in source


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


def test_subject_attribute_failure_is_clip_local() -> None:
    completed: list[str] = []

    def attributes(clip_uid: str) -> dict[str, int]:
        if clip_uid == "b":
            raise RuntimeError("attribute failure")
        return {"processed": 1}

    def downstream(clip_uid: str) -> dict[str, int]:
        completed.append(clip_uid)
        return {"processed": 1}

    result = StreamingDAGScheduler(
        [
            StreamingStage("subject_attributes", 2, attributes),
            StreamingStage("downstream", 2, downstream),
        ],
        cpu_workers=4,
    ).run(["a", "b", "c"])

    assert sorted(completed) == ["a", "c"]
    assert result.failed_tasks == [
        {
            "clip_uid": "b",
            "stage": "subject_attributes",
            "error_type": "RuntimeError",
            "reason": "attribute failure",
        }
    ]


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
    if request['type'] == 'attribute_probe':
        response = {'schema_version': 1, 'type': 'response', 'request_id': request['request_id'], 'status': 'ok', 'masks': [{'size': [2, 2], 'counts': [0, 1, 3]}]}
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
    masks = worker.attribute_probe(
        clip_uid="clip-a",
        frame_slot=1,
        source_frame_index=10,
        grounding_prompt="the red jacket",
    )
    second = worker.request("clip-b")
    worker.close()

    assert first == second == {"processed": 1}
    assert len(masks) == 1
    assert np.array_equal(masks[0], np.array([[True, False], [False, False]]))
    assert worker.starts == 1


def test_segment_pool_builds_two_isolated_cuda_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            stage_workers=replace(config.runtime.stage_workers, segment=2),
            gpu_workers=replace(
                config.runtime.gpu_workers,
                segment_pool=("5", "7"),
            ),
        ),
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("unused: true\n", encoding="utf-8")

    workers = runtime_segment_pool_worker_configs(
        config,
        config_path=config_path,
        overwrite=False,
        profile=True,
    )

    assert len(workers) == 2
    assert [worker.cuda_visible_devices for worker in workers] == ["5", "7"]
    assert all(worker.stage == "segment" for worker in workers)
    assert all(worker.attribute_probe_only is False for worker in workers)


def test_segment_pool_dispatches_to_available_worker_without_overlap() -> None:
    state_lock = threading.Lock()
    active: dict[str, int] = {"5": 0, "7": 0}
    maximum: dict[str, int] = {"5": 0, "7": 0}
    global_active = 0
    global_maximum = 0
    started: queue.Queue[tuple[str, str]] = queue.Queue()
    release = {"5": threading.Event(), "7": threading.Event()}

    class Worker:
        def __init__(self, gpu: str) -> None:
            self.config = SimpleNamespace(cuda_visible_devices=gpu)
            self.starts = 0
            self.closes = 0

        def start(self) -> None:
            self.starts += 1

        def terminate(self) -> None:
            pass

        def close(self) -> None:
            self.closes += 1

        def _execute(self, request_kind: str) -> None:
            nonlocal global_active, global_maximum
            gpu = self.config.cuda_visible_devices
            with state_lock:
                active[gpu] += 1
                maximum[gpu] = max(maximum[gpu], active[gpu])
                global_active += 1
                global_maximum = max(global_maximum, global_active)
            started.put((gpu, request_kind))
            try:
                assert release[gpu].wait(timeout=2)
            finally:
                with state_lock:
                    active[gpu] -= 1
                    global_active -= 1

        def request(self, _clip_uid: str) -> dict[str, object]:
            self._execute("main")
            return {"processed": 1}

        def attribute_probe(self, **_kwargs: object) -> tuple[np.ndarray, ...]:
            self._execute("attribute")
            return ()

    gpu5 = Worker("5")
    gpu7 = Worker("7")
    pool = PersistentStageProcessPool([gpu5, gpu7])  # type: ignore[list-item]
    pool.start()
    with ThreadPoolExecutor(max_workers=3) as executor:
        main = executor.submit(pool.request, "clip-a")
        attribute = executor.submit(
            pool.attribute_probe,
            clip_uid="clip-b",
            frame_slot=1,
            source_frame_index=10,
            grounding_prompt="the red jacket",
        )
        initial = [started.get(timeout=1), started.get(timeout=1)]
        assert {gpu for gpu, _kind in initial} == {"5", "7"}
        assert {kind for _gpu, kind in initial} == {"main", "attribute"}

        third = executor.submit(pool.request, "clip-c")
        assert not third.done()
        released_gpu = initial[0][0]
        release[released_gpu].set()
        assert started.get(timeout=1) == (released_gpu, "main")
        release[initial[1][0]].set()

        assert main.result() == {"processed": 1}
        assert attribute.result() == ()
        assert third.result() == {"processed": 1}
    counters = pool.performance_counters()
    pool.close()

    assert gpu5.starts == gpu7.starts == 1
    assert gpu5.closes == gpu7.closes == 1
    assert maximum == {"5": 1, "7": 1}
    assert global_maximum == 2
    assert counters["segment_worker_pool_size"] == 2
    assert sum(counters["segment_worker_requests_by_gpu"].values()) == 2
    assert sum(counters["sam_pool_main_requests_by_gpu"].values()) == 2
    assert sum(counters["sam_pool_attribute_probe_requests_by_gpu"].values()) == 1
    assert sum(counters["sam_pool_main_service_seconds_by_gpu"].values()) > 0
    assert sum(counters["sam_pool_attribute_service_seconds_by_gpu"].values()) > 0
    assert counters["sam_pool_main_wait_seconds_total"] > 0
    assert counters["sam_pool_attribute_wait_seconds_total"] >= 0
    assert counters["sam_pool_max_concurrent_requests"] == 2


def test_segment_pool_runs_two_attribute_probes_concurrently() -> None:
    barrier = threading.Barrier(2)
    observed: list[str] = []
    state_lock = threading.Lock()

    class Worker:
        def __init__(self, gpu: str) -> None:
            self.config = SimpleNamespace(cuda_visible_devices=gpu)

        def start(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def close(self) -> None:
            return None

        def attribute_probe(self, **_kwargs: object) -> tuple[np.ndarray, ...]:
            with state_lock:
                observed.append(self.config.cuda_visible_devices)
            barrier.wait(timeout=2)
            return ()

    pool = PersistentStageProcessPool(  # type: ignore[list-item]
        [Worker("5"), Worker("7")]
    )
    pool.start()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                pool.attribute_probe,
                clip_uid=f"clip-{index}",
                frame_slot=1,
                source_frame_index=10,
                grounding_prompt="the red jacket",
            )
            for index in range(2)
        ]
        assert [future.result() for future in futures] == [(), ()]
    counters = pool.performance_counters()
    pool.close()

    assert sorted(observed) == ["5", "7"]
    assert counters["sam_pool_attribute_probe_requests_by_gpu"] == {
        "5": 1,
        "7": 1,
    }
    assert counters["sam_pool_max_concurrent_requests"] == 2


def test_completion_resegmentation_uses_shared_segment_pool() -> None:
    observed: list[str] = []

    class Worker:
        def __init__(self, gpu: str) -> None:
            self.config = SimpleNamespace(cuda_visible_devices=gpu)

        def start(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def close(self) -> None:
            return None

        def attribute_completion_probe(self, **_kwargs):
            observed.append(self.config.cuda_visible_devices)
            return (np.ones((2, 2), dtype=bool),)

    pool = PersistentStageProcessPool(  # type: ignore[list-item]
        [Worker("5"), Worker("7")]
    )
    pool.start()
    first = pool.attribute_completion_probe(
        image_path=Path("/run/subject_attributes/completion_candidates/a.png"),
        grounding_prompt="hair",
    )
    second = pool.attribute_completion_probe(
        image_path=Path("/run/subject_attributes/completion_candidates/b.png"),
        grounding_prompt="face",
    )
    counters = pool.performance_counters()
    pool.close()

    assert len(first) == len(second) == 1
    assert sorted(observed) == ["5", "7"]
    assert counters["sam_pool_attribute_probe_requests_by_gpu"] == {
        "5": 1,
        "7": 1,
    }


def test_attribute_completion_disables_thinking_and_instruction_rewrite(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "subject_attributes" / "completion_candidates"
    source_path = sidecar / "clip" / "raw" / "source.png"
    output_path = sidecar / "clip" / "generated" / "candidate.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), "gray").save(source_path)
    observed: dict[str, object] = {}

    class Backend:
        def edit(self, **kwargs):
            observed.update(kwargs)
            buffer = io.BytesIO()
            Image.new("RGB", (kwargs["width"], kwargs["height"]), "gray").save(
                buffer,
                format="PNG",
            )
            return SimpleNamespace(png_bytes=buffer.getvalue())

    runtime = object.__new__(_StageRuntime)
    runtime.stage = "reference_edit"
    runtime.storage = SimpleNamespace(root=tmp_path)
    runtime._reference_edit_backend = Backend()
    runtime.config = SimpleNamespace(
        reference_edit=SimpleNamespace(
            target_area=1024 * 1024,
            alignment=16,
            model_path=Path("/model/boogu"),
        )
    )

    result = runtime.attribute_completion(
        source_path=source_path,
        output_path=output_path,
        instruction="补全单一主体",
        seed=17,
    )

    assert output_path.is_file()
    assert observed["thinking_enabled"] is False
    assert observed["instruction_rewrite_enabled"] is False
    assert observed["seed"] == 17
    assert result["thinking_enabled"] is False
    assert result["instruction_rewrite_enabled"] is False


def test_attribute_background_runtime_request_path_is_removed() -> None:
    assert not hasattr(_StageRuntime, "attribute_background")
    assert not hasattr(PersistentStageProcess, "attribute_background")


def test_segment_pool_returns_worker_after_attribute_probe_exception() -> None:
    class Worker:
        config = SimpleNamespace(cuda_visible_devices="5")

        def __init__(self) -> None:
            self.starts = 0
            self.closes = 0
            self.main_requests = 0

        def start(self) -> None:
            self.starts += 1

        def terminate(self) -> None:
            return None

        def close(self) -> None:
            self.closes += 1

        def request(self, _clip_uid: str) -> dict[str, object]:
            self.main_requests += 1
            return {"processed": 1}

        def attribute_probe(self, **_kwargs: object) -> tuple[np.ndarray, ...]:
            raise RuntimeError("probe failed")

    worker = Worker()
    pool = PersistentStageProcessPool([worker])  # type: ignore[list-item]
    pool.start()
    with pytest.raises(RuntimeError, match="probe failed"):
        pool.attribute_probe(
            clip_uid="clip-a",
            frame_slot=1,
            source_frame_index=10,
            grounding_prompt="the red jacket",
        )
    assert pool.request("clip-b") == {"processed": 1}
    counters = pool.performance_counters()
    pool.close()

    assert worker.starts == worker.closes == 1
    assert worker.main_requests == 1
    assert counters["sam_pool_attribute_probe_requests_by_gpu"] == {"5": 1}
    assert counters["sam_pool_main_requests_by_gpu"] == {"5": 1}


def test_main_segment_and_attribute_probe_share_one_request_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        )
    )
    active = 0
    maximum = 0
    state_lock = threading.Lock()

    def fake_exchange(payload: dict[str, object]) -> dict[str, object]:
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            if payload["type"] == "attribute_probe":
                return {"status": "ok", "masks": []}
            return {"status": "ok", "counts": {"processed": 1}}
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(worker, "_exchange", fake_exchange)
    with ThreadPoolExecutor(max_workers=2) as executor:
        main_request = executor.submit(worker.request, "clip-a")
        probe_request = executor.submit(
            worker.attribute_probe,
            clip_uid="clip-b",
            frame_slot=1,
            source_frame_index=10,
            grounding_prompt="the red jacket",
        )
        assert main_request.result() == {"processed": 1}
        assert probe_request.result() == ()

    assert maximum == 1


def test_persistent_segment_worker_reports_recall_counter_deltas_per_clip() -> None:
    class CumulativeBackend:
        def __init__(self) -> None:
            self.anchor = {"anchor_probe_calls": 0}
            self.recall = {
                "multi_instance_rescue_attempted": 0,
                "multi_instance_rescue_selected": 0,
                "partial_track_salvage_ready": 0,
                "partial_track_salvage_insufficient": 0,
            }

        def anchor_search_counters(self) -> dict[str, int]:
            return dict(self.anchor)

        def recall_rescue_counters(self) -> dict[str, int]:
            return dict(self.recall)

    class Counts:
        def __init__(self, values: dict[str, int]) -> None:
            self.values = values

        def to_dict(self) -> dict[str, int]:
            return dict(self.values)

    backend = CumulativeBackend()
    call_index = 0

    def handler(_scoped: object) -> Counts:
        nonlocal call_index
        call_index += 1
        backend.anchor["anchor_probe_calls"] += 2
        backend.recall["multi_instance_rescue_attempted"] += 1
        backend.recall["multi_instance_rescue_selected"] += 1
        values = {
            "processed": 1,
            "anchor_probe_calls": backend.anchor["anchor_probe_calls"],
            "multi_instance_rescue_attempted": backend.recall[
                "multi_instance_rescue_attempted"
            ],
            "multi_instance_rescue_selected": backend.recall[
                "multi_instance_rescue_selected"
            ],
            "partial_track_salvage_ready": int(call_index == 1),
            "partial_track_salvage_insufficient": int(call_index == 2),
        }
        return Counts(values)

    runtime = object.__new__(_StageRuntime)
    runtime.storage = object()
    runtime.stage = "segment"
    runtime._shared_lock = threading.Lock()
    runtime._segment_backend = backend
    runtime._handler = handler
    runtime._first_reference_edit_request = False

    first = runtime.run_clip("clip-a")
    second = runtime.run_clip("clip-b")

    assert first["anchor_probe_calls"] == 2
    assert second["anchor_probe_calls"] == 2
    assert first["multi_instance_rescue_attempted"] == 1
    assert second["multi_instance_rescue_attempted"] == 1
    assert first["multi_instance_rescue_selected"] == 1
    assert second["multi_instance_rescue_selected"] == 1
    assert first["partial_track_salvage_ready"] == 1
    assert first["partial_track_salvage_insufficient"] == 0
    assert second["partial_track_salvage_ready"] == 0
    assert second["partial_track_salvage_insufficient"] == 1
    assert sum(
        counts["multi_instance_rescue_attempted"] for counts in (first, second)
    ) == 2


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
