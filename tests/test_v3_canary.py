from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import load_config
from tools.run_v3_canary import (
    CANARY_STAGES,
    CanaryPipelineError,
    PipelineExecution,
    build_canary_paths,
    prepare_canary_artifacts,
    run_canary,
    run_pipeline_process,
    select_source_records,
)


def _roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    models = writable / "models"
    for path in (writable, dataset, pretrained, models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", models)
    return writable, dataset, pretrained


def _record(
    parent: str,
    shot_index: int,
    *,
    source_video_path: str,
) -> dict[str, object]:
    return {
        "video_path": f"{parent}_{shot_index}.mp4",
        "source_video_id": parent,
        "source_video_path": source_video_path,
        "shot_index": shot_index,
        "start_frame": shot_index * 10,
        "end_frame": shot_index * 10 + 9,
        "num_frames": 10,
        "start_time": float(shot_index),
        "end_time": float(shot_index + 1),
        "duration": 1.0,
        "caption": f"caption {parent} {shot_index}",
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    groups: list[tuple[str, str, int]],
) -> dict[str, Path]:
    writable, dataset, pretrained = _roots(tmp_path, monkeypatch)
    root = dataset / "jea"
    clips = root / "clips_clean_cropped"
    videos = root / "source_videos"
    clips.mkdir(parents=True)
    videos.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for parent, source_video_path, count in groups:
        source_file = videos / source_video_path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"source-video")
        for shot_index in range(count):
            record = _record(
                parent,
                shot_index,
                source_video_path=source_video_path,
            )
            (clips / str(record["video_path"])).write_bytes(b"processed-shot")
            records.append(record)
    source = root / "shots_f03_motion.jsonl"
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    base_config = writable / "r2v_v3_configs" / "base.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(
        yaml.safe_dump(
            {
                "dataset_json": str(source),
                "run_root": str(writable / "runs" / "base"),
                "export_root": str(writable / "exports" / "base"),
                "source": {"limit": 1},
                "qwen": {
                    "annotation": {"model": str(pretrained / "Qwen" / "judge")},
                    "instruction_writer": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                    "candidate_judge": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                    "background_remove_judge": {
                        "model": str(pretrained / "Qwen" / "judge")
                    },
                },
                "remove": {
                    "base_model_path": str(pretrained / "Qwen" / "edit"),
                    "adapter_path": str(writable / "models" / "remover"),
                },
                "runtime": {"mode": "streaming_v1"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "writable": writable,
        "dataset": dataset,
        "source": source,
        "clips": clips,
        "videos": videos,
        "base_config": base_config,
    }


def _selection_arguments(fixture: dict[str, Path]) -> dict[str, Path]:
    return {
        "source_jsonl": fixture["source"],
        "clips_root": fixture["clips"],
        "source_videos_root": fixture["videos"],
    }


def test_selection_excludes_unicode_and_uses_next_full_contiguous_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[
            ("movie-a", "01/丁宝桢/01 4K.mkv", 20),
            ("movie-b", "02/short.mov", 5),
            ("movie-c", "03/另一部电影/正片.webm", 20),
        ],
    )

    selection = select_source_records(
        **_selection_arguments(fixture),
        count=20,
        exclude_source_names=["丁宝桢"],
    )

    assert selection.source_video == (
        fixture["videos"] / "03/另一部电影/正片.webm"
    )
    assert [record["source_index"] for record in selection.records] == list(
        range(25, 45)
    )
    assert len({record["parent_video_id"] for record in selection.records}) == 1
    assert {
        record["metadata"]["source_relative_source_video_path"]
        for record in selection.records
    } == {"03/另一部电影/正片.webm"}


@pytest.mark.parametrize("absolute", (False, True))
def test_selection_supports_explicit_relative_or_absolute_source_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute: bool,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[
            ("movie-a", "甲/电影甲.mkv", 3),
            ("movie-b", "乙/电影乙.customvideo", 3),
        ],
    )
    relative = Path("乙/电影乙.customvideo")
    requested = fixture["videos"] / relative if absolute else relative

    selection = select_source_records(
        **_selection_arguments(fixture),
        count=2,
        source_video=requested,
    )

    assert selection.source_video == fixture["videos"] / relative
    assert [record["source_index"] for record in selection.records] == [3, 4]


def test_selection_ignores_partial_non_newline_eof_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "电影/源.mkv", 1)],
    )
    partial = _record("movie-a", 1, source_video_path="电影/源.mkv")
    (fixture["clips"] / "movie-a_1.mp4").write_bytes(b"processed-shot")
    with fixture["source"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(partial, ensure_ascii=False)[:-5])

    with pytest.raises(ValueError, match="no source video has 2 consecutive"):
        select_source_records(
            **_selection_arguments(fixture),
            count=2,
        )


def test_canary_artifacts_are_isolated_and_use_fixed_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 2)],
    )
    formal_cursor = (
        fixture["writable"]
        / "r2v_v3_configs"
        / "production"
        / "jea_motion_v1"
        / "prod-v1"
        / "state"
        / "cursor.json"
    )
    formal_cursor.parent.mkdir(parents=True)
    formal_cursor.write_bytes(b"formal-cursor-sentinel")
    selection = select_source_records(**_selection_arguments(fixture), count=2)
    paths = build_canary_paths(
        selection,
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 0, tzinfo=timezone.utc),
    )

    prepare_canary_artifacts(
        selection=selection,
        paths=paths,
        base_config=fixture["base_config"],
        **_selection_arguments(fixture),
    )

    assert paths.tag == "canary-e2e2-jea-20260820-150000-s000000000-000000001"
    assert "prod-v1" not in paths.config_root.parts
    assert formal_cursor.read_bytes() == b"formal-cursor-sentinel"
    assert not (paths.config_root / "state" / "cursor.json").exists()
    assert {path.name for path in paths.config_root.iterdir()} == {
        "source.yaml",
        "selection.jsonl",
        "shard-000000000-000000001.yaml",
    }
    descriptor = yaml.safe_load(paths.source_yaml.read_text(encoding="utf-8"))
    assert set(descriptor) == {
        "schema_version",
        "source_adapter",
        "source_jsonl",
        "base_config_path",
        "base_config_sha256",
        "base_config_fingerprint",
        "clips_root",
        "source_videos_root",
        "path_probe_records",
        "shard_size",
    }
    assert descriptor["source_adapter"] == "jea_video_motion_v1"
    assert descriptor["source_jsonl"] == str(fixture["source"])
    assert descriptor["base_config_path"] == str(fixture["base_config"])
    assert descriptor["clips_root"] == str(fixture["clips"])
    assert descriptor["source_videos_root"] == str(fixture["videos"])
    assert descriptor["path_probe_records"] == 2
    assert descriptor["shard_size"] == 2
    config = load_config(paths.shard_config)
    assert config.source.selection_mode == "fixed_selection_v1"
    assert config.source.selection_manifest == paths.selection_manifest
    assert config.source.start_index == 0
    assert config.source.limit is None
    assert config.run_root == paths.run_root
    assert config.export_root == paths.shard_export_root


def test_sam3_compile_flag_changes_only_isolated_canary_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 2)],
    )
    selection = select_source_records(**_selection_arguments(fixture), count=2)
    base_before = fixture["base_config"].read_bytes()
    base_value = yaml.safe_load(base_before)
    paths = build_canary_paths(
        selection,
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 0, tzinfo=timezone.utc),
        sam3_compile=True,
    )

    prepare_canary_artifacts(
        selection=selection,
        paths=paths,
        base_config=fixture["base_config"],
        sam3_compile=True,
        **_selection_arguments(fixture),
    )

    assert paths.tag.endswith("-sam3compile")
    assert fixture["base_config"].read_bytes() == base_before
    generated = yaml.safe_load(paths.shard_config.read_text(encoding="utf-8"))
    assert generated["runtime"]["sam3_compile_enabled"] is True
    assert generated.get("subject_attribute_gme") == base_value.get(
        "subject_attribute_gme"
    )
    assert load_config(paths.shard_config).runtime.sam3_compile_enabled is True


def test_attribute_completion_flag_writes_isolated_completion_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "source.mkv", 2)],
    )
    base = yaml.safe_load(fixture["base_config"].read_text(encoding="utf-8"))
    boogu_root = fixture["writable"] / "boogu"
    base["reference_edit"] = {
        "python_executable": str(boogu_root / "python"),
        "code_root": str(boogu_root / "code"),
        "model_path": str(boogu_root / "model"),
    }
    fixture["base_config"].write_text(
        yaml.safe_dump(base, sort_keys=False),
        encoding="utf-8",
    )
    selection = select_source_records(**_selection_arguments(fixture), count=2)
    paths = build_canary_paths(
        selection,
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 2, tzinfo=timezone.utc),
    )
    prepare_canary_artifacts(
        selection=selection,
        paths=paths,
        base_config=fixture["base_config"],
        attribute_completion=True,
        **_selection_arguments(fixture),
    )

    generated = load_config(paths.shard_config)
    assert generated.subject_attributes.completion.enabled is True
    assert generated.reference_edit.enabled is False
    assert generated.runtime.gpu_workers.subject_attributes_completion == "6"
    assert generated.subject_attribute_gme.enabled is False


def test_dual_sam3_and_qwen_overrides_change_only_isolated_canary_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 2)],
    )
    selection = select_source_records(**_selection_arguments(fixture), count=2)
    base_before = fixture["base_config"].read_bytes()
    base_config = load_config(fixture["base_config"])
    paths = build_canary_paths(
        selection,
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 3, tzinfo=timezone.utc),
    )

    prepare_canary_artifacts(
        selection=selection,
        paths=paths,
        base_config=fixture["base_config"],
        dual_main_sam3=True,
        qwen_max_inflight=4,
        qwen_stage_workers=4,
        **_selection_arguments(fixture),
    )

    generated = load_config(paths.shard_config)
    assert fixture["base_config"].read_bytes() == base_before
    assert generated.runtime.gpu_workers.segment_pool == ("5", "7")
    assert generated.runtime.stage_workers.segment == 2
    assert generated.runtime.sam3_compile_enabled is False
    assert generated.runtime.subject_attributes_deferred is False
    assert generated.runtime.qwen_max_inflight == 4
    assert generated.runtime.stage_workers.annotate == 4
    assert generated.runtime.stage_workers.pair == 4
    assert generated.runtime.stage_workers.reference_integrity == 4
    assert generated.runtime.stage_workers.instruct == 4
    assert generated.runtime.stage_workers.subject_attributes == 4
    for stage in ("frames", "rank", "background", "remove", "reference_edit"):
        assert getattr(generated.runtime.stage_workers, stage) == getattr(
            base_config.runtime.stage_workers, stage
        )
    assert generated.subject_attribute_gme == base_config.subject_attribute_gme

    inline_paths = build_canary_paths(
        selection,
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 4, tzinfo=timezone.utc),
    )
    prepare_canary_artifacts(
        selection=selection,
        paths=inline_paths,
        base_config=fixture["base_config"],
        qwen_stage_workers=4,
        **_selection_arguments(fixture),
    )
    inline = load_config(inline_paths.shard_config)
    assert inline.runtime.stage_workers.subject_attributes == 4
    assert inline.runtime.stage_workers.segment == 1
    assert inline.runtime.gpu_workers.segment_pool == ()


def test_dual_main_sam3_allows_inline_attributes_before_source_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_canary(
            dual_main_sam3=True,
            source_jsonl=tmp_path / "missing.jsonl",
        )


def test_deferred_canary_reports_zero_canonical_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 1)],
    )

    def pipeline(
        command: list[str],
        *,
        log_path: Path,
        **_kwargs: object,
    ) -> PipelineExecution:
        config = load_config(Path(command[command.index("--config") + 1]))
        config.export_root.mkdir(parents=True)
        (config.export_root / "dataset.json").write_text(
            json.dumps({"sample_count": 1, "reference_count": 1}),
            encoding="utf-8",
        )
        (config.export_root / "samples.jsonl").write_text("", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return PipelineExecution(
            returncode=0,
            output="{}\n",
            result={
                "runtime": {"failed_tasks": []},
                "subject_attributes": {"deferred": True},
            },
        )

    def compact(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_root"]))
        (output / "samples.jsonl").write_text("", encoding="utf-8")
        (output / "references").mkdir()
        return {
            "total_samples": 1,
            "total_visual_references": 1,
            "total_attribute_references": 0,
            "total_enriched_samples": 0,
        }

    summary = run_canary(
        **_selection_arguments(fixture),
        base_config=fixture["base_config"],
        count=1,
        now=datetime(2026, 8, 20, 15, 0, 5, tzinfo=timezone.utc),
        pipeline_runner=pipeline,
        compactor=compact,
        defer_subject_attributes=True,
    )

    assert summary["subject_attributes_deferred"] is True
    assert summary["canonical_attribute_references"] == 0


def test_pipeline_failure_prevents_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 2)],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "configured-by-caller")
    compact_calls: list[dict[str, object]] = []

    def failed_pipeline(
        command: list[str],
        *,
        log_path: Path,
        cwd: Path,
        environment: dict[str, str],
    ) -> PipelineExecution:
        assert cwd.name == "R2V_DATA_V2"
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["CUDA_VISIBLE_DEVICES"] == "configured-by-caller"
        assert command[-1] == "--profile"
        assert command[command.index("--stages") + 1] == ",".join(CANARY_STAGES)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("pipeline failed\n", encoding="utf-8")
        return PipelineExecution(returncode=7, output="pipeline failed\n", result=None)

    with pytest.raises(CanaryPipelineError) as error:
        run_canary(
            **_selection_arguments(fixture),
            base_config=fixture["base_config"],
            count=2,
            now=datetime(2026, 8, 20, 15, 0, 1, tzinfo=timezone.utc),
            pipeline_runner=failed_pipeline,
            compactor=lambda **kwargs: compact_calls.append(kwargs) or {},
        )

    captured = capsys.readouterr()
    assert error.value.returncode == 7
    assert compact_calls == []
    assert "Selected source video:" in captured.out
    assert "source_index: 0..1" in captured.out
    assert "status: FAIL" in captured.err
    assert "canary.log" in captured.err


def test_pipeline_runner_does_not_precreate_nonempty_run_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-root"
    script = (
        "import pathlib,sys; "
        "root=pathlib.Path(sys.argv[1]); "
        "assert not root.exists(); "
        "root.mkdir(parents=True); "
        "(root/'run.json').write_text('{}'); "
        "print('pipeline output')"
    )

    execution = run_pipeline_process(
        [sys.executable, "-c", script, str(run_root)],
        log_path=run_root / "canary.log",
        cwd=tmp_path,
        environment=os.environ.copy(),
    )

    assert execution.returncode == 0
    assert execution.output == "pipeline output\n"
    assert (run_root / "canary.log").read_text(encoding="utf-8") == (
        "pipeline output\n"
    )


def test_successful_mocked_pipeline_compacts_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        groups=[("movie-a", "目录/源视频.mkv", 2)],
    )
    compact_calls: list[dict[str, object]] = []

    def successful_pipeline(
        command: list[str],
        *,
        log_path: Path,
        cwd: Path,
        environment: dict[str, str],
    ) -> PipelineExecution:
        del cwd, environment
        config = load_config(Path(command[command.index("--config") + 1]))
        config.export_root.mkdir(parents=True)
        (config.export_root / "dataset.json").write_text(
            json.dumps({"sample_count": 1, "reference_count": 2}),
            encoding="utf-8",
        )
        (config.export_root / "samples.jsonl").write_text(
            json.dumps(
                {
                    "references": [
                        {"type": "entity"},
                        {"type": "background"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("pipeline passed\n", encoding="utf-8")
        profiling = config.resolved_run_root / "profiling"
        profiling.mkdir(parents=True)
        (profiling / "summary.json").write_text(
            json.dumps(
                {
                    "qwen_calls": 9,
                    "qwen_gate_wait_seconds_total": 1.8,
                    "qwen_gate_wait_seconds_mean": 0.2,
                    "qwen_gate_wait_seconds_max": 0.7,
                    "qwen_max_local_inflight_observed": 4,
                    "qwen_slot_usage": {"0": 3, "1": 2, "2": 2, "3": 2},
                }
            ),
            encoding="utf-8",
        )
        return PipelineExecution(
            returncode=0,
            output="{}\n",
            result={
                "runtime": {"failed_tasks": []},
                "segment": {
                    "sam3_compile_requested": True,
                    "sam3_compile_effective": True,
                    "sam3_compile_fallbacks": 0,
                    "sam3_compile_failure_reason": None,
                    "sam3_predictor_startup_seconds": 3.0,
                    "sam3_compile_warmup_seconds": 4.0,
                    "sam3_segment_model_call_time_seconds": 8.0,
                    "sam3_segment_clips": 2,
                    "sam3_segment_entities": 3,
                    "first_segment_clip_seconds": 6.0,
                    "steady_state_segment_mean_seconds": 2.0,
                    "segment_worker_pool_size": 2,
                    "segment_worker_requests_by_gpu": {"5": 1, "7": 1},
                    "segment_worker_service_seconds_by_gpu": {
                        "5": 7.0,
                        "7": 6.0,
                    },
                    "sam_pool_main_requests_by_gpu": {"5": 1, "7": 1},
                    "sam_pool_attribute_probe_requests_by_gpu": {
                        "5": 2,
                        "7": 1,
                    },
                    "sam_pool_main_service_seconds_by_gpu": {
                        "5": 7.0,
                        "7": 6.0,
                    },
                    "sam_pool_attribute_service_seconds_by_gpu": {
                        "5": 3.0,
                        "7": 2.0,
                    },
                    "sam_pool_main_wait_seconds_total": 0.4,
                    "sam_pool_attribute_wait_seconds_total": 0.7,
                    "sam_pool_max_concurrent_requests": 2,
                },
                "remove": {
                    "candidates_generated": 3,
                    "candidates_rejected": 2,
                    "ready_removed": 1,
                },
                "subject_attributes_summary": {
                    "gme_calls": 5,
                    "gme_candidates_screened": 4,
                    "gme_candidates_passed": 3,
                    "gme_candidates_rejected": 1,
                    "gme_retried_next_frame": 1,
                    "gme_failures": 1,
                    "gme_model_call_time_seconds": 2.75,
                },
            },
        )

    def successful_compactor(**kwargs: object) -> dict[str, object]:
        compact_calls.append(kwargs)
        output_root = Path(str(kwargs["output_root"]))
        (output_root / "samples.jsonl").write_text(
            json.dumps(
                {
                    "references": [
                        {"kind": "subject"},
                        {"kind": "background"},
                        {"kind": "attribute"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (output_root / "references").mkdir()
        return {
            "total_samples": 1,
            "total_visual_references": 2,
            "total_attribute_references": 1,
            "total_enriched_samples": 1,
        }

    clock_values = iter((100.0, 112.25))
    summary = run_canary(
        **_selection_arguments(fixture),
        base_config=fixture["base_config"],
        count=2,
        now=datetime(2026, 8, 20, 15, 0, 2, tzinfo=timezone.utc),
        pipeline_runner=successful_pipeline,
        compactor=successful_compactor,
        clock=lambda: next(clock_values),
    )

    assert len(compact_calls) == 1
    call = compact_calls[0]
    assert Path(str(call["runs_root"])).name == summary["tag"]
    assert Path(str(call["source_yaml"])).name == "source.yaml"
    assert summary["status"] == "PASS"
    assert summary["selected_source_indices"] == "0-1"
    assert summary["input_clips"] == 2
    assert summary["elapsed_seconds"] == 12.25
    assert summary["elapsed_hms"] == "00:00:12"
    assert summary["sam3_compile_requested"] is True
    assert summary["sam3_compile_effective"] is True
    assert summary["sam3_compile_fallbacks"] == 0
    assert summary["sam3_predictor_startup_seconds"] == 3.0
    assert summary["sam3_compile_warmup_seconds"] == 4.0
    assert summary["sam3_segment_model_call_time_seconds"] == 8.0
    assert summary["sam3_segment_clips"] == 2
    assert summary["sam3_segment_entities"] == 3
    assert summary["first_segment_clip_seconds"] == 6.0
    assert summary["steady_state_segment_mean_seconds"] == 2.0
    assert summary["segment_worker_pool_size"] == 2
    assert summary["segment_worker_requests_by_gpu"] == {"5": 1, "7": 1}
    assert summary["sam_pool_main_requests_by_gpu"] == {"5": 1, "7": 1}
    assert summary["sam_pool_attribute_probe_requests_by_gpu"] == {
        "5": 2,
        "7": 1,
    }
    assert summary["sam_pool_main_service_seconds_by_gpu"] == {
        "5": 7.0,
        "7": 6.0,
    }
    assert summary["sam_pool_attribute_service_seconds_by_gpu"] == {
        "5": 3.0,
        "7": 2.0,
    }
    assert summary["sam_pool_main_wait_seconds_total"] == 0.4
    assert summary["sam_pool_attribute_wait_seconds_total"] == 0.7
    assert summary["sam_pool_max_concurrent_requests"] == 2
    assert summary["qwen_calls"] == 9
    assert summary["qwen_gate_wait_seconds_total"] == 1.8
    assert summary["qwen_gate_wait_seconds_mean"] == 0.2
    assert summary["qwen_gate_wait_seconds_max"] == 0.7
    assert summary["qwen_max_local_inflight_observed"] == 4
    assert summary["qwen_slot_usage"] == {"0": 3, "1": 2, "2": 2, "3": 2}
    assert summary["visual_samples"] == 1
    assert summary["visual_references"] == 2
    assert summary["visual_background_references"] == 1
    assert summary["canonical_samples"] == 1
    assert summary["canonical_background_references"] == 1
    assert summary["canonical_attribute_references"] == 1
    assert summary["samples_with_background"] == 1
    assert summary["remove_candidates_generated"] == 3
    assert summary["remove_candidates_accepted"] == 1
    assert summary["remove_candidates_rejected"] == 2
    assert summary["ready_removed"] == 1
    assert summary["gme_calls"] == 5
    assert summary["gme_candidates_screened"] == 4
    assert summary["gme_candidates_passed"] == 3
    assert summary["gme_candidates_rejected"] == 1
    assert summary["gme_retried_next_frame"] == 1
    assert summary["gme_failures"] == 1
    assert summary["gme_model_call_time_seconds"] == 2.75
    assert summary["failed_tasks"] == []
    summary_path = Path(str(summary["export_root"])) / "canary_summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert Path(str(summary["references_root"])).is_dir()
