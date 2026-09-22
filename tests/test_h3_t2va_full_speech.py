from __future__ import annotations

import importlib
import json
import os
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import sam_audio_stem_shadow as frozen
from tests import test_h3_t2va_shadow as fixtures
from tests.test_h3_sam_audio_stem_shadow import _Diarization, _Qwen

finalized = fixtures.finalized
ffmpeg = fixtures.ffmpeg
setup = fixtures.setup


@pytest.fixture
def asr_jobs(tmp_path):
    path = tmp_path / "speech.wav"
    path.write_bytes(b"fake source audio")
    return [
        {
            "job_id": f"job-{index}",
            "source_audio_sha256": frozen.sha256_file(path),
            "segment": {
                "clip_uid": "clip",
                "clip_display_path": "clip.mp4",
                "media_collection_relpath": "collection",
                "media_collection_name": "collection",
                "episode_name": "episode",
                "clip_name": "clip",
                "shard_id": "shard",
                "segment_id": f"segment-{index}",
                "speaker_cluster_id": "speaker",
                "source_audio_path": str(path),
                "source_start_sample": index * 32000,
                "source_end_sample": (index + 1) * 32000,
                "start_time": float(index),
                "end_time": float(index + 1),
            },
        }
        for index in range(3)
    ]


def test_asr_prefetch_overlaps_only_next_load_and_preserves_results(
    speech, asr_jobs, tmp_path, monkeypatch
):
    before = json.dumps(asr_jobs, sort_keys=True)
    caller = threading.current_thread()
    infer_started = [threading.Event() for _ in asr_jobs]
    load_started = [threading.Event() for _ in asr_jobs]
    loaded = [threading.Event() for _ in asr_jobs]
    loader_threads = set()
    active = maximum = 0
    check_overlap = True

    def load(path, start, end, *, ffmpeg):
        index = int(start)
        assert path == Path(asr_jobs[index]["segment"]["source_audio_path"])
        assert end == start + 1 and ffmpeg == "test-ffmpeg"
        loader_threads.add(threading.current_thread())
        load_started[index].set()
        if index and check_overlap:
            assert infer_started[index - 1].wait(5)
        loaded[index].set()
        return index, 16000

    def transcribe(*, waveform, sample_rate_hz):
        nonlocal active, maximum
        assert threading.current_thread() is caller
        assert sample_rate_hz == 16000
        active += 1
        maximum = max(maximum, active)
        try:
            infer_started[waveform].set()
            if waveform < 2 and check_overlap:
                assert loaded[waveform + 1].wait(5)
            if waveform == 0 and check_overlap:
                assert not load_started[2].is_set()
            return f" exact {waveform} ", "English"
        finally:
            active -= 1

    monkeypatch.setattr(speech.asr, "load_qwen3_asr_model_input", load)
    backend = SimpleNamespace(
        _process=SimpleNamespace(poll=lambda: None), transcribe=transcribe
    )
    worker = speech._ASRWorker(backend, "test-ffmpeg")
    with worker.batch_jobs(asr_jobs):
        results = [worker.process(job, tmp_path) for job in asr_jobs]
    assert results == [
        {"text": " exact 0 ", "language": "English"},
        {"text": " exact 1 ", "language": "English"},
        {"text": " exact 2 ", "language": "English"},
    ]
    assert maximum == 1
    assert len(loader_threads) == 1 and caller not in loader_threads
    assert all(not thread.is_alive() for thread in loader_threads)
    assert json.dumps(asr_jobs, sort_keys=True) == before
    # The same worker can enter another request and retain exact direct behavior.
    check_overlap = False
    with worker.batch_jobs(asr_jobs):
        repeated = [worker.process(job, tmp_path) for job in asr_jobs]
    assert repeated == results
    assert [worker.process(job, tmp_path) for job in asr_jobs] == results


def test_asr_worker_hashes_each_source_audio_once(
    speech, asr_jobs, tmp_path, monkeypatch
):
    original = frozen.sha256_file
    calls = Counter()

    def counted(path):
        calls[str(path)] += 1
        return original(path)

    monkeypatch.setattr(frozen, "sha256_file", counted)
    monkeypatch.setattr(
        speech.asr,
        "load_qwen3_asr_model_input",
        lambda path, start, end, *, ffmpeg: (int(start), 16000),
    )
    worker = speech._ASRWorker(
        SimpleNamespace(
            _process=SimpleNamespace(poll=lambda: None),
            transcribe=lambda **kwargs: (str(kwargs["waveform"]), None),
        ),
        "test-ffmpeg",
    )
    with worker.batch_jobs(asr_jobs):
        [worker.process(job, tmp_path) for job in asr_jobs]
    source = asr_jobs[0]["segment"]["source_audio_path"]
    assert calls[source] == 1


@pytest.mark.parametrize("failure", ["loader", "hash", "inference"])
def test_asr_prefetch_failure_belongs_to_current_job(
    speech, asr_jobs, tmp_path, monkeypatch, failure
):
    failed = threading.Event()
    inferred = []
    if failure == "hash":
        asr_jobs[1]["source_audio_sha256"] = "0" * 64

    def load(path, start, end, *, ffmpeg):
        if start == 1 and failure == "loader":
            failed.set()
            raise ValueError("crop failed")
        return int(start), 16000

    def transcribe(*, waveform, sample_rate_hz):
        inferred.append(waveform)
        if waveform == 0 and failure == "loader":
            assert failed.wait(5)
        if waveform == 1 and failure == "inference":
            raise ValueError("bad sample")
        return str(waveform), None

    monkeypatch.setattr(speech.asr, "load_qwen3_asr_model_input", load)
    worker = speech._ASRWorker(
        SimpleNamespace(_process=SimpleNamespace(poll=lambda: None), transcribe=transcribe),
        "test-ffmpeg",
    )
    with worker.batch_jobs(asr_jobs):
        assert worker.process(asr_jobs[0], tmp_path) == {"text": "0", "language": None}
        expected = {
            "loader": "crop failed",
            "hash": "audio changed",
            "inference": "bad sample",
        }
        with pytest.raises(ValueError, match=expected[failure]):
            worker.process(asr_jobs[1], tmp_path)
        assert worker.process(asr_jobs[2], tmp_path) == {"text": "2", "language": None}
    assert inferred == ([0, 1, 2] if failure == "inference" else [0, 2])


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
def test_asr_prefetch_cleans_up_after_base_exception(
    speech, asr_jobs, tmp_path, monkeypatch, error
):
    next_loaded = threading.Event()
    threads = set()

    def load(path, start, end, *, ffmpeg):
        threads.add(threading.current_thread())
        if start == 1:
            next_loaded.set()
        return int(start), 16000

    def transcribe(**kwargs):
        assert next_loaded.wait(5)
        raise error("abort")

    monkeypatch.setattr(speech.asr, "load_qwen3_asr_model_input", load)
    backend = SimpleNamespace(
        _process=SimpleNamespace(poll=lambda: None), transcribe=transcribe
    )
    worker = speech._ASRWorker(backend, "test-ffmpeg")
    with pytest.raises(error, match="abort"), worker.batch_jobs(asr_jobs):
        worker.process(asr_jobs[0], tmp_path)
    assert threads and all(not thread.is_alive() for thread in threads)
    backend.transcribe = lambda **kwargs: ("recovered", None)
    with worker.batch_jobs([asr_jobs[2]]):
        assert worker.process(asr_jobs[2], tmp_path)["text"] == "recovered"
    with worker.batch_jobs([]):
        pass


@pytest.fixture
def speech(monkeypatch):
    module = importlib.import_module("r2v_data_v2.h3.t2va_full_speech")
    # Factories normally run in child processes; direct tests must restore env.
    monkeypatch.setenv("DIARIZEN_DEVICE", "cuda:5")
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cuda:6")
    monkeypatch.setattr(
        module,
        "_diar_configuration",
        lambda: {
            "provenance": _Diarization.provenance.model_dump(mode="json"),
            "environment": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_asr_configuration",
        lambda: {
            "configuration": _Qwen.configuration.model_dump(mode="json"),
            "environment": {},
        },
    )
    return module


class Executor:
    """CPU worker/cache substitute; only ready process results are retained."""

    def __init__(self):
        self.cache = {}
        self.calls = Counter()
        self.fail = set()
        self.batches = []

    def __call__(self, *, stage_root, jobs, gpu_ids, factory, configuration, **kwargs):
        self.batches.append((factory, jobs, configuration, kwargs))
        assert gpu_ids == ["2", "7"]
        results = {}
        for job in jobs:
            key = json.dumps([factory, configuration, job], sort_keys=True)
            if key not in self.cache:
                self.calls[job["job_id"]] += 1
                if job["job_id"] in self.fail:
                    results[job["job_id"]] = {
                        "status": "failed",
                        "result": None,
                        "failure_reason": "sample failure",
                    }
                    continue
                if factory.endswith(":diarizen_worker"):
                    result = {
                        "segments": [
                            {
                                "start_time": 0.0,
                                "end_time": 1.02,
                                "speaker_label": "A",
                            }
                        ]
                    }
                else:
                    result = {"text": "  exact transcript!  ", "language": "English"}
                self.cache[key] = {
                    "status": "ready",
                    "result": result,
                    "failure_reason": None,
                }
            results[job["job_id"]] = self.cache[key]
        return results


def test_frozen_lineage_partial_failure_resume(speech, finalized, tmp_path, ffmpeg):
    audio, run_id = finalized
    shadow = frozen.stem_shadow_root(audio, run_id)
    source_before = (audio / "diarization/inventory.json").read_bytes()
    executor = Executor()
    args = (audio, run_id, tmp_path / "workers", ["2", "7"], True)
    speech.run_diarizen(*args, ffmpeg=ffmpeg, execute=executor)
    jobs = executor.batches[-1][1]
    assert len(jobs) == 2  # The frozen fixture has one failed upstream separation.
    assert all(not j["target"]["visual_references"] for j in jobs)
    assert all(j["target"]["target_audio_binding_path"] is None for j in jobs)
    executor.fail.add(jobs[1]["job_id"])
    executor.cache.clear()
    speech.run_diarizen(*args, ffmpeg=ffmpeg, execute=executor)
    provenance, _, _ = frozen.validate_stem_diarization_lineage(
        shadow / "diarization",
        expected_shadow_root=shadow,
    )
    assert provenance.failed_clip_uids == [jobs[1]["job_id"]]
    assert provenance.binding_evidence_mode == "legacy_lr_asd"
    assert provenance.schema_version.endswith(".5")
    before = executor.calls.copy()
    executor.fail.clear()
    speech.run_diarizen(*args, ffmpeg=ffmpeg, execute=executor)
    assert executor.calls - before == Counter({jobs[1]["job_id"]: 1})
    assert executor.batches[0][1] == executor.batches[-1][1]
    raw = [
        json.loads(line)
        for line in (shadow / "diarization/raw_segments.jsonl").read_text().splitlines()
    ]
    assert all(row["end_time"] == 1.0 for row in raw)
    assert all(row["boundary_reconciliation"]["end_clamped"] for row in raw)
    speech.run_asr(*args, ffmpeg=ffmpeg, execute=executor)
    asr_jobs = executor.batches[-1][1]
    executor.fail.add(asr_jobs[0]["job_id"])
    executor.cache.clear()
    speech.run_asr(*args, ffmpeg=ffmpeg, execute=executor)
    rows = [
        json.loads(line)
        for line in (shadow / "asr/segments.jsonl").read_text().splitlines()
    ]
    assert Counter(row["status"] for row in rows) == {"failed": 1, "transcribed": 1}
    before = executor.calls.copy()
    executor.fail.clear()
    speech.run_asr(*args, ffmpeg=ffmpeg, execute=executor)
    assert executor.calls - before == Counter({asr_jobs[0]["job_id"]: 1})
    asr, diar = frozen.validate_stem_asr_lineage(
        shadow / "asr", expected_shadow_root=shadow
    )
    assert asr.clip_uids == diar.usable_clip_uids
    assert asr.asr_source_kind == "resolved_speech_stem_segments"
    rows = [
        json.loads(line)
        for line in (shadow / "asr/segments.jsonl").read_text().splitlines()
    ]
    assert all(row["text"] == "  exact transcript!  " for row in rows)
    assert (audio / "diarization/inventory.json").read_bytes() == source_before


@pytest.mark.parametrize(
    "error,process",
    [
        (TimeoutError("timeout"), SimpleNamespace(poll=lambda: None)),
        (RuntimeError("exited"), SimpleNamespace(poll=lambda: 9)),
        (RuntimeError("gone"), None),
        (BrokenPipeError("pipe"), SimpleNamespace(poll=lambda: None)),
        (RuntimeError("CUDA out of memory"), SimpleNamespace(poll=lambda: None)),
        (RuntimeError("CUDA OUT OF MEMORY"), SimpleNamespace(poll=lambda: None)),
    ],
)
def test_nested_worker_failure_is_fatal(speech, error, process):
    backend = SimpleNamespace(_process=process)

    def fail():
        raise error

    with pytest.raises(SystemExit):
        speech._inference(backend, fail)


def test_live_worker_sample_failure_is_retryable(speech):
    backend = SimpleNamespace(_process=SimpleNamespace(poll=lambda: None))

    def fail():
        raise ValueError("bad sample")

    with pytest.raises(ValueError, match="bad sample"):
        speech._inference(backend, fail)


def test_factories_load_once_and_use_frozen_crops(
    speech, finalized, tmp_path, ffmpeg, monkeypatch
):
    from tools import run_h3_diarization_binding as diar_cli
    from tools import run_h3_stem_qwen3_asr_shadow as asr_cli

    audio, run_id = finalized
    executor = Executor()
    args = (audio, run_id, tmp_path / "workers", ["2", "7"], True)
    parent_environment = dict(os.environ)
    speech.run_diarizen(*args, ffmpeg=ffmpeg, execute=executor)
    diar_jobs = executor.batches[-1][1]
    speech.run_asr(*args, ffmpeg=ffmpeg, execute=executor)
    asr_jobs = executor.batches[-1][1]
    assert dict(os.environ) == parent_environment

    class DiarBackend(_Diarization):
        starts = 0
        closes = 0

        def __init__(self):
            super().__init__()
            self.environment = {"CUDA_VISIBLE_DEVICES": "0"}

        def __enter__(self):
            self.starts += 1
            self._process = SimpleNamespace(poll=lambda: None)
            return self

        def close(self):
            self.closes += 1

    class ASRBackend(_Qwen):
        starts = 0
        closes = 0
        calls = 0

        def __enter__(self):
            self.starts += 1
            self._process = SimpleNamespace(poll=lambda: None)
            return self

        def close(self, **kwargs):
            self.closes += 1

        def transcribe(self, *, waveform, sample_rate_hz):
            self.calls += 1
            assert (
                waveform.size == 16000
            )  # Frozen loader cropped the clamped 1s segment.
            return super().transcribe(waveform=waveform, sample_rate_hz=sample_rate_hz)

    db, ab = DiarBackend(), ASRBackend()

    def runtime(*, output_root, input_profile):
        assert input_profile == "canonical_32k_stereo"
        assert os.environ["DIARIZEN_DEVICE"] == "cuda:0"
        return db, output_root

    monkeypatch.setattr(diar_cli, "_runtime_backend", runtime)
    monkeypatch.setattr(asr_cli, "_isolated_backend", lambda: ab)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with speech.diarizen_worker(speech._diar_configuration()) as worker:
        for job in diar_jobs:
            assert worker.process(job, tmp_path)["segments"]
        assert db.environment["CUDA_VISIBLE_DEVICES"] == "7"
    with speech.asr_worker({**speech._asr_configuration(), "ffmpeg": ffmpeg}) as worker:
        for job in asr_jobs:
            assert worker.process(job, tmp_path)["text"] == "exact transcript"
    assert (db.starts, db.closes, len(db.paths)) == (1, 1, 2)
    assert (ab.starts, ab.closes, ab.calls) == (1, 1, 2)


def test_invalid_diarization_not_cacheable(speech, finalized, tmp_path, ffmpeg):
    audio, run_id = finalized
    executor = Executor()
    speech.run_diarizen(
        audio,
        run_id,
        tmp_path / "workers",
        ["2", "7"],
        True,
        ffmpeg=ffmpeg,
        execute=executor,
    )
    job = executor.batches[-1][1][0]
    backend = _Diarization()
    backend._process = SimpleNamespace(poll=lambda: None)
    backend.diarize = lambda **kwargs: [
        speech.diar.DiarizationBackendSegment(
            start_time=2.0,
            end_time=3.0,
            speaker_label="A",
        )
    ]
    with pytest.raises(speech.diar.DiarizationBackendFailure, match="EOF"):
        speech._DiarWorker(backend).process(job, tmp_path)


@pytest.mark.parametrize(
    "result", [None, {}, {"text": 2, "language": None}, {"text": "ok", "language": []}]
)
def test_invalid_asr_not_cacheable(speech, result):
    with pytest.raises(ValueError, match="invalid Qwen3"):
        speech._asr_result(result)


def test_model_configuration_changes_cache_identity(monkeypatch, tmp_path):
    speech = importlib.import_module("r2v_data_v2.h3.t2va_full_speech")
    speech._diar_configuration_cached.cache_clear()
    speech._asr_configuration_cached.cache_clear()
    (tmp_path / "diar").mkdir()
    (tmp_path / "asr").mkdir()

    monkeypatch.setenv("DIARIZEN_MODEL_PATH", str(tmp_path / "diar"))
    monkeypatch.setenv("DIARIZEN_DEVICE", "cuda:5")
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", str(tmp_path / "asr"))
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cuda:6")
    monkeypatch.setenv("QWEN3_ASR_DTYPE", "bfloat16")
    before = dict(os.environ)

    dc = speech._diar_configuration()
    ac = speech._asr_configuration()
    assert dc == speech._diar_configuration()
    assert ac == speech._asr_configuration()
    assert ac["configuration"]["device"] == "cuda:0"
    assert dc["environment"]["DIARIZEN_DEVICE"] == "cuda:0"
    assert dict(os.environ) == before

    # Runtime provenance is logical identity only; it never reads model weights.
    assert dc["provenance"]["model_fingerprint"] == speech._logical_model_fingerprint(
        tmp_path / "diar",
        dc["provenance"]["model_identifier"],
    )
    assert ac["model_fingerprint"] == speech._logical_model_fingerprint(
        tmp_path / "asr",
        "qwen3-asr",
    )

    monkeypatch.setenv("QWEN3_ASR_DTYPE", "float32")
    changed_asr = speech._asr_configuration()
    assert ac != changed_asr
    assert ac["model_fingerprint"] == changed_asr["model_fingerprint"]

    monkeypatch.setenv("DIARIZEN_MODEL_IDENTIFIER", "changed-model")
    changed_diar = speech._diar_configuration()
    assert dc != changed_diar
    assert dc["provenance"]["model_fingerprint"] != changed_diar["provenance"]["model_fingerprint"]


def test_startup_failure_is_fatal(speech, monkeypatch):
    from tools import run_h3_stem_qwen3_asr_shadow as cli

    def fail():
        raise RuntimeError("startup failed")

    monkeypatch.setattr(cli, "_isolated_backend", fail)
    with (
        pytest.raises(SystemExit, match="initialization failed"),
        speech.asr_worker(speech._asr_configuration()),
    ):
        pytest.fail("worker must not yield")


@contextmanager
def cpu_diar_worker(configuration):
    """Importable fake model for exercising the real executor's receipt cache."""
    speech = importlib.import_module("r2v_data_v2.h3.t2va_full_speech")

    class Backend(_Diarization):
        def diarize(self, *, clip_uid, audio_path):
            with Path(configuration["calls"]).open("a") as handle:
                handle.write(clip_uid + "\n")
            if (
                clip_uid == configuration["fail"]
                and not Path(configuration["recover"]).exists()
            ):
                return [
                    speech.diar.DiarizationBackendSegment(
                        start_time=9.0,
                        end_time=10.0,
                        speaker_label="invalid",
                    )
                ]
            return super().diarize(clip_uid=clip_uid, audio_path=audio_path)

    backend = Backend()
    backend._process = SimpleNamespace(poll=lambda: None)
    yield speech._DiarWorker(backend)


def test_real_executor_retries_semantic_failures(speech, finalized, tmp_path, ffmpeg):
    from r2v_data_v2.h3.t2va_full_workers import execute_stage

    audio, run_id = finalized
    executor = Executor()
    speech.run_diarizen(
        audio,
        run_id,
        tmp_path / "plan",
        ["2", "7"],
        True,
        ffmpeg=ffmpeg,
        execute=executor,
    )
    jobs = executor.batches[-1][1]
    calls, recover = tmp_path / "calls", tmp_path / "recover"
    arguments = {
        "stage_root": tmp_path / "real-workers",
        "jobs": jobs,
        "gpu_ids": ["2", "7"],
        "factory": f"{__name__}:cpu_diar_worker",
        "configuration": {
            "fail": jobs[0]["job_id"],
            "calls": str(calls),
            "recover": str(recover),
        },
        "environment": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1])
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        },
    }
    first = execute_stage(**arguments)
    assert first[jobs[0]["job_id"]]["status"] == "failed"
    assert first[jobs[1]["job_id"]]["status"] == "ready"
    recover.touch()
    second = execute_stage(**arguments)
    assert all(row["status"] == "ready" for row in second.values())
    assert Counter(calls.read_text().splitlines()) == {
        jobs[0]["job_id"]: 2,
        jobs[1]["job_id"]: 1,
    }


def test_all_upstream_speech_failures_publish_empty_stages(
    speech, setup, tmp_path, ffmpeg, monkeypatch
):
    from r2v_data_v2.h3 import t2va_full_workers as workers

    production = fixtures.prepare_t2va_audio(
        fixtures.select_t2va_shots(fixtures.shot_manifest(tmp_path)),
        output_root=tmp_path / "all-failed-audio",
        audio_backend=fixtures.Audio(),
    )
    inventory = fixtures.auk_tests.auk.build_auk_inventory(
        audio_production_root=production,
        shadow_run_id=setup.shadow_run_id,
        case_manifest_path=production / "case_manifest.json",
        configuration=setup.model_configuration,
    )
    monkeypatch.setattr(
        fixtures.auk_tests.FakeBackend,
        "generate",
        lambda *args: {
            "status": "failed",
            "reason": "all upstream speech inference failed",
            "model_runtime_seconds": 0.1,
        },
    )
    root, (_, records, _) = fixtures._resolve(inventory, tmp_path, ffmpeg)
    assert len(records) == 3
    assert all(record.status == "failed" for record in records)
    calls = []

    def empty_execute(*, jobs, factory, **kwargs):
        assert jobs == []
        calls.append(factory)
        return {}

    monkeypatch.setattr(workers, "execute_stage", empty_execute)
    args = (production, setup.shadow_run_id, tmp_path / "workers", ["2", "7"], True)
    diar_result = speech.run_diarizen(*args, ffmpeg=ffmpeg)
    asr_result = speech.run_asr(*args, ffmpeg=ffmpeg)
    assert diar_result["job_count"] == asr_result["job_count"] == 0
    assert asr_result["summary"]["segment_count"] == 0
    asr_provenance, diar_provenance = frozen.validate_stem_asr_lineage(
        root.parent / "asr", expected_shadow_root=root.parent
    )
    assert diar_provenance.usable_clip_uids == []
    assert asr_provenance.clip_uids == []
    assert len(diar_provenance.skipped_clips) == 3
    assert len(asr_provenance.skipped_clips) == 3
    for relative in (
        "diarization/raw_segments.jsonl",
        "diarization/readable_targets.jsonl",
        "diarization/readable_segments.jsonl",
        "asr/segments.jsonl",
    ):
        assert (root.parent / relative).read_bytes() == b""
    assert calls == [f"{speech.__name__}:diarizen_worker", f"{speech.__name__}:asr_worker"]
