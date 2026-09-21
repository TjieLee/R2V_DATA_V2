from pathlib import Path
from typing import ClassVar

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import atomic_json, phase, read_json


def prepared_cases(tmp_path, count=3):
    cases = []
    for i in range(count):
        case = {"case_id":str(i),"directory":str(tmp_path/str(i)),"row_sha256":str(i)}
        atomic_json(Path(case["directory"])/"preparation/prepared.json",{
            "case_id":str(i),"row_sha256":str(i),"identity":"same","source":str(tmp_path/f"{i}.mp4"),
            "variant":"text_two_person","prompt":"six sections","frames":[311,124,226][i],
            "width":1568,"height":672,"seed":42})
        cases.append(case)
    return {"cases":cases,"pair_id":0,"identity":"same",
            "limits":{str(i):{"prepare":2,"generate":1} for i in range(count)}}


class RootChannel:
    rank = 0
    def __init__(self):
        self.sent = []
    def broadcast(self, value):
        self.sent.append(value)
        return value
    def gather(self, value):
        return [value,value]


def test_persistent_model_once_three_jobs_publish_failure_isolation(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    config = prepared_cases(tmp_path)
    config.update(group_size=4,ulysses_degree=2)
    channel = RootChannel()
    channel.gather = lambda value:[value] * 4
    events = []
    class Backend:
        metadata: ClassVar = {"model_load_count":1,"pdd_apply_count":1,
                             "distributed_init_count":1,"parallel_setup_count":1}
        def __init__(self):
            events.append("model_and_pdd_init")
        def prepare(self, job):
            return job
        def infer(self, job, reference):
            events.append(job["frames"])
            return b"video"
        def encode(self, result, path):
            if path.parent.parent.parent.name == "1":
                raise ValueError("bad encode")
            path.write_bytes(result)
        def memory(self):
            return {"peak_allocated_bytes":123}
    monkeypatch.setattr(module,"validate_output",lambda path,job:None)
    report = module.generate_loop(config,channel,Backend,tmp_path/"worker.json")
    assert events == ["model_and_pdd_init",311,124,226]
    assert report["jobs_succeeded"] == 2 and report["jobs_failed"] == 1
    assert report["model_load_count"] == report["pdd_apply_count"] == 1
    assert report["distributed_init_count"] == report["parallel_setup_count"] == 1
    manifest = read_json(tmp_path/"0/generation/manifest.json")
    assert len(manifest["rank_memory"]) == 4
    for key in ("reference_prepare_wall_seconds","pipeline_infer_wall_seconds","encode_wall_seconds","total_case_wall_seconds"):
        assert manifest[key] >= 0 and report[key] >= 0
    assert phase(config["cases"][0]) == phase(config["cases"][2]) == "done"
    assert phase(config["cases"][1]) == "generate"
    assert [x["kind"] for x in channel.sent].count("job") == 3
    assert channel.sent[-1]["kind"] == "stop"
    before = (tmp_path/"0/generation/manifest.json").read_bytes()
    again = module.generate_loop(config,RootChannel(),lambda: (_ for _ in ()).throw(AssertionError("loaded")),tmp_path/"again.json")
    assert again["jobs_attempted"] == 0
    assert (tmp_path/"0/generation/manifest.json").read_bytes() == before


def test_nonzero_rank_only_receives_infers_never_publishes(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    messages = iter([{"kind":"job","job":{"frames":311}},{"kind":"continue"},{"kind":"stop"}])
    class Channel:
        rank = 1
        def broadcast(self, value):
            assert value is None
            return next(messages)
        def gather(self, value):
            return [value,value]
    class Backend:
        metadata: ClassVar = {}
        def prepare(self, job):
            return job
        def infer(self, job, reference):
            return "result"
        def memory(self):
            return {}
    for name in ("atomic_json","publish_generated","begin_attempt","fail_attempt"):
        monkeypatch.setattr(module,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError("rank1 write")))
    module.generate_loop({},Channel(),Backend,tmp_path/"worker.json")
    assert not list(tmp_path.iterdir())


def test_inference_failure_consumes_one_attempt_requests_clean_worker_restart(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    config = prepared_cases(tmp_path,1)
    class Backend:
        metadata: ClassVar = {"model_load_count":1,"pdd_apply_count":1}
        def prepare(self, job):
            return job
        def infer(self, job, reference):
            raise RuntimeError("CUDA OOM")
        def memory(self):
            return {}
    report = module.generate_loop(config,RootChannel(),Backend,tmp_path/"worker.json")
    assert report["restart_required"] is True
    failures = list((tmp_path/"0/failures").glob("*.json"))
    assert len(failures) == 1
    assert "CUDA OOM" in read_json(failures[0])["error"]
    assert phase(config["cases"][0]) == "generate"


def test_bad_media_does_not_reload_persistent_model(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    config = prepared_cases(tmp_path)
    calls = []
    class Backend:
        metadata: ClassVar = {"model_load_count":1,"pdd_apply_count":1}
        def __init__(self):
            calls.append("load")
        def prepare(self, job):
            if job["frames"] == 124:
                raise ValueError("decode failed")
            return "decoded"
        def infer(self, job, reference):
            calls.append(job["frames"])
            return b"video"
        def encode(self, result, path):
            path.write_bytes(result)
        def memory(self):
            return {}
    monkeypatch.setattr(module,"validate_output",lambda *a:None)
    report = module.generate_loop(config,RootChannel(),Backend,tmp_path/"stats.json")
    assert calls == ["load",311,226]
    assert report["jobs_failed"] == 1 and report["jobs_succeeded"] == 2
    assert report["restart_required"] is False


def test_asymmetric_oom_broken_collective_reconciles_attempt_not_user_interrupt(tmp_path, capsys):
    from r2v_data_v2.person_replacement import h3_pair_generation as module
    from r2v_data_v2.person_replacement.h3_pair_executor import reconcile_failed_worker
    from r2v_data_v2.person_replacement.h3_pair_state import failure_count

    config = prepared_cases(tmp_path,1)
    stats, log = tmp_path/"stats.json",tmp_path/"log.txt"
    class Channel(RootChannel):
        def gather(self, value):
            if "memory" in value:
                raise RuntimeError("peer stuck in FSDP collective")
            return super().gather(value)
    class Backend:
        metadata: ClassVar = {}
        def prepare(self, job):
            return None
        def infer(self, *args):
            raise RuntimeError("CUDA out of memory")
        def memory(self):
            return {}
    with pytest.raises(RuntimeError,match="peer stuck"):
        module.generate_loop(config,Channel(),Backend,stats)
    output = capsys.readouterr().out
    log.write_text("")  # kill without an inference error must not count as failed case
    assert not reconcile_failed_worker(config,stats,log)
    assert failure_count(config["cases"][0],"generate") == 0
    log.write_text(output)
    assert reconcile_failed_worker(config,stats,log)
    assert failure_count(config["cases"][0],"generate") == 1


def test_interrupted_generation_resumes_suffix_and_ignores_stale_video(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    config = prepared_cases(tmp_path)
    interrupted = True
    seen = []
    class Backend:
        metadata: ClassVar = {"model_load_count":1,"pdd_apply_count":1}
        def prepare(self, job):
            return job
        def infer(self, job, reference):
            seen.append(job["frames"])
            if interrupted and job["frames"] == 124:
                # Simulate a killed session with an unfinished temporary video.
                (tmp_path/"1/tmp/generate-001/raw.mp4").write_bytes(b"partial")
                raise KeyboardInterrupt
            return b"valid"
        def encode(self, result, path):
            path.write_bytes(result)
        def memory(self):
            return {}
    monkeypatch.setattr(module,"validate_output",lambda *a:None)
    with pytest.raises(KeyboardInterrupt):
        module.generate_loop(config,RootChannel(),Backend,tmp_path/"first.json")
    committed = (tmp_path/"0/generation/manifest.json").read_bytes()
    assert phase(config["cases"][0]) == "done" and phase(config["cases"][1]) == "generate"
    interrupted = False
    seen.clear()
    report = module.generate_loop(config,RootChannel(),Backend,tmp_path/"resumed.json")
    assert seen == [124,226] and report["jobs_succeeded"] == 2
    assert (tmp_path/"0/generation/manifest.json").read_bytes() == committed
    assert (tmp_path/"1/generation/raw.mp4").read_bytes() == b"valid"
    assert (tmp_path/"1/tmp/generate-001/raw.mp4").read_bytes() == b"partial"
