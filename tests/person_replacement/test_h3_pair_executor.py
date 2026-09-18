import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_old_prepared_and_done_survive_group_change(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_executor as module
    from r2v_data_v2.person_replacement.h3_pair_state import (
        atomic_json,
        begin_attempt,
        fail_attempt,
    )
    from tools.person_replacement.run_h3_pdd_pair_executor import arguments

    clips = tmp_path/"clips"
    clips.mkdir()
    source = tmp_path/"source.jsonl"
    source.write_text('{"video_path":"a.mp4"}\n{"video_path":"b.mp4"}\n')
    argv = ["--input-jsonl",str(source),"--clips-root",str(clips),"--output-root",str(tmp_path/"out")]
    root,old = module.make_config(arguments(argv))
    assert old["identity_details"]["contract"] == "text_two_person_pdd_fsdp2_pair_v8"
    legacy_details = {**old["identity_details"],"contract":"text_two_person_pdd_fsdp2_pair_v7"}
    legacy_identity = hashlib.sha256(json.dumps(legacy_details,sort_keys=True).encode()).hexdigest()
    legacy_root = tmp_path/"legacy-v7"
    atomic_json(legacy_root/"identity.json",{"identity":legacy_identity})
    with pytest.raises(ValueError,match="identity mismatch"):
        module.validate_identity(legacy_root,old)
    module.validate_identity(root,old)
    done,pending = old["cases"]
    done_path = Path(done["directory"])/"generation/manifest.json"
    prepared = Path(pending["directory"])/"preparation/prepared.json"
    atomic_json(done_path,{"original":True})
    atomic_json(prepared,{"identity":old["identity"],"prompt":"unchanged"})
    for _ in range(2):
        fail_attempt(pending,begin_attempt(pending,"generate",{}),RuntimeError("OOM"))
    original = prepared.read_bytes(),done_path.read_bytes()
    _,new = module.make_config(arguments(argv+["--group-size","4","--ulysses-degree","2","--retry-failed","--max-generate-attempts","1"]))
    assert new["identity"] == old["identity"]
    assert new["ulysses_degree"] == 2 and old["ulysses_degree"] == 1
    module.validate_identity(root,new)
    assert new["limits"][pending["case_id"]]["generate"] == 3
    calls = []
    def children(specs, fd):
        calls.extend(specs)
        assert len(specs) == 1 and "--nproc_per_node=4" in specs[0]["command"]
        atomic_json(Path(pending["directory"])/"generation/manifest.json",{})
        return [0]
    monkeypatch.setattr(module,"run_children",children)
    module.execute_phases(new,root,"0,1,2,3",123)
    assert len(calls) == 1
    assert (prepared.read_bytes(),done_path.read_bytes()) == original


@pytest.mark.parametrize("log_exists", [True, False])
def test_infrastructure_failure_reports_worker_log_tail(tmp_path, monkeypatch, capsys, log_exists):
    from r2v_data_v2.person_replacement import h3_pair_executor as module
    from r2v_data_v2.person_replacement.h3_pair_state import atomic_json

    case = {"case_id":"case", "directory":str(tmp_path/"case")}
    atomic_json(Path(case["directory"])/"preparation/prepared.json", {})
    config = {"cases":[case], "limits":{"case":{"prepare":2,"generate":2}},
              "h3_python":sys.executable}
    logs = []
    def failed_worker(specs, lock_fd):
        log = Path(specs[0]["log"])
        logs.append(log)
        if log_exists:
            log.write_text("".join(f"worker-line-{i:03d}\n" for i in range(120)))
        return [1]
    monkeypatch.setattr(module, "run_children", failed_worker)
    with pytest.raises(RuntimeError, match="H3 infrastructure failed"):
        module.execute_phases(config, tmp_path/"shard", "0,1", 123)
    stderr = capsys.readouterr().err
    assert str(logs[0].resolve()) in stderr
    if log_exists:
        assert [line for line in stderr.splitlines() if line.startswith("worker-line-")] == [
            f"worker-line-{i:03d}" for i in range(40,120)]
    else:
        assert "Unable to read worker log" in stderr


@pytest.mark.parametrize("group_size", [2,4,8])
def test_two_prepare_children_then_one_torchrun_resume_skips_done(tmp_path, monkeypatch, group_size):
    from r2v_data_v2.person_replacement import h3_pair_executor as module
    from r2v_data_v2.person_replacement.h3_pair_state import atomic_json

    case = {"case_id":"case","directory":str(tmp_path/"shard-000000/case"),"row_sha256":"row"}
    config = {"cases":[case],"limits":{"case":{"prepare":2,"generate":2}},"pair_id":0,
              "identity":"identity","h3_python":sys.executable,"group_size":group_size}
    calls = []
    def children(specs, lock_fd, **kwargs):
        calls.append(specs)
        atomic_json(Path(case["directory"])/("preparation/prepared.json" if len(specs)==group_size else "generation/manifest.json"),{})
        return [0]*len(specs)
    monkeypatch.setattr(module,"run_children",children)
    devices = ",".join(str(i) for i in range(group_size))
    module.execute_phases(config,tmp_path/"shard-000000",devices,123)
    assert len(calls) == 2 and len(calls[0]) == group_size and len(calls[1]) == 1
    assert [spec["env"]["CUDA_VISIBLE_DEVICES"] for spec in calls[0]] == devices.split(",")
    assert calls[1][0]["env"]["CUDA_VISIBLE_DEVICES"] == devices
    assert f"--nproc_per_node={group_size}" in calls[1][0]["command"]
    calls.clear()
    module.execute_phases(config,tmp_path/"shard-000000",devices,123)
    assert calls == []


def test_dry_run_is_read_only_and_pair_size_does_not_overlap(tmp_path, monkeypatch, capsys):
    from tools.person_replacement import run_h3_pdd_pair_executor as cli

    clips = tmp_path/"clips"
    clips.mkdir()
    source = tmp_path/"source.jsonl"
    source.write_text("".join(json.dumps({"video_path":f"{i}.mp4"})+"\n" for i in range(6)))
    out = tmp_path/"out"
    monkeypatch.setattr(cli,"run_pair",lambda *a,**k:pytest.fail("launched"))
    for pair, expected in ((0,[0,1,2]),(1,[3,4,5])):
        assert cli.main(["--input-jsonl",str(source),"--clips-root",str(clips),"--output-root",str(out),
                         "--pair-id",str(pair),"--pair-size","3","--dry-run"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert [x["source_index"] for x in report["cases"]] == expected
    assert not out.exists()


def test_shard_lock_excludes_second_executor_and_identity_mismatch(tmp_path):
    from r2v_data_v2.person_replacement.h3_pair_executor import (
        pair_lock,
        validate_identity,
    )

    with pair_lock(tmp_path), pytest.raises(RuntimeError,match="active"), pair_lock(tmp_path):
        pass
    validate_identity(tmp_path,{"identity":"a"})
    with pytest.raises(ValueError,match="identity"):
        validate_identity(tmp_path,{"identity":"b"})


@pytest.mark.parametrize("group_size", [2,4,8])
def test_node_maps_four_nonoverlapping_pairs_without_barrier(tmp_path, monkeypatch, group_size):
    from tools.person_replacement import run_h3_pdd_node as cli

    calls = []
    monkeypatch.setattr(cli,"run_children",lambda specs,*a,**k:calls.extend(specs) or [0]*4)
    assert cli.main(["--gpus","0,1,2,3,4,5,6,7","--pair-start","4",
                     "--output-root",str(tmp_path),"--pair-size","1000","--resume",
                     "--group-size",str(group_size),"--ulysses-degree","2"]) == 0
    expected = {2:["0,1","2,3","4,5","6,7"],4:["0,1,2,3","4,5,6,7"],8:["0,1,2,3,4,5,6,7"]}
    assert [x["env"]["CUDA_VISIBLE_DEVICES"] for x in calls] == expected[group_size]
    assert [x["command"][x["command"].index("--pair-id")+1] for x in calls] == [str(4+i) for i in range(8//group_size)]
    assert all(x["command"][x["command"].index("--group-size")+1] == str(group_size) for x in calls)
    assert all(x["command"][x["command"].index("--ulysses-degree")+1] == "2" for x in calls)


def test_group_validation_before_launch_and_allocator_override(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_executor as module
    from tools.person_replacement.run_h3_pdd_node import main

    monkeypatch.setattr(module,"run_children",lambda *a,**k:pytest.fail("launched"))
    with pytest.raises(ValueError,match="CUDA_VISIBLE_DEVICES"):
        module.execute_phases({"group_size":4},tmp_path/"unused","0,1",123)
    assert not (tmp_path/"unused").exists()
    for gpus in ("0,1,2", "0,0,1,2"):
        with pytest.raises(SystemExit):
            main(["--gpus",gpus,"--group-size","4","--output-root",str(tmp_path)])
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF",raising=False)
    assert module.worker_environment(tmp_path,"0,1")["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:False")
    assert module.worker_environment(tmp_path,"0,1")["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_shutdown_kills_descendant_after_leader_exits(tmp_path):
    from r2v_data_v2.person_replacement.h3_pair_executor import run_children

    marker = tmp_path/"child.pid"
    heartbeat = tmp_path/"heartbeat"
    child = ("import signal,os,time; from pathlib import Path; "
             "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             f"Path({str(marker)!r}).write_text(str(os.getpid())); "
             f"p=Path({str(heartbeat)!r})\n"
             "while True:\n with p.open('ab') as f: f.write(b'x')\n time.sleep(.01)\n")
    parent = ("import subprocess,sys,time; from pathlib import Path; "
              f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
              f"p=Path({str(marker)!r}); "
              "\nwhile not p.exists(): time.sleep(.01)\n")
    codes = run_children([{"command":[sys.executable,"-c",parent],"env":os.environ.copy(),
                           "log":tmp_path/"log"}],shutdown_seconds=0.1)
    assert codes == [0]
    pid = int(marker.read_text())
    try:
        before = heartbeat.read_bytes()
        time.sleep(.1)
        assert heartbeat.read_bytes() == before
    finally:
        try:
            os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_sigterm_parent_cleans_children(tmp_path):
    marker = tmp_path/"pid"
    child = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    script = ("import os,sys; from r2v_data_v2.person_replacement.h3_pair_executor import run_children; "
              f"run_children([{{'command':[sys.executable,'-c',{child!r}], 'env':dict(os.environ), "
              f"'log':{str(tmp_path/'log')!r}}}],shutdown_seconds=.2)")
    parent = subprocess.Popen([sys.executable,"-c",script],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(200):
            if marker.exists():
                break
            time.sleep(.01)
        assert marker.exists()
        parent.terminate()
        parent.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()),0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
