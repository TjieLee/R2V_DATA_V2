"""Two-phase pair orchestration. Fixed shards; case artifacts remain authoritative."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from .h3_pair_state import (
    atomic_json,
    eligible,
    fail_attempt,
    failure_count,
    inventory,
    read_json,
    retry_limit,
    select_shard,
)
from .h3_pdd import PDDBackend
from .pipeline import validate_output_root

TOOLS = Path(__file__).resolve().parents[2]/"tools/person_replacement"


def visible_pair(value, group_size=2):
    devices = [device.strip() for device in value.split(",")]
    if group_size not in (2,4,8) or len(devices) != group_size or len(set(devices)) != group_size or not all(devices):
        raise ValueError(f"CUDA_VISIBLE_DEVICES must identify exactly {group_size} distinct GPUs (supported: 2/4/8)")
    return devices


def make_config(args):
    root = validate_output_root(args.output_root)
    source = args.input_jsonl.expanduser().resolve(strict=True)
    clips = args.clips_root.expanduser().resolve(strict=True)
    if not clips.is_dir() or root == clips or clips in root.parents or root in clips.parents or root in source.parents:
        raise ValueError("Output must not overlap inputs")
    cases = select_shard(source,root,args.pair_id,args.pair_size)
    values = {key:str(getattr(args,key).expanduser().absolute()) if getattr(args,key) is not None else None
              for key in ("qwen_model","h3_python","h3_model_root","pdd_code_root","pdd_lora")}
    identity = {"input":str(source),"clips_root":str(clips),"pair_id":args.pair_id,
                "pair_size":args.pair_size,"seed":args.seed,"resources":values,
                "rows":[(case["source_index"],case["row_sha256"]) for case in cases],
                "contract":"text_two_person_pdd_fsdp2_pair_v3"}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config = {**values,"identity":digest,"identity_details":identity,"cases":cases,"group_size":args.group_size,
              "ulysses_degree":args.ulysses_degree,
              "clips_root":str(clips),"seed":args.seed,"pair_id":args.pair_id,
              "limits":{case["case_id"]:{stage:retry_limit(case,stage,maximum,retry_failed=args.retry_failed)
                        for stage,maximum in (("prepare",args.max_prepare_attempts),("generate",args.max_generate_attempts))}
                        for case in cases}}
    return root/f"shard-{args.pair_id:06d}",config


@contextmanager
def pair_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/"pair.lock").open("a") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Pair is already active; no duplicate executor allowed") from exc
        # Children inherit the lock descriptor, preventing overlap even if this
        # supervisor is SIGKILLed before all descendants terminate.
        yield handle.fileno()


def validate_identity(root, config, *, read_only=False):
    path = root/"identity.json"
    if path.is_file():
        if read_json(path)["identity"] != config["identity"]:
            raise ValueError("Pair input/config identity mismatch; use a new output root")
    elif not read_only:
        atomic_json(path,{"identity":config["identity"],"details":config.get("identity_details")})


def worker_environment(directory, devices):
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    for key in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV","TRANSFORMERS_CACHE"):
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=devices,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",
               TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    for key,folder in (("HF_HOME","hf"),("HF_HUB_CACHE","hf/hub"),("HUGGINGFACE_HUB_CACHE","hf/hub"),
                       ("TORCH_HOME","torch"),("XDG_CACHE_HOME","cache"),("TRITON_CACHE_DIR","triton"),
                       ("TORCHINDUCTOR_CACHE_DIR","inductor"),("TMPDIR","tmp")):
        path = directory/"runtime"/folder
        path.mkdir(parents=True,exist_ok=True)
        env[key] = str(path)
    return env


def _group_alive(pid):
    try:
        os.killpg(pid,0)
        return True
    except ProcessLookupError:
        return False


def run_children(specs, lock_fd=None, *, stop_on_error=True, shutdown_seconds=60):
    processes, logs = [], []
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous[sig] = signal.signal(sig,interrupted)
        for spec in specs:
            path = Path(spec["log"])
            path.parent.mkdir(parents=True,exist_ok=True)
            log = path.open("a")
            logs.append(log)
            command = [sys.executable,str(TOOLS/"h3_pair_child.py"),str(os.getpid()),*spec["command"]]
            processes.append(subprocess.Popen(command,env=spec["env"],stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=() if lock_fd is None else (lock_fd,)))
        while any(p.poll() is None for p in processes):
            if stop_on_error and any(p.poll() not in (None,0) for p in processes):
                break
            time.sleep(0.2)
    finally:
        for process in processes:
            # Kill the group, including descendants of a failed leader.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        # A leader can exit before a TERM-resistant descendant. Check the
        # process group too. Give torchrun time to close its own rank sessions.
        deadline = time.monotonic()+shutdown_seconds
        while time.monotonic() < deadline:
            if not any((p.poll() is None) or _group_alive(p.pid) for p in processes):
                break
            time.sleep(0.05)
        for process in processes:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log in logs:
            log.close()
        for sig,handler in previous.items():
            signal.signal(sig,handler)
    return [p.returncode for p in processes]


def reconcile_failed_worker(config, stats_path, log_path):
    """Only an explicit inference error receipt can consume an interrupted attempt.

    User interruption never reaches this code; missing receipts and model-load/
    NCCL infrastructure failures remain fatal, not guessed case failures.
    """
    if not stats_path.is_file():
        return False
    stats = read_json(stats_path)
    if stats.get("phase") != "inference" or stats.get("active_case") not in config["cases"]:
        return False
    case,attempt = stats["active_case"],stats["active_attempt"]
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("H3_PAIR_INFERENCE_ERROR "):
            continue
        try:
            error = json.loads(line.split(" ",1)[1])
        except ValueError:
            continue
        if (error.get("case_id") == case["case_id"] and error.get("attempt") == attempt["attempt"]
                and error.get("error_type") not in ("DistBackendError","DistNetworkError")):
            fail_attempt(case,attempt,RuntimeError(f"Failed distributed inference: {error}"))
            atomic_json(stats_path,{**stats,"restart_required":True,"reconciled_worker_failure":True})
            return True
    return False


def report_worker_log(log_path):
    path = Path(log_path).resolve()
    print(f"H3 worker log: {path}\n--- last 80 lines ---", file=sys.stderr, flush=True)
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            tail = "".join(deque(handle, maxlen=80))
        print(tail, file=sys.stderr, end="" if tail.endswith("\n") else "\n", flush=True)
    except OSError as exc:
        print(f"Unable to read worker log: {exc}", file=sys.stderr, flush=True)


def execute_phases(config, root, devices, lock_fd, *, prepare_only=False):
    group_size = config.get("group_size",2)
    pair = visible_pair(devices,group_size)
    session = uuid.uuid4().hex
    session_root = root/"sessions"/session
    session_root.mkdir(parents=True)
    config_path = session_root/"config.json"
    atomic_json(config_path,config)  # freezes retry allowance across worker restarts
    def pending(stage):
        return any(eligible(c,stage,config["limits"][c["case_id"]][stage]) for c in config["cases"])
    if pending("prepare"):
        specs = [{"command":[sys.executable,str(TOOLS/"h3_pair_prepare_worker.py"),"--config",str(config_path),
                             "--worker",str(i)],
                  "env":worker_environment(session_root/f"qwen-{i}",gpu),
                  "log":session_root/f"qwen-{i}.log"} for i,gpu in enumerate(pair)]
        if any(run_children(specs,lock_fd)):
            raise RuntimeError("Qwen preparation infrastructure failed; completed preparations remain resumable")
    restart = 0
    while not prepare_only and pending("generate"):
        before = sum(failure_count(c,"generate") for c in config["cases"])
        stats = session_root/f"h3-{restart}.json"
        specs = [{"command":[config["h3_python"],"-m","torch.distributed.run","--standalone",
                              f"--nproc_per_node={group_size}","--max-restarts=0",str(TOOLS/"h3_pdd_fsdp_worker.py"),
                              "--config",str(config_path),"--stats",str(stats)],
                  "env":worker_environment(session_root/f"h3-{restart}",devices),
                  "log":session_root/f"h3-{restart}.log"}]
        codes = run_children(specs,lock_fd)
        # torchrun maps rank exit codes to its own failure code. A rank-zero
        # durable restart marker plus a new case failure authorizes restart.
        restart_requested = stats.is_file() and read_json(stats).get("restart_required") is True
        if any(codes) and not restart_requested:
            restart_requested = reconcile_failed_worker(config,stats,Path(specs[0]["log"]))
        if restart_requested:
            after = sum(failure_count(c,"generate") for c in config["cases"])
            if after <= before:
                report_worker_log(specs[0]["log"])
                raise RuntimeError("Worker restart made no durable failure progress")
            restart += 1  # bounded by the invocation's finite per-case retry budgets
            continue
        if any(codes):
            report_worker_log(specs[0]["log"])
            raise RuntimeError("H3 infrastructure failed; rerun --resume after addressing worker log")
        break
    return inventory(config["cases"],config["limits"])


def run_pair(root, config, args):
    visible_pair(os.environ.get("CUDA_VISIBLE_DEVICES",""),config.get("group_size",2))
    if root.exists() and not args.resume:
        raise FileExistsError("Existing shard requires --resume; never overwrite")
    with pair_lock(root) as fd:
        validate_identity(root,config)
        counts = inventory(config["cases"],config["limits"])
        if counts["done"] == len(config["cases"]):
            return counts
        # Validate existing resources without importing any GPU framework.
        PDDBackend(config["h3_python"],config["h3_model_root"],config["pdd_code_root"],config["pdd_lora"]).validate()
        return execute_phases(config,root,os.environ.get("CUDA_VISIBLE_DEVICES",""),fd,
                              prepare_only=args.prepare_only)
