"""Per-node launch and shared endpoint discovery, without distributed election."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


def local_coordinator_address():
    # UDP connect selects the local default route; it sends no network packet.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as stream:
        stream.connect(("192.0.2.1", 9))
        return stream.getsockname()[0]


def coordinator_token(token_file=None):
    path = Path(token_file or Path.home() / ".config/r2va/coordinator.token")
    if path.stat().st_mode & 0o077:
        raise ValueError(f"Coordinator credential must be private (0600): {path}")
    token = path.read_text().strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError(f"Invalid Coordinator credential file: {path}")
    return token


def publish_endpoint(run_root, host, port):
    """Called only after the designated Coordinator owns its local lock and TCP."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".coordinator_endpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"url": f"http://{host}:{port}"}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "coordinator_endpoint.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def wait_for_endpoint(run_root, timeout, coordinator=None, *, token):
    path = Path(run_root) / "coordinator_endpoint.json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if coordinator is not None and coordinator.poll() is not None:
            raise RuntimeError("Coordinator exited during startup")
        try:
            endpoint = json.loads(path.read_text())
            request = urllib.request.Request(endpoint["url"] + "/v1/status",
                headers={"Authorization": f"Bearer {token}"})
            with opener.open(request, timeout=min(1, max(.01, deadline - time.monotonic()))) as response:
                if json.load(response).get("run_root") != str(Path(run_root).absolute()):
                    raise ValueError("Coordinator discovery points to a different run")
                return endpoint
        except (FileNotFoundError, OSError):
            time.sleep(.1)
    raise TimeoutError(f"Coordinator discovery timed out: {path}; start the designated coordinator with the same run-id")


def node_configuration(repo_root, environment):
    path = environment.get("R2VA_CONFIGURATION")
    values = json.loads(Path(path or Path(repo_root) / "docs/examples/ra2va_group_native_node.json").read_text())
    if path:
        return values
    values["gpu_ids"] = environment.get("GPU_IDS", "0,1,2,3,4,5,6,7").split(",")
    values["mimo"]["gpu_groups"] = environment.get("MIMO_GPU_GROUPS", "0,1,2,3;4,5,6,7").split(";")
    values["mimo"]["ports"] = [int(port) for port in environment.get("MIMO_PORTS", "8094,8095").split(",")]
    values["mimo"]["mem_fraction_static"] = float(environment.get("MIMO_MEM_FRACTION_STATIC", ".65"))
    values["asr"]["environment"]["QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE"] = environment.get("ASR_BATCH_SIZE", "8")
    return values


def _runtime_environment(repo_root, run_root, environment):
    env = dict(environment)
    deps = env.get("SAM_DEPS_ROOT", "/mnt/workspace/litengjie/data/audio_deps")
    overlay = ":".join(f"{deps}/{part}" for part in (
        "sam-audio-src", "perception-models-src", "dacvae-src", "sam-audio-pydeps"))
    env.update(R2VA_RUN_ROOT=str(run_root), HOSTNAME=socket.gethostname(),
        SAM_AUDIO_RUNTIME_PYTHONPATH=overlay, PYTHONPATH=f"{repo_root}:{overlay}",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="*", no_proxy="*")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    return env


def _start_media(python, values, environment, repo_root):
    url = urllib.parse.urlsplit(values["mimo"]["media_base_url"])
    if url.hostname not in {"127.0.0.1", "localhost"}:
        return None
    port = url.port or 80
    try:
        with socket.create_connection((url.hostname, port), timeout=1):
            print(f"media: using existing {url.geturl()}", flush=True)
            return None
    except OSError:
        command = [python, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                   "--directory", values["mimo"]["media_root"]]
        process = subprocess.Popen(command, cwd=repo_root, env=environment, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with socket.create_connection((url.hostname, port), timeout=.2):
                    print(f"media: started {url.geturl()}", flush=True)
                    return process
            except OSError:
                time.sleep(.1)
        raise RuntimeError(f"Media server failed to start: {url.geturl()}")
    except BaseException:
        process.terminate()
        process.wait(timeout=5)
        raise


def launch_node(repo_root, environment, *, dry_run=False, fake=False):
    from r2v_data_v2.h3.ra2va_group_fake import OUTPUT_ROOT, SOURCE_ROOT, pilot_run_root
    from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process
    from r2v_data_v2.h3.t2va_full_workers import _descendants
    from r2v_data_v2.h3.t2va_production import atomic_json

    role = environment.get("R2VA_ROLE", "worker")
    if role not in {"coordinator", "worker"}:
        raise ValueError("R2VA_ROLE must be coordinator or worker")
    run_id = environment.get("R2VA_RUN_ID", "ra2va-group-pilot20-v1")
    output_root = Path(environment.get("R2VA_OUTPUT_ROOT", OUTPUT_ROOT))
    run_root = pilot_run_root(output_root, run_id)
    python = environment.get("R2V_PYTHON", str(repo_root / ".venv/bin/python"))
    tool = str(repo_root / "tools/run_h3_ra2va_group_production.py")
    values = node_configuration(repo_root, environment)
    workers = int(environment.get("REQUEST_WORKERS", len(values["gpu_ids"])))
    node = environment.get("R2VA_NODE_ID", socket.gethostname())
    configuration = run_root / "node-logs" / node / "launch_configuration.json"
    coordinator_command = [python, tool, "coordinator", "--execution-mode", "fake" if fake else "native",
        "--source-root", str(environment.get("R2VA_SOURCE_ROOT", SOURCE_ROOT)), "--output-root", str(output_root),
        "--run-id", run_id, "--limit", environment.get("R2VA_LIMIT", "20"),
        "--start-row", environment.get("R2VA_START_ROW", "0"),
        "--max-groups", environment.get("R2VA_MAX_GROUPS", "1"),
        "--host", environment.get("R2VA_COORDINATOR_HOST", "auto"),
        "--port", environment.get("R2VA_COORDINATOR_PORT", "8780"),
        "--local-lock-path", f"/tmp/{run_id}.coordinator.lock", "--publish-endpoint"]
    plan = {"role": role, "run_root": str(run_root), "node_id": node, "workers": workers,
        "start_row": int(environment.get("R2VA_START_ROW", "0")),
        "limit": int(environment.get("R2VA_LIMIT", "20")),
        "gpu_ids": values["gpu_ids"], "mimo_gpu_groups": values["mimo"]["gpu_groups"],
        "mimo_ports": values["mimo"]["ports"], "call_mode": "two_step",
        "execution": "fake" if fake else "native",
        "coordinator": coordinator_command if role == "coordinator" else None}
    print(json.dumps(plan), flush=True)
    if dry_run:
        return 0
    env = _runtime_environment(repo_root, run_root, environment)
    if not env.get("R2V_GROUP_COORDINATOR_TOKEN"):
        token_file = Path(env.get("R2VA_TOKEN_FILE", Path.home() / ".config/r2va/coordinator.token")).resolve()
        if token_file.is_relative_to(Path(values["mimo"]["media_root"]).resolve()):
            raise ValueError("Coordinator credential must be outside the media root")
        env["R2V_GROUP_COORDINATOR_TOKEN"] = coordinator_token(token_file)
    coordinator = worker = media = None
    worker_descendants = set()
    handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[sig] = signal.signal(sig, interrupted)
    try:
        if role == "coordinator":
            coordinator = subprocess.Popen(coordinator_command, cwd=repo_root, env=env, start_new_session=True)
        endpoint = wait_for_endpoint(run_root, float(env.get("R2VA_WAIT_SECONDS", "300")), coordinator,
                                     token=env["R2V_GROUP_COORDINATOR_TOKEN"])
        print(f"coordinator: {endpoint['url']}; node={node}", flush=True)
        command = [python, tool, "worker", "--fake" if fake else "--native",
                   "--coordinator-url", endpoint["url"], "--node-id", node, "--workers", str(workers)]
        if fake:
            command += ["--fake-delay-seconds", env.get("R2VA_FAKE_DELAY_SECONDS", "0")]
        else:
            atomic_json(configuration, values)
            media = _start_media(python, values, env, repo_root)
            command += ["--configuration", str(configuration)]
        worker = subprocess.Popen(command, cwd=repo_root, env=env, start_new_session=True)
        while worker.poll() is None:
            observed = _descendants({worker.pid})
            if worker.poll() is None:
                worker_descendants = observed
            else:
                worker_descendants.update(observed)
            if coordinator is not None and coordinator.poll() is not None:
                raise RuntimeError("Coordinator exited before local Workers completed")
            if media is not None and media.poll() is not None:
                raise RuntimeError("Owned media server exited before local Workers completed")
            time.sleep(.1)
        worker_status = worker.returncode
        stop_owned_process(worker, worker_descendants, grace=float(env.get("R2VA_CLEANUP_SECONDS", "60")))
        worker = None
        if worker_status == 0 and coordinator is not None:
            print("Local Workers finished; Coordinator remains available for other nodes. Ctrl+C stops it.", flush=True)
            return coordinator.wait()
        return worker_status
    except KeyboardInterrupt:
        return 130
    finally:
        # Native Worker closes its own model resources before Coordinator stops.
        for process in (worker, media, coordinator):
            if process is not None:
                descendants = worker_descendants if process is worker else set()
                stop_owned_process(process, descendants, grace=float(env.get("R2VA_CLEANUP_SECONDS", "60")))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
