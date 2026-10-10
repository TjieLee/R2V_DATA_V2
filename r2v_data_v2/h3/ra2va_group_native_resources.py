"""Node-owned process cleanup and shared, stage-scoped MiMo TP4 replicas."""

from __future__ import annotations

import signal
import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from r2v_data_v2.h3.t2va_full_workers import _descendants, _signal_pid
from r2v_data_v2.h3.t2va_mimo_stage_server import (
    build_mimo_serve_command,
    mimo_v26_server_env,
)


def stop_owned_process(process, descendants, *, grace=2):
    """Also stop observed nested backends which detached from the parent's PGID."""
    def signal_group(sig):
        # Children can retain the owned PGID after their leader exits.
        try:
            _signal_pid(process.pid, sig, group=True)
        except PermissionError:
            if process.poll() is None:
                raise

    descendants.update(_descendants({process.pid}))
    for pid in descendants:
        _signal_pid(pid, signal.SIGTERM)
    signal_group(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    for pid in descendants:
        _signal_pid(pid, signal.SIGKILL)
    signal_group(signal.SIGKILL)
    process.wait(timeout=5)


class SGLangProcess:
    def __init__(self, configuration, gpu_group, port, log_root):
        self.configuration, self.port = configuration, port
        self.command = build_mimo_serve_command(configuration["sglang"], configuration["checkpoint"],
            served_model_name="mimo-v2.6-flash-rl", port=port,
            mem_fraction_static=configuration.get("mem_fraction_static", .65))
        self.environment = mimo_v26_server_env(gpu_group)
        self.log_path = Path(log_root) / f"sglang-{port}.log"
        self.process = self.log = None
        self.descendants = set()

    def launch(self):
        # Never interpret somebody else's already-running endpoint as ours.
        with socket.socket() as available:
            available.bind(("127.0.0.1", self.port))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a")
        try:
            self.process = subprocess.Popen(self.command, env=self.environment, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def start(self, allowed):
        self.launch()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + self.configuration.get("startup_timeout_seconds", 1800)
        try:
            while allowed() and time.monotonic() < deadline:
                self.descendants.update(_descendants({self.process.pid}))
                if self.process.poll() is not None:
                    raise RuntimeError(f"SGLang exited; see {self.log_path}")
                try:
                    with opener.open(f"http://127.0.0.1:{self.port}/v1/models", timeout=2) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
                time.sleep(.2)
            raise RuntimeError(f"SGLang startup interrupted or timed out; see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            stop_owned_process(self.process, self.descendants)
        if self.log is not None:
            self.log.close()


class NodeMimoServices:
    def __init__(self, configuration, *, log_root, server_factory=None):
        self.configuration, self.log_root = configuration, Path(log_root)
        groups, ports = configuration["gpu_groups"], configuration["ports"]
        ids = [gpu.strip() for group in groups for gpu in group.split(",")]
        if (not groups or any(len(group.split(",")) != 4 for group in groups)
                or len(set(ids)) != len(ids)):
            raise ValueError("MiMo requires disjoint TP4 GPU groups, without overlap")
        if len(groups) != len(ports) or len(set(ports)) != len(ports):
            raise ValueError("each TP4 replica needs a distinct port")
        self.server_factory = server_factory or (lambda group, port: SGLangProcess(
            configuration, group, port, self.log_root / f"generation-{self.generation}"))
        self.lock = threading.Lock()
        self.generation, self.users, self.servers = None, 0, []
        self.slots = [threading.Lock() for _ in ports]

    def join(self, generation, allowed):
        with self.lock:
            if self.users and generation != self.generation:
                raise RuntimeError("previous node-local MiMo generation is still active")
            if not self.users:
                self.generation = generation
                try:
                    for group, port in zip(self.configuration["gpu_groups"], self.configuration["ports"], strict=True):
                        server = self.server_factory(group, port)
                        self.servers.append(server)
                        server.start(allowed)
                except BaseException:
                    self._close()
                    raise
            self.users += 1

    def _close(self):
        for server in reversed(self.servers):
            server.close()
        self.servers = []

    def leave(self, generation):
        with self.lock:
            if generation != self.generation or self.users <= 0:
                raise RuntimeError("MiMo release generation differs")
            self.users -= 1
            if not self.users:
                self._close()

    @contextmanager
    def lease(self, index, allowed):
        slot = self.slots[index]
        while not slot.acquire(timeout=.05):
            if not allowed():
                raise RuntimeError("MiMo endpoint wait interrupted")
        try:
            if not allowed():
                raise RuntimeError("MiMo endpoint permission expired")
            process = getattr(self.servers[index], "process", None)
            if process is not None and process.poll() is not None:
                raise RuntimeError("Node-owned SGLang process exited; no implicit restart")
            yield
        finally:
            slot.release()
