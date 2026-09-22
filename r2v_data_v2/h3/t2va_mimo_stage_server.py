"""Lazy per-shard lifecycle for the local MiMo SGLang endpoint."""

from __future__ import annotations

import http.client
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


def build_mimo_serve_command(
    sglang: str | Path,
    checkpoint: str | Path,
    *,
    served_model_name: str = "mimo-v2.5",
    mem_fraction_static: float = 0.65,
) -> list[str]:
    if not 0 < mem_fraction_static <= 1:
        raise ValueError("MiMo mem fraction must be in (0, 1]")
    if not served_model_name.strip():
        raise ValueError("MiMo served model name must be non-empty")
    command = [
        str(sglang),
        "serve",
        "--model-path",
        str(checkpoint),
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        "8092",
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mm-enable-dp-encoder",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "fa3",
        "--mm-attention-backend",
        "fa3",
        "--context-length",
        "131072",
        "--mem-fraction-static",
        f"{mem_fraction_static:g}",
        "--chunked-prefill-size",
        "16384",
        "--max-running-requests",
        "8",
        "--reasoning-parser",
        "mimo",
        "--tool-call-parser",
        "mimo",
        "--constrained-json-disable-any-whitespace",
        "--enable-deterministic-inference",
    ]
    if served_model_name == "mimo-v2.6-flash-rl":
        command.extend(
            [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--enable-multi-layer-eagle",
            ]
        )
    return command


def _descendants(parent: int) -> set[int]:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
        pairs = [tuple(map(int, line.split())) for line in output.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()
    found = {parent}
    while True:
        children = {pid for pid, ppid in pairs if ppid in found}
        if children <= found:
            return found - {parent}
        found.update(children)


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


class _Completions:
    def __init__(self, owner: "StageMimoClient") -> None:
        self.owner = owner

    def create(self, **kwargs):
        return self.owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: "StageMimoClient") -> None:
        self.completions = _Completions(owner)


class StageMimoClient:
    """OpenAI-compatible client that starts MiMo only on the first stage request."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        serve_command: list[str],
        log_root: str | Path,
        startup_polls: int = 360,
        poll_interval: float = 5.0,
        cleanup_grace_seconds: float = 30.0,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.port is None
            or not api_key
            or timeout_seconds <= 0
            or not serve_command
            or startup_polls < 1
            or poll_interval < 0
            or cleanup_grace_seconds <= 0
        ):
            raise ValueError("invalid MiMo stage-server configuration")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.serve_command = list(serve_command)
        self.log_root = Path(log_root)
        self.startup_polls = startup_polls
        self.poll_interval = poll_interval
        self.cleanup_grace_seconds = cleanup_grace_seconds
        self.host = parsed.hostname
        self.port = parsed.port
        self.chat = _Chat(self)

        self._lock = threading.Lock()
        self._stage_shard: int | None = None
        self._process: subprocess.Popen | None = None
        self._client = None
        self._log_handle = None
        self._log_path: Path | None = None
        self._started_once = False

    @contextmanager
    def stage(self, shard_id: int):
        with self._lock:
            if self._stage_shard is not None:
                raise RuntimeError("MiMo stage lifecycle is already active")
            self._stage_shard = shard_id
            self._started_once = False
        try:
            yield self
        finally:
            with self._lock:
                self._stop_locked()
                self._stage_shard = None
                self._started_once = False

    def _probe(self, path: str) -> bool:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            return 200 <= response.status < 300
        except OSError:
            return False
        finally:
            connection.close()

    def _port_available(self) -> bool:
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.bind((self.host, self.port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _new_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def _ensure_started_locked(self) -> None:
        if self._stage_shard is None:
            raise RuntimeError("MiMo request made outside a shard MiMo stage")
        if self._started_once:
            if self._process is None or self._process.poll() is not None:
                raise ConnectionError(
                    "connection refused: MiMo stage server exited during the shard"
                )
            return
        if not self._port_available():
            raise ConnectionError(
                f"connection refused: MiMo stage port {self.host}:{self.port} is busy"
            )

        self.log_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = self.log_root / (
            f"mimo-shard-{self._stage_shard:04d}-{stamp}-{os.getpid()}.log"
        )
        self._log_handle = self._log_path.open("ab")
        started = time.monotonic()
        try:
            self._process = subprocess.Popen(
                self.serve_command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            self._close_log_locked()
            raise ConnectionError(
                f"connection refused: MiMo stage server could not start: {exc}"
            ) from exc
        self._started_once = True
        print(
            f"shard={self._stage_shard} mimo_server_started "
            f"pid={self._process.pid} log={self._log_path}",
            flush=True,
        )

        for attempt in range(self.startup_polls):
            if self._process.poll() is not None:
                code = self._process.returncode
                path = self._log_path
                self._stop_locked()
                raise ConnectionError(
                    f"connection refused: MiMo stage server exited {code}; log={path}"
                )
            if self._probe("/v1/models") and self._probe("/model_info"):
                self._client = self._new_client()
                print(
                    f"shard={self._stage_shard} mimo_server_ready "
                    f"startup_seconds={time.monotonic() - started:.3f}",
                    flush=True,
                )
                return
            if attempt + 1 < self.startup_polls:
                time.sleep(self.poll_interval)

        path = self._log_path
        self._stop_locked()
        raise ConnectionError(
            f"connection refused: MiMo stage server readiness timeout; log={path}"
        )

    def _close_log_locked(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
        self._log_handle = None

    def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            self._client = None
            self._close_log_locked()
            return

        descendants = _descendants(process.pid)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.cleanup_grace_seconds)
            except subprocess.TimeoutExpired:
                for pid in descendants:
                    _signal(pid, signal.SIGKILL)
                process.kill()
                process.wait()

        # SGLang normally reaps its own ranks. Clean any surviving descendants
        # captured before shutdown without creating a separate process session;
        # this also lets the outer launcher kill the whole supervisor group.
        live = []
        for pid in descendants:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            live.append(pid)
            _signal(pid, signal.SIGTERM)
        if live:
            time.sleep(0.1)
            for pid in live:
                _signal(pid, signal.SIGKILL)

        shard_id = self._stage_shard
        self._process = None
        self._client = None
        self._close_log_locked()
        print(f"shard={shard_id} mimo_server_stopped", flush=True)

    def _create(self, **kwargs):
        with self._lock:
            self._ensure_started_locked()
            client = self._client
        return client.chat.completions.create(**kwargs)
