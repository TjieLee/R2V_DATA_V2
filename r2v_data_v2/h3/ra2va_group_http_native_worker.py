"""Native Group stages behind the existing HTTP claim/heartbeat/receipt loop."""

from __future__ import annotations

from pathlib import Path

from r2v_data_v2.h3.ra2va_group_http_worker import run_http_worker
from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
from r2v_data_v2.h3.ra2va_group_production import read_json
from r2v_data_v2.h3.ra2va_group_source import GroupTask


class NativeStageExecution:
    def __init__(self, session, *, backend_factory, ffmpeg, execution):
        self.execution = execution
        self.worker = GroupStageWorker(session["stage"], Path("."), backend_factory=backend_factory, ffmpeg=ffmpeg)

    def __enter__(self):
        self.worker.__enter__()
        return self

    def __exit__(self, *args):
        return self.worker.__exit__(*args)

    def execute(self, task):
        rows = {stage: read_json(Path(path)) for stage, path in task["upstream_results"].items()}
        if any(row["status"] != "ready" for row in rows.values()):
            return "skipped", {"execution": self.execution, "reason": "upstream_terminal"}
        self.worker.root = Path(task["artifact_root"])
        try:
            payload = self.worker.process(GroupTask(**task["task"]), {k: r["payload"] for k, r in rows.items()})
            status = "failed" if payload.get("product_failed_count", 0) else payload.get("status", "ready")
        except (ValueError, OSError) as error:
            status, payload = "failed", {"failure_reason": f"{type(error).__name__}: {error}"}
        return status, payload | {"execution": self.execution}


def run_http_native_worker(client, *, node_id, worker_instance, stop_event, backend_factory,
                           ffmpeg="ffmpeg", execution="native"):
    return run_http_worker(client, node_id=node_id, worker_instance=worker_instance, stop_event=stop_event,
        execution_mode=execution, executor_factory=lambda session, heartbeat: NativeStageExecution(
            session, backend_factory=backend_factory, ffmpeg=ffmpeg, execution=execution))
