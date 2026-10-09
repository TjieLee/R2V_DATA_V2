"""Bounded RA2VA group CPU scheduling pilot. No model or real export runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.ra2va_group_fake import (
    OUTPUT_ROOT,
    SOURCE_ROOT,
    pilot_run_root,
    pilot_summary,
    run_fake_pilot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "coordinator"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
        command.add_argument("--run-id", required=True)
        if name in ("run", "coordinator"):
            command.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
            command.add_argument("--limit", type=int, required=True)
            command.add_argument("--max-groups", type=int, default=1)
        if name == "run":
            command.add_argument("--fake", action="store_true", required=True)
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--poll-seconds", type=float, default=0.05)
        if name == "coordinator":
            command.add_argument("--host", required=True)
            command.add_argument("--port", type=int, required=True)
            command.add_argument("--local-lock-path", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--fake", action="store_true", required=True)
    worker.add_argument("--coordinator-url", required=True)
    worker.add_argument("--node-id", required=True)
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--fake-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.command == "worker":
        from r2v_data_v2.h3.ra2va_group_http_worker import run_http_fake_workers
        print(json.dumps(run_http_fake_workers(coordinator_url=args.coordinator_url,
            node_id=args.node_id, workers=args.workers, fake_delay_seconds=args.fake_delay_seconds),
            indent=2), flush=True)
        return
    root = pilot_run_root(args.output_root, args.run_id)
    if args.command == "coordinator":
        from r2v_data_v2.h3.ra2va_group_http import (
            HttpGroupCoordinator,
            build_http_server,
            coordinator_process_guard,
        )
        from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator

        def factory():
            return HttpGroupCoordinator(GroupCoordinator(root, args.source_root, args.limit,
                max_groups=args.max_groups, transport="http", coordinator_host=socket.gethostname()))

        def stop(signum, frame):
            raise KeyboardInterrupt

        with coordinator_process_guard(args.local_lock_path):
            server = build_http_server(args.host, args.port, service_factory=factory,
                                       bearer_token=os.environ["R2V_GROUP_COORDINATOR_TOKEN"])
            previous = signal.signal(signal.SIGTERM, stop)
            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                server.service.close()
                signal.signal(signal.SIGTERM, previous)
        return
    if args.command == "status":
        result = pilot_summary(root)
    else:
        result = run_fake_pilot(source_root=args.source_root, run_root=root,
                                workers=args.workers, limit=args.limit,
                                max_groups=args.max_groups, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
