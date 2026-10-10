"""Bounded RA2VA Group pilots: explicit Fake or Native HTTP execution."""

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
            command.add_argument("--start-row", type=int, default=0)
            command.add_argument("--max-groups", type=int, default=1)
        if name == "run":
            command.add_argument("--fake", action="store_true", required=True)
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--poll-seconds", type=float, default=0.05)
        if name == "coordinator":
            command.add_argument("--execution-mode", choices=("fake", "native", "cpu_fixture"), default="fake")
            command.add_argument("--host", required=True)
            command.add_argument("--port", type=int, required=True)
            command.add_argument("--local-lock-path", type=Path, required=True)
            command.add_argument("--publish-endpoint", action="store_true")
    worker = subparsers.add_parser("worker")
    modes = worker.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fake", action="store_true")
    modes.add_argument("--native", action="store_true")
    worker.add_argument("--configuration", type=Path)
    worker.add_argument("--cpu-fixtures", type=Path)
    worker.add_argument("--ffmpeg", default="ffmpeg")
    worker.add_argument("--coordinator-url", required=True)
    worker.add_argument("--node-id", required=True)
    worker.add_argument("--workers", type=int, default=1)
    worker.add_argument("--fake-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.command == "worker":
        if args.native:
            from r2v_data_v2.h3.ra2va_group_http_native_worker import (
                run_http_native_workers,
            )
            print(json.dumps(run_http_native_workers(coordinator_url=args.coordinator_url, node_id=args.node_id,
                workers=args.workers, configuration_path=args.configuration, cpu_fixtures=args.cpu_fixtures,
                ffmpeg=args.ffmpeg), indent=2), flush=True)
            return
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
        from r2v_data_v2.h3.ra2va_group_launch import (
            coordinator_token,
            local_coordinator_address,
            publish_endpoint,
        )
        from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator

        if args.host == "auto":
            args.host = local_coordinator_address()

        def factory():
            return HttpGroupCoordinator(GroupCoordinator(root, args.source_root, args.limit,
                max_groups=args.max_groups, start_row=args.start_row,
                transport="http", coordinator_host=socket.gethostname(),
                mode={"fake": "cpu_fake_pilot", "native": "http_native_pilot",
                      "cpu_fixture": "http_native_cpu_pilot"}[args.execution_mode],
                coordinator_identity={"guard_path": str(args.local_lock_path.resolve()),
                                      "bind_host": args.host, "bind_port": args.port}))

        def stop(signum, frame):
            raise KeyboardInterrupt

        with coordinator_process_guard(args.local_lock_path):
            token = os.environ.get("R2V_GROUP_COORDINATOR_TOKEN")
            if args.publish_endpoint and not token:
                token = coordinator_token(os.environ.get("R2VA_TOKEN_FILE"))
            server = build_http_server(args.host, args.port, service_factory=factory,
                                       bearer_token=token)
            previous = signal.signal(signal.SIGTERM, stop)
            try:
                if args.publish_endpoint:
                    publish_endpoint(root, args.host, server.server_port)
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
                                max_groups=args.max_groups, start_row=args.start_row,
                                poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
