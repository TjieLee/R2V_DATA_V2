"""Rank-zero durable scheduling; all ranks run the same persistent inference."""

import gc
import json
import time
from pathlib import Path

from .h3_pair_state import (
    atomic_json,
    begin_attempt,
    eligible,
    fail_attempt,
    now,
    publish_generated,
    read_json,
)
from .timeline import inspect_video_timeline


def validate_output(path, job):
    actual = inspect_video_timeline(path)
    if (actual.frame_count != job["frames"] or actual.rate != 24
            or (actual.width,actual.height) != (job["width"],job["height"])
            or abs(actual.duration_seconds-job["frames"]/24) > 1/24):
        raise ValueError("Generated video violates native count/FPS/resolution/duration")


def _next_job(config):
    for case in config["cases"]:
        if not eligible(case,"generate",config["limits"][case["case_id"]]["generate"]):
            continue
        job = read_json(Path(case["directory"])/"preparation/prepared.json")
        for key, expected in (("case_id",case["case_id"]),("row_sha256",case["row_sha256"]),
                              ("identity",config["identity"])):
            if job.get(key) != expected:
                raise ValueError(f"Prepared case {key} mismatch")
        attempt = begin_attempt(case,"generate",{"pair_id":config["pair_id"],"rank":0})
        temporary = Path(case["directory"])/"tmp"/f"generate-{attempt['attempt']:03d}"/"raw.mp4"
        temporary.parent.mkdir(parents=True,exist_ok=False)
        return {"kind":"job","case":case,"job":job,"attempt":attempt,"temporary":str(temporary)}
    return {"kind":"stop"}


def generate_loop(config, channel, backend_factory, stats_path):
    backend = None
    stats = {"worker_started_at":now(),"model_load_count":0,"pdd_apply_count":0,
             "jobs_attempted":0,"jobs_succeeded":0,"jobs_failed":0,"restart_required":False}
    while True:
        message = channel.broadcast(_next_job(config) if channel.rank == 0 else None)
        if message["kind"] == "stop":
            break
        if backend is None:
            if channel.rank == 0:
                atomic_json(stats_path,{**stats,"phase":"model_loading"})
            backend = backend_factory()  # once per healthy torchrun session; init failures are fatal
            stats.update(backend.metadata)
        started = time.monotonic()
        stats["jobs_attempted"] += 1
        result, reference, error = None, None, None
        try:
            reference = backend.prepare(message["job"])
        except Exception as exc:  # noqa: BLE001 -- decode errors have not touched FSDP state
            error = {"error_type":type(exc).__name__,"error":str(exc)}
        prepared = channel.gather({"error":error})
        prepare_failed = any(r["error"] is not None for r in prepared)
        if not prepare_failed:
            if channel.rank == 0:
                atomic_json(stats_path,{**stats,"phase":"inference",
                    "active_case":message["case"],"active_attempt":message["attempt"]})
            try:
                result = backend.infer(message["job"],reference)
            except Exception as exc:  # noqa: BLE001 -- FSDP forward state may be partially advanced
                error = {"error_type":type(exc).__name__,"error":str(exc)}
                # Out-of-band receipt survives a rank-asymmetric OOM that also
                # breaks the subsequent collective. No nonzero-rank file writes.
                print("H3_PAIR_INFERENCE_ERROR "+json.dumps({
                    "case_id":message["case"]["case_id"],"attempt":message["attempt"]["attempt"],
                    "rank":channel.rank,**error}),flush=True)
        else:
            error = next(r["error"] for r in prepared if r["error"] is not None)
        reference = None
        reports = channel.gather({"error":error,"memory":backend.memory()})
        decision = None
        if channel.rank == 0:
            failures = [report["error"] for report in reports if report["error"] is not None]
            if failures:
                fail_attempt(message["case"],message["attempt"],RuntimeError(str(failures)))
                stats["jobs_failed"] += 1
                stats["restart_required"] = not prepare_failed
                decision = {"kind":"continue" if prepare_failed else "restart"}
            else:
                try:
                    temporary = Path(message["temporary"])
                    backend.encode(result,temporary)
                    manifest = {**message["job"],**backend.metadata,"variant":"text_two_person",
                                "references":[{"type":"video","path":message["job"]["source"]}],
                                "attempt":message["attempt"],"rank_memory":[r["memory"] for r in reports],
                                "wall_seconds":time.monotonic()-started,
                                "worker_started_at":stats["worker_started_at"],
                                "production_timeline_compatible":False}
                    publish_generated(message["case"],temporary,manifest,
                                      lambda path,job=message["job"]:validate_output(path,job))
                    stats["jobs_succeeded"] += 1
                except Exception as exc:  # noqa: BLE001 -- codec/validation failures do not poison model
                    fail_attempt(message["case"],message["attempt"],exc)
                    stats["jobs_failed"] += 1
                decision = {"kind":"continue"}
            atomic_json(stats_path,{**stats,"phase":"idle"})
        result = None
        gc.collect()
        decision = channel.broadcast(decision)
        if decision["kind"] == "restart":
            stats["restart_required"] = True
            break
    if channel.rank == 0:
        atomic_json(stats_path,stats)
    return stats
