"""CPU-only replay/export of an already frozen Group pilot; never requests a model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.ra2va_group_export import export_group_clip
from r2v_data_v2.h3.ra2va_group_local import _collect_export, read_json_lines
from r2v_data_v2.h3.ra2va_group_mimo import reconcile_group_job
from r2v_data_v2.h3.ra2va_group_pipeline import GroupFrozenJob
from r2v_data_v2.h3.ra2va_group_source import GroupTask
from r2v_data_v2.h3.t2va_production import atomic_json


def _stage_rows(root, stage):
    return {row["clip_uid"]: (path, row) for path in sorted((root / "stages").glob(f"*-{stage}/results/*.json"))
            for row in [json.loads(path.read_text())]}


def replay_group(*, source_group: Path, output_root: Path, ffmpeg="ffmpeg") -> dict:
    source_group, output_root = Path(source_group), Path(output_root)
    tasks = [GroupTask(**row) for row in read_json_lines(source_group / "inventory/tasks.jsonl")]
    asr, resolved, original = (_stage_rows(source_group, stage) for stage in ("asr", "resolve", "mimo"))
    output_root.mkdir(parents=True, exist_ok=False)
    started, clips = time.monotonic(), []
    for task in tasks:
        clip_started = time.monotonic()
        source_path, source_row = original[task.clip_uid]
        previous = source_row["payload"]
        provenance = previous["backend_provenance"]
        config = MimoBackendConfig(api_key="offline-replay", **{key: provenance[key] for key in (
            "transport", "base_url", "model", "temperature", "max_completion_tokens", "video_fps",
            "media_resolution", "http_max_attempts")}, thinking=provenance["thinking"]["type"],
            media_resolver=MimoMediaResolver(mode=provenance["media_mode"],
                media_root=Path(provenance["media_root"]), media_base_url=provenance["media_base_url"]))
        job = GroupFrozenJob.model_validate(asr[task.clip_uid][1]["payload"]["job"])
        stems = resolved[task.clip_uid][1]["payload"]
        clip_root = output_root / "clips" / task.clip_uid
        result = reconcile_group_job(root=clip_root / "reconcile", config=config, client=None, job=job,
            stems={kind: stems[kind]["path"] for kind in ("speech", "music", "sfx")},
            replay_roots={turn: Path(previous["response_sources"][turn]).parent for turn in ("visual", "joint")})
        result["source_record_path"] = str(source_path)
        result["source_postprocessing_version"] = previous.get("postprocessing_version")
        atomic_json(clip_root / "reconcile/reconcile.json", result)
        exported = {"reason": "upstream_terminal"}
        if result["status"] == "ready":
            exported = export_group_clip(root=clip_root / "bundle", task=task, job=job,
                                         result=result, stems=stems, ffmpeg=ffmpeg)
        for stage, payload in (("000006-asr", asr[task.clip_uid][1]["payload"]),
                               ("000007-mimo", result), ("000008-export", exported)):
            atomic_json(output_root / "stages" / stage / "results" / f"{task.ordinal:08d}.json",
                        {"clip_uid": task.clip_uid, "payload": payload})
        clip = {"clip_uid": task.clip_uid, "reconcile_status": result["status"],
            "product_ready_count": exported.get("product_ready_count", 0),
            "product_failed_count": exported.get("product_failed_count", 0),
            "failure_reason": result.get("failure_reason"), "failure_issues": result.get("failure_issues", []),
            "product_failure_reasons": exported.get("failure_reasons", []),
            "correction_counts": result.get("deterministic_correction_counts", {}),
            "identity_exclusions": exported.get("identity_exclusions", []),
            "new_request_count": result["new_request_count"], "response_replay_count": result["response_replay_count"],
            "historical_model_call_count": result["model_call_count"],
            "elapsed_seconds": round(time.monotonic() - clip_started, 3)}
        clips.append(clip)
        print(json.dumps(clip, ensure_ascii=False), flush=True)
    summary = {**_collect_export(output_root), "source_group": str(source_group), "clips": clips,
        "clip_count": len(clips), "video_count": sum(c["product_ready_count"] > 0 for c in clips),
        "reconcile_ready_count": sum(c["reconcile_status"] == "ready" for c in clips),
        "reconcile_failed_count": sum(c["reconcile_status"] == "failed" for c in clips),
        **{key: sum(c[key] for c in clips) for key in (
            "new_request_count", "response_replay_count", "historical_model_call_count")},
        "elapsed_seconds": round(time.monotonic() - started, 3)}
    atomic_json(output_root / "offline_replay.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-group", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    print(json.dumps(replay_group(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
