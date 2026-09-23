"""Print same-population T2VA/TA2VA MiMo A/B commands or a read-only report."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.t2va_mimo_stage_server import build_mimo_serve_command
from r2v_data_v2.h3.t2va_shadow import (
    T2VACaseManifest,
    T2VAInventory,
    T2VARecord,
    read_rows,
    sha256_file,
    t2va_root,
)
from r2v_data_v2.h3.ta2va_shadow import TA2VAProduct


def _case_ids(path: Path) -> list[str]:
    ids = T2VACaseManifest.model_validate_json(path.read_text()).clip_uids
    if len(ids) != len(set(ids)):
        raise ValueError("A/B case manifest contains duplicate clip IDs")
    return ids


def build_plan(
    *,
    shot_manifest: Path,
    clips_root: Path,
    source_videos_root: Path,
    shot_index_root: Path,
    audio_production_root: Path,
    audio_shadow_run_id: str,
    case_manifest: Path,
    base_url: str,
    model: str,
    media_root: Path,
    sglang_bin: str,
    checkpoint: str,
    run_prefix: str,
    allow_unverified: bool = False,
) -> dict:
    clip_uids = _case_ids(case_manifest)
    endpoint = build_mimo_serve_command(sglang_bin, checkpoint, served_model_name=model)
    common = [
        "--shot-manifest",
        str(shot_manifest),
        "--clips-root",
        str(clips_root),
        "--source-videos-root",
        str(source_videos_root),
        "--shot-index-root",
        str(shot_index_root),
        "--audio-production-root",
        str(audio_production_root),
        "--audio-shadow-run-id",
        audio_shadow_run_id,
        "--case-manifest",
        str(case_manifest),
    ]
    runtime = [
        "--base-url",
        base_url,
        "--transport",
        "sglang",
        "--model",
        model,
        "--media-root",
        str(media_root),
    ]
    result = {"clip_uids": clip_uids, "endpoint": endpoint}
    for mode in ("multi", "single"):
        t2va_run_id = f"{run_prefix}-{mode}-t2va"
        ta2va_run_id = f"{run_prefix}-{mode}-ta2va"
        result[f"{mode}_t2va"] = [
            "python",
            "tools/run_h3_t2va_shadow.py",
            *common,
            "--t2va-run-id",
            t2va_run_id,
            *runtime,
            "--call-mode",
            mode,
        ]
        result[f"{mode}_ta2va"] = [
            "python",
            "tools/run_h3_ta2va_shadow.py",
            "--t2va-root",
            str(t2va_root(audio_production_root, t2va_run_id)),
            "--ta2va-run-id",
            ta2va_run_id,
            *runtime,
            *(["--allow-unverified"] if allow_unverified else []),
        ]
    return result


def _load_t2va(root: Path) -> tuple[T2VAInventory, list[T2VARecord]]:
    inventory = T2VAInventory.model_validate_json((root / "inventory.json").read_text())
    records = read_rows(root / "records.jsonl", T2VARecord)
    if [record.clip_uid for record in records] != inventory.clip_uids:
        raise ValueError("A/B T2VA record order differs from inventory")
    return inventory, records


def _load_ta2va(
    root: Path, inventory: T2VAInventory, t2va_run_root: Path
) -> list[TA2VAProduct]:
    source = json.loads((root / "inventory.json").read_text())
    records = read_rows(root / "records.jsonl", TA2VAProduct)
    if (
        source["source_t2va_root"] != str(t2va_run_root.resolve())
        or source["source_inventory_fingerprint"] != inventory.inventory_fingerprint
    ):
        raise ValueError("A/B TA2VA source differs from selected T2VA run")
    ready_ids = [
        record.clip_uid
        for record in read_rows(t2va_run_root / "records.jsonl", T2VARecord)
        if record.status == "ready"
    ]
    if source["clip_uids"] != ready_ids:
        raise ValueError("A/B TA2VA clip inventory differs from ready T2VA")
    keys = [(record.clip_uid, record.variant) for record in records]
    if len(keys) != len(set(keys)) or {uid for uid, _ in keys} - set(ready_ids):
        raise ValueError("A/B TA2VA product inventory differs")
    return records


def _side(root: Path, ta_root: Path, records, products) -> dict:
    by_clip: dict[str, list[TA2VAProduct]] = {}
    for product in products:
        by_clip.setdefault(product.clip_uid, []).append(product)
    clips = {}
    for record in records:
        items = by_clip.get(record.clip_uid, [])
        ta_status = (
            "skipped"
            if record.status != "ready"
            else "not_published"
            if not items
            else "failed"
            if any(item.status == "failed" for item in items)
            else "ready"
        )
        clips[record.clip_uid] = {
            "t2va_status": record.status,
            "t2va_model_call_count": record.model_call_count,
            "ta2va_status": ta_status,
            "ta2va_model_call_count": sum(item.model_call_count for item in items),
            "ta2va_variants": {item.variant: item.status for item in items},
            "raw_path": str(root / "raw" / f"{record.clip_uid}.json"),
            "core_path": str(root / "core" / f"{record.clip_uid}.json")
            if record.status == "ready"
            else None,
            "prompt_path": str(root / "prompts" / f"{record.clip_uid}.txt")
            if record.status == "ready"
            else None,
            "ta2va_prompt_paths": [
                str(ta_root / "prompts" / f"{item.clip_uid}.{item.variant}.txt")
                for item in items
                if item.status == "ready"
            ],
        }
    return clips


def build_report(
    *,
    case_manifest: Path,
    multi_t2va_root: Path,
    multi_ta2va_root: Path,
    single_t2va_root: Path,
    single_ta2va_root: Path,
) -> dict:
    ids = _case_ids(case_manifest)
    roots = {
        "multi": (multi_t2va_root, multi_ta2va_root),
        "single": (single_t2va_root, single_ta2va_root),
    }
    inventories = {}
    sides = {}
    for mode, (t2va_run_root, ta_root) in roots.items():
        inventory, records = _load_t2va(t2va_run_root)
        if (
            inventory.clip_uids != ids
            or inventory.shot_selection.case_manifest_path
            != str(case_manifest.resolve())
            or inventory.shot_selection.case_manifest_sha256
            != sha256_file(case_manifest)
        ):
            raise ValueError("A/B ordered case manifest differs from T2VA inventory")
        expected_mode = "single" if mode == "single" else "multi"
        actual_mode = getattr(inventory.backend, "call_mode", "multi")
        if actual_mode != expected_mode:
            raise ValueError("A/B T2VA call mode differs from selected side")
        products = _load_ta2va(ta_root, inventory, t2va_run_root)
        inventories[mode] = inventory
        sides[mode] = _side(t2va_run_root, ta_root, records, products)
    multi, single = inventories["multi"], inventories["single"]
    if (
        multi.shot_selection != single.shot_selection
        or multi.source_hashes != single.source_hashes
        or multi.audio_production_root != single.audio_production_root
        or multi.audio_shadow_run_id != single.audio_shadow_run_id
        or multi.jobs != single.jobs
        or multi.backend.model != single.backend.model
    ):
        raise ValueError("A/B upstream source, model, or ordered jobs differ")
    clips = [
        {"clip_uid": uid, "multi": sides["multi"][uid], "single": sides["single"][uid]}
        for uid in ids
    ]
    totals = {}
    for mode in ("multi", "single"):
        rows = [row[mode] for row in clips]
        totals[mode] = {
            "t2va_ready_count": sum(r["t2va_status"] == "ready" for r in rows),
            "t2va_failed_count": sum(r["t2va_status"] == "failed" for r in rows),
            "ta2va_ready_count": sum(r["ta2va_status"] == "ready" for r in rows),
            "ta2va_failed_count": sum(r["ta2va_status"] == "failed" for r in rows),
            "ready_clips": [
                row["clip_uid"] for row in clips if row[mode]["ta2va_status"] == "ready"
            ],
            "failed_clips": [
                row["clip_uid"]
                for row in clips
                if row[mode]["t2va_status"] == "failed"
                or row[mode]["ta2va_status"] == "failed"
            ],
            "model_call_count": sum(
                r["t2va_model_call_count"] + r["ta2va_model_call_count"] for r in rows
            ),
        }
    return {
        "case_manifest": str(case_manifest.resolve()),
        "clip_uids": ids,
        "model": multi.backend.model,
        "clips": clips,
        "totals": totals,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--shot-manifest", type=Path, required=True)
    plan.add_argument("--clips-root", type=Path, required=True)
    plan.add_argument("--source-videos-root", type=Path, required=True)
    plan.add_argument("--shot-index-root", type=Path, required=True)
    plan.add_argument("--audio-production-root", type=Path, required=True)
    plan.add_argument("--audio-shadow-run-id", required=True)
    plan.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    plan.add_argument("--model", default="mimo-v2.6-flash-rl")
    plan.add_argument("--media-root", type=Path, required=True)
    plan.add_argument("--sglang-bin", required=True)
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--run-prefix", default="v26-random20")
    plan.add_argument("--allow-unverified", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--multi-t2va-root", type=Path, required=True)
    report.add_argument("--multi-ta2va-root", type=Path, required=True)
    report.add_argument("--single-t2va-root", type=Path, required=True)
    report.add_argument("--single-ta2va-root", type=Path, required=True)
    for command in (plan, report):
        command.add_argument("--case-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    values = vars(args)
    action = values.pop("action")
    return build_plan(**values) if action == "plan" else build_report(**values)


if __name__ == "__main__":
    output = main()
    if "endpoint" in output:
        print("Case order:", ", ".join(output["clip_uids"]))
        for label in (
            "endpoint",
            "multi_t2va",
            "multi_ta2va",
            "single_t2va",
            "single_ta2va",
        ):
            print(f"{label}:\n{shlex.join(output[label])}\n")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))
