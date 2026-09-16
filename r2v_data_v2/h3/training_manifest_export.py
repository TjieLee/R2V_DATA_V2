from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TASK_ORDER = ("r2va", "ra2va", "t2va", "ta2va")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jobs_by_clip(inventory_path: Path) -> dict[str, dict]:
    jobs = _read_json(inventory_path).get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"inventory jobs missing: {inventory_path}")
    result = {}
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("clip_uid"), str):
            raise ValueError(f"invalid job in {inventory_path}")
        clip_uid = job["clip_uid"]
        if clip_uid in result:
            raise ValueError(f"duplicate clip_uid in {inventory_path}: {clip_uid}")
        result[clip_uid] = job
    return result


def _caption_row(video: str, images: list[str], audios: list[str], caption: str) -> dict:
    if not isinstance(video, str) or not video:
        raise ValueError("training manifest row requires video path")
    if not isinstance(caption, str) or not caption:
        raise ValueError("training manifest row requires caption")
    if any(not isinstance(path, str) or not path for path in [*images, *audios]):
        raise ValueError("training manifest media paths must be non-empty strings")
    return {"video": video, "images": images, "audios": audios, "caption": caption}


def _export_ra2va(shadow: Path) -> tuple[list[dict], list[dict]]:
    prepared = shadow / "h3_audio_reuse_prepared_v1"
    products = shadow / "h3_audio_reuse_products_v1"
    jobs = _jobs_by_clip(prepared / "inventory.json")
    r2va, ra2va = [], []
    for product in _read_jsonl(products / "records.jsonl"):
        if product.get("status") != "ready":
            continue
        clip_uid = product.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(f"R(A)2VA product clip missing from inventory: {clip_uid}")
        references = job.get("reference_images", [])
        if not isinstance(references, list):
            raise ValueError(f"invalid reference_images for {clip_uid}")
        images = [
            item["image_artifact_path"]
            for item in sorted(references, key=lambda item: item.get("image_index", 0))
        ]
        variant = product.get("conditioning_variant")
        caption = product.get("rendered_h3_prompt")
        if variant == "visual_only":
            r2va.append(_caption_row(job["target_video_path"], images, [], caption))
            continue
        audio_references = product.get("audio_references", [])
        if not isinstance(audio_references, list):
            raise ValueError(f"invalid audio_references for {clip_uid}")
        audios = []
        for reference in audio_references:
            contract = reference.get("contract") if isinstance(reference, dict) else None
            path = contract.get("path") if isinstance(contract, dict) else None
            if not isinstance(path, str) or not path:
                raise ValueError(f"RA2VA Audio path missing for {clip_uid}")
            audios.append(path)
        if audios:
            ra2va.append(_caption_row(job["target_video_path"], images, audios, caption))
    return r2va, ra2va


def _export_t2va(root: Path) -> list[dict]:
    jobs = _jobs_by_clip(root / "inventory.json")
    result = []
    for record in _read_jsonl(root / "records.jsonl"):
        if record.get("status") != "ready":
            continue
        clip_uid = record.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(f"T2VA record clip missing from inventory: {clip_uid}")
        caption = (root / "prompts" / f"{clip_uid}.txt").read_text(encoding="utf-8")
        result.append(_caption_row(job["target_video_path"], [], [], caption))
    return result


def _export_ta2va(root: Path) -> list[dict]:
    inventory = _read_json(root / "inventory.json")
    source_root = inventory.get("source_t2va_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("TA2VA inventory requires source_t2va_root")
    jobs = _jobs_by_clip(Path(source_root) / "inventory.json")
    result = []
    for product in _read_jsonl(root / "records.jsonl"):
        if product.get("status") != "ready":
            continue
        clip_uid = product.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(f"TA2VA product clip missing from T2VA inventory: {clip_uid}")
        references = product.get("audio_references", [])
        if not isinstance(references, list):
            raise ValueError(f"invalid TA2VA audio_references for {clip_uid}")
        audios = []
        for reference in references:
            path = reference.get("path") if isinstance(reference, dict) else None
            if not isinstance(path, str) or not path:
                raise ValueError(f"TA2VA Audio path missing for {clip_uid}")
            audios.append(path)
        result.append(_caption_row(job["target_video_path"], [], audios, product.get("prompt")))
    return result


def export_training_manifests(
    *,
    output_root: Path,
    ra2va_shadow_root: Path | None = None,
    t2va_root: Path | None = None,
    ta2va_root: Path | None = None,
) -> Path:
    output = output_root.expanduser()
    if output.exists():
        raise FileExistsError(output)

    rows = {task: [] for task in TASK_ORDER}
    if ra2va_shadow_root is not None:
        rows["r2va"], rows["ra2va"] = _export_ra2va(ra2va_shadow_root.expanduser())
    if t2va_root is not None:
        rows["t2va"] = _export_t2va(t2va_root.expanduser())
    if ta2va_root is not None:
        rows["ta2va"] = _export_ta2va(ta2va_root.expanduser())

    output.mkdir(parents=True)
    for task in TASK_ORDER:
        _write_jsonl(output / f"{task}.jsonl", rows[task])

    tasks_by_video: dict[str, set[str]] = defaultdict(set)
    for task in TASK_ORDER:
        for row in rows[task]:
            tasks_by_video[row["video"]].add(task)
    summary = [
        {
            "video": video,
            "tasks": [task for task in TASK_ORDER if task in tasks],
        }
        for video, tasks in sorted(tasks_by_video.items())
    ]
    _write_jsonl(output / "videos.jsonl", summary)
    return output
