from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

VISUAL_MODES = ("reference", "first_frame", "last_frame", "first_last_frame")
TASK_ORDER = (
    "r2va_reference",
    "r2va_first_frame",
    "r2va_last_frame",
    "r2va_first_last_frame",
    "ra2va_reference_full_audio",
    "ra2va_reference_speech_bgm",
    "ra2va_first_frame_full_audio",
    "ra2va_first_frame_speech_bgm",
    "ra2va_last_frame_full_audio",
    "ra2va_last_frame_speech_bgm",
    "ra2va_first_last_frame_full_audio",
    "ra2va_first_last_frame_speech_bgm",
    "t2va",
    "ta2va_full_audio",
    "ta2va_speech_bgm",
)


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
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
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


def _audio_paths(
    references: object, *, clip_uid: object, nested_contract: bool
) -> list[str]:
    if not isinstance(references, list):
        raise ValueError(f"invalid audio_references for {clip_uid}")
    result = []
    for reference in references:
        item = (
            reference.get("contract")
            if nested_contract and isinstance(reference, dict)
            else reference
        )
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError(f"Audio path missing for {clip_uid}")
        result.append(path)
    return result


def _reference_images(job: dict, clip_uid: object) -> list[str]:
    references = job.get("reference_images", [])
    if not isinstance(references, list):
        raise ValueError(f"invalid reference_images for {clip_uid}")
    return [
        item["image_artifact_path"]
        for item in sorted(references, key=lambda item: item.get("image_index", 0))
    ]


def _frame_images(product: dict) -> list[str]:
    references = product.get("frame_references", [])
    if not isinstance(references, list):
        raise ValueError(f"invalid frame_references for {product.get('clip_uid')}")
    return [
        item["image_path"]
        for item in sorted(references, key=lambda item: item.get("picture_index", 0))
    ]


def _audio_task(mode: str, variant: object) -> str | None:
    suffix = {
        "full_audio_reuse": "full_audio",
        "target_speech_reuse": "speech_bgm",
    }.get(variant)
    return None if suffix is None else f"ra2va_{mode}_{suffix}"


def _export_ra2va(shadow: Path) -> dict[str, list[dict]]:
    rows = {
        task: []
        for task in TASK_ORDER
        if task.startswith(("r2va_", "ra2va_"))
    }
    prepared = shadow / "audio_reuse_prepared_v1"
    products = shadow / "h3_audio_reuse_products_v1"
    jobs = _jobs_by_clip(prepared / "inventory.json")

    for product in _read_jsonl(products / "records.jsonl"):
        if product.get("status") != "ready":
            continue
        clip_uid = product.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(
                f"R(A)2VA product clip missing from inventory: {clip_uid}"
            )
        images = _reference_images(job, clip_uid)
        variant = product.get("conditioning_variant")
        caption = product.get("rendered_h3_prompt")
        if variant == "visual_only":
            rows["r2va_reference"].append(
                _caption_row(job["target_video_path"], images, [], caption)
            )
            continue
        task = _audio_task("reference", variant)
        if task is not None:
            audios = _audio_paths(
                product.get("audio_references", []),
                clip_uid=clip_uid,
                nested_contract=True,
            )
            rows[task].append(
                _caption_row(job["target_video_path"], images, audios, caption)
            )

    frame_records = shadow / "h3_frame_conditioned_products_v1" / "records.jsonl"
    if frame_records.is_file():
        for product in _read_jsonl(frame_records):
            mode = product.get("visual_reference_mode")
            if mode not in VISUAL_MODES[1:]:
                continue
            video = product.get("target_video_path")
            images = _frame_images(product)
            caption = product.get("rendered_h3_prompt")
            variant = product.get("conditioning_variant")
            if variant == "visual_only":
                rows[f"r2va_{mode}"].append(
                    _caption_row(video, images, [], caption)
                )
                continue
            task = _audio_task(mode, variant)
            if task is not None:
                audios = _audio_paths(
                    product.get("audio_references", []),
                    clip_uid=product.get("clip_uid"),
                    nested_contract=True,
                )
                rows[task].append(
                    _caption_row(video, images, audios, caption)
                )
    return rows


def _export_t2va(root: Path) -> list[dict]:
    jobs = _jobs_by_clip(root / "inventory.json")
    result = []
    for record in _read_jsonl(root / "records.jsonl"):
        if record.get("status") != "ready":
            continue
        clip_uid = record.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(
                f"T2VA record clip missing from inventory: {clip_uid}"
            )
        caption = (root / "prompts" / f"{clip_uid}.txt").read_text(
            encoding="utf-8"
        )
        result.append(
            _caption_row(job["target_video_path"], [], [], caption)
        )
    return result


def _export_ta2va(root: Path) -> dict[str, list[dict]]:
    inventory = _read_json(root / "inventory.json")
    source_root = inventory.get("source_t2va_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("TA2VA inventory requires source_t2va_root")
    jobs = _jobs_by_clip(Path(source_root) / "inventory.json")
    result = {"ta2va_full_audio": [], "ta2va_speech_bgm": []}
    task_by_variant = {
        "full_audio_reuse": "ta2va_full_audio",
        "target_speech_reuse": "ta2va_speech_bgm",
    }
    for product in _read_jsonl(root / "records.jsonl"):
        if product.get("status") != "ready":
            continue
        task = task_by_variant.get(product.get("variant"))
        if task is None:
            continue
        clip_uid = product.get("clip_uid")
        job = jobs.get(clip_uid)
        if job is None:
            raise ValueError(
                f"TA2VA product clip missing from T2VA inventory: {clip_uid}"
            )
        audios = _audio_paths(
            product.get("audio_references", []),
            clip_uid=clip_uid,
            nested_contract=False,
        )
        result[task].append(
            _caption_row(
                job["target_video_path"], [], audios, product.get("prompt")
            )
        )
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
        rows.update(_export_ra2va(ra2va_shadow_root.expanduser()))
    if t2va_root is not None:
        rows["t2va"] = _export_t2va(t2va_root.expanduser())
    if ta2va_root is not None:
        rows.update(_export_ta2va(ta2va_root.expanduser()))

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
