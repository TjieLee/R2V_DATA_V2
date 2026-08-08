from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files
from tools.build_v3_disagreement_review_board import (
    _candidate_images,
    _safe_component,
)

ReviewMode = Literal[
    "darkest",
    "blur",
    "no_face",
    "small_face",
    "extreme_pose",
    "all",
]

ALLOWED_REVIEW_ROOT = Path("/mnt/workspace/litengjie/data/r2v_v3_reviews")
_BOARD_WIDTH = 1320
_HEADER_HEIGHT = 150
_PANEL_WIDTH = 610
_PANEL_HEIGHT = 420
_METRICS_HEIGHT = 190
_MARGIN = 30
_PAGE_BACKGROUND = (250, 250, 248)
_PANEL_BACKGROUND = (236, 239, 242)
_TEXT = (38, 42, 48)
_MUTED = (92, 98, 108)

_MODE_LISTS: dict[str, tuple[str, ...]] = {
    "darkest": ("darkest_candidates", "highest_dark_fraction_candidates"),
    "blur": ("lowest_laplacian_candidates", "lowest_tenengrad_candidates"),
    "no_face": ("no_face_candidates",),
    "small_face": ("smallest_face_candidates",),
    "extreme_pose": ("largest_abs_yaw_candidates",),
    "all": (
        "darkest_candidates",
        "highest_dark_fraction_candidates",
        "lowest_laplacian_candidates",
        "lowest_tenengrad_candidates",
        "lowest_contrast_candidates",
        "no_face_candidates",
        "smallest_face_candidates",
        "largest_abs_yaw_candidates",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only V3 reference-filter extreme review board",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("darkest", "blur", "no_face", "small_face", "extreme_pose", "all"),
        default="all",
    )
    return parser


def _validated_input_root(path: Path, *, label: str) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory")
    return root


def _validated_output_root(output_root: Path, inputs: tuple[Path, ...]) -> Path:
    output = output_root.expanduser().resolve(strict=False)
    allowed = ALLOWED_REVIEW_ROOT.expanduser().resolve(strict=False)
    if output == allowed or allowed not in output.parents:
        raise ValueError("review output must be below the allowed review root")
    if any(
        output == source or source in output.parents or output in source.parents
        for source in inputs
    ):
        raise ValueError("review output must be separate from all input roots")
    if output.exists():
        raise FileExistsError(f"review output already exists: {output}")
    return output


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _load_records(audit_root: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in (audit_root / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise TypeError("audit JSONL records must be objects")
    return records


def _case_key(value: Mapping[str, object]) -> tuple[str, str, str]:
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("extreme review entry must identify a candidate")
    return str(value["clip_uid"]), str(value["entity_id"]), candidate_id


def _selected_cases(
    records: list[dict[str, object]],
    summary: Mapping[str, object],
    *,
    mode: ReviewMode,
) -> list[dict[str, object]]:
    review_lists = summary.get("review_lists")
    if not isinstance(review_lists, Mapping):
        raise TypeError("audit summary is missing review_lists")
    record_by_key = {
        _case_key(record): record
        for record in records
        if record.get("artifact_scope") == "candidate"
    }
    selected: dict[tuple[str, str, str], dict[str, object]] = {}
    for list_name in _MODE_LISTS[mode]:
        values = review_lists.get(list_name)
        if not isinstance(values, list):
            raise TypeError(f"audit summary is missing review list: {list_name}")
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError("review list entries must be objects")
            key = _case_key(value)
            record = record_by_key.get(key)
            if record is None:
                raise ValueError("review list references a missing audit candidate")
            case = selected.setdefault(
                key,
                {
                    "clip_uid": key[0],
                    "entity_id": key[1],
                    "candidate_id": key[2],
                    "phrase": record.get("phrase"),
                    "reference_type": record.get("reference_type"),
                    "frame_slot": record.get("frame_slot"),
                    "source_frame_index": record.get("source_frame_index"),
                    "qwen_selected": bool(record.get("is_current_selected")),
                    "review_reasons": [],
                    "audit_record": record,
                },
            )
            reasons = case["review_reasons"]
            assert isinstance(reasons, list)
            reasons.append(list_name)
    return list(selected.values())


def _fit_panel(image: Image.Image) -> Image.Image:
    return ImageOps.contain(
        image.convert("RGB"),
        (_PANEL_WIDTH, _PANEL_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )


def _paste_centered(
    board: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    panel = _fit_panel(image)
    board.paste(
        panel,
        (
            left + (right - left - panel.width) // 2,
            top + (bottom - top - panel.height) // 2,
        ),
    )


def _text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: object,
    *,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    content = str(value)
    try:
        draw.text(position, content, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(
            position,
            content.encode("ascii", "replace").decode("ascii"),
            fill=fill,
            font=font,
        )


def _metric(section: object, field: str) -> object:
    return section.get(field) if isinstance(section, Mapping) else None


def _format_metric(value: object) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "n/a"


def _embedding_label(record: Mapping[str, object]) -> str:
    embedding = record.get("embedding")
    if not isinstance(embedding, Mapping) or embedding.get("status") != "succeeded":
        return "embedding representativeness=n/a"
    backend = str(embedding.get("backend", "embedding"))
    score = _format_metric(embedding.get("representativeness_score"))
    return f"{backend} representativeness={score}"


def _render_case(
    case: dict[str, object],
    *,
    run_root: Path,
    destination: Path,
    crop_padding_ratio: float,
    evidence_cache: dict[str, object],
) -> None:
    record = case["audit_record"]
    if not isinstance(record, Mapping):
        raise TypeError("review case audit_record must be an object")
    context, isolated, source_path = _candidate_images(
        run_root,
        case,
        case,
        crop_padding_ratio=crop_padding_ratio,
        evidence_cache=evidence_cache,
    )
    case["source_image_path"] = source_path
    board_height = _HEADER_HEIGHT + _PANEL_HEIGHT + _METRICS_HEIGHT
    board = Image.new("RGB", (_BOARD_WIDTH, board_height), _PAGE_BACKGROUND)
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    _text(
        draw,
        (_MARGIN, 22),
        f"{case['clip_uid']} / {case['entity_id']} / {case['candidate_id']}",
        fill=_TEXT,
        font=font,
    )
    _text(
        draw,
        (_MARGIN, 48),
        f"{case['reference_type']} / {case['phrase']}",
        fill=_TEXT,
        font=font,
    )
    _text(
        draw,
        (_MARGIN, 74),
        f"review reasons: {', '.join(case['review_reasons'])}",
        fill=_MUTED,
        font=font,
    )
    _text(
        draw,
        (_MARGIN, 100),
        f"Qwen selected? {'yes' if case['qwen_selected'] else 'no'}",
        fill=_MUTED,
        font=font,
    )
    left_box = (_MARGIN, _HEADER_HEIGHT, _MARGIN + _PANEL_WIDTH, _HEADER_HEIGHT + _PANEL_HEIGHT)
    right_box = (
        _BOARD_WIDTH - _MARGIN - _PANEL_WIDTH,
        _HEADER_HEIGHT,
        _BOARD_WIDTH - _MARGIN,
        _HEADER_HEIGHT + _PANEL_HEIGHT,
    )
    draw.rectangle(left_box, fill=_PANEL_BACKGROUND, outline=(188, 193, 200))
    draw.rectangle(right_box, fill=_PANEL_BACKGROUND, outline=(188, 193, 200))
    _paste_centered(board, context, left_box)
    _paste_centered(board, isolated, right_box)
    _text(draw, (left_box[0] + 8, left_box[1] + 8), "context image", fill=_MUTED, font=font)
    _text(draw, (right_box[0] + 8, right_box[1] + 8), "isolated crop", fill=_MUTED, font=font)

    technical = record.get("technical_quality")
    pose = record.get("subject_pose")
    lines = [
        (
            "luma_mean="
            f"{_format_metric(_metric(technical, 'luma_mean'))}   "
            "dark_fraction_32="
            f"{_format_metric(_metric(technical, 'dark_fraction_32'))}   "
            "rms_contrast="
            f"{_format_metric(_metric(technical, 'rms_contrast'))}"
        ),
        (
            "laplacian_variance="
            f"{_format_metric(_metric(technical, 'laplacian_variance'))}   "
            "tenengrad_mean="
            f"{_format_metric(_metric(technical, 'tenengrad_mean'))}"
        ),
        _embedding_label(record),
    ]
    if case["reference_type"] == "subject":
        lines.append(
            "face_detected="
            f"{_metric(pose, 'face_detected')}   face_bbox_area_ratio="
            f"{_format_metric(_metric(pose, 'face_bbox_area_ratio'))}   "
            f"yaw={_format_metric(_metric(pose, 'yaw'))}   "
            f"pitch={_format_metric(_metric(pose, 'pitch'))}   "
            f"roll={_format_metric(_metric(pose, 'roll'))}"
        )
    text_top = _HEADER_HEIGHT + _PANEL_HEIGHT + 20
    for index, line in enumerate(lines):
        _text(
            draw,
            (_MARGIN, text_top + index * 30),
            line,
            fill=_TEXT if index < 2 else _MUTED,
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, format="PNG")


def _render_html(cases: list[dict[str, object]], summary: Mapping[str, object]) -> str:
    cards = []
    for case in cases:
        board_path = html.escape(str(case["board_path"]), quote=True)
        title = html.escape(
            f"{case['clip_uid']} / {case['entity_id']} / {case['candidate_id']}"
        )
        reasons = html.escape(", ".join(case["review_reasons"]))
        cards.append(
            f'<article><a href="{board_path}"><img src="{board_path}" '
            f'alt="{title}"></a><h2>{title}</h2><p>{reasons}</p></article>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V3 Reference Filter Extreme Review</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #24282e; background: #f5f5f2; }}
header {{ padding: 24px 32px; background: #fff; border-bottom: 1px solid #d8dadd; }}
h1 {{ margin: 0 0 10px; font-size: 28px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; padding: 24px 32px; }}
article {{ min-width: 0; padding: 12px; background: #fff; border: 1px solid #d8dadd; }}
article img {{ display: block; width: 100%; aspect-ratio: 1.7; object-fit: cover; border: 1px solid #ececef; }}
article h2 {{ margin: 12px 0 6px; font-size: 16px; overflow-wrap: anywhere; }}
article p {{ margin: 0; color: #626873; }}
</style>
</head>
<body>
<header><h1>V3 Reference Filter Extreme Review</h1><p>Mode: {html.escape(str(summary['mode']))} / Cases: {summary['case_count']}</p></header>
<main>{''.join(cards)}</main>
</body>
</html>
"""


def build_extreme_review_board(
    *,
    run_root: Path,
    audit_root: Path,
    output_root: Path,
    mode: ReviewMode = "all",
) -> dict[str, object]:
    if mode not in _MODE_LISTS:
        raise ValueError(f"unsupported review mode: {mode}")
    source = _validated_input_root(run_root, label="run_root")
    audit = _validated_input_root(audit_root, label="audit_root")
    inputs = (source, audit)
    output = _validated_output_root(output_root, inputs)
    before = {root: snapshot_run_files(root) for root in inputs}
    audit_summary = _load_json(audit / "audit.summary.json")
    records = _load_records(audit)
    cases = _selected_cases(records, audit_summary, mode=mode)
    crop_ratios = {
        float(case["audit_record"].get("crop_padding_ratio"))
        for case in cases
        if isinstance(case.get("audit_record"), Mapping)
        and isinstance(case["audit_record"].get("crop_padding_ratio"), (int, float))
    }
    if len(crop_ratios) > 1:
        raise ValueError("audit candidates use inconsistent crop padding ratios")
    crop_padding_ratio = next(iter(crop_ratios), 0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"review temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        evidence_cache: dict[str, object] = {}
        for case in cases:
            filename = (
                f"{_safe_component(case['clip_uid'])}__"
                f"{_safe_component(case['entity_id'])}__"
                f"{_safe_component(case['candidate_id'])}.png"
            )
            board_path = (Path("cases") / filename).as_posix()
            case["board_path"] = board_path
            _render_case(
                case,
                run_root=source,
                destination=temporary / board_path,
                crop_padding_ratio=crop_padding_ratio,
                evidence_cache=evidence_cache,
            )
        summary = {
            "schema_version": 1,
            "audit_only": True,
            "qwen_calls_added": 0,
            "mode": mode,
            "case_count": len(cases),
            "run_root": str(source),
            "audit_root": str(audit),
            "crop_padding_ratio": crop_padding_ratio,
            "all_inputs_unchanged": True,
        }
        (temporary / "review_cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "review_index.html").write_text(
            _render_html(cases, summary),
            encoding="utf-8",
        )
        if any(snapshot_run_files(root) != before[root] for root in inputs):
            raise RuntimeError("an input root changed during review generation")
        temporary.replace(output)
        return summary
    except Exception as exc:
        changed = any(snapshot_run_files(root) != before[root] for root in inputs)
        if temporary.exists():
            shutil.rmtree(temporary)
        if changed:
            raise RuntimeError(
                "an input root changed during failed review generation"
            ) from exc
        raise


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = build_extreme_review_board(
        run_root=arguments.run_root,
        audit_root=arguments.audit_root,
        output_root=arguments.output_root,
        mode=arguments.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
