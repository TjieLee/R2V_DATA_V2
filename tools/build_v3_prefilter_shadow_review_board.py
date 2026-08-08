from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
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
    "near_silhouette",
    "relative_blur",
    "qwen_selected_flagged",
    "all_candidates_flagged",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only V3 prefilter shadow review board",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "near_silhouette",
            "relative_blur",
            "qwen_selected_flagged",
            "all_candidates_flagged",
            "all",
        ),
        default="all",
    )
    return parser


def _validated_input_root(path: Path, *, label: str) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory")
    return root


def _validated_simulation(path: Path) -> Path:
    simulation = path.expanduser().resolve(strict=True)
    if not simulation.is_file():
        raise ValueError("simulation must be a JSON file")
    return simulation


def _validated_output_root(
    output_root: Path,
    *,
    run_root: Path,
    audit_root: Path,
    simulation: Path,
) -> Path:
    output = output_root.expanduser().resolve(strict=False)
    allowed = ALLOWED_REVIEW_ROOT.expanduser().resolve(strict=False)
    if output == allowed or allowed not in output.parents:
        raise ValueError("review output must be below the allowed review root")
    if any(
        output == source or source in output.parents or output in source.parents
        for source in (run_root, audit_root)
    ):
        raise ValueError("review output must be separate from source roots")
    if output == simulation:
        raise ValueError("review output must not replace the simulation")
    if output.exists():
        raise FileExistsError(f"review output already exists: {output}")
    return output


def _load_simulation(path: Path, audit_root: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("audit_only") is not True:
        raise ValueError("simulation must be an audit-only result")
    source = Path(str(value.get("source_audit_root"))).expanduser().resolve(
        strict=False
    )
    if source != audit_root:
        raise ValueError("simulation source audit does not match audit_root")
    candidates = value.get("candidates")
    entities = value.get("entities")
    if not isinstance(candidates, list) or not isinstance(entities, list):
        raise TypeError("simulation candidate and entity results must be lists")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise TypeError("simulation candidates must be objects")
    if not all(isinstance(entity, dict) for entity in entities):
        raise TypeError("simulation entities must be objects")
    return value


def _case_key(value: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(value["clip_uid"]),
        str(value["entity_id"]),
        str(value["candidate_id"]),
    )


def _selected_cases(
    simulation: Mapping[str, object],
    *,
    mode: ReviewMode,
) -> list[dict[str, object]]:
    candidates_value = simulation["candidates"]
    entities_value = simulation["entities"]
    assert isinstance(candidates_value, list)
    assert isinstance(entities_value, list)
    all_flagged_entities = {
        (str(entity["clip_uid"]), str(entity["entity_id"]))
        for entity in entities_value
        if isinstance(entity, Mapping)
        and entity.get("shadow_state") == "all_candidates_flagged"
    }
    cases: list[dict[str, object]] = []
    for candidate in candidates_value:
        assert isinstance(candidate, dict)
        entity_key = (str(candidate["clip_uid"]), str(candidate["entity_id"]))
        reasons = []
        if candidate.get("near_silhouette_flag") is True:
            reasons.append("subject_near_silhouette_v1")
        if candidate.get("relative_blur_flag") is True:
            reasons.append("subject_relative_blur_v1")
        qwen_selected_flagged = bool(
            candidate.get("current_qwen_selected")
            and candidate.get("shadow_flagged")
        )
        all_candidates_flagged = entity_key in all_flagged_entities
        include = {
            "near_silhouette": bool(candidate.get("near_silhouette_flag")),
            "relative_blur": bool(candidate.get("relative_blur_flag")),
            "qwen_selected_flagged": qwen_selected_flagged,
            "all_candidates_flagged": all_candidates_flagged,
            "all": bool(candidate.get("shadow_flagged")),
        }[mode]
        if not include:
            continue
        case = dict(candidate)
        case["shadow_rules"] = reasons
        case["qwen_selected_flagged"] = qwen_selected_flagged
        case["entity_all_candidates_flagged"] = all_candidates_flagged
        cases.append(case)
    return sorted(cases, key=_case_key)


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


def _draw_text(
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


def _format_metric(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    numeric = float(value)
    return f"{numeric:.6f}" if math.isfinite(numeric) else "n/a"


def _mapping_metric(section: object, field: str) -> object:
    return section.get(field) if isinstance(section, Mapping) else None


def _render_case(
    case: dict[str, object],
    *,
    run_root: Path,
    destination: Path,
    evidence_cache: dict[str, object],
) -> None:
    crop_padding_ratio = case.get("crop_padding_ratio")
    padding = float(crop_padding_ratio) if isinstance(crop_padding_ratio, (int, float)) else 0.08
    context, isolated, source_path = _candidate_images(
        run_root,
        case,
        case,
        crop_padding_ratio=padding,
        evidence_cache=evidence_cache,
    )
    case["source_image_path"] = source_path
    board_height = _HEADER_HEIGHT + _PANEL_HEIGHT + _METRICS_HEIGHT
    board = Image.new("RGB", (_BOARD_WIDTH, board_height), _PAGE_BACKGROUND)
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    rules = ", ".join(str(value) for value in case["shadow_rules"]) or "none"
    title_lines = (
        f"{case['clip_uid']} / {case['entity_id']} / {case['candidate_id']}",
        f"{case['reference_type']} / {case.get('phrase')}",
        f"shadow rules: {rules}",
        f"Qwen selected? {'yes' if case['current_qwen_selected'] else 'no'}",
    )
    for index, line in enumerate(title_lines):
        _draw_text(
            draw,
            (_MARGIN, 20 + index * 28),
            line,
            fill=_TEXT if index < 2 else _MUTED,
            font=font,
        )
    left_box = (
        _MARGIN,
        _HEADER_HEIGHT,
        _MARGIN + _PANEL_WIDTH,
        _HEADER_HEIGHT + _PANEL_HEIGHT,
    )
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
    _draw_text(
        draw,
        (left_box[0] + 8, left_box[1] + 8),
        "context image",
        fill=_MUTED,
        font=font,
    )
    _draw_text(
        draw,
        (right_box[0] + 8, right_box[1] + 8),
        "isolated crop",
        fill=_MUTED,
        font=font,
    )
    technical = case.get("technical_metrics")
    pose = case.get("subject_pose_evidence")
    lines = [
        (
            f"luma={_format_metric(_mapping_metric(technical, 'luma_mean'))}   "
            "dark_fraction_32="
            f"{_format_metric(_mapping_metric(technical, 'dark_fraction_32'))}"
        ),
        (
            "laplacian="
            f"{_format_metric(_mapping_metric(technical, 'laplacian_variance'))}   "
            "tenengrad="
            f"{_format_metric(_mapping_metric(technical, 'tenengrad_mean'))}"
        ),
        (
            f"lap_ratio={_format_metric(case.get('laplacian_ratio'))}   "
            f"tenengrad_ratio={_format_metric(case.get('tenengrad_ratio'))}"
        ),
    ]
    if case["reference_type"] == "subject":
        lines.append(
            f"face_detected={_mapping_metric(pose, 'face_detected')}   "
            "face area="
            f"{_format_metric(_mapping_metric(pose, 'face_bbox_area_ratio'))}   "
            f"yaw={_format_metric(_mapping_metric(pose, 'yaw'))}"
        )
    text_top = _HEADER_HEIGHT + _PANEL_HEIGHT + 20
    for index, line in enumerate(lines):
        _draw_text(
            draw,
            (_MARGIN, text_top + index * 30),
            line,
            fill=_TEXT if index < 3 else _MUTED,
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
        rules = html.escape(", ".join(str(value) for value in case["shadow_rules"]))
        cards.append(
            f'<article><a href="{board_path}"><img src="{board_path}" '
            f'alt="{title}"></a><h2>{title}</h2><p>{rules}</p></article>'
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V3 Prefilter Shadow Review</title>
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
<header><h1>V3 Prefilter Shadow Review</h1><p>Mode: {html.escape(str(summary['mode']))} / Cases: {summary['case_count']}</p></header>
<main>{''.join(cards)}</main>
</body>
</html>
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_prefilter_shadow_review_board(
    *,
    run_root: Path,
    audit_root: Path,
    simulation: Path,
    output_root: Path,
    mode: ReviewMode = "all",
) -> dict[str, object]:
    if mode not in {
        "near_silhouette",
        "relative_blur",
        "qwen_selected_flagged",
        "all_candidates_flagged",
        "all",
    }:
        raise ValueError(f"unsupported shadow review mode: {mode}")
    source_run = _validated_input_root(run_root, label="run_root")
    source_audit = _validated_input_root(audit_root, label="audit_root")
    source_simulation = _validated_simulation(simulation)
    destination = _validated_output_root(
        output_root,
        run_root=source_run,
        audit_root=source_audit,
        simulation=source_simulation,
    )
    before_roots = {
        root: snapshot_run_files(root) for root in (source_run, source_audit)
    }
    before_simulation = _sha256(source_simulation)
    simulation_value = _load_simulation(source_simulation, source_audit)
    cases = _selected_cases(simulation_value, mode=mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"review temporary output exists: {temporary}")
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
                run_root=source_run,
                destination=temporary / board_path,
                evidence_cache=evidence_cache,
            )
        summary = {
            "schema_version": 1,
            "audit_only": True,
            "qwen_calls_added": 0,
            "mode": mode,
            "case_count": len(cases),
            "run_root": str(source_run),
            "audit_root": str(source_audit),
            "simulation": str(source_simulation),
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
        roots_changed = any(
            snapshot_run_files(root) != before_roots[root]
            for root in before_roots
        )
        simulation_changed = _sha256(source_simulation) != before_simulation
        if roots_changed or simulation_changed:
            raise RuntimeError("an input changed during shadow review generation")
        temporary.replace(destination)
        return summary
    except Exception as exc:
        roots_changed = any(
            snapshot_run_files(root) != before_roots[root]
            for root in before_roots
        )
        simulation_changed = _sha256(source_simulation) != before_simulation
        if temporary.exists():
            shutil.rmtree(temporary)
        if roots_changed or simulation_changed:
            raise RuntimeError(
                "an input changed during failed shadow review generation"
            ) from exc
        raise


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = build_prefilter_shadow_review_board(
        run_root=arguments.run_root,
        audit_root=arguments.audit_root,
        simulation=arguments.simulation,
        output_root=arguments.output_root,
        mode=arguments.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
