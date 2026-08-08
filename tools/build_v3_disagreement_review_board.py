from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.config import PairConfig
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.pair import (
    build_candidate_context_image,
    build_reference_crop,
)
from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files
from r2v_data_v2.v3.schemas import SampledFramesArtifact, TrackedMasksArtifact
from tools.compare_v3_reference_embedding_audits import (
    FOCUS_CASES,
    _candidate_groups,
    _group_result,
    _load_audit,
)

ReviewMode = Literal[
    "all",
    "dino_eq_siglip_ne_qwen",
    "qwen_rank3_by_dino",
    "object_only",
    "focus_cases",
]

_BOARD_WIDTH = 1340
_HEADER_HEIGHT = 170
_PANEL_WIDTH = 620
_PANEL_HEIGHT = 300
_ROW_TEXT_HEIGHT = 120
_ROW_GAP = 20
_MARGIN = 30
_TITLE_COLOR = (28, 31, 36)
_TEXT_COLOR = (42, 46, 52)
_MUTED_COLOR = (95, 101, 110)
_PANEL_BACKGROUND = (237, 239, 242)
_PAGE_BACKGROUND = (250, 250, 248)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only V3 embedding disagreement review board",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dinov2-audit-root", type=Path, required=True)
    parser.add_argument("--siglip2-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "dino_eq_siglip_ne_qwen",
            "qwen_rank3_by_dino",
            "object_only",
            "focus_cases",
        ),
        default="all",
    )
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=PairConfig().crop_padding_ratio,
    )
    return parser


def _validated_input_root(input_root: Path, *, label: str) -> Path:
    root = input_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} must be a directory")
    return root


def _validated_output_root(output_root: Path, input_roots: tuple[Path, ...]) -> Path:
    output = output_root.expanduser().resolve(strict=False)
    if any(
        output == source or source in output.parents or output in source.parents
        for source in input_roots
    ):
        raise ValueError("review output must be separate from all input roots")
    if output.exists():
        raise FileExistsError(f"review output already exists: {output}")
    return output


def _resolve_source_path(run_root: Path, relative: Path) -> Path:
    path = (run_root / relative).resolve(strict=True)
    if run_root not in path.parents or not path.is_file():
        raise ValueError("review source artifact is outside run_root or not a file")
    return path


def _score_map(group: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in group:
        embedding = record.get("embedding")
        if not isinstance(embedding, Mapping) or embedding.get("status") != "succeeded":
            raise ValueError("review board requires successful candidate embeddings")
        score = embedding.get("representativeness_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("candidate representativeness must be finite")
        values[str(record["candidate_id"])] = float(score)
    return values


def _ranked_candidate_ids(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))


def _case_from_groups(
    key: tuple[str, str],
    dino_group: list[dict[str, object]],
    siglip_group: list[dict[str, object]],
) -> dict[str, object]:
    dino = _group_result(dino_group)
    siglip = _group_result(siglip_group)
    dino_records = {
        str(record["candidate_id"]): record for record in dino_group
    }
    siglip_records = {
        str(record["candidate_id"]): record for record in siglip_group
    }
    if set(dino_records) != set(siglip_records):
        raise ValueError("DINOv2 and SigLIP2 audits have different candidates")
    if (
        dino["production_selected_candidate_id"]
        != siglip["production_selected_candidate_id"]
        or dino["reference_type"] != siglip["reference_type"]
        or dino["phrase"] != siglip["phrase"]
    ):
        raise ValueError("DINOv2 and SigLIP2 audit baselines do not match")
    for candidate_id in sorted(dino_records):
        dino_record = dino_records[candidate_id]
        siglip_record = siglip_records[candidate_id]
        identity = ("frame_slot", "source_frame_index")
        if any(dino_record.get(field) != siglip_record.get(field) for field in identity):
            raise ValueError("audit candidate source identities do not match")
    dino_scores = _score_map(dino_group)
    siglip_scores = _score_map(siglip_group)
    dino_ranking = _ranked_candidate_ids(dino_scores)
    siglip_ranking = _ranked_candidate_ids(siglip_scores)
    qwen_selected = str(dino["production_selected_candidate_id"])
    dino_best = dino_ranking[0]
    siglip_best = siglip_ranking[0]
    dino_eq_siglip_ne_qwen = dino_best == siglip_best != qwen_selected
    is_disagreement = not (dino_best == siglip_best == qwen_selected)
    baseline = dino_records[qwen_selected].get("production_baseline")
    baseline_value = dict(baseline) if isinstance(baseline, Mapping) else None
    candidates = []
    for candidate_id in sorted(dino_records):
        record = dino_records[candidate_id]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "frame_slot": record["frame_slot"],
                "source_frame_index": record["source_frame_index"],
                "dinov2_representativeness_score": dino_scores[candidate_id],
                "siglip2_representativeness_score": siglip_scores[candidate_id],
                "qwen_selected": candidate_id == qwen_selected,
                "dinov2_top_1": candidate_id == dino_best,
                "siglip2_top_1": candidate_id == siglip_best,
                "dinov2_audit_record": record,
                "siglip2_audit_record": siglip_records[candidate_id],
            }
        )
    return {
        "clip_uid": key[0],
        "entity_id": key[1],
        "phrase": dino["phrase"],
        "reference_type": dino["reference_type"],
        "qwen_selected_candidate_id": qwen_selected,
        "dinov2_best_candidate_id": dino_best,
        "siglip2_best_candidate_id": siglip_best,
        "qwen_rank_by_dinov2": dino_ranking.index(qwen_selected) + 1,
        "is_disagreement": is_disagreement,
        "dino_eq_siglip_ne_qwen": dino_eq_siglip_ne_qwen,
        "is_object": dino["reference_type"] == "object",
        "qwen_rank3_by_dino": dino_ranking.index(qwen_selected) + 1 == 3,
        "is_focus_case": key in FOCUS_CASES,
        "production_baseline": baseline_value,
        "candidates": candidates,
    }


def _include_case(case: Mapping[str, object], mode: ReviewMode) -> bool:
    if mode == "all":
        return bool(case["is_disagreement"])
    if mode == "dino_eq_siglip_ne_qwen":
        return bool(case["dino_eq_siglip_ne_qwen"])
    if mode == "qwen_rank3_by_dino":
        return bool(case["qwen_rank3_by_dino"])
    if mode == "object_only":
        return bool(case["is_disagreement"] and case["is_object"])
    if mode == "focus_cases":
        return bool(case["is_focus_case"])
    raise ValueError(f"unsupported review mode: {mode}")


def _case_sort_key(case: Mapping[str, object]) -> tuple[int, str, str]:
    if case["dino_eq_siglip_ne_qwen"]:
        priority = 0
    elif case["is_object"]:
        priority = 1
    elif case["qwen_rank3_by_dino"]:
        priority = 2
    else:
        priority = 3
    return priority, str(case["clip_uid"]), str(case["entity_id"])


def _safe_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not component:
        raise ValueError("review case identifier is not filesystem-safe")
    return component


def _load_clip_evidence(
    run_root: Path,
    clip_uid: str,
) -> tuple[SampledFramesArtifact, TrackedMasksArtifact]:
    clip_dir = run_root / "clips" / _safe_component(clip_uid)
    frames_path = _resolve_source_path(run_root, clip_dir.relative_to(run_root) / "frames" / "frames.json")
    masks_path = _resolve_source_path(run_root, clip_dir.relative_to(run_root) / "masks.rle.json")
    frames = SampledFramesArtifact.model_validate_json(
        frames_path.read_text(encoding="utf-8")
    )
    masks = TrackedMasksArtifact.model_validate_json(
        masks_path.read_text(encoding="utf-8")
    )
    if frames.clip_uid != clip_uid or masks.clip_uid != clip_uid:
        raise ValueError("review evidence clip_uid does not match its path")
    return frames, masks


def _candidate_images(
    run_root: Path,
    case: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    crop_padding_ratio: float,
    evidence_cache: dict[str, tuple[SampledFramesArtifact, TrackedMasksArtifact]],
) -> tuple[Image.Image, Image.Image, str]:
    clip_uid = str(case["clip_uid"])
    entity_id = str(case["entity_id"])
    if clip_uid not in evidence_cache:
        evidence_cache[clip_uid] = _load_clip_evidence(run_root, clip_uid)
    frames, masks = evidence_cache[clip_uid]
    slot = int(candidate["frame_slot"])
    frame = frames.frames[slot]
    if frame.source_frame_index != int(candidate["source_frame_index"]):
        raise ValueError("audit candidate source frame index changed")
    tracked = masks.entities.get(entity_id)
    if tracked is None or tracked.status != "ready":
        raise ValueError("review entity tracked masks are not ready")
    tracked_frame = tracked.frames[slot]
    if not tracked_frame.present:
        raise ValueError("review candidate mask is absent")
    mask = decode_binary_mask(tracked_frame.rle)
    relative_frame_path = Path("clips") / clip_uid / frame.image_path
    source_path = _resolve_source_path(run_root, relative_frame_path)
    with Image.open(source_path) as opened:
        opened.load()
        source = opened.convert("RGB")
    context = build_candidate_context_image(source, mask)
    crop, _ = build_reference_crop(
        source,
        mask,
        crop_padding_ratio=crop_padding_ratio,
    )
    isolated = Image.new("RGB", crop.size, (255, 255, 255))
    isolated.paste(crop, mask=crop.getchannel("A"))
    return context, isolated, source_path.relative_to(run_root).as_posix()


def _fit_panel(image: Image.Image) -> Image.Image:
    return ImageOps.contain(
        image.convert("RGB"),
        (_PANEL_WIDTH, _PANEL_HEIGHT),
        method=Image.Resampling.LANCZOS,
    )


def _draw_safe_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: object,
    *,
    fill: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    text = str(value)
    try:
        draw.text(position, text, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(
            position,
            text.encode("ascii", "replace").decode("ascii"),
            fill=fill,
            font=font,
        )


def _paste_centered(
    board: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    panel = _fit_panel(image)
    x = left + (right - left - panel.width) // 2
    y = top + (bottom - top - panel.height) // 2
    board.paste(panel, (x, y))


def _render_case_board(
    case: dict[str, object],
    *,
    run_root: Path,
    destination: Path,
    crop_padding_ratio: float,
    evidence_cache: dict[str, tuple[SampledFramesArtifact, TrackedMasksArtifact]],
) -> None:
    candidates = case["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("review case has no candidates")
    row_height = _PANEL_HEIGHT + _ROW_TEXT_HEIGHT + _ROW_GAP
    board_height = _HEADER_HEIGHT + row_height * len(candidates) + _MARGIN
    board = Image.new("RGB", (_BOARD_WIDTH, board_height), _PAGE_BACKGROUND)
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    title_lines = [
        f"{case['clip_uid']} / {case['entity_id']} / {case['reference_type']}",
        *textwrap.wrap(str(case["phrase"]), width=110),
        (
            f"Qwen selected: {case['qwen_selected_candidate_id']}   "
            f"DINO top-1: {case['dinov2_best_candidate_id']}   "
            f"SigLIP top-1: {case['siglip2_best_candidate_id']}"
        ),
    ]
    for index, line in enumerate(title_lines[:5]):
        _draw_safe_text(
            draw,
            (_MARGIN, 20 + index * 25),
            line,
            fill=_TITLE_COLOR if index == 0 else _TEXT_COLOR,
            font=font,
        )
    baseline = case.get("production_baseline")
    baseline_value = baseline if isinstance(baseline, Mapping) else {}
    for row_index, candidate_value in enumerate(candidates):
        if not isinstance(candidate_value, dict):
            raise TypeError("review candidate metadata must be an object")
        candidate = candidate_value
        context, isolated, source_path = _candidate_images(
            run_root,
            case,
            candidate,
            crop_padding_ratio=crop_padding_ratio,
            evidence_cache=evidence_cache,
        )
        candidate["source_image_path"] = source_path
        row_top = _HEADER_HEIGHT + row_index * row_height
        left_box = (
            _MARGIN,
            row_top,
            _MARGIN + _PANEL_WIDTH,
            row_top + _PANEL_HEIGHT,
        )
        right_box = (
            _BOARD_WIDTH - _MARGIN - _PANEL_WIDTH,
            row_top,
            _BOARD_WIDTH - _MARGIN,
            row_top + _PANEL_HEIGHT,
        )
        draw.rectangle(left_box, fill=_PANEL_BACKGROUND, outline=(190, 194, 201))
        draw.rectangle(right_box, fill=_PANEL_BACKGROUND, outline=(190, 194, 201))
        _paste_centered(board, context, left_box)
        _paste_centered(board, isolated, right_box)
        _draw_safe_text(
            draw,
            (left_box[0] + 8, left_box[1] + 8),
            "context image",
            fill=_MUTED_COLOR,
            font=font,
        )
        _draw_safe_text(
            draw,
            (right_box[0] + 8, right_box[1] + 8),
            "isolated crop",
            fill=_MUTED_COLOR,
            font=font,
        )
        labels = [
            str(candidate["candidate_id"]),
            (
                "DINO representativeness: "
                f"{float(candidate['dinov2_representativeness_score']):.6f}   "
                "SigLIP representativeness: "
                f"{float(candidate['siglip2_representativeness_score']):.6f}"
            ),
            (
                f"Qwen selected? {'yes' if candidate['qwen_selected'] else 'no'}   "
                f"DINO top-1? {'yes' if candidate['dinov2_top_1'] else 'no'}   "
                f"SigLIP top-1? {'yes' if candidate['siglip2_top_1'] else 'no'}"
            ),
            (
                f"completeness={baseline_value.get('completeness')}   "
                f"reference_scope={baseline_value.get('reference_scope')}   "
                f"viewpoint={baseline_value.get('viewpoint')}   "
                f"truncation_severity={baseline_value.get('truncation_severity')}"
            ),
        ]
        text_top = row_top + _PANEL_HEIGHT + 8
        for line_index, label in enumerate(labels):
            _draw_safe_text(
                draw,
                (_MARGIN, text_top + line_index * 24),
                label,
                fill=_TEXT_COLOR if line_index < 3 else _MUTED_COLOR,
                font=font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination, format="PNG")


def _summary(cases: list[dict[str, object]], *, mode: ReviewMode) -> dict[str, object]:
    return {
        "schema_version": 1,
        "audit_only": True,
        "qwen_calls_added": 0,
        "mode": mode,
        "case_count": len(cases),
        "dino_eq_siglip_ne_qwen_count": sum(
            bool(case["dino_eq_siglip_ne_qwen"]) for case in cases
        ),
        "object_count": sum(bool(case["is_object"]) for case in cases),
        "qwen_rank3_by_dino_count": sum(
            bool(case["qwen_rank3_by_dino"]) for case in cases
        ),
        "focus_case_count": sum(bool(case["is_focus_case"]) for case in cases),
    }


def _render_html(cases: list[dict[str, object]], summary: Mapping[str, object]) -> str:
    cards = []
    for case in cases:
        board_path = html.escape(str(case["board_path"]), quote=True)
        title = html.escape(
            f"{case['clip_uid']} / {case['entity_id']} / {case['reference_type']}"
        )
        phrase = html.escape(str(case["phrase"]))
        cards.append(
            f'<article><a href="{board_path}"><img src="{board_path}" '
            f'alt="{title}"></a><h2>{title}</h2><p>{phrase}</p></article>'
        )
    metrics = "".join(
        f"<div><strong>{html.escape(label)}</strong><span>{summary[key]}</span></div>"
        for key, label in (
            ("case_count", "Case count"),
            ("dino_eq_siglip_ne_qwen_count", "DINO = SigLIP != Qwen"),
            ("object_count", "Object cases"),
            ("qwen_rank3_by_dino_count", "Qwen rank 3 by DINO"),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V3 Disagreement Review Board</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #20242a; background: #f5f5f2; }}
header {{ padding: 24px 32px; background: #fff; border-bottom: 1px solid #d8dadd; }}
h1 {{ margin: 0 0 18px; font-size: 28px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }}
.summary div {{ padding: 12px; border: 1px solid #d8dadd; background: #fafafa; }}
.summary strong, .summary span {{ display: block; }}
.summary span {{ margin-top: 6px; font-size: 24px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; padding: 24px 32px; }}
article {{ min-width: 0; padding: 12px; background: #fff; border: 1px solid #d8dadd; }}
article img {{ display: block; width: 100%; aspect-ratio: 1.55; object-fit: cover; border: 1px solid #ececef; }}
article h2 {{ margin: 12px 0 6px; font-size: 16px; overflow-wrap: anywhere; }}
article p {{ margin: 0; color: #626873; }}
@media (max-width: 720px) {{ .summary {{ grid-template-columns: repeat(2, 1fr); }} main {{ padding: 16px; }} }}
</style>
</head>
<body>
<header><h1>V3 Disagreement Review Board</h1><section class="summary">{metrics}</section></header>
<main>{''.join(cards)}</main>
</body>
</html>
"""


def build_disagreement_review_board(
    *,
    run_root: Path,
    dinov2_audit_root: Path,
    siglip2_audit_root: Path,
    output_root: Path,
    mode: ReviewMode = "all",
    crop_padding_ratio: float = PairConfig().crop_padding_ratio,
) -> dict[str, object]:
    if mode not in {
        "all",
        "dino_eq_siglip_ne_qwen",
        "qwen_rank3_by_dino",
        "object_only",
        "focus_cases",
    }:
        raise ValueError(f"unsupported review mode: {mode}")
    if not math.isfinite(crop_padding_ratio) or not 0 <= crop_padding_ratio <= 0.5:
        raise ValueError("crop_padding_ratio must be between zero and 0.5")
    source = _validated_input_root(run_root, label="run_root")
    dino_root = _validated_input_root(
        dinov2_audit_root,
        label="dinov2_audit_root",
    )
    siglip_root = _validated_input_root(
        siglip2_audit_root,
        label="siglip2_audit_root",
    )
    input_roots = (source, dino_root, siglip_root)
    destination = _validated_output_root(output_root, input_roots)
    before = {root: snapshot_run_files(root) for root in input_roots}
    dino_records, _ = _load_audit(dino_root)
    siglip_records, _ = _load_audit(siglip_root)
    dino_groups = _candidate_groups(dino_records)
    siglip_groups = _candidate_groups(siglip_records)
    if set(dino_groups) != set(siglip_groups):
        raise ValueError("DINOv2 and SigLIP2 audits contain different entities")
    all_cases = [
        _case_from_groups(key, dino_groups[key], siglip_groups[key])
        for key in sorted(dino_groups)
    ]
    cases = sorted(
        [case for case in all_cases if _include_case(case, mode)],
        key=_case_sort_key,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"review temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        evidence_cache: dict[
            str,
            tuple[SampledFramesArtifact, TrackedMasksArtifact],
        ] = {}
        for case in cases:
            filename = (
                f"{_safe_component(case['clip_uid'])}__"
                f"{_safe_component(case['entity_id'])}.png"
            )
            relative_board_path = (Path("cases") / filename).as_posix()
            case["board_path"] = relative_board_path
            _render_case_board(
                case,
                run_root=source,
                destination=temporary / relative_board_path,
                crop_padding_ratio=crop_padding_ratio,
                evidence_cache=evidence_cache,
            )
        summary = _summary(cases, mode=mode)
        summary.update(
            {
                "run_root": str(source),
                "dinov2_audit_root": str(dino_root),
                "siglip2_audit_root": str(siglip_root),
                "crop_padding_ratio": crop_padding_ratio,
                "all_inputs_unchanged": True,
                "source_run_unchanged": True,
            }
        )
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
        if any(snapshot_run_files(root) != before[root] for root in input_roots):
            raise RuntimeError("an input root changed during review board generation")
        temporary.replace(destination)
        return summary
    except Exception as exc:
        input_changed = any(
            snapshot_run_files(root) != before[root] for root in input_roots
        )
        if temporary.exists():
            shutil.rmtree(temporary)
        if input_changed:
            raise RuntimeError(
                "an input root changed during failed review board generation"
            ) from exc
        raise


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = build_disagreement_review_board(
        run_root=arguments.run_root,
        dinov2_audit_root=arguments.dinov2_audit_root,
        siglip2_audit_root=arguments.siglip2_audit_root,
        output_root=arguments.output_root,
        mode=arguments.mode,
        crop_padding_ratio=arguments.crop_padding_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
