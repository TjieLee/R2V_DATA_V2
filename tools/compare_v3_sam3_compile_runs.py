#!/usr/bin/env python3
"""Compare eager and compiled V3 SAM3 runs without equivalence tolerances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.sam3_backend import mask_iou


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"comparison artifact must be an object: {path}")
    return value


def _clip_artifacts(root: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    artifacts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for clip_path in sorted((root / "clips").glob("*/clip.json")):
        masks_path = clip_path.with_name("masks.rle.json")
        if not masks_path.is_file():
            raise FileNotFoundError(f"missing masks artifact: {masks_path}")
        clip = _read_object(clip_path)
        masks = _read_object(masks_path)
        clip_uid = clip.get("clip_uid")
        if not isinstance(clip_uid, str) or not clip_uid:
            raise ValueError(f"clip artifact has no clip_uid: {clip_path}")
        if masks.get("clip_uid") != clip_uid:
            raise ValueError(f"clip/mask identity mismatch: {clip_path.parent}")
        artifacts[clip_uid] = (clip, masks)
    return artifacts


def _retained_ids(clip: dict[str, Any]) -> list[str]:
    pairing = clip.get("pairing")
    if not isinstance(pairing, dict):
        return []
    value = pairing.get("retained_entity_ids", [])
    return sorted(str(item) for item in value) if isinstance(value, list) else []


def _reference_provenance(clip: dict[str, Any]) -> list[dict[str, Any]]:
    references = clip.get("references")
    entities = references.get("entities", []) if isinstance(references, dict) else []
    if not isinstance(entities, list):
        return []
    fields = (
        "entity_id",
        "status",
        "reason",
        "source_clip_uid",
        "source_entity_id",
        "source_frame_index",
        "source_frame_slot",
        "source_image_path",
    )
    return [
        {name: item.get(name) for name in fields}
        for item in entities
        if isinstance(item, dict)
    ]


def _entity_mask_comparison(
    eager: dict[str, Any] | None,
    compiled: dict[str, Any] | None,
) -> dict[str, Any]:
    if eager is None or compiled is None:
        return {
            "exact": False,
            "masks_exact": False,
            "status_reason_exact": False,
            "provenance_exact": False,
            "minimum_mask_iou": None,
        }
    eager_frames = eager.get("frames", [])
    compiled_frames = compiled.get("frames", [])
    ious: list[float] = []
    mask_differences: list[dict[str, Any]] = []
    if isinstance(eager_frames, list) and isinstance(compiled_frames, list):
        compiled_by_slot = {
            item.get("slot"): item for item in compiled_frames if isinstance(item, dict)
        }
        for frame in eager_frames:
            if not isinstance(frame, dict):
                continue
            peer = compiled_by_slot.get(frame.get("slot"))
            if not isinstance(peer, dict):
                continue
            first = decode_binary_mask(frame["rle"])
            second = decode_binary_mask(peer["rle"])
            if first.shape != second.shape:
                iou = 0.0
            else:
                iou = mask_iou(first, second)
            ious.append(iou)
            if frame.get("rle") != peer.get("rle"):
                mask_differences.append(
                    {
                        "slot": frame.get("slot"),
                        "iou": iou,
                        "eager_area_pixels": frame.get("area_pixels"),
                        "compile_area_pixels": peer.get("area_pixels"),
                    }
                )
    provenance_fields = ("reference_type", "grounding_prompt", "backend_object_ids")
    masks_exact = [
        (item.get("slot"), item.get("rle"))
        for item in eager_frames
        if isinstance(item, dict)
    ] == [
        (item.get("slot"), item.get("rle"))
        for item in compiled_frames
        if isinstance(item, dict)
    ]
    eager_present = [
        item.get("slot")
        for item in eager_frames
        if isinstance(item, dict) and item.get("present")
    ]
    compiled_present = [
        item.get("slot")
        for item in compiled_frames
        if isinstance(item, dict) and item.get("present")
    ]
    eager_valid = [
        item.get("slot")
        for item in eager_frames
        if isinstance(item, dict) and item.get("track_valid")
    ]
    compiled_valid = [
        item.get("slot")
        for item in compiled_frames
        if isinstance(item, dict) and item.get("track_valid")
    ]
    eager_areas = [
        (item.get("slot"), item.get("area_pixels"))
        for item in eager_frames
        if isinstance(item, dict)
    ]
    compiled_areas = [
        (item.get("slot"), item.get("area_pixels"))
        for item in compiled_frames
        if isinstance(item, dict)
    ]
    return {
        "exact": eager == compiled,
        "masks_exact": masks_exact,
        "status_reason_exact": (
            eager.get("status"), eager.get("reason")
        ) == (compiled.get("status"), compiled.get("reason")),
        "provenance_exact": all(
            eager.get(name) == compiled.get(name) for name in provenance_fields
        ),
        "present_frame_slots_exact": eager_present == compiled_present,
        "valid_frame_slots_exact": eager_valid == compiled_valid,
        "mask_areas_exact": eager_areas == compiled_areas,
        "differing_masks": mask_differences,
        "minimum_mask_iou": min(ious) if ious else None,
    }


def compare_runs(eager_root: str | Path, compiled_root: str | Path) -> dict[str, Any]:
    eager_path = Path(eager_root).expanduser().resolve(strict=True)
    compiled_path = Path(compiled_root).expanduser().resolve(strict=True)
    if not eager_path.is_dir() or not compiled_path.is_dir():
        raise NotADirectoryError("both comparison roots must be directories")
    if eager_path == compiled_path:
        raise ValueError("eager and compiled run roots must be distinct")
    eager_clips = _clip_artifacts(eager_path)
    compiled_clips = _clip_artifacts(compiled_path)
    eager_ids = set(eager_clips)
    compiled_ids = set(compiled_clips)
    details: list[dict[str, Any]] = []
    entity_total = entity_exact = mask_exact = 0
    nonexact_ious: list[float] = []
    differing_reference_selections = 0
    differing_retained_entity_sets = 0
    for clip_uid in sorted(eager_ids & compiled_ids):
        eager_clip, eager_masks = eager_clips[clip_uid]
        compiled_clip, compiled_masks = compiled_clips[clip_uid]
        eager_references = _reference_provenance(eager_clip)
        compiled_references = _reference_provenance(compiled_clip)
        eager_retained = _retained_ids(eager_clip)
        compiled_retained = _retained_ids(compiled_clip)
        eager_entities = eager_masks.get("entities", {})
        compiled_entities = compiled_masks.get("entities", {})
        if not isinstance(eager_entities, dict) or not isinstance(
            compiled_entities, dict
        ):
            raise TypeError("masks entities must be objects")
        entity_details: dict[str, Any] = {}
        for entity_id in sorted(set(eager_entities) | set(compiled_entities)):
            first = eager_entities.get(entity_id)
            second = compiled_entities.get(entity_id)
            comparison = _entity_mask_comparison(
                first if isinstance(first, dict) else None,
                second if isinstance(second, dict) else None,
            )
            entity_details[entity_id] = comparison
            entity_total += 1
            entity_exact += int(comparison["exact"])
            mask_exact += int(comparison["masks_exact"])
            if (
                not comparison["masks_exact"]
                and comparison["minimum_mask_iou"] is not None
            ):
                nonexact_ious.append(float(comparison["minimum_mask_iou"]))
        coverage_exact = eager_clip.get("coverage") == compiled_clip.get("coverage")
        references_exact = eager_references == compiled_references
        retained_exact = eager_retained == compiled_retained
        pairing_exact = eager_clip.get("pairing") == compiled_clip.get("pairing")
        differing_reference_selections += int(not references_exact)
        differing_retained_entity_sets += int(not retained_exact)
        exact = (
            eager_masks == compiled_masks
            and coverage_exact
            and references_exact
            and retained_exact
            and pairing_exact
        )
        details.append(
            {
                "clip_uid": clip_uid,
                "exact": exact,
                "coverage_exact": coverage_exact,
                "references_exact": references_exact,
                "eager_reference_selections": eager_references,
                "compile_reference_selections": compiled_references,
                "retained_entity_set_exact": retained_exact,
                "eager_retained_entity_ids": eager_retained,
                "compile_retained_entity_ids": compiled_retained,
                "pairing_exact": pairing_exact,
                "entities": entity_details,
            }
        )
    return {
        "schema_version": 1,
        "eager_root": str(eager_path),
        "compiled_root": str(compiled_path),
        "compared_clips": len(details),
        "eager_only_clips": sorted(eager_ids - compiled_ids),
        "compiled_only_clips": sorted(compiled_ids - eager_ids),
        "exactly_equal_clips": sum(bool(item["exact"]) for item in details),
        "differing_clips": sum(not bool(item["exact"]) for item in details),
        "compared_entity_tracks": entity_total,
        "exactly_equal_entity_tracks": entity_exact,
        "differing_entity_tracks": entity_total - entity_exact,
        "exact_masks": mask_exact,
        "minimum_nonexact_mask_iou": (
            min(nonexact_ious) if nonexact_ious else None
        ),
        "differing_reference_selections": differing_reference_selections,
        "differing_retained_entity_sets": differing_retained_entity_sets,
        "all_exact": (
            bool(details)
            and eager_ids == compiled_ids
            and all(bool(item["exact"]) for item in details)
        ),
        "clips": details,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-run-root", type=Path, required=True)
    parser.add_argument("--compile-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = compare_runs(args.eager_run_root, args.compile_run_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
