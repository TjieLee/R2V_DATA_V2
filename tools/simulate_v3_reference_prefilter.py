from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.reference_filter_audit import snapshot_run_files

RuleMode = Literal["near_silhouette", "relative_blur_v2", "all"]

ALLOWED_AUDIT_ROOT = Path("/mnt/workspace/litengjie/data/r2v_v3_audits")
NEAR_SILHOUETTE_RULE = "subject_near_silhouette_v1"
RELATIVE_BLUR_V2_RULE = "subject_relative_blur_v2"
REFERENCE_TYPES = ("subject", "object", "group")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate conservative V3 reference prefilters without filtering",
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rule",
        choices=("near_silhouette", "relative_blur_v2", "all"),
        default="all",
    )
    return parser


def _validated_audit_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir() or not (root / "audit.jsonl").is_file():
        raise ValueError("audit_root must contain audit.jsonl")
    return root


def _validated_output(path: Path, audit_root: Path) -> Path:
    output = path.expanduser().resolve(strict=False)
    allowed = ALLOWED_AUDIT_ROOT.expanduser().resolve(strict=False)
    if output.suffix.lower() != ".json":
        raise ValueError("simulation output must use a .json suffix")
    if output == allowed or allowed not in output.parents:
        raise ValueError("simulation output must be below the allowed audit root")
    if output == audit_root or audit_root in output.parents:
        raise ValueError("simulation output must be outside the source audit root")
    if output.exists():
        raise FileExistsError(f"simulation output already exists: {output}")
    return output


def _load_candidate_records(audit_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in (audit_root / "audit.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("audit JSONL records must be objects")
        if value.get("artifact_scope") == "candidate":
            records.append(value)
    if not records:
        raise ValueError("audit contains no candidate records")
    return sorted(
        records,
        key=lambda value: (
            str(value.get("clip_uid")),
            str(value.get("entity_id")),
            str(value.get("candidate_id")),
        ),
    )


def _finite_metric(section: object, field: str) -> float | None:
    if not isinstance(section, Mapping) or section.get("status") != "succeeded":
        return None
    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _subject_near_silhouette(
    reference_type: str,
    technical: object,
) -> bool:
    if reference_type != "subject":
        return False
    luma = _finite_metric(technical, "luma_mean")
    dark_fraction = _finite_metric(technical, "dark_fraction_32")
    laplacian = _finite_metric(technical, "laplacian_variance")
    tenengrad = _finite_metric(technical, "tenengrad_mean")
    return bool(
        luma is not None
        and dark_fraction is not None
        and laplacian is not None
        and tenengrad is not None
        and luma <= 15
        and dark_fraction >= 0.95
        and laplacian <= 5
        and tenengrad <= 100
    )


def _safe_ratio(value: float | None, maximum: float | None) -> float | None:
    if value is None or maximum is None or maximum <= 0:
        return None
    return value / maximum


def _subject_relative_blur_v2(
    reference_type: str,
    laplacian_ratio: float | None,
    tenengrad_ratio: float | None,
    laplacian_variance: float | None,
    tenengrad_mean: float | None,
) -> bool:
    return bool(
        reference_type == "subject"
        and laplacian_ratio is not None
        and tenengrad_ratio is not None
        and laplacian_variance is not None
        and tenengrad_mean is not None
        and laplacian_ratio <= 0.35
        and tenengrad_ratio <= 0.50
        and laplacian_variance <= 50
        and tenengrad_mean <= 1500
    )


def _enabled_rules(mode: RuleMode) -> tuple[str, ...]:
    if mode == "near_silhouette":
        return (NEAR_SILHOUETTE_RULE,)
    if mode == "relative_blur_v2":
        return (RELATIVE_BLUR_V2_RULE,)
    if mode == "all":
        return (NEAR_SILHOUETTE_RULE, RELATIVE_BLUR_V2_RULE)
    raise ValueError(f"unsupported shadow rule mode: {mode}")


def _entity_state(before: int, after: int) -> str:
    if before != 3:
        raise ValueError("shadow simulation requires three candidates per entity")
    if after == before:
        return "unchanged"
    if after == 2:
        return "3_to_2"
    if after == 1:
        return "3_to_1"
    if after == 0:
        return "all_candidates_flagged"
    raise ValueError("shadow candidate counts are inconsistent")


def _empty_summary() -> dict[str, int]:
    return {
        "candidate_count": 0,
        "near_silhouette_flag_count": 0,
        "relative_blur_v2_flag_count": 0,
        "combined_flag_count": 0,
        "entity_count": 0,
        "entity_unchanged_count": 0,
        "entity_3_to_2_count": 0,
        "entity_3_to_1_count": 0,
        "entity_all_flagged_count": 0,
        "qwen_selected_flagged_count": 0,
        "potential_input_images_before": 0,
        "potential_input_images_after": 0,
        "potential_input_images_reduced": 0,
        "potential_qwen_calls_skipped": 0,
    }


def _increment_summary(
    summary: dict[str, int],
    candidates: list[dict[str, object]],
    entity: Mapping[str, object],
) -> None:
    summary["candidate_count"] += len(candidates)
    summary["near_silhouette_flag_count"] += sum(
        bool(candidate["near_silhouette_flag"]) for candidate in candidates
    )
    summary["relative_blur_v2_flag_count"] += sum(
        bool(candidate["relative_blur_v2_flag"]) for candidate in candidates
    )
    summary["combined_flag_count"] += int(entity["flagged_count"])
    summary["entity_count"] += 1
    state_key = {
        "unchanged": "entity_unchanged_count",
        "3_to_2": "entity_3_to_2_count",
        "3_to_1": "entity_3_to_1_count",
        "all_candidates_flagged": "entity_all_flagged_count",
    }[str(entity["shadow_state"])]
    summary[state_key] += 1
    summary["qwen_selected_flagged_count"] += sum(
        bool(candidate["current_qwen_selected"])
        and bool(candidate["shadow_flagged"])
        for candidate in candidates
    )
    summary["potential_input_images_before"] += int(entity["baseline_input_images"])
    summary["potential_input_images_after"] += int(entity["shadow_input_images"])
    summary["potential_input_images_reduced"] += int(
        entity["potential_image_reduction"]
    )
    summary["potential_qwen_calls_skipped"] += int(
        entity["potential_qwen_calls_skipped"]
    )


def _build_simulation(
    records: list[dict[str, object]],
    *,
    rule: RuleMode,
    audit_root: Path,
) -> dict[str, object]:
    enabled = _enabled_rules(rule)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        clip_uid = record.get("clip_uid")
        entity_id = record.get("entity_id")
        candidate_id = record.get("candidate_id")
        reference_type = record.get("reference_type")
        if not all(
            isinstance(value, str) and value
            for value in (clip_uid, entity_id, candidate_id, reference_type)
        ):
            raise ValueError("candidate identity fields must be non-empty strings")
        if reference_type not in REFERENCE_TYPES:
            raise ValueError("candidate reference_type is unsupported")
        groups[(clip_uid, entity_id)].append(record)

    candidate_results: list[dict[str, object]] = []
    entity_results: list[dict[str, object]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda value: str(value["candidate_id"]))
        if len({str(record["candidate_id"]) for record in group}) != len(group):
            raise ValueError("candidate IDs must be unique within an entity")
        reference_types = {str(record["reference_type"]) for record in group}
        if len(reference_types) != 1:
            raise ValueError("entity candidates must share one reference_type")
        selected_count = sum(bool(record.get("is_current_selected")) for record in group)
        if selected_count > 1:
            raise ValueError("entity candidates contain multiple Qwen selections")

        laplacians = [
            _finite_metric(record.get("technical_quality"), "laplacian_variance")
            for record in group
        ]
        tenengrads = [
            _finite_metric(record.get("technical_quality"), "tenengrad_mean")
            for record in group
        ]
        valid_laplacians = [value for value in laplacians if value is not None]
        valid_tenengrads = [value for value in tenengrads if value is not None]
        max_laplacian = max(valid_laplacians) if valid_laplacians else None
        max_tenengrad = max(valid_tenengrads) if valid_tenengrads else None

        group_results: list[dict[str, object]] = []
        for record, laplacian, tenengrad in zip(
            group,
            laplacians,
            tenengrads,
            strict=True,
        ):
            reference_type = str(record["reference_type"])
            laplacian_ratio = _safe_ratio(laplacian, max_laplacian)
            tenengrad_ratio = _safe_ratio(tenengrad, max_tenengrad)
            near_condition = _subject_near_silhouette(
                reference_type,
                record.get("technical_quality"),
            )
            blur_condition = _subject_relative_blur_v2(
                reference_type,
                laplacian_ratio,
                tenengrad_ratio,
                laplacian,
                tenengrad,
            )
            near_flag = near_condition and NEAR_SILHOUETTE_RULE in enabled
            blur_flag = blur_condition and RELATIVE_BLUR_V2_RULE in enabled
            flagged_by = []
            if near_flag:
                flagged_by.append(NEAR_SILHOUETTE_RULE)
            if blur_flag:
                flagged_by.append(RELATIVE_BLUR_V2_RULE)
            technical = record.get("technical_quality")
            pose = record.get("subject_pose")
            result = {
                "clip_uid": key[0],
                "entity_id": key[1],
                "candidate_id": str(record["candidate_id"]),
                "reference_type": reference_type,
                "phrase": record.get("phrase"),
                "frame_slot": record.get("frame_slot"),
                "source_frame_index": record.get("source_frame_index"),
                "crop_padding_ratio": record.get("crop_padding_ratio"),
                "current_qwen_selected": bool(record.get("is_current_selected")),
                "near_silhouette_flag": near_flag,
                "relative_blur_v2_flag": blur_flag,
                "shadow_flagged": bool(flagged_by),
                "flagged_by": flagged_by,
                "technical_metrics": (
                    dict(technical) if isinstance(technical, Mapping) else None
                ),
                "laplacian_ratio": laplacian_ratio,
                "tenengrad_ratio": tenengrad_ratio,
                "subject_pose_evidence": (
                    dict(pose) if isinstance(pose, Mapping) else None
                ),
            }
            group_results.append(result)
            candidate_results.append(result)

        flagged_count = sum(bool(result["shadow_flagged"]) for result in group_results)
        before_count = len(group_results)
        after_count = before_count - flagged_count
        remaining_ids = [
            str(result["candidate_id"])
            for result in group_results
            if not result["shadow_flagged"]
        ]
        baseline_images = before_count * 2
        shadow_images = after_count * 2
        reduction = baseline_images - shadow_images
        entity_results.append(
            {
                "clip_uid": key[0],
                "entity_id": key[1],
                "reference_type": next(iter(reference_types)),
                "candidate_count_before": before_count,
                "flagged_count": flagged_count,
                "candidate_count_after": after_count,
                "remaining_candidate_ids": remaining_ids,
                "current_qwen_selected_flagged": any(
                    bool(result["current_qwen_selected"])
                    and bool(result["shadow_flagged"])
                    for result in group_results
                ),
                "shadow_state": _entity_state(before_count, after_count),
                "estimated_qwen_call_skippable": after_count == 0,
                "baseline_input_images": baseline_images,
                "shadow_input_images": shadow_images,
                "potential_image_reduction": reduction,
                "potential_image_reduction_rate": (
                    reduction / baseline_images if baseline_images else 0.0
                ),
                "potential_qwen_calls_skipped": int(after_count == 0),
            }
        )

    summary = _empty_summary()
    by_type = {reference_type: _empty_summary() for reference_type in REFERENCE_TYPES}
    candidates_by_entity = {
        (str(candidate["clip_uid"]), str(candidate["entity_id"])): []
        for candidate in candidate_results
    }
    for candidate in candidate_results:
        candidates_by_entity[
            (str(candidate["clip_uid"]), str(candidate["entity_id"]))
        ].append(candidate)
    for entity in entity_results:
        key = (str(entity["clip_uid"]), str(entity["entity_id"]))
        entity_candidates = candidates_by_entity[key]
        _increment_summary(summary, entity_candidates, entity)
        _increment_summary(
            by_type[str(entity["reference_type"])],
            entity_candidates,
            entity,
        )
    before_images = summary["potential_input_images_before"]
    summary["potential_input_image_reduction_rate"] = (
        summary["potential_input_images_reduced"] / before_images
        if before_images
        else 0.0
    )
    summary["flagged_candidate_rate"] = (
        summary["combined_flag_count"] / summary["candidate_count"]
        if summary["candidate_count"]
        else 0.0
    )
    for values in by_type.values():
        type_before = values["potential_input_images_before"]
        values["potential_input_image_reduction_rate"] = (
            values["potential_input_images_reduced"] / type_before
            if type_before
            else 0.0
        )
        values["flagged_candidate_rate"] = (
            values["combined_flag_count"] / values["candidate_count"]
            if values["candidate_count"]
            else 0.0
        )
    summary["by_reference_type"] = by_type

    near_cases = [
        candidate for candidate in candidate_results if candidate["near_silhouette_flag"]
    ]
    blur_cases = [
        candidate
        for candidate in candidate_results
        if candidate["relative_blur_v2_flag"]
    ]
    selected_cases = [
        candidate
        for candidate in candidate_results
        if candidate["current_qwen_selected"] and candidate["shadow_flagged"]
    ]
    all_flagged_cases = []
    for entity in entity_results:
        if entity["shadow_state"] != "all_candidates_flagged":
            continue
        key = (str(entity["clip_uid"]), str(entity["entity_id"]))
        all_flagged_cases.append(
            {
                **entity,
                "candidates": candidates_by_entity[key],
            }
        )

    return {
        "schema_version": 2,
        "audit_only": True,
        "production_filtering_applied": False,
        "qwen_calls_added": 0,
        "rule_mode": rule,
        "rules_enabled": list(enabled),
        "deprecated_rules": ["subject_relative_blur_v1"],
        "source_audit_root": str(audit_root),
        "candidates": candidate_results,
        "entities": entity_results,
        "summary": summary,
        "review_lists": {
            "near_silhouette_cases": near_cases,
            "relative_blur_v2_cases": blur_cases,
            "qwen_selected_flagged_cases": selected_cases,
            "all_candidates_flagged_cases": all_flagged_cases,
        },
    }


def simulate_reference_prefilter(
    *,
    audit_root: Path,
    output: Path,
    rule: RuleMode = "all",
) -> dict[str, object]:
    source = _validated_audit_root(audit_root)
    destination = _validated_output(output, source)
    before = snapshot_run_files(source)
    simulation = _build_simulation(
        _load_candidate_records(source),
        rule=rule,
        audit_root=source,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"simulation temporary output exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(simulation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if snapshot_run_files(source) != before:
            raise RuntimeError("source audit changed during shadow simulation")
        temporary.replace(destination)
    except Exception as exc:
        changed = snapshot_run_files(source) != before
        if temporary.exists():
            temporary.unlink()
        if changed:
            raise RuntimeError(
                "source audit changed during failed shadow simulation"
            ) from exc
        raise
    return simulation


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    simulation = simulate_reference_prefilter(
        audit_root=arguments.audit_root,
        output=arguments.output,
        rule=arguments.rule,
    )
    print(json.dumps(simulation["summary"], ensure_ascii=False, sort_keys=True))
    return simulation


if __name__ == "__main__":
    main()
