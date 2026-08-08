from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

FOCUS_CASES = frozenset(
    {
        ("15409fe27a23cb0a16bdd459", "e1"),
        ("251b44a75511156ff06222d0", "e1"),
        ("425f401670a4307b149f2420", "e1"),
        ("58c7d4523b65add330d71943", "e1"),
        ("82c20ca1c4f8e0c855a783de", "e1"),
        ("f374b11496cd99f988879a3e", "e1"),
        ("f374b11496cd99f988879a3e", "e2"),
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two read-only V3 visual embedding audits",
    )
    parser.add_argument("--dinov2-audit", type=Path, required=True)
    parser.add_argument("--siglip2-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _audit_paths(path: Path) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_dir():
        return resolved / "audit.jsonl", resolved / "audit.summary.json"
    if resolved.name != "audit.jsonl":
        raise ValueError("audit input must be an audit directory or audit.jsonl")
    return resolved, resolved.with_name("audit.summary.json")


def _load_audit(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    records_path, summary_path = _audit_paths(path)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise TypeError("audit JSONL records must be objects")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError("audit summary must be an object")
    return records, summary


def _candidate_groups(
    records: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        if record.get("artifact_scope") != "candidate":
            continue
        key = (str(record["clip_uid"]), str(record["entity_id"]))
        groups.setdefault(key, []).append(record)
    return groups


def _representativeness(record: Mapping[str, object]) -> float:
    embedding = record.get("embedding")
    if not isinstance(embedding, Mapping) or embedding.get("status") != "succeeded":
        raise ValueError("comparison requires successful candidate embeddings")
    value = embedding.get("representativeness_score")
    if not isinstance(value, (int, float)):
        raise TypeError("candidate representativeness score is missing")
    return float(value)


def _group_result(group: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(group, key=lambda record: str(record.get("candidate_id")))
    selected = [record for record in ordered if record.get("is_current_selected")]
    if len(selected) != 1:
        raise ValueError("audit entity must have exactly one production selection")
    scored = [
        {
            "candidate_id": record["candidate_id"],
            "representativeness_score": _representativeness(record),
        }
        for record in ordered
    ]
    best = min(
        scored,
        key=lambda item: (
            -float(item["representativeness_score"]),
            str(item["candidate_id"]),
        ),
    )
    return {
        "best_candidate_id": best["candidate_id"],
        "production_selected_candidate_id": selected[0]["candidate_id"],
        "reference_type": selected[0]["reference_type"],
        "phrase": selected[0]["phrase"],
        "candidate_scores": scored,
    }


def _agreement(cases: list[dict[str, object]], field: str) -> dict[str, object]:
    denominator = len(cases)
    count = sum(bool(case[field]) for case in cases)
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _backend_summary(
    summary: Mapping[str, object],
    qwen_agreement: dict[str, object],
) -> dict[str, object]:
    runtime = summary.get("runtime")
    visual_runtime = (
        runtime.get("visual_encoder") if isinstance(runtime, Mapping) else None
    )
    rank = summary.get("selected_representativeness_rank")
    by_reference_type = (
        rank.get("by_reference_type") if isinstance(rank, Mapping) else None
    )
    return {
        "runtime": dict(visual_runtime) if isinstance(visual_runtime, Mapping) else None,
        "embedding_dimensions": summary.get("embedding_dimensions"),
        "selected_rank_1_rate": (
            rank.get("selected_rank_1_rate") if isinstance(rank, Mapping) else None
        ),
        "selected_rank_1_rate_by_reference_type": {
            reference_type: (
                values.get("selected_rank_1_rate")
                if isinstance(values, Mapping)
                else None
            )
            for reference_type in ("subject", "object", "group")
            for values in (
                (
                    by_reference_type.get(reference_type)
                    if isinstance(by_reference_type, Mapping)
                    else None
                ),
            )
        },
        "agreement_with_qwen": qwen_agreement,
    }


def compare_embedding_audits(
    dinov2_audit: Path,
    siglip2_audit: Path,
    *,
    output_path: Path,
) -> dict[str, object]:
    dino_records, dino_summary = _load_audit(dinov2_audit)
    siglip_records, siglip_summary = _load_audit(siglip2_audit)
    dino_groups = _candidate_groups(dino_records)
    siglip_groups = _candidate_groups(siglip_records)
    if set(dino_groups) != set(siglip_groups):
        raise ValueError("embedding audits do not contain the same entity cases")
    cases: list[dict[str, object]] = []
    for key in sorted(dino_groups):
        dino = _group_result(dino_groups[key])
        siglip = _group_result(siglip_groups[key])
        if {
            value["candidate_id"] for value in dino["candidate_scores"]
        } != {value["candidate_id"] for value in siglip["candidate_scores"]}:
            raise ValueError("embedding audits do not contain the same candidates")
        if (
            dino["production_selected_candidate_id"]
            != siglip["production_selected_candidate_id"]
            or dino["reference_type"] != siglip["reference_type"]
        ):
            raise ValueError("embedding audit production baselines do not match")
        production_selected = dino["production_selected_candidate_id"]
        case = {
            "clip_uid": key[0],
            "entity_id": key[1],
            "reference_type": dino["reference_type"],
            "phrase": dino["phrase"],
            "dinov2_best_candidate_id": dino["best_candidate_id"],
            "siglip2_best_candidate_id": siglip["best_candidate_id"],
            "production_qwen_selected_candidate_id": production_selected,
            "dinov2_candidate_scores": dino["candidate_scores"],
            "siglip2_candidate_scores": siglip["candidate_scores"],
            "dinov2_qwen_agree": dino["best_candidate_id"] == production_selected,
            "siglip2_qwen_agree": siglip["best_candidate_id"] == production_selected,
            "dinov2_siglip2_agree": (
                dino["best_candidate_id"] == siglip["best_candidate_id"]
            ),
        }
        cases.append(case)
    dino_qwen = _agreement(cases, "dinov2_qwen_agree")
    siglip_qwen = _agreement(cases, "siglip2_qwen_agree")
    backend_agreement = _agreement(cases, "dinov2_siglip2_agree")
    object_cases = [case for case in cases if case["reference_type"] == "object"]
    comparison = {
        "schema_version": 1,
        "audit_only": True,
        "qwen_calls_added": 0,
        "entity_count": len(cases),
        "dinov2": _backend_summary(dino_summary, dino_qwen),
        "siglip2": _backend_summary(siglip_summary, siglip_qwen),
        "backend_agreement": backend_agreement,
        "object_only": {
            "entity_count": len(object_cases),
            "dinov2_selected_rank_1_rate": (
                dino_summary.get("selected_representativeness_rank", {})
                .get("by_reference_type", {})
                .get("object", {})
                .get("selected_rank_1_rate")
            ),
            "siglip2_selected_rank_1_rate": (
                siglip_summary.get("selected_representativeness_rank", {})
                .get("by_reference_type", {})
                .get("object", {})
                .get("selected_rank_1_rate")
            ),
            "dinov2_qwen_agreement": _agreement(
                object_cases,
                "dinov2_qwen_agree",
            ),
            "siglip2_qwen_agreement": _agreement(
                object_cases,
                "siglip2_qwen_agree",
            ),
            "dinov2_siglip2_agreement": _agreement(
                object_cases,
                "dinov2_siglip2_agree",
            ),
        },
        "disagreement_cases": [
            case
            for case in cases
            if not (
                case["dinov2_qwen_agree"]
                and case["siglip2_qwen_agree"]
                and case["dinov2_siglip2_agree"]
            )
        ],
        "focus_cases": [
            case
            for case in cases
            if (str(case["clip_uid"]), str(case["entity_id"])) in FOCUS_CASES
        ],
        "cases": cases,
    }
    output = output_path.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return comparison


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    result = compare_embedding_audits(
        arguments.dinov2_audit,
        arguments.siglip2_audit,
        output_path=arguments.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
