from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import parse_qwen_json_issues
from r2v_data_v2.v3.config import QwenServiceConfig, V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.pair import (
    EntityReferenceCandidate,
    build_entity_reference_candidates,
)
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
    EntityReferenceJudge,
    QwenEntityReferenceJudge,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    RawEntityReferenceDecision,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class BaselineDecision:
    selected_candidate_id: str | None
    completeness: str
    reference_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "completeness": self.completeness,
            "reference_scope": self.reference_scope,
        }


class ReadOnlyRunStorage(RunStorage):
    """RunStorage view whose candidate frame lookup never creates directories."""

    def frame_path(self, clip_uid: str, frame_slot: int) -> Path:
        self._require_clip(clip_uid)
        if not 0 <= frame_slot < self.config.frames.count:
            raise ValueError("frame_slot is outside configured frame range")
        return self.clip_dir(clip_uid) / "frames" / f"{frame_slot:02d}.jpg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_run_files(run_root: Path) -> dict[str, str]:
    root = run_root.expanduser().resolve(strict=True)
    return {
        path.relative_to(root).as_posix(): (
            "<directory>" if path.is_dir() else _sha256(path)
        )
        for path in sorted(root.rglob("*"))
        if path.is_dir() or path.is_file()
    }


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validated_output_path(output_path: Path, run_root: Path) -> Path:
    output = output_path.expanduser().resolve(strict=False)
    root = run_root.expanduser().resolve(strict=True)
    allowed = config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    if allowed not in output.parents:
        raise ValueError("benchmark output must be inside the writable data root")
    if _is_at_or_below(output, root):
        raise ValueError("benchmark output must be outside the source run_root")
    if output.exists() and output.is_dir():
        raise ValueError("benchmark output must be a file path")
    return output


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _baseline_path(
    storage: RunStorage,
    clip_uid: str,
    entity_id: str,
) -> Path:
    return (
        storage.clip_dir(clip_uid)
        / "debug"
        / "pair"
        / entity_id
        / "raw_responses.json"
    )


def load_baseline_decision(
    storage: RunStorage,
    *,
    clip_uid: str,
    entity_id: str,
) -> BaselineDecision | None:
    path = _baseline_path(storage, clip_uid, entity_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    responses = payload.get("responses") if isinstance(payload, dict) else None
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"baseline responses are missing for {clip_uid}/{entity_id}")
    final_response = responses[-1]
    if not isinstance(final_response, str):
        raise TypeError(
            f"baseline final response must be text for {clip_uid}/{entity_id}"
        )
    decision, issues = parse_qwen_json_issues(
        final_response,
        RawEntityReferenceDecision,
    )
    if decision is None or issues:
        details = "; ".join(issue.message for issue in issues)
        raise ValueError(
            f"baseline final response is invalid for {clip_uid}/{entity_id}: "
            f"{details}"
        )
    return BaselineDecision(
        selected_candidate_id=decision.selected_candidate_id,
        completeness=decision.completeness,
        reference_scope=decision.reference_scope,
    )


def _load_source_images(
    storage: RunStorage,
    candidates: list[EntityReferenceCandidate],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    root = storage.root.resolve(strict=True)
    for candidate in candidates:
        if candidate.image_path in images:
            continue
        path = (root / candidate.image_path).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("candidate source image is outside run_root")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.load()
        images[candidate.image_path] = image
    return images


def _result_record(
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    candidate_count: int,
    attempt: EntityReferenceDecisionAttempt,
    baseline: BaselineDecision | None,
    duration_seconds: float,
) -> dict[str, object]:
    decision = attempt.decision
    baseline_value = (
        baseline.to_dict()
        if baseline is not None
        else {
            "selected_candidate_id": None,
            "completeness": None,
            "reference_scope": None,
        }
    )
    return {
        "clip_uid": clip_uid,
        "entity_id": entity.entity_id,
        "phrase": entity.phrase,
        "candidate_count": candidate_count,
        "selected_candidate_id": decision.selected_candidate_id,
        "completeness": decision.completeness,
        "reference_scope": decision.reference_scope,
        "viewpoint": decision.viewpoint,
        "primary_identity_region_visible": (
            decision.primary_identity_region_visible
        ),
        "major_structure_visible": decision.major_structure_visible,
        "truncation_severity": decision.truncation_severity,
        "completion_needed_for_reference_use": (
            decision.completion_needed_for_reference_use
        ),
        "detached_target_fragments_present": (
            decision.detached_target_fragments_present
        ),
        "repair_attempts": attempt.repair_attempts,
        "duration_seconds": duration_seconds,
        "raw_response_count": len(attempt.raw_responses),
        "baseline": baseline_value,
    }


def _case_key(record: Mapping[str, object]) -> dict[str, str]:
    return {
        "clip_uid": str(record["clip_uid"]),
        "entity_id": str(record["entity_id"]),
    }


def _agreement(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary(
    *,
    model: str,
    records: list[dict[str, object]],
    total_seconds: float,
    profiling_summary: Mapping[str, object],
) -> dict[str, object]:
    components = profiling_summary.get("components")
    candidate_profile = (
        components.get("qwen_candidate_judge", {})
        if isinstance(components, Mapping)
        else {}
    )
    if not isinstance(candidate_profile, Mapping):
        candidate_profile = {}
    baseline_records = [
        record
        for record in records
        if isinstance(record.get("baseline"), Mapping)
        and record["baseline"].get("completeness") is not None
    ]
    changed_candidate_cases: list[dict[str, object]] = []
    changed_route_cases: list[dict[str, object]] = []
    candidate_matches = 0
    route_matches = 0
    for record in baseline_records:
        baseline = record["baseline"]
        assert isinstance(baseline, Mapping)
        candidate_matches += int(
            record["selected_candidate_id"] == baseline["selected_candidate_id"]
        )
        route_matches += int(
            record["completeness"] == baseline["completeness"]
            and record["reference_scope"] == baseline["reference_scope"]
        )
        if record["selected_candidate_id"] != baseline["selected_candidate_id"]:
            changed_candidate_cases.append(
                {
                    **_case_key(record),
                    "baseline": baseline["selected_candidate_id"],
                    "replay": record["selected_candidate_id"],
                }
            )
        if (
            record["completeness"] != baseline["completeness"]
            or record["reference_scope"] != baseline["reference_scope"]
        ):
            changed_route_cases.append(
                {
                    **_case_key(record),
                    "baseline": {
                        "completeness": baseline["completeness"],
                        "reference_scope": baseline["reference_scope"],
                    },
                    "replay": {
                        "completeness": record["completeness"],
                        "reference_scope": record["reference_scope"],
                    },
                }
            )
    repair_cases = [
        {**_case_key(record), "repair_attempts": record["repair_attempts"]}
        for record in records
        if int(record["repair_attempts"]) > 0
    ]
    initial_calls = int(candidate_profile.get("initial_calls", 0))
    repair_calls = int(candidate_profile.get("repair_calls", 0))
    result: dict[str, object] = {
        "model": model,
        "entity_count": len(records),
        "total_seconds": total_seconds,
        "mean_seconds_per_entity": (
            sum(float(record["duration_seconds"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "initial_calls": initial_calls,
        "repair_calls": repair_calls,
        "repair_rate": repair_calls / initial_calls if initial_calls else 0.0,
        "candidate_selection_agreement_with_baseline": _agreement(
            candidate_matches,
            len(baseline_records),
        ),
        "route_agreement_with_baseline": _agreement(
            route_matches,
            len(baseline_records),
        ),
        "reject_count": sum(
            record["reference_scope"] == "reject" for record in records
        ),
        "complete_count": sum(
            record["completeness"] == "complete" for record in records
        ),
        "local_usable_count": sum(
            record["completeness"] == "local_usable" for record in records
        ),
        "repairable_count": sum(
            record["completeness"] == "repairable" for record in records
        ),
        "severely_incomplete_count": sum(
            record["completeness"] == "severely_incomplete"
            for record in records
        ),
        "fragmented_count": sum(
            record["completeness"] == "fragmented" for record in records
        ),
        "changed_candidate_cases": changed_candidate_cases,
        "changed_route_cases": changed_route_cases,
        "repair_cases": repair_cases,
        "profiling": {"qwen_candidate_judge": dict(candidate_profile)},
    }
    return result


def _replay_config(
    config: V3Config,
    *,
    run_root: Path,
    base_url: str,
    model: str,
    api_key: str | None,
) -> tuple[V3Config, QwenServiceConfig]:
    original_judge = config.qwen.candidate_judge
    if original_judge is None:
        raise ValueError("original configuration has no candidate judge")
    judge_config = replace(
        original_judge,
        base_url=base_url,
        model=model,
        api_key=original_judge.api_key if api_key is None else api_key,
    )
    replay_config = replace(
        config,
        run_root=run_root,
        pair=replace(config.pair, max_candidates_per_entity=3),
        qwen=replace(config.qwen, candidate_judge=judge_config),
    )
    replay_config.validate()
    return replay_config, judge_config


def run_candidate_judge_replay(
    config: V3Config,
    *,
    run_root: Path,
    base_url: str,
    model: str,
    output_path: Path,
    api_key: str | None = None,
    save_raw: bool = False,
    judge: EntityReferenceJudge | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    root = run_root.expanduser().resolve(strict=True)
    output = _validated_output_path(output_path, root)
    summary_path = Path(f"{output}.summary.json")
    raw_path = Path(f"{output}.raw.jsonl")
    replay_config, judge_config = _replay_config(
        config,
        run_root=root,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    storage = ReadOnlyRunStorage(replay_config)
    run = storage.read_run()
    active_judge = judge
    owned_judge: QwenEntityReferenceJudge | None = None
    records: list[dict[str, object]] = []
    raw_records: list[dict[str, object]] = []
    benchmark_started = clock()
    with TemporaryDirectory(prefix="r2v-v3-candidate-replay-") as temporary:
        profiler = V3Profiler(
            Path(temporary),
            git_commit=run.git_commit,
            clock=clock,
        )
        try:
            with active_profiler(profiler):
                for initial_clip in storage.iter_clips():
                    clip = storage.read_clip(initial_clip.clip_uid)
                    if (
                        clip.annotation is None
                        or clip.annotation.status != "ready"
                        or clip.coverage is None
                        or not clip.coverage.passed
                    ):
                        continue
                    frames = validate_sampled_frames(storage, clip.clip_uid)
                    masks = storage.read_masks(clip.clip_uid)
                    for entity in clip.annotation.entities:
                        tracked = masks.entities.get(entity.entity_id)
                        if tracked is None or tracked.status != "ready":
                            continue
                        candidates = build_entity_reference_candidates(
                            replay_config,
                            storage,
                            clip_uid=clip.clip_uid,
                            entity=entity,
                            frames=frames,
                            masks=masks,
                        )
                        if not candidates:
                            continue
                        if active_judge is None:
                            owned_judge = QwenEntityReferenceJudge(
                                judge_config,
                                repair_retries=replay_config.pair.repair_retries,
                                crop_padding_ratio=(
                                    replay_config.pair.crop_padding_ratio
                                ),
                            )
                            active_judge = owned_judge
                        source_images = _load_source_images(storage, candidates)
                        started = clock()
                        attempt = active_judge.decide(
                            entity=entity,
                            candidates=candidates,
                            source_images=source_images,
                        )
                        duration = clock() - started
                        if duration < 0:
                            raise ValueError("benchmark clock moved backwards")
                        baseline = load_baseline_decision(
                            storage,
                            clip_uid=clip.clip_uid,
                            entity_id=entity.entity_id,
                        )
                        records.append(
                            _result_record(
                                clip_uid=clip.clip_uid,
                                entity=entity,
                                candidate_count=len(candidates),
                                attempt=attempt,
                                baseline=baseline,
                                duration_seconds=duration,
                            )
                        )
                        if save_raw:
                            raw_records.append(
                                {
                                    "clip_uid": clip.clip_uid,
                                    "entity_id": entity.entity_id,
                                    "raw_responses": list(attempt.raw_responses),
                                }
                            )
        finally:
            if owned_judge is not None:
                owned_judge.close()
        profiling_summary = profiler.write_summary()
    total_seconds = clock() - benchmark_started
    if total_seconds < 0:
        raise ValueError("benchmark clock moved backwards")
    summary = _summary(
        model=model,
        records=records,
        total_seconds=total_seconds,
        profiling_summary=profiling_summary,
    )
    _write_jsonl_atomic(output, records)
    write_json_atomic(summary_path, summary)
    if save_raw:
        _write_jsonl_atomic(raw_path, raw_records)
    return summary
