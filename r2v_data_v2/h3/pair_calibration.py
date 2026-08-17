from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    ClipRecord,
    EntityReferenceState,
    RunRecord,
    SampledFramesArtifact,
    SchemaModel,
    TrackedMasksArtifact,
)

PAIR_CALIBRATION_POLICY_VERSION = "h3_pair_calibration_selection_v2"
PAIR_CALIBRATION_PLAN_SCHEMA_VERSION = "r2v.h3.pair_calibration_plan.1"
PAIR_CALIBRATION_PAIR_SCHEMA_VERSION = "r2v.h3.visual_candidate_pair.1"
CalibrationSelectionSource = Literal[
    "seed_audio_pilot",
    "existing_v3_cross_pair_provenance",
    "same_parent_multi_clip",
]


class PairCalibrationRunProvenance(SchemaModel):
    run_root: str
    run_id: str
    git_commit: str
    config_hash: str
    run_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PairCalibrationSelectedClip(SchemaModel):
    clip_uid: str
    parent_video_id: str
    clip_suffix: str
    selection_source: CalibrationSelectionSource


class PairCalibrationSkippedPair(SchemaModel):
    left_clip_uid: str
    left_entity_id: str
    right_clip_uid: str
    right_entity_id: str
    reason: str


class PairCalibrationPlan(SchemaModel):
    schema_version: Literal["r2v.h3.pair_calibration_plan.1"] = (
        PAIR_CALIBRATION_PLAN_SCHEMA_VERSION
    )
    source_run: PairCalibrationRunProvenance
    seed_audio_pilot_root: str | None = None
    requested_max_clips: int = Field(gt=0)
    selected_clip_count: int = Field(ge=0)
    seed_clip_count: int = Field(ge=0)
    priority_a_clip_count: int = Field(ge=0)
    priority_b_clip_count: int = Field(ge=0)
    selected_parent_group_count: int = Field(ge=0)
    candidate_parent_group_count: int = Field(ge=0)
    max_clips_per_parent: int = Field(gt=0)
    visual_candidate_pair_count: int = Field(ge=0)
    skip_reason_counts: dict[str, int]
    selected_clips: list[PairCalibrationSelectedClip]
    skipped_priority_a_pairs: list[PairCalibrationSkippedPair]
    deterministic_selection_policy_version: Literal[
        "h3_pair_calibration_selection_v2"
    ] = PAIR_CALIBRATION_POLICY_VERSION
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> PairCalibrationPlan:
        if self.selected_clip_count != len(self.selected_clips):
            raise ValueError("selected clip count does not match selected_clips")
        source_counts = Counter(item.selection_source for item in self.selected_clips)
        if self.seed_clip_count != source_counts["seed_audio_pilot"]:
            raise ValueError("seed clip count does not match selected clips")
        if self.priority_a_clip_count != source_counts[
            "existing_v3_cross_pair_provenance"
        ]:
            raise ValueError("priority A clip count does not match selected clips")
        if self.priority_b_clip_count != source_counts["same_parent_multi_clip"]:
            raise ValueError("priority B clip count does not match selected clips")
        if self.selected_clip_count > self.requested_max_clips:
            raise ValueError("selected clips exceed requested maximum")
        if any(value < 0 for value in self.skip_reason_counts.values()):
            raise ValueError("skip reason counts must be non-negative")
        return self


class VisualCandidatePair(SchemaModel):
    schema_version: Literal["r2v.h3.visual_candidate_pair.1"] = (
        PAIR_CALIBRATION_PAIR_SCHEMA_VERSION
    )
    left_clip_uid: str
    left_entity_id: str
    right_clip_uid: str
    right_entity_id: str
    same_parent: bool
    candidate_source: Literal[
        "existing_v3_cross_pair_provenance",
        "same_parent_multi_clip",
    ]
    provenance: dict[str, object]
    same_person_label: None = None
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_pair(self) -> VisualCandidatePair:
        left = (self.left_clip_uid, self.left_entity_id)
        right = (self.right_clip_uid, self.right_entity_id)
        if left >= right:
            raise ValueError("visual candidate pair endpoints must be canonical")
        if self.left_clip_uid == self.right_clip_uid:
            raise ValueError("visual candidate pair endpoints must use distinct clips")
        if not self.same_parent:
            raise ValueError("V1 calibration candidate pairs must share a parent")
        return self


@dataclass(frozen=True)
class _EligibleSubject:
    entity: AnnotationEntity
    reference: EntityReferenceState
    canonical_reference_path: Path
    canonical_reference_sha256: str
    visual_integrity_provenance: dict[str, object]


@dataclass(frozen=True)
class _EligibleClip:
    clip: ClipRecord
    subjects: tuple[_EligibleSubject, ...]


@dataclass(frozen=True)
class EligibleVisualFaceOccurrence:
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    parent_video_id: str
    clip_suffix: str
    canonical_reference_path: Path
    canonical_reference_sha256: str
    canonical_reference_run_path: str
    existing_v3_cross_pair_provenance: dict[str, str] | None
    visual_integrity_provenance: dict[str, object]


@dataclass(frozen=True)
class VisualFaceOccurrenceLoadResult:
    run_root: Path
    occurrences: tuple[EligibleVisualFaceOccurrence, ...]
    skip_reason_counts: dict[str, int]


@dataclass(frozen=True)
class _PriorityARelation:
    target_clip_uid: str
    target_entity_id: str
    donor_clip_uid: str
    donor_entity_id: str
    target_image_path: str
    donor_image_path: str

    @property
    def canonical_endpoints(self) -> tuple[tuple[str, str], tuple[str, str]]:
        left, right = sorted(
            (
                (self.target_clip_uid, self.target_entity_id),
                (self.donor_clip_uid, self.donor_entity_id),
            )
        )
        return left, right


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _resolve_run_artifact(run_root: Path, value: str | None) -> Path:
    if value is None:
        raise ValueError("ready canonical reference is missing image_path")
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else run_root / path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("canonical reference must remain inside run_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError("canonical reference is not a file")
    return resolved


def _validate_roots(
    run_root: Path,
    output_root: Path,
    seed_audio_pilot_root: Path | None,
) -> tuple[Path, Path, Path | None]:
    source = run_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("calibration plan output must be separate from run_root")
    if not (source / "run.json").is_file() or not (source / "clips").is_dir():
        raise ValueError("run_root is not an initialized V3 run")
    if output.exists():
        raise FileExistsError(f"calibration plan output already exists: {output}")
    seed = None
    if seed_audio_pilot_root is not None:
        seed = seed_audio_pilot_root.expanduser().resolve(strict=True)
        if output == seed or seed in output.parents or output in seed.parents:
            raise ValueError("calibration plan output must be separate from seed pilot")
    return source, output, seed


def _integrity_allows_subject(clip: ClipRecord, entity_id: str) -> bool:
    integrity = clip.reference_integrity
    if integrity is None:
        return True
    if integrity.status != "ready":
        return False
    matching = [item for item in integrity.entities if item.entity_id == entity_id]
    return len(matching) == 1 and matching[0].status in {"accepted", "skipped"}


def _integrity_provenance(clip: ClipRecord, entity_id: str) -> dict[str, object]:
    integrity = clip.reference_integrity
    if integrity is None:
        return {"stage_status": "not_present", "entity_status": "not_reviewed"}
    matching = [item for item in integrity.entities if item.entity_id == entity_id]
    if len(matching) != 1:
        return {"stage_status": integrity.status, "entity_status": "missing"}
    item = matching[0]
    return {
        "stage_status": integrity.status,
        "entity_status": item.status,
        "reviewed": item.reviewed,
        "final_reference_path": item.final_reference_path,
        "review_verdict": item.review.verdict if item.review is not None else None,
        "reason": item.reason,
    }


def _load_eligible_clip(run_root: Path, clip_path: Path) -> _EligibleClip:
    clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
    if clip.clip_uid != clip_path.parent.name:
        raise ValueError("clip.json UID does not match its directory")
    if clip.annotation is None or clip.annotation.status != "ready":
        raise ValueError("annotation_not_ready")
    if clip.coverage is None or not clip.coverage.passed:
        raise ValueError("coverage_not_passed")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise ValueError("pairing_not_ready")

    source_video = Path(clip.source.video_path).expanduser()
    if not source_video.is_absolute() or not source_video.resolve(strict=True).is_file():
        raise ValueError("source_video_unavailable")

    frames = SampledFramesArtifact.model_validate_json(
        (clip_path.parent / "frames" / "frames.json").read_text(encoding="utf-8")
    )
    masks = TrackedMasksArtifact.model_validate_json(
        (clip_path.parent / "masks.rle.json").read_text(encoding="utf-8")
    )
    if frames.clip_uid != clip.clip_uid or masks.clip_uid != clip.clip_uid:
        raise ValueError("visual_evidence_identity_mismatch")

    entities = {entity.entity_id: entity for entity in clip.annotation.entities}
    references = {
        reference.entity_id: reference
        for reference in clip.references.entities
        if reference.status == "ready"
    }
    subjects: list[_EligibleSubject] = []
    for entity_id in clip.pairing.retained_entity_ids:
        entity = entities.get(entity_id)
        reference = references.get(entity_id)
        tracked = masks.entities.get(entity_id)
        if (
            entity is None
            or entity.reference_type != "subject"
            or reference is None
            or tracked is None
            or tracked.status != "ready"
            or not _integrity_allows_subject(clip, entity_id)
        ):
            continue
        canonical_path = _resolve_run_artifact(run_root, reference.image_path)
        subjects.append(
            _EligibleSubject(
                entity=entity,
                reference=reference,
                canonical_reference_path=canonical_path,
                canonical_reference_sha256=_sha256(canonical_path),
                visual_integrity_provenance=_integrity_provenance(clip, entity_id),
            )
        )
    if not subjects:
        raise ValueError("no_ready_human_subject")
    return _EligibleClip(clip=clip, subjects=tuple(subjects))


def _skip_reason(exc: Exception) -> str:
    message = str(exc).strip()
    recognized = {
        "annotation_not_ready",
        "coverage_not_passed",
        "pairing_not_ready",
        "source_video_unavailable",
        "visual_evidence_identity_mismatch",
        "no_ready_human_subject",
    }
    return message if message in recognized else "invalid_visual_clip"


def load_eligible_visual_face_occurrences(
    run_root: Path,
) -> VisualFaceOccurrenceLoadResult:
    source = run_root.expanduser().resolve(strict=True)
    if not (source / "run.json").is_file() or not (source / "clips").is_dir():
        raise ValueError("run_root is not an initialized V3 run")
    skip_reasons: Counter[str] = Counter()
    occurrences: list[EligibleVisualFaceOccurrence] = []
    for clip_path in sorted((source / "clips").glob("*/clip.json")):
        try:
            eligible = _load_eligible_clip(source, clip_path)
        except Exception as exc:  # noqa: BLE001 - invalid clips are exclusions
            skip_reasons[_skip_reason(exc)] += 1
            continue
        for subject in eligible.subjects:
            reference = subject.reference
            source_clip_uid = reference.source_clip_uid
            provenance = None
            if source_clip_uid not in {None, eligible.clip.clip_uid}:
                assert reference.source_entity_id is not None
                provenance = {
                    "target_clip_uid": eligible.clip.clip_uid,
                    "target_entity_id": subject.entity.entity_id,
                    "donor_clip_uid": source_clip_uid,
                    "donor_entity_id": reference.source_entity_id,
                }
            occurrences.append(
                EligibleVisualFaceOccurrence(
                    entity_occurrence_id=(
                        f"{eligible.clip.clip_uid}/{subject.entity.entity_id}"
                    ),
                    clip_uid=eligible.clip.clip_uid,
                    entity_id=subject.entity.entity_id,
                    parent_video_id=eligible.clip.source.parent_video_id,
                    clip_suffix=eligible.clip.source.clip_suffix,
                    canonical_reference_path=subject.canonical_reference_path,
                    canonical_reference_sha256=(
                        subject.canonical_reference_sha256
                    ),
                    canonical_reference_run_path=str(reference.image_path),
                    existing_v3_cross_pair_provenance=provenance,
                    visual_integrity_provenance=(
                        subject.visual_integrity_provenance
                    ),
                )
            )
    occurrences.sort(key=lambda item: item.entity_occurrence_id)
    return VisualFaceOccurrenceLoadResult(
        run_root=source,
        occurrences=tuple(occurrences),
        skip_reason_counts=dict(sorted(skip_reasons.items())),
    )


def _seed_clip_ids(seed_root: Path | None) -> list[str]:
    if seed_root is None:
        return []
    clip_ids: list[str] = []
    clips_root = seed_root / "clips"
    if clips_root.is_dir():
        for path in sorted(clips_root.glob("*/audio_binding.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            clip_uid = payload.get("clip_uid")
            if clip_uid != path.parent.name:
                raise ValueError("seed Audio binding clip identity mismatch")
            clip_ids.append(str(clip_uid))
    elif (seed_root / "audio_bindings.jsonl").is_file():
        for line in (seed_root / "audio_bindings.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                clip_ids.append(str(json.loads(line)["clip_uid"]))
    else:
        raise ValueError("seed Audio pilot contains no binding artifacts")
    return list(dict.fromkeys(clip_ids))


def _priority_a_relations(
    eligible: dict[str, _EligibleClip],
) -> list[_PriorityARelation]:
    relations: dict[
        tuple[tuple[str, str], tuple[str, str]], _PriorityARelation
    ] = {}
    for clip_uid in sorted(eligible):
        candidate = eligible[clip_uid]
        for subject in candidate.subjects:
            donor_clip_uid = subject.reference.source_clip_uid
            donor_entity_id = subject.reference.source_entity_id
            if donor_clip_uid in {None, clip_uid} or donor_entity_id is None:
                continue
            donor_clip = eligible.get(donor_clip_uid)
            if donor_clip is None:
                continue
            donor_subject = next(
                (
                    item
                    for item in donor_clip.subjects
                    if item.entity.entity_id == donor_entity_id
                ),
                None,
            )
            if donor_subject is None:
                continue
            if (
                donor_clip.clip.source.parent_video_id
                != candidate.clip.source.parent_video_id
            ):
                continue
            relation = _PriorityARelation(
                target_clip_uid=clip_uid,
                target_entity_id=subject.entity.entity_id,
                donor_clip_uid=donor_clip_uid,
                donor_entity_id=donor_entity_id,
                target_image_path=str(subject.reference.image_path),
                donor_image_path=str(donor_subject.reference.image_path),
            )
            relations.setdefault(relation.canonical_endpoints, relation)
    return [relations[key] for key in sorted(relations)]


def _can_add(
    clip_uids: list[str],
    *,
    selected: set[str],
    parent_counts: Counter[str],
    eligible: dict[str, _EligibleClip],
    max_clips: int,
    max_clips_per_parent: int,
) -> str | None:
    missing = [clip_uid for clip_uid in clip_uids if clip_uid not in selected]
    if len(selected) + len(missing) > max_clips:
        return "max_clips_reached"
    additions = Counter(
        eligible[clip_uid].clip.source.parent_video_id for clip_uid in missing
    )
    if any(
        parent_counts[parent] + count > max_clips_per_parent
        for parent, count in additions.items()
    ):
        return "max_clips_per_parent_reached"
    return None


def _candidate_pairs(
    *,
    eligible: dict[str, _EligibleClip],
    selected: set[str],
    priority_a: list[_PriorityARelation],
) -> list[VisualCandidatePair]:
    pairs: dict[
        tuple[tuple[str, str], tuple[str, str]], VisualCandidatePair
    ] = {}
    for relation in priority_a:
        left, right = relation.canonical_endpoints
        if left[0] not in selected or right[0] not in selected:
            continue
        target = eligible[relation.target_clip_uid].clip
        donor = eligible[relation.donor_clip_uid].clip
        pairs[(left, right)] = VisualCandidatePair(
            left_clip_uid=left[0],
            left_entity_id=left[1],
            right_clip_uid=right[0],
            right_entity_id=right[1],
            same_parent=(
                target.source.parent_video_id == donor.source.parent_video_id
            ),
            candidate_source="existing_v3_cross_pair_provenance",
            provenance={
                "target_clip_uid": relation.target_clip_uid,
                "target_entity_id": relation.target_entity_id,
                "donor_clip_uid": relation.donor_clip_uid,
                "donor_entity_id": relation.donor_entity_id,
                "target_reference_image_path": relation.target_image_path,
                "donor_reference_image_path": relation.donor_image_path,
                "parent_video_id": target.source.parent_video_id,
            },
        )

    selected_by_parent: dict[str, list[_EligibleClip]] = defaultdict(list)
    for clip_uid in sorted(selected):
        item = eligible[clip_uid]
        selected_by_parent[item.clip.source.parent_video_id].append(item)
    for parent_video_id in sorted(selected_by_parent):
        clips = sorted(
            selected_by_parent[parent_video_id],
            key=lambda item: (
                _natural_key(item.clip.source.clip_suffix),
                item.clip.clip_uid,
            ),
        )
        for left_clip, right_clip in combinations(clips, 2):
            for left_subject, right_subject in product(
                left_clip.subjects,
                right_clip.subjects,
            ):
                left, right = sorted(
                    (
                        (
                            left_clip.clip.clip_uid,
                            left_subject.entity.entity_id,
                        ),
                        (
                            right_clip.clip.clip_uid,
                            right_subject.entity.entity_id,
                        ),
                    )
                )
                key = (left, right)
                if key in pairs:
                    continue
                pairs[key] = VisualCandidatePair(
                    left_clip_uid=left[0],
                    left_entity_id=left[1],
                    right_clip_uid=right[0],
                    right_entity_id=right[1],
                    same_parent=True,
                    candidate_source="same_parent_multi_clip",
                    provenance={
                        "parent_video_id": parent_video_id,
                        "clip_suffixes": {
                            left_clip.clip.clip_uid: (
                                left_clip.clip.source.clip_suffix
                            ),
                            right_clip.clip.clip_uid: (
                                right_clip.clip.source.clip_suffix
                            ),
                        },
                    },
                )
    return [pairs[key] for key in sorted(pairs)]


def _write_outputs(
    *,
    destination: Path,
    plan: PairCalibrationPlan,
    pairs: list[VisualCandidatePair],
) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "clip_ids.txt").write_text(
            "".join(f"{item.clip_uid}\n" for item in plan.selected_clips),
            encoding="utf-8",
        )
        (temporary / "plan.json").write_text(
            json.dumps(
                plan.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "visual_candidate_pairs.jsonl").write_text(
            "".join(
                json.dumps(
                    pair.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for pair in pairs
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def plan_h3_pair_calibration(
    *,
    run_root: Path,
    output_root: Path,
    seed_audio_pilot_root: Path | None = None,
    max_clips: int = 60,
    max_clips_per_parent: int = 4,
) -> PairCalibrationPlan:
    if max_clips <= 0 or max_clips_per_parent <= 0:
        raise ValueError("calibration clip limits must be positive")
    source, destination, seed_root = _validate_roots(
        run_root,
        output_root,
        seed_audio_pilot_root,
    )
    run_path = source / "run.json"
    run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    skip_reasons: Counter[str] = Counter()
    eligible: dict[str, _EligibleClip] = {}
    for clip_path in sorted((source / "clips").glob("*/clip.json")):
        try:
            item = _load_eligible_clip(source, clip_path)
        except Exception as exc:  # noqa: BLE001 - invalid clips are plan exclusions
            skip_reasons[_skip_reason(exc)] += 1
            continue
        eligible[item.clip.clip_uid] = item

    parent_groups: dict[str, list[str]] = defaultdict(list)
    for clip_uid, item in eligible.items():
        parent_groups[item.clip.source.parent_video_id].append(clip_uid)
    parent_groups = {
        parent: sorted(
            clip_uids,
            key=lambda clip_uid: (
                _natural_key(eligible[clip_uid].clip.source.clip_suffix),
                clip_uid,
            ),
        )
        for parent, clip_uids in parent_groups.items()
        if len(clip_uids) >= 2
    }

    selected: list[str] = []
    selected_set: set[str] = set()
    selected_sources: dict[str, CalibrationSelectionSource] = {}
    selected_parent_counts: Counter[str] = Counter()
    expansion_parent_counts: Counter[str] = Counter()

    def add(clip_uid: str, selection_source: CalibrationSelectionSource) -> None:
        if clip_uid in selected_set:
            return
        selected.append(clip_uid)
        selected_set.add(clip_uid)
        selected_sources[clip_uid] = selection_source
        parent = eligible[clip_uid].clip.source.parent_video_id
        selected_parent_counts[parent] += 1
        if selection_source != "seed_audio_pilot":
            expansion_parent_counts[parent] += 1

    eligible_seed_ids: list[str] = []
    for clip_uid in _seed_clip_ids(seed_root):
        if clip_uid not in eligible:
            skip_reasons["seed_clip_not_eligible"] += 1
            continue
        eligible_seed_ids.append(clip_uid)
    if len(eligible_seed_ids) > max_clips:
        raise ValueError(
            "eligible seed clip count exceeds max_clips: "
            f"{len(eligible_seed_ids)} > {max_clips}"
        )
    for clip_uid in eligible_seed_ids:
        add(clip_uid, "seed_audio_pilot")

    priority_a = _priority_a_relations(eligible)
    skipped_pairs: list[PairCalibrationSkippedPair] = []
    priority_a_blocked_clips: set[str] = set()
    for relation in priority_a:
        endpoints = [endpoint[0] for endpoint in relation.canonical_endpoints]
        reason = _can_add(
            endpoints,
            selected=selected_set,
            parent_counts=expansion_parent_counts,
            eligible=eligible,
            max_clips=max_clips,
            max_clips_per_parent=max_clips_per_parent,
        )
        if reason is not None:
            left, right = relation.canonical_endpoints
            skipped_pairs.append(
                PairCalibrationSkippedPair(
                    left_clip_uid=left[0],
                    left_entity_id=left[1],
                    right_clip_uid=right[0],
                    right_entity_id=right[1],
                    reason=reason,
                )
            )
            skip_reasons[f"priority_a_{reason}"] += 1
            priority_a_blocked_clips.update(
                clip_uid for clip_uid in endpoints if clip_uid not in selected_set
            )
            continue
        for clip_uid in endpoints:
            add(clip_uid, "existing_v3_cross_pair_provenance")

    group_positions = {parent: 0 for parent in parent_groups}
    for parent in sorted(parent_groups):
        needed = max(0, 2 - selected_parent_counts[parent])
        if needed == 0:
            continue
        clip_uids = parent_groups[parent]
        candidates = [
            clip_uid
            for clip_uid in clip_uids
            if clip_uid not in selected_set
            and clip_uid not in priority_a_blocked_clips
        ][:needed]
        if len(candidates) != needed:
            skip_reasons["priority_b_pair_unavailable"] += 1
            continue
        reason = _can_add(
            candidates,
            selected=selected_set,
            parent_counts=expansion_parent_counts,
            eligible=eligible,
            max_clips=max_clips,
            max_clips_per_parent=max_clips_per_parent,
        )
        if reason is not None:
            skip_reasons[f"priority_b_pair_{reason}"] += 1
            continue
        for clip_uid in candidates:
            add(clip_uid, "same_parent_multi_clip")

    while len(selected) < max_clips:
        changed = False
        for parent in sorted(parent_groups):
            clip_uids = parent_groups[parent]
            position = group_positions[parent]
            while position < len(clip_uids) and (
                clip_uids[position] in selected_set
                or clip_uids[position] in priority_a_blocked_clips
            ):
                position += 1
            group_positions[parent] = position
            if position >= len(clip_uids):
                continue
            clip_uid = clip_uids[position]
            reason = _can_add(
                [clip_uid],
                selected=selected_set,
                parent_counts=expansion_parent_counts,
                eligible=eligible,
                max_clips=max_clips,
                max_clips_per_parent=max_clips_per_parent,
            )
            group_positions[parent] += 1
            if reason is not None:
                skip_reasons[f"priority_b_{reason}"] += 1
                continue
            add(clip_uid, "same_parent_multi_clip")
            changed = True
            if len(selected) >= max_clips:
                break
        if not changed:
            break

    pairs = _candidate_pairs(
        eligible=eligible,
        selected=selected_set,
        priority_a=priority_a,
    )
    selected_clips = [
        PairCalibrationSelectedClip(
            clip_uid=clip_uid,
            parent_video_id=eligible[clip_uid].clip.source.parent_video_id,
            clip_suffix=eligible[clip_uid].clip.source.clip_suffix,
            selection_source=selected_sources[clip_uid],
        )
        for clip_uid in selected
    ]
    selected_parent_group_count = sum(
        selected_parent_counts[parent] >= 2 for parent in parent_groups
    )
    plan = PairCalibrationPlan(
        source_run=PairCalibrationRunProvenance(
            run_root=str(source),
            run_id=run.run_id,
            git_commit=run.git_commit,
            config_hash=run.config_hash,
            run_json_sha256=_sha256(run_path),
        ),
        seed_audio_pilot_root=str(seed_root) if seed_root is not None else None,
        requested_max_clips=max_clips,
        selected_clip_count=len(selected_clips),
        seed_clip_count=sum(
            item.selection_source == "seed_audio_pilot" for item in selected_clips
        ),
        priority_a_clip_count=sum(
            item.selection_source == "existing_v3_cross_pair_provenance"
            for item in selected_clips
        ),
        priority_b_clip_count=sum(
            item.selection_source == "same_parent_multi_clip"
            for item in selected_clips
        ),
        selected_parent_group_count=selected_parent_group_count,
        candidate_parent_group_count=len(parent_groups),
        max_clips_per_parent=max_clips_per_parent,
        visual_candidate_pair_count=len(pairs),
        skip_reason_counts=dict(sorted(skip_reasons.items())),
        selected_clips=selected_clips,
        skipped_priority_a_pairs=skipped_pairs,
    )
    _write_outputs(destination=destination, plan=plan, pairs=pairs)
    return plan
