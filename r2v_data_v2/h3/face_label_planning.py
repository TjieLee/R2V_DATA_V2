from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.face_identity_mining import FaceIdentityCandidate
from r2v_data_v2.h3.schemas import SchemaModel


class HumanFaceIdentityLabel(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    same_person_label: Literal["same", "different", "uncertain"]
    face_similarity: float = Field(ge=-1, le=1)
    left_to_right_rank: int = Field(gt=0)
    right_to_left_rank: int = Field(gt=0)
    mutual_top_k: bool
    parent_video_id: str

    @model_validator(mode="after")
    def validate_label(self) -> HumanFaceIdentityLabel:
        if self.left_occurrence_id >= self.right_occurrence_id:
            raise ValueError("human face label endpoints must be canonical")
        if not self.parent_video_id.strip():
            raise ValueError("human face label parent provenance must not be empty")
        return self


class ConfirmedHumanFacePair(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    left_clip_uid: str
    right_clip_uid: str
    same_person_label: Literal["same"] = "same"
    face_similarity: float = Field(ge=-1, le=1)
    left_to_right_rank: int = Field(gt=0)
    right_to_left_rank: int = Field(gt=0)
    mutual_top_k: bool
    parent_video_id: str
    label_source: Literal["human_face_identity_review"] = (
        "human_face_identity_review"
    )
    label_file_path: str
    label_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thresholds_calibrated: Literal[False] = False


class FaceLabelAudioPlan(SchemaModel):
    schema_version: Literal["r2v.h3.face_label_audio_plan.1"] = (
        "r2v.h3.face_label_audio_plan.1"
    )
    face_mining_root: str
    labels_path: str
    labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_max_clips: int | None = Field(default=None, gt=0)
    label_count: int = Field(ge=0)
    human_same_pair_count: int = Field(ge=0)
    excluded_different_pair_count: int = Field(ge=0)
    excluded_uncertain_pair_count: int = Field(ge=0)
    skipped_same_pair_due_to_global_limit_count: int = Field(ge=0)
    confirmed_face_pair_count: int = Field(ge=0)
    selected_clip_count: int = Field(ge=0)
    selected_clip_ids: list[str]
    selection_policy: Literal["human_same_pair_endpoints_v1"] = (
        "human_same_pair_endpoints_v1"
    )
    parent_count_quota_applied: Literal[False] = False
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> FaceLabelAudioPlan:
        if self.selected_clip_count != len(self.selected_clip_ids):
            raise ValueError("face-label Audio plan clip count is inconsistent")
        if len(self.selected_clip_ids) != len(set(self.selected_clip_ids)):
            raise ValueError("face-label Audio plan clip IDs must be unique")
        if self.confirmed_face_pair_count > self.human_same_pair_count:
            raise ValueError("confirmed face pairs exceed HUMAN SAME labels")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(
    path: Path,
    model: type[HumanFaceIdentityLabel | FaceIdentityCandidate],
):
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clip_uid(occurrence_id: str) -> str:
    parts = occurrence_id.split("/", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("face occurrence ID must use clip_uid/entity_id")
    return parts[0]


def _matching_candidate(
    label: HumanFaceIdentityLabel,
    candidate: FaceIdentityCandidate,
) -> bool:
    return (
        label.face_similarity == candidate.face_similarity
        and label.left_to_right_rank == candidate.left_to_right_rank
        and label.right_to_left_rank == candidate.right_to_left_rank
        and label.mutual_top_k == candidate.mutual_top_k
        and label.parent_video_id == candidate.parent_video_id
    )


def plan_audio_from_face_labels(
    *,
    face_mining_root: Path,
    labels_path: Path,
    output_root: Path,
    max_clips: int | None = None,
) -> FaceLabelAudioPlan:
    if max_clips is not None and max_clips <= 0:
        raise ValueError("face-label Audio max_clips must be positive")
    mining = face_mining_root.expanduser().resolve(strict=True)
    labels_file = labels_path.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    if not labels_file.is_file():
        raise FileNotFoundError("face identity labels path is not a file")
    if not (mining / "summary.json").is_file():
        raise ValueError("face mining root is incomplete")
    if destination == mining or mining in destination.parents or destination in mining.parents:
        raise ValueError("face-label Audio plan must be separate from face mining root")
    if destination.exists():
        raise FileExistsError(f"face-label Audio plan already exists: {destination}")

    candidates = _read_jsonl(
        mining / "face_candidates.jsonl", FaceIdentityCandidate
    )
    candidate_by_pair = {
        (item.left_occurrence_id, item.right_occurrence_id): item
        for item in candidates
        if item.candidate_pool == "same_parent"
    }
    labels = _read_jsonl(labels_file, HumanFaceIdentityLabel)
    labels.sort(key=lambda item: (item.left_occurrence_id, item.right_occurrence_id))
    pairs = [(item.left_occurrence_id, item.right_occurrence_id) for item in labels]
    if len(pairs) != len(set(pairs)):
        raise ValueError("face identity labels contain duplicate pairs")
    for label in labels:
        pair = (label.left_occurrence_id, label.right_occurrence_id)
        candidate = candidate_by_pair.get(pair)
        if candidate is None or not _matching_candidate(label, candidate):
            raise ValueError("human face label does not match mining evidence")

    label_sha256 = _sha256(labels_file)
    selected: list[str] = []
    selected_set: set[str] = set()
    confirmed: list[ConfirmedHumanFacePair] = []
    skipped_due_to_limit = 0
    for label in labels:
        if label.same_person_label != "same":
            continue
        left_clip = _clip_uid(label.left_occurrence_id)
        right_clip = _clip_uid(label.right_occurrence_id)
        needed = [
            clip_uid
            for clip_uid in (left_clip, right_clip)
            if clip_uid not in selected_set
        ]
        if max_clips is not None and len(selected) + len(needed) > max_clips:
            skipped_due_to_limit += 1
            continue
        for clip_uid in needed:
            selected.append(clip_uid)
            selected_set.add(clip_uid)
        confirmed.append(
            ConfirmedHumanFacePair(
                left_occurrence_id=label.left_occurrence_id,
                right_occurrence_id=label.right_occurrence_id,
                left_clip_uid=left_clip,
                right_clip_uid=right_clip,
                face_similarity=label.face_similarity,
                left_to_right_rank=label.left_to_right_rank,
                right_to_left_rank=label.right_to_left_rank,
                mutual_top_k=label.mutual_top_k,
                parent_video_id=label.parent_video_id,
                label_file_path=str(labels_file),
                label_file_sha256=label_sha256,
            )
        )

    plan = FaceLabelAudioPlan(
        face_mining_root=str(mining),
        labels_path=str(labels_file),
        labels_sha256=label_sha256,
        requested_max_clips=max_clips,
        label_count=len(labels),
        human_same_pair_count=sum(
            item.same_person_label == "same" for item in labels
        ),
        excluded_different_pair_count=sum(
            item.same_person_label == "different" for item in labels
        ),
        excluded_uncertain_pair_count=sum(
            item.same_person_label == "uncertain" for item in labels
        ),
        skipped_same_pair_due_to_global_limit_count=skipped_due_to_limit,
        confirmed_face_pair_count=len(confirmed),
        selected_clip_count=len(selected),
        selected_clip_ids=selected,
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "clip_ids.txt").write_text(
            "".join(f"{clip_uid}\n" for clip_uid in selected), encoding="utf-8"
        )
        (temporary / "confirmed_face_pairs.jsonl").write_text(
            "".join(
                json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for item in confirmed
            ),
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
        temporary.replace(destination)
        return plan
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
