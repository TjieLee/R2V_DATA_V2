from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.embedding_pilot import EmbeddingPilotOccurrence
from r2v_data_v2.h3.face_identity_mining import FaceMiningOccurrence
from r2v_data_v2.h3.schemas import SchemaModel

PAIR_POLICY_REVIEW_SCHEMA_VERSION = "r2v.h3.pair_policy_review.1"
PAIR_POLICY_REPORT_SCHEMA_VERSION = "r2v.h3.pair_policy_calibration_report.1"


class PairPolicyEvidence(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    left_clip_uid: str
    right_clip_uid: str
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    face_left_to_right_rank: int = Field(gt=0)
    face_right_to_left_rank: int = Field(gt=0)
    voice_left_to_right_rank: int = Field(gt=0)
    voice_right_to_left_rank: int = Field(gt=0)
    face_top1_top2_margin_left: float | None = Field(default=None, ge=0)
    face_top1_top2_margin_right: float | None = Field(default=None, ge=0)
    voice_top1_top2_margin_left: float | None = Field(default=None, ge=0)
    voice_top1_top2_margin_right: float | None = Field(default=None, ge=0)
    same_parent: bool | None = None
    parent_video_id: str | None = None
    left_parent_video_id: str | None = None
    right_parent_video_id: str | None = None
    left_clip_suffix: str | None = None
    right_clip_suffix: str | None = None
    left_face_crop_path: str
    right_face_crop_path: str
    left_primary_voice_path: str
    right_primary_voice_path: str

    @model_validator(mode="after")
    def validate_evidence(self) -> PairPolicyEvidence:
        if self.left_occurrence_id >= self.right_occurrence_id:
            raise ValueError("pair-policy endpoints must be canonical")
        if self.left_clip_uid == self.right_clip_uid:
            raise ValueError("pair-policy evidence must be cross-clip")
        if self.same_parent is True:
            if (
                self.parent_video_id is None
                or self.left_parent_video_id != self.parent_video_id
                or self.right_parent_video_id != self.parent_video_id
            ):
                raise ValueError("same-parent provenance is inconsistent")
        elif self.parent_video_id is not None:
            raise ValueError("non-same-parent evidence cannot publish one parent ID")
        return self


class PairPolicyReviewCandidate(PairPolicyEvidence):
    schema_version: Literal["r2v.h3.pair_policy_review.1"] = (
        PAIR_POLICY_REVIEW_SCHEMA_VERSION
    )
    risk_tier: Literal[1, 2, 3, 4]
    risk_reason: Literal[
        "mutual_face_top3",
        "face_and_voice_rank_support",
        "mutual_voice_top3_with_face_retrieval_support",
        "remaining_unknown_retrieval_risk",
    ]
    same_person_label: None = None
    thresholds_calibrated: Literal[False] = False


class PairPolicyReviewSummary(SchemaModel):
    schema_version: Literal["r2v.h3.pair_policy_review_summary.1"] = (
        "r2v.h3.pair_policy_review_summary.1"
    )
    embedding_root: str
    confirmed_face_pairs_path: str
    face_mining_root: str | None = None
    both_embedding_occurrence_count: int = Field(ge=0)
    all_cross_clip_pair_count: int = Field(ge=0)
    direct_human_same_pair_count: int = Field(ge=0)
    direct_human_same_excluded_pair_count: int = Field(ge=0)
    component_implied_same_excluded_pair_count: int = Field(ge=0)
    unknown_pair_count: int = Field(ge=0)
    emitted_review_candidate_count: int = Field(ge=0)
    requested_top: int = Field(gt=0)
    risk_tier_counts: dict[str, int]
    thresholds_calibrated: Literal[False] = False
    production_pair_acceptance_enabled: Literal[False] = False


class PairPolicyHumanLabel(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    same_person_label: Literal["same", "different", "uncertain"]
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    face_left_to_right_rank: int = Field(gt=0)
    face_right_to_left_rank: int = Field(gt=0)
    voice_left_to_right_rank: int = Field(gt=0)
    voice_right_to_left_rank: int = Field(gt=0)
    face_top1_top2_margin_left: float | None = Field(default=None, ge=0)
    face_top1_top2_margin_right: float | None = Field(default=None, ge=0)
    voice_top1_top2_margin_left: float | None = Field(default=None, ge=0)
    voice_top1_top2_margin_right: float | None = Field(default=None, ge=0)
    same_parent: bool | None = None
    parent_video_id: str | None = None
    risk_tier: Literal[1, 2, 3, 4]

    @model_validator(mode="after")
    def validate_label(self) -> PairPolicyHumanLabel:
        if self.left_occurrence_id >= self.right_occurrence_id:
            raise ValueError("pair-policy label endpoints must be canonical")
        return self


class MetricDistribution(SchemaModel):
    count: int = Field(ge=0)
    minimum: float | None = None
    p10: float | None = None
    median: float | None = None
    p90: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> MetricDistribution:
        values = (self.minimum, self.p10, self.median, self.p90, self.maximum)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty metric distribution cannot publish values")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty metric distribution requires all values")
        return self


class PairRankDiagnostics(SchemaModel):
    pair_count: int = Field(ge=0)
    mutual_face_top1_count: int = Field(ge=0)
    mutual_face_top1_rate: float = Field(ge=0, le=1)
    mutual_face_top3_count: int = Field(ge=0)
    mutual_face_top3_rate: float = Field(ge=0, le=1)
    mutual_voice_top1_count: int = Field(ge=0)
    mutual_voice_top1_rate: float = Field(ge=0, le=1)
    mutual_voice_top3_count: int = Field(ge=0)
    mutual_voice_top3_rate: float = Field(ge=0, le=1)
    face_top1_top2_margin_distribution: MetricDistribution
    voice_top1_top2_margin_distribution: MetricDistribution


class ThresholdSimulationPolicy(SchemaModel):
    minimum_face_cosine: float = Field(ge=-1, le=1)
    minimum_voice_cosine: float | None = Field(default=None, ge=-1, le=1)
    maximum_face_mutual_rank: int | None = Field(default=None, gt=0)
    maximum_voice_mutual_rank: int | None = Field(default=None, gt=0)
    minimum_face_margin: float | None = Field(default=None, ge=0)
    minimum_voice_margin: float | None = Field(default=None, ge=0)


class ThresholdSimulationResult(SchemaModel):
    policy: ThresholdSimulationPolicy
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    evaluated_pair_count: int = Field(ge=0)
    analysis_only: Literal[True] = True


class PairPolicyCalibrationReport(SchemaModel):
    schema_version: Literal["r2v.h3.pair_policy_calibration_report.1"] = (
        PAIR_POLICY_REPORT_SCHEMA_VERSION
    )
    embedding_root: str
    confirmed_face_pairs_path: str
    hard_negative_labels_path: str
    confirmed_same_pair_count: int = Field(ge=0)
    confirmed_different_pair_count: int = Field(ge=0)
    uncertain_pair_count: int = Field(ge=0)
    directly_human_labeled_same_pair_count: int = Field(ge=0)
    component_implied_same_pair_count: int = Field(ge=0)
    component_implied_same_pairs: list[tuple[str, str]]
    face_positive_distribution: MetricDistribution
    face_negative_distribution: MetricDistribution
    voice_positive_distribution: MetricDistribution
    voice_negative_distribution: MetricDistribution
    positive_rank_diagnostics: PairRankDiagnostics
    negative_rank_diagnostics: PairRankDiagnostics
    lowest_face_confirmed_positives: list[dict[str, object]]
    highest_face_confirmed_negatives: list[dict[str, object]]
    lowest_voice_confirmed_positives: list[dict[str, object]]
    highest_voice_confirmed_negatives: list[dict[str, object]]
    highest_risk_false_match_candidates: list[dict[str, object]]
    threshold_simulation: ThresholdSimulationResult | None = None
    thresholds_calibrated: Literal[False] = False
    production_policy_selected: Literal[False] = False


@dataclass(frozen=True)
class _OccurrenceEvidence:
    record: EmbeddingPilotOccurrence
    face_vector: np.ndarray
    voice_vector: np.ndarray
    face_crop_path: Path
    primary_voice_path: Path
    parent_video_id: str | None
    clip_suffix: str | None


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self._parent.setdefault(value, value)
        if self._parent[value] != value:
            self._parent[value] = self.find(self._parent[value])
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_embedding_asset(root: Path, relative_path: str, sha256: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("embedding artifact path must be relative")
    resolved = (root / relative).resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file() or _sha256(resolved) != sha256:
        raise ValueError("embedding artifact provenance mismatch")
    return resolved


def _load_vector(path: Path) -> np.ndarray:
    vector = np.load(path, allow_pickle=False)
    if vector.dtype != np.float32:
        raise ValueError("pair-policy embeddings must be float32")
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pair-policy embedding must be finite and non-empty")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise ValueError("pair-policy embedding must already be L2-normalized")
    return np.ascontiguousarray(values, dtype=np.float32)


def _face_mining_provenance(
    face_mining_root: Path | None,
) -> dict[str, FaceMiningOccurrence]:
    if face_mining_root is None:
        return {}
    root = face_mining_root.expanduser().resolve(strict=True)
    rows = [
        FaceMiningOccurrence.model_validate_json(line)
        for line in (root / "occurrences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {item.entity_occurrence_id: item for item in rows}
    if len(by_id) != len(rows):
        raise ValueError("face mining provenance contains duplicate occurrences")
    return by_id


def _load_occurrences(
    *,
    embedding_root: Path,
    face_mining_root: Path | None,
) -> tuple[Path, dict[str, _OccurrenceEvidence]]:
    root = embedding_root.expanduser().resolve(strict=True)
    provenance = _face_mining_provenance(face_mining_root)
    rows = [
        EmbeddingPilotOccurrence.model_validate_json(line)
        for line in (root / "occurrences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: dict[str, _OccurrenceEvidence] = {}
    for item in sorted(rows, key=lambda value: value.entity_occurrence_id):
        if item.face.status != "available" or item.speaker.status != "available":
            continue
        if (
            item.face.crop_asset is None
            or item.face.embedding_asset is None
            or item.speaker.embedding_asset is None
        ):
            raise ValueError("available embedding occurrence is incomplete")
        face_embedding = _resolve_embedding_asset(
            root,
            item.face.embedding_asset.path,
            item.face.embedding_asset.sha256,
        )
        voice_embedding = _resolve_embedding_asset(
            root,
            item.speaker.embedding_asset.path,
            item.speaker.embedding_asset.sha256,
        )
        crop = _resolve_embedding_asset(
            root,
            item.face.crop_asset.path,
            item.face.crop_asset.sha256,
        )
        voice = Path(item.primary_voice_reference_path).expanduser().resolve(strict=True)
        if not voice.is_file() or _sha256(voice) != item.primary_voice_reference_sha256:
            raise ValueError("primary voice reference provenance mismatch")
        parent = provenance.get(item.entity_occurrence_id)
        output[item.entity_occurrence_id] = _OccurrenceEvidence(
            record=item,
            face_vector=_load_vector(face_embedding),
            voice_vector=_load_vector(voice_embedding),
            face_crop_path=crop,
            primary_voice_path=voice,
            parent_video_id=parent.parent_video_id if parent is not None else None,
            clip_suffix=parent.clip_suffix if parent is not None else None,
        )
    return root, output


def _similarity(
    vectors: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    return {
        (left, right): float(np.dot(vectors[left], vectors[right]))
        for left, right in combinations(sorted(vectors), 2)
    }


def _pair_value(
    values: dict[tuple[str, str], float], left: str, right: str
) -> float:
    return values[tuple(sorted((left, right)))]


def _ranks_and_margins(
    *,
    vectors: dict[str, np.ndarray],
    clip_by_id: dict[str, str],
    similarities: dict[tuple[str, str], float],
) -> tuple[dict[str, dict[str, int]], dict[str, float | None]]:
    ranks: dict[str, dict[str, int]] = {}
    margins: dict[str, float | None] = {}
    for anchor in sorted(vectors):
        candidates = [
            item
            for item in vectors
            if item != anchor and clip_by_id[item] != clip_by_id[anchor]
        ]
        candidates.sort(
            key=lambda item: (-_pair_value(similarities, anchor, item), item)
        )
        ranks[anchor] = {
            item: index for index, item in enumerate(candidates, start=1)
        }
        margins[anchor] = (
            _pair_value(similarities, anchor, candidates[0])
            - _pair_value(similarities, anchor, candidates[1])
            if len(candidates) >= 2
            else None
        )
    return ranks, margins


def build_pair_policy_evidence(
    *,
    embedding_root: Path,
    face_mining_root: Path | None = None,
) -> tuple[Path, dict[str, _OccurrenceEvidence], list[PairPolicyEvidence]]:
    root, occurrences = _load_occurrences(
        embedding_root=embedding_root,
        face_mining_root=face_mining_root,
    )
    face_vectors = {key: value.face_vector for key, value in occurrences.items()}
    voice_vectors = {key: value.voice_vector for key, value in occurrences.items()}
    clip_by_id = {key: value.record.clip_uid for key, value in occurrences.items()}
    face_similarity = _similarity(face_vectors)
    voice_similarity = _similarity(voice_vectors)
    face_ranks, face_margins = _ranks_and_margins(
        vectors=face_vectors,
        clip_by_id=clip_by_id,
        similarities=face_similarity,
    )
    voice_ranks, voice_margins = _ranks_and_margins(
        vectors=voice_vectors,
        clip_by_id=clip_by_id,
        similarities=voice_similarity,
    )
    rows: list[PairPolicyEvidence] = []
    for left, right in combinations(sorted(occurrences), 2):
        left_item = occurrences[left]
        right_item = occurrences[right]
        if left_item.record.clip_uid == right_item.record.clip_uid:
            continue
        same_parent = (
            None
            if left_item.parent_video_id is None or right_item.parent_video_id is None
            else left_item.parent_video_id == right_item.parent_video_id
        )
        rows.append(
            PairPolicyEvidence(
                left_occurrence_id=left,
                right_occurrence_id=right,
                left_clip_uid=left_item.record.clip_uid,
                right_clip_uid=right_item.record.clip_uid,
                face_similarity=_pair_value(face_similarity, left, right),
                voice_similarity=_pair_value(voice_similarity, left, right),
                face_left_to_right_rank=face_ranks[left][right],
                face_right_to_left_rank=face_ranks[right][left],
                voice_left_to_right_rank=voice_ranks[left][right],
                voice_right_to_left_rank=voice_ranks[right][left],
                face_top1_top2_margin_left=face_margins[left],
                face_top1_top2_margin_right=face_margins[right],
                voice_top1_top2_margin_left=voice_margins[left],
                voice_top1_top2_margin_right=voice_margins[right],
                same_parent=same_parent,
                parent_video_id=(
                    left_item.parent_video_id if same_parent is True else None
                ),
                left_parent_video_id=left_item.parent_video_id,
                right_parent_video_id=right_item.parent_video_id,
                left_clip_suffix=left_item.clip_suffix,
                right_clip_suffix=right_item.clip_suffix,
                left_face_crop_path=str(left_item.face_crop_path),
                right_face_crop_path=str(right_item.face_crop_path),
                left_primary_voice_path=str(left_item.primary_voice_path),
                right_primary_voice_path=str(right_item.primary_voice_path),
            )
        )
    return root, occurrences, rows


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if not left or not right or left == right:
        raise ValueError("pair label endpoints must be distinct and non-empty")
    return tuple(sorted((left, right)))


def _load_confirmed_same(path: Path) -> list[tuple[str, str]]:
    source = path.expanduser().resolve(strict=True)
    pairs: list[tuple[str, str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("same_person_label") != "same":
            raise ValueError("confirmed face pairs must contain only HUMAN SAME labels")
        pairs.append(
            _canonical_pair(
                str(payload["left_occurrence_id"]),
                str(payload["right_occurrence_id"]),
            )
        )
    if len(pairs) != len(set(pairs)):
        raise ValueError("confirmed face pairs contain duplicates")
    return sorted(pairs)


def _same_closure(
    direct_pairs: list[tuple[str, str]],
    valid_pairs: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    direct = set(direct_pairs)
    disjoint = _DisjointSet()
    for left, right in direct_pairs:
        disjoint.union(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for item in sorted({value for pair in direct_pairs for value in pair}):
        members[disjoint.find(item)].append(item)
    implied: set[tuple[str, str]] = set()
    for component in members.values():
        for pair in combinations(sorted(component), 2):
            if pair in valid_pairs and pair not in direct:
                implied.add(pair)
    return direct & valid_pairs, implied


def _risk(candidate: PairPolicyEvidence) -> tuple[int, str]:
    face_max = max(
        candidate.face_left_to_right_rank,
        candidate.face_right_to_left_rank,
    )
    face_min = min(
        candidate.face_left_to_right_rank,
        candidate.face_right_to_left_rank,
    )
    voice_max = max(
        candidate.voice_left_to_right_rank,
        candidate.voice_right_to_left_rank,
    )
    voice_min = min(
        candidate.voice_left_to_right_rank,
        candidate.voice_right_to_left_rank,
    )
    if face_max <= 3:
        return 1, "mutual_face_top3"
    if face_min <= 3 and voice_min <= 3:
        return 2, "face_and_voice_rank_support"
    if voice_max <= 3 and face_min <= 10:
        return 3, "mutual_voice_top3_with_face_retrieval_support"
    return 4, "remaining_unknown_retrieval_risk"


def _risk_key(candidate: PairPolicyEvidence) -> tuple[object, ...]:
    tier, _ = _risk(candidate)
    return (
        tier,
        max(candidate.face_left_to_right_rank, candidate.face_right_to_left_rank),
        min(candidate.face_left_to_right_rank, candidate.face_right_to_left_rank),
        -candidate.face_similarity,
        max(candidate.voice_left_to_right_rank, candidate.voice_right_to_left_rank),
        min(candidate.voice_left_to_right_rank, candidate.voice_right_to_left_rank),
        -candidate.voice_similarity,
        candidate.left_occurrence_id,
        candidate.right_occurrence_id,
    )


def _candidate(candidate: PairPolicyEvidence) -> PairPolicyReviewCandidate:
    tier, reason = _risk(candidate)
    return PairPolicyReviewCandidate(
        **candidate.model_dump(mode="python"),
        risk_tier=tier,
        risk_reason=reason,
    )


def _jsonl(values: list[SchemaModel]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in values
    )


def _review_url(path: str, output_root: Path) -> str:
    relative = Path(os.path.relpath(Path(path), output_root))
    return quote(relative.as_posix(), safe="/..")


def _review_html(candidates: list[PairPolicyReviewCandidate], output_root: Path) -> str:
    rows = []
    for item in candidates:
        payload = item.model_dump(mode="json")
        payload["left_face_url"] = _review_url(item.left_face_crop_path, output_root)
        payload["right_face_url"] = _review_url(item.right_face_crop_path, output_root)
        payload["left_voice_url"] = _review_url(
            item.left_primary_voice_path, output_root
        )
        payload["right_voice_url"] = _review_url(
            item.right_primary_voice_path, output_root
        )
        rows.append(payload)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 PairPolicy Hard-Negative Review</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#f4f4f5;color:#18181b}}
header{{padding:16px 24px;background:white;border-bottom:1px solid #d4d4d8}}
main{{max-width:1120px;margin:20px auto;padding:0 20px}}
.faces{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.face{{background:white;border:1px solid #d4d4d8;padding:10px}}
img{{width:100%;height:390px;object-fit:contain;background:#e4e4e7}}
audio{{width:100%;margin-top:10px}}
.meta{{background:white;border:1px solid #d4d4d8;padding:14px;margin:14px 0;white-space:pre-wrap}}
.actions{{display:flex;gap:8px;flex-wrap:wrap}}
button{{padding:10px 15px;background:white;border:1px solid #71717a;cursor:pointer}}
button.active{{background:#18181b;color:white}}
@media(max-width:700px){{.faces{{grid-template-columns:1fr}}img{{height:300px}}}}
</style></head><body>
<header><strong>H3 PairPolicy Hard-Negative Review</strong> <span id="progress"></span>
<p>Are these two entity occurrences the same physical person? Audio is supporting evidence only.</p></header>
<main><div class="faces">
<div class="face"><strong id="left-id"></strong><img id="left-face"><audio id="left-audio" controls></audio></div>
<div class="face"><strong id="right-id"></strong><img id="right-face"><audio id="right-audio" controls></audio></div>
</div><div class="meta" id="metrics"></div><div class="actions">
<button data-label="same">1 SAME</button><button data-label="different">2 DIFFERENT</button>
<button data-label="uncertain">3 UNCERTAIN</button><button id="previous">Previous (k/left)</button>
<button id="next">Next (j/right)</button><button id="export">Export JSONL</button>
</div></main><script>
const cases={payload};const key="r2v-h3-pair-policy-labels-v1";
const labels=JSON.parse(localStorage.getItem(key)||"{{}}");let index=0;
function id(x){{return `${{x.left_occurrence_id}}|${{x.right_occurrence_id}}`}}
function show(){{const x=cases[index];document.getElementById("progress").textContent=` ${{cases.length?index+1:0}}/${{cases.length}}`;if(!x)return;
document.getElementById("left-id").textContent=x.left_occurrence_id;document.getElementById("right-id").textContent=x.right_occurrence_id;
document.getElementById("left-face").src=x.left_face_url;document.getElementById("right-face").src=x.right_face_url;
document.getElementById("left-audio").src=x.left_voice_url;document.getElementById("right-audio").src=x.right_voice_url;
document.getElementById("metrics").textContent=`tier=${{x.risk_tier}} ${{x.risk_reason}}\nface=${{x.face_similarity.toFixed(6)}} ranks=${{x.face_left_to_right_rank}}/${{x.face_right_to_left_rank}} margins=${{x.face_top1_top2_margin_left}}/${{x.face_top1_top2_margin_right}}\nvoice=${{x.voice_similarity.toFixed(6)}} ranks=${{x.voice_left_to_right_rank}}/${{x.voice_right_to_left_rank}} margins=${{x.voice_top1_top2_margin_left}}/${{x.voice_top1_top2_margin_right}}\nsame_parent=${{x.same_parent}} parent=${{x.parent_video_id}}`;
document.querySelectorAll("button[data-label]").forEach(b=>b.classList.toggle("active",b.dataset.label===labels[id(x)]));}}
function label(v){{if(!cases.length)return;labels[id(cases[index])]=v;localStorage.setItem(key,JSON.stringify(labels));show()}}
function move(d){{if(cases.length){{index=(index+d+cases.length)%cases.length;show()}}}}
document.querySelectorAll("button[data-label]").forEach(b=>b.onclick=()=>label(b.dataset.label));
document.getElementById("previous").onclick=()=>move(-1);document.getElementById("next").onclick=()=>move(1);
document.getElementById("export").onclick=()=>{{const rows=cases.filter(x=>labels[id(x)]).map(x=>JSON.stringify({{
left_occurrence_id:x.left_occurrence_id,right_occurrence_id:x.right_occurrence_id,same_person_label:labels[id(x)],face_similarity:x.face_similarity,voice_similarity:x.voice_similarity,
face_left_to_right_rank:x.face_left_to_right_rank,face_right_to_left_rank:x.face_right_to_left_rank,voice_left_to_right_rank:x.voice_left_to_right_rank,voice_right_to_left_rank:x.voice_right_to_left_rank,
face_top1_top2_margin_left:x.face_top1_top2_margin_left,face_top1_top2_margin_right:x.face_top1_top2_margin_right,voice_top1_top2_margin_left:x.voice_top1_top2_margin_left,voice_top1_top2_margin_right:x.voice_top1_top2_margin_right,
same_parent:x.same_parent,parent_video_id:x.parent_video_id,risk_tier:x.risk_tier}})).join("\\n")+"\\n";
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows],{{type:"application/x-ndjson"}}));a.download="pair_policy_review_labels.jsonl";a.click();URL.revokeObjectURL(a.href)}};
document.addEventListener("keydown",e=>{{if(e.key==="1")label("same");else if(e.key==="2")label("different");else if(e.key==="3")label("uncertain");else if(e.key==="j"||e.key==="ArrowRight"||e.key==="ArrowDown")move(1);else if(e.key==="k"||e.key==="ArrowLeft"||e.key==="ArrowUp")move(-1)}});show();
</script></body></html>"""


def build_pair_policy_review(
    *,
    embedding_root: Path,
    confirmed_face_pairs: Path,
    output_root: Path,
    top: int = 50,
    face_mining_root: Path | None = None,
    overwrite: bool = False,
) -> PairPolicyReviewSummary:
    if top <= 0:
        raise ValueError("pair-policy review top must be positive")
    root, occurrences, evidence = build_pair_policy_evidence(
        embedding_root=embedding_root,
        face_mining_root=face_mining_root,
    )
    confirmed_path = confirmed_face_pairs.expanduser().resolve(strict=True)
    direct_pairs = _load_confirmed_same(confirmed_path)
    valid_pairs = {
        (item.left_occurrence_id, item.right_occurrence_id) for item in evidence
    }
    direct_excluded, implied_excluded = _same_closure(direct_pairs, valid_pairs)
    excluded = direct_excluded | implied_excluded
    unknown = [
        item
        for item in evidence
        if (item.left_occurrence_id, item.right_occurrence_id) not in excluded
    ]
    unknown.sort(key=_risk_key)
    candidates = [_candidate(item) for item in unknown[:top]]
    destination = output_root.expanduser().resolve(strict=False)
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("pair-policy review output must be separate from embedding root")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"pair-policy review output already exists: {destination}")
    tiers = Counter(str(item.risk_tier) for item in candidates)
    summary = PairPolicyReviewSummary(
        embedding_root=str(root),
        confirmed_face_pairs_path=str(confirmed_path),
        face_mining_root=(
            str(face_mining_root.expanduser().resolve(strict=True))
            if face_mining_root is not None
            else None
        ),
        both_embedding_occurrence_count=len(occurrences),
        all_cross_clip_pair_count=len(evidence),
        direct_human_same_pair_count=len(direct_pairs),
        direct_human_same_excluded_pair_count=len(direct_excluded),
        component_implied_same_excluded_pair_count=len(implied_excluded),
        unknown_pair_count=len(unknown),
        emitted_review_candidate_count=len(candidates),
        requested_top=top,
        risk_tier_counts=dict(sorted(tiers.items())),
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "review_candidates.jsonl").write_text(
            _jsonl(candidates), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(
                summary.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "review.html").write_text(
            _review_html(candidates, destination), encoding="utf-8"
        )
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            destination.replace(backup)
            published = False
            try:
                temporary.replace(destination)
                published = True
            finally:
                if published:
                    shutil.rmtree(backup)
                elif backup.exists() and not destination.exists():
                    backup.replace(destination)
        else:
            temporary.replace(destination)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _distribution(values: list[float]) -> MetricDistribution:
    if not values:
        return MetricDistribution(count=0)
    array = np.asarray(values, dtype=np.float64)
    return MetricDistribution(
        count=len(values),
        minimum=float(np.min(array)),
        p10=float(np.quantile(array, 0.10, method="linear")),
        median=float(np.quantile(array, 0.50, method="linear")),
        p90=float(np.quantile(array, 0.90, method="linear")),
        maximum=float(np.max(array)),
    )


def _rank_diagnostics(rows: list[PairPolicyEvidence]) -> PairRankDiagnostics:
    count = len(rows)
    face_top1 = sum(
        max(item.face_left_to_right_rank, item.face_right_to_left_rank) <= 1
        for item in rows
    )
    face_top3 = sum(
        max(item.face_left_to_right_rank, item.face_right_to_left_rank) <= 3
        for item in rows
    )
    voice_top1 = sum(
        max(item.voice_left_to_right_rank, item.voice_right_to_left_rank) <= 1
        for item in rows
    )
    voice_top3 = sum(
        max(item.voice_left_to_right_rank, item.voice_right_to_left_rank) <= 3
        for item in rows
    )
    face_margins = [
        value
        for item in rows
        for value in (
            item.face_top1_top2_margin_left,
            item.face_top1_top2_margin_right,
        )
        if value is not None
    ]
    voice_margins = [
        value
        for item in rows
        for value in (
            item.voice_top1_top2_margin_left,
            item.voice_top1_top2_margin_right,
        )
        if value is not None
    ]
    denominator = count or 1
    return PairRankDiagnostics(
        pair_count=count,
        mutual_face_top1_count=face_top1,
        mutual_face_top1_rate=face_top1 / denominator if count else 0.0,
        mutual_face_top3_count=face_top3,
        mutual_face_top3_rate=face_top3 / denominator if count else 0.0,
        mutual_voice_top1_count=voice_top1,
        mutual_voice_top1_rate=voice_top1 / denominator if count else 0.0,
        mutual_voice_top3_count=voice_top3,
        mutual_voice_top3_rate=voice_top3 / denominator if count else 0.0,
        face_top1_top2_margin_distribution=_distribution(face_margins),
        voice_top1_top2_margin_distribution=_distribution(voice_margins),
    )


def _evidence_summary(item: PairPolicyEvidence) -> dict[str, object]:
    tier, reason = _risk(item)
    return {
        "left_occurrence_id": item.left_occurrence_id,
        "right_occurrence_id": item.right_occurrence_id,
        "face_similarity": item.face_similarity,
        "voice_similarity": item.voice_similarity,
        "face_ranks": [
            item.face_left_to_right_rank,
            item.face_right_to_left_rank,
        ],
        "voice_ranks": [
            item.voice_left_to_right_rank,
            item.voice_right_to_left_rank,
        ],
        "risk_tier": tier,
        "risk_reason": reason,
    }


def simulate_pair_policy_thresholds(
    *,
    positive_rows: list[PairPolicyEvidence],
    negative_rows: list[PairPolicyEvidence],
    policy: ThresholdSimulationPolicy,
) -> ThresholdSimulationResult:
    def accepted(item: PairPolicyEvidence) -> bool:
        checks = [item.face_similarity >= policy.minimum_face_cosine]
        if policy.minimum_voice_cosine is not None:
            checks.append(item.voice_similarity >= policy.minimum_voice_cosine)
        if policy.maximum_face_mutual_rank is not None:
            checks.append(
                max(
                    item.face_left_to_right_rank,
                    item.face_right_to_left_rank,
                )
                <= policy.maximum_face_mutual_rank
            )
        if policy.maximum_voice_mutual_rank is not None:
            checks.append(
                max(
                    item.voice_left_to_right_rank,
                    item.voice_right_to_left_rank,
                )
                <= policy.maximum_voice_mutual_rank
            )
        if policy.minimum_face_margin is not None:
            margins = (
                item.face_top1_top2_margin_left,
                item.face_top1_top2_margin_right,
            )
            checks.append(
                all(value is not None for value in margins)
                and min(value for value in margins if value is not None)
                >= policy.minimum_face_margin
            )
        if policy.minimum_voice_margin is not None:
            margins = (
                item.voice_top1_top2_margin_left,
                item.voice_top1_top2_margin_right,
            )
            checks.append(
                all(value is not None for value in margins)
                and min(value for value in margins if value is not None)
                >= policy.minimum_voice_margin
            )
        return all(checks)

    true_positive = sum(accepted(item) for item in positive_rows)
    false_negative = len(positive_rows) - true_positive
    false_positive = sum(accepted(item) for item in negative_rows)
    true_negative = len(negative_rows) - false_positive
    predicted_positive = true_positive + false_positive
    return ThresholdSimulationResult(
        policy=policy,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=(
            true_positive / predicted_positive if predicted_positive else None
        ),
        recall=(true_positive / len(positive_rows) if positive_rows else None),
        evaluated_pair_count=len(positive_rows) + len(negative_rows),
    )


def _verify_review_label(
    label: PairPolicyHumanLabel,
    evidence: PairPolicyEvidence,
) -> None:
    pairs = (
        (label.face_similarity, evidence.face_similarity),
        (label.voice_similarity, evidence.voice_similarity),
    )
    if any(not math.isclose(left, right, abs_tol=1e-9) for left, right in pairs):
        raise ValueError("hard-negative label similarity does not match evidence")
    fields = (
        "face_left_to_right_rank",
        "face_right_to_left_rank",
        "voice_left_to_right_rank",
        "voice_right_to_left_rank",
        "face_top1_top2_margin_left",
        "face_top1_top2_margin_right",
        "voice_top1_top2_margin_left",
        "voice_top1_top2_margin_right",
        "same_parent",
        "parent_video_id",
    )
    if any(getattr(label, field) != getattr(evidence, field) for field in fields):
        raise ValueError("hard-negative label diagnostics do not match evidence")
    if label.risk_tier != _risk(evidence)[0]:
        raise ValueError("hard-negative label risk tier does not match evidence")


def report_pair_policy_calibration(
    *,
    embedding_root: Path,
    confirmed_face_pairs: Path,
    hard_negative_labels: Path,
    output_root: Path,
    simulation_policy: ThresholdSimulationPolicy | None = None,
    face_mining_root: Path | None = None,
) -> PairPolicyCalibrationReport:
    root, _, evidence = build_pair_policy_evidence(
        embedding_root=embedding_root,
        face_mining_root=face_mining_root,
    )
    by_pair = {
        (item.left_occurrence_id, item.right_occurrence_id): item
        for item in evidence
    }
    confirmed_path = confirmed_face_pairs.expanduser().resolve(strict=True)
    review_path = hard_negative_labels.expanduser().resolve(strict=True)
    confirmed = _load_confirmed_same(confirmed_path)
    labels = [
        PairPolicyHumanLabel.model_validate_json(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    review_pairs = [
        (item.left_occurrence_id, item.right_occurrence_id) for item in labels
    ]
    if len(review_pairs) != len(set(review_pairs)):
        raise ValueError("hard-negative review labels contain duplicate pairs")
    if set(confirmed) & set(review_pairs):
        raise ValueError("confirmed and hard-negative review labels overlap")
    for label in labels:
        pair = (label.left_occurrence_id, label.right_occurrence_id)
        if pair not in by_pair:
            raise ValueError("hard-negative label references unavailable evidence")
        _verify_review_label(label, by_pair[pair])
    if any(pair not in by_pair for pair in confirmed):
        raise ValueError("confirmed SAME pair references unavailable evidence")

    positive_pairs = confirmed + [
        pair
        for pair, label in zip(review_pairs, labels, strict=True)
        if label.same_person_label == "same"
    ]
    negative_pairs = [
        pair
        for pair, label in zip(review_pairs, labels, strict=True)
        if label.same_person_label == "different"
    ]
    uncertain_pairs = [
        pair
        for pair, label in zip(review_pairs, labels, strict=True)
        if label.same_person_label == "uncertain"
    ]
    positive_rows = [by_pair[pair] for pair in positive_pairs]
    negative_rows = [by_pair[pair] for pair in negative_pairs]
    _, implied = _same_closure(positive_pairs, set(by_pair))
    boundary_count = 5
    report = PairPolicyCalibrationReport(
        embedding_root=str(root),
        confirmed_face_pairs_path=str(confirmed_path),
        hard_negative_labels_path=str(review_path),
        confirmed_same_pair_count=len(positive_rows),
        confirmed_different_pair_count=len(negative_rows),
        uncertain_pair_count=len(uncertain_pairs),
        directly_human_labeled_same_pair_count=len(positive_rows),
        component_implied_same_pair_count=len(implied),
        component_implied_same_pairs=sorted(implied),
        face_positive_distribution=_distribution(
            [item.face_similarity for item in positive_rows]
        ),
        face_negative_distribution=_distribution(
            [item.face_similarity for item in negative_rows]
        ),
        voice_positive_distribution=_distribution(
            [item.voice_similarity for item in positive_rows]
        ),
        voice_negative_distribution=_distribution(
            [item.voice_similarity for item in negative_rows]
        ),
        positive_rank_diagnostics=_rank_diagnostics(positive_rows),
        negative_rank_diagnostics=_rank_diagnostics(negative_rows),
        lowest_face_confirmed_positives=[
            _evidence_summary(item)
            for item in sorted(
                positive_rows,
                key=lambda value: (
                    value.face_similarity,
                    value.left_occurrence_id,
                    value.right_occurrence_id,
                ),
            )[:boundary_count]
        ],
        highest_face_confirmed_negatives=[
            _evidence_summary(item)
            for item in sorted(
                negative_rows,
                key=lambda value: (
                    -value.face_similarity,
                    value.left_occurrence_id,
                    value.right_occurrence_id,
                ),
            )[:boundary_count]
        ],
        lowest_voice_confirmed_positives=[
            _evidence_summary(item)
            for item in sorted(
                positive_rows,
                key=lambda value: (
                    value.voice_similarity,
                    value.left_occurrence_id,
                    value.right_occurrence_id,
                ),
            )[:boundary_count]
        ],
        highest_voice_confirmed_negatives=[
            _evidence_summary(item)
            for item in sorted(
                negative_rows,
                key=lambda value: (
                    -value.voice_similarity,
                    value.left_occurrence_id,
                    value.right_occurrence_id,
                ),
            )[:boundary_count]
        ],
        highest_risk_false_match_candidates=[
            _evidence_summary(item)
            for item in sorted(negative_rows, key=_risk_key)[:boundary_count]
        ],
        threshold_simulation=(
            simulate_pair_policy_thresholds(
                positive_rows=positive_rows,
                negative_rows=negative_rows,
                policy=simulation_policy,
            )
            if simulation_policy is not None
            else None
        ),
    )
    destination_root = output_root.expanduser().resolve(strict=False)
    if (
        destination_root == root
        or root in destination_root.parents
        or destination_root in root.parents
    ):
        raise ValueError(
            "pair-policy calibration report output must be separate from "
            "embedding root"
        )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "pair_policy_calibration_report.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report
