from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.h3 import audio_pairing
from r2v_data_v2.h3.audio_backends import (
    EmbeddingResult,
    FaceEmbeddingResult,
    PrecomputedEmbeddingBackend,
)
from r2v_data_v2.h3.embedding_pilot import (
    EmbeddingPilotInput,
    EmbeddingPilotOccurrence,
    run_embedding_pilot,
)
from r2v_data_v2.h3.face_identity_mining import FaceMiningOccurrence
from r2v_data_v2.h3.pair_policy_calibration import (
    PairPolicyHumanLabel,
    ThresholdSimulationPolicy,
    build_pair_policy_evidence,
    build_pair_policy_review,
    report_pair_policy_calibration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class _FaceBackend:
    model_identifier = "test/buffalo_l"
    checkpoint_sha256 = "a" * 64

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        unavailable: set[str] | None = None,
    ) -> None:
        self.vectors = vectors
        self.unavailable = unavailable or set()

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        if entity_occurrence_id in self.unavailable:
            return FaceEmbeddingResult(status="unavailable", reason="face_not_found")
        return FaceEmbeddingResult(
            status="available",
            embedding=EmbeddingResult(
                vector=self.vectors[entity_occurrence_id],
                model_identifier=self.model_identifier,
                checkpoint_sha256=self.checkpoint_sha256,
            ),
            face_crop=Image.open(image_path).convert("RGB"),
        )


def _input(root: Path, occurrence_id: str) -> EmbeddingPilotInput:
    clip_uid, entity_id = occurrence_id.split("/")
    visual = root / "visual" / clip_uid / f"{entity_id}.png"
    voice = root / "voice" / clip_uid / f"{entity_id}.flac"
    visual.parent.mkdir(parents=True, exist_ok=True)
    voice.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (40, 80, 120)).save(visual)
    voice.write_bytes(f"voice:{occurrence_id}".encode())
    return EmbeddingPilotInput(
        entity_occurrence_id=occurrence_id,
        clip_uid=clip_uid,
        entity_id=entity_id,
        visual_reference_path=visual,
        visual_reference_sha256=_sha256(visual),
        primary_voice_reference_path=voice,
        primary_voice_reference_sha256=_sha256(voice),
        primary_voice_duration_seconds=1.5,
        local_binding_valid=True,
    )


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[EmbeddingPilotOccurrence]]:
    occurrence_ids = [
        "clip-a/e1",
        "clip-a/e2",
        "clip-b/e1",
        "clip-c/e1",
        "clip-d/e1",
        "clip-e/e1",
    ]
    inputs = [_input(tmp_path, item) for item in occurrence_ids]
    face_vectors = {
        "clip-a/e1": np.array([1.0, 0.0, 0.0]),
        "clip-a/e2": np.array([-1.0, 0.0, 0.0]),
        "clip-b/e1": np.array([0.98, 0.20, 0.0]),
        "clip-c/e1": np.array([0.90, 0.40, 0.0]),
        "clip-d/e1": np.array([0.0, 1.0, 0.0]),
        "clip-e/e1": np.array([0.0, 0.0, 1.0]),
    }
    voice_vectors = {
        "clip-a/e1": np.array([1.0, 0.0, 0.0]),
        "clip-a/e2": np.array([-1.0, 0.0, 0.0]),
        "clip-b/e1": np.array([0.85, 0.52, 0.0]),
        "clip-c/e1": np.array([0.75, 0.66, 0.0]),
        "clip-d/e1": np.array([0.0, 1.0, 0.0]),
        "clip-e/e1": np.array([0.0, 0.0, 1.0]),
    }
    embedding_root = tmp_path / "embedding"
    run_embedding_pilot(
        inputs=inputs,
        output_root=embedding_root,
        face_backend=_FaceBackend(face_vectors, unavailable={"clip-e/e1"}),
        speaker_backend=PrecomputedEmbeddingBackend(
            voice_vectors,
            model_identifier="test/ecapa",
            checkpoint_sha256="b" * 64,
        ),
        top_k=5,
    )
    occurrences = [
        EmbeddingPilotOccurrence.model_validate_json(line)
        for line in (embedding_root / "occurrences.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    face_mining_root = tmp_path / "face_mining"
    face_mining_root.mkdir()
    parent_by_clip = {
        "clip-a": ("parent-1", "1"),
        "clip-b": ("parent-1", "2"),
        "clip-c": ("parent-1", "3"),
        "clip-d": ("parent-2", "1"),
        "clip-e": ("parent-3", "1"),
    }
    mining_rows = []
    for item in occurrences:
        parent, suffix = parent_by_clip[item.clip_uid]
        mining_rows.append(
            FaceMiningOccurrence(
                entity_occurrence_id=item.entity_occurrence_id,
                clip_uid=item.clip_uid,
                entity_id=item.entity_id,
                parent_video_id=parent,
                clip_suffix=suffix,
                canonical_reference_path=item.visual_reference_path,
                canonical_reference_sha256=item.visual_reference_sha256,
                canonical_reference_run_path=f"clips/{item.clip_uid}/selected/{item.entity_id}.png",
                visual_integrity_provenance={"stage_status": "ready"},
                face=item.face,
            )
        )
    (face_mining_root / "occurrences.jsonl").write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
            for row in mining_rows
        )
    )
    confirmed = tmp_path / "confirmed_face_pairs.jsonl"
    confirmed.write_text(
        json.dumps(
            {
                "left_occurrence_id": "clip-a/e1",
                "right_occurrence_id": "clip-b/e1",
                "same_person_label": "same",
            }
        )
        + "\n"
        + json.dumps(
            {
                "left_occurrence_id": "clip-b/e1",
                "right_occurrence_id": "clip-c/e1",
                "same_person_label": "same",
            }
        )
        + "\n"
    )
    return embedding_root, face_mining_root, confirmed, occurrences


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_review_excludes_direct_and_component_same_pairs_and_is_deterministic(
    tmp_path: Path,
) -> None:
    embedding, mining, confirmed, occurrences = _fixture(tmp_path)
    before = _tree_hashes(embedding)
    first = tmp_path / "review-1"
    summary = build_pair_policy_review(
        embedding_root=embedding,
        confirmed_face_pairs=confirmed,
        output_root=first,
        face_mining_root=mining,
        top=50,
    )

    assert summary.both_embedding_occurrence_count == 5
    assert summary.all_cross_clip_pair_count == 9
    assert summary.direct_human_same_excluded_pair_count == 2
    assert summary.component_implied_same_excluded_pair_count == 1
    assert summary.unknown_pair_count == 6
    rows = _read_jsonl(first / "review_candidates.jsonl")
    pairs = {
        (str(row["left_occurrence_id"]), str(row["right_occurrence_id"]))
        for row in rows
    }
    assert len(rows) == len(pairs) == 6
    assert ("clip-a/e1", "clip-b/e1") not in pairs
    assert ("clip-b/e1", "clip-c/e1") not in pairs
    assert ("clip-a/e1", "clip-c/e1") not in pairs
    assert all(
        left.split("/", maxsplit=1)[0] != right.split("/", maxsplit=1)[0]
        for left, right in pairs
    )
    assert all("clip-e/e1" not in pair for pair in pairs)
    assert all(row["same_person_label"] is None for row in rows)
    assert all(row["thresholds_calibrated"] is False for row in rows)
    assert [int(row["risk_tier"]) for row in rows] == sorted(
        int(row["risk_tier"]) for row in rows
    )
    assert _tree_hashes(embedding) == before

    second = tmp_path / "review-2"
    build_pair_policy_review(
        embedding_root=embedding,
        confirmed_face_pairs=confirmed,
        output_root=second,
        face_mining_root=mining,
        top=50,
    )
    assert (first / "review_candidates.jsonl").read_bytes() == (
        second / "review_candidates.jsonl"
    ).read_bytes()
    assert (first / "summary.json").read_bytes() == (
        second / "summary.json"
    ).read_bytes()
    html = (first / "review.html").read_text()
    assert "same physical person" in html
    assert html.count("<audio") == 2
    assert "localStorage" in html and "Export JSONL" in html
    assert all(item.entity_occurrence_id for item in occurrences)


def test_pair_evidence_ranks_and_endpoint_margins_are_exact(tmp_path: Path) -> None:
    embedding, mining, _, _ = _fixture(tmp_path)
    _, _, evidence = build_pair_policy_evidence(
        embedding_root=embedding,
        face_mining_root=mining,
    )
    by_pair = {
        (item.left_occurrence_id, item.right_occurrence_id): item
        for item in evidence
    }
    ab = by_pair[("clip-a/e1", "clip-b/e1")]
    ac = by_pair[("clip-a/e1", "clip-c/e1")]
    ad = by_pair[("clip-a/e1", "clip-d/e1")]
    assert ab.face_left_to_right_rank == 1
    assert ac.face_left_to_right_rank == 2
    assert ad.face_left_to_right_rank == 3
    assert ad.face_top1_top2_margin_left == pytest.approx(
        ab.face_similarity - ac.face_similarity
    )
    assert ad.same_parent is False
    assert ab.same_parent is True
    assert ab.parent_video_id == "parent-1"


def _label_from_candidate(
    candidate: dict[str, object], label: str
) -> PairPolicyHumanLabel:
    fields = PairPolicyHumanLabel.model_fields
    return PairPolicyHumanLabel.model_validate(
        {
            key: candidate[key]
            for key in fields
            if key != "same_person_label"
        }
        | {"same_person_label": label}
    )


def test_report_uses_human_different_and_excludes_uncertain_from_statistics(
    tmp_path: Path,
) -> None:
    embedding, mining, confirmed, _ = _fixture(tmp_path)
    source_before = _tree_hashes(embedding)
    review = tmp_path / "review"
    build_pair_policy_review(
        embedding_root=embedding,
        confirmed_face_pairs=confirmed,
        output_root=review,
        face_mining_root=mining,
        top=50,
    )
    candidates = _read_jsonl(review / "review_candidates.jsonl")
    by_pair = {
        (row["left_occurrence_id"], row["right_occurrence_id"]): row
        for row in candidates
    }
    different = _label_from_candidate(
        by_pair[("clip-a/e1", "clip-d/e1")], "different"
    )
    uncertain = _label_from_candidate(
        by_pair[("clip-b/e1", "clip-d/e1")], "uncertain"
    )
    labels = tmp_path / "pair_policy_review_labels.jsonl"
    labels.write_text(
        different.model_dump_json() + "\n" + uncertain.model_dump_json() + "\n"
    )
    output = tmp_path / "report"
    report = report_pair_policy_calibration(
        embedding_root=embedding,
        confirmed_face_pairs=confirmed,
        hard_negative_labels=labels,
        output_root=output,
        face_mining_root=mining,
        simulation_policy=ThresholdSimulationPolicy(minimum_face_cosine=0.5),
    )

    assert report.confirmed_same_pair_count == 2
    assert report.confirmed_different_pair_count == 1
    assert report.uncertain_pair_count == 1
    assert report.directly_human_labeled_same_pair_count == 2
    assert report.component_implied_same_pair_count == 1
    assert report.component_implied_same_pairs == [
        ("clip-a/e1", "clip-c/e1")
    ]
    assert report.face_positive_distribution.count == 2
    assert report.face_negative_distribution.count == 1
    assert report.voice_positive_distribution.count == 2
    assert report.voice_negative_distribution.count == 1
    assert report.face_negative_distribution.minimum == pytest.approx(
        different.face_similarity
    )
    _, _, evidence = build_pair_policy_evidence(
        embedding_root=embedding,
        face_mining_root=mining,
    )
    by_evidence_pair = {
        (item.left_occurrence_id, item.right_occurrence_id): item
        for item in evidence
    }
    positive_face = sorted(
        (
            by_evidence_pair[("clip-a/e1", "clip-b/e1")].face_similarity,
            by_evidence_pair[("clip-b/e1", "clip-c/e1")].face_similarity,
        )
    )
    face_gap = positive_face[1] - positive_face[0]
    assert report.face_positive_distribution.minimum == pytest.approx(
        positive_face[0]
    )
    assert report.face_positive_distribution.p10 == pytest.approx(
        positive_face[0] + 0.1 * face_gap
    )
    assert report.face_positive_distribution.median == pytest.approx(
        positive_face[0] + 0.5 * face_gap
    )
    assert report.face_positive_distribution.p90 == pytest.approx(
        positive_face[0] + 0.9 * face_gap
    )
    assert report.face_positive_distribution.maximum == pytest.approx(
        positive_face[1]
    )
    assert report.positive_rank_diagnostics.pair_count == 2
    assert report.negative_rank_diagnostics.pair_count == 1
    assert report.threshold_simulation is not None
    assert report.threshold_simulation.true_positive == 2
    assert report.threshold_simulation.false_positive == 0
    assert report.threshold_simulation.false_negative == 0
    assert report.threshold_simulation.true_negative == 1
    assert report.threshold_simulation.precision == 1.0
    assert report.threshold_simulation.recall == 1.0
    payload = json.loads((output / "pair_policy_calibration_report.json").read_text())
    assert payload["thresholds_calibrated"] is False
    assert payload["production_policy_selected"] is False
    assert "best_threshold" not in json.dumps(payload)
    assert _tree_hashes(embedding) == source_before


def test_calibration_closure_is_not_imported_by_production_pairing() -> None:
    source = inspect.getsource(audio_pairing)
    assert "pair_policy_calibration" not in source
    assert "_same_closure" not in source
    assert "max_clips_per_parent" not in source
