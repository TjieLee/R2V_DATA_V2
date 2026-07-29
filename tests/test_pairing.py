from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import r2v_data_v2.pairing as pairing_module
from r2v_data_v2.config import PairingConfig, PipelineConfig
from r2v_data_v2.pairing import (
    build_cross_pair_index,
    build_pairs,
    choose_cross_pair,
    cross_pair_coarse_similarity,
    cross_pair_passes,
    indexed_cross_pair_candidates,
    is_same_parent_cross_candidate,
)
from r2v_data_v2.reference_binding import assign_reference_tokens
from r2v_data_v2.schemas import CrossPairJudgeResult, QwenAnnotationResult
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
)
from tests.test_caption_validation import _valid_payload


def _reference(
    tmp_path: Path,
    *,
    clip_uid: str,
    parent: str,
    suffix: str,
    label: str = "watch",
) -> dict[str, object]:
    image = tmp_path / f"{clip_uid}.jpg"
    pixels = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.circle(pixels, (32, 32), 18, (200, 100, 50), -1)
    assert cv2.imwrite(str(image), pixels)
    return {
        "clip_uid": clip_uid,
        "parent_video_id": parent,
        "clip_suffix": suffix,
        "entity_id": "e1",
        "phrase": "a silver watch",
        "canonical_label": label,
        "category": "product",
        "genericity": "descriptive",
        "name_evidence": "none",
        "canonical_path": str(image),
    }


class _FakeJudge:
    def __init__(self, result: CrossPairJudgeResult) -> None:
        self.result = result
        self.calls = 0
        self.seen_clip_uids: list[str] = []

    def judge(
        self,
        *,
        target: dict[str, object],
        candidate: dict[str, object],
    ) -> CrossPairJudgeResult:
        del target
        self.calls += 1
        self.seen_clip_uids.append(str(candidate["clip_uid"]))
        return self.result


class _FailingJudge:
    def judge(
        self,
        *,
        target: dict[str, object],
        candidate: dict[str, object],
    ) -> CrossPairJudgeResult:
        del target, candidate
        raise StructuredOutputFailure(
            raw_responses=["not json", '{"still": "invalid"}'],
            issues=[ValidationIssue("schema_missing_field", "confidence", "missing")],
        )


def _judgment(
    *,
    same: str = "yes",
    confidence: float = 0.95,
    near_duplicate: bool = False,
) -> CrossPairJudgeResult:
    return CrossPairJudgeResult(
        same_exact_instance=same,
        confidence=confidence,
        context_difference="large",
        near_duplicate=near_duplicate,
        conflicting_attributes=[],
        reason="matching product details in a different scene",
    )


def test_cross_pair_requires_same_parent_and_different_complete_suffix(
    tmp_path: Path,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    valid = _reference(
        tmp_path,
        clip_uid="valid",
        parent="movie",
        suffix="11_1",
    )
    same_suffix = _reference(
        tmp_path,
        clip_uid="same-suffix",
        parent="movie",
        suffix="11_0",
    )
    other_parent = _reference(
        tmp_path,
        clip_uid="other",
        parent="different",
        suffix="11_1",
    )

    assert is_same_parent_cross_candidate(target, valid)
    assert not is_same_parent_cross_candidate(target, same_suffix)
    assert not is_same_parent_cross_candidate(target, other_parent)


def test_cross_pair_index_only_queries_same_parent_category_and_label(
    tmp_path: Path,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    same_bucket = _reference(
        tmp_path,
        clip_uid="same-bucket",
        parent="movie",
        suffix="11_1",
    )
    same_suffix = _reference(
        tmp_path,
        clip_uid="same-suffix",
        parent="movie",
        suffix="11_0",
    )
    wrong_label = _reference(
        tmp_path,
        clip_uid="wrong-label",
        parent="movie",
        suffix="12_0",
        label="bracelet",
    )
    wrong_parent = _reference(
        tmp_path,
        clip_uid="wrong-parent",
        parent="other",
        suffix="12_0",
    )
    wrong_category = _reference(
        tmp_path,
        clip_uid="wrong-category",
        parent="movie",
        suffix="13_0",
    )
    wrong_category["category"] = "object"
    index = build_cross_pair_index(
        {
            "target": [target],
            "same": [same_bucket],
            "suffix": [same_suffix],
            "label": [wrong_label],
            "parent": [wrong_parent],
            "category": [wrong_category],
        }
    )

    candidates = indexed_cross_pair_candidates(index, target)

    assert [candidate["clip_uid"] for candidate in candidates] == ["same-bucket"]


def test_named_cross_pair_requires_exact_name_and_valid_evidence(
    tmp_path: Path,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
        label="Michael Jordan",
    )
    target.update({"genericity": "named", "name_evidence": "draft_caption"})
    candidate = _reference(
        tmp_path,
        clip_uid="candidate",
        parent="movie",
        suffix="12_0",
        label="michael jordan",
    )
    candidate.update({"genericity": "named", "name_evidence": "metadata"})

    assert not is_same_parent_cross_candidate(target, candidate)
    candidate["canonical_label"] = "Michael Jordan"
    assert is_same_parent_cross_candidate(target, candidate)
    candidate["name_evidence"] = "visible_text"
    assert not is_same_parent_cross_candidate(target, candidate)


def test_verified_cross_pair_is_selected(tmp_path: Path) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    sibling = _reference(
        tmp_path,
        clip_uid="sibling",
        parent="movie",
        suffix="12_0",
    )
    judge = _FakeJudge(_judgment())

    selected = choose_cross_pair(
        target=target,
        candidates=[sibling],
        config=PairingConfig(),
        judge=judge,  # type: ignore[arg-type]
    )

    assert selected is not None
    assert selected[0]["clip_uid"] == "sibling"
    assert judge.calls == 1


def test_cross_pair_prefers_cached_dino_embedding_for_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    far = _reference(
        tmp_path,
        clip_uid="far",
        parent="movie",
        suffix="12_0",
    )
    close = _reference(
        tmp_path,
        clip_uid="close",
        parent="movie",
        suffix="13_0",
    )
    for reference, embedding in (
        (target, [1.0, 0.0]),
        (far, [0.0, 1.0]),
        (close, [1.0, 0.0]),
    ):
        path = tmp_path / f"{reference['clip_uid']}.npy"
        np.save(path, np.asarray(embedding, dtype=np.float16))
        reference["dinov3_embedding_path"] = str(path)
    monkeypatch.setattr(
        pairing_module,
        "visual_histogram_similarity",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("histogram fallback must not run")
        ),
    )
    judge = _FakeJudge(_judgment())

    selected = choose_cross_pair(
        target=target,
        candidates=[far, close],
        config=PairingConfig(maximum_candidates_per_entity=1),
        judge=judge,  # type: ignore[arg-type]
    )

    assert selected is not None
    assert selected[0]["clip_uid"] == "close"
    assert selected[2] == pytest.approx(1.0)
    assert judge.seen_clip_uids == ["close"]


def test_cross_pair_falls_back_to_histogram_when_embedding_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    candidate = _reference(
        tmp_path,
        clip_uid="candidate",
        parent="movie",
        suffix="12_0",
    )
    monkeypatch.setattr(
        pairing_module,
        "visual_histogram_similarity",
        lambda *_: 0.42,
    )

    assert cross_pair_coarse_similarity(target, candidate) == 0.42


def test_high_dino_similarity_does_not_bypass_qwen_identity_judgment(
    tmp_path: Path,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    candidate = _reference(
        tmp_path,
        clip_uid="candidate",
        parent="movie",
        suffix="12_0",
    )
    for reference in (target, candidate):
        path = tmp_path / f"{reference['clip_uid']}.npy"
        np.save(path, np.asarray([1.0, 0.0], dtype=np.float16))
        reference["dinov3_embedding_path"] = str(path)
    judge = _FakeJudge(_judgment(same="no"))

    selected = choose_cross_pair(
        target=target,
        candidates=[candidate],
        config=PairingConfig(),
        judge=judge,  # type: ignore[arg-type]
    )

    assert selected is None
    assert judge.calls == 1


def test_invalid_cross_pair_judge_output_is_recorded_and_skipped(
    tmp_path: Path,
) -> None:
    target = _reference(
        tmp_path,
        clip_uid="target",
        parent="movie",
        suffix="11_0",
    )
    sibling = _reference(
        tmp_path,
        clip_uid="sibling",
        parent="movie",
        suffix="12_0",
    )
    failures: list[dict[str, object]] = []

    selected = choose_cross_pair(
        target=target,
        candidates=[sibling],
        config=PairingConfig(),
        judge=_FailingJudge(),  # type: ignore[arg-type]
        structured_failures=failures,
    )

    assert selected is None
    assert failures[0]["raw_responses"] == [
        "not json",
        '{"still": "invalid"}',
    ]


def test_uncertain_or_near_duplicate_cross_pair_falls_back() -> None:
    assert not cross_pair_passes(
        _judgment(same="uncertain"),
        minimum_confidence=0.9,
    )
    assert not cross_pair_passes(
        _judgment(near_duplicate=True),
        minimum_confidence=0.9,
    )
    assert not cross_pair_passes(
        _judgment(confidence=0.89),
        minimum_confidence=0.9,
    )


def test_no_ready_reference_sample_is_skipped_without_failure(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True)
    annotation = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_valid_payload())
    )
    record = {
        **annotation.model_dump(mode="json"),
        "clip_uid": "clip-1",
        "video_path": "/read-only/video.mp4",
        "parent_video_id": "parent",
        "clip_suffix": "1_0",
    }
    (manifests / "annotations.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    (manifests / "references.jsonl").write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "reference_type": "entity",
                "entity_id": "e1",
                "status": "pending_inpainting",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = build_pairs(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        judge=_FakeJudge(_judgment()),  # type: ignore[arg-type]
    )

    assert stats.failed == 0
    assert stats.skipped_no_ready_reference == 1
    assert (manifests / "final_samples.jsonl").read_text(encoding="utf-8") == ""
    assert not (output_root / "logs" / "pairing_failed.jsonl").exists()


def test_missing_cross_pair_falls_back_to_in_pair(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    manifests = output_root / "manifests"
    manifests.mkdir(parents=True)
    annotation = assign_reference_tokens(
        QwenAnnotationResult.model_validate(_valid_payload())
    )
    annotation_record = {
        **annotation.model_dump(mode="json"),
        "clip_uid": "clip-1",
        "video_path": "/read-only/video.mp4",
        "parent_video_id": "parent",
        "clip_suffix": "1_0",
    }
    (manifests / "annotations.jsonl").write_text(
        json.dumps(annotation_record) + "\n",
        encoding="utf-8",
    )
    reference_image = output_root / "references" / "clip-1" / "e1" / "canonical.jpg"
    reference_image.parent.mkdir(parents=True)
    assert cv2.imwrite(
        str(reference_image),
        np.full((32, 48, 3), 128, dtype=np.uint8),
    )
    mask_path = reference_image.with_name("mask.png")
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[4:28, 8:40] = 255
    assert cv2.imwrite(str(mask_path), mask)
    reference_record = {
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "phrase": annotation.entities[0].phrase,
        "canonical_label": annotation.entities[0].canonical_label,
        "category": annotation.entities[0].category,
        "ref_token": annotation.entities[0].ref_token,
        "genericity": annotation.entities[0].genericity,
        "name_evidence": annotation.entities[0].name_evidence,
        "canonical_path": str(reference_image),
        "mask_path": str(mask_path),
        "source_frame_index": 4,
        "ranking_score": 0.9,
    }
    (manifests / "references.jsonl").write_text(
        json.dumps(reference_record) + "\n",
        encoding="utf-8",
    )

    stats = build_pairs(
        PipelineConfig(
            dataset_json=tmp_path / "source.jsonl",
            output_root=output_root,
        ),
        judge=_FakeJudge(_judgment()),  # type: ignore[arg-type]
    )

    sample = json.loads((manifests / "final_samples.jsonl").read_text(encoding="utf-8"))
    assert stats.in_pair_count == 1
    assert stats.fallback_count == 1
    assert stats.cross_pair_count == 0
    assert sample["references"][0]["pair_type"] == "in_pair"
    assert sample["entities"][0]["entity_id"] == "e1"
    assert sample["entities"][0]["reference_worthy"]


def test_cross_pair_index_only_contains_ready_references(
    tmp_path: Path,
) -> None:
    ready = _reference(
        tmp_path,
        clip_uid="ready",
        parent="parent",
        suffix="1_0",
    )
    pending = {
        **_reference(
            tmp_path,
            clip_uid="pending",
            parent="parent",
            suffix="2_0",
        ),
        "status": "pending_inpainting",
    }
    rejected = {
        **_reference(
            tmp_path,
            clip_uid="rejected",
            parent="parent",
            suffix="3_0",
        ),
        "status": "rejected",
        "rejected": True,
    }

    index = build_cross_pair_index(
        {
            "ready": [ready],
            "pending": [pending],
            "rejected": [rejected],
        }
    )

    assert [item["clip_uid"] for item in next(iter(index.values()))] == [
        "ready"
    ]
