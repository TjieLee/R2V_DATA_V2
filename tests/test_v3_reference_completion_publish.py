from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.v3.reference_completion_publish import (
    IdentityReview,
    LocalityReview,
    PublicationConfig,
    Sam3LocalizedCompletionSegmenter,
    SegmentationMaskCandidate,
    UsabilityReview,
    build_candidate_reference,
    compute_foreground_metrics,
    evaluate_background_gate,
    evaluate_improvement_gate,
    evaluate_mask_gate,
    run_reference_completion_publication,
    select_segmented_mask,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rgba() -> Image.Image:
    pixels = np.full((100, 100, 4), 255, dtype=np.uint8)
    pixels[:, :, 3] = 0
    pixels[25:75, 0:40, :3] = (35, 80, 125)
    pixels[25:75, 0:40, 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _good_mask() -> Image.Image:
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[20:80, 10:56] = 255
    return Image.fromarray(pixels, mode="L")


def _group_mask(*, isolated: bool = False) -> Image.Image:
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[20:80, 10:30] = 255
    start = 75 if isolated else 35
    pixels[20:80, start : start + 20] = 255
    return Image.fromarray(pixels, mode="L")


def _candidate(mask: Image.Image | None = None) -> Image.Image:
    selected = _good_mask() if mask is None else mask
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    foreground = np.asarray(selected) == 255
    yy, xx = np.indices(foreground.shape)
    pixels[foreground, 0] = (40 + xx[foreground]) % 180
    pixels[foreground, 1] = (70 + yy[foreground]) % 190
    pixels[foreground, 2] = 130
    return Image.fromarray(pixels, mode="RGB")


def _identity_accept() -> IdentityReview:
    return IdentityReview(
        verdict="accept",
        same_entity=True,
        identity_preserved=True,
        no_redesign=True,
        no_extra_instance=True,
        reference_type_consistent=True,
        certain=True,
        reason="same entity",
    )


def _locality_accept() -> LocalityReview:
    return LocalityReview(
        verdict="accept",
        local_missing_part_only=True,
        visible_content_preserved=True,
        no_composition_expansion=True,
        no_new_scene_or_unrelated_content=True,
        missing_region_improved=True,
        certain=True,
        reason="localized repair",
    )


def _usability_accept() -> UsabilityReview:
    return UsabilityReview(
        verdict="accept",
        reference_usable=True,
        boundary_clean=True,
        geometry_and_structure_plausible=True,
        no_ghosting_or_duplicate_structure=True,
        no_fracture_extra_limb_or_fragment=True,
        no_text_logo_or_watermark=True,
        certain=True,
        reason="usable reference",
    )


def _reject(review: Any) -> Any:
    payload = review.model_dump()
    for key, value in payload.items():
        if key not in {"verdict", "reason"} and value is True:
            payload[key] = False
            break
    payload["verdict"] = "reject"
    payload["reason"] = "rejected by test"
    return type(review).model_validate(payload)


class _Segmenter:
    def __init__(self, response: object | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def segment(self, **kwargs: object) -> list[SegmentationMaskCandidate]:
        self.calls.append(kwargs)
        response = self.response
        if callable(response):
            response = response(kwargs)
        if isinstance(response, Exception):
            raise response
        mask = _good_mask() if response is None else response
        if isinstance(mask, list):
            return mask
        return [SegmentationMaskCandidate(mask=mask, confidence=0.9, object_id="1")]


class _Judge:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@dataclass
class _Environment:
    data_root: Path
    publication_root: Path
    source: Path
    candidate: Path
    localized_result: Path
    manifest: Path
    config: PublicationConfig


def _write_localized_result(
    environment: _Environment,
    *,
    path: Path | None = None,
    source: Path | None = None,
    candidate: Path | None = None,
    backend: str = "qwen_image_edit_2511",
    mode: str = "localized_raw",
    hard_passed: bool = True,
) -> Path:
    result_path = environment.localized_result if path is None else path
    source_path = environment.source if source is None else source
    candidate_path = environment.candidate if candidate is None else candidate
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "mode": mode,
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        "hard_check": {
            "status": "passed" if hard_passed else "failed",
            "checks": {"candidate_valid": hard_passed},
            "reasons": [] if hard_passed else ["candidate_invalid"],
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _write_manifest(
    environment: _Environment,
    records: list[dict[str, object]] | None = None,
) -> None:
    selected = records or [
        {
            "sample_id": "sample-1",
            "clip_uid": "video_001_clip_0001",
            "entity_id": "e1",
            "reference_type": "subject",
            "entity_phrase": "the blue subject",
            "source_rgba_path": str(environment.source.resolve()),
            "localized_result_path": str(environment.localized_result.resolve()),
        }
    ]
    environment.manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in selected),
        encoding="utf-8",
    )


@pytest.fixture
def environment(tmp_path: Path) -> _Environment:
    data_root = (tmp_path / "workspace" / "data").resolve()
    artifacts = data_root / "localized"
    artifacts.mkdir(parents=True)
    source = artifacts / "source.png"
    candidate = artifacts / "candidate.png"
    _source_rgba().save(source, format="PNG")
    _candidate().save(candidate, format="PNG")
    publication_root = data_root / "reference_completion_publication_benchmarks"
    value = _Environment(
        data_root=data_root,
        publication_root=publication_root,
        source=source,
        candidate=candidate,
        localized_result=artifacts / "result.json",
        manifest=data_root / "publication.jsonl",
        config=PublicationConfig(
            allowed_input_root=data_root,
            publication_root=publication_root,
        ),
    )
    _write_localized_result(value)
    _write_manifest(value)
    return value


def _run(
    environment: _Environment,
    *,
    segmenter: _Segmenter | None = None,
    identity: _Judge | None = None,
    locality: _Judge | None = None,
    usability: _Judge | None = None,
    run_id: str = "run-1",
) -> tuple[Path, object, tuple[_Judge, _Judge, _Judge]]:
    judges = (
        identity or _Judge(_identity_accept()),
        locality or _Judge(_locality_accept()),
        usability or _Judge(_usability_accept()),
    )
    root = environment.publication_root / run_id
    stats = run_reference_completion_publication(
        manifest_path=environment.manifest,
        benchmark_root=root,
        config=environment.config,
        segmenter=segmenter or _Segmenter(),
        identity_judge=judges[0],
        locality_judge=judges[1],
        usability_judge=judges[2],
    )
    return root, stats, judges


def _result(root: Path, sample_id: str = "sample-1") -> dict[str, object]:
    return json.loads((root / sample_id / "result.json").read_text(encoding="utf-8"))


def test_unanimous_accept_publishes_verified_rgba_and_summary(
    environment: _Environment,
) -> None:
    source_hash = _sha256(environment.source)
    candidate_hash = _sha256(environment.candidate)
    root, stats, judges = _run(environment)

    assert stats.to_dict() == {"processed": 1, "auto_published": 1, "rejected": 0}
    result = _result(root)
    assert result["status"] == "auto_published"
    assert result["candidate_reference_path"] == "candidate_reference.png"
    reference_path = root / "sample-1" / "candidate_reference.png"
    assert result["candidate_reference_sha256"] == _sha256(reference_path)
    with Image.open(reference_path) as reference:
        reference.load()
        pixels = np.asarray(reference)
        assert reference.mode == "RGBA"
        assert reference.size == (100, 100)
    assert set(np.unique(pixels[:, :, 3])) == {0, 255}
    assert np.all(pixels[:, :, :3][pixels[:, :, 3] == 0] == 255)
    assert _sha256(environment.source) == source_hash
    assert _sha256(environment.candidate) == candidate_hash
    assert all(len(judge.calls) == 1 for judge in judges)
    summary = json.loads((root / "publication_summary.json").read_text())
    assert summary == {
        "processed": 1,
        "auto_published": 1,
        "rejected": 0,
        "reference_type_counts": {"subject": 1},
        "rejection_reason_counts": {},
        "judge_rejection_counts": {"identity": 0, "locality": 0, "usability": 0},
    }
    assert not list(root.glob(".*.tmp-*"))
    assert "manual_review_pending" not in json.dumps(result)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("backend", "other", "backend"),
        ("mode", "whole_canvas", "localized_raw"),
        ("hard_passed", False, "hard-check failure"),
    ],
)
def test_batch_preflight_failure_creates_no_output_root(
    environment: _Environment,
    field: str,
    value: object,
    error: str,
) -> None:
    updates = {field: value}
    _write_localized_result(environment, **updates)
    root = environment.publication_root / "preflight-failure"
    with pytest.raises(ValueError, match=error):
        run_reference_completion_publication(
            manifest_path=environment.manifest,
            benchmark_root=root,
            config=environment.config,
            segmenter=_Segmenter(),
            identity_judge=_Judge(_identity_accept()),
            locality_judge=_Judge(_locality_accept()),
            usability_judge=_Judge(_usability_accept()),
        )
    assert not root.exists()


def test_source_and_candidate_hash_or_path_mismatch_fails_preflight(
    environment: _Environment,
) -> None:
    payload = json.loads(environment.localized_result.read_text())
    payload["candidate_sha256"] = "0" * 64
    environment.localized_result.write_text(json.dumps(payload))
    root = environment.publication_root / "bad-hash"
    with pytest.raises(ValueError, match="candidate SHA-256"):
        _run(environment, run_id="bad-hash")
    assert not root.exists()


@pytest.mark.parametrize(
    ("mask", "reason"),
    [
        (Image.new("L", (100, 100), 0), "mask_empty"),
        (Image.new("L", (100, 100), 255), "mask_full"),
    ],
)
def test_empty_and_full_masks_reject_without_running_judges(
    environment: _Environment,
    mask: Image.Image,
    reason: str,
) -> None:
    root, stats, judges = _run(environment, segmenter=_Segmenter(mask))
    result = _result(root)
    assert stats.rejected == 1
    assert result["status"] == "rejected"
    assert reason in result["rejection_reasons"]
    assert result["candidate_reference_path"] is None
    assert not (root / "sample-1" / "candidate_reference.png").exists()
    assert all(not judge.calls for judge in judges)


def test_secondary_component_rejects_subject_but_group_allows_members(
    environment: _Environment,
) -> None:
    strict_mask = np.zeros((100, 100), dtype=np.uint8)
    strict_mask[20:70, 10:60] = 255
    strict_mask[75:90, 75:90] = 255
    root, _, _ = _run(
        environment,
        segmenter=_Segmenter(Image.fromarray(strict_mask, mode="L")),
    )
    assert "secondary_component_ratio_above_max" in _result(root)[
        "rejection_reasons"
    ]

    _write_manifest(
        environment,
        [
            {
                "sample_id": "group-1",
                "clip_uid": "clip-group",
                "entity_id": "e1",
                "reference_type": "group",
                "entity_phrase": "the same group",
                "source_rgba_path": str(environment.source),
                "localized_result_path": str(environment.localized_result),
            }
        ],
    )
    group_root, group_stats, _ = _run(
        environment,
        segmenter=_Segmenter(_group_mask()),
        run_id="group-run",
    )
    assert group_stats.auto_published == 1
    assert _result(group_root, "group-1")["status"] == "auto_published"


def test_far_significant_group_component_rejects(
    environment: _Environment,
) -> None:
    _write_manifest(
        environment,
        [
            {
                "sample_id": "group-far",
                "clip_uid": "clip-group",
                "entity_id": "e1",
                "reference_type": "group",
                "entity_phrase": "the same group",
                "source_rgba_path": str(environment.source),
                "localized_result_path": str(environment.localized_result),
            }
        ],
    )
    root, _, _ = _run(
        environment,
        segmenter=_Segmenter(_group_mask(isolated=True)),
    )
    result = _result(root, "group-far")
    assert "group_significant_component_isolated" in result["rejection_reasons"]


def test_object_uses_strict_single_entity_gate(environment: _Environment) -> None:
    _write_manifest(
        environment,
        [
            {
                "sample_id": "object-1",
                "clip_uid": "clip-object",
                "entity_id": "e1",
                "reference_type": "object",
                "entity_phrase": "the blue object",
                "source_rgba_path": str(environment.source),
                "localized_result_path": str(environment.localized_result),
            }
        ],
    )
    root, stats, _ = _run(environment)
    assert stats.auto_published == 1
    assert _result(root, "object-1")["reference_type"] == "object"


def test_metrics_are_normalized_and_edge_improvement_is_detected() -> None:
    source = Image.new("L", (10, 20), 0)
    source_array = np.asarray(source).copy()
    source_array[5:15, 0:4] = 255
    source = Image.fromarray(source_array, mode="L")
    candidate = Image.new("L", (20, 40), 0)
    candidate_array = np.asarray(candidate).copy()
    candidate_array[8:32, 2:12] = 255
    candidate = Image.fromarray(candidate_array, mode="L")
    source_metrics = compute_foreground_metrics(source)
    candidate_metrics = compute_foreground_metrics(candidate)
    report, passed, reasons = evaluate_improvement_gate(
        source_metrics,
        candidate_metrics,
        reference_type="subject",
        config=PublicationConfig(),
    )
    assert source_metrics["foreground_area_ratio"] == 0.2
    assert candidate_metrics["foreground_area_ratio"] == 0.3
    assert report["improvements"]["edge_touches_reduced"] is True
    assert passed is True
    assert reasons == []


@pytest.mark.parametrize(
    ("bounds", "reason"),
    [
        ((25, 75, 20, 40), "candidate_area_ratio_below_min"),
        ((5, 95, 5, 85), "candidate_area_ratio_above_max"),
    ],
)
def test_abnormal_area_shrink_and_growth_reject(
    environment: _Environment,
    bounds: tuple[int, int, int, int],
    reason: str,
) -> None:
    top, bottom, left, right = bounds
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[top:bottom, left:right] = 255
    root, _, _ = _run(
        environment,
        segmenter=_Segmenter(Image.fromarray(pixels, mode="L")),
    )
    assert reason in _result(root)["rejection_reasons"]


def test_background_gate_allows_white_shadow_and_rejects_complex() -> None:
    mask = _good_mask()
    simple = np.asarray(_candidate(mask)).copy()
    simple[82:88, 20:45] = 230
    metrics, passed, reasons = evaluate_background_gate(
        Image.fromarray(simple, mode="RGB"),
        mask,
        config=PublicationConfig(),
    )
    assert passed is True
    assert reasons == []
    assert metrics["outside_near_white_ratio"] >= 0.8

    rng = np.random.default_rng(7)
    complex_pixels = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    complex_pixels[np.asarray(mask) == 255] = (80, 90, 100)
    _, passed, reasons = evaluate_background_gate(
        Image.fromarray(complex_pixels, mode="RGB"),
        mask,
        config=PublicationConfig(),
    )
    assert passed is False
    assert reasons


@pytest.mark.parametrize("kind", ["identity", "locality", "usability"])
@pytest.mark.parametrize("failure_mode", ["reject", "raise"])
def test_any_judge_reject_or_error_fails_closed(
    environment: _Environment,
    kind: str,
    failure_mode: str,
) -> None:
    accepts = {
        "identity": _identity_accept(),
        "locality": _locality_accept(),
        "usability": _usability_accept(),
    }
    response: object = (
        RuntimeError("service unavailable")
        if failure_mode == "raise"
        else _reject(accepts[kind])
    )
    judges = {name: _Judge(response if name == kind else review) for name, review in accepts.items()}
    root, _, _ = _run(
        environment,
        identity=judges["identity"],
        locality=judges["locality"],
        usability=judges["usability"],
    )
    result = _result(root)
    expected = f"{kind}_judge_{'failed' if failure_mode == 'raise' else 'rejected'}"
    assert result["status"] == "rejected"
    assert result["reason"] == expected
    assert result["candidate_reference_path"] is None
    assert all(len(judge.calls) == 1 for judge in judges.values())
    summary = json.loads((root / "publication_summary.json").read_text())
    assert summary["judge_rejection_counts"][kind] == 1


def test_binary_mask_ranking_is_deterministic() -> None:
    low = Image.new("L", (20, 20), 0)
    low_pixels = np.asarray(low).copy()
    low_pixels[2:18, 2:18] = 255
    low = Image.fromarray(low_pixels, mode="L")
    high_small = Image.new("L", (20, 20), 0)
    high_small_pixels = np.asarray(high_small).copy()
    high_small_pixels[5:10, 5:10] = 255
    high_small = Image.fromarray(high_small_pixels, mode="L")
    selected, provenance = select_segmented_mask(
        [
            SegmentationMaskCandidate(low, 0.8, "large-low"),
            SegmentationMaskCandidate(high_small, 0.9, "small-high"),
            SegmentationMaskCandidate(high_small, 0.9, "tie-later"),
        ],
        expected_size=(20, 20),
    )
    assert selected.getbbox() == high_small.getbbox()
    assert provenance["selected_original_index"] == 1
    assert provenance["selected_object_id"] == "small-high"


def test_nonbinary_segmentation_fails_closed(environment: _Environment) -> None:
    invalid = Image.new("L", (100, 100), 128)
    root, _, judges = _run(environment, segmenter=_Segmenter(invalid))
    result = _result(root)
    assert result["reason"] == "segmentation_failed"
    assert result["candidate_reference_path"] is None
    assert all(not judge.calls for judge in judges)


def test_build_candidate_reference_never_resizes() -> None:
    rgb = Image.new("RGB", (37, 23), (10, 20, 30))
    mask = Image.new("L", (37, 23), 0)
    mask_pixels = np.asarray(mask).copy()
    mask_pixels[4:18, 7:29] = 255
    mask = Image.fromarray(mask_pixels, mode="L")
    reference = build_candidate_reference(rgb, mask)
    assert reference.size == (37, 23)
    assert reference.mode == "RGBA"
    pixels = np.asarray(reference)
    assert np.all(pixels[:, :, :3][pixels[:, :, 3] == 0] == 255)


def test_failure_continues_transactionally_for_remaining_samples(
    environment: _Environment,
) -> None:
    second_result = _write_localized_result(
        environment,
        path=environment.localized_result.parent / "result-2.json",
    )
    records = [
        {
            "sample_id": "a-fails",
            "clip_uid": "clip-a",
            "entity_id": "e1",
            "reference_type": "subject",
            "entity_phrase": "fail segmentation",
            "source_rgba_path": str(environment.source),
            "localized_result_path": str(environment.localized_result),
        },
        {
            "sample_id": "b-passes",
            "clip_uid": "clip-b",
            "entity_id": "e1",
            "reference_type": "subject",
            "entity_phrase": "pass segmentation",
            "source_rgba_path": str(environment.source),
            "localized_result_path": str(second_result),
        },
    ]
    _write_manifest(environment, records)

    def response(kwargs: dict[str, object]) -> object:
        if kwargs["entity_phrase"] == "fail segmentation":
            return RuntimeError("fake SAM3 failure")
        return _good_mask()

    root, stats, _ = _run(environment, segmenter=_Segmenter(response))
    assert stats.to_dict() == {"processed": 2, "auto_published": 1, "rejected": 1}
    assert _result(root, "a-fails")["status"] == "rejected"
    assert _result(root, "b-passes")["status"] == "auto_published"
    assert not list(root.glob(".*.tmp-*"))


def test_review_schemas_are_strict_and_uncertainty_cannot_accept() -> None:
    payload = _identity_accept().model_dump()
    payload["certain"] = False
    with pytest.raises(ValidationError, match="verdict"):
        IdentityReview.model_validate(payload)
    payload = _identity_accept().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        IdentityReview.model_validate(payload)


def test_fake_sam3_adapter_uses_local_checkpoint_and_phrase(tmp_path: Path) -> None:
    code_root = tmp_path / "sam3"
    code_root.mkdir()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"fake")
    calls: dict[str, object] = {}

    def builder(*, checkpoint_path: str, device: str) -> object:
        calls["builder"] = (checkpoint_path, device)
        return object()

    class Processor:
        def set_image(self, image: Image.Image) -> str:
            calls["image"] = image.copy()
            return "state"

        def set_text_prompt(self, *, state: str, prompt: str) -> dict[str, object]:
            calls["prompt"] = (state, prompt)
            return {
                "masks": np.ones((1, 10, 12), dtype=bool),
                "scores": np.asarray([0.75]),
                "object_ids": np.asarray([7]),
            }

    segmenter = Sam3LocalizedCompletionSegmenter(
        code_root=code_root,
        checkpoint_path=checkpoint,
        device="cpu",
        builder=builder,
        processor_factory=lambda model: Processor(),
    )
    results = segmenter.segment(
        candidate_rgb=Image.new("RGB", (12, 10), "white"),
        entity_phrase="the target object",
        reference_type="object",
    )
    assert calls["builder"] == (str(checkpoint.resolve()), "cpu")
    assert calls["prompt"] == ("state", "the target object")
    assert len(results) == 1
    assert results[0].object_id == "7"


def test_output_must_stay_in_dedicated_publication_root(
    environment: _Environment,
) -> None:
    forbidden = environment.data_root / "r2v_v3_runs" / "bad"
    with pytest.raises(ValueError, match="strictly below"):
        run_reference_completion_publication(
            manifest_path=environment.manifest,
            benchmark_root=forbidden,
            config=environment.config,
            segmenter=_Segmenter(),
            identity_judge=_Judge(_identity_accept()),
            locality_judge=_Judge(_locality_accept()),
            usability_judge=_Judge(_usability_accept()),
        )
    assert not forbidden.exists()


def test_mask_gate_records_configured_thresholds() -> None:
    config = PublicationConfig(
        largest_component_ratio=0.92,
        max_secondary_component_ratio=0.03,
    )
    metrics, passed, reasons = evaluate_mask_gate(
        _good_mask(),
        reference_type="subject",
        config=config,
    )
    assert passed is True
    assert reasons == []
    assert metrics["thresholds"]["largest_component_ratio"] == 0.92
    assert metrics["thresholds"]["max_secondary_component_ratio"] == 0.03
