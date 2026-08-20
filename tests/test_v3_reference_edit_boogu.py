from __future__ import annotations

import importlib
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.reference_edit_boogu as boogu_module
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguBackgroundReview,
    BooguCompletionReview,
    BooguEditOutput,
    BooguSamReview,
    BooguSubprocessBackend,
    BooguWorkerConfig,
    QwenBooguReferenceEditJudge,
    Sam3BooguReferenceReviewer,
    resolve_boogu_1k_size,
    run_boogu_reference_edit,
)
from r2v_data_v2.v3.reference_geometry import (
    content_geometry_from_rgba,
    tiny_content_reason,
)
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
)


def _png_bytes(
    size: tuple[int, int],
    *,
    color: tuple[int, ...] = (27, 44, 91),
    mode: str = "RGB",
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _completion_review(*, accept: bool = True) -> BooguCompletionReview:
    values = {
        "same_physical_entity": accept,
        "identity_preserved": accept,
        "original_visible_attributes_preserved": accept,
        "exactly_one_entity": accept,
        "missing_parts_plausibly_completed": accept,
        "no_duplicate_entity": accept,
        "no_unrelated_entity": accept,
        "no_severe_structure_artifact": accept,
        "style_coherent": accept,
        "resolution_usable": accept,
        "reference_usable": accept,
        "certain": accept,
    }
    return BooguCompletionReview(
        verdict="accept" if accept else "reject",
        reason="usable" if accept else "identity drift",
        **values,
    )


def _background_review(
    *,
    accept: bool = True,
    updates: dict[str, bool] | None = None,
) -> BooguBackgroundReview:
    values = {
        "exactly_one_target_entity": accept,
        "identity_preserved": accept,
        "entity_appearance_consistent": accept,
        "no_duplicate_entity": accept,
        "no_added_salient_entity": accept,
        "no_unintended_completion_or_extension": accept,
        "subject_scale_preserved": accept,
        "subject_layout_preserved": accept,
        "background_coherent": accept,
        "background_improves_reference": accept,
        "prefer_candidate_over_source": accept,
        "background_style_consistent": accept,
        "no_halo_or_seam": accept,
        "subject_not_severely_redrawn": accept,
        "reference_usable": accept,
        "certain": accept,
    }
    values.update(updates or {})
    accepted = all(values.values())
    return BooguBackgroundReview(
        verdict="accept" if accepted else "reject",
        reason="usable" if accepted else "candidate is not preferable",
        **values,
    )


class _Backend:
    def __init__(
        self,
        *,
        returned_size: tuple[int, int] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.returned_size = returned_size
        self.events = events
        self.calls: list[dict[str, object]] = []
        self.output_bytes: bytes | None = None

    def edit(self, **kwargs: object) -> BooguEditOutput:
        if self.events is not None:
            self.events.append("boogu")
        self.calls.append(kwargs)
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        output_size = self.returned_size or (width, height)
        self.output_bytes = _png_bytes(output_size, color=(191, 22, 43))
        thinking = bool(kwargs["thinking_enabled"])
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=self.output_bytes,
            original_instruction=instruction,
            rewritten_instruction="rewritten" if thinking else None,
            effective_instruction="rewritten" if thinking else instruction,
            worker_metadata={"returned_size": list(output_size)},
        )


class _Judge:
    def __init__(
        self,
        *,
        accept: bool = True,
        background_updates: dict[str, bool] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.accept = accept
        self.background_updates = background_updates
        self.events = events
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> BooguCompletionReview | BooguBackgroundReview:
        if self.events is not None:
            self.events.append("qwen")
        self.calls.append(kwargs)
        if kwargs["operation"] == "complete_entity":
            return _completion_review(accept=self.accept)
        return _background_review(
            accept=self.accept,
            updates=self.background_updates,
        )


class _SamReviewer:
    def __init__(
        self,
        *,
        passed: bool = True,
        failure_kind: str | None = None,
        geometry: dict[str, object] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.failure_kind = failure_kind or ("none" if passed else "fragmented")
        self.geometry = geometry or {}
        self.events = events
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> BooguSamReview:
        if self.events is not None:
            self.events.append("sam")
        self.calls.append(kwargs)
        kind = self.failure_kind
        target_present = kind not in {"not_found", "backend_failure"}
        return BooguSamReview(
            passed=kind == "none",
            target_entity_present=target_present,
            exactly_one_target_instance=(
                target_present and kind != "multiple_instances"
            ),
            area_growth_acceptable=(
                target_present and kind != "excessive_area_growth"
            ),
            fragmentation_acceptable=(target_present and kind != "fragmented"),
            reason="valid" if kind == "none" else kind,
            diagnostics={
                "mask_path": "review-only.png",
                "failure_kind": kind,
                **self.geometry,
            },
        )


class _FailIfCalledJudge:
    def review(self, **kwargs: object) -> BooguCompletionReview | BooguBackgroundReview:
        raise AssertionError(f"Qwen review must be skipped: {kwargs['operation']}")


def _environment(
    tmp_path: Path,
    *,
    size: tuple[int, int] = (192, 128),
) -> tuple[Path, Path, bytes]:
    run_root = tmp_path / "run"
    canonical = run_root / "clips" / "clip-1" / "selected" / "e1.png"
    canonical.parent.mkdir(parents=True)
    image = Image.new("RGBA", size, (20, 180, 70, 255))
    image.putpixel((0, 0), (10, 20, 30, 0))
    image.save(canonical, format="PNG")
    payload = canonical.read_bytes()
    return run_root, canonical, payload


def _geometry_diagnostics(
    reason: str | None,
    *,
    scale_ratio: float = 1.0,
    center_shift: float = 0.0,
) -> dict[str, object]:
    return {
        "source_normalized_bbox": [0.0, 0.0, 1.0, 1.0],
        "candidate_normalized_bbox": [0.0, 0.0, 1.0, 1.0],
        "source_normalized_bbox_area": 1.0,
        "candidate_normalized_bbox_area": scale_ratio,
        "candidate_scale_ratio": scale_ratio,
        "source_center_xy": [0.5, 0.5],
        "candidate_center_xy": [0.5, 0.5],
        "center_shift": center_shift,
        "geometry_gate_passed": reason is None,
        "geometry_rejection_reason": reason,
    }


@pytest.mark.parametrize(
    ("source_size", "expected"),
    [
        ((100, 100), (1024, 1024)),
        ((160, 90), (1360, 768)),
        ((90, 160), (768, 1360)),
        ((150, 100), (1248, 832)),
    ],
)
def test_resolve_boogu_1k_size_preserves_ratio_and_alignment(
    source_size: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    resolved = resolve_boogu_1k_size(*source_size)

    assert resolved == expected
    assert resolved[0] % 16 == 0
    assert resolved[1] % 16 == 0
    assert abs(resolved[0] * resolved[1] - 1024 * 1024) / (1024 * 1024) < 0.02
    source_ratio = source_size[0] / source_size[1]
    assert abs(resolved[0] / resolved[1] - source_ratio) / source_ratio < 0.01


@pytest.mark.parametrize(
    "values",
    [(0, 1, 1024, 16), (1, -1, 1024, 16), (1, 1, 0, 16), (1, 1, 1024, 0)],
)
def test_resolve_boogu_1k_size_rejects_invalid_values(
    values: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_boogu_1k_size(
            values[0],
            values[1],
            target_area=values[2],
            alignment=values[3],
        )


def test_completion_publishes_native_1k_output_without_paste_back(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)
    backend = _Backend()
    judge = _Judge()
    sam = _SamReviewer()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete the same entity.",
        entity_phrase="green object",
        reference_type="object",
        backend=backend,
        judge=judge,
        sam_reviewer=sam,
    )

    assert result.status == "accepted"
    assert backend.calls[0]["thinking_enabled"] is True
    assert "alpha_mask" not in backend.calls[0]
    assert "mask" not in backend.calls[0]
    assert canonical.read_bytes() == canonical_bytes
    assert result.candidate_path is not None
    assert result.final_reference_path is not None
    assert result.candidate_path.read_bytes() == backend.output_bytes
    assert result.final_reference_path.read_bytes() == backend.output_bytes
    with Image.open(result.final_reference_path) as final:
        assert final.mode == "RGB"
        assert final.size == (1248, 832)
        assert final.getpixel((10, 10)) == (191, 22, 43)
    assert sam.calls
    assert "candidate_rgb" in sam.calls[0]


def test_tiny_source_fails_before_backend_generation(tmp_path: Path) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path, size=(64, 64))
    backend = _Backend()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="small person",
        reference_type="subject",
        backend=backend,
        judge=_Judge(),
    )

    assert result.status == "rejected"
    assert backend.calls == []
    assert canonical.read_bytes() == canonical_bytes
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_content_area_pixels"] == 64 * 64
    assert metadata["source_tiny"] is True
    assert metadata["source_gate_reason"] != "passed"
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert rejection["reason"] == "tiny_source_entity"


def test_long_thin_source_is_not_rejected_by_short_side_alone() -> None:
    source = Image.new("RGBA", (16, 2048), (20, 30, 40, 255))
    geometry = content_geometry_from_rgba(source)

    assert geometry.area_pixels == 16 * 2048
    assert (
        tiny_content_reason(
            geometry,
            minimum_area_pixels=128 * 128,
            minimum_long_side_pixels=128,
        )
        is None
    )


def test_sam_review_uses_entity_phrase_instead_of_scene_grounding_prompt(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    sam = _SamReviewer()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete the same entity.",
        entity_phrase="ornate wooden panel",
        grounding_prompt="ornate wooden panel behind a seated man",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(),
        sam_reviewer=sam,
    )

    assert result.status == "accepted"
    assert sam.calls[0]["entity_phrase"] == "ornate wooden panel"
    assert (
        sam.calls[0]["entity_phrase"]
        != "ornate wooden panel behind a seated man"
    )


def test_white_rgb_input_preserves_opaque_pixels_and_whitens_alpha(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    backend = _Backend()

    run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete the entity.",
        entity_phrase="object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
    )

    source_rgb = backend.calls[0]["source_rgb"]
    assert isinstance(source_rgb, Image.Image)
    assert source_rgb.mode == "RGB"
    assert source_rgb.getpixel((0, 0)) == (255, 255, 255)
    assert source_rgb.getpixel((1, 1)) == (20, 180, 70)


def test_background_uses_thinking_false_and_no_source_restoration(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)
    backend = _Backend()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a quiet studio background.",
        entity_phrase="green object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
    )

    assert result.status == "accepted"
    assert backend.calls[0]["thinking_enabled"] is False
    assert canonical.read_bytes() == canonical_bytes
    assert result.final_reference_path is not None
    assert result.final_reference_path.read_bytes() == backend.output_bytes


@pytest.mark.parametrize(
    ("reason", "scale_ratio", "center_shift"),
    [
        ("entity_scale_collapsed", 0.4, 0.0),
        ("entity_shifted_off_layout", 1.0, 0.35),
    ],
)
def test_composition_geometry_failure_rejects_candidate_publication(
    tmp_path: Path,
    reason: str,
    scale_ratio: float,
    center_shift: float,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    sam = _SamReviewer(
        geometry=_geometry_diagnostics(
            reason,
            scale_ratio=scale_ratio,
            center_shift=center_shift,
        )
    )

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_FailIfCalledJudge(),
        sam_reviewer=sam,
    )

    assert result.status == "rejected"
    assert result.final_reference_path is None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["geometry_gate_passed"] is False
    assert metadata["geometry_rejection_reason"] == reason
    assert metadata["qwen_review"] is None
    assert metadata["qwen_review_skipped_reason"] == (
        f"geometry_hard_failure:{reason}"
    )
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert rejection["reason"] == reason


def test_small_composition_change_passes_geometry_gate(tmp_path: Path) -> None:
    run_root, _, _ = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(),
        sam_reviewer=_SamReviewer(
            geometry=_geometry_diagnostics(None, scale_ratio=0.9, center_shift=0.1)
        ),
    )

    assert result.status == "accepted"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["geometry_gate_passed"] is True


@pytest.mark.parametrize(
    "background_updates",
    [
        {"background_improves_reference": False},
        {"prefer_candidate_over_source": False},
    ],
)
def test_background_must_be_beneficial_and_preferred_over_source(
    tmp_path: Path,
    background_updates: dict[str, bool],
) -> None:
    run_root, _, _ = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(background_updates=background_updates),
        sam_reviewer=_SamReviewer(),
    )

    assert result.status == "rejected"
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert rejection["reason"] == "background_not_beneficial"


def test_background_review_prompt_requires_source_comparison() -> None:
    prompt = boogu_module._BACKGROUND_REVIEW_PROMPT.lower()
    assert "scale" in prompt
    assert "layout" in prompt
    assert "large meaningless" in prompt
    assert "clearly preferable" in prompt


def test_wrong_native_output_size_fails_closed_and_preserves_canonical(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(returned_size=(640, 640)),
        judge=_Judge(),
    )

    assert result.status == "rejected"
    assert result.final_reference_path is None
    assert canonical.read_bytes() == canonical_bytes
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert "returned_size=(640, 640)" in rejection["reason"]


def test_rejected_candidate_uses_legacy_canonical_fallback(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(accept=False),
        fallback_status="canonical_local_fallback",
    )

    assert result.status == "rejected"
    assert result.fallback_status == "canonical_local_fallback"
    assert result.candidate_path is not None
    assert result.final_reference_path is None
    assert canonical.read_bytes() == canonical_bytes
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["fallback_status"] == "canonical_local_fallback"


def test_failed_sam_review_rejects_but_never_changes_candidate_pixels(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    backend = _Backend()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
        sam_reviewer=_SamReviewer(passed=False),
    )

    assert result.status == "rejected"
    assert result.candidate_path is not None
    assert result.candidate_path.read_bytes() == backend.output_bytes
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sam_mask_usage"] == "review_only"


def test_background_qwen_accept_and_sam_not_found_accepts_with_warning(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    judge = _Judge()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=judge,
        sam_reviewer=_SamReviewer(failure_kind="not_found"),
    )

    assert result.status == "accepted"
    assert len(judge.calls) == 1
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sam_review"]["diagnostics"]["failure_kind"] == "not_found"
    assert metadata["sam_warning"] == "target_not_found"
    assert metadata["qwen_review_skipped_reason"] is None


def test_background_qwen_reject_and_sam_not_found_rejects(tmp_path: Path) -> None:
    run_root, _, _ = _environment(tmp_path)
    judge = _Judge(accept=False)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=judge,
        sam_reviewer=_SamReviewer(failure_kind="not_found"),
    )

    assert result.status == "rejected"
    assert len(judge.calls) == 1
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sam_warning"] is None
    assert metadata["qwen_review_skipped_reason"] is None


def test_completion_qwen_accept_and_sam_not_found_rejects(tmp_path: Path) -> None:
    run_root, _, _ = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(),
        sam_reviewer=_SamReviewer(failure_kind="not_found"),
    )

    assert result.status == "rejected"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sam_warning"] is None


@pytest.mark.parametrize(
    "failure_kind",
    [
        "multiple_instances",
        "excessive_area_growth",
        "fragmented",
        "backend_failure",
    ],
)
def test_background_sam_hard_failure_rejects_without_qwen(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    events: list[str] = []

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(events=events),
        judge=_FailIfCalledJudge(),
        sam_reviewer=_SamReviewer(failure_kind=failure_kind, events=events),
    )

    assert result.status == "rejected"
    assert result.candidate_path is not None
    assert events == ["boogu", "sam"]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["qwen_review"] is None
    assert metadata["qwen_review_skipped_reason"] == (
        f"sam_hard_failure:{failure_kind}"
    )
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert rejection["reason"] == f"sam_hard_failure:{failure_kind}"


def test_qwen_reject_overrides_passing_sam_review(tmp_path: Path) -> None:
    run_root, _, _ = _environment(tmp_path)
    judge = _Judge(accept=False)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=judge,
        sam_reviewer=_SamReviewer(),
    )

    assert result.status == "rejected"
    assert len(judge.calls) == 1


def test_boogu_review_order_is_operation_specific(tmp_path: Path) -> None:
    background_events: list[str] = []
    background_root, _, _ = _environment(tmp_path / "background")
    background = run_boogu_reference_edit(
        run_root=background_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(events=background_events),
        judge=_Judge(events=background_events),
        sam_reviewer=_SamReviewer(events=background_events),
    )

    completion_events: list[str] = []
    completion_root, _, _ = _environment(tmp_path / "completion")
    completion = run_boogu_reference_edit(
        run_root=completion_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(events=completion_events),
        judge=_Judge(events=completion_events),
        sam_reviewer=_SamReviewer(events=completion_events),
    )

    assert background.status == "accepted"
    assert background_events == ["boogu", "sam", "qwen"]
    assert completion.status == "accepted"
    assert completion_events == ["boogu", "qwen", "sam"]


def test_metadata_records_source_output_dimensions_and_provenance(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_dimensions"] == {"width": 192, "height": 128}
    assert metadata["resolved_output_dimensions"] == {"width": 1248, "height": 832}
    assert metadata["target_area"] == 1024 * 1024
    assert metadata["alignment"] == 16
    assert metadata["output_pixel_count"] == 1248 * 832
    assert metadata["canonical_source_sha256"] == boogu_module._sha256_bytes(
        canonical_bytes
    )
    assert metadata["output_sha256"] == boogu_module._sha256_bytes(
        result.candidate_path.read_bytes()
    )
    assert canonical.read_bytes() == canonical_bytes


def test_boogu_operations_and_sam_reviews_are_profiled_separately(
    tmp_path: Path,
) -> None:
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")
    complete_root, _, _ = _environment(tmp_path / "complete")
    background_root, _, _ = _environment(tmp_path / "background")

    with active_profiler(profiler):
        complete = run_boogu_reference_edit(
            run_root=complete_root,
            clip_uid="clip-1",
            entity_id="e1",
            operation="complete_entity",
            instruction="Complete it.",
            entity_phrase="object",
            reference_type="object",
            backend=_Backend(),
            judge=_Judge(),
            sam_reviewer=_SamReviewer(),
        )
        background = run_boogu_reference_edit(
            run_root=background_root,
            clip_uid="clip-1",
            entity_id="e1",
            operation="add_entity_background",
            instruction="Add a background.",
            entity_phrase="object",
            reference_type="object",
            backend=_Backend(),
            judge=_Judge(),
            sam_reviewer=_SamReviewer(),
        )

    assert complete.status == "accepted"
    assert background.status == "accepted"
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["component"] for event in events] == [
        "boogu_complete_entity",
        "sam3_boogu_review",
        "boogu_add_entity_background",
        "sam3_boogu_review",
    ]
    assert [
        event["operation"]
        for event in events
        if event["component"] == "sam3_boogu_review"
    ] == ["complete_entity", "add_entity_background"]
    assert events[0]["metadata"]["thinking_enabled"] is True
    assert events[0]["metadata"]["width"] == 1248
    assert events[0]["metadata"]["height"] == 832


def test_background_hard_gate_omits_only_qwen_profile_event(tmp_path: Path) -> None:
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")
    normal_root, _, _ = _environment(tmp_path / "normal")
    hard_failure_root, _, _ = _environment(tmp_path / "hard-failure")
    completions = _ReviewCompletions(
        _background_review().model_dump(mode="json")
    )
    judge = QwenBooguReferenceEditJudge(
        QwenServiceConfig(model="/models/qwen"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            close=lambda: None,
        ),
    )

    with active_profiler(profiler):
        normal = run_boogu_reference_edit(
            run_root=normal_root,
            clip_uid="clip-1",
            entity_id="e1",
            operation="add_entity_background",
            instruction="Add a background.",
            entity_phrase="object",
            reference_type="object",
            backend=_Backend(),
            judge=judge,
            sam_reviewer=_SamReviewer(),
        )
        hard_failure = run_boogu_reference_edit(
            run_root=hard_failure_root,
            clip_uid="clip-1",
            entity_id="e1",
            operation="add_entity_background",
            instruction="Add a background.",
            entity_phrase="object",
            reference_type="object",
            backend=_Backend(),
            judge=_FailIfCalledJudge(),
            sam_reviewer=_SamReviewer(failure_kind="fragmented"),
        )

    assert normal.status == "accepted"
    assert hard_failure.status == "rejected"
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["component"] for event in events] == [
        "boogu_add_entity_background",
        "sam3_boogu_review",
        "qwen_boogu_background_review",
        "boogu_add_entity_background",
        "sam3_boogu_review",
    ]


def test_subprocess_backend_invokes_configured_python_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = Path(sys.executable).resolve()
    code_root = tmp_path / "vendor" / "Boogu-Image"
    model = tmp_path / "models" / "boogu"
    worker = tmp_path / "repo" / "worker.py"
    code_root.mkdir(parents=True)
    model.mkdir(parents=True)
    worker.parent.mkdir(parents=True)
    events_path = tmp_path / "events.jsonl"
    worker.write_text(
        """import json
import os
import sys
from pathlib import Path
from PIL import Image

events = Path(os.environ["FAKE_BOOGU_EVENTS"])
def emit(value):
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\\n")

emit({"type": "startup", "argv": sys.argv, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")})
print(json.dumps({"schema_version": 1, "type": "ready", "status": "ok"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    emit(request)
    if request["type"] == "shutdown":
        print(json.dumps({"schema_version": 1, "type": "shutdown", "request_id": request["request_id"], "status": "ok"}), flush=True)
        break
    Image.new("RGB", (request["width"], request["height"]), (1, 2, 3)).save(request["output_image_path"], format="PNG")
    print(json.dumps({
        "schema_version": 1,
        "type": "response",
        "request_id": request["request_id"],
        "status": "ok",
        "original_instruction": request["instruction"],
        "rewritten_instruction": None,
        "effective_instruction": request["instruction"],
        "returned_size": [request["width"], request["height"]],
    }), flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_BOOGU_EVENTS", str(events_path))
    backend = BooguSubprocessBackend(
        BooguWorkerConfig(
            python_executable=python,
            code_root=code_root.resolve(),
            model_path=model.resolve(),
            cuda_visible_devices="3",
            allowed_server_root=Path("/"),
            temporary_root=(tmp_path / "temporary").resolve(),
            worker_script=worker.resolve(),
        )
    )

    backend.start(stderr_log_path=tmp_path / "worker.stderr.log")
    first = backend.edit(
        source_rgb=Image.new("RGB", (160, 90)),
        instruction="Add a background.",
        width=1360,
        height=768,
        thinking_enabled=False,
    )
    second = backend.edit(
        source_rgb=Image.new("RGB", (160, 90)),
        instruction="Complete the entity.",
        width=1360,
        height=768,
        thinking_enabled=True,
    )
    seeded = backend.edit(
        source_rgb=Image.new("RGB", (160, 90)),
        instruction="Remove an entity.",
        width=1360,
        height=768,
        thinking_enabled=False,
        seed=17,
    )
    backend.close()

    assert first.effective_instruction == "Add a background."
    assert second.effective_instruction == "Complete the entity."
    assert seeded.effective_instruction == "Remove an entity."
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "startup",
        "edit",
        "edit",
        "edit",
        "shutdown",
    ]
    assert events[0]["argv"][:2] == [str(worker.resolve()), "--serve"]
    assert events[0]["cuda"] == "3"
    assert events[1]["request_id"] != events[2]["request_id"]
    assert "seed" not in events[1]
    assert "seed" not in events[2]
    assert events[3]["seed"] == 17


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("invalid_json", "invalid JSON"),
        ("exit", "exited without a response"),
        ("timeout", "timed out"),
    ],
)
def test_jsonl_worker_protocol_failures_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    match: str,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json
import os
import sys
import time

print(json.dumps({"schema_version": 1, "type": "ready", "status": "ok"}), flush=True)
for line in sys.stdin:
    json.loads(line)
    mode = os.environ["FAKE_PROTOCOL_MODE"]
    if mode == "invalid_json":
        print("not-json", flush=True)
        time.sleep(10)
    elif mode == "exit":
        raise SystemExit(3)
    else:
        time.sleep(10)
""",
        encoding="utf-8",
    )
    code_root = tmp_path / "code"
    model_path = tmp_path / "model"
    code_root.mkdir()
    model_path.mkdir()
    monkeypatch.setenv("FAKE_PROTOCOL_MODE", mode)
    backend = BooguSubprocessBackend(
        BooguWorkerConfig(
            python_executable=Path(sys.executable).resolve(),
            code_root=code_root.resolve(),
            model_path=model_path.resolve(),
            worker_script=worker.resolve(),
            allowed_server_root=Path("/"),
            temporary_root=(tmp_path / "temporary").resolve(),
            timeout_seconds=1,
        )
    )
    backend.start(stderr_log_path=tmp_path / "stderr.log")

    with pytest.raises((RuntimeError, TimeoutError), match=match):
        backend.edit(
            source_rgb=Image.new("RGB", (8, 8)),
            instruction="Edit the entity.",
            width=32,
            height=32,
            thinking_enabled=False,
        )

    assert backend.started is False


def test_single_generation_error_does_not_terminate_jsonl_worker(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json
import sys
from PIL import Image

print(json.dumps({"schema_version": 1, "type": "ready", "status": "ok"}), flush=True)
edit_count = 0
for line in sys.stdin:
    request = json.loads(line)
    if request["type"] == "shutdown":
        print(json.dumps({"schema_version": 1, "type": "shutdown", "request_id": request["request_id"], "status": "ok"}), flush=True)
        break
    edit_count += 1
    if edit_count == 1:
        print(json.dumps({"schema_version": 1, "type": "response", "request_id": request["request_id"], "status": "error", "reason": "generation failed"}), flush=True)
        continue
    Image.new("RGB", (request["width"], request["height"]), (1, 2, 3)).save(request["output_image_path"], format="PNG")
    print(json.dumps({
        "schema_version": 1,
        "type": "response",
        "request_id": request["request_id"],
        "status": "ok",
        "original_instruction": request["instruction"],
        "rewritten_instruction": None,
        "effective_instruction": request["instruction"],
        "returned_size": [request["width"], request["height"]],
    }), flush=True)
""",
        encoding="utf-8",
    )
    code_root = tmp_path / "code"
    model_path = tmp_path / "model"
    code_root.mkdir()
    model_path.mkdir()
    backend = BooguSubprocessBackend(
        BooguWorkerConfig(
            python_executable=Path(sys.executable).resolve(),
            code_root=code_root.resolve(),
            model_path=model_path.resolve(),
            worker_script=worker.resolve(),
            allowed_server_root=Path("/"),
            temporary_root=(tmp_path / "temporary").resolve(),
        )
    )
    backend.start(stderr_log_path=tmp_path / "stderr.log")

    with pytest.raises(RuntimeError, match="generation failed"):
        backend.edit(
            source_rgb=Image.new("RGB", (8, 8)),
            instruction="First edit.",
            width=32,
            height=32,
            thinking_enabled=False,
        )
    assert backend.started is True

    output = backend.edit(
        source_rgb=Image.new("RGB", (8, 8)),
        instruction="Second edit.",
        width=32,
        height=32,
        thinking_enabled=False,
    )
    backend.close()

    assert output.effective_instruction == "Second edit."
    assert backend.started is False


def test_worker_module_import_does_not_import_torch_or_boogu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if name == "torch" or name == "boogu" or name.startswith("boogu."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("tools.run_v3_boogu_reference_edit_worker")

    assert module is not None
    assert "torch" not in sys.modules
    assert not any(name == "boogu" or name.startswith("boogu.") for name in sys.modules)


def test_worker_passes_explicit_size_and_thinking_to_fake_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = importlib.import_module("tools.run_v3_boogu_reference_edit_worker")
    code_root = tmp_path / "vendor"
    model = tmp_path / "model"
    code_root.mkdir()
    model.mkdir()
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    result_path = tmp_path / "result.json"
    Image.new("RGB", (19, 23), (1, 2, 3)).save(input_path)
    captured: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            captured["generator_device"] = device

        def manual_seed(self, seed: int) -> FakeGenerator:
            captured["seed"] = seed
            return self

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakePipeline:
            captured["model_path"] = path
            captured["load_kwargs"] = kwargs
            return cls()

        def to(self, device: str) -> None:
            captured["device"] = device

        def devices_manager(self, **kwargs: object) -> None:
            captured["devices_manager"] = kwargs

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            captured["call"] = kwargs
            rewrite_path = Path(str(kwargs["save_rewritten_instruction_path"]))
            rewrite_path.write_text(
                json.dumps(
                    {
                        "ori_instruction": ["Complete it."],
                        "rewritten_instruction": ["Complete the same object."],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(images=[Image.new("RGB", (1360, 768))])

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    torch_module.Generator = FakeGenerator
    pipeline_module = types.ModuleType("boogu.pipelines.boogu.pipeline_boogu_turbo")
    pipeline_module.BooguImageTurboPipeline = FakePipeline
    for name in ("boogu", "boogu.pipelines", "boogu.pipelines.boogu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "boogu.pipelines.boogu.pipeline_boogu_turbo",
        pipeline_module,
    )
    payload = {
        "schema_version": 1,
        "code_root": str(code_root.resolve()),
        "model_path": str(model.resolve()),
        "model_name": "Boogu-Image-0.1-Edit-Turbo",
        "model_revision": "hotfix-1k-20260708",
        "device": "cuda:0",
        "seed": 17,
        "input_image_path": str(input_path.resolve()),
        "output_image_path": str(output_path.resolve()),
        "result_path": str(result_path.resolve()),
        "instruction": "Complete it.",
        "thinking_enabled": True,
        "width": 1360,
        "height": 768,
    }

    response = worker_module.run_request(payload)

    call = captured["call"]
    assert isinstance(call, dict)
    assert call["width"] == 1360
    assert call["height"] == 768
    assert call["device"] == "cuda:0"
    assert call["rewriter_device"] == "cuda:0"
    assert call["unload_rewriter_level"] == "keep"
    assert call["enable_inner_devices_manager"] is False
    assert call["align_res"] is False
    assert call["use_rewrite_text_instruction"] is True
    assert call["input_images"][0][0].size == (19, 23)
    assert call["input_images"][0][0].mode == "RGB"
    assert response["rewritten_instruction"] == "Complete the same object."
    assert response["effective_instruction"] == "Complete the same object."
    assert captured["devices_manager"] == {
        "instant_rewriter_device": "cuda:0",
        "user_set_pipe_device": "cuda:0",
        "user_set_rewriter_device": "cuda:0",
        "execution_device": "cuda:0",
        "unload_rewriter_level": "keep",
    }
    with Image.open(output_path) as output:
        assert output.size == (1360, 768)
        assert output.mode == "RGB"


def test_jsonl_worker_loads_pipeline_once_for_multiple_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = importlib.import_module("tools.run_v3_boogu_reference_edit_worker")
    input_path = tmp_path / "input.png"
    Image.new("RGB", (19, 23), (1, 2, 3)).save(input_path)
    load_calls = 0
    inference_calls: list[dict[str, object]] = []

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> FakeGenerator:
            self.seed = seed
            return self

    class FakePipeline:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            inference_calls.append(kwargs)
            return SimpleNamespace(
                images=[
                    Image.new(
                        "RGB",
                        (int(kwargs["width"]), int(kwargs["height"])),
                    )
                ]
            )

    def fake_load_pipeline(**kwargs: object) -> tuple[FakePipeline, object]:
        nonlocal load_calls
        del kwargs
        load_calls += 1
        return FakePipeline(), SimpleNamespace(Generator=FakeGenerator)

    requests = [
        {
            "schema_version": 1,
            "type": "edit",
            "request_id": "entity_1",
            "input_image_path": str(input_path.resolve()),
            "output_image_path": str((tmp_path / "entity_1.png").resolve()),
            "instruction": "Edit entity_1.",
            "thinking_enabled": False,
            "width": 32,
            "height": 32,
        },
        {
            "schema_version": 1,
            "type": "edit",
            "request_id": "entity_2",
            "input_image_path": str(input_path.resolve()),
            "output_image_path": str((tmp_path / "entity_2.png").resolve()),
            "instruction": "Edit entity_2.",
            "thinking_enabled": False,
            "width": 32,
            "height": 32,
            "seed": 17,
        },
    ]
    requests.append(
        {
            "schema_version": 1,
            "type": "shutdown",
            "request_id": "shutdown_1",
        }
    )
    stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    stdout = io.StringIO()
    monkeypatch.setattr(worker_module, "_load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(worker_module.sys, "stdin", stdin)
    monkeypatch.setattr(worker_module.sys, "stdout", stdout)
    args = SimpleNamespace(
        code_root=tmp_path,
        model_path=tmp_path,
        model_name="Boogu-Image-0.1-Edit-Turbo",
        model_revision="hotfix-1k-20260708",
        device="cuda:0",
        seed=0,
    )

    assert worker_module.serve(args) == 0

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert load_calls == 1
    assert len(inference_calls) == 2
    assert [call["generator"].seed for call in inference_calls] == [0, 17]
    assert [call["num_inference_steps"] for call in inference_calls] == [4, 4]
    assert [item["request_id"] for item in responses[1:3]] == [
        "entity_1",
        "entity_2",
    ]
    assert responses[-1]["type"] == "shutdown"


class _ReviewCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.payload))
                )
            ]
        )


class _SequencedReviewCompletions:
    def __init__(self, payloads: list[dict[str, object] | str]) -> None:
        self.payloads = iter(payloads)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        payload = next(self.payloads)
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )


def test_production_qwen_boogu_reviewer_uses_structured_two_image_review() -> None:
    payload = _completion_review().model_dump(mode="json")
    completions = _ReviewCompletions(payload)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        close=lambda: None,
    )
    judge = QwenBooguReferenceEditJudge(
        QwenServiceConfig(model="/models/qwen"),
        client=client,
    )

    review = judge.review(
        operation="complete_entity",
        source_rgba=Image.new("RGBA", (12, 10), (1, 2, 3, 255)),
        source_input_rgb=Image.new("RGB", (12, 10), (1, 2, 3)),
        candidate_rgb=Image.new("RGB", (32, 32), (4, 5, 6)),
        entity_phrase="a person in a blue coat",
        reference_type="subject",
    )

    assert review.verdict == "accept"
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    user_content = call["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in user_content) == 2
    assert not any("mask" in str(item).lower() for item in user_content)


def test_qwen_boogu_review_repair_and_operations_are_profiled(
    tmp_path: Path,
) -> None:
    completions = _SequencedReviewCompletions(
        [
            "{}",
            _completion_review().model_dump(mode="json"),
            _background_review().model_dump(mode="json"),
        ]
    )
    judge = QwenBooguReferenceEditJudge(
        QwenServiceConfig(model="/models/qwen"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            close=lambda: None,
        ),
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")
    images = {
        "source_rgba": Image.new("RGBA", (12, 10), (1, 2, 3, 255)),
        "source_input_rgb": Image.new("RGB", (12, 10), (1, 2, 3)),
        "candidate_rgb": Image.new("RGB", (32, 32), (4, 5, 6)),
    }

    with active_profiler(profiler):
        completion = judge.review(
            operation="complete_entity",
            entity_phrase="a person in a blue coat",
            reference_type="subject",
            **images,
        )
        background = judge.review(
            operation="add_entity_background",
            entity_phrase="a person in a blue coat",
            reference_type="subject",
            **images,
        )

    assert completion.verdict == "accept"
    assert background.verdict == "accept"
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["component"] for event in events] == [
        "qwen_boogu_completion_review",
        "qwen_boogu_completion_review",
        "qwen_boogu_background_review",
    ]
    assert [event["retry_index"] for event in events] == [0, 1, 0]
    assert all(event["input_image_count"] == 2 for event in events)


class _SamTrackBackend:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.calls: list[dict[str, object]] = []

    def track(self, **kwargs: object) -> EntityTrackResult:
        self.calls.append(kwargs)
        return EntityTrackResult(
            status="ready",
            observations=(
                BackendMaskObservation(
                    slot=5,
                    mask=self.mask,
                    confidence=0.9,
                    object_id="target",
                ),
            ),
        )


class _SamNotFoundBackend:
    def track(self, **kwargs: object) -> EntityTrackResult:
        del kwargs
        return EntityTrackResult(status="not_found")


class _SamEmptyMaskBackend:
    def track(self, **kwargs: object) -> EntityTrackResult:
        del kwargs
        return EntityTrackResult(
            status="ready",
            observations=(
                BackendMaskObservation(
                    slot=5,
                    mask=np.zeros((10, 10), dtype=bool),
                    confidence=0.9,
                    object_id="target",
                ),
            ),
        )


class _SamFailingBackend:
    def track(self, **kwargs: object) -> EntityTrackResult:
        del kwargs
        raise RuntimeError("sam unavailable")


class _SamAmbiguousInstancesBackend:
    def track(self, **kwargs: object) -> EntityTrackResult:
        del kwargs
        return EntityTrackResult(
            status="failed",
            reason=(
                "SAM3 returned multiple ambiguous instances for a "
                "single-entity prompt"
            ),
        )


def test_production_sam3_boogu_reviewer_is_review_only_and_tracks_ten_frames(
    tmp_path: Path,
) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    backend = _SamTrackBackend(mask)
    reviewer = Sam3BooguReferenceReviewer(
        backend,
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )
    source = Image.new("RGBA", (10, 10), (1, 2, 3, 0))
    source_alpha = np.zeros((10, 10), dtype=np.uint8)
    source_alpha[2:6, 2:6] = 255
    source.putalpha(Image.fromarray(source_alpha, mode="L"))

    review = reviewer.review(
        operation="complete_entity",
        source_rgba=source,
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the blue object",
        reference_type="object",
    )

    assert review.passed is True
    assert review.diagnostics["failure_kind"] == "none"
    assert review.diagnostics["mask_usage"] == "review_only"
    assert len(backend.calls[0]["frame_paths"]) == 10
    assert backend.calls[0]["grounding_prompt"] == "the blue object"


def test_production_sam3_boogu_reviewer_rejects_excessive_area_growth(
    tmp_path: Path,
) -> None:
    mask = np.ones((10, 10), dtype=bool)
    backend = _SamTrackBackend(mask)
    reviewer = Sam3BooguReferenceReviewer(
        backend,
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )
    source = Image.new("RGBA", (10, 10), (1, 2, 3, 0))
    source.putpixel((5, 5), (1, 2, 3, 255))

    review = reviewer.review(
        operation="add_entity_background",
        source_rgba=source,
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is False
    assert review.area_growth_acceptable is False
    assert review.diagnostics["failure_kind"] == "excessive_area_growth"


def test_production_sam3_empty_mask_is_not_found_without_fake_geometry(
    tmp_path: Path,
) -> None:
    reviewer = Sam3BooguReferenceReviewer(
        _SamEmptyMaskBackend(),
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )

    review = reviewer.review(
        operation="add_entity_background",
        source_rgba=Image.new("RGBA", (10, 10), (1, 2, 3, 255)),
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is False
    assert review.diagnostics["failure_kind"] == "not_found"
    assert "geometry_gate_passed" not in review.diagnostics


def test_production_sam3_review_records_scale_collapse_geometry(
    tmp_path: Path,
) -> None:
    candidate_mask = np.zeros((10, 10), dtype=bool)
    candidate_mask[:4, :] = True
    reviewer = Sam3BooguReferenceReviewer(
        _SamTrackBackend(candidate_mask),
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )

    review = reviewer.review(
        operation="add_entity_background",
        source_rgba=Image.new("RGBA", (10, 10), (1, 2, 3, 255)),
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is True
    assert review.diagnostics["candidate_scale_ratio"] == pytest.approx(0.4)
    assert review.diagnostics["geometry_gate_passed"] is False
    assert review.diagnostics["geometry_rejection_reason"] == (
        "entity_scale_collapsed"
    )


def test_production_sam3_review_records_layout_shift_geometry(
    tmp_path: Path,
) -> None:
    source = Image.new("RGBA", (10, 10), (1, 2, 3, 0))
    source_alpha = np.zeros((10, 10), dtype=np.uint8)
    source_alpha[3:7, 3:7] = 255
    source.putalpha(Image.fromarray(source_alpha, mode="L"))
    candidate_mask = np.zeros((10, 10), dtype=bool)
    candidate_mask[:4, :4] = True
    reviewer = Sam3BooguReferenceReviewer(
        _SamTrackBackend(candidate_mask),
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )

    review = reviewer.review(
        operation="complete_entity",
        source_rgba=source,
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is True
    assert review.diagnostics["candidate_scale_ratio"] == pytest.approx(1.0)
    assert review.diagnostics["center_shift"] > 0.20
    assert review.diagnostics["geometry_rejection_reason"] == (
        "entity_shifted_off_layout"
    )


def test_production_sam3_review_accepts_small_geometry_change(
    tmp_path: Path,
) -> None:
    source = Image.new("RGBA", (10, 10), (1, 2, 3, 0))
    source_alpha = np.zeros((10, 10), dtype=np.uint8)
    source_alpha[2:6, 2:6] = 255
    source.putalpha(Image.fromarray(source_alpha, mode="L"))
    candidate_mask = np.zeros((10, 10), dtype=bool)
    candidate_mask[3:7, 3:7] = True
    reviewer = Sam3BooguReferenceReviewer(
        _SamTrackBackend(candidate_mask),
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )

    review = reviewer.review(
        operation="add_entity_background",
        source_rgba=source,
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is True
    assert review.diagnostics["center_shift"] < 0.20
    assert review.diagnostics["geometry_gate_passed"] is True
    assert review.diagnostics["geometry_rejection_reason"] is None


@pytest.mark.parametrize(
    ("backend", "expected_failure_kind"),
    [
        (_SamNotFoundBackend(), "not_found"),
        (_SamAmbiguousInstancesBackend(), "multiple_instances"),
        (_SamFailingBackend(), "backend_failure"),
    ],
)
def test_production_sam3_boogu_reviewer_classifies_tracking_failure(
    tmp_path: Path,
    backend: object,
    expected_failure_kind: str,
) -> None:
    reviewer = Sam3BooguReferenceReviewer(
        backend,
        temporary_root=tmp_path,
        max_area_growth_ratio=2.0,
        max_significant_components=2,
    )

    review = reviewer.review(
        operation="add_entity_background",
        source_rgba=Image.new("RGBA", (10, 10), (1, 2, 3, 255)),
        candidate_rgb=Image.new("RGB", (10, 10), (4, 5, 6)),
        entity_phrase="the object",
        reference_type="object",
    )

    assert review.passed is False
    assert review.diagnostics["failure_kind"] == expected_failure_kind
