from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3 import subject_attributes
from r2v_data_v2.v3.pair import EntityReferenceCandidate
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipRecord,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedMasksArtifact,
    render_inline_instruction_text,
)
from r2v_data_v2.v3.subject_attributes import (
    DiscoveredSubjectAttribute,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    OwnershipGeometry,
    SubjectAttributeDiscovery,
    SubjectAttributeRecord,
    SubjectAttributeReview,
    SubjectAttributeReviewBatch,
    evaluate_owner_eligibility,
    evaluate_ownership_geometry,
    prefer_attribute_candidate_frames,
)


def _visibility(*, qualifies: bool) -> EntityVisibilitySummary:
    visible_slots = list(range(7)) if qualifies else []
    return EntityVisibilitySummary(
        status="ready" if qualifies else "not_found",
        visible_frame_slots=visible_slots,
        visible_frame_count=len(visible_slots),
        coverage_ratio=len(visible_slots) / 10,
        qualifies=qualifies,
        per_frame_area_ratio=[0.2 if slot in visible_slots else 0.0 for slot in range(10)],
        per_frame_confidence=[0.9 if slot in visible_slots else None for slot in range(10)],
    )


def _ready_reference(entity_id: str, *, source_frame_index: int = 0) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=entity_id,
        status="ready",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="clear",
        image_path=f"clips/clip-1/selected/{entity_id}.png",
        source_frame_index=source_frame_index,
    )


def _clip(
    *,
    entity_types: tuple[str, ...] = ("subject",),
    retained_ids: tuple[str, ...] | None = None,
) -> ClipRecord:
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=reference_type,
            phrase=f"person {index}" if reference_type == "subject" else f"item {index}",
            grounding_prompt=(
                f"the person {index}"
                if reference_type == "subject"
                else f"the item {index}"
            ),
        )
        for index, reference_type in enumerate(entity_types, start=1)
    ]
    retained_ids = retained_ids or tuple(entity.entity_id for entity in entities)
    type_counts = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity in entities:
        if entity.entity_id not in retained_ids:
            continue
        type_counts[entity.reference_type] += 1
        tokens[entity.entity_id] = (
            f"<ref_{entity.reference_type}_{type_counts[entity.reference_type]}>"
        )
    body = " and ".join(
        f"{{{{image_{index}}}}}"
        for index in range(1, len(retained_ids) + 1)
    ) + " move through the scene."
    legend = [
        InstructionLegendEntry(image_id=f"image_{index}", description="clear reference")
        for index in range(1, len(retained_ids) + 1)
    ]
    qualifying = retained_ids[-1]
    references = [
        (
            _ready_reference(entity.entity_id)
            if entity.entity_id in retained_ids
            else EntityReferenceState(
                entity_id=entity.entity_id,
                status="rejected",
                reference_scope="reject",
                visible_region="custom",
                whole_entity_recognizable=False,
                identity_features_visible=False,
                scope_reason="not retained",
            )
        )
        for entity in entities
    ]
    return ClipRecord(
        clip_uid="clip-1",
        source=ClipSource(
            video_path="/mnt/workspace/public/dataset/example.mp4",
            parent_video_id="parent",
            clip_suffix="001",
            source_index=0,
            caption_raw="caption",
            metadata={},
        ),
        annotation=AnnotationState(
            status="ready",
            instruction_template=" ".join(
                f"{{{{entity_{index}}}}}"
                for index in range(1, len(entities) + 1)
            ),
            entities=entities,
        ),
        coverage=CoverageState(
            passed=True,
            qualifying_entity_ids=[qualifying],
            entity_visibility_summary={
                entity.entity_id: _visibility(
                    qualifies=entity.entity_id == qualifying
                )
                for entity in entities
            },
        ),
        references=ReferencesState(entities=references),
        pairing=PairingState(
            status="ready",
            retained_entity_ids=list(retained_ids),
            tokens=tokens,
        ),
        instruction=InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_inline_instruction_text(body),
        ),
        export=ExportState(accepted=True, reason=None),
    )


def _candidate(
    candidate_id: str,
    *,
    slot: int,
    source_frame_index: int,
    mask: np.ndarray | None = None,
) -> EntityReferenceCandidate:
    binary = np.zeros((100, 100), dtype=bool) if mask is None else mask
    if not binary.any():
        binary[20:80, 20:80] = True
    return EntityReferenceCandidate(
        candidate_id=candidate_id,
        entity_id="e1",
        frame_slot=slot,
        source_frame_index=source_frame_index,
        image_path=f"clips/clip-1/frames/{slot:02d}.jpg",
        mask=binary,
        bbox_xyxy=(20, 20, 80, 80),
        area_pixels=int(binary.sum()),
        area_ratio=float(binary.mean()),
        bbox_fill_ratio=1.0,
        border_contact_count=0,
        normalized_center_distance=0.0,
        sharpness_score=1.0,
    )


def _accepted_review(attribute_id: str) -> SubjectAttributeReview:
    return SubjectAttributeReview(
        attribute_id=attribute_id,
        matches_attribute=True,
        owner_binding_correct=True,
        recognizable=True,
        characteristic_appearance_visible=True,
        usable_as_attribute_condition=True,
        reason="clear and useful",
    )


def _geometry() -> OwnershipGeometry:
    return OwnershipGeometry(
        passed=True,
        reason="passed",
        owner_overlap_ratio=1.0,
        maximum_other_owner_overlap_ratio=0.0,
        attribute_to_owner_area_ratio=0.1,
        near_owner_region=True,
        attribute_area_pixels=100,
        attribute_long_side_pixels=10,
        significant_component_count=1,
        largest_component_ratio=1.0,
        second_largest_component_ratio=0.0,
    )


def _attribute_record(
    attribute_id: str,
    owner_entity_id: str,
    attribute_type: str,
    *,
    accepted: bool = True,
) -> SubjectAttributeRecord:
    discovered = DiscoveredSubjectAttribute(
        attribute_type=attribute_type,
        phrase=f"{attribute_type} phrase",
        grounding_prompt=f"the {attribute_type} belonging to {owner_entity_id}",
    )
    if not accepted:
        return SubjectAttributeRecord(
            attribute_id=attribute_id,
            owner_entity_id=owner_entity_id,
            attribute_type=discovered.attribute_type,
            phrase=discovered.phrase,
            grounding_prompt=discovered.grounding_prompt,
            status="rejected",
            sam3_prompt=discovered.grounding_prompt,
            reason="recognizability:fragment",
        )
    return SubjectAttributeRecord(
        attribute_id=attribute_id,
        owner_entity_id=owner_entity_id,
        attribute_type=discovered.attribute_type,
        phrase=discovered.phrase,
        grounding_prompt=discovered.grounding_prompt,
        status="accepted",
        image_path=f"references/clip-1/{attribute_id}.png",
        source_frame_index=10,
        source_frame_slot=1,
        owner_candidate_id="candidate_2",
        same_frame_as_owner_reference=False,
        sam3_prompt=discovered.grounding_prompt,
        ownership_geometry=_geometry(),
        review=_accepted_review(attribute_id),
        reason="accepted",
    )


@pytest.mark.parametrize("reference_type", ["object", "group"])
def test_owner_eligibility_accepts_only_retained_ready_subjects(
    reference_type: str,
) -> None:
    subject_clip = _clip()
    assert evaluate_owner_eligibility(
        subject_clip,
        entity_id="e1",
        has_usable_candidate_evidence=True,
    ) == subject_attributes.OwnerEligibility(True, "eligible")
    assert evaluate_owner_eligibility(
        subject_clip,
        entity_id="e1",
        has_usable_candidate_evidence=False,
    ).reason == "usable_owner_candidate_evidence_missing"

    object_clip = _clip(entity_types=(reference_type,))
    assert evaluate_owner_eligibility(
        object_clip,
        entity_id="e1",
        has_usable_candidate_evidence=True,
    ).reason == "owner_is_not_subject"

    nonretained = _clip(
        entity_types=("subject", "subject"),
        retained_ids=("e2",),
    )
    assert evaluate_owner_eligibility(
        nonretained,
        entity_id="e1",
        has_usable_candidate_evidence=True,
    ).reason == "subject_not_retained"

    integrity_rejected = subject_clip.model_copy(
        update={
            "reference_integrity": SimpleNamespace(
                status="ready",
                entities=[SimpleNamespace(entity_id="e1", status="rejected")],
            )
        }
    )
    assert evaluate_owner_eligibility(
        integrity_rejected,
        entity_id="e1",
        has_usable_candidate_evidence=True,
    ).reason == "reference_integrity_rejected"


def test_discovery_schema_enforces_owner_and_three_attribute_bound() -> None:
    payload = SubjectAttributeDiscovery(
        owner_entity_id="e1",
        owner_is_human=True,
        attributes=[
            DiscoveredSubjectAttribute(
                attribute_type="hair",
                phrase="long black hair",
                grounding_prompt="the long black hair of the woman",
            )
        ],
    )
    assert payload.attributes[0].attribute_type == "hair"
    with pytest.raises(ValidationError):
        SubjectAttributeDiscovery(
            owner_entity_id="owner-1",
            owner_is_human=True,
            attributes=[],
        )
    with pytest.raises(ValidationError):
        SubjectAttributeDiscovery(
            owner_entity_id="e1",
            owner_is_human=True,
            attributes=[
                DiscoveredSubjectAttribute(
                    attribute_type=attribute_type,
                    phrase=attribute_type,
                    grounding_prompt=f"the {attribute_type} of the person",
                )
                for attribute_type in (
                    "face",
                    "hair",
                    "headwear",
                    "glasses",
                )
            ],
        )
    assert SubjectAttributeDiscovery(
        owner_entity_id="e1",
        owner_is_human=False,
        attributes=[],
    ).attributes == []
    with pytest.raises(ValidationError):
        SubjectAttributeDiscovery(
            owner_entity_id="e1",
            owner_is_human=False,
            attributes=payload.attributes,
        )


def test_different_owner_frame_is_preferred_and_same_frame_is_fallback() -> None:
    same = _candidate("candidate_1", slot=0, source_frame_index=0)
    different = _candidate("candidate_2", slot=1, source_frame_index=10)
    ordered = prefer_attribute_candidate_frames(
        [same, different],
        owner_reference_source_frame_index=0,
    )
    assert [candidate.candidate_id for candidate in ordered] == [
        "candidate_2",
        "candidate_1",
    ]
    assert prefer_attribute_candidate_frames(
        [same],
        owner_reference_source_frame_index=0,
    ) == [same]


def test_ownership_geometry_accepts_intended_owner_and_rejects_wrong_owner() -> None:
    owner = np.zeros((100, 100), dtype=bool)
    owner[20:80, 10:50] = True
    other = np.zeros_like(owner)
    other[20:80, 55:95] = True
    intended_attribute = np.zeros_like(owner)
    intended_attribute[30:50, 20:40] = True
    accepted = evaluate_ownership_geometry(
        intended_attribute,
        owner,
        {"e2": other},
        owner_padding_ratio=0.08,
    )
    assert accepted.passed is True
    assert accepted.owner_overlap_ratio == 1.0

    wrong_attribute = np.zeros_like(owner)
    wrong_attribute[30:50, 65:85] = True
    rejected = evaluate_ownership_geometry(
        wrong_attribute,
        owner,
        {"e2": other},
        owner_padding_ratio=0.08,
    )
    assert rejected.passed is False
    assert rejected.reason == "attribute_primarily_belongs_to_other_subject"


def test_recognizability_requires_every_review_boolean() -> None:
    assert _accepted_review("a1").accepted is True
    fragment = _accepted_review("a1").model_copy(update={"recognizable": False})
    assert fragment.accepted is False


def test_owner_aware_rendering_keeps_attributes_after_correct_subject() -> None:
    clip = _clip(entity_types=("subject", "subject"))
    attributes = [
        _attribute_record("a1", "e1", "hair"),
        _attribute_record("a2", "e2", "upper_clothing"),
    ]
    enriched, references = subject_attributes._render_enriched_instruction(
        clip,
        attributes,
    )
    assert [(reference.kind, reference.owner_entity_id) for reference in references] == [
        ("subject", None),
        ("attribute", "e1"),
        ("subject", None),
        ("attribute", "e2"),
    ]
    assert "<Image 1>, with the hairstyle shown in <Image 2>" in enriched
    assert "<Image 3>, wearing the clothing shown in <Image 4>" in enriched
    assert clip.instruction is not None
    assert clip.instruction.r2v_instruction == "<Image 1> and <Image 2> move through the scene."

    without_second, remaining = subject_attributes._render_enriched_instruction(
        clip,
        [attributes[0]],
    )
    assert "clothing shown" not in without_second
    assert [reference.kind for reference in remaining] == [
        "subject",
        "attribute",
        "subject",
    ]
    assert set(subject_attributes._ANGLE_IMAGE_LABEL.findall(without_second)) == {
        "1",
        "2",
        "3",
    }


class _FakeStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def frame_path(self, clip_uid: str, slot: int) -> Path:
        return self.root / "clips" / clip_uid / "frames" / f"{slot:02d}.jpg"


class _DiscoveryClient:
    calls = 0

    def discover(self, *, owner, owner_candidates, source_images):
        self.calls += 1
        assert owner.entity_id == "e1"
        assert len(owner_candidates) == 2
        assert set(source_images) == {candidate.image_path for candidate in owner_candidates}
        return SubjectAttributeDiscovery(
            owner_entity_id="e1",
            owner_is_human=True,
            attributes=[
                DiscoveredSubjectAttribute(
                    attribute_type="hair",
                    phrase="long black hair",
                    grounding_prompt="the long black hair of person 1",
                ),
                DiscoveredSubjectAttribute(
                    attribute_type="upper_clothing",
                    phrase="red jacket",
                    grounding_prompt="the red jacket worn by person 1",
                ),
            ],
        )


class _ReviewClient:
    calls = 0

    def review(self, *, owner, candidates):
        self.calls += 1
        assert owner.entity_id == "e1"
        return SubjectAttributeReviewBatch(
            owner_entity_id="e1",
            reviews=[_accepted_review(candidate.attribute_id) for candidate in candidates],
        )


class _SegmentationBackend:
    def __init__(self) -> None:
        self.calls = 0

    def track(self, **kwargs):
        self.calls += 1
        assert kwargs["reference_type"] == "object"
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:45, 30:45] = True
        return EntityTrackResult(
            status="ready",
            observations=(
                BackendMaskObservation(
                    slot=0,
                    mask=mask,
                    confidence=0.9,
                    object_id="attribute",
                ),
                BackendMaskObservation(
                    slot=1,
                    mask=mask,
                    confidence=0.9,
                    object_id="attribute",
                ),
            ),
        )


def _frames(root: Path) -> SampledFramesArtifact:
    directory = root / "clips" / "clip-1" / "frames"
    directory.mkdir(parents=True)
    frames = []
    for slot in range(10):
        path = directory / f"{slot:02d}.jpg"
        Image.new("RGB", (100, 100), (slot * 10, 50, 100)).save(path)
        frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot + 1),
                image_path=f"frames/{slot:02d}.jpg",
                sha256="0" * 64,
            )
        )
    return SampledFramesArtifact(
        clip_uid="clip-1",
        width=100,
        height=100,
        frames=frames,
    )


def test_owner_processing_batches_qwen_and_prefers_different_frame(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "run")
    owner_mask = np.zeros((100, 100), dtype=bool)
    owner_mask[20:80, 20:80] = True
    candidates = [
        _candidate("candidate_1", slot=0, source_frame_index=0, mask=owner_mask),
        _candidate("candidate_2", slot=1, source_frame_index=10, mask=owner_mask),
    ]
    discovery = _DiscoveryClient()
    review = _ReviewClient()
    segmenter = _SegmentationBackend()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(pair=SimpleNamespace(crop_padding_ratio=0.08)),
        storage=_FakeStorage(tmp_path / "run"),
        output_root=tmp_path / "output",
        clip=_clip(),
        owner=_clip().annotation.entities[0],
        owner_candidates=candidates,
        frames=frames,
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={},
        ),
        attribute_id_start=1,
        discovery_client=discovery,
        review_client=review,
        segmentation_backend=segmenter,
    )
    assert discovery.calls == 1
    assert review.calls == 1
    assert segmenter.calls == 2
    assert artifact.metrics.discovery_calls == 1
    assert artifact.metrics.review_calls == 1
    assert artifact.metrics.sam3_attempts == 2
    assert artifact.metrics.accepted_attributes == 2
    assert {record.source_frame_index for record in artifact.records} == {10}
    assert all(record.same_frame_as_owner_reference is False for record in artifact.records)
    for record in artifact.records:
        assert record.image_path is not None
        with Image.open(tmp_path / "output" / record.image_path) as opened:
            assert opened.mode == "RGBA"


def test_valid_owner_artifact_is_restart_safe(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    image_path = output_root / "references" / "clip-1" / "a1.png"
    image_path.parent.mkdir(parents=True)
    rgba = np.full((10, 10, 4), 255, dtype=np.uint8)
    rgba[:2, :, :3] = 255
    rgba[:2, :, 3] = 0
    Image.fromarray(rgba, mode="RGBA").save(image_path)
    record = _attribute_record("a1", "e1", "hair")
    artifact = OwnerEnrichmentArtifact(
        sample_id="clip-1",
        owner_entity_id="e1",
        owner_is_human=True,
        attribute_id_start=1,
        owner_phrase="person 1",
        owner_grounding_prompt="the person 1",
        records=[record],
        metrics=OwnerEnrichmentMetrics(
            discovery_calls=1,
            review_calls=1,
            sam3_attempts=1,
            discovered_by_type={"hair": 1},
            accepted_attributes=1,
            different_frame_accepted=1,
        ),
    )
    artifact_path = output_root / "owners" / "clip-1" / "e1.json"
    artifact_path.parent.mkdir(parents=True)
    write_json_atomic(artifact_path, artifact.model_dump(mode="json"))
    assert subject_attributes._load_cached_owner_artifact(
        artifact_path,
        output_root=output_root,
        sample_id="clip-1",
        owner_entity_id="e1",
        attribute_id_start=1,
    ) == artifact
    assert subject_attributes._load_cached_owner_artifact(
        artifact_path,
        output_root=output_root,
        sample_id="clip-1",
        owner_entity_id="e1",
        attribute_id_start=2,
    ) is None


def test_attribute_output_cannot_overlap_source_run_or_visual_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", tmp_path)
    run_root = tmp_path / "run"
    export_root = tmp_path / "visual-export"
    subject_attributes._validate_output_root(
        run_root,
        export_root,
        tmp_path / "attribute-enrichment",
    )
    with pytest.raises(ValueError, match="source run_root"):
        subject_attributes._validate_output_root(
            run_root,
            export_root,
            run_root / "attributes",
        )
    with pytest.raises(ValueError, match="Visual export_root"):
        subject_attributes._validate_output_root(
            run_root,
            export_root,
            export_root,
        )
