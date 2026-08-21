from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3 import subject_attributes
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    SubjectAttributeCompletionConfig,
    V3Config,
)
from r2v_data_v2.v3.reference_edit_boogu import BooguCompletionReview
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.pair import EntityReferenceCandidate
from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundReferenceState,
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
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_inline_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage
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


def _review_candidate(
    attribute_id: str,
    owner_candidate: EntityReferenceCandidate,
) -> subject_attributes.PendingAttributeCandidate:
    return subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="hair",
            phrase="long black hair",
            grounding_prompt="the long black hair of person 1",
        ),
        attribute_id=attribute_id,
        owner_entity_id="e1",
        owner_candidate=owner_candidate,
        attribute_mask=np.ones((100, 100), dtype=bool),
        source_image=Image.new("RGB", (100, 100)),
        crop=Image.new("RGBA", (20, 20)),
        geometry=_geometry(),
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
    with pytest.raises(ValidationError):
        SubjectAttributeDiscovery.model_validate(
            {
                **payload.model_dump(),
                "owner_phrase": "person 1",
            }
        )


def test_discovery_prompt_requires_exact_top_level_contract() -> None:
    prompt = " ".join(subject_attributes.DISCOVERY_SYSTEM_PROMPT.split())
    assert (
        "top level contains exactly owner_entity_id, owner_is_human, and attributes"
        in prompt
    )
    assert "Do not return owner_phrase, owner_grounding_prompt" in prompt
    assert "attributes must be a JSON array" in prompt
    assert "return owner_is_human=false and attributes=[]" in prompt


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


def _rectangular_mask(*, width: int, height: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10 : 10 + height, 10 : 10 + width] = True
    return mask


def test_clothing_owner_ratio_079_is_rejected() -> None:
    geometry = _geometry().model_copy(
        update={"attribute_to_owner_area_ratio": 0.79}
    )
    assert subject_attributes._clothing_geometry_rejection_reason(
        "upper_clothing", _rectangular_mask(width=20, height=20), geometry
    ) == "clothing_mask_too_owner_like"


def test_clothing_owner_ratio_073_is_allowed() -> None:
    geometry = _geometry().model_copy(
        update={"attribute_to_owner_area_ratio": 0.73}
    )
    assert (
        subject_attributes._clothing_geometry_rejection_reason(
            "lower_clothing", _rectangular_mask(width=20, height=20), geometry
        )
        is None
    )


def test_nonclothing_owner_ratio_079_is_not_rejected_by_clothing_rule() -> None:
    geometry = _geometry().model_copy(
        update={"attribute_to_owner_area_ratio": 0.79}
    )
    assert (
        subject_attributes._clothing_geometry_rejection_reason(
            "hair", _rectangular_mask(width=20, height=20), geometry
        )
        is None
    )


def test_clothing_aspect_ratio_40_is_rejected() -> None:
    assert subject_attributes._clothing_geometry_rejection_reason(
        "dress_or_skirt",
        _rectangular_mask(width=40, height=10),
        _geometry(),
    ) == "clothing_mask_too_strip_like"


def test_clothing_aspect_ratio_18_is_allowed() -> None:
    assert (
        subject_attributes._clothing_geometry_rejection_reason(
            "upper_clothing",
            _rectangular_mask(width=18, height=10),
            _geometry(),
        )
        is None
    )


@pytest.mark.parametrize(
    ("attribute_type", "long_side", "expected_reason"),
    [
        ("hair", 146, "hair_attribute_too_small"),
        ("hair", 191, "hair_attribute_too_small"),
        ("hair", 192, None),
        ("hair", 193, None),
        ("headwear", 56, "headwear_attribute_too_small"),
        ("headwear", 127, "headwear_attribute_too_small"),
        ("headwear", 128, None),
        ("headwear", 129, None),
        ("glasses", 57, None),
        ("face", 57, None),
        ("face", 500, None),
        ("upper_clothing", 57, None),
    ],
)
def test_type_specific_minimum_recognizable_size(
    attribute_type: str,
    long_side: int,
    expected_reason: str | None,
) -> None:
    geometry = _geometry().model_copy(
        update={"attribute_long_side_pixels": long_side}
    )
    assert (
        subject_attributes._type_specific_size_rejection_reason(
            attribute_type,
            geometry,
        )
        == expected_reason
    )


def _pending_mask_stub(
    attribute_id: str,
    attribute_type: str,
    mask: np.ndarray,
) -> SimpleNamespace:
    return SimpleNamespace(
        attribute_id=attribute_id,
        discovered=SimpleNamespace(attribute_type=attribute_type),
        attribute_mask=mask,
    )


def test_different_attribute_types_with_mask_iou_090_conflict() -> None:
    first = _rectangular_mask(width=20, height=20)
    second = first.copy()
    second[10, 10:20] = False
    assert subject_attributes._duplicate_attribute_mask_conflicts(
        [
            _pending_mask_stub("a1", "upper_clothing", first),
            _pending_mask_stub("a2", "lower_clothing", second),
        ]
    ) == {"a1", "a2"}


def test_clearly_different_attribute_masks_do_not_conflict() -> None:
    first = _rectangular_mask(width=20, height=20)
    second = np.roll(first, shift=50, axis=1)
    assert not subject_attributes._duplicate_attribute_mask_conflicts(
        [
            _pending_mask_stub("a1", "hair", first),
            _pending_mask_stub("a2", "upper_clothing", second),
        ]
    )


def test_recognizability_requires_every_review_boolean() -> None:
    assert _accepted_review("a1").accepted is True
    fragment = _accepted_review("a1").model_copy(update={"recognizable": False})
    assert fragment.accepted is False


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((1920, 1080), (768, 432)),
        ((1080, 1920), (432, 768)),
        ((640, 480), (640, 480)),
    ],
)
def test_qwen_input_resize_preserves_aspect_ratio_without_upscaling(
    source_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    original = Image.new("RGB", source_size, (12, 34, 56))
    original_pixels = original.tobytes()

    resized = subject_attributes._resize_qwen_input_image(original)

    assert resized.size == expected_size
    assert resized.mode == "RGB"
    assert resized is not original
    assert original.size == source_size
    assert original.tobytes() == original_pixels


def test_qwen_input_resize_preserves_rgba_without_mutating_original() -> None:
    original = Image.new("RGBA", (1024, 512), (12, 34, 56, 78))
    original_pixels = original.tobytes()

    resized = subject_attributes._resize_qwen_input_image(original)

    assert resized.size == (768, 384)
    assert resized.mode == "RGBA"
    assert original.size == (1024, 512)
    assert original.mode == "RGBA"
    assert original.tobytes() == original_pixels


def test_review_prompt_requires_canonical_batch_shape() -> None:
    prompt = " ".join(subject_attributes.REVIEW_SYSTEM_PROMPT.split())
    assert "top level contains exactly owner_entity_id and reviews" in prompt
    assert "reviews must be a JSON array" in prompt
    assert "Every review array item must contain exactly attribute_id" in prompt
    assert "preserving the supplied attribute order" in prompt
    assert "Do not return an object keyed by a1, a2, a3" in prompt
    assert "isolated transparent crop is the CROP-ONLY QUALITY TARGET" in prompt
    assert "owner image is OWNERSHIP-ONLY CONTEXT" in prompt
    assert "may be used only for owner_binding_correct" in prompt
    assert "neutral viewer" in prompt
    assert "contour-only hair" in prompt
    assert "isolated sleeve, cuff, shoulder, hem, or trouser edge" in prompt
    assert "Do not require whole-object completeness" in prompt
    assert "dominant visible content of the isolated crop" in prompt
    assert "substantial unrelated body regions, another garment" in prompt
    assert "recognizable requires sufficient component structure" in prompt
    assert "Hair needs a coherent hairstyle region or silhouette" in prompt
    assert "Face needs enough facial structure to function independently" in prompt
    assert "clothing needs coherent garment structure" in prompt
    assert "beyond the main subject reference" in prompt
    assert "pose-dominated cutout" in prompt
    assert "meaning relies on the requested attribute_type or owner context" in prompt


def test_review_input_labels_separate_quality_from_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = subject_attributes.QwenSubjectAttributeClient(
        QwenServiceConfig(),
        client=SimpleNamespace(),
    )
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return SubjectAttributeReviewBatch(
            owner_entity_id="e1",
            reviews=[_accepted_review("a1")],
        ).model_dump_json()

    monkeypatch.setattr(client, "_request", fake_request)
    candidate = _review_candidate(
        "a1",
        _candidate(
            "candidate_1",
            slot=1,
            source_frame_index=10,
        ),
    )
    owner = _clip().annotation.entities[0]
    client.review(owner=owner, candidates=[candidate])

    content = captured["content"]
    assert isinstance(content, list)
    metadata = json.loads(content[0]["text"])
    assert metadata == {
        "owner_entity_id": "e1",
        "attributes": [
            {
                "attribute_id": "a1",
                "attribute_type": "hair",
            }
        ],
    }
    labels = [
        item["text"]
        for item in content
        if item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    assert (
        "OWNERSHIP-ONLY CONTEXT for attributes a1\n"
        "(use only for owner_binding_correct)"
    ) in labels
    assert any(label.startswith("a1 CROP-ONLY QUALITY TARGET") for label in labels)


def _captured_review_content(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[subject_attributes.PendingAttributeCandidate],
) -> tuple[list[dict[str, object]], SubjectAttributeReviewBatch]:
    client = subject_attributes.QwenSubjectAttributeClient(
        QwenServiceConfig(),
        client=SimpleNamespace(),
    )
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return SubjectAttributeReviewBatch(
            owner_entity_id="e1",
            reviews=[
                _accepted_review(candidate.attribute_id) for candidate in candidates
            ],
        ).model_dump_json()

    monkeypatch.setattr(client, "_request", fake_request)
    payload = client.review(owner=_clip().annotation.entities[0], candidates=candidates)
    content = captured["content"]
    assert isinstance(content, list)
    return content, payload


def test_review_deduplicates_shared_owner_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_owner_candidate = _candidate(
        "candidate_1",
        slot=1,
        source_frame_index=10,
    )
    content, _ = _captured_review_content(
        monkeypatch,
        [
            _review_candidate("a1", shared_owner_candidate),
            _review_candidate("a2", shared_owner_candidate),
        ],
    )
    image_count = sum(item.get("type") == "image_url" for item in content)
    labels = [
        item["text"]
        for item in content
        if item.get("type") == "text" and isinstance(item.get("text"), str)
    ]

    assert image_count == 3
    assert labels.count(
        "OWNERSHIP-ONLY CONTEXT for attributes a1, a2\n"
        "(use only for owner_binding_correct)"
    ) == 1
    assert sum("CROP-ONLY QUALITY TARGET" in label for label in labels) == 2


def test_review_keeps_distinct_owner_contexts_and_output_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _review_candidate(
            attribute_id,
            _candidate(
                f"candidate_{index}",
                slot=index,
                source_frame_index=index * 10,
            ),
        )
        for index, attribute_id in enumerate(("a1", "a2", "a3"), start=1)
    ]
    content, payload = _captured_review_content(monkeypatch, candidates)
    context_labels = [
        item["text"]
        for item in content
        if item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and item["text"].startswith("OWNERSHIP-ONLY CONTEXT")
    ]

    assert sum(item.get("type") == "image_url" for item in content) == 6
    assert len(context_labels) == 3
    assert [review.attribute_id for review in payload.reviews] == ["a1", "a2", "a3"]


def test_owner_aware_rendering_keeps_attributes_after_correct_subject() -> None:
    clip = _clip(entity_types=("subject", "subject"))
    attributes = [
        _attribute_record("a1", "e1", "hair"),
        _attribute_record("a2", "e2", "upper_clothing"),
    ]
    enriched, references = subject_attributes._render_enriched_instruction(
        clip,
        attributes,
        source_run_root=Path("/source-run"),
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
        source_run_root=Path("/source-run"),
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


def test_clean_raw_background_visual_path_is_source_run_relative(tmp_path: Path) -> None:
    clip = _clip()
    assert clip.instruction is not None
    clip = clip.model_copy(
        update={
            "references": clip.references.model_copy(
                update={
                    "background": BackgroundReferenceState(
                        status="clean_raw",
                        source_image_path="frames/05.jpg",
                        output_image_path="frames/05.jpg",
                        source_frame_slot=5,
                        source_frame_index=50,
                        source_foreground_area_pixels=0,
                        source_foreground_area_ratio=0.0,
                    )
                }
            ),
            "pairing": clip.pairing.model_copy(
                update={"background_token": "<ref_bg_1>"}
            ),
            "instruction": clip.instruction.model_copy(
                update={
                    "instruction_body_template": (
                        "{{image_1}} crosses {{image_2}}."
                    )
                }
            ),
        }
    )
    _, references = subject_attributes._render_enriched_instruction(
        clip,
        [],
        source_run_root=tmp_path / "run",
    )
    background = next(reference for reference in references if reference.kind == "background")
    assert background.origin == "visual_run"
    assert background.image_path == "clips/clip-1/frames/05.jpg"


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
                    attribute_type="accessory",
                    phrase="silver necklace",
                    grounding_prompt="the silver necklace worn by person 1",
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
    def __init__(self, mask: np.ndarray | None = None) -> None:
        self.calls: list[int] = []
        self.mask = mask

    def track(self, **kwargs):
        raise AssertionError("attribute enrichment must not call full track")

    def segment_frame(self, *, frame_path, frame_slot, grounding_prompt):
        assert frame_path.name == f"{frame_slot:02d}.jpg"
        assert grounding_prompt
        self.calls.append(frame_slot)
        if self.mask is not None:
            return (self.mask,)
        mask = np.zeros((100, 100), dtype=bool)
        if "jacket" in grounding_prompt:
            mask[50:65, 50:65] = True
        else:
            mask[30:45, 30:45] = True
        return (mask,)


class _PromptOnlyPredictor:
    def __init__(self) -> None:
        self.request_types: list[str] = []
        self.stream_calls = 0

    def handle_request(self, request):
        self.request_types.append(request["type"])
        if request["type"] == "start_session":
            return {"session_id": "attribute-probe"}
        if request["type"] == "add_prompt":
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[30:45, 30:45] = 1
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_binary_masks": mask[None, ...],
                    "out_probs": np.array([0.9]),
                    "out_obj_ids": np.array([1]),
                },
            }
        if request["type"] == "close_session":
            return {}
        raise AssertionError(f"unexpected SAM3 request: {request['type']}")

    def handle_stream_request(self, request):
        self.stream_calls += 1
        raise AssertionError("attribute enrichment must not propagate in video")


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


def _tracked_subject(mask: np.ndarray) -> TrackedEntityMasks:
    height, width = mask.shape
    bbox = (0, 0, width, height)
    frames: list[TrackedMaskFrame] = []
    for slot in range(10):
        encoded = encode_binary_mask(mask)
        frames.append(
            TrackedMaskFrame(
                slot=slot,
                present=True,
                confidence=0.9,
                backend_confidences=[0.9],
                backend_object_ids=["subject"],
                area_pixels=int(mask.sum()),
                area_ratio=float(mask.mean()),
                bbox_xyxy=bbox,
                rle=encoded,
            )
        )
    return TrackedEntityMasks(
        status="ready",
        reference_type="subject",
        grounding_prompt="the other person",
        backend_object_ids=["subject"],
        frames=frames,
    )


def test_subject_attribute_sam3_helper_never_propagates(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "run")
    predictor = _PromptOnlyPredictor()
    core_backend = Sam3SegmentationBackend(
        Sam3Config(),
        predictor=predictor,
    )
    segmenter = subject_attributes.Sam3AttributeFrameSegmenter(
        Sam3Config(),
        backend=core_backend,
    )
    masks = segmenter.segment_frame(
        frame_path=tmp_path / "run" / "clips" / "clip-1" / frames.frames[1].image_path,
        frame_slot=1,
        grounding_prompt="the red jacket worn by person 1",
    )
    assert len(masks) == 1
    assert predictor.request_types == ["start_session", "add_prompt", "close_session"]
    assert predictor.stream_calls == 0


def test_persistent_worker_adapter_sends_exact_frame_provenance(tmp_path: Path) -> None:
    frames = _frames(tmp_path / "run")
    calls: list[dict[str, object]] = []

    class _ProbeClient:
        def attribute_probe(self, **kwargs):
            calls.append(kwargs)
            return (np.ones((2, 2), dtype=bool),)

    storage = SimpleNamespace(
        root=tmp_path / "run",
        read_frames=lambda _clip_uid: frames,
        clip_dir=lambda clip_uid: tmp_path / "run" / "clips" / clip_uid,
    )
    adapter = subject_attributes.PersistentWorkerAttributeFrameSegmenter(
        storage,
        _ProbeClient(),
    )
    masks = adapter.segment_frame(
        frame_path=tmp_path / "run" / "clips" / "clip-1" / "frames" / "01.jpg",
        frame_slot=1,
        grounding_prompt="the red jacket worn by person 1",
    )

    assert len(masks) == 1
    assert calls == [
        {
            "clip_uid": "clip-1",
            "frame_slot": 1,
            "source_frame_index": 10,
            "grounding_prompt": "the red jacket worn by person 1",
        }
    ]


def test_only_owner_candidate_frames_are_probed(tmp_path: Path) -> None:
    _frames(tmp_path / "run")
    candidates = [
        _candidate("candidate_1", slot=2, source_frame_index=20),
        _candidate("candidate_2", slot=5, source_frame_index=50),
    ]

    class _CandidateOnlySegmenter:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def segment_frame(self, *, frame_path, frame_slot, grounding_prompt):
            self.calls.append(frame_slot)
            if frame_slot != 5:
                return ()
            mask = np.zeros((100, 100), dtype=bool)
            mask[30:45, 30:45] = True
            return (mask,)

    segmenter = _CandidateOnlySegmenter()
    clip = _clip()
    selected = subject_attributes._select_attribute_candidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="accessory",
            phrase="silver necklace",
            grounding_prompt="the silver necklace worn by person 1",
        ),
        attribute_id="a1",
        clip_uid=clip.clip_uid,
        owner=clip.annotation.entities[0],
        owner_candidates=candidates,
        owner_reference=_ready_reference("e1"),
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={},
        ),
        storage=_FakeStorage(tmp_path / "run"),
        other_subject_ids=[],
        crop_padding_ratio=0.08,
        segmentation_backend=segmenter,
    )
    assert isinstance(selected, subject_attributes.PendingAttributeCandidate)
    assert selected.owner_candidate.frame_slot == 5
    assert segmenter.calls == [2, 5]


class _GmeScreener:
    def __init__(self, outcomes: list[bool | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, int]] = []
        self.config = SimpleNamespace(model_name="test-gme")

    def screen(self, *, crop, phrase, attribute_type):
        self.calls.append((phrase, attribute_type, int(np.asarray(crop)[..., 3].sum())))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        positive = 0.5 if outcome else 0.1
        negative = 0.4 if outcome else 0.2
        return subject_attributes.GmeRelativeMarginResult(
            positive_score=positive,
            negative_scores={"owner_body": negative},
            max_negative_score=negative,
            margin=positive - negative,
            passed=outcome,
            reason=(
                "gme_relative_margin_pass"
                if outcome
                else "gme_semantic_quality_reject"
            ),
        )


def _select_with_gme(
    tmp_path: Path,
    *,
    candidates: list[EntityReferenceCandidate],
    segmenter,
    gme_screener,
    masks: TrackedMasksArtifact | None = None,
    other_subject_ids: list[str] | None = None,
):
    clip = _clip()
    return subject_attributes._select_attribute_candidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="accessory",
            phrase="silver necklace",
            grounding_prompt="the silver necklace worn by person 1",
        ),
        attribute_id="a1",
        clip_uid=clip.clip_uid,
        owner=clip.annotation.entities[0],
        owner_candidates=candidates,
        owner_reference=_ready_reference("e1"),
        masks=masks
        or TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={},
        ),
        storage=_FakeStorage(tmp_path / "run"),
        other_subject_ids=other_subject_ids or [],
        crop_padding_ratio=0.08,
        segmentation_backend=segmenter,
        gme_screener=gme_screener,
    )


def test_gme_reject_retries_next_owner_candidate_and_pass_stops(
    tmp_path: Path,
) -> None:
    _frames(tmp_path / "run")
    candidates = [
        _candidate("candidate_1", slot=1, source_frame_index=10),
        _candidate("candidate_2", slot=2, source_frame_index=20),
        _candidate("candidate_3", slot=3, source_frame_index=30),
    ]
    segmenter = _SegmentationBackend()
    gme = _GmeScreener([False, True])
    metrics = subject_attributes._GmeSelectionMetrics()
    clip = _clip()

    selected = subject_attributes._select_attribute_candidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="accessory",
            phrase="silver necklace",
            grounding_prompt="the silver necklace worn by person 1",
        ),
        attribute_id="a1",
        clip_uid=clip.clip_uid,
        owner=clip.annotation.entities[0],
        owner_candidates=candidates,
        owner_reference=_ready_reference("e1"),
        masks=TrackedMasksArtifact(
            clip_uid="clip-1", height=100, width=100, entities={}
        ),
        storage=_FakeStorage(tmp_path / "run"),
        other_subject_ids=[],
        crop_padding_ratio=0.08,
        segmentation_backend=segmenter,
        gme_screener=gme,
        gme_metrics=metrics,
    )

    assert isinstance(selected, subject_attributes.PendingAttributeCandidate)
    assert selected.owner_candidate.candidate_id == "candidate_2"
    assert segmenter.calls == [1, 2]
    assert len(gme.calls) == 2
    assert [attempt.status for attempt in selected.gme_attempts] == [
        "rejected",
        "passed",
    ]
    assert selected.selected_gme_attempt_index == 1
    assert metrics.retried_next_frame == 1


def test_geometry_and_wrong_owner_rejections_happen_before_gme(
    tmp_path: Path,
) -> None:
    _frames(tmp_path / "run")
    candidate = _candidate("candidate_1", slot=1, source_frame_index=10)
    outside = np.zeros((100, 100), dtype=bool)
    outside[:5, :5] = True
    gme = _GmeScreener([True])
    geometry_rejected = _select_with_gme(
        tmp_path,
        candidates=[candidate],
        segmenter=_SegmentationBackend(outside),
        gme_screener=gme,
    )
    assert isinstance(geometry_rejected, SubjectAttributeRecord)
    assert geometry_rejected.reason.startswith("ownership_geometry:")
    assert gme.calls == []

    owner_mask = np.zeros((100, 100), dtype=bool)
    owner_mask[20:80, 10:50] = True
    other_mask = np.zeros_like(owner_mask)
    other_mask[20:80, 55:95] = True
    wrong_attribute = np.zeros_like(owner_mask)
    wrong_attribute[30:50, 65:85] = True
    wrong_owner_gme = _GmeScreener([True])
    wrong_owner = _select_with_gme(
        tmp_path,
        candidates=[
            _candidate(
                "candidate_1",
                slot=1,
                source_frame_index=10,
                mask=owner_mask,
            )
        ],
        segmenter=_SegmentationBackend(wrong_attribute),
        gme_screener=wrong_owner_gme,
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={"e2": _tracked_subject(other_mask)},
        ),
        other_subject_ids=["e2"],
    )
    assert isinstance(wrong_owner, SubjectAttributeRecord)
    assert "other_subject" in wrong_owner.reason
    assert wrong_owner_gme.calls == []


def test_bbox_fill_ratio_is_recorded_without_a_new_threshold() -> None:
    owner = np.zeros((100, 100), dtype=bool)
    owner[10:90, 10:90] = True
    ring = np.zeros_like(owner)
    ring[30:70, 30:70] = True
    ring[35:65, 35:65] = False

    geometry = evaluate_ownership_geometry(
        ring,
        owner,
        {},
        owner_padding_ratio=0.08,
    )

    assert geometry.passed is True
    assert geometry.bbox_fill_ratio == pytest.approx(700 / 1600)
    assert geometry.largest_component_ratio == 1.0


def test_gme_crop_preserves_all_mask_components_without_repair(tmp_path: Path) -> None:
    _frames(tmp_path / "run")
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:40, 30:40] = True
    mask[60:63, 60:63] = True
    gme = _GmeScreener([True])

    selected = _select_with_gme(
        tmp_path,
        candidates=[_candidate("candidate_1", slot=1, source_frame_index=10)],
        segmenter=_SegmentationBackend(mask),
        gme_screener=gme,
    )

    assert isinstance(selected, subject_attributes.PendingAttributeCandidate)
    assert gme.calls[0][2] // 255 == int(mask.sum())
    assert selected.geometry.second_largest_component_ratio > 0


def test_discovery_prompt_requires_english_attribute_text() -> None:
    prompt = subject_attributes.DISCOVERY_SYSTEM_PROMPT
    assert "phrase and grounding_prompt" in prompt
    assert "written in English" in prompt


def test_owner_processing_batches_qwen_and_prefers_different_frame(tmp_path: Path) -> None:
    _frames(tmp_path / "run")
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
    assert segmenter.calls == [1, 1]
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


@pytest.mark.parametrize(
    ("gme_outcome", "qwen_accepts", "expected_status", "expected_review_calls"),
    [
        (RuntimeError("worker unavailable"), True, "accepted", 1),
        (True, False, "rejected", 1),
        (True, True, "accepted", 1),
        (False, True, "rejected", 0),
    ],
)
def test_gme_prefilter_keeps_qwen_as_final_review_and_fails_open(
    tmp_path: Path,
    gme_outcome: bool | Exception,
    qwen_accepts: bool,
    expected_status: str,
    expected_review_calls: int,
) -> None:
    _frames(tmp_path / "run")

    class _OneAttributeDiscovery:
        def discover(self, *, owner, owner_candidates, source_images):
            return SubjectAttributeDiscovery(
                owner_entity_id=owner.entity_id,
                owner_is_human=True,
                attributes=[
                    DiscoveredSubjectAttribute(
                        attribute_type="accessory",
                        phrase="silver necklace",
                        grounding_prompt="the silver necklace worn by person 1",
                    )
                ],
            )

    class _FinalReview:
        def __init__(self) -> None:
            self.calls = 0

        def review(self, *, owner, candidates):
            self.calls += 1
            reviews = []
            for candidate in candidates:
                review = _accepted_review(candidate.attribute_id)
                if not qwen_accepts:
                    review = review.model_copy(
                        update={
                            "recognizable": False,
                            "reason": "fragmentary",
                        }
                    )
                reviews.append(review)
            return SubjectAttributeReviewBatch(
                owner_entity_id=owner.entity_id,
                reviews=reviews,
            )

    review = _FinalReview()
    clip = _clip()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(pair=SimpleNamespace(crop_padding_ratio=0.08)),
        storage=_FakeStorage(tmp_path / "run"),
        output_root=tmp_path / "output",
        clip=clip,
        owner=clip.annotation.entities[0],
        owner_candidates=[
            _candidate("candidate_1", slot=1, source_frame_index=10)
        ],
        masks=TrackedMasksArtifact(
            clip_uid="clip-1", height=100, width=100, entities={}
        ),
        attribute_id_start=1,
        discovery_client=_OneAttributeDiscovery(),
        review_client=review,
        segmentation_backend=_SegmentationBackend(),
        gme_screener=_GmeScreener([gme_outcome]),
    )

    record = artifact.records[0]
    assert record.status == expected_status
    assert review.calls == expected_review_calls
    assert artifact.metrics.gme_calls == 1
    if isinstance(gme_outcome, Exception):
        assert artifact.metrics.gme_failures == 1
        assert record.gme_attempts[0].status == "failed"
        assert record.selected_gme_attempt_index == 0
    elif not gme_outcome:
        assert record.reason == "gme_semantic_quality_reject"
        assert artifact.metrics.gme_candidates_rejected == 1
    elif qwen_accepts:
        assert record.review is not None and record.review.accepted
    else:
        assert record.reason == "recognizability:fragmentary"


@pytest.mark.parametrize(
    ("attribute_type", "expected_reason"),
    [
        ("hair", "hair_attribute_too_small"),
        ("headwear", "headwear_attribute_too_small"),
    ],
)
def test_type_specific_size_rejection_counts_as_deterministic(
    tmp_path: Path,
    attribute_type: str,
    expected_reason: str,
) -> None:
    _frames(tmp_path / "run")

    class _OneAttributeDiscovery:
        def discover(self, *, owner, owner_candidates, source_images):
            return SubjectAttributeDiscovery(
                owner_entity_id=owner.entity_id,
                owner_is_human=True,
                attributes=[
                    DiscoveredSubjectAttribute(
                        attribute_type=attribute_type,
                        phrase=attribute_type,
                        grounding_prompt=f"the {attribute_type} of person 1",
                    )
                ],
            )

    review = _ReviewClient()
    clip = _clip()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(pair=SimpleNamespace(crop_padding_ratio=0.08)),
        storage=_FakeStorage(tmp_path / "run"),
        output_root=tmp_path / "output",
        clip=clip,
        owner=clip.annotation.entities[0],
        owner_candidates=[
            _candidate("candidate_1", slot=1, source_frame_index=10)
        ],
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={},
        ),
        attribute_id_start=1,
        discovery_client=_OneAttributeDiscovery(),
        review_client=review,
        segmentation_backend=_SegmentationBackend(),
    )
    assert review.calls == 0
    assert artifact.records[0].reason == f"ownership_geometry:{expected_reason}"
    assert artifact.metrics.deterministic_ownership_rejects == 1


def test_nonretained_subject_mask_rejects_wrong_owner_attribute(tmp_path: Path) -> None:
    _frames(tmp_path / "run")
    owner_mask = np.zeros((100, 100), dtype=bool)
    owner_mask[20:80, 10:50] = True
    other_mask = np.zeros_like(owner_mask)
    other_mask[20:80, 55:95] = True
    wrong_attribute = np.zeros_like(owner_mask)
    wrong_attribute[30:50, 65:85] = True
    clip = _clip(
        entity_types=("subject", "subject"),
        retained_ids=("e1",),
    )

    class _OneAttributeDiscovery:
        def discover(self, *, owner, owner_candidates, source_images):
            return SubjectAttributeDiscovery(
                owner_entity_id=owner.entity_id,
                owner_is_human=True,
                attributes=[
                    DiscoveredSubjectAttribute(
                        attribute_type="upper_clothing",
                        phrase="red jacket",
                        grounding_prompt="the red jacket worn by person 1",
                    )
                ],
            )

    review = _ReviewClient()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(pair=SimpleNamespace(crop_padding_ratio=0.08)),
        storage=_FakeStorage(tmp_path / "run"),
        output_root=tmp_path / "output",
        clip=clip,
        owner=clip.annotation.entities[0],
        owner_candidates=[
            _candidate(
                "candidate_1",
                slot=1,
                source_frame_index=10,
                mask=owner_mask,
            )
        ],
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={"e2": _tracked_subject(other_mask)},
        ),
        attribute_id_start=1,
        discovery_client=_OneAttributeDiscovery(),
        review_client=review,
        segmentation_backend=_SegmentationBackend(wrong_attribute),
    )
    assert review.calls == 0
    assert len(artifact.records) == 1
    assert artifact.records[0].reason == (
        "ownership_geometry:attribute_primarily_belongs_to_other_subject"
    )


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


def test_clip_primitive_uses_cached_owner_without_touching_visual_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip()
    run_root = tmp_path / "run"
    clip_path = run_root / "clips" / clip.clip_uid / "clip.json"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_text(clip.model_dump_json(), encoding="utf-8")
    visual_bytes = clip_path.read_bytes()
    output_root = tmp_path / "attributes"
    cached = OwnerEnrichmentArtifact(
        sample_id=clip.clip_uid,
        owner_entity_id="e1",
        owner_is_human=True,
        attribute_id_start=1,
        owner_phrase="person 1",
        owner_grounding_prompt="the person 1",
        records=[_attribute_record("a1", "e1", "hair", accepted=False)],
        metrics=OwnerEnrichmentMetrics(
            discovery_calls=1,
            discovered_by_type={"hair": 1},
        ),
    )
    artifact_path = output_root / "owners" / clip.clip_uid / "e1.json"
    artifact_path.parent.mkdir(parents=True)
    write_json_atomic(artifact_path, cached.model_dump(mode="json"))
    monkeypatch.setattr(
        subject_attributes,
        "build_entity_reference_candidates",
        lambda *_args, **_kwargs: [
            _candidate("candidate_1", slot=1, source_frame_index=10)
        ],
    )

    class _Storage:
        root = run_root

        def read_clip(self, _clip_uid):
            return clip

        def read_frames(self, _clip_uid):
            return SimpleNamespace()

        def read_masks(self, _clip_uid):
            return TrackedMasksArtifact(
                clip_uid=clip.clip_uid,
                height=100,
                width=100,
                entities={},
            )

        def clip_path(self, _clip_uid):
            return clip_path

    class _ForbiddenModels:
        calls = 0

        def discover(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip Qwen")

        def review(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip Qwen")

        def segment_frame(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip SAM3")

    forbidden = _ForbiddenModels()
    result = subject_attributes.process_subject_attribute_clip(
        SimpleNamespace(pair=SimpleNamespace(crop_padding_ratio=0.08)),
        storage=_Storage(),
        output_root=output_root,
        clip=clip,
        discovery_client=forbidden,
        review_client=forbidden,
        segmentation_backend=forbidden,
    )

    assert result.totals.skipped_existing_owners == 1
    assert forbidden.calls == 0
    assert clip_path.read_bytes() == visual_bytes
    assert (output_root / "samples" / f"{clip.clip_uid}.json").is_file()


def test_sidecar_reconciliation_is_deterministic_after_out_of_order_completion(
    tmp_path: Path,
) -> None:
    clips = [
        _clip().model_copy(update={"clip_uid": "clip-a"}),
        _clip().model_copy(update={"clip_uid": "clip-b"}),
    ]
    output_root = tmp_path / "attributes"
    artifacts = {
        "clip-a": OwnerEnrichmentArtifact(
            sample_id="clip-a",
            owner_entity_id="e1",
            owner_is_human=True,
            attribute_id_start=1,
            owner_phrase="person 1",
            owner_grounding_prompt="the person 1",
            records=[_attribute_record("a1", "e1", "hair", accepted=False)],
            metrics=OwnerEnrichmentMetrics(discovery_calls=1),
        ),
        "clip-b": OwnerEnrichmentArtifact(
            sample_id="clip-b",
            owner_entity_id="e1",
            owner_is_human=True,
            attribute_id_start=1,
            owner_phrase="person 1",
            owner_grounding_prompt="the person 1",
            records=[_attribute_record("a1", "e1", "face", accepted=False)],
            metrics=OwnerEnrichmentMetrics(discovery_calls=1),
        ),
    }
    for clip_uid in ("clip-b", "clip-a"):
        path = output_root / "owners" / clip_uid / "e1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, artifacts[clip_uid].model_dump(mode="json"))

    storage = SimpleNamespace(
        root=tmp_path / "run",
        iter_clips=lambda: iter(clips),
        read_run=lambda: SimpleNamespace(git_commit="test-commit"),
    )
    subject_attributes.reconcile_subject_attribute_outputs(
        storage=storage,
        output_root=output_root,
        owner_limit=None,
        invocation_wall_time_seconds=0.0,
    )
    first_attributes = (output_root / "attributes.jsonl").read_bytes()
    first_samples = (output_root / "enriched_samples.jsonl").read_bytes()
    attribute_types = [
        json.loads(line)["attribute_type"]
        for line in first_attributes.decode("utf-8").splitlines()
    ]

    subject_attributes.reconcile_subject_attribute_outputs(
        storage=storage,
        output_root=output_root,
        owner_limit=None,
        invocation_wall_time_seconds=0.0,
    )

    assert attribute_types == ["hair", "face"]
    assert (output_root / "attributes.jsonl").read_bytes() == first_attributes
    assert (output_root / "enriched_samples.jsonl").read_bytes() == first_samples


def _source_run_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(v3_config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(v3_config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(v3_config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    qwen_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "visual-source",
        export_root=writable / "datasets" / "visual-source-v1",
        source=SourceConfig(limit=100),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(qwen_model)),
            instruction_writer=QwenServiceConfig(model=str(qwen_model)),
            candidate_judge=QwenServiceConfig(model=str(qwen_model)),
            background_remove_judge=QwenServiceConfig(model=str(qwen_model)),
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "Qwen-Image-Edit-2511-Object-Remover",
        ),
    )
    config.validate()
    return config


def test_standalone_enrichment_rejects_run_local_outputs_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source_run_config(tmp_path, monkeypatch)
    RunStorage(config).initialize(git_commit="visual-source")
    forbidden_models = SimpleNamespace()

    for output_root in (
        config.resolved_run_root / "foo",
        config.resolved_run_root / "subject_attributes",
    ):
        with pytest.raises(ValueError, match="separate from source run_root"):
            subject_attributes.run_subject_attribute_enrichment(
                config,
                run_root=config.resolved_run_root,
                output_root=output_root,
                discovery_client=forbidden_models,
                review_client=forbidden_models,
                segmentation_backend=forbidden_models,
            )


def test_run_local_sidecar_reuses_owner_artifacts_and_reaches_enrichment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source_run_config(tmp_path, monkeypatch)
    clips = [
        _clip().model_copy(update={"clip_uid": "clip-a"}),
        _clip().model_copy(update={"clip_uid": "clip-b"}),
    ]
    output_root = config.resolved_run_root / "subject_attributes"
    for clip, attribute_type in zip(clips, ("hair", "face"), strict=True):
        artifact = OwnerEnrichmentArtifact(
            sample_id=clip.clip_uid,
            owner_entity_id="e1",
            owner_is_human=True,
            attribute_id_start=1,
            owner_phrase="person 1",
            owner_grounding_prompt="the person 1",
            records=[
                _attribute_record("a1", "e1", attribute_type, accepted=False)
            ],
            metrics=OwnerEnrichmentMetrics(discovery_calls=1),
        )
        path = output_root / "owners" / clip.clip_uid / "e1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, artifact.model_dump(mode="json"))
    monkeypatch.setattr(
        subject_attributes,
        "build_entity_reference_candidates",
        lambda *_args, **_kwargs: [
            _candidate("candidate_1", slot=1, source_frame_index=10)
        ],
    )

    class _Storage:
        root = config.resolved_run_root

        def iter_clips(self):
            return iter(clips)

        def read_clip(self, clip_uid):
            return next(clip for clip in clips if clip.clip_uid == clip_uid)

        def read_frames(self, _clip_uid):
            return SimpleNamespace()

        def read_masks(self, clip_uid):
            return TrackedMasksArtifact(
                clip_uid=clip_uid,
                height=100,
                width=100,
                entities={},
            )

        def clip_path(self, clip_uid):
            return self.root / "clips" / clip_uid / "clip.json"

        def read_run(self):
            return SimpleNamespace(
                git_commit="visual-source",
                config_hash=config.fingerprint(),
                model_identifiers=config.model_identifiers(),
                source_manifest_path=str(config.dataset_json.resolve()),
            )

    class _ForbiddenModels:
        calls = 0

        def discover(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip Qwen")

        def review(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip Qwen")

        def segment_frame(self, **_kwargs):
            self.calls += 1
            raise AssertionError("cached owner must skip SAM3")

    storage = _Storage()
    forbidden = _ForbiddenModels()
    monkeypatch.setattr(subject_attributes, "RunStorage", lambda _config: storage)

    summary = subject_attributes.run_subject_attribute_enrichment(
        config,
        run_root=config.run_root,
        output_root=output_root,
        max_owners=1,
        discovery_client=forbidden,
        review_client=forbidden,
        segmentation_backend=forbidden,
        allow_run_local_sidecar=True,
    )

    records = [
        json.loads(line)
        for line in (output_root / "attributes.jsonl").read_text().splitlines()
    ]
    assert summary["eligible_human_owner_count"] == 1
    assert summary["skipped_existing_owner_count"] == 1
    assert [record["attribute_type"] for record in records] == ["hair"]
    assert forbidden.calls == 0


def test_wrong_source_config_fails_before_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config = _source_run_config(tmp_path, monkeypatch)
    RunStorage(source_config).initialize(git_commit="visual-source")
    assert source_config.qwen.candidate_judge is not None
    wrong_service = replace(
        source_config.qwen.candidate_judge,
        model=str(
            source_config.dataset_json.parent.parent
            / "pretrained"
            / "Qwen"
            / "wrong-model"
        ),
    )
    wrong_config = replace(
        source_config,
        qwen=replace(source_config.qwen, candidate_judge=wrong_service),
    )

    class _ForbiddenModelCalls:
        def __init__(self) -> None:
            self.calls = 0

        def discover(self, **kwargs):
            self.calls += 1
            raise AssertionError("Qwen discovery must not be called")

        def review(self, **kwargs):
            self.calls += 1
            raise AssertionError("Qwen review must not be called")

        def segment_frame(self, **kwargs):
            self.calls += 1
            raise AssertionError("SAM3 must not be called")

    forbidden = _ForbiddenModelCalls()
    with pytest.raises(ValueError, match="config_hash.*model_identifiers"):
        subject_attributes.run_subject_attribute_enrichment(
            wrong_config,
            run_root=source_config.run_root,
            output_root=(
                v3_config_module.ALLOWED_WRITABLE_ROOT / "attribute-enrichment"
            ),
            discovery_client=forbidden,
            review_client=forbidden,
            segmentation_backend=forbidden,
        )
    assert forbidden.calls == 0


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
    for output_root in (run_root / "foo", run_root / "subject_attributes"):
        with pytest.raises(ValueError, match="source run_root"):
            subject_attributes._validate_output_root(
                run_root,
                export_root,
                output_root,
            )

    subject_attributes._validate_output_root(
        run_root,
        export_root,
        run_root / "subject_attributes",
        allow_run_local_sidecar=True,
    )
    for output_root in (
        run_root / "foo",
        run_root,
        export_root,
        export_root / "subject_attributes",
    ):
        with pytest.raises(ValueError, match="must be exactly"):
            subject_attributes._validate_output_root(
                run_root,
                export_root,
                output_root,
                allow_run_local_sidecar=True,
            )

    with pytest.raises(ValueError, match="Visual export_root"):
        subject_attributes._validate_output_root(
            run_root,
            run_root / "subject_attributes",
            run_root / "subject_attributes",
            allow_run_local_sidecar=True,
        )
    with pytest.raises(ValueError, match="Visual export_root"):
        subject_attributes._validate_output_root(
            run_root,
            export_root,
            export_root,
        )


def _completion_review(*, accepted: bool) -> BooguCompletionReview:
    values = {
        "same_physical_entity": accepted,
        "identity_preserved": accepted,
        "original_visible_attributes_preserved": accepted,
        "exactly_one_entity": accepted,
        "missing_parts_plausibly_completed": accepted,
        "no_duplicate_entity": accepted,
        "no_unrelated_entity": accepted,
        "no_severe_structure_artifact": accepted,
        "style_coherent": accepted,
        "resolution_usable": accepted,
        "reference_usable": accepted,
        "certain": accepted,
    }
    return BooguCompletionReview(
        verdict="accept" if accepted else "reject",
        reason="usable" if accepted else "identity changed",
        **values,
    )


@pytest.mark.parametrize("attribute_type", ["face", "upper_clothing"])
def test_repairable_attribute_uses_boogu_then_sam3(
    tmp_path: Path,
    attribute_type: str,
) -> None:
    raw_mask = np.zeros((100, 100), dtype=bool)
    raw_mask[30:70, 30:70] = True
    crop_pixels = np.full((64, 64, 4), 255, dtype=np.uint8)
    crop_pixels[..., 3] = 0
    crop_pixels[16:48, 16:48, :3] = np.indices((32, 32))[0, ..., None] * 7
    crop_pixels[16:48, 16:48, 3] = 255
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type=attribute_type,
            phrase="distinct component",
            grounding_prompt="distinct component worn by person 1",
        ),
        attribute_id="a1",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_1", slot=1, source_frame_index=10),
        attribute_mask=raw_mask,
        source_image=Image.new("RGB", (100, 100), "white"),
        crop=Image.fromarray(crop_pixels, mode="RGBA"),
        geometry=_geometry(),
    )

    class Backend:
        calls = 0

        def attribute_completion(self, *, output_path, **_kwargs):
            self.calls += 1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop_pixels[..., :3], mode="RGB").save(output_path)
            return {"model_call_time_seconds": 0.25}

    class Segmenter:
        calls = 0

        def segment_generated_frame(self, **_kwargs):
            self.calls += 1
            mask = np.zeros((64, 64), dtype=bool)
            mask[16:48, 16:48] = True
            return (mask,)

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(accepted=True)

    backend = Backend()
    segmenter = Segmenter()
    metrics = subject_attributes._CompletionSelectionMetrics()
    config = replace(
        SubjectAttributeCompletionConfig(enabled=True),
        minimum_sharpness_score=-1.0,
        maximum_area_growth_ratio=4.0,
        face_maximum_area_growth_ratio=4.0,
        maximum_bbox_area_growth_ratio=4.0,
        face_maximum_bbox_area_growth_ratio=4.0,
    )
    completed, reason, failed = subject_attributes._attempt_attribute_completion(
        candidate,
        clip_uid="clip-1",
        output_root=tmp_path / "subject_attributes",
        completion_config=config,
        completion_backend=backend,
        completion_judge=Judge(),
        segmentation_backend=segmenter,
        metrics=metrics,
    )

    assert completed is not None
    assert completed.completion_attempted is True
    assert reason == "completion_accepted"
    assert failed is False
    assert backend.calls == segmenter.calls == 1
    assert metrics.attempts == metrics.accepted == 1


def test_unrepairable_blank_attribute_never_calls_completion(tmp_path: Path) -> None:
    crop = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
    crop.putpixel((32, 32), (255, 255, 255, 255))
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="face",
            phrase="face",
            grounding_prompt="face of person 1",
        ),
        attribute_id="a1",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_1", slot=1, source_frame_index=10),
        attribute_mask=np.ones((10, 10), dtype=bool),
        source_image=Image.new("RGB", (100, 100), "white"),
        crop=crop,
        geometry=_geometry(),
    )

    class Forbidden:
        def attribute_completion(self, **_kwargs):
            pytest.fail("unrepairable candidate reached Boogu")

    completed, reason, _ = subject_attributes._attempt_attribute_completion(
        candidate,
        clip_uid="clip-1",
        output_root=tmp_path,
        completion_config=SubjectAttributeCompletionConfig(enabled=True),
        completion_backend=Forbidden(),
        completion_judge=SimpleNamespace(),
        segmentation_backend=SimpleNamespace(),
        metrics=subject_attributes._CompletionSelectionMetrics(),
    )
    assert completed is None
    assert reason == "completion_precheck:blank_area_too_large"


def test_completion_review_rejects_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject_attributes,
        "_masked_crop_quality_rejection",
        lambda *_args, **_kwargs: None,
    )
    raw_mask = np.zeros((32, 32), dtype=bool)
    raw_mask[8:24, 8:24] = True
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="hair",
            phrase="hair",
            grounding_prompt="hair of person 1",
        ),
        attribute_id="a1",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_1", slot=1, source_frame_index=10),
        attribute_mask=raw_mask,
        source_image=Image.new("RGB", (100, 100), "white"),
        crop=Image.new("RGBA", (32, 32), (100, 100, 100, 255)),
        geometry=_geometry(),
    )

    class Backend:
        def attribute_completion(self, *, output_path, **_kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), "gray").save(output_path)
            return {"model_call_time_seconds": 0.1}

    class Segmenter:
        def segment_generated_frame(self, **_kwargs):
            return (raw_mask.copy(),)

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(accepted=False)

    metrics = subject_attributes._CompletionSelectionMetrics()
    completed, reason, _ = subject_attributes._attempt_attribute_completion(
        candidate,
        clip_uid="clip-1",
        output_root=tmp_path,
        completion_config=replace(
            SubjectAttributeCompletionConfig(enabled=True),
            maximum_alpha_coverage_without_completion=1.0,
        ),
        completion_backend=Backend(),
        completion_judge=Judge(),
        segmentation_backend=Segmenter(),
        metrics=metrics,
    )
    assert completed is None
    assert reason == "completion_qwen_review_reject"
    assert metrics.qwen_review_rejects == 1
