from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError
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
    SubjectAttributesConfig,
    V3Config,
)
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
        structure_complete=True,
        completion_recommended=False,
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
    with pytest.raises(ValidationError, match="must not repeat"):
        SubjectAttributeDiscovery(
            owner_entity_id="e1",
            owner_is_human=True,
            attributes=[payload.attributes[0], payload.attributes[0]],
        )
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
    assert "Each returned attribute_type must be unique" in prompt
    assert "return owner_is_human=false and attributes=[]" in prompt


def _discovery_client_with_raw_response(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> tuple[subject_attributes.QwenSubjectAttributeClient, list[dict[str, object]]]:
    client = subject_attributes.QwenSubjectAttributeClient(
        QwenServiceConfig(),
        client=SimpleNamespace(),
    )
    calls: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> str:
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(client, "_request", fake_request)
    return client, calls


def _discover_from_client(
    client: subject_attributes.QwenSubjectAttributeClient,
) -> SubjectAttributeDiscovery:
    candidate = _candidate("candidate_1", slot=1, source_frame_index=10)
    return client.discover(
        owner=_clip().annotation.entities[0],
        owner_candidates=[candidate],
        source_images={candidate.image_path: Image.new("RGB", (100, 100))},
    )


def test_discovery_duplicate_type_is_first_wins_in_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _discovery_client_with_raw_response(
        monkeypatch,
        json.dumps(
            {
                "owner_entity_id": "e1",
                "owner_is_human": True,
                "attributes": [
                    {
                        "attribute_type": "upper_clothing",
                        "phrase": "gray robe",
                        "grounding_prompt": "gray robe worn by the woman",
                    },
                    {
                        "attribute_type": "upper_clothing",
                        "phrase": "black vest",
                        "grounding_prompt": "black vest worn by the woman",
                    },
                    {
                        "attribute_type": "hair",
                        "phrase": "dark hair",
                        "grounding_prompt": "dark hair worn by the woman",
                    },
                ],
            }
        ),
    )

    payload = _discover_from_client(client)

    assert [attribute.attribute_type for attribute in payload.attributes] == [
        "upper_clothing",
        "hair",
    ]
    assert [attribute.phrase for attribute in payload.attributes] == [
        "gray robe",
        "dark hair",
    ]
    assert [attribute.grounding_prompt for attribute in payload.attributes] == [
        "gray robe worn by the woman",
        "dark hair worn by the woman",
    ]
    assert len(calls) == 1


def test_discovery_distinct_types_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_attributes = [
        {
            "attribute_type": "upper_clothing",
            "phrase": "gray robe",
            "grounding_prompt": "gray robe worn by the woman",
        },
        {
            "attribute_type": "hair",
            "phrase": "dark hair",
            "grounding_prompt": "dark hair worn by the woman",
        },
    ]
    client, calls = _discovery_client_with_raw_response(
        monkeypatch,
        json.dumps(
            {
                "owner_entity_id": "e1",
                "owner_is_human": True,
                "attributes": raw_attributes,
            }
        ),
    )

    payload = _discover_from_client(client)

    assert [attribute.model_dump() for attribute in payload.attributes] == raw_attributes
    assert len(calls) == 1


def test_discovery_malformed_payload_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _discovery_client_with_raw_response(
        monkeypatch,
        json.dumps(
            {
                "owner_entity_id": "e1",
                "owner_is_human": True,
                "attributes": [
                    {
                        "attribute_type": "hair",
                        "phrase": "",
                        "grounding_prompt": "dark hair worn by the woman",
                    }
                ],
            }
        ),
    )

    with pytest.raises(ValidationError):
        _discover_from_client(client)
    assert len(calls) == 1


def test_discovery_owner_id_mismatch_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _discovery_client_with_raw_response(
        monkeypatch,
        json.dumps(
            {
                "owner_entity_id": "e2",
                "owner_is_human": True,
                "attributes": [],
            }
        ),
    )

    with pytest.raises(ValueError, match="wrong owner_entity_id"):
        _discover_from_client(client)
    assert len(calls) == 1


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


def test_incomplete_semantically_correct_review_must_recommend_completion() -> None:
    with pytest.raises(
        ValidationError,
        match="semantically correct incomplete structure must recommend completion",
    ):
        SubjectAttributeReview(
            attribute_id="a1",
            matches_attribute=True,
            owner_binding_correct=True,
            recognizable=True,
            characteristic_appearance_visible=True,
            usable_as_attribute_condition=True,
            structure_complete=False,
            completion_recommended=False,
            reason="visible missing region",
        )


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
    assert "at least 50% of the frontal facial structure visible" in prompt
    assert "side profile" in prompt
    assert "only when attribute_type is face" in prompt
    assert "do not apply it to hair, headwear, glasses, clothing" in prompt
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


def _bbox_review_payload(*, accepted: bool = True) -> dict[str, object]:
    return {
        "correct_attribute": accepted,
        "owner_binding_correct": accepted,
        "target_is_dominant_and_identifiable": accepted,
        "no_large_competing_attribute_or_entity": accepted,
        "context_is_limited_and_supportive": accepted,
        "no_strong_owner_pose_or_scene_leakage": accepted,
        "no_severe_blur_or_artifact": accepted,
        "usable_as_attribute_condition": accepted,
        "certain": accepted,
        "reason": "usable" if accepted else "not usable",
        "verdict": "accept" if accepted else "reject",
    }


class _SequencedAttributeCompletions:
    def __init__(self, responses: list[dict[str, object] | BaseException]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response))
                )
            ]
        )


def _bbox_review_client(
    completions: _SequencedAttributeCompletions,
) -> subject_attributes.QwenSubjectAttributeClient:
    return subject_attributes.QwenSubjectAttributeClient(
        QwenServiceConfig(model="/models/qwen"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            close=lambda: None,
        ),
    )


def _call_bbox_review(
    client: subject_attributes.QwenSubjectAttributeClient,
) -> subject_attributes.SubjectAttributeBboxReview:
    return client.review_attribute_bbox(
        bbox_candidate=Image.new("RGB", (32, 24), "gray"),
        owner_context=Image.new("RGB", (48, 40), "white"),
        attribute_type="upper_clothing",
        attribute_phrase="gray robe",
    )


def test_bbox_review_uses_strict_json_schema() -> None:
    completions = _SequencedAttributeCompletions([_bbox_review_payload()])

    review = _call_bbox_review(_bbox_review_client(completions))

    assert review.verdict == "accept"
    response_format = completions.calls[0]["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "v3_subject_attribute_bbox_review",
            "strict": True,
            "schema": subject_attributes.SubjectAttributeBboxReview.model_json_schema(),
        },
    }


def test_bbox_review_json_schema_falls_back_with_full_schema() -> None:
    bad_request = BadRequestError(
        "json_schema unsupported",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        ),
        body={},
    )
    completions = _SequencedAttributeCompletions(
        [bad_request, _bbox_review_payload()]
    )

    review = _call_bbox_review(_bbox_review_client(completions))

    assert review.verdict == "accept"
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    fallback_prompt = completions.calls[1]["messages"][-1]["content"]
    assert json.dumps(
        subject_attributes.SubjectAttributeBboxReview.model_json_schema(),
        ensure_ascii=False,
    ) in fallback_prompt
    assert "Every required field must be present" in fallback_prompt
    assert "Do not omit booleans" in fallback_prompt


def test_bbox_review_missing_fields_gets_exactly_one_structural_repair() -> None:
    completions = _SequencedAttributeCompletions(
        [
            {"reason": "usable", "verdict": "accept"},
            _bbox_review_payload(),
        ]
    )

    review = _call_bbox_review(_bbox_review_client(completions))

    assert review.verdict == "accept"
    assert len(completions.calls) == 2
    repair_prompt = completions.calls[1]["messages"][-1]["content"]
    assert '"reason": "usable"' in repair_prompt
    assert "Validation error" in repair_prompt
    assert "Required schema" in repair_prompt


def test_bbox_review_second_invalid_response_fails_after_one_repair() -> None:
    completions = _SequencedAttributeCompletions(
        [
            {"reason": "usable", "verdict": "accept"},
            {"reason": "still incomplete", "verdict": "accept"},
        ]
    )

    with pytest.raises(ValueError, match="invalid structured"):
        _call_bbox_review(_bbox_review_client(completions))

    assert len(completions.calls) == 2


@pytest.mark.parametrize(
    ("false_flag", "model_verdict", "expected_verdict"),
    [
        (None, "reject", "accept"),
        ("no_severe_blur_or_artifact", "accept", "reject"),
        ("no_strong_owner_pose_or_scene_leakage", "accept", "reject"),
    ],
)
def test_bbox_review_verdict_is_normalized_from_complete_flags(
    false_flag: str | None,
    model_verdict: str,
    expected_verdict: str,
) -> None:
    payload = _bbox_review_payload()
    if false_flag is not None:
        payload[false_flag] = False
    payload["verdict"] = model_verdict
    completions = _SequencedAttributeCompletions([payload])

    review = _call_bbox_review(_bbox_review_client(completions))

    assert review.verdict == expected_verdict
    assert len(completions.calls) == 1


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


@pytest.mark.parametrize(
    ("attribute_type", "entity_name"),
    [
        ("face", "人脸"),
        ("headwear", "帽子"),
        ("accessory", "配饰"),
        ("upper_clothing", "衣服"),
        ("lower_clothing", "下装"),
        ("dress_or_skirt", "裙子"),
    ],
)
def test_attribute_completion_prompt_is_short_and_type_specific(
    attribute_type: str,
    entity_name: str,
) -> None:
    assert subject_attributes.attribute_completion_prompt(attribute_type) == (
        f"补全成完整的{entity_name}，去除掉破碎的部分，不添加其他内容"
    )


def test_hair_and_glasses_are_not_completion_eligible() -> None:
    eligible = SubjectAttributeCompletionConfig().eligible_types
    assert "hair" not in eligible
    assert "glasses" not in eligible


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
    assert segmenter.calls == [1, 0, 1, 0]
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

    legacy_completion_metrics = artifact.metrics.model_copy(
        update={
            "completion_attempts": 1,
            "completion_accepted": 1,
            "completion_attempts_by_type": {"hair": 1},
            "completion_accepted_by_type": {"hair": 1},
        }
    )
    legacy_completion_artifact = artifact.model_copy(
        update={
            "completion_mode": "boogu_completion_v1",
            "metrics": legacy_completion_metrics,
        }
    )
    write_json_atomic(
        artifact_path,
        legacy_completion_artifact.model_dump(mode="json"),
    )
    assert subject_attributes._load_durable_owner_artifact(
        artifact_path,
        output_root=output_root,
        sample_id="clip-1",
        owner_entity_id="e1",
    ) == legacy_completion_artifact

    completion_metrics = artifact.metrics.model_copy(
        update={
            "completion_attempts": 1,
            "completion_accepted": 1,
            "completion_raw_usable_attempts": 1,
            "completion_selected_completed": 1,
            "completion_sam_multi_mask": 1,
            "completion_sam_masks_returned_total": 2,
            "repaired_attribute_final_review_accepted": 1,
            "completion_attempts_by_type": {"hair": 1},
            "completion_accepted_by_type": {"hair": 1},
        }
    )
    completion_artifact = artifact.model_copy(
        update={
            "completion_mode": "boogu_completion_v1",
            "metrics": completion_metrics,
        }
    )
    write_json_atomic(
        artifact_path,
        completion_artifact.model_dump(mode="json"),
    )
    assert subject_attributes._load_durable_owner_artifact(
        artifact_path,
        output_root=output_root,
        sample_id="clip-1",
        owner_entity_id="e1",
    ) == completion_artifact

    storage = SimpleNamespace(
        root=tmp_path / "run",
        iter_clips=lambda: iter([_clip()]),
        read_run=lambda: SimpleNamespace(git_commit="test-commit"),
    )
    summary = subject_attributes.reconcile_subject_attribute_outputs(
        storage=storage,
        output_root=output_root,
        owner_limit=None,
        invocation_wall_time_seconds=0.0,
    )
    assert summary["eligible_human_owner_count"] == 1
    assert summary["attribute_record_count"] == 1
    assert summary["completion_attempts"] == 1
    assert summary["completion_selected_completed"] == 1
    assert summary["completion_sam_multi_mask"] == 1
    assert summary["completion_sam_masks_returned_total"] == 2
    assert subject_attributes._load_cached_owner_artifact(
        artifact_path,
        output_root=output_root,
        sample_id="clip-1",
        owner_entity_id="e1",
        attribute_id_start=2,
    ) is None


def test_attribute_png_validation_separates_raw_rgba_and_completed_rgb(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    raw_path = output_root / "references" / "clip-1" / "a1.png"
    completed_path = output_root / "references" / "clip-1" / "a2.png"
    raw_path.parent.mkdir(parents=True)
    raw_pixels = np.full((12, 12, 4), 255, dtype=np.uint8)
    raw_pixels[:2, :, 3] = 0
    Image.fromarray(raw_pixels, mode="RGBA").save(raw_path)
    Image.new("RGB", (12, 12), (170, 170, 170)).save(completed_path)

    subject_attributes._validate_attribute_png(
        output_root,
        raw_path.relative_to(output_root).as_posix(),
        final_selection="raw",
    )
    subject_attributes._validate_attribute_png(
        output_root,
        completed_path.relative_to(output_root).as_posix(),
        final_selection="completed",
    )
    with pytest.raises(ValueError, match="raw attribute artifact must be RGBA"):
        subject_attributes._validate_attribute_png(
            output_root,
            completed_path.relative_to(output_root).as_posix(),
            final_selection="raw",
        )
    with pytest.raises(ValueError, match="completed attribute artifact must be RGB"):
        subject_attributes._validate_attribute_png(
            output_root,
            raw_path.relative_to(output_root).as_posix(),
            final_selection="completed",
        )


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


def _completion_review(
    *,
    accepted: bool,
) -> subject_attributes.SubjectAttributeCompletionReview:
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
    return subject_attributes.SubjectAttributeCompletionReview(
        verdict="accept" if accepted else "reject",
        reason="usable" if accepted else "identity changed",
        **values,
    )


@pytest.mark.parametrize(
    "false_flag",
    [
        "original_visible_details_preserved",
        "materially_more_complete",
        "no_wrong_new_instance",
        "no_duplicate_component",
        "no_unrelated_content",
        "no_structural_distortion",
        "target_clear_and_prominent",
        "candidate_preferred_over_alpha",
    ],
)
def test_attribute_completion_comparative_flags_fail_closed(
    false_flag: str,
) -> None:
    payload = _completion_review(accepted=True).model_dump(mode="json")
    payload[false_flag] = False
    payload["verdict"] = "reject"

    review = subject_attributes.SubjectAttributeCompletionReview.model_validate(payload)

    assert review.verdict == "reject"
    assert getattr(review, false_flag) is False


def test_attribute_completion_prompt_allows_layout_change_but_requires_improvement() -> None:
    prompt = " ".join(
        subject_attributes.ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT.split()
    )
    assert "material and useful structural completeness improvement" in prompt
    assert "merely redraws an already sufficient source" in prompt
    assert "Reasonable spatial changes are allowed" in prompt
    assert "moved, was recentered, changed size moderately" in prompt
    assert "extreme corner or edge" in prompt
    assert "stretched, compressed, warped, duplicated" in prompt


def test_attribute_completion_qwen_uses_strict_schema_and_normalizes_verdict() -> None:
    payload = _completion_review(accepted=True).model_dump(mode="json")
    payload["verdict"] = "reject"

    class Completions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    completions = Completions()
    judge = subject_attributes.QwenSubjectAttributeCompletionJudge(
        QwenServiceConfig(model="/models/qwen"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            close=lambda: None,
        ),
    )

    review = judge.review(
        source_attribute=Image.new("RGBA", (32, 24), (80, 90, 100, 255)),
        generated_candidate=Image.new("RGB", (40, 36), (80, 90, 100)),
        attribute_type="upper_clothing",
        attribute_phrase="gray robe",
    )

    assert review.verdict == "accept"
    response_format = completions.calls[0]["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "v3_boogu_reference_edit_review",
            "strict": True,
            "schema": (
                subject_attributes.SubjectAttributeCompletionReview.model_json_schema()
            ),
        },
    }


def _attempt_completion_direct(
    tmp_path: Path,
    *,
    judge_accepts: bool = True,
    generated: Image.Image | None = None,
) -> tuple[
    subject_attributes.PendingAttributeCandidate | None,
    str,
    subject_attributes._CompletionSelectionMetrics,
    object,
]:
    raw_alpha = np.ones((64, 64), dtype=bool)
    raw_pixels = np.full((64, 64, 4), 120, dtype=np.uint8)
    raw_pixels[..., 3] = raw_alpha.astype(np.uint8) * 255
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="upper_clothing",
            phrase="gray traditional robe with dark sash",
            grounding_prompt="gray traditional robe with dark sash worn by the man",
        ),
        attribute_id="a3",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_2", slot=1, source_frame_index=10),
        attribute_mask=raw_alpha,
        source_image=Image.new("RGB", (64, 64), "gray"),
        crop=Image.fromarray(raw_pixels, mode="RGBA"),
        geometry=_geometry(),
    )

    if generated is None:
        generated = Image.new("RGB", (64, 64), (180, 180, 180))
    trace = SimpleNamespace(calls=0, events=[], reviewed_image=None)

    class Backend:
        def attribute_completion(self, *, output_path, **_kwargs):
            trace.events.append("boogu")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generated.save(output_path)
            return {"model_call_time_seconds": 0.1}

    class Judge:
        def review(self, *, generated_candidate, **_kwargs):
            trace.events.append("qwen_review")
            trace.calls += 1
            trace.reviewed_image = generated_candidate.copy()
            return _completion_review(accepted=judge_accepts)

    judge = Judge()
    metrics = subject_attributes._CompletionSelectionMetrics()
    completed, reason, _, completion_review, completion_seed = (
        subject_attributes._attempt_attribute_completion(
            candidate,
            clip_uid="42038d7dc619cfa7bebee437",
            output_root=tmp_path,
            completion_config=SubjectAttributeCompletionConfig(enabled=True),
            completion_backend=Backend(),
            completion_judge=judge,
            metrics=metrics,
            raw_usable=True,
        )
    )
    trace.completion_review = completion_review
    trace.completion_seed = completion_seed
    return completed, reason, metrics, trace


def test_completion_uses_boogu_gray_robe_rgb_without_sam_resegmentation(
    tmp_path: Path,
) -> None:
    pixels = np.full((64, 64, 3), 245, dtype=np.uint8)
    pixels[8:56, 12:52] = (170, 170, 170)
    pixels[36:44, 12:52] = (35, 35, 35)
    generated = Image.fromarray(pixels, mode="RGB")
    completed, reason, metrics, trace = _attempt_completion_direct(
        tmp_path,
        generated=generated,
    )

    assert completed is not None
    assert reason == "completion_identity_accepted"
    assert completed.crop.mode == "RGB"
    assert np.array_equal(np.asarray(completed.crop), pixels)
    assert trace.reviewed_image.mode == "RGB"
    assert np.array_equal(np.asarray(trace.reviewed_image), pixels)
    assert trace.events == ["boogu", "qwen_review"]
    assert trace.completion_review == _completion_review(accepted=True)
    assert metrics.sam3_time_seconds == 0.0
    assert metrics.sam_zero_mask_rejects == 0
    assert metrics.sam_single_mask == 0
    assert metrics.sam_multi_mask == 0
    assert metrics.sam_masks_returned_total == 0


def test_completion_qwen_rejects_without_repaired_candidate(tmp_path: Path) -> None:
    completed, reason, metrics, trace = _attempt_completion_direct(
        tmp_path,
        judge_accepts=False,
    )

    assert completed is None
    assert reason == "completion_qwen_review_reject"
    assert metrics.identity_review_rejects == 1
    assert trace.events == ["boogu", "qwen_review"]
    assert "segmentation_prompt" not in _completion_review(
        accepted=True
    ).model_dump()
    assert "segmentation_prompt" not in (
        subject_attributes.ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT
    )


@pytest.mark.parametrize("attribute_type", ["face", "upper_clothing"])
def test_repairable_attribute_uses_boogu_without_completion_sam(
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

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(accepted=True)

    backend = Backend()
    metrics = subject_attributes._CompletionSelectionMetrics()
    config = replace(
        SubjectAttributeCompletionConfig(enabled=True),
        minimum_sharpness_score=-1.0,
        maximum_area_growth_ratio=4.0,
        face_maximum_area_growth_ratio=4.0,
        maximum_bbox_area_growth_ratio=4.0,
        face_maximum_bbox_area_growth_ratio=4.0,
    )
    completed, reason, failed, completion_review, completion_seed = (
        subject_attributes._attempt_attribute_completion(
            candidate,
            clip_uid="clip-1",
            output_root=tmp_path / "subject_attributes",
            completion_config=config,
            completion_backend=backend,
            completion_judge=Judge(),
            metrics=metrics,
            raw_usable=True,
        )
    )

    assert completed is not None
    assert completed.completion_attempted is True
    assert reason == "completion_identity_accepted"
    assert failed is False
    assert completion_review == _completion_review(accepted=True)
    assert isinstance(completion_seed, int)
    assert backend.calls == 1
    assert completed.crop.mode == "RGB"
    assert metrics.sam3_time_seconds == 0.0
    assert metrics.sam_masks_returned_total == 0
    assert metrics.attempts == 1
    assert metrics.accepted == 0


def test_alpha_coverage_does_not_control_completion_routing(
    tmp_path: Path,
) -> None:
    crop = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="face",
            phrase="face",
            grounding_prompt="face of person 1",
        ),
        attribute_id="a1",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_1", slot=1, source_frame_index=10),
        attribute_mask=np.ones((64, 64), dtype=bool),
        source_image=Image.new("RGB", (100, 100), "white"),
        crop=crop,
        geometry=_geometry(),
    )

    captured: dict[str, object] = {}

    class Backend:
        def attribute_completion(self, *, output_path, **kwargs):
            captured.update(kwargs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), "gray").save(output_path)
            return {"model_call_time_seconds": 0.1}

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(accepted=True)

    completed, reason, _, _, _ = subject_attributes._attempt_attribute_completion(
        candidate,
        clip_uid="clip-1",
        output_root=tmp_path,
        completion_config=SubjectAttributeCompletionConfig(enabled=True),
        completion_backend=Backend(),
        completion_judge=Judge(),
        metrics=subject_attributes._CompletionSelectionMetrics(),
        raw_usable=False,
    )
    assert completed is not None
    assert reason == "completion_identity_accepted"
    assert captured["instruction"] == (
        "补全成完整的人脸，去除掉破碎的部分，不添加其他内容"
    )


def test_completion_review_rejects_fail_closed(
    tmp_path: Path,
) -> None:
    raw_mask = np.zeros((32, 32), dtype=bool)
    raw_mask[8:24, 8:24] = True
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="face",
            phrase="face",
            grounding_prompt="face of person 1",
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

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(accepted=False)

    metrics = subject_attributes._CompletionSelectionMetrics()
    completed, reason, _, completion_review, _ = (
        subject_attributes._attempt_attribute_completion(
            candidate,
            clip_uid="clip-1",
            output_root=tmp_path,
            completion_config=replace(
                SubjectAttributeCompletionConfig(enabled=True),
                maximum_alpha_coverage_without_completion=1.0,
            ),
            completion_backend=Backend(),
            completion_judge=Judge(),
            metrics=metrics,
            raw_usable=False,
        )
    )
    assert completed is None
    assert reason == "completion_qwen_review_reject"
    assert metrics.qwen_review_rejects == 1
    assert completion_review == _completion_review(accepted=False)


def _routing_review(attribute_id: str, kind: str) -> SubjectAttributeReview:
    accepted = _accepted_review(attribute_id)
    if kind == "accepted_repair":
        return accepted.model_copy(
            update={
                "structure_complete": False,
                "completion_recommended": True,
                "reason": "usable but visibly incomplete",
            }
        )
    if kind == "unusable_repair":
        return accepted.model_copy(
            update={
                "recognizable": False,
                "structure_complete": False,
                "completion_recommended": True,
                "reason": "correct repairable component",
            }
        )
    if kind == "wrong_semantic":
        return accepted.model_copy(
            update={
                "matches_attribute": False,
                "structure_complete": False,
                "reason": "wrong component",
            }
        )
    if kind == "wrong_owner":
        return accepted.model_copy(
            update={
                "owner_binding_correct": False,
                "structure_complete": False,
                "reason": "wrong owner",
            }
        )
    if kind == "unusable_no_repair":
        return accepted.model_copy(
            update={
                "recognizable": False,
                "reason": "unusable and not repairable",
            }
        )
    if kind == "accepted_complete":
        return accepted
    raise AssertionError(f"unknown review kind: {kind}")


def _run_completion_routing_case(
    tmp_path: Path,
    *,
    raw_kind: str,
    completion_stage: str = "success",
    attribute_count: int = 1,
    first_attribute_type: str = "face",
) -> tuple[OwnerEnrichmentArtifact, object, object, object]:
    _frames(tmp_path / "run")

    class Discovery:
        def discover(self, *, owner, **_kwargs):
            return SubjectAttributeDiscovery(
                owner_entity_id=owner.entity_id,
                owner_is_human=True,
                attributes=[
                    DiscoveredSubjectAttribute(
                        attribute_type=(
                            first_attribute_type if index == 0 else "accessory"
                        ),
                        phrase=(
                            first_attribute_type if index == 0 else "necklace"
                        ),
                        grounding_prompt=(
                            f"{first_attribute_type} of person 1"
                            if index == 0
                            else "necklace worn by person 1"
                        ),
                    )
                    for index in range(attribute_count)
                ],
            )

    class Review:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.crops: list[list[Image.Image]] = []

        def review(self, *, owner, candidates):
            self.calls.append([candidate.attribute_id for candidate in candidates])
            self.crops.append([candidate.crop.copy() for candidate in candidates])
            if len(self.calls) != 1:
                raise AssertionError("completed attributes must not be reviewed again")
            return SubjectAttributeReviewBatch(
                owner_entity_id=owner.entity_id,
                reviews=[
                    _routing_review(candidate.attribute_id, raw_kind)
                    for candidate in candidates
                ],
            )

    class Backend:
        def __init__(self) -> None:
            self.calls = 0
            self.background_calls = 0

        def attribute_completion(self, *, output_path, **_kwargs):
            self.calls += 1
            if completion_stage == "backend_failure":
                raise RuntimeError("backend unavailable")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((100, 100, 3), 245, dtype=np.uint8)
            pixels[15:90, 20:80] = (170, 170, 170)
            pixels[58:68, 20:80] = (35, 35, 35)
            Image.fromarray(pixels, mode="RGB").save(output_path)
            return {"model_call_time_seconds": 0.1}

        def attribute_background(self, **_kwargs):
            self.background_calls += 1
            raise AssertionError("attribute background is disabled by policy")

    class Segmenter:
        def __init__(self) -> None:
            self.frame_calls = 0
            self.generated_calls = 0

        def segment_frame(self, **kwargs):
            self.frame_calls += 1
            mask = np.zeros((100, 100), dtype=bool)
            if "necklace" in kwargs["grounding_prompt"]:
                mask[45:65, 25:75] = True
            else:
                mask[30:70, 30:70] = True
            return (mask,)

        def segment_generated_frame(self, **kwargs):
            self.generated_calls += 1
            raise AssertionError("completion path must not call SAM3")

    class Judge:
        def review(self, **_kwargs):
            return _completion_review(
                accepted=completion_stage != "identity_reject"
            )

    review = Review()
    backend = Backend()
    segmenter = Segmenter()
    clip = _clip()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(
            pair=SimpleNamespace(crop_padding_ratio=0.08),
            subject_attributes=SubjectAttributesConfig(
                completion=replace(
                    SubjectAttributeCompletionConfig(enabled=True),
                    minimum_sharpness_score=-1.0,
                    maximum_area_growth_ratio=4.0,
                    face_maximum_area_growth_ratio=4.0,
                    maximum_bbox_area_growth_ratio=4.0,
                    face_maximum_bbox_area_growth_ratio=4.0,
                )
            ),
        ),
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
        discovery_client=Discovery(),
        review_client=review,
        segmentation_backend=segmenter,
        completion_backend=backend,
        completion_judge=Judge(),
    )
    return artifact, review, backend, segmenter


def _run_two_candidate_routing_case(
    tmp_path: Path,
    *,
    raw_kinds: tuple[str, str],
    completion_accepts: tuple[bool, ...] = (),
    attribute_type: str = "face",
    bbox_accepts: bool = False,
):
    _frames(tmp_path / "run")

    class Discovery:
        def discover(self, *, owner, **_kwargs):
            return SubjectAttributeDiscovery(
                owner_entity_id=owner.entity_id,
                owner_is_human=True,
                attributes=[
                    DiscoveredSubjectAttribute(
                        attribute_type=attribute_type,
                        phrase=f"distinct {attribute_type}",
                        grounding_prompt=f"distinct {attribute_type} of person 1",
                    )
                ],
            )

    class Review:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.candidate_calls: list[list[str]] = []
            self.bbox_calls = 0

        def review(self, *, owner, candidates):
            self.calls.append([candidate.attribute_id for candidate in candidates])
            self.candidate_calls.append(
                [candidate.owner_candidate.candidate_id for candidate in candidates]
            )
            return SubjectAttributeReviewBatch(
                owner_entity_id=owner.entity_id,
                reviews=[
                    _routing_review(
                        candidate.attribute_id,
                        raw_kinds[
                            0
                            if candidate.owner_candidate.candidate_id == "candidate_1"
                            else 1
                        ],
                    )
                    for candidate in candidates
                ],
            )

        def review_attribute_bbox(self, **_kwargs):
            self.bbox_calls += 1
            return _attribute_variant_review(accepted=bbox_accepts)

    class Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def attribute_completion(self, *, source_path, output_path, **_kwargs):
            self.calls.append(source_path.parents[1].name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 100), (180, 180, 180)).save(output_path)
            return {"model_call_time_seconds": 0.1}

    class Judge:
        def __init__(self) -> None:
            self.calls = 0

        def review(self, **_kwargs):
            accepted = completion_accepts[self.calls]
            self.calls += 1
            return _completion_review(accepted=accepted)

    class Segmenter:
        def segment_frame(self, **_kwargs):
            mask = np.zeros((100, 100), dtype=bool)
            mask[30:70, 30:70] = True
            return (mask,)

    review = Review()
    backend = Backend()
    judge = Judge()
    clip = _clip()
    artifact = subject_attributes._process_owner(
        config=SimpleNamespace(
            pair=SimpleNamespace(crop_padding_ratio=0.08),
            subject_attributes=SubjectAttributesConfig(
                completion=replace(
                    SubjectAttributeCompletionConfig(enabled=True),
                    minimum_sharpness_score=-1.0,
                )
            ),
        ),
        storage=_FakeStorage(tmp_path / "run"),
        output_root=tmp_path / "output",
        clip=clip,
        owner=clip.annotation.entities[0],
        owner_candidates=[
            _candidate("candidate_1", slot=1, source_frame_index=10),
            _candidate("candidate_2", slot=2, source_frame_index=20),
        ],
        masks=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=100,
            width=100,
            entities={},
        ),
        attribute_id_start=1,
        discovery_client=Discovery(),
        review_client=review,
        segmentation_backend=Segmenter(),
        completion_backend=backend,
        completion_judge=judge,
    )
    return artifact, review, backend, judge


def test_candidate2_completion_is_tried_after_candidate1_qwen_reject(
    tmp_path: Path,
) -> None:
    artifact, review, backend, judge = _run_two_candidate_routing_case(
        tmp_path,
        raw_kinds=("accepted_complete", "accepted_complete"),
        completion_accepts=(False, True),
    )

    record = artifact.records[0]
    assert record.final_selection == "completed"
    assert record.owner_candidate_id == "candidate_2"
    assert review.candidate_calls == [["candidate_1"], ["candidate_2"]]
    assert backend.calls == ["candidate_1", "candidate_2"]
    assert judge.calls == 2
    assert artifact.metrics.review_calls == 2
    assert artifact.metrics.attribute_source_candidates_considered == 2
    assert artifact.metrics.attribute_second_candidate_attempts == 1
    assert artifact.metrics.attribute_completion_candidate1_accepted == 0
    assert artifact.metrics.attribute_completion_candidate2_accepted == 1


def test_two_completion_rejects_fall_back_to_highest_ranked_accepted_alpha(
    tmp_path: Path,
) -> None:
    artifact, review, backend, _ = _run_two_candidate_routing_case(
        tmp_path,
        raw_kinds=("accepted_complete", "accepted_complete"),
        completion_accepts=(False, False),
    )

    record = artifact.records[0]
    assert record.final_selection == "raw"
    assert record.owner_candidate_id == "candidate_1"
    assert review.bbox_calls == 0
    assert backend.calls == ["candidate_1", "candidate_2"]
    assert artifact.metrics.completion_fallback_to_raw == 1
    assert artifact.metrics.attribute_bbox_fallback_attempts == 0


def test_wrong_semantic_candidate1_skips_boogu_and_candidate2_can_complete(
    tmp_path: Path,
) -> None:
    artifact, review, backend, judge = _run_two_candidate_routing_case(
        tmp_path,
        raw_kinds=("wrong_semantic", "accepted_complete"),
        completion_accepts=(True,),
    )

    record = artifact.records[0]
    assert record.final_selection == "completed"
    assert record.owner_candidate_id == "candidate_2"
    assert review.candidate_calls == [["candidate_1"], ["candidate_2"]]
    assert backend.calls == ["candidate_2"]
    assert judge.calls == 1


def test_noncompletion_type_uses_candidate2_without_boogu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject_attributes, "HAIR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS", 4)
    artifact, review, backend, judge = _run_two_candidate_routing_case(
        tmp_path,
        raw_kinds=("wrong_semantic", "accepted_complete"),
        attribute_type="hair",
    )

    record = artifact.records[0]
    assert record.final_selection == "raw"
    assert record.owner_candidate_id == "candidate_2"
    assert review.candidate_calls == [["candidate_1"], ["candidate_2"]]
    assert backend.calls == []
    assert judge.calls == 0


def test_bbox_is_reviewed_only_after_both_candidates_lack_accepted_alpha(
    tmp_path: Path,
) -> None:
    artifact, review, backend, _ = _run_two_candidate_routing_case(
        tmp_path,
        raw_kinds=("unusable_repair", "unusable_repair"),
        completion_accepts=(False, False),
        bbox_accepts=True,
    )

    record = artifact.records[0]
    assert record.final_selection == "bbox"
    assert record.default_variant == "bbox"
    assert record.accepted_base_image_path is None
    assert review.bbox_calls == 1
    assert backend.calls == ["candidate_1", "candidate_2"]
    assert artifact.metrics.attribute_bbox_fallback_attempts == 1
    assert artifact.metrics.attribute_bbox_fallback_accepted == 1


def test_raw_accepted_complete_still_compares_boogu(tmp_path: Path) -> None:
    artifact, review, backend, segmenter = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_complete",
    )

    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.final_selection == "completed"
    assert record.completion_attempted is True
    assert backend.calls == 1
    assert segmenter.generated_calls == 0
    assert backend.background_calls == 0
    assert review.calls == [["a1"]]


def test_hair_repair_recommendation_never_calls_boogu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject_attributes,
        "HAIR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS",
        4,
    )
    artifact, _, backend, _ = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_repair",
        first_attribute_type="hair",
    )

    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.final_selection == "raw"
    assert record.completion_attempted is False
    assert record.image_path is not None
    assert (tmp_path / "output" / record.image_path).is_file()
    assert backend.calls == 0
    assert backend.background_calls == 0


def test_glasses_publish_raw_without_boogu(tmp_path: Path) -> None:
    artifact, _, backend, _ = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_complete",
        first_attribute_type="glasses",
    )

    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.final_selection == "raw"
    assert record.completion_attempted is False
    assert record.image_path is not None
    assert (tmp_path / "output" / record.image_path).is_file()
    assert backend.calls == 0
    assert backend.background_calls == 0


@pytest.mark.parametrize(
    "attribute_type",
    [
        "face",
        "hair",
        "headwear",
        "glasses",
        "upper_clothing",
        "lower_clothing",
        "dress_or_skirt",
        "shoes",
        "bag",
        "accessory",
    ],
)
def test_fresh_attribute_never_calls_background_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute_type: str,
) -> None:
    monkeypatch.setattr(subject_attributes, "HAIR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS", 4)
    monkeypatch.setattr(
        subject_attributes,
        "HEADWEAR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS",
        4,
    )

    artifact, _, backend, _ = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_complete",
        first_attribute_type=attribute_type,
    )

    assert artifact.records[0].status == "accepted"
    assert backend.background_calls == 0
    assert artifact.metrics.attribute_background_variants_attempted == 0
    assert artifact.metrics.attribute_background_variants_accepted == 0
    assert artifact.metrics.attribute_bbox_reviews_skipped_background_accepted == 0


@pytest.mark.parametrize(
    "failure_stage",
    [
        "backend_failure",
        "identity_reject",
    ],
)
def test_raw_accepted_repair_failures_fallback_to_raw(
    tmp_path: Path,
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject_attributes, "new_boogu_seed", lambda: 2718)
    artifact, review, backend, segmenter = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_repair",
        completion_stage=failure_stage,
    )

    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.final_selection == "raw"
    assert record.completion_attempted is True
    assert record.completion_seed == 2718
    assert artifact.metrics.completion_fallback_to_raw == 1
    assert artifact.metrics.completion_selected_completed == 0
    assert artifact.metrics.completion_review_calls == int(
        failure_stage == "identity_reject"
    )
    assert backend.calls == 1
    assert backend.background_calls == 0
    assert segmenter.generated_calls == 0
    assert review.calls == [["a1"]]
    assert record.completion_review == (
        _completion_review(accepted=False)
        if failure_stage == "identity_reject"
        else None
    )
    assert record.image_path is not None
    with Image.open(tmp_path / "output" / record.image_path) as published:
        assert published.mode == "RGBA"


def test_raw_accepted_repair_success_selects_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject_attributes, "new_boogu_seed", lambda: 314159)
    artifact, review, backend, segmenter = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_repair",
    )

    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.final_selection == "completed"
    assert record.completion_outcome == "selected_completed"
    assert record.completion_seed == 314159
    assert artifact.metrics.review_calls == 1
    assert artifact.metrics.completion_selected_completed == 1
    assert artifact.metrics.completion_accepted == 1
    assert artifact.metrics.completion_review_calls == 1
    assert artifact.metrics.completion_final_review_rejects == 0
    assert artifact.metrics.repaired_attribute_final_review_accepted == 0
    assert artifact.metrics.repaired_attribute_final_review_rejected == 0
    assert artifact.metrics.completion_sam_zero_mask_rejects == 0
    assert artifact.metrics.completion_sam_single_mask == 0
    assert artifact.metrics.completion_sam_multi_mask == 0
    assert artifact.metrics.completion_sam_masks_returned_total == 0
    assert artifact.metrics.completion_attempts_by_type == {"face": 1}
    assert artifact.metrics.completion_accepted_by_type == {"face": 1}
    assert review.calls == [["a1"]]
    assert backend.calls == 1
    assert backend.background_calls == 0
    assert segmenter.generated_calls == 0
    assert review.crops[0][0].mode == "RGBA"
    assert record.completion_review == _completion_review(accepted=True)
    assert record.image_path is not None
    with Image.open(tmp_path / "output" / record.image_path) as published:
        published.load()
        assert published.mode == "RGB"
        expected = np.full((100, 100, 3), 245, dtype=np.uint8)
        expected[15:90, 20:80] = (170, 170, 170)
        expected[58:68, 20:80] = (35, 35, 35)
        assert np.array_equal(np.asarray(published), expected)


def test_raw_review_is_owner_batched_without_repaired_review(tmp_path: Path) -> None:
    artifact, review, backend, segmenter = _run_completion_routing_case(
        tmp_path,
        raw_kind="accepted_repair",
        attribute_count=2,
    )

    assert [record.final_selection for record in artifact.records] == [
        "completed",
        "completed",
    ]
    assert review.calls == [["a1", "a2"]]
    assert artifact.metrics.review_calls == 1
    assert backend.calls == 2
    assert backend.background_calls == 0
    assert segmenter.generated_calls == 0


@pytest.mark.parametrize(
    ("completion_stage", "expected_status"),
    [
        ("success", "accepted"),
        ("identity_reject", "rejected"),
        ("backend_failure", "rejected"),
    ],
)
def test_raw_unusable_repair_is_completed_or_rejected_fail_closed(
    tmp_path: Path,
    completion_stage: str,
    expected_status: str,
) -> None:
    artifact, _, _, _ = _run_completion_routing_case(
        tmp_path,
        raw_kind="unusable_repair",
        completion_stage=completion_stage,
    )

    record = artifact.records[0]
    assert record.status == expected_status
    assert artifact.metrics.completion_raw_unusable_attempts == 1
    assert artifact.metrics.completion_review_calls == int(
        completion_stage != "backend_failure"
    )
    if expected_status == "accepted":
        assert record.final_selection == "completed"
        assert record.completion_review == _completion_review(accepted=True)
    else:
        assert record.image_path is None
        assert artifact.metrics.completion_rejected == 1
        if completion_stage == "identity_reject":
            assert record.completion_review == _completion_review(accepted=False)


@pytest.mark.parametrize(
    "raw_kind",
    ["wrong_semantic", "wrong_owner"],
)
def test_hard_rejected_or_nonrepairable_raw_never_reaches_boogu(
    tmp_path: Path,
    raw_kind: str,
) -> None:
    artifact, review, backend, segmenter = _run_completion_routing_case(
        tmp_path,
        raw_kind=raw_kind,
    )

    assert artifact.records[0].status == "rejected"
    assert artifact.metrics.completion_attempts == 0
    assert artifact.metrics.raw_attribute_review_hard_rejected == 1
    assert backend.calls == segmenter.generated_calls == 0
    assert backend.background_calls == 0
    assert review.calls == [["a1"]]


def _attribute_variant_review(*, accepted: bool):
    fields = (
        "correct_attribute",
        "owner_binding_correct",
        "target_is_dominant_and_identifiable",
        "no_large_competing_attribute_or_entity",
        "context_is_limited_and_supportive",
        "no_strong_owner_pose_or_scene_leakage",
        "no_severe_blur_or_artifact",
        "usable_as_attribute_condition",
        "certain",
    )
    return subject_attributes.SubjectAttributeBboxReview(
        **{field: accepted for field in fields},
        verdict="accept" if accepted else "reject",
        reason="usable" if accepted else "not usable",
    )


def _run_attribute_variant_case(
    tmp_path: Path,
    *,
    bbox_accepts: bool,
    bbox_failure: bool = False,
    completed_base: bool = False,
):
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 25:75] = True
    raw_crop = Image.new("RGBA", (54, 44), (120, 120, 120, 255))
    candidate = subject_attributes.PendingAttributeCandidate(
        discovered=DiscoveredSubjectAttribute(
            attribute_type="upper_clothing",
            phrase="gray robe",
            grounding_prompt="gray robe worn by person 1",
        ),
        attribute_id="a1",
        owner_entity_id="e1",
        owner_candidate=_candidate("candidate_1", slot=1, source_frame_index=10),
        attribute_mask=mask,
        source_image=Image.new("RGB", (100, 100), (240, 240, 240)),
        crop=(
            Image.new("RGB", (64, 64), (140, 140, 140))
            if completed_base
            else raw_crop
        ),
        raw_crop=raw_crop,
        geometry=_geometry(),
    )
    raw_review = (
        _routing_review("a1", "unusable_repair")
        if completed_base
        else _accepted_review("a1")
    )
    record = subject_attributes._accepted_record(
        candidate,
        output_root=tmp_path,
        sample_id="clip-1",
        owner_reference=_ready_reference("e1"),
        review=raw_review,
        completion_review=(
            _completion_review(accepted=True) if completed_base else None
        ),
        final_selection="completed" if completed_base else "raw",
        completion_attempted=completed_base,
        completion_outcome=(
            "selected_completed" if completed_base else "not_attempted"
        ),
        completion_seed=17 if completed_base else None,
        crop_padding_ratio=0.08,
    )

    class Judge:
        bbox_calls = 0

        def review_attribute_bbox(self, **_kwargs):
            self.bbox_calls += 1
            if bbox_failure:
                raise RuntimeError("invalid structured review")
            return _attribute_variant_review(accepted=bbox_accepts)

    judge = Judge()
    metrics = subject_attributes._AttributeVariantMetrics()
    selected = subject_attributes._with_attribute_variants(
        record,
        candidate,
        output_root=tmp_path,
        judge=judge,
        metrics=metrics,
    )
    return selected, judge, metrics


@pytest.mark.parametrize("bbox_accepts", [True, False])
def test_attribute_variants_use_bbox_and_never_generate_background(
    tmp_path: Path,
    bbox_accepts: bool,
) -> None:
    record, judge, metrics = _run_attribute_variant_case(
        tmp_path,
        bbox_accepts=bbox_accepts,
    )

    assert judge.bbox_calls == 0
    assert record.default_variant == "accepted_base"
    assert record.variants is not None
    assert record.variants.alpha.image_path is not None
    assert (tmp_path / record.variants.alpha.image_path).is_file()
    assert record.variants.bbox.image_path is not None
    assert (tmp_path / record.variants.bbox.image_path).is_file()
    assert record.variants.generated_background.model_dump(mode="json") == {
        "image_path": None,
        "status": "unavailable",
        "reviewed": False,
        "review_status": "not_applicable",
        "reason": "attribute_background_disabled_by_policy",
        "synthetic": True,
        "metadata_path": None,
        "source_frame_index": 10,
    }
    assert metrics.bbox_variants_materialized == 1
    assert metrics.bbox_reviews_attempted == 0


def test_attribute_bbox_judge_failure_preserves_available_path_and_base_default(
    tmp_path: Path,
) -> None:
    record, judge, _ = _run_attribute_variant_case(
        tmp_path,
        bbox_accepts=False,
        bbox_failure=True,
    )

    assert judge.bbox_calls == 0
    assert record.variants is not None
    assert record.variants.bbox.status == "available"
    assert record.variants.bbox.reviewed is False
    assert record.variants.bbox.review_status == "not_reviewed"
    assert record.variants.bbox.image_path is not None
    assert (tmp_path / record.variants.bbox.image_path).is_file()
    assert record.default_variant == "accepted_base"
    assert record.image_path == record.accepted_base_image_path


def test_completed_attribute_variant_fallback_preserves_accepted_base(
    tmp_path: Path,
) -> None:
    record, judge, _ = _run_attribute_variant_case(
        tmp_path,
        bbox_accepts=False,
        completed_base=True,
    )

    assert judge.bbox_calls == 0
    assert record.final_selection == "completed"
    assert record.default_variant == "accepted_base"
    assert record.image_path == record.accepted_base_image_path
    assert record.variants is not None
    assert record.variants.alpha.status == "rejected"
    with Image.open(tmp_path / record.image_path) as opened:
        assert opened.mode == "RGB"


def test_legacy_attribute_generated_background_is_read_but_not_selected(
    tmp_path: Path,
) -> None:
    record, _, _ = _run_attribute_variant_case(tmp_path, bbox_accepts=False)
    assert record.variants is not None
    legacy_background = tmp_path / "variants/clip-1/a1/generated_background.png"
    legacy_background.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "blue").save(legacy_background)
    payload = record.model_dump(mode="json")
    payload["image_path"] = legacy_background.relative_to(tmp_path).as_posix()
    payload["default_variant"] = "generated_background"
    payload["default_image_path"] = payload["image_path"]
    payload["default_reason"] = "attribute_background_review_accepted"
    payload["variants"]["generated_background"] = {
        "image_path": payload["image_path"],
        "status": "accepted",
        "reviewed": True,
        "review_status": "accept",
        "reason": "legacy accepted background",
        "synthetic": True,
        "metadata_path": None,
        "source_frame_index": 10,
    }

    normalized = SubjectAttributeRecord.model_validate(payload)

    assert legacy_background.is_file()
    assert normalized.default_variant == "accepted_base"
    assert normalized.image_path == normalized.accepted_base_image_path
    assert normalized.variants is not None
    assert normalized.variants.generated_background.image_path is None
    assert normalized.variants.generated_background.status == "unavailable"
