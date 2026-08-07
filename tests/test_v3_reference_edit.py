from r2v_data_v2.v3.reference_edit import _rejected_reference
from r2v_data_v2.v3.schemas import EntityReferenceState


def test_reference_edit_rejection_preserves_objective_source_evidence() -> None:
    source = EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="local",
        visible_region="central",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="minor crop",
        image_path="clips/clip-1/selected/e1.png",
        source_frame_index=5,
        image_quality="acceptable",
        completeness="repairable",
        viewpoint="front",
        independent_reference_value=True,
        requires_substantial_invention=False,
        primary_identity_region_visible=True,
        major_structure_visible=True,
        truncation_severity="minor",
        discrete_foreground_instance=True,
        mask_matches_target=True,
        completion_needed_for_reference_use=True,
        detached_target_fragments_present=True,
    )

    rejected = _rejected_reference(source, "repairable_completion_rejected:test")

    assert rejected.status == "rejected"
    assert rejected.primary_identity_region_visible is True
    assert rejected.major_structure_visible is True
    assert rejected.truncation_severity == "minor"
    assert rejected.discrete_foreground_instance is True
    assert rejected.mask_matches_target is True
    assert rejected.completion_needed_for_reference_use is True
    assert rejected.detached_target_fragments_present is True
    restored_source = EntityReferenceState.model_validate(
        source.model_dump(mode="json")
    )
    assert restored_source.completion_needed_for_reference_use is True
    assert restored_source.detached_target_fragments_present is True
