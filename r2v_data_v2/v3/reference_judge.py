from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from openai import BadRequestError, OpenAI
from PIL import Image

from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    build_structured_repair_prompt,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import (
    get_model_profile_context,
    model_profile_context,
    profiled_openai_call,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    RawEntityReferenceDecision,
)

if TYPE_CHECKING:
    from r2v_data_v2.v3.pair import EntityReferenceCandidate

SYSTEM_PROMPT = """You select the best reference image for one known entity.

Candidate IDs and candidate order are immutable. Judge only visible evidence and
never infer a body, structure, accessory, or identity feature outside the
images. A full reference may still have minor edge contact or small peripheral
parts outside the frame; do not downgrade only because of border contact,
natural holes, or multiple natural connected components. Local means a
coherent, reusable, identity-bearing but incomplete visual region; it may still
be recognizable as the whole known entity when enough identity and structure
remain visible. Reject when identity is not visible, fragmentation or occlusion
is severe, or segmentation is wrong. The context image shows scene placement;
the isolated crop shows the
exact proposed reference content.

Classify viewpoint as front, three_quarter, side, rear, or not_applicable.
Subjects must use a directional viewpoint; objects and groups must use
not_applicable. Back visibility is not identity visibility, and clothing alone
is not identity evidence. Reject rear-only subjects. A side-view subject is
usable only when the face or other explicit identity features remain visible.

independent_reference_value means the isolated visual content is independently
useful as a generation condition. Reject wall patches, scenery fragments,
background regions, and non-target mask fragments that have no independent
reference value. requires_substantial_invention means successful completion
would need the model to invent major identity or structure. Reject whenever it
is true; later Boogu capability is never a reason to mark such a crop usable.

Classify image_quality as high, acceptable, or poor. Classify completeness as
one routing outcome: complete, repairable, local_usable, severely_incomplete,
or fragmented. complete is mostly whole with clear identity. repairable is
limited to minor, local, low-risk missing content, such as a small limb
terminal or small object part, while the source preserves identity and most of
the main object structure and completion is genuinely required before the crop
is reusable. local_usable includes natural local framing: it is a stable,
structurally coherent, independently reusable, identity-bearing local view that
requires no generative completion. It need not show a complete body or object.
A person shown clearly from the head through the torso or waist can be a valid
local_usable reference even when the legs or lower body are outside the camera
framing. Do not classify a coherent portrait, head-and-shoulders view, upper-body
crop, side view, or three-quarter view as repairable merely because the whole
body is not visible. The distinction is whether the visible local reference is
itself reusable, not whether the full physical entity is present.
severely_incomplete means identity, the head, most of the body, most clothing,
or another major structure is missing, or completion would require large
invention. Crops cut through the torso, hips, buttocks, most legs or arms, and
torso-only or back-only fragments belong here; only a small limb terminal may
remain repairable. fragmented means a
bad mask, environment fragment, or non-target content. severely_incomplete and
fragmented must be rejected. Poor image quality must also be rejected. A
single connected component is not evidence that the entity is complete.

Report objective visual evidence separately. primary_identity_region_visible
means a subject's face, head, or another genuinely identity-bearing region is
visible; a back, clothing, trousers, or torso silhouette alone is insufficient.
For objects and groups it means the primary recognizable visual region is
present. major_structure_visible means most of the coherent subject or object
structure remains, rather than only a torso, back, partial clothing, or
disconnected pieces. Classify truncation_severity as none, minor, or major.
repairable permits only minor truncation. Missing identity regions, back-only or
torso-only subjects, and substantial missing body or object structure are major
and must reject. Boogu's ability to invent content is never a reason to retain a
majorly truncated source.

discrete_foreground_instance is false for environment regions, wall or terrain
patches, water surfaces, background architecture or murals, negative space, and
other scene fragments that are not independently reusable foreground instances.
Judge visual structure, not an object-name blacklist. mask_matches_target is
false when the crop mainly depicts another entity or contains large unrelated
regions, such as a snorkel mask dominated by unrelated diving equipment.
Multiple large separated components in a non-group mask strongly suggest
fragmentation or target mismatch. Use the supplied component diagnostics along
with both the context image and isolated crop.

completion_needed_for_reference_use states whether generative completion is
strictly necessary before this crop can serve as a training reference. Missing
legs or lower body outside natural camera framing is not by itself a defect and
must be false when the visible portrait or local object region is already
coherent and reusable. Set it true only for a small, real, low-risk truncation,
such as a clipped hand or foot terminal, short limb edge, or small object edge,
whose repair is necessary for reference use without guessing identity or major
structure. Minor truncation does not automatically mean repairable.

Natural local framing and detached target fragments are different. A coherent
portrait or upper-body crop can be local_usable when the lower body is simply
outside the camera framing and no detached target body parts remain elsewhere
in the mask. If the isolated crop contains non-trivial pieces of the same target
that are visibly separated from the main body or object, such as floating legs,
hands, limbs, or clothing/body pieces, this is not natural local framing. Do not
call a crop local_usable merely because its largest component forms a good
portrait when non-trivial detached pieces of the same subject remain elsewhere
in the isolated crop.

When identity and most major structure remain and detached pieces are limited
and repairable, set detached_target_fragments_present=true,
truncation_severity=minor, completion_needed_for_reference_use=true, and
completeness=repairable. Reject when fragmentation or missing structure is
extensive. Transparent background, natural holes, tiny noise, and naturally
separate members of a group are not detached target fragments. Use
significant_component_count, largest_component_ratio,
second_largest_component_ratio, and nontrivial_detached_component_signal as
supporting evidence.

Do not output tokens or crop coordinates. Return one strict JSON object only
with selected_candidate_id, image_quality, completeness, reference_scope,
visible_region, whole_entity_recognizable, identity_features_visible,
viewpoint, independent_reference_value, requires_substantial_invention,
primary_identity_region_visible, major_structure_visible,
truncation_severity, discrete_foreground_instance, mask_matches_target,
completion_needed_for_reference_use, detached_target_fragments_present, and
scope_reason."""


@dataclass(frozen=True)
class EntityReferenceDecisionAttempt:
    decision: RawEntityReferenceDecision
    raw_responses: tuple[str, ...]
    repair_attempts: int


class EntityReferenceJudgeFailure(StructuredOutputFailure):
    pass


class EntityReferenceJudge(Protocol):
    def decide(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> EntityReferenceDecisionAttempt: ...


EvidencePresentationMode = Literal["separate", "paired_card"]


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_paired_candidate_evidence_card(
    context: Image.Image,
    isolated_crop: Image.Image,
    *,
    panel_max_side: int,
) -> Image.Image:
    if (
        isinstance(panel_max_side, bool)
        or not isinstance(panel_max_side, int)
        or panel_max_side <= 0
    ):
        raise ValueError("candidate card panel_max_side must be a positive integer")
    if not isinstance(context, Image.Image) or not isinstance(
        isolated_crop,
        Image.Image,
    ):
        raise TypeError("candidate card inputs must be PIL images")

    card = Image.new(
        "RGB",
        (panel_max_side * 2, panel_max_side),
        (255, 255, 255),
    )
    for panel_index, source in enumerate((context, isolated_crop)):
        thumbnail = source.convert("RGB").copy()
        thumbnail.thumbnail(
            (panel_max_side, panel_max_side),
            Image.Resampling.LANCZOS,
        )
        panel_x = panel_index * panel_max_side
        destination = (
            panel_x + (panel_max_side - thumbnail.width) // 2,
            (panel_max_side - thumbnail.height) // 2,
        )
        card.paste(thumbnail, destination)
    return card


def subject_has_nontrivial_detached_component(
    candidate: EntityReferenceCandidate,
) -> bool:
    existing_large_signal = (
        candidate.significant_component_count >= 2
        and 0.05 <= candidate.second_largest_component_ratio <= 0.20
        and candidate.largest_component_ratio >= 0.70
    )
    if existing_large_signal:
        return True

    from r2v_data_v2.v3.pair import _foreground_components

    components = _foreground_components(candidate.mask)
    if len(components) < 2:
        return False
    main, secondary = components[:2]
    total_foreground = sum(component.area_pixels for component in components)
    secondary_ratio = secondary.area_pixels / total_foreground
    if not (0.005 <= secondary_ratio < 0.05) or secondary.area_pixels < 64:
        return False

    main_x1, main_y1, main_x2, main_y2 = main.bbox_xyxy
    secondary_x1, secondary_y1, secondary_x2, secondary_y2 = secondary.bbox_xyxy
    main_long_side = max(main_x2 - main_x1, main_y2 - main_y1)
    secondary_long_side = max(
        secondary_x2 - secondary_x1,
        secondary_y2 - secondary_y1,
    )
    if secondary_long_side < max(12.0, 0.08 * main_long_side):
        return False

    gap_x = max(
        main_x1 - secondary_x2,
        secondary_x1 - main_x2,
        0,
    )
    gap_y = max(
        main_y1 - secondary_y2,
        secondary_y1 - main_y2,
        0,
    )
    if math.hypot(gap_x, gap_y) < max(4.0, 0.02 * main_long_side):
        return False

    union_x1 = min(main_x1, secondary_x1)
    union_y1 = min(main_y1, secondary_y1)
    union_x2 = max(main_x2, secondary_x2)
    union_y2 = max(main_y2, secondary_y2)
    main_bbox_area = (main_x2 - main_x1) * (main_y2 - main_y1)
    union_bbox_area = (union_x2 - union_x1) * (union_y2 - union_y1)
    maximum_extension = max(
        main_x1 - union_x1,
        main_y1 - union_y1,
        union_x2 - main_x2,
        union_y2 - main_y2,
    )
    return (
        union_bbox_area / main_bbox_area >= 1.15
        or maximum_extension >= 0.10 * main_long_side
    )


def build_entity_reference_request_payload(
    entity: AnnotationEntity,
    candidates: list[EntityReferenceCandidate],
) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "phrase": entity.phrase,
        "grounding_prompt": entity.grounding_prompt,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "frame_slot": candidate.frame_slot,
                "source_frame_index": candidate.source_frame_index,
                "area_ratio": candidate.area_ratio,
                "bbox_fill_ratio": candidate.bbox_fill_ratio,
                "border_contact_count": candidate.border_contact_count,
                "sharpness_score": candidate.sharpness_score,
                "significant_component_count": (
                    candidate.significant_component_count
                ),
                "largest_component_ratio": candidate.largest_component_ratio,
                "second_largest_component_ratio": (
                    candidate.second_largest_component_ratio
                ),
                "nontrivial_detached_component_signal": (
                    entity.reference_type == "subject"
                    and subject_has_nontrivial_detached_component(candidate)
                ),
            }
            for candidate in candidates
        ],
    }


def validate_entity_reference_decision(
    decision: RawEntityReferenceDecision,
    *,
    candidate_ids: set[str],
    reference_type: str,
    candidate_by_id: dict[str, EntityReferenceCandidate] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    selected = decision.selected_candidate_id
    if selected is not None and selected not in candidate_ids:
        issues.append(
            ValidationIssue(
                code="unknown_candidate_id",
                field="selected_candidate_id",
                message=f"unknown candidate ID: {selected}",
            )
        )
    selected_candidate = (
        candidate_by_id.get(selected)
        if candidate_by_id is not None and selected is not None
        else None
    )
    selected_subject_has_detached_signal = (
        reference_type == "subject"
        and selected_candidate is not None
        and subject_has_nontrivial_detached_component(selected_candidate)
    )
    expected_scope = {
        "complete": "full",
        "repairable": "local",
        "local_usable": "local",
        "severely_incomplete": "reject",
        "fragmented": "reject",
    }[decision.completeness]
    if decision.reference_scope != expected_scope:
        issues.append(
            ValidationIssue(
                code="completeness_scope_mismatch",
                field="reference_scope",
                message="reference scope must match completeness routing",
            )
        )
    if decision.image_quality == "poor" and decision.reference_scope != "reject":
        issues.append(
            ValidationIssue(
                code="poor_quality_not_rejected",
                field="image_quality",
                message="poor image quality must reject the reference",
            )
        )
    if reference_type == "subject":
        if decision.viewpoint == "not_applicable":
            issues.append(
                ValidationIssue(
                    code="subject_viewpoint_required",
                    field="viewpoint",
                    message="subject reference requires a directional viewpoint",
                )
            )
        elif decision.viewpoint == "rear" and decision.reference_scope != "reject":
            issues.append(
                ValidationIssue(
                    code="rear_subject_not_rejected",
                    field="viewpoint",
                    message="rear-only subject reference must be rejected",
                )
            )
        elif (
            decision.viewpoint == "side"
            and not decision.identity_features_visible
            and decision.reference_scope != "reject"
        ):
            issues.append(
                ValidationIssue(
                    code="side_subject_requires_identity",
                    field="identity_features_visible",
                    message="side subject requires visible identity features",
                )
            )
    elif decision.viewpoint != "not_applicable":
        issues.append(
            ValidationIssue(
                code="non_subject_viewpoint_not_applicable",
                field="viewpoint",
                message="object and group viewpoints must be not_applicable",
            )
        )
    if (
        not decision.independent_reference_value
        and decision.reference_scope != "reject"
    ):
        issues.append(
            ValidationIssue(
                code="no_independent_reference_value",
                field="independent_reference_value",
                message="candidate without independent reference value must reject",
            )
        )
    if (
        decision.requires_substantial_invention
        and decision.reference_scope != "reject"
    ):
        issues.append(
            ValidationIssue(
                code="substantial_invention_not_rejected",
                field="requires_substantial_invention",
                message="candidate requiring substantial invention must reject",
            )
        )
    if (
        not decision.primary_identity_region_visible
        and decision.reference_scope != "reject"
    ):
        issues.append(
            ValidationIssue(
                code="primary_identity_region_not_visible",
                field="primary_identity_region_visible",
                message="non-reject reference requires its primary identity region",
            )
        )
    if not decision.major_structure_visible and decision.reference_scope != "reject":
        issues.append(
            ValidationIssue(
                code="major_structure_not_visible",
                field="major_structure_visible",
                message="non-reject reference requires visible major structure",
            )
        )
    if (
        not decision.discrete_foreground_instance
        and decision.reference_scope != "reject"
    ):
        issues.append(
            ValidationIssue(
                code="non_discrete_foreground_instance",
                field="discrete_foreground_instance",
                message="non-discrete scene content must reject",
            )
        )
    if not decision.mask_matches_target and decision.reference_scope != "reject":
        issues.append(
            ValidationIssue(
                code="mask_target_mismatch",
                field="mask_matches_target",
                message="a mask that does not match the target must reject",
            )
        )
    if decision.truncation_severity == "major" and (
        decision.reference_scope != "reject"
        or decision.selected_candidate_id is not None
    ):
        issues.append(
            ValidationIssue(
                code="major_truncation_not_rejected",
                field="truncation_severity",
                message="major truncation must reject without selecting a candidate",
            )
        )
    if (
        decision.completeness == "complete"
        and decision.truncation_severity != "none"
    ):
        issues.append(
            ValidationIssue(
                code="complete_truncation_mismatch",
                field="truncation_severity",
                message="complete reference requires truncation_severity=none",
            )
        )
    if (
        decision.completeness == "repairable"
        and decision.truncation_severity != "minor"
    ):
        issues.append(
            ValidationIssue(
                code="repairable_requires_minor_truncation",
                field="truncation_severity",
                message="repairable reference requires minor truncation",
            )
        )
    if (
        decision.completeness == "complete"
        and decision.completion_needed_for_reference_use
    ):
        issues.append(
            ValidationIssue(
                code="complete_must_not_need_completion",
                field="completion_needed_for_reference_use",
                message="complete reference cannot require completion",
            )
        )
    if (
        decision.completeness == "complete"
        and decision.detached_target_fragments_present
    ):
        issues.append(
            ValidationIssue(
                code="complete_has_detached_target_fragments",
                field="detached_target_fragments_present",
                message="complete reference cannot contain detached target fragments",
            )
        )
    if (
        decision.completeness == "complete"
        and selected_subject_has_detached_signal
    ):
        issues.append(
            ValidationIssue(
                code="complete_has_detached_subject_fragments",
                field="selected_candidate_id",
                message=(
                    "complete subject cannot select a candidate with a "
                    "non-trivial detached component"
                ),
            )
        )
    if (
        decision.completeness == "local_usable"
        and decision.completion_needed_for_reference_use
    ):
        issues.append(
            ValidationIssue(
                code="local_usable_must_not_need_completion",
                field="completion_needed_for_reference_use",
                message="local_usable reference cannot require completion",
            )
        )
    if (
        decision.completeness == "local_usable"
        and decision.detached_target_fragments_present
    ):
        issues.append(
            ValidationIssue(
                code="local_usable_has_detached_target_fragments",
                field="detached_target_fragments_present",
                message=(
                    "local_usable reference cannot contain detached target fragments"
                ),
            )
        )
    if (
        decision.completeness == "local_usable"
        and selected_subject_has_detached_signal
    ):
        issues.append(
            ValidationIssue(
                code="local_usable_has_detached_subject_fragments",
                field="selected_candidate_id",
                message=(
                    "local_usable subject cannot select a candidate with a "
                    "non-trivial detached component"
                ),
            )
        )
    if (
        decision.completeness == "repairable"
        and not decision.completion_needed_for_reference_use
    ):
        issues.append(
            ValidationIssue(
                code="repairable_requires_completion",
                field="completion_needed_for_reference_use",
                message="repairable reference must require completion for use",
            )
        )
    if decision.reference_scope == "full":
        if decision.visible_region != "whole":
            issues.append(
                ValidationIssue(
                    code="full_requires_whole",
                    field="visible_region",
                    message="full reference must use visible_region=whole",
                )
            )
        if not decision.whole_entity_recognizable:
            issues.append(
                ValidationIssue(
                    code="full_requires_recognizable",
                    field="whole_entity_recognizable",
                    message="full reference must recognize the whole entity",
                )
            )
        if not decision.identity_features_visible:
            issues.append(
                ValidationIssue(
                    code="full_requires_identity",
                    field="identity_features_visible",
                    message="full reference requires visible identity features",
                )
            )
    elif decision.reference_scope == "local":
        if decision.visible_region == "whole":
            issues.append(
                ValidationIssue(
                    code="local_must_be_non_whole",
                    field="visible_region",
                    message="local reference cannot use visible_region=whole",
                )
            )
        if not decision.identity_features_visible:
            issues.append(
                ValidationIssue(
                    code="local_requires_identity",
                    field="identity_features_visible",
                    message="local reference requires visible identity features",
                )
            )
    return issues


class QwenEntityReferenceJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        repair_retries: int = 1,
        crop_padding_ratio: float = 0.08,
        evidence_mode: EvidencePresentationMode = "separate",
        card_panel_max_side: int = 512,
        client: Any | None = None,
    ) -> None:
        if evidence_mode not in {"separate", "paired_card"}:
            raise ValueError("candidate judge evidence_mode is invalid")
        if (
            isinstance(card_panel_max_side, bool)
            or not isinstance(card_panel_max_side, int)
            or card_panel_max_side <= 0
        ):
            raise ValueError(
                "candidate judge card_panel_max_side must be a positive integer"
            )
        self.config = config
        self.repair_retries = repair_retries
        self.crop_padding_ratio = crop_padding_ratio
        self.evidence_mode = evidence_mode
        self.card_panel_max_side = card_panel_max_side
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
        request_text: str,
    ) -> list[dict[str, object]]:
        from r2v_data_v2.v3.pair import (
            build_candidate_context_image,
            build_reference_crop,
        )

        content: list[dict[str, object]] = [
            {"type": "text", "text": request_text},
        ]
        for candidate in candidates:
            source = source_images.get(candidate.image_path)
            if source is None:
                raise KeyError(
                    f"missing source image for candidate: {candidate.image_path}"
                )
            context = build_candidate_context_image(source, candidate.mask)
            crop, _ = build_reference_crop(
                source,
                candidate.mask,
                crop_padding_ratio=self.crop_padding_ratio,
            )
            isolated = Image.new("RGB", crop.size, (255, 255, 255))
            isolated.paste(crop, mask=crop.getchannel("A"))
            evidence: tuple[tuple[str, Image.Image], ...]
            if self.evidence_mode == "paired_card":
                label = (
                    f"Candidate {candidate.candidate_id} paired evidence card "
                    + "(left panel = scene context; right panel = isolated "
                    + "proposed reference)"
                )
                evidence = (
                    (
                        label,
                        build_paired_candidate_evidence_card(
                            context,
                            isolated,
                            panel_max_side=self.card_panel_max_side,
                        ),
                    ),
                )
            else:
                evidence = (
                    (f"Candidate {candidate.candidate_id} context", context),
                    (
                        f"Candidate {candidate.candidate_id} isolated crop",
                        isolated,
                    ),
                )
            for label, image in evidence:
                content.append({"type": "text", "text": label})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(image)},
                    }
                )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        profile_context = get_model_profile_context()
        retry_index = profile_context.retry_index
        candidate_count = int(profile_context.metadata.get("candidate_count", 0))
        evidence_mode = str(
            profile_context.metadata.get("evidence_mode", self.evidence_mode)
        )
        card_panel_max_side = profile_context.metadata.get(
            "card_panel_max_side",
            self.card_panel_max_side if evidence_mode == "paired_card" else None,
        )
        profile_metadata = {
            "candidate_count": candidate_count,
            "context_image_count": candidate_count,
            "isolated_crop_count": candidate_count,
            "paired_card_count": (
                candidate_count if evidence_mode == "paired_card" else 0
            ),
            "evidence_mode": evidence_mode,
            "card_panel_max_side": card_panel_max_side,
        }
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_entity_reference_decision",
                            "strict": True,
                            "schema": RawEntityReferenceDecision.model_json_schema(),
                        },
                    },
                ),
                component="qwen_candidate_judge",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    **profile_metadata,
                    "response_format": "json_schema",
                },
            )
        except BadRequestError:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                ),
                component="qwen_candidate_judge",
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    **profile_metadata,
                    "response_format": "json_object",
                },
            )
        result = response.choices[0].message.content
        if not result:
            raise RuntimeError("Qwen returned an empty entity reference decision")
        return str(result)

    def decide(
        self,
        *,
        entity: AnnotationEntity,
        candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> EntityReferenceDecisionAttempt:
        if not candidates:
            raise ValueError("entity reference judge requires candidates")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if candidate_ids != [
            f"candidate_{index}" for index in range(1, len(candidates) + 1)
        ]:
            raise ValueError(
                "entity reference candidates must be contiguous and ordered"
            )
        payload = build_entity_reference_request_payload(entity, candidates)
        original_request = (
            "Select one immutable candidate or reject all candidates. "
            "Return strict JSON for this payload:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.repair_retries + 1):
            request_text = original_request
            if attempt:
                request_text = build_structured_repair_prompt(
                    original_request=original_request,
                    invalid_response=raw_responses[-1],
                    validation_issues=issues,
                    json_schema=(RawEntityReferenceDecision.model_json_schema()),
                )
            try:
                with model_profile_context(
                    retry_index=attempt,
                    metadata={
                        "candidate_count": len(candidates),
                        "evidence_mode": self.evidence_mode,
                        "card_panel_max_side": (
                            self.card_panel_max_side
                            if self.evidence_mode == "paired_card"
                            else None
                        ),
                    },
                ):
                    raw = self._request(
                        self._messages(
                            entity=entity,
                            candidates=candidates,
                            source_images=source_images,
                            request_text=request_text,
                        )
                    )
            except Exception as exc:
                raise EntityReferenceJudgeFailure(
                    raw_responses=raw_responses,
                    issues=[
                        ValidationIssue(
                            code="qwen_request_failed",
                            field=None,
                            message=str(exc),
                        )
                    ],
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            decision, issues = parse_qwen_json_issues(
                raw,
                RawEntityReferenceDecision,
            )
            if decision is not None:
                issues = validate_entity_reference_decision(
                    decision,
                    candidate_ids=set(candidate_ids),
                    reference_type=entity.reference_type,
                    candidate_by_id={
                        candidate.candidate_id: candidate
                        for candidate in candidates
                    },
                )
            if decision is not None and not issues:
                return EntityReferenceDecisionAttempt(
                    decision=decision,
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                )
        raise EntityReferenceJudgeFailure(
            raw_responses=raw_responses,
            issues=issues,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
