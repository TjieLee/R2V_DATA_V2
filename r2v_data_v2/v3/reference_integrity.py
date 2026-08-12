from __future__ import annotations

import base64
import io
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageFilter

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import QwenServiceConfig, V3Config
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.profiling import profiled_openai_call
from r2v_data_v2.v3.schemas import (
    EntityReferenceState,
    PairingState,
    ReferenceIntegrityEntityState,
    ReferenceIntegrityReview,
    ReferenceIntegrityState,
    ReferencesState,
    ReferenceTopologyDiagnostics,
)
from r2v_data_v2.v3.storage import RunStorage

SYSTEM_PROMPT = """You are the final integrity reviewer for one entity reference.
Compare the highlighted target in the source context with the final reference.
The final reference must denote the same complete entity as the annotation
phrase. Do not reinterpret the target as a convenient sub-entity. When a
container-and-contents relationship is part of the annotated entity identity,
the reference must preserve that defining relationship; recognizable contents
alone are insufficient. Identity-defining nouns and structure must remain,
although modifiers describing only a transient state need not all remain
visually obvious. For example, "a clay pot of stew" must reject when only stew
remains, and "a bowl of noodles" must reject when only noodles remain. For the
object "a camera", the camera itself must remain. For the subject "a man in a
white t-shirt", an unrelated held bowl or chopsticks may disappear if the
stable subject identity remains.
For a subject, require enough stable identity-bearing appearance for the stated
scope, but do not require a visible face in every valid back view, masked person,
character, or animal. Reject a torso, arms, or clothing fragment when the source
contains substantially more useful identity evidence and the final image loses
it. For an object, require its recognizable structural core. Reject major
missing surfaces, large unnatural holes, disconnected remnants, identity-changing
completion artifacts, severe destructive truncation, or an unrelated dominant
entity. Legitimate source-matching cutouts, handles, wheels, scissors, brackets,
frames, and truss structures are not defects merely because they contain holes.
Return JSON only and make verdict accept if and only if every boolean is true."""


@dataclass(frozen=True)
class ReferenceIntegrityReviewAttempt:
    review: ReferenceIntegrityReview
    raw_response: str


class ReferenceIntegrityJudgeFailure(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ReferenceIntegrityJudge(Protocol):
    def review(
        self,
        *,
        source_context: Image.Image,
        final_reference: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        synthetic: bool,
    ) -> ReferenceIntegrityReviewAttempt: ...


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


class QwenReferenceIntegrityJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def review(
        self,
        *,
        source_context: Image.Image,
        final_reference: Image.Image,
        reference_type: str,
        phrase: str,
        grounding_prompt: str,
        reference_scope: str,
        synthetic: bool,
    ) -> ReferenceIntegrityReviewAttempt:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Reference type: {reference_type}\n"
                            f"Phrase: {phrase}\n"
                            f"Grounding prompt: {grounding_prompt}\n"
                            f"Scope: {reference_scope}\n"
                            f"Synthetic: {str(synthetic).lower()}\n"
                            "Image 1 is source context with the target highlighted. "
                            "Image 2 is the final reference."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(source_context)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(final_reference)},
                    },
                ],
            },
        ]
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        raw: str | None = None
        metadata = {
            "reference_type": reference_type,
            "reference_scope": reference_scope,
            "synthetic": synthetic,
            "image_count": 2,
            "mode": "targeted_qwen_v1",
        }
        try:
            try:
                response = profiled_openai_call(
                    lambda: self.client.chat.completions.create(
                        **parameters,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "v3_reference_integrity_review",
                                "strict": True,
                                "schema": ReferenceIntegrityReview.model_json_schema(),
                            },
                        },
                    ),
                    component="qwen_reference_integrity",
                    operation="initial",
                    retry_index=0,
                    model=self.config.model,
                    messages=messages,
                    metadata={**metadata, "response_format": "json_schema"},
                )
            except BadRequestError:
                response = profiled_openai_call(
                    lambda: self.client.chat.completions.create(
                        **parameters,
                        response_format={"type": "json_object"},
                    ),
                    component="qwen_reference_integrity",
                    operation="initial",
                    retry_index=0,
                    model=self.config.model,
                    messages=messages,
                    metadata={**metadata, "response_format": "json_object"},
                )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Qwen returned an empty integrity review")
            raw = str(content)
            review = ReferenceIntegrityReview.model_validate_json(raw)
        except Exception as exc:
            raise ReferenceIntegrityJudgeFailure(str(exc), raw_response=raw) from exc
        return ReferenceIntegrityReviewAttempt(review=review, raw_response=raw)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _component_areas(mask: np.ndarray) -> tuple[list[int], list[bool]]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary)
    areas: list[int] = []
    touches_border: list[bool] = []
    for y, x in np.argwhere(binary):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        area = 0
        border = False
        while queue:
            cy, cx = queue.popleft()
            area += 1
            border = border or cy in {0, height - 1} or cx in {0, width - 1}
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        areas.append(area)
        touches_border.append(border)
    return areas, touches_border


def reference_topology_diagnostics(image: Image.Image) -> ReferenceTopologyDiagnostics:
    if "A" not in image.getbands():
        return ReferenceTopologyDiagnostics(
            alpha_available=False,
            significant_component_count=0,
            largest_component_ratio=0.0,
            second_component_ratio=0.0,
            bbox_fill_ratio=0.0,
            border_contact=False,
            enclosed_transparent_hole_count=0,
            largest_enclosed_hole_area=0,
            enclosed_hole_bbox_ratio=0.0,
            suspicious=False,
        )
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8) > 0
    foreground_area = int(alpha.sum())
    if foreground_area == 0:
        return ReferenceTopologyDiagnostics(
            alpha_available=True,
            significant_component_count=0,
            largest_component_ratio=0.0,
            second_component_ratio=0.0,
            bbox_fill_ratio=0.0,
            border_contact=False,
            enclosed_transparent_hole_count=0,
            largest_enclosed_hole_area=0,
            enclosed_hole_bbox_ratio=0.0,
            suspicious=True,
            suspicion_reasons=["empty_alpha"],
        )
    ys, xs = np.nonzero(alpha)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = (x2 - x1) * (y2 - y1)
    component_areas, _ = _component_areas(alpha)
    significant_minimum = max(16, round(foreground_area * 0.01))
    significant = sorted(
        (area for area in component_areas if area >= significant_minimum),
        reverse=True,
    )
    hole_areas, hole_borders = _component_areas(~alpha[y1:y2, x1:x2])
    enclosed = sorted(
        (
            area
            for area, border in zip(hole_areas, hole_borders)
            if not border
        ),
        reverse=True,
    )
    reasons: list[str] = []
    largest_ratio = significant[0] / foreground_area if significant else 0.0
    second_ratio = significant[1] / foreground_area if len(significant) > 1 else 0.0
    hole_ratio = enclosed[0] / bbox_area if enclosed else 0.0
    bbox_fill = foreground_area / bbox_area
    if len(significant) > 3 or largest_ratio < 0.80:
        reasons.append("fragmented_alpha_topology")
    if hole_ratio >= 0.05:
        reasons.append("large_enclosed_alpha_hole")
    if bbox_fill < 0.12:
        reasons.append("low_alpha_bbox_fill")
    border_contact = bool(
        alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any()
    )
    return ReferenceTopologyDiagnostics(
        alpha_available=True,
        significant_component_count=len(significant),
        largest_component_ratio=largest_ratio,
        second_component_ratio=second_ratio,
        bbox_fill_ratio=bbox_fill,
        border_contact=border_contact,
        enclosed_transparent_hole_count=len(enclosed),
        largest_enclosed_hole_area=enclosed[0] if enclosed else 0,
        enclosed_hole_bbox_ratio=hole_ratio,
        suspicious=bool(reasons),
        suspicion_reasons=reasons,
    )


def _resolve_run_artifact(storage: RunStorage, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference artifact path must be run-relative")
    path = (storage.root / relative).resolve(strict=False)
    try:
        path.relative_to(storage.root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("reference artifact must remain inside run_root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"reference artifact is missing: {path}")
    return path


def _source_context(
    storage: RunStorage,
    *,
    clip_uid: str,
    reference: EntityReferenceState,
) -> Image.Image:
    source_clip_uid = reference.source_clip_uid or clip_uid
    source_entity_id = reference.source_entity_id or reference.entity_id
    frames = storage.read_frames(source_clip_uid)
    frame = next(
        (
            item
            for item in frames.frames
            if item.source_frame_index == reference.source_frame_index
        ),
        None,
    )
    if frame is None:
        raise ValueError("reference source frame is absent from sampled frames")
    frame_path = storage.clip_dir(source_clip_uid) / frame.image_path
    with Image.open(frame_path) as opened:
        context = opened.convert("RGB")
        context.load()
    masks = storage.read_masks(source_clip_uid)
    track = masks.entities.get(source_entity_id)
    if track is None:
        raise ValueError("reference source entity mask is missing")
    mask_frame = next((item for item in track.frames if item.slot == frame.slot), None)
    if mask_frame is None or not mask_frame.present:
        raise ValueError("reference source mask is absent at selected frame")
    mask = decode_binary_mask(mask_frame.rle)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    ring = np.asarray(mask_image.filter(ImageFilter.MaxFilter(7))) > 0
    ring &= ~mask
    pixels = np.asarray(context).copy()
    pixels[ring] = (255, 0, 0)
    return Image.fromarray(pixels)


def _write_context_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _rejected_reference(
    reference: EntityReferenceState,
    reason: str,
) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=reference.entity_id,
        status="rejected",
        reference_scope="reject",
        visible_region=reference.visible_region,
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason=reason,
        image_quality=reference.image_quality,
        completeness=reference.completeness,
        viewpoint=reference.viewpoint,
        independent_reference_value=False,
        requires_substantial_invention=reference.requires_substantial_invention,
        primary_identity_region_visible=False,
        major_structure_visible=False,
        truncation_severity=reference.truncation_severity,
        discrete_foreground_instance=reference.discrete_foreground_instance,
        mask_matches_target=reference.mask_matches_target,
        completion_needed_for_reference_use=reference.completion_needed_for_reference_use,
        detached_target_fragments_present=reference.detached_target_fragments_present,
    )


def _tokens_for_retained(
    retained: list[str],
    reference_types: dict[str, str],
) -> dict[str, str]:
    counters = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity_id in retained:
        reference_type = reference_types[entity_id]
        counters[reference_type] += 1
        tokens[entity_id] = f"<ref_{reference_type}_{counters[reference_type]}>"
    return tokens


@dataclass(frozen=True)
class ReferenceIntegrityStats:
    processed: int = 0
    skipped_disabled: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    entities_reviewed: int = 0
    entities_skipped_review: int = 0
    entities_accepted: int = 0
    entities_rejected: int = 0
    judge_failed: int = 0
    topology_suspicious: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def reference_integrity_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    judge: ReferenceIntegrityJudge | None = None,
) -> ReferenceIntegrityStats:
    counters = {field: 0 for field in ReferenceIntegrityStats.__dataclass_fields__}
    if not config.reference_integrity.enabled:
        counters["skipped_disabled"] = sum(1 for _ in storage.iter_clips())
        stats = ReferenceIntegrityStats(**counters)
        storage.update_stage_counts("reference_integrity", stats.to_dict())
        return stats
    owned_judge: QwenReferenceIntegrityJudge | None = None
    active_judge = judge
    if active_judge is None:
        service = config.qwen.reference_integrity_judge
        if service is None:
            raise ValueError("reference integrity Qwen judge is not configured")
        owned_judge = QwenReferenceIntegrityJudge(service)
        active_judge = owned_judge
    try:
        for clip in storage.iter_clips():
            if clip.pairing is None or clip.pairing.status != "ready":
                counters["skipped_not_ready"] += 1
                continue
            if config.reference_edit.enabled and (
                clip.reference_edit is None or clip.reference_edit.status != "ready"
            ):
                counters["skipped_not_ready"] += 1
                continue
            if (
                clip.reference_integrity is not None
                and clip.reference_integrity.status == "ready"
                and not overwrite
            ):
                counters["skipped_existing"] += 1
                continue
            counters["processed"] += 1
            try:
                references_by_id = {
                    item.entity_id: item for item in clip.references.entities
                }
                entities_by_id = {
                    item.entity_id: item
                    for item in (
                        clip.annotation.entities if clip.annotation is not None else []
                    )
                }
                results: list[ReferenceIntegrityEntityState] = []
                rejected_ids: set[str] = set()
                for entity_id in clip.pairing.retained_entity_ids:
                    reference = references_by_id[entity_id]
                    entity = entities_by_id[entity_id]
                    assert reference.image_path is not None
                    final_path = _resolve_run_artifact(storage, reference.image_path)
                    with Image.open(final_path) as opened:
                        final_image = opened.copy()
                        final_image.load()
                    diagnostics = reference_topology_diagnostics(final_image)
                    counters["topology_suspicious"] += int(diagnostics.suspicious)
                    requires_review = (
                        reference.synthetic
                        or reference.reference_scope == "local"
                        or diagnostics.suspicious
                    )
                    if not requires_review:
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="skipped",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                diagnostics=diagnostics,
                                reviewed=False,
                                reason="clean_real_full_reference",
                            )
                        )
                        counters["entities_skipped_review"] += 1
                        counters["entities_accepted"] += 1
                        continue
                    context = _source_context(
                        storage, clip_uid=clip.clip_uid, reference=reference
                    )
                    context_path = storage.selected_path(
                        clip.clip_uid, f"integrity_context_{entity_id}.png"
                    )
                    _write_context_atomic(context_path, context)
                    context_relative = storage.relative_artifact_path(context_path)
                    counters["entities_reviewed"] += 1
                    try:
                        attempt = active_judge.review(
                            source_context=context,
                            final_reference=final_image,
                            reference_type=entity.reference_type,
                            phrase=entity.phrase,
                            grounding_prompt=entity.grounding_prompt,
                            reference_scope=reference.reference_scope,
                            synthetic=reference.synthetic,
                        )
                    except ReferenceIntegrityJudgeFailure as exc:
                        rejected_ids.add(entity_id)
                        counters["judge_failed"] += 1
                        counters["entities_rejected"] += 1
                        results.append(
                            ReferenceIntegrityEntityState(
                                entity_id=entity_id,
                                status="rejected",
                                input_reference=reference,
                                final_reference_path=reference.image_path,
                                source_context_path=context_relative,
                                diagnostics=diagnostics,
                                reviewed=True,
                                judge_failed=True,
                                reason=f"integrity_judge_failed:{exc}",
                            )
                        )
                        continue
                    if config.debug.save_diagnostics:
                        debug_path = storage.debug_path(
                            clip.clip_uid, f"reference_integrity_{entity_id}.json"
                        )
                        write_json_atomic(
                            debug_path,
                            {
                                "review": attempt.review.model_dump(mode="json"),
                                "raw_response": attempt.raw_response,
                            },
                        )
                    accepted = attempt.review.verdict == "accept"
                    if accepted:
                        counters["entities_accepted"] += 1
                    else:
                        rejected_ids.add(entity_id)
                        counters["entities_rejected"] += 1
                    results.append(
                        ReferenceIntegrityEntityState(
                            entity_id=entity_id,
                            status="accepted" if accepted else "rejected",
                            input_reference=reference,
                            final_reference_path=reference.image_path,
                            source_context_path=context_relative,
                            diagnostics=diagnostics,
                            reviewed=True,
                            review=attempt.review,
                            reason=attempt.review.reason,
                        )
                    )
                final_references = [
                    _rejected_reference(item, "reference_integrity_rejected")
                    if item.entity_id in rejected_ids
                    else item
                    for item in clip.references.entities
                ]
                retained = [
                    entity_id
                    for entity_id in clip.pairing.retained_entity_ids
                    if entity_id not in rejected_ids
                ]
                qualifying = set(
                    clip.coverage.qualifying_entity_ids
                    if clip.coverage is not None
                    else []
                )
                if not set(retained).intersection(qualifying):
                    pairing = PairingState(
                        status="rejected",
                        reason="no_qualifying_reference_after_integrity",
                    )
                else:
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(
                            retained,
                            {
                                item.entity_id: item.reference_type
                                for item in entities_by_id.values()
                            },
                        ),
                        background_token=clip.pairing.background_token,
                    )
                storage.write_reference_integrity_result(
                    clip.clip_uid,
                    ReferencesState(
                        entities=final_references,
                        background=clip.references.background,
                    ),
                    pairing,
                    ReferenceIntegrityState(status="ready", entities=results),
                )
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clip artifacts
                storage.write_reference_integrity_failure(clip.clip_uid, str(exc))
                storage.append_failure(
                    stage="reference_integrity",
                    clip_uid=clip.clip_uid,
                    reason=str(exc),
                    details={"exception_type": type(exc).__name__},
                )
                counters["failed"] += 1
    finally:
        if owned_judge is not None:
            owned_judge.close()
    stats = ReferenceIntegrityStats(**counters)
    storage.update_stage_counts("reference_integrity", stats.to_dict())
    return stats
