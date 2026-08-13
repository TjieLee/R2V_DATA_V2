from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.reference_integrity as integrity_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceIntegrityConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.entity_composition_audit import audit_entity_composition
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.reference_integrity import (
    SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    QwenReferenceIntegrityJudge,
    QwenSourceBboxFallbackJudge,
    ReferenceIntegrityJudgeFailure,
    ReferenceIntegrityReviewAttempt,
    SourceBboxFallbackJudgeFailure,
    SourceBboxFallbackReviewAttempt,
    reference_integrity_clips,
    reference_semantic_hard_reject_reason,
    reference_semantic_risk_reason,
    reference_topology_diagnostics,
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
    ReferenceEditEntityState,
    ReferenceEditState,
    ReferenceIntegrityEntityState,
    ReferenceIntegrityReview,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    SourceBboxFallbackReview,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_inline_instruction_text,
)
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage


def _review(
    *,
    accept: bool,
    reason: str = "reviewed",
    preserves_semantics: bool | None = None,
    preserves_primary_identity_region: bool | None = None,
    reference_entity_semantically_valid: bool | None = None,
    no_severe_reference_artifact: bool | None = None,
) -> ReferenceIntegrityReview:
    semantic_fidelity = accept if preserves_semantics is None else preserves_semantics
    identity_region = (
        accept
        if preserves_primary_identity_region is None
        else preserves_primary_identity_region
    )
    semantic_validity = (
        accept
        if reference_entity_semantically_valid is None
        else reference_entity_semantically_valid
    )
    artifact_free = (
        accept
        if no_severe_reference_artifact is None
        else no_severe_reference_artifact
    )
    return ReferenceIntegrityReview(
        matches_target=accept,
        reference_entity_semantically_valid=semantic_validity,
        preserves_annotated_entity_semantics=semantic_fidelity,
        preserves_primary_identity_region=identity_region,
        recognizable_as_named_entity=accept,
        structurally_complete_for_scope=accept,
        no_major_missing_regions=accept,
        no_unnatural_holes_or_surface_loss=accept,
        no_unrelated_entity_dominance=accept,
        no_severe_reference_artifact=artifact_free,
        usable_as_independent_reference=accept,
        verdict="accept" if accept else "reject",
        reason=reason,
    )


def _semantic_reinterpretation_review() -> ReferenceIntegrityReview:
    return ReferenceIntegrityReview(
        matches_target=True,
        reference_entity_semantically_valid=True,
        preserves_annotated_entity_semantics=False,
        preserves_primary_identity_region=True,
        recognizable_as_named_entity=True,
        structurally_complete_for_scope=True,
        no_major_missing_regions=True,
        no_unnatural_holes_or_surface_loss=True,
        no_unrelated_entity_dominance=True,
        no_severe_reference_artifact=True,
        usable_as_independent_reference=True,
        verdict="reject",
        reason="only the stew remains; the annotated clay pot is missing",
    )


def _primary_identity_loss_review() -> ReferenceIntegrityReview:
    return ReferenceIntegrityReview(
        matches_target=True,
        reference_entity_semantically_valid=True,
        preserves_annotated_entity_semantics=True,
        preserves_primary_identity_region=False,
        recognizable_as_named_entity=True,
        structurally_complete_for_scope=True,
        no_major_missing_regions=True,
        no_unnatural_holes_or_surface_loss=True,
        no_unrelated_entity_dominance=True,
        no_severe_reference_artifact=True,
        usable_as_independent_reference=True,
        verdict="reject",
        reason="the source shows the head but the final subject is cropped below it",
    )


def _invalid_reference_semantics_review(reason: str) -> ReferenceIntegrityReview:
    accepted = _review(accept=True).model_dump(mode="json")
    return ReferenceIntegrityReview.model_validate(
        {
            **accepted,
            "reference_entity_semantically_valid": False,
            "verdict": "reject",
            "reason": reason,
        }
    )


def _severe_reference_artifact_review() -> ReferenceIntegrityReview:
    accepted = _review(accept=True).model_dump(mode="json")
    return ReferenceIntegrityReview.model_validate(
        {
            **accepted,
            "no_severe_reference_artifact": False,
            "verdict": "reject",
            "reason": (
                "a large white bottle-shaped edit cavity cuts through the woman"
            ),
        }
    )


def _bbox_review(*, accept: bool) -> SourceBboxFallbackReview:
    return SourceBboxFallbackReview(
        same_target_entity=True,
        target_remains_dominant=True,
        extra_non_target_content_is_minor=True,
        no_competing_salient_entity=accept,
        no_severe_reference_artifact=True,
        bbox_is_preferable_to_failed_reference=True,
        usable_as_independent_reference=True,
        certain=True,
        verdict="accept" if accept else "reject",
        reason=(
            "raw source bbox is a clear artifact-free target reference"
            if accept
            else "a competing salient person dominates the raw bbox"
        ),
    )


@dataclass
class FakeIntegrityJudge:
    results: list[ReferenceIntegrityReview | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    def review(self, **kwargs: object) -> ReferenceIntegrityReviewAttempt:
        self.calls.append(kwargs)
        result = self.results[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return ReferenceIntegrityReviewAttempt(
            review=result,
            raw_response=result.model_dump_json(),
        )


@dataclass
class FakeSourceBboxFallbackJudge:
    results: list[SourceBboxFallbackReview | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    def review(self, **kwargs: object) -> SourceBboxFallbackReviewAttempt:
        self.calls.append(kwargs)
        result = self.results[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return SourceBboxFallbackReviewAttempt(
            review=result,
            raw_response=result.model_dump_json(),
            finish_reason="stop",
        )


@dataclass
class FakeIntegrityCompletions:
    responses: list[tuple[str, str | None]]
    calls: list[dict[str, object]] = field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        content, finish_reason = self.responses.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


def _qwen_integrity_judge(
    config: V3Config,
    responses: list[tuple[str, str | None]],
) -> tuple[QwenReferenceIntegrityJudge, FakeIntegrityCompletions]:
    completions = FakeIntegrityCompletions(responses)
    service = config.qwen.reference_integrity_judge
    assert service is not None
    judge = QwenReferenceIntegrityJudge(
        service,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        ),
    )
    return judge, completions


def _run_qwen_integrity_review(
    judge: QwenReferenceIntegrityJudge,
) -> ReferenceIntegrityReviewAttempt:
    return judge.review(
        source_context=Image.new("RGB", (12, 10), (20, 40, 60)),
        final_reference=Image.new("RGB", (8, 8), (80, 100, 120)),
        reference_type="object",
        phrase="a green laser pointer",
        grounding_prompt="green laser pointer near center",
        reference_scope="full",
        synthetic=False,
    )


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    source = dataset_root / "source.jsonl"
    source.write_text("", encoding="utf-8")
    model = str(pretrained / "Qwen" / "judge")
    config = V3Config(
        dataset_json=source,
        run_root=writable / "runs" / "integrity",
        export_root=writable / "datasets" / "integrity",
        source=SourceConfig(limit=1),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=model),
            instruction_writer=QwenServiceConfig(model=model),
            candidate_judge=QwenServiceConfig(model=model),
            background_remove_judge=QwenServiceConfig(model=model),
            reference_integrity_judge=QwenServiceConfig(model=model),
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "object-remover",
        ),
        reference_integrity=ReferenceIntegrityConfig(enabled=True),
    )
    config.validate()
    return config


def test_qwen_integrity_valid_first_response_uses_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _review(accept=True)
    judge, completions = _qwen_integrity_judge(
        _config(tmp_path, monkeypatch),
        [(expected.model_dump_json(), "stop")],
    )

    attempt = _run_qwen_integrity_review(judge)

    assert attempt.review == expected
    assert attempt.raw_responses == (expected.model_dump_json(),)
    assert attempt.finish_reasons == ("stop",)
    assert len(completions.calls) == 1


def test_integrity_review_schema_terminates_with_reason_then_verdict() -> None:
    schema = ReferenceIntegrityReview.model_json_schema()

    assert list(schema["properties"])[-2:] == ["reason", "verdict"]
    assert set(schema["required"]) == {
        "matches_target",
        "reference_entity_semantically_valid",
        "preserves_annotated_entity_semantics",
        "preserves_primary_identity_region",
        "recognizable_as_named_entity",
        "structurally_complete_for_scope",
        "no_major_missing_regions",
        "no_unnatural_holes_or_surface_loss",
        "no_unrelated_entity_dominance",
        "no_severe_reference_artifact",
        "usable_as_independent_reference",
        "reason",
        "verdict",
    }


def test_source_bbox_review_schema_terminates_with_reason_then_verdict() -> None:
    schema = SourceBboxFallbackReview.model_json_schema()

    assert list(schema["properties"])[-2:] == ["reason", "verdict"]
    assert set(schema["required"]) == {
        "same_target_entity",
        "target_remains_dominant",
        "extra_non_target_content_is_minor",
        "no_competing_salient_entity",
        "no_severe_reference_artifact",
        "bbox_is_preferable_to_failed_reference",
        "usable_as_independent_reference",
        "certain",
        "reason",
        "verdict",
    }


@pytest.mark.parametrize(
    ("review_type", "review"),
    (
        (ReferenceIntegrityReview, _review(accept=True)),
        (SourceBboxFallbackReview, _bbox_review(accept=True)),
    ),
)
def test_review_schemas_accept_historical_verdict_before_reason_json(
    review_type: type[ReferenceIntegrityReview | SourceBboxFallbackReview],
    review: ReferenceIntegrityReview | SourceBboxFallbackReview,
) -> None:
    payload = review.model_dump(mode="json")
    reason = payload.pop("reason")
    verdict = payload.pop("verdict")
    historical_payload = {**payload, "verdict": verdict, "reason": reason}

    assert review_type.model_validate_json(json.dumps(historical_payload)) == review


def test_qwen_integrity_repairs_truncated_response_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _review(accept=True)
    profiled: list[tuple[str, int]] = []

    def profile(request, *, operation, retry_index, **_kwargs):
        profiled.append((operation, retry_index))
        return request()

    monkeypatch.setattr(integrity_module, "profiled_openai_call", profile)
    judge, completions = _qwen_integrity_judge(
        _config(tmp_path, monkeypatch),
        [(' {"matches_target": true', "length"), (expected.model_dump_json(), "stop")],
    )

    attempt = _run_qwen_integrity_review(judge)

    assert attempt.review == expected
    assert attempt.raw_responses == (
        ' {"matches_target": true',
        expected.model_dump_json(),
    )
    assert attempt.finish_reasons == ("length", "stop")
    assert profiled == [("initial", 0), ("repair", 1)]
    assert len(completions.calls) == 2
    repair_messages = completions.calls[1]["messages"]
    assert isinstance(repair_messages, list)
    image_items = [
        item
        for message in repair_messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]
    assert len(image_items) == 2
    repair_text = str(repair_messages[-1]["content"])
    assert "Validation error:" in repair_text
    assert "Do not return markdown, explanation, chain-of-thought" in repair_text
    assert "Follow the supplied schema key order" in repair_text
    assert "Emit reason before verdict" in repair_text
    assert "make verdict the final key" in repair_text
    assert "close the JSON object immediately after verdict" in repair_text
    assert "Do not emit trailing whitespace" in repair_text


def test_qwen_integrity_repairs_schema_invalid_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _review(accept=True)
    judge, completions = _qwen_integrity_judge(
        _config(tmp_path, monkeypatch),
        [("{}", "stop"), (expected.model_dump_json(), "stop")],
    )

    attempt = _run_qwen_integrity_review(judge)

    assert attempt.review == expected
    assert len(completions.calls) == 2


def test_qwen_integrity_fails_closed_with_both_invalid_raw_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge, completions = _qwen_integrity_judge(
        _config(tmp_path, monkeypatch),
        [("{", "length"), ("{}", "stop")],
    )

    with pytest.raises(ReferenceIntegrityJudgeFailure) as raised:
        _run_qwen_integrity_review(judge)

    assert raised.value.raw_responses == ("{", "{}")
    assert raised.value.raw_response == "{}"
    assert raised.value.finish_reasons == ("length", "stop")
    assert len(completions.calls) == 2


def test_qwen_integrity_repaired_output_still_enforces_hard_booleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _severe_reference_artifact_review()
    judge, _ = _qwen_integrity_judge(
        _config(tmp_path, monkeypatch),
        [("{}", "stop"), (expected.model_dump_json(), "stop")],
    )

    attempt = _run_qwen_integrity_review(judge)

    assert attempt.review.verdict == "reject"
    assert attempt.review.no_severe_reference_artifact is False


def _visibility() -> EntityVisibilitySummary:
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(7)),
        visible_frame_count=7,
        coverage_ratio=0.7,
        qualifies=True,
        per_frame_area_ratio=[0.25] * 7 + [0.0] * 3,
        per_frame_confidence=[0.9] * 7 + [None] * 3,
    )


def _ready_reference(
    entity_id: str,
    image_path: str,
    *,
    reference_scope: str = "full",
    synthetic: bool = False,
) -> EntityReferenceState:
    local = reference_scope == "local"
    return EntityReferenceState(
        entity_id=entity_id,
        status="ready",
        reference_scope=reference_scope,
        visible_region="upper_body" if local else "whole",
        whole_entity_recognizable=not local,
        identity_features_visible=True,
        scope_reason="usable source evidence",
        image_path=image_path,
        source_frame_index=0,
        source_clip_uid="clip-1" if synthetic else None,
        source_entity_id=entity_id if synthetic else None,
        image_quality="acceptable" if synthetic else None,
        completeness=("local_usable" if local else "complete") if synthetic else None,
        synthetic=synthetic,
        generation_metadata_path=(
            "clips/clip-1/selected/final.json" if synthetic else None
        ),
        generation_source_sha256=("a" * 64 if synthetic else None),
        generation_output_sha256=("b" * 64 if synthetic else None),
    )


def _storage_with_ready_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    second_scope: str = "local",
    second_synthetic: bool = False,
    second_hole: bool = False,
    second_phrase: str = "a metal bracket",
) -> RunStorage:
    storage = RunStorage(_config(tmp_path, monkeypatch))
    storage.initialize(git_commit="integrity-test")
    video = storage.config.dataset_json.parent / "clip.mp4"
    video.write_bytes(b"video")
    storage.create_clip(
        clip_uid="clip-1",
        source=ClipSource(
            video_path=str(video),
            parent_video_id="parent",
            clip_suffix="0_1",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entities = [
        AnnotationEntity(
            entity_id="e1",
            reference_type="subject",
            phrase="a person in an orange cap",
            grounding_prompt="person in orange cap near center",
        ),
        AnnotationEntity(
            entity_id="e2",
            reference_type="object",
            phrase=second_phrase,
            grounding_prompt=f"{second_phrase} beside the person",
        ),
    ]
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            instruction_template="{{entity_1}} holds {{entity_2}}.",
            entities=entities,
        ),
    )
    storage.write_coverage(
        "clip-1",
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1", "e2"],
            required_visible_frames=7,
            entity_visibility_summary={"e1": _visibility(), "e2": _visibility()},
        ),
    )
    frames_dir = storage.frames_dir("clip-1")
    frames_dir.mkdir(parents=True)
    frame_records: list[SampledFrame] = []
    for slot in range(10):
        frame_path = frames_dir / f"{slot:02d}.jpg"
        Image.new("RGB", (32, 24), (80, 100, 120)).save(frame_path)
        frame_records.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot,
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            )
        )
    write_json_atomic(
        storage.frames_manifest_path("clip-1"),
        SampledFramesArtifact(
            clip_uid="clip-1", width=32, height=24, frames=frame_records
        ).model_dump(mode="json"),
    )
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:20, 6:26] = True
    empty = encode_binary_mask(np.zeros_like(mask))
    tracks: dict[str, TrackedEntityMasks] = {}
    for entity in entities:
        tracks[entity.entity_id] = TrackedEntityMasks(
            status="ready",
            reference_type=entity.reference_type,
            grounding_prompt=entity.grounding_prompt,
            backend_object_ids=["1"],
            frames=[
                TrackedMaskFrame(
                    slot=slot,
                    present=slot == 0,
                    confidence=0.9 if slot == 0 else None,
                    backend_confidences=[0.9] if slot == 0 else [],
                    backend_object_ids=["1"] if slot == 0 else [],
                    area_pixels=int(mask.sum()) if slot == 0 else 0,
                    area_ratio=float(mask.mean()) if slot == 0 else 0.0,
                    bbox_xyxy=(6, 4, 26, 20) if slot == 0 else None,
                    rle=encode_binary_mask(mask) if slot == 0 else empty,
                )
                for slot in range(10)
            ],
        )
    storage.write_masks(
        "clip-1",
        TrackedMasksArtifact(clip_uid="clip-1", width=32, height=24, entities=tracks),
    )
    storage.write_coverage(
        "clip-1",
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1", "e2"],
            required_visible_frames=7,
            entity_visibility_summary={"e1": _visibility(), "e2": _visibility()},
        ),
    )
    references: list[EntityReferenceState] = []
    for index, entity_id in enumerate(("e1", "e2"), start=1):
        path = storage.selected_path("clip-1", f"{entity_id}.png")
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[8:56, 8:56] = 255
        if entity_id == "e2" and second_hole:
            alpha[20:44, 20:44] = 0
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[..., :3] = (30 * index, 90, 140)
        rgba[..., 3] = alpha
        Image.fromarray(rgba).save(path)
        references.append(
            _ready_reference(
                entity_id,
                storage.relative_artifact_path(path),
                reference_scope=(second_scope if entity_id == "e2" else "full"),
                synthetic=(second_synthetic if entity_id == "e2" else False),
            )
        )
    storage.write_references_and_pairing(
        "clip-1",
        ReferencesState(entities=references),
        PairingState(
            status="ready",
            retained_entity_ids=["e1", "e2"],
            tokens={"e1": "<ref_subject_1>", "e2": "<ref_object_1>"},
        ),
    )
    body = "{{image_1}} holds {{image_2}}."
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=[
                InstructionLegendEntry(image_id="image_1", description="person"),
                InstructionLegendEntry(image_id="image_2", description="bracket"),
            ],
            r2v_instruction=render_inline_instruction_text(body),
        ),
    )
    storage.write_export("clip-1", ExportState(accepted=True, reason=None))
    return storage


def test_integrity_schema_requires_semantic_fidelity_for_acceptance() -> None:
    accepted = _review(accept=True).model_dump(mode="json")

    missing = dict(accepted)
    missing.pop("preserves_annotated_entity_semantics")
    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate(missing)

    non_boolean = {
        **accepted,
        "preserves_annotated_entity_semantics": 1,
    }
    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate(non_boolean)

    contradicted = {
        **accepted,
        "preserves_annotated_entity_semantics": False,
    }
    with pytest.raises(ValueError, match="must match all integrity checks"):
        ReferenceIntegrityReview.model_validate(contradicted)


def test_integrity_schema_requires_primary_identity_region_for_acceptance() -> None:
    accepted = _review(accept=True).model_dump(mode="json")

    missing = dict(accepted)
    missing.pop("preserves_primary_identity_region")
    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate(missing)

    non_boolean = {
        **accepted,
        "preserves_primary_identity_region": 1,
    }
    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate(non_boolean)

    contradicted = {
        **accepted,
        "preserves_primary_identity_region": False,
    }
    with pytest.raises(ValueError, match="must match all integrity checks"):
        ReferenceIntegrityReview.model_validate(contradicted)


@pytest.mark.parametrize(
    "field_name",
    (
        "reference_entity_semantically_valid",
        "no_severe_reference_artifact",
    ),
)
def test_integrity_schema_requires_final_semantic_and_artifact_gates(
    field_name: str,
) -> None:
    accepted = _review(accept=True).model_dump(mode="json")

    missing = dict(accepted)
    missing.pop(field_name)
    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate(missing)

    with pytest.raises(ValueError):
        ReferenceIntegrityReview.model_validate({**accepted, field_name: 1})

    with pytest.raises(ValueError, match="must match all integrity checks"):
        ReferenceIntegrityReview.model_validate(
            {**accepted, field_name: False}
        )


@pytest.mark.parametrize(
    "description",
    (
        "person viewed from behind with the head present",
        "helmeted or masked person with the head present",
        "side-profile person with the head and upper body present",
        "normal portrait with a recognizable head region",
    ),
)
def test_human_subject_with_recognizable_head_region_can_pass(
    description: str,
) -> None:
    review = _review(accept=True, reason=description)

    assert review.preserves_primary_identity_region is True
    assert review.verdict == "accept"


def test_headless_human_subject_cannot_pass_integrity_schema() -> None:
    accepted = _review(accept=True).model_dump(mode="json")
    headless = {
        **accepted,
        "preserves_primary_identity_region": False,
        "verdict": "reject",
        "reason": "the source shows the head but the final reference is headless",
    }

    review = ReferenceIntegrityReview.model_validate(headless)

    assert review.preserves_primary_identity_region is False
    assert review.verdict == "reject"


def test_object_integrity_behavior_is_unchanged() -> None:
    review = _review(
        accept=True,
        reason="the camera preserves its recognizable structural core",
    )

    assert review.preserves_primary_identity_region is True
    assert review.verdict == "accept"


def test_integrity_prompt_forbids_sub_entity_reinterpretation() -> None:
    prompt = " ".join(SYSTEM_PROMPT.lower().split())

    for contract in (
        "same complete entity as the annotation phrase",
        "do not reinterpret the target as a convenient sub-entity",
        "recognizable contents alone are insufficient",
        '"a clay pot of stew" must reject when only stew remains',
        '"a bowl of noodles" must reject when only noodles remain',
        'for the object "a camera", the camera itself must remain',
        'for the subject "a man in a white t-shirt"',
        "an unrelated held bowl or chopsticks may disappear",
    ):
        assert contract in prompt


def test_integrity_prompt_requires_human_head_region_without_requiring_face() -> None:
    prompt = " ".join(SYSTEM_PROMPT.lower().split())

    for contract in (
        "the final reference must preserve a recognizable head region",
        "a visible face is not required",
        "person viewed from behind with the head present",
        "helmeted or masked person with the head present",
        "side-profile person with the head and upper body present",
        "chef reference containing only coat and arms",
        "person reference cropped completely below the neck",
        "clothing-only fragment labeled as a subject",
        "set preserves_primary_identity_region to false",
        "without imposing human anatomy",
        "apply this human head-region rule only to human subjects",
    ):
        assert contract in prompt


def test_integrity_prompt_enforces_reference_semantics_and_severe_artifacts() -> None:
    prompt = " ".join(SYSTEM_PROMPT.lower().split())

    for contract in (
        "set reference_entity_semantically_valid to false",
        "a living animal or creature is labeled as an object",
        "clearly cooked or prepared culinary food may remain a valid object",
        "amorphous sauce, liquid, smoke, steam, fog, light",
        "a static scene structure such as a cathedral, building, bridge, or tree",
        "represented content",
        "set no_severe_reference_artifact to false",
        "large white or transparent erased-object-shaped cavity",
        "a severe blank patch or edit scar",
        "large white bottle-shaped hole",
        "the bottle need not remain for her identity",
    ):
        assert contract in prompt


@pytest.mark.parametrize(
    "system_prompt",
    (SYSTEM_PROMPT, SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT),
)
def test_integrity_system_prompts_require_immediate_schema_ordered_termination(
    system_prompt: str,
) -> None:
    prompt = " ".join(system_prompt.lower().split())

    for contract in (
        "return exactly one compact json object",
        "follow the supplied schema key order",
        "reason must be one concise sentence",
        "emit reason before verdict",
        "make verdict the final key",
        "close the json object immediately after verdict",
        "do not emit trailing whitespace, markdown, or explanation",
    ):
        assert contract in prompt


@pytest.mark.parametrize(
    ("phrase", "expected_reason"),
    (
        ("a brown dog", "object_creature_semantic_risk"),
        ("a living clam", "object_creature_semantic_risk"),
        ("thick red sauce", "amorphous_object_semantic_risk"),
        ("a stone cathedral", "scene_structure_object_semantic_risk"),
        ("a person on a screen", "represented_content_semantic_risk"),
    ),
)
def test_clean_full_semantic_risk_objects_are_routed_to_integrity_review(
    phrase: str,
    expected_reason: str,
) -> None:
    assert reference_semantic_risk_reason(
        reference_type="object",
        phrase=phrase,
        grounding_prompt=phrase,
    ) == expected_reason


def test_valid_cooked_lobster_and_normal_object_are_not_semantic_risk() -> None:
    for phrase in ("a cooked lobster dish", "a black camera"):
        assert (
            reference_semantic_risk_reason(
                reference_type="object",
                phrase=phrase,
                grounding_prompt=phrase,
            )
            is None
        )


@pytest.mark.parametrize(
    ("phrase", "expected_reason"),
    (
        ("a thick golden-brown sauce", "semantic_policy:amorphous_object"),
        (
            "a large domed cathedral with multiple smaller domes and arched windows",
            "semantic_policy:scene_structure_object",
        ),
        ("a light-colored dog with long fur", "semantic_policy:living_creature_object"),
        (
            "a giant clam with a textured shell and blue-spotted mantle",
            "semantic_policy:living_creature_object",
        ),
    ),
)
def test_semantic_policy_rejects_without_qwen_or_bbox_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phrase: str,
    expected_reason: str,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
        second_phrase=phrase,
    )
    judge = FakeIntegrityJudge([])
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    clip = storage.read_clip("clip-1")
    assert judge.calls == []
    assert bbox_judge.calls == []
    assert stats.entities_reviewed == 0
    assert stats.entities_rejected == 1
    assert stats.semantic_policy_rejected == 1
    assert clip.references.entities[1].status == "rejected"
    assert clip.reference_integrity is not None
    result = clip.reference_integrity.entities[1]
    assert result.status == "rejected"
    assert result.reviewed is False
    assert result.review is None
    assert result.semantic_policy_reason == expected_reason
    assert result.reason == expected_reason
    assert result.final_reference_path == result.input_reference.image_path
    assert result.source_bbox_fallback_review is None
    assert clip.pairing is not None
    assert clip.pairing.retained_entity_ids == ["e1"]
    assert ClipRecord.model_validate(clip.model_dump(mode="json")) == clip


@pytest.mark.parametrize(
    "phrase",
    (
        "a cooked red lobster on a wooden cutting board",
        "a cooked whole fish in a wok",
        "a bottle of water",
        "an oil bottle",
        "a green laser pointer emitting bright green light",
        "a dog toy",
        "a clam shell",
        "a cathedral model",
        "a model cathedral",
        "a tree branch",
    ),
)
def test_semantic_policy_hard_reject_avoids_ambiguous_false_positives(
    phrase: str,
) -> None:
    assert (
        reference_semantic_hard_reject_reason(
            reference_type="object",
            phrase=phrase,
        )
        is None
    )


def test_represented_content_remains_qwen_review_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phrase = "a person on a screen"
    assert (
        reference_semantic_hard_reject_reason(
            reference_type="object",
            phrase=phrase,
        )
        is None
    )
    assert reference_semantic_risk_reason(
        reference_type="object",
        phrase=phrase,
        grounding_prompt=phrase,
    ) == "represented_content_semantic_risk"
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
        second_phrase=phrase,
    )
    judge = FakeIntegrityJudge(
        [_invalid_reference_semantics_review("represented content is not physical")]
    )

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    result = storage.read_clip("clip-1").reference_integrity
    assert result is not None
    assert len(judge.calls) == 1
    assert stats.semantic_policy_rejected == 0
    assert result.entities[1].reviewed is True
    assert result.entities[1].review is not None
    assert result.entities[1].semantic_policy_reason is None


def test_reference_integrity_entity_state_accepts_legacy_json_without_policy_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch, second_scope="full")
    reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([]),
    )
    integrity = storage.read_clip("clip-1").reference_integrity
    assert integrity is not None
    state = integrity.entities[0]
    payload = state.model_dump(mode="json")
    assert payload.pop("semantic_policy_reason") is None

    assert ReferenceIntegrityEntityState.model_validate(payload) == state


@pytest.mark.parametrize(
    "phrase",
    (
        "a cooked lobster dish",
        "a cooked whole fish in a wok",
        "a black camera",
    ),
)
def test_clean_full_valid_object_behavior_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phrase: str,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
        second_phrase=phrase,
    )
    judge = FakeIntegrityJudge([])
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    assert judge.calls == []
    assert bbox_judge.calls == []
    assert stats.entities_skipped_review == 2
    assert stats.entities_rejected == 0
    assert all(
        reference.status == "ready"
        for reference in storage.read_clip("clip-1").references.entities
    )


def test_clean_full_human_reference_behavior_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
    )
    judge = FakeIntegrityJudge([])

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    assert judge.calls == []
    assert stats.entities_skipped_review == 2
    assert storage.read_clip("clip-1").references.entities[0].status == "ready"


def test_bottle_shaped_white_cavity_rejects_human_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
    )
    clip = storage.read_clip("clip-1")
    human = clip.references.entities[0].model_copy(
        update={
            "reference_scope": "local",
            "visible_region": "upper_body",
            "whole_entity_recognizable": False,
        }
    )
    storage.write_references_and_pairing(
        "clip-1",
        ReferencesState(entities=[human, clip.references.entities[1]]),
        clip.pairing,
    )
    assert human.image_path is not None
    image_path = storage.root / human.image_path
    with Image.open(image_path) as opened:
        damaged = opened.convert("RGBA")
        damaged.load()
    draw = ImageDraw.Draw(damaged)
    draw.rectangle((27, 23, 39, 51), fill=(255, 255, 255, 255))
    draw.rectangle((31, 16, 35, 23), fill=(255, 255, 255, 255))
    damaged.save(image_path)
    judge = FakeIntegrityJudge([_severe_reference_artifact_review()])
    bbox_judge = FakeSourceBboxFallbackJudge([_bbox_review(accept=False)])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    updated = storage.read_clip("clip-1")
    assert stats.entities_rejected == 1
    assert updated.references.entities[0].status == "rejected"
    assert updated.reference_integrity is not None
    review = updated.reference_integrity.entities[0].review
    assert review is not None
    assert review.preserves_annotated_entity_semantics is True
    assert review.preserves_primary_identity_region is True
    assert review.no_severe_reference_artifact is False
    assert review.verdict == "reject"
    assert len(bbox_judge.calls) == 1


def test_artifact_only_human_publishes_real_source_bbox_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
    )
    clip = storage.read_clip("clip-1")
    original = clip.references.entities[0].model_copy(
        update={
            "reference_scope": "local",
            "visible_region": "upper_body",
            "whole_entity_recognizable": False,
        }
    )
    storage.write_references_and_pairing(
        "clip-1",
        ReferencesState(entities=[original, clip.references.entities[1]]),
        clip.pairing,
    )
    bbox_judge = FakeSourceBboxFallbackJudge([_bbox_review(accept=True)])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([_severe_reference_artifact_review()]),
        bbox_fallback_judge=bbox_judge,
    )

    updated = storage.read_clip("clip-1")
    published = updated.references.entities[0]
    assert stats.source_bbox_fallback_attempted == 1
    assert stats.source_bbox_fallback_accepted == 1
    assert stats.entities_rejected == 0
    assert published.status == "ready"
    assert published.synthetic is False
    assert published.source_bbox_fallback is True
    assert published.source_clip_uid == "clip-1"
    assert published.source_entity_id == "e1"
    assert published.source_frame_index == 0
    assert published.source_bbox_xyxy == (4, 2, 28, 22)
    assert published.image_path is not None
    assert published.source_bbox_metadata_path is not None
    candidate_path = storage.root / published.image_path
    with Image.open(storage.frame_path("clip-1", 0)) as source_frame:
        expected_pixels = np.asarray(source_frame.convert("RGB"))[2:22, 4:28]
    with Image.open(candidate_path) as candidate:
        assert candidate.mode == "RGB"
        assert candidate.size == (24, 20)
        assert np.array_equal(np.asarray(candidate), expected_pixels)
    metadata_path = storage.root / published.source_bbox_metadata_path
    assert metadata_path.is_file()
    assert updated.pairing is not None
    assert updated.pairing.retained_entity_ids == ["e1", "e2"]
    assert updated.reference_integrity is not None
    result = updated.reference_integrity.entities[0]
    assert result.status == "accepted"
    assert result.review is not None and result.review.verdict == "reject"
    assert result.source_bbox_fallback_review is not None
    assert result.source_bbox_fallback_review.verdict == "accept"
    assert result.final_reference_path == published.image_path
    assert updated.instruction is None
    assert updated.export == ExportState()
    body = "{{image_1}} holds {{image_2}}."
    storage.write_instruction(
        "clip-1",
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=[
                InstructionLegendEntry(image_id="image_1", description="person"),
                InstructionLegendEntry(image_id="image_2", description="bracket"),
            ],
            r2v_instruction=render_inline_instruction_text(body),
        ),
    )
    dataset = DatasetExporter(storage.config, storage).export()
    assert dataset.sample_count == 1
    exported = storage.config.resolved_export_root / "references/clip-1/subject_1.png"
    with Image.open(exported) as exported_image:
        assert np.array_equal(np.asarray(exported_image), expected_pixels)


def test_cross_pair_artifact_failure_never_uses_source_bbox_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    clip = storage.read_clip("clip-1")
    cross_reference = clip.references.entities[1].model_copy(
        update={"source_clip_uid": "donor", "source_entity_id": "e2"}
    )
    storage.write_references_and_pairing(
        "clip-1",
        ReferencesState(entities=[clip.references.entities[0], cross_reference]),
        clip.pairing,
    )
    original_loader = integrity_module._source_evidence

    def load_local_source(storage_arg, *, clip_uid, reference):
        local_reference = reference.model_copy(
            update={"source_clip_uid": None, "source_entity_id": None}
        )
        return original_loader(
            storage_arg,
            clip_uid=clip_uid,
            reference=local_reference,
        )

    monkeypatch.setattr(integrity_module, "_source_evidence", load_local_source)
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([_severe_reference_artifact_review()]),
        bbox_fallback_judge=bbox_judge,
    )

    assert stats.source_bbox_fallback_attempted == 0
    assert bbox_judge.calls == []
    assert storage.read_clip("clip-1").references.entities[1].status == "rejected"


def test_source_bbox_judge_failure_preserves_original_fail_closed_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([_severe_reference_artifact_review()]),
        bbox_fallback_judge=FakeSourceBboxFallbackJudge(
            [SourceBboxFallbackJudgeFailure("malformed bbox review")]
        ),
    )

    updated = storage.read_clip("clip-1")
    assert stats.source_bbox_fallback_attempted == 1
    assert stats.source_bbox_fallback_judge_failed == 1
    assert stats.source_bbox_fallback_rejected == 1
    assert updated.references.entities[1].status == "rejected"
    assert updated.reference_integrity is not None
    assert updated.reference_integrity.entities[1].status == "rejected"


def test_source_bbox_fallback_updates_reference_edit_provenance_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
        second_synthetic=True,
    )
    clip = storage.read_clip("clip-1")
    first = clip.references.entities[0]
    generated = clip.references.entities[1]
    assert generated.image_path is not None
    raw_path = storage.selected_path("clip-1", "e2_source.png")
    raw_path.write_bytes((storage.root / generated.image_path).read_bytes())
    raw_relative = storage.relative_artifact_path(raw_path)
    source_reference = generated.model_copy(
        update={
            "image_path": raw_relative,
            "synthetic": False,
            "generation_metadata_path": None,
            "generation_source_sha256": None,
            "generation_output_sha256": None,
        }
    )
    original_generation_metadata = generated.generation_metadata_path
    assert original_generation_metadata is not None
    reference_edit = ReferenceEditState(
        status="ready",
        entities=[
            ReferenceEditEntityState(
                entity_id="e1",
                route="complete",
                status="not_required",
                source_reference=first,
                source_image_path=first.image_path,
                output_image_path=first.image_path,
            ),
            ReferenceEditEntityState(
                entity_id="e2",
                route="complete",
                status="accepted",
                source_reference=source_reference,
                source_image_path=raw_relative,
                output_image_path=generated.image_path,
                operation="add_entity_background",
                operations=["add_entity_background"],
                metadata_path=original_generation_metadata,
            ),
        ],
    )
    storage.write_reference_edit_result(
        "clip-1",
        clip.references,
        clip.pairing,
        reference_edit,
    )

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([_severe_reference_artifact_review()]),
        bbox_fallback_judge=FakeSourceBboxFallbackJudge(
            [_bbox_review(accept=True)]
        ),
    )

    updated = storage.read_clip("clip-1")
    assert stats.source_bbox_fallback_accepted == 1
    final_reference = updated.references.entities[1]
    assert final_reference.source_bbox_fallback is True
    assert final_reference.synthetic is False
    assert updated.reference_edit is not None
    final_edit = updated.reference_edit.entities[1]
    assert final_edit.status == "fallback"
    assert final_edit.fallback_policy == "source_bbox_fallback"
    assert final_edit.output_image_path == final_reference.image_path
    assert (
        final_edit.source_bbox_fallback_metadata_path
        == final_reference.source_bbox_metadata_path
    )
    assert final_edit.metadata_path == original_generation_metadata
    assert final_edit.operations == ["add_entity_background"]
    ClipRecord.model_validate(updated.model_dump(mode="json"))


def test_source_bbox_fields_are_backward_compatible_for_existing_clip_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    for reference in payload["references"]["entities"]:
        reference.pop("source_bbox_fallback")
        reference.pop("source_bbox_xyxy")
        reference.pop("source_bbox_metadata_path")

    restored = ClipRecord.model_validate(payload)

    assert all(
        reference.source_bbox_fallback is False
        for reference in restored.references.entities
    )


def test_source_bbox_fallback_review_requires_every_strict_boolean() -> None:
    accepted = _bbox_review(accept=True).model_dump(mode="json")

    with pytest.raises(ValueError):
        SourceBboxFallbackReview.model_validate(
            {**accepted, "target_remains_dominant": 1}
        )
    with pytest.raises(ValueError, match="must match all strict checks"):
        SourceBboxFallbackReview.model_validate(
            {**accepted, "target_remains_dominant": False}
        )


def test_qwen_source_bbox_reviewer_uses_three_images_and_strict_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _bbox_review(accept=True)
    completions = FakeIntegrityCompletions(
        [(expected.model_dump_json(), "stop")]
    )
    service = _config(tmp_path, monkeypatch).qwen.reference_integrity_judge
    assert service is not None
    judge = QwenSourceBboxFallbackJudge(
        service,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        ),
    )

    attempt = judge.review(
        source_context=Image.new("RGB", (12, 10)),
        failed_reference=Image.new("RGB", (8, 8)),
        source_bbox_candidate=Image.new("RGB", (9, 7)),
        reference_type="subject",
        phrase="a woman in a red top",
        grounding_prompt="woman in red top near center",
        reference_scope="local",
    )

    assert attempt.review == expected
    call = completions.calls[0]
    response_format = call["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    messages = call["messages"]
    assert isinstance(messages, list)
    assert sum(
        item.get("type") == "image_url"
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict)
    ) == 3


def test_large_enclosed_alpha_hole_is_review_suspicion_not_rejection() -> None:
    rgba = np.zeros((80, 80, 4), dtype=np.uint8)
    rgba[8:72, 8:72, :3] = 80
    rgba[8:72, 8:72, 3] = 255
    rgba[28:52, 28:52, 3] = 0

    diagnostics = reference_topology_diagnostics(Image.fromarray(rgba))

    assert diagnostics.suspicious is True
    assert diagnostics.enclosed_transparent_hole_count == 1
    assert "large_enclosed_alpha_hole" in diagnostics.suspicion_reasons


def test_integrity_rejects_entity_and_invalidates_instruction_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    judge = FakeIntegrityJudge([_review(accept=False, reason="major surface missing")])

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    clip = storage.read_clip("clip-1")
    assert stats.entities_skipped_review == 1
    assert stats.entities_rejected == 1
    assert clip.pairing is not None
    assert clip.pairing.retained_entity_ids == ["e1"]
    assert clip.references.entities[1].status == "rejected"
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_integrity_rejects_reference_reinterpreted_as_contained_food(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_phrase="a steaming clay pot of stew",
    )
    judge = FakeIntegrityJudge([_semantic_reinterpretation_review()])

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    clip = storage.read_clip("clip-1")
    assert judge.calls[0]["phrase"] == "a steaming clay pot of stew"
    assert stats.entities_rejected == 1
    assert clip.references.entities[1].status == "rejected"
    assert clip.reference_integrity is not None
    result = clip.reference_integrity.entities[1]
    assert result.review is not None
    assert result.review.preserves_annotated_entity_semantics is False
    assert result.status == "rejected"


def test_integrity_rejects_headless_human_subject_from_final_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
    )
    clip = storage.read_clip("clip-1")
    human = clip.references.entities[0].model_copy(
        update={
            "reference_scope": "local",
            "visible_region": "upper_body",
            "whole_entity_recognizable": False,
        }
    )
    storage.write_references_and_pairing(
        "clip-1",
        ReferencesState(entities=[human, clip.references.entities[1]]),
        clip.pairing,
    )
    judge = FakeIntegrityJudge([_primary_identity_loss_review()])
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    updated = storage.read_clip("clip-1")
    assert stats.entities_rejected == 1
    assert updated.references.entities[0].status == "rejected"
    assert updated.pairing is not None
    assert updated.pairing.retained_entity_ids == ["e2"]
    result = updated.reference_integrity
    assert result is not None
    assert result.entities[0].review is not None
    assert result.entities[0].review.preserves_primary_identity_region is False
    assert bbox_judge.calls == []


@pytest.mark.parametrize(
    ("reason", "reference_type"),
    (
        ("only disconnected scallion fragments remain", "object"),
        ("a major object surface is missing", "object"),
        ("the subject retains only torso and arms without identity evidence", "subject"),
    ),
)
def test_integrity_semantic_failures_reject_only_the_reviewed_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    reference_type: str,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full" if reference_type == "subject" else "local",
    )
    clip = storage.read_clip("clip-1")
    entity_id = "e1" if reference_type == "subject" else "e2"
    if entity_id == "e1":
        first = clip.references.entities[0].model_copy(
            update={
                "reference_scope": "local",
                "visible_region": "upper_body",
                "whole_entity_recognizable": False,
            }
        )
        storage.write_references_and_pairing(
            "clip-1",
            ReferencesState(entities=[first, clip.references.entities[1]]),
            clip.pairing,
        )
    judge = FakeIntegrityJudge([_review(accept=False, reason=reason)])
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    updated = storage.read_clip("clip-1")
    rejected = next(item for item in updated.references.entities if item.entity_id == entity_id)
    assert rejected.status == "rejected"
    assert stats.entities_rejected == 1
    assert bbox_judge.calls == []


def test_legitimate_bracket_cutout_is_reviewed_and_may_be_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope="full",
        second_hole=True,
    )
    judge = FakeIntegrityJudge([_review(accept=True, reason="source-matching cutout")])

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    assert stats.topology_suspicious == 1
    assert stats.entities_accepted == 2
    assert storage.read_clip("clip-1").references.entities[1].status == "ready"


@pytest.mark.parametrize(
    ("scope", "synthetic", "hole"),
    (("local", False, False), ("full", True, False), ("full", False, True)),
)
def test_local_synthetic_or_suspicious_reference_always_uses_qwen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    synthetic: bool,
    hole: bool,
) -> None:
    storage = _storage_with_ready_pair(
        tmp_path,
        monkeypatch,
        second_scope=scope,
        second_synthetic=synthetic,
        second_hole=hole,
    )
    judge = FakeIntegrityJudge([_review(accept=True)])

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    assert len(judge.calls) == 1
    assert stats.entities_reviewed == 1
    assert storage.read_clip("clip-1").references.entities[1].status == "ready"


def test_integrity_judge_failure_fails_closed_for_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    judge = FakeIntegrityJudge(
        [ReferenceIntegrityJudgeFailure("structured output invalid")]
    )
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    clip = storage.read_clip("clip-1")
    assert stats.judge_failed == 1
    assert clip.reference_integrity is not None
    state = clip.reference_integrity.entities[1]
    assert state.judge_failed is True
    assert state.status == "rejected"
    assert bbox_judge.calls == []


def test_malformed_integrity_repair_failure_never_uses_bbox_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    judge, completions = _qwen_integrity_judge(
        storage.config,
        [("{", "length"), ("{}", "stop")],
    )
    bbox_judge = FakeSourceBboxFallbackJudge([])

    stats = reference_integrity_clips(
        storage.config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )

    assert len(completions.calls) == 2
    assert stats.judge_failed == 1
    assert stats.source_bbox_fallback_attempted == 0
    assert bbox_judge.calls == []
    assert storage.read_clip("clip-1").references.entities[1].status == "rejected"


def test_audit_reports_stage_density_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_with_ready_pair(tmp_path, monkeypatch)
    reference_integrity_clips(
        storage.config,
        storage,
        judge=FakeIntegrityJudge([_review(accept=False)]),
    )

    summary = audit_entity_composition(
        run_root=storage.root,
        output_root=tmp_path / "audit",
    )

    densities = summary["reference_density_by_stage"]
    assert densities["post_pair"]["histogram"]["2"] == 1
    assert densities["post_reference_edit"]["histogram"]["2"] == 1
    assert densities["post_reference_integrity"]["histogram"]["1"] == 1
    assert densities["final_export"]["histogram"]["0"] == 0
    assert summary["funnel_by_type"]["final_ready"]["subject"] == 1
    assert summary["integrity_rejections"]["count"] == 1
