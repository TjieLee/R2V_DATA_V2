from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
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
    SYSTEM_PROMPT,
    ReferenceIntegrityJudgeFailure,
    ReferenceIntegrityReviewAttempt,
    reference_integrity_clips,
    reference_topology_diagnostics,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    ReferenceIntegrityReview,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_inline_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage


def _review(
    *,
    accept: bool,
    reason: str = "reviewed",
    preserves_semantics: bool | None = None,
) -> ReferenceIntegrityReview:
    semantic_fidelity = accept if preserves_semantics is None else preserves_semantics
    return ReferenceIntegrityReview(
        matches_target=accept,
        preserves_annotated_entity_semantics=semantic_fidelity,
        recognizable_as_named_entity=accept,
        structurally_complete_for_scope=accept,
        no_major_missing_regions=accept,
        no_unnatural_holes_or_surface_loss=accept,
        no_unrelated_entity_dominance=accept,
        usable_as_independent_reference=accept,
        verdict="accept" if accept else "reject",
        reason=reason,
    )


def _semantic_reinterpretation_review() -> ReferenceIntegrityReview:
    return ReferenceIntegrityReview(
        matches_target=True,
        preserves_annotated_entity_semantics=False,
        recognizable_as_named_entity=True,
        structurally_complete_for_scope=True,
        no_major_missing_regions=True,
        no_unnatural_holes_or_surface_loss=True,
        no_unrelated_entity_dominance=True,
        usable_as_independent_reference=True,
        verdict="reject",
        reason="only the stew remains; the annotated clay pot is missing",
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

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    updated = storage.read_clip("clip-1")
    rejected = next(item for item in updated.references.entities if item.entity_id == entity_id)
    assert rejected.status == "rejected"
    assert stats.entities_rejected == 1


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

    stats = reference_integrity_clips(storage.config, storage, judge=judge)

    clip = storage.read_clip("clip-1")
    assert stats.judge_failed == 1
    assert clip.reference_integrity is not None
    state = clip.reference_integrity.entities[1]
    assert state.judge_failed is True
    assert state.status == "rejected"


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
