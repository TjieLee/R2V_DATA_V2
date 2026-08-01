from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.v3.config import (
    BackgroundConfig,
    FramesConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceScopeConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundReferenceState,
    BackgroundRemovalAttempt,
    BackgroundRemovalReview,
    ClipRecord,
    ClipSource,
    CoverageState,
    DatasetReference,
    DatasetSample,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    ReferencesState,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage
from run_pipeline_v3 import run_pipeline_v3


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug: bool = False,
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
    annotation_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    remove_model = pretrained / "Qwen" / "Qwen-Image-Edit-2511"
    adapter = user_models / "Qwen-Image-Edit-2511-Object-Remover"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "pilot",
        export_root=writable / "datasets" / "pilot-v1",
        source=SourceConfig(limit=100),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(annotation_model)),
            instruction_writer=QwenServiceConfig(model=str(annotation_model)),
        ),
        sam3=v3_config_module.Sam3Config(
            model_path=user_models / "sam3" / "checkpoint.pt"
        ),
        remove=RemoveConfig(
            base_model_path=remove_model,
            adapter_path=adapter,
        ),
        debug=v3_config_module.DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _entity(entity_id: str, phrase: str) -> AnnotationEntity:
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type="subject" if entity_id == "e1" else "object",
        phrase=phrase,
        grounding_prompt=phrase.lower(),
    )


def _visibility_summary(
    visible_frame_count: int,
    *,
    qualifies: bool,
) -> EntityVisibilitySummary:
    return EntityVisibilitySummary(
        status="ready" if visible_frame_count else "not_found",
        visible_frame_slots=list(range(visible_frame_count)),
        visible_frame_count=visible_frame_count,
        coverage_ratio=visible_frame_count / 10,
        qualifies=qualifies,
        per_frame_area_ratio=[
            0.1 if slot < visible_frame_count else 0.0
            for slot in range(10)
        ],
        per_frame_confidence=[
            0.9 if slot < visible_frame_count else None
            for slot in range(10)
        ],
    )


def _instruction_state(*, include_background: bool = False) -> InstructionState:
    body = (
        "Use the stable appearance of {{image_1}} and place it at the center "
        "of the frame"
    )
    legend = [
        InstructionLegendEntry(
            image_id="image_1",
            description="the stable appearance of the woman in a yellow coat",
        )
    ]
    if include_background:
        body += " while using {{image_2}} as the overall background."
        legend.append(
            InstructionLegendEntry(
                image_id="image_2",
                description="the bright plaza environment",
            )
        )
    else:
        body += "."
    return InstructionState(
        status="ready",
        instruction_body_template=body,
        reference_legend=legend,
        r2v_instruction=render_instruction_text(body, legend),
    )


def _source_video_path(storage: RunStorage, clip_uid: str) -> str:
    return str(
        storage.config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"
    )


def _clip_source(storage: RunStorage, clip_uid: str) -> ClipSource:
    return ClipSource(
        video_path=_source_video_path(storage, clip_uid),
        parent_video_id="parent",
        clip_suffix="1_0",
        source_index=0,
        caption_raw="",
        metadata={},
    )


def _tracked_masks(
    clip_uid: str,
    *,
    counts: str = "encoded",
) -> TrackedMasksArtifact:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    confidence = 0.9 if counts == "encoded" else 0.8
    empty = encode_binary_mask(np.zeros_like(mask))
    return TrackedMasksArtifact(
        clip_uid=clip_uid,
        height=4,
        width=5,
        entities={
            "e1": TrackedEntityMasks(
                status="ready",
                reference_type="subject",
                grounding_prompt="a woman",
                backend_object_ids=["7"],
                frames=[
                    TrackedMaskFrame(
                        slot=slot,
                        present=slot == 0,
                        confidence=confidence if slot == 0 else None,
                        backend_confidences=(
                            [confidence] if slot == 0 else []
                        ),
                        backend_object_ids=["7"] if slot == 0 else [],
                        area_pixels=int(mask.sum()) if slot == 0 else 0,
                        area_ratio=(
                            float(mask.mean()) if slot == 0 else 0.0
                        ),
                        bbox_xyxy=(2, 1, 4, 3) if slot == 0 else None,
                        rle=encode_binary_mask(mask) if slot == 0 else empty,
                    )
                    for slot in range(10)
                ],
            )
        },
    )


def _create_exportable_clip(
    storage: RunStorage,
    *,
    clip_uid: str = "clip-1",
    include_background: bool = True,
) -> None:
    storage.create_clip(
        clip_uid=clip_uid,
        source=_clip_source(storage, clip_uid),
    )
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="A woman walks beside a small table.",
            entities=[
                _entity("e1", "A woman"),
                _entity("e2", "a small table"),
            ],
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            required_visible_frames=7,
            entity_visibility_summary={
                "e1": _visibility_summary(7, qualifies=True),
                "e2": _visibility_summary(3, qualifies=False),
            },
        ),
    )
    entity_path = storage.selected_path(clip_uid, "e1.png")
    Image.new("RGBA", (12, 10), (10, 20, 30, 128)).save(entity_path)
    ready = EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="local",
        visible_region="upper_body",
        whole_entity_recognizable=False,
        identity_features_visible=True,
        scope_reason="coherent upper body",
        image_path=storage.relative_artifact_path(entity_path),
        source_frame_index=2,
    )
    rejected = EntityReferenceState(
        entity_id="e2",
        status="rejected",
        reference_scope="reject",
        visible_region="central",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason="fragmented mask",
    )
    background = None
    background_token = None
    if include_background:
        frame_path = storage.frame_path(clip_uid, 3)
        Image.new("RGB", (14, 9), (40, 50, 60)).save(
            frame_path,
            format="JPEG",
        )
        background = BackgroundReferenceState(
            status="clean_raw",
            source_image_path=storage.relative_artifact_path(frame_path),
            output_image_path=storage.relative_artifact_path(frame_path),
            source_frame_slot=3,
            source_frame_index=3,
            source_foreground_area_pixels=0,
            source_foreground_area_ratio=0.0,
        )
        background_token = "<ref_bg_1>"
    storage.write_references(
        clip_uid,
        ReferencesState(entities=[ready, rejected], background=background),
    )
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
            background_token=background_token,
        ),
    )
    storage.write_instruction(
        clip_uid,
        _instruction_state(include_background=include_background),
    )
    storage.write_export(
        clip_uid,
        ExportState(accepted=True, reason=None),
    )


def _initialize_storage_with_complete_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_masks: bool = False,
) -> RunStorage:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    if with_masks:
        storage.create_clip(
            clip_uid="clip-1",
            source=_clip_source(storage, "clip-1"),
        )
        storage.write_masks("clip-1", _tracked_masks("clip-1"))
    _create_exportable_clip(storage, include_background=False)
    return storage


def test_v3_config_loads_32b_defaults_without_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 100\n"
        "sam3:\n"
        f"  model_path: {config.sam3.model_path}\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.qwen.annotation.model.endswith("Qwen3-VL-32B-Instruct")
    assert loaded.frames.count == 10
    assert loaded.coverage.required_visible_frames == 7
    assert loaded.background.raw_foreground_area_ratio == 0.0
    assert loaded.background.max_pending_remove_area_ratio == 0.50
    assert loaded.remove.fallback_to_raw is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "frames",
            FramesConfig(count=8),
            "exactly 10",
        ),
        (
            "reference_scope",
            ReferenceScopeConfig(allow_synthetic_completion=True),
            "synthetic entity completion",
        ),
        (
            "background",
            BackgroundConfig(raw_foreground_area_ratio=0.01),
            "raw_foreground_area_ratio",
        ),
        (
            "remove",
            RemoveConfig(fallback_to_raw=True),
            "fallback_to_raw",
        ),
    ],
)
def test_v3_config_rejects_non_negotiable_policy_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=message):
        replace(config, **{field: value}).validate()


def test_v3_config_rejects_unknown_sam3_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="unsupported V3 SAM3 backend"):
        replace(
            config,
            sam3=v3_config_module.Sam3Config(backend="unknown"),
        ).validate()


def test_v3_config_allows_missing_sam3_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        sam3=v3_config_module.Sam3Config(model_path=None),
    )

    changed.validate()


def test_v3_config_rejects_sam3_model_path_outside_allowed_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match="sam3.model_path must be inside an allowed model root",
    ):
        replace(
            config,
            sam3=v3_config_module.Sam3Config(
                model_path=tmp_path / "outside-model.pt"
            ),
        ).validate()



@pytest.mark.parametrize("required", [0, 11])
def test_coverage_required_visible_frames_must_be_in_range(
    required: int,
) -> None:
    with pytest.raises(ValidationError, match="greater than or equal|less than"):
        CoverageState(
            passed=False,
            required_visible_frames=required,
        )


def test_coverage_default_is_seven_and_eight_is_also_valid() -> None:
    assert CoverageState(passed=False).required_visible_frames == 7
    assert (
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            required_visible_frames=8,
            entity_visibility_summary={
                "e1": _visibility_summary(8, qualifies=True)
            },
        ).required_visible_frames
        == 8
    )


def test_qualifying_coverage_requires_visible_frame_counts() -> None:
    with pytest.raises(
        ValidationError,
        match="visible-frame summaries",
    ):
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
        )


def test_v3_config_rejects_output_outside_writable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="run_root must be inside"):
        replace(config, run_root=tmp_path / "outside").validate()


def test_single_clip_json_lifecycle_and_single_mask_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123", created_at="2026-07-30T00:00:00+00:00")
    source = _clip_source(storage, "clip-1")

    first = storage.create_clip(clip_uid="clip-1", source=source)
    second = storage.create_clip(clip_uid="clip-1", source=source)
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            t2v_caption="A woman walks through a plaza.",
            entities=[_entity("e1", "A woman")],
        ),
    )
    storage.write_coverage(
        "clip-1",
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={
                "e1": _visibility_summary(7, qualifies=True)
            },
        ),
    )
    masks_path = storage.write_masks(
        "clip-1",
        _tracked_masks("clip-1"),
    )

    assert first == second
    assert storage.read_clip("clip-1").annotation is not None
    assert masks_path.name == "masks.rle.json"
    clip_files = {
        path.name
        for path in storage.clip_dir("clip-1").iterdir()
        if path.is_file()
    }
    assert clip_files == {"clip.json", "masks.rle.json"}
    assert len(list(config.resolved_run_root.rglob("clip.json"))) == 1
    assert not (config.resolved_run_root / "manifests").exists()
    assert not (config.resolved_run_root / "samples").exists()
    assert not list(config.resolved_run_root.rglob("annotations.json"))
    assert not list(config.resolved_run_root.rglob("ranking_metadata.json"))
    assert not list(config.resolved_run_root.rglob("*.jsonl"))
    assert not (storage.clip_dir("clip-1") / "debug").exists()
    with pytest.raises(RuntimeError, match="debug artifact saving is disabled"):
        storage.debug_path("clip-1", "candidate.png")


@pytest.mark.parametrize(
    ("upstream", "cleared_sections"),
    [
        (
            "annotation",
            {
                "coverage",
                "references",
                "pairing",
                "reference_finalization",
                "instruction",
                "export",
            },
        ),
        (
            "masks",
            {
                "coverage",
                "references",
                "pairing",
                "reference_finalization",
                "instruction",
                "export",
            },
        ),
        (
            "coverage",
            {
                "references",
                "pairing",
                "reference_finalization",
                "instruction",
                "export",
            },
        ),
        (
            "references",
            {"pairing", "reference_finalization", "instruction", "export"},
        ),
        ("pairing", {"reference_finalization", "instruction", "export"}),
        ("instruction", {"export"}),
    ],
)
def test_changed_upstream_content_invalidates_only_downstream_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: str,
    cleared_sections: set[str],
) -> None:
    storage = _initialize_storage_with_complete_clip(
        tmp_path,
        monkeypatch,
        with_masks=upstream == "masks",
    )
    before = storage.read_clip("clip-1")

    if upstream == "annotation":
        assert before.annotation is not None
        storage.write_annotation(
            "clip-1",
            before.annotation.model_copy(
                update={
                    "t2v_caption": (
                        before.annotation.t2v_caption + " The shot then ends."
                    )
                }
            ),
        )
    elif upstream == "masks":
        storage.write_masks(
            "clip-1",
            _tracked_masks("clip-1", counts="changed"),
        )
    elif upstream == "coverage":
        assert before.coverage is not None
        summary = before.coverage.entity_visibility_summary["e1"]
        confidence = list(summary.per_frame_confidence)
        confidence[0] = 0.85
        storage.write_coverage(
            "clip-1",
            before.coverage.model_copy(
                update={
                    "entity_visibility_summary": {
                        **before.coverage.entity_visibility_summary,
                        "e1": summary.model_copy(
                            update={"per_frame_confidence": confidence}
                        ),
                    }
                }
            ),
        )
    elif upstream == "references":
        ready = before.references.entities[0].model_copy(
            update={"scope_reason": "updated local scope"}
        )
        storage.write_references(
            "clip-1",
            before.references.model_copy(
                update={"entities": [ready, *before.references.entities[1:]]}
            ),
        )
    elif upstream == "pairing":
        assert before.pairing is not None
        storage.write_pairing(
            "clip-1",
            PairingState(status="rejected", reason="updated pairing"),
        )
    else:
        assert before.instruction is not None
        body = (
            before.instruction.instruction_body_template
            + " Keep the camera steady."
        )
        storage.write_instruction(
            "clip-1",
            before.instruction.model_copy(
                update={
                    "instruction_body_template": body,
                    "r2v_instruction": render_instruction_text(
                        body,
                        before.instruction.reference_legend,
                    ),
                }
            ),
        )

    after = storage.read_clip("clip-1")
    for section in cleared_sections:
        if section == "references":
            assert after.references == ReferencesState()
        elif section == "export":
            assert after.export == ExportState()
        else:
            assert getattr(after, section) is None
    if upstream == "masks":
        masks = TrackedMasksArtifact.model_validate_json(
            (storage.clip_dir("clip-1") / "masks.rle.json").read_text(
                encoding="utf-8"
            )
        )
        assert masks == _tracked_masks("clip-1", counts="changed")
    else:
        assert getattr(after, upstream) != getattr(before, upstream)


def test_repeated_identical_writes_preserve_all_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(
        tmp_path,
        monkeypatch,
        with_masks=True,
    )
    before = storage.read_clip("clip-1")
    clip_bytes = storage.clip_path("clip-1").read_bytes()
    masks = _tracked_masks("clip-1")

    assert before.annotation is not None
    assert before.coverage is not None
    assert before.pairing is not None
    assert before.instruction is not None
    storage.write_annotation("clip-1", before.annotation)
    storage.write_masks("clip-1", masks)
    storage.write_coverage("clip-1", before.coverage)
    storage.write_references("clip-1", before.references)
    storage.write_pairing("clip-1", before.pairing)
    storage.write_instruction("clip-1", before.instruction)
    storage.write_export("clip-1", before.export)

    assert storage.read_clip("clip-1") == before
    assert storage.clip_path("clip-1").read_bytes() == clip_bytes


@pytest.mark.parametrize(
    "missing_field",
    ["source_image_path", "source_frame_slot", "source_frame_index"],
)
def test_pending_remove_requires_complete_source_provenance(
    missing_field: str,
) -> None:
    values: dict[str, object] = {
        "status": "pending_remove",
        "source_image_path": "clips/clip-1/frames/03.jpg",
        "source_frame_slot": 3,
        "source_frame_index": 30,
        "source_mask_path": "clips/clip-1/masks.rle.json",
        "source_foreground_area_pixels": 2,
        "source_foreground_area_ratio": 0.02,
    }
    del values[missing_field]

    with pytest.raises(ValidationError, match="requires source_image_path"):
        BackgroundReferenceState.model_validate(values)


def test_ready_pairing_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="at least one retained entity"):
        PairingState(status="ready")


def test_rejected_pairing_cannot_retain_tokens() -> None:
    with pytest.raises(ValidationError, match="must clear retained IDs and tokens"):
        PairingState(
            status="rejected",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
            reason="no usable pair",
        )


def test_entity_pairing_token_cannot_use_background_token() -> None:
    with pytest.raises(ValidationError, match="entity tokens cannot use"):
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_bg_1>"},
        )


def test_accepted_export_requires_all_cross_section_state() -> None:
    with pytest.raises(ValidationError, match="requires ready annotation"):
        ClipRecord(
            clip_uid="clip-1",
            source=ClipSource(
                video_path="/mnt/workspace/public/dataset/video.mp4",
                parent_video_id="parent",
                clip_suffix="1_0",
                source_index=0,
                caption_raw="",
                metadata={},
            ),
            export=ExportState(accepted=True, reason=None),
        )


def test_clip_cross_section_validator_rejects_inconsistent_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    record = storage.read_clip("clip-1")
    payload = record.model_dump(mode="json")
    payload["instruction"]["instruction_body_template"] = (
        "Use {{image_1}} and {{image_2}} to generate the shot."
    )
    payload["instruction"]["reference_legend"].append(
        {
            "image_id": "image_2",
            "description": "an extra reference",
        }
    )
    payload["instruction"]["r2v_instruction"] = (
        "Use Image 1 and Image 2 to generate the shot.\n\n"
        "Image 1: the stable appearance of the woman in a yellow coat\n"
        "Image 2: an extra reference"
    )

    with pytest.raises(ValidationError, match="legend must match final pairing"):
        ClipRecord.model_validate(payload)


def test_clip_cross_section_validator_rejects_unknown_qualifying_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    payload["coverage"]["qualifying_entity_ids"] = ["missing"]
    payload["coverage"]["entity_visibility_summary"]["missing"] = (
        payload["coverage"]["entity_visibility_summary"].pop("e1")
    )

    with pytest.raises(ValidationError, match="must exist in annotation"):
        ClipRecord.model_validate(payload)


def test_clip_cross_section_validator_rejects_unknown_reference_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    payload["references"]["entities"][0]["entity_id"] = "missing"

    with pytest.raises(ValidationError, match="correspond to annotation"):
        ClipRecord.model_validate(payload)


def test_clip_cross_section_validator_requires_ready_retained_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    payload["pairing"]["retained_entity_ids"] = ["e2"]
    payload["pairing"]["tokens"] = {"e2": "<ref_object_1>"}

    with pytest.raises(ValidationError, match="retain every ready"):
        ClipRecord.model_validate(payload)


def test_clip_cross_section_validator_requires_retained_qualifying_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    payload = storage.read_clip("clip-1").model_dump(mode="json")
    first = payload["references"]["entities"][0]
    first.update(
        {
            "status": "rejected",
            "reference_scope": "reject",
            "visible_region": "custom",
            "whole_entity_recognizable": False,
            "identity_features_visible": False,
            "scope_reason": "not reusable",
            "image_path": None,
            "source_frame_index": None,
        }
    )
    second = payload["references"]["entities"][1]
    second.update(
        {
            "status": "ready",
            "reference_scope": "local",
            "visible_region": "central",
            "whole_entity_recognizable": False,
            "identity_features_visible": True,
            "scope_reason": "coherent local object region",
            "image_path": "clips/clip-1/selected/e2.png",
            "source_frame_index": 2,
        }
    )
    payload["pairing"]["retained_entity_ids"] = ["e2"]
    payload["pairing"]["tokens"] = {"e2": "<ref_object_1>"}

    with pytest.raises(ValidationError, match="at least one qualifying entity"):
        ClipRecord.model_validate(payload)


def test_create_clip_rejects_video_outside_public_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")

    with pytest.raises(ValueError, match="source.video_path must be inside"):
        storage.create_clip(
            clip_uid="clip-1",
            source=ClipSource(
                video_path=str((tmp_path / "private" / "video.mp4").resolve()),
                parent_video_id="parent",
                clip_suffix="1_0",
                source_index=0,
                caption_raw="",
                metadata={},
            ),
        )


def test_exporter_reads_only_background_output_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _initialize_storage_with_complete_clip(tmp_path, monkeypatch)
    clip = storage.read_clip("clip-1")
    output_path = storage.selected_path("clip-1", "bg_removed.png")
    Image.new("RGB", (9, 7), (70, 80, 90)).save(output_path)
    background = BackgroundReferenceState(
        status="ready_removed",
        source_image_path="clips/clip-1/frames/missing-source.jpg",
        output_image_path=storage.relative_artifact_path(output_path),
        source_frame_slot=3,
        source_frame_index=30,
        source_mask_path="clips/clip-1/source-mask.png",
        generation_mask_path="clips/clip-1/generation-mask.png",
        source_foreground_area_pixels=2,
        source_foreground_area_ratio=0.02,
        removal_backend="qwen_image_edit_2511_object_remover",
        removal_seed=0,
        generation_mask_dilation_pixels=0,
        generation_mask_area_pixels=2,
        generation_mask_area_ratio=0.02,
        output_sha256="a" * 64,
        removal_attempts=[
            BackgroundRemovalAttempt(
                seed=0,
                status="accepted",
                runtime_seconds=1.0,
                candidate_sha256="a" * 64,
                review=BackgroundRemovalReview(
                    verdict="accept",
                    foreground_absent=True,
                    foreground_not_reconstructed=True,
                    no_new_salient_entity=True,
                    background_only_in_repaired_region=True,
                    background_continuity_ok=True,
                    no_visible_artifacts=True,
                    reason="accepted for exporter path test",
                ),
            )
        ],
    )
    storage.write_references(
        "clip-1",
        clip.references.model_copy(update={"background": background}),
    )
    storage.write_pairing(
        "clip-1",
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
            background_token="<ref_bg_1>",
        ),
    )
    storage.write_instruction(
        "clip-1",
        _instruction_state(include_background=True),
    )
    storage.write_export("clip-1", ExportState(accepted=True, reason=None))

    dataset = DatasetExporter(storage.config, storage).export()

    assert dataset.reference_count == 2
    exported_background = (
        storage.config.resolved_export_root
        / "references"
        / "clip-1"
        / "background_1.png"
    )
    assert exported_background.is_file()


def test_run_metadata_is_resumable_and_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)

    created = storage.initialize(
        git_commit="abc123",
        created_at="2026-07-30T00:00:00+00:00",
    )
    resumed = storage.initialize(
        git_commit="abc123",
        created_at="later-is-ignored",
    )

    assert resumed == created
    with pytest.raises(ValueError, match="does not match"):
        storage.initialize(git_commit="different")


def test_failures_use_one_structured_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")

    storage.append_failure(
        clip_uid="clip-1",
        stage="rank",
        reason="no valid candidate",
        created_at="2026-07-30T00:00:00+00:00",
    )
    storage.append_failure(
        clip_uid="clip-2",
        stage="remove",
        reason="validation failed",
        created_at="2026-07-30T00:00:01+00:00",
    )

    records = [
        json.loads(line)
        for line in storage.failures_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["clip_uid"] for record in records] == ["clip-1", "clip-2"]
    assert list(config.resolved_run_root.glob("*.jsonl")) == [
        storage.failures_path
    ]


def test_compact_export_contains_only_accepted_training_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(
        git_commit="abc123",
        created_at="2026-07-30T00:00:00+00:00",
    )
    _create_exportable_clip(storage)
    storage.create_clip(
        clip_uid="rejected-clip",
        source=_clip_source(storage, "rejected-clip").model_copy(
            update={"clip_suffix": "2_0", "source_index": 1}
        ),
    )

    dataset = DatasetExporter(config, storage).export(
        created_at="2026-07-30T01:00:00+00:00"
    )

    assert dataset.sample_count == 1
    assert dataset.reference_count == 2
    assert {path.name for path in config.resolved_export_root.iterdir()} == {
        "dataset.json",
        "samples.jsonl",
        "references",
    }
    sample = json.loads(
        (config.resolved_export_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    assert sample["sample_id"] == "clip-1"
    assert {reference["entity_id"] for reference in sample["references"]} == {
        "e1",
        None,
    }
    assert all(
        not Path(reference["image_path"]).is_absolute()
        and (config.resolved_export_root / reference["image_path"]).is_file()
        for reference in sample["references"]
    )
    assert {
        path.relative_to(config.resolved_export_root).as_posix()
        for path in (config.resolved_export_root / "references").rglob("*.png")
    } == {reference["image_path"] for reference in sample["references"]}
    entity_path = config.resolved_export_root / next(
        reference["image_path"]
        for reference in sample["references"]
        if reference["type"] == "entity"
    )
    background_path = config.resolved_export_root / next(
        reference["image_path"]
        for reference in sample["references"]
        if reference["type"] == "background"
    )
    with Image.open(entity_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (12, 10)
    with Image.open(background_path) as image:
        assert image.mode == "RGB"
        assert image.size == (14, 9)
    dataset_text = (config.resolved_export_root / "dataset.json").read_text(
        encoding="utf-8"
    )
    assert "127.0.0.1" not in dataset_text
    assert str(config.resolved_run_root) not in dataset_text


def test_entity_la_png_export_preserves_alpha(tmp_path: Path) -> None:
    source = tmp_path / "entity-la.png"
    destination = tmp_path / "exported" / "entity.png"
    Image.new("LA", (2, 1), (120, 37)).save(source)

    DatasetExporter._copy_png(source, destination, background=False)

    with Image.open(destination) as image:
        assert image.mode == "LA"
        assert image.getbands() == ("L", "A")
        assert list(image.getchannel("A").getdata()) == [37, 37]


def test_export_refuses_to_destroy_existing_dataset_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    _create_exportable_clip(storage, include_background=False)
    config.resolved_export_root.mkdir(parents=True)
    sentinel = config.resolved_export_root / "keep.txt"
    sentinel.write_text("valuable existing dataset", encoding="utf-8")

    with pytest.raises(FileExistsError, match="dataset_root already exists"):
        DatasetExporter(config, storage).export()

    assert sentinel.read_text(encoding="utf-8") == "valuable existing dataset"
    assert list(config.resolved_export_root.iterdir()) == [sentinel]


def test_failed_overwrite_build_preserves_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    _create_exportable_clip(storage, include_background=False)
    selected = storage.selected_path("clip-1", "e1.png")
    selected.unlink()
    config.resolved_export_root.mkdir(parents=True)
    sentinel = config.resolved_export_root / "keep.txt"
    sentinel.write_text("valuable existing dataset", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="reference image is missing"):
        DatasetExporter(config, storage).export(overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "valuable existing dataset"
    assert not list(config.resolved_export_root.parent.glob(".*.tmp-*"))


def test_dataset_sample_schema_rejects_unbound_image_label() -> None:
    reference = DatasetReference(
        token="<ref_subject_1>",
        type="entity",
        entity_id="e1",
        scope="local",
        visible_region="upper_body",
        image_path="references/clip-1/subject_1.png",
        source_frame_index=2,
        synthetic=False,
    )

    with pytest.raises(ValidationError, match="image labels must match"):
        DatasetSample(
            sample_id="clip-1",
            target_video="/mnt/workspace/public/dataset/video.mp4",
            t2v_caption="A woman walks.",
            r2v_instruction="Use Image 2 to generate the shot.",
            references=[reference],
            source={"parent_video_id": "parent", "clip_suffix": "1_0"},
        )


def test_v3_entrypoint_initializes_storage_without_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 100\n",
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=(),
        git_commit="abc123",
    )

    assert result["completed_stages"] == []
    assert (config.resolved_run_root / "run.json").is_file()
    assert not (config.resolved_run_root / "clips").exists()
    with pytest.raises(
        FileNotFoundError,
        match="requires manifest stage",
    ):
        run_pipeline_v3(
            config_path=config_path,
            stages=("annotate",),
            git_commit="abc123",
        )
