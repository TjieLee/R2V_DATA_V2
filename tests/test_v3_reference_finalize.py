from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import (
    ReferenceFinalizeConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.reference_finalize import (
    finalize_references,
    normalize_entity_reference_image,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundReferenceState,
    ClipRecord,
    ClipSource,
    CoverageState,
    DatasetSample,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    ReferencesState,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage
from run_pipeline_v3 import STAGE_ORDER, run_pipeline_v3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reference_finalize: ReferenceFinalizeConfig | None = None,
) -> V3Config:
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
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "finalize-test",
        export_root=writable / "datasets" / "finalize-test",
        source=SourceConfig(limit=10),
        reference_finalize=(
            reference_finalize or ReferenceFinalizeConfig()
        ),
    )
    config.validate()
    return config


def _entity_image(
    size: tuple[int, int] = (600, 500),
    bbox: tuple[int, int, int, int] = (50, 50, 550, 450),
) -> Image.Image:
    width, height = size
    pixels = np.full((height, width, 4), 255, dtype=np.uint8)
    pixels[..., 3] = 0
    x1, y1, x2, y2 = bbox
    pixels[y1:y2, x1:x2, :3] = (30, 90, 170)
    pixels[y1:y2, x1:x2, 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _visibility(*, qualifies: bool) -> EntityVisibilitySummary:
    count = 7 if qualifies else 3
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(count)),
        visible_frame_count=count,
        coverage_ratio=count / 10,
        qualifies=qualifies,
        per_frame_area_ratio=[0.1] * count + [0.0] * (10 - count),
        per_frame_confidence=[0.9] * count + [None] * (10 - count),
    )


def _write_ready_clip(
    storage: RunStorage,
    *,
    clip_uid: str = "clip-1",
    scopes: tuple[str, ...] = ("full",),
    images: tuple[Image.Image, ...] | None = None,
    include_background: bool = False,
) -> list[Path]:
    images = images or tuple(_entity_image() for _ in scopes)
    if len(images) != len(scopes):
        raise ValueError("images and scopes must have equal lengths")
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(
                storage.config.dataset_json.parent
                / "videos"
                / f"{clip_uid}.mp4"
            ),
            parent_video_id="parent",
            clip_suffix=clip_uid,
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type="subject",
            phrase=f"entity {index}",
            grounding_prompt=f"visible entity {index}",
        )
        for index in range(1, len(scopes) + 1)
    ]
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="One or more entities move through the scene.",
            entities=entities,
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={
                entity.entity_id: _visibility(qualifies=index == 0)
                for index, entity in enumerate(entities)
            },
        ),
    )
    references: list[EntityReferenceState] = []
    source_paths: list[Path] = []
    for index, (scope, image) in enumerate(zip(scopes, images), start=1):
        entity_id = f"e{index}"
        path = storage.selected_entity_path(clip_uid, entity_id)
        image.save(path, format="PNG", optimize=False, compress_level=9)
        source_paths.append(path)
        is_full = scope == "full"
        references.append(
            EntityReferenceState(
                entity_id=entity_id,
                status="ready",
                reference_scope=scope,
                visible_region="whole" if is_full else "upper_body",
                whole_entity_recognizable=is_full,
                identity_features_visible=True,
                scope_reason="deterministic test reference",
                image_path=storage.relative_artifact_path(path),
                source_frame_index=index,
            )
        )
    background = None
    background_token = None
    if include_background:
        background_path = storage.frame_path(clip_uid, 3)
        Image.new("RGB", (37, 19), (40, 50, 60)).save(
            background_path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        background = BackgroundReferenceState(
            status="clean_raw",
            source_image_path=storage.relative_artifact_path(background_path),
            output_image_path=storage.relative_artifact_path(background_path),
            source_frame_slot=3,
            source_frame_index=30,
            source_foreground_area_pixels=0,
            source_foreground_area_ratio=0.0,
        )
        background_token = "<ref_bg_1>"
    storage.write_references(
        clip_uid,
        ReferencesState(entities=references, background=background),
    )
    tokens = {
        entity.entity_id: f"<ref_subject_{index}>"
        for index, entity in enumerate(entities, start=1)
    }
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=[entity.entity_id for entity in entities],
            tokens=tokens,
            background_token=background_token,
        ),
    )
    return source_paths


def _make_exportable(storage: RunStorage, clip_uid: str) -> None:
    clip = storage.read_clip(clip_uid)
    assert clip.pairing is not None
    binding_count = len(clip.pairing.retained_entity_ids)
    if clip.pairing.background_token is not None:
        binding_count += 1
    body = "Keep " + " and ".join(
        f"{{{{image_{index}}}}}" for index in range(1, binding_count + 1)
    ) + " consistent."
    legend = [
        InstructionLegendEntry(
            image_id=f"image_{index}",
            description=f"stable reference {index}",
        )
        for index in range(1, binding_count + 1)
    ]
    storage.write_instruction(
        clip_uid,
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=render_instruction_text(body, legend),
        ),
    )
    storage.write_export(clip_uid, ExportState(accepted=True, reason=None))


@pytest.mark.parametrize(
    "size",
    [(600, 200), (200, 600), (600, 600)],
)
def test_normalization_handles_horizontal_vertical_and_square(
    size: tuple[int, int],
) -> None:
    image = _entity_image(size=size, bbox=(0, 0, *size))

    result = normalize_entity_reference_image(
        image,
        canvas_size=1024,
        content_max_side=896,
        max_upscale=2.0,
    )

    assert result.image.size == (1024, 1024)
    assert result.image.mode == "RGBA"
    assert max(
        result.normalized_content_width,
        result.normalized_content_height,
    ) == 896
    source_ratio = size[0] / size[1]
    normalized_ratio = (
        result.normalized_content_width / result.normalized_content_height
    )
    assert normalized_ratio == pytest.approx(source_ratio, rel=0.005)
    x1, y1, x2, y2 = result.normalized_content_bbox_xyxy
    assert abs(x1 - (1024 - x2)) <= 1
    assert abs(y1 - (1024 - y2)) <= 1


def test_normalization_caps_upscale_and_marks_low_resolution() -> None:
    image = _entity_image(size=(100, 60), bbox=(0, 0, 100, 60))

    result = normalize_entity_reference_image(
        image,
        canvas_size=1024,
        content_max_side=896,
        max_upscale=2.0,
    )

    assert result.scale_factor == 2.0
    assert result.normalized_content_width == 200
    assert result.normalized_content_height == 120
    assert result.low_resolution is True


def test_normalization_outputs_binary_alpha_and_white_transparent_rgb() -> None:
    result = normalize_entity_reference_image(
        _entity_image(),
        canvas_size=1024,
        content_max_side=896,
        max_upscale=2.0,
    )
    pixels = np.asarray(result.image)
    alpha = pixels[..., 3]

    assert set(np.unique(alpha)) == {0, 255}
    assert np.all(pixels[..., :3][alpha == 0] == 255)


def test_quality_tiers_are_conservative_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage, scopes=("full", "local"))

    stats = finalize_references(config, storage)

    state = storage.read_clip("clip-1").reference_finalization
    assert state is not None and state.status == "ready"
    assert [entity.quality_tier for entity in state.entities] == ["A", "B"]
    assert state.entities[0].quality_flags == []
    assert state.entities[1].quality_flags == [
        "local_reference",
        "non_whole_visible_region",
    ]
    assert stats.entities_tier_a == 1
    assert stats.entities_tier_b == 1


def test_border_contact_forces_tier_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    image = _entity_image(size=(600, 500), bbox=(0, 50, 500, 450))
    _write_ready_clip(storage, images=(image,))

    finalize_references(config, storage)

    state = storage.read_clip("clip-1").reference_finalization
    assert state is not None
    entity = state.entities[0]
    assert entity.source_border_contact_count == 1
    assert entity.quality_tier == "B"
    assert entity.quality_flags == ["border_contact"]


def test_low_resolution_metadata_uses_capped_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    image = _entity_image(size=(120, 80), bbox=(10, 10, 110, 70))
    _write_ready_clip(storage, images=(image,))

    finalize_references(config, storage)

    state = storage.read_clip("clip-1").reference_finalization
    assert state is not None
    entity = state.entities[0]
    assert entity.scale_factor == 2.0
    assert entity.normalized_content_width == 200
    assert entity.quality_tier == "B"
    assert entity.quality_flags == ["low_resolution"]


def test_empty_alpha_fails_closed_and_later_clip_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    empty = np.full((20, 30, 4), 255, dtype=np.uint8)
    empty[..., 3] = 0
    _write_ready_clip(
        storage,
        clip_uid="bad-clip",
        images=(Image.fromarray(empty, mode="RGBA"),),
    )
    _write_ready_clip(storage, clip_uid="good-clip")

    stats = finalize_references(config, storage)

    bad = storage.read_clip("bad-clip").reference_finalization
    good = storage.read_clip("good-clip").reference_finalization
    assert stats.processed == 2
    assert stats.failed == 1
    assert stats.entities_rejected == 1
    assert stats.entities_finalized == 1
    assert bad is not None and bad.status == "failed"
    assert bad.entities[0].quality_tier == "reject"
    assert good is not None and good.status == "ready"
    failures = [
        json.loads(line)
        for line in storage.failures_path.read_text(encoding="utf-8").splitlines()
    ]
    assert failures[-1]["stage"] == "reference_finalize"
    assert failures[-1]["clip_uid"] == "bad-clip"


def test_source_is_unchanged_and_rebuild_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    source = _write_ready_clip(storage)[0]
    source_bytes = source.read_bytes()

    first = finalize_references(config, storage)
    output = storage.finalized_entity_path("clip-1", "e1")
    first_hash = _sha256(output)
    skipped = finalize_references(config, storage)
    rebuilt = finalize_references(config, storage, overwrite=True)

    assert first.entities_finalized == 1
    assert skipped.skipped_existing == 1
    assert rebuilt.processed == 1
    assert source.read_bytes() == source_bytes
    assert _sha256(output) == first_hash


def test_existing_finalized_artifact_is_validated_before_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage)
    finalize_references(config, storage)
    output = storage.finalized_entity_path("clip-1", "e1")
    output.write_bytes(b"not a png")

    stats = finalize_references(config, storage)

    assert stats.skipped_existing == 0
    assert stats.failed == 1


def test_background_is_only_validated_and_keeps_original_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage, include_background=True)
    clip = storage.read_clip("clip-1")
    assert clip.references.background is not None
    background_path = storage.root / clip.references.background.output_image_path
    before = background_path.read_bytes()

    stats = finalize_references(config, storage)

    state = storage.read_clip("clip-1").reference_finalization
    assert state is not None and state.background is not None
    assert (state.background.width, state.background.height) == (37, 19)
    assert state.background.mode == "RGB"
    assert state.background.sha256 == hashlib.sha256(before).hexdigest()
    assert background_path.read_bytes() == before
    assert stats.backgrounds_validated == 1
    assert not (storage.finalized_dir("clip-1") / "background.png").exists()


def test_pairing_change_invalidates_finalization_and_export_only_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage)
    finalize_references(config, storage)
    _make_exportable(storage, "clip-1")
    before = storage.read_clip("clip-1")
    assert before.instruction is not None
    assert before.reference_finalization is not None

    storage.write_pairing(
        "clip-1",
        PairingState(status="rejected", reason="changed pairing"),
    )

    after = storage.read_clip("clip-1")
    assert after.reference_finalization is None
    assert after.instruction is None
    assert after.export == ExportState()
    assert not storage.finalized_dir("clip-1").exists()


def test_finalization_change_invalidates_export_but_not_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage)
    finalize_references(config, storage)
    _make_exportable(storage, "clip-1")
    before = storage.read_clip("clip-1")
    assert before.reference_finalization is not None
    assert before.instruction is not None
    entity = before.reference_finalization.entities[0].model_copy(
        update={"normalized_sha256": "0" * 64}
    )
    changed = before.reference_finalization.model_copy(
        update={"entities": [entity]}
    )

    storage.write_reference_finalization("clip-1", changed)

    after = storage.read_clip("clip-1")
    assert after.instruction == before.instruction
    assert after.export == ExportState()


def test_exporter_writes_source_and_normalized_entity_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    source = _write_ready_clip(storage, include_background=True)[0]
    finalize_references(config, storage)
    _make_exportable(storage, "clip-1")

    DatasetExporter(config, storage).export()

    sample = DatasetSample.model_validate_json(
        (config.resolved_export_root / "samples.jsonl").read_text(
            encoding="utf-8"
        )
    )
    entity = next(reference for reference in sample.references if reference.type == "entity")
    background = next(
        reference for reference in sample.references if reference.type == "background"
    )
    normalized = config.resolved_export_root / entity.image_path
    assert entity.source_image_path is not None
    exported_source = config.resolved_export_root / entity.source_image_path
    assert normalized.name == "subject_1.png"
    assert exported_source.name == "subject_1.source.png"
    assert exported_source.read_bytes() == source.read_bytes()
    with Image.open(normalized) as image:
        assert image.mode == "RGBA"
        assert image.size == (1024, 1024)
    assert entity.width == 1024
    assert entity.height == 1024
    assert entity.normalization_profile == "entity_1024_v1"
    assert entity.sha256 == _sha256(normalized)
    assert entity.source_sha256 == _sha256(exported_source)
    assert background.width == 37
    assert background.height == 19
    assert background.sha256 == _sha256(
        config.resolved_export_root / background.image_path
    )
    assert not (
        config.resolved_export_root
        / "references"
        / "clip-1"
        / "background_1.source.png"
    ).exists()


def test_legacy_clip_and_dataset_sample_still_load_without_new_fields() -> None:
    clip = ClipRecord.model_validate(
        {
            "clip_uid": "legacy",
            "source": {
                "video_path": "/mnt/workspace/public/dataset/legacy.mp4",
                "parent_video_id": "parent",
                "clip_suffix": "1_0",
                "source_index": 0,
                "caption_raw": "",
                "metadata": {},
            },
        }
    )
    sample = DatasetSample.model_validate(
        {
            "sample_id": "legacy",
            "target_video": "/mnt/workspace/public/dataset/legacy.mp4",
            "t2v_caption": "A person walks.",
            "r2v_instruction": (
                "Keep <Image 1> consistent.\n\n<Image 1>: a person"
            ),
            "references": [
                {
                    "token": "<ref_subject_1>",
                    "type": "entity",
                    "entity_id": "e1",
                    "scope": "full",
                    "visible_region": "whole",
                    "image_path": "references/legacy/subject_1.png",
                    "source_frame_index": 1,
                    "synthetic": False,
                }
            ],
            "source": {"parent_video_id": "parent", "clip_suffix": "1_0"},
        }
    )

    assert clip.reference_finalization is None
    assert sample.references[0].source_image_path is None
    assert sample.references[0].quality_flags == []


def test_non_rgba_source_is_rejected_without_publishing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(
        storage,
        images=(Image.new("RGB", (40, 30), (10, 20, 30)),),
    )

    stats = finalize_references(config, storage)

    state = storage.read_clip("clip-1").reference_finalization
    assert stats.failed == 1
    assert stats.entities_rejected == 1
    assert state is not None and state.status == "failed"
    assert state.entities[0].quality_tier == "reject"
    assert "must be RGBA" in (state.entities[0].reason or "")
    assert not storage.finalized_entity_path("clip-1", "e1").exists()


def test_finalized_entity_schema_checks_metrics_and_flag_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage)
    finalize_references(config, storage)
    state = storage.read_clip("clip-1").reference_finalization
    assert state is not None
    entity = state.entities[0]

    inconsistent = entity.model_dump(mode="json")
    inconsistent["normalized_content_width"] += 1
    with pytest.raises(ValueError, match="content dimensions"):
        type(entity).model_validate(inconsistent)

    unstable_flags = entity.model_dump(mode="json")
    unstable_flags["quality_flags"] = ["border_contact", "low_resolution"]
    with pytest.raises(ValueError, match="stable canonical order"):
        type(entity).model_validate(unstable_flags)

    wrong_profile_size = entity.model_dump(mode="json")
    wrong_profile_size["normalized_width"] = 512
    wrong_profile_size["normalized_height"] = 512
    with pytest.raises(ValueError, match="1024x1024"):
        type(entity).model_validate(wrong_profile_size)


@pytest.mark.parametrize(
    "reference_finalize",
    [
        ReferenceFinalizeConfig(enabled=1),
        ReferenceFinalizeConfig(entity_canvas_size=True),
        ReferenceFinalizeConfig(entity_canvas_size=0),
        ReferenceFinalizeConfig(entity_content_max_side=0),
        ReferenceFinalizeConfig(entity_content_max_side=1025),
        ReferenceFinalizeConfig(entity_max_upscale=True),
        ReferenceFinalizeConfig(entity_max_upscale=float("inf")),
        ReferenceFinalizeConfig(entity_max_upscale=0),
        ReferenceFinalizeConfig(normalization_profile="../unsafe"),
    ],
)
def test_reference_finalize_config_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_finalize: ReferenceFinalizeConfig,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        replace(config, reference_finalize=reference_finalize).validate()


def test_reference_finalize_config_loads_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config_path = tmp_path / "reference-finalize.yaml"
    config_path.write_text(
        f"dataset_json: {json.dumps(str(config.dataset_json))}\n"
        f"run_root: {json.dumps(str(config.run_root))}\n"
        f"export_root: {json.dumps(str(config.export_root))}\n"
        "source:\n"
        "  limit: 10\n"
        "reference_finalize:\n"
        "  enabled: true\n"
        "  entity_canvas_size: 768\n"
        "  entity_content_max_side: 700\n"
        "  entity_max_upscale: 1.5\n"
        "  normalization_profile: entity_768_v1\n",
        encoding="utf-8",
    )

    loaded = config_module.load_config(config_path)

    assert loaded.reference_finalize == ReferenceFinalizeConfig(
        enabled=True,
        entity_canvas_size=768,
        entity_content_max_side=700,
        entity_max_upscale=1.5,
        normalization_profile="entity_768_v1",
    )


def test_reference_finalize_config_changes_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    changed = replace(
        config,
        reference_finalize=replace(
            config.reference_finalize,
            entity_content_max_side=800,
        ),
    )

    assert changed.fingerprint() != config.fingerprint()


def test_cli_stage_order_and_dispatch_include_reference_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="test")
    _write_ready_clip(storage)
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 10\n",
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("reference_finalize",),
        git_commit="test",
    )

    assert STAGE_ORDER.index("pair") < STAGE_ORDER.index("reference_finalize")
    assert STAGE_ORDER.index("reference_finalize") < STAGE_ORDER.index("instruct")
    assert result["completed_stages"] == ["reference_finalize"]
    assert result["reference_finalize"]["entities_finalized"] == 1


def test_disabled_reference_finalize_stage_fails_before_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        reference_finalize=ReferenceFinalizeConfig(enabled=False),
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="test")

    with pytest.raises(ValueError, match="reference_finalize stage is disabled"):
        finalize_references(config, storage)