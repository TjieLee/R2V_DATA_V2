from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

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
    V3Config,
    load_config,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundReferenceState,
    ClipSource,
    CoverageState,
    DatasetReference,
    DatasetSample,
    EntityReferenceState,
    ExportState,
    InstructionState,
    PairingState,
    ReferencesState,
    TrackedMasksArtifact,
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
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(annotation_model)),
            instruction_writer=QwenServiceConfig(model=str(annotation_model)),
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
        phrase=phrase,
        grounding_prompt=phrase.lower(),
        canonical_label=phrase.lower(),
        category="person" if entity_id == "e1" else "object",
        reference_worthy=True,
        salience="primary" if entity_id == "e1" else "secondary",
        genericity="generic",
        name_evidence="none",
        separability="independent",
        selection_reason="visible entity",
    )


def _create_exportable_clip(
    storage: RunStorage,
    *,
    clip_uid: str = "clip-1",
    include_background: bool = True,
) -> None:
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path="/mnt/workspace/public/dataset/video.mp4",
            parent_video_id="parent",
            clip_suffix="1_0",
        ),
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
            required_visible_frames=8,
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
            image_path=storage.relative_artifact_path(frame_path),
            source_frame_index=3,
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
    instruction = (
        "Generate a continuous shot using the visible appearance from "
        "<ref_subject_1> and the clean setting from <ref_bg_1>."
        if include_background
        else (
            "Generate a continuous shot using the visible appearance from "
            "<ref_subject_1>."
        )
    )
    storage.write_instruction(
        clip_uid,
        InstructionState(status="ready", r2v_instruction=instruction),
    )
    storage.write_export(
        clip_uid,
        ExportState(accepted=True, reason=None),
    )


def test_v3_config_loads_32b_defaults_without_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.qwen.annotation.model.endswith("Qwen3-VL-32B-Instruct")
    assert loaded.frames.count == 10
    assert loaded.background.raw_foreground_area_ratio == 0.0
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
    source = ClipSource(
        video_path="/mnt/workspace/public/dataset/video.mp4",
        parent_video_id="parent",
        clip_suffix="1_0",
    )

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
        CoverageState(passed=True, qualifying_entity_ids=["e1"]),
    )
    masks_path = storage.write_masks(
        "clip-1",
        TrackedMasksArtifact(
            clip_uid="clip-1",
            entities={
                "e1": {
                    "slots": {
                        "0": {"mask_available": True, "counts": "encoded"}
                    }
                }
            },
        ),
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
        source=ClipSource(
            video_path="/mnt/workspace/public/dataset/rejected.mp4",
            parent_video_id="parent",
            clip_suffix="2_0",
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


def test_dataset_sample_schema_rejects_unbound_instruction_token() -> None:
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

    with pytest.raises(ValidationError, match="tokens must exactly match"):
        DatasetSample(
            sample_id="clip-1",
            target_video="/mnt/workspace/public/dataset/video.mp4",
            t2v_caption="A woman walks.",
            r2v_instruction="Generate a shot with <ref_subject_2>.",
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
        f"export_root: {config.export_root}\n",
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
    with pytest.raises(NotImplementedError, match="Commit 1"):
        run_pipeline_v3(
            config_path=config_path,
            stages=("annotate",),
            git_commit="abc123",
        )
