from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.candidate_judge_replay import snapshot_run_files
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceEditConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.scale_collapse_fallback_guard import (
    ScaleCollapseFallbackReview,
    ScaleCollapseFallbackReviewAttempt,
)
from r2v_data_v2.v3.scale_collapse_fallback_guard_replay import (
    run_scale_collapse_fallback_guard_replay,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipRecord,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    PairingState,
    ReferenceEditEntityState,
    ReferenceEditState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage


def _config(tmp_path: Path, monkeypatch) -> V3Config:
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
    service = QwenServiceConfig(model=str(pretrained / "Qwen" / "judge"))
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "source-run",
        export_root=writable / "exports" / "source-run",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=service.model),
            instruction_writer=service,
            candidate_judge=service,
            background_remove_judge=service,
            reference_edit_judge=service,
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "remover",
        ),
        reference_edit=ReferenceEditConfig(
            enabled=True,
            python_executable=writable / "venvs" / "boogu" / "python",
            code_root=writable / "vendor" / "Boogu-Image",
            model_path=writable / "models" / "Boogu-Image",
        ),
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
        per_frame_area_ratio=[0.1] * 7 + [0.0] * 3,
        per_frame_confidence=[0.9] * 7 + [None] * 3,
    )


def _add_fallback_clip(storage: RunStorage, config: V3Config) -> Path:
    clip_uid = "clip-1"
    source = ClipSource(
        video_path=str(config.dataset_json.parent / "clip-1.mp4"),
        parent_video_id="parent",
        clip_suffix="1",
        source_index=0,
        caption_raw="",
        metadata={},
    )
    storage.create_clip(clip_uid=clip_uid, source=source)
    image_path = storage.clip_dir(clip_uid) / "selected" / "e1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 6), (20, 40, 60, 170)).save(
        image_path,
        format="PNG",
    )
    relative_image = storage.relative_artifact_path(image_path)
    reference = EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="local",
        visible_region="upper_body",
        whole_entity_recognizable=False,
        identity_features_visible=True,
        scope_reason="coherent local view",
        image_path=relative_image,
        source_frame_index=0,
        image_quality="acceptable",
        completeness="local_usable",
        synthetic=False,
    )
    annotation = AnnotationState(
        status="ready",
        t2v_caption="A seated person faces the camera.",
        entities=[
            AnnotationEntity(
                entity_id="e1",
                reference_type="subject",
                phrase="a seated person",
                grounding_prompt="the seated person",
            )
        ],
    )
    record = ClipRecord(
        clip_uid=clip_uid,
        source=source,
        annotation=annotation,
        coverage=CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={"e1": _visibility()},
        ),
        references=ReferencesState(entities=[reference]),
        pairing=PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
        ),
        reference_edit=ReferenceEditState(
            status="ready",
            entities=[
                ReferenceEditEntityState(
                    entity_id="e1",
                    route="local_usable",
                    status="fallback",
                    source_reference=reference,
                    source_image_path=relative_image,
                    output_image_path=relative_image,
                    operation="add_entity_background",
                    metadata_path=(
                        "clips/clip-1/reference_edit/e1/final_metadata.json"
                    ),
                    operations=["add_entity_background"],
                    background_metadata_path=(
                        "clips/clip-1/reference_edit/e1/background_metadata.json"
                    ),
                    fallback_policy="keep_source",
                    reason="entity_scale_collapsed",
                )
            ],
        ),
    )
    write_json_atomic(storage.clip_path(clip_uid), record.model_dump(mode="json"))
    return image_path


class _ReplayJudge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> ScaleCollapseFallbackReviewAttempt:
        self.calls.append(kwargs)
        review = ScaleCollapseFallbackReview(
            verdict="reject",
            identity_or_primary_region_visible=False,
            coherent_structure=False,
            independent_reference_value=False,
            not_severely_fragmented=False,
            reason="headless fragmented torso",
        )
        return ScaleCollapseFallbackReviewAttempt(
            review=review,
            raw_responses=(review.model_dump_json(),),
        )


def test_scale_collapse_fallback_replay_is_read_only(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    image_path = _add_fallback_clip(storage, config)
    before = snapshot_run_files(storage.root)
    output = config.run_root.parent / "replays" / "scale-collapse.jsonl"
    judge = _ReplayJudge()

    summary = run_scale_collapse_fallback_guard_replay(
        config,
        run_root=storage.root,
        output_path=output,
        judge=judge,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0] | {
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "reference_type": "subject",
        "status": "succeeded",
        "verdict": "reject",
    } == records[0]
    assert summary | {
        "candidate_count": 1,
        "accepted": 0,
        "rejected": 1,
        "failed": 0,
        "subject_count": 1,
        "object_count": 0,
        "group_count": 0,
    } == summary
    assert len(judge.calls) == 1
    reviewed = judge.calls[0]["image"]
    with Image.open(image_path) as source:
        assert reviewed.mode == source.mode
        assert reviewed.size == source.size
        assert reviewed.tobytes() == source.tobytes()
    assert snapshot_run_files(storage.root) == before
    assert json.loads(
        Path(f"{output}.summary.json").read_text(encoding="utf-8")
    ) == summary
