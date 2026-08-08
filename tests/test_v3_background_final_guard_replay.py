from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background_final_guard import (
    FinalBackgroundJudgeFailure,
    FinalBackgroundReviewAttempt,
)
from r2v_data_v2.v3.background_final_guard_replay import (
    run_background_final_guard_replay,
)
from r2v_data_v2.v3.candidate_judge_replay import snapshot_run_files
from r2v_data_v2.v3.config import (
    PairConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    ClipRecord,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    FinalBackgroundReview,
    PairingState,
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
            background_final_judge=service,
        ),
        pair=PairConfig(background_final_guard_mode="qwen_v1"),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "remover",
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


def _add_bound_clip(
    storage: RunStorage,
    config: V3Config,
    clip_uid: str,
    *,
    bound: bool = True,
) -> None:
    source = ClipSource(
        video_path=str(config.dataset_json.parent / f"{clip_uid}.mp4"),
        parent_video_id="parent",
        clip_suffix=clip_uid,
        source_index=0,
        caption_raw="",
        metadata={},
    )
    storage.create_clip(clip_uid=clip_uid, source=source)
    frame_path = storage.clip_dir(clip_uid) / "frames" / "00.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (30, 60, 90)).save(frame_path, format="JPEG")
    entity_path = storage.clip_dir(clip_uid) / "selected" / "e1.png"
    entity_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (6, 6), (10, 20, 30, 255)).save(
        entity_path,
        format="PNG",
    )
    annotation = AnnotationState(
        status="ready",
        t2v_caption="A person walks through a stone courtyard.",
        entities=[
            AnnotationEntity(
                entity_id="e1",
                reference_type="subject",
                phrase="a person",
                grounding_prompt="the visible person",
            )
        ],
        background=BackgroundAnnotation(
            phrase=f"stone courtyard {clip_uid}",
            grounding_prompt="the stone courtyard and surrounding walls",
        ),
    )
    references = ReferencesState(
        entities=[
            EntityReferenceState(
                entity_id="e1",
                status="ready",
                reference_scope="local",
                visible_region="upper_body",
                whole_entity_recognizable=False,
                identity_features_visible=True,
                scope_reason="clear upper body",
                image_path=storage.relative_artifact_path(entity_path),
                source_frame_index=0,
            )
        ],
        background=BackgroundReferenceState(
            status="clean_raw",
            source_image_path="frames/00.jpg",
            output_image_path="frames/00.jpg",
            source_frame_slot=0,
            source_frame_index=0,
            source_foreground_area_pixels=0,
            source_foreground_area_ratio=0.0,
        ),
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
        references=references,
        pairing=PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
            background_token="<ref_bg_1>" if bound else None,
        ),
    )
    write_json_atomic(storage.clip_path(clip_uid), record.model_dump(mode="json"))


def _review(verdict: str) -> FinalBackgroundReview:
    accepted = verdict == "accept"
    return FinalBackgroundReview(
        verdict=verdict,
        background_matches_description=True,
        no_unexpected_foreground_subject=accepted,
        usable_background_information=True,
        no_obvious_artifacts=True,
        reason="usable" if accepted else "foreground remains",
    )


class _ReplayJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def review(self, **kwargs: object) -> FinalBackgroundReviewAttempt:
        phrase = str(kwargs["background_phrase"])
        self.calls.append(phrase)
        if phrase.endswith("clip-b"):
            raise FinalBackgroundJudgeFailure("structured output failed")
        review = _review("reject" if phrase.endswith("clip-c") else "accept")
        return FinalBackgroundReviewAttempt(
            review=review,
            raw_response=review.model_dump_json(),
        )


def test_background_final_guard_replay_is_read_only_and_summarizes_all_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    for clip_uid in ("clip-c", "clip-a", "clip-b"):
        _add_bound_clip(storage, config, clip_uid)
    _add_bound_clip(storage, config, "clip-unbound", bound=False)
    before = snapshot_run_files(storage.root)
    output = config.run_root.parent / "replays" / "background-final.jsonl"
    judge = _ReplayJudge()

    summary = run_background_final_guard_replay(
        config,
        run_root=storage.root,
        output_path=output,
        judge=judge,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["clip_uid"] for record in records] == [
        "clip-a",
        "clip-b",
        "clip-c",
    ]
    assert [record["status"] for record in records] == [
        "succeeded",
        "failed",
        "succeeded",
    ]
    assert [record["verdict"] for record in records] == [
        "accept",
        None,
        "reject",
    ]
    assert len(judge.calls) == 3
    assert all("clip-unbound" not in phrase for phrase in judge.calls)
    assert summary["candidate_background_count"] == 3
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["failed"] == 1
    assert summary["clean_raw_count"] == 3
    assert summary["ready_removed_count"] == 0
    assert json.loads(
        Path(f"{output}.summary.json").read_text(encoding="utf-8")
    ) == summary
    assert snapshot_run_files(storage.root) == before
