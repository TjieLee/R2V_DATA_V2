from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.candidate_judge_replay as replay_module
import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import ValidationIssue
from r2v_data_v2.v3.candidate_judge_replay import (
    load_baseline_decision,
    run_candidate_judge_replay,
    snapshot_run_files,
)
from r2v_data_v2.v3.config import (
    DebugConfig,
    PairConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceEditConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
    EntityReferenceJudgeFailure,
    QwenEntityReferenceJudge,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    CoverageState,
    EntityVisibilitySummary,
    RawEntityReferenceDecision,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage
from tools.replay_v3_candidate_judge import _parser

WIDTH = 64
HEIGHT = 48


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    model = str(pretrained / "Qwen" / "baseline")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "source-run",
        export_root=writable / "datasets" / "source-export",
        source=SourceConfig(limit=2),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=model),
            instruction_writer=QwenServiceConfig(model=model),
            candidate_judge=QwenServiceConfig(model=model),
            background_remove_judge=QwenServiceConfig(model=model),
        ),
        pair=PairConfig(max_candidates_per_entity=3, repair_retries=1),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "edit",
            adapter_path=user_models / "remover",
        ),
        reference_edit=ReferenceEditConfig(
            min_source_content_area_pixels=1,
            min_source_content_long_side_pixels=1,
        ),
        debug=DebugConfig(save_diagnostics=True),
    )
    config.validate()
    return config


def _decision(
    *,
    selected_candidate_id: str | None = "candidate_1",
    route: str = "complete",
) -> RawEntityReferenceDecision:
    rejected = route in {"fragmented", "severely_incomplete"}
    return RawEntityReferenceDecision(
        selected_candidate_id=None if rejected else selected_candidate_id,
        image_quality="acceptable" if rejected else "high",
        completeness=route,
        reference_scope="reject" if rejected else "full",
        visible_region="custom" if rejected else "whole",
        whole_entity_recognizable=not rejected,
        identity_features_visible=not rejected,
        viewpoint="not_applicable",
        independent_reference_value=True,
        requires_substantial_invention=False,
        primary_identity_region_visible=not rejected,
        major_structure_visible=not rejected,
        truncation_severity="major" if rejected else "none",
        discrete_foreground_instance=not rejected,
        mask_matches_target=not rejected,
        completion_needed_for_reference_use=False,
        detached_target_fragments_present=False,
        scope_reason="usable object" if not rejected else "fragmented object",
    )


def _tracked_frame(slot: int, mask: np.ndarray) -> TrackedMaskFrame:
    rows, columns = np.nonzero(mask)
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        track_valid=True,
        confidence=0.9,
        backend_confidences=[0.9],
        backend_object_ids=["object-1"],
        area_pixels=int(mask.sum()),
        area_ratio=float(mask.mean()),
        bbox_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        rle=encode_binary_mask(mask),
    )


def _visibility() -> EntityVisibilitySummary:
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(10)),
        visible_frame_count=10,
        coverage_ratio=1.0,
        qualifies=True,
        per_frame_area_ratio=[0.25] * 10,
        per_frame_confidence=[0.9] * 10,
    )


def _add_clip(storage: RunStorage, clip_uid: str) -> None:
    config = storage.config
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"),
            parent_video_id="parent",
            clip_suffix=clip_uid,
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entity = AnnotationEntity(
        entity_id="e1",
        reference_type="object",
        phrase=f"ornate panel in {clip_uid}",
        grounding_prompt=f"the ornate panel in {clip_uid}",
    )
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption=f"An ornate panel in {clip_uid} remains visible.",
            entities=[entity],
        ),
    )
    frames: list[SampledFrame] = []
    for slot in range(10):
        yy, xx = np.indices((HEIGHT, WIDTH))
        pixels = np.stack(
            (
                (xx * (slot + 1) + 17) % 256,
                (yy * 7 + slot * 13) % 256,
                ((xx + yy) * 5 + slot) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        path = storage.frame_path(clip_uid, slot)
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot + 1),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=_sha256(path),
            )
        )
    write_json_atomic(
        storage.frames_manifest_path(clip_uid),
        SampledFramesArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            frames=frames,
        ).model_dump(mode="json"),
    )
    tracked_frames: list[TrackedMaskFrame] = []
    for slot in range(10):
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[10:38, 16 + slot % 2 : 48 + slot % 2] = True
        tracked_frames.append(_tracked_frame(slot, mask))
    storage.write_masks(
        clip_uid,
        TrackedMasksArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            entities={
                "e1": TrackedEntityMasks(
                    status="ready",
                    reference_type="object",
                    grounding_prompt=entity.grounding_prompt,
                    backend_object_ids=["object-1"],
                    frames=tracked_frames,
                )
            },
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={"e1": _visibility()},
        ),
    )
    baseline_dir = storage.clip_dir(clip_uid) / "debug" / "pair" / "e1"
    baseline_dir.mkdir(parents=True)
    write_json_atomic(
        baseline_dir / "raw_responses.json",
        {
            "responses": [
                json.dumps(_decision(selected_candidate_id="candidate_3").model_dump()),
                json.dumps(_decision().model_dump()),
            ]
        },
    )


class _Completions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(self.responses))
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


def test_candidate_judge_replay_is_read_only_and_summarizes_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-b")
    _add_clip(storage, "clip-a")
    before = snapshot_run_files(storage.root)
    completions = _Completions(
        [
            "{}",
            json.dumps(
                _decision(selected_candidate_id="candidate_2").model_dump()
            ),
            json.dumps(_decision(route="fragmented").model_dump()),
        ]
    )
    assert config.qwen.candidate_judge is not None
    judge = QwenEntityReferenceJudge(
        config.qwen.candidate_judge,
        repair_retries=config.pair.repair_retries,
        crop_padding_ratio=config.pair.crop_padding_ratio,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    builder_calls: list[tuple[str, str]] = []
    production_builder = replay_module.build_entity_reference_candidates

    def tracked_builder(*args: object, **kwargs: object):
        builder_calls.append((str(kwargs["clip_uid"]), kwargs["entity"].entity_id))
        return production_builder(*args, **kwargs)

    monkeypatch.setattr(
        replay_module,
        "build_entity_reference_candidates",
        tracked_builder,
    )
    output = config.run_root.parent / "benchmarks" / "replay.jsonl"
    model = str(config_module.ALLOWED_PRETRAINED_ROOT / "Qwen" / "new-model")

    summary = run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model=model,
        output_path=output,
        judge=judge,
    )

    after = snapshot_run_files(storage.root)
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert before == after
    assert builder_calls == [("clip-a", "e1"), ("clip-b", "e1")]
    assert [(item["clip_uid"], item["entity_id"]) for item in records] == [
        ("clip-a", "e1"),
        ("clip-b", "e1"),
    ]
    assert [item["candidate_count"] for item in records] == [3, 3]
    assert [item["status"] for item in records] == ["succeeded", "succeeded"]
    assert records[0]["baseline"] == {
        "selected_candidate_id": "candidate_1",
        "completeness": "complete",
        "reference_scope": "full",
        "viewpoint": "not_applicable",
        "identity_features_visible": True,
        "primary_identity_region_visible": True,
        "truncation_severity": "none",
        "completion_needed_for_reference_use": False,
        "detached_target_fragments_present": False,
        "valid": True,
    }
    assert [item["repair_attempts"] for item in records] == [1, 0]
    assert [item["raw_response_count"] for item in records] == [2, 1]
    assert all("raw_responses" not in item for item in records)
    assert [item["input_image_count"] for item in records] == [6, 6]
    assert [item["prompt_tokens"] for item in records] == [200, 100]
    assert all(item["variant"]["valid"] is True for item in records)
    assert summary["evidence_mode"] == "baseline"
    assert summary["card_panel_max_side"] is None
    assert summary["attempt_count"] == 2
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 0
    assert summary["entity_count"] == 2
    assert summary["succeeded_entity_count"] == 2
    assert summary["failed_entity_count"] == 0
    assert summary["structured_failure_count"] == 0
    assert summary["structured_failure_rate"] == 0.0
    assert summary["agreement_denominator"] == 2
    assert summary["initial_calls"] == 2
    assert summary["repair_calls"] == 1
    assert summary["repair_rate"] == pytest.approx(0.5)
    assert summary["avg_input_image_count"] == 6.0
    assert summary["avg_prompt_tokens"] == 100.0
    assert summary["total_prompt_tokens"] == 300
    assert summary["avg_completion_tokens"] == 20.0
    assert summary["candidate_selection_agreement_with_baseline"] == 0.0
    assert summary["selected_candidate_agreement"] == 0.0
    assert summary["reference_scope_agreement"] == pytest.approx(0.5)
    assert summary["completeness_agreement"] == pytest.approx(0.5)
    assert summary["full_decision_exact_agreement"] == 0.0
    assert summary["route_agreement_with_baseline"] == pytest.approx(0.5)
    assert summary["reject_count"] == 1
    assert summary["complete_count"] == 1
    assert summary["fragmented_count"] == 1
    assert [item["clip_uid"] for item in summary["changed_candidate_cases"]] == [
        "clip-a",
        "clip-b",
    ]
    assert [item["clip_uid"] for item in summary["changed_route_cases"]] == [
        "clip-b"
    ]
    assert [
        item["clip_uid"]
        for item in summary["baseline_accept_variant_reject"]
    ] == ["clip-b"]
    assert summary["baseline_reject_variant_accept"] == []
    assert summary["repair_cases"] == [
        {"clip_uid": "clip-a", "entity_id": "e1", "repair_attempts": 1}
    ]
    profile = summary["profiling"]["qwen_candidate_judge"]
    assert profile["calls"] == 3
    assert profile["input_images_total"] == 18
    assert profile["prompt_tokens_total"] == 300
    assert Path(f"{output}.summary.json").is_file()
    assert not Path(f"{output}.raw.jsonl").exists()


class _CaseJudge:
    def __init__(
        self,
        outcomes: dict[str, EntityReferenceDecisionAttempt | BaseException],
    ) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def decide(self, *, entity, candidates, source_images):
        del candidates, source_images
        clip_uid = entity.phrase.rsplit(" ", 1)[-1]
        self.calls.append(clip_uid)
        outcome = self.outcomes[clip_uid]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _attempt(
    *,
    selected_candidate_id: str = "candidate_1",
) -> EntityReferenceDecisionAttempt:
    return EntityReferenceDecisionAttempt(
        decision=_decision(selected_candidate_id=selected_candidate_id),
        raw_responses=("successful raw response",),
        repair_attempts=0,
    )


def _structured_failure() -> EntityReferenceJudgeFailure:
    return EntityReferenceJudgeFailure(
        raw_responses=["invalid raw one", "invalid raw two"],
        issues=[
            ValidationIssue(
                code="primary_identity_region_not_visible",
                field="primary_identity_region_visible",
                message="primary identity region is not visible",
            ),
            ValidationIssue(
                code="primary_identity_region_not_visible",
                field="reference_scope",
                message="full scope requires the primary identity region",
            ),
            ValidationIssue(
                code="local_requires_identity",
                field="identity_features_visible",
                message="local reference requires visible identity features",
            ),
        ],
    )


def test_structured_failure_is_recorded_and_later_cases_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    for clip_uid in ("clip-c", "clip-a", "clip-b"):
        _add_clip(storage, clip_uid)
    before = snapshot_run_files(storage.root)
    judge = _CaseJudge(
        {
            "clip-a": _attempt(),
            "clip-b": _structured_failure(),
            "clip-c": _attempt(selected_candidate_id="candidate_2"),
        }
    )
    output = config.run_root.parent / "benchmarks" / "failure-replay.jsonl"

    summary = run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model="new-model",
        output_path=output,
        judge=judge,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert judge.calls == ["clip-a", "clip-b", "clip-c"]
    assert [record["status"] for record in records] == [
        "succeeded",
        "failed",
        "succeeded",
    ]
    failed = records[1]
    assert failed["clip_uid"] == "clip-b"
    assert failed["failure"]["type"] == "structured_output_failure"
    assert failed["failure"]["attempt_count"] == 2
    assert [issue["code"] for issue in failed["failure"]["issues"]] == [
        "primary_identity_region_not_visible",
        "primary_identity_region_not_visible",
        "local_requires_identity",
    ]
    assert all("raw_responses" not in record for record in records)
    assert "invalid raw one" not in output.read_text(encoding="utf-8")
    assert summary["entity_count"] == 3
    assert summary["succeeded_entity_count"] == 2
    assert summary["failed_entity_count"] == 1
    assert summary["structured_failure_count"] == 1
    assert summary["structured_failure_rate"] == pytest.approx(1 / 3)
    assert summary["failure_issue_histogram"] == {
        "local_requires_identity": 1,
        "primary_identity_region_not_visible": 2,
    }
    assert summary["failed_cases"] == [
        {
            "clip_uid": "clip-b",
            "entity_id": "e1",
            "phrase": "ornate panel in clip-b",
            "attempt_count": 2,
            "issue_codes": [
                "primary_identity_region_not_visible",
                "primary_identity_region_not_visible",
                "local_requires_identity",
            ],
        }
    ]
    assert summary["agreement_denominator"] == 2
    assert summary["candidate_selection_agreement_with_baseline"] == 0.5
    assert summary["route_agreement_with_baseline"] == 1.0
    assert snapshot_run_files(storage.root) == before
    assert not Path(f"{output}.raw.jsonl").exists()

    raw_output = config.run_root.parent / "benchmarks" / "failure-raw.jsonl"
    raw_judge = _CaseJudge(
        {
            "clip-a": _attempt(),
            "clip-b": _structured_failure(),
            "clip-c": _attempt(selected_candidate_id="candidate_2"),
        }
    )
    run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model="new-model",
        output_path=raw_output,
        save_raw=True,
        judge=raw_judge,
    )
    raw_records = [
        json.loads(line)
        for line in Path(f"{raw_output}.raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert raw_records[1] == {
        "clip_uid": "clip-b",
        "entity_id": "e1",
        "status": "failed",
        "raw_responses": ["invalid raw one", "invalid raw two"],
    }
    assert snapshot_run_files(storage.root) == before


def test_real_judge_failure_keeps_http_profiling_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-a")
    _add_clip(storage, "clip-b")
    completions = _Completions(
        [
            "{}",
            "{}",
            json.dumps(_decision().model_dump()),
        ]
    )
    assert config.qwen.candidate_judge is not None
    judge = QwenEntityReferenceJudge(
        config.qwen.candidate_judge,
        repair_retries=config.pair.repair_retries,
        crop_padding_ratio=config.pair.crop_padding_ratio,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    output = config.run_root.parent / "benchmarks" / "profiled-failure.jsonl"

    summary = run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model="new-model",
        output_path=output,
        judge=judge,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == ["failed", "succeeded"]
    assert summary["initial_calls"] == 2
    assert summary["repair_calls"] == 1
    assert summary["structured_failure_count"] == 1
    profile = summary["profiling"]["qwen_candidate_judge"]
    assert profile["calls"] == 3
    assert profile["successful_calls"] == 3
    assert profile["input_images_total"] == 18


@pytest.mark.parametrize(
    ("failure", "fail_fast"),
    [
        (_structured_failure(), True),
        (RuntimeError("programming failure"), False),
    ],
)
def test_fail_fast_and_non_structured_errors_still_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    fail_fast: bool,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-a")
    output = config.run_root.parent / "benchmarks" / "abort.jsonl"
    judge = _CaseJudge({"clip-a": failure})

    with pytest.raises(type(failure), match=str(failure)):
        run_candidate_judge_replay(
            config,
            run_root=storage.root,
            base_url="http://127.0.0.1:8001/v1",
            model="new-model",
            output_path=output,
            fail_fast=fail_fast,
            judge=judge,
        )

    assert not output.exists()
    assert not Path(f"{output}.summary.json").exists()


def test_baseline_uses_final_response_and_accepts_json_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-a")
    path = storage.clip_dir("clip-a") / "debug" / "pair" / "e1" / "raw_responses.json"
    final = json.dumps(_decision(selected_candidate_id="candidate_2").model_dump())
    write_json_atomic(
        path,
        {"responses": ["{}", f"```json\n{final}\n```"]},
    )

    baseline = load_baseline_decision(
        storage,
        clip_uid="clip-a",
        entity_id="e1",
    )

    assert baseline is not None
    assert baseline.selected_candidate_id == "candidate_2"
    assert baseline.completeness == "complete"
    assert baseline.reference_scope == "full"


def test_replay_constructs_production_judge_and_saves_raw_only_on_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-a")
    created: dict[str, object] = {}

    class FakeProductionJudge:
        def __init__(
            self,
            service_config: QwenServiceConfig,
            *,
            repair_retries: int,
            crop_padding_ratio: float,
            evidence_mode: str,
            card_panel_max_side: int,
        ) -> None:
            created.update(
                {
                    "service_config": service_config,
                    "repair_retries": repair_retries,
                    "crop_padding_ratio": crop_padding_ratio,
                    "evidence_mode": evidence_mode,
                    "card_panel_max_side": card_panel_max_side,
                }
            )

        def decide(self, *, entity, candidates, source_images):
            assert entity.entity_id == "e1"
            assert len(candidates) == 3
            assert set(source_images) == {
                candidate.image_path for candidate in candidates
            }
            return EntityReferenceDecisionAttempt(
                decision=_decision(),
                raw_responses=("private raw response",),
                repair_attempts=0,
            )

        def close(self) -> None:
            created["closed"] = True

    monkeypatch.setattr(
        replay_module,
        "QwenEntityReferenceJudge",
        FakeProductionJudge,
    )
    output = config.run_root.parent / "benchmarks" / "raw-replay.jsonl"
    model = str(config_module.ALLOWED_PRETRAINED_ROOT / "Qwen" / "new-model")

    run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model=model,
        output_path=output,
        api_key="benchmark-key",
        save_raw=True,
    )

    service_config = created["service_config"]
    assert isinstance(service_config, QwenServiceConfig)
    assert service_config.base_url == "http://127.0.0.1:8001/v1"
    assert service_config.model == model
    assert service_config.api_key == "benchmark-key"
    assert config.qwen.candidate_judge is not None
    assert service_config.temperature == config.qwen.candidate_judge.temperature
    assert service_config.max_tokens == config.qwen.candidate_judge.max_tokens
    assert created["repair_retries"] == config.pair.repair_retries
    assert created["crop_padding_ratio"] == config.pair.crop_padding_ratio
    assert created["evidence_mode"] == "separate"
    assert created["card_panel_max_side"] == 512
    assert created["closed"] is True
    assert "private raw response" not in output.read_text(encoding="utf-8")
    raw_output = Path(f"{output}.raw.jsonl")
    assert "private raw response" in raw_output.read_text(encoding="utf-8")


def test_replay_paired_card_mode_uses_three_images_and_stays_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, "clip-a")
    before = snapshot_run_files(storage.root)
    completions = _Completions([json.dumps(_decision().model_dump())])
    assert config.qwen.candidate_judge is not None
    judge = QwenEntityReferenceJudge(
        config.qwen.candidate_judge,
        repair_retries=config.pair.repair_retries,
        crop_padding_ratio=config.pair.crop_padding_ratio,
        evidence_mode="paired_card",
        card_panel_max_side=384,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    output = config.run_root.parent / "benchmarks" / "paired-card.jsonl"

    summary = run_candidate_judge_replay(
        config,
        run_root=storage.root,
        base_url="http://127.0.0.1:8001/v1",
        model="new-model",
        output_path=output,
        evidence_mode="paired_card",
        card_panel_max_side=384,
        judge=judge,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    content = completions.calls[0]["messages"][1]["content"]
    labels = [item["text"] for item in content if item["type"] == "text"][1:]
    assert [label.split()[1] for label in labels] == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
    ]
    assert sum(item["type"] == "image_url" for item in content) == 3
    assert record["candidate_count"] == 3
    assert record["input_image_count"] == 3
    assert summary["evidence_mode"] == "paired_card"
    assert summary["card_panel_max_side"] == 384
    assert summary["avg_input_image_count"] == 3.0
    assert snapshot_run_files(storage.root) == before


def test_replay_rejects_output_inside_source_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")

    with pytest.raises(ValueError, match="outside the source run_root"):
        run_candidate_judge_replay(
            config,
            run_root=storage.root,
            base_url="http://127.0.0.1:8001/v1",
            model="new-model",
            output_path=storage.root / "replay.jsonl",
            judge=SimpleNamespace(),
        )


def test_replay_cli_defaults_to_baseline_and_accepts_card_sizes() -> None:
    required = [
        "--config",
        "config.yaml",
        "--run-root",
        "run",
        "--base-url",
        "http://127.0.0.1:8001/v1",
        "--model",
        "model",
        "--output",
        "results.jsonl",
    ]

    baseline = _parser().parse_args(required)
    paired_384 = _parser().parse_args(
        [*required, "--evidence-mode", "paired_card", "--card-panel-max-side", "384"]
    )
    paired_512 = _parser().parse_args(
        [*required, "--evidence-mode", "paired_card", "--card-panel-max-side", "512"]
    )

    assert baseline.evidence_mode == "baseline"
    assert baseline.card_panel_max_side == 512
    assert paired_384.evidence_mode == "paired_card"
    assert paired_384.card_panel_max_side == 384
    assert paired_512.card_panel_max_side == 512
