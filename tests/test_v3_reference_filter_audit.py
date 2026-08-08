from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.pair as pair_module
import r2v_data_v2.v3.reference_filter_audit as audit_module
from r2v_data_v2.reconciliation import write_json_atomic
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
from r2v_data_v2.v3.reference_filter_audit import (
    EmbeddingObservation,
    ExternalReferenceFilterScorer,
    QualityObservation,
    SubjectPoseObservation,
    cosine_similarity,
    discover_local_models,
    run_reference_filter_audit,
    snapshot_run_files,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    RawEntityReferenceDecision,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage
from tools.audit_v3_reference_filters import _parser

WIDTH = 96
HEIGHT = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    audit_root = (writable / "r2v_v3_audits").resolve()
    for path in (writable, dataset_root, pretrained, user_models, audit_root):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    monkeypatch.setattr(audit_module, "ALLOWED_AUDIT_ROOT", audit_root)
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    model = str(pretrained / "Qwen" / "baseline")
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "source-run",
        export_root=writable / "datasets" / "source-export",
        source=SourceConfig(limit=3),
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


def _tracked_frame(slot: int, mask: np.ndarray, object_id: str) -> TrackedMaskFrame:
    rows, columns = np.nonzero(mask)
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        track_valid=True,
        confidence=0.9,
        backend_confidences=[0.9],
        backend_object_ids=[object_id],
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
        per_frame_area_ratio=[0.1] * 10,
        per_frame_confidence=[0.9] * 10,
    )


def _decision(reference_type: str) -> RawEntityReferenceDecision:
    return RawEntityReferenceDecision(
        selected_candidate_id="candidate_1",
        image_quality="high",
        completeness="complete",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        viewpoint="front" if reference_type == "subject" else "not_applicable",
        independent_reference_value=True,
        requires_substantial_invention=False,
        primary_identity_region_visible=True,
        major_structure_visible=True,
        truncation_severity="none",
        discrete_foreground_instance=True,
        mask_matches_target=True,
        completion_needed_for_reference_use=False,
        detached_target_fragments_present=False,
        scope_reason="clear complete reference",
    )


def _add_clip(
    storage: RunStorage,
    clip_uid: str = "clip-a",
    reference_types: tuple[str, ...] = ("subject", "object", "group"),
) -> None:
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(storage.config.dataset_json.parent / f"{clip_uid}.mp4"),
            parent_video_id="parent",
            clip_suffix="1",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=reference_type,
            phrase=f"{reference_type} entity {index}",
            grounding_prompt=f"the {reference_type} entity {index}",
        )
        for index, reference_type in enumerate(reference_types, start=1)
    ]
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="Three distinct foreground entities remain visible.",
            entities=entities,
        ),
    )
    frames: list[SampledFrame] = []
    for slot in range(10):
        yy, xx = np.indices((HEIGHT, WIDTH))
        pixels = np.stack(
            (
                (xx * (slot + 2) + 11) % 256,
                (yy * 9 + slot * 17) % 256,
                ((xx + yy) * 7 + slot * 3) % 256,
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
                timestamp_seconds=float(slot),
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
    starts = (4, 35, 66)
    tracked: dict[str, TrackedEntityMasks] = {}
    for entity, start in zip(entities, starts[: len(entities)], strict=True):
        tracked_frames: list[TrackedMaskFrame] = []
        for slot in range(10):
            mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
            shift = slot % 2
            mask[14:50, start + shift : start + 24 + shift] = True
            tracked_frames.append(
                _tracked_frame(slot, mask, f"object-{entity.entity_id}")
            )
        tracked[entity.entity_id] = TrackedEntityMasks(
            status="ready",
            reference_type=entity.reference_type,
            grounding_prompt=entity.grounding_prompt,
            backend_object_ids=[f"object-{entity.entity_id}"],
            frames=tracked_frames,
        )
    storage.write_masks(
        clip_uid,
        TrackedMasksArtifact(
            clip_uid=clip_uid,
            width=WIDTH,
            height=HEIGHT,
            entities=tracked,
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=[entity.entity_id for entity in entities],
            entity_visibility_summary={
                entity.entity_id: _visibility() for entity in entities
            },
        ),
    )
    references: list[EntityReferenceState] = []
    for index, entity in enumerate(entities):
        selected_path = storage.selected_entity_path(clip_uid, entity.entity_id)
        rgba = np.full((36, 24, 4), 255, dtype=np.uint8)
        rgba[..., :3] = (40 + index * 60, 90 + index * 20, 130)
        Image.fromarray(rgba, mode="RGBA").save(selected_path, format="PNG")
        references.append(
            EntityReferenceState(
                entity_id=entity.entity_id,
                status="ready",
                reference_scope="full",
                visible_region="whole",
                whole_entity_recognizable=True,
                identity_features_visible=True,
                scope_reason="complete source-faithful reference",
                image_path=(
                    selected_path.relative_to(storage.root).as_posix()
                ),
                source_frame_index=0,
                image_quality="high",
                completeness="complete",
                viewpoint=(
                    "front"
                    if entity.reference_type == "subject"
                    else "not_applicable"
                ),
                independent_reference_value=True,
                requires_substantial_invention=False,
                primary_identity_region_visible=True,
                major_structure_visible=True,
                truncation_severity="none",
                discrete_foreground_instance=True,
                mask_matches_target=True,
                completion_needed_for_reference_use=False,
                detached_target_fragments_present=False,
            )
        )
        debug = storage.clip_dir(clip_uid) / "debug" / "pair" / entity.entity_id
        debug.mkdir(parents=True)
        raw_decision = json.dumps(_decision(entity.reference_type).model_dump())
        if index == 0:
            raw_decision = f"```json\n{raw_decision}\n```"
        write_json_atomic(
            debug / "raw_responses.json",
            {"responses": [raw_decision]},
        )
    storage.write_references(clip_uid, ReferencesState(entities=references))


class FakeQualityScorer:
    backend = "fake_iqa"
    model_name = "fake-quality"

    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def score(self, image: Image.Image) -> QualityObservation:
        del image
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("fake quality failure")
        return QualityObservation(
            quality_score=0.9,
            quality_scale_min=0.0,
            quality_scale_max=1.0,
            aesthetic_score=0.2,
            aesthetic_scale_min=0.0,
            aesthetic_scale_max=1.0,
            backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=0.01,
            raw_metrics={"orientation_metadata": "rear_three_quarter"},
        )


class FakeEmbeddingScorer:
    backend = "fake_dino"
    model_name = "fake-embedding"
    fingerprint = "fake-embedding-v1"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = iter(vectors)
        self.calls = 0

    def embed(self, image: Image.Image) -> EmbeddingObservation:
        del image
        self.calls += 1
        return EmbeddingObservation(
            embedding=next(self.vectors),
            backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=0.02,
            raw_metrics={},
        )


class FakePoseScorer:
    backend = "fake_pose"
    model_name = "fake-subject-pose"

    def inspect(self, image: Image.Image) -> SubjectPoseObservation:
        del image
        return SubjectPoseObservation(
            face_detected=True,
            face_detection_confidence=0.95,
            face_bbox_area_ratio=0.12,
            head_visible=True,
            yaw=-35.0,
            pitch=4.0,
            roll=1.0,
            pose_backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=0.03,
            subject_view_quality_score=0.8,
        )


def _records(output_root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_root / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _embedding_vectors() -> list[list[float]]:
    return [
        [1.0, 0.0],
        [0.8, 0.6],
        [0.8, -0.6],
        [-1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]


def test_audit_is_read_only_and_reuses_production_candidate_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage)
    before = snapshot_run_files(storage.root)
    calls: list[tuple[str, str]] = []
    production_builder = pair_module.build_entity_reference_candidates

    def tracked_builder(*args: object, **kwargs: object):
        calls.append((str(kwargs["clip_uid"]), kwargs["entity"].entity_id))
        return production_builder(*args, **kwargs)

    monkeypatch.setattr(pair_module, "build_entity_reference_candidates", tracked_builder)
    monkeypatch.setattr(
        pair_module,
        "QwenEntityReferenceJudge",
        lambda *args, **kwargs: pytest.fail("audit instantiated production judge"),
    )
    output = audit_module.ALLOWED_AUDIT_ROOT / "audit-read-only"
    summary = run_reference_filter_audit(
        config,
        run_root=storage.root,
        output_root=output,
        artifact_scope="both",
    )

    records = _records(output)
    assert snapshot_run_files(storage.root) == before
    assert calls == [("clip-a", "e1"), ("clip-a", "e2"), ("clip-a", "e3")]
    assert summary["candidate_count"] == 9
    assert summary["final_reference_count"] == 3
    assert summary["qwen_calls_added"] == 0
    candidates = [item for item in records if item["artifact_scope"] == "candidate"]
    assert [item["candidate_id"] for item in candidates] == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
    ] * 3
    finals = [item for item in records if item["artifact_scope"] == "final"]
    assert [item["image_path"] for item in finals] == [
        f"clips/clip-a/selected/e{index}.png" for index in range(1, 4)
    ]
    source = inspect.getsource(audit_module)
    assert "QwenEntityReferenceJudge(" not in source
    assert "OpenAI(" not in source


def test_quality_embedding_and_pose_features_remain_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage)
    output = audit_module.ALLOWED_AUDIT_ROOT / "audit-features"
    run_reference_filter_audit(
        config,
        run_root=storage.root,
        output_root=output,
        artifact_scope="candidates",
        quality_backend="fake_iqa",
        embedding_backend="fake_dino",
        subject_pose_backend="fake_pose",
        quality_scorer=FakeQualityScorer(),
        embedding_scorer=FakeEmbeddingScorer(_embedding_vectors()),
        subject_pose_scorer=FakePoseScorer(),
    )
    records = _records(output)
    first = records[0]
    assert first["quality"]["quality_score"] == 0.9
    assert first["quality"]["aesthetic_score"] == 0.2
    assert first["embedding"]["same_entity_mean_similarity"] == pytest.approx(0.8)
    assert first["embedding"]["representativeness_score"] == pytest.approx(0.8)
    assert first["embedding"]["max_other_entity_similarity"] == pytest.approx(0.0)
    assert first["embedding"]["inter_entity_margin"] == pytest.approx(0.8)
    assert first["subject_pose"]["yaw"] == -35.0
    object_records = [item for item in records if item["reference_type"] == "object"]
    group_records = [item for item in records if item["reference_type"] == "group"]
    assert all(item["subject_pose"]["status"] == "not_applicable" for item in object_records)
    assert all(item["subject_pose"]["status"] == "not_applicable" for item in group_records)
    assert all(item["quality"]["quality_score"] == 0.9 for item in object_records)
    assert all(
        item["quality"]["raw_metrics"]["orientation_metadata"]
        == "rear_three_quarter"
        for item in object_records
    )
    forbidden = {
        "final_score",
        "reference_score",
        "keep_score",
        "drop_score",
        "ranking_score",
        "should_keep",
        "should_drop",
        "recommended_action",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert not (keys(records) & forbidden)
    summary = json.loads((output / "audit.summary.json").read_text())
    assert summary["thresholds_applied"] is False
    assert summary["production_filtering_applied"] is False
    assert summary["selected_vs_non_selected"][
        "selected_is_highest_quality_denominator"
    ] == 3
    assert summary["embedding_dimensions"] == [2]
    assert summary["representativeness_definition"] == {
        "formula": "mean cosine similarity to other same-entity candidates",
        "centroid_value_equivalent": False,
        "centroid_rank_equivalent": True,
        "centroid_rank_equivalence_scope": (
            "same normalized candidate set with a nonzero centroid"
        ),
    }
    rank = summary["selected_representativeness_rank"]
    assert rank["entity_count"] == 3
    assert rank["selected_rank_1_count"] == 3
    assert rank["selected_rank_1_rate"] == 1.0
    assert rank["by_reference_type"]["object"]["selected_rank_1_rate"] == 1.0
    assert rank["selected_rank_last_cases"] == []
    assert summary["runtime"]["visual_encoder"][
        "estimated_seconds_per_three_candidate_entity"
    ] == pytest.approx(0.06)


def test_single_entity_margin_is_null_and_cosine_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, reference_types=("object",))
    output = audit_module.ALLOWED_AUDIT_ROOT / "audit-single"
    run_reference_filter_audit(
        config,
        run_root=storage.root,
        output_root=output,
        artifact_scope="candidates",
        embedding_backend="fake_dino",
        embedding_scorer=FakeEmbeddingScorer(_embedding_vectors()[:3]),
    )
    records = _records(output)
    assert len(records) == 3
    assert all(
        item["embedding"]["max_other_entity_similarity"] is None
        and item["embedding"]["inter_entity_margin"] is None
        for item in records
    )


def test_scorer_failure_is_isolated_and_fail_fast_is_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage)
    before = snapshot_run_files(storage.root)
    output = audit_module.ALLOWED_AUDIT_ROOT / "audit-failure"
    run_reference_filter_audit(
        config,
        run_root=storage.root,
        output_root=output,
        artifact_scope="candidates",
        quality_backend="fake_iqa",
        quality_scorer=FakeQualityScorer(fail_call=2),
    )
    records = _records(output)
    assert len(records) == 9
    assert [item["quality"]["status"] for item in records].count("failed") == 1
    assert snapshot_run_files(storage.root) == before

    fast_output = audit_module.ALLOWED_AUDIT_ROOT / "audit-fail-fast"
    with pytest.raises(RuntimeError, match="fake quality failure"):
        run_reference_filter_audit(
            config,
            run_root=storage.root,
            output_root=fast_output,
            artifact_scope="candidates",
            quality_backend="fake_iqa",
            quality_scorer=FakeQualityScorer(fail_call=1),
            fail_fast=True,
        )
    assert not fast_output.exists()
    assert snapshot_run_files(storage.root) == before


def test_embedding_cache_and_model_discovery_stay_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = RunStorage(config)
    storage.initialize(git_commit="source-run")
    _add_clip(storage, reference_types=("object",))
    before = snapshot_run_files(storage.root)
    output = audit_module.ALLOWED_AUDIT_ROOT / "audit-cache"
    run_reference_filter_audit(
        config,
        run_root=storage.root,
        output_root=output,
        artifact_scope="candidates",
        embedding_backend="fake_dino",
        embedding_scorer=FakeEmbeddingScorer(_embedding_vectors()[:3]),
    )
    assert len(list((output / "cache").glob("*.npy"))) == 3
    assert snapshot_run_files(storage.root) == before

    discovery_root = tmp_path / "models"
    (discovery_root / "one" / "two" / "dinov2-small").mkdir(parents=True)
    too_deep = discovery_root / "one" / "two" / "three" / "musiq-hidden"
    too_deep.mkdir(parents=True)
    found = discover_local_models([discovery_root], max_depth=3)
    assert any(item["backend"] == "dinov2" for item in found)
    assert all("musiq-hidden" not in item["path"] for item in found)
    with pytest.raises(ValueError, match="max_depth"):
        discover_local_models([discovery_root], max_depth=4)


def test_external_worker_uses_explicit_local_adapter(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "adapter"
    code_root.mkdir()
    (code_root / "r2v_reference_filter_adapter.py").write_text(
        """
class Scorer:
    def eval(self):
        return self

    def score(self, image):
        assert image.mode == "RGB"
        return {
            "quality_score": 0.7,
            "quality_scale_min": 0.0,
            "quality_scale_max": 1.0,
            "aesthetic_score": 0.3,
            "aesthetic_scale_min": 0.0,
            "aesthetic_scale_max": 1.0,
            "raw_metrics": {"offline": True},
        }

def load_scorer(*, kind, backend, model_path, local_files_only):
    assert kind == "quality"
    assert backend == "fake_iqa"
    assert model_path.is_file()
    assert local_files_only is True
    return Scorer()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"local-only")
    scorer = ExternalReferenceFilterScorer(
        kind="quality",
        backend="fake_iqa",
        python_executable=Path(__file__).parents[1] / ".venv" / "bin" / "python",
        code_root=code_root,
        model_path=model_path,
    )
    try:
        result = scorer.score(Image.new("RGBA", (7, 5), (1, 2, 3, 128)))
    finally:
        scorer.close()
    assert result.quality_score == 0.7
    assert result.aesthetic_score == 0.3
    assert result.raw_metrics == {"offline": True}


def test_cli_defaults_all_model_backends_to_none() -> None:
    arguments = _parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--run-root",
            "source-run",
            "--output-root",
            "audit-output",
        ]
    )
    assert arguments.artifact_scope == "both"
    assert arguments.quality_backend == "none"
    assert arguments.embedding_backend == "none"
    assert arguments.subject_pose_backend == "none"
    assert arguments.discover_local_models is False


def test_selected_representativeness_rank_records_last_place_anomaly() -> None:
    records = []
    for candidate_id, score in (
        ("candidate_1", 0.1),
        ("candidate_2", 0.9),
        ("candidate_3", 0.5),
    ):
        records.append(
            {
                "clip_uid": "clip-a",
                "entity_id": "e1",
                "reference_type": "object",
                "artifact_scope": "candidate",
                "candidate_id": candidate_id,
                "is_current_selected": candidate_id == "candidate_1",
                "embedding": {
                    "status": "succeeded",
                    "representativeness_score": score,
                },
                "production_baseline": {
                    "completeness": "complete",
                    "reference_scope": "full",
                    "viewpoint": "not_applicable",
                },
            }
        )

    result = audit_module._selected_representativeness_rank_analysis(records)

    assert result["selected_rank_3_count"] == 1
    assert result["selected_rank_1_rate"] == 0.0
    assert result["by_reference_type"]["object"]["selected_rank_3_count"] == 1
    assert result["selected_rank_last_cases"] == result["cases"]
    assert result["cases"][0]["representativeness_values"] == [
        {"candidate_id": "candidate_1", "representativeness_score": 0.1},
        {"candidate_id": "candidate_2", "representativeness_score": 0.9},
        {"candidate_id": "candidate_3", "representativeness_score": 0.5},
    ]
