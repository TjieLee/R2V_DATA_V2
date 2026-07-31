from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.annotation import (
    SYSTEM_PROMPT,
    AnnotationStats,
    QwenAnnotationClient,
    annotate_clips,
)
from r2v_data_v2.v3.config import (
    DebugConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.schemas import (
    ClipRecord,
    CoverageState,
    EntityReferenceState,
    ExportState,
    InstructionState,
    PairingState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug: bool = False,
    repair_retries: int = 1,
    source_start_index: int = 0,
    source_limit: int | None = 100,
    source_allow_full_run: bool = False,
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
    annotation_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    remove_model = pretrained / "Qwen" / "Qwen-Image-Edit-2511"
    adapter = user_models / "Qwen-Image-Edit-2511-Object-Remover"
    config = V3Config(
        dataset_json=dataset_root / "source.json",
        run_root=writable / "runs" / "annotation",
        export_root=writable / "datasets" / "annotation-v1",
        source=SourceConfig(
            start_index=source_start_index,
            limit=source_limit,
            allow_full_run=source_allow_full_run,
        ),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(
                model=str(annotation_model),
                repair_retries=repair_retries,
            ),
            instruction_writer=QwenServiceConfig(model=str(annotation_model)),
        ),
        remove=RemoveConfig(
            base_model_path=remove_model,
            adapter_path=adapter,
        ),
        debug=DebugConfig(save_diagnostics=debug),
    )
    config.validate()
    return config


def _config_path(config: V3Config, tmp_path: Path) -> Path:
    path = tmp_path / "v3-annotation.yaml"
    source_limit = (
        ""
        if config.source.limit is None
        else f"  limit: {config.source.limit}\n"
    )
    path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        f"  start_index: {config.source.start_index}\n"
        f"{source_limit}"
        "  allow_full_run: "
        f"{str(config.source.allow_full_run).lower()}\n"
        "qwen:\n"
        "  annotation:\n"
        f"    model: {config.qwen.annotation.model}\n"
        f"    repair_retries: {config.qwen.annotation.repair_retries}\n"
        "debug:\n"
        f"  save_diagnostics: {str(config.debug.save_diagnostics).lower()}\n",
        encoding="utf-8",
    )
    return path


def _video(config: V3Config, name: str = "scene_1_0.mp4") -> Path:
    path = config.dataset_json.parent / "videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mock video")
    return path


def _write_source(
    config: V3Config,
    records: list[dict[str, object]],
) -> None:
    config.dataset_json.write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )


def _payload(
    *,
    caption: str | None = None,
) -> dict[str, object]:
    return {
        "t2v_caption": caption
        or (
            "A woman in a yellow coat walks beside a wooden table through a "
            "sunlit plaza as the camera tracks backward."
        ),
        "entities": [
            {
                "entity_id": "e1",
                "phrase": "A woman in a yellow coat",
                "grounding_prompt": "woman wearing a yellow coat",
                "canonical_label": "woman",
                "category": "person",
                "reference_worthy": True,
                "salience": "primary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "independent",
                "selection_reason": "stable primary subject",
            },
            {
                "entity_id": "e2",
                "phrase": "a wooden table",
                "grounding_prompt": "wooden table",
                "canonical_label": "table",
                "category": "object",
                "reference_worthy": True,
                "salience": "secondary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "important_independent_object",
                "selection_reason": "distinct visible object",
            },
        ],
        "relations": [
            {
                "subject_id": "e1",
                "predicate": "walking beside",
                "object_id": "e2",
            }
        ],
        "background": {
            "phrase": "a sunlit plaza",
            "grounding_prompt": "empty sunlit plaza",
            "reference_worthy": True,
        },
    }


class _FakeQwenClient(QwenAnnotationClient):
    def __init__(
        self,
        config: QwenAnnotationConfig,
        responses: list[str | Exception],
    ) -> None:
        self.config = config
        self._responses: Iterator[str | Exception] = iter(responses)
        self.requests: list[list[dict[str, object]]] = []

    def _request(self, messages: list[dict[str, object]]) -> str:
        self.requests.append(messages)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _storage_with_manifest(
    config: V3Config,
) -> tuple[RunStorage, str]:
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")
    stats = build_manifest(config, storage)
    assert stats.processed == 1
    clip_uid = next(storage.iter_clips()).clip_uid
    return storage, clip_uid


def _annotate_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, object]],
    *,
    repair_retries: int = 0,
    source_fields: dict[str, object] | None = None,
) -> tuple[AnnotationStats, ClipRecord, _FakeQwenClient]:
    config = _config(
        tmp_path,
        monkeypatch,
        repair_retries=repair_retries,
    )
    video = _video(config)
    source_record: dict[str, object] = {"file_path": str(video)}
    source_record.update(source_fields or {})
    _write_source(config, [source_record])
    storage, clip_uid = _storage_with_manifest(config)
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(payload) for payload in payloads],
    )
    stats = annotate_clips(config, storage, client=client)
    return stats, storage.read_clip(clip_uid), client


def _seed_ready_downstream(storage: RunStorage, clip_uid: str) -> None:
    storage.write_coverage(
        clip_uid,
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
        ),
    )
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=[
                EntityReferenceState(
                    entity_id="e1",
                    status="ready",
                    reference_scope="local",
                    visible_region="upper_body",
                    whole_entity_recognizable=False,
                    identity_features_visible=True,
                    scope_reason="coherent upper body",
                    image_path=f"clips/{clip_uid}/selected/e1.png",
                    source_frame_index=2,
                )
            ]
        ),
    )
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=["e1"],
            tokens={"e1": "<ref_subject_1>"},
        ),
    )
    storage.write_instruction(
        clip_uid,
        InstructionState(
            status="ready",
            r2v_instruction="Generate a shot using <ref_subject_1>.",
        ),
    )
    storage.write_export(
        clip_uid,
        ExportState(accepted=True, reason=None),
    )


def test_manifest_creates_one_clip_json_without_stage_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video), "text": "draft"}])
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    stats = build_manifest(config, storage)

    identity = parse_clip_identity(video)
    clip = storage.read_clip(identity.clip_uid)
    assert stats.to_dict() == {
        "processed": 1,
        "skipped_existing": 0,
        "failed": 0,
    }
    assert clip.source.video_path == str(video.resolve())
    assert clip.source.source_index == 0
    assert clip.source.caption_raw == "draft"
    assert clip.source.metadata == {"text": "draft"}
    assert list(config.resolved_run_root.rglob("clip.json")) == [
        storage.clip_path(identity.clip_uid)
    ]
    assert not list(config.resolved_run_root.rglob("*.mp4"))
    assert not list(config.resolved_run_root.rglob("*.jsonl"))
    assert storage.read_run().counts["manifest.processed"] == 1


def test_manifest_selection_limit_creates_only_five_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, source_limit=5)
    records = [
        {"file_path": str(_video(config, f"scene_{index}_0.mp4"))}
        for index in range(8)
    ]
    _write_source(config, records)
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    stats = build_manifest(config, storage)

    clips = list(storage.iter_clips())
    assert stats.processed == 5
    assert len(clips) == 5
    assert {clip.source.source_index for clip in clips} == set(range(5))


def test_source_limit_is_required_without_full_run_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="source.limit is required"):
        _config(
            tmp_path,
            monkeypatch,
            source_limit=None,
            source_allow_full_run=False,
        )


def test_source_selection_changes_config_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    changed = replace(
        config,
        source=replace(config.source, start_index=2, limit=5),
    )

    assert changed.fingerprint() != config.fingerprint()


def test_manifest_rerun_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"video_path": str(video)}])
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    first = build_manifest(config, storage)
    clip_path = next(config.resolved_run_root.rglob("clip.json"))
    original = clip_path.read_bytes()
    second = build_manifest(config, storage)

    assert first.processed == 1
    assert second.to_dict() == {
        "processed": 0,
        "skipped_existing": 1,
        "failed": 0,
    }
    assert clip_path.read_bytes() == original


def test_bad_manifest_source_does_not_block_valid_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [
            {"text": "missing path"},
            {"file_path": str(video), "text": "valid"},
        ],
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="annotation-test")

    stats = build_manifest(config, storage)

    assert stats.processed == 1
    assert stats.failed == 1
    assert len(list(storage.iter_clips())) == 1
    failure = json.loads(storage.failures_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "manifest"
    assert failure["details"]["source_index"] == 0


def test_manifest_source_identity_conflict_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    payload = storage.read_clip(clip_uid).model_dump(mode="json")
    payload["source"]["parent_video_id"] = "conflicting-parent"
    write_json_atomic(storage.clip_path(clip_uid), payload)

    stats = build_manifest(config, storage)

    assert stats.failed == 1
    assert stats.skipped_existing == 0
    assert (
        storage.read_clip(clip_uid).source.parent_video_id
        == "conflicting-parent"
    )


def test_annotation_writes_semantic_fields_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [
            {
                "file_path": str(video),
                "text": "Draft evidence.",
                "title": "Metadata evidence",
            }
        ],
    )
    storage, clip_uid = _storage_with_manifest(config)
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )

    stats = annotate_clips(config, storage, client=client)

    clip = storage.read_clip(clip_uid)
    assert stats.processed == 1
    assert clip.annotation is not None
    assert clip.annotation.t2v_caption.startswith("A woman")
    assert [entity.entity_id for entity in clip.annotation.entities] == [
        "e1",
        "e2",
    ]
    assert clip.annotation.relations[0].predicate == "walking beside"
    assert clip.annotation.background is not None
    assert clip.annotation.background.phrase == "a sunlit plaza"
    assert clip.references.entities == []
    assert clip.pairing is None
    assert clip.instruction is None
    serialized = storage.clip_path(clip_uid).read_text(encoding="utf-8")
    assert "<ref_" not in serialized
    assert "prompt_with_refs" not in serialized
    assert "r2v_instruction" not in serialized


def test_annotation_with_reference_token_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=0)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    payload = _payload(
        caption="A woman <ref_subject_1> walks through a sunlit plaza."
    )
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(payload)],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.failed == 1
    annotation = storage.read_clip(clip_uid).annotation
    assert annotation is not None
    assert annotation.status == "failed"
    assert annotation.reason == "reference_token_in_annotation"
    failure = json.loads(storage.failures_path.read_text(encoding="utf-8"))
    assert failure["reason"] == "reference_token_in_annotation"


def test_unknown_relation_is_dropped_without_failing_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    payload = _payload()
    payload["relations"] = [
        {
            "subject_id": "e1",
            "predicate": "near",
            "object_id": "missing",
        }
    ]
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(payload)],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.processed == 1
    annotation = storage.read_clip(clip_uid).annotation
    assert annotation is not None
    assert annotation.relations == []


def test_changed_annotation_invalidates_all_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    first = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )
    annotate_clips(config, storage, client=first)
    _seed_ready_downstream(storage, clip_uid)
    changed = _payload(
        caption=(
            "A woman in a yellow coat walks beside a wooden table through a "
            "sunlit plaza, then pauses as the camera pans left."
        )
    )

    stats = annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(changed)],
        ),
    )

    clip = storage.read_clip(clip_uid)
    assert stats.processed == 1
    assert clip.coverage is None
    assert clip.references == ReferencesState()
    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_identical_annotation_rerun_preserves_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    raw = json.dumps(_payload())
    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )
    _seed_ready_downstream(storage, clip_uid)
    before = storage.read_clip(clip_uid)
    clip_bytes = storage.clip_path(clip_uid).read_bytes()

    stats = annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )

    assert stats.processed == 1
    assert storage.read_clip(clip_uid) == before
    assert storage.clip_path(clip_uid).read_bytes() == clip_bytes


def test_qwen_request_failure_uses_unified_failure_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    client = _FakeQwenClient(
        config.qwen.annotation,
        [RuntimeError("endpoint unavailable")],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.failed == 1
    annotation = storage.read_clip(clip_uid).annotation
    assert annotation is not None
    assert annotation.status == "failed"
    assert annotation.reason == "qwen_request_failed"
    failure = json.loads(storage.failures_path.read_text(encoding="utf-8"))
    assert failure["clip_uid"] == clip_uid
    assert failure["stage"] == "annotate"
    assert failure["reason"] == "qwen_request_failed"
    assert not list(config.resolved_run_root.rglob("qwen_failed.jsonl"))


def test_failed_annotation_overwrite_invalidates_downstream_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=0)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(_payload())],
        ),
    )
    _seed_ready_downstream(storage, clip_uid)

    stats = annotate_clips(
        config,
        storage,
        overwrite=True,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [RuntimeError("endpoint unavailable")],
        ),
    )

    clip = storage.read_clip(clip_uid)
    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "failed"
    assert clip.coverage is None
    assert clip.references == ReferencesState()
    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_annotation_uses_clip_source_evidence_without_rescanning_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [
            {
                "file_path": str(video),
                "text": "Stored draft evidence.",
                "title": "Stored metadata evidence",
            }
        ],
    )
    storage, clip_uid = _storage_with_manifest(config)
    config.dataset_json.unlink()
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.processed == 1
    request = str(client.requests[0])
    assert "Stored draft evidence." in request
    assert "Stored metadata evidence" in request
    assert storage.read_clip(clip_uid).annotation is not None


def test_invalid_structured_output_can_be_repaired_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, repair_retries=1)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    client = _FakeQwenClient(
        config.qwen.annotation,
        ["not json", json.dumps(_payload())],
    )

    stats = annotate_clips(config, storage, client=client)

    assert stats.processed == 1
    assert stats.repaired == 1
    assert len(client.requests) == 2
    repair_content = client.requests[1][-1]["content"]
    assert isinstance(repair_content, list)
    assert "Repair only" in str(repair_content[-1]["text"])
    assert storage.read_clip(clip_uid).annotation is not None
    assert not (storage.clip_dir(clip_uid) / "debug").exists()


def test_raw_responses_are_saved_only_when_debug_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch, debug=True)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    raw = json.dumps(_payload())

    annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(config.qwen.annotation, [raw]),
    )

    diagnostic = json.loads(
        (storage.clip_dir(clip_uid) / "debug" / "annotation_raw.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["raw_responses"] == [raw]


class _CompletionsStub:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.raw)
                )
            ]
        )


class _StrictFallbackCompletionsStub(_CompletionsStub):
    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise BadRequestError(
                "strict schema unsupported",
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST",
                        "http://127.0.0.1:8000/v1/chat/completions",
                    ),
                ),
                body={},
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.raw)
                )
            ]
        )


def test_qwen_request_uses_full_video_and_strict_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    completions = _CompletionsStub(json.dumps(_payload()))
    openai_stub = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=openai_stub,
    )

    result = client.annotate(
        video_path=video,
        caption_raw="draft",
        metadata={"title": "evidence"},
    )

    assert result.annotation.status == "ready"
    request = completions.calls[0]
    assert request["model"] == str(
        config.dataset_json.parent.parent
        / "pretrained"
        / "Qwen"
        / "Qwen3-VL-32B-Instruct"
    )
    assert request["temperature"] == 0.0
    assert request["extra_body"] == {
        "mm_processor_kwargs": {
            "fps": 2.0,
            "do_sample_frames": False,
        }
    }
    response_format = request["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["strict"] is True
    messages = request["messages"]
    assert isinstance(messages, list)
    user_content = messages[-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["video_url"]["url"] == video.resolve().as_uri()


def test_qwen_falls_back_to_json_object_when_strict_schema_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    completions = _StrictFallbackCompletionsStub(json.dumps(_payload()))
    client = QwenAnnotationClient(
        config.qwen.annotation,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )

    result = client.annotate(
        video_path=video,
        caption_raw="draft",
        metadata={"title": "evidence"},
    )

    assert result.annotation.status == "ready"
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert completions.calls[1]["response_format"] == {"type": "json_object"}


def test_identity_prompt_limits_names_to_explicit_source_evidence() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split())
    assert "Never identify a person from appearance." in normalized
    assert "draft caption or metadata" in normalized
    assert "set name_evidence to draft_caption or metadata" in normalized


def test_system_prompt_forbids_cross_shot_relations() -> None:
    normalized = " ".join(SYSTEM_PROMPT.split()).casefold()
    assert "simultaneously visible in the same shot or time segment" in normalized
    assert "never create a spatial relation across a cut" in normalized


def test_explicit_person_name_from_metadata_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(
        config,
        [
            {
                "file_path": str(video),
                "person_name": "Alice",
            }
        ],
    )
    storage, clip_uid = _storage_with_manifest(config)
    payload = _payload(
        caption=(
            "Alice in a yellow coat walks beside a wooden table through a "
            "sunlit plaza as the camera tracks backward."
        )
    )
    first_entity = payload["entities"][0]
    assert isinstance(first_entity, dict)
    first_entity.update(
        {
            "phrase": "Alice in a yellow coat",
            "canonical_label": "Alice",
            "genericity": "named",
            "name_evidence": "metadata",
        }
    )

    stats = annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(payload)],
        ),
    )

    assert stats.processed == 1
    annotation = storage.read_clip(clip_uid).annotation
    assert annotation is not None
    assert annotation.entities[0].canonical_label == "Alice"
    assert annotation.entities[0].name_evidence == "metadata"


def test_generic_entity_with_name_evidence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    entity = payload["entities"][0]
    assert isinstance(entity, dict)
    entity["name_evidence"] = "draft_caption"

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
        source_fields={"text": "A woman appears."},
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "failed"
    assert clip.annotation.reason == "unexpected_name_evidence"


def test_reference_worthy_incidental_entity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    entity = payload["entities"][1]
    assert isinstance(entity, dict)
    entity["salience"] = "incidental"

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.reason == "invalid_reference_salience"


def test_bronze_statue_soldiers_cannot_use_person_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(
        caption=(
            "Bronze statue soldiers stand beside a wooden table in a sunlit "
            "plaza as the camera tracks backward."
        )
    )
    entity = payload["entities"][0]
    assert isinstance(entity, dict)
    entity.update(
        {
            "phrase": "Bronze statue soldiers",
            "grounding_prompt": "bronze statue soldiers",
            "canonical_label": "soldiers",
            "category": "person",
        }
    )

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.reason == "depicted_person_category"


def test_real_soldier_is_not_rejected_as_a_depicted_person(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(
        caption=(
            "A soldier in a green uniform walks beside a wooden table through "
            "a sunlit plaza as the camera tracks backward."
        )
    )
    entity = payload["entities"][0]
    assert isinstance(entity, dict)
    entity.update(
        {
            "phrase": "A soldier in a green uniform",
            "grounding_prompt": "soldier wearing a green uniform",
            "canonical_label": "soldier",
            "category": "person",
        }
    )

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.processed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


def test_reference_entity_phrase_cannot_duplicate_background_phrase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    entity = payload["entities"][1]
    assert isinstance(entity, dict)
    entity.update(
        {
            "phrase": "a sunlit plaza",
            "grounding_prompt": "sunlit plaza",
            "canonical_label": "plaza",
        }
    )

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.reason == "reference_background_overlap"


def test_independent_building_entity_does_not_overlap_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(
        caption=(
            "A woman in a yellow coat walks beside a brick building through a "
            "sunlit plaza as the camera tracks backward."
        )
    )
    entity = payload["entities"][1]
    assert isinstance(entity, dict)
    entity.update(
        {
            "phrase": "a brick building",
            "grounding_prompt": "brick building",
            "canonical_label": "building",
        }
    )

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.processed == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"


@pytest.mark.parametrize(
    ("field", "forbidden_text"),
    [
        ("caption", "serene"),
        ("selection_reason", "determination"),
        ("relation", "shouting"),
    ],
)
def test_forbidden_inference_language_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forbidden_text: str,
) -> None:
    payload = _payload()
    if field == "caption":
        payload["t2v_caption"] = (
            "A woman in a yellow coat walks beside a wooden table through a "
            f"{forbidden_text} plaza as the camera tracks backward."
        )
    elif field == "selection_reason":
        entity = payload["entities"][0]
        assert isinstance(entity, dict)
        entity["selection_reason"] = f"shows {forbidden_text}"
    else:
        relation = payload["relations"][0]
        assert isinstance(relation, dict)
        relation["predicate"] = f"{forbidden_text} near"

    stats, clip, _ = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [payload],
    )

    assert stats.failed == 1
    assert clip.annotation is not None
    assert clip.annotation.reason == "forbidden_inference_language"


def test_semantic_validation_issue_can_be_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _payload()
    entity = invalid["entities"][0]
    assert isinstance(entity, dict)
    entity["name_evidence"] = "draft_caption"

    stats, clip, client = _annotate_payloads(
        tmp_path,
        monkeypatch,
        [invalid, _payload()],
        repair_retries=1,
        source_fields={"text": "A woman appears."},
    )

    assert stats.processed == 1
    assert stats.repaired == 1
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"
    assert len(client.requests) == 2
    assert "unexpected_name_evidence" in str(client.requests[1])


def test_pipeline_runs_requested_stages_in_v3_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video), "text": "draft"}])
    client = _FakeQwenClient(
        config.qwen.annotation,
        [json.dumps(_payload())],
    )

    result = run_pipeline_v3(
        config_path=_config_path(config, tmp_path),
        stages=("annotate", "manifest"),
        git_commit="annotation-test",
        annotation_client=client,
    )

    assert result["completed_stages"] == ["manifest", "annotate"]
    assert result["manifest"] == {
        "processed": 1,
        "skipped_existing": 0,
        "failed": 0,
    }
    assert result["annotate"] == {
        "processed": 1,
        "skipped_existing": 0,
        "failed": 0,
        "repaired": 0,
    }
    run = RunStorage(config).read_run()
    assert run.counts["manifest.processed"] == 1
    assert run.counts["annotate.processed"] == 1


def test_reference_worthy_candidates_are_capped_after_phrase_sanitizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    video = _video(config)
    _write_source(config, [{"file_path": str(video)}])
    storage, clip_uid = _storage_with_manifest(config)
    payload = _payload()
    entities = list(payload["entities"])
    entities.extend(
        [
            {
                **entities[1],
                "entity_id": "e3",
                "phrase": "yellow coat",
                "grounding_prompt": "yellow coat",
                "canonical_label": "coat",
                "salience": "primary",
            },
            {
                **entities[1],
                "entity_id": "e4",
                "phrase": "an absent red bicycle",
                "grounding_prompt": "red bicycle",
                "canonical_label": "bicycle",
                "salience": "primary",
            },
            {
                **entities[1],
                "entity_id": "e5",
                "phrase": "the camera",
                "grounding_prompt": "camera viewpoint",
                "canonical_label": "camera",
                "salience": "primary",
            },
        ]
    )
    payload["entities"] = entities

    stats = annotate_clips(
        config,
        storage,
        client=_FakeQwenClient(
            config.qwen.annotation,
            [json.dumps(payload)],
        ),
    )

    annotation = storage.read_clip(clip_uid).annotation
    assert stats.processed == 1
    assert annotation is not None
    assert sum(entity.reference_worthy for entity in annotation.entities) <= 3
    absent = next(
        entity
        for entity in annotation.entities
        if entity.entity_id == "e4"
    )
    assert absent.reference_worthy is False


@pytest.mark.parametrize(
    ("video", "message"),
    [
        (
            v3_config_module.QwenVideoConfig(fps=1.0),
            "fps to be 2.0",
        ),
        (
            v3_config_module.QwenVideoConfig(do_sample_frames=True),
            "must disable HF video re-sampling",
        ),
    ],
)
def test_v3_annotation_requires_complete_video_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video: v3_config_module.QwenVideoConfig,
    message: str,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=message):
        replace(
            config,
            qwen=replace(
                config.qwen,
                annotation=replace(
                    config.qwen.annotation,
                    video=video,
                ),
            ),
        ).validate()


def test_negative_annotation_repair_retry_count_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(
            config,
            qwen=replace(
                config.qwen,
                annotation=replace(
                    config.qwen.annotation,
                    repair_retries=-1,
                ),
            ),
        ).validate()
