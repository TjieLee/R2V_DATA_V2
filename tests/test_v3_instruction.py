from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.instruction import (
    QwenInstructionClient,
    build_instruction_bindings,
    instruct_clips,
    source_transcript_from_metadata,
    validate_instruction_output,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    PairingState,
    RawInstructionOutput,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3


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
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(v3_config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(v3_config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(v3_config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_root / "source.json",
        run_root=writable / "runs" / "instruction",
        export_root=writable / "datasets" / "instruction-v1",
        source=SourceConfig(limit=5),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(model)),
            instruction_writer=QwenServiceConfig(model=str(model)),
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=(
                user_models / "Qwen-Image-Edit-2511-Object-Remover"
            ),
        ),
    )
    config.validate()
    return config


def _config_path(config: V3Config, tmp_path: Path) -> Path:
    path = tmp_path / "v3-instruction.yaml"
    path.write_text(
        f"dataset_json: {config.dataset_json}\n"
        f"run_root: {config.run_root}\n"
        f"export_root: {config.export_root}\n"
        "source:\n"
        "  limit: 5\n"
        "qwen:\n"
        "  annotation:\n"
        f"    model: {config.qwen.annotation.model}\n"
        "  instruction_writer:\n"
        f"    model: {config.qwen.instruction_writer.model}\n",
        encoding="utf-8",
    )
    return path


def _ready_storage(
    config: V3Config,
    *,
    metadata: dict[str, object] | None = None,
) -> tuple[RunStorage, str]:
    storage = RunStorage(config)
    storage.initialize(git_commit="instruction-test")
    clip_uid = "clip-1"
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(config.dataset_json.parent / "videos" / "clip.mp4"),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata=metadata or {},
        ),
    )
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption=(
                "A woman in a yellow coat walks beside a red bicycle through "
                "a bright plaza as the camera tracks backward."
            ),
            entities=[
                AnnotationEntity(
                    entity_id="e1",
                    reference_type="subject",
                    phrase="a woman in a yellow coat",
                    grounding_prompt="the woman wearing a yellow coat",
                ),
                AnnotationEntity(
                    entity_id="e2",
                    reference_type="object",
                    phrase="a red bicycle",
                    grounding_prompt="the red bicycle beside the woman",
                ),
            ],
            background=BackgroundAnnotation(
                phrase="a bright plaza",
                grounding_prompt="the empty bright plaza",
            ),
        ),
    )
    storage.write_coverage(
        clip_uid,
        CoverageState(passed=True, qualifying_entity_ids=["e1"]),
    )
    references = [
        EntityReferenceState(
            entity_id=entity_id,
            status="ready",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=True,
            identity_features_visible=True,
            scope_reason="clear reference",
            image_path=f"clips/{clip_uid}/selected/{entity_id}.png",
            source_frame_index=index,
        )
        for index, entity_id in enumerate(("e1", "e2"), start=1)
    ]
    storage.write_references(
        clip_uid,
        ReferencesState(
            entities=references,
            background=BackgroundReferenceState(
                status="clean_raw",
                source_image_path=f"clips/{clip_uid}/frames/03.jpg",
                output_image_path=f"clips/{clip_uid}/frames/03.jpg",
                source_frame_slot=3,
                source_frame_index=30,
            ),
        ),
    )
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=["e2", "e1"],
            tokens={
                "e2": "<ref_object_1>",
                "e1": "<ref_subject_1>",
            },
            background_token="<ref_bg_1>",
        ),
    )
    return storage, clip_uid


def _raw_output(
    *,
    body: str | None = None,
    legend_ids: list[str] | None = None,
    descriptions: list[str] | None = None,
) -> RawInstructionOutput:
    ids = legend_ids or ["image_1", "image_2", "image_3"]
    values = descriptions or [
        "\u7ea2\u8272\u81ea\u884c\u8f66\u7684\u5916\u89c2",
        "\u9ec4\u8272\u5916\u5957\u5973\u5b50\u7684\u5916\u89c2",
        "\u660e\u4eae\u7684\u5e7f\u573a\u73af\u5883",
    ]
    return RawInstructionOutput(
        instruction_body_template=body
        or (
            "\u4ee5{{image_3}}\u4f5c\u4e3a\u6574\u4f53\u80cc\u666f\uff0c"
            "{{image_2}}\u63a8\u7740{{image_1}}\u5411\u524d\u884c\u8d70\uff0c"
            "\u955c\u5934\u7a33\u5b9a\u540e\u9000\u3002"
        ),
        reference_legend=[
            {"image_id": image_id, "description": description}
            for image_id, description in zip(ids, values)
        ],
    )


class _FakeInstructionClient(QwenInstructionClient):
    def __init__(
        self,
        config: QwenServiceConfig,
        responses: list[dict[str, object] | str | Exception],
        *,
        repair_retries: int = 1,
    ) -> None:
        self.config = config
        self.repair_retries = repair_retries
        self._responses: Iterator[
            dict[str, object] | str | Exception
        ] = iter(responses)
        self.requests: list[str] = []

    def _request(self, request_text: str) -> str:
        self.requests.append(request_text)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)


def _issue_codes(
    output: RawInstructionOutput,
    *,
    transcript: str | None = None,
) -> set[str]:
    bindings = [
        {
            "image_id": f"image_{index}",
            "image_index": index,
            "reference_type": reference_type,
            "entity_id": None if index == 3 else f"e{index}",
            "phrase": "reference",
            "grounding_prompt": "visible reference",
        }
        for index, reference_type in enumerate(
            ("subject", "object", "background"),
            start=1,
        )
    ]
    from r2v_data_v2.v3.schemas import InstructionBinding

    issues = validate_instruction_output(
        output,
        t2v_caption="An English source caption.",
        bindings=[
            InstructionBinding.model_validate(binding)
            for binding in bindings
        ],
        source_transcript=transcript,
    )
    return {issue.code for issue in issues}


def test_bindings_follow_pairing_order_and_put_background_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, clip_uid = _ready_storage(_config(tmp_path, monkeypatch))

    bindings = build_instruction_bindings(storage.read_clip(clip_uid))

    assert [binding.image_id for binding in bindings] == [
        "image_1",
        "image_2",
        "image_3",
    ]
    assert [binding.image_index for binding in bindings] == [1, 2, 3]
    assert [binding.entity_id for binding in bindings] == ["e2", "e1", None]
    assert [binding.reference_type for binding in bindings] == [
        "object",
        "subject",
        "background",
    ]


def test_instruction_output_schema_uses_english_identifiers_only() -> None:
    schema = RawInstructionOutput.model_json_schema()

    assert set(schema["properties"]) == {
        "instruction_body_template",
        "reference_legend",
    }
    legend_schema = schema["$defs"]["RawInstructionLegend"]
    assert set(legend_schema["properties"]) == {
        "image_id",
        "description",
    }
    serialized = json.dumps(schema, ensure_ascii=False)
    assert "display_id" not in serialized
    assert "\u56fe1" not in serialized


def test_repeated_placeholder_is_allowed() -> None:
    output = _raw_output(
        body=(
            "{{image_1}}\u4f4d\u4e8e{{image_2}}\u65c1\u8fb9\uff0c"
            "\u955c\u5934\u518d\u6b21\u5bf9\u51c6{{image_1}}\uff0c"
            "\u5e76\u4ee5{{image_3}}\u4e3a\u80cc\u666f\u3002"
        )
    )

    assert _issue_codes(output) == set()


@pytest.mark.parametrize(
    ("output", "code"),
    [
        (
            _raw_output(
                body=(
                    "{{image_1}}\u548c{{image_2}}\u4f4d\u4e8e"
                    "{{image_3}}\u4e2d\uff0c{{image_4}}\u51fa\u73b0\u3002"
                )
            ),
            "unknown_image_placeholder",
        ),
        (
            _raw_output(
                body=(
                    "{{image_1}}\u4f4d\u4e8e{{image_3}}\u4e2d\u3002"
                )
            ),
            "missing_image_placeholder",
        ),
        (
            _raw_output(legend_ids=["image_1", "image_2"]),
            "legend_count_mismatch",
        ),
        (
            _raw_output(
                legend_ids=["image_2", "image_1", "image_3"],
            ),
            "legend_order_mismatch",
        ),
        (
            _raw_output(
                descriptions=[
                    "",
                    "\u9ec4\u8272\u5916\u5957\u5973\u5b50",
                    "\u5e7f\u573a",
                ]
            ),
            "empty_legend_description",
        ),
        (
            _raw_output(
                body=(
                    "{{image_1}}\u548c{{image_2}}\u4f4d\u4e8e"
                    "{{image_3}}\u4e2d\uff0c\u4f7f\u7528"
                    "<ref_subject_1>\u3002"
                )
            ),
            "reference_token_in_instruction",
        ),
        (
            _raw_output(
                body=(
                    "\u56fe1\u548c{{image_2}}\u4f4d\u4e8e"
                    "{{image_3}}\u4e2d\u3002"
                )
            ),
            "direct_chinese_image_label",
        ),
        (
            _raw_output(
                body=(
                    "{{ image_1 }}\u548c{{image_2}}\u4f4d\u4e8e"
                    "{{image_3}}\u4e2d\u3002"
                )
            ),
            "invalid_image_placeholder",
        ),
    ],
)
def test_instruction_validation_rejects_invalid_structured_output(
    output: RawInstructionOutput,
    code: str,
) -> None:
    assert code in _issue_codes(output)


def test_quoted_dialogue_requires_source_transcript() -> None:
    output = _raw_output(
        body=(
            "{{image_2}}\u5bf9{{image_1}}\u8bf4\uff1a\u300c\u8ddf\u6211\u6765\u3002\u300d"
            "\uff0c{{image_3}}\u4f5c\u4e3a\u80cc\u666f\u3002"
        )
    )

    assert "quoted_dialogue_without_transcript" in _issue_codes(output)
    assert _issue_codes(output, transcript="\u8ddf\u6211\u6765\u3002") == set()


def test_source_transcript_reads_only_explicit_metadata_fields() -> None:
    assert source_transcript_from_metadata(
        {"title": "not dialogue", "transcript": "  hello  "}
    ) == "hello"
    assert source_transcript_from_metadata({"title": "not dialogue"}) is None


def test_unknown_placeholder_can_be_repaired_and_rendered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, clip_uid = _ready_storage(config)
    invalid = _raw_output(
        body=(
            "{{image_1}}\u548c{{image_2}}\u4f4d\u4e8e"
            "{{image_3}}\u4e2d\uff0c{{image_4}}\u51fa\u73b0\u3002"
        )
    )
    valid = _raw_output()
    client = _FakeInstructionClient(
        config.qwen.instruction_writer,
        [
            invalid.model_dump(mode="json"),
            valid.model_dump(mode="json"),
        ],
    )

    stats = instruct_clips(config, storage, client=client)

    instruction = storage.read_clip(clip_uid).instruction
    assert stats.processed == 1
    assert stats.repaired == 1
    assert len(client.requests) == 2
    assert "unknown_image_placeholder" in client.requests[1]
    assert instruction is not None
    assert instruction.reference_legend[0].image_id == "image_1"
    assert instruction.r2v_instruction == (
        "\u4ee5\u56fe3\u4f5c\u4e3a\u6574\u4f53\u80cc\u666f\uff0c"
        "\u56fe2\u63a8\u7740\u56fe1\u5411\u524d\u884c\u8d70\uff0c"
        "\u955c\u5934\u7a33\u5b9a\u540e\u9000\u3002\n\n"
        "\u56fe1\uff1a\u7ea2\u8272\u81ea\u884c\u8f66\u7684\u5916\u89c2\n"
        "\u56fe2\uff1a\u9ec4\u8272\u5916\u5957\u5973\u5b50\u7684\u5916\u89c2\n"
        "\u56fe3\uff1a\u660e\u4eae\u7684\u5e7f\u573a\u73af\u5883"
    )


def test_instruction_failure_preserves_ready_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, clip_uid = _ready_storage(config)
    before = storage.read_clip(clip_uid).annotation
    invalid = _raw_output(
        body=(
            "{{image_1}}\u4f4d\u4e8e{{image_3}}\u4e2d\u3002"
        )
    )
    client = _FakeInstructionClient(
        config.qwen.instruction_writer,
        [invalid.model_dump(mode="json")],
        repair_retries=0,
    )

    stats = instruct_clips(config, storage, client=client)

    clip = storage.read_clip(clip_uid)
    assert stats.failed == 1
    assert clip.annotation == before
    assert clip.annotation is not None
    assert clip.annotation.status == "ready"
    assert clip.instruction is not None
    assert clip.instruction.status == "failed"
    assert clip.instruction.reason == "missing_image_placeholder"


def test_instruction_input_uses_english_ids_and_explicit_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, _ = _ready_storage(
        config,
        metadata={"transcript": "\u8ddf\u6211\u6765\u3002"},
    )
    client = _FakeInstructionClient(
        config.qwen.instruction_writer,
        [_raw_output().model_dump(mode="json")],
    )

    instruct_clips(config, storage, client=client)

    request = client.requests[0]
    assert '"image_id": "image_1"' in request
    assert '"image_index": 1' in request
    assert '"reference_type": "background"' in request
    assert '"entity_id": null' in request
    assert '"source_transcript": "\u8ddf\u6211\u6765\u3002"' in request
    assert "\u56fe1" not in request


def test_pipeline_runs_instruct_stage_with_fake_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, _ = _ready_storage(config)
    client = _FakeInstructionClient(
        config.qwen.instruction_writer,
        [_raw_output().model_dump(mode="json")],
    )

    result = run_pipeline_v3(
        config_path=_config_path(config, tmp_path),
        stages=("instruct",),
        git_commit="instruction-test",
        instruction_client=client,
    )

    assert result["completed_stages"] == ["instruct"]
    assert result["instruct"]["processed"] == 1
    assert storage.read_clip("clip-1").instruction is not None
