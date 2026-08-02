from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
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
    EntityVisibilitySummary,
    InstructionLegendEntry,
    InstructionState,
    PairingState,
    RawInstructionOutput,
    ReferencesState,
    render_instruction_text,
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
        sam3=Sam3Config(
            model_path=user_models / "sam3" / "checkpoint.pt"
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
        "sam3:\n"
        f"  model_path: {config.sam3.model_path}\n"
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
        CoverageState(
            passed=True,
            qualifying_entity_ids=["e1"],
            entity_visibility_summary={
                "e1": EntityVisibilitySummary(
                    status="ready",
                    visible_frame_slots=list(range(7)),
                    visible_frame_count=7,
                    coverage_ratio=0.7,
                    qualifies=True,
                    per_frame_area_ratio=[0.1] * 7 + [0.0] * 3,
                    per_frame_confidence=[0.9] * 7 + [None] * 3,
                ),
                "e2": EntityVisibilitySummary(
                    status="ready",
                    visible_frame_slots=list(range(3)),
                    visible_frame_count=3,
                    coverage_ratio=0.3,
                    qualifies=False,
                    per_frame_area_ratio=[0.1] * 3 + [0.0] * 7,
                    per_frame_confidence=[0.9] * 3 + [None] * 7,
                ),
            },
        ),
    )
    references = [
        EntityReferenceState(
            entity_id=entity_id,
            status="ready",
            reference_scope="local",
            visible_region="central",
            whole_entity_recognizable=False,
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
                source_foreground_area_pixels=0,
                source_foreground_area_ratio=0.0,
            ),
        ),
    )
    storage.write_pairing(
        clip_uid,
        PairingState(
            status="ready",
            retained_entity_ids=["e1", "e2"],
            tokens={
                "e1": "<ref_subject_1>",
                "e2": "<ref_object_1>",
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
        "the stable appearance of the woman in a yellow coat",
        "the stable appearance of the red bicycle",
        "the bright plaza environment",
    ]
    return RawInstructionOutput(
        instruction_body_template=body
        or (
            "Use {{image_3}} as the overall background while {{image_1}} "
            "walks forward pushing {{image_2}}, and the camera tracks "
            "steadily backward."
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
    assert [binding.entity_id for binding in bindings] == ["e1", "e2", None]
    assert [binding.reference_type for binding in bindings] == [
        "subject",
        "object",
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


def test_english_body_and_legend_are_accepted() -> None:
    output = _raw_output(
        body=(
            "At dusk—{{image_1}} walks beside {{image_2}} in {{image_3}}, "
            "while the camera tracks backward."
        )
    )

    assert _issue_codes(output) == set()


def test_chinese_body_is_rejected() -> None:
    output = _raw_output(
        body="\u5728{{image_3}}\u4e2d\uff0c{{image_1}}\u63a8\u7740{{image_2}}\u524d\u8fdb\u3002"
    )

    assert "non_english_instruction_text" in _issue_codes(output)


def test_chinese_legend_description_is_rejected() -> None:
    output = _raw_output(
        descriptions=[
            "\u9ec4\u8272\u5916\u5957\u5973\u5b50",
            "the red bicycle",
            "the bright plaza",
        ]
    )

    assert "non_english_legend_description" in _issue_codes(output)


@pytest.mark.parametrize("cjk_text", ["日本語", "한국어"])
def test_other_cjk_scripts_are_rejected(cjk_text: str) -> None:
    body = (
        cjk_text
        + " {{image_1}} stands beside {{image_2}} in {{image_3}}."
    )
    output = _raw_output(body=body)

    assert "non_english_instruction_text" in _issue_codes(output)


def test_repeated_placeholder_is_allowed() -> None:
    output = _raw_output(
        body=(
            "{{image_1}} stands beside {{image_2}}; the camera returns to "
            "{{image_1}} with {{image_3}} in the background."
        )
    )

    assert _issue_codes(output) == set()


@pytest.mark.parametrize(
    ("output", "code"),
    [
        (
            _raw_output(
                body=(
                    "{{image_1}} and {{image_2}} stand in {{image_3}}, while "
                    "{{image_4}} appears."
                )
            ),
            "unknown_image_placeholder",
        ),
        (
            _raw_output(body="{{image_1}} stands in {{image_3}}."),
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
                descriptions=["", "the red bicycle", "the bright plaza"]
            ),
            "empty_legend_description",
        ),
        (
            _raw_output(
                body=(
                    "{{image_1}} and {{image_2}} stand in {{image_3}} beside "
                    "<ref_subject_1>."
                )
            ),
            "reference_token_in_instruction",
        ),
        (
            _raw_output(
                body="\u56fe1 stands beside {{image_2}} in {{image_3}}."
            ),
            "direct_chinese_image_label",
        ),
        (
            _raw_output(
                body=(
                    "<Image 1> stands beside {{image_1}} and {{image_2}} in "
                    "{{image_3}}."
                )
            ),
            "direct_english_image_label",
        ),
        (
            _raw_output(
                body=(
                    "Image 1 stands beside {{image_1}} and {{image_2}} in "
                    "{{image_3}}."
                )
            ),
            "direct_english_image_label",
        ),
        (
            _raw_output(
                body=(
                    "{{ image_1 }} stands beside {{image_2}} in {{image_3}}."
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
            "{{image_2}} tells {{image_1}}, \u201cFollow me.\u201d while "
            "{{image_3}} remains behind them."
        )
    )

    assert "quoted_dialogue_without_transcript" in _issue_codes(output)
    assert _issue_codes(output, transcript="Follow me.") == set()


def test_renderer_uses_angle_bracket_english_labels() -> None:
    legend = [
        InstructionLegendEntry(
            image_id="image_1",
            description="the stable appearance of a red bicycle",
        )
    ]

    rendered = render_instruction_text("Move {{image_1}} forward.", legend)

    assert rendered == (
        "Move <Image 1> forward.\n\n"
        "<Image 1>: the stable appearance of a red bicycle"
    )
    assert "\nImage 1:" not in rendered
    assert "\u56fe1" not in rendered


def test_legacy_plain_english_instruction_state_can_be_loaded() -> None:
    state = InstructionState.model_validate(
        {
            "status": "ready",
            "instruction_body_template": "Move {{image_1}} forward.",
            "reference_legend": [
                {
                    "image_id": "image_1",
                    "description": "a red bicycle",
                }
            ],
            "r2v_instruction": (
                "Move Image 1 forward.\n\nImage 1: a red bicycle"
            ),
        }
    )

    assert state.status == "ready"
    assert state.r2v_instruction.startswith("Move Image 1")


def test_legacy_chinese_instruction_state_can_be_loaded() -> None:
    state = InstructionState.model_validate(
        {
            "status": "ready",
            "instruction_body_template": "\u8ba9{{image_1}}\u5411\u524d\u79fb\u52a8\u3002",
            "reference_legend": [
                {
                    "image_id": "image_1",
                    "description": "\u9ec4\u8272\u5916\u5957\u5973\u5b50",
                }
            ],
            "r2v_instruction": (
                "\u8ba9\u56fe1\u5411\u524d\u79fb\u52a8\u3002\n\n"
                "\u56fe1\uff1a\u9ec4\u8272\u5916\u5957\u5973\u5b50"
            ),
        }
    )

    assert state.status == "ready"
    assert state.r2v_instruction.startswith("\u8ba9\u56fe1")


def test_new_instruction_state_requires_angle_bracket_rendering() -> None:
    body = "Move {{image_1}} forward."
    legend = [
        InstructionLegendEntry(image_id="image_1", description="a red bicycle")
    ]

    state = InstructionState(
        status="ready",
        instruction_body_template=body,
        reference_legend=legend,
        r2v_instruction=render_instruction_text(body, legend),
    )

    assert state.r2v_instruction == (
        "Move <Image 1> forward.\n\n<Image 1>: a red bicycle"
    )
    with pytest.raises(ValidationError, match="angle-bracket English rendering"):
        InstructionState(
            status="ready",
            instruction_body_template=body,
            reference_legend=legend,
            r2v_instruction=(
                "Move <Image 1> forward.\n\nImage 1: a red bicycle"
            ),
        )


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
            "{{image_1}} and {{image_2}} stand in {{image_3}}, while "
            "{{image_4}} appears."
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
        "Use <Image 3> as the overall background while <Image 1> walks "
        "forward pushing <Image 2>, and the camera tracks steadily "
        "backward.\n\n"
        "<Image 1>: the stable appearance of the woman in a yellow coat\n"
        "<Image 2>: the stable appearance of the red bicycle\n"
        "<Image 3>: the bright plaza environment"
    )


def test_instruction_failure_preserves_ready_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, clip_uid = _ready_storage(config)
    before = storage.read_clip(clip_uid).annotation
    invalid = _raw_output(body="{{image_1}} stands in {{image_3}}.")
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


@pytest.mark.parametrize("legacy_style", ["plain_english", "chinese"])
def test_overwrite_replaces_legacy_instruction_with_angle_brackets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_style: str,
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage, clip_uid = _ready_storage(config)
    if legacy_style == "plain_english":
        raw = _raw_output()
        legacy_payload = {
            "status": "ready",
            "instruction_body_template": raw.instruction_body_template,
            "reference_legend": [
                entry.model_dump(mode="json")
                for entry in raw.reference_legend
            ],
            "r2v_instruction": (
                "Use Image 3 as the overall background while Image 1 walks "
                "forward pushing Image 2, and the camera tracks steadily "
                "backward.\n\n"
                "Image 1: the stable appearance of the woman in a yellow coat\n"
                "Image 2: the stable appearance of the red bicycle\n"
                "Image 3: the bright plaza environment"
            ),
        }
    else:
        legacy_payload = {
            "status": "ready",
            "instruction_body_template": (
                "\u4ee5{{image_3}}\u4e3a\u80cc\u666f\uff0c{{image_1}}\u63a8\u7740"
                "{{image_2}}\u524d\u8fdb\u3002"
            ),
            "reference_legend": [
                {"image_id": "image_1", "description": "\u9ec4\u8272\u5916\u5957\u5973\u5b50"},
                {"image_id": "image_2", "description": "\u7ea2\u8272\u81ea\u884c\u8f66"},
                {"image_id": "image_3", "description": "\u660e\u4eae\u5e7f\u573a"},
            ],
            "r2v_instruction": (
                "\u4ee5\u56fe3\u4e3a\u80cc\u666f\uff0c\u56fe1\u63a8\u7740\u56fe2\u524d\u8fdb\u3002\n\n"
                "\u56fe1\uff1a\u9ec4\u8272\u5916\u5957\u5973\u5b50\n"
                "\u56fe2\uff1a\u7ea2\u8272\u81ea\u884c\u8f66\n"
                "\u56fe3\uff1a\u660e\u4eae\u5e7f\u573a"
            ),
        }
    storage.write_instruction(
        clip_uid,
        InstructionState.model_validate(legacy_payload),
    )
    client = _FakeInstructionClient(
        config.qwen.instruction_writer,
        [_raw_output().model_dump(mode="json")],
    )

    stats = instruct_clips(
        config,
        storage,
        overwrite=True,
        client=client,
    )

    instruction = storage.read_clip(clip_uid).instruction
    assert stats.processed == 1
    assert stats.skipped_existing == 0
    assert instruction is not None
    assert instruction.instruction_body_template == (
        _raw_output().instruction_body_template
    )
    assert instruction.r2v_instruction == render_instruction_text(
        instruction.instruction_body_template,
        instruction.reference_legend,
    )
    assert "<Image 1>" in instruction.r2v_instruction
    assert "\nImage 1:" not in instruction.r2v_instruction
    assert "\u56fe1" not in instruction.r2v_instruction


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
