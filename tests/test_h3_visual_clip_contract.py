from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.visual_clip_contract import (
    VisualClipRecord,
    load_visual_clip_record,
)


def _clip_payload() -> dict[str, object]:
    return {
        "schema_version": "r2v.v3.clip.2",
        "clip_uid": "clip-1",
        "source": {
            "video_path": "/public/processed/clip-1.mp4",
            "parent_video_id": "parent",
            "clip_suffix": "1",
            "source_index": 0,
            "caption_raw": "",
            "metadata": {
                "source_relative_video_path": "01/show/season/clip-1.mp4",
                "source_relative_source_video_path": "01/show/season/episode.mkv",
                "visual_internal_evidence": {"version": 3},
            },
        },
        "annotation": {
            "status": "ready",
            "instruction_template": "{{entity_1}} and {{entity_2}}",
            "entities": [
                {
                    "entity_id": "e1",
                    "reference_type": "subject",
                    "phrase": "a person",
                    "grounding_prompt": "a person near the door",
                    "visual_internal_label": "primary",
                },
                {
                    "entity_id": "e2",
                    "reference_type": "object",
                    "phrase": "a suitcase",
                    "grounding_prompt": "a suitcase beside the person",
                },
            ],
        },
        "coverage": {
            "passed": True,
            "qualifying_entity_ids": ["e1"],
            "entity_visibility_summary": {"e1": {"future": "shape"}},
        },
        "references": {
            "entities": [
                {
                    "entity_id": "e1",
                    "status": "ready",
                    "image_path": "clips/clip-1/selected/e1.png",
                    "reference_scope": "full",
                    "visible_region": "whole",
                },
                {
                    "entity_id": "e2",
                    "status": "ready",
                    "image_path": "clips/clip-1/selected/e2.png",
                    "reference_scope": "full",
                    "visible_region": "whole",
                },
            ],
            "background": {"status": "clean_raw"},
        },
        "pairing": {
            "status": "ready",
            "retained_entity_ids": ["e1", "e2"],
            "tokens": {"e1": "<ref_subject_1>", "e2": "<ref_object_1>"},
        },
        "reference_edit": {
            "status": "ready",
            "entities": [
                {
                    "entity_id": "e1",
                    "accepted_base_image_path": None,
                    "future_visual_field": [1, 2, 3],
                }
            ],
        },
        "subject_attributes": {"records": [{"review": {"new_check": True}}]},
        "diagnostics": {"visual_only": True},
    }


def test_visual_clip_projection_ignores_visual_internal_sections() -> None:
    clip = VisualClipRecord.model_validate(_clip_payload())

    assert clip.clip_uid == "clip-1"
    assert clip.pairing is not None
    assert clip.pairing.retained_entity_ids == ["e1", "e2"]
    assert [reference.image_path for reference in clip.references.entities] == [
        "clips/clip-1/selected/e1.png",
        "clips/clip-1/selected/e2.png",
    ]
    assert "reference_edit" not in type(clip).model_fields
    assert "subject_attributes" not in type(clip).model_fields
    assert "diagnostics" not in type(clip).model_fields


@pytest.mark.parametrize(
    "path",
    [
        ("source", "video_path"),
        ("source", "metadata", "source_relative_video_path"),
        ("source", "metadata", "source_relative_source_video_path"),
    ],
)
def test_visual_clip_projection_requires_downstream_source_fields(
    path: tuple[str, ...],
) -> None:
    payload = _clip_payload()
    node: dict[str, object] = payload
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    del node[path[-1]]

    with pytest.raises(ValidationError):
        VisualClipRecord.model_validate(payload)


def test_visual_clip_loader_rejects_expected_uid_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "different" / "clip.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_clip_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match expected clip_uid"):
        load_visual_clip_record(path, expected_clip_uid="different")


def test_visual_clip_projection_rejects_duplicate_annotation_entity() -> None:
    payload = _clip_payload()
    annotation = payload["annotation"]
    assert isinstance(annotation, dict)
    entities = annotation["entities"]
    assert isinstance(entities, list)
    entities.append(copy.deepcopy(entities[0]))

    with pytest.raises(ValidationError, match="entity IDs must be unique"):
        VisualClipRecord.model_validate(payload)


def test_visual_clip_projection_rejects_unknown_retained_entity() -> None:
    payload = _clip_payload()
    pairing = payload["pairing"]
    assert isinstance(pairing, dict)
    pairing["retained_entity_ids"] = ["e1", "e9"]

    with pytest.raises(ValidationError, match="unknown annotation entities"):
        VisualClipRecord.model_validate(payload)


def test_visual_clip_projection_rejects_duplicate_reference() -> None:
    payload = _clip_payload()
    references = payload["references"]
    assert isinstance(references, dict)
    entities = references["entities"]
    assert isinstance(entities, list)
    entities.append(copy.deepcopy(entities[0]))

    with pytest.raises(ValidationError, match="reference IDs must be unique"):
        VisualClipRecord.model_validate(payload)


def test_visual_clip_projection_requires_ready_retained_reference() -> None:
    payload = _clip_payload()
    references = payload["references"]
    assert isinstance(references, dict)
    entities = references["entities"]
    assert isinstance(entities, list)
    second = entities[1]
    assert isinstance(second, dict)
    second["status"] = "rejected"
    second["image_path"] = None

    with pytest.raises(ValidationError, match="matching ready references"):
        VisualClipRecord.model_validate(payload)
