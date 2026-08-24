from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.run_h3_jea_production as jea_cli
from r2v_data_v2.h3.jea_audio_production import (
    JEAOccurrenceEmbedding,
    audio_binding_path,
    build_jea_pairs,
    full_audio_path,
    primary_voice_path,
)
from r2v_data_v2.h3.qwen3_asr import (
    Qwen3ASRBackend,
    Qwen3ASRConfiguration,
    Qwen3ASRSegment,
)
from r2v_data_v2.h3.visual_production_source import (
    ReadableClipIdentity,
    derive_readable_clip_identity,
    load_visual_production_inventory,
)
from r2v_data_v2.v3.subject_attributes import (
    EnrichedSample,
    SubjectAttributeRecord,
)


class _StageResult:
    def __init__(self, **values: object) -> None:
        self.values = values

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.values)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _attribute_record(
    *,
    image_path: str,
    default_variant: str | None = None,
    final_selection: str = "raw",
) -> SubjectAttributeRecord:
    completed = final_selection == "completed"
    payload: dict[str, object] = {
        "attribute_id": "a1",
        "owner_entity_id": "e1",
        "attribute_type": "hair",
        "phrase": "dark hair",
        "grounding_prompt": "the person's dark hair",
        "status": "accepted",
        "image_path": image_path,
        "source_frame_index": 6,
        "source_frame_slot": 6,
        "owner_candidate_id": "candidate_1",
        "same_frame_as_owner_reference": False,
        "sam3_prompt": "dark hair",
        "ownership_geometry": {
            "passed": True,
            "reason": "passed",
            "owner_overlap_ratio": 1.0,
            "maximum_other_owner_overlap_ratio": 0.0,
            "attribute_to_owner_area_ratio": 0.2,
            "near_owner_region": True,
            "attribute_area_pixels": 100,
            "attribute_long_side_pixels": 20,
            "significant_component_count": 1,
            "largest_component_ratio": 1.0,
            "second_largest_component_ratio": 0.0,
        },
        "review": {
            "attribute_id": "a1",
            "matches_attribute": True,
            "owner_binding_correct": True,
            "recognizable": True,
            "characteristic_appearance_visible": True,
            "usable_as_attribute_condition": True,
            "structure_complete": True,
            "completion_recommended": False,
            "reason": "accepted",
        },
        "final_selection": final_selection,
        "completion_attempted": completed,
        "completion_outcome": (
            "selected_completed" if completed else "not_attempted"
        ),
        "reason": "accepted",
    }
    if completed:
        payload["completion_seed"] = 17
        payload["completion_review"] = {
            "verdict": "accept",
            "same_physical_entity": True,
            "identity_preserved": True,
            "original_visible_attributes_preserved": True,
            "exactly_one_entity": True,
            "missing_parts_plausibly_completed": True,
            "no_duplicate_entity": True,
            "no_unrelated_entity": True,
            "no_severe_structure_artifact": True,
            "style_coherent": True,
            "resolution_usable": True,
            "reference_usable": True,
            "certain": True,
            "reason": "accepted",
        }
    if default_variant is not None:
        alpha_path = (
            image_path
            if default_variant == "alpha"
            else "references/ordinary/hair-alpha.png"
        )
        bbox_path = (
            image_path
            if default_variant == "bbox"
            else "references/ordinary/hair-bbox.png"
        )
        generated_path = (
            image_path
            if default_variant == "generated_background"
            else "references/ordinary/hair-generated.png"
        )

        def variant(
            *,
            name: str,
            path: str,
            synthetic: bool,
        ) -> dict[str, object]:
            selected = default_variant == name
            return {
                "image_path": path,
                "status": "accepted" if selected else "available",
                "reviewed": selected,
                "review_status": "accepted" if selected else "not_reviewed",
                "reason": "selected default" if selected else "available",
                "synthetic": synthetic,
                "source_frame_index": 6,
            }

        payload.update(
            {
                "variants": {
                    "alpha": variant(
                        name="alpha",
                        path=alpha_path,
                        synthetic=False,
                    ),
                    "bbox": variant(
                        name="bbox",
                        path=bbox_path,
                        synthetic=False,
                    ),
                    "generated_background": variant(
                        name="generated_background",
                        path=generated_path,
                        synthetic=True,
                    ),
                },
                "default_variant": default_variant,
                "default_image_path": image_path,
                "default_reason": "Visual-selected default",
                "accepted_base_image_path": (
                    image_path
                    if default_variant in {"alpha", "accepted_base"}
                    else "references/ordinary/hair-accepted-base.png"
                ),
            }
        )
    return SubjectAttributeRecord.model_validate(payload)


def _sample(
    tmp_path: Path,
    *,
    clip_uid: str,
    shard_id: str,
    clip_relative_path: str,
    source_relative_path: str,
    parent_video_id: str = "legacy-parent",
    with_attribute: bool = False,
    with_latest_reference_fields: bool = False,
) -> tuple[dict[str, object], Path, Path]:
    production = tmp_path / "visual-production"
    runs = tmp_path / "visual-runs"
    target = tmp_path / "processed" / f"{clip_uid}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"processed-mp4")
    subject_path = production / "references" / clip_uid / "subject.png"
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    subject_path.write_bytes(b"subject")
    references: list[dict[str, object]] = [
        {
            "image_id": "image_1",
            "image_index": 1,
            "kind": "subject",
            "entity_id": "entity_1",
            "image_path": subject_path.relative_to(production).as_posix(),
            "source_frame_index": 0,
            "scope": "full",
            "synthetic": False,
        }
    ]
    if with_attribute:
        attribute_path = production / "references" / clip_uid / "hair.png"
        attribute_path.write_bytes(b"hair")
        references.append(
            {
                "image_id": "image_2",
                "image_index": 2,
                "kind": "attribute",
                "attribute_id": "attribute_1",
                "owner_entity_id": "entity_1",
                "attribute_type": "hair",
                "image_path": attribute_path.relative_to(production).as_posix(),
                "source_frame_index": 0,
                "synthetic": False,
            }
        )
    sample = {
        "schema_version": "r2v.v3.production_sample.1",
        "sample_id": clip_uid,
        "clip_uid": clip_uid,
        "target_video": str(target),
        "t2v_caption": f"caption for {clip_uid}",
        "r2v_instruction": "Use "
        + " and ".join(f"Image {index}" for index in range(1, len(references) + 1)),
        "references": references,
        "source": {
            "parent_video_id": parent_video_id,
            "clip_suffix": clip_uid,
            "shard_id": shard_id,
        },
    }
    clip = {
        "schema_version": "r2v.v3.clip.2",
        "clip_uid": clip_uid,
        "source": {
            "video_path": str(target),
            "parent_video_id": parent_video_id,
            "clip_suffix": clip_uid,
            "source_index": 0,
            "caption_raw": "",
            "metadata": {
                "source_relative_video_path": clip_relative_path,
                "source_relative_source_video_path": source_relative_path,
            },
        },
    }
    if with_latest_reference_fields:
        references[0]["entity_id"] = "e1"
        selected_path = f"clips/{clip_uid}/selected/subject.png"
        ready_reference = {
            "entity_id": "e1",
            "status": "ready",
            "reference_scope": "full",
            "visible_region": "whole",
            "whole_entity_recognizable": True,
            "identity_features_visible": True,
            "scope_reason": "complete subject",
            "image_path": selected_path,
            "source_frame_index": 0,
            "synthetic": False,
        }
        clip.update(
            {
                "annotation": {
                    "status": "ready",
                    "t2v_caption": "A person stands in view.",
                    "entities": [
                        {
                            "entity_id": "e1",
                            "reference_type": "subject",
                            "phrase": "A person",
                            "grounding_prompt": "a person",
                        }
                    ],
                },
                "coverage": {
                    "passed": True,
                    "qualifying_entity_ids": ["e1"],
                    "required_visible_frames": 7,
                    "entity_visibility_summary": {
                        "e1": {
                            "status": "ready",
                            "visible_frame_slots": list(range(7)),
                            "visible_frame_count": 7,
                            "coverage_ratio": 0.7,
                            "qualifies": True,
                            "per_frame_area_ratio": [0.1] * 7 + [0.0] * 3,
                            "per_frame_confidence": [0.9] * 7 + [None] * 3,
                        }
                    },
                },
                "references": {
                    "entities": [ready_reference],
                    "background": None,
                },
                "pairing": {
                    "status": "ready",
                    "retained_entity_ids": ["e1"],
                    "tokens": {"e1": "<ref_subject_1>"},
                    "background_token": None,
                },
                "reference_edit": {
                    "status": "ready",
                    "entities": [
                        {
                            "entity_id": "e1",
                            "route": "complete",
                            "status": "not_required",
                            "source_reference": ready_reference,
                            "source_image_path": selected_path,
                            "variants": {
                                "alpha": {
                                    "image_path": selected_path,
                                    "status": "accepted",
                                    "reviewed": True,
                                    "review_status": "accepted",
                                    "reason": "raw accepted",
                                    "synthetic": False,
                                    "source_frame_index": 0,
                                },
                                "bbox": {
                                    "image_path": (
                                        f"clips/{clip_uid}/selected/subject-bbox.png"
                                    ),
                                    "status": "available",
                                    "reviewed": False,
                                    "review_status": "not_reviewed",
                                    "reason": "available",
                                    "synthetic": False,
                                    "source_frame_index": 0,
                                },
                                "generated_background": {
                                    "status": "unavailable",
                                    "reviewed": False,
                                    "review_status": "not_generated",
                                    "reason": "not generated",
                                    "synthetic": True,
                                    "source_frame_index": 0,
                                },
                            },
                            "default_variant": "alpha",
                            "default_image_path": selected_path,
                            "default_reason": "raw reference accepted",
                            "output_image_path": selected_path,
                        }
                    ],
                },
            }
        )
    clip_path = runs / shard_id / "clips" / clip_uid / "clip.json"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
    return sample, production, runs


def _inventory(
    tmp_path: Path,
    specs: list[dict[str, object]],
) -> object:
    rows = []
    production = runs = None
    for spec in specs:
        row, production, runs = _sample(tmp_path, **spec)
        rows.append(row)
    assert production is not None and runs is not None
    _jsonl(production / "samples.jsonl", rows)
    return load_visual_production_inventory(
        visual_production_root=production,
        visual_runs_root=runs,
    )


def _dataset_inventory(
    tmp_path: Path,
    *,
    with_enriched: bool = False,
    attribute_default_variant: str | None = None,
    attribute_final_selection: str = "raw",
    legacy_enriched: bool = False,
) -> object:
    export_root = tmp_path / "visual-export"
    run_root = tmp_path / "single-visual-run"
    target = tmp_path / "processed" / "ordinary.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"processed-mp4")
    references = []
    reference_specs = [
        ("subject", "<ref_subject_1>", "entity", "e1", "full", "whole"),
        ("object", "<ref_object_1>", "entity", "e2", "full", "whole"),
        ("group", "<ref_group_1>", "entity", "e3", "local", "central"),
        ("background", "<ref_bg_1>", "background", None, "scene", "whole"),
    ]
    for index, (kind, token, ref_type, entity_id, scope, visible_region) in enumerate(
        reference_specs, start=1
    ):
        image_path = f"references/ordinary/{kind}.png"
        artifact = export_root / image_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(kind.encode())
        references.append(
            {
                "token": token,
                "type": ref_type,
                "entity_id": entity_id,
                "scope": scope,
                "visible_region": visible_region,
                "image_path": image_path,
                "source_frame_index": index,
                "synthetic": False,
            }
        )
    sample = {
        "schema_version": "r2v.v3.sample.1",
        "sample_id": "ordinary",
        "target_video": str(target),
        "t2v_caption": "A person stands beside an object and a group.",
        "r2v_instruction": "Use "
        + ", ".join(f"<Image {index}>" for index in range(1, 5)),
        "references": references,
        "source": {"parent_video_id": "parent", "clip_suffix": "00216"},
    }
    _jsonl(export_root / "samples.jsonl", [sample])
    clip = {
        "schema_version": "r2v.v3.clip.2",
        "clip_uid": "ordinary",
        "source": {
            "video_path": str(target),
            "parent_video_id": "parent",
            "clip_suffix": "00216",
            "source_index": 0,
            "caption_raw": "",
            "metadata": {
                "source_relative_video_path": (
                    "01/法证先锋/法证先锋1/法证先锋1_22/clip_00216.mp4"
                ),
                "source_relative_source_video_path": (
                    "01/法证先锋/法证先锋1/法证先锋1_22/source.mp4"
                ),
            },
        },
    }
    clip_path = run_root / "clips" / "ordinary" / "clip.json"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_text(json.dumps(clip, ensure_ascii=False), encoding="utf-8")
    if with_enriched:
        for kind in ("subject", "object", "group", "background"):
            visual_reference = (
                run_root / f"clips/ordinary/selected/{kind}.png"
            )
            visual_reference.parent.mkdir(parents=True, exist_ok=True)
            visual_reference.write_bytes(f"enriched-{kind}".encode())
        attribute_path = "references/ordinary/hair.png"
        attribute = run_root / "subject_attributes" / attribute_path
        attribute.parent.mkdir(parents=True)
        attribute.write_bytes(b"hair")
        record = _attribute_record(
            image_path=attribute_path,
            default_variant=attribute_default_variant,
            final_selection=attribute_final_selection,
        )
        enriched = EnrichedSample(
            sample_id="ordinary",
            clip_uid="ordinary",
            source_run_root=str(run_root),
            original_visual={
                "target_video": str(target),
                "source": sample["source"],
            },
            original_instruction=sample["r2v_instruction"],
            enriched_instruction=(
                "Use <Image 1> with <Image 2>, <Image 3>, <Image 4>, and "
                "<Image 5>."
            ),
            references=[
                {
                    "image_id": "image_1",
                    "image_index": 1,
                    "kind": "subject",
                    "origin": "visual_run",
                    "entity_id": "e1",
                    "image_path": "clips/ordinary/selected/subject.png",
                    "source_frame_index": 1,
                },
                {
                    "image_id": "image_2",
                    "image_index": 2,
                    "kind": "attribute",
                    "origin": "attribute_enrichment",
                    "attribute_id": "a1",
                    "owner_entity_id": "e1",
                    "attribute_type": "hair",
                    "image_path": attribute_path,
                    "source_frame_index": 6,
                },
                {
                    "image_id": "image_3",
                    "image_index": 3,
                    "kind": "object",
                    "origin": "visual_run",
                    "entity_id": "e2",
                    "image_path": "clips/ordinary/selected/object.png",
                    "source_frame_index": 2,
                },
                {
                    "image_id": "image_4",
                    "image_index": 4,
                    "kind": "group",
                    "origin": "visual_run",
                    "entity_id": "e3",
                    "image_path": "clips/ordinary/selected/group.png",
                    "source_frame_index": 3,
                },
                {
                    "image_id": "image_5",
                    "image_index": 5,
                    "kind": "background",
                    "origin": "visual_run",
                    "image_path": "clips/ordinary/selected/background.png",
                    "source_frame_index": 4,
                },
            ],
            accepted_attributes=[record],
        )
        enriched_payload = enriched.model_dump(mode="json")
        if legacy_enriched:
            enriched_payload["references"][1].pop("attribute_type")
            for field in (
                "completion_seed",
                "variants",
                "default_variant",
                "default_image_path",
                "default_reason",
                "accepted_base_image_path",
            ):
                enriched_payload["accepted_attributes"][0].pop(field)
        _jsonl(
            run_root / "subject_attributes" / "enriched_samples.jsonl",
            [enriched_payload],
        )
    return load_visual_production_inventory(
        visual_production_root=export_root,
        visual_runs_root=run_root,
    )


def test_canonical_source_loads_multiple_shards(tmp_path: Path) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "clip-a",
                "shard_id": "shard-a",
                "clip_relative_path": "节目/集合/ep-a_0.mp4",
                "source_relative_path": "节目/集合/ep-a.mkv",
            },
            {
                "clip_uid": "clip-b",
                "shard_id": "shard-b",
                "clip_relative_path": "节目/集合/ep-b_0.mp4",
                "source_relative_path": "节目/集合/ep-b.mkv",
            },
        ],
    )
    assert inventory.canonical_sample_count == 2
    assert inventory.shard_count == 2
    assert inventory.visual_input_schema == "r2v.v3.production_sample.1"
    assert inventory.visual_input_mode == "compacted_production"
    assert [item.identity.clip_uid for item in inventory.clips] == ["clip-a", "clip-b"]


def test_audio_loader_reads_latest_variant_aware_clip_record(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "clip-latest",
                "shard_id": "shard-latest",
                "clip_relative_path": "节目/集合/ep-latest_0.mp4",
                "source_relative_path": "节目/集合/ep-latest.mkv",
                "with_latest_reference_fields": True,
            }
        ],
    )

    reference_edit = inventory.clips[0].clip.reference_edit
    assert reference_edit is not None
    latest = reference_edit.entities[0]
    assert latest.default_variant == "alpha"
    assert latest.default_image_path == latest.source_image_path
    assert latest.variants is not None
    assert latest.variants.alpha.status == "accepted"
    assert latest.variants.bbox.status == "available"


def test_dataset_sample_single_run_normalizes_references_and_identity(
    tmp_path: Path,
) -> None:
    inventory = _dataset_inventory(tmp_path)

    assert inventory.visual_input_schema == "r2v.v3.sample.1"
    assert inventory.visual_input_mode == "single_run_export"
    assert inventory.shard_count == 1
    clip = inventory.clips[0]
    assert clip.identity.clip_uid == "ordinary"
    assert clip.sample.clip_uid == "ordinary"
    assert clip.identity.shard_id == "single-visual-run"
    assert Path(clip.clip_record_path) == (
        tmp_path / "single-visual-run/clips/ordinary/clip.json"
    )
    assert [reference.kind for reference in clip.sample.references] == [
        "subject",
        "object",
        "group",
        "background",
    ]
    assert [reference.kind for reference in clip.subject_references] == ["subject"]
    assert clip.sample.r2v_instruction.startswith("Use <Image 1>")
    assert clip.identity.media_collection_relpath == "01/法证先锋"
    assert clip.identity.clip_display_path.endswith("clip_00216")
    assert all(Path(item.artifact_path).is_file() for item in clip.sample.references)


def test_dataset_sample_enriched_sidecar_preserves_attribute_provenance(
    tmp_path: Path,
) -> None:
    inventory = _dataset_inventory(tmp_path, with_enriched=True)
    clip = inventory.clips[0]

    assert clip.sample.r2v_instruction.startswith("Use <Image 1> with <Image 2>")
    assert [reference.kind for reference in clip.sample.references] == [
        "subject",
        "attribute",
        "object",
        "group",
        "background",
    ]
    attribute = clip.sample.references[1]
    assert attribute.attribute_id == "a1"
    assert attribute.owner_entity_id == "e1"
    assert attribute.attribute_type == "hair"
    assert attribute.source_frame_index == 6
    assert attribute.image_path == "references/ordinary/hair.png"
    assert Path(attribute.artifact_path).read_bytes() == b"hair"
    assert [reference.entity_id for reference in clip.subject_references] == ["e1"]


def test_legacy_enriched_attribute_without_variant_fields_still_loads(
    tmp_path: Path,
) -> None:
    inventory = _dataset_inventory(
        tmp_path,
        with_enriched=True,
        attribute_final_selection="completed",
        legacy_enriched=True,
    )
    sidecar = json.loads(
        (
            tmp_path
            / "single-visual-run/subject_attributes/enriched_samples.jsonl"
        ).read_text(encoding="utf-8")
    )
    legacy_record = sidecar["accepted_attributes"][0]
    assert "variants" not in legacy_record
    assert "default_variant" not in legacy_record
    assert "default_image_path" not in legacy_record

    attribute = inventory.clips[0].sample.references[1]
    assert attribute.image_path == legacy_record["image_path"]
    assert attribute.synthetic is True


@pytest.mark.parametrize(
    ("default_variant", "final_selection", "expected_synthetic"),
    [
        ("generated_background", "raw", True),
        ("bbox", "raw", False),
        ("accepted_base", "completed", True),
        ("alpha", "raw", False),
        ("accepted_base", "raw", False),
    ],
)
def test_variant_aware_attribute_uses_visual_selected_image_and_synthetic_rule(
    tmp_path: Path,
    default_variant: str,
    final_selection: str,
    expected_synthetic: bool,
) -> None:
    inventory = _dataset_inventory(
        tmp_path,
        with_enriched=True,
        attribute_default_variant=default_variant,
        attribute_final_selection=final_selection,
    )

    attribute = inventory.clips[0].sample.references[1]
    assert attribute.image_path == "references/ordinary/hair.png"
    assert Path(attribute.artifact_path).read_bytes() == b"hair"
    assert attribute.synthetic is expected_synthetic
    assert "default_variant" not in attribute.model_dump()
    assert "variants" not in attribute.model_dump()


def test_dataset_sample_dry_run_reports_detected_input_layout(tmp_path: Path) -> None:
    inventory = _dataset_inventory(tmp_path)
    result = jea_cli.main(
        [
            "--visual-production-root",
            inventory.visual_production_root,
            "--visual-runs-root",
            inventory.visual_runs_root,
            "--audio-production-root",
            str(tmp_path / "audio-production"),
            "--dry-run",
        ]
    )

    assert result["visual_input_schema"] == "r2v.v3.sample.1"
    assert result["visual_input_mode"] == "single_run_export"


def test_canonical_samples_jsonl_is_the_allowlist(tmp_path: Path) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "included",
                "shard_id": "shard-a",
                "clip_relative_path": "collection/included_0.mp4",
                "source_relative_path": "collection/included.mkv",
            }
        ],
    )
    extra = Path(inventory.visual_runs_root) / "shard-a/clips/not-listed/clip.json"
    extra.parent.mkdir(parents=True)
    extra.write_text("{}", encoding="utf-8")
    reloaded = load_visual_production_inventory(
        visual_production_root=Path(inventory.visual_production_root),
        visual_runs_root=Path(inventory.visual_runs_root),
    )
    assert [item.identity.clip_uid for item in reloaded.clips] == ["included"]


def test_readable_identity_derivation_preserves_unicode_and_spaces() -> None:
    identity = derive_readable_clip_identity(
        clip_uid="opaque",
        shard_id="shard-1",
        source_relative_video_path="栏目 A/电影 集/第一集_0003.mp4",
        source_relative_source_video_path="栏目 A/电影 集/第一集.mkv",
    )
    assert identity.media_collection_relpath == "栏目 A/电影 集"
    assert identity.media_collection_name == "电影 集"
    assert identity.episode_name == "第一集"
    assert identity.clip_name == "第一集_0003"
    assert identity.clip_display_path == "栏目 A/电影 集/第一集_0003"


@pytest.mark.parametrize(
    ("video_path", "expected_collection"),
    [
        (
            "01/爱情公寓/01.爱情公寓1 4K（2009）/ep01/v_a.mp4",
            "01/爱情公寓",
        ),
        (
            "01/爱情公寓/04.爱情公寓4 4K（2014）/ep02/v_b.mp4",
            "01/爱情公寓",
        ),
        (
            "01/法证先锋/法证先锋1/法证先锋1_22/v_a.mp4",
            "01/法证先锋",
        ),
        (
            "01/法证先锋/法证先锋3/法证先锋3_10/v_b.mp4",
            "01/法证先锋",
        ),
        (
            "02/忠犬八公物语/season/episode/v_a.mp4",
            "02/忠犬八公物语",
        ),
    ],
)
def test_readable_identity_groups_by_dataset_category_and_work(
    video_path: str,
    expected_collection: str,
) -> None:
    identity = derive_readable_clip_identity(
        clip_uid="opaque",
        shard_id="shard-1",
        source_relative_video_path=video_path,
        source_relative_source_video_path="legacy/source/episode.mkv",
    )

    assert identity.media_collection_relpath == expected_collection
    assert identity.media_collection_name == expected_collection.split("/")[1]


@pytest.mark.parametrize(
    "value",
    ["../escape.mp4", "/absolute.mp4", "a\\b.mp4", "single.mp4"],
)
def test_readable_identity_rejects_unsafe_relative_paths(value: str) -> None:
    with pytest.raises(ValueError):
        derive_readable_clip_identity(
            clip_uid="clip",
            shard_id="shard",
            source_relative_video_path=value,
            source_relative_source_video_path="collection/episode.mkv",
        )


def test_readable_audio_and_primary_voice_paths() -> None:
    identity = ReadableClipIdentity(
        clip_uid="opaque",
        clip_display_path="栏目/集合/片段 01",
        media_collection_relpath="栏目/集合",
        media_collection_name="集合",
        episode_name="片段",
        clip_name="片段 01",
        shard_id="shard",
    )
    assert (
        audio_binding_path(Path("audio"), identity)
        .as_posix()
        .endswith("audio/clips/栏目/集合/片段 01/audio_binding.json")
    )
    assert (
        full_audio_path(Path("audio"), identity)
        .as_posix()
        .endswith("audio/full_audio/栏目/集合/片段 01.flac")
    )
    assert (
        primary_voice_path(Path("primary_voice"), identity, entity_id="hero")
        .as_posix()
        .endswith("primary_voice/栏目/集合/片段 01/hero.flac")
    )


def _occurrences(inventory: object) -> list[JEAOccurrenceEmbedding]:
    rows = []
    for index, clip in enumerate(inventory.clips):
        rows.append(
            JEAOccurrenceEmbedding(
                occurrence_id=f"{clip.identity.clip_uid}/entity_1",
                clip_uid=clip.identity.clip_uid,
                entity_id="entity_1",
                subject_index=1,
                identity=clip.identity,
            visual_reference_path=str(
                clip.subject_references[0].artifact_path
            ),
                primary_voice_reference_path=str(
                    primary_voice_path(
                        Path("primary_voice"), clip.identity, entity_id="entity_1"
                    )
                ),
                face_embedding=[1.0, 0.0, float(index) * 0.0],
                voice_embedding=[1.0, 0.0, float(index) * 0.0],
            )
        )
    return rows


def test_cross_pairs_only_use_exact_full_media_collection(tmp_path: Path) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "target",
                "shard_id": "s1",
                "clip_relative_path": (
                    "01/爱情公寓/爱情公寓1/ep01/target_0.mp4"
                ),
                "source_relative_path": "01/爱情公寓/爱情公寓1/ep01.mkv",
            },
            {
                "clip_uid": "wrong",
                "shard_id": "s2",
                "clip_relative_path": (
                    "01/法证先锋/法证先锋1/ep01/wrong_0.mp4"
                ),
                "source_relative_path": "01/法证先锋/法证先锋1/ep01.mkv",
            },
        ],
    )
    summary = build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    assert summary.in_pair_count == 2
    assert summary.cross_pair_count == 0


def test_cross_pairs_ignore_legacy_parent_video_id(tmp_path: Path) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "a",
                "shard_id": "s1",
                "clip_relative_path": "show/collection/a_0.mp4",
                "source_relative_path": "show/collection/a.mkv",
                "parent_video_id": "old-parent-a",
            },
            {
                "clip_uid": "b",
                "shard_id": "s2",
                "clip_relative_path": "show/collection/b_0.mp4",
                "source_relative_path": "show/collection/b.mkv",
                "parent_video_id": "old-parent-b",
            },
        ],
    )
    summary = build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    assert summary.cross_pair_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "pairs/cross_pairs.jsonl").read_text().splitlines()
    ]
    assert {row["media_collection_relpath"] for row in rows} == {"show/collection"}


def test_pair_policy_keeps_frozen_thresholds_and_no_extra_gates(tmp_path: Path) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "a",
                "shard_id": "s1",
                "clip_relative_path": "c/collection/a_0.mp4",
                "source_relative_path": "c/a.mkv",
            },
            {
                "clip_uid": "b",
                "shard_id": "s2",
                "clip_relative_path": "c/collection/b_0.mp4",
                "source_relative_path": "c/b.mkv",
            },
        ],
    )
    summary = build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    assert (summary.face_threshold, summary.voice_threshold) == (0.72, 0.20)
    assert not summary.rank_gate_enabled
    assert not summary.margin_gate_enabled
    assert not summary.text_gate_enabled


def test_multi_subject_matching_maximizes_complete_face_assignment(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "target",
                "shard_id": "s1",
                "clip_relative_path": "show/collection/target_0.mp4",
                "source_relative_path": "show/collection/target.mkv",
            },
            {
                "clip_uid": "donor-a",
                "shard_id": "s2",
                "clip_relative_path": "show/collection/a_0.mp4",
                "source_relative_path": "show/collection/a.mkv",
            },
            {
                "clip_uid": "donor-b",
                "shard_id": "s3",
                "clip_relative_path": "show/collection/b_0.mp4",
                "source_relative_path": "show/collection/b.mkv",
            },
        ],
    )
    clip_by_uid = {item.identity.clip_uid: item for item in inventory.clips}

    def vector(angle: float) -> list[float]:
        radians = math.radians(angle)
        return [math.cos(radians), math.sin(radians)]

    def occurrence(clip_uid: str, entity_id: str, index: int, angle: float):
        clip = clip_by_uid[clip_uid]
        return JEAOccurrenceEmbedding(
            occurrence_id=f"{clip_uid}/{entity_id}",
            clip_uid=clip_uid,
            entity_id=entity_id,
            subject_index=index,
            identity=clip.identity,
            visual_reference_path=f"/{clip_uid}/{entity_id}.png",
            primary_voice_reference_path=f"/{clip_uid}/{entity_id}.flac",
            face_embedding=vector(angle),
            voice_embedding=[1.0, 0.0],
        )

    rows = [
        occurrence("target", "e1", 1, 0),
        occurrence("target", "e2", 2, 30),
        occurrence("donor-a", "e1", 1, 10),
        occurrence("donor-b", "e1", 1, -10),
    ]
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=rows,
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    cross_pairs = [
        json.loads(line)
        for line in (tmp_path / "pairs/cross_pairs.jsonl").read_text().splitlines()
    ]
    target = next(item for item in cross_pairs if item["target_clip_uid"] == "target")
    assert [item["donor_occurrence_id"] for item in target["mappings"]] == [
        "donor-b/e1",
        "donor-a/e1",
    ]


@pytest.mark.parametrize("input_layout", ["compacted", "single_run"])
def test_cli_wires_all_seven_stages_for_both_visual_layouts_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_layout: str,
) -> None:
    if input_layout == "compacted":
        inventory = _inventory(
            tmp_path,
            [
                {
                    "clip_uid": "clip-a",
                    "shard_id": "shard-a",
                    "clip_relative_path": "show/collection/a_0.mp4",
                    "source_relative_path": "show/collection/a.mkv",
                },
                {
                    "clip_uid": "clip-b",
                    "shard_id": "shard-b",
                    "clip_relative_path": "show/collection/b_0.mp4",
                    "source_relative_path": "show/collection/b.mkv",
                },
            ],
        )
    else:
        inventory = _dataset_inventory(tmp_path)
    run_roots = {Path(clip.clip_record_path).parents[2] for clip in inventory.clips}
    for run_root in run_roots:
        (run_root / "run.json").write_text("{}")
    output = tmp_path / "audio-production"
    calls: list[str] = []

    monkeypatch.setattr(jea_cli, "load_visual_production_inventory", lambda **_: inventory)
    monkeypatch.setattr(jea_cli, "FFmpegAudioMediaBackend", lambda **_: object())
    monkeypatch.setattr(jea_cli, "_path_argument", lambda *_, **__: tmp_path)
    monkeypatch.setattr(jea_cli, "LRASDRuntimeConfig", lambda **_: object())
    monkeypatch.setattr(jea_cli, "SileroVADRuntimeConfig", lambda **_: object())
    monkeypatch.setattr(jea_cli, "LRASDSubprocessBackend", lambda _: object())
    monkeypatch.setattr(jea_cli, "SileroVADSubprocessBackend", lambda _: object())
    monkeypatch.setattr(jea_cli, "ExternalReviewMediaBackend", lambda **_: object())

    def audio_stage(**kwargs: object) -> _StageResult:
        calls.append("audio")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        (root / "summary.json").write_text("{}")
        return _StageResult(clip_count=2)

    def primary_stage(**kwargs: object) -> _StageResult:
        calls.append("primary-voice")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        (root / "primary_voice_references.jsonl").write_text("{}\n")
        return _StageResult(reference_count=2)

    def embedding_stage(**kwargs: object) -> _StageResult:
        calls.append("embedding")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        _jsonl(
            root / "occurrences.jsonl",
            [item.model_dump(mode="json") for item in _occurrences(inventory)],
        )
        return _StageResult(occurrence_count=2)

    def pair_stage(**kwargs: object) -> _StageResult:
        calls.append("pair")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        (root / "in_pairs.jsonl").write_text("{}\n")
        return _StageResult(pair_count=2)

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def close(self) -> None:
            return None

    monkeypatch.setattr(jea_cli, "run_jea_audio_stage", audio_stage)
    monkeypatch.setattr(jea_cli, "run_jea_primary_voice_stage", primary_stage)
    monkeypatch.setattr(jea_cli, "run_jea_embedding_stage", embedding_stage)
    monkeypatch.setattr(jea_cli, "build_jea_pairs", pair_stage)
    monkeypatch.setattr(jea_cli, "_embedding_backends", lambda *_: (_Backend(), _Backend()))
    monkeypatch.setattr(jea_cli, "build_jea_diarization_inventory", lambda **_: object())
    monkeypatch.setattr(
        jea_cli,
        "_runtime_backend",
        lambda **_: (_Backend(), tmp_path / "missing-diagnostics"),
    )

    def diarization_stage(**kwargs: object) -> _StageResult:
        calls.append("diarization")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        (root / "inventory.json").write_text("{}")
        (root / "bound_segments.jsonl").write_text("{}\n")
        return _StageResult(segment_count=2)

    monkeypatch.setattr(jea_cli, "run_diarization_binding_pilot", diarization_stage)
    monkeypatch.setattr(
        jea_cli,
        "publish_readable_diarization_metadata",
        lambda **_: _StageResult(segment_count=2),
    )
    monkeypatch.setenv("QWEN3_ASR_ENV", str(tmp_path / "qwen"))
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", str(tmp_path / "qwen-model"))
    monkeypatch.setattr(jea_cli, "Qwen3ASRBackend", lambda _: object())

    def asr_stage(**kwargs: object) -> _StageResult:
        calls.append("qwen3-asr")
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True)
        (root / "segments.jsonl").write_text("{}\n")
        return _StageResult(segment_count=2)

    def h3_stage(**kwargs: object) -> _StageResult:
        calls.append("h3")
        Path(kwargs["output_root"]).mkdir(parents=True)
        return _StageResult(sample_count=2)

    monkeypatch.setattr(jea_cli, "run_qwen3_asr", asr_stage)
    monkeypatch.setattr(jea_cli, "render_jea_final_samples", h3_stage)
    result = jea_cli.main(
        [
            "--visual-production-root",
            inventory.visual_production_root,
            "--visual-runs-root",
            inventory.visual_runs_root,
            "--audio-production-root",
            str(output),
            "--stages",
            "all",
            "--workers",
            "1",
        ]
    )
    assert calls == [
        "audio",
        "primary-voice",
        "embedding",
        "pair",
        "diarization",
        "qwen3-asr",
        "h3",
    ]
    assert list(result["stage_results"]) == calls


class _FakeQwenModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return [SimpleNamespace(text=" raw transcript ", language="zh")]


def test_qwen3_backend_uses_official_api_and_exact_waveform() -> None:
    model = _FakeQwenModel()
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def factory(*args: object, **kwargs: object) -> _FakeQwenModel:
        factory_calls.append((args, kwargs))
        return model

    configuration = Qwen3ASRConfiguration(local_model_path="/local/qwen3")
    backend = Qwen3ASRBackend(configuration, model_factory=factory)
    waveform = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    text, language = backend.transcribe(waveform=waveform, sample_rate_hz=16000)
    assert (text, language) == (" raw transcript ", "zh")
    assert factory_calls[0][0] == ("/local/qwen3",)
    assert factory_calls[0][1]["device_map"] == "cuda:0"
    call = model.calls[0]
    captured_waveform, captured_rate = call["audio"]
    np.testing.assert_array_equal(captured_waveform, waveform)
    assert captured_rate == 16000
    assert call["context"] == ""
    assert call["language"] is None
    assert call["return_time_stamps"] is False


def test_qwen3_backend_loads_model_once_for_multiple_segments() -> None:
    model = _FakeQwenModel()
    count = 0

    def factory(*args: object, **kwargs: object) -> _FakeQwenModel:
        nonlocal count
        del args, kwargs
        count += 1
        return model

    backend = Qwen3ASRBackend(
        Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
        model_factory=factory,
    )
    backend.transcribe(waveform=np.ones(2, dtype=np.float32), sample_rate_hz=16000)
    backend.transcribe(waveform=np.ones(3, dtype=np.float32), sample_rate_hz=16000)
    assert count == 1
    assert len(model.calls) == 2


def test_qwen3_empty_and_failed_schemas_publish_no_confidence() -> None:
    common = {
        "clip_uid": "clip",
        "clip_display_path": "collection/clip",
        "media_collection_relpath": "collection",
        "media_collection_name": "collection",
        "episode_name": "episode",
        "clip_name": "clip",
        "shard_id": "shard",
        "segment_id": "segment_0001",
        "speaker_cluster_id": "speaker_1",
        "source_audio_path": "/audio.flac",
        "source_start_sample": 0,
        "source_end_sample": 100,
        "source_sample_rate_hz": 16000,
        "start_time": 0.0,
        "end_time": 0.00625,
        "configuration": Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
    }
    empty = Qwen3ASRSegment(status="empty", **common)
    failed = Qwen3ASRSegment(
        status="failed", failure_reason="RuntimeError:test", **common
    )
    for row in (empty, failed):
        payload = row.model_dump(mode="json")
        assert "confidence" not in payload
        assert "language_probability" not in payload
        assert payload["text"] is None
