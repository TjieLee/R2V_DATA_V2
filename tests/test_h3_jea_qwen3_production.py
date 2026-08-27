from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import r2v_data_v2.h3.jea_audio_production as jea_audio
import tools.backfill_h3_canonical_audio as canonical_audio_backfill
import tools.run_h3_jea_production as jea_cli
import tools.run_h3_qwen3_asr as qwen_cli
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    JEAOccurrenceEmbedding,
    audio_binding_path,
    build_jea_pairs,
    full_audio_path,
    jea_production_paths,
    materialize_canonical_audio_clips,
    primary_voice_path,
)
from r2v_data_v2.h3.qwen3_asr import (
    Qwen3ASRBackend,
    Qwen3ASRConfiguration,
    Qwen3ASRSegment,
    load_official_diarizen_waveform,
    run_qwen3_asr,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioTrackMetadata,
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
    with_subject: bool = True,
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
            "kind": "subject" if with_subject else "object",
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
                            "accepted_base_image_path": None,
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
                "subject_attributes": {
                    "internal_schema": "future-visual-only",
                },
                "diagnostics": {
                    "visual_only_counter": 7,
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
    enriched_mutator: Callable[[dict[str, object]], None] | None = None,
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
        if enriched_mutator is not None:
            enriched_mutator(enriched_payload)
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


def test_visual_inventory_preserves_canonical_clips_without_subjects(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "subject",
                "shard_id": "shard-a",
                "clip_relative_path": "show/work/subject.mp4",
                "source_relative_path": "show/work/source-a.mkv",
            },
            {
                "clip_uid": "object-only",
                "shard_id": "shard-b",
                "clip_relative_path": "show/work/object.mp4",
                "source_relative_path": "show/work/source-b.mkv",
                "with_subject": False,
            },
        ],
    )

    assert inventory.canonical_sample_count == len(inventory.canonical_clips) == 2
    assert inventory.eligible_clip_count == len(inventory.clips) == 1
    assert [item.identity.clip_uid for item in inventory.canonical_clips] == [
        "subject",
        "object-only",
    ]
    assert [item.identity.clip_uid for item in inventory.clips] == ["subject"]
    assert inventory.skip_reason_counts == {"no_subject_reference": 1}


def test_canonical_audio_manifest_covers_subject_and_no_subject_clips(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "subject",
                "shard_id": "shard-a",
                "clip_relative_path": "show/work/subject.mp4",
                "source_relative_path": "show/work/source-a.mkv",
            },
            {
                "clip_uid": "object-only",
                "shard_id": "shard-b",
                "clip_relative_path": "show/work/object.mp4",
                "source_relative_path": "show/work/source-b.mkv",
                "with_subject": False,
            },
        ],
    )
    calls: list[str] = []

    class _AudioBackend:
        def materialize_full_audio(self, **kwargs: object) -> SimpleNamespace:
            clip_uid = str(kwargs["clip_uid"])
            destination = Path(kwargs["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(f"audio:{clip_uid}".encode())
            calls.append(clip_uid)
            return SimpleNamespace(
                path=destination,
                stream=SimpleNamespace(duration_seconds=2.5),
            )

    audio_root = tmp_path / "audio-production" / "audio"
    summary = materialize_canonical_audio_clips(
        visual_inventory=inventory,
        audio_root=audio_root,
        audio_backend=_AudioBackend(),  # type: ignore[arg-type]
    )
    rows = [
        CanonicalAudioClip.model_validate_json(line)
        for line in (audio_root / "canonical_clips.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert calls == ["subject", "object-only"]
    assert summary.visual_canonical_clip_count == 2
    assert summary.canonical_audio_clip_count == 2
    assert [row.clip_uid for row in rows] == ["subject", "object-only"]
    no_subject = rows[1]
    assert no_subject.subject_reference_count == 0
    assert no_subject.target_audio_binding_path is None
    assert no_subject.target_audio_binding_sha256 is None

    sentinel = audio_root.parent / "pairs" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("unchanged", encoding="utf-8")
    calls.clear()
    materialize_canonical_audio_clips(
        visual_inventory=inventory,
        audio_root=audio_root,
        audio_backend=_AudioBackend(),  # type: ignore[arg-type]
    )
    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_canonical_audio_backfill_uses_official_audio_stage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "subject",
                "shard_id": "shard-a",
                "clip_relative_path": "show/work/subject.mp4",
                "source_relative_path": "show/work/source-a.mkv",
            }
        ],
    )
    production_root = tmp_path / "audio-production"
    production_root.mkdir()
    paths = jea_production_paths(production_root)
    item = inventory.clips[0]
    expected_audio = full_audio_path(paths.audio, item.identity)
    binding_path = audio_binding_path(paths.audio, item.identity)
    binding_path.parent.mkdir(parents=True)
    binding = AudioBindingSidecar(
        clip_uid=item.identity.clip_uid,
        source_run_root=inventory.visual_runs_root,
        source_video_path=item.sample.target_video,
        status="ineligible",
        reason="fixture has no clean speech",
        evidence=AudioBindingEvidence(
            clip_uid=item.identity.clip_uid,
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=item.sample.target_video,
                full_audio_path=str(expected_audio),
                duration_seconds=2.5,
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
    )
    binding_path.write_text(binding.model_dump_json(indent=2), encoding="utf-8")
    calls: list[Path] = []

    class _AudioBackend:
        def materialize_full_audio(self, **kwargs: object) -> SimpleNamespace:
            destination = Path(kwargs["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"canonical flac")
            calls.append(destination)
            return SimpleNamespace(
                path=destination,
                stream=SimpleNamespace(duration_seconds=2.5),
            )

    monkeypatch.setattr(
        canonical_audio_backfill,
        "load_visual_production_inventory",
        lambda **_kwargs: inventory,
    )
    monkeypatch.setattr(
        canonical_audio_backfill,
        "FFmpegAudioMediaBackend",
        lambda **_kwargs: _AudioBackend(),
    )
    arguments = [
        "--visual-production-root",
        inventory.visual_production_root,
        "--visual-runs-root",
        inventory.visual_runs_root,
        "--audio-production-root",
        str(production_root),
    ]

    result = canonical_audio_backfill.main(arguments)
    second_result = canonical_audio_backfill.main(arguments)

    assert calls == [expected_audio]
    assert result == second_result
    assert result["manifest_path"] == str(paths.audio / "canonical_clips.jsonl")
    assert result["summary_path"] == str(
        paths.audio / "canonical_clips_summary.json"
    )
    assert expected_audio.is_file()
    record = CanonicalAudioClip.model_validate_json(
        (paths.audio / "canonical_clips.jsonl").read_text(encoding="utf-8")
    )
    assert record.target_audio_binding_path == str(binding_path.resolve(strict=True))
    assert not (production_root / "canonical_clips.jsonl").exists()
    assert not (production_root / "canonical_clips_summary.json").exists()
    assert not (production_root / "full_audio").exists()


def test_audio_stage_binds_subject_subset_but_materializes_canonical_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "subject",
                "shard_id": "shard-a",
                "clip_relative_path": "show/work/subject.mp4",
                "source_relative_path": "show/work/source-a.mkv",
            },
            {
                "clip_uid": "object-only",
                "shard_id": "shard-b",
                "clip_relative_path": "show/work/object.mp4",
                "source_relative_path": "show/work/source-b.mkv",
                "with_subject": False,
            },
        ],
    )
    bound: list[str] = []

    def fake_pilot(**kwargs: object) -> SimpleNamespace:
        bound.extend(
            Path(item.clip_path).parent.name  # type: ignore[attr-defined]
            for item in kwargs["explicit_clips"]  # type: ignore[union-attr]
        )
        return SimpleNamespace()

    monkeypatch.setattr(jea_audio, "run_h3_audio_binding_pilot", fake_pilot)

    class _AudioBackend:
        def materialize_full_audio(self, **kwargs: object) -> SimpleNamespace:
            destination = Path(kwargs["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(str(kwargs["clip_uid"]).encode())
            return SimpleNamespace(
                path=destination,
                stream=SimpleNamespace(duration_seconds=2.5),
            )

    output = tmp_path / "audio-stage"
    jea_audio.run_jea_audio_stage(
        visual_inventory=inventory,
        output_root=output,
        lr_asd_backend=object(),  # type: ignore[arg-type]
        speech_backend=object(),  # type: ignore[arg-type]
        review_media_backend=object(),  # type: ignore[arg-type]
        audio_backend=_AudioBackend(),  # type: ignore[arg-type]
    )

    assert bound == ["subject"]
    rows = (output / "canonical_clips.jsonl").read_text(encoding="utf-8")
    assert [
        json.loads(line)["clip_uid"] for line in rows.splitlines()
    ] == ["subject", "object-only"]


def test_audio_loader_ignores_latest_visual_internal_clip_sections(
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

    clip = inventory.clips[0].clip
    assert clip.clip_uid == "clip-latest"
    assert clip.pairing is not None
    assert clip.pairing.retained_entity_ids == ["e1"]
    assert "reference_edit" not in type(clip).model_fields
    assert "subject_attributes" not in type(clip).model_fields
    assert "diagnostics" not in type(clip).model_fields


def test_visual_inventory_rejects_target_video_mismatch(tmp_path: Path) -> None:
    sample, production, runs = _sample(
        tmp_path,
        clip_uid="clip-target-mismatch",
        shard_id="shard-target-mismatch",
        clip_relative_path="节目/集合/ep-target_0.mp4",
        source_relative_path="节目/集合/ep-target.mkv",
    )
    different_target = tmp_path / "processed" / "different.mp4"
    different_target.write_bytes(b"different")
    sample["target_video"] = str(different_target)
    _jsonl(production / "samples.jsonl", [sample])

    with pytest.raises(
        ValueError,
        match="canonical target_video differs from clip source video_path",
    ):
        load_visual_production_inventory(
            visual_production_root=production,
            visual_runs_root=runs,
        )


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


def test_historical_visual_review_checklists_are_not_h3_schema_dependencies(
    tmp_path: Path,
) -> None:
    def historical_review(payload: dict[str, object]) -> None:
        attributes = payload["accepted_attributes"]
        assert isinstance(attributes, list)
        record = attributes[0]
        assert isinstance(record, dict)
        record["review"] = {
            "attribute_id": "a1",
            "sufficient_source_evidence": True,
            "historical_internal_check": "not consumed by H3",
        }
        record["completion_review"] = {
            "same_physical_attribute": True,
            "original_visible_details_preserved": True,
            "no_wrong_new_instance": True,
            "no_duplicate_component": True,
            "no_unrelated_content": True,
            "no_structural_distortion": True,
            "target_clear_and_prominent": True,
            "candidate_better_than_alpha": True,
        }

    inventory = _dataset_inventory(
        tmp_path,
        with_enriched=True,
        attribute_final_selection="completed",
        enriched_mutator=historical_review,
    )

    attribute = inventory.clips[0].sample.references[1]
    assert attribute.attribute_id == "a1"
    assert attribute.owner_entity_id == "e1"
    assert attribute.source_frame_index == 6


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["accepted_attributes"][0].pop("attribute_id"),
            "attribute_id",
        ),
        (
            lambda payload: payload["accepted_attributes"][0].update(
                {"owner_entity_id": "e2"}
            ),
            "attribute provenance mismatch",
        ),
        (
            lambda payload: payload["references"][1].update(
                {"image_path": "references/ordinary/different.png"}
            ),
            "attribute provenance mismatch",
        ),
        (
            lambda payload: payload["accepted_attributes"][0].update(
                {"source_frame_index": 7}
            ),
            "attribute provenance mismatch",
        ),
        (
            lambda payload: payload["accepted_attributes"].append(
                dict(payload["accepted_attributes"][0])
            ),
            "enriched attribute references must match accepted records",
        ),
        (
            lambda payload: payload["original_visual"].update(
                {"target_video": "/wrong/target.mp4"}
            ),
            "enriched target_video mismatch",
        ),
        (
            lambda payload: payload.update({"source_run_root": "/wrong/run"}),
            "enriched source_run_root mismatch",
        ),
    ],
)
def test_enriched_projection_preserves_required_provenance_failures(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _dataset_inventory(
            tmp_path,
            with_enriched=True,
            enriched_mutator=lambda payload: mutation(payload),
        )


def test_enriched_projection_rejects_duplicate_visual_reference(
    tmp_path: Path,
) -> None:
    def duplicate_reference(payload: dict[str, object]) -> None:
        references = payload["references"]
        assert isinstance(references, list)
        duplicate = dict(references[0])
        duplicate.update({"image_id": "image_6", "image_index": 6})
        references.append(duplicate)
        payload["enriched_instruction"] += " and <Image 6>"

    with pytest.raises(ValueError, match="no export match"):
        _dataset_inventory(
            tmp_path,
            with_enriched=True,
            enriched_mutator=duplicate_reference,
        )


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
        (root / "readable_segments.jsonl").write_text("{}\n")
        return _StageResult(segment_count=2)

    monkeypatch.setattr(jea_cli, "run_diarization_binding_pilot", diarization_stage)
    monkeypatch.setattr(
        jea_cli,
        "publish_readable_diarization_metadata",
        lambda **_: _StageResult(segment_count=2),
    )
    qwen_environment = tmp_path / "qwen"
    qwen_python = qwen_environment / "bin/python"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.write_text("fake python", encoding="utf-8")
    monkeypatch.setenv("QWEN3_ASR_ENV", str(qwen_environment))
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", str(tmp_path / "qwen-model"))

    def asr_subprocess(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append("qwen3-asr")
        assert command[0] == str(qwen_python)
        assert kwargs["env"]["QWEN3_ASR_MODEL_PATH"] == str(
            tmp_path / "qwen-model"
        )
        root = output / "asr"
        root.mkdir(parents=True)
        (root / "segments.jsonl").write_text("{}\n")
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "segment_count": 2,
                    "transcribed_count": 2,
                    "empty_count": 0,
                    "failed_count": 0,
                    "clip_count": 2,
                    "language_counts": {},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def h3_stage(**kwargs: object) -> _StageResult:
        calls.append("h3")
        Path(kwargs["output_root"]).mkdir(parents=True)
        return _StageResult(sample_count=2)

    monkeypatch.setattr(jea_cli.subprocess, "run", asr_subprocess)
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


def test_dedicated_qwen_cli_imports_without_visual_or_openai() -> None:
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'openai' or name == 'r2v_data_v2.v3.subject_attributes':
        raise AssertionError(f'forbidden isolated ASR import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import tools.run_h3_qwen3_asr
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_dedicated_qwen_cli_uses_roots_only_as_lightweight_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visual_root = tmp_path / "visual-production"
    visual_runs_root = tmp_path / "visual-runs"
    audio_root = tmp_path / "audio-production"
    for path in (visual_root, visual_runs_root, audio_root / "diarization"):
        path.mkdir(parents=True)
    qwen_environment = tmp_path / "qwen-env"
    monkeypatch.setenv("QWEN3_ASR_ENV", str(qwen_environment))
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", str(tmp_path / "qwen-model"))
    configuration = Qwen3ASRConfiguration(local_model_path=str(tmp_path / "qwen-model"))
    monkeypatch.setattr(qwen_cli, "Qwen3ASRBackend", lambda value: ("backend", value))
    calls: list[dict[str, object]] = []

    def run_stage(**kwargs: object) -> _StageResult:
        calls.append(kwargs)
        return _StageResult(segment_count=3)

    monkeypatch.setattr(qwen_cli, "run_qwen3_asr", run_stage)
    monkeypatch.setattr(
        qwen_cli.Qwen3ASRConfiguration,
        "from_environment",
        lambda: configuration,
    )
    result = qwen_cli.main(
        [
            "--visual-production-root",
            str(visual_root),
            "--visual-runs-root",
            str(visual_runs_root),
            "--audio-production-root",
            str(audio_root),
        ]
    )

    assert len(calls) == 1
    assert calls[0]["diarization_root"] == audio_root / "diarization"
    assert calls[0]["source_visual_production_root"] == str(visual_root.resolve())
    assert calls[0]["backend"] == ("backend", configuration)
    assert result["visual_runs_root"] == str(visual_runs_root.resolve())


def _isolated_qwen_orchestrator_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path]:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "clip-asr",
                "shard_id": "shard-asr",
                "clip_relative_path": "01/节目/第一季/片段.mp4",
                "source_relative_path": "01/节目/第一季/第一集.mkv",
            }
        ],
    )
    output = tmp_path / "audio-production"
    diarization = output / "diarization"
    diarization.mkdir(parents=True)
    (diarization / "readable_segments.jsonl").write_text("{}\n", encoding="utf-8")
    qwen_environment = tmp_path / "qwen-env"
    qwen_python = qwen_environment / "bin/python"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.write_text("fake python", encoding="utf-8")
    monkeypatch.setenv("QWEN3_ASR_ENV", str(qwen_environment))
    monkeypatch.setenv("QWEN3_ASR_MODEL_PATH", str(tmp_path / "qwen-model"))
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "cuda:3")
    monkeypatch.setenv("QWEN3_ASR_DTYPE", "bfloat16")
    monkeypatch.setenv("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(
        jea_cli,
        "load_visual_production_inventory",
        lambda **_: inventory,
    )
    return inventory, output, qwen_python


def test_jea_qwen_stage_launches_one_isolated_subprocess_and_reads_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, output, qwen_python = _isolated_qwen_orchestrator_fixture(
        tmp_path,
        monkeypatch,
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run_child(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        asr = output / "asr"
        asr.mkdir()
        (asr / "summary.json").write_text(
            json.dumps(
                {
                    "segment_count": 3,
                    "transcribed_count": 2,
                    "empty_count": 1,
                    "failed_count": 0,
                    "clip_count": 1,
                    "language_counts": {"zh": 2},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(jea_cli.subprocess, "run", run_child)
    result = jea_cli.main(
        [
            "--visual-production-root",
            inventory.visual_production_root,
            "--visual-runs-root",
            inventory.visual_runs_root,
            "--audio-production-root",
            str(output),
            "--stages",
            "qwen3-asr",
        ]
    )

    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == str(qwen_python)
    assert command[1].endswith("tools/run_h3_qwen3_asr.py")
    assert options["env"]["QWEN3_ASR_DEVICE"] == "cuda:3"
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert result["stage_results"]["qwen3-asr"]["segment_count"] == 3


def test_jea_qwen_subprocess_failure_prevents_asr_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, output, _qwen_python = _isolated_qwen_orchestrator_fixture(
        tmp_path,
        monkeypatch,
    )
    calls = 0

    def fail_child(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=7, stdout="", stderr="model load failed")

    monkeypatch.setattr(jea_cli.subprocess, "run", fail_child)
    with pytest.raises(RuntimeError, match="model load failed"):
        jea_cli.main(
            [
                "--visual-production-root",
                inventory.visual_production_root,
                "--visual-runs-root",
                inventory.visual_runs_root,
                "--audio-production-root",
                str(output),
                "--stages",
                "qwen3-asr",
            ]
        )

    assert calls == 1
    assert not (output / "asr").exists()


class _FakeQwenModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return [SimpleNamespace(text=" raw transcript ", language="zh")]


def test_diarizen_waveform_loader_uses_soundfile_first_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    stereo = np.asarray(
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        dtype=np.float32,
    )

    def read(path: str, *, dtype: str, always_2d: bool):
        calls.append((path, dtype, always_2d))
        return stereo, 48000

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=read))
    source = tmp_path / "diarizen.flac"
    waveform, sample_rate = load_official_diarizen_waveform(source)

    assert calls == [(str(source), "float32", True)]
    np.testing.assert_array_equal(waveform, stereo[:, 0])
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert sample_rate == 48000


@pytest.mark.parametrize("shape", [(0, 1), (3, 0)])
def test_diarizen_waveform_loader_requires_samples_and_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            read=lambda *_args, **_kwargs: (
                np.empty(shape, dtype=np.float32),
                16000,
            )
        ),
    )

    with pytest.raises(ValueError, match="no samples or channels"):
        load_official_diarizen_waveform(tmp_path / "empty.flac")


def test_qwen3_backend_import_error_names_required_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(
        RuntimeError,
        match=r"usable PyTorch runtime and qwen-asr==0\.0\.6",
    ):
        Qwen3ASRBackend(
            Qwen3ASRConfiguration(local_model_path="/local/qwen3")
        )


def test_qwen3_backend_rejects_unavailable_requested_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Model:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object):
            model_calls.append((args, kwargs))
            return _FakeQwenModel()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16=object(),
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        SimpleNamespace(Qwen3ASRModel=_Model),
    )

    with pytest.raises(
        RuntimeError,
        match=r"torch\.cuda\.is_available\(\) is false",
    ):
        Qwen3ASRBackend(
            Qwen3ASRConfiguration(local_model_path="/local/qwen3")
        )
    assert model_calls == []


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


def _write_readable_diarization_artifacts(
    *,
    diarization_root: Path,
    inventory: object,
    raw_segments: list[dict[str, object]],
    bound_segments: list[dict[str, object]],
) -> None:
    identity_by_clip = {
        item.identity.clip_uid: item.identity for item in inventory.clips
    }
    bound_by_key = {
        (str(item["target_clip_uid"]), str(item["segment_id"])): item
        for item in bound_segments
    }
    readable = []
    for raw in raw_segments:
        clip_uid = str(raw["target_clip_uid"])
        bound = bound_by_key[(clip_uid, str(raw["segment_id"]))]
        readable.append(
            {
                **identity_by_clip[clip_uid].model_dump(mode="json"),
                "schema_version": "r2v.h3.jea_diarization_segment.1",
                "segment_id": raw["segment_id"],
                "speaker_cluster_id": raw["speaker_cluster_id"],
                "entity_id": bound.get("entity_id"),
                "entity_occurrence_id": bound.get("entity_occurrence_id"),
                "source_audio_path": raw["source_audio_path"],
                "source_start_sample": raw["source_start_sample"],
                "source_end_sample": raw["source_end_sample"],
                "source_sample_rate_hz": raw["source_sample_rate_hz"],
                "start_time": raw["start_time"],
                "end_time": raw["end_time"],
                "raw_schema_version": "r2v.h3.diarization_segment.2",
                "bound_schema_version": "r2v.h3.diarization_bound_segment.1",
                "mapping_policy_version": "h3_diarizen_sparse_anchor_policy_v1",
                "segmentation_changed": False,
                "numeric_mapping_thresholds_changed": False,
            }
        )
    _jsonl(diarization_root / "readable_segments.jsonl", readable)
    (diarization_root / "readable_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "r2v.h3.jea_diarization_summary.1",
                "target_count": len(identity_by_clip),
                "segment_count": len(readable),
                "media_collection_count": len(
                    {
                        item.media_collection_relpath
                        for item in identity_by_clip.values()
                    }
                ),
                "segmentation_changed": False,
                "numeric_mapping_thresholds_changed": False,
            }
        ),
        encoding="utf-8",
    )


def _single_qwen_diarization_fixture(
    tmp_path: Path,
) -> tuple[object, Path, Path]:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "clip-readable",
                "shard_id": "shard-readable",
                "clip_relative_path": "01/节目 系列/第一季/片段 01.mp4",
                "source_relative_path": "01/节目 系列/第一季/第一集.mkv",
            }
        ],
    )
    diarization = tmp_path / "diarization"
    diarization.mkdir()
    source_audio = tmp_path / "source.flac"
    source_audio.write_bytes(b"source audio")
    raw_segments = [
        {
            "target_clip_uid": "clip-readable",
            "segment_id": "segment_0001",
            "speaker_cluster_id": "speaker_1",
            "start_time": 0.000125,
            "end_time": 0.0003125,
            "source_start_sample": 2,
            "source_end_sample": 5,
            "source_audio_path": str(source_audio),
            "source_sample_rate_hz": 16000,
        }
    ]
    bound_segments = [
        {
            "target_clip_uid": "clip-readable",
            "segment_id": "segment_0001",
            "speaker_cluster_id": "speaker_1",
            "start_time": 0.000125,
            "end_time": 0.0003125,
            "source_start_sample": 2,
            "source_end_sample": 5,
            "entity_id": "entity_1",
            "entity_occurrence_id": "clip-readable/entity_1",
        }
    ]
    _jsonl(diarization / "raw_segments.jsonl", raw_segments)
    _jsonl(diarization / "bound_segments.jsonl", bound_segments)
    _write_readable_diarization_artifacts(
        diarization_root=diarization,
        inventory=inventory,
        raw_segments=raw_segments,
        bound_segments=bound_segments,
    )
    return inventory, diarization, source_audio


def test_qwen3_consumes_readable_diarization_without_visual_inventory(
    tmp_path: Path,
) -> None:
    inventory, diarization, source_audio = _single_qwen_diarization_fixture(tmp_path)
    model = _FakeQwenModel()
    backend = Qwen3ASRBackend(
        Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
        model_factory=lambda *_args, **_kwargs: model,
    )
    loader_paths: list[Path] = []

    def audio_loader(path: Path) -> tuple[np.ndarray, int]:
        loader_paths.append(path)
        return np.arange(10, dtype=np.float32), 16000

    output = tmp_path / "asr"
    summary = run_qwen3_asr(
        diarization_root=diarization,
        source_visual_production_root=inventory.visual_production_root,
        output_root=output,
        backend=backend,
        audio_loader=audio_loader,
    )

    assert summary.segment_count == 1
    assert loader_paths == [source_audio]
    waveform, sample_rate = model.calls[0]["audio"]
    np.testing.assert_array_equal(waveform, np.asarray([2, 3, 4], dtype=np.float32))
    assert sample_rate == 16000
    row = json.loads((output / "segments.jsonl").read_text(encoding="utf-8"))
    assert row["clip_display_path"] == "01/节目 系列/第一季/片段 01"
    assert row["media_collection_relpath"] == "01/节目 系列"
    assert row["source_start_sample"] == 2
    assert row["source_end_sample"] == 5
    assert row["speaker_cluster_id"] == "speaker_1"
    assert row["entity_id"] == "entity_1"
    assert row["entity_occurrence_id"] == "clip-readable/entity_1"


def test_qwen3_readable_raw_bound_mismatch_fails_closed(tmp_path: Path) -> None:
    inventory, diarization, _source_audio = _single_qwen_diarization_fixture(tmp_path)
    readable_path = diarization / "readable_segments.jsonl"
    readable = json.loads(readable_path.read_text(encoding="utf-8"))
    readable["source_end_sample"] = 6
    readable_path.write_text(json.dumps(readable) + "\n", encoding="utf-8")
    backend = Qwen3ASRBackend(
        Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
        model_factory=lambda *_args, **_kwargs: _FakeQwenModel(),
    )

    with pytest.raises(ValueError, match="differs from raw or bound"):
        run_qwen3_asr(
            diarization_root=diarization,
            source_visual_production_root=inventory.visual_production_root,
            output_root=tmp_path / "asr",
            backend=backend,
            audio_loader=lambda _path: (np.ones(10, dtype=np.float32), 16000),
        )


def test_qwen3_all_segment_infrastructure_failure_publishes_no_stage(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        [
            {
                "clip_uid": "clip-fail",
                "shard_id": "shard-fail",
                "clip_relative_path": "show/collection/fail_0.mp4",
                "source_relative_path": "show/collection/fail.mkv",
            }
        ],
    )
    diarization = tmp_path / "diarization"
    diarization.mkdir()
    source_audio = tmp_path / "source.flac"
    source_audio.write_bytes(b"source audio")
    inventory_payload = {
        "mode": "production",
        "source_pairs_path": str(tmp_path / "pairs.jsonl"),
        "source_pairs_sha256": "a" * 64,
        "inventory_fingerprint": "b" * 64,
        "source_target_count": 1,
        "selected_target_count": 1,
        "selection_mode": "complete_in_pair_target_inventory_v1",
        "bounded_selection_applied": False,
        "targets": [
            {
                "target_clip_uid": "clip-fail",
                "target_video_path": inventory.clips[0].sample.target_video,
                "source_audio_path": str(source_audio),
                "source_audio_sha256": "c" * 64,
                "source_sample_rate_hz": 16000,
                "source_channels": 1,
                "source_frame_count": 16000,
                "target_audio_binding_path": str(tmp_path / "audio_binding.json"),
                "visual_references": [],
            }
        ],
    }
    (diarization / "inventory.json").write_text(
        json.dumps(inventory_payload),
        encoding="utf-8",
    )
    raw_segments = []
    bound_segments = []
    for index, (start, end) in enumerate(((0.0, 0.5), (0.5, 1.0)), start=1):
        segment_id = f"segment_{index:04d}"
        start_sample = round(start * 16000)
        end_sample = round(end * 16000)
        raw_segments.append(
            {
                "target_clip_uid": "clip-fail",
                "segment_id": segment_id,
                "speaker_cluster_id": "speaker_1",
                "backend_speaker_label": "speaker_1",
                "backend_reported_start_time": start,
                "backend_reported_end_time": end,
                "backend_reported_start_sample": start_sample,
                "backend_reported_end_sample": end_sample,
                "start_time": start,
                "end_time": end,
                "source_start_sample": start_sample,
                "source_end_sample": end_sample,
                "source_audio_path": str(source_audio),
                "source_audio_sha256": "c" * 64,
                "source_sample_rate_hz": 16000,
                "backend": "fake-diarizen",
                "model_identifier": "fake/diarizen",
                "model_fingerprint": "d" * 64,
                "backend_configuration_fingerprint": "e" * 64,
                "boundary_reconciliation": {
                    "adjusted": False,
                    "end_clamped": False,
                    "end_overrun_samples": 0,
                    "end_overrun_seconds": 0.0,
                },
            }
        )
        bound_segments.append(
            {
                "target_clip_uid": "clip-fail",
                "segment_id": segment_id,
                "speaker_cluster_id": "speaker_1",
                "start_time": start,
                "end_time": end,
                "source_start_sample": start_sample,
                "source_end_sample": end_sample,
                "cluster_binding_status": "unbound",
                "direct_anchor_samples": 0,
                "direct_anchor_seconds": 0.0,
                "identity_scope": "unresolved",
            }
        )
    _jsonl(diarization / "raw_segments.jsonl", raw_segments)
    _jsonl(diarization / "bound_segments.jsonl", bound_segments)
    _write_readable_diarization_artifacts(
        diarization_root=diarization,
        inventory=inventory,
        raw_segments=raw_segments,
        bound_segments=bound_segments,
    )

    backend = Qwen3ASRBackend(
        Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
        model_factory=lambda *_args, **_kwargs: _FakeQwenModel(),
    )
    loader_calls = 0

    def failed_loader(_path: Path) -> tuple[np.ndarray, int]:
        nonlocal loader_calls
        loader_calls += 1
        raise OSError("audio infrastructure unavailable")

    output = tmp_path / "asr"
    with pytest.raises(
        RuntimeError,
        match="Qwen3 ASR failed for every diarization segment",
    ):
        run_qwen3_asr(
            diarization_root=diarization,
            source_visual_production_root=inventory.visual_production_root,
            output_root=output,
            backend=backend,
            audio_loader=failed_loader,
        )

    assert loader_calls == 2
    assert not output.exists()
    assert list(tmp_path.glob(".asr.tmp-*")) == []
