from __future__ import annotations

import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.diarization_binding import BoundDiarizationSegment
from r2v_data_v2.h3.jea_audio_production import (
    JEAOccurrenceEmbedding,
    build_jea_pairs,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalSubjectVoice,
    FinalVisualReference,
    render_jea_final_samples,
)
from r2v_data_v2.h3.qwen3_asr import (
    Qwen3ASRConfiguration,
    Qwen3ASRSegment,
)
from r2v_data_v2.h3.visual_production_source import (
    NormalizedVisualReference,
    NormalizedVisualSample,
    ReadableClipIdentity,
    VisualProductionClip,
    VisualProductionInventory,
)
from r2v_data_v2.v3.production_export import ProductionSample
from r2v_data_v2.v3.schemas import ClipRecord


def _identity(clip_uid: str, clip_name: str) -> ReadableClipIdentity:
    return ReadableClipIdentity(
        clip_uid=clip_uid,
        clip_display_path=f"节目/集合/{clip_name}",
        media_collection_relpath="节目/集合",
        media_collection_name="集合",
        episode_name=clip_name.split("_")[0],
        clip_name=clip_name,
        shard_id=f"shard-{clip_uid}",
    )


def _visual_clip(
    tmp_path: Path,
    clip_uid: str,
    clip_name: str,
    *,
    subject_scope: str = "full",
    subject_visible_region: str | None = None,
    subject_entity_ids: tuple[str, ...] = ("entity_1",),
) -> VisualProductionClip:
    identity = _identity(clip_uid, clip_name)
    target = tmp_path / f"{clip_uid}.mp4"
    target.write_bytes(b"processed")
    subjects = [
        tmp_path / f"{clip_uid}-subject-{index}.png"
        for index in range(1, len(subject_entity_ids) + 1)
    ]
    attribute = tmp_path / f"{clip_uid}-hair.png"
    for index, subject in enumerate(subjects, start=1):
        subject.write_bytes(f"subject-{index}".encode())
    attribute.write_bytes(b"hair")
    reference_rows = [
        {
            "image_id": f"image_{index}",
            "image_index": index,
            "kind": "subject",
            "entity_id": entity_id,
            "image_path": subject.name,
            "source_frame_index": 0,
            "scope": subject_scope,
            "visible_region": subject_visible_region,
            "synthetic": False,
        }
        for index, (entity_id, subject) in enumerate(
            zip(subject_entity_ids, subjects, strict=True), start=1
        )
    ]
    attribute_index = len(reference_rows) + 1
    reference_rows.append(
        {
            "image_id": f"image_{attribute_index}",
            "image_index": attribute_index,
            "kind": "attribute",
            "attribute_id": "attribute_hair",
            "owner_entity_id": "entity_1",
            "attribute_type": "hair",
            "image_path": attribute.name,
            "source_frame_index": 0,
            "synthetic": False,
        }
    )
    sample = ProductionSample.model_validate(
        {
            "sample_id": clip_uid,
            "clip_uid": clip_uid,
            "target_video": str(target),
            "t2v_caption": "canonical caption",
            "r2v_instruction": f"canonical {clip_uid}: "
            + " and ".join(
                f"Image {index}" for index in range(1, attribute_index + 1)
            ),
            "references": reference_rows,
            "source": {
                "parent_video_id": "legacy-parent",
                "clip_suffix": clip_uid,
                "shard_id": identity.shard_id,
            },
        }
    )
    clip = ClipRecord.model_validate(
        {
            "clip_uid": clip_uid,
            "source": {
                "video_path": str(target),
                "parent_video_id": "legacy-parent",
                "clip_suffix": clip_uid,
                "source_index": 0,
                "caption_raw": "",
                "metadata": {
                    "source_relative_video_path": f"节目/集合/{clip_name}.mp4",
                    "source_relative_source_video_path": f"节目/集合/{clip_name}.mkv",
                },
            },
        }
    )
    references = [
        NormalizedVisualReference(
            **reference.model_dump(mode="python"),
            artifact_path=str(tmp_path / reference.image_path),
        )
        for reference in sample.references
    ]
    normalized = NormalizedVisualSample(
        sample_id=sample.sample_id,
        clip_uid=sample.clip_uid,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=sample.r2v_instruction,
        references=references,
    )
    return VisualProductionClip(
        identity=identity,
        sample=normalized,
        clip=clip,
        clip_record_path=str(tmp_path / identity.shard_id / clip_uid / "clip.json"),
        subject_references=[
            reference for reference in references if reference.kind == "subject"
        ],
    )


def _visual_inventory(tmp_path: Path) -> VisualProductionInventory:
    clips = [
        _visual_clip(
            tmp_path,
            "clip-a",
            "episode_a_0001",
            subject_scope="local",
            subject_visible_region="upper_body",
        ),
        _visual_clip(
            tmp_path,
            "clip-b",
            "episode_b_0002",
            subject_visible_region="whole",
        ),
    ]
    return VisualProductionInventory(
        visual_production_root=str(tmp_path),
        visual_runs_root=str(tmp_path / "runs"),
        visual_input_schema="r2v.v3.production_sample.1",
        visual_input_mode="compacted_production",
        canonical_sample_count=2,
        eligible_clip_count=2,
        eligible_subject_occurrence_count=2,
        media_collection_count=1,
        media_collection_clip_counts={"节目/集合": 2},
        shard_count=2,
        canonical_clips=clips,
        clips=clips,
        skip_reason_counts={},
    )


def _occurrences(inventory: VisualProductionInventory) -> list[JEAOccurrenceEmbedding]:
    return [
        JEAOccurrenceEmbedding(
            occurrence_id=f"{clip.identity.clip_uid}/entity_1",
            clip_uid=clip.identity.clip_uid,
            entity_id="entity_1",
            subject_index=1,
            identity=clip.identity,
            visual_reference_path=str(
                Path(inventory.visual_production_root)
                / clip.subject_references[0].image_path
            ),
            primary_voice_reference_path=str(
                Path("primary_voice") / clip.identity.clip_display_path / "e1.flac"
            ),
            face_embedding=[1.0, 0.0],
            voice_embedding=[1.0, 0.0],
        )
        for clip in inventory.clips
    ]


def _jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _final_sample(
    tmp_path: Path,
    voices: list[FinalSubjectVoice],
    *,
    clip_uid: str = "clip-a",
    references: list[FinalVisualReference] | None = None,
) -> FinalH3SampleV2:
    if references is None:
        references = []
        for index in (1, 2):
            artifact = tmp_path / f"entity_{index}.png"
            artifact.write_bytes(f"entity-{index}".encode())
            references.append(
                FinalVisualReference(
                    image_id=f"image_{index}",
                    image_index=index,
                    kind="subject",
                    image_path=f"references/entity_{index}.png",
                    image_artifact_path=str(artifact),
                    entity_id=f"entity_{index}",
                    source_frame_index=0,
                    scope="full",
                    visible_region="whole",
                    synthetic=False,
                )
            )
    return FinalH3SampleV2(
        sample_id=f"{clip_uid}/in_pair",
        pair_id=f"in_pair/{clip_uid}",
        pair_type="in_pair",
        clip_uid=clip_uid,
        clip_display_path=f"节目/集合/{clip_uid}",
        media_collection_relpath="节目/集合",
        media_collection_name="集合",
        episode_name="episode",
        clip_name=clip_uid,
        shard_id="shard-a",
        target_video=f"{clip_uid}.mp4",
        target_full_audio_path=f"{clip_uid}.flac",
        r2v_instruction="Image 1 and Image 2",
        visual_references=references,
        subject_voices=voices,
        speech_segments=[],
    )


def _voice(
    entity_index: int,
    *,
    subject_index: int | None = None,
    clip_uid: str = "clip-a",
    target_occurrence_id: str | None = None,
) -> FinalSubjectVoice:
    entity_id = f"entity_{entity_index}"
    return FinalSubjectVoice(
        subject_index=entity_index if subject_index is None else subject_index,
        entity_id=entity_id,
        target_occurrence_id=target_occurrence_id
        or f"{clip_uid}/{entity_id}",
        voice_reference_path=f"voices/{entity_id}.flac",
        voice_source="target",
    )


def _render(tmp_path: Path) -> list[dict[str, object]]:
    inventory = _visual_inventory(tmp_path)
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    bound = []
    qwen = []
    for clip in inventory.clips:
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid=clip.identity.clip_uid,
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                start_time=0.0,
                end_time=0.1,
                source_start_sample=0,
                source_end_sample=1600,
                cluster_binding_status="candidate_mapped",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip.identity.clip_uid}/entity_1",
                direct_anchor_samples=100,
                direct_anchor_seconds=0.00625,
                identity_scope="direct_anchor_present",
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **clip.identity.model_dump(mode="python"),
                segment_id="segment_0001",
                speaker_cluster_id="speaker_1",
                entity_id="entity_1",
                entity_occurrence_id=f"{clip.identity.clip_uid}/entity_1",
                source_audio_path=str(tmp_path / f"{clip.identity.clip_uid}.flac"),
                source_start_sample=0,
                source_end_sample=1600,
                source_sample_rate_hz=16000,
                start_time=0.0,
                end_time=0.1,
                status="transcribed",
                text=f"raw {clip.identity.clip_uid}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)
    render_jea_final_samples(
        visual_inventory=inventory,
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )
    return [
        json.loads(line)
        for line in (tmp_path / "h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_final_visual_reference_preserves_canonical_visible_region_values(
    tmp_path: Path,
) -> None:
    (tmp_path / "local.png").write_bytes(b"local")
    (tmp_path / "full.png").write_bytes(b"full")
    local = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/local.png",
            entity_id="entity_1",
            source_frame_index=1,
            scope="local",
            visible_region="upper_body",
            synthetic=False,
            artifact_path=str(tmp_path / "local.png"),
        )
    )
    full = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_2",
            image_index=2,
            kind="object",
            image_path="references/full.png",
            entity_id="entity_2",
            source_frame_index=2,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(tmp_path / "full.png"),
        )
    )

    assert local.visible_region == "upper_body"
    assert isinstance(local.visible_region, str)
    assert full.visible_region == "whole"


def test_final_visual_reference_keeps_nullable_background_and_attribute_region(
    tmp_path: Path,
) -> None:
    (tmp_path / "background.png").write_bytes(b"background")
    (tmp_path / "hair.png").write_bytes(b"hair")
    background = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="background",
            image_path="references/background.png",
            source_frame_index=1,
            scope="scene",
            visible_region=None,
            synthetic=False,
            artifact_path=str(tmp_path / "background.png"),
        )
    )
    attribute = FinalVisualReference.from_visual(
        NormalizedVisualReference(
            image_id="image_2",
            image_index=2,
            kind="attribute",
            image_path="references/hair.png",
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=2,
            scope=None,
            visible_region=None,
            synthetic=False,
            artifact_path=str(tmp_path / "hair.png"),
        )
    )

    assert background.visible_region is None
    assert attribute.visible_region is None


def test_final_visual_reference_publishes_exact_normalized_artifact_paths(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "visual-export/references/ordinary.png"
    enriched = tmp_path / "visual-runs/clips/clip-a/selected/local.png"
    background_path = tmp_path / "visual-runs/clips/clip-a/selected/background.png"
    attribute_path = (
        tmp_path / "visual-runs/subject_attributes/clip-a/attribute_hair.png"
    )
    compacted = tmp_path / "compacted-production/references/subject.png"
    for path in (ordinary, enriched, background_path, attribute_path, compacted):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    values = [
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/ordinary.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(ordinary),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="clips/clip-a/selected/local.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="local",
            visible_region="upper_body",
            synthetic=True,
            artifact_path=str(enriched),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="background",
            image_path="clips/clip-a/selected/background.png",
            source_frame_index=0,
            scope="scene",
            synthetic=False,
            artifact_path=str(background_path),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="attribute",
            image_path="clip-a/attribute_hair.png",
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=0,
            synthetic=False,
            artifact_path=str(attribute_path),
        ),
        NormalizedVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/subject.png",
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
            artifact_path=str(compacted),
        ),
    ]

    published = [FinalVisualReference.from_visual(value) for value in values]

    assert [item.image_path for item in published] == [
        value.image_path for value in values
    ]
    assert [item.image_artifact_path for item in published] == [
        str(ordinary),
        str(enriched),
        str(background_path),
        str(attribute_path),
        str(compacted),
    ]


def test_final_sample_supports_visual_assets_from_different_owner_roots(
    tmp_path: Path,
) -> None:
    subject_path = tmp_path / "visual-export/references/subject.png"
    attribute_path = (
        tmp_path / "visual-runs/subject_attributes/clip-a/attribute_hair.png"
    )
    for path in (subject_path, attribute_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    references = [
        FinalVisualReference(
            image_id="image_1",
            image_index=1,
            kind="subject",
            image_path="references/subject.png",
            image_artifact_path=str(subject_path),
            entity_id="entity_1",
            source_frame_index=0,
            scope="full",
            visible_region="whole",
            synthetic=False,
        ),
        FinalVisualReference(
            image_id="image_2",
            image_index=2,
            kind="attribute",
            image_path="clip-a/attribute_hair.png",
            image_artifact_path=str(attribute_path),
            attribute_id="attribute_hair",
            owner_entity_id="entity_1",
            attribute_type="hair",
            source_frame_index=0,
            synthetic=False,
        ),
    ]

    sample = _final_sample(tmp_path, [], references=references)

    assert [Path(item.image_artifact_path).read_bytes() for item in sample.visual_references] == [
        b"subject.png",
        b"attribute_hair.png",
    ]


def test_final_visual_reference_rejects_relative_or_missing_artifact_path(
    tmp_path: Path,
) -> None:
    values = {
        "image_id": "image_1",
        "image_index": 1,
        "kind": "subject",
        "image_path": "references/subject.png",
        "entity_id": "entity_1",
        "source_frame_index": 0,
        "scope": "full",
        "visible_region": "whole",
        "synthetic": False,
    }
    with pytest.raises(ValueError, match="must be absolute"):
        FinalVisualReference(**values, image_artifact_path="subject.png")
    with pytest.raises(ValueError, match="must be an existing file"):
        FinalVisualReference(
            **values,
            image_artifact_path=str(tmp_path / "missing.png"),
        )


def test_resolved_path_contract_requires_new_final_sample_schema(
    tmp_path: Path,
) -> None:
    sample = _final_sample(tmp_path, [_voice(1)])
    payload = sample.model_dump(mode="json")

    assert sample.schema_version == "r2v.h3.final_sample.3"
    without_artifact = json.loads(json.dumps(payload))
    without_artifact["visual_references"][0].pop("image_artifact_path")
    with pytest.raises(ValueError, match="image_artifact_path"):
        FinalH3SampleV2.model_validate(without_artifact)

    old_version = json.loads(json.dumps(payload))
    old_version["schema_version"] = "r2v.h3.final_sample.2"
    with pytest.raises(ValueError, match="r2v.h3.final_sample.3"):
        FinalH3SampleV2.model_validate(old_version)


def test_final_sample_allows_partial_subject_voice_coverage(tmp_path: Path) -> None:
    sample = _final_sample(tmp_path, [_voice(1)])

    assert sample.schema_version == "r2v.h3.final_sample.3"
    assert [item.entity_id for item in sample.visual_references] == [
        "entity_1",
        "entity_2",
    ]
    assert [item.entity_id for item in sample.subject_voices] == ["entity_1"]


def test_final_sample_still_allows_all_subjects_to_have_voice(tmp_path: Path) -> None:
    sample = _final_sample(tmp_path, [_voice(1), _voice(2)])

    assert [item.entity_id for item in sample.subject_voices] == [
        "entity_1",
        "entity_2",
    ]


def test_final_sample_rejects_noncanonical_or_duplicate_voice_bindings(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only canonical subject references"):
        _final_sample(tmp_path, [_voice(3)])
    with pytest.raises(ValueError, match="unique entity IDs"):
        _final_sample(tmp_path, [_voice(1), _voice(1)])


def test_final_sample_validates_voice_occurrence_and_subject_index(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="target occurrence is inconsistent"):
        _final_sample(
            tmp_path,
            [_voice(1, target_occurrence_id="other/entity_1")],
        )
    with pytest.raises(ValueError, match="canonical subject order"):
        _final_sample(tmp_path, [_voice(1, subject_index=2)])


def test_final_renderer_uses_exact_canonical_instruction_and_ordered_references(
    tmp_path: Path,
) -> None:
    rows = _render(tmp_path)
    assert {row["schema_version"] for row in rows} == {"r2v.h3.final_sample.3"}
    summary = json.loads((tmp_path / "h3/summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "r2v.h3.final_summary.3"
    assert summary["final_sample_schema_version"] == "r2v.h3.final_sample.3"
    first = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    assert first["r2v_instruction"] == "canonical clip-a: Image 1 and Image 2"
    assert [item["kind"] for item in first["visual_references"]] == [
        "subject",
        "attribute",
    ]
    assert first["visual_references"][1]["attribute_id"] == "attribute_hair"
    assert first["visual_references"][1]["owner_entity_id"] == "entity_1"
    assert first["visual_references"][0]["scope"] == "local"
    assert first["visual_references"][0]["visible_region"] == "upper_body"
    assert not isinstance(first["visual_references"][0]["visible_region"], dict)
    assert first["visual_references"][0]["image_path"].endswith("subject-1.png")
    assert Path(first["visual_references"][0]["image_artifact_path"]).is_file()
    second = next(row for row in rows if row["sample_id"] == "clip-b/in_pair")
    assert second["visual_references"][0]["scope"] == "full"
    assert second["visual_references"][0]["visible_region"] == "whole"
    assert len(first["subject_voices"]) == 1


def test_final_renderer_preserves_unvoiced_subject_and_its_speech(tmp_path: Path) -> None:
    visual = _visual_clip(
        tmp_path,
        "clip-a",
        "episode_a_0001",
        subject_entity_ids=("entity_1", "entity_2"),
    )
    inventory = VisualProductionInventory(
        visual_production_root=str(tmp_path),
        visual_runs_root=str(tmp_path / "runs"),
        visual_input_schema="r2v.v3.production_sample.1",
        visual_input_mode="compacted_production",
        canonical_sample_count=1,
        eligible_clip_count=1,
        eligible_subject_occurrence_count=2,
        media_collection_count=1,
        media_collection_clip_counts={"节目/集合": 1},
        shard_count=1,
        canonical_clips=[visual],
        clips=[visual],
        skip_reason_counts={},
    )
    build_jea_pairs(
        visual_inventory=inventory,
        occurrences=_occurrences(inventory),
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    bound = []
    qwen = []
    for index in (1, 2):
        entity_id = f"entity_{index}"
        segment_id = f"segment_{index:04d}"
        start_sample = (index - 1) * 1600
        end_sample = index * 1600
        bound.append(
            BoundDiarizationSegment(
                target_clip_uid="clip-a",
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{index}",
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                cluster_binding_status="candidate_mapped",
                entity_id=entity_id,
                entity_occurrence_id=f"clip-a/{entity_id}",
                direct_anchor_samples=100,
                direct_anchor_seconds=0.00625,
                identity_scope="direct_anchor_present",
            )
        )
        qwen.append(
            Qwen3ASRSegment(
                **visual.identity.model_dump(mode="python"),
                segment_id=segment_id,
                speaker_cluster_id=f"speaker_{index}",
                entity_id=entity_id,
                entity_occurrence_id=f"clip-a/{entity_id}",
                source_audio_path=str(tmp_path / "clip-a.flac"),
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                source_sample_rate_hz=16000,
                start_time=(index - 1) * 0.1,
                end_time=index * 0.1,
                status="transcribed",
                text=f"speech from {entity_id}",
                language="en",
                configuration=Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
            )
        )
    _jsonl(tmp_path / "diarization/bound_segments.jsonl", bound)
    _jsonl(tmp_path / "asr/segments.jsonl", qwen)

    render_jea_final_samples(
        visual_inventory=inventory,
        pairs_root=tmp_path / "pairs",
        diarization_root=tmp_path / "diarization",
        qwen3_asr_root=tmp_path / "asr",
        output_root=tmp_path / "h3",
    )

    row = json.loads(
        (tmp_path / "h3/samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert [item["entity_id"] for item in row["visual_references"][:2]] == [
        "entity_1",
        "entity_2",
    ]
    assert [item["entity_id"] for item in row["subject_voices"]] == ["entity_1"]
    assert [item["entity_id"] for item in row["speech_segments"]] == [
        "entity_1",
        "entity_2",
    ]


def test_final_cross_pair_swaps_only_donor_voice(tmp_path: Path) -> None:
    rows = _render(tmp_path)
    in_pair = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    cross = next(row for row in rows if row["sample_id"] == "clip-a/cross_pair/1")
    for key in (
        "target_video",
        "target_full_audio_path",
        "r2v_instruction",
        "visual_references",
        "speech_segments",
    ):
        assert cross[key] == in_pair[key]
    assert (
        cross["subject_voices"][0]["voice_reference_path"]
        != in_pair["subject_voices"][0]["voice_reference_path"]
    )
    assert cross["subject_voices"][0]["voice_source"] == "cross_donor"


def test_final_renderer_publishes_qwen_text_without_confidence(tmp_path: Path) -> None:
    rows = _render(tmp_path)
    segment = rows[0]["speech_segments"][0]
    assert segment["text"].startswith("raw clip-")
    assert segment["language"] == "en"
    assert "confidence" not in segment
    assert "language_probability" not in segment


def test_multi_subject_cross_mapping_is_one_to_one_and_capped_at_one(
    tmp_path: Path,
) -> None:
    inventory = _visual_inventory(tmp_path)
    occurrences = []
    for clip in inventory.clips:
        for index in (1, 2):
            occurrences.append(
                JEAOccurrenceEmbedding(
                    occurrence_id=f"{clip.identity.clip_uid}/entity_{index}",
                    clip_uid=clip.identity.clip_uid,
                    entity_id=f"entity_{index}",
                    subject_index=index,
                    identity=clip.identity,
                    visual_reference_path=f"visual/{clip.identity.clip_uid}/e{index}.png",
                    primary_voice_reference_path=f"voice/{clip.identity.clip_uid}/e{index}.flac",
                    face_embedding=[1.0, 0.0],
                    voice_embedding=[1.0, 0.0],
                )
            )
    summary = build_jea_pairs(
        visual_inventory=inventory,
        occurrences=occurrences,
        audio_root=tmp_path / "audio",
        output_root=tmp_path / "pairs",
    )
    assert summary.cross_pair_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "pairs/cross_pairs.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["pair_id"].endswith("/1")
        donors = [item["donor_occurrence_id"] for item in row["mappings"]]
        assert len(donors) == len(set(donors)) == 2
