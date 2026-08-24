from __future__ import annotations

import json
from pathlib import Path

from r2v_data_v2.h3.diarization_binding import BoundDiarizationSegment
from r2v_data_v2.h3.jea_audio_production import (
    JEAOccurrenceEmbedding,
    build_jea_pairs,
)
from r2v_data_v2.h3.jea_final_renderer import render_jea_final_samples
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


def _visual_clip(tmp_path: Path, clip_uid: str, clip_name: str) -> VisualProductionClip:
    identity = _identity(clip_uid, clip_name)
    target = tmp_path / f"{clip_uid}.mp4"
    target.write_bytes(b"processed")
    subject = tmp_path / f"{clip_uid}-subject.png"
    attribute = tmp_path / f"{clip_uid}-hair.png"
    subject.write_bytes(b"subject")
    attribute.write_bytes(b"hair")
    sample = ProductionSample.model_validate(
        {
            "sample_id": clip_uid,
            "clip_uid": clip_uid,
            "target_video": str(target),
            "t2v_caption": "canonical caption",
            "r2v_instruction": f"canonical {clip_uid}: Image 1 and Image 2",
            "references": [
                {
                    "image_id": "image_1",
                    "image_index": 1,
                    "kind": "subject",
                    "entity_id": "entity_1",
                    "image_path": subject.name,
                    "source_frame_index": 0,
                    "scope": "full",
                    "synthetic": False,
                },
                {
                    "image_id": "image_2",
                    "image_index": 2,
                    "kind": "attribute",
                    "attribute_id": "attribute_hair",
                    "owner_entity_id": "entity_1",
                    "attribute_type": "hair",
                    "image_path": attribute.name,
                    "source_frame_index": 0,
                    "synthetic": False,
                },
            ],
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
        subject_references=[references[0]],
    )


def _visual_inventory(tmp_path: Path) -> VisualProductionInventory:
    clips = [
        _visual_clip(tmp_path, "clip-a", "episode_a_0001"),
        _visual_clip(tmp_path, "clip-b", "episode_b_0002"),
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


def test_final_renderer_uses_exact_canonical_instruction_and_ordered_references(
    tmp_path: Path,
) -> None:
    rows = _render(tmp_path)
    first = next(row for row in rows if row["sample_id"] == "clip-a/in_pair")
    assert first["r2v_instruction"] == "canonical clip-a: Image 1 and Image 2"
    assert [item["kind"] for item in first["visual_references"]] == [
        "subject",
        "attribute",
    ]
    assert first["visual_references"][1]["attribute_id"] == "attribute_hair"
    assert first["visual_references"][1]["owner_entity_id"] == "entity_1"
    assert len(first["subject_voices"]) == 1


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
