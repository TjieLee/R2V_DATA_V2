from __future__ import annotations

import pytest

from r2v_data_v2.h3.jea_diarization import JEAReadableDiarizationSegment


def test_readable_diarization_segment_preserves_exact_bounds_and_policy() -> None:
    row = JEAReadableDiarizationSegment(
        clip_uid="opaque",
        clip_display_path="节目/集合/clip_0001",
        media_collection_relpath="节目/集合",
        media_collection_name="集合",
        episode_name="clip",
        clip_name="clip_0001",
        shard_id="shard-1",
        segment_id="segment_0001",
        speaker_cluster_id="speaker_1",
        entity_id="entity_1",
        entity_occurrence_id="opaque/entity_1",
        source_audio_path="/audio/节目/集合/clip_0001.flac",
        source_start_sample=320,
        source_end_sample=3520,
        source_sample_rate_hz=32000,
        source_channels=2,
        start_time=0.01,
        end_time=0.11,
    )
    assert (row.source_start_sample, row.source_end_sample) == (320, 3520)
    assert row.mapping_policy_version == "h3_diarizen_sparse_anchor_policy_v1"
    assert not row.segmentation_changed
    assert not row.numeric_mapping_thresholds_changed


def test_readable_diarization_rejects_legacy_16k_schema_as_current() -> None:
    payload = JEAReadableDiarizationSegment(
        clip_uid="opaque",
        clip_display_path="show/work/clip",
        media_collection_relpath="show/work",
        media_collection_name="work",
        episode_name="episode",
        clip_name="clip",
        shard_id="shard-1",
        segment_id="segment_0001",
        speaker_cluster_id="speaker_1",
        source_audio_path="/audio/clip.flac",
        source_start_sample=0,
        source_end_sample=3200,
        start_time=0.0,
        end_time=0.1,
    ).model_dump(mode="json")
    payload["schema_version"] = "r2v.h3.jea_diarization_segment.1"

    with pytest.raises(ValueError, match="jea_diarization_segment.2"):
        JEAReadableDiarizationSegment.model_validate(payload)
