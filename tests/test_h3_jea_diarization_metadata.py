from __future__ import annotations

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
        source_start_sample=160,
        source_end_sample=1760,
        source_sample_rate_hz=16000,
        start_time=0.01,
        end_time=0.11,
    )
    assert (row.source_start_sample, row.source_end_sample) == (160, 1760)
    assert row.mapping_policy_version == "h3_diarizen_sparse_anchor_policy_v1"
    assert not row.segmentation_changed
    assert not row.numeric_mapping_thresholds_changed
