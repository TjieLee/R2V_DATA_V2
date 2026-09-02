from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    VoiceConsistencyAuditRecord,
    VoiceConsistencyAuditSummary,
    similarity_distribution,
)
from r2v_data_v2.h3.diarization_voice_consistency_review import (
    VoiceConsistencyReviewAnnotation,
    build_review_summary,
    current_reviews,
    initialize_review,
    load_review_context,
    make_review_handler,
    render_review_html,
    save_review,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalSubjectVoice,
    FinalVisualReference,
)

_HASH = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, marker: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes((marker.to_bytes(2, "little", signed=True)) * 1600)


def _sample(
    root: Path,
    *,
    target_video: Path,
    primary_voice: Path,
    sample_id: str = "clip-a/in_pair",
    pair_type: str = "in_pair",
) -> FinalH3SampleV2:
    artifact = root / "entity.png"
    artifact.write_bytes(b"entity")
    reference = FinalVisualReference(
        image_id="image_1",
        image_index=1,
        kind="subject",
        image_path="references/e1.png",
        image_artifact_path=str(artifact),
        entity_id="e1",
        source_frame_index=0,
        scope="full",
        visible_region="whole",
        synthetic=False,
    )
    canonical_audio = root / "full-audio.flac"
    canonical_audio.write_bytes(b"canonical-audio")
    voice = FinalSubjectVoice(
        subject_index=1,
        entity_id="e1",
        target_occurrence_id="clip-a/e1",
        voice_reference_path=str(primary_voice),
        voice_reference_sha256=_sha256(primary_voice),
        source_start=0.0,
        source_end=1.0,
        source_start_sample=0,
        source_end_sample=32000,
        sample_mapping_policy="round_time_seconds_times_32000_v1",
        voice_source=("target" if pair_type == "in_pair" else "cross_donor"),
        donor_occurrence_id=(None if pair_type == "in_pair" else "donor/e1"),
        donor_clip_uid=(None if pair_type == "in_pair" else "donor"),
        donor_clip_display_path=(None if pair_type == "in_pair" else "show/donor"),
    )
    return FinalH3SampleV2(
        sample_id=sample_id,
        pair_id=f"{pair_type}/clip-a",
        pair_type=pair_type,  # type: ignore[arg-type]
        clip_uid="clip-a",
        clip_display_path="category/show/clip-a",
        media_collection_relpath="category/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-a",
        shard_id="shard-a",
        target_video=str(target_video),
        target_full_audio_path=str(canonical_audio),
        target_full_audio_sha256=_sha256(canonical_audio),
        r2v_instruction="Use Image 1.",
        visual_references=[reference],
        subject_voices=[voice],
        speech_segments=[],
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in values
        ),
        encoding="utf-8",
    )


def _record(
    *,
    audit_root: Path,
    source_audio: Path,
    primary_voice: Path,
    index: int,
    scope: str,
    similarity: float,
) -> VoiceConsistencyAuditRecord:
    segment_id = f"segment_{index:04d}"
    segment_relative = Path("segment_audio/clip-a") / f"{segment_id}.wav"
    segment_path = audit_root / segment_relative
    _write_wav(segment_path, index)
    return VoiceConsistencyAuditRecord(
        clip_uid="clip-a",
        segment_id=segment_id,
        speaker_cluster_id="speaker_0",
        entity_id="e1",
        entity_occurrence_id="clip-a/e1",
        start_time=float(index),
        end_time=float(index + 1),
        duration_seconds=1.0,
        direct_anchor_seconds=(0.2 if scope == "direct_anchor_present" else 0.0),
        identity_scope=scope,  # type: ignore[arg-type]
        duration_bucket="1.0-2.0s",
        source_audio_path=str(source_audio),
        source_audio_sha256=_sha256(source_audio),
        source_start_sample=index * 32000,
        source_end_sample=(index + 1) * 32000,
        source_sample_rate_hz=32000,
        source_channels=2,
        segment_audio_path=segment_relative.as_posix(),
        segment_audio_sha256=_sha256(segment_path),
        primary_voice_reference_path=str(primary_voice),
        primary_voice_reference_sha256=_sha256(primary_voice),
        speaker_model_identifier="speechbrain/spkrec-ecapa-voxceleb",
        speaker_model_fingerprint="f" * 64,
        cosine_similarity=similarity,
    )


def _summary(
    *,
    production: Path,
    h3_samples: Path,
    records: list[VoiceConsistencyAuditRecord],
) -> VoiceConsistencyAuditSummary:
    scopes = ("direct_anchor_present", "cluster_propagated_only")
    buckets = ("<0.75s", "0.75-1.0s", "1.0-2.0s", ">=2.0s")
    return VoiceConsistencyAuditSummary(
        source_audio_production_root=str(production),
        source_artifact_sha256={
            "raw_segments": "1" * 64,
            "bound_segments": "2" * 64,
            "h3_samples": _sha256(h3_samples),
        },
        source_audio_set_fingerprint="3" * 64,
        target_primary_voice_set_fingerprint="4" * 64,
        bound_segment_count=len(records),
        mapped_segment_count=len(records),
        audited_segment_count=len(records),
        skipped_segment_count=0,
        direct_anchor_present_count=sum(
            item.identity_scope == "direct_anchor_present" for item in records
        ),
        cluster_propagated_only_count=sum(
            item.identity_scope == "cluster_propagated_only" for item in records
        ),
        skip_reason_counts={},
        speaker_model_identifier="speechbrain/spkrec-ecapa-voxceleb",
        speaker_model_fingerprint="f" * 64,
        primary_voice_embedding_call_count=1,
        segment_embedding_call_count=len(records),
        model_call_count=len(records) + 1,
        similarity_distributions={
            scope: similarity_distribution(
                [
                    item.cosine_similarity
                    for item in records
                    if item.identity_scope == scope
                ]
            )
            for scope in scopes
        },
        duration_bucket_distributions={
            scope: {
                bucket: similarity_distribution(
                    [
                        item.cosine_similarity
                        for item in records
                        if item.identity_scope == scope
                        and item.duration_bucket == bucket
                    ]
                )
                for bucket in buckets
            }
            for scope in scopes
        },
        review_candidate_count=0,
    )


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    production = tmp_path / "production"
    audit = production / "diarization_voice_consistency_audit_v1"
    source_audio = tmp_path / "source.wav"
    primary_voice = tmp_path / "voice.flac"
    target_video = tmp_path / "target.mp4"
    _write_wav(source_audio, 10)
    _write_wav(primary_voice, 20)
    target_video.write_bytes(b"video-bytes")
    h3_samples = production / "h3/samples.jsonl"
    _write_jsonl(
        h3_samples,
        [
            _sample(
                tmp_path,
                target_video=target_video,
                primary_voice=primary_voice,
            )
        ],
    )
    records = [
        _record(
            audit_root=audit,
            source_audio=source_audio,
            primary_voice=primary_voice,
            index=1,
            scope="cluster_propagated_only",
            similarity=0.1,
        ),
        _record(
            audit_root=audit,
            source_audio=source_audio,
            primary_voice=primary_voice,
            index=2,
            scope="cluster_propagated_only",
            similarity=0.2,
        ),
        *[
            _record(
                audit_root=audit,
                source_audio=source_audio,
                primary_voice=primary_voice,
                index=index,
                scope="direct_anchor_present",
                similarity=0.3 + (index - 3) * 0.01,
            )
            for index in range(3, 15)
        ],
    ]
    records_path = audit / "records.jsonl"
    _write_jsonl(records_path, records)
    summary_path = audit / "summary.json"
    summary_path.write_text(
        _summary(
            production=production,
            h3_samples=h3_samples,
            records=records,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    inputs = [
        records_path,
        summary_path,
        h3_samples,
        source_audio,
        primary_voice,
        target_video,
        *[audit / item.segment_audio_path for item in records],
    ]
    return audit, inputs


def _payload(context: object, index: int = 0) -> dict[str, object]:
    item = context.inventory_payload["items"][index]  # type: ignore[attr-defined,index]
    return {
        "clip_uid": item["clip_uid"],
        "segment_id": item["segment_id"],
        "record_fingerprint": item["record_fingerprint"],
        "source_audio_sha256": item["source_audio_sha256"],
        "segment_audio_sha256": item["segment_audio_sha256"],
        "primary_voice_reference_sha256": item[
            "primary_voice_reference_sha256"
        ],
        "decision": "same",
        "notes": "Same speaker.",
    }


def test_default_inventory_contains_all_propagated_and_lowest_ten_direct(
    tmp_path: Path,
) -> None:
    audit, _ = _fixture(tmp_path)
    context = load_review_context(audit)
    items = context.inventory_payload["items"]
    default = [item for item in items if item["default_review_inventory"]]
    assert len(default) == 12
    assert sum(
        item["identity_scope"] == "cluster_propagated_only" for item in default
    ) == 2
    direct = [
        item for item in default if item["identity_scope"] == "direct_anchor_present"
    ]
    assert [item["cosine_similarity"] for item in direct] == pytest.approx(
        [0.3 + index * 0.01 for index in range(10)]
    )


def test_annotation_roundtrip_and_deterministic_csv_summary(tmp_path: Path) -> None:
    audit, _ = _fixture(tmp_path)
    context, empty = initialize_review(audit)
    assert empty.reviewed == 0
    reviewed_at = datetime(2026, 9, 1, 3, 4, 5, tzinfo=UTC)
    annotation, summary = save_review(
        context,
        _payload(context),
        reviewed_at=reviewed_at,
    )
    assert annotation.decision == "same"
    assert annotation.reviewed_at == reviewed_at
    assert summary.reviewed == 1
    assert summary.decision_counts == {"same": 1, "different": 0, "uncertain": 0}
    assert summary.by_identity_scope["cluster_propagated_only"]["same"] == 1
    assert summary.by_duration_bucket["1.0-2.0s"]["same"] == 1
    assert summary.decision_similarity["same"].min == pytest.approx(0.1)
    before = {
        name: (audit / "human_review" / name).read_bytes()
        for name in ("annotations.jsonl", "annotations.csv", "summary.json")
    }
    reloaded, _ = initialize_review(audit)
    after = {
        name: (audit / "human_review" / name).read_bytes()
        for name in before
    }
    assert after == before
    restored = current_reviews(reloaded)
    assert restored["annotations"][0]["notes"] == "Same speaker."
    assert "record_fingerprint" in (audit / "human_review/annotations.csv").read_text()


def test_stale_fingerprint_annotation_is_not_current(tmp_path: Path) -> None:
    audit, _ = _fixture(tmp_path)
    context, _ = initialize_review(audit)
    payload = _payload(context)
    save_review(
        context,
        payload,
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    rows = [
        VoiceConsistencyAuditRecord.model_validate_json(line)
        for line in (audit / "records.jsonl").read_text().splitlines()
    ]
    rows[0] = rows[0].model_copy(update={"cosine_similarity": 0.11})
    _write_jsonl(audit / "records.jsonl", rows)
    changed = load_review_context(audit)
    reviews = current_reviews(changed)
    summary = build_review_summary(
        changed,
        [
            VoiceConsistencyReviewAnnotation.model_validate_json(line)
            for line in (audit / "human_review/annotations.jsonl")
            .read_text()
            .splitlines()
        ],
    )
    assert reviews == {"annotations": [], "stale_annotation_count": 1}
    assert summary.reviewed == 0
    assert summary.stale_annotation_count == 1
    with pytest.raises(ValueError, match="fingerprint is stale"):
        save_review(changed, payload)


def test_target_video_mapping_and_inconsistent_duplicates_fail_closed(
    tmp_path: Path,
) -> None:
    audit, _ = _fixture(tmp_path)
    context = load_review_context(audit)
    assert context.target_video_by_clip["clip-a"].name == "target.mp4"
    production = audit.parent
    h3_path = production / "h3/samples.jsonl"
    current = FinalH3SampleV2.model_validate_json(h3_path.read_text().splitlines()[0])
    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(b"different-video")
    other = current.model_copy(
        update={
            "sample_id": "clip-a/cross_pair",
            "pair_id": "cross_pair/clip-a",
            "pair_type": "cross_pair",
            "target_video": str(other_video),
        }
    )
    _write_jsonl(h3_path, [current, other])
    summary = VoiceConsistencyAuditSummary.model_validate_json(
        (audit / "summary.json").read_text()
    ).model_copy(
        update={
            "source_artifact_sha256": {
                **VoiceConsistencyAuditSummary.model_validate_json(
                    (audit / "summary.json").read_text()
                ).source_artifact_sha256,
                "h3_samples": _sha256(h3_path),
            }
        }
    )
    (audit / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    with pytest.raises(ValueError, match="disagree on target video"):
        load_review_context(audit)


def test_review_server_media_security_and_source_immutability(tmp_path: Path) -> None:
    audit, inputs = _fixture(tmp_path)
    before = {str(path): _sha256(path) for path in inputs}
    context, _ = initialize_review(audit)
    server = HTTPServer(("127.0.0.1", 0), make_review_handler(context))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        inventory = context.inventory_payload
        first = inventory["items"][0]
        with urlopen(base + first["media"]["segment_audio"], timeout=5) as response:
            assert response.headers["Content-Type"] == "audio/x-wav"
            assert response.headers["Cache-Control"] == "no-store"
        request = Request(
            base + first["media"]["primary_voice"],
            headers={"Range": "bytes=0-9"},
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Length"] == "10"
            assert len(response.read()) == 10
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/media/%2e%2e/records.jsonl", timeout=5)
        assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    after = {str(path): _sha256(path) for path in inputs}
    assert after == before


def test_review_html_has_fast_controls_and_no_score_preselection() -> None:
    review = render_review_html()
    for text in (
        "SAME",
        "DIFFERENT",
        "UNCERTAIN",
        "Previous",
        "Next",
        "Save &amp; Next",
        "Propagated only",
        "Direct anchor",
        "Similarity ascending",
        "Similarity descending",
    ):
        assert text in review
    assert "event.key === '1'" in review
    assert "event.key === '2'" in review
    assert "event.key === '3'" in review
    assert "event.key === 'Enter'" in review
    assert "state.pending = annotation ? annotation.decision : null" in review
    assert "state.pending = item.cosine_similarity" not in review


def test_generated_review_javascript_is_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    javascript_core = Path(
        "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"
    )
    if node is None and not javascript_core.is_file():
        pytest.skip("No JavaScript syntax checker is available")
    match = re.search(r"<script>(.*?)</script>", render_review_html(), flags=re.DOTALL)
    assert match is not None
    script = tmp_path / "voice-review.js"
    script.write_text(
        "function __syntax_check__() {\n" + match.group(1) + "\n}\n",
        encoding="utf-8",
    )
    command = (
        [node, "--check", str(script)]
        if node is not None
        else [str(javascript_core), f"--strict-file={script}", "-e", "0"]
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
