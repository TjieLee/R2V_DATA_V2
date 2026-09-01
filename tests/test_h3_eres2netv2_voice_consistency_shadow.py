from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3.audio_backends import EmbeddingResult
from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    VoiceConsistencyAuditRecord,
    VoiceConsistencyAuditSummary,
    duration_bucket,
    similarity_distribution,
)
from r2v_data_v2.h3.diarization_voice_consistency_review import (
    VoiceConsistencyReviewAnnotation,
    record_fingerprint,
)
from r2v_data_v2.h3.eres2netv2_voice_consistency_shadow import (
    ERes2NetV2ShadowError,
    ERes2NetV2ShadowRecord,
    ERes2NetV2ShadowSummary,
    binary_roc_auc,
    diagnostic_threshold_sweep,
    run_eres2netv2_voice_consistency_shadow,
    threshold_metrics,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalSubjectVoice,
    FinalVisualReference,
)
from tools.run_h3_eres2netv2_embedding_worker import (
    ERES2NETV2_CHECKPOINT_NAME,
    ERES2NETV2_MODEL_IDENTIFIER,
    resolve_eres2netv2_checkpoint,
    serve_jsonl_requests,
    validate_speakerlab_code_root,
)

_MODEL_HASH = "e" * 64
_SOURCE_HASH = "f" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _sample(root: Path, primary_voice: Path) -> FinalH3SampleV2:
    video = root / "target.mp4"
    video.write_bytes(b"target-video")
    image = root / "entity.png"
    image.write_bytes(b"entity-image")
    return FinalH3SampleV2(
        sample_id="clip-a/in_pair",
        pair_id="in_pair/clip-a",
        pair_type="in_pair",
        clip_uid="clip-a",
        clip_display_path="category/show/clip-a",
        media_collection_relpath="category/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-a",
        shard_id="shard-a",
        target_video=str(video),
        target_full_audio_path=str(root / "full.flac"),
        r2v_instruction="Use Image 1.",
        visual_references=[
            FinalVisualReference(
                image_id="image_1",
                image_index=1,
                kind="subject",
                image_path="references/e1.png",
                image_artifact_path=str(image),
                entity_id="e1",
                source_frame_index=0,
                scope="full",
                visible_region="whole",
                synthetic=False,
            )
        ],
        subject_voices=[
            FinalSubjectVoice(
                subject_index=1,
                entity_id="e1",
                target_occurrence_id="clip-a/e1",
                voice_reference_path=str(primary_voice),
                voice_source="target",
            )
        ],
        speech_segments=[],
    )


def _record(
    *,
    audit_root: Path,
    primary_voice: Path,
    index: int,
    duration: float,
    scope: str,
    ecapa_score: float,
    shared_segment_path: Path | None = None,
) -> VoiceConsistencyAuditRecord:
    segment_id = f"segment_{index:04d}"
    relative = (
        shared_segment_path.relative_to(audit_root)
        if shared_segment_path is not None
        else Path("segment_audio") / f"{segment_id}.wav"
    )
    segment_path = audit_root / relative
    if not segment_path.exists():
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        segment_path.write_bytes(f"segment-{index}".encode())
    source = audit_root / "source.wav"
    if not source.exists():
        source.write_bytes(b"source-audio")
    return VoiceConsistencyAuditRecord(
        clip_uid="clip-a",
        segment_id=segment_id,
        speaker_cluster_id="speaker_0",
        entity_id="e1",
        entity_occurrence_id="clip-a/e1",
        start_time=float(index),
        end_time=float(index) + duration,
        duration_seconds=duration,
        direct_anchor_seconds=(0.2 if scope == "direct_anchor_present" else 0.0),
        identity_scope=scope,  # type: ignore[arg-type]
        duration_bucket=duration_bucket(duration),
        source_audio_path=str(source),
        source_audio_sha256=_sha256(source),
        source_start_sample=index * 16000,
        source_end_sample=index * 16000 + round(duration * 16000),
        source_sample_rate_hz=16000,
        segment_audio_path=relative.as_posix(),
        segment_audio_sha256=_sha256(segment_path),
        primary_voice_reference_path=str(primary_voice),
        primary_voice_reference_sha256=_sha256(primary_voice),
        speaker_model_identifier="speechbrain/spkrec-ecapa-voxceleb",
        speaker_model_fingerprint="a" * 64,
        cosine_similarity=ecapa_score,
    )


def _annotation(
    record: VoiceConsistencyAuditRecord,
    decision: str,
    *,
    fingerprint: str | None = None,
) -> VoiceConsistencyReviewAnnotation:
    return VoiceConsistencyReviewAnnotation(
        clip_uid=record.clip_uid,
        segment_id=record.segment_id,
        entity_id=record.entity_id,
        identity_scope=record.identity_scope,
        cosine_similarity=record.cosine_similarity,
        source_audio_sha256=record.source_audio_sha256,
        segment_audio_sha256=record.segment_audio_sha256,
        primary_voice_reference_sha256=record.primary_voice_reference_sha256,
        record_fingerprint=fingerprint or record_fingerprint(record),
        decision=decision,  # type: ignore[arg-type]
        notes="human label",
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def _fixture(tmp_path: Path) -> tuple[Path, list[VoiceConsistencyAuditRecord], list[Path]]:
    production = tmp_path / "production"
    audit = production / "diarization_voice_consistency_audit_v1"
    audit.mkdir(parents=True)
    primary = tmp_path / "primary.flac"
    primary.write_bytes(b"primary-voice")
    sample_path = production / "h3/samples.jsonl"
    _write_jsonl(sample_path, [_sample(tmp_path, primary)])
    shared = audit / "segment_audio/shared.wav"
    records = [
        _record(
            audit_root=audit,
            primary_voice=primary,
            index=1,
            duration=0.5,
            scope="direct_anchor_present",
            ecapa_score=0.8,
            shared_segment_path=shared,
        ),
        _record(
            audit_root=audit,
            primary_voice=primary,
            index=2,
            duration=0.75,
            scope="cluster_propagated_only",
            ecapa_score=0.2,
            shared_segment_path=shared,
        ),
        _record(
            audit_root=audit,
            primary_voice=primary,
            index=3,
            duration=1.0,
            scope="cluster_propagated_only",
            ecapa_score=-0.1,
        ),
        _record(
            audit_root=audit,
            primary_voice=primary,
            index=4,
            duration=2.0,
            scope="direct_anchor_present",
            ecapa_score=0.1,
        ),
        _record(
            audit_root=audit,
            primary_voice=primary,
            index=5,
            duration=1.5,
            scope="cluster_propagated_only",
            ecapa_score=0.4,
        ),
    ]
    records_path = audit / "records.jsonl"
    _write_jsonl(records_path, records)
    scopes = ("direct_anchor_present", "cluster_propagated_only")
    buckets = ("<0.75s", "0.75-1.0s", "1.0-2.0s", ">=2.0s")
    summary = VoiceConsistencyAuditSummary(
        source_audio_production_root=str(production),
        source_artifact_sha256={
            "raw_segments": "1" * 64,
            "bound_segments": "2" * 64,
            "h3_samples": _sha256(sample_path),
        },
        source_audio_set_fingerprint="3" * 64,
        target_primary_voice_set_fingerprint="4" * 64,
        bound_segment_count=len(records),
        mapped_segment_count=len(records),
        audited_segment_count=len(records),
        skipped_segment_count=0,
        direct_anchor_present_count=2,
        cluster_propagated_only_count=3,
        skip_reason_counts={},
        speaker_model_identifier="speechbrain/spkrec-ecapa-voxceleb",
        speaker_model_fingerprint="a" * 64,
        primary_voice_embedding_call_count=1,
        segment_embedding_call_count=len(records),
        model_call_count=len(records) + 1,
        similarity_distributions={
            scope: similarity_distribution(
                [item.cosine_similarity for item in records if item.identity_scope == scope]
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
    summary_path = audit / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n")
    annotations = [
        _annotation(records[0], "same"),
        _annotation(records[1], "different"),
        _annotation(records[2], "same"),
        _annotation(records[3], "different"),
        _annotation(records[4], "uncertain"),
        _annotation(records[0], "different", fingerprint="9" * 64),
    ]
    annotations_path = audit / "human_review/annotations.jsonl"
    _write_jsonl(annotations_path, annotations)
    source_paths = [
        records_path,
        summary_path,
        annotations_path,
        sample_path,
        primary,
        audit / "source.wav",
        *{audit / item.segment_audio_path for item in records},
    ]
    return audit, records, source_paths


class _FakeERes2NetV2Backend:
    model_identifier = ERES2NETV2_MODEL_IDENTIFIER
    checkpoint_sha256 = _MODEL_HASH

    def __init__(self, vectors: dict[str, np.ndarray], *, fail_name: str | None = None):
        self.vectors = vectors
        self.fail_name = fail_name
        self.calls: list[Path] = []

    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult:
        del entity_occurrence_id
        self.calls.append(audio_path)
        if audio_path.name == self.fail_name:
            raise RuntimeError("fixture embedding failure")
        return EmbeddingResult(
            vector=self.vectors.get(audio_path.name, np.array([3.0, 0.0])),
            model_identifier=self.model_identifier,
            checkpoint_sha256=self.checkpoint_sha256,
            backend_metadata={"speakerlab_source_fingerprint": _SOURCE_HASH},
        )


def _read_records(path: Path) -> list[ERes2NetV2ShadowRecord]:
    return [
        ERes2NetV2ShadowRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line
    ]


def test_binary_roc_auc_perfect_reversed_and_ties() -> None:
    labels = ["different", "different", "same", "same"]
    assert binary_roc_auc([0.1, 0.2, 0.8, 0.9], labels) == 1.0
    assert binary_roc_auc([0.9, 0.8, 0.2, 0.1], labels) == 0.0
    assert binary_roc_auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5
    assert binary_roc_auc([0.2], ["same"]) is None


def test_threshold_metrics_and_sweep_are_deterministic() -> None:
    scores = [0.1, 0.4, 0.8, 0.9]
    decisions = ["different", "same", "different", "same"]
    metrics = threshold_metrics(scores, decisions, 0.5)
    assert (
        metrics.true_positive,
        metrics.false_positive,
        metrics.true_negative,
        metrics.false_negative,
    ) == (1, 1, 1, 1)
    assert metrics.same_precision == pytest.approx(0.5)
    assert metrics.different_recall == pytest.approx(0.5)
    first = diagnostic_threshold_sweep(scores, decisions)
    second = diagnostic_threshold_sweep(scores, decisions)
    assert first == second
    assert len(first) == len(set(scores)) + 1


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.749999, "<0.75s"),
        (0.75, "0.75-1.0s"),
        (1.0, "1.0-2.0s"),
        (2.0, ">=2.0s"),
    ],
)
def test_shadow_duration_bucket_boundaries(duration: float, expected: str) -> None:
    assert duration_bucket(duration) == expected


def test_shadow_joins_current_labels_caches_media_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    audit, _, inputs = _fixture(tmp_path)
    before = {str(path): _sha256(path) for path in inputs}
    backend = _FakeERes2NetV2Backend(
        {
            "primary.flac": np.array([3.0, 0.0]),
            "shared.wav": np.array([6.0, 0.0]),
            "segment_0003.wav": np.array([2.0, 0.0]),
            "segment_0004.wav": np.array([0.0, 4.0]),
        }
    )
    output = tmp_path / "shadow-a"
    summary = run_eres2netv2_voice_consistency_shadow(
        audit_root=audit,
        speaker_backend=backend,
        output_root=output,
    )
    assert {str(path): _sha256(path) for path in inputs} == before
    assert summary.current_annotation_count == 5
    assert summary.stale_annotation_count == 1
    assert summary.labeled_same_count == 2
    assert summary.labeled_different_count == 2
    assert summary.uncertain_count == 1
    assert summary.evaluated_record_count == 4
    assert summary.error_count == 0
    assert summary.primary_voice_embedding_call_count == 1
    assert summary.segment_embedding_call_count == 3
    assert summary.model_call_count == 4
    assert sum(path.name == "primary.flac" for path in backend.calls) == 1
    assert sum(path.name == "shared.wav" for path in backend.calls) == 1
    assert all(path.name != "segment_0005.wav" for path in backend.calls)
    assert summary.production_threshold_applied is False
    assert summary.binding_modified is False
    assert summary.production_artifacts_modified is False
    assert summary.speakerlab_source_fingerprint == _SOURCE_HASH
    records = _read_records(output / "records.jsonl")
    assert records[0].ecapa_cosine_similarity == pytest.approx(0.8)
    assert records[0].eres2netv2_cosine_similarity == pytest.approx(1.0)
    assert records[2].ecapa_cosine_similarity == pytest.approx(-0.1)
    assert records[2].eres2netv2_cosine_similarity == pytest.approx(1.0)
    assert summary.overall.models["eres2netv2"].roc_auc == pytest.approx(0.75)
    assert summary.by_duration_bucket["1.0-2.0s"].labeled_same_count == 1
    assert summary.by_duration_bucket["1.0-2.0s"].uncertain_count == 1
    assert ERes2NetV2ShadowSummary.model_validate_json(
        (output / "summary.json").read_text()
    ) == summary
    assert (output / "comparison.csv").read_text().splitlines()[0].startswith(
        "schema_version,clip_uid,segment_id"
    )

    second = tmp_path / "shadow-b"
    run_eres2netv2_voice_consistency_shadow(
        audit_root=audit,
        speaker_backend=_FakeERes2NetV2Backend(backend.vectors),
        output_root=second,
    )
    for name in ("records.jsonl", "errors.jsonl", "comparison.csv", "summary.json"):
        assert (output / name).read_bytes() == (second / name).read_bytes()


def test_shadow_embedding_failure_is_fail_soft(tmp_path: Path) -> None:
    audit, _, _ = _fixture(tmp_path)
    output = tmp_path / "shadow"
    summary = run_eres2netv2_voice_consistency_shadow(
        audit_root=audit,
        speaker_backend=_FakeERes2NetV2Backend(
            {"primary.flac": np.array([1.0, 0.0])},
            fail_name="segment_0004.wav",
        ),
        output_root=output,
    )
    assert summary.evaluated_record_count == 3
    assert summary.error_count == 1
    errors = [
        ERes2NetV2ShadowError.model_validate_json(line)
        for line in (output / "errors.jsonl").read_text().splitlines()
    ]
    assert errors[0].segment_id == "segment_0004"
    assert errors[0].stage == "segment_embedding"
    assert "fixture embedding failure" in errors[0].reason


class _StubProtocolWorker:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        if request.get("audio_path") == "bad.wav":
            raise ValueError("bad fixture")
        return {
            "status": "available",
            "model_identifier": ERES2NETV2_MODEL_IDENTIFIER,
            "model_fingerprint": _MODEL_HASH,
            "embedding": [1.0, 2.0],
            "dimension": 2,
            "dtype": "float32",
            "backend_metadata": {"backend_name": "fixture"},
        }


def test_worker_jsonl_protocol_is_persistent_and_request_failures_are_isolated() -> None:
    source = io.StringIO(
        '{"request_id":"one","operation":"speaker_embedding","audio_path":"a.wav"}\n'
        '{"request_id":"bad","operation":"speaker_embedding","audio_path":"bad.wav"}\n'
        '{"request_id":"two","operation":"speaker_embedding","audio_path":"b.wav"}\n'
        '{"request_id":"stop","operation":"shutdown"}\n'
    )
    output = io.StringIO()
    errors = io.StringIO()
    worker = _StubProtocolWorker()
    assert (
        serve_jsonl_requests(
            worker,
            input_stream=source,
            output_stream=output,
            error_stream=errors,
            model_identifier=ERES2NETV2_MODEL_IDENTIFIER,
            model_fingerprint=_MODEL_HASH,
        )
        == 0
    )
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [item["status"] for item in responses] == [
        "available",
        "failed",
        "available",
        "shutdown",
    ]
    assert worker.calls == 3
    assert "bad fixture" in errors.getvalue()


def test_worker_validates_explicit_offline_source_and_checkpoint(tmp_path: Path) -> None:
    code_root = tmp_path / "3D-Speaker"
    for relative in (
        "speakerlab/process/processor.py",
        "speakerlab/utils/builder.py",
        "speakerlab/models/eres2net/ERes2NetV2.py",
    ):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    model_root = tmp_path / "model"
    model_root.mkdir()
    checkpoint = model_root / ERES2NETV2_CHECKPOINT_NAME
    checkpoint.write_bytes(b"checkpoint")
    assert validate_speakerlab_code_root(code_root) == code_root.resolve()
    assert resolve_eres2netv2_checkpoint(model_root) == (
        model_root.resolve(),
        checkpoint.resolve(),
    )
    wrong_checkpoint = tmp_path / "wrong.ckpt"
    wrong_checkpoint.write_bytes(b"wrong")
    with pytest.raises(FileNotFoundError, match="pretrained_eres2netv2"):
        resolve_eres2netv2_checkpoint(wrong_checkpoint)
    with pytest.raises(FileNotFoundError, match="official ERes2NetV2"):
        validate_speakerlab_code_root(tmp_path)
