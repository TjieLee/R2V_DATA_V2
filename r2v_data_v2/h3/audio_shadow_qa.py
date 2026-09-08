from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClipResult,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2, FinalQwen3SpeechSegment
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest, build_mimo25_inventory
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MATERIALIZER_VERSION,
    MimoAVAnnotationDraft,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3MaterializationContractError,
    _materialize_sample,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    MimoStemReconcileRecord,
    MimoStemReconcileSummary,
    build_stem_reconcile_jobs,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    _publish_directory,
    load_stem_shadow,
    sha256_file,
    stem_separation_root,
    stem_shadow_root,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.structured_output import normalize_structured_json_envelope

QA_DATA_VERSION = "r2v.h3.audio_shadow_qa.7"
QA_REVIEW_VERSION = "r2v.h3.audio_shadow_human_qa.2"
QA_LABELS = (
    "better", "same", "worse", "speaker_wrong",
    "dialogue_wrong", "audio_wrong", "visual_hallucination",
)
_Model = TypeVar("_Model", bound=BaseModel)


@dataclass(frozen=True)
class _MaterializerInput:
    """Read-only projection of the stem record fields consumed by the materializer."""

    annotation: MimoAVAnnotationDraft
    request_fingerprint: str


def _direct_h3(record: object) -> dict[str, object] | None:
    if record is None:
        return None
    annotation = record.annotation
    if annotation is not None:
        return annotation.h3_semantics.model_dump(mode="json")
    # Failed final publication must not hide parseable model-authored prose.
    fields: dict[str, object] = {}
    for raw, keys in (
        (record.visual_raw_response, ("subject_definitions", "visual_retention_analysis", "style_opening")),
        (record.speech_av_raw_response, ("shot1_caption",)),
        (record.audio_finalize_raw_response, ("summary", "shot1_caption", "overall_soundscape", "non_diegetic_music")),
    ):
        if raw is None:
            continue
        try:
            payload = json.loads(normalize_structured_json_envelope(raw))
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        fields.update({key: payload[key] for key in keys if key in payload})
    return fields or None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _rows(path: Path, model: type[_Model]) -> list[_Model]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _destination(
    output_root: Path | None, shadow: Path, protected: list[Path], overwrite: bool,
) -> Path:
    path = (output_root or shadow / "qa").expanduser().absolute()
    if path.is_symlink():
        raise ValueError("QA output cannot be a symlink")
    destination = path.resolve(strict=False)
    default = shadow / "qa"
    if destination == default:
        if any(_overlaps(destination, root) for root in protected[1:]):
            raise ValueError("QA output overlaps Visual input")
    elif any(_overlaps(destination, root) for root in protected):
        raise ValueError("QA output cannot overlap production or shadow inputs")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        try:
            previous = json.loads((destination / "data.json").read_text(encoding="utf-8"))
            owned = (
                previous["schema_version"] == QA_DATA_VERSION
                and previous["source_shadow_root"] == str(shadow)
            )
        except (OSError, ValueError, KeyError, TypeError):
            owned = False
        if not owned:
            raise ValueError("cannot overwrite a directory not owned by this QA run")
    return destination


def build_audio_shadow_qa(
    *,
    visual_production_root: Path,
    visual_runs_root: Path,
    audio_production_root: Path,
    case_manifest: Path,
    shadow_run_id: str | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Read frozen artifacts and publish only an independently owned static QA page."""
    audio = audio_production_root.expanduser().resolve(strict=True)
    visual = visual_production_root.expanduser().resolve(strict=True)
    runs = visual_runs_root.expanduser().resolve(strict=True)
    shadow = stem_shadow_root(audio, shadow_run_id)
    destination = _destination(output_root, shadow, [audio, visual, runs], overwrite)
    case_path = case_manifest.expanduser().resolve(strict=True)
    if _overlaps(destination, case_path):
        raise ValueError("QA output overlaps its case manifest")
    cases = MimoCaseManifest.model_validate_json(case_path.read_text(encoding="utf-8"))
    separation = stem_separation_root(audio, shadow_run_id)
    diarization, asr = shadow / "diarization", shadow / "asr"
    reconcile_root = shadow / MIMO25_STEM_RECONCILE_STAGE
    # Hash the source JSON sidecars before loading, then verify the same snapshot at publication.
    samples_path = audio / "h3/samples.jsonl"
    sources = {case_path, samples_path}
    for stage in (separation, diarization, asr, reconcile_root):
        for pattern in ("*.json", "*.jsonl", "raw_responses/*.json"):
            sources.update(stage.glob(pattern))
    source_hashes = {str(path): sha256_file(path) for path in sorted(sources)}
    inventory, stems, _ = load_stem_shadow(separation)
    if inventory.clip_uids != cases.clip_uids:
        raise ValueError("QA case manifest differs from selected separation order")
    diar_provenance, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=shadow,
    )
    base = build_mimo25_inventory(
        visual_production_root=visual, visual_runs_root=runs,
        audio_production_root=audio, case_manifest_path=case_path,
    )
    if [job.clip_uid for job in base.jobs] != cases.clip_uids:
        raise ValueError("QA Visual/MiMo inventory differs from case order")
    if source_hashes[str(samples_path)] != base.source_h3_samples_sha256:
        raise ValueError("QA source H3 samples differ from MiMo inventory")
    samples = _rows(samples_path, FinalH3SampleV2)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("QA source H3 samples contain duplicate sample IDs")
    for job in base.jobs:
        if sorted(job.source_h3_sample_ids) != sorted(
            sample.sample_id for sample in samples if sample.clip_uid == job.clip_uid
        ):
            raise ValueError("QA source H3 variants differ from MiMo inventory")
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base, diar_provenance),
        stem_diarization_root=diarization,
        stem_asr_root=asr, route=diar_provenance.route,
    )
    jobs_by_clip = {job.clip_uid: job for job in jobs}
    stems_by_clip = {record.clip_uid: record for record in stems if record.route == diar_provenance.route}
    reconcile = _rows(reconcile_root / "records.jsonl", MimoStemReconcileRecord)
    summary = MimoStemReconcileSummary.model_validate_json(
        (reconcile_root / "summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.clip_uids != cases.clip_uids
        or summary.route != diar_provenance.route
        or summary.skipped_clips != diar_provenance.skipped_clips
        or summary.diarization_failed_clips != diar_provenance.diarization_failed_clips
        or summary.processed_clip_uids != [item.clip_uid for item in reconcile]
        or summary.processed_clip_uids != [item.clip_uid for item in jobs]
        or len({item.clip_uid for item in reconcile}) != len(reconcile)
        or summary.ready_count != sum(item.status == "ready" for item in reconcile)
        or summary.failed_count != sum(item.status == "failed" for item in reconcile)
        or summary.model_call_count != sum(item.model_call_count for item in reconcile)
        or summary.audio_model_call_count != sum(item.audio_model_call_count for item in reconcile)
        or summary.av_model_call_count != sum(item.av_model_call_count for item in reconcile)
        or summary.visual_model_call_count != sum(item.visual_model_call_count for item in reconcile)
        or summary.text_model_call_count != sum(item.text_model_call_count for item in reconcile)
    ):
        raise ValueError("QA reconcile summary differs from current run inventory")
    for record in reconcile:
        if (
            record.source_job_fingerprint != jobs_by_clip[record.clip_uid].request_fingerprint
            or record.source_stem_record_fingerprint
            != stems_by_clip[record.clip_uid].record_fingerprint
        ):
            raise ValueError("QA reconcile record has stale source provenance")
    reconcile_by_clip = {item.clip_uid: item for item in reconcile}
    upstream_failures = {
        item.clip_uid: item.model_dump(mode="json")
        for item in [*summary.skipped_clips, *summary.diarization_failed_clips]
    }
    raw = sorted(
        _rows(diarization / "raw_segments.jsonl", RawDiarizationSegment),
        key=lambda item: (item.start_time, item.end_time, item.segment_id),
    )
    bound = _rows(diarization / "bound_segments.jsonl", BoundDiarizationSegment)
    clusters = _rows(diarization / "cluster_bindings.jsonl", DiarizationClusterBinding)
    clip_results = _rows(diarization / "clip_results.jsonl", DiarizationClipResult)
    transcripts = _rows(asr / "segments.jsonl", Qwen3ASRSegment)
    media: dict[str, dict[str, str]] = {}

    def media_link(source: str, expected_sha256: str) -> str:
        path = Path(source).resolve(strict=True)
        if not path.is_file() or _overlaps(destination, path):
            raise ValueError("QA media is missing or overlaps QA output")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"QA source media changed: {path}")
        suffix = path.suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".wav", ".flac", ".mp3",
                          ".m4a", ".ogg", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"unsupported QA media extension: {suffix}")
        url = f"media/{_fingerprint([str(path), expected_sha256])}{suffix}"
        media[url] = {"source_path": str(path), "sha256": expected_sha256}
        return url

    clips = []
    for base_job, stem_job in zip(base.jobs, inventory.jobs, strict=True):
        clip = base_job.clip_uid
        if (
            base_job.target_video_sha256 != stem_job.target_video_sha256
            or Path(base_job.target_video_path).resolve() != Path(stem_job.target_video_path).resolve()
        ):
            raise ValueError("QA original target video differs from separation inventory")
        selected = [item for item in stems if item.clip_uid == clip and item.route == summary.route]
        if len(selected) != 1:
            raise ValueError("QA separation route does not contain exactly one record per clip")
        stem_record = selected[0]
        stem_media = {
            item.stem_type: media_link(item.canonical_stem_path, item.canonical_stem_sha256)
            for item in stem_record.stems
        }
        images = [
            {**image.model_dump(mode="json"),
             "url": media_link(image.image_artifact_path, image.image_sha256)}
            for image in base_job.reference_images
        ]
        record = reconcile_by_clip.get(clip)
        current = jobs_by_clip.get(clip)
        final_h3 = {
            "status": "unavailable", "materializer_version": MIMO25_MATERIALIZER_VERSION,
            "text": None, "reason": "reconcile_failed" if record else "upstream_failed",
            "variants": [],
        }
        if record is not None and record.status == "ready":
            assert current is not None and record.annotation is not None
            # Only the in-memory speech input changes: exact current shadow ASR,
            # not the unrelated frozen production speech inventory. No media is generated.
            speech = [FinalQwen3SpeechSegment(
                segment_id=item.segment_id,
                speaker_cluster_id=item.source_speaker_cluster_id,
                entity_id=item.current_entity_id,
                entity_occurrence_id=item.entity_occurrence_id,
                source_start_sample=item.source_start_sample,
                source_end_sample=item.source_end_sample,
                source_sample_rate_hz=item.source_sample_rate_hz,
                start_time=item.start_time, end_time=item.end_time,
                text=item.asr_text, language=item.asr_language,
            ).model_dump(mode="python") for item in current.segments
                if item.asr_status == "transcribed"]
            variants = []
            for sample_id in sorted(current.source_h3_sample_ids):
                source = samples_by_id[sample_id]
                sample = FinalH3SampleV2.model_validate({
                    **source.model_dump(mode="python"), "speech_segments": speech,
                })
                try:
                    _, text, warnings = _materialize_sample(
                        sample, current,
                        _MaterializerInput(record.annotation, record.source_job_fingerprint),
                    )
                    variant = {"status": "ready", "text": text, "warnings": warnings, "reason": None}
                except MimoH3MaterializationContractError as error:
                    variant = {"status": "unavailable", "text": None,
                               "reason": "materialization_contract_failed",
                               "issues": [item.to_dict() for item in error.issues]}
                variants.append({"sample_id": sample_id, "pair_type": source.pair_type, **variant})
            final_h3.update(variants[0], variants=variants)
        elif record is not None:
            final_h3.update(
                reason="AV reconcile failed: " + (record.failure_reason or "unavailable"),
                issues=[item.to_dict() for item in record.failure_issues],
            )
        row = {
            "clip_uid": clip,
            "target": {
                "clip_display_path": stem_job.clip_display_path,
                "video_url": media_link(base_job.target_video_path, base_job.target_video_sha256),
                "video_path": base_job.target_video_path,
                "video_sha256": base_job.target_video_sha256,
                "duration_seconds": base_job.target_duration_seconds,
            },
            "visual": {
                "r2v_instruction": base_job.r2v_instruction,
                "reference_images": images,
                "reference_subjects": [item.model_dump(mode="json") for item in base_job.reference_subjects],
                "reference_selection": base_job.reference_selection.model_dump(mode="json"),
                "source_h3_sample_ids": base_job.source_h3_sample_ids,
            },
            "diarization": {
                "clip_result": next((item.model_dump(mode="json") for item in clip_results
                                     if item.target_clip_uid == clip), None),
                "raw_segments": [item.model_dump(mode="json") for item in raw if item.target_clip_uid == clip],
            },
            "asr": {"segments": [item.model_dump(mode="json") for item in transcripts if item.clip_uid == clip]},
            "speaker_binding": {
                "frozen_production_evidence": [item.model_dump(mode="json") for item in base_job.segments],
                "stem_segment_evidence": [item.model_dump(mode="json") for item in current.segments] if current else [],
                "stem_bound_segments": [item.model_dump(mode="json") for item in bound if item.target_clip_uid == clip],
                "stem_cluster_bindings": [item.model_dump(mode="json") for item in clusters if item.target_clip_uid == clip],
            },
            "separation": {**stem_record.model_dump(mode="json"), "media": stem_media},
            "final_h3": final_h3,
            "direct_h3": _direct_h3(record),
            "production_h3": [
                {"sample_id": sample_id, "text": samples_by_id[sample_id].r2v_instruction}
                for sample_id in sorted(base_job.source_h3_sample_ids)
            ],
            "reconcile": record.model_dump(mode="json") if record else {
                "status": "upstream_failed", "upstream_failure": upstream_failures[clip],
                "model_call_count": 0,
            },
            "qa": {"labels": [], "notes": ""},
        }
        row["record_fingerprint"] = _fingerprint(row)
        clips.append(row)
    data = {
        "schema_version": QA_DATA_VERSION, "review_schema_version": QA_REVIEW_VERSION,
        "shadow_run_id": shadow_run_id, "source_shadow_root": str(shadow),
        "clip_uids": cases.clip_uids, "clips": clips, "qa_labels": list(QA_LABELS),
        "source_artifact_sha256": source_hashes, "media": media,
        "base_inventory_provenance": base.model_dump(mode="json", exclude={"jobs"}),
        "reconcile_summary": summary.model_dump(mode="json"),
    }
    data["dataset_fingerprint"] = _fingerprint(data)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        (temporary / "media").mkdir(parents=True)
        for url, source in media.items():
            (temporary / url).symlink_to(source["source_path"])
        (temporary / "data.json").write_text(_json(data) + "\n", encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("audio_shadow_qa.html"), temporary / "review.html")
        if any(sha256_file(Path(path)) != digest for path, digest in source_hashes.items()):
            raise ValueError("QA source artifacts changed during build")
        _publish_directory(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"output_root": str(destination), "clip_count": len(clips),
            "dataset_fingerprint": data["dataset_fingerprint"], "model_called": False}
