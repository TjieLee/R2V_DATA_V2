from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from r2v_data_v2.h3.audio_shadow_qa_variants import discover_products, finalize_registry
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClipResult,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2, FinalQwen3SpeechSegment
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoCaseManifest,
    MimoClipJob,
    _inventory,
    build_mimo25_inventory,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MATERIALIZER_VERSION,
    MimoAVAnnotationDraft,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3MaterializationContractError,
    _extra_audio_contract,
    _MaterializationContext,
    _materialize_sample,
    _prepare_materialization_context,
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

QA_DATA_VERSION = "r2v.h3.audio_shadow_qa.9"
QA_REVIEW_VERSION = "r2v.h3.audio_shadow_human_qa.4"
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
        (record.speech_av_raw_response, ("summary", "shot1_caption")),
        (record.audio_finalize_raw_response, ("overall_soundscape", "non_diegetic_music")),
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


def _reference_payload(
    context: _MaterializationContext,
    job: MimoClipJob,
    source_samples: Sequence[FinalH3SampleV2],
    media_link: Callable[[str, str], str],
    *,
    audio_provenance: Mapping[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Project the rendering contract, never infer references from rendered prose."""
    selection = job.reference_selection
    images = {image.picture_label: image for image in job.reference_images}
    promotions = {item.source_image_index: item for item in selection.face_promotions}
    pictures = []
    for picture in context.contract.pictures:
        source = images[picture.picture_label]
        if picture.image_sha256 != source.image_sha256:
            raise ValueError(f"QA source media changed: {picture.image_path}")
        promotion = promotions.get(source.source_image_index)
        pictures.append({
            **picture.model_dump(mode="json"),
            "source_image_index": source.source_image_index,
            "source_image_id": source.source_image_id,
            "source_image_label": source.source_image_label,
            "face_promotion": promotion.model_dump(mode="json") if promotion else None,
            "url": media_link(picture.image_path, picture.image_sha256),
        })
    subjects = [subject.model_dump(mode="json") for subject in context.contract.subjects]
    audios = []
    for audio in context.contract.audios:
        if audio.kind == "full_audio_reuse" and (
            Path(audio.path).resolve() != Path(job.target_full_audio_path).resolve()
            or audio.sha256 != job.target_full_audio_sha256
        ):
            raise ValueError("QA canonical full Audio provenance differs from current job")
        provenance = dict((audio_provenance or {}).get(audio.audio_label, {}))
        voices = [v for v in context.sample.subject_voices
                  if v.voice_reference_path == audio.path
                  and v.entity_id == audio.entity_id]
        if any(v.voice_reference_sha256 != audio.sha256 for v in voices):
            raise ValueError(f"QA source media changed: {audio.path}")
        if len(voices) == 1 and audio.kind in {"target_voice", "cross_voice"}:
            voice = voices[0]
            provenance = {
                "role": "voice_reference",
                "source_type": "cross_donor" if voice.voice_source == "cross_donor" else "existing_target_voice",
                **voice.model_dump(mode="json"), **provenance,
            }
        elif audio.kind == "full_audio_reuse":
            provenance = {"role": "full_audio_reuse", "source_type": "canonical_full_audio", **provenance}
        audios.append({
            **audio.model_dump(mode="json"), "provenance": provenance,
            "role": provenance.get("role"), "source_type": provenance.get("source_type"),
            "url": media_link(audio.path, audio.sha256),
        })
    dropped = []
    for item in selection.dropped_references:
        row = item.model_dump(mode="json")
        candidates = {
            ref.image_artifact_path
            for sample in source_samples for ref in sample.visual_references
            if ref.image_id == item.source_image_id and ref.image_index == item.source_image_index
        }
        if len(candidates) == 1:
            path = Path(next(iter(candidates)))
            if path.is_file():
                digest = sha256_file(path)
                row.update(url=media_link(str(path), digest), image_sha256=digest)
        dropped.append(row)
    return {
        "conditioning_variant": context.variant,
        "pictures": pictures, "subjects": subjects, "audios": audios,
        "visual_selection": selection.model_dump(mode="json"),
        "dropped_visual_references": dropped,
    }


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
                previous["schema_version"] in {"r2v.h3.audio_shadow_qa.7", "r2v.h3.audio_shadow_qa.8", QA_DATA_VERSION}
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
    effective_inventory = _inventory({
        **base.model_dump(mode="json", exclude={"inventory_fingerprint"}),
        "jobs": [j.model_dump(mode="json") for j in jobs],
        "clip_count": len(jobs),
    })
    published = discover_products(
        shadow=shadow, jobs=jobs, reconcile=reconcile, samples=samples,
        stems=[s for s in stems if s.route == summary.route], inventories=[base, effective_inventory],
        source_hashes=source_hashes, fingerprint=_fingerprint,
    )
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
            plans = [(sid, None) for sid in sorted(current.source_h3_sample_ids)]
            plans += [(sid, "full_audio_reuse") for sid in sorted(current.source_h3_sample_ids)
                      if samples_by_id[sid].pair_type == "canonical" and len(current.reference_images) < 12]
            for sample_id, requested_variant in plans:
                source = samples_by_id[sample_id]
                sample = FinalH3SampleV2.model_validate({
                    **source.model_dump(mode="python"), "speech_segments": speech,
                })
                try:
                    materializer_input = _MaterializerInput(record.annotation, record.source_job_fingerprint)
                    context = _prepare_materialization_context(
                        sample, current, materializer_input, conditioning_variant=requested_variant,
                    )
                except MimoH3MaterializationContractError:
                    context = None
                references = _reference_payload(
                    context, current, [samples_by_id[sid] for sid in current.source_h3_sample_ids], media_link,
                ) if context is not None else None
                try:
                    _, text, warnings = _materialize_sample(
                        sample, current, materializer_input, conditioning_variant=requested_variant,
                    )
                    variant = {"status": "ready", "text": text, "warnings": warnings, "reason": None}
                except MimoH3MaterializationContractError as error:
                    variant = {"status": "unavailable", "text": None,
                               "reason": "materialization_contract_failed",
                               "issues": [item.to_dict() for item in error.issues]}
                variant_name = requested_variant or {"canonical": "visual_only", "in_pair": "target_voice_reference",
                                                     "cross_pair": "cross_voice_reference"}[source.pair_type]
                variants.append({"sample_id": f"{clip}/audio_reuse" if requested_variant else sample_id,
                                 "source_h3_sample_id": sample_id, "pair_type": source.pair_type,
                                 "conditioning_variant": variant_name,
                                 "source_kind": "qa_derived" if requested_variant else "source_h3",
                                 "publication_status": "review_only",
                                 "references": references, **variant})
            for source_kind, product_root, product, source in published:
                if product.clip_uid != clip:
                    continue
                refs = None
                if product.status == "ready":
                    values = {**source.model_dump(mode="python"), "speech_segments": speech}
                    options = {"conditioning_variant": product.conditioning_variant}
                    if source_kind == "audio_reuse_product":
                        options["reuse_audio_contracts"] = [r.contract for r in product.audio_references]
                        provenance = {r.contract.audio_label: r.model_dump(mode="json") for r in product.audio_references}
                    else:
                        values["subject_voices"] = product.effective_subject_voices
                        provenance = {f"<Audio {r.audio_index}>": r.model_dump(mode="json") for r in product.audio_references}
                        for ref in product.audio_references:
                            recovery = next((r for r in product.recovered_voice_references
                                             if r.status == "selected" and r.source_segment_id == ref.source_segment_id
                                             and r.entity_id == ref.entity_id), None)
                            if recovery is not None:
                                provenance[f"<Audio {ref.audio_index}>"]["recovery"] = recovery.model_dump(mode="json")
                        extras = [r for r in product.audio_references if r.role != "voice_reference"]
                        if extras:
                            options["extra_audio_contract"] = _extra_audio_contract(extras[0], render_path=Path(extras[0].audio_path))
                    context = _prepare_materialization_context(
                        FinalH3SampleV2.model_validate(values), current, materializer_input, **options,
                    )
                    if context.corrected != product.corrected_speech_segments:
                        raise ValueError("QA published speech differs from current materialization contract")
                    if source_kind == "mimo_h3_shadow" and [
                        (a.path, a.sha256, a.entity_id, a.speaker_id) for a in context.contract.audios
                    ] != [(a.audio_path, a.audio_sha256, a.entity_id, a.speaker_id) for a in product.audio_references]:
                        raise ValueError("QA published Audio differs from effective reference contract")
                    refs = _reference_payload(context, current, [source], media_link, audio_provenance=provenance)
                variants.append({
                    "sample_id": product.sample_id, "source_h3_sample_id": product.source_h3_sample_id,
                    "pair_type": product.pair_type, "conditioning_variant": product.conditioning_variant,
                    "source_kind": source_kind, "source_root": str(product_root), "publication_status": "published",
                    "status": "ready" if product.status == "ready" else "unavailable",
                    "text": product.rendered_h3_prompt, "warnings": product.warnings,
                    "reason": product.failure_reason, "references": refs,
                })
            variants, _ = finalize_registry(variants, _fingerprint)
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
            "reference_variant_coverage": finalize_registry(final_h3["variants"], _fingerprint)[1],
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
        if any(sha256_file(Path(item["source_path"])) != item["sha256"] for item in media.values()):
            raise ValueError("QA source media changed during build")
        _publish_directory(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"output_root": str(destination), "clip_count": len(clips),
            "dataset_fingerprint": data["dataset_fingerprint"], "model_called": False}
