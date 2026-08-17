from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from r2v_data_v2.h3.audio_binding import load_clip_bindings
from r2v_data_v2.h3.audio_pairing import AudioPairingConfig, build_audio_pair_samples
from r2v_data_v2.h3.audio_schemas import (
    AudioDatasetManifest,
    AudioPairSample,
    H3DatasetManifest,
    H3PictureAsset,
    H3Sample,
    H3SampleAudioAsset,
    H3SubjectAsset,
    H3SubjectAudioBinding,
    ProducerProvenance,
    SourceVideoProvenance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"output root already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    published = False
    try:
        temporary.replace(destination)
        published = True
    finally:
        if published:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)


def _resolve_asset(root: Path, relative_path: str, expected_sha256: str) -> Path:
    source = (root / relative_path).resolve(strict=True)
    source.relative_to(root.resolve(strict=True))
    if _sha256(source) != expected_sha256:
        raise ValueError(f"asset SHA-256 mismatch: {relative_path}")
    return source


def _copy_checked(
    *,
    source_root: Path,
    source_path: str,
    source_sha256: str,
    destination_root: Path,
    destination_path: Path,
) -> str:
    source = _resolve_asset(source_root, source_path, source_sha256)
    destination = destination_root / destination_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256(destination) != source_sha256:
        raise ValueError("copied asset SHA-256 mismatch")
    return destination_path.as_posix()


def publish_audio_pair_dataset(
    *,
    audio_binding_root: Path,
    output_root: Path,
    config: AudioPairingConfig | None = None,
    overwrite: bool = False,
    report_only: bool = False,
) -> dict[str, object]:
    source = audio_binding_root.resolve(strict=True)
    destination = output_root.resolve(strict=False)
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("pair output must be outside source audio root")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output root already exists: {destination}")
    bindings = load_clip_bindings(source / "clip_bindings.jsonl")
    source_manifest = AudioDatasetManifest.model_validate_json(
        (source / "dataset.json").read_text(encoding="utf-8")
    )
    source_report_path = source / "pair_report.json"
    source_report = (
        json.loads(source_report_path.read_text(encoding="utf-8"))
        if source_report_path.is_file()
        else {}
    )
    accounting_keys = (
        "selected_clip_count",
        "clip_binding_count",
        "ineligible_clip_count",
        "failed_clip_count",
    )
    source_accounting = {
        key: source_report[key] for key in accounting_keys if key in source_report
    }
    samples, edges, report = build_audio_pair_samples(
        bindings,
        audio_root=source,
        config=config,
    )
    report = {
        **report,
        **source_accounting,
        "pair_sample_count": len(samples),
        "pairwise_edges": [item.model_dump(mode="json") for item in edges],
    }
    if report_only:
        return report
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copytree(source, temporary)
        _write_jsonl(temporary / "pair_samples.jsonl", samples)
        (temporary / "pair_report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        producer = (
            samples[0].producer_provenance
            if samples
            else ProducerProvenance(
                producer="r2v_data_v2.h3.audio_pairing",
                version="h3_pair_policy_v1",
                config_fingerprint=(config or AudioPairingConfig()).fingerprint(),
                thresholds_calibrated=True,
            )
        )
        manifest = AudioDatasetManifest(
            clip_binding_count=len(bindings),
            failed_clip_count=source_manifest.failed_clip_count,
            pair_sample_count=len(samples),
            producer_provenance=producer,
        )
        (temporary / "dataset.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _safe_sample_dir(pair_id: str) -> str:
    return hashlib.sha256(pair_id.encode()).hexdigest()[:20]


def _load_pair_samples(path: Path) -> list[AudioPairSample]:
    return [
        AudioPairSample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export_h3_audio_dataset(
    *,
    audio_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> list[H3Sample]:
    source = audio_root.resolve(strict=True)
    destination = output_root.resolve(strict=False)
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("H3 output must be outside source audio root")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output root already exists: {destination}")
    bindings = load_clip_bindings(source / "clip_bindings.jsonl")
    pair_samples = _load_pair_samples(source / "pair_samples.jsonl")
    occurrence_by_id = {
        occurrence.entity_occurrence_id: occurrence
        for binding in bindings
        for occurrence in binding.entity_occurrences
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    outputs: list[H3Sample] = []
    try:
        temporary.mkdir(parents=True)
        for pair in pair_samples:
            sample_dir = _safe_sample_dir(pair.pair_id)
            video_source = Path(pair.target.video.path).resolve(strict=True)
            if _sha256(video_source) != pair.target.video.sha256:
                raise ValueError("target video SHA-256 mismatch")
            video_relative = Path("videos") / f"{sample_dir}{video_source.suffix}"
            video_destination = temporary / video_relative
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video_source, video_destination)
            target_audio_suffix = Path(pair.target.full_audio.asset.path).suffix
            target_audio_relative = (
                Path("audio") / sample_dir / f"audio_1{target_audio_suffix}"
            )
            _copy_checked(
                source_root=source,
                source_path=pair.target.full_audio.asset.path,
                source_sha256=pair.target.full_audio.asset.sha256,
                destination_root=temporary,
                destination_path=target_audio_relative,
            )
            pictures: list[H3PictureAsset] = []
            subjects: list[H3SubjectAsset] = []
            voice_assets: list[H3SampleAudioAsset] = []
            mappings: list[H3SubjectAudioBinding] = []
            for index, subject_binding in enumerate(pair.subjects, start=1):
                voice_reference = pair.voice_references[index - 1]
                target_occurrence = occurrence_by_id[
                    subject_binding.target_entity_occurrence_id
                ]
                picture_relative = Path("pictures") / sample_dir / f"picture_{index}.png"
                picture_path = _copy_checked(
                    source_root=source,
                    source_path=target_occurrence.visual_reference.image_asset.path,
                    source_sha256=target_occurrence.visual_reference.image_asset.sha256,
                    destination_root=temporary,
                    destination_path=picture_relative,
                )
                subject_id = f"subject_{index}"
                picture_id = f"picture_{index}"
                audio_id = f"audio_{index + 1}"
                voice_suffix = Path(voice_reference.asset.path).suffix
                voice_relative = (
                    Path("audio") / sample_dir / f"{audio_id}{voice_suffix}"
                )
                voice_path = _copy_checked(
                    source_root=source,
                    source_path=voice_reference.asset.path,
                    source_sha256=voice_reference.asset.sha256,
                    destination_root=temporary,
                    destination_path=voice_relative,
                )
                pictures.append(
                    H3PictureAsset(
                        picture_id=picture_id,
                        subject_id=subject_id,
                        path=picture_path,
                        sha256=target_occurrence.visual_reference.image_asset.sha256,
                    )
                )
                subjects.append(
                    H3SubjectAsset(
                        subject_id=subject_id,
                        entity_occurrence_id=target_occurrence.entity_occurrence_id,
                        entity_id=target_occurrence.entity_id,
                        phrase=target_occurrence.phrase,
                        picture_id=picture_id,
                    )
                )
                voice_assets.append(
                    H3SampleAudioAsset(
                        audio_id=audio_id,
                        role="voice_reference",
                        path=voice_path,
                        sha256=voice_reference.asset.sha256,
                        entity_occurrence_id=voice_reference.entity_occurrence_id,
                    )
                )
                mappings.append(
                    H3SubjectAudioBinding(
                        subject_id=subject_id,
                        audio_id=audio_id,
                        entity_occurrence_id=voice_reference.entity_occurrence_id,
                    )
                )
            outputs.append(
                H3Sample(
                    sample_id=pair.pair_id,
                    pair_kind=pair.pair_kind,
                    target_video=SourceVideoProvenance(
                        path=video_relative.as_posix(),
                        sha256=pair.target.video.sha256,
                    ),
                    target_full_audio=H3SampleAudioAsset(
                        audio_id="audio_1",
                        role="target_full_audio",
                        path=target_audio_relative.as_posix(),
                        sha256=pair.target.full_audio.asset.sha256,
                    ),
                    pictures=pictures,
                    subjects=subjects,
                    voice_reference_audio=voice_assets,
                    subject_audio_bindings=mappings,
                    speech_turns=pair.speech_turns,
                    rendered_h3_draft_annotation=pair.annotation_draft,
                    source_clip_binding_ids=pair.source_clip_binding_ids,
                    pair_evidence=pair.pair_evidence,
                    producer_provenance=pair.producer_provenance,
                )
            )
        outputs.sort(key=lambda item: item.sample_id)
        _write_jsonl(temporary / "samples.jsonl", outputs)
        producer = (
            outputs[0].producer_provenance
            if outputs
            else ProducerProvenance(
                producer="r2v_data_v2.h3.audio_export",
                version="v1",
                config_fingerprint=hashlib.sha256(b"empty-h3-export-v1").hexdigest(),
            )
        )
        (temporary / "dataset.json").write_text(
            H3DatasetManifest(
                sample_count=len(outputs),
                producer_provenance=producer,
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
        source_report = source / "pair_report.json"
        if source_report.is_file():
            shutil.copyfile(source_report, temporary / "pair_report.json")
        else:
            (temporary / "pair_report.json").write_text("{}\n", encoding="utf-8")
        expected = {
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
        }
        declared = {"dataset.json", "samples.jsonl", "pair_report.json"}
        for sample in outputs:
            declared.add(sample.target_video.path)
            declared.add(sample.target_full_audio.path)
            declared.update(item.path for item in sample.pictures)
            declared.update(item.path for item in sample.voice_reference_audio)
        if expected != declared:
            raise ValueError("H3 export tree does not match samples.jsonl")
        _publish(temporary, destination, overwrite=overwrite)
        return outputs
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
