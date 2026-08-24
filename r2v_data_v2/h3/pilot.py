from __future__ import annotations

import json
import re
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from r2v_data_v2.h3.association import (
    FaceEntityAssociationPolicy,
    associate_face_tracks_to_entities,
)
from r2v_data_v2.h3.fusion import AudioBindingPolicy, build_audio_binding_sidecar
from r2v_data_v2.h3.lr_asd import (
    LRASDBackend,
    LRASDRuntimeError,
    SpeechActivityBackend,
    normalize_lr_asd_evidence,
)
from r2v_data_v2.h3.pilot_schemas import (
    H3AudioBindingPilotSummary,
    LRASDNativeArtifact,
    SpeechActivityArtifact,
    VoiceReferenceClipDiagnostics,
)
from r2v_data_v2.h3.review import ReviewMediaBackend, write_review_bundle
from r2v_data_v2.h3.schemas import H3TaskSpecification
from r2v_data_v2.h3.voice_quality import (
    build_voice_reference_quality_diagnostics,
    build_voice_reference_quality_report,
)
from r2v_data_v2.v3.schemas import (
    ClipRecord,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _validate_roots(run_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = run_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("pilot output must be separate from source run_root")
    if not (source / "run.json").is_file() or not (source / "clips").is_dir():
        raise ValueError("source run_root is not an initialized V3 run")
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")
    return source, output


def _selected_clip_paths(
    source: Path,
    *,
    clip_ids: list[str] | None,
    limit: int | None,
) -> list[Path]:
    if limit is not None and limit <= 0:
        raise ValueError("pilot limit must be positive")
    if not clip_ids and limit is None:
        raise ValueError("pilot requires explicit clip IDs or a bounded limit")
    if clip_ids:
        unique_ids = list(dict.fromkeys(clip_ids))
        if any(
            _SAFE_COMPONENT.fullmatch(clip_uid) is None
            or clip_uid in {".", ".."}
            for clip_uid in unique_ids
        ):
            raise ValueError("pilot clip IDs must be safe path components")
        paths = [source / "clips" / clip_uid / "clip.json" for clip_uid in unique_ids]
        missing = [path.parent.name for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"pilot clip IDs are absent from the run: {missing}")
    else:
        paths = sorted((source / "clips").glob("*/clip.json"))
    return paths[:limit] if limit is not None else paths


def _load_clip_artifacts(
    clip_path: Path,
) -> tuple[ClipRecord, SampledFramesArtifact, TrackedMasksArtifact]:
    clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
    if clip.clip_uid != clip_path.parent.name:
        raise ValueError("clip.json UID does not match its directory")
    if clip.annotation is None or clip.annotation.status != "ready":
        raise ValueError("pilot clip annotation is not ready")
    if clip.coverage is None or not clip.coverage.passed:
        raise ValueError("pilot clip coverage has not passed")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise ValueError("pilot clip pairing is not ready")
    frames = SampledFramesArtifact.model_validate_json(
        (clip_path.parent / "frames" / "frames.json").read_text(encoding="utf-8")
    )
    masks = TrackedMasksArtifact.model_validate_json(
        (clip_path.parent / "masks.rle.json").read_text(encoding="utf-8")
    )
    if frames.clip_uid != clip.clip_uid or masks.clip_uid != clip.clip_uid:
        raise ValueError("pilot visual artifacts do not match clip UID")
    if not any(entity.status == "ready" for entity in masks.entities.values()):
        raise ValueError("pilot clip has no ready tracked entity masks")
    return clip, frames, masks


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _published_path(value: str, *, temporary: Path, destination: Path) -> str:
    path = Path(value).expanduser().resolve(strict=False)
    try:
        relative = path.relative_to(temporary)
    except ValueError:
        return str(path)
    return str(destination / relative)


def _published_optional_path(
    value: str | None,
    *,
    temporary: Path,
    destination: Path,
) -> str | None:
    if value is None:
        return None
    return _published_path(value, temporary=temporary, destination=destination)


def _published_native_artifact(
    native: LRASDNativeArtifact,
    *,
    temporary: Path,
    destination: Path,
) -> LRASDNativeArtifact:
    payload = native.model_dump(mode="json")
    payload.update(
        {
            "model_video_path": _published_path(
                native.model_video_path,
                temporary=temporary,
                destination=destination,
            ),
            "audio_path": _published_path(
                native.audio_path,
                temporary=temporary,
                destination=destination,
            ),
            "official_visualization_path": _published_optional_path(
                native.official_visualization_path,
                temporary=temporary,
                destination=destination,
            ),
        }
    )
    return LRASDNativeArtifact.model_validate(payload)


def _published_speech_artifact(
    speech: SpeechActivityArtifact,
    *,
    source_audio_path: str,
) -> SpeechActivityArtifact:
    payload = speech.model_dump(mode="json")
    payload["source_audio_path"] = source_audio_path
    return SpeechActivityArtifact.model_validate(payload)


@dataclass(frozen=True)
class _PilotClipResult:
    clip_uid: str
    counters: dict[str, int]
    sidecar_payload: dict[str, object] | None = None
    voice_quality_payload: dict[str, object] | None = None
    failure: dict[str, object] | None = None


@dataclass(frozen=True)
class ExplicitPilotClip:
    """One explicitly selected V3 clip, possibly from a different shard run."""

    clip_path: Path
    source_run_root: Path
    artifact_relpath: Path

    def validated(self, *, source_root: Path) -> ExplicitPilotClip:
        clip_path = self.clip_path.expanduser().resolve(strict=True)
        shard_root = self.source_run_root.expanduser().resolve(strict=True)
        clip_path.relative_to(shard_root)
        shard_root.relative_to(source_root)
        if not (shard_root / "run.json").is_file():
            raise ValueError("explicit pilot shard root is not an initialized V3 run")
        relative = Path(self.artifact_relpath)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("explicit pilot artifact path must be safe and relative")
        return ExplicitPilotClip(
            clip_path=clip_path,
            source_run_root=shard_root,
            artifact_relpath=relative,
        )


def _empty_clip_counters() -> dict[str, int]:
    return {
        "clips_succeeded": 0,
        "clips_failed": 0,
        "clips_with_speech": 0,
        "bound_intervals": 0,
        "overlap_intervals": 0,
        "offscreen_intervals": 0,
        "ambiguous_intervals": 0,
        "no_speech_intervals": 0,
        "face_entity_association_failures": 0,
        "asd_runtime_failures": 0,
    }


def _run_pilot_clip(
    *,
    clip_path: Path,
    source: Path,
    temporary: Path,
    destination: Path,
    lr_asd_backend: LRASDBackend,
    speech_backend: SpeechActivityBackend,
    review_media_backend: ReviewMediaBackend,
    association_policy: FaceEntityAssociationPolicy | None,
    binding_policy: AudioBindingPolicy | None,
    artifact_relpath: Path | None = None,
) -> _PilotClipResult:
    clip_uid = clip_path.parent.name
    counters = _empty_clip_counters()
    stage = "load_visual_evidence"
    try:
        clip, frames, masks = _load_clip_artifacts(clip_path)
        source_video = Path(clip.source.video_path).expanduser()
        if not source_video.is_absolute():
            raise ValueError("pilot source video path must be absolute")
        source_video = source_video.resolve(strict=True)
        stage = "lr_asd"
        runtime_native = lr_asd_backend.analyze(
            clip_uid=clip_uid,
            source_video_path=source_video,
            work_dir=temporary / "runtime" / clip_uid / "lr_asd",
        )
        if runtime_native.clip_uid != clip_uid:
            raise ValueError("LR-ASD artifact clip UID does not match")
        if runtime_native.source_video_path != str(source_video):
            raise ValueError("LR-ASD artifact source video does not match")
        stage = "face_entity_association"
        associations = associate_face_tracks_to_entities(
            frames=frames,
            masks=masks,
            tracks=runtime_native.tracks,
            policy=association_policy,
        )
        stage = "speech_activity"
        speech = speech_backend.detect(
            clip_uid=clip_uid,
            audio_path=Path(runtime_native.audio_path),
            duration_seconds=runtime_native.duration_seconds,
            work_dir=temporary / "runtime" / clip_uid / "vad",
        )
        native = _published_native_artifact(
            runtime_native,
            temporary=temporary,
            destination=destination,
        )
        speech = _published_speech_artifact(
            speech,
            source_audio_path=native.audio_path,
        )
        _write_json(
            temporary / "runtime" / clip_uid / "lr_asd" / "lr_asd_native.json",
            native.model_dump(mode="json"),
        )
        _write_json(
            temporary / "runtime" / clip_uid / "vad" / "speech_activity.json",
            speech.model_dump(mode="json"),
        )
        evidence = normalize_lr_asd_evidence(native, speech, associations)
        stage = "fusion"
        sidecar = build_audio_binding_sidecar(
            clip,
            evidence,
            source_run_root=str(source),
            task=H3TaskSpecification(components=["reference_generation"]),
            policy=binding_policy,
        )
        if sidecar.status != "ready":
            raise ValueError(
                f"pilot reference-generation sidecar is {sidecar.status}: "
                f"{sidecar.reason}"
            )
        try:
            voice_quality = build_voice_reference_quality_diagnostics(
                native=native,
                sidecar=sidecar,
                source_audio_path=Path(runtime_native.audio_path),
                published_audio_path=native.audio_path,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics do not gate binding
            voice_quality = VoiceReferenceClipDiagnostics(
                clip_uid=clip_uid,
                source_audio_path=native.audio_path,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
        stage = "review_bundle"
        write_review_bundle(
            destination=temporary / "review" / clip_uid,
            source_video_path=source_video,
            native=native,
            associations=associations,
            sidecar=sidecar,
            media_backend=review_media_backend,
            source_audio_path=Path(runtime_native.audio_path),
        )
        sidecar_payload = sidecar.model_dump(mode="json")
        voice_quality_payload = voice_quality.model_dump(mode="json")
        _write_json(
            temporary / "review" / clip_uid / "voice_reference_quality.json",
            voice_quality_payload,
        )
        clip_output = artifact_relpath or Path(clip_uid)
        if clip_output.is_absolute() or ".." in clip_output.parts:
            raise ValueError("pilot clip artifact path must be safe and relative")
        _write_json(temporary / "clips" / clip_output / "audio_binding.json", sidecar_payload)
        _write_json(
            temporary / "clips" / clip_output / "voice_reference_quality.json",
            voice_quality_payload,
        )
        counters["clips_succeeded"] = 1
        counters["clips_with_speech"] = int(bool(speech.intervals))
        counters["face_entity_association_failures"] = sum(
            association.status != "matched" for association in associations
        )
        for status in (
            "bound",
            "overlap",
            "offscreen",
            "ambiguous",
            "no_speech",
        ):
            counters[f"{status}_intervals"] = sum(
                binding.status == status for binding in sidecar.bindings
            )
        return _PilotClipResult(
            clip_uid=clip_uid,
            counters=counters,
            sidecar_payload=sidecar_payload,
            voice_quality_payload=voice_quality_payload,
        )
    except Exception as exc:  # noqa: BLE001 - isolate pilot clips
        counters["clips_failed"] = 1
        if isinstance(exc, LRASDRuntimeError):
            counters["asd_runtime_failures"] = 1
        if stage == "face_entity_association":
            counters["face_entity_association_failures"] = 1
        return _PilotClipResult(
            clip_uid=clip_uid,
            counters=counters,
            failure={
                "clip_uid": clip_uid,
                "stage": stage,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            },
        )


def run_h3_audio_binding_pilot(
    *,
    run_root: Path,
    output_root: Path,
    lr_asd_backend: LRASDBackend,
    speech_backend: SpeechActivityBackend,
    review_media_backend: ReviewMediaBackend,
    clip_ids: list[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    association_policy: FaceEntityAssociationPolicy | None = None,
    binding_policy: AudioBindingPolicy | None = None,
    explicit_clips: list[ExplicitPilotClip] | None = None,
) -> H3AudioBindingPilotSummary:
    if workers <= 0:
        raise ValueError("pilot workers must be positive")
    if explicit_clips is None:
        source, destination = _validate_roots(run_root, output_root)
        selected = [
            ExplicitPilotClip(
                clip_path=clip_path,
                source_run_root=source,
                artifact_relpath=Path(clip_path.parent.name),
            )
            for clip_path in _selected_clip_paths(
                source, clip_ids=clip_ids, limit=limit
            )
        ]
    else:
        if clip_ids is not None or limit is not None:
            raise ValueError("explicit pilot clips cannot be combined with ID selection")
        source = run_root.expanduser().resolve(strict=True)
        destination = output_root.expanduser().resolve(strict=False)
        if (
            destination == source
            or source in destination.parents
            or destination in source.parents
        ):
            raise ValueError("pilot output must be separate from source run_root")
        if destination.exists():
            raise FileExistsError(f"pilot output already exists: {destination}")
        selected = [item.validated(source_root=source) for item in explicit_clips]
        clip_uids = [item.clip_path.parent.name for item in selected]
        artifact_paths = [item.artifact_relpath.as_posix() for item in selected]
        if len(clip_uids) != len(set(clip_uids)):
            raise ValueError("explicit pilot clip UIDs must be unique")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("explicit pilot artifact paths must be unique")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    counters = {
        "clips_attempted": len(selected),
        "clips_succeeded": 0,
        "clips_failed": 0,
        "clips_with_speech": 0,
        "bound_intervals": 0,
        "overlap_intervals": 0,
        "offscreen_intervals": 0,
        "ambiguous_intervals": 0,
        "no_speech_intervals": 0,
        "face_entity_association_failures": 0,
        "asd_runtime_failures": 0,
    }
    failures: list[dict[str, object]] = []
    canonical_sidecars: list[dict[str, object]] = []
    voice_quality_reports: list[VoiceReferenceClipDiagnostics] = []
    try:
        temporary.mkdir()
        clip_arguments = [
            {
                "clip_path": item.clip_path,
                "source": item.source_run_root,
                "temporary": temporary,
                "destination": destination,
                "lr_asd_backend": lr_asd_backend,
                "speech_backend": speech_backend,
                "review_media_backend": review_media_backend,
                "association_policy": association_policy,
                "binding_policy": binding_policy,
                "artifact_relpath": item.artifact_relpath,
            }
            for item in selected
        ]
        if workers == 1 or not clip_arguments:
            results = [_run_pilot_clip(**values) for values in clip_arguments]
        else:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(clip_arguments))
            ) as executor:
                futures = [
                    executor.submit(_run_pilot_clip, **values)
                    for values in clip_arguments
                ]
                results = [future.result() for future in futures]
        for result in sorted(results, key=lambda item: item.clip_uid):
            for name, value in result.counters.items():
                counters[name] += value
            if result.sidecar_payload is not None:
                canonical_sidecars.append(result.sidecar_payload)
            if result.voice_quality_payload is not None:
                voice_quality_reports.append(
                    VoiceReferenceClipDiagnostics.model_validate(
                        result.voice_quality_payload
                    )
                )
            if result.failure is not None:
                failures.append(result.failure)
        summary = H3AudioBindingPilotSummary(
            source_run_root=str(source),
            output_root=str(destination),
            **counters,
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        _write_jsonl(temporary / "failures.jsonl", failures)
        _write_jsonl(temporary / "audio_bindings.jsonl", canonical_sidecars)
        voice_quality_turns = [
            turn.model_dump(mode="json")
            for report in voice_quality_reports
            for turn in report.candidate_turns
        ]
        _write_jsonl(
            temporary / "voice_reference_quality.jsonl",
            voice_quality_turns,
        )
        voice_quality_summary = build_voice_reference_quality_report(
            voice_quality_reports
        )
        _write_json(
            temporary / "voice_reference_quality_summary.json",
            voice_quality_summary.model_dump(mode="json"),
        )
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
