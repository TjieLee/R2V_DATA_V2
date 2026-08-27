from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from r2v_data_v2.h3.association import (
    FaceEntityAssociationPolicy,
    associate_face_tracks_to_entities,
)
from r2v_data_v2.h3.fusion import AudioBindingPolicy, build_audio_binding_sidecar
from r2v_data_v2.h3.laser_asd import (
    LaserASDBackend,
    LaserASDRuntimeError,
    normalize_laser_asd_evidence,
)
from r2v_data_v2.h3.lr_asd import SpeechActivityBackend
from r2v_data_v2.h3.pilot import (
    _load_clip_artifacts,
    _published_diagnostic_value,
    _published_path,
    _published_speech_artifact,
    _selected_clip_paths,
    _validate_roots,
    _write_json,
    _write_jsonl,
)
from r2v_data_v2.h3.pilot_schemas import (
    LaserASDNativeArtifact,
    LaserASDPilotSummary,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingSidecar,
    EntityFaceAssociation,
    H3TaskSpecification,
)


@dataclass(frozen=True)
class _LaserPilotClipResult:
    clip_uid: str
    counters: dict[str, int]
    sidecar_payload: dict[str, object] | None = None
    failure: dict[str, object] | None = None


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


def _published_laser_artifact(
    native: LaserASDNativeArtifact,
    *,
    temporary: Path,
    destination: Path,
) -> LaserASDNativeArtifact:
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
            "debug_visualization_path": (
                _published_path(
                    native.debug_visualization_path,
                    temporary=temporary,
                    destination=destination,
                )
                if native.debug_visualization_path is not None
                else None
            ),
        }
    )
    return LaserASDNativeArtifact.model_validate(payload)


def _write_review_bundle(
    *,
    destination: Path,
    source_video_path: Path,
    runtime_native: LaserASDNativeArtifact,
    published_native: LaserASDNativeArtifact,
    associations: list[EntityFaceAssociation],
    sidecar: AudioBindingSidecar,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_video_path, destination / "source.mp4")
    if runtime_native.debug_visualization_path is not None:
        visualization = Path(runtime_native.debug_visualization_path).resolve(
            strict=True
        )
        shutil.copyfile(visualization, destination / "visualization.mp4")
    _write_json(destination / "audio_binding.json", sidecar.model_dump(mode="json"))
    _write_json(
        destination / "laser_asd_native.json",
        published_native.model_dump(mode="json"),
    )
    _write_json(
        destination / "face_entity_association.json",
        {
            "schema_version": "r2v.h3.face_entity_association.1",
            "clip_uid": published_native.clip_uid,
            "associations": [
                association.model_dump(mode="json")
                for association in associations
            ],
        },
    )
    association_by_track = {
        association.face_track_id: association for association in associations
    }
    timeline_samples = []
    for track in published_native.tracks:
        association = association_by_track.get(track.face_track_id)
        entity_id = (
            association.entity_id
            if association is not None and association.status == "matched"
            else None
        )
        for sample in track.samples:
            binding = next(
                (
                    item
                    for item in sidecar.bindings
                    if item.start_time
                    <= sample.timestamp_seconds
                    < item.end_time
                ),
                None,
            )
            timeline_samples.append(
                {
                    "face_track_id": track.face_track_id,
                    "entity_id": entity_id,
                    "frame_index": sample.frame_index,
                    "timestamp_seconds": sample.timestamp_seconds,
                    "bbox_xyxy": list(sample.bbox_xyxy),
                    "raw_backend_score": sample.raw_backend_score,
                    "backend_native_active": sample.backend_native_active,
                    "landmark_available": sample.landmark_available,
                    "binding_status": (
                        binding.status if binding is not None else "ambiguous"
                    ),
                }
            )
    timeline_samples.sort(
        key=lambda item: (item["timestamp_seconds"], item["face_track_id"])
    )
    _write_json(
        destination / "timeline.json",
        {
            "schema_version": "r2v.h3.laser_audio_binding_review_timeline.1",
            "clip_uid": published_native.clip_uid,
            "model_fps": published_native.model_fps,
            "score_semantics": published_native.score_semantics,
            "active_decision_rule": published_native.active_decision_rule,
            "samples": timeline_samples,
            "bindings": [
                binding.model_dump(mode="json") for binding in sidecar.bindings
            ],
        },
    )


def _run_clip(
    *,
    clip_path: Path,
    source: Path,
    temporary: Path,
    destination: Path,
    laser_backend: LaserASDBackend,
    speech_backend: SpeechActivityBackend,
    association_policy: FaceEntityAssociationPolicy | None,
    binding_policy: AudioBindingPolicy | None,
) -> _LaserPilotClipResult:
    clip_uid = clip_path.parent.name
    counters = _empty_clip_counters()
    stage = "load_visual_evidence"
    try:
        clip, frames, masks = _load_clip_artifacts(clip_path)
        source_video = Path(clip.source.video_path).expanduser()
        if not source_video.is_absolute():
            raise ValueError("LASER pilot source video path must be absolute")
        source_video = source_video.resolve(strict=True)
        stage = "laser_asd"
        runtime_native = laser_backend.analyze(
            clip_uid=clip_uid,
            source_video_path=source_video,
            work_dir=temporary / "runtime" / clip_uid / "laser_asd",
        )
        if runtime_native.clip_uid != clip_uid:
            raise ValueError("LASER artifact clip UID does not match")
        if runtime_native.source_video_path != str(source_video):
            raise ValueError("LASER artifact source video does not match")
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
        native = _published_laser_artifact(
            runtime_native,
            temporary=temporary,
            destination=destination,
        )
        speech = _published_speech_artifact(speech, source_audio_path=native.audio_path)
        _write_json(
            temporary / "runtime" / clip_uid / "laser_asd" / "laser_asd_native.json",
            native.model_dump(mode="json"),
        )
        _write_json(
            temporary / "runtime" / clip_uid / "vad" / "speech_activity.json",
            speech.model_dump(mode="json"),
        )
        evidence = normalize_laser_asd_evidence(native, speech, associations)
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
                f"LASER pilot reference-generation sidecar is {sidecar.status}: "
                f"{sidecar.reason}"
            )
        sidecar_payload = sidecar.model_dump(mode="json")
        _write_json(
            temporary / "clips" / clip_uid / "audio_binding.json",
            sidecar_payload,
        )
        stage = "review_bundle"
        try:
            _write_review_bundle(
                destination=temporary / "review" / clip_uid,
                source_video_path=source_video,
                runtime_native=runtime_native,
                published_native=native,
                associations=associations,
                sidecar=sidecar,
            )
        except Exception as exc:  # noqa: BLE001 - review media is diagnostic only
            _write_json(
                temporary / "review" / clip_uid / "review_error.json",
                {
                    "clip_uid": clip_uid,
                    "error_type": type(exc).__name__,
                    "reason": str(exc).replace(str(temporary), str(destination)),
                },
            )
        counters["clips_succeeded"] = 1
        counters["clips_with_speech"] = int(bool(speech.intervals))
        counters["face_entity_association_failures"] = sum(
            association.status != "matched" for association in associations
        )
        for status in ("bound", "overlap", "offscreen", "ambiguous", "no_speech"):
            counters[f"{status}_intervals"] = sum(
                binding.status == status for binding in sidecar.bindings
            )
        return _LaserPilotClipResult(
            clip_uid=clip_uid,
            counters=counters,
            sidecar_payload=sidecar_payload,
        )
    except Exception as exc:  # noqa: BLE001 - pilot clips are isolated
        counters["clips_failed"] = 1
        if isinstance(exc, LaserASDRuntimeError):
            counters["asd_runtime_failures"] = 1
        if stage == "face_entity_association":
            counters["face_entity_association_failures"] = 1
        return _LaserPilotClipResult(
            clip_uid=clip_uid,
            counters=counters,
            failure={
                "clip_uid": clip_uid,
                "stage": stage,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            },
        )


def run_h3_laser_audio_binding_pilot(
    *,
    run_root: Path,
    output_root: Path,
    laser_backend: LaserASDBackend,
    speech_backend: SpeechActivityBackend,
    clip_ids: list[str] | None = None,
    limit: int | None = None,
    association_policy: FaceEntityAssociationPolicy | None = None,
    binding_policy: AudioBindingPolicy | None = None,
) -> LaserASDPilotSummary:
    source, destination = _validate_roots(run_root, output_root)
    selected = _selected_clip_paths(source, clip_ids=clip_ids, limit=limit)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    counters = {"clips_attempted": len(selected), **_empty_clip_counters()}
    failures: list[dict[str, object]] = []
    sidecars: list[dict[str, object]] = []
    try:
        temporary.mkdir(parents=True)
        for clip_path in selected:
            result = _run_clip(
                clip_path=clip_path,
                source=source,
                temporary=temporary,
                destination=destination,
                laser_backend=laser_backend,
                speech_backend=speech_backend,
                association_policy=association_policy,
                binding_policy=binding_policy,
            )
            for name, value in result.counters.items():
                counters[name] += value
            if result.sidecar_payload is not None:
                sidecars.append(result.sidecar_payload)
            if result.failure is not None:
                failures.append(result.failure)
        sidecars.sort(key=lambda item: str(item["clip_uid"]))
        failures.sort(key=lambda item: str(item["clip_uid"]))
        summary = LaserASDPilotSummary(
            source_run_root=str(source),
            output_root=str(destination),
            **counters,
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        _write_jsonl(temporary / "audio_bindings.jsonl", sidecars)
        _write_jsonl(
            temporary / "failures.jsonl",
            [
                _published_diagnostic_value(
                    failure,
                    temporary=temporary,
                    destination=destination,
                )
                for failure in failures
            ],
        )
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
