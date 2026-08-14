from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from r2v_data_v2.h3.audio_backends import (
    AudioMediaBackend,
    FaceEmbeddingBackend,
    IdentityTextEmbeddingBackend,
    SpeakerEmbeddingBackend,
)
from r2v_data_v2.h3.audio_schemas import (
    AudioClipBinding,
    AudioDatasetManifest,
    EmbeddingAsset,
    EntityOccurrence,
    FileAsset,
    FullAudioArtifact,
    LocalBindingSummary,
    ProducerProvenance,
    SourceVideoProvenance,
    SpeechTurn,
    TranscriptProvenance,
    VisualReferenceProvenance,
    VisualSourceProvenance,
    VoiceReferenceArtifact,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, AudioEntityBinding


@dataclass(frozen=True)
class AudioBindingProductionConfig:
    speech_merge_gap_seconds: float = 0.08
    minimum_voice_reference_duration_seconds: float = 0.50
    full_audio_sample_rate_hz: int = 48000
    full_audio_channels: int = 2
    full_audio_format: str = "flac"
    voice_sample_rate_hz: int = 16000
    voice_format: str = "flac"
    minimum_binding_confidence: float = 0.80

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.speech_merge_gap_seconds)
            or self.speech_merge_gap_seconds < 0
        ):
            raise ValueError("speech merge gap must be non-negative")
        if (
            not math.isfinite(self.minimum_voice_reference_duration_seconds)
            or self.minimum_voice_reference_duration_seconds <= 0
        ):
            raise ValueError("minimum voice-reference duration must be positive")
        if self.full_audio_sample_rate_hz <= 0 or self.voice_sample_rate_hz <= 0:
            raise ValueError("audio sample rates must be positive")
        if self.full_audio_channels <= 0:
            raise ValueError("full audio channels must be positive")
        if self.full_audio_format not in {"flac", "wav"} or self.voice_format not in {
            "flac",
            "wav",
        }:
            raise ValueError("audio output formats must be flac or wav")
        if not 0 <= self.minimum_binding_confidence <= 1:
            raise ValueError("binding confidence must be in [0, 1]")

    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TranscriptSegment:
    start_time: float
    end_time: float
    text: str
    status: str = "auto"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start_time)
            or not math.isfinite(self.end_time)
            or self.start_time < 0
            or self.end_time <= self.start_time
        ):
            raise ValueError("transcript segment interval is invalid")
        if not self.text.strip() or self.status not in {"auto", "reviewed"}:
            raise ValueError("transcript segment text/status is invalid")


@dataclass(frozen=True)
class VisualEntityInput:
    entity_id: str
    reference_type: str
    phrase: str
    grounding_prompt: str
    token: str
    reference_scope: str
    visible_region: str
    image_path: Path
    source_frame_index: int | None
    source_clip_uid: str | None
    source_entity_id: str | None
    synthetic: bool


@dataclass(frozen=True)
class VisualClipInput:
    clip_uid: str
    sample_id: str
    parent_video_id: str
    clip_suffix: str
    source_video_path: Path
    t2v_caption: str
    r2v_instruction: str
    visual_sample_sha256: str
    entities: tuple[VisualEntityInput, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_asset(path: Path, root: Path, media_type: str) -> FileAsset:
    relative = path.relative_to(root).as_posix()
    return FileAsset(
        path=relative,
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        media_type=media_type,
    )


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    if flattened.size == 0 or not np.isfinite(flattened).all():
        raise ValueError("embedding must be a finite non-empty vector")
    norm = float(np.linalg.norm(flattened))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding norm must be positive")
    return np.ascontiguousarray(flattened / norm, dtype=np.float32)


def _save_embedding(
    *,
    vector: np.ndarray,
    path: Path,
    root: Path,
    model_identifier: str,
    checkpoint_sha256: str | None,
    backend_metadata: dict[str, object] | None,
) -> EmbeddingAsset:
    normalized = _normalize_embedding(vector)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        np.save(stream, normalized, allow_pickle=False)
    temporary.replace(path)
    file = _file_asset(path, root, "application/x-npy")
    return EmbeddingAsset(
        **file.model_dump(mode="python"),
        model_identifier=model_identifier,
        checkpoint_sha256=checkpoint_sha256,
        dimension=int(normalized.size),
        backend_metadata=backend_metadata or {},
    )


def _frame_range(binding: AudioEntityBinding, frame_rate: float) -> tuple[int, int]:
    return (
        max(0, round(binding.start_time * frame_rate)),
        max(0, math.ceil(binding.end_time * frame_rate) - 1),
    )


def _turn_text(
    start_time: float,
    end_time: float,
    transcript_segments: Sequence[TranscriptSegment],
) -> tuple[str | None, str]:
    matches = [
        item
        for item in transcript_segments
        if item.start_time < end_time and item.end_time > start_time
    ]
    if not matches:
        return None, "missing"
    text = " ".join(item.text.strip() for item in matches if item.text.strip()).strip()
    if not text:
        return None, "missing"
    status = "reviewed" if all(item.status == "reviewed" for item in matches) else "auto"
    return text, status


def coalesce_audio_bindings(
    bindings: Sequence[AudioEntityBinding],
    *,
    clip_uid: str,
    sample_rate_hz: int,
    maximum_gap_seconds: float,
    minimum_voice_reference_duration_seconds: float,
    frame_rate: float = 25.0,
    transcript_segments: Sequence[TranscriptSegment] = (),
) -> list[SpeechTurn]:
    if sample_rate_hz <= 0 or frame_rate <= 0:
        raise ValueError("speech turn sample/frame rates must be positive")
    if maximum_gap_seconds < 0 or minimum_voice_reference_duration_seconds <= 0:
        raise ValueError("speech turn merge/duration policy is invalid")
    ordered = sorted(bindings, key=lambda item: (item.start_time, item.end_time))
    if list(bindings) != ordered:
        raise ValueError("frame-level audio bindings must be chronologically ordered")
    if any(
        ordered[index].end_time > ordered[index + 1].start_time
        for index in range(len(ordered) - 1)
    ):
        raise ValueError("frame-level audio bindings must not overlap")

    groups: list[list[tuple[int, AudioEntityBinding]]] = []
    for index, binding in enumerate(ordered, start=1):
        if not groups:
            groups.append([(index, binding)])
            continue
        previous = groups[-1][-1][1]
        mergeable = (
            binding.status == "bound"
            and previous.status == "bound"
            and binding.entity_id == previous.entity_id
            and binding.face_track_id == previous.face_track_id
            and binding.evidence.clean_training_eligible
            == previous.evidence.clean_training_eligible
            and binding.start_time - previous.end_time <= maximum_gap_seconds
        )
        if mergeable:
            groups[-1].append((index, binding))
        else:
            groups.append([(index, binding)])

    turns: list[SpeechTurn] = []
    for turn_index, group in enumerate(groups, start=1):
        first = group[0][1]
        last = group[-1][1]
        frame_ranges = [_frame_range(binding, frame_rate) for _, binding in group]
        text, text_status = _turn_text(
            first.start_time,
            last.end_time,
            transcript_segments,
        )
        duration = last.end_time - first.start_time
        voice_eligible = (
            first.status == "bound"
            and all(binding.evidence.clean_training_eligible for _, binding in group)
            and duration >= minimum_voice_reference_duration_seconds
        )
        turns.append(
            SpeechTurn(
                turn_id=f"turn_{turn_index}",
                clip_uid=clip_uid,
                start_time=first.start_time,
                end_time=last.end_time,
                start_sample=round(first.start_time * sample_rate_hz),
                end_sample=round(last.end_time * sample_rate_hz),
                start_frame=frame_ranges[0][0],
                end_frame=frame_ranges[-1][1],
                entity_id=first.entity_id,
                face_track_id=first.face_track_id,
                status=first.status,
                binding_confidence=min(binding.confidence for _, binding in group),
                source_binding_ids=[f"binding_{index}" for index, _ in group],
                source_frame_ranges=frame_ranges,
                voice_reference_eligible=voice_eligible,
                text=text,
                text_status=text_status,
            )
        )
    return turns


def select_primary_voice_turns(
    turns: Sequence[SpeechTurn],
    *,
    entity_order: Sequence[str],
    minimum_binding_confidence: float,
) -> dict[str, SpeechTurn]:
    selected: dict[str, SpeechTurn] = {}
    for entity_id in entity_order:
        eligible = [
            turn
            for turn in turns
            if turn.entity_id == entity_id
            and turn.voice_reference_eligible
            and turn.binding_confidence >= minimum_binding_confidence
        ]
        if eligible:
            selected[entity_id] = min(
                eligible,
                key=lambda item: (
                    -item.binding_confidence,
                    -(item.end_time - item.start_time),
                    item.start_time,
                    item.turn_id,
                ),
            )
    return selected


def load_visual_clip_inputs(
    *,
    run_root: Path,
    visual_export_root: Path,
    clip_allowlist: set[str] | None = None,
    limit: int | None = None,
) -> list[VisualClipInput]:
    source = run_root.resolve(strict=True)
    export = visual_export_root.resolve(strict=True)
    samples_path = export / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError("visual export samples.jsonl is missing")
    records = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[VisualClipInput] = []
    for sample in records:
        clip_uid = str(sample["sample_id"])
        if clip_allowlist is not None and clip_uid not in clip_allowlist:
            continue
        clip_path = source / "clips" / clip_uid / "clip.json"
        raw = json.loads(clip_path.read_text(encoding="utf-8"))
        annotation_entities = {
            item["entity_id"]: item
            for item in (raw.get("annotation") or {}).get("entities", [])
        }
        retained = list((raw.get("pairing") or {}).get("retained_entity_ids", []))
        references = {
            item["entity_id"]: item
            for item in sample.get("references", [])
            if item.get("type") == "entity"
        }
        entities: list[VisualEntityInput] = []
        for entity_id in retained:
            reference = references.get(entity_id)
            entity = annotation_entities.get(entity_id)
            if reference is None or entity is None:
                raise ValueError("visual export and retained entities are inconsistent")
            image_path = (export / reference["image_path"]).resolve(strict=True)
            image_path.relative_to(export)
            entities.append(
                VisualEntityInput(
                    entity_id=entity_id,
                    reference_type=str(entity["reference_type"]),
                    phrase=str(entity["phrase"]),
                    grounding_prompt=str(entity["grounding_prompt"]),
                    token=str(reference["token"]),
                    reference_scope=str(reference["scope"]),
                    visible_region=str(reference["visible_region"]),
                    image_path=image_path,
                    source_frame_index=reference.get("source_frame_index"),
                    source_clip_uid=reference.get("source_clip_uid"),
                    source_entity_id=reference.get("source_entity_id"),
                    synthetic=bool(reference.get("synthetic", False)),
                )
            )
        serialized_sample = json.dumps(
            sample, sort_keys=True, separators=(",", ":")
        ).encode()
        selected.append(
            VisualClipInput(
                clip_uid=clip_uid,
                sample_id=clip_uid,
                parent_video_id=str(sample["source"]["parent_video_id"]),
                clip_suffix=str(sample["source"]["clip_suffix"]),
                source_video_path=Path(sample["target_video"]).resolve(strict=True),
                t2v_caption=str(sample["t2v_caption"]),
                r2v_instruction=str(sample["r2v_instruction"]),
                visual_sample_sha256=hashlib.sha256(serialized_sample).hexdigest(),
                entities=tuple(entities),
            )
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _copy_visual_reference(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
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
        raise FileExistsError(f"audio export already exists: {destination}")
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


def _build_audio_clip_binding_batch(
    *,
    run_root: Path,
    visual_export_root: Path,
    sidecar_root: Path,
    output_root: Path,
    audio_backend: AudioMediaBackend,
    face_backend: FaceEmbeddingBackend,
    speaker_backend: SpeakerEmbeddingBackend,
    text_backend: IdentityTextEmbeddingBackend | None = None,
    transcript_by_clip: dict[str, Sequence[TranscriptSegment]] | None = None,
    config: AudioBindingProductionConfig | None = None,
    clip_allowlist: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    visual_inputs: Sequence[VisualClipInput] | None = None,
) -> list[AudioClipBinding]:
    active = config or AudioBindingProductionConfig()
    source_run = run_root.resolve(strict=True)
    visual_export = visual_export_root.resolve(strict=True)
    sidecar = sidecar_root.resolve(strict=True)
    destination = output_root.resolve(strict=False)
    for source_root in (source_run, visual_export, sidecar):
        if (
            destination == source_root
            or source_root in destination.parents
            or destination in source_root.parents
        ):
            raise ValueError("audio output must be outside every source root")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"audio export already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    producer = ProducerProvenance(
        producer="r2v_data_v2.h3.audio_binding",
        version="v1",
        config_fingerprint=active.fingerprint(),
    )
    outputs: list[AudioClipBinding] = []
    try:
        active_visual_inputs = (
            list(visual_inputs)
            if visual_inputs is not None
            else load_visual_clip_inputs(
                run_root=source_run,
                visual_export_root=visual_export,
                clip_allowlist=clip_allowlist,
                limit=limit,
            )
        )
        for visual in active_visual_inputs:
            sidecar_path = sidecar / "clips" / visual.clip_uid / "audio_binding.json"
            sidecar_record = AudioBindingSidecar.model_validate_json(
                sidecar_path.read_text(encoding="utf-8")
            )
            if sidecar_record.status != "ready" or sidecar_record.evidence is None:
                continue
            full_path = temporary / "full_audio" / (
                f"{visual.clip_uid}.{active.full_audio_format}"
            )
            full = audio_backend.materialize_full_audio(
                clip_uid=visual.clip_uid,
                source_video_path=visual.source_video_path,
                destination=full_path,
                sample_rate_hz=active.full_audio_sample_rate_hz,
                channels=active.full_audio_channels,
                output_format=active.full_audio_format,
            )
            turns = coalesce_audio_bindings(
                sidecar_record.bindings,
                clip_uid=visual.clip_uid,
                sample_rate_hz=full.sample_rate_hz,
                maximum_gap_seconds=active.speech_merge_gap_seconds,
                minimum_voice_reference_duration_seconds=(
                    active.minimum_voice_reference_duration_seconds
                ),
                transcript_segments=(transcript_by_clip or {}).get(
                    visual.clip_uid, ()
                ),
            )
            selected_turns = select_primary_voice_turns(
                turns,
                entity_order=[entity.entity_id for entity in visual.entities],
                minimum_binding_confidence=active.minimum_binding_confidence,
            )
            diagnostics_path = (
                temporary
                / "diagnostics"
                / visual.clip_uid
                / "frame_bindings.json"
            )
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sidecar_path, diagnostics_path)
            occurrences: list[EntityOccurrence] = []
            for entity in visual.entities:
                occurrence_id = f"{visual.clip_uid}/{entity.entity_id}"
                visual_destination = (
                    temporary
                    / "visual_references"
                    / visual.clip_uid
                    / f"{entity.entity_id}.png"
                )
                _copy_visual_reference(entity.image_path, visual_destination)
                visual_asset = _file_asset(
                    visual_destination,
                    temporary,
                    "image/png",
                )
                bound_turns = [turn for turn in turns if turn.entity_id == entity.entity_id]
                best_confidence = max(
                    (turn.binding_confidence for turn in bound_turns),
                    default=0.0,
                )
                face_status = "unavailable"
                face_crop_asset = None
                face_embedding_asset = None
                reason_codes: list[str] = []
                if bound_turns and entity.reference_type == "subject":
                    face_result = face_backend.embed_face(
                        entity_occurrence_id=occurrence_id,
                        image_path=entity.image_path,
                    )
                    face_status = face_result.status
                    if (
                        face_result.status == "available"
                        and face_result.face_crop is not None
                        and face_result.embedding is not None
                    ):
                        crop_path = (
                            temporary
                            / "face_crops"
                            / visual.clip_uid
                            / f"{entity.entity_id}.png"
                        )
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        face_result.face_crop.convert("RGB").save(crop_path, format="PNG")
                        face_crop_asset = _file_asset(crop_path, temporary, "image/png")
                        face_embedding_asset = _save_embedding(
                            vector=face_result.embedding.vector,
                            path=(
                                temporary
                                / "embeddings"
                                / "face"
                                / visual.clip_uid
                                / f"{entity.entity_id}.npy"
                            ),
                            root=temporary,
                            model_identifier=face_result.embedding.model_identifier,
                            checkpoint_sha256=face_result.embedding.checkpoint_sha256,
                            backend_metadata=face_result.embedding.backend_metadata,
                        )
                    else:
                        reason_codes.append("face_identity_unavailable")
                else:
                    reason_codes.append("not_a_bound_human_subject")
                voice_reference = None
                voice_embedding = None
                selected_turn = selected_turns.get(entity.entity_id)
                if selected_turn is not None and entity.reference_type == "subject":
                    voice_path = (
                        temporary
                        / "voice_refs"
                        / visual.clip_uid
                        / entity.entity_id
                        / f"voice_ref_1.{active.voice_format}"
                    )
                    audio_backend.extract_voice_reference(
                        clip_uid=visual.clip_uid,
                        entity_id=entity.entity_id,
                        full_audio_path=full.path,
                        start_time=selected_turn.start_time,
                        end_time=selected_turn.end_time,
                        destination=voice_path,
                        sample_rate_hz=active.voice_sample_rate_hz,
                        output_format=active.voice_format,
                    )
                    voice_reference = VoiceReferenceArtifact(
                        voice_reference_id="voice_ref_1",
                        entity_occurrence_id=occurrence_id,
                        source_turn_id=selected_turn.turn_id,
                        source_start=selected_turn.start_time,
                        source_end=selected_turn.end_time,
                        source_start_sample=selected_turn.start_sample,
                        source_end_sample=selected_turn.end_sample,
                        asset=_file_asset(
                            voice_path,
                            temporary,
                            "audio/flac"
                            if active.voice_format == "flac"
                            else "audio/wav",
                        ),
                        quality_score=selected_turn.binding_confidence,
                        quality_metadata={"selection_policy": "confidence_duration_time_v1"},
                    )
                    speaker = speaker_backend.embed_speaker(
                        entity_occurrence_id=occurrence_id,
                        audio_path=voice_path,
                    )
                    voice_embedding = _save_embedding(
                        vector=speaker.vector,
                        path=(
                            temporary
                            / "embeddings"
                            / "voice"
                            / visual.clip_uid
                            / f"{entity.entity_id}.npy"
                        ),
                        root=temporary,
                        model_identifier=speaker.model_identifier,
                        checkpoint_sha256=speaker.checkpoint_sha256,
                        backend_metadata=speaker.backend_metadata,
                    )
                else:
                    reason_codes.append("primary_voice_reference_unavailable")
                text_embedding = None
                identity_text = entity.phrase
                identity_text_specific = len(identity_text.split()) >= 3
                if text_backend is not None and identity_text_specific:
                    text_result = text_backend.embed_text(
                        entity_occurrence_id=occurrence_id,
                        text=identity_text,
                    )
                    text_embedding = _save_embedding(
                        vector=text_result.vector,
                        path=(
                            temporary
                            / "embeddings"
                            / "text"
                            / visual.clip_uid
                            / f"{entity.entity_id}.npy"
                        ),
                        root=temporary,
                        model_identifier=text_result.model_identifier,
                        checkpoint_sha256=text_result.checkpoint_sha256,
                        backend_metadata=text_result.backend_metadata,
                    )
                in_pair_eligible = voice_reference is not None
                cross_pair_eligible = (
                    in_pair_eligible
                    and face_embedding_asset is not None
                    and voice_embedding is not None
                )
                occurrences.append(
                    EntityOccurrence(
                        entity_occurrence_id=occurrence_id,
                        entity_id=entity.entity_id,
                        reference_type=entity.reference_type,
                        phrase=entity.phrase,
                        grounding_prompt=entity.grounding_prompt,
                        identity_text=identity_text,
                        identity_text_specific=identity_text_specific,
                        visual_reference=VisualReferenceProvenance(
                            image_asset=visual_asset,
                            token=entity.token,
                            reference_scope=entity.reference_scope,
                            visible_region=entity.visible_region,
                            source_frame_index=entity.source_frame_index,
                            source_clip_uid=entity.source_clip_uid,
                            source_entity_id=entity.source_entity_id,
                            synthetic=entity.synthetic,
                        ),
                        face_identity_status=face_status,
                        face_crop_asset=face_crop_asset,
                        face_embedding_asset=face_embedding_asset,
                        identity_text_embedding_asset=text_embedding,
                        primary_voice_reference=voice_reference,
                        voice_embedding_asset=voice_embedding,
                        local_binding_summary=LocalBindingSummary(
                            bound_turn_ids=[turn.turn_id for turn in bound_turns],
                            total_bound_seconds=sum(
                                turn.end_time - turn.start_time for turn in bound_turns
                            ),
                            best_binding_confidence=best_confidence,
                            high_confidence_bound=(
                                best_confidence >= active.minimum_binding_confidence
                            ),
                        ),
                        in_pair_eligible=in_pair_eligible,
                        cross_pair_eligible=cross_pair_eligible,
                        reason_codes=sorted(set(reason_codes)),
                    )
                )
            transcript_segments = list(
                (transcript_by_clip or {}).get(visual.clip_uid, ())
            )
            transcript_status = (
                "missing"
                if not transcript_segments
                else "reviewed"
                if all(segment.status == "reviewed" for segment in transcript_segments)
                else "precomputed"
            )
            outputs.append(
                AudioClipBinding(
                    clip_binding_id=f"clip_binding/{visual.clip_uid}",
                    clip_uid=visual.clip_uid,
                    sample_id=visual.sample_id,
                    parent_video_id=visual.parent_video_id,
                    clip_suffix=visual.clip_suffix,
                    source_video=SourceVideoProvenance(
                        path=str(visual.source_video_path),
                        sha256=_sha256(visual.source_video_path),
                    ),
                    visual_source=VisualSourceProvenance(
                        run_root=str(source_run),
                        export_root=str(visual_export),
                        visual_sample_id=visual.sample_id,
                        visual_sample_sha256=visual.visual_sample_sha256,
                    ),
                    full_audio=FullAudioArtifact(
                        asset=_file_asset(
                            full.path,
                            temporary,
                            "audio/flac"
                            if active.full_audio_format == "flac"
                            else "audio/wav",
                        ),
                        stream=full.stream,
                        output_sample_rate_hz=full.sample_rate_hz,
                        output_channels=full.channels,
                        output_format=active.full_audio_format,
                    ),
                    raw_frame_bindings_path=diagnostics_path.relative_to(
                        temporary
                    ).as_posix(),
                    speech_turns=turns,
                    entity_occurrences=occurrences,
                    transcript_provenance=TranscriptProvenance(
                        backend=("precomputed" if transcript_status != "missing" else None),
                        status=transcript_status,
                    ),
                    visual_t2v_caption=visual.t2v_caption,
                    visual_r2v_instruction=visual.r2v_instruction,
                    producer_provenance=producer,
                )
            )
        outputs.sort(key=lambda item: item.clip_uid)
        _write_jsonl(temporary / "clip_bindings.jsonl", outputs)
        (temporary / "pair_samples.jsonl").write_text("", encoding="utf-8")
        (temporary / "pair_report.json").write_text(
            json.dumps({"pair_sample_count": 0}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = AudioDatasetManifest(
            clip_binding_count=len(outputs),
            pair_sample_count=0,
            producer_provenance=producer,
        )
        (temporary / "dataset.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return outputs
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_audio_clip_binding_dataset(
    *,
    run_root: Path,
    visual_export_root: Path,
    sidecar_root: Path,
    output_root: Path,
    audio_backend: AudioMediaBackend,
    face_backend: FaceEmbeddingBackend,
    speaker_backend: SpeakerEmbeddingBackend,
    text_backend: IdentityTextEmbeddingBackend | None = None,
    transcript_by_clip: dict[str, Sequence[TranscriptSegment]] | None = None,
    config: AudioBindingProductionConfig | None = None,
    clip_allowlist: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[AudioClipBinding]:
    active = config or AudioBindingProductionConfig()
    source_run = run_root.resolve(strict=True)
    visual_export = visual_export_root.resolve(strict=True)
    sidecar = sidecar_root.resolve(strict=True)
    destination = output_root.resolve(strict=False)
    for source_root in (source_run, visual_export, sidecar):
        if (
            destination == source_root
            or source_root in destination.parents
            or destination in source_root.parents
        ):
            raise ValueError("audio output must be outside every source root")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"audio export already exists: {destination}")
    visual_inputs = load_visual_clip_inputs(
        run_root=source_run,
        visual_export_root=visual_export,
        clip_allowlist=clip_allowlist,
        limit=limit,
    )
    staging = destination.with_name(f".{destination.name}.clips-{uuid.uuid4().hex}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    producer = ProducerProvenance(
        producer="r2v_data_v2.h3.audio_binding",
        version="v1",
        config_fingerprint=active.fingerprint(),
    )
    outputs: list[AudioClipBinding] = []
    failures: list[dict[str, object]] = []
    successful_roots: list[Path] = []
    try:
        staging.mkdir(parents=True)
        for visual in visual_inputs:
            clip_root = staging / visual.clip_uid
            try:
                clip_outputs = _build_audio_clip_binding_batch(
                    run_root=source_run,
                    visual_export_root=visual_export,
                    sidecar_root=sidecar,
                    output_root=clip_root,
                    audio_backend=audio_backend,
                    face_backend=face_backend,
                    speaker_backend=speaker_backend,
                    text_backend=text_backend,
                    transcript_by_clip=transcript_by_clip,
                    config=active,
                    overwrite=False,
                    visual_inputs=[visual],
                )
            except Exception as exc:  # noqa: BLE001 - isolate one source clip
                failures.append(
                    {
                        "clip_uid": visual.clip_uid,
                        "failure_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                )
                continue
            outputs.extend(clip_outputs)
            successful_roots.append(clip_root)

        temporary.mkdir(parents=True)
        root_metadata = {
            "clip_bindings.jsonl",
            "dataset.json",
            "pair_samples.jsonl",
            "pair_report.json",
        }
        for clip_root in successful_roots:
            for child in clip_root.iterdir():
                if child.name in root_metadata:
                    continue
                target = temporary / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(child, target)
        outputs.sort(key=lambda item: item.clip_uid)
        _write_jsonl(temporary / "clip_bindings.jsonl", outputs)
        (temporary / "pair_samples.jsonl").write_text("", encoding="utf-8")
        (temporary / "pair_report.json").write_text(
            json.dumps({"pair_sample_count": 0}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "failures.jsonl").write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in failures
            ),
            encoding="utf-8",
        )
        (temporary / "dataset.json").write_text(
            AudioDatasetManifest(
                clip_binding_count=len(outputs),
                failed_clip_count=len(failures),
                pair_sample_count=0,
                producer_provenance=producer,
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return outputs
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_clip_bindings(path: Path) -> list[AudioClipBinding]:
    return [
        AudioClipBinding.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
