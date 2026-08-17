from __future__ import annotations

import hashlib
import inspect
import json
import math
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.h3 import lr_asd as lr_asd_module
from r2v_data_v2.h3.association import (
    FaceEntityAssociationPolicy,
    associate_face_tracks_to_entities,
    nearest_track_sample,
)
from r2v_data_v2.h3.audio_backends import (
    PrecomputedAudioMediaBackend,
    PrecomputedEmbeddingBackend,
)
from r2v_data_v2.h3.audio_binding import build_audio_clip_binding_dataset
from r2v_data_v2.h3.audio_schemas import AudioStreamProvenance
from r2v_data_v2.h3.backends import (
    EntityFaceAssociationBackend,
    PrecomputedEvidenceBackend,
)
from r2v_data_v2.h3.fusion import (
    build_audio_binding_sidecar,
    fuse_audio_entity_bindings,
    render_h3_audio_instruction,
)
from r2v_data_v2.h3.lr_asd import (
    LRASDRuntimeConfig,
    LRASDRuntimeError,
    LRASDSubprocessBackend,
    PrecomputedLRASDBackend,
    PrecomputedSpeechActivityBackend,
    SileroVADRuntimeConfig,
    SileroVADSubprocessBackend,
    normalize_lr_asd_evidence,
)
from r2v_data_v2.h3.pilot import run_h3_audio_binding_pilot
from r2v_data_v2.h3.pilot_schemas import (
    LRASDNativeArtifact,
    LRASDNativeSample,
    LRASDNativeTrack,
    SpeechActivityArtifact,
    SpeechActivityInterval,
)
from r2v_data_v2.h3.review import ExternalReviewMediaBackend, build_review_timeline
from r2v_data_v2.h3.schemas import (
    ActiveSpeakerInterval,
    ASDModelProvenance,
    AudioBindingEvidence,
    AudioEntityBinding,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceGeometrySample,
    FaceTrack,
    H3AudioBindingIR,
    H3TaskSpecification,
    PrecomputedClipEvidence,
    PrecomputedEvidenceFile,
    VoiceReference,
    VoiceReferenceCandidate,
)
from r2v_data_v2.h3.sidecar import build_audio_binding_sidecar_run
from r2v_data_v2.h3.voice_quality import (
    _local_noise_metrics,
    recompute_voice_reference_quality_artifacts,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipRecord,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    PairingState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from run_pipeline_v3 import STAGE_ORDER
from tools.build_v3_h3_audio_binding_sidecar import main as sidecar_main
from tools.eval_h3_audio_binding_lr_asd import _parser as lr_asd_pilot_parser
from tools.recompute_h3_voice_quality import main as recompute_voice_quality_main


def _visibility() -> EntityVisibilitySummary:
    return EntityVisibilitySummary(
        status="ready",
        visible_frame_slots=list(range(10)),
        visible_frame_count=10,
        coverage_ratio=1.0,
        qualifies=True,
        per_frame_area_ratio=[0.1] * 10,
        per_frame_confidence=[0.95] * 10,
    )


def _clip(
    entity_count: int = 2,
    *,
    reference_types: list[str] | None = None,
) -> ClipRecord:
    active_types = reference_types or ["subject"] * entity_count
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type=active_types[index - 1],
            phrase=f"person {index}",
            grounding_prompt=f"person {index}",
        )
        for index in range(1, entity_count + 1)
    ]
    references = [
        EntityReferenceState(
            entity_id=entity.entity_id,
            status="ready",
            reference_scope="full",
            visible_region="whole",
            whole_entity_recognizable=True,
            identity_features_visible=True,
            scope_reason="test fixture",
            image_path=f"clips/clip-1/selected/{entity.entity_id}.png",
            source_frame_index=index,
            source_clip_uid="clip-1",
            source_entity_id=entity.entity_id,
            image_quality="high",
            completeness="complete",
            synthetic=False,
        )
        for index, entity in enumerate(entities)
    ]
    retained = [entity.entity_id for entity in entities]
    token_counters = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity in entities:
        token_counters[entity.reference_type] += 1
        tokens[entity.entity_id] = (
            f"<ref_{entity.reference_type}_{token_counters[entity.reference_type]}>"
        )
    return ClipRecord(
        clip_uid="clip-1",
        source=ClipSource(
            video_path="/public/video.mp4",
            parent_video_id="parent",
            clip_suffix="1",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
        annotation=AnnotationState(
            status="ready",
            instruction_template=" and ".join(
                f"{{{{entity_{index}}}}}" for index in range(1, entity_count + 1)
            ),
            entities=entities,
        ),
        coverage=CoverageState(
            passed=True,
            qualifying_entity_ids=retained,
            required_visible_frames=7,
            entity_visibility_summary={
                entity_id: _visibility() for entity_id in retained
            },
        ),
        references=ReferencesState(entities=references),
        pairing=PairingState(
            status="ready",
            retained_entity_ids=retained,
            tokens=tokens,
        ),
    )


def _audio(status: str = "ready") -> AudioTrackMetadata:
    if status == "ready":
        return AudioTrackMetadata(
            status="ready",
            source_video_path="/public/video.mp4",
            full_audio_path="audio/full.wav",
            duration_seconds=10.0,
            sample_rate_hz=16000,
            channels=1,
            quality_score=0.95,
        )
    return AudioTrackMetadata(
        status=status,
        source_video_path="/public/video.mp4",
        reason=f"{status} audio fixture",
    )


def _track(index: int) -> FaceTrack:
    return FaceTrack(
        face_track_id=f"face_{index}",
        start_time=0.0,
        end_time=10.0,
        sample_count=20,
        mean_detection_confidence=0.95,
        geometry_samples=[
            FaceGeometrySample(
                frame_index=0,
                timestamp=0.0,
                bbox_xyxy=(10.0, 20.0, 30.0, 50.0),
                confidence=0.96,
            ),
            FaceGeometrySample(
                frame_index=9,
                timestamp=9.0,
                bbox_xyxy=(12.0, 20.0, 32.0, 50.0),
                confidence=0.94,
            ),
        ],
    )


def _association(index: int) -> EntityFaceAssociation:
    return EntityFaceAssociation(
        face_track_id=f"face_{index}",
        entity_id=f"e{index}",
        confidence=0.95,
        method="test_fixture",
    )


def _interval(
    start: float,
    end: float,
    probabilities: dict[str, float],
    *,
    speech: bool = True,
    visible_face_track_ids: list[str] | None = None,
) -> ActiveSpeakerInterval:
    visible = (
        list(probabilities)
        if visible_face_track_ids is None
        else visible_face_track_ids
    )
    return ActiveSpeakerInterval(
        start_time=start,
        end_time=end,
        speech_present=speech,
        visible_face_track_ids=visible,
        face_speaking_probabilities=probabilities,
        asd_coverage_ratio=len(probabilities) / len(visible) if visible else 1.0,
        audio_quality_usable=True,
        synchronization_plausible=True,
    )


def _evidence(
    intervals: list[ActiveSpeakerInterval],
    *,
    associations: list[EntityFaceAssociation] | None = None,
    audio_status: str = "ready",
) -> AudioBindingEvidence:
    return AudioBindingEvidence(
        clip_uid="clip-1",
        audio=_audio(audio_status),
        face_tracks=[_track(1), _track(2)],
        associations=(
            [_association(1), _association(2)] if associations is None else associations
        ),
        active_speaker_intervals=intervals,
    )


def _voice(
    start: float, end: float, *, path: str = "audio/voice.wav"
) -> VoiceReferenceCandidate:
    return VoiceReferenceCandidate(
        path=path,
        source_start=start,
        source_end=end,
        quality_score=0.9,
        quality_metadata={"snr_db": 24.0},
    )


class _FakeVoiceReferenceBackend:
    def __init__(self, candidates: Sequence[VoiceReferenceCandidate]) -> None:
        self.candidates = list(candidates)
        self.calls: list[list[tuple[str | None, str]]] = []

    def extract(
        self,
        *,
        clip_uid: str,
        audio: AudioTrackMetadata,
        clean_bindings: Sequence[AudioEntityBinding],
    ) -> Sequence[VoiceReferenceCandidate]:
        assert clip_uid == "clip-1"
        assert audio.status == "ready"
        summarized = [(binding.entity_id, binding.status) for binding in clean_bindings]
        self.calls.append(summarized)
        return self.candidates


def _fuse(evidence: AudioBindingEvidence):
    return fuse_audio_entity_bindings(
        evidence,
        known_entity_ids={"e1", "e2"},
    )


def test_one_visible_person_speaking_binds_to_entity() -> None:
    bindings = _fuse(_evidence([_interval(1.0, 2.0, {"face_1": 0.95})]))

    assert [(item.status, item.entity_id) for item in bindings] == [("bound", "e1")]
    assert bindings[0].evidence.clean_training_eligible is True


def test_two_visible_people_only_e1_speaks() -> None:
    binding = _fuse(_evidence([_interval(1.0, 2.0, {"face_1": 0.96, "face_2": 0.03})]))[
        0
    ]

    assert binding.status == "bound"
    assert binding.entity_id == "e1"
    assert binding.face_track_id == "face_1"


def test_alternating_visible_speakers_bind_in_time_order() -> None:
    bindings = _fuse(
        _evidence(
            [
                _interval(1.0, 2.0, {"face_1": 0.96, "face_2": 0.02}),
                _interval(3.0, 4.0, {"face_1": 0.03, "face_2": 0.94}),
            ]
        )
    )

    assert [item.entity_id for item in bindings] == ["e1", "e2"]
    assert all(item.status == "bound" for item in bindings)


def test_simultaneous_speakers_are_overlap_not_clean_bound() -> None:
    binding = _fuse(_evidence([_interval(1.0, 2.0, {"face_1": 0.93, "face_2": 0.91})]))[
        0
    ]

    assert binding.status == "overlap"
    assert binding.entity_id is None
    assert binding.evidence.clean_training_eligible is False


def test_visible_silent_person_with_speech_is_offscreen() -> None:
    binding = _fuse(_evidence([_interval(1.0, 2.0, {"face_1": 0.04, "face_2": 0.02})]))[
        0
    ]

    assert binding.status == "offscreen"
    assert binding.entity_id is None


def test_incomplete_asd_visible_face_coverage_is_ambiguous() -> None:
    binding = _fuse(
        _evidence(
            [
                _interval(
                    1.0,
                    2.0,
                    {"face_1": 0.02},
                    visible_face_track_ids=["face_1", "face_2"],
                )
            ]
        )
    )[0]

    assert binding.status == "ambiguous"
    assert binding.evidence.reason_codes == ["asd_visible_face_coverage_incomplete"]


def test_ambiguous_asd_scores_are_not_force_bound() -> None:
    binding = _fuse(_evidence([_interval(1.0, 2.0, {"face_1": 0.75, "face_2": 0.70})]))[
        0
    ]

    assert binding.status == "ambiguous"
    assert binding.entity_id is None


def test_missing_face_track_association_is_ambiguous() -> None:
    binding = _fuse(
        _evidence(
            [_interval(1.0, 2.0, {"face_1": 0.95})],
            associations=[_association(2)],
        )
    )[0]

    assert binding.status == "ambiguous"
    assert binding.evidence.reason_codes == ["face_track_entity_association_missing"]


def test_no_speech_interval_remains_explicit() -> None:
    binding = _fuse(
        _evidence([_interval(1.0, 2.0, {"face_1": 0.0, "face_2": 0.0}, speech=False)])
    )[0]

    assert binding.status == "no_speech"
    assert binding.entity_id is None


@pytest.mark.parametrize("audio_status", ["missing", "corrupted"])
def test_missing_or_corrupted_audio_fails_without_binding(audio_status: str) -> None:
    sidecar = build_audio_binding_sidecar(
        _clip(),
        _evidence([], audio_status=audio_status),
        source_run_root="/run",
    )

    assert sidecar.status == "failed"
    assert sidecar.reason == f"audio_{audio_status}"
    assert sidecar.bindings == []


def test_h3_asset_numbering_and_rendering_are_deterministic() -> None:
    evidence = _evidence(
        [
            _interval(1.0, 2.0, {"face_1": 0.96, "face_2": 0.02}),
            _interval(3.0, 4.0, {"face_1": 0.03, "face_2": 0.94}),
        ],
    )
    voice_backend = _FakeVoiceReferenceBackend(
        [
            _voice(3.1, 3.9, path="audio/e2.wav"),
            _voice(1.1, 1.9, path="audio/e1.wav"),
        ]
    )

    first = build_audio_binding_sidecar(
        _clip(),
        evidence,
        source_run_root="/run",
        voice_reference_backend=voice_backend,
    )
    second = build_audio_binding_sidecar(
        _clip(),
        evidence,
        source_run_root="/run",
        voice_reference_backend=voice_backend,
    )

    assert first == second
    assert first.status == "ready"
    assert first.h3_ir is not None
    assert [item.picture_id for item in first.h3_ir.picture_assets] == [
        "picture_1",
        "picture_2",
    ]
    assert [item.subject_id for item in first.h3_ir.subjects] == [
        "subject_1",
        "subject_2",
    ]
    assert [item.audio_id for item in first.h3_ir.audio_assets] == [
        "audio_1",
        "audio_2",
    ]
    assert [item.entity_id for item in first.h3_ir.audio_assets] == ["e1", "e2"]
    assert all(item.role == "voice_reference" for item in first.h3_ir.audio_assets)
    assert first.h3_ir.task.components == [
        "reference_generation",
        "audio_reference",
    ]
    assert all(
        item.path != evidence.audio.full_audio_path for item in first.h3_ir.audio_assets
    )
    assert voice_backend.calls == [
        [("e1", "bound"), ("e2", "bound")],
        [("e1", "bound"), ("e2", "bound")],
    ]
    rendered = render_h3_audio_instruction(first.h3_ir)
    assert "<Subject 1>" in rendered
    assert "<Picture 1>" in rendered
    assert "<Audio 1>" in rendered
    assert "[reference generation + audio reference]" in rendered


def test_full_audio_provenance_is_only_published_for_explicit_audio_reuse() -> None:
    evidence = _evidence([_interval(1.0, 2.0, {"face_1": 0.95})])

    reference_only = build_audio_binding_sidecar(
        _clip(),
        evidence,
        source_run_root="/run",
        task=H3TaskSpecification(components=["reference_generation"]),
    )
    audio_reuse = build_audio_binding_sidecar(
        _clip(),
        evidence,
        source_run_root="/run",
        task=H3TaskSpecification(components=["reference_generation", "audio_reuse"]),
    )

    assert reference_only.status == "ready"
    assert reference_only.h3_ir is not None
    assert reference_only.h3_ir.audio_assets == []
    assert audio_reuse.status == "ready"
    assert audio_reuse.h3_ir is not None
    assert audio_reuse.h3_ir.task.components == [
        "reference_generation",
        "audio_reuse",
    ]
    assert [(item.role, item.path) for item in audio_reuse.h3_ir.audio_assets] == [
        ("full_audio", "audio/full.wav")
    ]


def test_combined_audio_tasks_require_an_explicit_combination() -> None:
    evidence = _evidence([_interval(1.0, 2.0, {"face_1": 0.95})])
    backend = _FakeVoiceReferenceBackend([_voice(1.1, 1.9)])
    combined = build_audio_binding_sidecar(
        _clip(),
        evidence,
        source_run_root="/run",
        voice_reference_backend=backend,
        task=H3TaskSpecification(
            components=[
                "reference_generation",
                "audio_reference",
                "audio_reuse",
            ]
        ),
    )

    assert combined.status == "ready"
    assert combined.h3_ir is not None
    assert [item.role for item in combined.h3_ir.audio_assets] == [
        "voice_reference",
        "full_audio",
    ]


def test_h3_task_components_require_canonical_order() -> None:
    with pytest.raises(ValidationError, match="deterministic order"):
        H3TaskSpecification(components=["audio_reuse", "reference_generation"])


def test_voice_extractor_receives_fused_clean_entity_bindings() -> None:
    backend = _FakeVoiceReferenceBackend([_voice(1.1, 1.9)])

    sidecar = build_audio_binding_sidecar(
        _clip(),
        _evidence([_interval(1.0, 2.0, {"face_1": 0.95})]),
        source_run_root="/run",
        voice_reference_backend=backend,
    )

    assert sidecar.status == "ready"
    assert backend.calls == [[("e1", "bound")]]
    assert sidecar.voice_references[0].entity_id == "e1"
    assert "entity_id" not in _voice(1.1, 1.9).model_dump()


@pytest.mark.parametrize("reference_type", ["object", "group"])
def test_v1_does_not_publish_object_or_group_voice_references(
    reference_type: str,
) -> None:
    backend = _FakeVoiceReferenceBackend([_voice(1.1, 1.9)])

    sidecar = build_audio_binding_sidecar(
        _clip(entity_count=1, reference_types=[reference_type]),
        _evidence(
            [_interval(1.0, 2.0, {"face_1": 0.95})],
            associations=[_association(1)],
        ),
        source_run_root="/run",
        voice_reference_backend=backend,
    )

    assert sidecar.status == "ineligible"
    assert sidecar.reason == "no_clean_entity_bound_audio"
    assert sidecar.voice_references == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("voice_reference_id", "path"),
    [("voice_1", "audio/voice.wav"), ("voice_reference_1", " ")],
)
def test_voice_reference_validates_id_and_path(
    voice_reference_id: str,
    path: str,
) -> None:
    with pytest.raises(ValidationError):
        VoiceReference(
            voice_reference_id=voice_reference_id,
            entity_id="e1",
            path=path,
            source_start=1.1,
            source_end=1.9,
            quality_score=0.9,
            binding_start=1.0,
            binding_end=2.0,
        )


def test_h3_ir_rejects_unknown_or_duplicate_voice_entities() -> None:
    sidecar = build_audio_binding_sidecar(
        _clip(),
        _evidence([_interval(1.0, 2.0, {"face_1": 0.95})]),
        source_run_root="/run",
        voice_reference_backend=_FakeVoiceReferenceBackend([_voice(1.1, 1.9)]),
    )
    assert sidecar.h3_ir is not None
    unknown = sidecar.h3_ir.model_dump(mode="json")
    unknown["audio_assets"][0]["entity_id"] = "e9"
    with pytest.raises(ValidationError, match="entity must exist"):
        H3AudioBindingIR.model_validate(unknown)

    duplicate = sidecar.h3_ir.model_dump(mode="json")
    second_asset = dict(duplicate["audio_assets"][0])
    second_asset["audio_id"] = "audio_2"
    second_asset["voice_reference_id"] = "voice_reference_2"
    duplicate["audio_assets"].append(second_asset)
    with pytest.raises(ValidationError, match="at most one voice reference"):
        H3AudioBindingIR.model_validate(duplicate)


def test_h3_ir_rejects_multiple_full_audio_assets() -> None:
    sidecar = build_audio_binding_sidecar(
        _clip(),
        _evidence([]),
        source_run_root="/run",
        task=H3TaskSpecification(components=["reference_generation", "audio_reuse"]),
    )
    assert sidecar.h3_ir is not None
    invalid = sidecar.h3_ir.model_dump(mode="json")
    second_asset = dict(invalid["audio_assets"][0])
    second_asset["audio_id"] = "audio_2"
    invalid["audio_assets"].append(second_asset)

    with pytest.raises(ValidationError, match="at most one full-audio"):
        H3AudioBindingIR.model_validate(invalid)


def test_face_track_geometry_is_bounded_and_association_has_mask_context() -> None:
    geometry = [
        FaceGeometrySample(
            frame_index=index,
            timestamp=index / 10,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
            confidence=0.9,
        )
        for index in range(33)
    ]
    with pytest.raises(ValidationError):
        FaceTrack(
            face_track_id="face_1",
            start_time=0.0,
            end_time=10.0,
            sample_count=33,
            mean_detection_confidence=0.9,
            geometry_samples=geometry,
        )

    parameters = inspect.signature(EntityFaceAssociationBackend.associate).parameters
    assert {"clip", "source_run_root", "tracked_masks_path", "face_tracks"} <= set(
        parameters
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_run(root: Path, clip: ClipRecord) -> None:
    (root / "clips" / clip.clip_uid).mkdir(parents=True)
    (root / "run.json").write_text("{}\n", encoding="utf-8")
    (root / "clips" / clip.clip_uid / "clip.json").write_text(
        clip.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_precomputed_sidecar_cli_is_read_only_and_preserves_v3_contract(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output_root = tmp_path / "sidecar"
    evidence_path = tmp_path / "evidence.json"
    clip = _clip(entity_count=1)
    evidence = _evidence(
        [_interval(1.0, 2.0, {"face_1": 0.95})],
        associations=[_association(1)],
    )
    _write_run(run_root, clip)
    evidence_file = PrecomputedEvidenceFile(
        clips=[
            PrecomputedClipEvidence(
                evidence=evidence,
                voice_reference_candidates=[_voice(1.1, 1.9, path="audio/e1.wav")],
            )
        ]
    )
    evidence_path.write_text(evidence_file.model_dump_json(indent=2), encoding="utf-8")
    before = _tree_hashes(run_root)

    result = sidecar_main(
        [
            "--run-root",
            str(run_root),
            "--evidence-json",
            str(evidence_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert _tree_hashes(run_root) == before
    assert result["clip_count"] == 1
    assert result["ready_count"] == 1
    assert (output_root / "summary.json").is_file()
    assert (output_root / "audio_bindings.jsonl").is_file()
    assert (output_root / "clips" / "clip-1" / "audio_binding.json").is_file()
    assert (
        ClipRecord.model_validate_json(
            (run_root / "clips" / "clip-1" / "clip.json").read_text("utf-8")
        )
        == clip
    )
    assert "audio_binding" not in STAGE_ORDER


def test_fake_evidence_backend_continues_after_missing_case(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output_root = tmp_path / "sidecar"
    clip = _clip(entity_count=1)
    _write_run(run_root, clip)
    backend = PrecomputedEvidenceBackend(PrecomputedEvidenceFile(clips=[]))

    summary = build_audio_binding_sidecar_run(
        run_root=run_root,
        output_root=output_root,
        backend=backend,
    )

    assert summary.failed_count == 1
    record = json.loads(
        (output_root / "clips" / "clip-1" / "audio_binding.json").read_text("utf-8")
    )
    assert record["status"] == "failed"
    assert "missing audio evidence" in record["reason"]


def test_sidecar_output_inside_source_run_is_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    clip = _clip(entity_count=1)
    _write_run(run_root, clip)
    before = _tree_hashes(run_root)

    with pytest.raises(ValueError, match="separate from source run_root"):
        build_audio_binding_sidecar_run(
            run_root=run_root,
            output_root=run_root / "audio_binding",
            backend=PrecomputedEvidenceBackend(PrecomputedEvidenceFile(clips=[])),
        )

    assert _tree_hashes(run_root) == before


def _pilot_clip(clip_uid: str, source_video: Path, *, entity_count: int = 1) -> ClipRecord:
    payload = _clip(entity_count=entity_count).model_dump(mode="json")
    payload["clip_uid"] = clip_uid
    payload["source"]["video_path"] = str(source_video)
    for reference in payload["references"]["entities"]:
        reference["source_clip_uid"] = clip_uid
        reference["image_path"] = (
            f"clips/{clip_uid}/selected/{reference['entity_id']}.png"
        )
    return ClipRecord.model_validate(payload)


def _tracked_mask_frame(slot: int, mask: np.ndarray, entity_id: str) -> TrackedMaskFrame:
    rows, columns = np.nonzero(mask)
    return TrackedMaskFrame(
        slot=slot,
        present=True,
        confidence=0.95,
        backend_confidences=[0.95],
        backend_object_ids=[f"object-{entity_id}"],
        area_pixels=int(mask.sum()),
        area_ratio=float(mask.mean()),
        bbox_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ),
        rle=encode_binary_mask(mask),
    )


def _visual_artifacts(
    clip_uid: str,
    masks_by_entity: dict[str, np.ndarray],
) -> tuple[SampledFramesArtifact, TrackedMasksArtifact]:
    height, width = next(iter(masks_by_entity.values())).shape
    frames = SampledFramesArtifact(
        clip_uid=clip_uid,
        width=width,
        height=height,
        frames=[
            SampledFrame(
                slot=slot,
                source_frame_index=slot,
                timestamp_seconds=slot / 25,
                image_path=f"frames/{slot:02d}.jpg",
                sha256="0" * 64,
            )
            for slot in range(10)
        ],
    )
    masks = TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=width,
        height=height,
        entities={
            entity_id: TrackedEntityMasks(
                status="ready",
                reference_type="subject",
                grounding_prompt=f"person {entity_id}",
                backend_object_ids=[f"object-{entity_id}"],
                frames=[
                    _tracked_mask_frame(slot, mask, entity_id)
                    for slot in range(10)
                ],
            )
            for entity_id, mask in masks_by_entity.items()
        },
    )
    return frames, masks


def _native_artifact(
    *,
    clip_uid: str,
    source_video: Path,
    audio_path: Path,
    logits_by_track: list[list[float]],
    bboxes: list[tuple[float, float, float, float]] | None = None,
    duration_seconds: float = 0.4,
) -> LRASDNativeArtifact:
    active_bboxes = bboxes or [(2.0, 2.0, 8.0, 8.0)] * len(logits_by_track)
    return LRASDNativeArtifact(
        clip_uid=clip_uid,
        source_video_path=str(source_video),
        model_video_path=str(source_video.parent / f"{clip_uid}.model.avi"),
        audio_path=str(audio_path),
        model_provenance=ASDModelProvenance(
            backend="lr_asd",
            model_identifier="Junhua-Liao/LR-ASD",
            checkpoint_path="/models/lr_asd.model",
            checkpoint_sha256="a" * 64,
        ),
        width=20,
        height=20,
        duration_seconds=duration_seconds,
        tracks=[
            LRASDNativeTrack(
                face_track_id=f"face_{track_index}",
                samples=[
                    LRASDNativeSample(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / 25,
                        bbox_xyxy=active_bboxes[track_index - 1],
                        detection_confidence=0.95,
                        raw_class1_logit=logit,
                        backend_native_active=logit >= 0,
                    )
                    for frame_index, logit in enumerate(logits)
                ],
            )
            for track_index, logits in enumerate(logits_by_track, start=1)
        ],
    )


def _speech_artifact(
    clip_uid: str,
    audio_path: Path,
    *,
    speech: bool,
    duration_seconds: float = 0.4,
) -> SpeechActivityArtifact:
    return SpeechActivityArtifact(
        clip_uid=clip_uid,
        backend="silero_vad",
        model_identifier="silero_vad.jit",
        source_audio_path=str(audio_path),
        duration_seconds=duration_seconds,
        intervals=(
            [SpeechActivityInterval(start_time=0.0, end_time=duration_seconds)]
            if speech
            else []
        ),
    )


def _write_pcm16_wav(path: Path, samples: list[int]) -> None:
    payload = array("h", samples)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        destination.writeframes(payload.tobytes())


def test_local_noise_uses_only_no_speech_and_robust_window_median() -> None:
    sample_rate = 16000
    samples = [0] * (3 * sample_rate)
    samples[: 400 * 16] = [1000] * (400 * 16)
    samples[: 20 * 16] = [30000] * (20 * 16)
    samples[400 * 16 : 800 * 16] = [25000] * (400 * 16)
    samples[2000 * 16 : 2400 * 16] = [1000] * (400 * 16)
    bindings = _fuse(
        _evidence(
            [
                _interval(0.0, 0.4, {}, speech=False, visible_face_track_ids=[]),
                _interval(0.4, 0.8, {"face_1": 0.01}),
                _interval(2.0, 2.4, {}, speech=False, visible_face_track_ids=[]),
            ]
        )
    )

    metrics = _local_noise_metrics(
        audio_samples=samples,
        sidecar=SimpleNamespace(bindings=bindings),
        turn_start_time=1.0,
        turn_end_time=2.0,
        sample_rate_hz=sample_rate,
    )

    assert [binding.status for binding in bindings] == [
        "no_speech",
        "offscreen",
        "no_speech",
    ]
    assert metrics["local_noise_context_available"] is True
    assert metrics["local_noise_sample_count"] == 12800
    assert metrics["local_noise_duration_seconds"] == pytest.approx(0.8)
    assert metrics["local_noise_rms_amplitude"] == pytest.approx(1000 / 32768)
    assert metrics["local_noise_rms_dbfs"] == pytest.approx(
        20 * math.log10(1000 / 32768)
    )


@pytest.mark.parametrize(
    ("duration", "expected_available"),
    [(0.199, False), (0.200, True)],
)
def test_local_noise_requires_at_least_point_two_seconds(
    duration: float,
    expected_available: bool,
) -> None:
    sample_rate = 16000
    bindings = _fuse(
        _evidence(
            [
                _interval(
                    0.0,
                    duration,
                    {},
                    speech=False,
                    visible_face_track_ids=[],
                )
            ]
        )
    )

    metrics = _local_noise_metrics(
        audio_samples=[1000] * (3 * sample_rate),
        sidecar=SimpleNamespace(bindings=bindings),
        turn_start_time=1.0,
        turn_end_time=2.0,
        sample_rate_hz=sample_rate,
    )

    assert metrics["local_noise_context_available"] is expected_available
    assert metrics["local_noise_sample_count"] == round(duration * sample_rate)
    if expected_available:
        assert metrics["local_noise_rms_amplitude"] == pytest.approx(1000 / 32768)
    else:
        assert metrics["local_noise_rms_amplitude"] is None
        assert metrics["local_noise_rms_dbfs"] is None
        assert metrics["estimated_snr_db"] is None


def _full_entity_mask() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:18, 1:12] = 1
    return mask


def test_lr_asd_native_scores_preserve_logit_decision_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.7] * 10],
    )
    native = LRASDNativeArtifact.model_validate_json(native.model_dump_json())
    associations = [_association(1)]
    evidence = normalize_lr_asd_evidence(
        native,
        _speech_artifact("clip-1", audio, speech=True),
        associations,
    )

    score = evidence.active_speaker_intervals[0].face_scores[0]
    assert score.raw_backend_score == pytest.approx(0.7)
    assert score.backend_native_active is True
    assert score.score_semantics == "lr_asd_native_class_1_logit"
    assert score.normalized_score is None
    assert evidence.active_speaker_intervals[0].model_provenance == (
        native.model_provenance
    )
    payload = native.model_dump(mode="json")
    payload["tracks"][0]["samples"][0]["backend_native_active"] = False
    with pytest.raises(ValidationError, match="score >= 0"):
        LRASDNativeArtifact.model_validate(payload)
    timestamp_payload = native.model_dump(mode="json")
    timestamp_payload["tracks"][0]["samples"][1]["timestamp_seconds"] = 0.05
    with pytest.raises(ValidationError, match="25-FPS"):
        LRASDNativeArtifact.model_validate(timestamp_payload)


def test_lr_asd_25fps_nearest_timestamp_mapping_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.1] * 10],
    )

    matched = nearest_track_sample(
        native.tracks[0],
        timestamp_seconds=0.081,
        maximum_delta_seconds=0.01,
    )
    assert matched is not None
    assert matched[0].frame_index == 2
    assert matched[0].timestamp_seconds == pytest.approx(2 / 25)
    assert matched[1] == pytest.approx(0.001)
    assert (
        nearest_track_sample(
            native.tracks[0],
            timestamp_seconds=0.081,
            maximum_delta_seconds=0.0005,
        )
        is None
    )


def test_face_track_associates_with_covering_entity_mask(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    frames, masks = _visual_artifacts("clip-1", {"e1": _full_entity_mask()})
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.5] * 10],
    )

    association = associate_face_tracks_to_entities(
        frames=frames,
        masks=masks,
        tracks=native.tracks,
    )[0]

    assert association.status == "matched"
    assert association.entity_id == "e1"
    assert association.candidates[0].matched_sampled_slots == 10
    assert association.candidates[0].mean_face_bbox_coverage == 1.0
    assert association.candidates[0].face_center_inside_count == 10
    diagnostics = association.evidence["slot_diagnostics"]
    assert {item["timestamp_delta_seconds"] for item in diagnostics} == {0.0}


def test_conflicting_entity_masks_make_face_association_ambiguous(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    mask = _full_entity_mask()
    frames, masks = _visual_artifacts("clip-1", {"e1": mask, "e2": mask})
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.5] * 10],
    )

    association = associate_face_tracks_to_entities(
        frames=frames,
        masks=masks,
        tracks=native.tracks,
    )[0]

    assert association.status == "ambiguous"
    assert association.entity_id is None
    assert association.reason == "conflicting_entity_masks"
    assert association.top1_top2_margin == 0.0


def test_insufficient_aligned_slots_make_association_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    frames, masks = _visual_artifacts("clip-1", {"e1": _full_entity_mask()})
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.5]],
    )

    association = associate_face_tracks_to_entities(
        frames=frames,
        masks=masks,
        tracks=native.tracks,
        policy=FaceEntityAssociationPolicy(
            maximum_timestamp_delta_seconds=0.01,
            minimum_matched_sampled_slots=2,
        ),
    )[0]

    assert association.status == "ambiguous"
    assert association.reason == "insufficient_aligned_sampled_slots"
    assert association.evidence["aligned_sampled_slots"] == 1


@pytest.mark.parametrize(
    ("speech", "logits_by_track", "expected_status"),
    [
        (True, [[-0.3] * 10], "offscreen"),
        (False, [[0.8] * 10], "no_speech"),
        (True, [[0.8] * 10, [0.2] * 10], "overlap"),
    ],
)
def test_native_asd_and_vad_produce_explicit_binding_states(
    tmp_path: Path,
    speech: bool,
    logits_by_track: list[list[float]],
    expected_status: str,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=logits_by_track,
    )
    associations = [_association(index) for index in range(1, len(logits_by_track) + 1)]
    evidence = normalize_lr_asd_evidence(
        native,
        _speech_artifact("clip-1", audio, speech=speech),
        associations,
    )

    binding = fuse_audio_entity_bindings(
        evidence,
        known_entity_ids={f"e{index}" for index in range(1, len(logits_by_track) + 1)},
    )[0]

    assert binding.status == expected_status
    if expected_status == "offscreen":
        assert "lr_asd_native_decision_unvalidated" in binding.evidence.reason_codes


class _FakeReviewMediaBackend:
    def render_visualization(
        self,
        *,
        source_video_path: Path,
        timeline_path: Path,
        destination_path: Path,
    ) -> None:
        assert source_video_path.read_bytes() == b"video"
        assert json.loads(timeline_path.read_text(encoding="utf-8"))["samples"]
        destination_path.write_bytes(b"visualization")

    def extract_audio(
        self,
        *,
        source_audio_path: Path,
        start_time: float,
        end_time: float,
        destination_path: Path,
    ) -> None:
        assert source_audio_path.is_file()
        destination_path.write_bytes(f"{start_time:.3f}-{end_time:.3f}".encode())


class _CountingLRASDBackend(PrecomputedLRASDBackend):
    def __init__(
        self,
        artifacts: dict[str, LRASDNativeArtifact],
        *,
        fail_clip_uid: str | None = None,
    ) -> None:
        super().__init__(artifacts)
        self.fail_clip_uid = fail_clip_uid
        self.calls: list[str] = []

    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LRASDNativeArtifact:
        self.calls.append(clip_uid)
        if clip_uid == self.fail_clip_uid:
            raise LRASDRuntimeError("fake isolated runtime failure")
        return super().analyze(
            clip_uid=clip_uid,
            source_video_path=source_video_path,
            work_dir=work_dir,
        )


class _OutputLocalLRASDBackend:
    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LRASDNativeArtifact:
        work_dir.mkdir(parents=True)
        audio_path = work_dir / "audio.wav"
        model_video_path = work_dir / "model.avi"
        visualization_path = work_dir / "official.avi"
        audio_path.write_bytes(b"audio")
        model_video_path.write_bytes(b"model")
        visualization_path.write_bytes(b"visualization")
        native = _native_artifact(
            clip_uid=clip_uid,
            source_video=source_video_path,
            audio_path=audio_path,
            logits_by_track=[[0.7] * 10],
        )
        payload = native.model_dump(mode="json")
        payload["model_video_path"] = str(model_video_path)
        payload["official_visualization_path"] = str(visualization_path)
        return LRASDNativeArtifact.model_validate(payload)


class _DynamicSpeechActivityBackend:
    def detect(
        self,
        *,
        clip_uid: str,
        audio_path: Path,
        duration_seconds: float,
        work_dir: Path,
    ) -> SpeechActivityArtifact:
        del work_dir
        assert duration_seconds == 0.4
        return _speech_artifact(clip_uid, audio_path, speech=True)


def _write_pilot_clip(
    run_root: Path,
    clip: ClipRecord,
    *,
    masks_by_entity: dict[str, np.ndarray],
) -> None:
    _write_run(run_root, clip)
    clip_dir = run_root / "clips" / clip.clip_uid
    frames, masks = _visual_artifacts(clip.clip_uid, masks_by_entity)
    (clip_dir / "frames").mkdir()
    (clip_dir / "frames" / "frames.json").write_text(
        frames.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (clip_dir / "masks.rle.json").write_text(
        masks.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def test_pilot_isolates_lr_asd_failure_and_runs_backend_once_per_clip(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_1 = tmp_path / "clip-1.mp4"
    source_2 = tmp_path / "clip-2.mp4"
    audio_1 = tmp_path / "clip-1.wav"
    audio_2 = tmp_path / "clip-2.wav"
    for path, content in (
        (source_1, b"video"),
        (source_2, b"video"),
        (audio_1, b"audio"),
        (audio_2, b"audio"),
    ):
        path.write_bytes(content)
    _write_pilot_clip(
        run_root,
        _pilot_clip("clip-1", source_1),
        masks_by_entity={"e1": _full_entity_mask()},
    )
    _write_pilot_clip(
        run_root,
        _pilot_clip("clip-2", source_2),
        masks_by_entity={"e1": _full_entity_mask()},
    )
    native_1 = _native_artifact(
        clip_uid="clip-1",
        source_video=source_1,
        audio_path=audio_1,
        logits_by_track=[[0.7] * 10],
    )
    native_2 = _native_artifact(
        clip_uid="clip-2",
        source_video=source_2,
        audio_path=audio_2,
        logits_by_track=[[0.7] * 10],
    )
    backend = _CountingLRASDBackend(
        {"clip-1": native_1, "clip-2": native_2},
        fail_clip_uid="clip-1",
    )
    before = _tree_hashes(run_root)

    summary = run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=tmp_path / "pilot",
        lr_asd_backend=backend,
        speech_backend=PrecomputedSpeechActivityBackend(
            {
                "clip-2": _speech_artifact("clip-2", audio_2, speech=True),
            }
        ),
        review_media_backend=_FakeReviewMediaBackend(),
        limit=2,
    )

    assert backend.calls == ["clip-1", "clip-2"]
    assert summary.clips_attempted == 2
    assert summary.clips_succeeded == 1
    assert summary.clips_failed == 1
    assert summary.asd_runtime_failures == 1
    assert (tmp_path / "pilot" / "review" / "clip-2" / "timeline.json").is_file()
    assert (
        tmp_path / "pilot" / "clips" / "clip-2" / "audio_binding.json"
    ).is_file()
    canonical_records = [
        json.loads(line)
        for line in (tmp_path / "pilot" / "audio_bindings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert [item["clip_uid"] for item in canonical_records] == ["clip-2"]
    failures = [
        json.loads(line)
        for line in (tmp_path / "pilot" / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures == [
        {
            "clip_uid": "clip-1",
            "error_type": "LRASDRuntimeError",
            "reason": "fake isolated runtime failure",
            "stage": "lr_asd",
        }
    ]
    assert _tree_hashes(run_root) == before


def test_parallel_pilot_matches_serial_outputs_and_failure_accounting(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    native_artifacts = {}
    speech_artifacts = {}
    waveform = [0, 16384, -16384, 32767, -32768] * 2560
    for clip_uid in ("clip-3", "clip-1", "clip-2"):
        source = tmp_path / f"{clip_uid}.mp4"
        audio = tmp_path / f"{clip_uid}.wav"
        source.write_bytes(b"video")
        _write_pcm16_wav(audio, waveform)
        _write_pilot_clip(
            run_root,
            _pilot_clip(clip_uid, source),
            masks_by_entity={"e1": _full_entity_mask()},
        )
        native_artifacts[clip_uid] = _native_artifact(
            clip_uid=clip_uid,
            source_video=source,
            audio_path=audio,
            logits_by_track=[[0.7] * 20],
            duration_seconds=0.8,
        )
        speech_artifacts[clip_uid] = _speech_artifact(
            clip_uid,
            audio,
            speech=True,
            duration_seconds=0.8,
        )

    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    serial_summary = run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=serial_root,
        lr_asd_backend=_CountingLRASDBackend(
            native_artifacts,
            fail_clip_uid="clip-2",
        ),
        speech_backend=PrecomputedSpeechActivityBackend(speech_artifacts),
        review_media_backend=_FakeReviewMediaBackend(),
        limit=3,
        workers=1,
    )
    parallel_summary = run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=parallel_root,
        lr_asd_backend=_CountingLRASDBackend(
            native_artifacts,
            fail_clip_uid="clip-2",
        ),
        speech_backend=PrecomputedSpeechActivityBackend(speech_artifacts),
        review_media_backend=_FakeReviewMediaBackend(),
        limit=3,
        workers=4,
    )

    serial_counters = serial_summary.model_dump(mode="json")
    parallel_counters = parallel_summary.model_dump(mode="json")
    serial_counters.pop("output_root")
    parallel_counters.pop("output_root")
    assert parallel_counters == serial_counters
    assert parallel_summary.clips_succeeded == 2
    assert parallel_summary.clips_failed == 1
    assert parallel_summary.asd_runtime_failures == 1
    assert (parallel_root / "audio_bindings.jsonl").read_bytes() == (
        serial_root / "audio_bindings.jsonl"
    ).read_bytes()
    assert (parallel_root / "failures.jsonl").read_bytes() == (
        serial_root / "failures.jsonl"
    ).read_bytes()
    assert (parallel_root / "voice_reference_quality.jsonl").read_bytes() == (
        serial_root / "voice_reference_quality.jsonl"
    ).read_bytes()
    assert (
        parallel_root / "voice_reference_quality_summary.json"
    ).read_bytes() == (
        serial_root / "voice_reference_quality_summary.json"
    ).read_bytes()
    records = [
        json.loads(line)
        for line in (parallel_root / "audio_bindings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["clip_uid"] for record in records] == ["clip-1", "clip-3"]
    quality_records = [
        json.loads(line)
        for line in (parallel_root / "voice_reference_quality.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["clip_uid"] for record in quality_records] == [
        "clip-1",
        "clip-3",
    ]
    quality = quality_records[0]
    assert "source_binding_ids" not in quality
    assert quality["duration_seconds"] == pytest.approx(0.8)
    assert quality["sample_count"] == 12800
    expected_rms = math.sqrt(
        (0**2 + 16384**2 + (-16384) ** 2 + 32767**2 + (-32768) ** 2) / 5
    ) / 32768
    assert quality["rms_amplitude"] == pytest.approx(expected_rms)
    assert quality["rms_dbfs"] == pytest.approx(20 * math.log10(expected_rms))
    assert quality["peak_amplitude"] == 1.0
    assert quality["peak_dbfs"] == 0.0
    assert quality["clipping_ratio"] == pytest.approx(0.4)
    assert quality["lr_asd_raw_native_score"] == pytest.approx(
        {"mean": 0.7, "min": 0.7, "p10": 0.7}
    )
    assert quality["association_confidence"] == pytest.approx(
        {"mean": 1.0, "min": 1.0}
    )
    assert quality["local_noise_context_available"] is False
    assert quality["local_noise_sample_count"] == 0
    assert quality["local_noise_duration_seconds"] == 0.0
    assert quality["local_noise_rms_amplitude"] is None
    assert quality["local_noise_rms_dbfs"] is None
    assert quality["estimated_snr_db"] is None
    quality_summary = json.loads(
        (parallel_root / "voice_reference_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert quality_summary["thresholds_calibrated"] is False
    assert quality_summary["candidate_turn_count"] == 2
    assert quality_summary["diagnostics_failed_clip_count"] == 0
    assert quality_summary["noise_context_available_count"] == 0
    assert quality_summary["noise_context_unavailable_count"] == 2
    assert quality_summary["noise_context_availability_rate"] == 0.0
    for clip_uid in ("clip-1", "clip-3"):
        assert (
            parallel_root / "clips" / clip_uid / "audio_binding.json"
        ).read_bytes() == (
            serial_root / "clips" / clip_uid / "audio_binding.json"
        ).read_bytes()
        assert (
            parallel_root
            / "clips"
            / clip_uid
            / "voice_reference_quality.json"
        ).read_bytes() == (
            serial_root
            / "clips"
            / clip_uid
            / "voice_reference_quality.json"
        ).read_bytes()
        assert (
            parallel_root
            / "review"
            / clip_uid
            / "face_entity_association.json"
        ).read_bytes() == (
            serial_root
            / "review"
            / clip_uid
            / "face_entity_association.json"
        ).read_bytes()


def test_voice_quality_noise_metrics_and_postprocess_are_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_root = tmp_path / "run"
    source = tmp_path / "clip-1.mp4"
    audio = tmp_path / "clip-1.wav"
    source.write_bytes(b"video")
    samples = [1000] * 32000
    samples[:320] = [30000] * 320
    samples[6400:25600] = [10000] * 19200
    _write_pcm16_wav(audio, samples)
    _write_pilot_clip(
        run_root,
        _pilot_clip("clip-1", source),
        masks_by_entity={"e1": _full_entity_mask()},
    )
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.7] * 50],
        duration_seconds=2.0,
    )
    lr_backend = _CountingLRASDBackend({"clip-1": native})
    pilot_root = tmp_path / "voice-quality-pilot20"
    run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=pilot_root,
        lr_asd_backend=lr_backend,
        speech_backend=PrecomputedSpeechActivityBackend(
            {
                "clip-1": SpeechActivityArtifact(
                    clip_uid="clip-1",
                    backend="silero_vad",
                    model_identifier="silero_vad.jit",
                    source_audio_path=str(audio),
                    duration_seconds=2.0,
                    intervals=[
                        SpeechActivityInterval(start_time=0.4, end_time=1.6)
                    ],
                )
            }
        ),
        review_media_backend=_FakeReviewMediaBackend(),
        clip_ids=["clip-1"],
    )

    quality_path = (
        pilot_root / "clips" / "clip-1" / "voice_reference_quality.json"
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["schema_version"] == "r2v.h3.voice_reference_quality.2"
    assert len(quality["candidate_turns"]) == 1
    turn = quality["candidate_turns"][0]
    assert turn["start_time"] == pytest.approx(0.4)
    assert turn["end_time"] == pytest.approx(1.6)
    assert turn["local_noise_context_available"] is True
    assert turn["local_noise_sample_count"] == 12800
    assert turn["local_noise_duration_seconds"] == pytest.approx(0.8)
    assert turn["local_noise_rms_amplitude"] == pytest.approx(1000 / 32768)
    assert turn["local_noise_rms_dbfs"] == pytest.approx(
        20 * math.log10(1000 / 32768)
    )
    assert turn["estimated_snr_db"] == pytest.approx(20.0)
    summary_path = pilot_root / "voice_reference_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["noise_context_available_count"] == 1
    assert summary["noise_context_unavailable_count"] == 0
    assert summary["noise_context_availability_rate"] == 1.0
    assert summary["metric_distributions"]["estimated_snr_db"]["mean"] == (
        pytest.approx(20.0)
    )

    non_diagnostic_before = {
        name: digest
        for name, digest in _tree_hashes(pilot_root).items()
        if "voice_reference_quality" not in name
    }
    for path in pilot_root.rglob("*voice_reference_quality*"):
        if path.is_file():
            path.write_text("{}\n", encoding="utf-8")

    recomputed = recompute_voice_reference_quality_artifacts(
        pilot_root=pilot_root,
    )

    assert recomputed.candidate_turn_count == 1
    assert lr_backend.calls == ["clip-1"]
    assert {
        name: digest
        for name, digest in _tree_hashes(pilot_root).items()
        if "voice_reference_quality" not in name
    } == non_diagnostic_before
    assert quality_path.read_bytes() == (
        pilot_root / "review" / "clip-1" / "voice_reference_quality.json"
    ).read_bytes()
    first_diagnostic_hashes = {
        name: digest
        for name, digest in _tree_hashes(pilot_root).items()
        if "voice_reference_quality" in name
    }

    cli_result = recompute_voice_quality_main(
        ["--pilot-root", str(pilot_root)]
    )

    assert cli_result["thresholds_calibrated"] is False
    assert json.loads(capsys.readouterr().out)["candidate_turn_count"] == 1
    assert {
        name: digest
        for name, digest in _tree_hashes(pilot_root).items()
        if "voice_reference_quality" in name
    } == first_diagnostic_hashes
    assert not list(pilot_root.rglob("*.tmp-*"))


def test_voice_quality_failure_does_not_change_binding_result(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source = tmp_path / "clip-1.mp4"
    invalid_audio = tmp_path / "clip-1.wav"
    source.write_bytes(b"video")
    invalid_audio.write_bytes(b"not-a-wave")
    _write_pilot_clip(
        run_root,
        _pilot_clip("clip-1", source),
        masks_by_entity={"e1": _full_entity_mask()},
    )
    output_root = tmp_path / "pilot"

    summary = run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=output_root,
        lr_asd_backend=PrecomputedLRASDBackend(
            {
                "clip-1": _native_artifact(
                    clip_uid="clip-1",
                    source_video=source,
                    audio_path=invalid_audio,
                    logits_by_track=[[0.7] * 20],
                    duration_seconds=0.8,
                )
            }
        ),
        speech_backend=PrecomputedSpeechActivityBackend(
            {
                "clip-1": _speech_artifact(
                    "clip-1",
                    invalid_audio,
                    speech=True,
                    duration_seconds=0.8,
                )
            }
        ),
        review_media_backend=_FakeReviewMediaBackend(),
        clip_ids=["clip-1"],
    )

    assert summary.clips_succeeded == 1
    assert summary.clips_failed == 0
    sidecar = json.loads(
        (output_root / "clips" / "clip-1" / "audio_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert {binding["status"] for binding in sidecar["bindings"]} == {"bound"}
    assert all(
        binding["evidence"]["audio_quality_usable"]
        and binding["evidence"]["synchronization_plausible"]
        for binding in sidecar["bindings"]
    )
    diagnostics = json.loads(
        (
            output_root
            / "clips"
            / "clip-1"
            / "voice_reference_quality.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == "failed"
    assert diagnostics["candidate_turns"] == []
    quality_summary = json.loads(
        (output_root / "voice_reference_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert quality_summary["diagnostics_failed_clip_count"] == 1
    assert quality_summary["candidate_turn_count"] == 0


def test_pilot_workers_cli_defaults_to_one_and_accepts_parallel_count() -> None:
    required = ["--run-root", "/run", "--output-root", "/output", "--limit", "1"]

    assert lr_asd_pilot_parser().parse_args(required).workers == 1
    assert lr_asd_pilot_parser().parse_args([*required, "--workers", "4"]).workers == 4
    with pytest.raises(SystemExit):
        lr_asd_pilot_parser().parse_args([*required, "--workers", "0"])


def test_audio_subprocesses_preserve_configured_python_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "uv-python"
    target.write_bytes(b"python")
    target.chmod(0o755)
    python_path = tmp_path / "venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(target)
    code_root = tmp_path / "LR-ASD"
    code_root.mkdir()
    (code_root / "Columbia_test.py").write_text("# vendor entry\n", encoding="utf-8")
    model_path = tmp_path / "pretrain.model"
    model_path.write_bytes(b"model")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run_logged_command(
        command: list[str],
        **kwargs: object,
    ) -> None:
        commands.append(command)
        error_type = kwargs["error_type"]
        raise error_type("stop after command capture")  # type: ignore[operator]

    monkeypatch.setattr(
        lr_asd_module,
        "_run_logged_command",
        fake_run_logged_command,
    )
    lr_backend = LRASDSubprocessBackend(
        LRASDRuntimeConfig(
            code_root=code_root,
            python_path=python_path,
            model_path=model_path,
        )
    )
    with pytest.raises(LRASDRuntimeError, match="command capture"):
        lr_backend.analyze(
            clip_uid="clip-1",
            source_video_path=source,
            work_dir=tmp_path / "lr-work",
        )
    assert commands[-1][0] == str(python_path)
    assert commands[-1][0] != str(target)

    silero_backend = SileroVADSubprocessBackend(
        SileroVADRuntimeConfig(
            python_path=python_path,
            model_path=model_path,
        )
    )
    with pytest.raises(lr_asd_module.SpeechActivityRuntimeError, match="command capture"):
        silero_backend.detect(
            clip_uid="clip-1",
            audio_path=source,
            duration_seconds=1.0,
            work_dir=tmp_path / "vad-work",
        )
    assert commands[-1][0] == str(python_path)
    assert commands[-1][0] != str(target)

    review_backend = ExternalReviewMediaBackend(python_path=python_path)
    review_commands: list[list[str]] = []
    monkeypatch.setattr(
        review_backend,
        "_run",
        lambda command, *, destination_path: review_commands.append(command),
    )
    review_backend.render_visualization(
        source_video_path=source,
        timeline_path=tmp_path / "timeline.json",
        destination_path=tmp_path / "review.mp4",
    )
    assert review_commands[0][0] == str(python_path)
    assert review_commands[0][0] != str(target)


def test_review_bundle_metadata_is_deterministic_and_keeps_native_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    run_root = tmp_path / "run"
    clip = _pilot_clip("clip-1", source)
    _write_pilot_clip(
        run_root,
        clip,
        masks_by_entity={"e1": _full_entity_mask()},
    )
    native = _native_artifact(
        clip_uid="clip-1",
        source_video=source,
        audio_path=audio,
        logits_by_track=[[0.7] * 10],
    )

    for output_name in ("pilot-a", "pilot-b"):
        run_h3_audio_binding_pilot(
            run_root=run_root,
            output_root=tmp_path / output_name,
            lr_asd_backend=PrecomputedLRASDBackend({"clip-1": native}),
            speech_backend=PrecomputedSpeechActivityBackend(
                {"clip-1": _speech_artifact("clip-1", audio, speech=True)}
            ),
            review_media_backend=_FakeReviewMediaBackend(),
            clip_ids=["clip-1"],
        )

    first = tmp_path / "pilot-a" / "review" / "clip-1"
    second = tmp_path / "pilot-b" / "review" / "clip-1"
    for name in (
        "timeline.json",
        "audio_binding.json",
        "lr_asd_native.json",
        "face_entity_association.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    timeline = json.loads((first / "timeline.json").read_text(encoding="utf-8"))
    assert timeline == build_review_timeline(
        native=native,
        associations=associate_face_tracks_to_entities(
            frames=_visual_artifacts("clip-1", {"e1": _full_entity_mask()})[0],
            masks=_visual_artifacts("clip-1", {"e1": _full_entity_mask()})[1],
            tracks=native.tracks,
        ),
        bindings=[
            AudioEntityBinding.model_validate(item) for item in timeline["bindings"]
        ],
    )
    assert (first / "source.mp4").read_bytes() == b"video"
    assert (first / "visualization.mp4").read_bytes() == b"visualization"
    assert list((first / "bound_audio").glob("e1_*.wav"))


def test_pilot_rebases_internal_runtime_paths_before_atomic_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    run_root = tmp_path / "run"
    _write_pilot_clip(
        run_root,
        _pilot_clip("clip-1", source),
        masks_by_entity={"e1": _full_entity_mask()},
    )
    output_root = tmp_path / "pilot"

    run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=output_root,
        lr_asd_backend=_OutputLocalLRASDBackend(),
        speech_backend=_DynamicSpeechActivityBackend(),
        review_media_backend=_FakeReviewMediaBackend(),
        clip_ids=["clip-1"],
    )

    native = json.loads(
        (output_root / "review" / "clip-1" / "lr_asd_native.json").read_text(
            encoding="utf-8"
        )
    )
    audio_path = Path(native["audio_path"])
    model_video_path = Path(native["model_video_path"])
    official_visualization_path = Path(native["official_visualization_path"])
    assert output_root in audio_path.parents
    assert audio_path.read_bytes() == b"audio"
    assert model_video_path.read_bytes() == b"model"
    assert official_visualization_path.read_bytes() == b"visualization"
    assert ".tmp-" not in native["audio_path"]
    sidecar = json.loads(
        (output_root / "review" / "clip-1" / "audio_binding.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["evidence"]["audio"]["full_audio_path"] == str(audio_path)


def test_pilot_canonical_sidecar_is_directly_consumed_by_bind_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    run_root = tmp_path / "run"
    clip = _pilot_clip("clip-1", source)
    _write_pilot_clip(
        run_root,
        clip,
        masks_by_entity={"e1": _full_entity_mask()},
    )
    pilot_root = tmp_path / "pilot"
    run_h3_audio_binding_pilot(
        run_root=run_root,
        output_root=pilot_root,
        lr_asd_backend=PrecomputedLRASDBackend(
            {
                "clip-1": _native_artifact(
                    clip_uid="clip-1",
                    source_video=source,
                    audio_path=audio,
                    logits_by_track=[[0.7] * 10],
                )
            }
        ),
        speech_backend=PrecomputedSpeechActivityBackend(
            {"clip-1": _speech_artifact("clip-1", audio, speech=True)}
        ),
        review_media_backend=_FakeReviewMediaBackend(),
        clip_ids=["clip-1"],
    )
    review_sidecar = pilot_root / "review" / "clip-1" / "audio_binding.json"
    canonical_sidecar = pilot_root / "clips" / "clip-1" / "audio_binding.json"
    assert canonical_sidecar.read_bytes() == review_sidecar.read_bytes()

    visual_root = tmp_path / "visual-export"
    reference_path = visual_root / "references" / "clip-1" / "subject_1.png"
    reference_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "navy").save(reference_path)
    (visual_root / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "clip-1",
                "target_video": str(source),
                "t2v_caption": "A person speaks.",
                "r2v_instruction": "Use the person reference.",
                "references": [
                    {
                        "type": "entity",
                        "entity_id": "e1",
                        "image_path": "references/clip-1/subject_1.png",
                        "token": "<ref_subject_1>",
                        "scope": "full",
                        "visible_region": "whole",
                        "source_frame_index": 5,
                        "source_clip_uid": "clip-1",
                        "source_entity_id": "e1",
                        "synthetic": False,
                    }
                ],
                "source": {"parent_video_id": "parent", "clip_suffix": "1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    full_audio = tmp_path / "full.flac"
    full_audio.write_bytes(b"full-audio")
    outputs = build_audio_clip_binding_dataset(
        run_root=run_root,
        visual_export_root=visual_root,
        sidecar_root=pilot_root,
        output_root=tmp_path / "audio-bindings",
        audio_backend=PrecomputedAudioMediaBackend(
            {"clip-1": full_audio},
            {},
            {
                "clip-1": AudioStreamProvenance(
                    stream_index=0,
                    codec_name="aac",
                    original_sample_rate_hz=16000,
                    original_channels=1,
                    duration_seconds=0.4,
                    time_base="1/16000",
                )
            },
        ),
        face_backend=PrecomputedEmbeddingBackend(
            {"clip-1/e1": np.asarray([1.0, 0.0], dtype=np.float32)},
            model_identifier="fake-face",
        ),
        speaker_backend=PrecomputedEmbeddingBackend(
            {},
            model_identifier="fake-speaker",
        ),
    )

    assert [item.clip_uid for item in outputs] == ["clip-1"]
