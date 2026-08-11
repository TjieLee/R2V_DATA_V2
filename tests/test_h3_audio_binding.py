from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.backends import PrecomputedEvidenceBackend
from r2v_data_v2.h3.fusion import (
    build_audio_binding_sidecar,
    fuse_audio_entity_bindings,
    render_h3_audio_instruction,
)
from r2v_data_v2.h3.schemas import (
    ActiveSpeakerInterval,
    AudioBindingEvidence,
    AudioTrackMetadata,
    EntityFaceAssociation,
    FaceTrack,
    PrecomputedEvidenceFile,
    VoiceReferenceCandidate,
)
from r2v_data_v2.h3.sidecar import build_audio_binding_sidecar_run
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
)
from run_pipeline_v3 import STAGE_ORDER
from tools.build_v3_h3_audio_binding_sidecar import main as sidecar_main


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


def _clip(entity_count: int = 2) -> ClipRecord:
    entities = [
        AnnotationEntity(
            entity_id=f"e{index}",
            reference_type="subject",
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
            entity_visibility_summary={entity_id: _visibility() for entity_id in retained},
        ),
        references=ReferencesState(entities=references),
        pairing=PairingState(
            status="ready",
            retained_entity_ids=retained,
            tokens={
                entity_id: f"<ref_subject_{index}>"
                for index, entity_id in enumerate(retained, start=1)
            },
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
) -> ActiveSpeakerInterval:
    return ActiveSpeakerInterval(
        start_time=start,
        end_time=end,
        speech_present=speech,
        face_speaking_probabilities=probabilities,
        audio_quality_usable=True,
        synchronization_plausible=True,
    )


def _evidence(
    intervals: list[ActiveSpeakerInterval],
    *,
    associations: list[EntityFaceAssociation] | None = None,
    audio_status: str = "ready",
    voice_candidates: list[VoiceReferenceCandidate] | None = None,
) -> AudioBindingEvidence:
    return AudioBindingEvidence(
        clip_uid="clip-1",
        audio=_audio(audio_status),
        face_tracks=[_track(1), _track(2)],
        associations=(
            [_association(1), _association(2)]
            if associations is None
            else associations
        ),
        active_speaker_intervals=intervals,
        voice_reference_candidates=voice_candidates or [],
    )


def _voice(entity_id: str, start: float, end: float) -> VoiceReferenceCandidate:
    return VoiceReferenceCandidate(
        entity_id=entity_id,
        path=f"audio/{entity_id}.wav",
        source_start=start,
        source_end=end,
        quality_score=0.9,
        quality_metadata={"snr_db": 24.0},
    )


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
    binding = _fuse(
        _evidence([_interval(1.0, 2.0, {"face_1": 0.96, "face_2": 0.03})])
    )[0]

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
    binding = _fuse(
        _evidence([_interval(1.0, 2.0, {"face_1": 0.93, "face_2": 0.91})])
    )[0]

    assert binding.status == "overlap"
    assert binding.entity_id is None
    assert binding.evidence.clean_training_eligible is False


def test_visible_silent_person_with_speech_is_offscreen() -> None:
    binding = _fuse(
        _evidence([_interval(1.0, 2.0, {"face_1": 0.04, "face_2": 0.02})])
    )[0]

    assert binding.status == "offscreen"
    assert binding.entity_id is None


def test_ambiguous_asd_scores_are_not_force_bound() -> None:
    binding = _fuse(
        _evidence([_interval(1.0, 2.0, {"face_1": 0.75, "face_2": 0.70})])
    )[0]

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
    assert binding.evidence.reason_codes == [
        "face_track_entity_association_missing"
    ]


def test_no_speech_interval_remains_explicit() -> None:
    binding = _fuse(
        _evidence(
            [_interval(1.0, 2.0, {"face_1": 0.0, "face_2": 0.0}, speech=False)]
        )
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
        voice_candidates=[_voice("e2", 3.1, 3.9), _voice("e1", 1.1, 1.9)],
    )

    first = build_audio_binding_sidecar(
        _clip(), evidence, source_run_root="/run"
    )
    second = build_audio_binding_sidecar(
        _clip(), evidence, source_run_root="/run"
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
        "audio_3",
    ]
    assert [item.entity_id for item in first.h3_ir.audio_assets[:2]] == ["e1", "e2"]
    assert first.h3_ir.audio_assets[2].role == "full_audio_reference"
    rendered = render_h3_audio_instruction(first.h3_ir)
    assert "<Subject 1>" in rendered
    assert "<Picture 1>" in rendered
    assert "<Audio 1>" in rendered
    assert "[reference generation + audio reference]" in rendered


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
        voice_candidates=[_voice("e1", 1.1, 1.9)],
    )
    _write_run(run_root, clip)
    evidence_file = PrecomputedEvidenceFile(clips=[evidence])
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
    assert ClipRecord.model_validate_json(
        (run_root / "clips" / "clip-1" / "clip.json").read_text("utf-8")
    ) == clip
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
