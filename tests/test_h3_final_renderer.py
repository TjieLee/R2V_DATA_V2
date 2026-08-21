from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import r2v_data_v2.h3.final_renderer as renderer_module
import tools.run_h3_final_renderer as renderer_cli
from r2v_data_v2.h3.asr_transcription import ASRDecoderDiagnostics
from r2v_data_v2.h3.asr_v2_transcription import (
    ASRV2SegmentRecord,
    run_asr_v2_transcription,
)
from r2v_data_v2.h3.asr_v2_transcription import _summary as _asr_v2_summary
from r2v_data_v2.h3.audio_production import (
    H3ProductionCrossPair,
    H3ProductionCrossPairMapping,
    H3ProductionInPair,
    H3ProductionInPairSubject,
    H3ProductionSummary,
)
from r2v_data_v2.h3.final_renderer import (
    FinalH3Inventory,
    FinalH3Sample,
    FinalH3Summary,
    plan_final_h3_renderer,
    publish_final_h3_renderer,
)
from r2v_data_v2.h3.text_usability import publish_text_usability
from r2v_data_v2.v3.schemas import ClipRecord
from tests.test_h3_asr_v2_transcription import (
    _build,
    _FakeBackend,
    _write_asr_v1,
    _write_diarization_source,
)
from tests.test_h3_audio_binding import _clip


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in values
        ),
        encoding="utf-8",
    )


def _tree_hashes(root: Path, *, exclude_h3: bool = False) -> dict[str, str]:
    values = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if exclude_h3 and relative.startswith("production/h3/"):
            continue
        values[relative] = _sha256(path)
    return values


def _pair_paths(root: Path, clip_uid: str) -> dict[str, str]:
    media = root / "fixture_media"
    return {
        "video": str(media / f"{clip_uid}.mp4"),
        "audio": str(media / f"{clip_uid}.wav"),
        "sidecar": str(
            root / "production" / "audio" / "clips" / clip_uid / "audio_binding.json"
        ),
        "voice": str(
            root
            / "production"
            / "primary_voice"
            / "clips"
            / clip_uid
            / "e1.flac"
        ),
    }


def _in_pair(
    root: Path, *, clip_uid: str, entity_id: str
) -> H3ProductionInPair:
    paths = _pair_paths(root, clip_uid)
    return H3ProductionInPair(
        pair_id=f"in_pair/{clip_uid}",
        target_clip_uid=clip_uid,
        target_video_path=paths["video"],
        target_full_audio_path=paths["audio"],
        target_audio_binding_path=paths["sidecar"],
        subjects=[
            H3ProductionInPairSubject(
                subject_index=1,
                target_occurrence_id=f"{clip_uid}/{entity_id}",
                target_entity_id=entity_id,
                target_visual_reference_path=str(
                    root
                    / "visual-run"
                    / "clips"
                    / clip_uid
                    / "selected"
                    / f"{entity_id}.png"
                ),
                target_primary_voice_reference_path=paths["voice"],
            )
        ],
    )


def _cross_pair(root: Path) -> H3ProductionCrossPair:
    target = _pair_paths(root, "clip-000")
    donor = _pair_paths(root, "clip-001")
    pair_id = "cross_pair/clip-000/1"
    return H3ProductionCrossPair(
        pair_id=pair_id,
        target_clip_uid="clip-000",
        target_video_path=target["video"],
        target_full_audio_path=target["audio"],
        target_audio_binding_path=target["sidecar"],
        mappings=[
            H3ProductionCrossPairMapping(
                mapping_id=f"{pair_id}/subject_1",
                subject_index=1,
                target_occurrence_id="clip-000/e1",
                donor_occurrence_id="clip-001/e2",
                target_clip_uid="clip-000",
                donor_clip_uid="clip-001",
                target_visual_reference_path=str(
                    root / "visual-run/clips/clip-000/selected/e1.png"
                ),
                donor_primary_voice_reference_path=donor["voice"],
                donor_visual_reference_path=str(
                    root / "visual-run/clips/clip-001/selected/e2.png"
                ),
                target_primary_voice_reference_path=target["voice"],
                face_similarity=0.9,
                voice_similarity=0.4,
            )
        ],
    )


def _write_pairs(root: Path) -> tuple[list[H3ProductionInPair], list[H3ProductionCrossPair]]:
    pairs = [
        _in_pair(root, clip_uid="clip-000", entity_id="e1"),
        _in_pair(root, clip_uid="clip-001", entity_id="e2"),
        _in_pair(root, clip_uid="clip-019", entity_id="e2"),
    ]
    cross_pairs = [_cross_pair(root)]
    pair_root = root / "production" / "pairs"
    _write_jsonl(pair_root / "in_pairs.jsonl", pairs)
    _write_jsonl(pair_root / "cross_pairs.jsonl", cross_pairs)
    summary = H3ProductionSummary(
        complete_visual_eligible_occurrence_count=3,
        audio_clips_attempted=75,
        audio_clips_succeeded=75,
        audio_clips_failed=0,
        primary_voice_available_count=3,
        primary_voice_unavailable_count=0,
        face_embedding_available_count=3,
        face_embedding_unavailable_count=0,
        speaker_embedding_available_count=3,
        speaker_embedding_unavailable_count=0,
        in_pair_clip_sample_count=3,
        cross_pair_candidate_clip_count=3,
        cross_pair_clip_sample_count=1,
        selected_target_donor_subject_mapping_count=1,
        clips_without_complete_cross_pair_mapping=2,
        rejection_reason_counts={},
        incomplete_cross_pair_reason_counts={},
    )
    _write_json(pair_root / "summary.json", summary.model_dump(mode="json"))
    return pairs, cross_pairs


def _visual_clip(root: Path, *, clip_uid: str) -> None:
    visual_root = root / "visual-run"
    video = root / "fixture_media" / f"{clip_uid}.mp4"
    payload = _clip(2).model_dump(mode="json")
    payload["clip_uid"] = clip_uid
    payload["source"]["video_path"] = str(video)
    payload["source"]["parent_video_id"] = f"parent-{clip_uid}"
    payload["source"]["clip_suffix"] = clip_uid
    for reference in payload["references"]["entities"]:
        entity_id = reference["entity_id"]
        reference["image_path"] = f"clips/{clip_uid}/selected/{entity_id}.png"
        reference["source_clip_uid"] = clip_uid
    payload["instruction"] = {
        "status": "ready",
        "instruction_body_template": "Use {{image_1}} beside {{image_2}}.",
        "reference_legend": [
            {"image_id": "image_1", "description": "person 1"},
            {"image_id": "image_2", "description": "person 2"},
        ],
        "r2v_instruction": "Use <Image 1> beside <Image 2>.",
    }
    payload["export"] = {"accepted": True, "reason": None}
    clip = ClipRecord.model_validate(payload)
    clip_root = visual_root / "clips" / clip_uid
    _write_json(clip_root / "clip.json", clip.model_dump(mode="json"))
    for entity_id in ("e1", "e2"):
        path = clip_root / "selected" / f"{entity_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"picture:{clip_uid}:{entity_id}".encode())
    sidecar = Path(_pair_paths(root, clip_uid)["sidecar"])
    _write_json(
        sidecar,
        {
            "clip_uid": clip_uid,
            "status": "ready",
            "source_run_root": str(visual_root),
            "source_video_path": str(video),
        },
    )
    for entity_id in ("e1", "e2"):
        voice = (
            root
            / "production"
            / "primary_voice"
            / "clips"
            / clip_uid
            / f"{entity_id}.flac"
        )
        voice.parent.mkdir(parents=True, exist_ok=True)
        voice.write_bytes(f"voice:{clip_uid}:{entity_id}".encode())


def _diagnostics(*, language: str, probability: float) -> ASRDecoderDiagnostics:
    return ASRDecoderDiagnostics(
        detected_language=language,
        language_probability=probability,
        avg_log_probability=-0.2,
        no_speech_probability=0.02,
        compression_ratio=1.1,
        decoder_segment_count=1,
    )


def _customize_asr(root: Path, inventory) -> None:
    path = root / "production" / "asr_v2" / "segments.jsonl"
    records = [
        ASRV2SegmentRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updates = {
        ("clip-000", "segment_001"): ("保留原文。", "zh", 0.99),
        ("clip-000", "segment_002"): ("Keep exact text.", "en", 0.90),
        ("clip-000", "segment_003"): ("Versteckter Text", "de", 0.50),
        ("clip-001", "segment_001"): ("Background speaker.", "en", 0.95),
        ("clip-019", "segment_002"): ("Unresolved speaker.", "en", 0.95),
    }
    customized = []
    for record in records:
        values = updates.get((record.target_clip_uid, record.segment_id))
        if values is None:
            customized.append(record)
            continue
        text, language, probability = values
        customized.append(
            record.model_copy(
                update={
                    "text": text,
                    "language": language,
                    "diagnostics": _diagnostics(
                        language=language,
                        probability=probability,
                    ),
                }
            )
        )
    summary = _asr_v2_summary(
        inventory=inventory,
        records=customized,
        backend_provenance=customized[0].backend_provenance,
        backend_call_count=len(customized),
        failure_counts=Counter(),
    )
    _write_jsonl(path, customized)
    _write_json(
        root / "production" / "asr_v2" / "summary.json",
        summary.model_dump(mode="json"),
    )


@pytest.fixture
def final_source(tmp_path: Path):
    root = tmp_path / "audio-run"
    _write_asr_v1(
        root,
        ordered_clip_ids=[f"clip-{index:03d}" for index in reversed(range(20))],
    )
    _write_pairs(root)
    diarization = _write_diarization_source(root)
    asr_v1 = json.loads(
        (root / "asr_pilot20" / "inventory.json").read_text(encoding="utf-8")
    )
    from r2v_data_v2.h3.asr_transcription import ASRInventory

    inventory = _build(
        root,
        mode="production",
        asr_v1=ASRInventory.model_validate(asr_v1),
        diarization=diarization,
    )
    run_asr_v2_transcription(
        inventory=inventory,
        output_root=root / "production" / "asr_v2",
        backend=_FakeBackend(),
    )
    _customize_asr(root, inventory)
    publish_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
    )
    for clip_uid in ("clip-000", "clip-001", "clip-019"):
        _visual_clip(root, clip_uid=clip_uid)
    return root, inventory.inventory_fingerprint


def _samples(root: Path) -> list[FinalH3Sample]:
    return [
        FinalH3Sample.model_validate(json.loads(line))
        for line in (root / "production/h3/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_final_renderer_maps_subjects_text_timeline_and_language(final_source) -> None:
    root, fingerprint = final_source
    before = _tree_hashes(root)

    summary = publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    samples = _samples(root)
    in_sample = next(item for item in samples if item.pair_id == "in_pair/clip-000")

    assert summary.total_sample_count == len(samples) == 4
    assert summary.in_pair_sample_count == 3
    assert summary.cross_pair_sample_count == 1
    assert in_sample.subjects[0].subject_id == "subject_1"
    assert in_sample.subjects[0].voice_occurrence_id == "clip-000/e1"
    assert in_sample.subjects[0].donor_occurrence_id is None
    assert [item.start_time for item in in_sample.speech_segments] == [0.0, 0.5, 1.5]
    assert in_sample.speech_segments[0].rendered_dialogue == (
        "<Subject 1> says <d>保留原文。</d>"
    )
    assert in_sample.speech_segments[1].rendered_dialogue == (
        "<Subject 1> says <d>Keep exact text.</d>"
    )
    assert in_sample.speech_segments[2].text_status == "hidden"
    assert in_sample.speech_segments[2].rendered_dialogue is None
    assert in_sample.detected_languages == ["de", "en", "zh"]
    assert in_sample.language_conditioning_applied is False
    assert "Language:" not in in_sample.rendered_annotation
    assert "dialogue is" not in in_sample.rendered_annotation
    assert in_sample.visual_r2v_instruction == "Use <Image 1> beside <Image 2>."
    assert _tree_hashes(root, exclude_h3=True) == before


def test_background_and_unresolved_trusted_text_stays_structured(final_source) -> None:
    root, fingerprint = final_source
    publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    by_id = {item.pair_id: item for item in _samples(root)}

    background = by_id["in_pair/clip-001"].speech_segments[0]
    unresolved = next(
        item
        for item in by_id["in_pair/clip-019"].speech_segments
        if item.segment_id == "segment_002"
    )
    assert background.identity_status == "background_identified"
    assert background.subject_id is None
    assert background.trusted_text == "Background speaker."
    assert background.dialogue_status == "trusted_text_unrendered_no_subject"
    assert background.rendered_dialogue is None
    assert unresolved.identity_status == "unresolved"
    assert unresolved.entity_id is None
    assert unresolved.subject_id is None
    assert unresolved.trusted_text == "Unresolved speaker."
    assert unresolved.dialogue_status == "trusted_text_unrendered_no_subject"


def test_cross_pair_changes_only_voice_and_keeps_target_timeline(final_source) -> None:
    root, fingerprint = final_source
    publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    by_id = {item.pair_id: item for item in _samples(root)}
    target = by_id["in_pair/clip-000"]
    cross = by_id["cross_pair/clip-000/1"]

    assert cross.target_video.path == target.target_video.path
    assert cross.target_full_audio.path == target.target_full_audio.path
    assert cross.subjects[0].picture.path == target.subjects[0].picture.path
    assert cross.subjects[0].voice_reference.path != (
        target.subjects[0].voice_reference.path
    )
    assert cross.subjects[0].voice_occurrence_id == "clip-001/e2"
    assert cross.subjects[0].donor_occurrence_id == "clip-001/e2"
    assert cross.speech_segments == target.speech_segments
    assert cross.rendered_annotation == target.rendered_annotation


def test_assets_are_content_addressed_deduplicated_and_schemas_are_new(final_source) -> None:
    root, fingerprint = final_source
    summary = publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    output = root / "production" / "h3"
    inventory = FinalH3Inventory.model_validate_json(
        (output / "inventory.json").read_text(encoding="utf-8")
    )
    stored_summary = FinalH3Summary.model_validate_json(
        (output / "summary.json").read_text(encoding="utf-8")
    )

    assert inventory.schema_version == "r2v.h3.final_inventory.1"
    assert all(item.schema_version == "r2v.h3.final_sample.1" for item in _samples(root))
    assert stored_summary == summary
    assert inventory.pair_row_count == 4
    assert inventory.parent_quota_applied is False
    assert inventory.model_calls == inventory.mllm_calls == 0
    assert output == root / "production/h3"
    assert summary.target_video_asset_count == 3
    assert summary.full_audio_asset_count == 1
    assert not list((root / "production").glob("h3-*"))
    for path in (output / "assets").rglob("*"):
        if path.is_file():
            assert path.stem == _sha256(path)


def test_dry_run_is_read_only_and_publication_detects_source_mutation(final_source) -> None:
    root, fingerprint = final_source
    before = _tree_hashes(root)
    plan = plan_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )

    assert plan["renderable_sample_count"] == 4
    assert plan["pair_row_count"] == 4
    assert plan["dry_run"] is True
    assert not (root / "production/h3").exists()
    assert _tree_hashes(root) == before

    source = root / "production/pairs/summary.json"

    def mutate_source() -> None:
        source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source changed"):
        publish_final_h3_renderer(
            audio_run_root=root,
            expected_asr_inventory_fingerprint=fingerprint,
            before_publish=mutate_source,
        )
    assert not (root / "production/h3").exists()
    assert not list((root / "production").glob(".h3.tmp-*"))


def test_overwrite_is_atomic_and_cli_has_no_model_options(final_source) -> None:
    root, fingerprint = final_source
    first = publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    first_hashes = _tree_hashes(root / "production/h3")
    second = publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
        overwrite=True,
    )
    parser = renderer_cli._parser()
    args = parser.parse_args(["--audio-run-root", str(root), "--dry-run"])

    assert second == first
    assert _tree_hashes(root / "production/h3") == first_hashes
    assert args.dry_run is True
    assert not list((root / "production").glob(".h3.old-*"))
    source = Path(renderer_module.__file__).read_text(encoding="utf-8").lower()
    assert "torch" not in source
    assert "cuda" not in source
    assert "whisperasrbackend" not in source
    assert "diarizenbackend" not in source
    assert "openai" not in source


def test_missing_frozen_visual_instruction_fails_only_affected_samples(
    final_source,
) -> None:
    root, fingerprint = final_source
    path = root / "visual-run/clips/clip-000/clip.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instruction"] = {
        "status": "failed",
        "reason": "fixture instruction unavailable",
    }
    payload["export"] = {
        "accepted": False,
        "reason": "instruction unavailable",
    }
    clip = ClipRecord.model_validate(payload)
    _write_json(path, clip.model_dump(mode="json"))

    summary = publish_final_h3_renderer(
        audio_run_root=root,
        expected_asr_inventory_fingerprint=fingerprint,
    )
    failures = [
        json.loads(line)
        for line in (root / "production/h3/failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert summary.total_sample_count == 2
    assert summary.failed_sample_count == 2
    assert summary.failure_reason_counts == {"visual_instruction_unavailable": 2}
    assert {item["pair_id"] for item in failures} == {
        "in_pair/clip-000",
        "cross_pair/clip-000/1",
    }
