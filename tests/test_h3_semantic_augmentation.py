from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.audio_production import (
    H3ProductionInPair,
    H3ProductionInPairSubject,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioEntityBinding,
    AudioTrackMetadata,
    BindingEvidence,
    H3AudioBindingIR,
    H3TaskSpecification,
    PictureAsset,
    SemanticSubject,
)
from r2v_data_v2.h3.semantic_augmentation import (
    DEFAULT_MAX_BASE64_BYTES,
    OmniSemanticResponse,
    OpenAIQwenOmniBackend,
    QwenOmniSemanticConfig,
    SemanticAugmentationFailure,
    SemanticBackendResult,
    SemanticInventoryItem,
    SemanticNonSpeechEvent,
    SemanticSpeechTurnTranscript,
    _response_validation_issues,
    build_semantic_inventory,
    run_semantic_augmentation,
    semantic_output_root,
)
from tools.run_h3_omni_semantic import _parser
from tools.run_h3_omni_semantic import main as semantic_main


def _binding(
    *,
    start: float,
    end: float,
    entity_id: str,
    face_track_id: str,
) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        entity_id=entity_id,
        face_track_id=face_track_id,
        status="bound",
        confidence=0.93,
        evidence=BindingEvidence(
            active_face_track_ids=[face_track_id],
            face_speaking_probabilities={face_track_id: 0.9},
            association_confidence=0.95,
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=True,
        ),
    )


def _write_target(
    pairs_root: Path,
    *,
    clip_uid: str,
    subject_count: int = 1,
) -> H3ProductionInPair:
    media = pairs_root.parent / "fixture_media"
    media.mkdir(parents=True, exist_ok=True)
    video = media / f"{clip_uid}.mp4"
    video.write_bytes(f"video-with-audio:{clip_uid}".encode())
    full_audio = media / f"{clip_uid}.flac"
    full_audio.write_bytes(f"audio:{clip_uid}".encode())
    bindings = [
        _binding(
            start=float(index - 1) * 1.2,
            end=float(index - 1) * 1.2 + 1.0,
            entity_id=f"e{index}",
            face_track_id=f"face_{index}",
        )
        for index in range(1, subject_count + 1)
    ]
    pictures = [
        PictureAsset(
            picture_id=f"picture_{index}",
            entity_id=f"e{index}",
            path=f"clips/{clip_uid}/e{index}.png",
        )
        for index in range(1, subject_count + 1)
    ]
    subjects = [
        SemanticSubject(
            subject_id=f"subject_{index}",
            entity_id=f"e{index}",
            reference_type="subject",
            phrase=f"person {index}",
            source_assets=[f"picture_{index}"],
        )
        for index in range(1, subject_count + 1)
    ]
    sidecar = AudioBindingSidecar(
        clip_uid=clip_uid,
        source_run_root="/read-only/visual-run",
        source_video_path=str(video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid=clip_uid,
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(video),
                full_audio_path=str(full_audio),
                duration_seconds=max(3.0, subject_count * 1.2),
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
        bindings=bindings,
        h3_ir=H3AudioBindingIR(
            clip_uid=clip_uid,
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=pictures,
            subjects=subjects,
            audio_assets=[],
            bindings=bindings,
        ),
    )
    sidecar_path = pairs_root.parent / "audio" / "clips" / clip_uid / "audio_binding.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return H3ProductionInPair(
        pair_id=f"in_pair/{clip_uid}",
        target_clip_uid=clip_uid,
        target_video_path=str(video),
        target_full_audio_path=str(full_audio),
        target_audio_binding_path=str(sidecar_path),
        subjects=[
            H3ProductionInPairSubject(
                subject_index=index,
                target_occurrence_id=f"{clip_uid}/e{index}",
                target_entity_id=f"e{index}",
                target_visual_reference_path=f"refs/{clip_uid}/e{index}.png",
                target_primary_voice_reference_path=f"voice/{clip_uid}/e{index}.flac",
            )
            for index in range(1, subject_count + 1)
        ],
    )


def _write_pairs(
    root: Path,
    specs: list[tuple[str, int]],
) -> Path:
    pairs_root = root / "production" / "pairs"
    pairs_root.mkdir(parents=True)
    pairs = [
        _write_target(pairs_root, clip_uid=clip_uid, subject_count=count)
        for clip_uid, count in specs
    ]
    (pairs_root / "in_pairs.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in reversed(pairs)),
        encoding="utf-8",
    )
    (pairs_root / "cross_pairs.jsonl").write_text(
        json.dumps(
            {
                "target_clip_uid": specs[0][0],
                "donor_clip_uid": "donor-secret",
                "donor_video_path": "/must/not/be/read.mp4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pairs_root


def _response(job: SemanticInventoryItem, *, uncertain: bool = False) -> OmniSemanticResponse:
    return OmniSemanticResponse(
        speech_turn_transcripts=[
            SemanticSpeechTurnTranscript(
                **turn.model_dump(mode="python"),
                status="uncertain" if uncertain else "transcribed",
                text=None if uncertain else f"Transcript for {turn.turn_id}.",
                language=None if uncertain else "English",
            )
            for turn in job.speech_turns
        ],
        non_speech_events=[
            SemanticNonSpeechEvent(
                start_time=0.0,
                end_time=0.5,
                category="environmental",
                description="Faint room noise is audible.",
            )
        ],
        audiovisual_summary="A person speaks in the target video.",
        warnings=[],
    )


class _FakeBackend:
    model_identifier = "test/qwen-omni"

    provenance = QwenOmniSemanticConfig(
        base_url="https://example.invalid/v1",
        api_key="not-recorded",
        model=model_identifier,
    ).provenance()

    def __init__(self, *, fail_clip: str | None = None) -> None:
        self.fail_clip = fail_clip
        self.calls: list[str] = []

    def augment(self, job: SemanticInventoryItem) -> SemanticBackendResult:
        self.calls.append(job.target_clip_uid)
        if job.target_clip_uid == self.fail_clip:
            raise SemanticAugmentationFailure(
                code="structured_output_failed",
                reason="fixture malformed output",
                raw_responses=["not json", "still not json"],
                attempt_count=2,
            )
        return SemanticBackendResult(
            response=_response(job),
            raw_responses=("{\"valid\":true}",),
        )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_inventory_deduplicates_targets_and_preserves_frozen_bound_turns(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-b", 1), ("clip-a", 2)])

    inventory = build_semantic_inventory(pairs_root=pairs, mode="production")

    assert [item.target_clip_uid for item in inventory.jobs] == ["clip-a", "clip-b"]
    assert inventory.source_target_count == inventory.selected_target_count == 2
    assert inventory.bounded_selection_applied is False
    turns = inventory.jobs[0].speech_turns
    assert [(item.turn_id, item.entity_id) for item in turns] == [
        ("turn_1", "e1"),
        ("turn_2", "e2"),
    ]
    assert [(item.start_time, item.end_time) for item in turns] == [
        (0.0, 1.0),
        (1.2, 2.2),
    ]


def test_pilot20_forces_multi_subject_then_fills_deterministically(tmp_path: Path) -> None:
    specs = [(f"clip-{index:02d}", 1) for index in range(22)]
    specs[-1] = ("clip-99", 2)
    pairs = _write_pairs(tmp_path, specs)

    first = build_semantic_inventory(pairs_root=pairs, mode="pilot20")
    second = build_semantic_inventory(pairs_root=pairs, mode="pilot20")

    assert first == second
    assert first.selected_target_count == 20
    assert first.jobs[0].target_clip_uid == "clip-99"
    assert [item.target_clip_uid for item in first.jobs[1:]] == [
        f"clip-{index:02d}" for index in range(19)
    ]


def test_production_calls_once_per_target_and_never_reads_cross_donor(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1), ("clip-b", 1)])
    before = _tree_hashes(pairs)
    inventory = build_semantic_inventory(pairs_root=pairs, mode="production")
    backend = _FakeBackend()

    summary = run_semantic_augmentation(
        inventory=inventory,
        output_root=tmp_path / "production" / "semantic",
        backend=backend,
    )

    assert backend.calls == ["clip-a", "clip-b"]
    assert "donor-secret" not in (tmp_path / "production" / "semantic" / "inventory.json").read_text()
    assert summary.semantic_record_count == 2
    assert summary.valid_pair_inputs_retained_count == 2
    assert summary.donor_media_used is False
    assert summary.backend_provenance.output_modalities == ["text"]
    assert "api_key" not in summary.backend_provenance.model_dump()
    assert _tree_hashes(pairs) == before


def test_failed_semantics_publish_null_dialogue_without_deleting_pair(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1), ("clip-b", 1)])
    before = _tree_hashes(pairs)
    inventory = build_semantic_inventory(pairs_root=pairs, mode="production")

    summary = run_semantic_augmentation(
        inventory=inventory,
        output_root=tmp_path / "production" / "semantic",
        backend=_FakeBackend(fail_clip="clip-b"),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "production" / "semantic" / "records.jsonl")
        .read_text()
        .splitlines()
    ]
    failed = next(item for item in records if item["target_clip_uid"] == "clip-b")
    assert failed["status"] == "failed"
    assert failed["failure"]["code"] == "structured_output_failed"
    assert failed["speech_turn_transcripts"][0]["text"] is None
    assert failed["audiovisual_summary"] is None
    assert summary.failed_count == 1
    assert summary.repair_call_count == 1
    assert _tree_hashes(pairs) == before


def test_frozen_turn_validation_rejects_unknown_missing_and_changed_fields(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1)])
    job = build_semantic_inventory(pairs_root=pairs, mode="production").jobs[0]
    valid = _response(job)

    unknown = valid.model_copy(
        update={
            "speech_turn_transcripts": [
                valid.speech_turn_transcripts[0].model_copy(update={"turn_id": "turn_9"})
            ]
        }
    )
    changed_time = valid.model_copy(
        update={
            "speech_turn_transcripts": [
                valid.speech_turn_transcripts[0].model_copy(update={"start_time": 0.1})
            ]
        }
    )
    changed_entity = valid.model_copy(
        update={
            "speech_turn_transcripts": [
                valid.speech_turn_transcripts[0].model_copy(update={"entity_id": "e2"})
            ]
        }
    )

    assert {item.code for item in _response_validation_issues(unknown, job)} == {
        "unknown_turn_id",
        "missing_turn_id",
    }
    assert [item.code for item in _response_validation_issues(changed_time, job)] == [
        "frozen_turn_field_changed"
    ]
    assert [item.code for item in _response_validation_issues(changed_entity, job)] == [
        "frozen_turn_field_changed"
    ]


def test_uncertain_speech_requires_null_text(tmp_path: Path) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1)])
    job = build_semantic_inventory(pairs_root=pairs, mode="production").jobs[0]

    response = _response(job, uncertain=True)

    assert response.speech_turn_transcripts[0].text is None
    with pytest.raises(ValueError, match="must not guess text"):
        SemanticSpeechTurnTranscript(
            **job.speech_turns[0].model_dump(mode="python"),
            status="uncertain",
            text="Maybe hello",
        )


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        midpoint = len(response) // 2
        return [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
            )
            for part in (response[:midpoint], response[midpoint:])
        ]


def _client(responses: list[str]) -> tuple[SimpleNamespace, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    return (
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        completions,
    )


def test_openai_backend_streams_text_only_and_repairs_malformed_json(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1)])
    job = build_semantic_inventory(pairs_root=pairs, mode="production").jobs[0]
    valid = _response(job).model_dump_json()
    client, completions = _client(["not json", valid])
    backend = OpenAIQwenOmniBackend(
        QwenOmniSemanticConfig(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            max_base64_bytes=DEFAULT_MAX_BASE64_BYTES,
        ),
        client=client,
    )

    result = backend.augment(job)

    assert result.response == _response(job)
    assert len(result.raw_responses) == len(completions.calls) == 2
    assert all(call["stream"] is True for call in completions.calls)
    assert all(call["modalities"] == ["text"] for call in completions.calls)
    assert all("audio" not in call for call in completions.calls)
    assert completions.calls[0]["messages"][1]["content"][0]["video_url"][
        "url"
    ].startswith("data:video/mp4;base64,")
    request_text = json.dumps(completions.calls[0]["messages"])
    assert "donor-secret" not in request_text
    assert "/must/not/be/read.mp4" not in request_text
    assert "Repair the previous JSON only" in completions.calls[1]["messages"][1][
        "content"
    ][1]["text"]


def test_openai_backend_both_malformed_responses_fail_closed(tmp_path: Path) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1)])
    job = build_semantic_inventory(pairs_root=pairs, mode="production").jobs[0]
    client, completions = _client(["bad", "still bad"])
    backend = OpenAIQwenOmniBackend(
        QwenOmniSemanticConfig(
            base_url="https://example.invalid/v1",
            api_key="test-key",
        ),
        client=client,
    )

    with pytest.raises(SemanticAugmentationFailure) as exc_info:
        backend.augment(job)

    assert exc_info.value.code == "structured_output_failed"
    assert exc_info.value.attempt_count == 2
    assert len(completions.calls) == 2


def test_base64_preflight_fails_before_api_without_changing_source(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1)])
    job = build_semantic_inventory(pairs_root=pairs, mode="production").jobs[0]
    source = Path(job.target_video_path)
    before = source.read_bytes()
    client, completions = _client([])
    backend = OpenAIQwenOmniBackend(
        QwenOmniSemanticConfig(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            max_base64_bytes=4,
        ),
        client=client,
    )

    with pytest.raises(
        SemanticAugmentationFailure,
        match="encoded target video exceeds",
    ) as exc_info:
        backend.augment(job)

    assert exc_info.value.code == "local_base64_media_limit_exceeded"
    assert completions.calls == []
    assert source.read_bytes() == before


def test_review_page_and_fixed_output_roots_cover_every_record(tmp_path: Path) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", 1), ("clip-b", 1)])
    inventory = build_semantic_inventory(pairs_root=pairs, mode="pilot20")
    output = semantic_output_root(tmp_path, mode="pilot20")

    run_semantic_augmentation(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(),
    )

    review = (output / "review.html").read_text(encoding="utf-8")
    assert output == tmp_path / "semantic_pilot20"
    assert semantic_output_root(tmp_path, mode="production") == (
        tmp_path / "production" / "semantic"
    )
    assert review.count("class='case'") == 2
    assert all(clip_uid in review for clip_uid in ("clip-a", "clip-b"))
    assert all(label in review for label in ("CORRECT", "WRONG", "UNCERTAIN"))
    assert (output / "media" / "clip-a.mp4").is_symlink()


def test_cli_has_no_limit_or_parent_quota_options() -> None:
    destinations = {action.dest for action in _parser()._actions}

    assert "limit" not in destinations
    assert "max_clips_per_parent" not in destinations
    assert {"audio_run_root", "mode", "dry_run", "overwrite"} <= destinations


def test_cli_dry_run_builds_inventory_without_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_pairs(tmp_path, [("clip-a", 1), ("clip-b", 1)])
    for name in ("DASHSCOPE_API_KEY", "QWEN_OMNI_BASE_URL", "QWEN_OMNI_MODEL"):
        monkeypatch.delenv(name, raising=False)

    result = semantic_main(
        [
            "--audio-run-root",
            str(tmp_path),
            "--mode",
            "pilot20",
            "--dry-run",
        ]
    )

    assert result["selected_target_count"] == 2
    assert result["model_identifier"] == "qwen3.5-omni-plus"
    assert result["donor_media_used"] is False
    assert json.loads(capsys.readouterr().out)["selected_target_count"] == 2
