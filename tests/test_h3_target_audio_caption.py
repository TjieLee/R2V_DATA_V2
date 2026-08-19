from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption import (
    DEFAULT_DOTS3_CHECKPOINT_ID,
    QA_FLAGS,
    QA_LABELS,
    BackgroundMusic,
    Dots3TargetAudioCaptionConfig,
    Dots3TargetAudioCaptionResponse,
    ModelSpeakerDelivery,
    OpenAIDots3TargetAudioCaptionBackend,
    SpeakerClusterEvidence,
    SpeakerTimeRange,
    TargetAudioCaptionBackendFailure,
    TargetAudioCaptionBackendResult,
    TargetAudioCaptionHumanQAExport,
    TargetAudioCaptionInventory,
    TargetAudioCaptionJob,
    _model_input,
    _response_issues,
    _validate_pilot_selection,
    render_audio_prompt_draft,
    run_target_audio_caption_pilot,
    target_audio_caption_output_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _response(job: TargetAudioCaptionJob) -> Dots3TargetAudioCaptionResponse:
    return Dots3TargetAudioCaptionResponse(
        ambient_scene="busy indoor public space",
        background_music=BackgroundMusic(present=True, style="soft upbeat pop"),
        sound_events=["crowd chatter", "dish clatter"],
        acoustic_style=["moderately noisy", "reverberant"],
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=cluster.speaker_cluster_id,
                delivery_style=["calm", "conversational"],
            )
            for cluster in job.speaker_clusters
        ],
    )


class _FakeBackend:
    model_identifier = DEFAULT_DOTS3_CHECKPOINT_ID

    def __init__(self, root: Path, *, fail_clip: str | None = None) -> None:
        self.fail_clip = fail_clip
        self.calls: list[str] = []
        self.provenance = Dots3TargetAudioCaptionConfig(
            base_url="https://example.invalid/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=root),
        ).provenance()

    def describe(self, job: TargetAudioCaptionJob) -> TargetAudioCaptionBackendResult:
        self.calls.append(job.target_clip_uid)
        if job.target_clip_uid == self.fail_clip:
            raise TargetAudioCaptionBackendFailure(
                code="structured_output_failed",
                reason="fixture malformed output",
                raw_responses=("not json", "still not json"),
                attempt_count=2,
            )
        return TargetAudioCaptionBackendResult(
            response=_response(job),
            raw_responses=(json.dumps({"valid": True}),),
        )


def _inventory(tmp_path: Path) -> TargetAudioCaptionInventory:
    sources = {
        name: _write(tmp_path / "source" / name, name.encode())
        for name in (
            "pilot.json",
            "diar.json",
            "diar_raw.jsonl",
            "diar_bound.jsonl",
            "diar_clusters.jsonl",
            "diar_summary.json",
            "asr_inventory.json",
            "asr_segments.jsonl",
            "asr_summary.json",
            "text_inventory.json",
            "text_segments.jsonl",
            "text_summary.json",
        )
    }
    (tmp_path / "source" / "diarization").mkdir()
    (tmp_path / "source" / "asr_v2").mkdir()
    (tmp_path / "source" / "text_usability").mkdir()
    (tmp_path / "production" / "audio").mkdir(parents=True)
    jobs = []
    for index in range(20):
        clip_uid = f"clip-{index:03d}"
        video = _write(
            tmp_path / "media" / f"{clip_uid}.mp4", f"video:{clip_uid}".encode()
        )
        audio = _write(
            tmp_path / "media" / f"{clip_uid}.flac", f"audio:{clip_uid}".encode()
        )
        sidecar = _write(
            tmp_path / "audio" / clip_uid / "audio_binding.json",
            f"sidecar:{clip_uid}".encode(),
        )
        jobs.append(
            TargetAudioCaptionJob(
                target_clip_uid=clip_uid,
                target_video_path=str(video),
                target_video_sha256=_sha256(video),
                target_full_audio_path=str(audio),
                target_full_audio_sha256=_sha256(audio),
                target_audio_binding_path=str(sidecar),
                target_audio_binding_sha256=_sha256(sidecar),
                speaker_clusters=[
                    SpeakerClusterEvidence(
                        speaker_cluster_id="speaker_0",
                        entity_id="e1",
                        active_time_ranges=[
                            SpeakerTimeRange(start_time=0.2, end_time=1.4)
                        ],
                    )
                ],
            )
        )
    values = {
        "schema_version": "r2v.h3.target_audio_caption_inventory.1",
        "mode": "pilot20",
        "source_pilot_inventory_path": str(sources["pilot.json"]),
        "source_pilot_inventory_sha256": _sha256(sources["pilot.json"]),
        "source_pilot_inventory_fingerprint": "1" * 64,
        "source_diarization_root": str(tmp_path / "source" / "diarization"),
        "source_diarization_inventory_path": str(sources["diar.json"]),
        "source_diarization_inventory_sha256": _sha256(sources["diar.json"]),
        "source_diarization_inventory_fingerprint": "2" * 64,
        "source_diarization_raw_segments_path": str(sources["diar_raw.jsonl"]),
        "source_diarization_raw_segments_sha256": _sha256(sources["diar_raw.jsonl"]),
        "source_diarization_bound_segments_path": str(sources["diar_bound.jsonl"]),
        "source_diarization_bound_segments_sha256": _sha256(
            sources["diar_bound.jsonl"]
        ),
        "source_diarization_cluster_bindings_path": str(sources["diar_clusters.jsonl"]),
        "source_diarization_cluster_bindings_sha256": _sha256(
            sources["diar_clusters.jsonl"]
        ),
        "source_diarization_summary_path": str(sources["diar_summary.json"]),
        "source_diarization_summary_sha256": _sha256(sources["diar_summary.json"]),
        "source_asr_v2_root": str(tmp_path / "source" / "asr_v2"),
        "source_asr_v2_inventory_path": str(sources["asr_inventory.json"]),
        "source_asr_v2_inventory_sha256": _sha256(sources["asr_inventory.json"]),
        "source_asr_v2_inventory_fingerprint": "3" * 64,
        "source_asr_v2_segments_path": str(sources["asr_segments.jsonl"]),
        "source_asr_v2_segments_sha256": _sha256(sources["asr_segments.jsonl"]),
        "source_asr_v2_summary_path": str(sources["asr_summary.json"]),
        "source_asr_v2_summary_sha256": _sha256(sources["asr_summary.json"]),
        "source_text_usability_root": str(tmp_path / "source" / "text_usability"),
        "source_text_usability_inventory_path": str(sources["text_inventory.json"]),
        "source_text_usability_inventory_sha256": _sha256(
            sources["text_inventory.json"]
        ),
        "source_text_usability_inventory_fingerprint": "4" * 64,
        "source_text_usability_segments_path": str(sources["text_segments.jsonl"]),
        "source_text_usability_segments_sha256": _sha256(
            sources["text_segments.jsonl"]
        ),
        "source_text_usability_summary_path": str(sources["text_summary.json"]),
        "source_text_usability_summary_sha256": _sha256(sources["text_summary.json"]),
        "source_audio_root": str(tmp_path / "production" / "audio"),
        "selected_target_count": 20,
        "selection_mode": "exact_asr_v2_pilot20_order_v1",
        "bounded_selection_applied": True,
        "parent_quota_applied": False,
        "transcript_supplied_to_model": False,
        "final_renderer_applied": False,
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return TargetAudioCaptionInventory(**values, inventory_fingerprint=fingerprint)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_exact_asr_v2_pilot20_selection_order_is_reused() -> None:
    pilot_ids = [f"clip-{index:03d}" for index in range(19, -1, -1)]
    production_ids = [f"clip-{index:03d}" for index in range(75)]

    assert (
        _validate_pilot_selection(
            pilot_ids=pilot_ids,
            production_ids=production_ids,
        )
        == pilot_ids
    )
    with pytest.raises(ValueError, match="20 unique"):
        _validate_pilot_selection(
            pilot_ids=pilot_ids[:-1],
            production_ids=production_ids,
        )


def test_model_input_has_only_cluster_ids_and_ranges(tmp_path: Path) -> None:
    job = _inventory(tmp_path).jobs[0]
    payload = _model_input(job)
    encoded = json.dumps(payload)

    assert "speaker_0" in encoded
    assert "entity_id" not in encoded
    assert "transcript" not in encoded
    assert "trusted_text" not in encoded


def test_model_schema_forbids_entity_id_and_supports_unknown_values() -> None:
    with pytest.raises(ValidationError, match="entity_id"):
        ModelSpeakerDelivery.model_validate(
            {
                "speaker_cluster_id": "speaker_0",
                "entity_id": "e1",
                "delivery_style": ["calm"],
            }
        )
    response = Dots3TargetAudioCaptionResponse(
        ambient_scene=None,
        background_music=BackgroundMusic(present=None, style=None),
        sound_events=[],
        acoustic_style=[],
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id="speaker_0",
                delivery_style=[],
            )
        ],
    )
    assert render_audio_prompt_draft(response) == "Audio semantics are unknown."


def test_unknown_missing_and_duplicate_clusters_fail_validation(tmp_path: Path) -> None:
    job = _inventory(tmp_path).jobs[0]
    unknown = _response(job).model_copy(
        update={
            "speaker_delivery": [
                ModelSpeakerDelivery(
                    speaker_cluster_id="speaker_unknown",
                    delivery_style=[],
                )
            ]
        }
    )
    missing = _response(job).model_copy(update={"speaker_delivery": []})
    duplicate = _response(job).model_copy(
        update={"speaker_delivery": _response(job).speaker_delivery * 2}
    )

    assert _response_issues(unknown, job)[0].code == "unknown_speaker_cluster"
    assert _response_issues(missing, job)[0].code == "missing_speaker_cluster"
    assert _response_issues(duplicate, job)[0].code == "duplicate_speaker_cluster"


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


def test_dots3_request_is_text_plus_native_video_only_and_repairs(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    job = inventory.jobs[0]
    valid = _response(job).model_dump_json()
    completions = _FakeCompletions(["not json", valid])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIDots3TargetAudioCaptionBackend(
        Dots3TargetAudioCaptionConfig(
            base_url="https://example.invalid/v1",
            media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        ),
        client=client,
    )

    result = backend.describe(job)

    assert len(result.raw_responses) == 2
    assert len(completions.requests) == 2
    for request in completions.requests:
        content = request["messages"][1]["content"]  # type: ignore[index]
        assert [item["type"] for item in content] == ["text", "video_url"]
        assert all(item["type"] != "audio_url" for item in content)
        prompt = content[0]["text"]
        assert job.target_full_audio_path not in prompt
        assert "entity_id" not in json.dumps(_model_input(job))


def test_pilot_publication_is_atomic_read_only_and_continues_failures(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)
    source_before = _tree_hashes(tmp_path / "source") | _tree_hashes(tmp_path / "media")
    output = target_audio_caption_output_root(tmp_path)
    backend = _FakeBackend(tmp_path, fail_clip="clip-010")

    summary = run_target_audio_caption_pilot(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    assert backend.calls == [item.target_clip_uid for item in inventory.jobs]
    assert summary.ready_count == 19
    assert summary.failed_count == 1
    assert summary.initial_call_count == 20
    assert summary.repair_call_count == 1
    assert source_before == _tree_hashes(tmp_path / "source") | _tree_hashes(
        tmp_path / "media"
    )
    assert (output / "records.jsonl").is_file()
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "Export QA JSON" in report
    assert "target_audio_caption_pilot20_human_qa.json" in report
    assert "dialogue_leakage" in report
    assert "localStorage.clear" not in report
    assert "final_renderer" not in report


def test_failed_atomic_build_does_not_publish_partial_output(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = target_audio_caption_output_root(tmp_path)

    class _ProgrammingErrorBackend(_FakeBackend):
        def describe(
            self, job: TargetAudioCaptionJob
        ) -> TargetAudioCaptionBackendResult:
            raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        run_target_audio_caption_pilot(
            inventory=inventory,
            output_root=output,
            backend=_ProgrammingErrorBackend(tmp_path),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".target_audio_caption_pilot20.tmp-*"))


def test_audio_prompt_draft_is_deterministic(tmp_path: Path) -> None:
    response = _response(_inventory(tmp_path).jobs[0])
    assert render_audio_prompt_draft(response) == (
        "Busy indoor public space. Soft upbeat pop is audible. "
        "Audible sound events include crowd chatter, dish clatter. "
        "The acoustic style is moderately noisy, reverberant. "
        "speaker_0 speaks in a calm, conversational manner."
    )


def test_human_qa_contract_has_exact_labels_flags_and_unlabeled() -> None:
    export = TargetAudioCaptionHumanQAExport(
        inventory_fingerprint="a" * 64,
        label_count=1,
        counts={"CORRECT": 1, "WRONG": 0, "UNCERTAIN": 0, "UNLABELED": 19},
        labels=[
            {
                "target_clip_uid": "clip-000",
                "label": "CORRECT",
                "failure_flags": ["hallucinated_music"],
            }
        ],
    )
    assert QA_LABELS == ("CORRECT", "WRONG", "UNCERTAIN")
    assert "dialogue_leakage" in QA_FLAGS
    assert export.counts["UNLABELED"] == 19
