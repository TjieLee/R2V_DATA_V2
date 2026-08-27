from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import r2v_data_v2.h3.specialized_audio_semantics as specialized
from r2v_data_v2.h3.jea_target_audio_caption import (
    FALLBACK_SYSTEM_PROMPT as TARGET_AUDIO_FALLBACK_SYSTEM_PROMPT,
)
from r2v_data_v2.h3.jea_target_audio_caption import (
    JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION,
    JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
    JEATargetAudioCaptionInventory,
    JEATargetAudioCaptionJob,
    _inventory_fingerprint,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.specialized_audio_semantics import (
    ASSEMBLED_RECORD_VERSION,
    CAPTIONER_POLICY_VERSION,
    GLOBAL_FALLBACK_PROMPT_VERSION,
    GLOBAL_FALLBACK_SYSTEM_PROMPT,
    GLOBAL_PROMPT_VERSION,
    GLOBAL_RECAPTION_RESCUE_POLICY_VERSION,
    GLOBAL_SYSTEM_PROMPT,
    LOCAL_FIELD_POLICY_VERSION,
    LOCAL_PROMPT_VERSION,
    LOCAL_SYSTEM_PROMPT,
    SPECIALIZED_ROOT_NAME,
    CaptionerBackendResult,
    CaptionerConfig,
    GlobalAudioSemanticsResponse,
    GlobalSemanticsConfig,
    LocalAudioSemanticsResponse,
    LocalSemanticsConfig,
    OpenAICaptionerBackend,
    OpenAIGlobalSemanticsBackend,
    OpenAILocalSemanticsBackend,
    SpecializedBackendFailure,
    SpecializedBackendResult,
    captioner_request_fingerprint,
    global_request_fingerprint,
    local_request_fingerprint,
    run_assemble_phase,
    run_captioner_phase,
    run_global_semantics_phase,
    run_local_semantics_phase,
    run_specialized_pipeline,
)
from r2v_data_v2.h3.target_audio_caption_contract import (
    ModelSpeakerDelivery,
    SpeakerClusterEvidence,
    SpeakerTimeRange,
    TargetAudioCaptionResponse,
    TemporalAudioEvent,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(tmp_path: Path, *, clip_count: int = 3) -> JEATargetAudioCaptionInventory:
    root = tmp_path / "production"
    root.mkdir(parents=True)
    sources = {
        "source_pairs_path": root / "pairs.jsonl",
        "source_readable_segments_path": root / "readable.jsonl",
        "source_qwen3_asr_segments_path": root / "asr.jsonl",
    }
    for name, path in sources.items():
        path.write_text(name + "\n", encoding="utf-8")
    jobs: list[JEATargetAudioCaptionJob] = []
    for index in range(clip_count):
        clip_uid = f"clip-{index:03d}"
        media = root / "media" / clip_uid
        media.mkdir(parents=True)
        video = media / "video.mp4"
        audio = media / "full_audio.flac"
        binding = media / "audio_binding.json"
        video.write_bytes(f"video-{clip_uid}".encode())
        audio.write_bytes(f"audio-{clip_uid}".encode())
        binding.write_text("{}\n", encoding="utf-8")
        jobs.append(
            JEATargetAudioCaptionJob(
                target_clip_uid=clip_uid,
                clip_display_path=f"01/series/{clip_uid}",
                target_video_path=str(video),
                target_video_sha256=_sha256(video),
                target_full_audio_path=str(audio),
                target_full_audio_sha256=_sha256(audio),
                target_duration_seconds=2.0,
                target_audio_binding_path=str(binding),
                target_audio_binding_sha256=_sha256(binding),
                speaker_clusters=[
                    SpeakerClusterEvidence(
                        speaker_cluster_id="speaker_0",
                        entity_id="e1" if index % 2 == 0 else None,
                        active_time_ranges=[SpeakerTimeRange(start_time=0.1, end_time=1.2)],
                    )
                ],
            )
        )
    values = {
        "source_audio_production_root": str(root),
        **{name: str(path) for name, path in sources.items()},
        **{
            name.replace("_path", "_sha256"): _sha256(path)
            for name, path in sources.items()
        },
        "target_clip_count": clip_count,
        "readable_segment_count": clip_count,
        "qwen3_asr_segment_count": clip_count,
        "jobs": jobs,
    }
    provisional = JEATargetAudioCaptionInventory.model_construct(
        **values,
        inventory_fingerprint="",
    )
    return JEATargetAudioCaptionInventory(
        **values,
        inventory_fingerprint=_inventory_fingerprint(provisional),
    )


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> SimpleNamespace:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        message = SimpleNamespace(content=response)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        return SimpleNamespace(choices=[choice], usage=usage)


def _client(responses: list[object]) -> tuple[SimpleNamespace, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _resolver(inventory: JEATargetAudioCaptionInventory) -> MediaURLResolver:
    return MediaURLResolver(
        mode="file",
        media_root=Path(inventory.source_audio_production_root),
    )


def _caption_config(
    inventory: JEATargetAudioCaptionInventory,
    *,
    model: str = "cap",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 16384,
) -> CaptionerConfig:
    return CaptionerConfig(
        base_url="http://captioner.invalid/v1",
        api_key="EMPTY",
        served_model_name=model,
        checkpoint_id=f"/models/{model}",
        media_resolver=_resolver(inventory),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _global_config(*, model: str = "vl") -> GlobalSemanticsConfig:
    return GlobalSemanticsConfig(
        base_url="http://global.invalid/v1",
        api_key="EMPTY",
        served_model_name=model,
        checkpoint_id=f"/models/{model}",
    )


def _local_config(
    inventory: JEATargetAudioCaptionInventory,
    *,
    include_video: bool = False,
    model: str = "local",
) -> LocalSemanticsConfig:
    return LocalSemanticsConfig(
        base_url="http://local.invalid/v1",
        api_key="EMPTY",
        served_model_name=model,
        checkpoint_id=f"/models/{model}",
        media_resolver=_resolver(inventory),
        include_video=include_video,
    )


def _global_json(description: str | None = "soft rain") -> str:
    return GlobalAudioSemanticsResponse(
        overall_audio_description=description,
        overall_soundscape=None,
        non_diegetic_music=None,
    ).model_dump_json()


def _local_response(*, style: str | None = "calm") -> LocalAudioSemanticsResponse:
    return LocalAudioSemanticsResponse(
        temporal_audio_events=[
            TemporalAudioEvent(start_time=0.2, end_time=0.8, description="a door closes"),
            TemporalAudioEvent(start_time=0.5, end_time=0.9, description="footsteps"),
        ],
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id="speaker_0",
                delivery_style=style,
            )
        ],
    )


def _target_audio_response(
    *,
    style: str | None = "calm",
    overall_soundscape: str | None = "quiet room tone",
    non_diegetic_music: str | None = "soft score",
) -> TargetAudioCaptionResponse:
    local = _local_response(style=style)
    return TargetAudioCaptionResponse(
        overall_soundscape=overall_soundscape,
        non_diegetic_music=non_diegetic_music,
        temporal_audio_events=local.temporal_audio_events,
        speaker_delivery=local.speaker_delivery,
    )


def _job_with_speaker_clusters(
    inventory: JEATargetAudioCaptionInventory,
    *cluster_ids: str,
    duration_seconds: float = 4.0,
) -> JEATargetAudioCaptionJob:
    return inventory.jobs[0].model_copy(
        update={
            "target_duration_seconds": duration_seconds,
            "speaker_clusters": [
                SpeakerClusterEvidence(
                    speaker_cluster_id=cluster_id,
                    entity_id=f"e{index + 1}",
                    active_time_ranges=[
                        SpeakerTimeRange(start_time=0.1, end_time=1.2)
                    ],
                )
                for index, cluster_id in enumerate(cluster_ids)
            ],
        }
    )


def test_captioner_uses_only_canonical_audio_and_preserves_raw_text(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    raw = "  Rich caption with dialogue-like evidence kept verbatim.\n"
    client, completions = _client([raw])
    backend = OpenAICaptionerBackend(_caption_config(inventory), client=client)

    result = backend.caption(inventory.jobs[0])

    assert result.raw_audio_caption == raw
    request = completions.requests[0]
    content = request["messages"][0]["content"]  # type: ignore[index]
    assert [item["type"] for item in content] == ["audio_url"]  # type: ignore[index]
    serialized = json.dumps(request)
    assert "video_url" not in serialized
    assert "speaker_0" not in serialized
    assert "entity_id" not in serialized
    assert "transcript" not in serialized
    assert request["temperature"] == 0.6
    assert request["top_p"] == 0.95
    assert request["max_tokens"] == 16384
    assert "top_k" not in request
    assert request["extra_body"] == {"top_k": 20}


def test_captioner_sampling_defaults_and_structured_stage_determinism(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    captioner = _caption_config(inventory)
    assert (
        captioner.temperature,
        captioner.top_p,
        captioner.top_k,
        captioner.max_tokens,
    ) == (0.6, 0.95, 20, 16384)
    captioner_provenance = captioner.provenance()
    assert captioner_provenance.temperature == 0.6
    assert captioner_provenance.top_p == 0.95
    assert captioner_provenance.top_k == 20
    for provenance in (
        _global_config().provenance(),
        _local_config(inventory).provenance(),
    ):
        assert provenance.temperature == 0
        assert provenance.top_p is None
        assert provenance.top_k is None


def test_captioner_transport_policy_bump_changes_fingerprint(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    config = _caption_config(inventory)
    current = config.provenance()
    previous = specialized._provenance(
        role="captioner",
        served_model_name=config.served_model_name,
        checkpoint_id=config.checkpoint_id,
        base_url=config.base_url,
        input_modality="canonical_full_audio_only",
        media_resolver=config.media_resolver,
        prompt_version="qwen3_omni_native_audio_caption_v1",
        fallback_prompt_version=None,
        fallback_policy_version="captioner_empty_retry_once_v1",
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
    )
    assert CAPTIONER_POLICY_VERSION == "qwen3_omni_native_audio_caption_v2"
    assert current.configuration_fingerprint != previous.configuration_fingerprint
    assert captioner_request_fingerprint(inventory.jobs[0], current) != (
        captioner_request_fingerprint(inventory.jobs[0], previous)
    )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("temperature", 0.0, "temperature"),
        ("top_p", 0.0, "top-p"),
        ("top_p", 1.1, "top-p"),
        ("top_k", 0, "top-k"),
        ("max_tokens", 0, "max tokens"),
    ],
)
def test_captioner_sampling_validation_fails_closed(
    tmp_path: Path,
    field_name: str,
    value: float,
    message: str,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    values: dict[str, object] = {
        "base_url": "http://captioner.invalid/v1",
        "api_key": "EMPTY",
        "served_model_name": "cap",
        "checkpoint_id": "/models/cap",
        "media_resolver": _resolver(inventory),
        field_name: value,
    }
    with pytest.raises(ValueError, match=message):
        CaptionerConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("first", [" \n\t", None])
def test_captioner_empty_or_non_string_retries_exactly_once(
    tmp_path: Path,
    first: object,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    client, completions = _client([first, "usable raw caption"])
    result = OpenAICaptionerBackend(
        _caption_config(inventory), client=client
    ).caption(inventory.jobs[0])
    assert result.raw_audio_caption == "usable raw caption"
    assert result.model_call_count == 2
    assert len(completions.requests) == 2


def test_captioner_infrastructure_error_does_not_retry(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    client, completions = _client([TimeoutError("offline")])
    backend = OpenAICaptionerBackend(_caption_config(inventory), client=client)
    with pytest.raises(SpecializedBackendFailure, match="TimeoutError"):
        backend.caption(inventory.jobs[0])
    assert len(completions.requests) == 1


def test_captioner_media_failure_is_isolated_per_clip(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    Path(inventory.jobs[0].target_full_audio_path).unlink()
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    client, completions = _client(["second clip caption"])
    records, summary = run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=OpenAICaptionerBackend(_caption_config(inventory), client=client),
    )
    assert [record.status for record in records] == ["failed", "ready"]
    assert records[0].failure is not None
    assert records[0].failure.code == "captioner_media_unavailable"
    assert summary.failed_count == 1 and summary.ready_count == 1
    assert len(completions.requests) == 1


def test_global_vl_is_text_only_and_repairs_invalid_json_once(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    client, completions = _client(["not-json", _global_json()])
    backend = OpenAIGlobalSemanticsBackend(_global_config(), client=client)
    raw_caption = "Possibly a gong or war horn behind Mandarin dialogue."

    result = backend.extract(inventory.jobs[0], raw_caption)

    assert result.response.overall_audio_description == "soft rain"
    assert result.model_call_count == 2
    serialized = json.dumps(completions.requests)
    assert raw_caption in serialized
    assert "audio_url" not in serialized and "video_url" not in serialized
    assert "response_format" in completions.requests[0]
    assert "human-language or vocal" in GLOBAL_SYSTEM_PROMPT
    assert "Never add facts" in GLOBAL_SYSTEM_PROMPT
    assert "Never choose one uncertain source" in GLOBAL_SYSTEM_PROMPT


def test_global_music_prompt_prefers_supported_music_recall() -> None:
    prompt = " ".join(GLOBAL_SYSTEM_PROMPT.split())
    assert GLOBAL_PROMPT_VERSION == "qwen3_vl_global_audio_extraction_v2"
    assert GLOBAL_FALLBACK_PROMPT_VERSION == (
        "qwen3_vl_global_audio_extraction_v2_recheck"
    )
    for phrase in (
        'Do not require the caption to explicitly say "background music"',
        '"BGM"',
        '"score"',
        "source is unspecified",
        "Prefer recall over abstention when evidence is clearly musical",
    ):
        assert phrase in prompt
    for evidence in (
        "melody",
        "melodic",
        "harmonic",
        "instrumental",
        "synth",
        "rhythmic musical backing",
    ):
        assert evidence in prompt


def test_global_music_prompt_preserves_diegetic_and_tonal_boundaries() -> None:
    prompt = " ".join(GLOBAL_SYSTEM_PROMPT.split())
    fallback_prompt = " ".join(GLOBAL_FALLBACK_SYSTEM_PROMPT.split())
    for source in (
        "radio",
        "television",
        "phone",
        "loudspeaker",
        "live band",
        "person playing guitar or piano",
    ):
        assert source in prompt
    for non_music in ("hum", "buzz", "whine", "tone", "electronic tone"):
        assert non_music in prompt
    assert "alone is not music" in prompt
    assert "source may be preserved as non_diegetic_music" in fallback_prompt
    assert "exclude it only when" in fallback_prompt
    assert "alone is not music" in fallback_prompt


def test_global_prompt_version_changes_only_global_fingerprints(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    job = inventory.jobs[0]
    global_config = _global_config(model="vl-a")
    current_global = global_config.provenance()
    previous_global = specialized._provenance(
        role="global_semantics",
        served_model_name=global_config.served_model_name,
        checkpoint_id=global_config.checkpoint_id,
        base_url=global_config.base_url,
        input_modality="captioner_text_only",
        media_resolver=None,
        prompt_version="qwen3_vl_global_audio_extraction_v1",
        fallback_prompt_version="qwen3_vl_global_audio_extraction_v1_recheck",
        fallback_policy_version=specialized.GLOBAL_FALLBACK_POLICY_VERSION,
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_tokens=global_config.max_tokens,
    )
    assert current_global.configuration_fingerprint != (
        previous_global.configuration_fingerprint
    )
    assert global_request_fingerprint("soft piano melody", current_global) != (
        global_request_fingerprint("soft piano melody", previous_global)
    )
    previous_rescue_policy = specialized._provenance(
        role="global_semantics",
        served_model_name=global_config.served_model_name,
        checkpoint_id=global_config.checkpoint_id,
        base_url=global_config.base_url,
        input_modality="captioner_text_only",
        media_resolver=None,
        prompt_version=GLOBAL_PROMPT_VERSION,
        fallback_prompt_version=GLOBAL_FALLBACK_PROMPT_VERSION,
        fallback_policy_version=specialized.GLOBAL_FALLBACK_POLICY_VERSION,
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_tokens=global_config.max_tokens,
    )
    assert GLOBAL_RECAPTION_RESCUE_POLICY_VERSION == (
        "global_missing_semantics_recaption_once_v1"
    )
    assert current_global.configuration_fingerprint != (
        previous_rescue_policy.configuration_fingerprint
    )
    assert global_request_fingerprint("soft piano melody", current_global) != (
        global_request_fingerprint("soft piano melody", previous_rescue_policy)
    )

    captioner = _caption_config(inventory).provenance()
    local = _local_config(inventory).provenance()
    assert captioner.prompt_version == CAPTIONER_POLICY_VERSION
    assert local.prompt_version == LOCAL_PROMPT_VERSION
    assert captioner_request_fingerprint(job, captioner) == (
        captioner_request_fingerprint(job, _caption_config(inventory).provenance())
    )
    assert local_request_fingerprint(job, local) == (
        local_request_fingerprint(job, _local_config(inventory).provenance())
    )


@pytest.mark.parametrize("initial", [_global_json(None), "  "])
def test_global_all_null_or_whitespace_rechecks_without_field_merge(
    tmp_path: Path,
    initial: str,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    fallback = GlobalAudioSemanticsResponse(
        overall_audio_description=None,
        overall_soundscape="quiet room tone",
        non_diegetic_music=None,
    ).model_dump_json()
    client, completions = _client([initial, fallback])
    result = OpenAIGlobalSemanticsBackend(
        _global_config(), client=client
    ).extract(inventory.jobs[0], "quiet room tone")
    assert result.response.overall_audio_description is None
    assert result.response.overall_soundscape == "quiet room tone"
    assert result.fallback_attempted is True
    assert len(completions.requests) == 2


def test_global_fallback_failure_preserves_primary_and_fallback_diagnostics(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    responses = [" ", "not-json", "still-not-json"]
    client, completions = _client(responses.copy())

    with pytest.raises(SpecializedBackendFailure) as captured:
        OpenAIGlobalSemanticsBackend(
            _global_config(), client=client
        ).extract(inventory.jobs[0], "quiet room tone")

    failure = captured.value
    assert failure.code == "global_semantics_structured_output_failed"
    assert failure.model_call_count == 3
    assert failure.raw_responses == tuple(responses)
    assert len(failure.diagnostics) == 3
    assert [item.raw_content_char_count for item in failure.diagnostics] == [
        len(item) for item in responses
    ]
    assert failure.issues
    assert len(completions.requests) == 3


def test_global_infrastructure_failure_has_no_semantic_recheck(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    client, completions = _client([ConnectionError("down")])
    with pytest.raises(SpecializedBackendFailure):
        OpenAIGlobalSemanticsBackend(
            _global_config(), client=client
        ).extract(inventory.jobs[0], "rain")
    assert len(completions.requests) == 1


def test_local_contract_cluster_validation_overlap_and_media_modes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    valid = _target_audio_response().model_dump_json()
    for include_video, expected_types in (
        (False, ["text", "audio_url"]),
        (True, ["text", "video_url", "audio_url"]),
    ):
        client, completions = _client([valid])
        backend = OpenAILocalSemanticsBackend(
            _local_config(inventory, include_video=include_video),
            client=client,
        )
        result = backend.describe(inventory.jobs[0])
        content = completions.requests[0]["messages"][1]["content"]  # type: ignore[index]
        assert [item["type"] for item in content] == expected_types  # type: ignore[index]
        assert len(result.response.temporal_audio_events) == 2
        assert set(result.response.model_dump()) == {
            "temporal_audio_events",
            "speaker_delivery",
        }
        assert not hasattr(result.response, "overall_soundscape")
        assert not hasattr(result.response, "non_diegetic_music")
        serialized = json.dumps(completions.requests)
        assert "overall_soundscape" in LOCAL_SYSTEM_PROMPT
        assert "non_diegetic_music" in LOCAL_SYSTEM_PROMPT
        assert completions.requests[0]["messages"][0]["content"] == (
            LOCAL_SYSTEM_PROMPT
        )
        assert len(completions.requests) == 1
        assert "SECRET TRANSCRIPT" not in serialized


def test_local_reuses_verified_target_audio_prompt_versions(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    config = _local_config(inventory)
    provenance = config.provenance()
    assert LOCAL_PROMPT_VERSION == JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION
    assert provenance.prompt_version == "h3_target_audio_semantics_v2"
    assert provenance.fallback_prompt_version == (
        JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
    )
    assert LOCAL_FIELD_POLICY_VERSION == "specialized_local_field_salvage_v2"
    previous = specialized._provenance(
        role="local_semantics",
        served_model_name=config.served_model_name,
        checkpoint_id=config.checkpoint_id,
        base_url=config.base_url,
        input_modality="canonical_full_audio_only",
        media_resolver=config.media_resolver,
        prompt_version=LOCAL_PROMPT_VERSION,
        fallback_prompt_version=JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION,
        fallback_policy_version=specialized.LOCAL_FALLBACK_POLICY_VERSION,
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_tokens=config.max_tokens,
    )
    assert provenance.configuration_fingerprint != previous.configuration_fingerprint
    assert local_request_fingerprint(inventory.jobs[0], provenance) != (
        local_request_fingerprint(inventory.jobs[0], previous)
    )


def test_local_full_response_drives_fallback_before_field_projection(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    full_response = TargetAudioCaptionResponse(
        overall_soundscape="steady rain outside",
        non_diegetic_music=None,
        temporal_audio_events=[],
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id="speaker_0",
                delivery_style=None,
            )
        ],
    )
    client, completions = _client([full_response.model_dump_json()])

    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(inventory.jobs[0])

    assert result.semantic_source == "primary"
    assert result.fallback_attempted is False
    assert result.response.temporal_audio_events == []
    assert result.response.speaker_delivery[0].delivery_style is None
    assert len(completions.requests) == 1


def test_local_normalizes_event_and_speaker_order_without_repair(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    job = _job_with_speaker_clusters(inventory, "speaker_0", "speaker_1")
    raw = json.dumps(
        {
            "overall_soundscape": "quiet room tone",
            "non_diegetic_music": None,
            "temporal_audio_events": [
                {"start_time": 3.3125, "end_time": 3.5, "description": "third"},
                {"start_time": 0.6725, "end_time": 0.9, "description": "first"},
                {"start_time": 2.2925, "end_time": 2.6, "description": "second"},
            ],
            "speaker_delivery": [
                {
                    "speaker_cluster_id": "speaker_1",
                    "delivery_style": "brief and neutral",
                },
                {
                    "speaker_cluster_id": "speaker_0",
                    "delivery_style": "calm and conversational",
                },
            ],
        }
    )
    client, completions = _client([raw])

    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(job)

    assert [item.start_time for item in result.response.temporal_audio_events] == [
        0.6725,
        2.2925,
        3.3125,
    ]
    assert [item.speaker_cluster_id for item in result.response.speaker_delivery] == [
        "speaker_0",
        "speaker_1",
    ]
    assert result.model_call_count == 1
    assert len(completions.requests) == 1


@pytest.mark.parametrize(
    "wrapped",
    [
        "Assistant: {payload}",
        "Assistant:\n```json\n{payload}\n```",
    ],
)
def test_local_accepts_exact_assistant_json_envelopes(
    tmp_path: Path,
    wrapped: str,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    raw = wrapped.format(payload=_target_audio_response().model_dump_json())
    client, completions = _client([raw])

    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(inventory.jobs[0])

    assert result.response.speaker_delivery[0].delivery_style == "calm"
    assert result.model_call_count == 1
    assert len(completions.requests) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        {"temporal_audio_events": [], "speaker_delivery": []},
        {
            "temporal_audio_events": [],
            "speaker_delivery": [
                {"speaker_cluster_id": "unknown", "delivery_style": None}
            ],
        },
        {
            "temporal_audio_events": [],
            "speaker_delivery": [
                {"speaker_cluster_id": "speaker_0", "delivery_style": None},
                {"speaker_cluster_id": "speaker_0", "delivery_style": None},
            ],
        },
        {
            "temporal_audio_events": [
                {"start_time": 1.9, "end_time": 2.2, "description": "impact"}
            ],
            "speaker_delivery": [
                {"speaker_cluster_id": "speaker_0", "delivery_style": None}
            ],
        },
    ],
)
def test_local_invalid_cluster_or_timing_repairs_once(
    tmp_path: Path,
    invalid: dict[str, object],
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    client, completions = _client([json.dumps(invalid), _local_response().model_dump_json()])
    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(inventory.jobs[0])
    assert result.model_call_count == 2
    assert len(completions.requests) == 2


def test_local_all_null_and_whitespace_recheck(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    null_response = TargetAudioCaptionResponse(
        overall_soundscape=None,
        non_diegetic_music=None,
        temporal_audio_events=[],
        speaker_delivery=[
            ModelSpeakerDelivery(speaker_cluster_id="speaker_0", delivery_style=None)
        ],
    ).model_dump_json()
    client, completions = _client([null_response, null_response])
    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(inventory.jobs[0])
    assert result.semantic_source == "fallback_all_null_confirmed"
    assert len(completions.requests) == 2
    assert completions.requests[1]["messages"][0]["content"] == (
        TARGET_AUDIO_FALLBACK_SYSTEM_PROMPT
    )

    client, completions = _client([" ", _target_audio_response().model_dump_json()])
    result = OpenAILocalSemanticsBackend(
        _local_config(inventory), client=client
    ).describe(inventory.jobs[0])
    assert result.response.speaker_delivery[0].delivery_style == "calm"
    assert len(completions.requests) == 2


def test_local_empty_fallback_preserves_primary_and_fallback_diagnostics(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    responses = [" ", "\t"]
    client, completions = _client(responses.copy())

    with pytest.raises(SpecializedBackendFailure) as captured:
        OpenAILocalSemanticsBackend(
            _local_config(inventory), client=client
        ).describe(inventory.jobs[0])

    failure = captured.value
    assert failure.code == "local_semantics_empty_response"
    assert failure.model_call_count == 2
    assert failure.raw_responses == tuple(responses)
    assert len(failure.diagnostics) == 2
    assert all(item.whitespace_only for item in failure.diagnostics)
    assert len(completions.requests) == 2


def test_response_schema_is_strict_and_local_has_only_two_fields() -> None:
    assert list(GlobalAudioSemanticsResponse.model_fields) == [
        "overall_audio_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]
    assert list(LocalAudioSemanticsResponse.model_fields) == [
        "temporal_audio_events",
        "speaker_delivery",
    ]
    with pytest.raises(ValidationError):
        GlobalAudioSemanticsResponse.model_validate(
            {
                "overall_audio_description": None,
                "overall_soundscape": None,
                "non_diegetic_music": None,
                "dialogue": "forbidden",
            }
        )


def test_request_fingerprints_cover_semantics_but_not_inflight(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    job = inventory.jobs[0]
    caption_a = _caption_config(inventory, model="cap-a").provenance()
    caption_b = _caption_config(inventory, model="cap-b").provenance()
    assert captioner_request_fingerprint(job, caption_a) != captioner_request_fingerprint(
        job, caption_b
    )
    for changed in (
        _caption_config(inventory, model="cap-a", temperature=0.7),
        _caption_config(inventory, model="cap-a", top_p=0.9),
        _caption_config(inventory, model="cap-a", top_k=30),
        _caption_config(inventory, model="cap-a", max_tokens=8192),
    ):
        changed_provenance = changed.provenance()
        assert changed_provenance.configuration_fingerprint != (
            caption_a.configuration_fingerprint
        )
        assert captioner_request_fingerprint(job, changed_provenance) != (
            captioner_request_fingerprint(job, caption_a)
        )
    global_a = _global_config(model="vl-a").provenance()
    global_b = _global_config(model="vl-b").provenance()
    assert global_request_fingerprint("rain", global_a) != global_request_fingerprint(
        "rain", global_b
    )
    assert global_request_fingerprint("rain", global_a) != global_request_fingerprint(
        "wind", global_a
    )
    local_audio = _local_config(inventory).provenance()
    local_video = _local_config(inventory, include_video=True).provenance()
    assert local_request_fingerprint(job, local_audio) != local_request_fingerprint(
        job, local_video
    )
    assert GLOBAL_PROMPT_VERSION in global_a.prompt_version
    assert LOCAL_PROMPT_VERSION in local_audio.prompt_version


def test_review_namespace_covers_stage_configuration_only(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    caption = _caption_config(inventory).provenance()
    global_semantics = _global_config().provenance()
    local = _local_config(inventory).provenance()
    baseline = specialized.review_configuration_fingerprint(
        caption,
        global_semantics,
        local,
    )
    assert baseline == specialized.review_configuration_fingerprint(
        caption,
        global_semantics,
        local,
    )
    assert baseline != specialized.review_configuration_fingerprint(
        _caption_config(inventory, temperature=0.7).provenance(),
        global_semantics,
        local,
    )
    assert baseline != specialized.review_configuration_fingerprint(
        caption,
        _global_config(model="vl-other").provenance(),
        local,
    )
    assert baseline != specialized.review_configuration_fingerprint(
        caption,
        global_semantics,
        _local_config(inventory, include_video=True).provenance(),
    )


@dataclass
class _FakeCaptioner:
    config: CaptionerConfig
    failures: set[str] = field(default_factory=set)
    caption_prefix: str = "raw"
    calls: list[str] = field(default_factory=list)
    delay_by_clip: dict[str, float] = field(default_factory=dict)
    tracker: _ActivityTracker | None = None

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def caption(self, job: JEATargetAudioCaptionJob) -> CaptionerBackendResult:
        self.calls.append(job.target_clip_uid)
        if self.tracker:
            self.tracker.enter("captioner")
        try:
            time.sleep(self.delay_by_clip.get(job.target_clip_uid, 0.0))
            if job.target_clip_uid in self.failures:
                raise SpecializedBackendFailure(
                    code="captioner_fake_failure",
                    reason="fake failure",
                    model_call_count=1,
                )
            raw = f"{self.caption_prefix} {job.target_clip_uid}"
            return CaptionerBackendResult(
                raw_audio_caption=raw,
                raw_responses=(),
                diagnostics=(),
                model_call_count=1,
            )
        finally:
            if self.tracker:
                self.tracker.leave("captioner")


@dataclass
class _FakeGlobal:
    config: GlobalSemanticsConfig
    failures: set[str] = field(default_factory=set)
    description_prefix: str = ""
    calls: list[str] = field(default_factory=list)
    first_started: threading.Event | None = None
    tracker: _ActivityTracker | None = None

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def extract(self, job: JEATargetAudioCaptionJob, raw_audio_caption: str):  # type: ignore[no-untyped-def]
        self.calls.append(job.target_clip_uid)
        if self.first_started:
            self.first_started.set()
        if self.tracker:
            self.tracker.enter("global")
        try:
            if job.target_clip_uid in self.failures:
                raise SpecializedBackendFailure(
                    code="global_fake_failure",
                    reason="fake failure",
                    model_call_count=1,
                )
            response = GlobalAudioSemanticsResponse(
                overall_audio_description=f"{self.description_prefix}{raw_audio_caption}",
                overall_soundscape=None,
                non_diegetic_music="soft score",
            )
            return SpecializedBackendResult(
                response=response,
                raw_responses=(),
                diagnostics=(),
                model_call_count=1,
            )
        finally:
            if self.tracker:
                self.tracker.leave("global")


def _completion_diagnostic(raw: str) -> specialized.CompletionDiagnostic:
    return specialized.CompletionDiagnostic(
        finish_reason="stop",
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        raw_content_char_count=len(raw),
        non_whitespace_content_char_count=sum(
            not character.isspace() for character in raw
        ),
        whitespace_only=not raw.strip(),
    )


@dataclass
class _SequencedCaptioner:
    config: CaptionerConfig
    responses: list[str | SpecializedBackendFailure]
    calls: list[str] = field(default_factory=list)

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def caption(self, job: JEATargetAudioCaptionJob) -> CaptionerBackendResult:
        self.calls.append(job.target_clip_uid)
        response = self.responses.pop(0)
        if isinstance(response, SpecializedBackendFailure):
            raise response
        return CaptionerBackendResult(
            raw_audio_caption=response,
            raw_responses=(response,),
            diagnostics=(_completion_diagnostic(response),),
            model_call_count=1,
        )


@dataclass
class _SequencedGlobal:
    config: GlobalSemanticsConfig
    responses: list[GlobalAudioSemanticsResponse | SpecializedBackendFailure]
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def extract(
        self,
        job: JEATargetAudioCaptionJob,
        raw_audio_caption: str,
    ) -> SpecializedBackendResult[GlobalAudioSemanticsResponse]:
        self.calls.append((job.target_clip_uid, raw_audio_caption))
        response = self.responses.pop(0)
        if isinstance(response, SpecializedBackendFailure):
            raise response
        raw = response.model_dump_json()
        return SpecializedBackendResult(
            response=response,
            raw_responses=(raw,),
            diagnostics=(_completion_diagnostic(raw),),
            model_call_count=1,
        )


@dataclass
class _FakeLocal:
    config: LocalSemanticsConfig
    failures: set[str] = field(default_factory=set)
    failure_raw_by_clip: dict[str, list[str]] = field(default_factory=dict)
    style_prefix: str = ""
    calls: list[str] = field(default_factory=list)
    delay: float = 0.0
    tracker: _ActivityTracker | None = None

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self.config.provenance()

    def describe(self, job: JEATargetAudioCaptionJob):  # type: ignore[no-untyped-def]
        self.calls.append(job.target_clip_uid)
        if self.tracker:
            self.tracker.enter("local")
        try:
            time.sleep(self.delay)
            if job.target_clip_uid in self.failures:
                raw_responses = self.failure_raw_by_clip.get(job.target_clip_uid, [])
                raise SpecializedBackendFailure(
                    code="local_fake_failure",
                    reason="fake failure",
                    raw_responses=raw_responses,
                    diagnostics=[
                        specialized.CompletionDiagnostic(
                            finish_reason="stop",
                            prompt_tokens=11,
                            completion_tokens=7,
                            total_tokens=18,
                            raw_content_char_count=len(raw),
                            non_whitespace_content_char_count=sum(
                                not character.isspace() for character in raw
                            ),
                            whitespace_only=not raw.strip(),
                        )
                        for raw in raw_responses
                    ],
                    model_call_count=max(1, len(raw_responses)),
                )
            response = _local_response(
                style=f"{self.style_prefix}{job.target_clip_uid}"
            )
            return SpecializedBackendResult(
                response=response,
                raw_responses=(),
                diagnostics=(),
                model_call_count=1,
            )
        finally:
            if self.tracker:
                self.tracker.leave("local")


class _ActivityTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: dict[str, int] = {"captioner": 0, "global": 0, "local": 0}
        self.peaks = dict(self.active)
        self.cross_role_peak = 0

    def enter(self, role: str) -> None:
        with self.lock:
            self.active[role] += 1
            self.peaks[role] = max(self.peaks[role], self.active[role])
            self.cross_role_peak = max(
                self.cross_role_peak,
                sum(value > 0 for value in self.active.values()),
            )

    def leave(self, role: str) -> None:
        with self.lock:
            self.active[role] -= 1


def _failure(code: str) -> SpecializedBackendFailure:
    raw = f"{code}-raw"
    return SpecializedBackendFailure(
        code=code,
        reason=f"{code} reason",
        raw_responses=(raw,),
        diagnostics=(_completion_diagnostic(raw),),
        model_call_count=1,
    )


def _primary_and_rescue(
    tmp_path: Path,
    *,
    primary: GlobalAudioSemanticsResponse | SpecializedBackendFailure,
    rescue_caption: str | SpecializedBackendFailure,
    rescue: GlobalAudioSemanticsResponse | SpecializedBackendFailure | None,
) -> tuple[
    specialized._ProcessedRecord[specialized.GlobalSemanticsRecord],
    _SequencedCaptioner,
    _SequencedGlobal,
]:
    inventory = _inventory(tmp_path, clip_count=1)
    job = inventory.jobs[0]
    canonical_captioner = _SequencedCaptioner(
        _caption_config(inventory),
        ["canonical caption"],
    )
    captioner_record = specialized._captioner_record(job, canonical_captioner).record
    global_backend = _SequencedGlobal(
        _global_config(),
        [primary, *([] if rescue is None else [rescue])],
    )
    primary_record = specialized._global_record(
        job,
        captioner_record,
        global_backend,
    )
    rescue_captioner = _SequencedCaptioner(
        _caption_config(inventory),
        [rescue_caption],
    )
    merged = specialized._run_recaption_rescues(
        jobs=inventory.jobs,
        processed=[primary_record],
        captioner_backend=rescue_captioner,
        global_backend=global_backend,
        captioner_max_inflight=1,
        global_max_inflight=1,
    )[0]
    return merged, rescue_captioner, global_backend


@pytest.mark.parametrize(
    ("status", "description", "soundscape", "music", "expected"),
    [
        ("ready", "description", "room", "music", False),
        ("ready", None, "room", "music", True),
        ("ready", "description", "room", None, True),
        ("ready", "description", None, "music", False),
        ("failed", None, None, None, True),
        ("blocked", None, None, None, False),
    ],
)
def test_recaption_rescue_trigger_contract(
    status: str,
    description: str | None,
    soundscape: str | None,
    music: str | None,
    expected: bool,
) -> None:
    record = specialized.GlobalSemanticsRecord.model_construct(
        status=status,
        overall_audio_description=description,
        overall_soundscape=soundscape,
        non_diegetic_music=music,
    )
    assert specialized._needs_recaption_rescue(record) is expected


@pytest.mark.parametrize(
    ("primary", "rescue", "expected"),
    [
        (
            GlobalAudioSemanticsResponse(
                overall_audio_description="A",
                overall_soundscape="B",
                non_diegetic_music=None,
            ),
            GlobalAudioSemanticsResponse(
                overall_audio_description="C",
                overall_soundscape="D",
                non_diegetic_music="M",
            ),
            ("A", "B", "M"),
        ),
        (
            GlobalAudioSemanticsResponse(
                overall_audio_description=None,
                overall_soundscape="B",
                non_diegetic_music="M1",
            ),
            GlobalAudioSemanticsResponse(
                overall_audio_description="A",
                overall_soundscape="D",
                non_diegetic_music="M2",
            ),
            ("A", "B", "M1"),
        ),
        (
            GlobalAudioSemanticsResponse(
                overall_audio_description=None,
                overall_soundscape=None,
                non_diegetic_music=None,
            ),
            GlobalAudioSemanticsResponse(
                overall_audio_description="A",
                overall_soundscape="B",
                non_diegetic_music="M",
            ),
            ("A", "B", "M"),
        ),
    ],
)
def test_recaption_rescue_only_fills_missing_fields(
    tmp_path: Path,
    primary: GlobalAudioSemanticsResponse,
    rescue: GlobalAudioSemanticsResponse,
    expected: tuple[str, str, str],
) -> None:
    merged, captioner, global_backend = _primary_and_rescue(
        tmp_path,
        primary=primary,
        rescue_caption="independent recaption",
        rescue=rescue,
    )
    assert (
        merged.record.overall_audio_description,
        merged.record.overall_soundscape,
        merged.record.non_diegetic_music,
    ) == expected
    assert merged.record.recaption_rescue_attempted is True
    assert merged.record.recaption_rescue_used is True
    assert captioner.calls == ["clip-000"]
    assert len(global_backend.calls) == 2


@pytest.mark.parametrize("failure_stage", ["captioner", "global"])
def test_failed_recaption_does_not_damage_ready_primary(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    primary = GlobalAudioSemanticsResponse(
        overall_audio_description="primary description",
        overall_soundscape="primary room",
        non_diegetic_music=None,
    )
    merged, _, global_backend = _primary_and_rescue(
        tmp_path,
        primary=primary,
        rescue_caption=(
            _failure("captioner_rescue_failed")
            if failure_stage == "captioner"
            else "independent recaption"
        ),
        rescue=(
            None
            if failure_stage == "captioner"
            else _failure("global_rescue_failed")
        ),
    )
    assert merged.record.status == "ready"
    assert merged.record.overall_audio_description == "primary description"
    assert merged.record.overall_soundscape == "primary room"
    assert merged.record.non_diegetic_music is None
    assert merged.record.recaption_rescue_attempted is True
    assert merged.record.recaption_rescue_used is False
    assert merged.record.failure is None
    assert merged.record.model_call_count == (2 if failure_stage == "captioner" else 3)
    assert merged.record.raw_response_count == merged.record.model_call_count
    assert len(merged.record.completion_diagnostics) == merged.record.model_call_count
    assert len(global_backend.calls) == (1 if failure_stage == "captioner" else 2)


def test_all_null_recaption_is_unused_and_primary_remains_ready(tmp_path: Path) -> None:
    merged, _, global_backend = _primary_and_rescue(
        tmp_path,
        primary=GlobalAudioSemanticsResponse(
            overall_audio_description="primary description",
            overall_soundscape=None,
            non_diegetic_music=None,
        ),
        rescue_caption="independent recaption",
        rescue=GlobalAudioSemanticsResponse(
            overall_audio_description=None,
            overall_soundscape=None,
            non_diegetic_music=None,
        ),
    )
    assert merged.record.status == "ready"
    assert merged.record.overall_audio_description == "primary description"
    assert merged.record.overall_soundscape is None
    assert merged.record.non_diegetic_music is None
    assert merged.record.recaption_rescue_attempted is True
    assert merged.record.recaption_rescue_used is False
    assert merged.record.model_call_count == 3
    assert len(global_backend.calls) == 2


def test_failed_primary_global_can_recover_once_from_recaption(tmp_path: Path) -> None:
    rescue = GlobalAudioSemanticsResponse(
        overall_audio_description="rescued description",
        overall_soundscape="rescued room",
        non_diegetic_music="rescued music",
    )
    merged, captioner, global_backend = _primary_and_rescue(
        tmp_path,
        primary=_failure("primary_global_failed"),
        rescue_caption="independent recaption",
        rescue=rescue,
    )
    assert merged.record.status == "ready"
    assert merged.record.overall_audio_description == "rescued description"
    assert merged.record.overall_soundscape == "rescued room"
    assert merged.record.non_diegetic_music == "rescued music"
    assert merged.record.recaption_rescue_used is True
    assert merged.record.failure is None
    assert merged.record.model_call_count == 3
    assert captioner.calls == ["clip-000"]
    assert len(global_backend.calls) == 2


def test_failed_primary_and_failed_recaption_remain_failed(tmp_path: Path) -> None:
    merged, _, _ = _primary_and_rescue(
        tmp_path,
        primary=_failure("primary_global_failed"),
        rescue_caption="independent recaption",
        rescue=_failure("rescue_global_failed"),
    )
    assert merged.record.status == "failed"
    assert merged.record.failure is not None
    assert merged.record.failure.code == "primary_global_failed"
    assert merged.record.recaption_rescue_attempted is True
    assert merged.record.recaption_rescue_used is False
    assert merged.record.model_call_count == 3
    assert merged.record.raw_response_count == 3


def test_standalone_global_recaption_preserves_primary_caption_artifact_and_audit(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    canonical_captioner = _SequencedCaptioner(
        _caption_config(inventory),
        ["canonical first caption"],
    )
    run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=canonical_captioner,
    )
    captioner_before = {
        path.relative_to(root / "captioner"): path.read_bytes()
        for path in (root / "captioner").rglob("*")
        if path.is_file()
    }
    primary = GlobalAudioSemanticsResponse(
        overall_audio_description="primary description",
        overall_soundscape="primary room",
        non_diegetic_music=None,
    )
    rescue = GlobalAudioSemanticsResponse(
        overall_audio_description="rescue description",
        overall_soundscape="rescue room",
        non_diegetic_music="rescued music",
    )
    global_backend = _SequencedGlobal(_global_config(), [primary, rescue])
    rescue_captioner = _SequencedCaptioner(
        _caption_config(inventory),
        ["independent second caption"],
    )

    records, summary = run_global_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=global_backend,
        captioner_backend=rescue_captioner,
        captioner_max_inflight=1,
        max_inflight=1,
    )

    record = records[0]
    assert record.overall_audio_description == "primary description"
    assert record.overall_soundscape == "primary room"
    assert record.non_diegetic_music == "rescued music"
    assert record.model_call_count == 3
    assert record.raw_response_count == 3
    assert len(record.completion_diagnostics) == 3
    assert summary.model_call_count == 3
    assert rescue_captioner.calls == ["clip-000"]
    assert len(global_backend.calls) == 2
    raw = json.loads(
        (root / "global_semantics/raw/clip-000.json").read_text(encoding="utf-8")
    )
    assert raw["primary_global"]["raw_responses"]
    assert raw["recaption_rescue"]["attempted"] is True
    assert raw["recaption_rescue"]["used"] is True
    assert raw["recaption_rescue"]["captioner"]["raw_audio_caption"] == (
        "independent second caption"
    )
    assert raw["recaption_rescue"]["global"]["raw_responses"]
    captioner_after = {
        path.relative_to(root / "captioner"): path.read_bytes()
        for path in (root / "captioner").rglob("*")
        if path.is_file()
    }
    assert captioner_after == captioner_before


def test_pipeline_applies_same_one_shot_recaption_contract(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    primary = GlobalAudioSemanticsResponse(
        overall_audio_description="primary description",
        overall_soundscape="primary room",
        non_diegetic_music=None,
    )
    rescue = GlobalAudioSemanticsResponse(
        overall_audio_description="rescue description",
        overall_soundscape="rescue room",
        non_diegetic_music="rescued music",
    )
    captioner = _SequencedCaptioner(
        _caption_config(inventory),
        ["canonical first caption", "independent second caption"],
    )
    global_backend = _SequencedGlobal(_global_config(), [primary, rescue])

    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=_FakeLocal(_local_config(inventory)),
        captioner_max_inflight=1,
        global_vl_max_inflight=1,
    )

    global_record = specialized.GlobalSemanticsRecord.model_validate_json(
        (root / "global_semantics/records.jsonl").read_text(encoding="utf-8")
    )
    captioner_record = specialized.CaptionerRecord.model_validate_json(
        (root / "captioner/records.jsonl").read_text(encoding="utf-8")
    )
    assert global_record.overall_audio_description == "primary description"
    assert global_record.overall_soundscape == "primary room"
    assert global_record.non_diegetic_music == "rescued music"
    assert global_record.recaption_rescue_attempted is True
    assert global_record.recaption_rescue_used is True
    assert global_record.model_call_count == 3
    assert result.global_semantics_summary.model_call_count == 3
    assert captioner_record.raw_audio_caption == "canonical first caption"
    assert captioner_record.model_call_count == 1
    assert captioner.calls == ["clip-000", "clip-000"]
    assert len(global_backend.calls) == 2


def test_phases_assemble_partial_results_and_preserve_upstream(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=3)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    captioner = _FakeCaptioner(_caption_config(inventory), failures={"clip-001"})
    global_backend = _FakeGlobal(_global_config(), failures={"clip-002"})
    local = _FakeLocal(_local_config(inventory), failures={"clip-000"})

    run_captioner_phase(inventory=inventory, output_root=root, backend=captioner)
    run_global_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=global_backend,
        captioner_backend=captioner,
    )
    run_local_semantics_phase(inventory=inventory, output_root=root, backend=local)
    upstream_before = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "assembled" not in path.parts
    }
    source_audio_before = {
        job.target_clip_uid: Path(job.target_full_audio_path).read_bytes()
        for job in inventory.jobs
    }
    records, summary = run_assemble_phase(inventory=inventory, output_root=root)

    assert [record.target_clip_uid for record in records] == [
        "clip-000",
        "clip-001",
        "clip-002",
    ]
    assert records[0].status == "partial" and records[0].local_semantics_status == "failed"
    assert records[1].global_semantics_status == "blocked"
    assert records[2].global_semantics_status == "failed"
    assert records[0].speaker_delivery == []
    assert records[2].speaker_delivery[0].entity_id == "e1"
    assert summary.model_call_count == 0
    assert all(record.schema_version == ASSEMBLED_RECORD_VERSION for record in records)
    review_html = (root / "assembled/review.html").read_text()
    assert "Captioner raw audio caption" in review_html
    assert "h3-specialized-audio-qa:" in review_html
    assert "file://" not in review_html
    for job in inventory.jobs:
        media_name = hashlib.sha256(job.target_clip_uid.encode()).hexdigest() + ".flac"
        review_audio = root / "assembled/media" / media_name
        assert f"src='media/{media_name}'" in review_html
        assert review_audio.read_bytes() == Path(job.target_full_audio_path).read_bytes()
        assert _sha256(review_audio) == job.target_full_audio_sha256
    upstream_after = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "assembled" not in path.parts
    }
    assert upstream_after == upstream_before
    assert {
        job.target_clip_uid: Path(job.target_full_audio_path).read_bytes()
        for job in inventory.jobs
    } == source_audio_before


def test_failed_local_salvages_strict_speaker_delivery_and_assembles_partial(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    malformed = """{
  "overall_soundscape": null,
  "non_diegetic_music": null,
  "temporal_audio_events": [
    {"start_time": 00:02.12, "end_time": 2.4, "description": "impact"}
  ],
  "speaker_delivery": [
    {"speaker_cluster_id": "speaker_0", "delivery_style": "calm [steady] \\"delivery\\""}
    ]
}"""
    prompt_echo = "user\nRepair the previous JSON only"
    client, completions = _client([malformed, prompt_echo])
    local = OpenAILocalSemanticsBackend(
        _local_config(inventory),
        client=client,
    )

    run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeCaptioner(_caption_config(inventory)),
    )
    run_global_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeGlobal(_global_config()),
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
    )
    local_records, local_summary = run_local_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=local,
    )
    assembled, _ = run_assemble_phase(inventory=inventory, output_root=root)

    local_record = local_records[0]
    assert local_record.status == "failed"
    assert local_record.failure is not None
    assert local_record.model_call_count == 2
    assert local_record.raw_response_count == 2
    assert len(local_record.completion_diagnostics) == 2
    assert len(completions.requests) == 2
    assert local_record.temporal_audio_events == []
    assert local_record.speaker_delivery[0].delivery_style == 'calm [steady] "delivery"'
    assert local_summary.failed_count == 1
    raw_payload = json.loads(
        (root / "local_semantics/raw/clip-000.json").read_text(encoding="utf-8")
    )
    assert raw_payload["raw_responses"] == [malformed, prompt_echo]

    record = assembled[0]
    assert record.status == "partial"
    assert record.local_semantics_status == "failed"
    assert record.temporal_audio_events == []
    assert record.speaker_delivery[0].entity_id == "e1"
    assert record.speaker_delivery[0].delivery_style == 'calm [steady] "delivery"'
    review = (root / "assembled/review.html").read_text(encoding="utf-8")
    event_section = review.split("<h3>Temporal audio events</h3>", 1)[1].split(
        "<h3>Speaker delivery</h3>", 1
    )[0]
    speaker_section = review.split("<h3>Speaker delivery</h3>", 1)[1].split(
        "<details>", 1
    )[0]
    assert "[unavailable]" in event_section
    assert "[unavailable]" not in speaker_section
    assert "calm [steady] &quot;delivery&quot;" in speaker_section


@pytest.mark.parametrize(
    "raw_responses",
    [
        [
            (
                '{"temporal_audio_events":[{"start_time":00:02.12}],'
                '"speaker_delivery":['
                '{"speaker_cluster_id":"speaker_0","delivery_style":"calm"},'
                '{"speaker_cluster_id":"speaker_0","delivery_style":"calm"}]}'
            )
        ],
        [
            (
                '{"temporal_audio_events":[{"start_time":00:02.12}],'
                '"speaker_delivery":['
                '{"speaker_cluster_id":"unknown","delivery_style":"calm"}]}'
            )
        ],
        [
            (
                '{"temporal_audio_events":[{"start_time":00:02.12}],'
                '"speaker_delivery":[]}'
            )
        ],
        ["", "\n\n"],
        ["user\nRepair the previous JSON only"],
    ],
)
def test_failed_local_does_not_salvage_untrusted_speaker_arrays(
    tmp_path: Path,
    raw_responses: list[str],
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    processed = specialized._local_record(
        inventory.jobs[0],
        _FakeLocal(
            _local_config(inventory),
            failures={"clip-000"},
            failure_raw_by_clip={"clip-000": raw_responses},
        ),
    )

    assert processed.record.status == "failed"
    assert processed.record.speaker_delivery == []
    assert processed.record.temporal_audio_events == []


def test_failed_local_salvage_reorders_complete_speaker_set(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    job = _job_with_speaker_clusters(inventory, "speaker_0", "speaker_1")
    malformed = (
        '{"temporal_audio_events":[{"start_time":00:02.12}],'
        '"speaker_delivery":['
        '{"speaker_cluster_id":"speaker_1","delivery_style":"brief"},'
        '{"speaker_cluster_id":"speaker_0","delivery_style":"calm"}]}'
    )

    delivery = specialized._salvage_speaker_delivery([malformed], job)

    assert [item.speaker_cluster_id for item in delivery] == [
        "speaker_0",
        "speaker_1",
    ]


def test_pipeline_streams_roles_bounds_work_and_resumes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=5)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    tracker = _ActivityTracker()
    global_started = threading.Event()
    captioner = _FakeCaptioner(
        _caption_config(inventory),
        delay_by_clip={"clip-000": 0.01, "clip-001": 0.15},
        tracker=tracker,
    )
    global_backend = _FakeGlobal(
        _global_config(), first_started=global_started, tracker=tracker
    )
    local = _FakeLocal(_local_config(inventory), delay=0.04, tracker=tracker)

    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=local,
        captioner_max_inflight=2,
        global_vl_max_inflight=2,
        local_instruct_max_inflight=1,
    )

    assert global_started.is_set()
    assert tracker.cross_role_peak > 1
    assert tracker.peaks == {"captioner": 2, "global": 1, "local": 1}
    assert result.peak_captioner_inflight <= 2
    assert result.peak_global_vl_inflight <= 2
    assert result.peak_local_instruct_inflight <= 1
    assert result.peak_global_backlog <= 4
    records = [
        json.loads(line)
        for line in (root / "assembled/records.jsonl").read_text().splitlines()
    ]
    assert [item["target_clip_uid"] for item in records] == [
        job.target_clip_uid for job in inventory.jobs
    ]

    resumed_captioner = _FakeCaptioner(_caption_config(inventory))
    resumed_global = _FakeGlobal(_global_config())
    resumed_local = _FakeLocal(_local_config(inventory))
    resumed = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=resumed_captioner,
        global_backend=resumed_global,
        local_backend=resumed_local,
    )
    assert resumed.captioner_reused is True
    assert resumed.global_semantics_reused is True
    assert resumed.local_semantics_reused is True
    assert resumed_captioner.calls == []
    assert resumed_global.calls == []
    assert resumed_local.calls == []


def test_pipeline_failure_does_not_cancel_other_roles(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    captioner = _FakeCaptioner(_caption_config(inventory), failures={"clip-000"})
    global_backend = _FakeGlobal(_global_config())
    local = _FakeLocal(_local_config(inventory))
    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=local,
    )
    assert local.calls == ["clip-000", "clip-001"]
    assert global_backend.calls == ["clip-001"]
    assert result.global_semantics_summary.blocked_count == 1
    assert result.local_semantics_summary.ready_count == 2


def test_pipeline_reuses_captioner_and_local_while_regenerating_global(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    shutil.rmtree(root / "global_semantics")
    captioner = _FakeCaptioner(_caption_config(inventory))
    global_backend = _FakeGlobal(_global_config())
    local = _FakeLocal(_local_config(inventory))

    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=local,
    )

    assert result.captioner_reused is True
    assert result.global_semantics_reused is False
    assert result.local_semantics_reused is True
    assert captioner.calls == []
    assert global_backend.calls == ["clip-000", "clip-001"]
    assert local.calls == []


def test_stale_global_fails_independently_and_pipeline_regenerates_dependencies(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeCaptioner(
            _caption_config(inventory),
            caption_prefix="changed",
        ),
        overwrite=True,
    )

    independent_global = _FakeGlobal(_global_config())
    with pytest.raises(ValueError, match="stale captioner"):
        run_global_semantics_phase(
            inventory=inventory,
            output_root=root,
            backend=independent_global,
            captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        )
    assert independent_global.calls == []

    captioner = _FakeCaptioner(_caption_config(inventory))
    global_backend = _FakeGlobal(_global_config())
    local = _FakeLocal(_local_config(inventory))
    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=local,
    )
    assert result.captioner_reused is True
    assert result.global_semantics_reused is False
    assert result.local_semantics_reused is True
    assert captioner.calls == []
    assert global_backend.calls == ["clip-000", "clip-001"]
    assert local.calls == []
    assembled = [
        json.loads(line)
        for line in (root / "assembled/records.jsonl").read_text().splitlines()
    ]
    assert [row["overall_audio_description"] for row in assembled] == [
        "changed clip-000",
        "changed clip-001",
    ]


@pytest.mark.parametrize("changed_role", ["captioner", "global", "local"])
def test_pipeline_configuration_changes_rerun_only_affected_dependencies(
    tmp_path: Path,
    changed_role: str,
) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    captioner = _FakeCaptioner(
        _caption_config(
            inventory,
            temperature=0.7 if changed_role == "captioner" else 0.6,
        )
    )
    global_backend = _FakeGlobal(
        _global_config(model="vl-new" if changed_role == "global" else "vl")
    )
    local = _FakeLocal(
        _local_config(
            inventory,
            include_video=changed_role == "local",
        )
    )

    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=captioner,
        global_backend=global_backend,
        local_backend=local,
    )

    if changed_role == "captioner":
        assert result.captioner_reused is False
        assert result.global_semantics_reused is False
        assert result.local_semantics_reused is True
        assert captioner.calls == ["clip-000", "clip-001"]
        assert global_backend.calls == ["clip-000", "clip-001"]
        assert local.calls == []
    elif changed_role == "global":
        assert result.captioner_reused is True
        assert result.global_semantics_reused is False
        assert result.local_semantics_reused is True
        assert captioner.calls == []
        assert global_backend.calls == ["clip-000", "clip-001"]
        assert local.calls == []
    else:
        assert result.captioner_reused is True
        assert result.global_semantics_reused is True
        assert result.local_semantics_reused is False
        assert captioner.calls == []
        assert global_backend.calls == []
        assert local.calls == ["clip-000", "clip-001"]


def test_stale_assembled_fails_independently_and_pipeline_rebuilds(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    assembled_before = (root / "assembled/records.jsonl").read_bytes()
    run_local_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeLocal(_local_config(inventory), style_prefix="changed-"),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="stale stage records"):
        run_assemble_phase(inventory=inventory, output_root=root)

    result = run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    assert result.captioner_reused is True
    assert result.global_semantics_reused is True
    assert result.local_semantics_reused is True
    assembled_after = (root / "assembled/records.jsonl").read_bytes()
    assert assembled_after != assembled_before
    assert b"changed-clip-000" in assembled_after


@pytest.mark.parametrize("changed_role", ["captioner", "global", "local"])
def test_independent_assemble_rejects_each_changed_upstream_record(
    tmp_path: Path,
    changed_role: str,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    if changed_role == "captioner":
        run_captioner_phase(
            inventory=inventory,
            output_root=root,
            backend=_FakeCaptioner(
                _caption_config(inventory),
                caption_prefix="changed",
            ),
            overwrite=True,
        )
        run_global_semantics_phase(
            inventory=inventory,
            output_root=root,
            backend=_FakeGlobal(_global_config()),
            captioner_backend=_FakeCaptioner(_caption_config(inventory)),
            overwrite=True,
        )
    elif changed_role == "global":
        run_global_semantics_phase(
            inventory=inventory,
            output_root=root,
            backend=_FakeGlobal(_global_config(), description_prefix="changed-"),
            captioner_backend=_FakeCaptioner(_caption_config(inventory)),
            overwrite=True,
        )
    else:
        run_local_semantics_phase(
            inventory=inventory,
            output_root=root,
            backend=_FakeLocal(_local_config(inventory), style_prefix="changed-"),
            overwrite=True,
        )

    with pytest.raises(ValueError, match="stale stage records"):
        run_assemble_phase(inventory=inventory, output_root=root)


def test_compatible_assembled_output_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )

    def unexpected_publish(*args: object, **kwargs: object) -> None:
        raise AssertionError("compatible assembled output should be reused")

    monkeypatch.setattr(specialized, "_publish_assembled", unexpected_publish)
    records, _ = run_assemble_phase(inventory=inventory, output_root=root)
    assert len(records) == 1


def test_assembled_publication_failure_preserves_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    assembled = root / "assembled"
    before = _tree_bytes(assembled)

    def fail_hardlink(*args: object, **kwargs: object) -> None:
        raise OSError("simulated cross-device link")

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated review media failure")

    monkeypatch.setattr(specialized.os, "link", fail_hardlink)
    monkeypatch.setattr(specialized.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="simulated review media failure"):
        run_assemble_phase(
            inventory=inventory,
            output_root=root,
            overwrite=True,
        )
    assert _tree_bytes(assembled) == before
    assert not list(root.glob(".assembled.tmp-*"))


def test_assembled_review_media_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeCaptioner(_caption_config(inventory)),
    )
    run_global_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeGlobal(_global_config()),
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
    )
    run_local_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeLocal(_local_config(inventory)),
    )

    def fail_hardlink(*args: object, **kwargs: object) -> None:
        raise OSError("simulated cross-device link")

    monkeypatch.setattr(specialized.os, "link", fail_hardlink)
    run_assemble_phase(inventory=inventory, output_root=root)
    media_name = hashlib.sha256(b"clip-000").hexdigest() + ".flac"
    assert (root / "assembled/media" / media_name).read_bytes() == (
        Path(inventory.jobs[0].target_full_audio_path).read_bytes()
    )


def test_pipeline_and_independent_phases_publish_same_semantic_bytes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=2)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    run_captioner_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeCaptioner(_caption_config(inventory)),
    )
    run_global_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeGlobal(_global_config()),
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
    )
    run_local_semantics_phase(
        inventory=inventory,
        output_root=root,
        backend=_FakeLocal(_local_config(inventory)),
    )
    run_assemble_phase(inventory=inventory, output_root=root)
    phased = (root / "assembled/records.jsonl").read_bytes()
    shutil.rmtree(root)

    run_specialized_pipeline(
        inventory=inventory,
        output_root=root,
        captioner_backend=_FakeCaptioner(_caption_config(inventory)),
        global_backend=_FakeGlobal(_global_config()),
        local_backend=_FakeLocal(_local_config(inventory)),
    )
    assert (root / "assembled/records.jsonl").read_bytes() == phased


def test_overwrite_fails_closed_on_unknown_stage_ownership(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=1)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    destination = root / "captioner"
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text('{"stage":"other"}\n')
    backend = _FakeCaptioner(_caption_config(inventory))
    with pytest.raises(ValueError, match="not owned"):
        run_captioner_phase(
            inventory=inventory,
            output_root=root,
            backend=backend,
            overwrite=True,
        )
    assert backend.calls == []


def test_legacy_target_audio_caption_remains_importable() -> None:
    from r2v_data_v2.h3.jea_target_audio_caption import (
        JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION,
    )

    assert JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION == "r2v.h3.target_audio_caption.8"
