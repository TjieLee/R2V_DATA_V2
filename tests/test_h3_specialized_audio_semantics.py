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
    JEATargetAudioCaptionInventory,
    JEATargetAudioCaptionJob,
    _inventory_fingerprint,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.specialized_audio_semantics import (
    ASSEMBLED_RECORD_VERSION,
    CAPTIONER_POLICY_VERSION,
    GLOBAL_PROMPT_VERSION,
    GLOBAL_SYSTEM_PROMPT,
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
    valid = _local_response().model_dump_json()
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
        serialized = json.dumps(completions.requests)
        assert "Do not produce" in LOCAL_SYSTEM_PROMPT
        assert "overall_soundscape" in LOCAL_SYSTEM_PROMPT
        assert "SECRET TRANSCRIPT" not in serialized


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
    null_response = LocalAudioSemanticsResponse(
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

    client, completions = _client([" ", _local_response().model_dump_json()])
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
                non_diegetic_music=None,
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


@dataclass
class _FakeLocal:
    config: LocalSemanticsConfig
    failures: set[str] = field(default_factory=set)
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
                raise SpecializedBackendFailure(
                    code="local_fake_failure",
                    reason="fake failure",
                    model_call_count=1,
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


def test_phases_assemble_partial_results_and_preserve_upstream(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, clip_count=3)
    root = Path(inventory.source_audio_production_root) / SPECIALIZED_ROOT_NAME
    captioner = _FakeCaptioner(_caption_config(inventory), failures={"clip-001"})
    global_backend = _FakeGlobal(_global_config(), failures={"clip-002"})
    local = _FakeLocal(_local_config(inventory), failures={"clip-000"})

    run_captioner_phase(inventory=inventory, output_root=root, backend=captioner)
    run_global_semantics_phase(
        inventory=inventory, output_root=root, backend=global_backend
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
            overwrite=True,
        )
    elif changed_role == "global":
        run_global_semantics_phase(
            inventory=inventory,
            output_root=root,
            backend=_FakeGlobal(_global_config(), description_prefix="changed-"),
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
