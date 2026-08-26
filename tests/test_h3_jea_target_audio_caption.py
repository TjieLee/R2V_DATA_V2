from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import r2v_data_v2.h3.jea_target_audio_caption as jea_caption
from r2v_data_v2.h3.jea_audio_production import JEAInPair
from r2v_data_v2.h3.jea_diarization import JEAReadableDiarizationSegment
from r2v_data_v2.h3.jea_target_audio_caption import (
    FALLBACK_SYSTEM_PROMPT,
    JEA_TARGET_AUDIO_CAPTION_FALLBACK_POLICY_VERSION,
    JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION,
    JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION,
    JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION,
    JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION,
    JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
    JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION,
    JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION,
    SYSTEM_PROMPT,
    JEATargetAudioCaptionBackendFailure,
    JEATargetAudioCaptionBackendResult,
    JEATargetAudioCaptionConfig,
    JEATargetAudioCaptionHumanQAExport,
    JEATargetAudioCaptionInventory,
    JEATargetAudioCaptionJob,
    JEATargetAudioCaptionRecord,
    JEATargetAudioCaptionSummary,
    OpenAIJEATargetAudioCaptionBackend,
    _fallback_repair_prompt,
    _fallback_user_prompt,
    _is_all_semantic_null,
    build_jea_target_audio_caption_inventory,
    run_jea_target_audio_caption,
    target_audio_caption_output_root,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration, Qwen3ASRSegment
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption_contract import (
    ModelSpeakerDelivery,
    TargetAudioCaptionResponse,
    TemporalAudioEvent,
)
from tools import run_h3_target_audio_caption as cli


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_summary_prompt_version(output: Path, prompt_version: str) -> None:
    path = output / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["backend_provenance"]["prompt_version"] = prompt_version
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_pcm16_wave(
    path: Path,
    *,
    sample_rate_hz: int = 16000,
    sample_count: int = 16000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(b"\x00\x00" * sample_count)


def _production_fixture(
    tmp_path: Path,
    *,
    clip_count: int = 2,
    segment_count: int = 4,
    distinct_audio_artifacts: bool = False,
) -> Path:
    root = tmp_path / "production"
    pairs: list[JEAInPair] = []
    readable: list[JEAReadableDiarizationSegment] = []
    asr_rows: list[Qwen3ASRSegment] = []
    base_segments, clips_with_extra = divmod(segment_count, clip_count)
    for clip_index in range(clip_count):
        count = base_segments + int(clip_index < clips_with_extra)
        clip_uid = f"clip-{clip_index:03d}"
        display = f"01/series/season/episode/{clip_uid}"
        media = root / "source" / clip_uid
        media.mkdir(parents=True)
        video = media / "target.mp4"
        readable_audio = media / "audio_source" / "clip.wav"
        canonical_audio = (
            media / "full_audio" / "clip.flac"
            if distinct_audio_artifacts
            else readable_audio
        )
        binding = media / "audio_binding.json"
        video.write_bytes(f"video-{clip_uid}".encode())
        _write_pcm16_wave(readable_audio)
        if canonical_audio != readable_audio:
            _write_pcm16_wave(canonical_audio)
        binding.write_text("{}\n", encoding="utf-8")
        pairs.append(
            JEAInPair(
                pair_id=f"in_pair/{clip_uid}",
                target_clip_uid=clip_uid,
                target_clip_display_path=display,
                media_collection_relpath="01/series",
                media_collection_name="series",
                episode_name="episode",
                clip_name=clip_uid,
                shard_id="shard-1",
                target_video_path=str(video),
                target_full_audio_path=str(canonical_audio),
                target_audio_binding_path=str(binding),
                subjects=[],
            )
        )
        for segment_index in range(count):
            segment_id = f"segment-{segment_index:04d}"
            speaker_cluster_id = (
                "speaker-1" if clip_index == 0 and segment_index == 1 else "speaker-0"
            )
            entity_id = None
            if clip_index == 0 and speaker_cluster_id == "speaker-0":
                entity_id = "e1" if segment_index == 0 else None
            start_sample = segment_index * 1600
            end_sample = start_sample + 800
            values = {
                "clip_uid": clip_uid,
                "clip_display_path": display,
                "media_collection_relpath": "01/series",
                "media_collection_name": "series",
                "episode_name": "episode",
                "clip_name": clip_uid,
                "shard_id": "shard-1",
                "segment_id": segment_id,
                "speaker_cluster_id": speaker_cluster_id,
                "entity_id": entity_id,
                "entity_occurrence_id": (
                    None if entity_id is None else f"{clip_uid}/{entity_id}"
                ),
                "source_audio_path": str(readable_audio),
                "source_start_sample": start_sample,
                "source_end_sample": end_sample,
                "source_sample_rate_hz": 16000,
                "start_time": start_sample / 16000,
                "end_time": end_sample / 16000,
            }
            readable.append(JEAReadableDiarizationSegment(**values))
            asr_rows.append(
                Qwen3ASRSegment(
                    **values,
                    status="transcribed",
                    text=f"SECRET TRANSCRIPT {clip_uid} {segment_id}",
                    language="Chinese",
                    configuration=Qwen3ASRConfiguration(
                        local_model_path="/models/Qwen3-ASR-1.7B"
                    ),
                )
            )
    _write_jsonl(root / "pairs/in_pairs.jsonl", pairs)
    _write_jsonl(root / "diarization/readable_segments.jsonl", readable)
    _write_jsonl(root / "asr/segments.jsonl", asr_rows)
    return root


@dataclass(frozen=True)
class _FakeCompletion:
    content: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class _FakeCompletions:
    def __init__(self, responses: list[str | _FakeCompletion]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        value = self.responses.pop(0)
        response = value if isinstance(value, str) else value.content
        finish_reason = None if isinstance(value, str) else value.finish_reason
        usage = None if isinstance(value, str) else value.usage
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response),
                    finish_reason=finish_reason,
                )
            ],
            usage=(None if usage is None else SimpleNamespace(**usage)),
        )


def _config(
    tmp_path: Path,
    *,
    family: str,
) -> JEATargetAudioCaptionConfig:
    return JEATargetAudioCaptionConfig(
        backend_family=family,  # type: ignore[arg-type]
        base_url="https://example.invalid/v1",
        api_key="secret-not-published",
        served_model_name="served-model",
        checkpoint_id="checkpoint",
        media_resolver=MediaURLResolver(mode="file", media_root=tmp_path),
        max_tokens=321,
    )


def _response(job_index: int = 0) -> TargetAudioCaptionResponse:
    cluster_ids = ["speaker-0", "speaker-1"] if job_index == 0 else ["speaker-0"]
    return TargetAudioCaptionResponse(
        overall_soundscape="faint music and room ambience",
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=cluster_id,
                delivery_style="calm and conversational",
            )
            for cluster_id in cluster_ids
        ],
    )


def _response_for_job(
    job: JEATargetAudioCaptionJob,
    *,
    background: str | None,
    delivery_styles: list[str | None],
    music: str | None = None,
    events: list[TemporalAudioEvent] | None = None,
) -> TargetAudioCaptionResponse:
    clusters = job.speaker_clusters
    assert len(clusters) == len(delivery_styles)
    return TargetAudioCaptionResponse(
        overall_soundscape=background,
        non_diegetic_music=music,
        temporal_audio_events=events or [],
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=cluster.speaker_cluster_id,
                delivery_style=delivery_style,
            )
            for cluster, delivery_style in zip(clusters, delivery_styles, strict=True)
        ],
    )


def _read_records(output: Path) -> list[JEATargetAudioCaptionRecord]:
    return [
        JEATargetAudioCaptionRecord.model_validate_json(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _run_semantic_sequence(
    tmp_path: Path,
    responses: list[str | _FakeCompletion],
    *,
    family: str = "qwen3_omni",
    max_concurrency: int = 1,
) -> tuple[
    JEATargetAudioCaptionSummary,
    JEATargetAudioCaptionRecord,
    dict[str, object],
    _FakeCompletions,
    JEATargetAudioCaptionJob,
]:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    completions = _FakeCompletions(responses)
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family=family),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    output = target_audio_caption_output_root(
        root,
        backend_family=family,  # type: ignore[arg-type]
    )
    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
        max_concurrency=max_concurrency,
    )
    record = _read_records(output)[0]
    raw = json.loads(
        (output / "raw" / f"{record.target_clip_uid}.json").read_text(encoding="utf-8")
    )
    return summary, record, raw, completions, inventory.jobs[0]


class _FakeBackend:
    def __init__(self, tmp_path: Path, *, family: str) -> None:
        self._config = _config(tmp_path, family=family)
        self.calls: list[str] = []

    @property
    def provenance(self):
        return self._config.provenance()

    def describe(self, job):
        self.calls.append(job.target_clip_uid)
        return JEATargetAudioCaptionBackendResult(
            response=TargetAudioCaptionResponse(
                overall_soundscape="faint music and room ambience",
                speaker_delivery=[
                    ModelSpeakerDelivery(
                        speaker_cluster_id=cluster.speaker_cluster_id,
                        delivery_style="calm",
                    )
                    for cluster in job.speaker_clusters
                ],
            ),
            raw_responses=("{}",),
        )


class _ConcurrentBackend(_FakeBackend):
    def __init__(
        self,
        tmp_path: Path,
        *,
        family: str,
        delays: dict[str, float] | None = None,
        backend_failure_clip: str | None = None,
        programming_failure_clip: str | None = None,
    ) -> None:
        super().__init__(tmp_path, family=family)
        self.delays = delays or {}
        self.backend_failure_clip = backend_failure_clip
        self.programming_failure_clip = programming_failure_clip
        self.completion_order: list[str] = []
        self.active_count = 0
        self.peak_active_count = 0
        self._lock = threading.Lock()

    def describe(self, job):
        with self._lock:
            self.calls.append(job.target_clip_uid)
            self.active_count += 1
            self.peak_active_count = max(self.peak_active_count, self.active_count)
        try:
            time.sleep(self.delays.get(job.target_clip_uid, 0.02))
            if job.target_clip_uid == self.backend_failure_clip:
                raise JEATargetAudioCaptionBackendFailure(
                    code="structured_output_failed",
                    reason="invalid after repair",
                    raw_responses=("bad", "still bad"),
                    attempt_count=2,
                )
            if job.target_clip_uid == self.programming_failure_clip:
                raise RuntimeError("programming failure")
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background="room ambience",
                    delivery_styles=["calm"] * len(job.speaker_clusters),
                ),
                raw_responses=("{}",),
            )
        finally:
            with self._lock:
                self.active_count -= 1
                self.completion_order.append(job.target_clip_uid)


def test_current_35_target_92_segment_inventory_and_partial_binding(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=35, segment_count=92)

    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    assert inventory.schema_version == JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION
    assert inventory.target_clip_count == 35
    assert inventory.readable_segment_count == 92
    assert inventory.qwen3_asr_segment_count == 92
    assert len(inventory.jobs) == 35
    first = inventory.jobs[0]
    assert [item.speaker_cluster_id for item in first.speaker_clusters] == [
        "speaker-0",
        "speaker-1",
    ]
    assert first.speaker_clusters[0].entity_id == "e1"
    assert len(first.speaker_clusters[0].active_time_ranges) == 2
    assert first.speaker_clusters[1].entity_id is None
    assert "SECRET TRANSCRIPT" not in inventory.model_dump_json()
    assert (
        JEATargetAudioCaptionInventory.model_validate_json(inventory.model_dump_json())
        == inventory
    )
    assert first.target_duration_seconds == pytest.approx(1.0)


def test_target_duration_participates_in_inventory_fingerprint(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=1)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    values = inventory.model_dump(mode="json")
    values["jobs"][0]["target_duration_seconds"] += 0.01  # type: ignore[index]
    values["inventory_fingerprint"] = jea_caption._inventory_fingerprint(values)

    changed = JEATargetAudioCaptionInventory.model_validate(values)

    assert changed.inventory_fingerprint != inventory.inventory_fingerprint


def test_inventory_allows_distinct_readable_and_canonical_audio_artifacts(
    tmp_path: Path,
) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )

    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    job = inventory.jobs[0]
    assert job.target_full_audio_path.endswith("/full_audio/clip.flac")
    readable = json.loads(
        (root / "diarization/readable_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert readable["source_audio_path"].endswith("/audio_source/clip.wav")
    assert readable["source_audio_path"] != job.target_full_audio_path


def test_inventory_rejects_incompatible_full_audio_timelines(tmp_path: Path) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )
    pair = json.loads(
        (root / "pairs/in_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    _write_pcm16_wave(Path(pair["target_full_audio_path"]), sample_count=32000)

    with pytest.raises(ValueError, match="timelines differ"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_inventory_accepts_lr_asd_audio_quantization_delta(tmp_path: Path) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )
    pair = json.loads(
        (root / "pairs/in_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    _write_pcm16_wave(Path(pair["target_full_audio_path"]), sample_count=16833)

    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    assert inventory.target_clip_count == 1


def test_inventory_rejects_missing_readable_source_audio(tmp_path: Path) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )
    readable = json.loads(
        (root / "diarization/readable_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    Path(readable["source_audio_path"]).unlink()

    with pytest.raises(FileNotFoundError, match="readable source audio is missing"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_inventory_rejects_missing_pair_canonical_audio(tmp_path: Path) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )
    pair = json.loads(
        (root / "pairs/in_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    Path(pair["target_full_audio_path"]).unlink()

    with pytest.raises(FileNotFoundError, match="target full audio is missing"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_inventory_rejects_inconsistent_readable_source_audio_within_clip(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    alternate = root / "source/clip-000/audio_source/alternate.wav"
    _write_pcm16_wave(alternate)
    for relative_path in (
        "diarization/readable_segments.jsonl",
        "asr/segments.jsonl",
    ):
        path = root / relative_path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["source_audio_path"] = str(alternate)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="source audio paths differ within clip"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_inventory_rejects_segment_outside_readable_source_audio(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=1)
    for relative_path in (
        "diarization/readable_segments.jsonl",
        "asr/segments.jsonl",
    ):
        path = root / relative_path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_end_sample"] = 16001
        rows[0]["end_time"] = 16001 / 16000
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="exceeds source audio samples"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_qwen_asr_and_readable_diarization_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path)
    path = root / "asr/segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_end_sample"] += 1
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence differs"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


def test_speaker_cluster_conflicting_entity_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path)
    for relative_path in (
        "diarization/readable_segments.jsonl",
        "asr/segments.jsonl",
    ):
        path = root / relative_path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["speaker_cluster_id"] = "speaker-0"
        rows[1]["entity_id"] = "e2"
        rows[1]["entity_occurrence_id"] = "clip-000/e2"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="conflicting entity IDs"):
        build_jea_target_audio_caption_inventory(audio_production_root=root)


@pytest.mark.parametrize(
    ("family", "media_types", "modality"),
    [
        (
            "dots3",
            ["text", "video_url"],
            "native_target_video_with_embedded_audio",
        ),
        (
            "qwen3_omni",
            ["text", "video_url", "audio_url"],
            "target_video_plus_canonical_full_audio",
        ),
    ],
)
def test_backends_share_schema_but_use_distinct_media_without_sensitive_evidence(
    tmp_path: Path,
    family: str,
    media_types: list[str],
    modality: str,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    job = inventory.jobs[0]
    valid = _response().model_dump_json()
    completions = _FakeCompletions([valid])
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family=family),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = backend.describe(job)

    assert isinstance(result.response, TargetAudioCaptionResponse)
    request = completions.requests[0]
    content = request["messages"][1]["content"]  # type: ignore[index]
    assert [item["type"] for item in content] == media_types
    user_text = content[0]["text"]
    model_input = json.loads(
        user_text.split("\nInput:\n", maxsplit=1)[1].split(
            "\nJSON schema:\n", maxsplit=1
        )[0]
    )
    assert set(model_input) == {"speaker_clusters", "target_duration_seconds"}
    assert model_input["target_duration_seconds"] == job.target_duration_seconds
    assert all(
        set(cluster) == {"speaker_cluster_id", "active_time_ranges"}
        for cluster in model_input["speaker_clusters"]
    )
    model_input_text = json.dumps(model_input, ensure_ascii=False)
    for forbidden in (
        "SECRET TRANSCRIPT",
        "entity_id",
        "gender",
        "reference_image",
        "primary_voice",
        "donor_media",
    ):
        assert forbidden not in model_input_text
    encoded = json.dumps(request, ensure_ascii=False)
    assert "SECRET TRANSCRIPT" not in encoded
    assert '"entity_id":' not in encoded
    assert backend.provenance.input_modality == modality
    assert "api_key" not in backend.provenance.model_dump(mode="json")
    if family == "qwen3_omni":
        assert request["modalities"] == ["text"]
        assert job.target_video_path in encoded
        assert job.target_full_audio_path in encoded
    else:
        assert "modalities" not in request
        assert job.target_full_audio_path not in encoded


def test_qwen_omni_uses_pair_canonical_audio_not_readable_source(
    tmp_path: Path,
) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    readable = json.loads(
        (root / "diarization/readable_segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    completions = _FakeCompletions([_response(job_index=1).model_dump_json()])
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="qwen3_omni"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    backend.describe(job)

    encoded = json.dumps(completions.requests[0], ensure_ascii=False)
    assert job.target_video_path in encoded
    assert job.target_full_audio_path in encoded
    assert readable["source_audio_path"] not in encoded
    assert "SECRET TRANSCRIPT" not in encoded
    assert '"entity_id":' not in encoded


def test_cluster_order_mismatch_repairs_once_and_second_failure_fails_closed(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    reordered = TargetAudioCaptionResponse(
        overall_soundscape=None,
        speaker_delivery=list(reversed(_response().speaker_delivery)),
    ).model_dump_json()
    valid = _response().model_dump_json()
    completions = _FakeCompletions([reordered, valid])
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="dots3"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert len(backend.describe(job).raw_responses) == 2
    assert len(completions.requests) == 2
    repair_text = completions.requests[1]["messages"][1]["content"][0]["text"]
    assert "Preserve the original four-pass audible-only analysis" in repair_text
    assert "Do not infer sound from visual content" in repair_text

    failed_completions = _FakeCompletions([reordered, reordered])
    failed_backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="dots3"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=failed_completions)),
    )
    with pytest.raises(RuntimeError, match="after one repair"):
        failed_backend.describe(job)
    assert len(failed_completions.requests) == 2


@pytest.mark.parametrize(
    ("response", "issue_code"),
    [
        (
            TargetAudioCaptionResponse(
                overall_soundscape=None,
                speaker_delivery=[
                    ModelSpeakerDelivery(
                        speaker_cluster_id="speaker-unknown",
                        delivery_style=None,
                    )
                ],
            ),
            "unknown_speaker_cluster",
        ),
        (
            TargetAudioCaptionResponse(
                overall_soundscape=None,
                speaker_delivery=[
                    ModelSpeakerDelivery(
                        speaker_cluster_id="speaker-0",
                        delivery_style=None,
                    )
                ],
            ),
            "missing_speaker_cluster",
        ),
    ],
)
def test_unknown_or_missing_cluster_fails_after_one_repair(
    tmp_path: Path,
    response: TargetAudioCaptionResponse,
    issue_code: str,
) -> None:
    root = _production_fixture(tmp_path)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    raw = response.model_dump_json()
    completions = _FakeCompletions([raw, raw])
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="dots3"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    with pytest.raises(JEATargetAudioCaptionBackendFailure) as failure:
        backend.describe(job)

    assert failure.value.attempt_count == 2
    assert [item.code for item in failure.value.issues] == [issue_code]


def test_separate_atomic_backend_outputs_keep_upstream_unchanged(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    before = {
        name: _tree_bytes(root / name)
        for name in ("pairs", "diarization", "asr", "source")
    }

    dots_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=target_audio_caption_output_root(root, backend_family="dots3"),
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    qwen_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=target_audio_caption_output_root(root, backend_family="qwen3_omni"),
        backend=_FakeBackend(tmp_path, family="qwen3_omni"),
    )

    assert dots_summary.inventory_fingerprint == qwen_summary.inventory_fingerprint
    assert dots_summary.backend_provenance.backend_family == "dots3"
    assert qwen_summary.backend_provenance.backend_family == "qwen3_omni"
    assert (root / "audio_caption/dots3/review.html").is_file()
    assert (root / "audio_caption/qwen3_omni/review.html").is_file()
    assert before == {
        name: _tree_bytes(root / name)
        for name in ("pairs", "diarization", "asr", "source")
    }
    record = JEATargetAudioCaptionRecord.model_validate_json(
        (root / "audio_caption/dots3/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert record.schema_version == JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION
    assert record.speaker_delivery[0].entity_id == "e1"
    review = (root / "audio_caption/dots3/review.html").read_text(encoding="utf-8")
    assert "hallucinated_music" in review
    assert "dialogue_leakage" in review
    assert JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION in review
    assert "backendProvenance" in review


def test_generated_review_javascript_is_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for JavaScript syntax validation")
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(root, backend_family="dots3")
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    review = (output / "review.html").read_text(encoding="utf-8")
    match = re.search(r"<script>(.*?)</script>", review, flags=re.DOTALL)
    assert match is not None
    script = tmp_path / "review.js"
    script.write_text(match.group(1), encoding="utf-8")

    result = subprocess.run(
        [node, "--check", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_review_uses_new_semantic_sections_and_configuration_qa_namespace(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=1)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    backend = _FakeBackend(tmp_path, family="qwen3_omni")
    output = target_audio_caption_output_root(root, backend_family="qwen3_omni")
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    review = (output / "review.html").read_text(encoding="utf-8")

    for heading in (
        "Overall soundscape",
        "Non-diegetic music",
        "Temporal audio events",
        "Speaker delivery",
    ):
        assert heading in review
    for flag in jea_caption.QA_FLAGS:
        assert flag in review
    assert "h3-target-audio-caption-v4" not in review
    assert "h3-target-audio-caption-v5" not in review
    assert JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION in review
    assert JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION in review
    assert backend.provenance.backend_family in review
    assert backend.provenance.configuration_fingerprint in review
    assert inventory.inventory_fingerprint in review


def test_programming_failure_leaves_no_partial_publication(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(root, backend_family="dots3")

    class BrokenBackend(_FakeBackend):
        def describe(self, job):
            raise RuntimeError("programming failure")

    with pytest.raises(RuntimeError, match="programming failure"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=BrokenBackend(tmp_path, family="dots3"),
        )

    assert not output.exists()
    assert not list(output.parent.glob(".dots3.tmp-*"))


def test_structured_failure_isolated_to_one_clip(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(root, backend_family="dots3")

    class PartiallyFailingBackend(_FakeBackend):
        def describe(self, job):
            if job.target_clip_uid == "clip-000":
                raise JEATargetAudioCaptionBackendFailure(
                    code="structured_output_failed",
                    reason="invalid after repair",
                    raw_responses=("bad", "still bad"),
                    attempt_count=2,
                )
            return super().describe(job)

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=PartiallyFailingBackend(tmp_path, family="dots3"),
    )

    assert summary.ready_count == 1
    assert summary.failed_count == 1
    assert summary.initial_call_count == 2
    assert summary.repair_call_count == 1
    assert len((output / "records.jsonl").read_text().splitlines()) == 2


def test_backend_cannot_overwrite_other_backend_custom_root(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/custom"
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family="qwen3_omni"),
    )
    dots_backend = _FakeBackend(tmp_path, family="dots3")

    with pytest.raises(ValueError, match="cannot overwrite"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=dots_backend,
            overwrite=True,
        )

    assert dots_backend.calls == []


@pytest.mark.parametrize("family", ["dots3", "qwen3_omni"])
def test_v5_same_backend_output_can_be_atomically_replaced_by_v6(
    tmp_path: Path,
    family: str,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(
        root,
        backend_family=family,  # type: ignore[arg-type]
    )
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family=family),
    )
    _rewrite_summary_prompt_version(output, "h3_target_audio_caption_v5")
    replacement = _FakeBackend(tmp_path, family=family)

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=replacement,
        overwrite=True,
    )

    assert replacement.calls == [job.target_clip_uid for job in inventory.jobs]
    assert summary.backend_provenance.prompt_version == "h3_target_audio_semantics_v1"
    published = JEATargetAudioCaptionSummary.model_validate_json(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert published.backend_provenance.prompt_version == (
        "h3_target_audio_semantics_v1"
    )


@pytest.mark.parametrize(
    ("existing_family", "replacement_family"),
    [("dots3", "qwen3_omni"), ("qwen3_omni", "dots3")],
)
def test_v5_output_cannot_be_overwritten_by_other_backend(
    tmp_path: Path,
    existing_family: str,
    replacement_family: str,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/custom"
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family=existing_family),
    )
    _rewrite_summary_prompt_version(output, "h3_target_audio_caption_v5")
    before = _tree_bytes(output)
    replacement = _FakeBackend(tmp_path, family=replacement_family)

    with pytest.raises(ValueError, match="cannot overwrite"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=replacement,
            overwrite=True,
        )

    assert replacement.calls == []
    assert _tree_bytes(output) == before


@pytest.mark.parametrize(
    "summary_payload",
    [
        "{malformed",
        [],
        {},
        {"backend_provenance": {}},
        {"backend_provenance": {"backend_family": "unknown"}},
    ],
    ids=("malformed", "not-object", "missing-provenance", "missing-family", "unknown"),
)
def test_unknown_existing_output_ownership_fails_before_inference(
    tmp_path: Path,
    summary_payload: object,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/custom"
    output.mkdir(parents=True)
    summary_text = (
        summary_payload
        if isinstance(summary_payload, str)
        else json.dumps(summary_payload)
    )
    (output / "summary.json").write_text(summary_text, encoding="utf-8")
    backend = _FakeBackend(tmp_path, family="dots3")

    with pytest.raises(ValueError, match="cannot establish.*ownership"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=backend,
            overwrite=True,
        )

    assert backend.calls == []
    assert (output / "summary.json").read_text(encoding="utf-8") == summary_text


def test_missing_existing_summary_fails_before_inference(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/custom"
    output.mkdir(parents=True)
    backend = _FakeBackend(tmp_path, family="dots3")

    with pytest.raises(ValueError, match="cannot establish.*ownership"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=backend,
            overwrite=True,
        )

    assert backend.calls == []
    assert output.is_dir()


def test_existing_output_without_overwrite_still_fails_before_inference(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(root, backend_family="dots3")
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    backend = _FakeBackend(tmp_path, family="dots3")

    with pytest.raises(FileExistsError, match="already exists"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=backend,
        )

    assert backend.calls == []


def test_failed_v6_regeneration_preserves_existing_v5_output(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = target_audio_caption_output_root(root, backend_family="dots3")
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    _rewrite_summary_prompt_version(output, "h3_target_audio_caption_v5")
    before = _tree_bytes(output)

    class BrokenBackend(_FakeBackend):
        def describe(self, job):
            raise RuntimeError("programming failure")

    with pytest.raises(RuntimeError, match="programming failure"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=BrokenBackend(tmp_path, family="dots3"),
            overwrite=True,
        )

    assert _tree_bytes(output) == before


@pytest.mark.parametrize(
    ("background", "delivery_styles"),
    [
        ("music and room tone", ["calm", "measured"]),
        (None, ["calm", None]),
        ("faint traffic", [None, None]),
        (None, [None, "questioning"]),
    ],
)
def test_qwen_partial_semantics_accept_v6_without_fallback(
    tmp_path: Path,
    background: str | None,
    delivery_styles: list[str | None],
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    response = _response_for_job(
        job,
        background=background,
        delivery_styles=delivery_styles,
    )

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [response.model_dump_json()],
    )

    assert len(completions.requests) == 1
    assert record.semantic_source == "primary"
    assert record.semantic_fallback_attempted is False
    assert record.semantic_fallback_trigger_reason is None
    assert summary.semantic_fallback_trigger_count == 0
    assert summary.semantic_fallback_initial_call_count == 0


@pytest.mark.parametrize(
    ("background", "music", "events", "delivery"),
    [
        ("Low indoor room tone.", None, [], None),
        (None, "Soft piano accompaniment.", [], None),
        (
            None,
            None,
            [
                TemporalAudioEvent(
                    start_time=0.10,
                    end_time=0.20,
                    description="A brief door close.",
                )
            ],
            None,
        ),
        (None, None, [], "quiet and measured"),
    ],
)
def test_each_new_audio_semantic_layer_is_independently_valid(
    tmp_path: Path,
    background: str | None,
    music: str | None,
    events: list[TemporalAudioEvent],
    delivery: str | None,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    response = _response_for_job(
        job,
        background=background,
        music=music,
        events=events,
        delivery_styles=[delivery, None],
    )

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [response.model_dump_json()],
    )

    assert len(completions.requests) == 1
    assert record.status == "ready"
    assert record.overall_soundscape == background
    assert record.non_diegetic_music == music
    assert record.temporal_audio_events == events
    assert summary.semantic_fallback_trigger_count == 0


def test_temporal_audio_event_range_is_strict() -> None:
    with pytest.raises(ValidationError):
        TemporalAudioEvent(start_time=-0.1, end_time=0.2, description="impact")
    with pytest.raises(ValidationError):
        TemporalAudioEvent(start_time=0.2, end_time=0.2, description="impact")


def test_temporal_audio_events_must_be_chronological_but_may_overlap() -> None:
    first = TemporalAudioEvent(
        start_time=0.1,
        end_time=0.4,
        description="Footsteps approach.",
    )
    overlapping = TemporalAudioEvent(
        start_time=0.2,
        end_time=0.5,
        description="A door closes.",
    )
    response = TargetAudioCaptionResponse(
        temporal_audio_events=[first, overlapping],
        speaker_delivery=[],
    )
    assert response.temporal_audio_events == [first, overlapping]
    with pytest.raises(ValidationError, match="chronological"):
        TargetAudioCaptionResponse(
            temporal_audio_events=[overlapping, first],
            speaker_delivery=[],
        )


def test_temporal_audio_event_beyond_duration_repairs_once(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    invalid = _response_for_job(
        job,
        background=None,
        events=[
            TemporalAudioEvent(
                start_time=0.8,
                end_time=1.2,
                description="A brief impact.",
            )
        ],
        delivery_styles=[None, None],
    )
    repaired = _response_for_job(
        job,
        background=None,
        events=[
            TemporalAudioEvent(
                start_time=0.8,
                end_time=1.0,
                description="A brief impact.",
            )
        ],
        delivery_styles=[None, None],
    )

    _, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [invalid.model_dump_json(), repaired.model_dump_json()],
        family="dots3",
    )

    assert len(completions.requests) == 2
    assert record.repair_count == 1
    assert record.temporal_audio_events == repaired.temporal_audio_events
    repair_text = completions.requests[1]["messages"][1]["content"][0]["text"]
    assert "audio_event_out_of_range" in repair_text


def test_qwen_all_null_uses_one_fallback_and_complete_v5_result(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    )
    recovered = _response_for_job(
        job,
        background="faint instrumental music",
        delivery_styles=[None, "soft and hesitant"],
    )

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [all_null.model_dump_json(), recovered.model_dump_json()],
    )

    assert len(completions.requests) == 2
    assert record.status == "ready"
    assert record.overall_soundscape == "faint instrumental music"
    assert [item.delivery_style for item in record.speaker_delivery] == [
        None,
        "soft and hesitant",
    ]
    assert record.semantic_source == "fallback"
    assert record.semantic_fallback_attempted is True
    assert record.semantic_fallback_trigger_reason == "all_null"
    assert (
        record.semantic_fallback_prompt_version
        == JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
    )
    assert summary.semantic_fallback_trigger_count == 1
    assert summary.semantic_fallback_all_null_trigger_count == 1
    assert summary.semantic_fallback_empty_response_trigger_count == 0
    assert summary.semantic_fallback_initial_call_count == 1
    assert summary.semantic_fallback_repair_call_count == 0
    assert summary.semantic_fallback_recovered_count == 1
    assert summary.semantic_fallback_still_all_null_count == 0
    assert summary.semantic_fallback_failed_count == 0
    assert summary.raw_response_count == 2
    assert raw["primary"]["raw_responses"] == [all_null.model_dump_json()]
    assert raw["semantic_fallback"]["raw_responses"] == [recovered.model_dump_json()]
    assert raw["primary"]["validated_response"] == all_null.model_dump(mode="json")
    assert raw["semantic_fallback"]["validated_response"] == recovered.model_dump(
        mode="json"
    )
    assert raw["raw_responses"] == [
        all_null.model_dump_json(),
        recovered.model_dump_json(),
    ]


def test_qwen_v5_all_null_confirms_ready_primary_without_third_attempt(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    ).model_dump_json()

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [all_null, all_null],
    )

    assert len(completions.requests) == 2
    assert record.status == "ready"
    assert record.overall_soundscape is None
    assert all(item.delivery_style is None for item in record.speaker_delivery)
    assert record.semantic_source == "primary_all_null_confirmed"
    assert record.semantic_fallback_trigger_reason == "all_null"
    assert summary.semantic_fallback_still_all_null_count == 1
    assert summary.semantic_fallback_recovered_count == 0
    assert summary.semantic_fallback_failed_count == 0


def test_qwen_v5_request_failure_preserves_valid_v6_all_null(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    ).model_dump_json()

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [all_null],
    )

    assert len(completions.requests) == 2
    assert record.status == "ready"
    assert record.semantic_source == "primary_all_null_fallback_failed"
    assert record.semantic_fallback_trigger_reason == "all_null"
    assert record.failure is None
    assert summary.ready_count == 1
    assert summary.failed_count == 0
    assert summary.semantic_fallback_failed_count == 1
    assert raw["semantic_fallback"]["failure"]["code"].endswith("vllm_request_failed")


def test_qwen_v5_malformed_then_repaired_counts_separately(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    )
    recovered = _response_for_job(
        job,
        background="quiet room tone",
        delivery_styles=[None, None],
    )

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [all_null.model_dump_json(), "not json", recovered.model_dump_json()],
    )

    assert len(completions.requests) == 3
    assert record.semantic_source == "fallback"
    assert record.repair_count == 0
    assert summary.initial_call_count == 1
    assert summary.repair_call_count == 0
    assert summary.semantic_fallback_initial_call_count == 1
    assert summary.semantic_fallback_repair_call_count == 1
    assert summary.semantic_fallback_recovered_count == 1
    assert summary.raw_response_count == 3
    assert raw["semantic_fallback"]["raw_responses"] == [
        "not json",
        recovered.model_dump_json(),
    ]


@pytest.mark.parametrize("fallback_expected", [False, True])
def test_primary_structured_repair_controls_semantic_fallback(
    tmp_path: Path,
    fallback_expected: bool,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    repaired = _response_for_job(
        job,
        background=None if fallback_expected else "room ambience",
        delivery_styles=[None, None],
    )
    responses = ["not json", repaired.model_dump_json()]
    if fallback_expected:
        responses.append(
            _response_for_job(
                job,
                background="faint music",
                delivery_styles=[None, None],
            ).model_dump_json()
        )

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        responses,
    )

    assert len(completions.requests) == 2 + int(fallback_expected)
    assert record.repair_count == 1
    assert summary.initial_call_count == 1
    assert summary.repair_call_count == 1
    assert summary.semantic_fallback_trigger_count == int(fallback_expected)
    assert record.semantic_source == ("fallback" if fallback_expected else "primary")


def test_primary_request_failure_does_not_trigger_semantic_fallback(
    tmp_path: Path,
) -> None:
    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path,
        [],
    )

    assert len(completions.requests) == 1
    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.code == "qwen3_omni_vllm_request_failed"
    assert record.semantic_source is None
    assert record.semantic_fallback_attempted is False
    assert record.semantic_fallback_trigger_reason is None
    assert summary.semantic_fallback_trigger_count == 0
    assert raw["semantic_fallback"]["attempted"] is False


def test_qwen_initial_whitespace_uses_v5_and_persists_completion_diagnostics(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    recovered = _response_for_job(
        job,
        background="faint orchestral music",
        delivery_styles=["measured", None],
    )
    whitespace = "\n \t" * 20

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [
            _FakeCompletion(
                whitespace,
                finish_reason="stop",
                usage={
                    "prompt_tokens": 4780,
                    "completion_tokens": 73,
                    "total_tokens": 4853,
                },
            ),
            _FakeCompletion(recovered.model_dump_json(), finish_reason="stop"),
        ],
    )

    assert len(completions.requests) == 2
    assert record.status == "ready"
    assert record.semantic_source == "fallback"
    assert record.semantic_fallback_trigger_reason == "empty_response"
    assert record.overall_soundscape == "faint orchestral music"
    assert summary.semantic_fallback_trigger_count == 1
    assert summary.semantic_fallback_all_null_trigger_count == 0
    assert summary.semantic_fallback_empty_response_trigger_count == 1
    assert summary.semantic_fallback_recovered_count == 1
    assert summary.raw_response_count == 2
    diagnostic = raw["primary"]["completion_diagnostics"][0]
    assert diagnostic == {
        "completion_tokens": 73,
        "finish_reason": "stop",
        "non_whitespace_content_char_count": 0,
        "prompt_tokens": 4780,
        "raw_content_char_count": len(whitespace),
        "total_tokens": 4853,
        "whitespace_only": True,
    }
    assert raw["semantic_fallback"]["trigger_reason"] == "empty_response"
    fallback_diagnostic = raw["semantic_fallback"]["completion_diagnostics"][0]
    assert fallback_diagnostic["finish_reason"] == "stop"
    assert fallback_diagnostic["prompt_tokens"] is None
    assert fallback_diagnostic["completion_tokens"] is None
    assert fallback_diagnostic["total_tokens"] is None


def test_qwen_initial_whitespace_then_v5_all_null_is_ready_without_third_call(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    )

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [" \n\n", all_null.model_dump_json()],
    )

    assert len(completions.requests) == 2
    assert record.status == "ready"
    assert record.semantic_source == "fallback"
    assert record.semantic_fallback_trigger_reason == "empty_response"
    assert record.overall_soundscape is None
    assert all(item.delivery_style is None for item in record.speaker_delivery)
    assert summary.semantic_fallback_still_all_null_count == 1
    assert summary.semantic_fallback_recovered_count == 0
    assert summary.semantic_fallback_failed_count == 0


def test_qwen_initial_whitespace_then_failed_v5_keeps_primary_failure(
    tmp_path: Path,
) -> None:
    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path,
        [" \n", "not json", "\n\n"],
    )

    assert len(completions.requests) == 3
    assert record.status == "failed"
    assert record.semantic_source is None
    assert record.semantic_fallback_attempted is True
    assert record.semantic_fallback_trigger_reason == "empty_response"
    assert record.failure is not None
    assert record.failure.code == "qwen3_omni_vllm_empty_response"
    assert summary.initial_call_count == 1
    assert summary.repair_call_count == 0
    assert summary.semantic_fallback_initial_call_count == 1
    assert summary.semantic_fallback_repair_call_count == 1
    assert summary.semantic_fallback_failed_count == 1
    assert summary.raw_response_count == 3
    assert raw["semantic_fallback"]["failure"]["code"] == (
        "qwen3_omni_vllm_empty_response"
    )


def test_qwen_repair_whitespace_preserves_initial_cluster_issue_and_recovers(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    incomplete = TargetAudioCaptionResponse(
        overall_soundscape="dramatic orchestral music",
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=job.speaker_clusters[0].speaker_cluster_id,
                delivery_style="forceful",
            )
        ],
    )
    recovered = _response_for_job(
        job,
        background="dramatic orchestral music",
        delivery_styles=["forceful", "quiet"],
    )

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [
            _FakeCompletion(incomplete.model_dump_json(), finish_reason="stop"),
            _FakeCompletion("\n" * 50, finish_reason="length"),
            _FakeCompletion(recovered.model_dump_json(), finish_reason="stop"),
        ],
    )

    assert len(completions.requests) == 3
    assert record.status == "ready"
    assert record.repair_count == 1
    assert record.semantic_source == "fallback"
    assert record.semantic_fallback_trigger_reason == "empty_response"
    assert summary.repair_call_count == 1
    assert raw["primary"]["failure"]["code"] == ("qwen3_omni_vllm_empty_response")
    assert [item["code"] for item in raw["primary"]["failure"]["issues"]] == [
        "missing_speaker_cluster"
    ]
    assert [
        item["finish_reason"] for item in raw["primary"]["completion_diagnostics"]
    ] == [
        "stop",
        "length",
    ]
    assert raw["primary"]["completion_diagnostics"][1]["whitespace_only"] is True


def test_qwen_repair_whitespace_and_failed_v5_keeps_initial_cluster_issue(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    incomplete = TargetAudioCaptionResponse(
        overall_soundscape="dramatic orchestral music",
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=job.speaker_clusters[0].speaker_cluster_id,
                delivery_style="forceful",
            )
        ],
    )

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [incomplete.model_dump_json(), "\n", "bad", "\t"],
    )

    assert len(completions.requests) == 4
    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.code == "qwen3_omni_vllm_empty_response"
    assert [item["code"] for item in record.failure.issues] == [
        "missing_speaker_cluster"
    ]
    assert raw["primary"]["failure"]["issues"] == record.failure.issues
    assert summary.repair_call_count == 1
    assert summary.semantic_fallback_repair_call_count == 1
    assert summary.semantic_fallback_failed_count == 1


def test_qwen_non_whitespace_length_completion_does_not_trigger_fallback(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    response = _response_for_job(
        job,
        background="room ambience",
        delivery_styles=[None, None],
    )

    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [_FakeCompletion(response.model_dump_json(), finish_reason="length")],
    )

    assert len(completions.requests) == 1
    assert record.status == "ready"
    assert record.semantic_source == "primary"
    assert record.semantic_fallback_attempted is False
    assert summary.semantic_fallback_trigger_count == 0
    assert raw["primary"]["completion_diagnostics"][0]["finish_reason"] == "length"
    assert raw["primary"]["completion_diagnostics"][0]["whitespace_only"] is False


def test_qwen_media_resolution_failure_does_not_trigger_semantic_fallback(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    completions = _FakeCompletions([])
    config = _config(unrelated_root, family="qwen3_omni")
    backend = OpenAIJEATargetAudioCaptionBackend(
        config,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=root / "audio_caption/qwen-media-failure",
        backend=backend,
    )
    record = _read_records(root / "audio_caption/qwen-media-failure")[0]

    assert completions.requests == []
    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.code == "qwen3_omni_vllm_request_failed"
    assert record.semantic_fallback_attempted is False
    assert summary.initial_call_count == 0
    assert summary.semantic_fallback_trigger_count == 0


def test_qwen_http_failure_does_not_trigger_semantic_fallback(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    class HTTPFailureCompletions:
        def __init__(self) -> None:
            self.call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            raise RuntimeError("HTTP 400: local media access rejected")

    completions = HTTPFailureCompletions()
    backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="qwen3_omni"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    output = root / "audio_caption/qwen-http-failure"

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )
    record = _read_records(output)[0]

    assert completions.call_count == 1
    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.code == "qwen3_omni_vllm_request_failed"
    assert record.semantic_fallback_attempted is False
    assert summary.initial_call_count == 1
    assert summary.semantic_fallback_trigger_count == 0


def test_dots3_whitespace_failure_does_not_use_qwen_semantic_fallback(
    tmp_path: Path,
) -> None:
    summary, record, raw, completions, _ = _run_semantic_sequence(
        tmp_path,
        ["\n\n\t"],
        family="dots3",
    )

    assert len(completions.requests) == 1
    assert record.status == "failed"
    assert record.failure is not None
    assert record.failure.code == "dots3_vllm_empty_response"
    assert record.semantic_fallback_attempted is False
    assert summary.semantic_fallback_trigger_count == 0
    assert raw["primary"]["completion_diagnostics"][0]["whitespace_only"] is True


def test_dots3_all_null_does_not_use_qwen_semantic_fallback(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    )

    summary, record, _, completions, _ = _run_semantic_sequence(
        tmp_path / "run",
        [all_null.model_dump_json()],
        family="dots3",
    )

    assert len(completions.requests) == 1
    assert record.semantic_source == "primary"
    assert summary.semantic_fallback_trigger_count == 0


def test_all_null_helper_handles_empty_cluster_list() -> None:
    assert _is_all_semantic_null(
        TargetAudioCaptionResponse(
            overall_soundscape=None,
            speaker_delivery=[],
        )
    )
    assert not _is_all_semantic_null(
        TargetAudioCaptionResponse(
            overall_soundscape="room tone",
            speaker_delivery=[],
        )
    )


def test_qwen_fallback_uses_new_schema_prompt_and_same_media(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    all_null = _response_for_job(
        job,
        background=None,
        delivery_styles=[None, None],
    )
    recovered = _response_for_job(
        job,
        background="music",
        delivery_styles=[None, None],
    )

    _, _, _, completions, actual_job = _run_semantic_sequence(
        tmp_path / "run",
        [all_null.model_dump_json(), recovered.model_dump_json()],
    )

    fallback_request = completions.requests[1]
    assert fallback_request["messages"][0]["content"] == FALLBACK_SYSTEM_PROMPT
    assert "same four-field schema" in FALLBACK_SYSTEM_PROMPT
    fallback_content = fallback_request["messages"][1]["content"]
    assert fallback_content[0]["text"] == _fallback_user_prompt(actual_job)
    assert "target_duration_seconds" in fallback_content[0]["text"]
    assert [item["type"] for item in fallback_content] == [
        "text",
        "video_url",
        "audio_url",
    ]
    assert fallback_request["temperature"] == 0.0
    assert fallback_request["max_tokens"] == 321
    assert fallback_request["modalities"] == ["text"]
    assert fallback_request["stream"] is False
    assert fallback_request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    encoded = json.dumps(fallback_request, ensure_ascii=False)
    assert actual_job.target_video_path in encoded
    assert actual_job.target_full_audio_path in encoded
    for forbidden in (
        "SECRET TRANSCRIPT",
        '"entity_id":',
        "reference_image",
        "primary_voice",
        "donor_media",
    ):
        assert forbidden not in encoded


def test_fallback_repair_prompt_preserves_new_schema(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    issues = [
        SimpleNamespace(
            to_dict=lambda: {
                "code": "invalid_json",
                "field": None,
                "message": "invalid JSON",
            }
        )
    ]
    actual = _fallback_repair_prompt(
        job=job,
        invalid_response="bad",
        issues=issues,  # type: ignore[arg-type]
    )
    assert "fallback four-pass audible-only policy" in actual
    assert "same new semantic schema" in actual
    assert _fallback_user_prompt(job) in actual
    assert '"code":"invalid_json"' in actual


def test_concurrent_completion_preserves_inventory_order_and_record_semantics(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    serial_output = root / "audio_caption/serial"
    concurrent_output = root / "audio_caption/concurrent"
    serial_backend = _ConcurrentBackend(tmp_path, family="dots3")
    concurrent_backend = _ConcurrentBackend(
        tmp_path,
        family="dots3",
        delays={
            "clip-000": 0.08,
            "clip-001": 0.06,
            "clip-002": 0.04,
            "clip-003": 0.01,
        },
    )

    serial_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=serial_output,
        backend=serial_backend,
        max_concurrency=1,
    )
    concurrent_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=concurrent_output,
        backend=concurrent_backend,
        max_concurrency=4,
    )

    expected_order = [job.target_clip_uid for job in inventory.jobs]
    concurrent_records = _read_records(concurrent_output)
    assert [record.target_clip_uid for record in concurrent_records] == expected_order
    assert concurrent_backend.completion_order != expected_order
    assert concurrent_backend.peak_active_count > 1
    assert (serial_output / "records.jsonl").read_bytes() == (
        concurrent_output / "records.jsonl"
    ).read_bytes()
    assert (serial_output / "review.html").read_bytes() == (
        concurrent_output / "review.html"
    ).read_bytes()
    assert [record.request_fingerprint for record in _read_records(serial_output)] == [
        record.request_fingerprint for record in concurrent_records
    ]
    assert (
        serial_summary.inventory_fingerprint == concurrent_summary.inventory_fingerprint
    )
    assert serial_summary.max_concurrency == 1
    assert concurrent_summary.max_concurrency == 4
    assert serial_summary.model_dump(exclude={"max_concurrency"}) == (
        concurrent_summary.model_dump(exclude={"max_concurrency"})
    )


def test_concurrent_backend_failure_isolated_without_reordering(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/concurrent"
    backend = _ConcurrentBackend(
        tmp_path,
        family="dots3",
        backend_failure_clip="clip-001",
    )

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
        max_concurrency=4,
    )

    records = _read_records(output)
    assert [record.target_clip_uid for record in records] == [
        job.target_clip_uid for job in inventory.jobs
    ]
    assert [record.status for record in records] == [
        "ready",
        "failed",
        "ready",
        "ready",
    ]
    assert summary.ready_count == 3
    assert summary.failed_count == 1
    assert summary.initial_call_count == 4
    assert summary.repair_call_count == 1
    assert summary.failure_reason_counts == {"structured_output_failed": 1}


def test_concurrent_qwen_whitespace_fallback_preserves_inventory_order(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    class ConcurrentWhitespaceBackend(_ConcurrentBackend):
        def __init__(self) -> None:
            super().__init__(
                tmp_path,
                family="qwen3_omni",
                delays={"clip-000": 0.08, "clip-003": 0.01},
            )
            self.fallback_calls: list[str] = []

        def describe(self, job):
            if job.target_clip_uid == "clip-001":
                raise JEATargetAudioCaptionBackendFailure(
                    code="qwen3_omni_vllm_empty_response",
                    reason="audio caption vLLM returned no non-whitespace text",
                    raw_responses=("\n\n",),
                    attempt_count=1,
                )
            return super().describe(job)

        def describe_semantic_fallback(self, job):
            self.fallback_calls.append(job.target_clip_uid)
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background="faint music",
                    delivery_styles=[None] * len(job.speaker_clusters),
                ),
                raw_responses=("fallback",),
            )

    output = root / "audio_caption/qwen-concurrent-whitespace"
    backend = ConcurrentWhitespaceBackend()
    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
        max_concurrency=4,
    )

    records = _read_records(output)
    assert [record.target_clip_uid for record in records] == [
        job.target_clip_uid for job in inventory.jobs
    ]
    assert backend.fallback_calls == ["clip-001"]
    assert records[1].semantic_source == "fallback"
    assert records[1].semantic_fallback_trigger_reason == "empty_response"
    assert summary.semantic_fallback_trigger_count == 1
    assert summary.semantic_fallback_all_null_trigger_count == 0
    assert summary.semantic_fallback_empty_response_trigger_count == 1


def test_concurrent_qwen_fallbacks_are_clip_local_and_counters_reconcile(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/qwen-concurrent"

    class ConcurrentFallbackBackend(_FakeBackend):
        def __init__(self) -> None:
            super().__init__(tmp_path, family="qwen3_omni")
            self.fallback_calls: list[str] = []
            self._lock = threading.Lock()

        def describe(self, job):
            with self._lock:
                self.calls.append(job.target_clip_uid)
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background=(
                        "room ambience" if job.target_clip_uid == "clip-000" else None
                    ),
                    delivery_styles=[None] * len(job.speaker_clusters),
                ),
                raw_responses=(f"primary-{job.target_clip_uid}",),
            )

        def describe_semantic_fallback(self, job):
            with self._lock:
                self.fallback_calls.append(job.target_clip_uid)
            if job.target_clip_uid == "clip-003":
                raise JEATargetAudioCaptionBackendFailure(
                    code="structured_output_failed",
                    reason="invalid after repair",
                    raw_responses=("fallback-bad",),
                    attempt_count=2,
                )
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background=(
                        "faint music" if job.target_clip_uid == "clip-001" else None
                    ),
                    delivery_styles=[None] * len(job.speaker_clusters),
                ),
                raw_responses=(f"fallback-{job.target_clip_uid}",),
            )

    backend = ConcurrentFallbackBackend()
    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=backend,
        max_concurrency=4,
    )

    records = _read_records(output)
    assert [record.semantic_source for record in records] == [
        "primary",
        "fallback",
        "primary_all_null_confirmed",
        "primary_all_null_fallback_failed",
    ]
    assert sorted(backend.fallback_calls) == ["clip-001", "clip-002", "clip-003"]
    assert summary.initial_call_count == 4
    assert summary.repair_call_count == 0
    assert summary.semantic_fallback_trigger_count == 3
    assert summary.semantic_fallback_all_null_trigger_count == 3
    assert summary.semantic_fallback_empty_response_trigger_count == 0
    assert summary.semantic_fallback_initial_call_count == 3
    assert summary.semantic_fallback_repair_call_count == 1
    assert summary.semantic_fallback_recovered_count == 1
    assert summary.semantic_fallback_still_all_null_count == 1
    assert summary.semantic_fallback_failed_count == 1
    assert summary.raw_response_count == 7
    assert summary.ready_count == 4
    assert summary.failed_count == 0


def test_dots3_all_null_has_no_fallback_under_concurrency(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)

    class ConcurrentAllNullDotsBackend(_FakeBackend):
        def describe(self, job):
            self.calls.append(job.target_clip_uid)
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background=None,
                    delivery_styles=[None] * len(job.speaker_clusters),
                ),
                raw_responses=("all-null",),
            )

        def describe_semantic_fallback(self, job):
            raise AssertionError("Dots3 must not use semantic fallback")

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=root / "audio_caption/dots-concurrent",
        backend=ConcurrentAllNullDotsBackend(tmp_path, family="dots3"),
        max_concurrency=4,
    )

    assert summary.ready_count == 4
    assert summary.semantic_fallback_trigger_count == 0


def test_unexpected_concurrent_error_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=4, segment_count=4)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    output = root / "audio_caption/dots3"
    run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output,
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    before = _tree_bytes(output)

    with pytest.raises(RuntimeError, match="programming failure"):
        run_jea_target_audio_caption(
            inventory=inventory,
            output_root=output,
            backend=_ConcurrentBackend(
                tmp_path,
                family="dots3",
                programming_failure_clip="clip-001",
            ),
            overwrite=True,
            max_concurrency=4,
        )

    assert _tree_bytes(output) == before
    assert not list(output.parent.glob(".dots3.tmp-*"))


def test_openai_backend_reuses_one_client_per_worker_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _production_fixture(tmp_path, clip_count=8, segment_count=8)
    inventory = build_jea_target_audio_caption_inventory(audio_production_root=root)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    client_threads: list[int] = []
    request_count = 0

    class ThreadCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            nonlocal request_count
            with lock:
                request_count += 1
            response = TargetAudioCaptionResponse(
                overall_soundscape="room ambience",
                speaker_delivery=[
                    ModelSpeakerDelivery(
                        speaker_cluster_id="speaker-0",
                        delivery_style="calm",
                    )
                ],
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=response.model_dump_json())
                    )
                ]
            )

    def client_factory(**kwargs: object) -> SimpleNamespace:
        with lock:
            client_threads.append(threading.get_ident())
        barrier.wait(timeout=2)
        return SimpleNamespace(chat=SimpleNamespace(completions=ThreadCompletions()))

    monkeypatch.setattr(jea_caption, "OpenAI", client_factory)
    backend = OpenAIJEATargetAudioCaptionBackend(_config(tmp_path, family="dots3"))

    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=root / "audio_caption/thread-local",
        backend=backend,
        max_concurrency=4,
    )

    assert summary.ready_count == 8
    assert request_count == 8
    assert len(client_threads) == 4
    assert len(set(client_threads)) == 4


def test_qwen_fallback_policy_changes_request_fingerprint_without_affecting_dots3(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=1)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]

    def old_policy_fingerprint(family: str) -> str:
        provenance = _config(tmp_path, family=family).provenance()
        return jea_caption._sha256_text(
            jea_caption._compact_json(
                {
                    "prompt_version": JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
                    "model_input": jea_caption._model_input(job),
                    "target_video_sha256": job.target_video_sha256,
                    "target_full_audio_sha256": job.target_full_audio_sha256,
                    "backend_configuration_fingerprint": (
                        provenance.configuration_fingerprint
                    ),
                    "semantic_fallback_prompt_version": (
                        JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
                        if family == "qwen3_omni"
                        else None
                    ),
                }
            )
        )

    qwen_provenance = _config(tmp_path, family="qwen3_omni").provenance()
    dots_provenance = _config(tmp_path, family="dots3").provenance()

    assert jea_caption.target_audio_caption_request_fingerprint(
        job, qwen_provenance
    ) != old_policy_fingerprint("qwen3_omni")
    assert jea_caption.target_audio_caption_request_fingerprint(
        job, dots_provenance
    ) == old_policy_fingerprint("dots3")


def test_old_contracts_do_not_validate_as_jea_multibackend_contract() -> None:
    with pytest.raises(ValidationError):
        JEATargetAudioCaptionRecord.model_validate(
            {
                "schema_version": "r2v.h3.target_audio_caption.2",
                "target_clip_uid": "clip",
            }
        )
    with pytest.raises(ValidationError):
        JEATargetAudioCaptionInventory.model_validate(
            {
                "schema_version": "r2v.h3.target_audio_caption_inventory.1",
                "target_clip_count": 1,
            }
        )
    assert (
        JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION
        == "r2v.h3.target_audio_caption_summary.6"
    )
    assert JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION == "r2v.h3.target_audio_caption.6"
    assert (
        JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION
        == "r2v.h3.target_audio_caption_inventory.3"
    )
    assert (
        JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION
        == "r2v.h3.target_audio_caption_human_qa.3"
    )
    assert JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION == "h3_target_audio_semantics_v1"
    assert (
        JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION
        == "h3_target_audio_semantics_v1"
    )
    assert (
        JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
        == "h3_target_audio_semantics_v1_recheck"
    )
    assert (
        JEA_TARGET_AUDIO_CAPTION_FALLBACK_POLICY_VERSION
        == "qwen_h3_audio_semantics_all_null_or_empty_recheck_v1"
    )
    assert set(TargetAudioCaptionResponse.model_json_schema()["properties"]) == {
        "overall_soundscape",
        "non_diegetic_music",
        "temporal_audio_events",
        "speaker_delivery",
    }


def test_current_records_and_summary_are_strict_new_versions(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=1)
    job = build_jea_target_audio_caption_inventory(audio_production_root=root).jobs[0]
    response = _response_for_job(
        job,
        background="room tone",
        delivery_styles=["measured"],
    )
    summary, record, _, _, _ = _run_semantic_sequence(
        tmp_path / "run",
        [response.model_dump_json()],
    )

    assert record.schema_version == "r2v.h3.target_audio_caption.6"
    assert summary.schema_version == "r2v.h3.target_audio_caption_summary.6"
    old_record = record.model_dump(mode="json")
    old_record["schema_version"] = "r2v.h3.target_audio_caption.5"
    with pytest.raises(ValidationError):
        JEATargetAudioCaptionRecord.model_validate(old_record)
    old_summary = summary.model_dump(mode="json")
    old_summary["schema_version"] = "r2v.h3.target_audio_caption_summary.5"
    with pytest.raises(ValidationError):
        JEATargetAudioCaptionSummary.model_validate(old_summary)


def test_h3_semantic_prompt_separates_four_audio_layers() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "AUDIBLE EVIDENCE IS THE SOURCE OF TRUTH" in prompt
    assert "Never infer a sound merely because" in prompt
    assert "four independent passes" in prompt
    assert "PASS 1 - OVERALL SOUNDSCAPE" in prompt
    assert "PASS 2 - NON-DIEGETIC MUSIC" in prompt
    assert "PASS 3 - TEMPORAL AUDIO EVENTS" in prompt
    assert "PASS 4 - SPEAKER DELIVERY" in prompt
    assert "overlapping events are allowed" in prompt
    for evidence in (
        "music",
        "ambience",
        "sound effects",
        "laughter",
        "human non-speech vocalizations",
        "traffic",
        "crowds",
        "footsteps",
        "doors",
        "machinery",
        "nature",
        "emotion",
        "pace",
        "energy",
        "loudness",
        "pitch tendency",
        "rhythm",
        "hesitation",
        "pauses",
        "whispering",
        "shouting",
        "questioning",
        "commanding",
    ):
        assert evidence in prompt
    assert "Never transcribe, quote, paraphrase" in prompt
    assert "identity, gender, age, nationality" in prompt
    assert "Visual evidence may only disambiguate" in prompt


def test_qa_export_carries_backend_provenance(tmp_path: Path) -> None:
    provenance = _config(tmp_path, family="qwen3_omni").provenance()
    qa = JEATargetAudioCaptionHumanQAExport(
        inventory_fingerprint="a" * 64,
        backend_provenance=provenance,
        label_count=1,
        total_clip_count=2,
        counts={"CORRECT": 1, "WRONG": 0, "UNCERTAIN": 0, "UNLABELED": 1},
        labels=[{"target_clip_uid": "clip-1", "label": "CORRECT"}],
    )
    assert qa.backend_provenance.backend_family == "qwen3_omni"
    assert qa.backend_provenance.served_model_name == "served-model"
    assert qa.backend_provenance.prompt_version == "h3_target_audio_semantics_v1"
    assert qa.backend_provenance == qa.backend_provenance.model_validate_json(
        qa.backend_provenance.model_dump_json()
    )


def test_cli_dry_run_uses_current_root_and_constructs_no_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _production_fixture(tmp_path)

    def forbidden_backend(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not construct a backend")

    monkeypatch.setattr(cli, "OpenAIJEATargetAudioCaptionBackend", forbidden_backend)
    monkeypatch.delenv("DOTS3_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN3_OMNI_BASE_URL", raising=False)

    result = cli.main(
        [
            "--audio-production-root",
            str(root),
            "--backend",
            "qwen3-omni",
            "--max-concurrency",
            "4",
            "--dry-run",
        ]
    )

    assert result["model_calls"] == 0
    assert result["max_concurrency"] == 4
    assert result["target_clip_count"] == 2
    assert result["output_root"] == str(root / "audio_caption/qwen3_omni")
    assert not (root / "audio_caption").exists()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_max_concurrency(value: str) -> None:
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            [
                "--audio-production-root",
                "/tmp/audio-production",
                "--backend",
                "dots3",
                "--max-concurrency",
                value,
            ]
        )


def test_cli_defaults_to_single_clip_concurrency() -> None:
    arguments = cli._parser().parse_args(
        [
            "--audio-production-root",
            "/tmp/audio-production",
            "--backend",
            "dots3",
        ]
    )

    assert arguments.max_concurrency == 1


def test_cli_model_calls_include_semantic_fallback_and_fallback_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _production_fixture(tmp_path, clip_count=1, segment_count=2)

    class CliFallbackBackend:
        def __init__(self, config: JEATargetAudioCaptionConfig) -> None:
            self._provenance = config.provenance()

        @property
        def provenance(self):
            return self._provenance

        def describe(self, job):
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background=None,
                    delivery_styles=[None, None],
                ),
                raw_responses=("primary",),
            )

        def describe_semantic_fallback(self, job):
            return JEATargetAudioCaptionBackendResult(
                response=_response_for_job(
                    job,
                    background="faint music",
                    delivery_styles=[None, None],
                ),
                raw_responses=("fallback-invalid", "fallback-valid"),
            )

    monkeypatch.setattr(
        cli,
        "OpenAIJEATargetAudioCaptionBackend",
        CliFallbackBackend,
    )

    result = cli.main(
        [
            "--audio-production-root",
            str(root),
            "--backend",
            "qwen3-omni",
            "--base-url",
            "https://example.invalid/v1",
            "--media-root",
            str(tmp_path),
        ]
    )

    assert result["model_calls"] == 3
    assert result["summary"]["initial_call_count"] == 1
    assert result["summary"]["repair_call_count"] == 0
    assert result["summary"]["semantic_fallback_initial_call_count"] == 1
    assert result["summary"]["semantic_fallback_repair_call_count"] == 1


@pytest.mark.parametrize(
    ("backend_name", "prefix", "family"),
    [
        ("dots3", "DOTS3", "dots3"),
        ("qwen3-omni", "QWEN3_OMNI", "qwen3_omni"),
    ],
)
def test_backend_environment_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    prefix: str,
    family: str,
) -> None:
    other_prefix = "QWEN3_OMNI" if prefix == "DOTS3" else "DOTS3"
    for suffix in (
        "BASE_URL",
        "API_KEY",
        "MODEL",
        "CHECKPOINT_ID",
        "MEDIA_MODE",
        "MEDIA_ROOT",
        "MEDIA_BASE_URL",
    ):
        monkeypatch.delenv(f"{other_prefix}_{suffix}", raising=False)
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv(f"{prefix}_MEDIA_ROOT", str(tmp_path))
    arguments = cli._parser().parse_args(
        [
            "--audio-production-root",
            str(tmp_path),
            "--backend",
            backend_name,
        ]
    )

    config = cli._backend_config(arguments)

    assert config.backend_family == family
    assert config.base_url == "https://example.invalid/v1"
