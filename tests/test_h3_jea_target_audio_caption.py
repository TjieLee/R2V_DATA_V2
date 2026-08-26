from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.jea_audio_production import JEAInPair
from r2v_data_v2.h3.jea_diarization import JEAReadableDiarizationSegment
from r2v_data_v2.h3.jea_target_audio_caption import (
    JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION,
    JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION,
    JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
    JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION,
    JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION,
    SYSTEM_PROMPT,
    JEATargetAudioCaptionBackendFailure,
    JEATargetAudioCaptionBackendResult,
    JEATargetAudioCaptionConfig,
    JEATargetAudioCaptionHumanQAExport,
    JEATargetAudioCaptionInventory,
    JEATargetAudioCaptionRecord,
    OpenAIJEATargetAudioCaptionBackend,
    build_jea_target_audio_caption_inventory,
    run_jea_target_audio_caption,
    target_audio_caption_output_root,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration, Qwen3ASRSegment
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption_contract import (
    ModelSpeakerDelivery,
    TargetAudioCaptionResponse,
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


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.responses.pop(0))
                )
            ]
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
        background_audio_prompt="faint music and room ambience",
        speaker_delivery=[
            ModelSpeakerDelivery(
                speaker_cluster_id=cluster_id,
                delivery_style="calm and conversational",
            )
            for cluster_id in cluster_ids
        ],
    )


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
                background_audio_prompt="faint music and room ambience",
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


def test_current_35_target_92_segment_inventory_and_partial_binding(
    tmp_path: Path,
) -> None:
    root = _production_fixture(tmp_path, clip_count=35, segment_count=92)

    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )

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
        JEATargetAudioCaptionInventory.model_validate_json(
            inventory.model_dump_json()
        )
        == inventory
    )


def test_inventory_allows_distinct_readable_and_canonical_audio_artifacts(
    tmp_path: Path,
) -> None:
    root = _production_fixture(
        tmp_path,
        clip_count=1,
        segment_count=1,
        distinct_audio_artifacts=True,
    )

    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )

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
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
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
    job = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    ).jobs[0]
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
    job = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    ).jobs[0]
    reordered = TargetAudioCaptionResponse(
        background_audio_prompt=None,
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

    failed_completions = _FakeCompletions([reordered, reordered])
    failed_backend = OpenAIJEATargetAudioCaptionBackend(
        _config(tmp_path, family="dots3"),
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=failed_completions)
        ),
    )
    with pytest.raises(RuntimeError, match="after one repair"):
        failed_backend.describe(job)
    assert len(failed_completions.requests) == 2


@pytest.mark.parametrize(
    ("response", "issue_code"),
    [
        (
            TargetAudioCaptionResponse(
                background_audio_prompt=None,
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
                background_audio_prompt=None,
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
    job = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    ).jobs[0]
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
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
    before = {
        name: _tree_bytes(root / name)
        for name in ("pairs", "diarization", "asr", "source")
    }

    dots_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=target_audio_caption_output_root(
            root, backend_family="dots3"
        ),
        backend=_FakeBackend(tmp_path, family="dots3"),
    )
    qwen_summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=target_audio_caption_output_root(
            root, backend_family="qwen3_omni"
        ),
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
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
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


def test_programming_failure_leaves_no_partial_publication(tmp_path: Path) -> None:
    root = _production_fixture(tmp_path)
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
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
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
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
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=root
    )
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
    assert JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION.endswith(".2")
    assert JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION == "h3_target_audio_caption_v5"
    assert set(TargetAudioCaptionResponse.model_json_schema()["properties"]) == {
        "background_audio_prompt",
        "speaker_delivery",
    }


def test_v5_prompt_prioritizes_generation_useful_audible_evidence() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "faint or partially masked background" in prompt
    for evidence in (
        "background music",
        "ambience",
        "sound effects",
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
    assert "never invent a sound merely because" in prompt


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
            "--dry-run",
        ]
    )

    assert result["model_calls"] == 0
    assert result["target_clip_count"] == 2
    assert result["output_root"] == str(root / "audio_caption/qwen3_omni")
    assert not (root / "audio_caption").exists()


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
