from __future__ import annotations

import hashlib
import math
import wave
from array import array
from pathlib import Path

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.background_audio_scout import (
    BackgroundAudioPilotSelection,
    BackgroundAudioScoutInventory,
    BackgroundAudioScoutJob,
    BackgroundAudioSelectionReview,
    SampleSpan,
    analyze_background_audio_job,
    background_audio_scout_output_root,
    build_background_audio_scout_inventory,
    complement_sample_spans,
    run_background_audio_scout,
    union_sample_spans,
)
from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    DiarizationTargetClip,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioTrackMetadata,
    H3AudioBindingIR,
    H3TaskSpecification,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, values: list[int], *, sample_rate: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array("h", values)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _write_source(root: Path) -> DiarizationInventory:
    pairs = root / "production" / "pairs" / "in_pairs.jsonl"
    pairs.parent.mkdir(parents=True)
    pairs.write_text("", encoding="utf-8")
    audio_root = root / "production" / "audio"
    targets = []
    for index in range(75):
        clip_uid = f"clip-{index:03d}"
        video = root / "media" / f"{clip_uid}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video:{clip_uid}".encode())
        audio = audio_root / "clips" / clip_uid / "audio.wav"
        _write_wav(audio, [index + 1] * 100)
        sidecar_path = audio_root / "clips" / clip_uid / "audio_binding.json"
        sidecar = AudioBindingSidecar(
            clip_uid=clip_uid,
            source_run_root="/read-only/visual",
            source_video_path=str(video),
            status="ready",
            evidence=AudioBindingEvidence(
                clip_uid=clip_uid,
                audio=AudioTrackMetadata(
                    status="ready",
                    source_video_path=str(video),
                    full_audio_path=str(audio),
                    duration_seconds=1.0,
                    sample_rate_hz=100,
                    channels=1,
                ),
            ),
            h3_ir=H3AudioBindingIR(
                clip_uid=clip_uid,
                task=H3TaskSpecification(components=["reference_generation"]),
                picture_assets=[],
                subjects=[],
                audio_assets=[],
                bindings=[],
            ),
        )
        sidecar_path.write_text(
            sidecar.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video),
                source_audio_path=str(audio),
                source_audio_sha256=_sha256(audio),
                source_sample_rate_hz=100,
                source_channels=1,
                source_frame_count=100,
                target_audio_binding_path=str(sidecar_path),
                visual_references=[],
            )
        )
    pairs_hash = _sha256(pairs)
    fingerprint = _diarization_inventory_fingerprint(
        source_pairs_sha256=pairs_hash,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
    )
    inventory = DiarizationInventory(
        mode="production",
        source_pairs_path=str(pairs),
        source_pairs_sha256=pairs_hash,
        inventory_fingerprint=fingerprint,
        source_target_count=75,
        selected_target_count=75,
        selection_mode="complete_in_pair_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )
    diar_root = root / "production" / "diarization"
    diar_root.mkdir(parents=True)
    (diar_root / "inventory.json").write_text(
        inventory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (diar_root / "raw_segments.jsonl").write_text("", encoding="utf-8")
    return inventory


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_build_enumerates_all_75_production_targets_without_selection(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)

    inventory = build_background_audio_scout_inventory(audio_run_root=tmp_path)

    assert inventory.target_clip_count == 75
    assert [item.target_clip_uid for item in inventory.jobs] == [
        item.target_clip_uid for item in source.targets
    ]
    assert inventory.selection_applied is False
    assert inventory.parent_quota_applied is False
    assert inventory.model_calls == 0
    assert inventory.automatic_background_selection_applied is False


def test_overlapping_diarization_intervals_use_temporal_union() -> None:
    assert union_sample_spans(
        [(20, 50), (40, 70), (10, 15), (15, 18)],
        frame_count=100,
    ) == [(10, 18), (20, 70)]
    assert complement_sample_spans(
        [(20, 50), (40, 70)],
        frame_count=100,
    ) == [(0, 20), (70, 100)]


def test_pcm_diagnostics_are_deterministic_and_use_non_speech_complement(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    values = [1000] * 20 + [3000] * 50 + [2000] * 30
    _write_wav(audio, values)
    sidecar = tmp_path / "audio_binding.json"
    sidecar.write_text("{}", encoding="utf-8")
    job = BackgroundAudioScoutJob(
        target_clip_uid="clip-a",
        target_full_audio_path=str(audio),
        target_full_audio_sha256=_sha256(audio),
        target_audio_binding_path=str(sidecar),
        target_audio_binding_sha256=_sha256(sidecar),
        source_sample_rate_hz=100,
        source_frame_count=100,
        diarized_speech_spans=[SampleSpan(start_sample=20, end_sample=70)],
    )

    first = analyze_background_audio_job(job)
    second = analyze_background_audio_job(job)

    expected_non_speech_rms = math.sqrt((20 * 1000**2 + 30 * 2000**2) / 50) / 32768
    assert first == second
    assert first.clip_duration_seconds == 1.0
    assert first.diarized_speaker_union_seconds == 0.5
    assert first.non_speech_seconds == 0.5
    assert first.non_speech_ratio == 0.5
    assert first.non_speech_rms_dbfs == pytest.approx(
        20 * math.log10(expected_non_speech_rms)
    )
    assert first.non_speech_peak_dbfs == pytest.approx(20 * math.log10(2000 / 32768))


def test_scout_is_read_only_model_free_fixed_root_and_atomic(tmp_path: Path) -> None:
    _write_source(tmp_path)
    inventory = build_background_audio_scout_inventory(audio_run_root=tmp_path)
    source_before = _tree_hashes(tmp_path / "production")
    clean_pilot = tmp_path / "target_audio_caption_pilot20" / "keep.json"
    clean_pilot.parent.mkdir()
    clean_pilot.write_text('{"keep":true}\n', encoding="utf-8")
    clean_hash = _sha256(clean_pilot)
    output = background_audio_scout_output_root(tmp_path)

    summary = run_background_audio_scout(
        inventory=inventory,
        output_root=output,
    )

    assert summary.target_clip_count == 75
    assert summary.diagnostic_record_count == 75
    assert summary.model_calls == summary.dots3_calls == 0
    assert summary.automatic_background_selection_applied is False
    assert source_before == _tree_hashes(tmp_path / "production")
    assert _sha256(clean_pilot) == clean_hash
    assert (
        sum(
            1
            for line in (output / "diagnostics.jsonl").read_text().splitlines()
            if line
        )
        == 75
    )
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "Highest non-speech RMS" in report
    assert "Highest non-speech ratio" in report
    assert "Longest non-speech duration" in report
    assert "High non-speech energy is not an automatic" in report
    assert "Export background-rich selection JSON" in report
    assert "Select at least one background-rich clip" in report
    assert "const keyPrefix='h3-background-audio-scout-'" in report
    assert "background_audio_pilot_selection.json" in report


def test_final_selection_accepts_dynamic_nonempty_manual_background_rich_clips() -> (
    None
):
    reviews = [
        BackgroundAudioSelectionReview(
            target_clip_uid=f"clip-{index:03d}",
            label="background-rich" if index < 20 else "clean",
            flags=["music"] if index == 0 else [],
        )
        for index in range(21)
    ]
    selection = BackgroundAudioPilotSelection(
        source_scout_inventory_fingerprint="1" * 64,
        source_diarization_inventory_fingerprint="2" * 64,
        source_audio_evidence_fingerprint="3" * 64,
        selection_count=20,
        selected_clip_ids=[item.target_clip_uid for item in reviews[:20]],
        reviews=reviews,
    )
    assert selection.selection_count == 20
    assert selection.selection_method == "manual_human_review_v1"

    one = BackgroundAudioPilotSelection(
        source_scout_inventory_fingerprint="1" * 64,
        source_diarization_inventory_fingerprint="2" * 64,
        source_audio_evidence_fingerprint="3" * 64,
        selection_count=1,
        selected_clip_ids=[reviews[0].target_clip_uid],
        reviews=[reviews[0]],
    )
    assert one.selected_clip_ids == ["clip-000"]

    with pytest.raises(ValidationError, match="selection count"):
        BackgroundAudioPilotSelection(
            source_scout_inventory_fingerprint="1" * 64,
            source_diarization_inventory_fingerprint="2" * 64,
            source_audio_evidence_fingerprint="3" * 64,
            selection_count=20,
            selected_clip_ids=[item.target_clip_uid for item in reviews[:19]],
            reviews=reviews,
        )

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        BackgroundAudioPilotSelection(
            source_scout_inventory_fingerprint="1" * 64,
            source_diarization_inventory_fingerprint="2" * 64,
            source_audio_evidence_fingerprint="3" * 64,
            selection_count=0,
            selected_clip_ids=[],
            reviews=[],
        )


def test_selection_contains_only_background_rich_reviews_in_source_order() -> None:
    reviews = [
        BackgroundAudioSelectionReview(target_clip_uid="clip-a", label="clean"),
        BackgroundAudioSelectionReview(
            target_clip_uid="clip-b", label="background-rich"
        ),
        BackgroundAudioSelectionReview(target_clip_uid="clip-c", label="uncertain"),
        BackgroundAudioSelectionReview(
            target_clip_uid="clip-d", label="background-rich"
        ),
    ]
    selection = BackgroundAudioPilotSelection(
        source_scout_inventory_fingerprint="1" * 64,
        source_diarization_inventory_fingerprint="2" * 64,
        source_audio_evidence_fingerprint="3" * 64,
        selection_count=2,
        selected_clip_ids=["clip-b", "clip-d"],
        reviews=reviews,
    )
    assert selection.selected_clip_ids == ["clip-b", "clip-d"]

    with pytest.raises(ValidationError, match="background-rich reviews"):
        BackgroundAudioPilotSelection(
            source_scout_inventory_fingerprint="1" * 64,
            source_diarization_inventory_fingerprint="2" * 64,
            source_audio_evidence_fingerprint="3" * 64,
            selection_count=2,
            selected_clip_ids=["clip-a", "clip-b"],
            reviews=reviews,
        )


def test_source_drift_cleans_temporary_output_without_publication(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path)
    inventory = build_background_audio_scout_inventory(audio_run_root=tmp_path)
    Path(inventory.jobs[-1].target_audio_binding_path).write_text(
        '{"changed":true}\n', encoding="utf-8"
    )
    output = background_audio_scout_output_root(tmp_path)

    with pytest.raises(ValueError, match="evidence changed"):
        run_background_audio_scout(inventory=inventory, output_root=output)

    assert not output.exists()
    assert not list(tmp_path.glob(".background_audio_scout.tmp-*"))


def test_inventory_rejects_any_automatic_selection_flag(tmp_path: Path) -> None:
    _write_source(tmp_path)
    inventory = build_background_audio_scout_inventory(audio_run_root=tmp_path)
    payload = inventory.model_dump(mode="json")
    payload["automatic_background_selection_applied"] = True

    with pytest.raises(ValidationError):
        BackgroundAudioScoutInventory.model_validate(payload)
