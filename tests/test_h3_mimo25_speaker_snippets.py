from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from r2v_data_v2.h3 import mimo25_backend as mb
from r2v_data_v2.h3.audio_backends import AudioFileProbe, FFmpegAudioMediaBackend
from tests.h3_mimo_two_turn_helpers import SnippetAudioBackend, split_annotation
from tests.test_h3_mimo25_av_shadow import _annotation, _backend, _job_fixture


def _fixture(tmp_path, source_rate=16000):
    job = _job_fixture(tmp_path)
    visual, speech, final = split_annotation(_annotation(resolution="uncertain").model_dump_json())
    speech = json.loads(speech)
    segments, decisions, groundings = [], [], []
    for index, group in enumerate(("g1", "g2", "g1")):
        start, end = index * source_rate // 4, (index + 1) * source_rate // 4
        segment_id = f"segment_{index + 1}"
        segments.append(job.segments[0].model_copy(update={
            "segment_id": segment_id, "start_time": start / source_rate,
            "end_time": end / source_rate, "source_start_sample": start,
            "source_end_sample": end, "source_sample_rate_hz": source_rate,
        }))
        decisions.append({**speech["audio_observation"]["segment_decisions"][0],
                          "segment_id": segment_id, "primary_speaker_group": group})
        bound = {**speech["av_grounding"]["segment_groundings"][0], "segment_id": segment_id}
        if group == "g1":
            bound.update(binding_status="offscreen", entity_id=None, speech_presentation="offscreen_spoken")
        groundings.append(bound)
    speech["audio_observation"]["segment_decisions"] = decisions
    speech["av_grounding"]["segment_groundings"] = groundings
    final = json.loads(final)
    final["speaker_voice_profiles"] = [
        {"speaker_group": group, "voice_characteristics": None} for group in ("g1", "g2")
    ]
    job = job.model_copy(update={"segments": segments})
    assembly = mb.MimoSpeechAVAssemblyDraft.model_validate(speech)
    return job, assembly, (visual, json.dumps(speech), json.dumps(final))


@pytest.mark.parametrize("source_rate", [16000, 32000, 44100])
@pytest.mark.parametrize("transport", ["xiaomi", "sglang"])
def test_exact_snippets_keep_real_uncertain_groups_all_intervals_and_source_bytes(tmp_path, source_rate, transport):
    job, assembly, raws = _fixture(tmp_path, source_rate)
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws], transport=transport)
    media = SnippetAudioBackend()
    backend._audio_media_backend = media
    stems = {}
    for kind in ("speech", "music", "sfx"):
        stems[kind] = tmp_path / f"{kind}.flac"
        stems[kind].write_bytes(kind.encode())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    result = backend._request(job, allowed_reference_labels={"<Subject 1>", "<Picture 1>"}, auxiliary_audio_paths=stems)
    assert len(calls.requests) == 3
    content = calls.requests[2]["messages"][-1]["content"]
    assert content[0]["type"] == "video_url"
    targets = mb._speaker_profile_targets(assembly, job)
    assert [(t["speaker_group"], t["speaker_id"]) for t in targets] == [("g1", "S1"), ("g2", "S2")]
    assert targets[0]["segments"][0]["final_binding"] == "offscreen"
    assert targets[1]["segments"][0]["final_binding"] == "<Subject 1>"
    target_text = content[1]["text"].split("FINALIZED SPEAKER-PROFILE TARGETS:\n")[1]
    assert json.loads(target_text.split("\nRESPONSE SCHEMA:")[0]) == targets
    assert [item["type"] for item in content] == ["video_url", "text"] + ["text", "audio_url"] * 6
    labels = [json.loads(content[i]["text"].removeprefix("speaker snippet: ")) for i in (4, 6, 8)]
    assert [(label["speaker_group"], label["speaker_id"], label["segment_id"]) for label in labels] == [
        ("g1", "S1", "segment_1"), ("g1", "S1", "segment_3"), ("g2", "S2", "segment_2"),
    ]
    assert job.segments[0].asr_text not in json.dumps(content)
    for label, extraction, position in zip(labels, media.extractions, (5, 7, 9), strict=True):
        segment = next(s for s in job.segments if s.segment_id == label["segment_id"])
        expected = (round(segment.source_start_sample * 32000 / source_rate),
                    round(segment.source_end_sample * 32000 / source_rate))
        assert (extraction["source_start_sample"], extraction["source_end_sample"]) == expected
        assert (extraction["start_time"], extraction["end_time"]) == (segment.start_time, segment.end_time)
        assert (label["speech_stem_start_sample"], label["speech_stem_end_sample"]) == expected
        assert extraction["full_audio_path"] == extraction["source_audio_path"] == stems["speech"]
        assert extraction["channels"] == 2 and extraction["output_format"] == "flac"
        payload = json.loads(base64.b64decode(content[position]["audio_url"]["url"].split(",", 1)[1]))
        assert payload == {"source_start_sample": expected[0], "source_end_sample": expected[1], "sample_rate_hz": 32000}
        assert not extraction["destination"].parent.exists()
    for position, kind in ((3, "speech"), (11, "music"), (13, "sfx")):
        assert content[position]["audio_url"]["url"] == backend.config.media_resolver.resolve(stems[kind])
    assert json.loads(result[0])["audio_observation"]["speaker_voice_profiles"] == [
        {"speaker_group": group, "voice_characteristics": None} for group in ("g1", "g2")
    ]
    repeated = backend._speaker_snippet_content(job, targets, stems["speech"])
    assert repeated == content[4:10]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == set(before)


def test_no_targets_does_not_probe_or_invent_snippets(tmp_path):
    job, assembly, raws = _fixture(tmp_path)
    for decision in assembly.audio_observation.segment_decisions:
        decision.primary_speaker_group = None
    backend, _ = _backend(tmp_path, [(raw, 8) for raw in raws])
    targets = mb._speaker_profile_targets(assembly, job)
    assert targets == []
    assert backend._speaker_snippet_content(job, targets, tmp_path / "missing.flac") == []


@pytest.mark.parametrize("failure", ["extract", "encode"])
def test_snippet_failure_cleans_temp_and_preserves_two_completed_turns(tmp_path, monkeypatch, failure):
    job, _, raws = _fixture(tmp_path)
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws])
    media = SnippetAudioBackend()
    backend._audio_media_backend = media
    if failure == "extract":
        extract = media.extract_voice_reference

        def fail(**kwargs):
            extract(**kwargs)
            raise OSError("synthetic crop failure")

        monkeypatch.setattr(media, "extract_voice_reference", fail)
    else:
        resolve = mb.MimoMediaResolver.resolve

        def fail(self, path):
            if path.name.startswith("snippet-"):
                raise ValueError("synthetic encoding failure")
            return resolve(self, path)

        monkeypatch.setattr(mb.MimoMediaResolver, "resolve", fail)
    with pytest.raises(mb.MimoBackendFailure) as error:
        backend._request(job, allowed_reference_labels={"<Subject 1>"}, auxiliary_audio_paths={
            kind: Path(job.target_full_audio_path) for kind in ("speech", "music", "sfx")
        })
    assert error.value.model_call_count == len(calls.requests) == 2
    assert error.value.raw_responses == raws[:2]
    assert len(error.value.diagnostics) == 2
    assert error.value.recheck_count == error.value.http_retry_count == 0
    assert len(media.extractions) == 1
    assert not media.extractions[0]["destination"].parent.exists()


def test_snippet_extraction_uses_existing_exact_sample_trim_without_padding(tmp_path, monkeypatch):
    media = FFmpegAudioMediaBackend()
    source, destination = tmp_path / "speech.flac", tmp_path / "snippet.flac"
    commands = []
    monkeypatch.setattr(media, "probe_audio_file", lambda path: AudioFileProbe(
        sample_rate_hz=32000, channels=2, frame_count=32000 if path == source else 8000,
        duration_seconds=1.0 if path == source else 0.25, format_name="flac",
    ))
    monkeypatch.setattr(media, "_publish_command", lambda command, path: commands.append((command, path)))
    media.extract_voice_reference(
        clip_uid="clip", entity_id="", full_audio_path=source, source_audio_path=source,
        source_start_sample=8000, source_end_sample=16000, start_time=0.25, end_time=0.5,
        destination=destination, sample_rate_hz=32000, channels=2, output_format="flac",
    )
    command, output = commands[0]
    assert output == destination and command[command.index("-i") + 1] == str(source)
    assert command[command.index("-af") + 1] == "atrim=start_sample=8000:end_sample=16000,asetpts=PTS-STARTPTS"
    assert not any(value in str(command) for value in ("apad", "atempo", "-ss", "-to"))
