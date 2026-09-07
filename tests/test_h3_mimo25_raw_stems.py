from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.mimo25_av_reconcile import _inventory
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_PROMPT_VERSION,
    SYSTEM_PROMPT,
    MimoAuxAudioDescription,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoH3Semantics,
    MimoMediaResolver,
    OpenAIMimo25Backend,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    StemAwareOpenAIMimo25Backend,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import load_stem_shadow
from tests.test_h3_audio_shadow_qa import _fixture, _snapshot
from tests.test_h3_mimo25_av_shadow import _annotation, _Completions
from tools import run_h3_mimo25_stem_reconcile_shadow as cli


def _subset_inventory(base, clip_ids):
    by_id = {job.clip_uid: job for job in base.jobs}
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
    if values["source_diarization_inventory_sha256"] is None:
        values.pop("source_diarization_inventory_sha256")
    values["jobs"] = [by_id[uid].model_dump(mode="json") for uid in clip_ids]
    values["clip_count"] = len(clip_ids)
    return _inventory(values)


def test_final_av_prompt_preserves_positive_auxiliary_observations():
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_unified_av_reconcile_v32"
    assert "FACTUAL CONTRADICTION FILTER" in SYSTEM_PROMPT
    assert "NOT A REQUIREMENT TO RE-PROVE EVERY AUXILIARY DETAIL" in SYSTEM_PROMPT
    assert "Preserve positive observations by default" in SYSTEM_PROMPT
    assert "Do NOT require every auxiliary detail to be independently re-proven" in SYSTEM_PROMPT
    assert "melancholic, reflective, tense, light" in SYSTEM_PROMPT
    assert "correct a mistaken source without deleting the sound" in SYSTEM_PROMPT
    assert "Auxiliary negative/absence claims never establish absence" in SYSTEM_PROMPT
    assert "Music never belongs in overall_soundscape" in SYSTEM_PROMPT
    assert "not merely because a positive candidate is weak or masked" in SYSTEM_PROMPT
    assert "Reinspect the ORIGINAL TARGET AV before using any claim" not in SYSTEM_PROMPT
    assert "unless the original AV supports them" not in SYSTEM_PROMPT


@pytest.mark.parametrize("clip_ids,mixed", [
    (["clip-a"], False),
    (["clip-z", "clip-a", "clip-m"], False),
    (["clip-z"], True),
])
def test_reconcile_ordered_subset_cli(tmp_path, monkeypatch, clip_ids, mixed):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, mixed=mixed)
    base = _subset_inventory(qa.build_mimo25_inventory(), clip_ids)
    monkeypatch.setattr(qa, "build_mimo25_inventory", lambda **kwargs: base)
    backend, completions, _, _ = _backend(
        tmp_path, shadow, [(_raw(), 8)] * len(clip_ids),
    )
    if mixed:
        monkeypatch.setattr(
            cli, "separation_skips",
            lambda **kwargs: [SimpleNamespace(clip_uid="clip-m")],
        )
    case_path = kwargs["case_manifest"]
    case = json.loads(case_path.read_text())
    case["clip_uids"] = clip_ids
    case_path.write_text(json.dumps(case))
    monkeypatch.setattr(cli, "build_mimo25_inventory", lambda **kwargs: base)
    monkeypatch.setattr(cli, "StemAwareOpenAIMimo25Backend", lambda *a, **k: backend)
    monkeypatch.setenv("MIMO_API_KEY", "fake")
    argv = [
        part for key, value in kwargs.items()
        for part in ("--" + key.replace("_", "-"), str(value))
    ]
    result = cli.main([
        *argv, "--media-root", str(tmp_path), "--allow-unverified", "--overwrite",
    ])
    summary = result["summary"]
    assert result["clip_uids"] == clip_ids
    assert summary["clip_uids"] == clip_ids
    assert summary["processed_clip_uids"] == clip_ids
    assert summary["skipped_clips"] == []
    assert summary["diarization_failed_clips"] == []
    assert [row["clip_uid"] for row in _records(shadow)] == clip_ids
    assert len(completions.requests) == 3 * len(clip_ids)
    assert summary["audio_model_call_count"] == 2 * len(clip_ids)


@pytest.mark.parametrize("clip_ids", [["clip-m", "clip-z"], ["unknown"], ["clip-a", "clip-a"]])
def test_reconcile_rejects_invalid_subsets(tmp_path, monkeypatch, clip_ids):
    _, shadow = _fixture(tmp_path, monkeypatch)
    inventory = load_stem_shadow(shadow / "separation")[0]
    with pytest.raises(ValueError, match="ordered subset"):
        cli._validate_stage_closure(
            case_manifest=SimpleNamespace(clip_uids=clip_ids),
            stem_inventory=inventory,
            separation_root=shadow / "separation",
            diarization_provenance=SimpleNamespace(source_stem_root=str(shadow / "separation")),
        )
    base = qa.build_mimo25_inventory()
    by_id = {job.clip_uid: job for job in base.jobs}
    requested = [
        by_id.get(uid, base.jobs[0].model_copy(update={"clip_uid": uid}))
        for uid in clip_ids
    ]
    with pytest.raises(ValueError, match="unique and usable|ordered usable subset"):
        build_stem_reconcile_jobs(
            base_inventory=SimpleNamespace(jobs=requested),
            stem_diarization_root=shadow / "diarization",
            stem_asr_root=shadow / "asr", route="music_first",
        )


def _raw():
    return _annotation().model_dump_json().replace("segment_1", "segment_0001")


def _backend(tmp_path, shadow, responses):
    stems = load_stem_shadow(shadow / "separation")[1]
    completions = _Completions(responses)
    original = completions.create

    def create(**request):
        if request["response_format"]["json_schema"]["name"] == "MimoAuxAudioDescription":
            completions.requests.append(request)
            return _Completions([('{"description":"An audible auxiliary sound."}', 4)]).create(**request)
        return original(**request)

    completions.create = create
    backend = StemAwareOpenAIMimo25Backend(
        MimoBackendConfig(
            api_key="fake",
            transport="sglang",
            icl="official_ref2va_v1",
            media_resolver=MimoMediaResolver(
                mode="http",
                media_root=tmp_path,
                media_base_url="http://media.invalid",
            ),
        ),
        stem_records_by_clip={r.clip_uid: r for r in stems},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    base = qa.build_mimo25_inventory()
    jobs = build_stem_reconcile_jobs(
        base_inventory=base,
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr",
        route="music_first",
    )
    return backend, completions, stems, jobs


def _run(shadow, backend, stems, jobs):
    return run_mimo25_stem_reconcile_shadow(
        jobs=jobs,
        stem_records=stems,
        backend=backend,
        output_root=shadow / MIMO25_STEM_RECONCILE_STAGE,
        route="music_first",
        allow_unverified=True,
        overwrite=True,
    )


def _records(shadow):
    path = shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _reconcile(backend, job):
    return backend.reconcile(
        job,
        segment_ids=[s.segment_id for s in job.segments],
        transcribed_segment_ids=[
            s.segment_id for s in job.segments if s.asr_status == "transcribed"
        ],
        allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
    )


@pytest.mark.parametrize("audio_tokens", [0, None])
def test_embedded_audio_accounting_is_warning_only(tmp_path, monkeypatch, audio_tokens):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, _, jobs = _backend(tmp_path, shadow, [(_raw(), audio_tokens)])
    result = _reconcile(backend, jobs[0])
    assert result.model_call_count == len(completions.requests) == 1
    assert result.recheck_count == result.http_retry_count == 0
    expected = "embedded_audio_tokens_zero" if audio_tokens == 0 else "audio_tokens_unavailable"
    assert expected in result.diagnostics[0].warnings
    assert not any(
        item["type"] == "audio_url"
        for item in completions.requests[0]["messages"][-1]["content"]
    )


@pytest.mark.parametrize("music_failure", [None, "api", "json"])
def test_parallel_auxiliary_results_keep_roles_and_allow_positive_recovery(
    tmp_path, monkeypatch, music_failure,
):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    positive = "Soft instrumental music is clearly audible."
    final = {"overall_soundscape": "A clink.", "non_diegetic_music": positive}
    payload["h3_semantics"].update(final)
    backend, completions, stems, jobs = _backend(
        tmp_path, shadow, [(json.dumps(payload), 0)],
    )
    jobs = jobs[:1]
    record = next(r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first")
    role_urls = {
        backend.config.media_resolver.resolve(Path(record.stem(role).canonical_stem_path)): role
        for role in ("music", "sfx")
    }
    barrier = Barrier(2)
    sfx_finished = Event()
    completion_order = []
    original = completions.create

    def create(**request):
        if request["response_format"]["json_schema"]["name"] != "MimoAuxAudioDescription":
            assert completion_order == ["sfx", "music"]
            return original(**request)
        role = role_urls[request["messages"][1]["content"][0]["audio_url"]["url"]]
        completions.requests.append(request)
        barrier.wait(timeout=5)
        if role == "music":
            assert sfx_finished.wait(timeout=5)
            completion_order.append(role)
            if music_failure == "api":
                raise OSError("synthetic auxiliary error")
            raw = "bad JSON" if music_failure == "json" else json.dumps({"description": positive})
        else:
            raw = '{"description":"A clink."}'
            completion_order.append(role)
            sfx_finished.set()
        return _Completions([(raw, None)]).create(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    result = _records(shadow)[0]
    assert completion_order == ["sfx", "music"]
    assert result["status"] == "ready" and result["annotation"]
    assert result["sfx_stem_description"] == "A clink."
    assert result["sfx_stem_error"] is None
    assert result["music_stem_description"] == (positive if music_failure is None else None)
    assert bool(result["music_stem_error"]) == (music_failure is not None)
    if music_failure == "json":
        assert result["music_stem_raw_response"] == "bad JSON"
    assert {key: result["annotation"]["h3_semantics"][key] for key in final} == final
    assert summary.av_model_call_count == 1
    assert summary.audio_model_call_count == 2
    assert summary.model_call_count == len(completions.requests) == 3
    text = completions.requests[-1]["messages"][-1]["content"][-1]["text"]
    assert "A clink." in text
    assert "music_separator_candidate" in text and "sfx_separator_candidate" in text
    assert (positive if music_failure is None else "SOURCE_UNAVAILABLE") in text
    assert result["diagnostics"][1]["input_modality"] == "auxiliary_audio_only"
    assert result["diagnostics"][0]["input_modality"] == "auxiliary_audio_only"
    assert result["diagnostics"][2]["input_modality"] == "target_video_with_embedded_audio"


def test_final_av_malformed_retains_stems_without_fourth_call(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [("bad JSON", 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    result = _records(shadow)[0]
    assert result["status"] == "failed" and result["annotation"] is None
    assert result["music_stem_description"] and result["sfx_stem_description"]
    assert "sound_partition" not in result and "text_model_call_count" not in result
    assert summary.model_call_count == len(completions.requests) == 3


def test_changed_auxiliary_file_is_not_sent(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    record = next(r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first")
    Path(record.stem("music").canonical_stem_path).write_bytes(b"changed fixture")
    summary = _run(shadow, backend, stems, jobs[:1])
    result = _records(shadow)[0]
    assert "auxiliary stem changed" in result["music_stem_error"]
    assert result["music_stem_description"] is None and result["sfx_stem_description"]
    assert summary.audio_model_call_count == 1
    assert summary.model_call_count == len(completions.requests) == 2


def test_auxiliary_media_preflight_has_no_model_attempt(tmp_path):
    completions = _Completions([])
    backend = OpenAIMimo25Backend(
        MimoBackendConfig(
            api_key="fake", transport="sglang",
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        ),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    result = backend.describe_auxiliary_audio(tmp_path / "missing.flac")
    assert result.error and result.description is None
    assert result.model_call_count == 0 and not completions.requests


@pytest.mark.parametrize(
    "candidate,soundscape,music,caption_suffix",
    [
        ("Soft piano music is audible.", "N/A", "Soft piano background music plays.", ""),
        ("Soft, slow, melancholic piano music is audible.", "N/A",
         "Soft, slow, melancholic piano background music plays.", ""),
        ("A low continuous rumble, possibly from machinery or an engine.",
         "A low continuous rumble or hum is audible.", "N/A", ""),
        ("Piano music is audible.", "N/A", "N/A", " A visible pianist plays the piano in the room."),
        ("A large engine in a moving vehicle fills a large empty space.",
         "A faint low-frequency hum is audible.", "N/A", ""),
    ],
)
def test_final_av_authors_sound_and_diegetic_caption_without_postprocessing(
    tmp_path, monkeypatch, candidate, soundscape, music, caption_suffix,
):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"].update(overall_soundscape=soundscape, non_diegetic_music=music)
    payload["h3_semantics"]["shot1_caption"] += caption_suffix
    backend, completions, stems, jobs = _backend(
        tmp_path, shadow, [(json.dumps(payload), 8)] * 3,
    )
    original = completions.create
    jobs_before = [job.model_dump(mode="json") for job in jobs]

    def create(**request):
        if request["response_format"]["json_schema"]["name"] == "MimoAuxAudioDescription":
            completions.requests.append(request)
            return _Completions([(json.dumps({"description": candidate}), 8)]).create(**request)
        assert candidate in request["messages"][-1]["content"][-1]["text"]
        return original(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    assert summary.ready_count == 3 and summary.model_call_count == 9
    assert summary.av_model_call_count == 3 and summary.audio_model_call_count == 6
    assert len(completions.requests) == 9
    assert backend.provenance.prompt_version == MIMO25_PROMPT_VERSION
    assert [job.model_dump(mode="json") for job in jobs] == jobs_before
    assert all(row["annotation"]["h3_semantics"] == payload["h3_semantics"] for row in _records(shadow))
    qa.build_audio_shadow_qa(**kwargs)
    clips = json.loads((shadow / "qa/data.json").read_text())["clips"]
    for clip in clips:
        assert clip["final_h3"]["status"] == "ready"
        final = clip["final_h3"]["text"]
        assert f"overall_soundscape:\n{soundscape}" in final
        assert f"non_diegetic_music:\n{music}" in final
        expected_detailed = payload["h3_semantics"]["style_opening"] + "\n[Shot 1] " + payload["h3_semantics"]["shot1_caption"]
        assert expected_detailed in final
        headings = ["subject_definitions:", "summary:", "retention_analysis:",
                    "detailed_description:", "overall_soundscape:", "non_diegetic_music:"]
        assert [final.index(heading) for heading in headings] == sorted(final.index(h) for h in headings)


def test_real_entry_without_facts_sends_two_audio_then_one_final_av(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(
        tmp_path,
        shadow,
        [(_raw(), 8)] * 3,
    )
    for name in ("mimo_reconcile_av_rawstems_sound_partition", "mimo_reconcile_av_stemtext_sound_partition",
                 "mimo_v29_oneclip_smoke", "mimo_v30_833_oneclip_smoke"):
        legacy_output = shadow / name
        legacy_output.mkdir()
        (legacy_output / "sentinel.json").write_text('{"preserved": true}')
    source_before = {
        path: data
        for path, data in _snapshot(tmp_path).items()
        if MIMO25_STEM_RECONCILE_STAGE not in path.parts
    }
    monkeypatch.setattr(cli, "build_mimo25_inventory", qa.build_mimo25_inventory)
    monkeypatch.setattr(cli, "StemAwareOpenAIMimo25Backend", lambda *a, **k: backend)
    argv = [
        part
        for key, value in kwargs.items()
        for part in ("--" + key.replace("_", "-"), str(value))
    ]
    with pytest.raises(ValueError, match="unverified"):
        cli.main([*argv, "--dry-run"])
    monkeypatch.setenv("MIMO_API_KEY", "fake")
    result = cli.main(
        [*argv, "--media-root", str(tmp_path), "--allow-unverified", "--overwrite"]
    )
    assert result["output_root"] == str(shadow / MIMO25_STEM_RECONCILE_STAGE)
    assert not (shadow / "mimo_stem_facts").exists()
    assert not (shadow / "stem_views").exists()
    assert result["summary"]["ready_count"] == 3
    assert result["summary"]["av_model_call_count"] == 3
    assert "text_model_call_count" not in result["summary"]
    assert result["summary"]["model_call_count"] == len(completions.requests) == 9
    assert result["summary"]["processed_clip_uids"] == ["clip-z", "clip-a", "clip-m"]

    audio_a, audio_b, first = completions.requests[:3]
    assert [r["response_format"]["json_schema"]["name"] for r in completions.requests] == [
        "MimoAuxAudioDescription", "MimoAuxAudioDescription", "MimoAVAnnotationDraft",
    ] * 3
    assert result["summary"]["audio_model_call_count"] == 6
    assert [m["role"] for m in first["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    media = first["messages"][-1]["content"]
    assert sum(p["type"] == "video_url" for p in media) == 1
    assert sum(p["type"] == "image_url" for p in media) == 1
    audio_urls = [p["audio_url"]["url"] for p in media if p["type"] == "audio_url"]
    selected = next(
        r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first"
    )
    assert audio_urls == []
    auxiliary_requests = [audio_a, audio_b]
    assert audio_a["messages"][0] == audio_b["messages"][0]
    expected_paths = {
        backend.config.media_resolver.resolve(Path(selected.stem(kind).canonical_stem_path))
        for kind in ("music", "sfx")
    }
    assert {
        request["messages"][1]["content"][0]["audio_url"]["url"]
        for request in auxiliary_requests
    } == expected_paths
    for request in auxiliary_requests:
        assert [m["role"] for m in request["messages"]] == ["system", "user"]
        assert len(request["messages"][1]["content"]) == 1
        assert request["messages"][1]["content"][0]["type"] == "audio_url"
        assert not any(word in request["messages"][0]["content"].lower() for word in ("music", "sfx", "asr", "caption", "reference", "icl"))
        assert request["response_format"]["json_schema"]["schema"] == MimoAuxAudioDescription.model_json_schema()
        assert request["temperature"] == 0.0 and request["max_completion_tokens"] == 1024
        assert request["stream"] is False and request["reasoning_effort"] == "none"
        assert request["extra_body"] == {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    full = json.dumps(media)
    assert "stem_facts" not in full and "STEM AUXILIARY EVIDENCE" not in full
    assert "input_audio" not in full and "<Audio " not in full
    assert (
        backend.config.media_resolver.resolve(Path(jobs[0].target_full_audio_path))
        not in full
    )
    assert (
        backend.config.media_resolver.resolve(
            Path(selected.stem("speech").canonical_stem_path)
        )
        not in full
    )
    assert "music_separator_candidate" in full and "sfx_separator_candidate" in full
    assert "An audible auxiliary sound." in full
    assert "Auxiliary separations" not in first["messages"][0]["content"]
    assert first["extra_body"]["use_audio_in_video"] is True
    assert first["extra_body"]["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }
    assert first["reasoning_effort"] == "none"

    schema = first["response_format"]["json_schema"]["schema"]
    assert "sound_description" not in schema["$defs"]["MimoH3Semantics"]["properties"]
    assert {"overall_soundscape", "non_diegetic_music"} <= set(schema["$defs"]["MimoH3Semantics"]["required"])
    assert set(schema["$defs"]["MimoH3Semantics"]["properties"]) == {
        "subject_definitions", "summary", "style_opening", "shot1_caption",
        "overall_soundscape", "non_diegetic_music", "visual_retention_analysis",
    }
    assert not hasattr(backend, "reconcile_sound_descriptions")
    assert not hasattr(backend, "partition_sound_description")
    for forbidden in (
        "overall_soundscape_status",
        "complete_silence_verified",
        "temporal_non_speech_events",
    ):
        assert forbidden not in json.dumps(schema)
    assert (
        "audio_semantics" not in schema["$defs"]["MimoAudioObservation"]["properties"]
    )
    assert all(path.read_bytes() == data for path, data in source_before.items())

    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    assert all(row["final_h3"]["status"] == "ready" for row in data["clips"])
    assert (
        data["clips"][0]["reconcile"]["diagnostics"][2]["input_modality"]
        == "target_video_with_embedded_audio"
    )
    final = data["clips"][0]["final_h3"]["text"]
    assert final.count("[Shot 1]") == 1 and "[[" not in final
    assert _annotation().h3_semantics.shot1_caption in final
    assert "overall_soundscape:\nA quiet room tone and a clink." in final
    assert "non_diegetic_music:\nN/A" in final


@pytest.mark.parametrize(
    "failure",
    [
        "semantic",
        "format",
        "schema",
        "articulation",
    ],
)
def test_failures_keep_raw_and_continue_without_retries(tmp_path, monkeypatch, failure):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    if failure == "semantic":
        payload["h3_semantics"]["subject_definitions"][0]["subject_label"] = (
            "<Subject 9>"
        )
    elif failure == "format":
        payload["h3_semantics"]["shot1_caption"] = "<d>[Chinese] unchanged</d>"
    elif failure == "articulation":
        payload["visual_observation"]["segment_views"][0]["entity_observations"][0][
            "speech_correlated_articulation"
        ] = "not_observed"
    elif failure == "schema":
        del payload["h3_semantics"]["subject_definitions"]
    first_raw = json.dumps(payload)
    backend, completions, stems, jobs = _backend(
        tmp_path,
        shadow,
        [
            (first_raw, 8),
            (_raw(), 8),
            (_raw(), 8),
        ],
    )
    summary = _run(shadow, backend, stems, jobs)
    assert summary.failed_count == 1 and summary.ready_count == 2
    assert summary.av_model_call_count == 3
    assert summary.model_call_count == len(completions.requests) == 9
    records = _records(shadow)
    assert records[0]["status"] == "failed"
    assert records[0]["raw_responses"] == [first_raw]
    assert records[0]["music_stem_description"] and records[0]["sfx_stem_description"]
    qa.build_audio_shadow_qa(**kwargs)
    clip = json.loads((shadow / "qa/data.json").read_text())["clips"][0]
    assert (
        clip["direct_h3"]["shot1_caption"] == payload["h3_semantics"]["shot1_caption"]
    )
    assert clip["direct_h3"]["overall_soundscape"]
    assert clip["final_h3"]["status"] == "unavailable"
    page = (shadow / "qa/review.html").read_text()
    assert "Final AV raw response" in page and "Final overall_soundscape" in page
    assert "Text-only sound partition" not in page


def test_av_http_failure_is_single_attempt_and_next_clip_runs(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(
        tmp_path,
        shadow,
        [(_raw(), 8)] * 2,
    )
    original = completions.create

    def create(**request):
        if len(completions.requests) == 2:
            completions.requests.append(request)
            raise OSError("synthetic AV timeout")
        return original(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    assert summary.ready_count == 2 and summary.failed_count == 1
    assert summary.av_model_call_count == 3
    assert len(completions.requests) == 9
    assert summary.audio_model_call_count == 6
    assert _records(shadow)[0]["raw_responses"] == []


def test_sdk_retries_disabled_even_for_injected_openai_client(tmp_path):
    from openai import OpenAI

    client = OpenAI(api_key="fake", base_url="http://unused.invalid/v1", max_retries=4)
    config = MimoBackendConfig(
        api_key="fake",
        media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
    )
    backend = OpenAIMimo25Backend(config, client=client)
    assert backend.client.max_retries == 0
    backend.client.close()
    client.close()
    owned = OpenAIMimo25Backend(config)
    assert owned.client.max_retries == 0
    owned.client.close()


def test_multiple_same_speaker_segments_may_share_one_unchanged_dialogue(
    tmp_path, monkeypatch
):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, _, _, jobs = _backend(tmp_path, shadow, [])
    job = jobs[0]
    values = job.model_dump(mode="json")
    segment = values["segments"][0]
    values["segments"] = []
    for i in range(1, 5):
        start, end = (i - 1) * 0.25, (i - 1) * 0.25 + 0.2
        values["segments"].append(
            {
                **segment,
                "segment_id": f"segment_{i:04d}",
                "start_time": start,
                "end_time": end,
                "source_start_sample": round(start * segment["source_sample_rate_hz"]),
                "source_end_sample": round(end * segment["source_sample_rate_hz"]),
            }
        )
    payload = json.loads(_raw())
    for section, key in (
        ("visual_observation", "segment_views"),
        ("audio_observation", "segment_decisions"),
        ("av_grounding", "segment_groundings"),
    ):
        template = payload[section][key][0]
        payload[section][key] = [
            {**template, "segment_id": s["segment_id"]} for s in values["segments"]
        ]
    caption = "(S1) <Subject 1> gestures. <d>[Chinese] Original model wording stays intact.</d>"
    payload["h3_semantics"]["shot1_caption"] = caption
    from r2v_data_v2.h3.mimo25_av_reconcile import _job

    values.pop("request_fingerprint")
    job = _job(values)
    backend.client.chat.completions.responses = [(json.dumps(payload), 8)]
    result = _reconcile(backend, job)
    assert result.model_call_count == 1
    assert result.annotation.h3_semantics.shot1_caption == caption
    assert len(result.annotation.segment_decisions) == 4


@pytest.mark.parametrize(
    "text",
    [
        "(S1) <d>[English] a <d>[English] b</d></d>",
        "(S1) <d>[English] a",
        "(S1) <d>no language</d>",
        "<d>[English] no source</d>",
        "(S9) <d>[English] unknown</d>",
        "(S1) <d>[English] a</d> <d>[English] b</d>",
        "[Shot 1] body",
        "[[segment:x]]",
    ],
)
def test_dialogue_format_failures_are_not_rewritten(text):
    unchanged, issues, warnings = protect_direct_dialogue(
        text,
        [{"speaker_id": "S1", "text": "not used"}],
        allowed_labels=set(),
    )
    assert unchanged == text and issues and warnings == []


def test_media_preflight_failure_has_zero_model_calls(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, _, jobs = _backend(tmp_path, shadow, [])
    monkeypatch.setattr(
        backend,
        "_media_content",
        lambda job: (_ for _ in ()).throw(ValueError("missing stem")),
    )
    with pytest.raises(MimoBackendFailure) as failure:
        _reconcile(backend, jobs[0])
    assert failure.value.model_call_count == 0 and not failure.value.diagnostics
    assert not completions.requests


@pytest.mark.parametrize(
    "payload, accepted",
    [
        ({"overall_soundscape": "", "non_diegetic_music": ""}, True),
        (
            {
                "overall_soundscape": "Music and speech, uncertain.",
                "non_diegetic_music": "N/A",
            },
            True,
        ),
        ({"overall_soundscape": 7, "non_diegetic_music": "N/A"}, False),
    ],
)
def test_final_sound_fields_check_only_string_shape(payload, accepted):
    values = _annotation().h3_semantics.model_dump(mode="json")
    values.update(payload)
    if accepted:
        result = MimoH3Semantics.model_validate(values)
        assert {key: getattr(result, key) for key in payload} == payload
    else:
        with pytest.raises(ValueError):
            MimoH3Semantics.model_validate(values)


def test_optional_music_reference_does_not_invent_timing(tmp_path):
    from r2v_data_v2.h3.mimo25_h3_materializer import _select_music_reference

    result = _select_music_reference(
        job=None,
        record=SimpleNamespace(annotation=_annotation()),
        final_root=tmp_path,
        audio_backend=None,
        temporary_root=tmp_path,
        analyzer=None,
    )
    assert result == (None, None, ["music_reference_timing_unavailable"])
