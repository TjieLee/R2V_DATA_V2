from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.mimo25_av_reconcile import _inventory, _job
from r2v_data_v2.h3.mimo25_backend import (
    AUDIO_FINALIZE_SYSTEM_PROMPT,
    AUXILIARY_AUDIO_PROMPT,
    MIMO25_BACKEND_VERSION,
    MIMO25_PROMPT_VERSION,
    SYSTEM_PROMPT,
    VISUAL_SYSTEM_PROMPT,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoH3Semantics,
    MimoMediaResolver,
    OpenAIMimo25Backend,
    _normalize_speaker_annotation,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    MimoStemReconcileRecord,
    StemAwareOpenAIMimo25Backend,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import load_stem_shadow
from tests.h3_mimo_two_turn_helpers import (
    SnippetAudioBackend,
    assembly_raw,
    split_annotation,
)
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


def test_finalizer_uses_raw_stems_without_evidence_gates():
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_speech_assembly_v46"
    assert "separated music and sfx audio views of that SAME target" in AUDIO_FINALIZE_SYSTEM_PROMPT
    assert "Original AV remains primary authority" in AUDIO_FINALIZE_SYSTEM_PROMPT
    assert "quiet in the original mix" in AUDIO_FINALIZE_SYSTEM_PROMPT
    assert "residuals are not automatically true" in AUDIO_FINALIZE_SYSTEM_PROMPT
    assert "threshold" not in AUDIO_FINALIZE_SYSTEM_PROMPT


def test_visual_coverage_stays_in_turn1_without_word_count_gate():
    for requirement in (
        "composition/framing", "foreground/background",
        "appearance", "lighting/color", "camera behavior",
        "gaze/expression", "observable actions/state changes",
        "early-to-middle-to-late", "No hard word-count requirement",
    ):
        assert requirement in VISUAL_SYSTEM_PROMPT
    assert "PRIMARY H3 WRITING TASK" not in SYSTEM_PROMPT
    assert "SHOT1 CAPTION / DETAILED DESCRIPTION" not in SYSTEM_PROMPT
    values = json.loads(_raw())["h3_semantics"]
    values["shot1_caption"] = "A person stands still."
    assert MimoH3Semantics.model_validate(values).shot1_caption == values["shot1_caption"]
    assert MimoH3Semantics.model_json_schema()["properties"]["shot1_caption"]["minLength"] == 1


def test_caption_field_boundaries_and_no_grounding_explanation():
    assert "Grounding rationale and internal evidence belong only in av_grounding, not in shot1_caption" in SYSTEM_PROMPT
    assert "Do not expose internal grounding fields or evidence codes in consumer-facing prose" in SYSTEM_PROMPT
    assert "Do not add non-dialogue audio" in SYSTEM_PROMPT
    assert "An offscreen female voice (S1) says" not in SYSTEM_PROMPT


def test_visual_subject_preservation_remains_in_frozen_turn1():
    assert "Naturally cite visible entity Subjects when they appear" in VISUAL_SYSTEM_PROMPT
    assert "Profile/back/occluded/silent subjects remain visible" in VISUAL_SYSTEM_PROMPT
    assert "Preserve its visual content and ordering" in SYSTEM_PROMPT
    assert "If the full AV reasonably indicates that a known visible entity is speaking, bind that entity directly" in SYSTEM_PROMPT
    assert "A visible listener must still not inherit speech when the AV clearly indicates another source" in SYSTEM_PROMPT
    assert "Use H3 reference labels ONLY from allowed_h3_reference_labels" in SYSTEM_PROMPT


def test_prompt_reference_vocal_and_positive_audio_contracts():
    assert "VISIBLE SPEAKER BINDING" in SYSTEM_PROMPT
    assert "supporting clues, NOT mandatory prerequisites" in SYSTEM_PROMPT
    assert "Absence of supporting evidence is NOT a contradiction" in SYSTEM_PROMPT
    assert "resolution == resolved is NOT a prerequisite" in SYSTEM_PROMPT
    assert "visible_entity requires presence" not in SYSTEM_PROMPT
    assert "may resolve Stage B needs_acoustic_refinement only" not in SYSTEM_PROMPT
    assert "Use H3 reference labels ONLY from allowed_h3_reference_labels" in SYSTEM_PROMPT
    assert "NEVER emit <Audio N>" in SYSTEM_PROMPT
    assert "They are NOT Subject numbers" in SYSTEM_PROMPT
    assert "A silent Subject must not consume a speaker ID" in SYSTEM_PROMPT
    assert "first actual vocal source -> S1" in SYSTEM_PROMPT
    assert "SAME continuing speaker may inherit" in SYSTEM_PROMPT
    assert "When the speaker changes, the new vocal source MUST be explicitly introduced" in SYSTEM_PROMPT
    assert "The first dialogue event from a vocal source must establish its (Sx)" in SYSTEM_PROMPT
    assert "If speaker continuity is uncertain, repeat the explicit (Sx)" in SYSTEM_PROMPT
    assert "before each <d> MUST contain" not in SYSTEM_PROMPT
    assert "Repeat the same (Sx) for separate dialogue blocks" not in SYSTEM_PROMPT
    assert "If allowed_segment_ids is empty, output empty" in SYSTEM_PROMPT
    assert "describe only POSITIVE audible content" in AUXILIARY_AUDIO_PROMPT
    assert "Do NOT summarize what is absent" in AUXILIARY_AUDIO_PROMPT
    assert "Do NOT infer a physical source or cause, room/location/environment" in AUXILIARY_AUDIO_PROMPT
    assert "Do not mention that this is a source-separation output" in AUXILIARY_AUDIO_PROMPT
    assert not any(role in AUXILIARY_AUDIO_PROMPT.lower() for role in (
        "music stem", "sfx stem", "music separator", "sfx separator",
    ))


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
    assert len(completions.requests) == 4 * len(clip_ids)
    assert summary["audio_model_call_count"] == len(clip_ids)


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


def _backend(tmp_path, shadow, responses, *, polish_responses=None):
    stems = load_stem_shadow(shadow / "separation")[1]
    completions = _Completions(responses)
    original = completions.create

    def create(**request):
        if request["response_format"]["json_schema"]["name"] == "MimoVisualDraft":
            completions.requests.append(request)
            try:
                visual = split_annotation(completions.responses[0][0])[0]
            except (ValueError, KeyError, TypeError):
                visual = split_annotation(_raw())[0]
            return _Completions([(visual, 0)]).create(**request)
        if request["response_format"]["json_schema"]["name"] == "MimoAuxAudioDescription":
            completions.requests.append(request)
            return _Completions([('{"description":"An audible auxiliary sound."}', 4)]).create(**request)
        if request["response_format"]["json_schema"]["name"] == "MimoSpeakerMarkerPolish":
            completions.requests.append(request)
            response = polish_responses.pop(0) if polish_responses is not None else json.dumps({
                "shot1_caption": json.loads(request["messages"][-1]["content"])["existing_shot_caption"],
                "needs_review": True,
            })
            if isinstance(response, Exception):
                raise response
            return _Completions([(response, 0)]).create(**request)
        response = completions.responses[0]
        if request["response_format"]["json_schema"]["name"] in {"MimoSpeechAVAssemblyDraft", "MimoSpeakerProfileDraft"}:
            response_index = 2 if request["response_format"]["json_schema"]["name"] == "MimoSpeakerProfileDraft" else 1
            completions.requests.append(request)
            try:
                speech = split_annotation(response[0])[response_index]
            except (ValueError, KeyError, TypeError):
                speech = split_annotation(_raw())[response_index]
            return _Completions([(speech, response[1])]).create(**request)
        completions.responses[0] = (assembly_raw(response[0]), *response[1:])
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
        audio_media_backend=SnippetAudioBackend(),
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
    stems = backend.stem_records_by_clip[job.clip_uid]
    return backend.reconcile(
        job,
        auxiliary_audio_paths={kind: Path(stems.stem(kind).canonical_stem_path) for kind in ("speech", "music", "sfx")},
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
    assert result.model_call_count == len(completions.requests) == 4
    assert result.recheck_count == result.http_retry_count == 0
    expected = "embedded_audio_tokens_zero" if audio_tokens == 0 else "audio_tokens_unavailable"
    assert expected in result.diagnostics[1].warnings
    assert expected not in result.diagnostics[0].warnings
    assert not any(
        item["type"] == "audio_url"
        for item in completions.requests[0]["messages"][-1]["content"]
    )


@pytest.mark.parametrize("stem_kind", ["speech", "music", "sfx"])
@pytest.mark.parametrize("source_state", ["valid", "changed", "missing"])
def test_raw_stem_assets_replace_standalone_calls(tmp_path, monkeypatch, source_state, stem_kind):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"].update(
        non_diegetic_music="A soft, slow piano melody.", overall_soundscape="A low continuous rumble.",
    )
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    record = next(r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first")
    path = Path(record.stem(stem_kind).canonical_stem_path)
    if source_state == "changed":
        path.write_bytes(b"changed")
    elif source_state == "missing":
        path.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("standalone auxiliary inference must not run")
    monkeypatch.setattr(backend, "describe_auxiliary_audio", forbidden)
    summary = _run(shadow, backend, stems, jobs[:1])
    result = _records(shadow)[0]
    assert summary.audio_model_call_count == int(source_state == "valid")
    assert result["music_stem_description"] is result["sfx_stem_description"] is None
    if source_state != "valid":
        assert not completions.requests and summary.model_call_count == 0
        assert result["failure_code"] == "mimo_stem_media_invalid"
        return
    assert summary.ready_count == 1
    assert summary.model_call_count == len(completions.requests) == 4
    assert result["annotation"]["h3_semantics"] == payload["h3_semantics"]
    media = completions.requests[-1]["messages"][-1]["content"]
    audio_urls = [item["audio_url"]["url"] for item in media if item["type"] == "audio_url"]
    assert len(audio_urls) == 2
    assert audio_urls == [
        backend.config.media_resolver.resolve(Path(record.stem(kind).canonical_stem_path))
        for kind in ("music", "sfx")
    ]
    assert not any(d["input_modality"] == "auxiliary_audio_only" for d in result["diagnostics"])
    assert sum(item["type"] == "video_url" for item in media) == 1


def test_final_av_malformed_retains_stems_without_fourth_call(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [("bad JSON", 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    result = _records(shadow)[0]
    assert result["status"] == "failed" and result["annotation"] is None
    assert result["music_stem_description"] is result["sfx_stem_description"] is None
    assert "sound_partition" not in result and result["text_model_call_count"] == 0
    assert summary.model_call_count == len(completions.requests) == 4


def test_changed_auxiliary_file_is_not_sent(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    record = next(r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first")
    Path(record.stem("music").canonical_stem_path).write_bytes(b"changed fixture")
    summary = _run(shadow, backend, stems, jobs[:1])
    result = _records(shadow)[0]
    assert "auxiliary stem changed" in result["failure_reason"]
    assert result["status"] == "failed"
    assert summary.audio_model_call_count == 0
    assert summary.model_call_count == len(completions.requests) == 0


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


def test_final_prompt_uses_exact_caller_reference_allowlist(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, _, jobs = _backend(tmp_path, shadow, [])
    allowed = {"<Subject 1>"}
    prompt = backend._prompt(jobs[0], allowed_reference_labels=allowed)
    contract = json.loads(prompt.split("AUTHORITATIVE INPUT:\n", 1)[1])
    assert contract["allowed_h3_reference_labels"] == ["<Subject 1>"]
    assert allowed == {"<Subject 1>"}
    assert prompt == backend._prompt(jobs[0], allowed_reference_labels=allowed)
    assert not completions.requests


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
        assert request["response_format"]["json_schema"]["name"] != "MimoAuxAudioDescription"
        if request["response_format"]["json_schema"]["name"] == "MimoAudioFinalizeDraft":
            assert sum(p["type"] == "audio_url" for p in request["messages"][-1]["content"]) == 2
        assert candidate not in json.dumps(request["messages"])
        return original(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    assert summary.ready_count == 3 and summary.model_call_count == 12
    assert summary.av_model_call_count == 6 and summary.audio_model_call_count == 3
    assert len(completions.requests) == 12
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


def test_real_entry_without_facts_sends_two_audio_then_visual_and_final_av(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(
        tmp_path,
        shadow,
        [(_raw(), 8)] * 3,
    )
    for name in ("mimo_reconcile_av_rawstems_sound_partition", "mimo_reconcile_av_stemtext_sound_partition",
                 "mimo_v29_oneclip_smoke", "mimo_v30_833_oneclip_smoke",
                 "mimo_reconcile_stemtext_final_av", "mimo_reconcile_stemtext_final_av_v33",
                 "mimo_reconcile_stemtext_final_av_v35", "mimo_reconcile_stemtext_final_av_v35_backend39"):
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
    assert result["summary"]["av_model_call_count"] == 6
    assert result["summary"]["text_model_call_count"] == 0
    assert result["summary"]["model_call_count"] == len(completions.requests) == 12
    assert result["summary"]["processed_clip_uids"] == ["clip-z", "clip-a", "clip-m"]

    visual, speech, profile, first = completions.requests[:4]
    assert [r["response_format"]["json_schema"]["name"] for r in completions.requests] == [
        "MimoVisualDraft", "MimoSpeechAVAssemblyDraft", "MimoSpeakerProfileDraft", "MimoAudioFinalizeDraft",
    ] * 3
    assert result["summary"]["audio_model_call_count"] == 3
    assert result["summary"]["visual_model_call_count"] == 3
    assert [m["role"] for m in visual["messages"]] == ["system", *(["user", "assistant"] * 4), "user"]
    assert sum(p["type"] == "image_url" for p in visual["messages"][-1]["content"]) == 1
    selected = next(r for r in stems if r.clip_uid == jobs[0].clip_uid and r.route == "music_first")
    for request in (speech, first):
        assert [m["role"] for m in request["messages"]] == ["system", "user"]
        media = request["messages"][-1]["content"]
        assert not any(p["type"] == "image_url" for p in media)
        assert sum(p["type"] == "video_url" for p in media) == 1
        assert request["extra_body"]["use_audio_in_video"] is True
        assert request["reasoning_effort"] == "none"
    assert visual["extra_body"]["use_audio_in_video"] is False
    expected_urls = [backend.config.media_resolver.resolve(Path(selected.stem(kind).canonical_stem_path))
                     for kind in ("music", "sfx")]
    actual_urls = [p["audio_url"]["url"] for p in first["messages"][-1]["content"] if p["type"] == "audio_url"]
    assert actual_urls == expected_urls
    assert len(actual_urls) == 2
    assert all(p["type"] != "video_url" for p in profile["messages"][-1]["content"])
    assert not any(p["type"] == "audio_url" for p in speech["messages"][-1]["content"])
    full = speech["messages"][-1]["content"][-1]["text"]
    contract = json.loads(full.split("AUTHORITATIVE INPUT:\n", 1)[1].split("\nTURN 1 VISUAL DRAFT:", 1)[0])
    assert contract["allowed_h3_reference_labels"] == ["<Picture 1>", "<Subject 1>"]
    assert contract["entity_subject_mapping"] == {"e1": "<Subject 1>"}
    assert "stem_facts" not in full and "separator_candidate" not in full
    schema = speech["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"summary", "audio_observation", "av_grounding", "shot1_caption", "warnings"}
    assert set(first["response_format"]["json_schema"]["schema"]["properties"]) == {"overall_soundscape", "non_diegetic_music"}
    assert not hasattr(backend, "reconcile_sound_descriptions")
    assert not hasattr(backend, "partition_sound_description")
    assert all(path.read_bytes() == data for path, data in source_before.items())

    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    assert all(row["final_h3"]["status"] == "ready" for row in data["clips"])
    assert (
        data["clips"][0]["reconcile"]["diagnostics"][1]["input_modality"]
        == "target_video_speech_assembly"
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
        payload["h3_semantics"]["shot1_caption"] = "(S1)<d>missing language</d>"
    elif failure == "articulation":
        payload["visual_observation"]["segment_views"][0]["entity_observations"][0][
            "speech_correlated_articulation"
        ] = "not_observed"
        payload["audio_observation"]["segment_decisions"][0]["audio_evidence_codes"] *= 2
        payload["h3_semantics"]["visual_retention_analysis"][0]["description"] = (
            "fully_preserved - the person remains visible."
        )
    elif failure == "schema":
        del payload["h3_semantics"]["non_diegetic_music"]
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
    expected_failed = 0 if failure == "articulation" else 1
    assert summary.failed_count == expected_failed and summary.ready_count == 3 - expected_failed
    assert summary.av_model_call_count == 6
    assert summary.model_call_count == len(completions.requests) == 12
    records = _records(shadow)
    assert records[0]["status"] == ("ready" if failure == "articulation" else "failed")
    assert records[0]["audio_finalize_raw_response"] == assembly_raw(first_raw)
    assert records[0]["music_stem_description"] is records[0]["sfx_stem_description"] is None
    if failure == "articulation":
        assert records[0]["failure_issues"] == []
        assert "stage_a_av_articulation_contradiction" in records[0]["diagnostics"][-1]["warnings"]
        annotation = records[0]["annotation"]
        assert annotation is not None
        assert annotation["av_grounding"] == payload["av_grounding"]
        assert annotation["audio_observation"]["segment_decisions"][0]["audio_evidence_codes"] == ["voice_continuity"]
        assert annotation["h3_semantics"]["visual_retention_analysis"][0]["description"] == "the person remains visible."
    qa.build_audio_shadow_qa(**kwargs)
    clip = json.loads((shadow / "qa/data.json").read_text())["clips"][0]
    assert (
        clip["direct_h3"]["shot1_caption"] == payload["h3_semantics"]["shot1_caption"]
    )
    assert clip["direct_h3"]["overall_soundscape"]
    if failure == "articulation":
        assert clip["direct_h3"] == records[0]["annotation"]["h3_semantics"]
    assert clip["final_h3"]["status"] == ("ready" if failure == "articulation" else "unavailable")
    page = (shadow / "qa/review.html").read_text()
    assert "Turn 2 speech AV raw response" in page
    assert "Turn 4 audio finalizer raw response" in page and "Final overall_soundscape" in page
    assert "Text-only sound partition" not in page


@pytest.mark.parametrize("field", ["subject_definitions", "visual_retention_analysis"])
@pytest.mark.parametrize("label", ["<Subject 1>", "Subject 1", "<Subject 99>"])
def test_repeated_subject_prose_parses_but_unknown_reference_stays_hard(
    tmp_path, monkeypatch, field, label,
):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    prose = f"{label} remains visible; {label} has a clear appearance."
    payload["h3_semantics"][field][0]["description"] = prose
    assert MimoAVAnnotationDraft.model_validate(payload).h3_semantics.model_dump()[field][0]["description"] == prose
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert row["annotation"]["h3_semantics"][field][0]["description"] == prose
    if label == "<Subject 99>":
        assert row["status"] == "failed"
        assert "draft_contains_unknown_reference" in {item["code"] for item in row["failure_issues"]}
    else:
        assert row["status"] == "ready"
    assert summary.model_call_count == len(completions.requests) == 4


@pytest.mark.parametrize("field", ["subject_definitions", "visual_retention_analysis"])
def test_reference_prose_still_requires_nonempty_description(field):
    payload = json.loads(_raw())
    payload["h3_semantics"][field][0]["description"] = " "
    with pytest.raises(ValueError, match="must not be empty"):
        MimoAVAnnotationDraft.model_validate(payload)
    payload = json.loads(_raw())
    payload["h3_semantics"]["subject_definitions"][0]["description"] = "From <Picture 1>."
    with pytest.raises(ValueError, match="cannot own Picture provenance"):
        MimoAVAnnotationDraft.model_validate(payload)


@pytest.mark.parametrize("caption,issue", [
    ("<Subject 1> (S1) speaks using <Audio 1>, <d>[Chinese] text</d>", "direct_unknown_reference"),
    ("A man (S2) says, <d>[Chinese] text</d>", "direct_unknown_speaker"),
    ("A man (S1) says, <d>text</d>", "direct_dialogue_language_missing"),
    ("A man (S1) says, <d>[Chinese] text", "direct_dialogue_format"),
])
def test_direct_format_hard_gates_keep_parseable_annotation(tmp_path, monkeypatch, caption, issue):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = caption
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert row["status"] == "failed" and row["annotation"] is not None
    assert issue in {item["code"] for item in row["failure_issues"]}
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == caption
    text_calls = int(issue == "direct_unknown_speaker")
    assert summary.text_model_call_count == text_calls
    assert summary.model_call_count == len(completions.requests) == 4 + text_calls


@pytest.mark.parametrize("has_transcribed,empty_inventory", [(False, True), (False, False), (True, False)])
def test_segment_inventory_is_review_only_without_transcribed_speech(
    tmp_path, monkeypatch, has_transcribed, empty_inventory,
):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    # Three mutually aligned internal stages, but not the authoritative inventory.
    for section, field in (
        ("audio_observation", "segment_decisions"),
        ("av_grounding", "segment_groundings"),
        ("visual_observation", "segment_views"),
    ):
        payload[section][field][0]["segment_id"] = "extra_segment"
    payload["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
    payload["h3_semantics"]["shot1_caption"] = "The person sits by a table."
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    if empty_inventory:
        values["segments"] = []
    elif not has_transcribed:
        for segment in values["segments"]:
            segment.update(asr_status="empty", asr_text=None, asr_language=None)
    job = _job(values)
    summary = _run(shadow, backend, stems, [job])
    row = _records(shadow)[0]
    inventory_codes = {"segment_inventory_mismatch", "visual_segment_inventory_mismatch"}
    if has_transcribed:
        assert row["status"] == "failed"
        assert inventory_codes <= {item["code"] for item in row["failure_issues"]}
    else:
        assert row["status"] == "ready"
        assert inventory_codes <= set(row["diagnostics"][-1]["warnings"])
        assert row["failure_issues"] == []
        source_samples = kwargs["audio_production_root"] / "h3/samples.jsonl"
        sample_values = json.loads(source_samples.read_text().splitlines()[0])
        sample_values["speech_segments"] = []
        sample = qa.FinalH3SampleV2.model_validate(sample_values)
        corrected, text, _ = qa._materialize_sample(
            sample, job,
            qa._MaterializerInput(
                MimoAVAnnotationDraft.model_validate(row["annotation"]),
                job.request_fingerprint,
            ),
        )
        assert corrected == []
        assert payload["visual_observation"]["visual_blocks"][0]["text"] in text
        assert "deterministic_correction:no_transcript_visual_caption_projection" in row["diagnostics"][-1]["warnings"]
        assert "overall_soundscape:\nA quiet room tone and a clink." in text
    assert row["annotation"]["audio_observation"]["segment_decisions"][0]["segment_id"] == "extra_segment"
    assert summary.model_call_count == len(completions.requests) == 3
    assert summary.audio_model_call_count == row["audio_model_call_count"] == 0
    assert row["speaker_profile_raw_response"] is None
    assert row["annotation"]["audio_observation"]["speaker_voice_profiles"] == []


def test_av_http_failure_is_single_attempt_and_next_clip_runs(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(
        tmp_path,
        shadow,
        [(_raw(), 8)] * 2,
    )
    original = completions.create

    def create(**request):
        if len(completions.requests) == 1:
            completions.requests.append(request)
            raise OSError("synthetic AV timeout")
        return original(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    assert summary.ready_count == 2 and summary.failed_count == 1
    assert summary.av_model_call_count == 5
    assert len(completions.requests) == 10
    assert summary.audio_model_call_count == 2
    failed = _records(shadow)[0]
    assert failed["raw_responses"] == [failed["visual_raw_response"]]
    assert failed["audio_finalize_raw_response"] is None


@pytest.mark.parametrize("stage", ["MimoVisualDraft", "MimoSpeechAVAssemblyDraft", "MimoSpeakerProfileDraft", "MimoAudioFinalizeDraft"])
@pytest.mark.parametrize("failure", ["http", "malformed"])
def test_failed_turn_remains_in_qa_and_next_clip_runs(tmp_path, monkeypatch, stage, failure):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)] * 3)
    original = completions.create
    failed_once = False

    def create(**request):
        nonlocal failed_once
        if not failed_once and request["response_format"]["json_schema"]["name"] == stage:
            failed_once = True
            completions.requests.append(request)
            completions.responses.pop(0)
            if failure == "http":
                raise OSError("synthetic transport failure")
            return _Completions([("not JSON", 0)]).create(**request)
        return original(**request)

    monkeypatch.setattr(completions, "create", create)
    summary = _run(shadow, backend, stems, jobs)
    visual_failed = stage == "MimoVisualDraft"
    remaining_turns = {"MimoVisualDraft": 3, "MimoSpeechAVAssemblyDraft": 2, "MimoSpeakerProfileDraft": 1, "MimoAudioFinalizeDraft": 0}[stage]
    assert summary.ready_count == 2 and summary.failed_count == 1
    assert summary.visual_model_call_count == 3
    assert summary.av_model_call_count == 6 - {"MimoVisualDraft": 2, "MimoSpeechAVAssemblyDraft": 1, "MimoSpeakerProfileDraft": 1, "MimoAudioFinalizeDraft": 0}[stage]
    assert summary.audio_model_call_count == (2 if remaining_turns >= 2 else 3)
    assert summary.model_call_count == len(completions.requests) == 12 - remaining_turns
    assert summary.text_model_call_count == 0
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    first = data["clips"][0]
    assert first["reconcile"]["status"] == "failed"
    assert first["final_h3"]["status"] == "unavailable"
    assert all(clip["final_h3"]["status"] == "ready" for clip in data["clips"][1:])
    if visual_failed:
        assert first["reconcile"]["audio_finalize_raw_response"] is None
    else:
        assert first["reconcile"]["visual_raw_response"]
    html = (shadow / "qa/review.html").read_text()
    assert "Turn 1 visual draft raw response" in html
    assert "Turn 2 speech AV raw response" in html
    assert "Turn 4 audio finalizer raw response" in html
    assert "visual_model_call_count" in html


@pytest.mark.parametrize("tamper", ["profile_raw_missing", "profile_modality_missing", "order", "audio_count"])
def test_stem_record_rejects_inconsistent_four_stage_history(tmp_path, monkeypatch, tamper):
    _, shadow = _fixture(tmp_path, monkeypatch)
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.model_call_count == len(completions.requests) == 4
    assert row["schema_version"] == "r2v.h3.mimo25_stem_reconcile.14"
    assert summary.schema_version == "r2v.h3.mimo25_stem_reconcile_summary.16"
    assert (row["visual_model_call_count"], row["av_model_call_count"], row["audio_model_call_count"], row["text_model_call_count"]) == (1, 2, 1, 0)
    assert row["raw_responses"] == [row[key] for key in (
        "visual_raw_response", "speech_av_raw_response", "speaker_profile_raw_response", "audio_finalize_raw_response",
    )]
    MimoStemReconcileRecord.model_validate(row)
    if tamper == "profile_raw_missing":
        row["speaker_profile_raw_response"] = None
    elif tamper == "profile_modality_missing":
        row["diagnostics"].pop(2)
    elif tamper == "order":
        row["raw_responses"][2:] = reversed(row["raw_responses"][2:])
    else:
        row["audio_model_call_count"] = 0
    with pytest.raises(ValueError, match="stem reconcile .*counts differ"):
        MimoStemReconcileRecord.model_validate(row)


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


@pytest.mark.parametrize("split_blocks", [False, True])
def test_multiple_same_speaker_segments_keep_merged_or_continuing_dialogue(
    tmp_path, monkeypatch, split_blocks,
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
    if split_blocks:
        caption += " ".join(
            f" <d>[Chinese] Model-owned continuation {i}.</d>" for i in range(2, 5)
        )
    payload["h3_semantics"]["shot1_caption"] = caption
    from r2v_data_v2.h3.mimo25_av_reconcile import _job

    values.pop("request_fingerprint")
    job = _job(values)
    backend.client.chat.completions.responses = [(json.dumps(payload), 8)]
    result = _reconcile(backend, job)
    assert result.model_call_count == 4
    assert result.annotation.h3_semantics.shot1_caption == caption
    assert len(result.annotation.segment_decisions) == 4


@pytest.mark.parametrize("speakers,caption,missing,unknown", [
    (["S1"], "A man says, <d>[Chinese] hello</d>", ["dialogue_1"], False),
    (["S1", "S1"], "A man (S1) says, <d>[Chinese] a</d> He continues, <d>[Chinese] b</d>", [], False),
    (["S1"] * 4, "<Subject 1> (S1) says, <d>[Chinese] a</d> He continues, <d>[Chinese] b</d> After a pause, he adds, <d>[Chinese] c</d> Finally, he concludes, <d>[Chinese] d</d>", [], False),
    (["S1", "S1"], "(S1) <d>[Chinese] a</d> <d>[Chinese] b</d>", [], False),
    (["S1", "S2"], "A man (S1) says, <d>[Chinese] a</d> A woman replies, <d>[Chinese] b</d>", ["dialogue_2"], False),
    (["S1", "S2"], "(S1) <d>[Chinese] a</d> <d>[Chinese] b</d>", ["dialogue_2"], False),
    (["S1", "S2"], "A man (S1) says, <d>[Chinese] a</d> A woman (S2) replies, <d>[Chinese] b</d>", [], False),
    (["S1"] * 4, "A man (S1) says, <d>[Chinese] merged same-speaker dialogue</d>", [], False),
    (["S1"], "A man (S1) says, <d>[Chinese] a</d> <d>[Chinese] b</d> <d>[Chinese] c</d>", ["dialogue_2", "dialogue_3"], False),
    (["S1"], "A man (S2) says, <d>[Chinese] a</d>", ["dialogue_1"], True),
    (["S1", "S1"], "<d>[Chinese] a</d> <d>[Chinese] b</d>", ["dialogue_1"], False),
    (["S1"], "(S1) <d>[Chinese] a</d> (S1) <d>[Chinese] b</d>", [], False),
])
def test_direct_dialogue_authoritative_continuity(speakers, caption, missing, unknown):
    unchanged, issues, warnings = protect_direct_dialogue(
        caption,
        [{"speaker_id": speaker} for speaker in speakers],
        allowed_labels={"<Subject 1>"},
    )
    single_speaker = len(set(speakers)) == 1
    assert unchanged == caption
    assert warnings == (["direct_single_speaker_marker_missing"] * len(missing) if single_speaker else [])
    assert [(issue.code, issue.field) for issue in issues] == (
        ([("direct_unknown_speaker", "shot1_caption")] if unknown else [])
        + ([("direct_dialogue_speaker_marker_missing", field) for field in missing] if not single_speaker else [])
    )


@pytest.mark.parametrize("speakers,caption,mismatched", [
    (["S1", "S2"], "(S1)<d>[English] a</d> (S1)<d>[English] b</d>", ["dialogue_2"]),
    (["S1", "S2"], "(S1)<d>[English] a</d> (S2)<d>[English] b</d>", []),
    (["S1", "S1", "S2"], "(S1)<d>[English] a</d> He continues <d>[English] b</d> (S1)<d>[English] c</d>", ["dialogue_3"]),
    (["S1", "S1"], "(S1)<d>[English] a</d> <d>[English] b</d>", []),
    (["S1", "S2"], "(S2)<d>[English] merged</d>", []),
    (["S1", "S2"], "(S1)<d>[English] a</d> (S1)<d>[English] b</d> (S1)<d>[English] c</d>", []),
    (["S1", "S2"], "(S2)<d>[English] a</d> (S2)<d>[English] b</d>", ["dialogue_1"]),
    (["S1", "S2"], "(S1)<d>[English] a</d> (S1) listens while (S2)<d>[English] b</d>", []),
])
def test_explicit_known_marker_must_match_only_with_equal_counts(speakers, caption, mismatched):
    unchanged, issues, warnings = protect_direct_dialogue(
        caption, [{"speaker_id": speaker} for speaker in speakers], allowed_labels=set(),
    )
    assert unchanged == caption and warnings == []
    assert [(issue.code, issue.field) for issue in issues] == [
        ("direct_dialogue_speaker_marker_mismatch", field) for field in mismatched
    ]
    assert all(
        issue.message == "explicit speaker marker disagrees with authoritative chronological speaker"
        for issue in issues
    )


@pytest.mark.parametrize("resolution,composition", [
    ("resolved", "single_speaker"),
    ("needs_acoustic_refinement", "single_speaker"),
    ("needs_acoustic_refinement", "uncertain"),
    ("uncertain", "uncertain"),
])
def test_visible_binding_without_supporting_evidence_is_retained(
    tmp_path, monkeypatch, resolution, composition,
):
    from r2v_data_v2.h3 import mimo25_backend

    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    observation = payload["visual_observation"]["segment_views"][0]["entity_observations"][0]
    observation.update(mouth_visibility="not_assessable", speech_correlated_articulation="not_assessable")
    audio = payload["audio_observation"]["segment_decisions"][0]
    audio.update(resolution=resolution, vocal_composition=composition)
    grounding = payload["av_grounding"]["segment_groundings"][0]
    grounding["evidence_codes"] = ["insufficient_evidence"]
    annotation = MimoAVAnnotationDraft.model_validate(payload)

    def forbidden_downgrade(*args, **kwargs):
        pytest.fail("evidence-sufficiency downgrade must not run")

    monkeypatch.setattr(mimo25_backend, "_conservative_visible_speaker_downgrade", forbidden_downgrade)
    normalized, corrections = _normalize_speaker_annotation(
        annotation, segment_ids=["segment_0001"],
        transcribed_segment_ids={"segment_0001"}, allowed_entity_ids={"e1"},
    )
    assert normalized.segment_decisions == annotation.segment_decisions
    assert "conservative_visible_speaker_downgrade" not in corrections
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.ready_count == 1 and summary.failed_count == 0
    assert row["annotation"]["av_grounding"]["segment_groundings"][0] == grounding
    assert row["failure_issues"] == []
    warnings = set(row["diagnostics"][-1]["warnings"])
    assert {
        "visible_entity_requires_confirmed_onscreen_speech",
        "onscreen_speech_requires_reliable_visible_speaker_evidence",
    } <= warnings
    if composition == "uncertain":
        assert "visible_entity_requires_resolved_audio" in warnings
    assert summary.model_call_count == len(completions.requests) == 4
    assert summary.audio_model_call_count == 1 and summary.av_model_call_count == 2

    sample = qa.FinalH3SampleV2.model_validate_json(
        (kwargs["audio_production_root"] / "h3/samples.jsonl").read_text().splitlines()[0],
    )
    source = jobs[0].segments[0]
    # Shadow ASR owns the in-memory speech input, as in the QA materializer path.
    sample = qa.FinalH3SampleV2.model_validate({
        **sample.model_dump(mode="python"),
        "speech_segments": [qa.FinalQwen3SpeechSegment(
            segment_id=source.segment_id, speaker_cluster_id=source.source_speaker_cluster_id,
            entity_id=source.current_entity_id, entity_occurrence_id=source.entity_occurrence_id,
            source_start_sample=source.source_start_sample, source_end_sample=source.source_end_sample,
            source_sample_rate_hz=source.source_sample_rate_hz,
            start_time=source.start_time, end_time=source.end_time,
            text=source.asr_text, language=source.asr_language,
        ).model_dump(mode="python")],
    })
    corrected, text, _ = qa._materialize_sample(
        sample, jobs[0], qa._MaterializerInput(
            MimoAVAnnotationDraft.model_validate(row["annotation"]), jobs[0].request_fingerprint,
        ),
    )
    assert corrected[0].entity_id == "e1"
    assert corrected[0].text == jobs[0].segments[0].asr_text
    assert "detailed_description:" in text


def test_incomplete_onscreen_grounding_is_review_only_without_rebinding(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    grounding = payload["av_grounding"]["segment_groundings"][0]
    grounding.update(binding_status="no_reliable_entity", entity_id=None)
    backend, _, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.ready_count == 1 and row["failure_issues"] == []
    assert "onscreen_grounding_incomplete" in row["diagnostics"][-1]["warnings"]
    assert row["annotation"]["av_grounding"]["segment_groundings"][0] == grounding


@pytest.mark.parametrize("contradiction,issue", [
    ("absent", "visible_entity_absent_from_visual_segment"),
    ("offscreen_audio", "visible_speaker_evidence_presentation_contradiction"),
    ("voice_over_context", "visible_speaker_evidence_presentation_contradiction"),
    ("device_playback_context", "visible_speaker_evidence_presentation_contradiction"),
])
def test_concrete_visible_speaker_contradictions_remain_hard(tmp_path, monkeypatch, contradiction, issue):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    view = payload["visual_observation"]["segment_views"][0]
    grounding = payload["av_grounding"]["segment_groundings"][0]
    if contradiction == "absent":
        view.update(visible_entity_ids=[], entity_observations=[])
        grounding["evidence_codes"] = ["insufficient_evidence"]
    else:
        grounding["evidence_codes"] = [contradiction]
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.failed_count == 1 and row["status"] == "failed"
    assert issue in {item["code"] for item in row["failure_issues"]}
    assert issue not in row["diagnostics"][-1]["warnings"]
    assert summary.model_call_count == len(completions.requests) == 4


def test_unknown_grounding_entity_still_cannot_publish(tmp_path, monkeypatch):
    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["av_grounding"]["segment_groundings"][0].update(
        entity_id="e99", evidence_codes=["insufficient_evidence"],
    )
    backend, _, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    grounding = row["annotation"]["av_grounding"]["segment_groundings"][0]
    assert grounding["entity_id"] is None and grounding["binding_status"] != "visible_entity"


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_confirmed_multi_speaker_cannot_publish_h3_identity(tmp_path, monkeypatch, composition):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["audio_observation"]["segment_decisions"][0].update(
        resolution="needs_acoustic_refinement", vocal_composition=composition,
        secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"},
    )
    backend, _, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    sample = qa.FinalH3SampleV2.model_validate_json(
        (kwargs["audio_production_root"] / "h3/samples.jsonl").read_text().splitlines()[0],
    )
    with pytest.raises(qa.MimoH3MaterializationContractError) as error:
        qa._materialize_sample(
            sample, jobs[0], qa._MaterializerInput(
                MimoAVAnnotationDraft.model_validate(row["annotation"]), jobs[0].request_fingerprint,
            ),
        )
    assert {issue.code for issue in error.value.issues} == {"multi_speaker_segment_requires_turn_refinement"}


@pytest.mark.parametrize("speakers,caption,hard_code", [
    (["S1"], "She says, <d>[Chinese] a</d>", None),
    (["S1"] * 3, "He says, <d>[Chinese] a</d> He continues, <d>[Chinese] b</d> He adds, <d>[Chinese] c</d>", None),
    (["S1"], "<d>[Chinese] a</d> <d>[Chinese] b</d>", None),
    (["S1", "S2"], "(S1)<d>[Chinese] a</d> Another voice <d>[Chinese] b</d>", "direct_dialogue_speaker_marker_missing"),
    (["S1"], "Another voice (S2)<d>[Chinese] a</d>", "direct_unknown_speaker"),
    (["S1", "S2"], "(S1)<d>[Chinese] a</d> (S1)<d>[Chinese] b</d>", "direct_dialogue_speaker_marker_mismatch"),
    ([], "<d>[Chinese] a</d>", "direct_dialogue_speaker_marker_missing"),
])
def test_marker_severity_uses_distinct_authoritative_speakers(
    tmp_path, monkeypatch, speakers, caption, hard_code,
):
    from r2v_data_v2.h3 import mimo25_backend

    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = caption
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)] * 3)
    monkeypatch.setattr(
        mimo25_backend, "direct_speech_facts",
        lambda annotation, segments: [{"speaker_id": speaker} for speaker in speakers],
    )
    summary = _run(shadow, backend, stems, jobs)
    row = _records(shadow)[0]
    assert row["annotation"]["h3_semantics"]["shot1_caption"] == caption
    assert row["audio_finalize_raw_response"] == assembly_raw(json.dumps(payload))
    assert row["annotation"]["av_grounding"] == payload["av_grounding"]
    text_calls = int(0 < caption.count("<d>") == len(speakers))
    assert summary.model_call_count == len(completions.requests) == (4 + text_calls) * len(jobs)
    assert row["model_call_count"] == 4 + text_calls
    assert row["text_model_call_count"] == text_calls
    assert row["audio_model_call_count"] == 1 and row["av_model_call_count"] == 2
    if hard_code:
        assert summary.failed_count == len(jobs) and row["status"] == "failed"
        assert hard_code in {issue["code"] for issue in row["failure_issues"]}
        assert hard_code not in row["diagnostics"][-1]["warnings"]
    else:
        assert summary.ready_count == len(jobs) and row["status"] == "ready"
        assert row["failure_issues"] == []
        assert "direct_single_speaker_marker_missing" in row["diagnostics"][3]["warnings"]
        qa.build_audio_shadow_qa(**kwargs)
        final = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]
        assert final["status"] == "ready"
        assert caption in final["text"]
        assert "direct_single_speaker_marker_missing" in final["variants"][0]["warnings"]


def test_explicit_marker_mismatch_remains_hard_in_backend(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import mimo25_backend

    _, shadow = _fixture(tmp_path, monkeypatch)
    payload = json.loads(_raw())
    payload["h3_semantics"]["shot1_caption"] = "(S1)<d>[English] a</d> (S1)<d>[English] b</d>"
    backend, completions, stems, jobs = _backend(tmp_path, shadow, [(json.dumps(payload), 8)])
    monkeypatch.setattr(
        mimo25_backend, "direct_speech_facts",
        lambda annotation, segments: [{"speaker_id": "S1"}, {"speaker_id": "S2"}],
    )
    summary = _run(shadow, backend, stems, jobs[:1])
    row = _records(shadow)[0]
    assert summary.failed_count == 1 and summary.ready_count == 0
    assert row["status"] == "failed" and row["annotation"] is not None
    assert [(issue["code"], issue["field"]) for issue in row["failure_issues"]] == [
        ("direct_dialogue_speaker_marker_mismatch", "dialogue_2"),
    ]
    assert "direct_dialogue_speaker_marker_mismatch" not in row["diagnostics"][-1]["warnings"]
    assert summary.model_call_count == len(completions.requests) == 5
    assert row["text_model_call_count"] == 1
    assert backend.provenance.schema_version == MIMO25_BACKEND_VERSION == "r2v.h3.mimo25_backend.61"
    assert backend.provenance.prompt_version == "h3_mimo25_speech_assembly_v46"


@pytest.mark.parametrize(
    "text",
    [
        "(S1) <d>[English] a <d>[English] b</d></d>",
        "(S1) <d>[English] a",
        "(S1) <d>no language</d>",
        "(S9) <d>[English] unknown</d>",
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
    assert unchanged == text and issues
    assert set(warnings) <= {"direct_single_speaker_marker_missing"}


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
