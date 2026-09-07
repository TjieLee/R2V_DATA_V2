from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoVisualBlock,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3MaterializationContractError,
    _materialize_sample,
    prepare_compositor_input,
    sample_with_job_speech,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.mimo25_text_compositor import (
    AtomShot,
    CompositorInput,
    CompositorOrdering,
    DescriptionAtom,
    ShotOrdering,
    make_composition,
    split_visual_sentences,
    validate_ordering,
)
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _backend,
    _playback_inputs,
    _record_fixture,
    _sample,
    _validate,
)
from tests.test_h3_sam_audio_stem_shadow import _mimo_inventory_fixture, _run_facts


def _initial(source):
    return CompositorOrdering(shots=[
        ShotOrdering(shot_index=s.shot_index, atom_ids=s.initial_order) for s in source.shots
    ])


def test_visual_schema_has_no_generated_times():
    schema = MimoVisualBlock.model_json_schema()
    assert set(schema["properties"]) == {"block_id", "text"}
    assert set(schema["required"]) == {"block_id", "text"}
    for field in ("start_time", "end_time"):
        with pytest.raises(ValidationError):
            MimoVisualBlock.model_validate({"block_id": "v1", "text": "A person turns.", field: 0})
    for value in ({"block_id": "wrong", "text": "A person turns."}, {"block_id": "v1", "text": ""}):
        with pytest.raises(ValidationError):
            MimoVisualBlock.model_validate(value)
    old = _annotation().model_dump(mode="json")
    old["schema_version"] = "r2v.h3.mimo25_av_annotation.14"
    with pytest.raises(ValidationError):
        MimoAVAnnotationDraft.model_validate(old)


def test_exact_sentence_splitter():
    assert split_visual_sentences("A woman turns toward the group. Two people look back at her.") == [
        "A woman turns toward the group.", "Two people look back at her.",
    ]
    assert split_visual_sentences('Dr. Smith turns. He pauses at 2.5 meters. "Why?"') == [
        "Dr. Smith turns.", "He pauses at 2.5 meters.", '"Why?"',
    ]


def _order_fixture():
    atoms = [DescriptionAtom(
        atom_id=f"{prefix}{i}", kind=kind, source_id=f"{kind}{i}", text=f"{kind} {i}.",
        start_time=None if kind == "visual" else float(i),
        end_time=None if kind == "visual" else float(i + 1),
    ) for kind, prefix in (("visual", "V"), ("speech", "SPEECH_"), ("event", "EVENT_"))
        for i in (1, 2)]
    return CompositorInput(
        source_job_fingerprint="a"*64, source_annotation_fingerprint="b"*64,
        shots=[AtomShot(shot_index=1, atoms=atoms, initial_order=[a.atom_id for a in atoms]),
               AtomShot(shot_index=2, atoms=[DescriptionAtom(
                   atom_id="V3", kind="visual", source_id="v3", text="Still.", start_time=None, end_time=None,
               )], initial_order=["V3"])],
    )


@pytest.mark.parametrize("mutation,code", [
    ("missing", "compositor_atom_inventory_mismatch"),
    ("duplicate", "compositor_duplicate_atom"),
    ("unknown", "compositor_unknown_atom"),
    ("visual", "compositor_visual_order_mismatch"),
    ("speech", "compositor_speech_order_mismatch"),
    ("event", "compositor_event_order_mismatch"),
    ("cross", "compositor_shot_membership_mismatch"),
    ("shot", "compositor_shot_inventory_mismatch"),
])
def test_invalid_ordering_fails_closed(mutation, code):
    source = _order_fixture()
    values = _initial(source).model_dump(mode="json")
    ids = values["shots"][0]["atom_ids"]
    if mutation == "missing":
        ids.pop()
    elif mutation == "duplicate":
        ids.append(ids[0])
    elif mutation == "unknown":
        ids.append("UNKNOWN")
    elif mutation == "cross":
        ids.append(values["shots"][1]["atom_ids"].pop())
        values["shots"][1]["atom_ids"] = ["V1"]
        ids.remove("V1")
    elif mutation == "shot":
        values["shots"].reverse()
    else:
        first = {"visual": 0, "speech": 2, "event": 4}[mutation]
        ids[first], ids[first + 1] = ids[first + 1], ids[first]
    ordering = CompositorOrdering.model_validate(values)
    assert code in {i.code for i in validate_ordering(source, ordering)}
    with pytest.raises(ValueError):
        make_composition(source, ordering)


def test_compositor_cannot_output_prose():
    with pytest.raises(ValidationError):
        CompositorOrdering.model_validate({
            "shots": [{"shot_index": 1, "atom_ids": ["V1"], "text": "new dialogue"}],
        })


@pytest.mark.parametrize("two_shots", [False, True])
def test_locked_atoms_and_six_sections(tmp_path, two_shots):
    sample, job, record = _playback_inputs(tmp_path, two_shots=two_shots)
    source = prepare_compositor_input(sample, job, record)
    initial = _initial(source)
    composition = make_composition(source, initial)
    _, before, _ = _materialize_sample(
        sample, job, record, composition=composition, conditioning_variant="visual_only",
    )
    changed = initial.model_dump(mode="json")
    ids = changed["shots"][0]["atom_ids"]
    ids.remove("SPEECH_1")
    ids.insert(1, "SPEECH_1")
    _, after, _ = _materialize_sample(
        sample, job, record, composition=make_composition(source, CompositorOrdering.model_validate(changed)),
        conditioning_variant="visual_only",
    )
    assert before != after
    names = ["subject_definitions", "summary", "retention_analysis", "detailed_description",
             "overall_soundscape", "non_diegetic_music"]
    for text in (before, after):
        assert [text.index(name + ":") for name in names] == sorted(text.index(name + ":") for name in names)
        assert "[[" not in text and "SPEECH_" not in text
        assert text.count("<d>[English] Exact, text!</d>") == 1
    a, b = before.split("detailed_description:"), after.split("detailed_description:")
    assert a[0] == b[0]
    assert a[1].split("overall_soundscape:")[1] == b[1].split("overall_soundscape:")[1]
    assert after.index("Opening view.") < after.index("<d>") < after.index("A stable medium")
    clause = next(a.text for s in source.shots for a in s.atoms if a.kind == "speech")
    assert clause in after
    with pytest.raises(MimoH3MaterializationContractError, match="contract"):
        _materialize_sample(sample, job, record)


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_compositor_cannot_bypass_multi_speaker_gate(tmp_path, composition):
    sample, job, record = _playback_inputs(tmp_path)
    values = record.annotation.model_dump(mode="json")
    values["audio_observation"]["segment_decisions"] = _annotation(
        composition=composition, resolution="needs_acoustic_refinement",
    ).model_dump(mode="json")["audio_observation"]["segment_decisions"]
    record = _record_fixture(tmp_path, MimoAVAnnotationDraft.model_validate(values), job=job)
    with pytest.raises(MimoH3MaterializationContractError) as raised:
        prepare_compositor_input(sample, job, record)
    assert raised.value.issues[0].code == "multi_speaker_segment_requires_turn_refinement"


def test_compositor_request_is_only_text_regardless_of_av_experiments(tmp_path, monkeypatch):
    source = _order_fixture()
    raw = _initial(source).model_dump_json()
    backend, calls = _backend(tmp_path, [(raw, None)], transport="sglang", thinking="enabled", icl="v1")
    create = calls.create
    def with_reasoning(**kwargs):
        result = create(**kwargs)
        result.choices[0].message.reasoning_content = "DO_NOT_STORE"
        return result
    monkeypatch.setattr(calls, "create", with_reasoning)
    composition, content, diagnostic = backend.compose(source)
    assert content == raw
    assert composition.ordering == _initial(source)
    request = calls.requests[0]
    assert len(request["messages"]) == 2
    assert all(isinstance(m["content"], str) for m in request["messages"])
    assert request["temperature"] == 0.0 and request["max_completion_tokens"] == 4096
    assert request["reasoning_effort"] == "none"
    assert request["extra_body"] == {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    assert request["response_format"]["json_schema"]["name"] == "CompositorOrdering"
    assert not any(key in json.dumps(request) for key in ("image_url", "video_url", "audio_url", "use_audio_in_video"))
    assert diagnostic.input_modality == "text_only_compositor"
    assert "DO_NOT_STORE" not in content + composition.model_dump_json() + diagnostic.model_dump_json()


@pytest.mark.parametrize("mode,av_calls,comp_calls", [
    ("ready", 1, 1), ("recheck", 2, 1), ("av_failed", 2, 0), ("compositor_failed", 1, 1),
])
def test_stage5_two_pass_accounting_and_failure_persistence(tmp_path, mode, av_calls, comp_calls):
    stem, facts = _run_facts(tmp_path)
    jobs = build_stem_reconcile_jobs(
        base_inventory=_mimo_inventory_fixture(tmp_path / "base"),
        stem_diarization_root=stem.parent / "diarization", stem_asr_root=stem.parent / "asr",
    )
    job = jobs[0]
    folder = tmp_path / "sample"
    folder.mkdir()
    values = _sample(folder).model_dump(mode="python")
    values.update(sample_id=job.source_h3_sample_ids[0], clip_uid=job.clip_uid, target_video=job.target_video_path)
    values["visual_references"][0]["image_artifact_path"] = job.reference_images[0].image_artifact_path
    sample = FinalH3SampleV2.model_validate(values)
    annotation = MimoAVAnnotationDraft.model_validate_json(
        _annotation().model_dump_json().replace("segment_1", "segment_0001"),
    )
    source = prepare_compositor_input(sample_with_job_speech(sample, job), job, SimpleNamespace(
        annotation=annotation, request_fingerprint=job.request_fingerprint,
    ))
    good = annotation.model_dump_json()
    responses = [("bad AV", 8), ("still bad AV", 8)] if mode == "av_failed" else (
        ([("bad AV", 8)] if mode == "recheck" else []) + [(good, 8)]
        + [("bad compositor" if mode == "compositor_failed" else _initial(source).model_dump_json(), None)]
    )
    backend, calls = _backend(tmp_path, list(responses), transport="sglang")
    output = stem.parent / "mimo_reconcile"
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_facts=[facts], source_samples=[sample], backend=backend,
        output_root=output, route="music_first", allow_unverified=True,
    )
    record = MimoStemReconcileRecord.model_validate_json((output / "records.jsonl").read_text())
    assert summary.model_call_count == record.model_call_count == len(calls.requests) == av_calls + comp_calls
    assert record.av_model_call_count == summary.av_model_call_count == av_calls
    assert record.compositor_model_call_count == summary.compositor_model_call_count == comp_calls
    assert (record.annotation is None) == (mode == "av_failed")
    assert (record.composition is None) == ("failed" in mode)
    assert record.failure_stage == (
        "av_reconcile" if mode == "av_failed" else "text_compositor" if mode == "compositor_failed" else None
    )
    assert record.raw_responses == [raw for raw, _ in responses]
    if "failed" in mode:
        assert record.failure_issues
    assert MimoStemReconcileRecord.model_validate_json(record.model_dump_json()) == record
    changed = record.model_dump(mode="json")
    changed["raw_responses"] = ["tampered"]
    with pytest.raises(ValueError, match="fingerprint"):
        MimoStemReconcileRecord.model_validate(changed)
    if comp_calls:
        assert all(isinstance(m["content"], str) for m in calls.requests[-1]["messages"])


def test_authoritative_timed_inventory_stays_strict(tmp_path):
    _, _, record = _playback_inputs(tmp_path, two_shots=True)
    values = record.annotation.model_dump(mode="json")
    values["h3_projection"]["shots"][0]["timeline_parts"].append(
        values["h3_projection"]["shots"][1]["timeline_parts"].pop(0),
    )
    annotation = MimoAVAnnotationDraft.model_validate(values)
    issues = _validate(annotation, segment_intervals={"segment_1": (1, 2)}, target_duration_seconds=5)
    assert "audio_event_placeholder_wrong_shot" in {i.code for i in issues}
