from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.mimo25_av_reconcile import _job
from r2v_data_v2.h3.mimo25_backend import (
    MimoAVAnnotationDraft,
    MimoBackendFailure,
    MimoBackendProvenance,
    _speaker_profile_targets,
    direct_speech_facts,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    MimoStemReconcileRecord,
)
from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_audio_shadow_qa import _fixture
from tests.test_h3_mimo25_av_shadow import _Completions
from tests.test_h3_mimo25_raw_stems import _backend, _raw, _records, _run
from tools import run_h3_mimo25_stem_reconcile_shadow as cli


def _drafts(payload=None):
    payload = copy.deepcopy(payload or json.loads(_raw()))
    visual, speech, profiles, finalized = map(json.loads, split_annotation(json.dumps(payload)))
    visual["shot1_visual_description"] = "<Subject 1> raises a hand, turns toward the doorway, then lowers it."
    speech["shot1_caption"] = (
        "<Subject 1> raises a hand, turns toward the doorway and (S1) says, "
        "<d>[English] exact transcript</d> then lowers it."
    )
    finalized.update(overall_soundscape="Low room ambience and a distant clink.", non_diegetic_music="Gentle piano score.")
    return visual, {"speech_av": speech, **profiles, "audio_finalize": finalized}


def _setup(tmp_path, monkeypatch, *, visual=None, joint=None):
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend

    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    multi, multi_completions, stems, jobs = _backend(tmp_path, shadow, [(_raw(), 8)])
    multi.config = replace(multi.config, model="mimo-v2.6-flash-rl")
    default_visual, default_joint = _drafts()
    completions = _Completions([
        (json.dumps(default_visual if visual is None else visual), 0),
        (json.dumps(default_joint if joint is None else joint), 12),
    ])
    backend = TwoStepOpenAIMimo26Backend(
        multi.config, stem_records_by_clip=multi.stem_records_by_clip,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return kwargs, shadow, backend, completions, stems, jobs, multi, multi_completions


def _reconcile(backend, job, stems):
    return backend.reconcile(
        job, segment_ids=[s.segment_id for s in job.segments],
        transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
        allowed_entity_ids={"e1"}, allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_paths={s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems},
    )


def test_runner_two_step_is_opt_in_and_multi_default_unchanged():
    action = next(a for a in cli._parser()._actions if a.dest == "call_mode")
    assert action.default == "multi"
    assert "two_step" in action.choices
    with pytest.raises(ValueError, match="single-call only"):
        cli._parse_arguments([
            "--visual-production-root", "/v", "--visual-runs-root", "/r",
            "--audio-production-root", "/a", "--case-manifest", "/c",
            "--call-mode", "two_step", "--single-contract", "compact3",
        ])


def test_two_step_reuses_visual_request_and_common_annotation(tmp_path, monkeypatch):
    _, _, backend, completions, stems, jobs, multi, multi_completions = _setup(tmp_path, monkeypatch)
    result = _reconcile(backend, jobs[0], stems)
    _reconcile(multi, jobs[0], stems)
    assert completions.requests[0] == multi_completions.requests[0]
    assert len(completions.requests) == result.model_call_count == 2
    assert result.http_attempt_count == 2 and result.recheck_count == result.http_retry_count == 0
    assert not result.speaker_marker_polish.attempted
    assert result.speaker_profile_raw_response is result.audio_finalize_raw_response is None
    assert result.raw_responses == (result.visual_raw_response, result.speech_av_raw_response)
    annotation = result.annotation
    assert annotation.schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert annotation.visual_observation.visual_blocks[0].text == _drafts()[0]["shot1_visual_description"]
    assert "raises a hand" in annotation.h3_semantics.shot1_caption
    assert "then lowers it" in annotation.h3_semantics.shot1_caption
    assert "piano" not in annotation.h3_semantics.shot1_caption
    assert "clink" not in annotation.h3_semantics.shot1_caption
    assert annotation.speaker_voice_profiles[0].speaker_group == "g1"
    request = completions.requests[1]
    assert request["extra_body"]["use_audio_in_video"] is True
    content = request["messages"][-1]["content"]
    assert sum(item["type"] == "video_url" for item in content) == 1
    assert sum(item["type"] == "audio_url" for item in content) == 3
    assert not any(item["type"] == "image_url" for item in content)
    assert "TURN 1 VISUAL DRAFT" in content[-1]["text"]
    assert "speaker snippet:" not in str(content)


@pytest.mark.parametrize("profiles", [[], [{"speaker_group": "g2", "voice_characteristics": None}],
    [{"speaker_group": "g1", "voice_characteristics": None}] * 2])
def test_joint_profile_inventory_is_checked_before_normalization(tmp_path, monkeypatch, profiles):
    _, joint = _drafts()
    joint["speaker_voice_profiles"] = profiles
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert [i.code for i in caught.value.issues] == ["speaker_voice_profile_inventory_mismatch"]
    assert len(completions.requests) == caught.value.model_call_count == 2
    assert caught.value.speaker_profile_raw_response is caught.value.audio_finalize_raw_response is None


@pytest.mark.parametrize("binding", ["offscreen", "no_reliable_entity"])
def test_offscreen_or_unbound_profile_can_be_null(tmp_path, monkeypatch, binding):
    _, joint = _drafts()
    grounding = joint["speech_av"]["av_grounding"]["segment_groundings"][0]
    grounding.update(binding_status=binding, entity_id=None,
                     speech_presentation="offscreen_spoken" if binding == "offscreen" else "uncertain",
                     evidence_codes=["offscreen_audio" if binding == "offscreen" else "insufficient_evidence"])
    joint["speech_av"]["shot1_caption"] = (
        "<Subject 1> raises a hand. An offscreen voice (S1) says, <d>[English] exact transcript</d>."
        if binding == "offscreen" else "<Subject 1> raises a hand. A voice (S1) says, <d>[English] exact transcript</d>."
    )
    joint["speaker_voice_profiles"][0]["voice_characteristics"] = None
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert result.annotation.segment_decisions[0].binding_status == binding
    assert result.annotation.speaker_voice_profiles[0].voice_characteristics is None
    assert len(completions.requests) == 2


@pytest.mark.parametrize("nonlexical", [False, True])
def test_no_transcript_does_not_create_profiles_sx_or_dialogue(tmp_path, monkeypatch, nonlexical):
    visual, joint = _drafts()
    joint["speaker_voice_profiles"] = []
    if nonlexical:
        joint["speech_av"]["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
        joint["speech_av"]["av_grounding"]["segment_groundings"][0].update(
            binding_status="no_reliable_entity", entity_id=None, speech_presentation="uncertain",
            evidence_codes=["insufficient_evidence"],
        )
    else:
        visual["segment_views"] = []
        joint["speech_av"]["audio_observation"]["segment_decisions"] = []
        joint["speech_av"]["av_grounding"]["segment_groundings"] = []
    joint["speech_av"]["shot1_caption"] = visual["shot1_visual_description"]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    if nonlexical:
        values["segments"][0].update(asr_status="empty", asr_text=None, asr_language=None)
    else:
        values["segments"] = []
    job = _job(values)
    result = _reconcile(backend, job, stems)
    assert result.annotation.speaker_voice_profiles == []
    assert direct_speech_facts(result.annotation, list(job.segments)) == []
    assert result.annotation.h3_semantics.shot1_caption == visual["shot1_visual_description"]
    assert "<d>" not in result.annotation.h3_semantics.shot1_caption
    assert "(S1)" not in result.annotation.h3_semantics.shot1_caption
    assert result.annotation.h3_semantics.non_diegetic_music == "Gentle piano score."
    assert len(completions.requests) == 2


@pytest.mark.parametrize("clip_uid", ["02750d804022e9e8ba5f2964", "aed328a5a4eb2ff45d0d4d66"])
def test_no_segments_unscoped_asr_warning_keeps_raw_and_qa_audit(tmp_path, monkeypatch, clip_uid):
    visual, joint = _drafts()
    visual["segment_views"] = []
    joint["speech_av"]["audio_observation"]["segment_decisions"] = []
    joint["speech_av"]["av_grounding"]["segment_groundings"] = []
    joint["speaker_voice_profiles"] = []
    joint["speech_av"]["shot1_caption"] = visual["shot1_visual_description"]
    joint["speech_av"]["warnings"] = [{"code": "possible_asr_conflict", "segment_id": None}]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    job = _job({**jobs[0].model_dump(mode="json", exclude={"request_fingerprint"}), "clip_uid": clip_uid, "segments": []})
    result = _reconcile(backend, job, stems)
    assert result.annotation.warnings == []
    assert result.annotation.audio_observation.segment_decisions == []
    assert result.annotation.av_grounding.segment_groundings == []
    assert result.annotation.h3_semantics.shot1_caption == visual["shot1_visual_description"]
    assert result.annotation.h3_semantics.non_diegetic_music == joint["audio_finalize"]["non_diegetic_music"]
    assert result.speech_av_raw_response == json.dumps(joint)
    assert result.visual_raw_response == json.dumps(visual)
    assert result.deterministic_correction_counts == {"unscoped_possible_asr_conflict_no_segments": 1}
    assert "unscoped_possible_asr_conflict_no_segments" in result.diagnostics[-1].warnings
    assert "deterministic_correction_count:unscoped_possible_asr_conflict_no_segments=1" in result.diagnostics[-1].warnings
    assert len(completions.requests) == result.model_call_count == 2


def test_null_asr_warning_with_frozen_segments_still_fails(tmp_path, monkeypatch):
    _, joint = _drafts()
    joint["speech_av"]["warnings"] = [{"code": "possible_asr_conflict", "segment_id": None}]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert caught.value.reason == "MiMo joint AV/audio draft failed structured validation"
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert "unscoped_possible_asr_conflict_no_segments" not in caught.value.diagnostics[-1].warnings
    assert len(completions.requests) == 2


def _visual_source_job(rows):
    return SimpleNamespace(reference_subjects=[
        SimpleNamespace(subject_label=f"<Subject {index}>", kind=kind, source_picture_labels=[picture])
        for index, (kind, picture) in enumerate(rows, 1)
    ])


@pytest.mark.parametrize("descriptions,sources,expected", [
    (
        ["A young woman with shoulder-length dark hair and straight bangs, shown in <Picture 1>.",
         "A young man with short dark hair, shown in <Picture 3>.",
         "The upper-clothing attribute shown in <Picture 2>, a deep red long-sleeved button-up shirt with a collar, chest pockets, gathered cuffs, and a rope belt."],
        [("entity", "<Picture 1>"), ("entity", "<Picture 3>"), ("attribute", "<Picture 2>")],
        ["A young woman with shoulder-length dark hair and straight bangs.",
         "A young man with short dark hair.",
         "A deep red long-sleeved button-up shirt with a collar, chest pockets, gathered cuffs, and a rope belt."],
    ),
    (
        ["The face of the man in <Picture 2>, showing his eyes, nose, mouth, and skin texture."],
        [("attribute", "<Picture 2>")],
        ["The face of the man, showing his eyes, nose, mouth, and skin texture."],
    ),
], ids=["86ca3660f631858590d00852", "aa28e97595dd82cbdc96457e"])
def test_visual_source_provenance_is_removed_without_changing_visual_facts(descriptions, sources, expected):
    from pydantic import ValidationError

    from r2v_data_v2.h3.mimo25_backend import MimoVisualDraft
    from r2v_data_v2.h3.mimo26_two_step_backend import (
        _canonicalize_two_step_visual_payload,
    )

    visual, _ = _drafts()
    visual["subject_definitions"] = [
        {"subject_label": f"<Subject {index}>", "description": text}
        for index, text in enumerate(descriptions, 1)
    ]
    visual["shot1_visual_description"] += " <Picture 1> preserves the appearance of <Subject 1>."
    raw = json.dumps(visual)
    with pytest.raises(ValidationError, match="cannot own Picture provenance"):
        MimoVisualDraft.model_validate_json(raw)  # The shared/Multi schema stays strict.
    canonical, corrections = _canonicalize_two_step_visual_payload(raw, _visual_source_job(sources))
    parsed = MimoVisualDraft.model_validate_json(canonical)
    assert [d.description for d in parsed.subject_definitions] == expected
    values = json.loads(canonical)
    values["subject_definitions"] = visual["subject_definitions"]
    assert values == visual  # No caption, retention, label or inventory edit.
    assert json.loads(raw) == visual
    assert corrections == {"visual_subject_picture_provenance_removed": len(descriptions)}


@pytest.mark.parametrize("description,picture", [
    ("A man shown in <Picture 99>.", "<Picture 1>"),
    ("A man shown in <Picture 2>.", "<Picture 1>"),
    ("shown in <Picture 1>.", "<Picture 1>"),
    ("A man standing beside <Picture 1>.", "<Picture 1>"),
])
def test_visual_source_cleanup_does_not_hide_unknown_mismatched_or_empty_facts(description, picture):
    from pydantic import ValidationError

    from r2v_data_v2.h3.mimo25_backend import MimoVisualDraft
    from r2v_data_v2.h3.mimo26_two_step_backend import (
        _canonicalize_two_step_visual_payload,
    )

    visual, _ = _drafts()
    visual["subject_definitions"][0]["description"] = description
    canonical, _ = _canonicalize_two_step_visual_payload(json.dumps(visual), _visual_source_job([("entity", picture)]))
    with pytest.raises(ValidationError):
        MimoVisualDraft.model_validate_json(canonical)


def test_visual_source_cleanup_preserves_raw_and_records_correction_through_two_requests(tmp_path, monkeypatch):
    visual, joint = _drafts()
    visual["subject_definitions"][0]["description"] = "A young woman with dark hair, shown in <Picture 1>."
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert result.annotation.h3_semantics.subject_definitions[0].description == "A young woman with dark hair."
    assert result.annotation.h3_semantics.shot1_caption == joint["speech_av"]["shot1_caption"]
    assert result.visual_raw_response == json.dumps(visual)
    assert result.speech_av_raw_response == json.dumps(joint)
    assert result.deterministic_correction_counts == {"visual_subject_picture_provenance_removed": 1}
    assert "deterministic_correction_count:visual_subject_picture_provenance_removed=1" in result.diagnostics[-1].warnings
    assert len(completions.requests) == 2


_RANDOM20_FORMAT_CASES = json.loads((Path(__file__).parent / "fixtures/h3_two_step_random20_format_cases.json").read_text())


@pytest.mark.parametrize("case", _RANDOM20_FORMAT_CASES, ids=lambda case: case["clip_uid"])
def test_supplied_random20_response_payload_cpu_replay(tmp_path, monkeypatch, case):
    from r2v_data_v2.h3.mimo25_backend import MimoVisualDraft
    from r2v_data_v2.h3.mimo26_two_step_backend import (
        _canonicalize_two_step_visual_payload,
    )

    visual_raw = json.dumps(case["visual"])
    source_job = _visual_source_job([(s[0], s[2]) for s in case["subjects"]])
    canonical, corrections = _canonicalize_two_step_visual_payload(visual_raw, source_job)
    visual = MimoVisualDraft.model_validate_json(canonical)
    assert visual.shot1_visual_description == case["visual"]["shot1_visual_description"]
    assert [r.model_dump(mode="json") for r in visual.visual_retention_analysis] == case["visual"]["visual_retention_analysis"]
    if case["joint"] is None:
        expected = 3 if case["clip_uid"].startswith("86ca") else 1
        assert corrections == {"visual_subject_picture_provenance_removed": expected}
        return  # Visual-only replay: no cached Joint response exists.

    _, _, backend, completions, stems, jobs, _, _ = _setup(
        tmp_path, monkeypatch, visual=case["visual"], joint=case["joint"],
    )
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    values.update(clip_uid=case["clip_uid"], segments=[], binding_evidence_mode="none",
                  target_duration_seconds=case["target_duration_seconds"])
    subjects, images = [], []
    for index, source in enumerate(case["subjects"], 1):
        kind, entity, picture, *attribute = source
        number = int(picture.removeprefix("<Picture ").removesuffix(">"))
        subject = {**values["reference_subjects"][0], "subject_index": index, "subject_label": f"<Subject {index}>", "kind": kind,
                   "entity_id": entity if kind == "entity" else None, "source_picture_labels": [picture]}
        image = {**values["reference_images"][0], "image_index": number, "picture_label": picture,
                 "source_image_index": number, "source_image_id": f"image_{number}", "source_image_label": f"<Image {number}>",
                 "kind": "subject" if kind == "entity" else kind, "entity_id": subject["entity_id"]}
        if attribute:
            details = {"owner_entity_id": entity, "attribute_id": f"a{index}", "attribute_type": attribute[0]}
            subject.update(details)
            image.update(details)
        subjects.append(subject)
        images.append(image)
    values.update(reference_subjects=subjects, reference_images=sorted(images, key=lambda image: image["image_index"]))
    count = len(images)
    values["reference_selection"].update(original_picture_count=count, selected_picture_count=count,
                                         selected_source_image_indexes=list(range(1, count + 1)),
                                         selected_source_image_ids=[f"image_{i}" for i in range(1, count + 1)])
    job = _job(values)
    labels = {s.subject_label for s in job.reference_subjects} | {r.picture_label for r in job.reference_images}
    kwargs = {"segment_ids": [], "transcribed_segment_ids": [], "allowed_reference_labels": labels,
              "allowed_entity_ids": {s.entity_id for s in job.reference_subjects if s.kind == "entity"},
              "auxiliary_audio_paths": {s.stem_type: Path(s.canonical_stem_path) for s in stems[0].stems}}
    if case["clip_uid"].startswith("02750"):
        with pytest.raises(MimoBackendFailure, match="non-visual or pipeline syntax") as caught:
            backend.reconcile(job, **kwargs)
        result = caught.value
        assert "(S1)" in json.loads(result.visual_raw_response)["shot1_visual_description"]
    else:
        result = backend.reconcile(job, **kwargs)
        assert result.annotation.h3_semantics.shot1_caption == case["visual"]["shot1_visual_description"]
        assert result.annotation.h3_semantics.overall_soundscape == case["joint"]["audio_finalize"]["overall_soundscape"]
        assert result.deterministic_correction_counts == {"unscoped_possible_asr_conflict_no_segments": 1}
    assert "unscoped_possible_asr_conflict_no_segments" in result.diagnostics[-1].warnings
    assert result.visual_raw_response == visual_raw
    assert result.speech_av_raw_response == json.dumps(case["joint"])
    assert len(completions.requests) == result.model_call_count == 2


@pytest.mark.parametrize("warning", [
    {"code": "possible_asr_conflict"},
    {"code": "possible_asr_conflict", "segment_id": None, "detail": "unscoped"},
    {"code": "another_warning", "segment_id": None},
])
def test_empty_inventory_cleanup_leaves_other_malformed_warnings_for_schema(warning):
    from pydantic import ValidationError

    from r2v_data_v2.h3.mimo26_two_step_backend import (
        MimoJointAVAudioDraft,
        _canonicalize_joint_payload,
    )

    _, joint = _drafts()
    joint["speech_av"]["warnings"] = [warning]
    audit = []
    canonical, corrections = _canonicalize_joint_payload(json.dumps(joint), SimpleNamespace(segments=[]), warnings=audit)
    assert json.loads(canonical)["speech_av"]["warnings"] == [warning]
    assert corrections == {} and audit == []
    with pytest.raises(ValidationError):
        MimoJointAVAudioDraft.model_validate_json(canonical)


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_multiple_speaker_safety_is_not_relaxed(tmp_path, monkeypatch, composition):
    visual, joint = _drafts()
    audio = joint["speech_av"]["audio_observation"]["segment_decisions"][0]
    audio.update(vocal_composition=composition, resolution="needs_acoustic_refinement",
                 secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"})
    grounding = joint["speech_av"]["av_grounding"]["segment_groundings"][0]
    grounding.update(binding_status="no_reliable_entity", entity_id=None, speech_presentation="uncertain",
                     evidence_codes=["insufficient_evidence"])
    kwargs, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert "non_single_speaker" in speaker_ownership_reasons(result.annotation.audio_observation.segment_decisions[0])
    with pytest.raises(qa.MimoH3MaterializationContractError) as caught:
        _materialize(kwargs, jobs[0], result.annotation)
    assert "multi_speaker_segment_requires_turn_refinement" in {i.code for i in caught.value.issues}
    assert len(completions.requests) == 2


def test_joint_schema_and_prompt_reuse_without_stage_conflicts():
    from r2v_data_v2.h3.mimo26_two_step_backend import (
        JOINT_SYSTEM_PROMPT,
        MimoJointAVAudioDraft,
    )

    _, joint = _drafts()
    parsed = MimoJointAVAudioDraft.model_validate(joint)
    assert [t["speaker_group"] for t in _speaker_profile_targets(parsed.speech_av, SimpleNamespace(
        segments=[SimpleNamespace(segment_id="segment_0001", asr_status="transcribed", start_time=0.0,
                                  end_time=1.0, source_speaker_cluster_id="c1")], reference_subjects=[],
    ))] == ["g1"]
    schema = MimoJointAVAudioDraft.model_json_schema()
    assert schema["properties"]["speech_av"]["$ref"].endswith("/MimoSpeechAVAssemblyDraft")
    assert schema["properties"]["audio_finalize"]["$ref"].endswith("/MimoAudioFinalizeDraft")
    assert schema["properties"]["speaker_voice_profiles"]["items"]["$ref"].endswith("/MimoSpeakerVoiceProfile")
    assert "Defer ALL non-dialogue audio to Turn 4" not in JOINT_SYSTEM_PROMPT
    assert "localized speech snippets" not in JOINT_SYSTEM_PROMPT
    assert "full speech stem" in JOINT_SYSTEM_PROMPT
    assert "voice_characteristics=null" in JOINT_SYSTEM_PROMPT
    assert "Do not add non-dialogue audio" in JOINT_SYSTEM_PROMPT
    assert "Original AV remains primary authority" in JOINT_SYSTEM_PROMPT
    assert "return ONLY" not in JOINT_SYSTEM_PROMPT
    assert "Return ONLY" not in JOINT_SYSTEM_PROMPT
    assert "first actual vocal source -> S1" not in JOINT_SYSTEM_PROMPT
    assert "Non-transcribed-only groups consume no Sx" in JOINT_SYSTEM_PROMPT


@pytest.mark.parametrize("requirements", [
    (
        "VISUAL PERFORMANCE FIDELITY",
        "authoritative visual action sequence",
        "intermediate actions", "gestures", "gaze shifts", "expressions",
        "object interactions", "body and head movements", "camera behavior", "temporal ordering",
        "minimal local grammatical changes", "Do not replace a sequence of actions with a generic scene summary",
        "Do not invent unsupported events", "No minimum word count",
    ),
    (
        "VOICE-SOURCE NARRATION CONSISTENCY",
        "different final speaker groups", "distinct vocal source",
        'Never use "he continues", "she continues", "the same speaker"',
        "AV grounding explicitly supports that visible entity",
        "offscreen or unresolved vocal source separately", "neutral voice description",
        "do not guess gender or identity", "Do not change, merge or renumber gN/Sx",
        "exact ASR text, language markers, punctuation and dialogue order",
    ),
    (
        "SOURCE RESPONSIBILITIES",
        "Frozen Pictures", "stable appearance", "Reference images do not determine actions or speaker identity",
        "Turn 1 Visual", "DiariZen + Qwen3-ASR", "Original AV", "Speech stem", "Music/SFX stems",
        "Picture order, Subject numbering or prominence, or grammatical proximity",
        "Do not reclassify Pictures or reassign Subjects",
        "shot1_caption owns visual action, speaker presentation and exact dialogue",
        "summary stays a short target-video overview",
        "without transcribed ASR", "use the Turn 1 Visual caption", "do not invent (Sx) or <d>",
    ),
], ids=["visual_performance", "speaker_narration", "source_ownership"])
def test_joint_request_supplies_caption_fidelity_guidance(tmp_path, monkeypatch, requirements):
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    result = _reconcile(backend, jobs[0], stems)
    prompt = completions.requests[1]["messages"][0]["content"]
    for requirement in requirements:
        assert requirement in prompt
    for foreign_contract in ("<Video 1>", "attribute_transfer", "[reference generation]"):
        assert foreign_contract not in prompt
    assert result.annotation.h3_semantics.shot1_caption == _drafts()[1]["speech_av"]["shot1_caption"]
    assert result.model_call_count == len(completions.requests) == 2


def _materialize(kwargs, job, annotation):
    sample = qa.FinalH3SampleV2.model_validate_json(
        (kwargs["audio_production_root"] / "h3/samples.jsonl").read_text().splitlines()[0],
    )
    source = job.segments[0]
    sample = sample.model_copy(update={"speech_segments": [qa.FinalQwen3SpeechSegment(
        segment_id=source.segment_id, speaker_cluster_id=source.source_speaker_cluster_id,
        entity_id=source.current_entity_id, entity_occurrence_id=source.entity_occurrence_id,
        source_start_sample=source.source_start_sample, source_end_sample=source.source_end_sample,
        source_sample_rate_hz=source.source_sample_rate_hz, start_time=source.start_time, end_time=source.end_time,
        text=source.asr_text, language=source.asr_language,
    )]})
    return qa._materialize_sample(sample, job, qa._MaterializerInput(annotation, job.request_fingerprint))


def test_two_step_record_and_existing_h3_materializer(tmp_path, monkeypatch):
    kwargs, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    summary = _run(shadow, backend, stems, jobs[:1])
    assert summary.ready_count == 1 and summary.model_call_count == 2
    row = _records(shadow)[0]
    record = MimoStemReconcileRecord.model_validate(row)
    assert record.visual_model_call_count == record.av_model_call_count == 1
    assert record.audio_model_call_count == record.text_model_call_count == 0
    assert len(record.raw_responses) == len(record.diagnostics) == len(completions.requests) == 2
    assert MimoBackendProvenance.model_validate_json(backend.provenance.model_dump_json()) == backend.provenance
    corrected, rendered, _ = _materialize(kwargs, jobs[0], MimoAVAnnotationDraft.model_validate(row["annotation"]))
    assert corrected[0].entity_id == "e1"
    assert "<d>[English] exact transcript</d>" in rendered
    assert "Gentle piano score." in rendered
    assert "overall_soundscape:" in rendered
    assert (shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl").is_file()


@pytest.mark.parametrize("first_transcribed", [False, True])
def test_groups_profiles_and_sx_use_their_existing_chronological_inventories(tmp_path, monkeypatch, first_transcribed):
    visual, joint = _drafts()
    speech = joint["speech_av"]
    second_view = copy.deepcopy(visual["segment_views"][0])
    second_view["segment_id"] = "segment_0002"
    visual["segment_views"].append(second_view)
    second_audio = copy.deepcopy(speech["audio_observation"]["segment_decisions"][0])
    second_audio.update(segment_id="segment_0002", primary_speaker_group="g2")
    speech["audio_observation"]["segment_decisions"].append(second_audio)
    second_grounding = copy.deepcopy(speech["av_grounding"]["segment_groundings"][0])
    second_grounding.update(segment_id="segment_0002", primary_speaker_group="g2", binding_status="offscreen",
                            entity_id=None, speech_presentation="offscreen_spoken", evidence_codes=["offscreen_audio"])
    speech["av_grounding"]["segment_groundings"].append(second_grounding)
    joint["speaker_voice_profiles"].append({"speaker_group": "g2", "voice_characteristics": "Low register with a soft texture."})
    if first_transcribed:
        speech["shot1_caption"] += " An offscreen voice (S2) replies, <d>[Chinese] 好，我知道了。</d>."
    else:
        speech["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
        joint["speaker_voice_profiles"] = joint["speaker_voice_profiles"][1:]
        speech["shot1_caption"] = "<Subject 1> turns toward the doorway. An offscreen voice (S1) says, <d>[Chinese] 好，我知道了。</d>."
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    values = jobs[0].model_dump(mode="json", exclude={"request_fingerprint"})
    second = copy.deepcopy(values["segments"][0])
    start = second["end_time"] + 0.1
    second.update(segment_id="segment_0002", source_speaker_cluster_id="cluster_2", current_entity_id=None,
                  start_time=start, end_time=start + 0.5, source_start_sample=round(start * 32000),
                  source_end_sample=round((start + 0.5) * 32000), asr_text="好，我知道了。", asr_language="Chinese")
    values["segments"].append(second)
    if not first_transcribed:
        values["segments"][0].update(asr_status="empty", asr_text=None, asr_language=None)
    job = _job(values)
    result = _reconcile(backend, job, stems)
    facts = direct_speech_facts(result.annotation, list(job.segments))
    assert [f["speaker_id"] for f in facts] == (["S1", "S2"] if first_transcribed else ["S1"])
    assert [p.speaker_group for p in result.annotation.speaker_voice_profiles] == (["g1", "g2"] if first_transcribed else ["g2"])
    assert [d.primary_speaker_group for d in result.annotation.audio_observation.segment_decisions] == ["g1", "g2"]
    assert "<d>[Chinese] 好，我知道了。</d>" in result.annotation.h3_semantics.shot1_caption
    assert len(completions.requests) == 2


@pytest.mark.parametrize("stage", ["visual", "joint", "missing_marker"])
def test_failed_stages_keep_real_raws_without_retry_or_polish(tmp_path, monkeypatch, stage):
    visual, joint = _drafts()
    if stage == "visual":
        visual.pop("style_opening")
    elif stage == "joint":
        joint.pop("audio_finalize")
    else:
        joint["speech_av"]["shot1_caption"] = "Another voice (S2) says, <d>[English] exact transcript</d>."
    _, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    summary = _run(shadow, backend, stems, jobs[:1])
    row = MimoStemReconcileRecord.model_validate(_records(shadow)[0])
    expected = 1 if stage == "visual" else 2
    assert summary.failed_count == 1 and row.model_call_count == len(completions.requests) == expected
    assert len(row.raw_responses) == expected
    assert not row.speaker_marker_polish_attempted
    assert row.speaker_profile_raw_response is row.audio_finalize_raw_response is None


def test_eligible_marker_warning_does_not_trigger_third_polish_call(tmp_path, monkeypatch):
    _, joint = _drafts()
    joint["speech_av"]["shot1_caption"] = "<Subject 1> turns and says, <d>[English] exact transcript</d>."
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    result = _reconcile(backend, jobs[0], stems)
    assert len(completions.requests) == result.model_call_count == 2
    assert not result.speaker_marker_polish.attempted
    assert "direct_single_speaker_marker_missing" in result.diagnostics[-1].warnings


def test_empty_segment_input_rejects_invented_av_inventory(tmp_path, monkeypatch):
    visual, joint = _drafts()
    visual["segment_views"] = []
    joint["speaker_voice_profiles"] = []
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    job = _job({**jobs[0].model_dump(mode="json", exclude={"request_fingerprint"}), "segments": []})
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, job, stems)
    assert {i.code for i in caught.value.issues} == {"segment_inventory_mismatch"}
    assert len(completions.requests) == 2


def test_integer_temperature_provenance_matches_float_and_two_step(tmp_path):
    from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend

    config_values = {
        "api_key": "local-no-key",
        "transport": "sglang",
        "model": "mimo-v2.6-flash-rl",
        "base_url": "http://127.0.0.1:8094/v1",
        "media_resolver": MimoMediaResolver(
            mode="http", media_root=tmp_path,
            media_base_url="http://127.0.0.1:8766/",
        ),
    }
    integer_config = MimoBackendConfig(**config_values, temperature=0)
    float_config = MimoBackendConfig(**config_values, temperature=0.0)

    assert integer_config.provenance() == float_config.provenance()
    assert integer_config.provenance().temperature == 0.0
    integer_backend = TwoStepOpenAIMimo26Backend(
        integer_config, stem_records_by_clip={}, client=SimpleNamespace(),
    )
    float_backend = TwoStepOpenAIMimo26Backend(
        float_config, stem_records_by_clip={}, client=SimpleNamespace(),
    )
    assert integer_backend.provenance == float_backend.provenance
    assert integer_backend.provenance.schema_version == "r2v.h3.mimo25_backend.83"


def test_two_step_has_independent_provenance_without_changing_multi(tmp_path, monkeypatch):
    from r2v_data_v2.h3.audio_reuse_prepared import FrozenReuseBackendProvenance

    _, _, backend, _, _, _, multi, _ = _setup(tmp_path, monkeypatch)
    assert backend.provenance.schema_version == "r2v.h3.mimo25_backend.83"
    assert backend.provenance.prompt_version == "h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"
    assert multi.provenance.schema_version == "r2v.h3.mimo25_backend.66"
    assert multi.provenance.prompt_version == "h3_mimo25_speech_assembly_v49"
    FrozenReuseBackendProvenance.model_validate_json(backend.provenance.model_dump_json())
    values = backend.provenance.model_dump()
    values["prompt_version"] = multi.provenance.prompt_version
    with pytest.raises(ValueError, match="provenance differs"):
        MimoBackendProvenance.model_validate(values)


@pytest.mark.parametrize("version,prompt", [
    ("r2v.h3.mimo25_backend.80", "h3_mimo26_ra2va_two_step_joint_v1"),
    ("r2v.h3.mimo25_backend.81", "h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"),
    ("r2v.h3.mimo25_backend.82", "h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"),
])
def test_historical_two_step_provenance_remains_readable(tmp_path, monkeypatch, version, prompt):
    from r2v_data_v2.h3.audio_reuse_prepared import FrozenReuseBackendProvenance
    from r2v_data_v2.h3.mimo25_backend import _compact_json, _sha256_text

    _, _, backend, _, _, _, _, _ = _setup(tmp_path, monkeypatch)
    values = backend.provenance.model_dump(mode="json", exclude={"configuration_fingerprint"})
    values.update(schema_version=version, prompt_version=prompt)
    values["configuration_fingerprint"] = _sha256_text(_compact_json(values))
    assert MimoBackendProvenance.model_validate(values).model_dump(mode="json") == values
    assert FrozenReuseBackendProvenance.model_validate(values).model_dump(mode="json") == values


def test_two_step_flows_through_existing_audio_reuse_and_h3_cpu_path(tmp_path):
    from r2v_data_v2.h3 import audio_reuse_materializer as product
    from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
    from r2v_data_v2.h3.audio_reuse_prepared import load_reconcile_sources
    from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
    from tests.test_h3_audio_reuse_materializer import _case

    args, sample, _ = _case(tmp_path / "source", music="Quiet piano score.")
    visual, speech, profiles, finalized = map(json.loads, split_annotation(args["annotation"].model_dump_json()))
    completions = _Completions([
        (json.dumps(visual), 0), (json.dumps({"speech_av": speech, **profiles, "audio_finalize": finalized}), 12),
    ])
    backend = TwoStepOpenAIMimo26Backend(
        MimoBackendConfig(api_key="fake", transport="sglang", model="mimo-v2.6-flash-rl",
                          media_resolver=MimoMediaResolver(mode="http", media_root=tmp_path, media_base_url="http://media.invalid")),
        stem_records_by_clip={args["job"].clip_uid: args["stem_record"]},
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    root = tmp_path / "joint-result"
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=[args["job"]], stem_records=[args["stem_record"]], backend=backend,
        output_root=root, route=args["stem_record"].route, allow_unverified=True,
    )
    assert summary.ready_count == 1 and summary.model_call_count == 2
    record = load_reconcile_sources(root)[0]
    values = {**args, "annotation": record.annotation, "output_root": tmp_path / "joint-reuse"}
    manifest = build_audio_reuse_assets(**values)
    assert len(manifest.speakers) == 1 and manifest.music is not None
    source = product.load_reuse_source(manifest_path=values["output_root"] / "manifest.json", job=args["job"],
                                       record=record, stem_record=args["stem_record"])
    refs, _ = product.select_reuse_audio(sample, source, {source.job.clip_uid: source})
    assert [r.contract.kind for r in refs] == ["speaker_speech_reuse", "music_reuse"]
    _, rendered, _ = qa._materialize_sample(sample, args["job"], record, conditioning_variant="target_speech_reuse",
                                           reuse_audio_contracts=[r.contract for r in refs])
    assert "<Audio 1>" in rendered and "<Audio 2>" in rendered
    assert "<d>[English] Exact, text!</d>" in rendered


def _segment_drafts(groups, texts, caption):
    visual, joint = _drafts()
    speech = joint["speech_av"]
    original_view = visual["segment_views"][0]
    original_audio = speech["audio_observation"]["segment_decisions"][0]
    original_grounding = speech["av_grounding"]["segment_groundings"][0]
    visual["segment_views"] = []
    speech["audio_observation"]["segment_decisions"] = []
    speech["av_grounding"]["segment_groundings"] = []
    for index, (group, text) in enumerate(zip(groups, texts, strict=True), start=1):
        segment_id = f"segment_{index:04d}"
        visual["segment_views"].append({**copy.deepcopy(original_view), "segment_id": segment_id})
        speech["audio_observation"]["segment_decisions"].append({
            **copy.deepcopy(original_audio), "segment_id": segment_id,
            "primary_speaker_group": group, "delivery_style": "calm" if text is not None else None,
        })
        speech["av_grounding"]["segment_groundings"].append({
            **copy.deepcopy(original_grounding), "segment_id": segment_id, "primary_speaker_group": group,
            "binding_status": "no_reliable_entity", "entity_id": None,
            "speech_presentation": "uncertain", "evidence_codes": ["insufficient_evidence"],
        })
    speech["shot1_caption"] = caption
    return visual, joint


def _segment_job(job, texts):
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    original = values["segments"][0]
    values["segments"] = []
    for index, text in enumerate(texts, start=1):
        start = (index - 1) * 0.3
        values["segments"].append({
            **copy.deepcopy(original), "segment_id": f"segment_{index:04d}",
            "start_time": start, "end_time": start + 0.2,
            "source_start_sample": round(start * 32000), "source_end_sample": round((start + 0.2) * 32000),
            "asr_status": "transcribed" if text is not None else "empty",
            "asr_text": text, "asr_language": "Chinese" if text is not None else None,
        })
    return _job(values)


@pytest.mark.parametrize("extra_traits", [None, "Soft low-register vocal texture."], ids=["null", "non_null"])
def test_25755_non_transcribed_profile_is_excluded_without_losing_audio_facts(tmp_path, monkeypatch, extra_traits):
    texts = ["你好，老师。", None, "请坐下来。", None]
    caption = "A woman says, <d>[Chinese] 你好，老师。</d> then continues, <d>[Chinese] 请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g2", "g1", "g2"], texts, caption)
    joint["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "Clear middle register."},
        {"speaker_group": "g2", "voice_characteristics": extra_traits},
    ]
    _, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    job = _segment_job(jobs[0], texts)
    summary = _run(shadow, backend, stems, [job])
    record = MimoStemReconcileRecord.model_validate(_records(shadow)[0])
    assert summary.ready_count == 1 and len(completions.requests) == record.model_call_count == 2, record.failure_reason
    assert [p.model_dump(mode="json") for p in record.annotation.speaker_voice_profiles] == [joint["speaker_voice_profiles"][0]]
    assert record.annotation.h3_semantics.shot1_caption == caption
    assert [d.model_dump(mode="json") for d in record.annotation.audio_observation.segment_decisions] == (
        joint["speech_av"]["audio_observation"]["segment_decisions"]
    )
    assert [d.model_dump(mode="json") for d in record.annotation.av_grounding.segment_groundings] == (
        joint["speech_av"]["av_grounding"]["segment_groundings"]
    )
    assert [fact["speaker_id"] for fact in direct_speech_facts(record.annotation, job.segments)] == ["S1", "S1"]
    assert [fact["text"] for fact in direct_speech_facts(record.annotation, job.segments)] == [texts[0], texts[2]]
    assert "direct_single_speaker_marker_missing" in record.diagnostics[-1].warnings
    correction = "joint_non_transcribed_null_profile_removed" if extra_traits is None else "joint_non_transcribed_non_null_profile_excluded"
    assert f"deterministic_correction_count:{correction}=1" in record.diagnostics[-1].warnings
    profile_warnings = [w for w in record.diagnostics[-1].warnings if w.startswith("non_transcribed_only_voice_profile_excluded:")]
    assert profile_warnings == ([] if extra_traits is None else [
        ("non_transcribed_only_voice_profile_excluded:g2:acoustic traits preserved in speech_av_raw_response; "
         "not published as an identity-related voice profile"),
    ])
    assert record.speech_av_raw_response == json.dumps(joint)
    assert json.loads(record.speech_av_raw_response)["speaker_voice_profiles"][-1] == joint["speaker_voice_profiles"][-1]
    text = completions.requests[1]["messages"][-1]["content"][-1]["text"]
    contract = json.loads(text.split("AUTHORITATIVE INPUT:\n", 1)[1].split("\nTURN 1 VISUAL DRAFT:", 1)[0])
    assert contract["required_segment_ids_in_order"] == ["segment_0001", "segment_0002", "segment_0003", "segment_0004"]
    assert contract["required_dialogue_blocks_in_order"] == [
        "<d>[Chinese] 你好，老师。</d>", "<d>[Chinese] 请坐下来。</d>",
    ]


def test_195457_profiles_reorder_by_transcribed_appearance_without_reassigning_traits(tmp_path, monkeypatch):
    texts = [None, "你好，老师。", "请坐下来。"]
    caption = "A voice (S1) says, <d>[Chinese] 你好，老师。</d>. She continues (S2), <d>[Chinese] 请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g2", "g1"], texts, caption)
    joint["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "Low register."},
        {"speaker_group": "g2", "voice_characteristics": "Bright middle register."},
    ]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    job = _segment_job(jobs[0], texts)
    result = _reconcile(backend, job, stems)
    assert [(p.speaker_group, p.voice_characteristics) for p in result.annotation.speaker_voice_profiles] == [
        ("g2", "Bright middle register."), ("g1", "Low register."),
    ]
    assert [d.primary_speaker_group for d in result.annotation.audio_observation.segment_decisions] == ["g1", "g2", "g1"]
    assert [d.primary_speaker_group for d in result.annotation.av_grounding.segment_groundings] == ["g1", "g2", "g1"]
    assert [fact["speaker_id"] for fact in direct_speech_facts(result.annotation, job.segments)] == ["S1", "S2"]
    assert result.annotation.h3_semantics.shot1_caption == caption
    assert result.deterministic_correction_counts == {"joint_voice_profile_order_normalization": 2}
    assert result.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == 2


@pytest.mark.parametrize("duplicate_audio", [False, True], ids=["4e0506", "2b9292"])
def test_exact_chinese_dialogue_gets_only_missing_language_and_identical_audio_dedup(tmp_path, monkeypatch, duplicate_audio):
    texts = ["你好，老师。", "请坐下来。"]
    caption = "A woman (S1) says, <d>你好，老师。</d> and then continues, <d>请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g1"], texts, caption)
    if duplicate_audio:
        rows = joint["speech_av"]["audio_observation"]["segment_decisions"]
        rows.insert(1, copy.deepcopy(rows[0]))
        assert rows[0] == rows[1]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    result = _reconcile(backend, _segment_job(jobs[0], texts), stems)
    assert result.annotation.h3_semantics.shot1_caption == (
        "A woman (S1) says, <d>[Chinese] 你好，老师。</d> and then continues, <d>[Chinese] 请坐下来。</d>."
    )
    assert [d.segment_id for d in result.annotation.audio_observation.segment_decisions] == ["segment_0001", "segment_0002"]
    assert result.deterministic_correction_counts["joint_dialogue_language_marker_restored"] == 2
    assert result.deterministic_correction_counts.get("joint_audio_segment_duplicate_removed", 0) == int(duplicate_audio)
    assert result.speech_av_raw_response == json.dumps(joint)
    if duplicate_audio:
        raw_rows = json.loads(result.speech_av_raw_response)["speech_av"]["audio_observation"]["segment_decisions"]
        assert len(raw_rows) == 3 and raw_rows[0] == raw_rows[1]
    assert len(completions.requests) == 2
    text = completions.requests[1]["messages"][-1]["content"][-1]["text"]
    contract = json.loads(text.split("AUTHORITATIVE INPUT:\n", 1)[1].split("\nTURN 1 VISUAL DRAFT:", 1)[0])
    assert contract["required_segment_ids_in_order"] == ["segment_0001", "segment_0002"]
    assert contract["required_dialogue_blocks_in_order"] == [
        "<d>[Chinese] 你好，老师。</d>", "<d>[Chinese] 请坐下来。</d>",
    ]


@pytest.mark.parametrize("profiles", [
    [{"speaker_group": "g1", "voice_characteristics": None}, {"speaker_group": "g2", "voice_characteristics": "Low register."},
     {"speaker_group": "g2", "voice_characteristics": "Low register."}],
    [{"speaker_group": "g1", "voice_characteristics": None}, {"speaker_group": "g3", "voice_characteristics": None}],
    [{"speaker_group": "g1", "voice_characteristics": None}, {"speaker_group": "g3", "voice_characteristics": "Low register."}],
    [{"speaker_group": "g2", "voice_characteristics": None}],
    [{"speaker_group": "g1", "voice_characteristics": None}] * 2,
])
def test_profile_normalization_does_not_hide_unsafe_inventory(tmp_path, monkeypatch, profiles):
    texts = ["你好，老师。", None]
    visual, joint = _segment_drafts(["g1", "g2"], texts, "A voice (S1) says, <d>[Chinese] 你好，老师。</d>.")
    joint["speaker_voice_profiles"] = profiles
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, _segment_job(jobs[0], texts), stems)
    assert "speaker_voice_profile_inventory_mismatch" in {i.code for i in caught.value.issues}
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == 2


@pytest.mark.parametrize("blocks", [
    ["你好，老师。", "请你坐下。"], ["你好，老师。"],
    ["你好，老师。", "你好，老师。", "请坐下来。"], ["请坐下来。", "你好，老师。"],
    ["[English] 你好，老师。", "请坐下来。"], ["你好，老师。 ", "请坐下来。"],
    ["[Chinese] 你好，老师。", "[Chinese] 你好，老师。", "[Chinese] 请坐下来。"],
    ["[Chinese] 请坐下来。", "[Chinese] 你好，老师。"],
])
def test_language_normalization_never_repairs_changed_or_inconsistent_dialogue(tmp_path, monkeypatch, blocks):
    texts = ["你好，老师。", "请坐下来。"]
    caption = "A voice (S1) says, " + " and continues, ".join(f"<d>{text}</d>" for text in blocks)
    visual, joint = _segment_drafts(["g1", "g1"], texts, caption)
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, _segment_job(jobs[0], texts), stems)
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert not any("joint_dialogue_language_marker_restored" in warning for d in caught.value.diagnostics for warning in d.warnings)
    assert len(completions.requests) == 2


def test_language_normalization_keeps_correct_blocks_and_surrounding_prose_verbatim(tmp_path, monkeypatch):
    texts = ["你好，老师。", "请坐下来。"]
    caption = "A woman (S1) says, <d>[Chinese] 你好，老师。</d>; she raises a hand.\nThen: <d>请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g1"], texts, caption)
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    result = _reconcile(backend, _segment_job(jobs[0], texts), stems)
    assert result.annotation.h3_semantics.shot1_caption == (
        "A woman (S1) says, <d>[Chinese] 你好，老师。</d>; she raises a hand.\nThen: <d>[Chinese] 请坐下来。</d>."
    )
    assert result.deterministic_correction_counts["joint_dialogue_language_marker_restored"] == 1
    assert len(completions.requests) == 2


@pytest.mark.parametrize("difference", ["speaker_group", "evidence_duplicate"])
def test_conflicting_audio_duplicate_is_not_collapsed(tmp_path, monkeypatch, difference):
    visual, joint = _drafts()
    rows = joint["speech_av"]["audio_observation"]["segment_decisions"]
    duplicate = copy.deepcopy(rows[0])
    if difference == "speaker_group":
        duplicate["primary_speaker_group"] = "g2"
    else:
        duplicate["audio_evidence_codes"] *= 2
    rows.append(duplicate)
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert "segment_inventory_mismatch" in {i.code for i in caught.value.issues}
    assert "conflicting" in str(caught.value)
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == 2


@pytest.mark.parametrize("inventory", ["audio", "grounding"])
def test_joint_inventories_remain_exact_even_without_transcripts(tmp_path, monkeypatch, inventory):
    visual, joint = _segment_drafts(["g1", "g1"], [None, None], "A woman turns toward the doorway.")
    joint["speaker_voice_profiles"] = []
    speech = joint["speech_av"]
    rows = speech["audio_observation"]["segment_decisions"] if inventory == "audio" else speech["av_grounding"]["segment_groundings"]
    rows.reverse()
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, _segment_job(jobs[0], [None, None]), stems)
    assert "segment_inventory_mismatch" in {i.code for i in caught.value.issues}
    assert len(completions.requests) == 2


def test_same_visible_entity_two_groups_is_reported_not_merged(tmp_path, monkeypatch):
    texts = ["你好，老师。", "请坐下来。"]
    caption = "<Subject 1> (S1) says, <d>[Chinese] 你好，老师。</d>. She (S2) adds, <d>[Chinese] 请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g2"], texts, caption)
    template = _drafts()[1]["speech_av"]["av_grounding"]["segment_groundings"][0]
    for row in joint["speech_av"]["av_grounding"]["segment_groundings"]:
        row.update({key: template[key] for key in ("binding_status", "entity_id", "speech_presentation", "evidence_codes")})
    joint["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "Middle register."},
        {"speaker_group": "g2", "voice_characteristics": "Low register."},
    ]
    _, shadow, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    summary = _run(shadow, backend, stems, [_segment_job(jobs[0], texts)])
    record = MimoStemReconcileRecord.model_validate(_records(shadow)[0])
    assert summary.failed_count == 1
    assert "visible_entity_speaker_group_contradiction" in {i.code for i in record.failure_issues}
    assert [d.primary_speaker_group for d in record.annotation.audio_observation.segment_decisions] == ["g1", "g2"]
    assert record.annotation.h3_semantics.shot1_caption == caption
    assert record.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == record.model_call_count == 2


def test_final_annotation_schema_error_keeps_structured_failure_issues(tmp_path, monkeypatch):
    _, joint = _drafts()
    joint["speech_av"]["summary"] = ""
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert caught.value.code == "mimo_structured_output_failed"
    assert "schema_validation" in {i.code for i in caught.value.issues}
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == 2


def test_identity_failure_also_keeps_missing_speaker_marker_diagnostics(tmp_path, monkeypatch):
    texts = ["你好，老师。", "请坐下来。"]
    caption = "<Subject 1> says, <d>[Chinese] 你好，老师。</d>. She adds, <d>[Chinese] 请坐下来。</d>."
    visual, joint = _segment_drafts(["g1", "g2"], texts, caption)
    template = _drafts()[1]["speech_av"]["av_grounding"]["segment_groundings"][0]
    for row in joint["speech_av"]["av_grounding"]["segment_groundings"]:
        row.update({key: template[key] for key in ("binding_status", "entity_id", "speech_presentation", "evidence_codes")})
    joint["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": None},
        {"speaker_group": "g2", "voice_characteristics": None},
    ]
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, _segment_job(jobs[0], texts), stems)
    assert {"visible_entity_speaker_group_contradiction", "direct_dialogue_speaker_marker_missing"}.issubset(
        {i.code for i in caught.value.issues},
    )
    assert caught.value.annotation.h3_semantics.shot1_caption == caption
    assert len(completions.requests) == 2


def test_visual_correction_audit_survives_conflicting_joint_audio_rows(tmp_path, monkeypatch):
    visual, joint = _drafts()
    visual["segment_views"].append(copy.deepcopy(visual["segment_views"][0]))
    rows = joint["speech_av"]["audio_observation"]["segment_decisions"]
    rows.append({**copy.deepcopy(rows[0]), "primary_speaker_group": "g2"})
    _, _, backend, completions, stems, jobs, _, _ = _setup(tmp_path, monkeypatch, visual=visual, joint=joint)
    with pytest.raises(MimoBackendFailure) as caught:
        _reconcile(backend, jobs[0], stems)
    assert "deterministic_correction_count:visual_segment_row_merge=1" in caught.value.diagnostics[-1].warnings
    assert caught.value.visual_raw_response == json.dumps(visual)
    assert caught.value.speech_av_raw_response == json.dumps(joint)
    assert len(completions.requests) == 2
