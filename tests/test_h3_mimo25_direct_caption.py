from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from r2v_data_v2.h3 import mimo25_backend as mb
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_ICL_VERSION,
    SYSTEM_PROMPT,
    VISUAL_SYSTEM_PROMPT,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    _official_detailed_description_icl_messages,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _backend,
    _job_fixture,
    _record_fixture,
    _sample,
    _validate,
)

LABELS = {"<Subject 1>", "<Picture 1>"}
SPEECH = [{"segment_id": "segment_1", "speaker_id": "S1", "language": "English", "text": "Exact, text!"}]


@pytest.mark.parametrize("groups,clusters,expected", [
    (["g1", "g1", "g1"], ["speaker_0", "speaker_1", "speaker_1"], ["S1", "S1", "S1"]),
    (["g1", "g2", "g1"], ["speaker_0", "speaker_1", "speaker_1"], ["S1", "S2", "S1"]),
    ([None, None], ["speaker_0", "speaker_1"], ["S1", "S2"]),
    ([None, "g1"], ["speaker_0", "speaker_1"], ["S1", "S2"]),
])
def test_final_group_sx_projection_ignores_resolution_and_binding(tmp_path, groups, clusters, expected):
    from r2v_data_v2.h3.mimo25_h3_materializer import (
        _audio_facts,
        _corrected_segments,
        _speaker_ids,
    )
    from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract

    job, sample, annotation = _job_fixture(tmp_path), _sample(tmp_path), _annotation()
    source, speech = job.segments[0], sample.speech_segments[0]
    decision = annotation.audio_observation.segment_decisions[0]
    grounding = annotation.segment_decisions[0]
    job.segments = []
    sample.speech_segments = []
    annotation.audio_observation.segment_decisions = []
    annotation.av_grounding.segment_groundings = []
    for i, (group, cluster) in enumerate(zip(groups, clusters, strict=True)):
        common = {"segment_id": f"segment_{i+1}", "start_time": float(i), "end_time": float(i+1),
                  "source_start_sample": i*32000, "source_end_sample": (i+1)*32000}
        job.segments.append(source.model_copy(update={**common, "source_speaker_cluster_id": cluster}))
        sample.speech_segments.append(speech.model_copy(update={**common, "speaker_cluster_id": cluster}))
        annotation.audio_observation.segment_decisions.append(mb.MimoUncertainAudioSegmentDecision.model_validate({
            **decision.model_dump(), "segment_id": common["segment_id"],
            "primary_speaker_group": group, "resolution": "uncertain",
        }))
        annotation.av_grounding.segment_groundings.append(grounding.model_copy(update={
            "segment_id": common["segment_id"], "primary_speaker_group": group,
            "binding_status": "no_reliable_entity", "entity_id": None, "speech_presentation": "uncertain",
        }))
    before = annotation.model_dump()
    facts = mb.direct_speech_facts(annotation, job.segments)
    assert [f["speaker_id"] for f in facts] == expected
    targets = mb._speaker_profile_targets(annotation, job)
    assert [(t["speaker_group"], t["speaker_id"]) for t in targets] == list(dict.fromkeys(
        (group, sx) for group, sx in zip(groups, expected, strict=True) if group is not None
    ))
    corrected, warnings = _corrected_segments(sample, job, SimpleNamespace(annotation=annotation))
    ids = _speaker_ids(corrected)
    assert [ids[s.speaker_cluster_id] for s in corrected] == expected
    assert all(s.entity_id is None for s in corrected)
    assert len(warnings) == len(groups)
    materialized_facts = _audio_facts(
        sample=sample, corrected=corrected, record=SimpleNamespace(annotation=annotation),
        contract=build_reference_contract(sample, "visual_only"),
    )
    assert [f.speaker_id for f in materialized_facts.speech] == expected
    assert all(f.locked_dialogue_block == "<d>[English] Exact, text!</d>" for f in materialized_facts.speech)
    assert annotation.model_dump() == before


@pytest.mark.parametrize("evidence,articulation,allowed", [
    (["av_temporal_alignment"], "not_observed", True),
    (["av_temporal_alignment"], "not_assessable", True),
    (["visible_lip_motion"], "observed", True),
    (["voice_continuity"], "not_observed", False),
    (["insufficient_evidence"], "not_observed", False),
    (["av_temporal_alignment", "no_visible_lip_motion"], "not_observed", False),
    (["av_temporal_alignment", "offscreen_audio"], "not_observed", False),
    (["av_temporal_alignment", "voice_over_context"], "not_observed", False),
    (["av_temporal_alignment", "device_playback_context"], "not_observed", False),
])
def test_sparse_av_binding_legality(evidence, articulation, allowed):
    annotation = _annotation()
    view = annotation.visual_observation.segment_views[0]
    view.entity_observations[0].speech_correlated_articulation = articulation
    grounding = annotation.segment_decisions[0]
    grounding.evidence_codes = evidence
    assert mb._visible_binding_is_permitted(grounding, view) == allowed
    codes = {issue.code for issue in _validate(annotation)}
    assert ("visible_entity_binding_not_permitted" not in codes) == allowed
    if allowed:
        assert "onscreen_speech_requires_reliable_visible_speaker_evidence" not in codes
        assert "visible_entity_requires_confirmed_onscreen_speech" not in codes
        normalized, count = mb._conservative_visible_speaker_downgrade(annotation, allowed_entity_ids={"e1"})
        assert count == 0 and normalized.model_dump() == annotation.model_dump()
    assert "visible_entity_binding_not_permitted" not in mb._REVIEW_ONLY_ISSUES


@pytest.mark.parametrize("cue", ["av_temporal_alignment", "voice_continuity"])
def test_occluded_binding_still_permitted(cue):
    annotation = _annotation()
    view = annotation.visual_observation.segment_views[0]
    observation = view.entity_observations[0]
    observation.mouth_visibility = "occluded"
    observation.speech_correlated_articulation = "not_assessable"
    annotation.segment_decisions[0].evidence_codes = ["speaker_visible_mouth_occluded", cue]
    assert mb._visible_binding_is_permitted(annotation.segment_decisions[0], view)
    assert not _validate(annotation)


def test_competing_articulation_and_exact_visibility_remain_exclusions():
    annotation = _annotation()
    view = annotation.visual_observation.segment_views[0]
    other = view.entity_observations[0].model_copy(update={"entity_id": "e2"})
    view.entity_observations[0].speech_correlated_articulation = "not_observed"
    view.entity_observations.append(other)
    view.visible_entity_ids.append("e2")
    annotation.segment_decisions[0].evidence_codes = ["av_temporal_alignment"]
    assert not mb._visible_binding_is_permitted(annotation.segment_decisions[0], view)
    assert "visible_entity_binding_not_permitted" in {i.code for i in _validate(annotation, allowed_entity_ids={"e1", "e2"})}
    view.visible_entity_ids = ["e2"]
    view.entity_observations = [other]
    assert "visible_entity_absent_from_visual_segment" in {i.code for i in _validate(annotation, allowed_entity_ids={"e1", "e2"})}


def test_temporal_cue_does_not_enable_strong_group_merging():
    annotation = _annotation()
    annotation.segment_decisions[0].evidence_codes = ["av_temporal_alignment"]
    annotation.visual_observation.segment_views[0].entity_observations[0].speech_correlated_articulation = "not_observed"
    payload = annotation.model_dump()
    for section, field in (("audio_observation", "segment_decisions"), ("av_grounding", "segment_groundings")):
        payload[section][field].append({**payload[section][field][0], "segment_id": "segment_2", "primary_speaker_group": "g2"})
    payload["visual_observation"]["segment_views"].append({
        **payload["visual_observation"]["segment_views"][0], "segment_id": "segment_2",
    })
    annotation = MimoAVAnnotationDraft.model_validate(payload)
    result, corrections, groups = mb._canonicalize_same_visible_entity_speaker_groups(
        annotation, segment_ids=["segment_1", "segment_2"], allowed_entity_ids={"e1"},
    )
    assert result.model_dump() == annotation.model_dump()
    assert corrections == {} and groups is None


@pytest.mark.parametrize("composition", ["overlapping_secondary_speech", "sequential_multi_speaker_speech"])
def test_sparse_cue_does_not_relax_multi_speaker_exclusion(composition):
    annotation = _annotation(composition=composition, resolution="needs_acoustic_refinement")
    annotation.segment_decisions[0].evidence_codes = ["av_temporal_alignment"]
    issues = _validate(annotation)
    assert "visible_entity_requires_resolved_audio" in {i.code for i in issues}
    assert not mb._audio_allows_visible_entity_resolution(annotation.audio_observation.segment_decisions[0])


@pytest.mark.parametrize("blocks,missing", [
    (["哪里不着急？"], []),
    (["我。", "哪里不着急？"], []),
    (["哪里不着急？", "我。"], ["segment_0003"]),
])
def test_exact_short_dialogue_inventory(blocks, missing):
    facts = [
        {"segment_id": "segment_0001", "speaker_id": "S1", "language": "Chinese", "text": "我。"},
        {"segment_id": "segment_0003", "speaker_id": "S2", "language": "Chinese", "text": "哪里不着急？"},
    ]
    caption = "A brief vocalization is heard. " + " ".join(f"<d>[Chinese] {text}</d>" for text in blocks)
    issues = mb._validate_direct_transcribed_dialogue(caption, facts)
    assert [(i.code, i.field) for i in issues] == [("direct_transcribed_dialogue_missing", sid) for sid in missing]


@pytest.mark.parametrize("text,count", [
    ("我。", 1), ("好。", 1), ("谢谢。", 2), ("不要走。", 3), ("那我先就过去帮忙了。", 9),
    ("yes", 1), ("thank you", 2), ("come with me", 3), ("", 0), ("  ...!? ", 0),
    ("ｔｈａｎｋ　ｙｏｕ", 2), ("cafe\u0301 noir", 2), ("我say谢谢123", 5), ("ありがとう", 5),
])
def test_lexical_units_and_missing_dialogue_policy(text, count):
    assert mb._lexical_unit_count(text) == count
    fact = {"segment_id": "segment_1", "language": "Chinese", "text": text}
    issues = mb._validate_direct_transcribed_dialogue("A quiet visual scene.", [fact])
    assert [i.code for i in issues] == (["direct_transcribed_dialogue_missing"] if count >= 3 else [])
    assert not mb._validate_direct_transcribed_dialogue(f"<d>[Chinese] {text}</d>", [fact])


def test_tiny_omission_keeps_substantive_order_and_other_hard_guards():
    facts = [{"segment_id": f"segment_{i}", "language": "Chinese", "text": text, "speaker_id": "S1"}
             for i, text in enumerate(("我。", "不要走。", "那我先就过去帮忙了。"))]
    text = "(S1)<d>[Chinese] 那我先就过去帮忙了。</d> (S1)<d>[Chinese] 不要走。</d>"
    assert [i.field for i in mb._validate_direct_transcribed_dialogue(text, facts)] == ["segment_2"]
    for caption, code in [("(S9)<d>[Chinese] 好。</d>", "direct_unknown_speaker"),
                          ("(S1)<d>好。</d>", "direct_dialogue_language_missing"),
                          ("(S1)<d>[Chinese] 好。", "direct_dialogue_format"),
                          ("<Picture 99>", "direct_unknown_reference")]:
        _, issues, _ = protect_direct_dialogue(caption, facts, allowed_labels=LABELS)
        assert code in {i.code for i in issues}


def test_duplicate_locked_dialogue_requires_distinct_ordered_occurrences():
    facts = [{**SPEECH[0], "text": "Exact spoken text!", "segment_id": f"segment_{i}"} for i in (1, 2)]
    block = "<d>[English] Exact spoken text!</d>"
    assert not mb._validate_direct_transcribed_dialogue(block + block, facts)
    assert [i.field for i in mb._validate_direct_transcribed_dialogue(block, facts)] == ["segment_2"]


@pytest.mark.parametrize("caption", ["A brief vocalization is heard.", "(S1)<d>[English] paraphrased</d>"])
def test_missing_exact_asr_is_hard_and_never_marker_polished(tmp_path, caption):
    visual, speech, profile, finalized = map(json.loads, split_annotation(_annotation().model_dump_json()))
    speech["shot1_caption"] = caption
    backend, calls = _backend(tmp_path, [(json.dumps(item), 8) for item in (visual, speech, profile, finalized)])
    job = _job_fixture(tmp_path)
    job.segments[0].asr_text = "Exact spoken text!"
    with pytest.raises(MimoBackendFailure) as exc:
        _run(backend, job)
    assert "direct_transcribed_dialogue_missing" in {i.code for i in exc.value.issues}
    assert len(calls.requests) == 4
    assert not exc.value.speaker_marker_polish.attempted


def test_sparse_binding_and_locked_dialogue_prompt_contract():
    for rule in (
        "Visible lip motion is strong positive evidence, not a mandatory prerequisite",
        "At 4 FPS", "UNKNOWN, not no_visible_lip_motion",
        "affirmative evidence", "voice_continuity alone is insufficient",
        "Every supplied transcribed segment is a locked speech fact",
        "one chronological <d>...</d> block per segment",
        "even if the text is extremely short",
    ):
        assert rule in SYSTEM_PROMPT
    assert "may share one natural <d> block" not in SYSTEM_PROMPT


@pytest.mark.parametrize("omit_short", [False, True])
def test_25755_short_asr_survives_without_lip_frames_or_extra_calls(tmp_path, omit_short):
    job = _job_fixture(tmp_path)
    template = job.segments[0]
    parts = [("segment_0001", "我。", 0.0125, 0.1925), ("segment_0003", "哪里不着急？", 0.3, 0.9)]
    job = job.model_copy(update={"segments": [
        template.model_copy(update={
            "segment_id": sid, "asr_language": "Chinese", "asr_text": words,
            "start_time": start, "end_time": end,
            "source_start_sample": round(start * template.source_sample_rate_hz),
            "source_end_sample": round(end * template.source_sample_rate_hz),
        }) for sid, words, start, end in parts
    ]})
    values = _annotation().model_dump()
    for section, key in (("visual_observation", "segment_views"), ("audio_observation", "segment_decisions"),
                         ("av_grounding", "segment_groundings")):
        base = values[section][key][0]
        values[section][key] = [{**base, "segment_id": sid} for sid, *_ in parts]
    for view in values["visual_observation"]["segment_views"]:
        view["entity_observations"] = []
    values["av_grounding"]["segment_groundings"][0]["evidence_codes"] = ["av_temporal_alignment"]
    # The later source is offscreen; no entity/group identity merge is implied.
    values["audio_observation"]["segment_decisions"][1]["primary_speaker_group"] = "g2"
    values["av_grounding"]["segment_groundings"][1].update(
        primary_speaker_group="g2", binding_status="offscreen", entity_id=None,
        speech_presentation="offscreen_spoken", evidence_codes=["offscreen_audio"],
    )
    values["audio_observation"]["speaker_voice_profiles"].append({
        "speaker_group": "g2", "voice_characteristics": "A clear low register.",
    })
    caption = "(S1)<d>[Chinese] 我。</d> " if not omit_short else "A brief vocalization is heard. "
    caption += "A voice (S2) asks, <d>[Chinese] 哪里不着急？</d>"
    values["h3_semantics"]["shot1_caption"] = caption
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in split_annotation(json.dumps(values))])
    kwargs = {
        "segment_ids": [p[0] for p in parts], "transcribed_segment_ids": [p[0] for p in parts],
        "allowed_entity_ids": {"e1"}, "allowed_reference_labels": LABELS,
        "auxiliary_audio_paths": {kind: Path(job.target_full_audio_path) for kind in ("speech", "music", "sfx")},
    }
    result = backend.reconcile(job, **kwargs)
    assert result.annotation.h3_semantics.shot1_caption == caption
    assert result.annotation.segment_decisions[0].entity_id == "e1"
    assert result.model_call_count == 4 and not result.speaker_marker_polish.attempted
    assert all("direct_transcribed_dialogue_missing" not in d.warnings for d in result.diagnostics)
    assert len(calls.requests) == 4
    assert backend.config.video_fps == 4.0


@pytest.mark.parametrize("invented", [False, True])
def test_no_transcript_caption_uses_visual_ownership(tmp_path, invented):
    job = _job_fixture(tmp_path)
    job = job.model_copy(update={"segments": [
        segment.model_copy(update={"asr_status": "empty", "asr_text": None})
        for segment in job.segments
    ]})
    visual, speech, _, finalized = map(json.loads, split_annotation(_annotation().model_dump_json()))
    visual["shot1_visual_description"] = "VISUAL_ONLY_SENTINEL"
    speech["shot1_caption"] = (
        "A voice (S1) says, <d>[English] invented dialogue</d>"
        if invented else visual["shot1_visual_description"]
    )
    speech["audio_observation"]["segment_decisions"][0]["delivery_style"] = None
    speech_raw = json.dumps(speech)
    backend, calls = _backend(tmp_path, [(json.dumps(visual), 0), (speech_raw, 8), (json.dumps(finalized), 8)])
    before = job.model_dump()
    result = backend.reconcile(
        job, segment_ids=["segment_1"], transcribed_segment_ids=[],
        allowed_entity_ids={"e1"}, allowed_reference_labels=LABELS,
        auxiliary_audio_paths={kind: Path(job.target_full_audio_path) for kind in ("speech", "music", "sfx")},
    )
    caption = result.annotation.h3_semantics.shot1_caption
    assert caption == visual["shot1_visual_description"]
    assert "(S1)" not in caption and "<d>" not in caption
    assert result.speech_av_raw_response == speech_raw
    assert result.annotation.av_grounding.model_dump() == speech["av_grounding"]
    assert result.annotation.h3_semantics.summary == speech["summary"]
    assert [d.model_dump() for d in result.annotation.audio_observation.segment_decisions] == speech["audio_observation"]["segment_decisions"]
    for key, value in finalized.items():
        assert getattr(result.annotation.h3_semantics, key) == value
    assert result.deterministic_correction_counts.get("no_transcript_visual_caption_projection", 0) == int(invented)
    assert result.model_call_count == len(calls.requests) == 3
    assert result.text_model_call_count == result.audio_model_call_count == 0
    assert result.speaker_profile_raw_response is None
    assert not result.speaker_marker_polish.attempted
    assert job.model_dump() == before


def test_unknown_sx_uses_marker_polish_without_changing_dialogue(tmp_path):
    visual, speech, profile, finalized = map(json.loads, split_annotation(_annotation().model_dump_json()))
    original = "(S2) says, <d>[English] Exact, text!</d>"
    candidate = "(S1) says, <d>[English] Exact, text!</d>"
    speech["shot1_caption"] = original
    responses = [json.dumps(item) for item in (visual, speech, profile, finalized)]
    responses.append(json.dumps({"shot1_caption": candidate, "needs_review": False}))
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in responses])
    result = _run(backend, _job_fixture(tmp_path))
    assert result.speaker_marker_polish.attempted and result.speaker_marker_polish.applied
    assert result.text_model_call_count == 1 and result.model_call_count == len(calls.requests) == 5
    assert result.annotation.h3_semantics.shot1_caption == candidate
    assert result.speech_av_raw_response == responses[1]
    assert re.findall(r"<d>.*?</d>", original) == re.findall(r"<d>.*?</d>", candidate)
    assert original.replace("(S2)", "") == candidate.replace("(S1)", "")
    _, issues, _ = protect_direct_dialogue(candidate, SPEECH, allowed_labels=LABELS)
    assert not issues
    payload = json.loads(calls.requests[-1]["messages"][-1]["content"])
    assert "direct_unknown_speaker" in payload["speaker_marker_projection_issues"]
    assert payload["dialogue_marker_targets"] == [{"dialogue_index": 1, "speaker_id": "S1"}]


def test_transcribed_job_with_missing_decision_does_not_use_visual_caption(tmp_path):
    visual, speech, _, finalized = map(json.loads, split_annotation(_annotation().model_dump_json()))
    visual["shot1_visual_description"] = "VISUAL_ONLY_SENTINEL"
    speech["audio_observation"]["segment_decisions"] = []
    responses = [json.dumps(item) for item in (visual, speech, finalized)]
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in responses])
    with pytest.raises(MimoBackendFailure) as caught:
        _run(backend, _job_fixture(tmp_path))
    assert "segment_inventory_mismatch" in {issue.code for issue in caught.value.issues}
    assert caught.value.annotation.h3_semantics.shot1_caption == speech["shot1_caption"]
    assert caught.value.model_call_count == len(calls.requests) == 3
    assert caught.value.text_model_call_count == 0


@pytest.mark.parametrize("failure", ["unaligned", "unrelated"])
def test_unknown_sx_still_fails_closed_outside_polish_scope(tmp_path, failure):
    visual, speech, profile, finalized = map(json.loads, split_annotation(_annotation().model_dump_json()))
    speech["shot1_caption"] = "(S2) says, <d>[English] Exact, text!</d>"
    if failure == "unaligned":
        speech["shot1_caption"] += " then <d>[English] extra</d>"
    else:
        speech["summary"] = "<Subject 99> stands in the room."
    responses = [json.dumps(item) for item in (visual, speech, profile, finalized)]
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in responses])
    with pytest.raises(MimoBackendFailure) as caught:
        _run(backend, _job_fixture(tmp_path))
    assert "direct_unknown_speaker" in {issue.code for issue in caught.value.issues}
    if failure == "unrelated":
        assert "direct_unknown_reference" in {issue.code for issue in caught.value.issues}
    assert not caught.value.speaker_marker_polish.attempted
    assert caught.value.text_model_call_count == 0
    assert caught.value.model_call_count == len(calls.requests) == 4


def _run(backend, job):
    return backend.reconcile(
        job, segment_ids=["segment_1"], transcribed_segment_ids=["segment_1"],
        allowed_entity_ids={"e1"}, allowed_reference_labels=LABELS,
        auxiliary_audio_paths={kind: Path(job.target_full_audio_path) for kind in ("speech", "music", "sfx")},
    )


def test_official_examples_preserved_in_two_turn_prefix(tmp_path):
    job = _job_fixture(tmp_path)
    visual, speech, profile, finalized = split_annotation(_annotation().model_dump_json())
    backend, calls = _backend(tmp_path, [(visual, 0), (speech, 8), (profile, 8), (finalized, 8)],
                             icl="official_ref2va_v1")
    result = _run(backend, job)
    defaults = MimoBackendConfig(media_resolver=backend.config.media_resolver, api_key="test")
    assert (defaults.thinking, defaults.icl, defaults.temperature) == (
        "disabled", "official_ref2va_v1", 0.0,
    )
    assert result.model_call_count == len(calls.requests) == 4
    messages = calls.requests[0]["messages"]
    assert [m["role"] for m in calls.requests[1]["messages"]] == ["system", "user"]
    assert not any(m in calls.requests[1]["messages"] for m in messages[1:-1])
    assert [m["role"] for m in messages] == ["system", *(["user", "assistant"] * 4), "user"]
    assert messages[1:-1] == _official_detailed_description_icl_messages()
    guide = (Path(__file__).parents[1] / "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md").read_text()
    example = guide.split("## 7. Complete Example", 1)[1].split("```text", 1)[1].split("```", 1)[0].strip()
    detailed = example.split("detailed_description:\n", 1)[1]
    opening, body = detailed.split("\n[Shot 1] ", 1)
    body = body.split("\n[Shot 2]", 1)[0]
    assert json.loads(messages[2]["content"]) == {
        "style_opening": opening, "shot1_caption": body,
        "overall_soundscape": example.split("\noverall_soundscape:\n", 1)[1].split("\n\nnon_diegetic_music:\n", 1)[0],
        "non_diegetic_music": example.split("\nnon_diegetic_music:\n", 1)[1],
    }
    assert "official H3 writing/field-boundary demonstration" in messages[1]["content"]
    assert "not the response-schema definition" in messages[1]["content"]
    for excluded in ("subject_definitions:", "summary:", "retention_analysis:",
                     "overall_soundscape:", "non_diegetic_music:", "[Shot 1]", "[Shot 2]", "[Shot 3]"):
        assert excluded not in messages[2]["content"]
    assert not re.search(r"\d{2}:\d{2}(?:\.\d+)?", messages[2]["content"])
    base = (Path(__file__).parents[1] / "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md").read_text()
    for number, assistant in zip((2, 3, 4), messages[4:9:2], strict=True):
        section = base.split(f"### Case {number}:", 1)[1].split("### Case ", 1)[0]
        source = section.split("```text\n", 1)[1].split("\n```", 1)[0]
        lines = source.splitlines()
        expected = {
            "shot1_caption": next(line.removeprefix("integrated_multimodal_description: [Shot 1] ") for line in lines if line.startswith("integrated_multimodal_description:")),
            "overall_soundscape": next(line.removeprefix("overall_soundscape: ") for line in lines if line.startswith("overall_soundscape:")),
            "non_diegetic_music": next(line.removeprefix("non_diegetic_music: ") for line in lines if line.startswith("non_diegetic_music:")),
        }
        assert json.loads(assistant["content"]) == expected
    examples = [json.loads(message["content"]) for message in messages[2:9:2]]
    assert any(item["non_diegetic_music"] == "N/A" for item in examples)
    positive = [item for item in examples if item["non_diegetic_music"] != "N/A"]
    assert len(positive) == 2
    assert all(item["non_diegetic_music"] not in item["shot1_caption"] for item in positive)
    assert "cello" in examples[1]["non_diegetic_music"] and "cello" not in examples[1]["shot1_caption"]
    assert "electronic pulse" in examples[3]["non_diegetic_music"] and "electronic pulse" not in examples[3]["shot1_caption"]
    for message in messages[1:-1]:
        assert not re.search(r"\[Shot \d+\]|\d{2}:\d{2}|\d+\.\d+[- ]second", message["content"])
        assert "For the target video, at" not in message["content"]
        assert "How the reference pictures align" not in message["content"]
        assert "Case 1: T2VA" not in message["content"]
    assert backend.provenance.icl_version == MIMO25_ICL_VERSION == "h3_official_ref2va_detailed_shot1_v4"
    assert backend.provenance.prompt_version == "h3_mimo25_speech_assembly_v46"
    assert backend.provenance.schema_version == "r2v.h3.mimo25_backend.61"
    assert backend.provenance.annotation_schema_version == "r2v.h3.mimo25_av_annotation.20"
    assert backend.provenance.materializer_version == "h3_mimo25_materializer_v26"
    assert "The pipeline owns [Shot 1]" in SYSTEM_PROMPT
    assert "Do not repeat the marker before every utterance" not in SYSTEM_PROMPT
    assert "style_opening is one concise global style/camera/lighting sentence" in VISUAL_SYSTEM_PROMPT


def test_direct_caption_materializes_without_reconstructing_prose(tmp_path):
    job = _job_fixture(tmp_path)
    sample = _sample(tmp_path)
    annotation = _annotation()
    record = _record_fixture(tmp_path, annotation, job=job)
    raw = annotation.h3_semantics.shot1_caption
    _, final, warnings = _materialize_sample(sample, job, record)
    direct = final.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    opening = annotation.h3_semantics.style_opening + " [Shot 1] "
    assert direct.startswith(opening)
    assert direct.count("[Shot 1]") == 1 and "[Shot 2]" not in direct
    direct = direct[len(opening):]
    assert re.sub(r"<d>.*?</d>", "<d></d>", direct) == re.sub(r"<d>.*?</d>", "<d></d>", raw)
    assert "<d>[English] Exact, text!</d>" in direct
    assert "asr_dialogue_payload_corrected" in warnings
    assert annotation.h3_semantics.shot1_caption == raw
    sections = ["subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape", "non_diegetic_music"]
    offsets = [final.index(name + ":\n") for name in sections]
    assert offsets == sorted(offsets)
    assert "[[" not in final
    assert "h3_projection" not in MimoAVAnnotationDraft.model_json_schema()["properties"]


@pytest.mark.parametrize("label", ["<Subject 9>", "<Audio 1>", "(S9)", "<Picture 0>"])
def test_unknown_labels_detected(label):
    text = f"{label} <Subject 1> (S1) asks, <d>[English] Exact, text!</d>"
    corrected, issues, _ = protect_direct_dialogue(text, SPEECH, allowed_labels=LABELS)
    assert issues and corrected == text


@pytest.mark.parametrize("text", [
    "Only a room.",
    "(S1) asks <d>[English] Exact, text!</d> then (S1) says <d>extra</d>",
])
def test_dialogue_count_cannot_be_guessed(text):
    corrected, issues, _ = protect_direct_dialogue(text, SPEECH, allowed_labels=LABELS)
    assert corrected == text
    assert "direct_dialogue_inventory_mismatch" in {issue.code for issue in issues}


def test_recognizable_reordered_dialogue_is_not_rewritten():
    speech = [*SPEECH, {**SPEECH[0], "segment_id": "segment_2", "text": "Second."}]
    text = "(S1) says <d>[English] Second.</d> then (S1) replies <d>[English] Exact, text!</d>"
    corrected, issues, _ = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert issues and corrected == text


def test_one_full_av_recheck_remains_the_limit(tmp_path):
    job = _job_fixture(tmp_path)
    valid = _annotation().model_dump_json()
    backend, calls = _backend(tmp_path, [("not JSON", 8), (valid, 8)])
    assert _run(backend, job).model_call_count == len(calls.requests) == 2
    backend, calls = _backend(tmp_path, [("not JSON", 8), ("still not JSON", 8)])
    with pytest.raises(MimoBackendFailure):
        _run(backend, job)
    assert len(calls.requests) == 2


def test_internal_absence_and_audio_wording_do_not_block_caption(tmp_path):
    job = _job_fixture(tmp_path)
    values = _annotation().model_dump(mode="json")
    values["audio_observation"]["audio_semantics"].update(
        overall_soundscape_status="absent", overall_soundscape=None, complete_silence_verified=False,
    )
    values["h3_semantics"]["overall_soundscape"] = "Dialogue and background music fill the room."
    annotation = MimoAVAnnotationDraft.model_validate(values)
    backend, calls = _backend(tmp_path, [(annotation.model_dump_json(), 8)])
    result = _run(backend, job)
    assert result.model_call_count == len(calls.requests) == 1
    assert "direct_soundscape_contains_music_or_speech" in result.diagnostics[0].warnings


def test_one_shot_schema_has_no_model_owned_timing():
    schema = MimoAVAnnotationDraft.model_json_schema()
    visual = schema["$defs"]["MimoVisualObservation"]
    assert set(visual["properties"]) == {"visual_blocks", "segment_views"}
    assert "MimoVisualShotObservation" not in schema["$defs"]
    assert "start_time" not in json.dumps(visual)
    semantics = schema["$defs"]["MimoH3Semantics"]["properties"]
    assert {"style_opening", "shot1_caption"} <= set(semantics)
    assert "detailed_description" not in semantics


@pytest.mark.parametrize("field,marker", [
    ("style_opening", "[Shot 1]"), ("shot1_caption", "[Shot 2]"),
])
def test_model_shot_markers_fail_and_use_only_existing_recheck(tmp_path, field, marker):
    job = _job_fixture(tmp_path)
    values = _annotation().model_dump(mode="json")
    values["h3_semantics"][field] += " " + marker
    with pytest.raises(ValidationError, match="pipeline-owned shot markers"):
        MimoAVAnnotationDraft.model_validate(values)
    backend, calls = _backend(tmp_path, [
        (json.dumps(values), 8), (_annotation().model_dump_json(), 8),
    ])
    assert _run(backend, job).model_call_count == len(calls.requests) == 2


@pytest.mark.parametrize("speakers,introductions,valid", [
    (["S1"] * 4, ["<Subject 1> (S1) speaks softly,", "He (S1) continues,", "He (S1) then adds,", "Finally, he (S1) concludes,"], True),
    (["S1"] * 4, ["The woman speaks,"] * 4, False),
    (["S1", "S1"], ["(S1) speaks,", "He continues,"], False),
    (["S1", "S1", "S2", "S2"], ["(S1) speaks,", "He (S1) adds,", "An off-screen voice (S2) replies,", "It (S2) continues,"], True),
    (["S1", "S2"], ["(S1) listens to (S2), then speaks,", "(S2) answers (S1) with a pause,"], True),
    (["S1", "S2"], ["(S1) speaks,", "A voice replies,"], False),
    (["S1", "S1"], ["He speaks,", "(S1) continues,"], False),
])
def test_each_dialogue_needs_its_speaker_in_event_lead_in(speakers, introductions, valid):
    speech = [
        {"segment_id": f"segment_{i}", "speaker_id": speaker,
         "language": "Chinese", "text": f"原文{i}"}
        for i, speaker in enumerate(speakers)
    ]
    text = " ".join(f"{intro} <d>[Chinese] 原文{i}</d>"
                    for i, intro in enumerate(introductions))
    corrected, issues, _ = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert corrected == text
    assert (not issues) == valid
    if not valid:
        assert {issue.code for issue in issues} == {"direct_dialogue_speaker_marker_missing"}


def test_long_natural_lead_in_preserved_with_exact_asr_payload():
    lead = ("After watching the other person for a moment, <Subject 1> (S1) "
            "turns away and answers quietly, ")
    tail = " while lowering his gaze."
    text = lead + "<d>[Chinese] wrong text</d>" + tail
    speech = [{**SPEECH[0], "language": "Chinese", "text": "EXACT ASR"}]
    corrected, issues, warnings = protect_direct_dialogue(text, speech, allowed_labels=LABELS)
    assert not issues
    assert corrected == lead + "<d>[Chinese] EXACT ASR</d>" + tail
    assert warnings == ["asr_dialogue_payload_corrected"]


def test_old_caption_excluded_and_single_authoritative_payload(tmp_path):
    job = _job_fixture(tmp_path)
    sentinel = "OLD_CAPTION_MUST_NOT_ANCHOR_RECAPTION"
    old_instruction = f"summary:\nOld scene\ndetailed_description:\n{sentinel}\noverall_soundscape:\nOld audio"
    job = job.model_copy(update={"r2v_instruction": old_instruction})
    backend, calls = _backend(tmp_path, [("bad JSON", 8), (_annotation().model_dump_json(), 8)],
                              transport="sglang", icl="official_ref2va_v1")
    result = _run(backend, job)
    assert result.model_call_count == len(calls.requests) == 2
    assert job.r2v_instruction == old_instruction
    for request in calls.requests:
        assert sentinel not in json.dumps(request["messages"])
        text = request["messages"][-1]["content"][-1]["text"]
        assert text.count('"subject_definition_requirements"') == 1
        assert text.count('"reference_image_mapping"') == 1
        assert text.count('"segments"') == 1
        assert "authoritative_speech_facts" not in text
        assert "MANDATORY MACHINE CONTRACT" not in text
        assert "r2v_instruction" not in text
    contract = backend.build_compact_task_contract(job)
    assert contract["segments"] == [segment.model_dump(mode="json") for segment in job.segments]
    assert calls.requests[1]["messages"][-1]["content"][-1]["text"].startswith(
        "Reinspect the same AV and fix ONLY these hard issues:"
    )
