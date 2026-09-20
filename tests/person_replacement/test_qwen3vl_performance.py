"""Qwen two-person source call with compact non-temporal performance anchors."""

import pytest

from r2v_data_v2.person_replacement.qwen3vl import (
    TWO_SOURCE_PROMPT,
    TWO_SOURCE_WITH_PERFORMANCE_PROMPT,
)

LABELS = ("SOURCE_SUBJECT_1","SOURCE_SUBJECT_2","SOURCE_PERFORMANCE_1",
          "SOURCE_PERFORMANCE_2","SHOT_DESCRIPTION")
RESPONSE = (
    "SOURCE_SUBJECT_1: a man in a blue jacket on the left\n"
    "SOURCE_SUBJECT_2: a woman with short hair holding a black clipboard\n"
    "SOURCE_PERFORMANCE_1: holding and handling a hockey stick\n"
    "SOURCE_PERFORMANCE_2: holding a black clipboard with both hands\n"
    "SHOT_DESCRIPTION: At 00:02.000 <Subject 1> lifts the stick, then <Subject 2> writes."
)


def qwen(text):
    from r2v_data_v2.person_replacement.qwen3vl import LocalQwen

    instance = LocalQwen.__new__(LocalQwen)
    instance._text = lambda content: text
    return instance


def test_pair_source_call_parses_five_labelled_fields(tmp_path):
    video = tmp_path/"v.mp4"
    video.touch()
    values = qwen(RESPONSE).describe_two_with_performance(video)
    assert values == ("a man in a blue jacket on the left",
                      "a woman with short hair holding a black clipboard",
                      "holding and handling a hockey stick",
                      "holding a black clipboard with both hands",
                      "At 00:02.000 <Subject 1> lifts the stick, then <Subject 2> writes.")


def test_old_describe_two_signature_and_prompt_are_untouched():
    from r2v_data_v2.person_replacement.qwen3vl import LocalQwen

    assert "SHOT_DESCRIPTION" in TWO_SOURCE_PROMPT
    assert "Return exactly three nonempty single-line labelled fields" in TWO_SOURCE_PROMPT
    assert "SOURCE_PERFORMANCE_1" not in TWO_SOURCE_PROMPT
    assert hasattr(LocalQwen,"describe_two")


def test_performance_prompt_requests_non_temporal_object_anchors():
    prompt = TWO_SOURCE_WITH_PERFORMANCE_PROMPT
    for label in LABELS:
        assert label in prompt
    assert "Return exactly five nonempty single-line labelled fields" in prompt
    assert "then, before, after, initially or later" in prompt
    assert "holding a black clipboard with both hands" in prompt
    assert "throughout" in prompt  # explicitly forbids false whole-shot claims
    assert "output exactly NONE and nothing else" in prompt  # no-object sentinel
    for banned in ("standing","sitting","standing still","walking","facial expression","gaze",
                   "emotion","mouth state","clothing","uniform","hair","background","camera"):
        assert banned in prompt  # present only inside the explicit prohibition list
    assert "You may add one simple" not in prompt  # general-state allowance removed
    assert "object-only non-temporal anchors" in prompt
    assert "SHOT_DESCRIPTION still keeps the complete temporal sequence" in prompt


def test_performance_prompt_disambiguates_props_against_shot_description():
    prompt = TWO_SOURCE_WITH_PERFORMANCE_PROMPT
    # Subjects must not carry temporary props, performance fields must carry them,
    # and the same prop may legitimately appear in both anchor and timed sequence.
    assert "must not contain temporary props or temporary actions" in prompt
    assert "non-temporal record of the visible held, carried, touched" in prompt
    assert "SHOT_DESCRIPTION still keeps the complete temporal sequence" in prompt
    assert "not a contradiction" in prompt
    assert "temporary actions and held/touched props in SHOT_DESCRIPTION" in TWO_SOURCE_PROMPT
    assert "not in conflict" in prompt


@pytest.mark.parametrize("broken", [
    RESPONSE.replace("SOURCE_PERFORMANCE_2:","BAD_LABEL:"),
    "\n".join(RESPONSE.splitlines()[:4]),
    RESPONSE.replace("holding a black clipboard with both hands",""),
])
def test_malformed_responses_fail_closed(tmp_path, broken):
    video = tmp_path/"v.mp4"
    video.touch()
    with pytest.raises(ValueError):
        qwen(broken).describe_two_with_performance(video)


def detailed(prompt):
    return prompt.split("detailed_description:")[1]


def test_performance_anchors_are_object_only_and_target_centric():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    shot = "UNIQUE_SHOT_MARKER_1234"
    prompt = build_prompt("a man","a woman",shot,"an old man","a young woman",
                          performance1="holding and handling a hockey stick",
                          performance2="holding a black clipboard with both hands")
    section = detailed(prompt)
    assert "<Subject 3> must preserve this source object interaction: " \
           "holding and handling a hockey stick." in section
    assert "<Subject 4> must preserve this source object interaction: " \
           "holding a black clipboard with both hands." in section
    assert "Any source object held, carried, touched" in section
    assert shot not in prompt  # SHOT_DESCRIPTION stays provenance only


@pytest.mark.parametrize("value", ["holding a clipboard.","holding a clipboard..",
                                   "holding a clipboard...","holding a clipboard"])
def test_dynamic_anchor_never_doubles_sentence_punctuation(value):
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    section = detailed(build_prompt("a man","a woman","shot","an old man","a young woman",
                                    performance1=value))
    assert "holding a clipboard." in section
    assert "clipboard.." not in section
    assert ".." not in section


def test_none_anchor_produces_no_target_sentence_and_no_sentinel():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    section = detailed(build_prompt("a man","a woman","shot","an old man","a young woman",
                                    performance1="NONE",performance2="none"))
    assert "NONE" not in section and "none " not in section.lower().replace("non-person","")
    assert "must preserve this source object interaction" not in section
    assert "Any source object held, carried, touched" in section  # generic rule stays


def test_detailed_section_never_reactivates_source_labels():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    for frame0 in (False,True):
        section = detailed(build_prompt("a man","a woman","shot","an old man","a young woman",
                                        frame0=frame0,
                                        performance1="holding a black clipboard with both hands",
                                        performance2="NONE"))
        assert "<Subject 3>" in section and "<Subject 4>" in section
        assert "<Subject 1>" not in section
        assert "<Subject 2>" not in section
        for phrase in ("Edit <Video 1> in place","Do not add any new person",
                       "duplicate either replacement","keep a source identity as an additional person",
                       "same visible-person count and occupancy as <Video 1>",
                       "facial expression, gaze, mouth movement","directly from <Video 1>"):
            assert phrase in section
        assert "exactly two visible people" not in section


def test_definitions_declare_source_only_and_in_place_replacement():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    definitions = build_prompt("a man","a woman","shot","an old man","a young woman").split(
        "subject_definitions:")[1].split("summary:")[0]
    assert definitions.count("source-only performer identity") == 2
    assert definitions.count("must not appear as an additional person") == 2
    assert definitions.count("replaces <Subject 1> in place") == 1
    assert definitions.count("replaces <Subject 2> in place") == 1
    assert definitions.count("must never coexist as separate people") == 2
    summary = build_prompt("a man","a woman","shot","an old man","a young woman").split(
        "summary:")[1].split("retention_analysis:")[0]
    assert "The replacement happens in place." in summary
    assert "Do not retain a source performer beside its replacement" in summary
    assert "duplicate a replacement, or add any new person" in summary


def test_without_performance_arguments_prompt_is_unchanged():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    base = build_prompt("a man","a woman","shot","an old man","a young woman")
    assert "preserves <Subject 1>'s source performance" not in base
    assert "Any object held, carried, touched" not in base
    assert base == build_prompt("a man","a woman","shot","an old man","a young woman")


def test_performance_anchor_also_applies_to_frame0():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    prompt = build_prompt("a man","a woman","shot","an old man","a young woman",frame0=True,
                          performance1="holding a black clipboard with both hands")
    assert "appearance from <Picture 1>" in prompt
    assert "holding a black clipboard with both hands" in prompt
