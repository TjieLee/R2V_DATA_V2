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


def test_h3_prompt_takes_no_performance_arguments():
    """The production H3 prompt is the reverted V13 contract: no anchors at all."""
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    base = build_prompt("a man","a woman","shot","an old man","a young woman")
    assert "source performance and object interactions" not in base
    assert "Any source object held, carried, touched" not in base
    assert base == build_prompt("a man","a woman","shot","an old man","a young woman")


def test_frame0_prompt_keeps_its_original_picture_anchor():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    prompt = build_prompt("a man","a woman","shot","an old man","a young woman",frame0=True)
    assert "<Picture 1> is the edited first-frame target appearance and composition anchor." in prompt
    assert "<Picture 1> additionally supplies an edited first-frame visual anchor" in prompt
