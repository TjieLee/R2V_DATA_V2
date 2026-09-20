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
    assert "<Video 1> alone decides timing" in prompt


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


def test_performance_anchors_reach_detailed_description_but_shot_does_not():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    shot = "UNIQUE_SHOT_MARKER_1234"
    prompt = build_prompt("a man","a woman",shot,"an old man","a young woman",
                          performance1="holding and handling a hockey stick",
                          performance2="holding a black clipboard with both hands")
    assert "<Subject 3> preserves <Subject 1>'s source performance and object interactions " \
           "from <Video 1>: holding and handling a hockey stick." in prompt
    assert "<Subject 4> preserves <Subject 2>'s source performance and object interactions " \
           "from <Video 1>: holding a black clipboard with both hands." in prompt
    assert "Any object held, carried, touched, or otherwise interacted with by <Subject 1> " \
           "or <Subject 2> in <Video 1> must remain the same source object" in prompt
    assert shot not in prompt  # SHOT_DESCRIPTION stays provenance only
    assert prompt.index("holding and handling a hockey stick") > prompt.index("detailed_description")


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
