from r2v_data_v2.person_replacement import pipeline, qwen3vl


def test_bernini_prompt_makes_person_replacement_appearance_only():
    prompt = pipeline.build_prompt(
        "the man in a dark shirt with both hands in his trouser pockets",
        "a different middle-aged woman in a light jacket and trousers",
    )

    assert "appearance-only person replacement" in prompt
    assert "Do not invent, reinterpret, simplify, or adjust" in prompt
    assert "motion frame by frame" in prompt
    assert "hand-body and hand-object contact states" in prompt
    assert "hand is inside a pocket" in prompt
    assert "same contact state and placement" in prompt


def test_qwen_source_prompt_requests_stable_pose_and_contact_anchor():
    assert "stable pose or action" in qwen3vl.SOURCE_PROMPT
    assert "hand-body or hand-object contact" in qwen3vl.SOURCE_PROMPT
    assert "hands inside trouser pockets" in qwen3vl.SOURCE_PROMPT
