from r2v_data_v2.person_replacement.h3_prompt import build_h3_ref2va_replacement_prompt


def test_six_sections_and_appearance_only_edit():
    prompt = build_h3_ref2va_replacement_prompt("man in gray", "woman in blue")
    headers = [line[:-1] for line in prompt.splitlines() if line.endswith(":" )]
    assert headers == ["subject_definitions", "summary", "retention_analysis",
                       "detailed_description", "overall_soundscape", "non_diegetic_music"]
    for text in ["<Video 1>", "<Subject 1>", "man in gray", "woman in blue",
                 "direct source video", "pose", "hand placement", "hand-body contact",
                 "hand-object contact", "expression", "gaze", "background", "camera", "timing"]:
        assert text in prompt
    assert "<Picture" not in prompt
    assert prompt.endswith("non_diegetic_music:\nN/A")
