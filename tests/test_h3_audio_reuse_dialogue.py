from __future__ import annotations

import re

import pytest

pytest.importorskip("soundfile")

from r2v_data_v2.h3.audio_reuse_materializer import (
    AUDIO_REUSE_MATERIALIZER_VERSION,
    AudioReuseProduct,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    _materialize_sample,
    project_authoritative_dialogue,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionSpeechFact,
    build_reference_contract,
)
from r2v_data_v2.h3.speech_presentation import semantic_speech_source
from tests.test_h3_audio_reuse_materializer import _case, _prepared, _render


def _blocks(text):
    return re.findall(r"<d>[\s\S]*?</d>", text)


def _detail(prompt):
    return prompt.split("[Shot 1] ", 1)[1].split("\n\noverall_soundscape:", 1)[0]


def test_translated_model_dialogue_replaced_without_annotation_mutation(tmp_path):
    caption = "<Subject 1> (S1) says, <d>[English] Ah.</d> She waits."
    _, sample, source = _case(tmp_path, caption=caption, speech_payloads=[("Chinese", "\u554a\u3002")])
    before = source.record.model_dump()
    _, _, _, prompt = _render(sample, source)
    assert _blocks(prompt) == ["<d>[Chinese] \u554a\u3002</d>"]
    assert "Ah." not in prompt and "She waits." in prompt
    assert source.record.model_dump() == before
    # The same frozen annotation still renders literally through legacy v26.
    _, legacy, _ = _materialize_sample(sample, source.job, source.record, conditioning_variant="visual_only")
    assert _blocks(legacy) == ["<d>[English] Ah.</d>"]


def test_missing_second_speaker_has_real_clause_before_audio_projection(tmp_path):
    chinese = "\u6211\u4ece\u6765\u6ca1\u6709\u7ed9\u522b\u4eba\u5199\u8fc7\u60c5\u4e66\u3002"
    _, sample, source = _case(
        tmp_path, ("g1", "g2"),
        caption=f"<Subject 1> (S1) says, <d>[Chinese] {chinese}</d>",
        speech_payloads=[("Chinese", chinese), ("Malay", "eh")],
    )
    refs, _, _, prompt = _render(sample, source)
    assert _blocks(prompt) == [f"<d>[Chinese] {chinese}</d>", "<d>[Malay] eh</d>"]
    assert [r.contract.speaker_id for r in refs] == ["S1", "S2"]
    assert "An off-screen voice (S2), with the synchronized speech signal copied directly from <Audio 2>, says," in prompt
    assert prompt.count("copied directly from <Audio 2>") == 1


@pytest.mark.parametrize("caption", [
    "(S1) says, <d>[English] merged a and b</d>",
    "(S1) says, <d>[English] a</d>",
])
def test_merged_or_missing_same_speaker_expands_exact_segments(tmp_path, caption):
    _, sample, source = _case(tmp_path, ("g1", "g1"), caption=caption,
                             speech_payloads=[("Chinese", "\u554a\u3002"), ("Chinese", "\u4e00\u5b9a\u7ed3\u4e86\u5a5a\uff0c\u518d\u5728\u3002")])
    _, _, _, prompt = _render(sample, source)
    assert _blocks(prompt) == [f"<d>[{s.asr_language}] {s.asr_text}</d>" for s in source.job.segments]
    assert prompt.count("copied directly from <Audio 1>") == 1


def test_missing_middle_speaker_inserted_before_next_aligned_slot(tmp_path):
    caption = "(S1) says, <d>[English] a</d> Then (S3) responds, <d>[English] c</d> The camera stays still."
    _, sample, source = _case(tmp_path, ("g1", "g2", "g3"), caption=caption,
                             speech_payloads=[("English", "a"), ("Malay", "b"), ("English", "c")])
    _, _, _, prompt = _render(sample, source)
    assert _blocks(prompt) == ["<d>[English] a</d>", "<d>[Malay] b</d>", "<d>[English] c</d>"]
    assert prompt.index("(S2)") < prompt.index("Then (S3)")
    assert "The camera stays still." in prompt


@pytest.mark.parametrize("groups,caption", [
    (("g1",), "(S1)<d>[English] a</d> (S1)<d>[English] extra</d>"),
    (("g1", "g2"), "(S2)<d>[English] b</d> (S1)<d>[English] a</d>"),
    (("g1", "g2", "g3"), "(S2)<d>[English] b</d> (S1)<d>[English] a</d>"),
    (("g1", "g2", "g3"), "(S1) and (S2)<d>[English] ambiguous</d>"),
])
def test_extra_or_contradictory_slots_fail_closed(tmp_path, groups, caption):
    _, sample, source = _case(tmp_path, groups, caption=caption)
    with pytest.raises(ValueError, match="slot_|marker_mismatch"):
        _render(sample, source)


def test_exact_existing_prose_and_continuation_are_byte_stable(tmp_path):
    caption = "<Subject 1> (S1) says, <d>[English] a</d>  He continues, <d>[English] b</d>\nHe waits."
    _, sample, source = _case(tmp_path, ("g1", "g1"), caption=caption, speech_payloads=[("English", "a"), ("English", "b")])
    _, prompt, _ = _materialize_sample(sample, source.job, source.record, reuse_audio_contracts=[])
    assert _detail(prompt) == caption


@pytest.mark.parametrize("presentation", [
    "onscreen_spoken", "offscreen_spoken", "voice_over", "message_voice_over", "device_playback", "uncertain",
])
def test_missing_clause_reuses_semantic_speech_source(tmp_path, presentation):
    _, sample, _ = _case(tmp_path)
    subject = "<Subject 1>" if presentation == "onscreen_spoken" else None
    speech = RecaptionSpeechFact(
        fact_id="segment", segment_id="segment", speaker_cluster_id="g1",
        entity_id="e1" if subject else None, entity_subject_label=subject, speaker_id="S1",
        start_time=0, end_time=.2, text="exact", language="English",
        locked_dialogue_block="<d>[English] exact</d>",
    )
    caption = project_authoritative_dialogue("The room is still.", [speech], {"segment": presentation},
                                            build_reference_contract(sample, "visual_only"))
    source, action = semantic_speech_source(speaker_id="S1", subject_label=subject, presentation=presentation)
    assert caption == f"The room is still.\n{source} {action} {speech.locked_dialogue_block}"


def test_all_product_variants_project_target_not_donor_dialogue(tmp_path):
    _, sample, source = _case(tmp_path / "target", caption="(S1) says, <d>[English] translation</d>",
                             speech_payloads=[("Chinese", "\u554a\u3002")])
    _, _, donor = _case(tmp_path / "donor", clip="donor", text="Donor must never enter target.")
    from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2

    values = sample.model_dump(mode="json")
    values["pair_type"] = "cross_pair"
    values["subject_voices"][0].update(voice_source="cross_donor", donor_clip_uid="donor",
                                     donor_occurrence_id="donor/e1", donor_clip_display_path="01/show/donor")
    cross = FinalH3SampleV2.model_validate(values)
    _, _, _, cross_prompt = _render(cross, source, {source.job.clip_uid: source, "donor": donor})
    from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionAudioContract

    audio = RecaptionAudioContract(audio_index=1, audio_label="<Audio 1>", kind="full_audio_reuse",
                                  path=source.job.target_full_audio_path, sha256=source.job.target_full_audio_sha256,
                                  retention_marker="fully_copy")
    for variant, audios in (("visual_only", []), ("full_audio_reuse", [audio])):
        _, prompt, _ = _materialize_sample(sample, source.job, source.record, conditioning_variant=variant, reuse_audio_contracts=audios)
        assert _blocks(prompt) == ["<d>[Chinese] \u554a\u3002</d>"]
    assert _blocks(cross_prompt) == ["<d>[Chinese] \u554a\u3002</d>"]
    assert "Donor must never enter target." not in cross_prompt
    assert "voice characteristics referenced from <Audio 1>" in cross_prompt


def test_published_product_validator_enforces_exact_dialogue(tmp_path):
    from r2v_data_v2.h3 import audio_reuse_materializer as product

    args = _prepared(tmp_path)
    product.materialize_audio_reuse_products(**args)
    rows = product._rows(args["output_root"] / "records.jsonl", AudioReuseProduct)
    assert AUDIO_REUSE_MATERIALIZER_VERSION == "h3_mimo25_audio_reuse_materializer_v2"
    values = rows[0].model_dump(mode="json")
    values["rendered_h3_prompt"] = values["rendered_h3_prompt"].replace("Exact, text!", "Translated text")
    values["record_fingerprint"] = product._hash({k: v for k, v in values.items() if k != "record_fingerprint"})
    with pytest.raises(ValueError, match="authoritative_dialogue_mismatch"):
        AudioReuseProduct.model_validate(values)
