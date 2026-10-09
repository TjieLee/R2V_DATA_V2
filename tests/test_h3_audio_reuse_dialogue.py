from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("soundfile")

from r2v_data_v2.h3.audio_reuse_materializer import (
    AUDIO_REUSE_MATERIALIZER_VERSION,
    AudioReuseProduct,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    _materialize_sample,
    project_authoritative_dialogue,
    prune_summary_dialogue,
    validate_product_dialogue_sections,
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


def _real_marker_case(tmp_path, case_index):
    from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
    from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
    from r2v_data_v2.h3.mimo25_stem_shadow import MimoStemReconcileRecord
    from tests.test_h3_mimo25_av_shadow import _sample

    fixture = Path(__file__).parent / "fixtures/two_step_dialogue_slot_real_cases_v1/real_cases.json"
    case = json.loads(fixture.read_text())["cases"][case_index]
    job = MimoClipJob.model_validate(case["frozen_job"])
    record = MimoStemReconcileRecord.model_validate(case["reconcile_record"])
    sample = _sample(tmp_path)
    # Only media paths use local stand-ins; all frozen text/ownership/timing stays real.
    image = sample.visual_references[0].image_artifact_path
    references = []
    for reference in [*job.reference_images, *job.reference_selection.dropped_references]:
        references.append({
            "image_id": reference.source_image_id, "image_index": reference.source_image_index,
            **{key: getattr(reference, key) for key in (
                "kind", "entity_id", "attribute_id", "owner_entity_id", "attribute_type",
            )},
            "image_artifact_path": image, "image_path": "selected/reference.png",
            "source_frame_index": 0, "scope": None if reference.kind == "attribute" else "full",
            "visible_region": None if reference.kind == "attribute" else "whole", "synthetic": False,
        })
    local_job = job.model_copy(update={"reference_images": [
        reference.model_copy(update={"image_artifact_path": image}) for reference in job.reference_images
    ]})
    values = sample.model_dump(mode="json")
    values.update(
        sample_id=job.source_h3_sample_ids[0], pair_id=f"canonical/{job.clip_uid}", pair_type="canonical",
        clip_uid=job.clip_uid, target_video=job.target_video_path,
        target_full_audio_path=job.target_full_audio_path, target_full_audio_sha256=job.target_full_audio_sha256,
        visual_references=sorted(references, key=lambda r: r["image_index"]), subject_voices=[],
        speech_segments=[{
            **{key: getattr(segment, key) for key in (
                "segment_id", "source_start_sample", "source_end_sample", "source_sample_rate_hz",
                "start_time", "end_time",
            )},
            "speaker_cluster_id": segment.source_speaker_cluster_id,
            "text": segment.asr_text, "language": segment.asr_language,
        } for segment in job.segments],
    )
    return job, local_job, record, FinalH3SampleV2.model_validate(values)


@pytest.mark.parametrize("case_index,count,old_lead,new_lead", [
    (0, 3, ", and asks, ", "; <Subject 1> (S1) asks, "),
    (1, 2, ", saying, ", "; <Subject 1> (S1) says, "),
])
@pytest.mark.parametrize("variant", ["visual_only", "target_speech_reuse", "full_audio_reuse"])
def test_real_two_step_marker_captions_all_product_variants(tmp_path, case_index, count, old_lead, new_lead, variant):
    from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionAudioContract

    frozen_job, job, record, sample = _real_marker_case(tmp_path, case_index)
    before = (frozen_job.model_dump(), record.model_dump(), sample.model_dump())
    assert record.status == "ready" and record.model_call_count == 2
    assert record.source_job_fingerprint == frozen_job.request_fingerprint
    assert len(record.raw_responses) == 2
    assert record.raw_responses == [record.visual_raw_response, record.speech_av_raw_response]
    subject = next(s for s in job.reference_subjects if s.kind == "entity" and s.entity_id == "e1")
    assert subject.subject_label == "<Subject 1>"
    caption = record.annotation.h3_semantics.shot1_caption
    expected_blocks = [f"<d>[{s.asr_language}] {s.asr_text}</d>" for s in job.segments]
    assert len(expected_blocks) == count and _blocks(caption) == expected_blocks
    expected = caption.replace(old_lead, new_lead, 1)
    audios = []
    if variant != "visual_only":
        audios = [RecaptionAudioContract(
            audio_index=1, audio_label="<Audio 1>",
            kind="speaker_speech_reuse" if variant == "target_speech_reuse" else "full_audio_reuse",
            path=job.target_full_audio_path, sha256=job.target_full_audio_sha256,
            retention_marker="partially_copy" if variant == "target_speech_reuse" else "fully_copy",
            **({"speaker_id": "S1", "entity_id": "e1", "subject_label": subject.subject_label}
               if variant == "target_speech_reuse" else {}),
        )]
        if variant == "target_speech_reuse" and record.annotation.h3_semantics.non_diegetic_music != "N/A":
            audios.append(RecaptionAudioContract(
                audio_index=2, audio_label="<Audio 2>", kind="music_reuse",
                path=job.target_full_audio_path, sha256=job.target_full_audio_sha256, retention_marker="partially_copy",
            ))
    corrected, prompt, warnings = _materialize_sample(
        sample, job, record, conditioning_variant=variant, reuse_audio_contracts=audios,
    )
    detail = _detail(prompt)
    relation = ", with the synchronized speech signal copied directly from <Audio 1>,"
    expected_detail = expected.replace("(S1)", "(S1)" + relation, 1) if variant == "target_speech_reuse" else expected
    assert detail == expected_detail
    assert detail.count("(S1)") == 1 and _blocks(detail) == expected_blocks
    assert "(S1)" not in detail.split("; <Subject 1> (S1)", 1)[0]
    if variant == "target_speech_reuse":
        assert detail.count("copied directly from <Audio 1>") == 1
    assert set(re.findall(r"<Audio \d+>", prompt)) == {audio.audio_label for audio in audios}
    if len(audios) == 2:
        assert "music signal copied directly from <Audio 2>" in prompt
    assert "deterministic_correction_count:two_step_first_speaker_marker_inserted=1" in warnings
    assert [(s.segment_id, s.text, s.language, s.start_time, s.end_time, s.source_start_sample,
             s.source_end_sample, s.speaker_cluster_id, s.entity_id) for s in corrected] == [
        (s.segment_id, s.asr_text, s.asr_language, s.start_time, s.end_time, s.source_start_sample,
         s.source_end_sample, "g1", "e1") for s in job.segments
    ]
    validate_product_dialogue_sections(prompt, expected_blocks)
    assert (frozen_job.model_dump(), record.model_dump(), sample.model_dump()) == before


@pytest.mark.parametrize("lead,count,expected", [
    ("<Subject 1> asks, ", 3, "<Subject 1> (S1) asks, "),
    ("<Subject 1> leans forward, saying, ", 2, "<Subject 1> leans forward; <Subject 1> (S1) says, "),
])
def test_two_step_missing_first_marker_all_three_products_ready(tmp_path, lead, count, expected):
    from r2v_data_v2.h3 import audio_reuse_materializer as product
    from r2v_data_v2.h3.audio_reuse_prepared import AudioReusePreparedSource
    from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
    from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
    from tests.test_h3_audio_reuse import _seal

    blocks = [f"<d>[English] Line {i}.</d>" for i in range(count)]
    opening = "<Subject 1> sits at the table. "
    caption = opening + lead + blocks[0] + "".join(f" He continues, {b}" for b in blocks[1:])
    caption += " The camera stays still."
    args = _prepared(tmp_path, case_kwargs={
        "groups": ("g1",) * count, "all_visible": True, "caption": caption,
        "speech_payloads": [("English", f"Line {i}.") for i in range(count)],
    })
    path = args["mimo_root"] / "records.jsonl"
    record = AudioReusePreparedSource.model_validate_json(path.read_text())
    values = record.model_dump(mode="json")
    values["source_backend_provenance"] = TwoStepOpenAIMimo26Backend(MimoBackendConfig(
        api_key="fixture", transport="sglang", media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
    ), stem_records_by_clip={}).provenance.model_dump(mode="json")
    record = _seal(AudioReusePreparedSource, values, "prepared_fingerprint")
    path.write_text(record.model_dump_json() + "\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    summary = product.materialize_audio_reuse_products(**args, enable_full_audio_reuse=True)
    assert (summary.ready_count, summary.failed_count, summary.model_call_count) == (3, 0, 0)
    rows = product._rows(args["output_root"] / "records.jsonl", AudioReuseProduct)
    assert [r.conditioning_variant for r in rows] == ["visual_only", "target_speech_reuse", "full_audio_reuse"]
    for row in rows:
        assert row.status == "ready", row.failure_reason
        detail = _detail(row.rendered_h3_prompt)
        assert detail.startswith(opening) and "(S1)" not in opening
        assert detail.count("(S1)") == 1 and _blocks(detail) == blocks
        assert " The camera stays still." in detail
        assert {s.speaker_cluster_id for s in row.corrected_speech_segments} == {"g1"}
        assert {s.entity_id for s in row.corrected_speech_segments} == {"e1"}
        assert "deterministic_correction_count:two_step_first_speaker_marker_inserted=1" in row.warnings
        if row.conditioning_variant == "target_speech_reuse":
            assert detail.count("copied directly from <Audio 1>") == 1
            assert row.audio_references[0].contract.speaker_id == "S1"
        else:
            assert detail == caption.replace(lead, expected, 1)
    assert before == {p: p.read_bytes() for p in before}


def test_missing_first_marker_is_not_repaired_for_multi_single_or_legacy_materialization(tmp_path):
    caption = "<Subject 1> sits. <Subject 1> asks, <d>[English] Exact, text!</d>"
    _, sample, source = _case(tmp_path, caption=caption)
    for prompt_version in ("h3_mimo25_speech_assembly_v49", "h3_mimo26_ra2va_single_v10_speaker_subject_consistency"):
        record = source.record.model_copy(update={
            "source_backend_provenance": SimpleNamespace(prompt_version=prompt_version),
        })
        with pytest.raises(ValueError, match="audio_reuse_dialogue_slot_ambiguous"):
            _materialize_sample(sample, source.job, record, reuse_audio_contracts=[])
    record = source.record.model_copy(update={"source_backend_provenance": SimpleNamespace(
        prompt_version="h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity",
    )})
    _, prompt, _ = _materialize_sample(sample, source.job, record)
    assert _detail(prompt) == caption


@pytest.mark.parametrize("case", [
    "multiple_groups", "missing_asr", "wrong_order", "wrong_text", "conflicting_marker",
    "unbound", "missing_entity", "negative_evidence", "competing_speaker", "subject_mapping", "unknown_lead", "other_actor",
    "ambiguous_pronoun_asks", "ambiguous_pronoun_saying",
])
def test_two_step_first_marker_refuses_ambiguous_or_inconsistent_inputs(tmp_path, case):
    from r2v_data_v2.h3.jea_final_renderer import FinalQwen3SpeechSegment
    from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
    from r2v_data_v2.h3.mimo25_h3_materializer import (
        _audio_facts,
        _two_step_first_speaker_marker,
    )

    caption = "<Subject 1> sits. <Subject 1> asks, <d>[English] One.</d> He continues, <d>[English] Two.</d>"
    _, sample, source = _case(tmp_path, ("g1", "g1"), caption=caption, all_visible=True,
                             speech_payloads=[("English", "One."), ("English", "Two.")])
    payload = source.record.annotation.model_dump(mode="json")
    if case == "multiple_groups":
        payload["audio_observation"]["segment_decisions"][1]["primary_speaker_group"] = "g2"
        payload["av_grounding"]["segment_groundings"][1]["primary_speaker_group"] = "g2"
    elif case == "missing_asr":
        caption = caption.replace("<d>[English] Two.</d>", "")
    elif case == "wrong_order":
        caption = caption.replace("One.", "Temp.").replace("Two.", "One.").replace("Temp.", "Two.")
    elif case == "wrong_text":
        caption = caption.replace("One.", "Rewritten.")
    elif case == "conflicting_marker":
        caption = caption.replace("<Subject 1> asks", "<Subject 1> (S2) asks")
    elif case == "unbound":
        payload["av_grounding"]["segment_groundings"][0].update(
            binding_status="no_reliable_entity", entity_id=None, speech_presentation="uncertain",
        )
    elif case == "missing_entity":
        payload["visual_observation"]["segment_views"][0]["visible_entity_ids"] = []
        payload["visual_observation"]["segment_views"][0]["entity_observations"] = []
    elif case == "negative_evidence":
        payload["av_grounding"]["segment_groundings"][0]["evidence_codes"] = ["offscreen_audio"]
    elif case == "competing_speaker":
        payload["audio_observation"]["segment_decisions"][0]["vocal_composition"] = "uncertain"
    elif case == "unknown_lead":
        caption = caption.replace("<Subject 1> asks", "<Subject 1> mutters")
    elif case == "other_actor":
        caption = caption.replace("<Subject 1> asks", "Another person leans forward, saying")
    elif case == "ambiguous_pronoun_asks":
        caption = caption.replace("<Subject 1> asks", "Another person enters. He asks")
    elif case == "ambiguous_pronoun_saying":
        caption = caption.replace("<Subject 1> asks", "Another person enters. He leans forward, saying")
    job = source.job
    if case == "subject_mapping":
        job = job.model_copy(update={"reference_subjects": [
            s.model_copy(update={"entity_id": "e2"}) if s.kind == "entity" else s for s in job.reference_subjects
        ]})
    record = source.record.model_copy(update={
        "annotation": MimoAVAnnotationDraft.model_validate(payload),
        "source_backend_provenance": SimpleNamespace(prompt_version="h3_mimo26_ra2va_two_step_joint_v2_caption_fidelity"),
    })
    corrected = [FinalQwen3SpeechSegment.model_validate({
        **s.model_dump(), "speaker_cluster_id": "g1", "entity_id": "e1", "entity_occurrence_id": f"{job.clip_uid}/e1",
    }) for s in sample.speech_segments]
    contract = build_reference_contract(sample, "visual_only")
    facts = _audio_facts(sample=sample, corrected=corrected, record=record, contract=contract)
    assert _two_step_first_speaker_marker(caption, job, record, facts.speech) == caption
    with pytest.raises(ValueError, match="audio_reuse_dialogue_slot_ambiguous"):
        project_authoritative_dialogue(caption, facts.speech,
                                      {g.segment_id: g.speech_presentation for g in record.annotation.segment_decisions},
                                      contract)


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
    caption = "<Subject 1> stands nearby. (S1) says, <d>[English] a</d> Then (S3) responds, <d>[English] c</d> The camera stays still."
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
    _, sample, source = _case(tmp_path / "target", caption="<Subject 1> stands nearby. (S1) says, <d>[English] translation</d>",
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
    assert AUDIO_REUSE_MATERIALIZER_VERSION == "h3_mimo25_audio_reuse_materializer_v7"
    values = rows[0].model_dump(mode="json")
    values["rendered_h3_prompt"] = values["rendered_h3_prompt"].replace("Exact, text!", "Translated text")
    values["record_fingerprint"] = product._hash({k: v for k, v in values.items() if k != "record_fingerprint"})
    with pytest.raises(ValueError, match="authoritative_dialogue_mismatch"):
        AudioReuseProduct.model_validate(values)


def test_zero_speech_stale_summary_pruned_product_ready_and_legacy_unchanged(tmp_path):
    from r2v_data_v2.h3 import audio_reuse_materializer as product

    visual = "A woman stands on a stage."
    stale = " A woman (S1) speaks to the audience, saying, <d>[Chinese] stale speech.</d>."
    _, sample, source = _case(tmp_path, (), summary=visual + stale)
    before = source.record.model_dump()
    _, _, corrected, prompt = _render(sample, source)
    assert corrected == []
    assert prompt.split("\n\nsummary:\n", 1)[1].split("\n\nretention_analysis:\n", 1)[0].endswith(visual)
    assert "<d>" not in prompt and "</d>" not in prompt and "(S1)" not in prompt
    assert source.record.model_dump() == before
    _, legacy, _ = _materialize_sample(sample, source.job, source.record)
    assert visual + stale in legacy
    # The persisted product validator also accepts this zero-ASR result.
    args = _prepared(tmp_path / "prepared")
    product.materialize_audio_reuse_products(**args)
    row = product._rows(args["output_root"] / "records.jsonl", AudioReuseProduct)[0]
    values = row.model_dump(mode="json")
    values.update(rendered_h3_prompt=prompt, corrected_speech_segments=[])
    values["record_fingerprint"] = product._hash({k: v for k, v in values.items() if k != "record_fingerprint"})
    assert AudioReuseProduct.model_validate(values).status == "ready"


def test_clean_zero_speech_visual_summary_is_byte_stable(tmp_path):
    visual = "A woman appears to be speaking on stage.  The camera stays still."
    assert prune_summary_dialogue(visual) == visual
    _, sample, source = _case(tmp_path, (), summary=visual)
    _, _, _, prompt = _render(sample, source)
    assert visual in prompt
    validate_product_dialogue_sections(prompt, [])


def test_real_speech_summary_pruning_preserves_high_level_sx_and_exact_asr(tmp_path):
    retained = "A woman stands indoors. (S1) speaks offscreen."
    stale = " A voice says <d>[English] copied. Another sentence!</d>."
    _, sample, source = _case(tmp_path, summary=retained + stale)
    _, _, _, prompt = _render(sample, source)
    assert retained in prompt and "Another sentence!" not in prompt
    assert _blocks(prompt) == ["<d>[English] Exact, text!</d>"]
    validate_product_dialogue_sections(prompt, _blocks(prompt))


@pytest.mark.parametrize("section", [
    "subject_definitions", "summary", "retention_analysis", "overall_soundscape", "non_diegetic_music",
])
@pytest.mark.parametrize("tag", ["<d>", "</d>"])
def test_dialogue_tags_in_other_sections_fail_closed(tmp_path, section, tag):
    _, sample, source = _case(tmp_path, ())
    _, _, _, prompt = _render(sample, source)
    prompt = prompt.replace(section + ":\n", section + ":\n" + tag, 1)
    with pytest.raises(ValueError, match="dialogue_outside_detailed_description"):
        validate_product_dialogue_sections(prompt, [])


def test_summary_pruning_preserves_retained_slices_and_fails_if_empty():
    assert prune_summary_dialogue("Visual. Speech (S1) <d>[English] a. b!</d>.  More visual.") == "Visual.  More visual."
    with pytest.raises(ValueError, match="summary_empty_after_dialogue_pruning"):
        prune_summary_dialogue("(S1) says <d>[Chinese] a</d>.")


def test_zero_speech_detail_rejects_dialogue_and_bad_section_order(tmp_path):
    _, sample, source = _case(tmp_path, ())
    _, _, _, prompt = _render(sample, source)
    with pytest.raises(ValueError, match="authoritative_dialogue_mismatch"):
        validate_product_dialogue_sections(prompt.replace("\n[Shot 1] ", "\n[Shot 1] <d>[English] extra</d>"), [])
    with pytest.raises(ValueError, match="section_structure_invalid"):
        validate_product_dialogue_sections(prompt.replace("summary:\n", "unknown:\n"), [])
