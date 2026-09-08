from __future__ import annotations

import json

import pytest

from r2v_data_v2.h3 import mimo25_backend as mb
from r2v_data_v2.h3 import mimo25_h3_materializer as mm
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import _job
from r2v_data_v2.h3.mimo25_recovered_voice import recover_mimo_target_voices
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionAudioContract,
    _canonical_audio_definition,
    _canonical_audio_retention,
)
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _as_asd_miss,
    _backend,
    _job_fixture,
    _job_for_reference_inventory,
    _record_fixture,
    _RecoveredVoiceAnalyzer,
    _RecoveredVoiceMediaBackend,
    _sample,
    _sample_with_reference_inventory,
    _visual_reference_inventory,
)


def test_turn3_profile_reaches_mimo_only_recovered_s2_audio_reference(tmp_path):
    job = _as_asd_miss(_job_fixture(tmp_path))
    values = job.model_dump(mode="json", exclude={"request_fingerprint"})
    values["segments"].append({
        **values["segments"][0], "segment_id": "segment_2", "start_time": 1.0,
        "end_time": 2.0, "source_start_sample": 32000, "source_end_sample": 64000,
        "source_speaker_cluster_id": "speaker_1", "asr_text": "Reply.",
    })
    job = _job(values)
    payload = _annotation().model_dump(mode="json")
    payload["audio_observation"]["segment_decisions"].append({
        **payload["audio_observation"]["segment_decisions"][0],
        "segment_id": "segment_2", "primary_speaker_group": "g2",
    })
    first_bound = payload["av_grounding"]["segment_groundings"][0]
    payload["av_grounding"]["segment_groundings"].append({
        **first_bound, "segment_id": "segment_2", "primary_speaker_group": "g2",
    })
    first_bound.update(binding_status="offscreen", speech_presentation="offscreen_spoken",
                       entity_id=None, evidence_codes=["offscreen_audio"])
    payload["visual_observation"]["segment_views"].append({
        **payload["visual_observation"]["segment_views"][0], "segment_id": "segment_2",
    })
    profile = "low register, dry texture, brisk cadence"
    payload["audio_observation"]["speaker_voice_profiles"] = [
        {"speaker_group": "g1", "voice_characteristics": "high register, soft texture"},
        {"speaker_group": "g2", "voice_characteristics": profile},
    ]
    payload["h3_semantics"]["shot1_caption"] = (
        "<Subject 1> stands. An offscreen voice (S1) says, <d>[English] Exact, text!</d> "
        "<Subject 1> (S2) replies, <d>[English] Reply.</d>"
    )
    raws = split_annotation(json.dumps(payload))
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws])
    result = backend.reconcile(
        job, segment_ids=["segment_1", "segment_2"],
        transcribed_segment_ids=["segment_1", "segment_2"], allowed_entity_ids={"e1"},
        allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_paths={kind: tmp_path / "full.flac" for kind in ("speech", "music", "sfx")},
    )
    assert result.model_call_count == len(calls.requests) == 3
    assert all(s.current_entity_id is None and s.direct_anchor_seconds == 0 for s in job.segments)
    assert result.annotation.av_grounding.segment_groundings[0].binding_status == "offscreen"
    assert result.annotation.av_grounding.segment_groundings[1].entity_id == "e1"
    text = calls.requests[2]["messages"][1]["content"][1]["text"]
    targets = json.loads(text.split("FINALIZED SPEAKER-PROFILE TARGETS:\n")[1].split("\nRESPONSE SCHEMA:", 1)[0])
    assert [(t["speaker_group"], t["speaker_id"]) for t in targets] == [("g1", "S1"), ("g2", "S2")]
    assert [t["segments"][0]["final_binding"] for t in targets] == ["offscreen", "<Subject 1>"]
    assert [t["segments"][0]["start_time"] for t in targets] == [0.0, 1.0]
    assert json.loads(raws[1])["audio_observation"]["speaker_voice_profiles"] == []
    assert result.annotation.speaker_voice_profiles[1].voice_characteristics == profile
    record = _record_fixture(tmp_path, result.annotation, job=job)
    analyzer = _RecoveredVoiceAnalyzer()
    analyzer.analysis.mono_pcm16[32000:64000] = 3000
    recovered, voices, _ = recover_mimo_target_voices(
        job=job, record=record, subject_index_by_entity={"e1": 1}, existing_entity_ids=set(),
        reference_capacity=3, temporary_root=tmp_path / "voice-temp", final_root=tmp_path / "voices",
        audio_backend=_RecoveredVoiceMediaBackend(), analyzer=analyzer,
    )
    assert len(voices) == 1, recovered
    assert recovered[0].status == "selected" and recovered[0].source_segment_id == "segment_2"
    sample = _sample(tmp_path)
    sample_values = sample.model_dump(mode="python")
    sample_values["subject_voices"] = [v.model_dump(mode="python") for v in voices]
    original = sample_values["speech_segments"][0]
    sample_values["speech_segments"] = [{
        **original, "segment_id": s.segment_id, "speaker_cluster_id": s.source_speaker_cluster_id,
        "entity_id": None, "entity_occurrence_id": None,
        "start_time": s.start_time, "end_time": s.end_time,
        "source_start_sample": s.source_start_sample, "source_end_sample": s.source_end_sample,
        "text": s.asr_text,
    } for s in job.segments]
    sample = FinalH3SampleV2.model_validate(sample_values)
    _, rendered, _ = mm._materialize_sample(sample, job, record)
    assert f"<Audio 1> is the voice-timbre reference for <Subject 1> (S2), featuring {profile}." in rendered
    assert "<Audio 1>: reference -" in rendered
    assert mb.MIMO25_MATERIALIZER_VERSION == "h3_mimo25_materializer_v25"


@pytest.mark.parametrize("variant", [
    "visual_only", "target_voice_reference", "cross_voice_reference",
    "music_reference", "full_audio_reuse",
])
def test_final_contract_audio_definitions_and_retention(tmp_path, monkeypatch, variant):
    job = _job_fixture(tmp_path)
    sample = _sample(tmp_path)
    if variant == "cross_voice_reference":
        values = sample.model_dump()
        values["pair_type"] = "cross_pair"
        values["subject_voices"][0].update(
            voice_source="cross_donor", donor_occurrence_id="donor/e1",
            donor_clip_uid="donor", donor_clip_display_path="01/show/donor",
        )
        sample = FinalH3SampleV2.model_validate(values)
    record = _record_fixture(tmp_path, _annotation(), job=job)
    original_record = record.model_dump()
    original_sample = sample.model_dump()
    contracts, rendered = [], []
    audio_facts, render = mm._audio_facts, mm.render_h3_prompt

    def capture_facts(**kwargs):
        contracts.append(kwargs["contract"])
        return audio_facts(**kwargs)

    def capture_render(structured):
        rendered.append(structured)
        return render(structured)

    monkeypatch.setattr(mm, "_audio_facts", capture_facts)
    monkeypatch.setattr(mm, "render_h3_prompt", capture_render)
    _, visual_text, _ = mm._materialize_sample(
        sample, job, record, conditioning_variant="visual_only",
    )
    assert contracts[-1].audios == []
    assert "<Audio " not in visual_text
    visual = rendered[-1]
    extra = None
    if variant == "music_reference":
        extra = RecaptionAudioContract(
            audio_index=1, audio_label="<Audio 1>", kind="music_reference",
            path=sample.target_full_audio_path, sha256=sample.target_full_audio_sha256,
            retention_marker="reference", music_characteristics="a gentle piano melody",
        )
    _, text, _ = mm._materialize_sample(
        sample, job, record, conditioning_variant=variant, extra_audio_contract=extra,
    )
    contract, final = contracts[-1], rendered[-1]
    definitions = [_canonical_audio_definition(audio) for audio in contract.audios]
    retentions = [_canonical_audio_retention(audio) for audio in contract.audios]
    assert final.subject_definitions == visual.subject_definitions + definitions
    assert final.retention_analysis == visual.retention_analysis + retentions
    assert final.detailed_description == visual.detailed_description
    assert final.overall_soundscape == visual.overall_soundscape
    assert final.non_diegetic_music == visual.non_diegetic_music
    assert all(line in text for line in definitions + retentions)
    if variant == "visual_only":
        assert not contract.audios and "<Audio " not in text
    else:
        assert len(contract.audios) == 1
        audio = contract.audios[0]
        assert text.count("<Audio 1>") == 2
        if variant in {"target_voice_reference", "cross_voice_reference"}:
            assert audio.kind == ("target_voice" if variant == "target_voice_reference" else "cross_voice")
            assert (audio.subject_label, audio.speaker_id) == ("<Subject 1>", "S1")
            assert "voice-timbre reference for <Subject 1> (S1)" in definitions[0]
            assert retentions[0].startswith("<Audio 1>: reference -")
        elif variant == "music_reference":
            assert "a gentle piano melody" in definitions[0]
            assert "without copying the source waveform" in retentions[0]
            assert audio.subject_label is None
        else:
            assert audio.kind == "full_audio_reuse"
            assert definitions == ["<Audio 1> is the supplied full-audio reference for the target."]
            assert retentions == ["<Audio 1>: fully_copy - the supplied full audio is reused in full."]
    assert record.model_dump() == original_record
    assert sample.model_dump() == original_sample


@pytest.mark.parametrize("owner_entity_id", ["e1", "e2"])
def test_attribute_owner_is_materialized_from_frozen_contract(tmp_path, owner_entity_id):
    references = _visual_reference_inventory(
        tmp_path, ["subject", "subject", "face", "hair", "upper_clothing", "background"],
    )
    for reference in references:
        if reference.kind == "attribute":
            reference.owner_entity_id = owner_entity_id
    sample = _sample_with_reference_inventory(tmp_path, references)
    job = _job_for_reference_inventory(tmp_path, sample)
    payload = _annotation().model_dump()
    descriptions = {
        "entity": "an older man in a grey shirt",
        "attribute": "a visible reference detail",
        "background": "a room with plain walls",
    }
    payload["h3_semantics"]["subject_definitions"] = [
        {"subject_label": item.subject_label, "description": descriptions[item.kind]}
        for item in job.reference_subjects
    ]
    payload["h3_semantics"]["visual_retention_analysis"] = [
        {"subject_label": item.subject_label, "marker": "fully_preserved",
         "description": "The referenced detail remains visible."}
        for item in job.reference_subjects
    ]
    annotation = type(_annotation()).model_validate(payload)
    record = _record_fixture(tmp_path, annotation, job=job)
    _, text, _ = mm._materialize_sample(sample, job, record)
    definitions = text.split("subject_definitions:\n", 1)[1].split("\n\nsummary:", 1)[0]
    owner_label = next(item.subject_label for item in job.reference_subjects if item.entity_id == owner_entity_id)
    for item in job.reference_subjects:
        if item.kind == "attribute":
            kind = item.attribute_type.replace("_", " ")
            assert f"{item.subject_label} is the {kind} of {owner_label}" in definitions
            assert "described as a visible reference detail" in definitions
            assert item.source_picture_labels[0] in definitions
        elif item.kind == "background":
            line = next(line for line in definitions.splitlines() if line.startswith(item.subject_label))
            assert "<Subject 1>" not in line and "<Subject 2>" not in line
            assert "depicted in" in line
    assert "subject_definitions" in text and "[Shot 1]" in text
