from __future__ import annotations

import pytest

from r2v_data_v2.h3 import mimo25_h3_materializer as mm
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionAudioContract,
    _canonical_audio_definition,
    _canonical_audio_retention,
)
from tests.test_h3_mimo25_av_shadow import (
    _annotation,
    _job_fixture,
    _record_fixture,
    _sample,
)


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
