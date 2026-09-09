from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse_materializer as product
from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob, _inventory
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from r2v_data_v2.h3.mimo25_h3_materializer import _materialize_sample
from r2v_data_v2.h3.mimo25_recovered_voice import _semantic_reasons
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionAudioContract,
    _canonical_audio_definition,
    _canonical_audio_retention,
    audio_task_prefix,
    build_reference_contract,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import SAMAudioStemRecord, sha256_file
from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons
from tests.test_h3_audio_reuse import _fixture, _seal
from tests.test_h3_mimo25_av_shadow import _record_fixture, _sample


def _case(tmp_path, groups=("g1",), *, music="N/A", offscreen=False, clip="clip-1", composition=None,
          text="Exact, text!", profile="Target bright voice", all_visible=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    intervals = [(i * .3, i * .3 + .2, group) for i, group in enumerate(groups)]
    args = _fixture(tmp_path, intervals, offscreen=offscreen)
    values = args["job"].model_dump(mode="json")
    values["clip_uid"] = clip
    for segment in values["segments"]:
        segment.update(source_start_sample=round(segment["start_time"] * 32000),
                       source_end_sample=round(segment["end_time"] * 32000), source_sample_rate_hz=32000, asr_text=text)
    args["job"] = _seal(MimoClipJob, values, "request_fingerprint")
    stem_values = args["stem_record"].model_dump(mode="json")
    stem_values["clip_uid"] = clip
    args["stem_record"] = _seal(SAMAudioStemRecord, stem_values, "record_fingerprint")
    payload = args["annotation"].model_dump(mode="json")
    if all_visible:
        for grounding in payload["av_grounding"]["segment_groundings"]:
            grounding.update(binding_status="visible_entity", entity_id="e1", speech_presentation="onscreen_spoken",
                             evidence_codes=["visible_lip_motion"])
    speakers = list(dict.fromkeys(groups))
    payload["audio_observation"]["speaker_voice_profiles"] = [
        {"speaker_group": group, "voice_characteristics": profile} for group in speakers
    ]
    payload["h3_semantics"]["shot1_caption"] = "<Subject 1> stands in a room. " + " ".join(
        f"{'An offscreen voice' if offscreen or group != 'g1' else '<Subject 1>'} (S{speakers.index(group)+1}) says, "
        f"<d>[English] {text}</d>" for group in groups
    )
    payload["h3_semantics"]["non_diegetic_music"] = music
    if composition:
        payload["audio_observation"]["segment_decisions"][0].update(
            vocal_composition=composition,
            secondary_vocal_activity={"present": True, "speaker_relation": "same_speaker" if composition == "same_speaker_nonlexical" else "uncertain", "kind": "sigh"},
        )
    args["annotation"] = MimoAVAnnotationDraft.model_validate(payload)
    sample = _sample(tmp_path)
    values = args["job"].model_dump(mode="json")
    values["target_video_sha256"] = sha256_file(Path(values["target_video_path"]))
    for reference in values["reference_images"]:
        reference["image_sha256"] = sha256_file(Path(reference["image_artifact_path"]))
    args["job"] = _seal(MimoClipJob, values, "request_fingerprint")
    sample_values = sample.model_dump(mode="json")
    sample_values.update(clip_uid=clip, sample_id=f"{clip}/in_pair", target_full_audio_path=args["job"].target_full_audio_path,
                         target_full_audio_sha256=args["job"].target_full_audio_sha256)
    for voice in sample_values["subject_voices"]:
        voice["target_occurrence_id"] = f"{clip}/{voice['entity_id']}"
    template = sample_values["speech_segments"][0]
    sample_values["speech_segments"] = [{
        **template, "segment_id": s.segment_id, "speaker_cluster_id": s.source_speaker_cluster_id,
        "source_start_sample": s.source_start_sample, "source_end_sample": s.source_end_sample,
        "source_sample_rate_hz": s.source_sample_rate_hz, "start_time": s.start_time, "end_time": s.end_time,
        "text": s.asr_text, "language": s.asr_language,
    } for s in args["job"].segments]
    sample = FinalH3SampleV2.model_validate(sample_values)
    record = _record_fixture(tmp_path, args["annotation"], job=args["job"])
    manifest = build_audio_reuse_assets(**args)
    path = args["output_root"] / "manifest.json"
    source = product.load_reuse_source(manifest_path=path, job=args["job"], record=record, stem_record=args["stem_record"])
    assert source.manifest == manifest
    return args, sample, source


def _render(sample, source, sources=None):
    refs, warnings = product.select_reuse_audio(sample, source, sources or {source.job.clip_uid: source})
    corrected, prompt, render_warnings = _materialize_sample(
        sample, source.job, source.record, conditioning_variant="target_speech_reuse",
        reuse_audio_contracts=[r.contract for r in refs],
    )
    return refs, warnings + render_warnings, corrected, prompt


@pytest.mark.parametrize("count,music", [(0, True), (1, False), (1, True), (2, True), (3, True), (4, True)])
def test_slot_capacity_exact_dialogue_and_projection(tmp_path, count, music):
    groups = tuple(f"g{i+1}" for i in range(count))
    _, sample, source = _case(tmp_path, groups, music="A gentle piano melody continues." if music else "N/A")
    before = (sample.model_dump(), source.record.model_dump())
    refs, warnings, corrected, prompt = _render(sample, source)
    expected = min(3, count + int(music))
    assert len(refs) == expected
    assert [r.contract.audio_index for r in refs] == list(range(1, expected + 1))
    assert all(r.contract.retention_marker == "partially_copy" for r in refs)
    assert [r.contract.speaker_id for r in refs if r.role == "speaker_speech_reuse"] == [f"S{i+1}" for i in range(min(3, count))]
    assert len(corrected) == count
    assert re.findall(r"<d>.*?</d>", prompt) == ["<d>[English] Exact, text!</d>"] * count
    detail = prompt.split("detailed_description:\n")[1].split("\n\noverall_soundscape:")[0]
    for ref in refs:
        if ref.role == "speaker_speech_reuse":
            assert detail.count(f"copied directly from {ref.contract.audio_label}") == 1
            assert "featuring" not in _canonical_audio_definition(ref.contract)
            assert ref.reuse_asset.source_sample_ranges[0].end_time - ref.reuse_asset.source_sample_ranges[0].start_time < 1
        else:
            assert ref.role == "music_reuse" and count < 3
            assert f"copied directly from {ref.contract.audio_label}" in prompt.split("non_diegetic_music:\n")[1]
    if music:
        assert "A gentle piano melody continues." in prompt
    if count >= 3 and music:
        assert "audio_capacity_omitted_music" in warnings
    if count > 3:
        assert "S4:audio_capacity_omitted_speaker" in warnings
    assert (sample.model_dump(), source.record.model_dump()) == before
    assert audio_task_prefix([r.contract for r in refs]) in prompt
    assert len(build_reference_contract(sample, "visual_only").pictures) + len(refs) <= 12


def test_repeated_speaker_relation_only_at_first_speech_and_offscreen_no_entity(tmp_path):
    _, sample, source = _case(tmp_path, ("g1", "g1"), offscreen=True)
    refs, _, _, prompt = _render(sample, source)
    assert refs[0].contract.entity_id is None and refs[0].contract.subject_label is None
    assert refs[0].contract.speaker_id == "S1"
    assert "synchronized speech track for (S1)" in prompt
    assert prompt.count("with the synchronized speech signal copied directly from <Audio 1>") == 1
    assert "An offscreen voice (S1), with the synchronized speech signal" in prompt
    assert prompt.count("<d>[English] Exact, text!</d>") == 2


@pytest.mark.parametrize("composition,allowed", [("same_speaker_nonlexical", True), ("secondary_non_speech_vocalization", False)])
def test_same_speaker_nonlexical_reuse_but_legacy_voice_unchanged(tmp_path, composition, allowed):
    _, sample, source = _case(tmp_path, composition=composition)
    decision = source.record.annotation.audio_observation.segment_decisions[0]
    grounding = source.record.annotation.segment_decisions[0]
    assert (speaker_ownership_reasons(decision) == []) is allowed
    assert _semantic_reasons(decision, grounding) == ["non_single_speaker", "secondary_vocal_activity"]
    refs, _, _, _ = _render(sample, source)
    assert len(refs) == int(allowed)


@pytest.mark.parametrize("donor_profile", [None, "A dry low voice."])
def test_cross_donor_profile_signal_and_mixed_music_ownership(tmp_path, donor_profile):
    _, sample, source = _case(tmp_path / "target", music="Quiet piano continues.")
    _, _, donor = _case(tmp_path / "donor", clip="donor", text="Unrelated donor dialogue.", profile=donor_profile)
    values = sample.model_dump(mode="json")
    values["pair_type"] = "cross_pair"
    values["subject_voices"][0].update(voice_source="cross_donor", donor_clip_uid="donor", donor_occurrence_id="donor/e1", donor_clip_display_path="01/show/donor")
    sample = FinalH3SampleV2.model_validate(values)
    refs, _, _, prompt = _render(sample, source, {source.job.clip_uid: source, "donor": donor})
    audio = refs[0]
    assert audio.contract.kind == "cross_voice" and audio.contract.retention_marker == "reference"
    assert audio.contract.path == donor.manifest.speakers[0].output_path
    assert audio.contract.path != source.manifest.speakers[0].output_path
    assert audio.contract.voice_characteristics == donor_profile
    assert audio.contract.entity_id == "e1" and audio.contract.speaker_id == "S1"
    assert audio.donor_occurrence_id == "donor/e1" and audio.donor_entity_id == "e1"
    assert audio.reuse_manifest_sha256 == donor.manifest_sha256
    assert "[reference generation + audio reference + audio reuse]" in prompt
    assert "<Subject 1> (S1), using the voice characteristics referenced from <Audio 1>, says" in prompt
    assert "<d>[English] Exact, text!</d>" in prompt
    assert "Unrelated donor dialogue" not in prompt
    assert "Target bright voice" not in _canonical_audio_definition(audio.contract)
    assert ".." not in _canonical_audio_definition(audio.contract)
    # Missing donor never substitutes target voice or another speaker.
    refs, warnings = product.select_reuse_audio(sample, source, {source.job.clip_uid: source})
    assert [r.contract.kind for r in refs] == ["music_reuse"]
    assert "S1:donor_reuse_unavailable" in warnings


@pytest.mark.parametrize("kind,retention", [("full_audio_reuse", "fully_copy"), ("music_reference", "reference")])
def test_legacy_audio_contract_wording_and_music_projection(tmp_path, kind, retention):
    _, sample, source = _case(tmp_path, music="Gentle piano continues.")
    audio = RecaptionAudioContract(audio_index=1, audio_label="<Audio 1>", kind=kind,
                                   path=source.job.target_full_audio_path, sha256=source.job.target_full_audio_sha256,
                                   retention_marker=retention, music_characteristics="gentle piano" if kind == "music_reference" else None)
    _, prompt, _ = _materialize_sample(sample, source.job, source.record, reuse_audio_contracts=[audio])
    if kind == "full_audio_reuse":
        _, legacy, _ = _materialize_sample(sample, source.job, source.record, conditioning_variant=kind)
        assert prompt == legacy
        assert _canonical_audio_retention(audio) == "<Audio 1>: fully_copy - the supplied full audio is reused in full."
    else:
        assert "follows the musical characteristics referenced from <Audio 1>. Gentle piano continues." in prompt
        assert "<Audio 1>: reference -" in prompt


@pytest.mark.parametrize("field", ["clip_uid", "source_job_fingerprint", "source_annotation_sha256", "source_stem_record_fingerprint"])
def test_manifest_lineage_mismatch_fails_closed(tmp_path, field):
    args, _, source = _case(tmp_path)
    values = source.manifest.model_dump(mode="json")
    values[field] = "wrong" if field == "clip_uid" else "0" * 64
    source.manifest_path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="lineage"):
        product.load_reuse_source(manifest_path=source.manifest_path, job=source.job, record=source.record, stem_record=args["stem_record"])


def test_asset_hash_and_range_mismatch_fail_closed(tmp_path):
    args, _, source = _case(tmp_path)
    values = source.manifest.model_dump(mode="json")
    values["speakers"][0]["source_sample_ranges"][0]["start_time"] = .01
    source.manifest_path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="ownership/timing"):
        product.load_reuse_source(manifest_path=source.manifest_path, job=source.job, record=source.record, stem_record=args["stem_record"])
    source.manifest_path.write_text(source.manifest.model_dump_json())
    Path(source.manifest.speakers[0].output_path).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        product.load_reuse_source(manifest_path=source.manifest_path, job=source.job, record=source.record, stem_record=args["stem_record"])


def _prepared(tmp_path):
    args, sample, source = _case(tmp_path / "case")
    canonical = sample.model_copy(update={"sample_id": "clip-1/canonical", "pair_type": "canonical", "pair_id": "canonical/clip-1", "subject_voices": []})
    root = tmp_path / "prepared"
    h3, mimo, stems, reuse = [root / name for name in ("h3", "mimo", "separation", "reuse")]
    for path in (h3, mimo, stems, reuse):
        path.mkdir(parents=True)
    samples_file = h3 / "samples.jsonl"
    samples_file.write_text(canonical.model_dump_json() + "\n" + sample.model_dump_json() + "\n")
    values = source.job.model_dump(mode="json")
    values["source_h3_sample_ids"] = [canonical.sample_id, sample.sample_id]
    job = _seal(MimoClipJob, values, "request_fingerprint")
    iv = {name: "a" * 64 for name in (
        "source_visual_inventory_sha256", "source_canonical_audio_manifest_sha256", "source_diarization_raw_segments_sha256",
        "source_diarization_bound_segments_sha256", "source_qwen3_asr_segments_sha256", "source_binding_audit_segments_sha256",
    )}
    inventory = _inventory({**iv, "schema_version": "r2v.h3.mimo25_inventory.4", "inventory_scope": "explicit_case_subset",
                            "canonical_wide_coverage": False, "source_h3_samples_sha256": sha256_file(samples_file),
                            "clip_count": 1, "jobs": [job.model_dump(mode="json")]})
    record = _record_fixture(tmp_path / "case", source.record.annotation, job=job, inventory_fingerprint=inventory.inventory_fingerprint)
    (mimo / "inventory.json").write_text(inventory.model_dump_json())
    (mimo / "records.jsonl").write_text(record.model_dump_json() + "\n")
    (stems / "records.jsonl").write_text(args["stem_record"].model_dump_json() + "\n")
    build_audio_reuse_assets(**{**args, "job": job, "output_root": reuse / job.clip_uid})
    return {"mimo_root": mimo, "source_h3_root": h3, "separation_root": stems, "reuse_root": reuse,
            "audio_production_root": args["audio_production_root"],
            "output_root": args["audio_production_root"] / "sam_audio_stem_shadow_v1/runs/test/products"}


def test_model_free_cli_atomic_products_and_input_immutability(tmp_path):
    args = _prepared(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    summary = product.materialize_audio_reuse_products(**args, enable_full_audio_reuse=True)
    assert (summary.sample_count, summary.ready_count, summary.failed_count) == (3, 3, 0)
    assert summary.model_call_count == 0 and summary.production_artifacts_modified is False
    rows = product._rows(args["output_root"] / "records.jsonl", product.AudioReuseProduct)
    assert [r.conditioning_variant for r in rows] == ["visual_only", "target_speech_reuse", "full_audio_reuse"]
    assert summary.audio_kind_counts == {"full_audio_reuse": 1, "speaker_speech_reuse": 1}
    assert before == {p: sha256_file(p) for p in before}
    with pytest.raises(FileExistsError):
        product.materialize_audio_reuse_products(**args)
    from tools.materialize_h3_audio_reuse_shadow import main
    argv = [part for key, value in args.items() for part in ("--" + key.replace("_", "-"), str(value))]
    argv[-1] += "-second"
    assert main(argv)["model_call_count"] == 0


def test_no_publication_when_lineage_changes(tmp_path):
    args = _prepared(tmp_path)
    manifest_path = args["reuse_root"] / "clip-1/manifest.json"
    values = json.loads(manifest_path.read_text())
    values["source_annotation_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="lineage"):
        product.materialize_audio_reuse_products(**args)
    assert not args["output_root"].exists()


def test_ambiguous_donor_asset_is_excluded_without_fallback(tmp_path):
    _, sample, source = _case(tmp_path / "target")
    _, _, donor = _case(tmp_path / "donor", ("g1", "g2"), clip="donor", all_visible=True)
    values = sample.model_dump(mode="json")
    values["pair_type"] = "cross_pair"
    values["subject_voices"][0].update(voice_source="cross_donor", donor_clip_uid="donor", donor_occurrence_id="donor/e1", donor_clip_display_path="01/show/donor")
    refs, warnings = product.select_reuse_audio(FinalH3SampleV2.model_validate(values), source, {"donor": donor})
    assert refs == []
    assert warnings == ["S1:donor_speaker_reuse_not_unique"]


@pytest.mark.parametrize("music", ["N/A", "unknown", ""])
def test_music_stem_existence_is_not_semantic_presence(tmp_path, music):
    _, sample, source = _case(tmp_path, (), music=music)
    assert Path(source.manifest.music.output_path).is_file()
    if not music:
        assert product.select_reuse_audio(sample, source, {})[0] == []
        with pytest.raises(ValueError, match="must not be empty"):
            _render(sample, source)
        return
    refs, _, _, prompt = _render(sample, source)
    assert refs == [] and "<Audio " not in prompt
    assert "[reference generation]" in prompt


@pytest.mark.parametrize("updates", [
    {"speaker_id": None}, {"retention_marker": "reference"},
    {"voice_characteristics": "low voice"}, {"entity_id": "e1", "subject_label": None},
])
def test_signal_reuse_contract_rejects_identity_reference_semantics(updates):
    values = {"audio_index": 1, "audio_label": "<Audio 1>", "kind": "speaker_speech_reuse",
              "path": "/synthetic/reuse.flac", "sha256": "a" * 64, "speaker_id": "S1",
              "retention_marker": "partially_copy"}
    with pytest.raises(ValueError):
        RecaptionAudioContract(**{**values, **updates})


def test_manifest_frame_and_sample_format_mismatch(tmp_path):
    import soundfile as sf

    args, _, source = _case(tmp_path)
    path = Path(source.manifest.speakers[0].output_path)
    pcm, rate = sf.read(path, dtype="int16", always_2d=True)
    for data, sample_rate, message in [(pcm[:-1], rate, "frame count"), (pcm, 16000, "32 kHz")]:
        sf.write(path, data, sample_rate, subtype="PCM_16")
        values = source.manifest.model_dump(mode="json")
        values["speakers"][0]["output_sha256"] = sha256_file(path)
        source.manifest_path.write_text(json.dumps(values))
        with pytest.raises(ValueError, match=message):
            product.load_reuse_source(manifest_path=source.manifest_path, job=source.job, record=source.record, stem_record=args["stem_record"])


def test_late_input_change_cannot_publish_partial_product_stage(tmp_path, monkeypatch):
    args = _prepared(tmp_path)
    original = product._materialize_sample

    def mutate(*a, **kw):
        result = original(*a, **kw)
        (args["mimo_root"] / "records.jsonl").write_text("changed fixture")
        return result

    monkeypatch.setattr(product, "_materialize_sample", mutate)
    with pytest.raises(ValueError, match="changed during materialization"):
        product.materialize_audio_reuse_products(**args)
    assert not args["output_root"].exists()
