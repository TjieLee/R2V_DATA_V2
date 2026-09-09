from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse as reuse
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob, MimoSegmentEvidence
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemRecord,
    StemArtifact,
    sha256_file,
)
from tests.test_h3_mimo25_av_shadow import _annotation, _job_fixture


def _seal(model, values, field):
    values = {k: v for k, v in values.items() if k != field}
    values[field] = hashlib.sha256(json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return model.model_validate(values)


def _signal(frames, offset=0):
    samples = np.arange(frames, dtype=np.int32)
    return np.column_stack(((samples + offset) % 20001 - 10000,
                            (samples * 3 + offset) % 21001 - 10500)).astype(np.int16)


def _write(path, pcm, *, container="FLAC"):
    sf.write(str(path), pcm, 32000, subtype="PCM_16", format=container)
    return sha256_file(path)


def _fixture(tmp_path, intervals=((0.1, 0.3, "g1"), (0.9, 1.1, "g1")), *, drift=0, offscreen=False, mixed=False,
             target_subtype="PCM_16", target_format="FLAC", target_rate=32000, target_channels=2):
    production = tmp_path / "production"
    production.mkdir()
    target = production / "target.flac"
    sf.write(str(target), _signal(64000)[:, :target_channels], target_rate,
             subtype=target_subtype, format=target_format)
    target_hash = sha256_file(target)
    job = _job_fixture(tmp_path)
    template = _annotation().model_dump(mode="json")
    segments, decisions, groundings, views = [], [], [], []
    for i, (start, end, group) in enumerate(intervals, 1):
        sid = f"segment_{i}"
        segment = job.segments[0].model_dump(mode="json")
        segment.update(segment_id=sid, start_time=start, end_time=end,
                       source_start_sample=round(start * 16000), source_end_sample=round(end * 16000),
                       source_sample_rate_hz=16000, direct_anchor_seconds=0,
                       identity_scope="unresolved", cluster_binding_status="unbound")
        segments.append(MimoSegmentEvidence.model_validate(segment).model_dump(mode="json"))
        decision = copy.deepcopy(template["audio_observation"]["segment_decisions"][0])
        decision.update(segment_id=sid, primary_speaker_group=group)
        if mixed and i == 1:
            decision.update(vocal_composition="sequential_multi_speaker_speech", resolution="needs_acoustic_refinement",
                            secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"})
        decisions.append(decision)
        grounding = copy.deepcopy(template["av_grounding"]["segment_groundings"][0])
        grounding.update(segment_id=sid, primary_speaker_group=group)
        if offscreen or group != "g1":
            grounding.update(binding_status="offscreen", entity_id=None,
                             speech_presentation="offscreen_spoken", evidence_codes=["offscreen_audio"])
        groundings.append(grounding)
        view = copy.deepcopy(template["visual_observation"]["segment_views"][0])
        view["segment_id"] = sid
        views.append(view)
    values = job.model_dump(mode="json")
    values.update(target_full_audio_path=str(target), target_full_audio_sha256=target_hash,
                  target_duration_seconds=2.0, segments=segments)
    job = _seal(MimoClipJob, values, "request_fingerprint")
    template["audio_observation"]["segment_decisions"] = decisions
    template["av_grounding"]["segment_groundings"] = groundings
    template["visual_observation"]["segment_views"] = views
    annotation = MimoAVAnnotationDraft.model_validate(template)
    stems = []
    frames = 64000 + round(drift * 32000)
    for kind, offset in (("music", 7), ("speech", 11), ("sfx", 19)):
        path = production / f"{kind}.wav"
        digest = _write(path, _signal(frames, offset), container="WAV")
        raw_path = production / f"{kind}-raw.flac"
        raw_digest = _write(raw_path, _signal(frames, offset)[:, 0])
        duration = frames / 32000
        delta = duration - 2.0
        stems.append(StemArtifact(
            stem_type=kind, source_audio_path=str(target), source_audio_sha256=target_hash,
            source_end_sample=64000, source_duration_seconds=2.0,
            raw_stem_path=str(raw_path), raw_stem_sha256=raw_digest, raw_sample_rate_hz=32000,
            raw_channels=1, raw_frame_count=frames, raw_duration_seconds=duration,
            canonical_stem_path=str(path), canonical_stem_sha256=digest,
            canonical_frame_count=frames, canonical_duration_seconds=duration,
            raw_duration_delta_seconds=delta, canonical_duration_delta_seconds=delta,
            canonical_uncovered_tail_seconds=max(0, -delta),
            alignment_state="accepted_with_alignment_warning" if abs(delta) > .05 else "normal_bounded_drift",
            conversion_settings=["synthetic PCM fixture"],
        ).model_dump(mode="json"))
    calls = [{"pass_index": i, "prompt": prompt, "input_path": str(target), "input_sha256": target_hash,
                  "target_path": str(target), "target_sha256": target_hash,
                  "residual_path": str(target), "residual_sha256": target_hash,
                  "verification_state": "unverified", "candidate_count": None, "selected_candidate_index": None,
                  "judge_scores": None, "clap_score": None, "runtime_seconds": 0.0, "backend_metadata": {}}
             for i, prompt in ((1, "music soundtrack"), (2, "human voices"))]
    record = _seal(SAMAudioStemRecord, {
        "schema_version": "r2v.h3.sam_audio_stem_record.3", "clip_uid": job.clip_uid, "route": "music_first",
        "inventory_fingerprint": "a" * 64, "model_configuration_fingerprint": "b" * 64,
        "separation_state": "unverified", "model_call_count": 2, "calls": calls, "stems": stems, "failure_reason": None,
    }, "record_fingerprint")
    return {"job": job, "annotation": annotation, "stem_record": record,
                "audio_production_root": production, "output_root": tmp_path / "reuse-shadow", "allow_unverified": True}


def _pcm(path, *, container="FLAC"):
    pcm, rate = sf.read(str(path), dtype="int16", always_2d=True)
    assert rate == 32000 and pcm.shape[1] == 2
    assert sf.info(str(path)).format == container
    return pcm


def test_multiple_short_intervals_preserve_exact_samples_and_silence(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import mimo25_recovered_voice

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy voice quality must not run")

    monkeypatch.setattr(mimo25_recovered_voice, "_quality_reasons", forbidden)
    args = _fixture(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    input_models = {k: v.model_dump() for k, v in args.items() if hasattr(v, "model_dump")}
    result = reuse.build_audio_reuse_assets(**args)
    assert len(result.speakers) == 1
    asset = result.speakers[0]
    assert (asset.speaker_group, asset.speaker_id, asset.entity_id, asset.subject_index) == ("g1", "S1", "e1", 1)
    assert asset.source_segment_ids == ["segment_1", "segment_2"]
    assert asset.target_frame_count == 64000 and asset.target_duration_seconds == 2
    assert asset.timeline_preserved and asset.silence_fill
    source = _pcm(asset.source_stem_path, container="WAV")
    expected = np.zeros((64000, 2), dtype=np.int16)
    for r in asset.source_sample_ranges:
        assert r.target_start_sample == r.source_start_sample * 2
        assert r.target_end_sample == r.source_end_sample * 2
        assert r.target_end_sample - r.target_start_sample == 6400  # 0.2 s, no 1 s gate.
        expected[r.target_start_sample:r.target_end_sample] = source[r.target_start_sample:r.target_end_sample]
    np.testing.assert_array_equal(_pcm(asset.output_path), expected)
    assert asset.output_sha256 == sha256_file(Path(asset.output_path))
    assert result.production_artifacts_modified is False and result.model_call_count == 0
    assert not result.exclusions
    assert reuse.AudioReuseManifest.model_validate_json((args["output_root"] / "manifest.json").read_text()) == result
    assert {p: sha256_file(p) for p in before} == before
    assert {k: args[k].model_dump() for k in input_models} == input_models
    with pytest.raises(FileExistsError):
        reuse.build_audio_reuse_assets(**args)


def test_two_speakers_are_isolated_and_offscreen_is_not_identity_bound(tmp_path):
    args = _fixture(tmp_path, ((0.1, .3, "g1"), (.5, .7, "g2"), (1.0, 1.1, "g1")), offscreen=True)
    result = reuse.build_audio_reuse_assets(**args)
    assert [(s.speaker_group, s.speaker_id) for s in result.speakers] == [("g1", "S1"), ("g2", "S2")]
    for asset in result.speakers:
        assert asset.entity_id is None and asset.subject_index is None
        expected = np.zeros((64000, 2), dtype=np.int16)
        for r in asset.source_sample_ranges:
            expected[r.target_start_sample:r.target_end_sample] = _pcm(asset.source_stem_path, container="WAV")[r.target_start_sample:r.target_end_sample]
        np.testing.assert_array_equal(_pcm(asset.output_path), expected)
    assert not np.any(np.any(_pcm(result.speakers[0].output_path) != 0, axis=1)
                      & np.any(_pcm(result.speakers[1].output_path) != 0, axis=1))


@pytest.mark.parametrize("drift", [-.025, .032, .08, 0])
def test_music_target_timeline_padding_and_truncation(tmp_path, drift):
    result = reuse.build_audio_reuse_assets(**_fixture(tmp_path, drift=drift))
    asset = result.music
    expected = np.zeros((64000, 2), dtype=np.int16)
    source = _pcm(asset.source_stem_path, container="WAV")
    copied = min(64000, len(source))
    expected[:copied] = source[:copied]
    np.testing.assert_array_equal(_pcm(asset.output_path), expected)
    assert asset.copied_frame_count == copied
    assert asset.silence_tail_frame_count == max(0, -round(drift * 32000))
    assert asset.truncated_tail_frame_count == max(0, round(drift * 32000))
    assert asset.semantic_presence_evaluated is False


def test_speech_outside_short_stem_is_not_truncated_or_published(tmp_path):
    args = _fixture(tmp_path, ((1.8, 2.0, "g1"),), drift=-.025)
    result = reuse.build_audio_reuse_assets(**args)
    assert result.speakers == []
    assert [e.reason for e in result.exclusions] == ["speech_range_outside_stem"]
    assert list(args["output_root"].glob("S*.flac")) == []


def test_multi_speaker_interval_excluded_without_removing_upstream_fact(tmp_path):
    args = _fixture(tmp_path, mixed=True)
    result = reuse.build_audio_reuse_assets(**args)
    assert result.speakers[0].source_segment_ids == ["segment_2"]
    assert {e.reason for e in result.exclusions} == {"non_single_speaker", "secondary_vocal_activity"}
    assert not np.any(_pcm(result.speakers[0].output_path)[3200:9600])
    assert args["annotation"].audio_observation.segment_decisions[0].vocal_composition == "sequential_multi_speaker_speech"


def test_cross_speaker_overlap_excludes_both_tracks(tmp_path):
    result = reuse.build_audio_reuse_assets(**_fixture(tmp_path, ((.1, .5, "g1"), (.4, .6, "g2"))))
    assert result.speakers == []
    assert {(e.segment_id, e.reason) for e in result.exclusions} == {
        ("segment_1", "cross_speaker_overlap"), ("segment_2", "cross_speaker_overlap"),
    }


def test_production_root_cannot_be_output(tmp_path):
    args = _fixture(tmp_path)
    from r2v_data_v2.h3.jea_audio_production import jea_production_paths

    paths = jea_production_paths(args["audio_production_root"])
    for path in (paths.root, tmp_path, paths.audio, paths.primary_voice, paths.embedding,
                 paths.pairs, paths.diarization, paths.asr, paths.h3, paths.h3 / "new-assets"):
        with pytest.raises(ValueError, match="separate"):
            reuse.build_audio_reuse_assets(**{**args, "output_root": path})


def test_shadow_output_under_production_is_allowed_and_immutable(tmp_path):
    args = _fixture(tmp_path)
    production = args["audio_production_root"]
    before = {p: sha256_file(p) for p in production.rglob("*") if p.is_file()}
    output = production / "sam_audio_stem_shadow_v1/runs/pilot/audio_reuse_assets" / args["job"].clip_uid
    result = reuse.build_audio_reuse_assets(**{**args, "output_root": output})
    assert result.production_artifacts_modified is False
    assert result.speakers and (output / "manifest.json").is_file()
    assert before == {p: sha256_file(p) for p in before}
    published = {p: sha256_file(p) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        reuse.build_audio_reuse_assets(**{**args, "output_root": output})
    assert published == {p: sha256_file(p) for p in published}


def test_source_media_collision_and_symlink_production_collision(tmp_path):
    args = _fixture(tmp_path)
    source = Path(args["stem_record"].stem("speech").canonical_stem_path)
    before = sha256_file(source)
    with pytest.raises(FileExistsError):
        reuse.build_audio_reuse_assets(**{**args, "output_root": source})
    with pytest.raises(ValueError, match="source media"):
        reuse.build_audio_reuse_assets(**{**args, "output_root": source / "assets"})
    protected = args["audio_production_root"] / "audio"
    protected.mkdir()
    alias = tmp_path / "shadow-alias"
    alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="production stages"):
        reuse.build_audio_reuse_assets(**{**args, "output_root": alias / "assets"})
    assert sha256_file(source) == before
    assert not (protected / "assets").exists()


@pytest.mark.parametrize("group", [None, "g1"])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
def test_shared_ownership_preserves_legacy_semantic_reasons(group, single, secondary):
    from r2v_data_v2.h3.mimo25_recovered_voice import _semantic_reasons
    from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons

    annotation = _annotation()
    original = annotation.audio_observation.segment_decisions[0]
    decision = original.model_copy(update={
        "primary_speaker_group": group,
        "vocal_composition": "single_speaker" if single else "sequential_multi_speaker_speech",
        "secondary_vocal_activity": original.secondary_vocal_activity.model_copy(update={"present": secondary}),
    })
    expected = ([] if group else ["unresolved"])
    expected += [] if single else ["non_single_speaker"]
    expected += ["secondary_vocal_activity"] if secondary else []
    assert speaker_ownership_reasons(decision) == expected
    grounding = annotation.av_grounding.segment_groundings[0]
    assert _semantic_reasons(decision, grounding) == expected
    offscreen = grounding.model_copy(update={
        "binding_status": "offscreen", "entity_id": None, "speech_presentation": "offscreen_spoken",
    })
    assert _semantic_reasons(decision, offscreen) == (
        ([] if group else ["unresolved"]) + ["not_visible_entity", "not_onscreen_spoken"]
        + ([] if single else ["non_single_speaker"])
        + (["secondary_vocal_activity"] if secondary else [])
    )


def test_tampered_source_fails_without_publication(tmp_path):
    args = _fixture(tmp_path)
    Path(args["stem_record"].stem("speech").canonical_stem_path).write_bytes(b"changed")
    with pytest.raises(reuse.AudioReuseIntegrityError, match="hash_mismatch"):
        reuse.build_audio_reuse_assets(**args)
    assert not args["output_root"].exists()


def test_unverified_requires_opt_in(tmp_path):
    args = _fixture(tmp_path)
    with pytest.raises(ValueError, match="allow_unverified"):
        reuse.build_audio_reuse_assets(**{**args, "allow_unverified": False})


def test_write_verification_failure_is_atomic(tmp_path, monkeypatch):
    args = _fixture(tmp_path)
    before = {p: sha256_file(p) for p in args["audio_production_root"].rglob("*") if p.is_file()}
    original = sf.write

    def corrupt(path, pcm, *a, **kw):
        original(path, pcm * 0, *a, **kw)

    monkeypatch.setattr(sf, "write", corrupt)
    with pytest.raises(reuse.AudioReuseIntegrityError, match="decoded_samples_differ"):
        reuse.build_audio_reuse_assets(**args)
    assert not args["output_root"].exists()
    assert not list(tmp_path.glob(".audio-reuse-*"))
    assert {p: sha256_file(p) for p in before} == before


def test_output_pcm_and_metadata_deterministic(tmp_path):
    args = _fixture(tmp_path)
    first = reuse.build_audio_reuse_assets(**args)
    second = reuse.build_audio_reuse_assets(**{**args, "output_root": tmp_path / "other-shadow"})
    for a, b in zip([*first.speakers, first.music], [*second.speakers, second.music], strict=True):
        assert a.model_dump(exclude={"output_path"}) == b.model_dump(exclude={"output_path"})


@pytest.mark.parametrize("duration", [.1, .2, .7])
def test_short_speech_has_no_legacy_minimum_duration(tmp_path, duration):
    result = reuse.build_audio_reuse_assets(**_fixture(tmp_path, ((0.0, duration, "g1"),)))
    assert len(result.speakers) == 1 and not result.exclusions
    assert result.speakers[0].source_sample_ranges[0].target_end_sample == round(duration * 32000)


def test_nontranscribed_interval_not_copied(tmp_path):
    args = _fixture(tmp_path)
    values = args["job"].model_dump(mode="json")
    values["segments"][0].update(asr_status="empty", asr_text=None, asr_language=None)
    args["job"] = _seal(MimoClipJob, values, "request_fingerprint")
    result = reuse.build_audio_reuse_assets(**args)
    assert result.speakers[0].source_segment_ids == ["segment_2"]
    assert not np.any(_pcm(result.speakers[0].output_path)[3200:9600])


def test_nontranscribed_other_speaker_overlap_is_not_copied(tmp_path):
    args = _fixture(tmp_path, ((.1, .5, "g1"), (.4, .6, "g2")))
    values = args["job"].model_dump(mode="json")
    values["segments"][1].update(asr_status="empty", asr_text=None, asr_language=None)
    args["job"] = _seal(MimoClipJob, values, "request_fingerprint")
    result = reuse.build_audio_reuse_assets(**args)
    assert result.speakers == []
    assert "cross_speaker_overlap" in {e.reason for e in result.exclusions}


def test_speech_outside_target_is_never_truncated(tmp_path):
    result = reuse.build_audio_reuse_assets(**_fixture(tmp_path, ((1.9, 2.025, "g1"),), drift=.032))
    assert not result.speakers
    assert [e.reason for e in result.exclusions] == ["speech_range_outside_target"]


def test_bound_stem_drift_limit_not_relaxed(tmp_path):
    with pytest.raises(ValueError, match="duration differs"):
        _fixture(tmp_path, drift=.12)


def test_conflicting_entity_provenance_does_not_infer_identity(tmp_path):
    args = _fixture(tmp_path)
    values = args["annotation"].model_dump(mode="json")
    values["av_grounding"]["segment_groundings"][1]["entity_id"] = "e2"
    args["annotation"] = MimoAVAnnotationDraft.model_validate(values)
    result = reuse.build_audio_reuse_assets(**args)
    assert result.speakers == []
    assert {e.reason for e in result.exclusions} == {"conflicting_entity_provenance"}


def test_missing_segment_decision_fails_closed(tmp_path):
    args = _fixture(tmp_path)
    values = args["annotation"].model_dump(mode="json")
    values["audio_observation"]["segment_decisions"].pop()
    args["annotation"] = MimoAVAnnotationDraft.model_validate(values)
    with pytest.raises(reuse.AudioReuseIntegrityError, match="segment_inventory_mismatch"):
        reuse.build_audio_reuse_assets(**args)
    assert not args["output_root"].exists()


def test_pcm24_target_is_probed_without_pcm_decode(tmp_path, monkeypatch):
    args = _fixture(tmp_path, target_subtype="PCM_24")
    target = Path(args["job"].target_full_audio_path)
    before = {p: sha256_file(p) for p in args["audio_production_root"].rglob("*") if p.is_file()}
    original_read = sf.read

    def read_stem_only(path, *a, **kw):
        assert Path(path) != target, "target timeline must not be decoded to PCM16"
        return original_read(path, *a, **kw)

    monkeypatch.setattr(sf, "read", read_stem_only)
    result = reuse.build_audio_reuse_assets(**args)
    assert sf.info(target).subtype == "PCM_24"
    for asset in [*result.speakers, result.music]:
        stem_info = sf.info(asset.source_stem_path)
        assert (stem_info.format, stem_info.subtype, stem_info.samplerate, stem_info.channels) == (
            "WAV", "PCM_16", 32000, 2,
        )
        info = sf.info(asset.output_path)
        assert (info.frames, info.format, info.samplerate, info.channels, info.subtype) == (
            sf.info(target).frames, "FLAC", 32000, 2, "PCM_16",
        )
        assert asset.target_frame_count == 64000
    expected = np.zeros((64000, 2), dtype=np.int16)
    speaker = result.speakers[0]
    source = _pcm(speaker.source_stem_path, container="WAV")
    for interval in speaker.source_sample_ranges:
        start, end = interval.target_start_sample, interval.target_end_sample
        expected[start:end] = source[start:end]
    np.testing.assert_array_equal(_pcm(speaker.output_path), expected)
    np.testing.assert_array_equal(_pcm(result.music.output_path), _pcm(result.music.source_stem_path, container="WAV"))
    assert before == {p: sha256_file(p) for p in before}


@pytest.mark.parametrize("options", [
    {"target_format": "WAV"}, {"target_rate": 16000}, {"target_channels": 1},
])
def test_invalid_canonical_target_format_fails(tmp_path, options):
    args = _fixture(tmp_path, **options)
    with pytest.raises(reuse.AudioReuseIntegrityError, match="target_requires_32k_stereo_flac"):
        reuse.build_audio_reuse_assets(**args)
    assert not args["output_root"].exists()


def test_canonical_target_hash_is_verified(tmp_path):
    args = _fixture(tmp_path, target_subtype="PCM_24")
    target = Path(args["job"].target_full_audio_path)
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(reuse.AudioReuseIntegrityError, match="hash_mismatch"):
        reuse.build_audio_reuse_assets(**args)


@pytest.mark.parametrize("kind", ["speech", "music"])
@pytest.mark.parametrize("container,subtype", [("WAV", "PCM_24"), ("FLAC", "PCM_16")])
def test_invalid_stem_rejected_before_generation(tmp_path, kind, container, subtype):
    args = _fixture(tmp_path, target_subtype="PCM_24")
    values = args["stem_record"].model_dump(mode="json")
    stem = next(s for s in values["stems"] if s["stem_type"] == kind)
    path = Path(stem["canonical_stem_path"])
    sf.write(path, _pcm(path, container="WAV"), 32000, subtype=subtype, format=container)
    stem["canonical_stem_sha256"] = sha256_file(path)
    args["stem_record"] = _seal(SAMAudioStemRecord, values, "record_fingerprint")
    with pytest.raises(reuse.AudioReuseIntegrityError, match="PCM16 WAV"):
        reuse.build_audio_reuse_assets(**args)
    assert not args["output_root"].exists()
