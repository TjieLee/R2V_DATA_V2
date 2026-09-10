"""Frozen, typed product fixtures for unified QA; no inference or extraction."""
import json
from collections import Counter
from pathlib import Path

import pytest

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.audio_reuse import (
    AudioReuseManifest,
    MusicReuseAsset,
    SpeakerSpeechReuseAsset,
)
from r2v_data_v2.h3.audio_reuse_materializer import (
    AudioReuseProduct,
    AudioReuseProductSummary,
    ReuseAudioReference,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    load_reconcile_sources,
    project_prepared_samples,
)
from r2v_data_v2.h3.audio_shadow_qa_variants import finalize_registry
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3ShadowRecord,
    MimoH3ShadowSummary,
    _extra_audio_contract,
)
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionAudioContract, audio_task_prefix
from tests.test_h3_audio_shadow_qa import _fixture, _snapshot


def _inputs(kwargs, shadow):
    base = qa.build_mimo25_inventory()
    provenance, _, _ = qa.validate_stem_diarization_lineage(shadow / "diarization")
    jobs = qa.build_stem_reconcile_jobs(
        base_inventory=qa.usable_stem_reconcile_inventory(base, provenance),
        stem_diarization_root=shadow / "diarization", stem_asr_root=shadow / "asr", route=provenance.route,
    )
    samples = qa._rows(kwargs["audio_production_root"] / "h3/samples.jsonl", qa.FinalH3SampleV2)
    records = qa._rows(shadow / qa.MIMO25_STEM_RECONCILE_STAGE / "records.jsonl", qa.MimoStemReconcileRecord)
    return base, jobs, samples, records


def _seal(model, **values):
    # Include default fields in the publisher's canonical fingerprint.
    defaults = {key: f.default for key, f in model.model_fields.items() if not f.is_required()}
    values = {**defaults, **values}
    values.pop("record_fingerprint", None)
    return model(**values, record_fingerprint=qa._fingerprint(values))


def _write_products(kwargs, shadow):
    base, jobs, samples, records = _inputs(kwargs, shadow)
    prepared = shadow / "prepared-fixture"
    (prepared / "h3").mkdir(parents=True)
    projected = project_prepared_samples(jobs, samples)
    sample_path = prepared / "h3/samples.jsonl"
    sample_path.write_text("".join(s.model_dump_json() + "\n" for s in projected))
    inv = qa._inventory({**base.model_dump(mode="json", exclude={"inventory_fingerprint"}),
                         "jobs": [j.model_dump(mode="json") for j in jobs],
                         "source_h3_samples_sha256": qa.sha256_file(sample_path)})
    (prepared / "inventory.json").write_text(inv.model_dump_json())
    prepared_records = load_reconcile_sources(shadow / qa.MIMO25_STEM_RECONCILE_STAGE)
    (prepared / "records.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in prepared_records))
    job, record = jobs[0], records[0]
    sample = next(s for s in projected if s.clip_uid == job.clip_uid and s.pair_type == "canonical")
    _, _, stems = qa.validate_stem_diarization_lineage(shadow / "diarization")
    stem = next(s for s in stems if s.clip_uid == job.clip_uid)
    assets_root = shadow / "audio_reuse_assets_v1" / job.clip_uid
    assets_root.mkdir(parents=True)
    assets = {}
    for kind, model in (("speech", SpeakerSpeechReuseAsset), ("music", MusicReuseAsset)):
        source = stem.stem(kind)
        output = assets_root / (kind + ".flac")
        sf = pytest.importorskip("soundfile")
        pcm, rate = sf.read(source.canonical_stem_path, dtype="int16", always_2d=True)
        sf.write(output, pcm, rate, subtype="PCM_16", format="FLAC")
        values = {"clip_uid": job.clip_uid, "source_stem_path": source.canonical_stem_path,
                  "source_stem_sha256": source.canonical_stem_sha256, "source_stem_record_fingerprint": stem.record_fingerprint,
                  "source_stem_verification_state": stem.separation_state, "source_stem_frame_count": len(pcm),
                  "target_audio_path": job.target_full_audio_path, "target_audio_sha256": job.target_full_audio_sha256,
                  "target_frame_count": len(pcm), "target_duration_seconds": len(pcm)/32000,
                  "output_path": str(output), "output_sha256": qa.sha256_file(output)}
        if kind == "speech":
            segment = job.segments[0]
            values.update(speaker_group="g1", speaker_id="S1", entity_id="e1", subject_index=1,
                          source_segment_ids=[segment.segment_id], source_job_fingerprint=job.request_fingerprint,
                          source_annotation_sha256=qa._fingerprint(record.annotation.model_dump(mode="json")),
                          source_sample_ranges=[{**{k: getattr(segment, k) for k in (
                              "segment_id", "source_start_sample", "source_end_sample", "source_sample_rate_hz", "start_time", "end_time")},
                              "target_start_sample": round(segment.start_time*32000), "target_end_sample": round(segment.end_time*32000)}])
        else:
            values.update(copied_frame_count=len(pcm), silence_tail_frame_count=0, truncated_tail_frame_count=0)
        assets[kind] = model(**values)
    manifest = AudioReuseManifest(
        clip_uid=job.clip_uid, source_job_fingerprint=job.request_fingerprint,
        source_annotation_sha256=qa._fingerprint(record.annotation.model_dump(mode="json")),
        source_stem_record_fingerprint=stem.record_fingerprint, speakers=[assets["speech"]], music=assets["music"], exclusions=[],
    )
    manifest_path = assets_root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    products = []
    for kind, variant in (("speaker_speech_reuse", "target_speech_reuse"), ("music_reuse", "target_speech_reuse"),
                          ("cross_voice", "cross_voice_reference"), ("full_audio_reuse", "full_audio_reuse")):
        asset = None if kind == "full_audio_reuse" else assets["music" if kind == "music_reuse" else "speech"]
        contract = RecaptionAudioContract(
            audio_label="<Audio 1>", audio_index=1, kind=kind,
            path=asset.output_path if asset else job.target_full_audio_path,
            sha256=asset.output_sha256 if asset else job.target_full_audio_sha256,
            retention_marker="fully_copy" if asset is None else "reference" if kind == "cross_voice" else "partially_copy",
            **({"entity_id": "e1", "subject_label": "<Subject 1>", "speaker_id": "S1"}
               if kind in {"cross_voice", "speaker_speech_reuse"} else {}),
        )
        ref = ReuseAudioReference(
            contract=contract, role="voice_reference" if kind == "cross_voice" else kind,
            source_type={"cross_voice": "donor_speaker_reuse_asset", "speaker_speech_reuse": "target_speaker_reuse_asset",
                         "music_reuse": "target_music_reuse_asset", "full_audio_reuse": "canonical_full_audio"}[kind],
            source_clip_uid=job.clip_uid, source_mimo_record_fingerprint=record.record_fingerprint,
            **({"reuse_asset": asset, "reuse_manifest_path": str(manifest_path),
                "reuse_manifest_sha256": qa.sha256_file(manifest_path),
                "reuse_manifest_fingerprint": qa._fingerprint(manifest.model_dump(mode="json"))} if asset else {}),
            **({"donor_occurrence_id": "donor/e1", "donor_entity_id": "e1", "donor_clip_display_path": "01/show/donor"}
               if kind == "cross_voice" else {}),
        )
        refs = [ref]
        if kind == "music_reuse":
            # Music is a companion, never a standalone target-speech product.
            refs = [products[0].audio_references[0], ref.model_copy(update={
                "contract": contract.model_copy(update={"audio_index": 2, "audio_label": "<Audio 2>"})})]
        corrected, text, warnings = qa._materialize_sample(
            sample, job, qa._MaterializerInput(record.annotation, job.request_fingerprint),
            conditioning_variant=variant, reuse_audio_contracts=[r.contract for r in refs],
        )
        products.append(_seal(AudioReuseProduct,
            sample_id=job.clip_uid + "/" + kind, source_h3_sample_id=sample.sample_id,
            source_h3_sample_sha256=qa._fingerprint(sample.model_dump(mode="json")), clip_uid=job.clip_uid,
            pair_type="canonical", conditioning_variant=variant, source_mimo_record_fingerprint=record.record_fingerprint,
            status="ready", audio_references=[r.model_dump(mode="json") for r in refs], corrected_speech_segments=[s.model_dump(mode="json") for s in corrected],
            rendered_h3_prompt=text, task_prefix=audio_task_prefix([r.contract for r in refs]), warnings=warnings,
        ))
    root = shadow / "h3_audio_reuse_products_v1"
    root.mkdir()
    (root / "records.jsonl").write_text("".join(p.model_dump_json() + "\n" for p in products))
    paths = [prepared / "inventory.json", prepared / "records.jsonl", sample_path, shadow / "separation/records.jsonl"]
    summary = AudioReuseProductSummary(
        source_hashes={str(p): qa.sha256_file(p) for p in paths}, clip_uids=[j.clip_uid for j in jobs],
        sample_count=len(products), ready_count=len(products), failed_count=0,
        audio_kind_counts=dict(Counter(a.contract.kind for p in products for a in p.audio_references)),
        task_prefix_counts=dict(Counter(p.task_prefix for p in products)), warning_counts={},
    )
    (root / "summary.json").write_text(summary.model_dump_json())
    return root


def _write_mimo(kwargs, shadow, *, kind="full_audio_reuse"):
    base, jobs, samples, records = _inputs(kwargs, shadow)
    job, record = jobs[0], records[0]
    source = next(s for s in samples if s.clip_uid == job.clip_uid and s.pair_type == "canonical")
    sample = project_prepared_samples([job], [source])[0]
    from r2v_data_v2.h3.mimo25_h3_materializer import MimoShadowAudioReference
    audio = MimoShadowAudioReference(
        audio_index=1, role=kind, source_type="canonical_full_audio" if kind == "full_audio_reuse" else "mimo_non_diegetic_music_event",
        audio_path=job.target_full_audio_path, audio_sha256=job.target_full_audio_sha256,
        **({"interval_provenance": "canonical_full_audio_exact_v1"} if kind == "full_audio_reuse" else {
            "interval_provenance": "mimo_approximate_event_times_rounded_to_32000_v1", "source_event_id": "event_1",
            "source_start_time": 0, "source_end_time": 1, "source_start_sample": 0, "source_end_sample": 32000,
            "music_description": "Gentle piano", "rms_dbfs": -20, "clipping_ratio": 0,
        }),
    )
    corrected, text, warnings = qa._materialize_sample(
        sample, job, qa._MaterializerInput(record.annotation, job.request_fingerprint), conditioning_variant=kind,
        extra_audio_contract=_extra_audio_contract(audio, render_path=Path(audio.audio_path)),
    )
    product = _seal(MimoH3ShadowRecord,
        sample_id=job.clip_uid + ("/audio_reuse" if kind == "full_audio_reuse" else "/music_reference"),
        source_h3_sample_id=source.sample_id, clip_uid=job.clip_uid, pair_type="canonical", derived_from_pair_type="canonical",
        conditioning_variant=kind, status="ready", source_mimo_record_fingerprint=record.record_fingerprint,
        source_h3_sample_sha256=qa._fingerprint(source.model_dump(mode="json")),
        corrected_speech_segments=[s.model_dump(mode="json") for s in corrected], effective_subject_voices=[],
        audio_references=[audio.model_dump(mode="json")], recovered_voice_references=[], rendered_h3_prompt=text, warnings=warnings,
    )
    root = shadow / "mimo25_h3_shadow_v5"
    root.mkdir()
    (root / "records.jsonl").write_text(product.model_dump_json()+"\n")
    values = {name: 0 for name in (
        "materialization_failure_count", "target_clip_annotation_reuse_count", "existing_target_voice_reference_count",
        "recovered_voice_candidate_count", "recovered_voice_accepted_count", "recovered_voice_rejected_count", "recovered_voice_entity_count",
        "derived_in_pair_count", "enriched_existing_in_pair_count", "full_audio_reuse_count", "music_reference_count")}
    values[kind+"_count"] = 1
    summary = MimoH3ShadowSummary(
        source_mimo_inventory_fingerprint=qa._inventory({
            **base.model_dump(mode="json", exclude={"inventory_fingerprint"}),
            "jobs": [j.model_dump(mode="json") for j in jobs], "clip_count": len(jobs),
        }).inventory_fingerprint,
        source_mimo_records_sha256=qa.sha256_file(shadow / qa.MIMO25_STEM_RECONCILE_STAGE / "records.jsonl"),
        source_h3_samples_sha256=base.source_h3_samples_sha256, sample_count=1, ready_count=1, failed_count=0,
        materialization_failure_code_counts={}, warning_counts={}, recovery_rejection_reason_counts={},
        music_reference_rejection_reason_counts={}, **values,
    )
    (root / "summary.json").write_text(summary.model_dump_json())
    return root


@pytest.mark.parametrize("kind", ["full_audio_reuse", "music_reference"])
def test_mimo_published_discovery_and_full_dedup(tmp_path, monkeypatch, kind):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True, cross_variant=True)
    _write_mimo(kwargs, shadow, kind=kind)
    before = _snapshot(tmp_path)
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    variants = data["clips"][0]["final_h3"]["variants"]
    published = [v for v in variants if v["conditioning_variant"] == kind]
    assert len(published) == 1 and published[0]["source_kind"] == "mimo_h3_shadow"
    assert published[0]["publication_status"] == "published"
    assert published[0]["references"]["audios"][0]["kind"] == kind
    assert all(p.read_bytes() == content for p, content in before.items())


def test_reuse_products_discovered_and_distinct_sources_preserved(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True, cross_variant=True)
    _write_products(kwargs, shadow)
    _write_mimo(kwargs, shadow)
    before = _snapshot(tmp_path)
    result = qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    clip = data["clips"][0]
    variants = clip["final_h3"]["variants"]
    audios = [a for v in variants for a in (v["references"] or {})["audios"]]
    assert {a["kind"] for a in audios} >= {"target_voice", "cross_voice", "speaker_speech_reuse", "music_reuse", "full_audio_reuse"}
    full = [v for v in variants if v["conditioning_variant"] == "full_audio_reuse"]
    # Audio reuse owns exact ASR projection, so genuinely different text stays visible.
    assert len(full) == 2 and len({v["text"] for v in full}) == 2
    assert {v["source_kind"] for v in full} == {"audio_reuse_product", "mimo_h3_shadow"}
    assert len([a for a in audios if a["kind"] == "cross_voice"]) == 2  # distinct donor contracts
    assert clip["reference_variant_coverage"]["music_reference"] == "not generated"
    assert clip["reference_variant_coverage"]["speaker_speech_reuse"] == "available"
    assert result["model_called"] is False
    assert all(p.read_bytes() == content for p, content in before.items())
    first = (shadow / "qa/data.json").read_bytes()
    qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    assert (shadow / "qa/data.json").read_bytes() == first


@pytest.mark.parametrize("mutation", ["fingerprint", "summary", "media", "source_hash", "source_sample", "materializer"])
def test_current_run_product_corruption_fails_closed(tmp_path, monkeypatch, mutation):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    root = _write_mimo(kwargs, shadow)
    path = root / "records.jsonl"
    values = json.loads(path.read_text())
    if mutation == "summary":
        summary = json.loads((root / "summary.json").read_text())
        summary["source_mimo_records_sha256"] = "f"*64
        (root / "summary.json").write_text(json.dumps(summary))
    elif mutation == "media":
        Path(values["audio_references"][0]["audio_path"]).write_bytes(b"tampered")
    else:
        field = {"fingerprint": "source_mimo_record_fingerprint", "source_hash": "source_h3_sample_sha256",
                 "source_sample": "source_h3_sample_id", "materializer": "materializer_version"}[mutation]
        values[field] = "h3_mimo25_materializer_v27" if mutation == "materializer" else "f"*64
        values["record_fingerprint"] = qa._fingerprint({k: v for k, v in values.items() if k != "record_fingerprint"})
        path.write_text(json.dumps(values)+"\n")
    with pytest.raises(ValueError):
        qa.build_audio_shadow_qa(**kwargs)
    assert not (shadow / "qa").exists()


def test_same_variant_different_text_is_not_deduplicated():
    common = {"sample_id": "clip/audio_reuse", "conditioning_variant": "full_audio_reuse", "status": "ready",
              "references": {"pictures": [], "subjects": [], "audios": []}, "warnings": []}
    variants = [dict(common, text="one", source_kind="qa_derived", publication_status="review_only"),
                dict(common, text="two", source_kind="mimo_h3_shadow", publication_status="published")]
    assert len(finalize_registry(variants, qa._fingerprint)[0]) == 2
    variants.append(dict(common, text="one", source_kind="audio_reuse_product", publication_status="published"))
    result, _ = finalize_registry(variants, qa._fingerprint)
    assert len(result) == 2
    assert next(v for v in result if v["text"] == "one")["source_kind"] == "audio_reuse_product"


def test_missing_products_show_coverage_without_extraction(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True, cross_variant=True)
    before = _snapshot(tmp_path)
    assert qa.build_audio_shadow_qa(**kwargs)["model_called"] is False
    clip = json.loads((shadow / "qa/data.json").read_text())["clips"][0]
    assert clip["reference_variant_coverage"] == {
        "visual_only": "available", "target_voice_reference": "available", "cross_voice_reference": "available",
        "full_audio_reuse": "review-only available", "music_reference": "not generated",
        "speaker_speech_reuse": "not generated", "music_reuse": "not generated",
    }
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (shadow / "audio_reuse_assets_v1").exists()


def test_published_recovered_voice_keeps_source_identity(tmp_path, monkeypatch):
    from r2v_data_v2.h3.mimo25_h3_materializer import _audio_references
    from r2v_data_v2.h3.mimo25_recovered_voice import (
        MimoRecoveredVoiceQualityPolicy,
        MimoRecoveredVoiceReference,
    )

    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    root = _write_mimo(kwargs, shadow)
    _, jobs, samples, records = _inputs(kwargs, shadow)
    job, record = jobs[0], records[0]
    source = next(s for s in samples if s.clip_uid == job.clip_uid and s.pair_type == "in_pair")
    sample = project_prepared_samples([job], [source])[0]
    voice = sample.subject_voices[0]
    segment = job.segments[0]
    recovery = MimoRecoveredVoiceReference(
        clip_uid=job.clip_uid, entity_id="e1", subject_index=1, speaker_group="g1",
        source_segment_id=segment.segment_id, source_start=segment.start_time, source_end=segment.end_time,
        source_start_sample=round(segment.start_time*32000), source_end_sample=round(segment.end_time*32000),
        mimo_record_fingerprint=record.record_fingerprint, evidence_codes=["visible_lip_motion"], confidence="high",
        quality_metrics={"duration_seconds": segment.end_time-segment.start_time, "sample_count": 16000,
                         "rms_amplitude": .1, "peak_amplitude": .2, "clipping_ratio": 0,
                         "local_noise_sample_count": 3200, "local_noise_duration_seconds": .1},
        quality_policy_fingerprint=MimoRecoveredVoiceQualityPolicy().fingerprint(),
        status="selected", reason_codes=[], output_path=voice.voice_reference_path, output_sha256=voice.voice_reference_sha256,
    )
    corrected, text, warnings = qa._materialize_sample(sample, job, qa._MaterializerInput(record.annotation, job.request_fingerprint))
    audio = _audio_references(voices=[voice], corrected=corrected, recovered=[recovery])
    product = _seal(MimoH3ShadowRecord,
        sample_id=sample.sample_id, source_h3_sample_id=sample.sample_id, clip_uid=job.clip_uid, pair_type="in_pair",
        conditioning_variant="target_voice_reference", status="ready", source_mimo_record_fingerprint=record.record_fingerprint,
        source_h3_sample_sha256=qa._fingerprint(source.model_dump(mode="json")),
        corrected_speech_segments=[s.model_dump(mode="json") for s in corrected],
        effective_subject_voices=[voice.model_dump(mode="json")], audio_references=[a.model_dump(mode="json") for a in audio],
        recovered_voice_references=[recovery.model_dump(mode="json")], rendered_h3_prompt=text, warnings=warnings,
    )
    (root / "records.jsonl").write_text(product.model_dump_json()+"\n")
    qa.build_audio_shadow_qa(**kwargs)
    variants = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]["variants"]
    recovered = next(v for v in variants if v["source_kind"] == "mimo_h3_shadow")
    ref = recovered["references"]["audios"][0]
    assert ref["source_type"] == "mimo_recovered_target_voice"
    assert ref["provenance"]["source_segment_id"] == segment.segment_id
    assert ref["provenance"]["recovery"]["output_sha256"] == voice.voice_reference_sha256


def test_standalone_mimo_record_source_is_verified(tmp_path, monkeypatch):
    from r2v_data_v2.h3.mimo25_av_reconcile import MimoRecord

    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    root = _write_mimo(kwargs, shadow)
    base, jobs, _, records = _inputs(kwargs, shadow)
    inventory = qa._inventory({**base.model_dump(mode="json", exclude={"inventory_fingerprint"}),
                               "jobs": [jobs[0].model_dump(mode="json")], "clip_count": 1})
    source = _seal(MimoRecord,
        clip_uid=jobs[0].clip_uid, request_fingerprint=jobs[0].request_fingerprint,
        inventory_fingerprint=inventory.inventory_fingerprint, status="ready",
        backend_provenance=records[0].backend_provenance.model_dump(mode="json"),
        annotation=records[0].annotation.model_dump(mode="json"), input_modality="target_video_with_embedded_audio",
        model_call_count=3, http_attempt_count=3, raw_response_count=3, http_retry_count=0, recheck_count=0,
        deterministic_correction_counts={},
    )
    origin = shadow / "mimo25_av_reconcile_v5"
    origin.mkdir()
    (origin / "inventory.json").write_text(inventory.model_dump_json())
    (origin / "records.jsonl").write_text(source.model_dump_json()+"\n")
    summary = json.loads((root / "summary.json").read_text())
    summary.update(source_mimo_inventory_fingerprint=inventory.inventory_fingerprint,
                   source_mimo_records_sha256=qa.sha256_file(origin / "records.jsonl"))
    (root / "summary.json").write_text(json.dumps(summary))
    product = json.loads((root / "records.jsonl").read_text())
    product["source_mimo_record_fingerprint"] = source.record_fingerprint
    product["record_fingerprint"] = qa._fingerprint({k: v for k, v in product.items() if k != "record_fingerprint"})
    (root / "records.jsonl").write_text(json.dumps(product)+"\n")
    qa.build_audio_shadow_qa(**kwargs)
    variants = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]["variants"]
    assert any(v["source_kind"] == "mimo_h3_shadow" for v in variants)
    (origin / "records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        qa.build_audio_shadow_qa(**kwargs, overwrite=True)


@pytest.mark.parametrize("target", ["prepared", "reuse_media", "manifest"])
def test_reuse_dependency_tamper_fails_closed(tmp_path, monkeypatch, target):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    _write_products(kwargs, shadow)
    path = {"prepared": shadow / "prepared-fixture/records.jsonl",
            "reuse_media": shadow / "audio_reuse_assets_v1/clip-z/speech.flac",
            "manifest": shadow / "audio_reuse_assets_v1/clip-z/manifest.json"}[target]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        qa.build_audio_shadow_qa(**kwargs)
    assert not (shadow / "qa").exists()


@pytest.mark.parametrize("clip", ["711b", "aa28", "c54"])
def test_existing_target_cross_shapes_survive_registry(clip):
    variants = [{"sample_id": f"{clip}/{pair}", "conditioning_variant": variant,
                 "source_kind": "source_h3", "publication_status": "review_only", "status": "ready",
                 "text": pair, "references": {"audios": []}}
                for pair, variant in (("canonical", "visual_only"), ("in_pair", "target_voice_reference"),
                                      ("cross_pair/1", "cross_voice_reference"))]
    variants.append({"sample_id": f"{clip}/audio_reuse", "conditioning_variant": "full_audio_reuse",
                     "source_kind": "qa_derived", "publication_status": "review_only", "status": "ready",
                     "text": "reuse", "references": {"audios": []}})
    result, _ = finalize_registry(variants, qa._fingerprint)
    assert {v["sample_id"] for v in result} == {v["sample_id"] for v in variants}
