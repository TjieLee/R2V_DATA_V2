from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import _inventory, _job
from r2v_data_v2.h3.mimo25_backend import (
    MimoAuxAudioDescriptionCall,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    MimoUsage,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    load_stem_shadow,
    sha256_file,
    validate_stem_diarization_lineage,
)
from tests.h3_mimo_two_turn_helpers import split_annotation
from tests.test_h3_mimo25_av_shadow import _annotation, _sample
from tests.test_h3_sam_audio_stem_shadow import (
    _mimo_inventory_for_clip_order,
    _MixedDiarization,
    _run_three_clip_shadow_to_asr,
)
from tools.build_h3_audio_shadow_qa import main


class _Reconcile:
    def __init__(self, tmp_path):
        self.provenance = MimoBackendConfig(
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
            api_key="fake", transport="sglang",
        ).provenance()

    def describe_auxiliary_audio(self, path):
        return MimoAuxAudioDescriptionCall(
            description="A quiet sound.", raw_response='{"description":"A quiet sound."}',
            diagnostic=MimoCompletionDiagnostic(
                input_modality="auxiliary_audio_only", usage=MimoUsage(), http_attempt_count=1,
            ),
        )

    def reconcile(self, job, **kwargs):
        annotation = MimoAVAnnotationDraft.model_validate_json(
            _annotation().model_dump_json().replace("segment_1", "segment_0001")
        )
        values = annotation.model_dump(mode="json")
        if not job.segments:
            values["visual_observation"]["segment_views"] = []
            values["audio_observation"]["segment_decisions"] = []
            values["audio_observation"]["speaker_voice_profiles"] = []
            values["av_grounding"]["segment_groundings"] = []
            values["h3_semantics"]["shot1_caption"] = "The seated person remains still."
        annotation = MimoAVAnnotationDraft.model_validate(values)
        visual_raw, speech_raw, profile_raw, finalized_raw = split_annotation(annotation.model_dump_json())
        if not job.segments:
            profile_raw = None
        diagnostics = tuple(
            MimoCompletionDiagnostic(input_modality=modality, usage=MimoUsage(), http_attempt_count=1)
            for modality in (
                "target_video_visual_only", "target_video_speech_assembly",
                *(("speaker_profile_audio_only",) if profile_raw else ()),
                "target_video_audio_finalize",
            )
        )
        fields = {
            "model_call_count": len(diagnostics), "visual_model_call_count": 1,
            "audio_model_call_count": int(profile_raw is not None),
            "visual_raw_response": visual_raw, "speech_av_raw_response": speech_raw,
            "speaker_profile_raw_response": profile_raw, "audio_finalize_raw_response": finalized_raw,
            "raw_responses": tuple(raw for raw in (visual_raw, speech_raw, profile_raw, finalized_raw) if raw is not None),
            "diagnostics": diagnostics,
        }
        if job.clip_uid == "clip-a":
            raise MimoBackendFailure(
                code="synthetic_failed", reason="failed <script>not markup</script>", **fields,
            )
        return SimpleNamespace(annotation=annotation, **fields)


def _fixture(tmp_path, monkeypatch, *, mixed=False, variants=False, cross_variant=False):
    root = tmp_path / "production"
    root.mkdir()
    separation, diarization, asr, inventory, order = _run_three_clip_shadow_to_asr(
        root, shadow_run_id="random10-v1",
        diarization_backend=_MixedDiarization() if mixed else None,
    )
    shadow = separation.parent
    base = _mimo_inventory_for_clip_order(tmp_path / "base", order)
    jobs = []
    for old, stem in zip(base.jobs, inventory.jobs, strict=True):
        values = old.model_dump(mode="json", exclude={"request_fingerprint"})
        values.update(
            target_video_path=stem.target_video_path,
            target_video_sha256=stem.target_video_sha256,
            target_full_audio_path=stem.source_audio_path,
            target_full_audio_sha256=stem.source_audio_sha256,
        )
        if variants:
            values["source_h3_sample_ids"].append(f"{old.clip_uid}/canonical")
        if cross_variant:
            values["source_h3_sample_ids"].append(f"{old.clip_uid}/cross_pair")
        for ref in values["reference_images"]:
            Image.new("RGB", (160, 200), (98, 150, 140)).save(ref["image_artifact_path"])
            ref["image_sha256"] = sha256_file(Path(ref["image_artifact_path"]))
        jobs.append(_job(values))
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
    values["jobs"] = [job.model_dump(mode="json") for job in jobs]
    sample_root = tmp_path / "sample-template"
    sample_root.mkdir()
    template = _sample(sample_root)
    voice_path = sample_root / "voice.wav"
    with wave.open(str(voice_path), "wb") as stream:
        stream.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 128000)
    template.subject_voices[0].voice_reference_path = str(voice_path)
    template.subject_voices[0].voice_reference_sha256 = sha256_file(voice_path)
    samples_path = root / "h3/samples.jsonl"
    samples_path.parent.mkdir()
    samples = []
    for job in jobs:
        sample_values = template.model_dump(mode="json")
        sample_values.update(sample_id=job.source_h3_sample_ids[0], clip_uid=job.clip_uid,
                             target_video=job.target_video_path,
                             target_full_audio_path=job.target_full_audio_path,
                             target_full_audio_sha256=job.target_full_audio_sha256)
        sample_values["visual_references"][0]["image_artifact_path"] = job.reference_images[0].image_artifact_path
        sample_values["subject_voices"][0]["target_occurrence_id"] = f"{job.clip_uid}/e1"
        samples.append(FinalH3SampleV2.model_validate(sample_values))
        if variants:
            samples.append(FinalH3SampleV2.model_validate({
                **sample_values, "sample_id": f"{job.clip_uid}/canonical",
                "pair_type": "canonical", "subject_voices": [],
            }))
        if cross_variant:
            donor_path = sample_root / "donor.wav"
            donor_path.write_bytes(voice_path.read_bytes())
            voice = {**sample_values["subject_voices"][0], "voice_source": "cross_donor",
                     "voice_reference_path": str(donor_path), "donor_clip_uid": "donor",
                     "donor_occurrence_id": "donor/e3", "donor_clip_display_path": "01/show/donor"}
            samples.append(FinalH3SampleV2.model_validate({
                **sample_values, "sample_id": f"{job.clip_uid}/cross_pair",
                "pair_type": "cross_pair", "subject_voices": [voice],
            }))
    samples_path.write_text("".join(sample.model_dump_json() + "\n" for sample in samples))
    values["source_h3_samples_sha256"] = sha256_file(samples_path)
    base = _inventory(values)
    monkeypatch.setattr(qa, "build_mimo25_inventory", lambda **kwargs: base)
    assert not (shadow / "mimo_stem_facts").exists()
    provenance, _, _ = validate_stem_diarization_lineage(diarization)
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.usable_stem_reconcile_inventory(base, provenance),
        stem_diarization_root=diarization,
        stem_asr_root=asr, route="music_first",
    )
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_records=load_stem_shadow(separation)[1],
        backend=_Reconcile(tmp_path), output_root=shadow / MIMO25_STEM_RECONCILE_STAGE,
        source_clip_uids=order, route="music_first",
        diarization_failed_clips=provenance.diarization_failed_clips,
        allow_unverified=True,
    )
    visual, runs = tmp_path / "visual", tmp_path / "runs"
    visual.mkdir()
    runs.mkdir()
    kwargs = {
        "visual_production_root": visual, "visual_runs_root": runs,
        "audio_production_root": root, "case_manifest": root / "case.json",
        "shadow_run_id": "random10-v1",
    }
    return kwargs, shadow


def _snapshot(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_builder_ready_failed_order_media_and_sources_unchanged(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    materialized = []
    original = qa._materialize_sample

    def capture(sample, job, record, **kwargs):
        result = original(sample, job, record, **kwargs)
        materialized.append((sample, job, record, result[1]))
        return result

    monkeypatch.setattr(qa, "_materialize_sample", capture)
    before = _snapshot(tmp_path)
    result = qa.build_audio_shadow_qa(**kwargs)
    assert result["model_called"] is False
    output = shadow / "qa"
    assert result["output_root"] == str(output)
    data = json.loads((output / "data.json").read_text())
    reconcile_summary = json.loads((shadow / MIMO25_STEM_RECONCILE_STAGE / "summary.json").read_text())
    assert reconcile_summary["schema_version"] == "r2v.h3.mimo25_stem_reconcile_summary.16"
    assert reconcile_summary["current_mimo_versions_modified"] is True
    assert data["clip_uids"] == ["clip-z", "clip-a", "clip-m"]
    assert [clip["reconcile"]["status"] for clip in data["clips"]] == ["ready", "failed", "ready"]
    failed = data["clips"][1]["reconcile"]
    assert failed["failure_code"] == "synthetic_failed"
    assert failed["model_call_count"] == 4
    assert failed["visual_raw_response"] and failed["speech_av_raw_response"] and failed["audio_finalize_raw_response"]
    assert failed["visual_model_call_count"] == 1
    assert data["clips"][1]["direct_h3"]["style_opening"]
    assert data["clips"][1]["direct_h3"]["shot1_caption"]
    assert data["clips"][0]["reconcile"]["annotation"]["audio_observation"]
    assert [item[1].clip_uid for item in materialized] == ["clip-z", "clip-m"]
    for clip, call in zip((data["clips"][0], data["clips"][2]), materialized, strict=True):
        final = clip["final_h3"]
        assert final["status"] == "ready"
        assert final["materializer_version"] == qa.MIMO25_MATERIALIZER_VERSION
        assert final["text"] == call[3] == original(*call[:3])[1]
        assert final["variants"][0]["text"] == final["text"]
        assert "[[" not in final["text"]
        assert clip["direct_h3"]["shot1_caption"]
        assert clip["production_h3"]
        assert "MiMo DIRECT H3" in (output / "review.html").read_text()
        assert "<Picture 1>" in final["text"]
        sections = ["subject_definitions", "summary", "retention_analysis",
                    "detailed_description", "overall_soundscape", "non_diegetic_music"]
        positions = [final["text"].index(section + ":\n") for section in sections]
        assert positions == sorted(positions)
        assert clip["direct_h3"]["shot1_caption"] in final["text"]
        assert final["text"].count("\n[Shot 1] ") == 1
    assert data["clips"][1]["final_h3"]["status"] == "unavailable"
    assert data["clips"][1]["final_h3"]["text"] is None
    assert data["clips"][1]["final_h3"]["reason"] == "AV reconcile failed: failed <script>not markup</script>"
    assert data["qa_labels"] == list(qa.QA_LABELS)
    assert all(path.read_bytes() == content for path, content in before.items())
    for url, source in data["media"].items():
        link = output / url
        assert link.is_symlink()
        assert link.resolve() == Path(source["source_path"])
        assert link.read_bytes() == Path(source["source_path"]).read_bytes()
    html = (output / "review.html").read_text()
    assert 'fetch("data.json", {cache: "no-store"})' in html
    assert "Export QA JSON" in html and "Import QA JSON" in html
    assert "localStorage.clear" not in html
    assert "innerHTML" not in html
    assert "frozen_production_evidence" in data["clips"][0]["speaker_binding"]
    assert "stem_segment_evidence" in data["clips"][0]["speaker_binding"]
    assert "stem_facts" not in data["clips"][0]
    assert data["clips"][1]["direct_h3"]["overall_soundscape"]
    assert data["clips"][1]["direct_h3"]["non_diegetic_music"]
    assert not (shadow / "mimo_stem_facts").exists()
    first_data = (output / "data.json").read_bytes()
    qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    assert (output / "data.json").read_bytes() == first_data


def test_upstream_failed_and_empty_clips_remain_visible(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, mixed=True)
    qa.build_audio_shadow_qa(**kwargs)
    clips = json.loads((shadow / "qa/data.json").read_text())["clips"]
    assert [clip["diarization"]["clip_result"]["status"] for clip in clips] == ["ready", "failed", "empty"]
    assert clips[1]["reconcile"]["status"] == "upstream_failed"
    assert clips[1]["reconcile"]["model_call_count"] == 0
    assert "stem_facts" not in clips[1]
    assert clips[1]["asr"]["segments"] == []
    assert clips[1]["final_h3"]["status"] == "unavailable"
    assert clips[1]["final_h3"]["text"] is None
    assert clips[1]["final_h3"]["reason"] == "upstream_failed"
    assert clips[2]["asr"]["segments"] == []


def test_final_text_changes_review_fingerprints(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    qa.build_audio_shadow_qa(**kwargs)
    before = json.loads((shadow / "qa/data.json").read_text())
    original = qa._materialize_sample

    def changed_text(*args, **kwargs):
        corrected, text, warnings = original(*args, **kwargs)
        return corrected, text + "\nsynthetic materializer change", warnings

    monkeypatch.setattr(qa, "_materialize_sample", changed_text)
    qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    after = json.loads((shadow / "qa/data.json").read_text())
    assert before["dataset_fingerprint"] != after["dataset_fingerprint"]
    for old, new in zip(before["clips"], after["clips"], strict=True):
        assert (old["record_fingerprint"] == new["record_fingerprint"]) == (
            old["reconcile"]["status"] == "failed"
        )


def test_final_h3_keeps_conditioning_variants_separate(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    qa.build_audio_shadow_qa(**kwargs)
    final = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]
    assert [row["sample_id"] for row in final["variants"]] == [
        "clip-z/canonical", "clip-z/in_pair", "clip-z/audio_reuse",
    ]
    canonical, voice, full = final["variants"]
    assert full["source_kind"] == "qa_derived" and full["publication_status"] == "review_only"
    assert full["references"]["audios"][0]["kind"] == "full_audio_reuse"
    assert full["references"]["audios"][0]["retention_marker"] == "fully_copy"
    assert final["text"] == canonical["text"]
    assert "[reference generation]" in canonical["text"]
    assert "<Audio 1>" not in canonical["text"]
    assert "[reference generation + audio reference]" in voice["text"]
    assert voice["text"].split("detailed_description:", 1)[1] == canonical["text"].split("detailed_description:", 1)[1]
    assert canonical["references"]["audios"] == []
    assert canonical["references"]["conditioning_variant"] == "visual_only"
    assert voice["references"]["audios"][0]["audio_label"] == "<Audio 1>"
    assert voice["references"]["audios"][0]["kind"] == "target_voice"
    assert voice["references"]["audios"][0]["subject_label"] == "<Subject 1>"
    assert voice["references"]["audios"][0]["speaker_id"] == "S1"


def test_reference_media_and_cross_provenance_are_variant_owned(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True, cross_variant=True)
    before = _snapshot(tmp_path)
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    canonical, cross, target, full = data["clips"][0]["final_h3"]["variants"]
    assert full["references"]["audios"][0]["sha256"] == qa._rows(
        kwargs["audio_production_root"] / "h3/samples.jsonl", FinalH3SampleV2,
    )[0].target_full_audio_sha256
    for variant in (canonical, cross, target):
        refs = variant["references"]
        assert refs["subjects"][0]["source_picture_labels"] == ["<Picture 1>"]
        assert refs["pictures"][0]["url"] in data["media"]
        assert not any("video" in ref.get("kind", "") for ref in refs["pictures"])
    assert canonical["references"]["audios"] == []
    cross_audio, target_audio = cross["references"]["audios"][0], target["references"]["audios"][0]
    assert cross_audio["kind"] == "cross_voice" and cross_audio["source_type"] == "cross_donor"
    assert cross_audio["provenance"]["donor_occurrence_id"] == "donor/e3"
    assert cross_audio["provenance"]["donor_clip_uid"] == "donor"
    assert target_audio["source_type"] == "existing_target_voice"
    assert cross_audio["url"] != target_audio["url"]
    for item in (cross_audio, target_audio):
        assert data["media"][item["url"]]["source_path"] == item["path"]
        assert (shadow / "qa" / item["url"]).is_symlink()
    assert not set(data["clips"][0]["separation"]["media"].values()) & {cross_audio["url"], target_audio["url"]}
    assert all(path.read_bytes() == content for path, content in before.items())


def test_variant_coverage_warnings_are_preserved(tmp_path, monkeypatch):
    original = _Reconcile.reconcile

    def reconcile(self, job, **kwargs):
        result = original(self, job, **kwargs)
        values = result.annotation.model_dump(mode="json")
        values["h3_semantics"]["shot1_caption"] = values["h3_semantics"]["shot1_caption"].replace(
            "<Subject 1>", "The person",
        )
        result.annotation = MimoAVAnnotationDraft.model_validate(values)
        return result

    monkeypatch.setattr(_Reconcile, "reconcile", reconcile)
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    qa.build_audio_shadow_qa(**kwargs)
    final = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]
    warning = "<Subject 1>:preserved_visual_subject_missing_from_detailed_description"
    for variant in final["variants"]:
        assert variant["status"] == "ready"
        assert warning in variant["warnings"]
        assert variant["references"]["subjects"][0]["subject_label"] == "<Subject 1>"


@pytest.mark.parametrize("kind", ["target_voice", "cross_voice", "speaker_speech_reuse",
                                  "music_reuse", "music_reference", "full_audio_reuse"])
def test_reference_projection_supports_current_typed_audio_kinds(tmp_path, kind):
    from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionAudioContract
    from tests.test_h3_audio_reuse_materializer import _case

    _, sample, source = _case(tmp_path)
    ownership = ({"subject_label": "<Subject 1>", "entity_id": "e1", "speaker_id": "S1"}
                 if kind in {"target_voice", "cross_voice", "speaker_speech_reuse"} else {})
    audio = RecaptionAudioContract(
        audio_index=1, audio_label="<Audio 1>", kind=kind, path=source.job.target_full_audio_path,
        sha256=source.job.target_full_audio_sha256,
        retention_marker="fully_copy" if kind == "full_audio_reuse" else (
            "partially_copy" if kind.endswith("_reuse") else "reference"),
        music_characteristics="Gentle piano" if kind == "music_reference" else None,
        **ownership,
    )
    context = qa._prepare_materialization_context(sample, source.job, source.record,
                                                 reuse_audio_contracts=[audio])
    media = {}

    def link(path, digest):
        assert sha256_file(Path(path)) == digest
        url = f"media/{digest}{Path(path).suffix}"
        media[url] = path
        return url

    source_type = {"target_voice": "mimo_recovered_target_voice", "cross_voice": "cross_donor",
                   "speaker_speech_reuse": "target_speaker_reuse_asset", "music_reuse": "target_music_reuse_asset",
                   "music_reference": "mimo_non_diegetic_music_event", "full_audio_reuse": "canonical_full_audio"}[kind]
    provenance = {"role": "voice_reference" if kind in {"target_voice", "cross_voice"} else kind,
                  "source_type": source_type, "source_segment_id": "segment_1"}
    refs = qa._reference_payload(context, source.job, [sample], link,
                                 audio_provenance={"<Audio 1>": provenance})
    result = refs["audios"][0]
    assert {key: result[key] for key in audio.model_dump()} == audio.model_dump(mode="json")
    assert result["provenance"]["source_segment_id"] == "segment_1"
    assert result["url"] in media


@pytest.mark.parametrize("media_kind", ["voice", "image"])
def test_added_reference_media_hash_mismatch_fails_closed(tmp_path, monkeypatch, media_kind):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    sample = qa._rows(kwargs["audio_production_root"] / "h3/samples.jsonl", FinalH3SampleV2)[0]
    path = sample.subject_voices[0].voice_reference_path if media_kind == "voice" else sample.visual_references[0].image_artifact_path
    Path(path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="QA source media changed"):
        qa.build_audio_shadow_qa(**kwargs)
    assert not (shadow / "qa").exists()


def test_promoted_and_dropped_references_remain_separate(tmp_path, monkeypatch):
    from r2v_data_v2.h3.mimo25_av_reconcile import project_mimo_h3_sample_references
    from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract
    from tests.test_h3_mimo25_av_shadow import (
        _augmentation_draws,
        _job_for_reference_inventory,
        _sample_with_reference_inventory,
        _visual_reference_inventory,
    )

    originals = _visual_reference_inventory(tmp_path, ["subject", "face", "hair", "clothing"])
    sample = _sample_with_reference_inventory(tmp_path, originals)
    _augmentation_draws(monkeypatch, [.8, .9])
    job = _job_for_reference_inventory(tmp_path, sample)
    projected = project_mimo_h3_sample_references(sample, reference_images=job.reference_images,
                                                 reference_selection=job.reference_selection)
    context = qa._MaterializationContext(projected, [], [], "visual_only",
                                        build_reference_contract(projected, "visual_only"))
    links = {}

    def link(path, digest):
        assert sha256_file(Path(path)) == digest
        url = "media/" + digest + ".png"
        links[url] = path
        return url

    before = sample.model_dump_json()
    refs = qa._reference_payload(context, job, [sample], link)
    promoted = next(p for p in refs["pictures"] if p["face_promotion"])
    assert promoted["kind"] == "subject" and promoted["source_image_index"] == 2
    assert promoted["face_promotion"]["replaced_source_image_indexes"] == [1]
    assert {r["source_image_index"] for r in refs["dropped_visual_references"]} == {1, 3}
    assert all(r["url"] in links for r in refs["dropped_visual_references"])
    assert {p["source_image_index"] for p in refs["pictures"]} == {2, 4}
    assert refs["audios"] == [] and sample.model_dump_json() == before


def test_output_safety_and_atomic_failure_preserve_existing_qa(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    for protected in (
        kwargs["audio_production_root"], kwargs["visual_production_root"],
        kwargs["visual_runs_root"], shadow, shadow / "asr",
        shadow.parent / "another-run/qa",
    ):
        with pytest.raises(ValueError, match="overlap"):
            qa.build_audio_shadow_qa(**kwargs, output_root=protected, overwrite=True)
    qa.build_audio_shadow_qa(**kwargs)
    with pytest.raises(FileExistsError):
        qa.build_audio_shadow_qa(**kwargs)
    external = tmp_path / "unknown-output"
    external.mkdir()
    (external / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="not owned"):
        qa.build_audio_shadow_qa(**kwargs, output_root=external, overwrite=True)
    assert (external / "sentinel").read_text() == "keep"
    before = _snapshot(shadow / "qa")
    original_copy = qa.shutil.copyfile

    def fail_template(source, destination):
        if Path(source).suffix == ".html":
            raise RuntimeError("synthetic publication failure")
        return original_copy(source, destination)

    monkeypatch.setattr(qa.shutil, "copyfile", fail_template)
    with pytest.raises(RuntimeError, match="publication"):
        qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(shadow.glob(".qa.tmp-*"))


def test_stale_reconcile_fingerprint_rejected_before_publication(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    path = shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["source_job_fingerprint"] = "f" * 64
    values = {k: v for k, v in records[0].items() if k != "record_fingerprint"}
    records[0]["record_fingerprint"] = qa._fingerprint(values)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="stale source provenance"):
        qa.build_audio_shadow_qa(**kwargs)
    assert not (shadow / "qa").exists()


def test_cli_and_manifest_order_fail_closed(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    argv = [part for key, value in kwargs.items() for part in ("--" + key.replace("_", "-"), str(value))]
    result = main(argv)
    assert result["clip_count"] == 3
    case = kwargs["case_manifest"]
    value = json.loads(case.read_text())
    value["clip_uids"].reverse()
    case.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="separation order"):
        main([*argv, "--overwrite"])
    assert (shadow / "qa/review.html").is_file()


@pytest.mark.parametrize("blocked, expected", [
    ("multi", "multi_speaker_segment_requires_turn_refinement"),
    ("placeholder", "direct_internal_syntax"),
])
def test_ready_annotation_with_blocked_materialization_is_unavailable(tmp_path, monkeypatch, blocked, expected):
    original = _Reconcile.reconcile

    def reconcile(self, job, **kwargs):
        result = original(self, job, **kwargs)
        values = result.annotation.model_dump(mode="json")
        if blocked == "multi":
            for decision in values["audio_observation"]["segment_decisions"]:
                decision.update(
                    vocal_composition="sequential_multi_speaker_speech",
                    resolution="needs_acoustic_refinement",
                    secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"},
                )
            values["audio_observation"]["speaker_voice_profiles"] = []
            for grounding in values["av_grounding"]["segment_groundings"]:
                grounding.update(binding_status="no_reliable_entity", entity_id=None,
                                 speech_presentation="uncertain", confidence="low",
                                 evidence_codes=["insufficient_evidence"])
        else:
            values["h3_semantics"]["shot1_caption"] += " [[unknown]]"
        result.annotation = MimoAVAnnotationDraft.model_validate(values)
        return result

    monkeypatch.setattr(_Reconcile, "reconcile", reconcile)
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    before = _snapshot(tmp_path)
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    assert data["schema_version"] == qa.QA_DATA_VERSION
    for clip in (data["clips"][0], data["clips"][2]):
        assert clip["reconcile"]["status"] == "ready"
        final = clip["final_h3"]
        assert clip["direct_h3"]["shot1_caption"]
        assert final["status"] == "unavailable" and final["text"] is None
        assert final["reason"] == "materialization_contract_failed"
        for variant in final["variants"]:
            assert variant["text"] is None
            assert expected in {issue["code"] for issue in variant["issues"]}
            assert (variant["references"] is None) == (blocked == "multi")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_html_expected_schema_matches_qa_data_version():
    html = Path(qa.__file__).with_suffix(".html").read_text()
    expected = re.findall(r'payload\.schema_version !== "([^"]+)"', html)
    assert expected == [qa.QA_DATA_VERSION]


def test_generated_javascript_and_qa_roundtrip(tmp_path, monkeypatch):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for generated JavaScript execution")
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    qa.build_audio_shadow_qa(**kwargs)
    html = (shadow / "qa/review.html").read_text()
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL).group(1)
    js = tmp_path / "generated.js"
    js.write_text(script)
    subprocess.run([node, "--check", str(js)], check=True, capture_output=True, text=True)
    # Execute generated QA functions with a minimal DOM/storage, no network or model.
    prelude = r'''
const fs = require("fs");
const dataset = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const elements = new Map(), saved = new Map();
const inputs = dataset.qa_labels.map(label => ({dataset:{qaLabel:label}, checked:false}));
function el(id) {
  if (!elements.has(id)) elements.set(id, {value: "", hidden: false, textContent: "", children: [],
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; }});
  return elements.get(id);
}
const document = {
  getElementById: el,
  querySelector: selector => inputs.find(input => selector.includes('"' + input.dataset.qaLabel + '"')),
  querySelectorAll: selector => selector.includes("data-qa-label") ? inputs : [],
  createElement: tag => ({tagName:tag, textContent:"", children:[], append(...children) {this.children.push(...children);}}),
  createDocumentFragment: () => ({children:[], append(...children) {this.children.push(...children);}}),
  addEventListener: () => {}
};
const localStorage = {
  getItem: key => saved.get(key) ?? null,
  setItem: (key, value) => saved.set(key, value)
};
const fetch = () => new Promise(() => {});
'''
    checks = r'''
data = dataset; index = 1;
$("notes").value = "review <script> text";
render = function() { updateProgress(); };
function change(label, checked) {
  const input = inputs.find(input => input.dataset.qaLabel === label);
  input.checked = checked; labelChanged(input);
}
function expectLabels(labels) {
  if (JSON.stringify(stateFor(data.clips[index]).labels) !== JSON.stringify(labels))
    throw Error("selection was not immediately persisted");
  if (JSON.stringify(exportPayload().annotations[0].labels) !== JSON.stringify(labels))
    throw Error("export selection differs");
}
change("better", true); expectLabels(["better"]);
change("same", true); expectLabels(["same"]);
change("same", false); expectLabels([]);
change("dialogue_wrong", true); expectLabels(["dialogue_wrong"]);
change("audio_wrong", true); expectLabels(["dialogue_wrong", "audio_wrong"]);
change("dialogue_wrong", false); change("audio_wrong", false); expectLabels([]);
change("dialogue_wrong", true); change("audio_wrong", true);
change("dialogue_wrong", false); expectLabels(["audio_wrong"]);
const first = exportPayload();
if (first.annotations[0].clip_uid !== data.clips[1].clip_uid) throw Error("wrong clip");
if (first.annotations[0].labels.join(",") !== "audio_wrong") throw Error("DOM state was not exported");
if (stateFor(data.clips[1]).notes !== $("notes").value) throw Error("notes not restored");
if ($("progress").textContent !== "Labeled 1 / 3") throw Error("progress wrong");
importPayload(first);
if (JSON.stringify(exportPayload()) !== JSON.stringify(first)) throw Error("roundtrip unstable");
const old = JSON.stringify(first);
first.dataset_fingerprint = "stale";
let rejected = false; try { importPayload(first); } catch (e) { rejected = true; }
if (!rejected) throw Error("stale import accepted");
const invalid = JSON.parse(old); invalid.annotations[0].labels = ["invented"];
rejected = false; try { importPayload(invalid); } catch (e) { rejected = true; }
if (!rejected) throw Error("unknown label accepted");
const contradictory = JSON.parse(old); contradictory.annotations[0].labels = ["better", "same"];
const beforeImport = JSON.stringify([...saved]);
rejected = false; try { importPayload(contradictory); } catch (e) {
  rejected = e.message === "Invalid, duplicate, or stale QA annotation.";
}
if (!rejected || JSON.stringify([...saved]) !== beforeImport) throw Error("contradictory import accepted or rewritten");
const storageKey = key(data.clips[index]), previousState = saved.get(storageKey);
saved.set(storageKey, JSON.stringify({labels:["better", "same"], notes:"stale"}));
rejected = false; try { stateFor(data.clips[index]); } catch (e) {
  rejected = e.message.startsWith("Invalid saved QA state");
}
if (!rejected) throw Error("contradictory saved state accepted");
saved.set(storageKey, previousState);
inputs.find(input => input.dataset.qaLabel === "better").checked = true;
inputs.find(input => input.dataset.qaLabel === "same").checked = true;
rejected = false; try { exportPayload(); } catch (e) { rejected = true; }
if (!rejected || saved.get(storageKey) !== previousState) throw Error("contradictory DOM exported or persisted");
inputs.find(input => input.dataset.qaLabel === "better").checked = false;
inputs.find(input => input.dataset.qaLabel === "same").checked = false;
const duplicate = JSON.parse(old); duplicate.annotations.push(duplicate.annotations[0]);
rejected = false; try { importPayload(duplicate); } catch (e) { rejected = true; }
if (!rejected) throw Error("duplicate accepted");
const staleRecord = JSON.parse(old); staleRecord.annotations[0].record_fingerprint = "stale";
rejected = false; try { importPayload(staleRecord); } catch (e) { rejected = true; }
if (!rejected) throw Error("stale record accepted");
index = 0; $("notes").value = ""; saveCurrent();
if (exportPayload().annotations.map(x => x.clip_uid).join(",") !== "clip-z,clip-a")
  throw Error("export did not retain authoritative order");
index = 0; $("final-variant").value = "0";
data.clips[0].final_h3.variants[0] = {status:"unavailable", text:null,
  issues:[{code:"multi_speaker_segment_requires_turn_refinement", message:"retain ASR"}]};
renderFinalVariant();
if ($("final-unavailable").hidden || !$("copy-final").disabled ||
    $("final-text").textContent !== "" ||
    !$("final-unavailable").textContent.includes("multi_speaker_segment_requires_turn_refinement"))
  throw Error("blocked materialization was not displayed");
'''
    harness = tmp_path / "test.cjs"
    harness.write_text(prelude + "\n" + script + "\n" + checks)
    subprocess.run(
        [node, str(harness), str(shadow / "qa/data.json")],
        check=True, capture_output=True, text=True,
    )


def test_synthetic_browser_review_desktop_mobile(tmp_path, monkeypatch):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional preinstalled Playwright not configured")
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True, cross_variant=True)
    from tests.test_h3_audio_shadow_qa_variants import _write_mimo, _write_products

    _write_products(kwargs, shadow)
    _write_mimo(kwargs, shadow, kind="music_reference")
    qa.build_audio_shadow_qa(**kwargs)
    script = tmp_path / "browser.cjs"
    script.write_text(r'''
const fs = require("fs"), path = require("path"), assert = require("assert");
const {chromium} = require(process.argv[2]);
(async () => {
  const browser = await chromium.launch({headless: true, channel:process.env.QA_BROWSER_CHANNEL});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors = []; page.on("pageerror", error => errors.push(error.message));
    const root = process.argv[3];
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== "qa.invalid") throw Error("Unexpected network request");
      const filename = path.join(root, url.pathname);
      if (!fs.existsSync(filename)) return route.fulfill({status:404, body:"fixture only"});
      const type = filename.endsWith(".html") ? "text/html" : filename.endsWith(".json") ? "application/json" : filename.endsWith(".wav") ? "audio/wav" : filename.endsWith(".png") ? "image/png" : "application/octet-stream";
      return route.fulfill({body: fs.readFileSync(filename), contentType:type});
    });
    await page.goto("http://qa.invalid/review.html");
    await page.locator("#main").waitFor({state:"visible"});
    await page.waitForFunction(() => [...document.querySelectorAll("#references img")].every(image => image.naturalWidth > 0));
    assert.strictEqual(await page.locator("#clip-id").textContent(), "clip-z");
    const dataset = JSON.parse(fs.readFileSync(path.join(root, "data.json"), "utf8"));
    const expectedFinal = dataset.clips[0].final_h3.text;
    assert.strictEqual(await page.locator("#final-text").textContent(), expectedFinal);
    assert(await page.locator("#final-text").isVisible());
    assert.strictEqual(await page.locator("#final-h3").evaluate(el => el.closest("details")), null);
    assert.strictEqual(await page.locator("#materializer-version").textContent(), dataset.clips[0].final_h3.materializer_version);
    assert.strictEqual(await page.locator("#audio-references audio").count(), dataset.clips[0].final_h3.variants[0].references.audios.length);
    assert.strictEqual(await page.locator("#references img").count(), dataset.clips[0].final_h3.variants[0].references.pictures.length);
    const variants = dataset.clips[0].final_h3.variants;
    for (let i = 0; i < variants.length; i++) {
      await page.locator("#final-variant").selectOption(String(i));
      assert.strictEqual(await page.locator("#final-text").textContent(), variants[i].text);
      assert.strictEqual(await page.locator("#audio-references audio").count(), variants[i].references.audios.length);
      assert((await page.locator("#variant-source").textContent()).includes(variants[i].source_kind));
      assert.deepStrictEqual(await page.locator("#references img").evaluateAll(images => images.map(x => x.getAttribute("src"))),
        variants[i].references.pictures.map(p => p.url));
      if (variants[i].conditioning_variant === "full_audio_reuse") {
        assert.match(await page.locator("#audio-references").textContent(), /fully_copy.*canonical_full_audio/s);
        assert(!/Subject:|Speaker:/.test(await page.locator("#audio-references").textContent()));
      }
      assert.strictEqual(await page.locator("#variant-coverage .selected").count() > 0, true);
    }
    await page.locator("#final-variant").selectOption("0");
    assert.match(await page.locator("#subject-graph").textContent(), /<Subject 1>.*<Picture 1>/s);
    const crossIndex = variants.findIndex(v => v.references.audios.some(a => a.kind === "cross_voice" && a.provenance.donor_occurrence_id === "donor/e3"));
    const targetIndex = variants.findIndex(v => v.references.audios.some(a => a.kind === "target_voice" && a.source_type === "existing_target_voice"));
    const visualIndex = variants.findIndex(v => v.references.audios.length === 0);
    assert(crossIndex >= 0 && targetIndex >= 0 && visualIndex >= 0);
    await page.locator("#final-variant").selectOption(String(crossIndex));
    assert.strictEqual(await page.locator("#final-text").textContent(), variants[crossIndex].text);
    assert.strictEqual(await page.locator("#audio-references audio").count(), 1);
    assert.match(await page.locator("#audio-references").textContent(), /cross_voice.*donor\/e3/s);
    await page.locator("#final-variant").selectOption(String(targetIndex));
    assert.match(await page.locator("#audio-references").textContent(), /target_voice.*existing_target_voice/s);
    await page.waitForFunction(() => document.querySelector("#audio-references audio").readyState >= 1);
    const audioURL = variants[targetIndex].references.audios[0].url;
    assert((await page.locator("#audio-references audio").getAttribute("src")).endsWith(audioURL));
    await page.locator("#final-variant").selectOption(String(visualIndex));
    assert.strictEqual(await page.locator("#audio-references audio").count(), 0);
    const label = name => page.locator('input[data-qa-label="' + name + '"]');
    async function assertSelection(expected) {
      assert.deepStrictEqual(await page.locator("input[data-qa-label]:checked").evaluateAll(
        inputs => inputs.map(input => input.dataset.qaLabel)), expected);
      assert.deepStrictEqual(await page.evaluate(() => stateFor(data.clips[index]).labels), expected);
    }
    await label("better").check();
    await label("same").check(); await assertSelection(["same"]);
    await label("same").uncheck(); await assertSelection([]);
    await label("dialogue_wrong").check(); await assertSelection(["dialogue_wrong"]);
    await label("audio_wrong").check(); await assertSelection(["dialogue_wrong", "audio_wrong"]);
    await label("dialogue_wrong").uncheck(); await label("audio_wrong").uncheck(); await assertSelection([]);
    await page.locator('input[data-qa-label="audio_wrong"]').check();
    await page.locator("#notes").fill("synthetic human note");
    await page.locator("#next").click();
    assert.strictEqual(await page.locator("#status").textContent(), "failed");
    assert(await page.locator("#final-unavailable").isVisible());
    assert.match(await page.locator("#final-unavailable").textContent(), /Final H3 Prompt unavailable.*AV reconcile failed/s);
    assert.strictEqual(await page.locator("#audio-references audio").count(), 0);
    assert(await page.locator("#reference-unavailable").isVisible());
    assert.strictEqual(await page.locator("#final-text").textContent(), "");
    assert(await page.locator("#copy-final").isDisabled());
    assert.match(await page.locator("#reconcile-body").textContent(), /synthetic_failed/);
    await page.locator("#previous").click();
    assert.strictEqual(await page.locator("#final-text").textContent(), expectedFinal);
    assert(await page.locator("#final-text").isVisible());
    assert(!await page.locator("#final-unavailable").isVisible());
    assert.strictEqual(await page.locator("#notes").inputValue(), "synthetic human note");
    const downloaded = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloaded;
    const payload = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.deepStrictEqual(payload.annotations[0].labels, ["audio_wrong"]);
    await page.evaluate(() => localStorage.removeItem(key(data.clips[0])));
    await page.reload(); await page.locator("#main").waitFor({state:"visible"});
    await page.locator("#import-file").setInputFiles({name:"qa.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(payload))});
    await page.waitForFunction(() => document.getElementById("notes").value === "synthetic human note");
    const contradictory = JSON.parse(JSON.stringify(payload));
    contradictory.annotations[0].labels = ["better", "same"];
    await page.locator("#import-file").setInputFiles({name:"invalid.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(contradictory))});
    await page.waitForFunction(() => document.getElementById("message").textContent === "Invalid, duplicate, or stale QA annotation.");
    await assertSelection(["audio_wrong"]);
    await page.locator("#final-variant").selectOption("2");
    await page.evaluate(() => {
      const variant = data.clips[index].final_h3.variants[2];
      variant.warnings = ["<Subject 1>:preserved_visual_subject_missing_from_detailed_description"];
      const template = variant.references.audios[0];
      variant.references.audios = ["target_voice", "cross_voice", "speaker_speech_reuse", "music_reuse", "music_reference", "full_audio_reuse", "future_kind"].map((kind, i) => ({
        ...template, audio_label:"<Audio " + (i+1) + ">", kind,
        voice_characteristics:"<script>window.injected=true</script>", provenance:{donor_clip_uid:null}
      }));
      renderFinalVariant();
    });
    assert.strictEqual(await page.locator("#audio-references audio").count(), 7);
    assert.match(await page.locator("#variant-warnings").textContent(), /preserved_visual_subject_missing/);
    assert.strictEqual(await page.locator("#variant-warnings.failed").count(), 0);
    assert.strictEqual(await page.evaluate(() => window.injected), undefined);
    assert.strictEqual(await page.locator("#audio-references script").count(), 0);
    await page.evaluate(clip => { data.clips[0] = clip; document.getElementById("message").textContent = ""; }, dataset.clips[0]);
    const fullIndex = variants.findIndex(v => v.conditioning_variant === "full_audio_reuse" && v.publication_status === "published");
    await page.locator("#final-variant").selectOption(String(fullIndex));
    await page.getByText("Reference Variant Coverage", {exact:true}).click();
    await page.screenshot({path:path.join(process.argv[4], "qa-desktop.png"), fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(process.argv[4], "qa-mobile.png"), fullPage:true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.evaluate(() => localStorage.setItem(key(data.clips[0]),
      JSON.stringify({labels:["better", "same"], notes:"injected"})));
    await page.reload();
    await page.waitForFunction(() => document.getElementById("message").textContent.startsWith("Invalid saved QA state"));
    assert.strictEqual(await page.locator("#main").isVisible(), false);
    assert.deepStrictEqual(errors, []);
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
''')
    result = subprocess.run(
        [node, str(script), playwright, str(shadow / "qa"), str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
