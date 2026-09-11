from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.h3 import t2va_shadow as t2va
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_CANONICAL_ABSENT_SOUNDSCAPE,
    MimoAudioFinalizeDraft,
    MimoMediaResolver,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    run_stem_diarization_shadow,
    run_stem_qwen3_asr_shadow,
)
from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoBackend, T2VAMimoConfig
from r2v_data_v2.h3.t2va_qa import build_t2va_qa
from r2v_data_v2.h3.t2va_source import prepare_t2va_audio, select_t2va_shots
from r2v_data_v2.naming import clip_uid
from tests import test_h3_auk_speech_shadow as auk_tests
from tests.test_h3_resolved_audio_stems import _resolve
from tests.test_h3_sam_audio_stem_shadow import (
    _Diarization,
    _Qwen,
)

setup = auk_tests.setup
ffmpeg = auk_tests.ffmpeg


class Audio:
    def materialize_full_audio(self, *, source_video_path, destination, **kwargs):
        import soundfile as sf

        assert source_video_path.suffix == ".mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.full((32000, 2), 0.125), 32000, subtype="PCM_24")


def shot_ids(tmp_path):
    return [clip_uid(tmp_path / f"movie_{i}.mp4") for i in (1, 2, 3)]


def shot_manifest(tmp_path):
    rows = []
    for i in (1, 2, 3):
        path = tmp_path / f"movie_{i}.mp4"
        path.write_bytes(f"original AV {i}".encode())
        rows.append(
            {
                "video_path": str(path),
                "source_video_path": str(tmp_path / "movie.mp4"),
                "source_video_id": "movie",
                "shot_index": i,
                "duration": 1,
            }
        )
    manifest = tmp_path / "shots_f03_motion.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


@pytest.fixture
def job(tmp_path):
    video = tmp_path / "original.mp4"
    video.write_bytes(b"original AV fixture")
    evidence = {
        "resolved_root": str(tmp_path / "resolved_stems_v1"),
        "resolved_record_fingerprint": "b" * 64,
    }
    for kind in ("full_audio", "speech", "music", "sfx"):
        path = tmp_path / f"{kind}.wav"
        import soundfile as sf

        sf.write(path, np.zeros((160000, 2)), 32000, subtype="PCM_16")
        evidence[f"{kind}_path"] = str(path)
        evidence[f"{kind}_sha256"] = t2va.sha256_file(path)
    return t2va.T2VAJob(
        clip_uid="clip",
        clip_display_path="collection/episode/clip",
        target_video_path=str(video),
        target_video_sha256=t2va.sha256_file(video),
        target_duration_seconds=5,
        audio_evidence=t2va.T2VAAudioEvidence(**evidence),
        speech_facts=[
            t2va.T2VASpeechFact(
                segment_id=f"segment_{i}",
                source_speaker_cluster=f"cluster_{i}",
                start_time=i,
                end_time=i + 0.5,
                text=text,
                language="Chinese",
            )
            for i, text in enumerate(["你好。", "好的！"], 1)
        ],
    )


def draft_for(job):
    speakers = {}
    assignments, parts = (
        [],
        [{"kind": "prose", "text": "[Shot 1] Live-action, a static medium shot."}],
    )
    for fact in job.speech_facts:
        speaker = speakers.setdefault(
            fact.source_speaker_cluster, f"S{len(speakers) + 1}"
        )
        assignments.append(
            {
                "segment_id": fact.segment_id,
                "source_speaker_cluster": fact.source_speaker_cluster,
                "speaker_id": speaker,
                "speech_presentation": "onscreen_spoken",
            }
        )
        if fact.text is not None:
            parts.append(
                {
                    "kind": "speech",
                    "segment_id": fact.segment_id,
                    "lead_in": "A person says",
                }
            )
    return t2va.T2VAMimoDraft(
        summary="People converse in a quiet setting.",
        speaker_assignments=assignments,
        integrated_sequence=parts,
        warnings=[],
    )


def audio_for():
    return MimoAudioFinalizeDraft(
        overall_soundscape=MIMO25_CANONICAL_ABSENT_SOUNDSCAPE, non_diegetic_music="N/A"
    )


def materialize(job, draft):
    return t2va.validate_t2va_draft(job, draft, audio_for())


def config(tmp_path):
    return T2VAMimoConfig(
        media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        base_url="http://127.0.0.1:8092/v1",
        api_key="fake",
    )


class Client:
    def __init__(self, responses, *, audio_response=None):
        self.responses = iter(responses)
        self.audio_response = (
            audio_response
            if audio_response is not None
            else audio_for().model_dump_json()
        )
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(request)
        finalizer = (
            request["messages"][0]["content"] == t2va.AUDIO_FINALIZE_SYSTEM_PROMPT
        )
        response = self.audio_response if finalizer else next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": response}}],
            "usage": {"prompt_tokens_details": {"video_tokens": 20, "audio_tokens": 5}},
        }


def test_core_renderer_exact_and_reusable(job):
    core = materialize(job, draft_for(job))
    restored = t2va.H3NoReferenceAVCore.model_validate_json(core.model_dump_json())
    assert t2va.render_t2va_prompt(core) == t2va.render_t2va_prompt(restored)
    prompt = t2va.render_t2va_prompt(restored)
    assert [line[:-1] for line in prompt.splitlines() if line.endswith(":")] == list(
        t2va.SECTIONS
    )
    assert "<d>[Chinese] 你好。</d>" in prompt
    assert "non_diegetic_music:\nN/A" in prompt


@pytest.mark.parametrize("label", ["Picture", "Subject", "Audio", "Video"])
@pytest.mark.parametrize("field", ["prose", "lead_in", *t2va.SECTIONS[1:], "warnings"])
def test_no_reference_conditioning(job, label, field):
    if field in t2va.SECTIONS[1:]:
        audio = audio_for()
        setattr(audio, field, f"<{label} 1>")
        with pytest.raises(ValueError, match="syntax"):
            t2va.validate_t2va_draft(job, draft_for(job), audio)
        return
    data = draft_for(job).model_dump()
    if field in {"prose", "lead_in"}:
        data["integrated_sequence"][0 if field == "prose" else 1][
            "text" if field == "prose" else "lead_in"
        ] = f"<{label} 1>"
    else:
        data[field] = [f"<{label} 1>"] if field == "warnings" else f"<{label} 1>"
    with pytest.raises(ValueError, match="conditioning"):
        t2va.T2VAMimoDraft.model_validate(data)


@pytest.mark.parametrize(
    "header",
    [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        *t2va.SECTIONS,
    ],
)
def test_no_embedded_section_headers(job, header):
    data = draft_for(job).model_dump()
    data["integrated_sequence"][0]["text"] += f"\n{header}: text"
    with pytest.raises(ValueError, match="headers"):
        t2va.T2VAMimoDraft.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "language",
        "drop",
        "duplicate",
        "reorder",
        "speaker",
        "late_marker",
        "nested",
    ],
)
def test_exact_dialogue_fails_without_repair(job, change):
    draft = draft_for(job)
    caption = materialize(job, draft).integrated_multimodal_description
    if change == "text":
        caption = caption.replace("你好。", "Hello.")
    elif change == "language":
        caption = caption.replace("[Chinese]", "[English]")
    elif change == "drop":
        caption = "[Shot 1] No dialogue."
    elif change == "duplicate":
        caption += " (S2) <d>[Chinese] 好的！</d>"
    elif change == "reorder":
        caption = (
            caption.replace("你好。", "TMP")
            .replace("好的！", "你好。")
            .replace("TMP", "好的！")
        )
    elif change == "speaker":
        caption = caption.replace("(S2)", "(S1)")
    elif change == "late_marker":
        caption = caption.replace("(S2)", "") + " (S2)"
    else:
        caption = caption.replace("你好。", "<d>你好。</d>")
    with pytest.raises(ValueError):
        t2va.validate_t2va_dialogue(
            job.speech_facts, draft.speaker_assignments, caption
        )


def test_speaker_cluster_cannot_split_and_ids_contiguous(job):
    draft = draft_for(job)
    job.speech_facts[1].source_speaker_cluster = "cluster_1"
    draft.speaker_assignments[1].source_speaker_cluster = "cluster_1"
    with pytest.raises(ValueError, match="cannot split"):
        materialize(job, draft)
    clean = draft_for(job)
    assert len({a.speaker_id for a in materialize(job, clean).speaker_assignments}) == 1
    clean.speaker_assignments[0].speaker_id = "S3"
    clean.speaker_assignments[1].speaker_id = "S3"
    with pytest.raises(ValueError, match="contiguous"):
        materialize(job, clean)


def test_acoustic_clusters_can_merge_without_entity_binding(job):
    draft = draft_for(job)
    draft.speaker_assignments[1].speaker_id = "S1"
    core = materialize(job, draft)
    assert [a.speaker_id for a in core.speaker_assignments] == ["S1", "S1"]
    assert "entity_id" not in core.model_dump_json()


def test_empty_asr_never_invents_dialogue(job):
    job.speech_facts = []
    draft = draft_for(job)
    materialize(job, draft)
    draft.integrated_sequence[0].text += " <d>[English] Invented</d>"
    with pytest.raises(ValueError, match="syntax"):
        materialize(job, draft)


def test_nontranscribed_vocal_segment_keeps_assignment_without_dialogue(job):
    job.speech_facts[0].text = job.speech_facts[0].language = None
    core = materialize(job, draft_for(job))
    assert len(core.speaker_assignments) == 2
    assert core.integrated_multimodal_description.count("<d>") == 1


@pytest.mark.parametrize("field", ["overall_soundscape", "non_diegetic_music"])
def test_no_dialogue_in_audio_fields(job, field):
    audio = audio_for()
    setattr(audio, field, "<d>[Chinese] copied</d>")
    with pytest.raises(ValueError, match="syntax"):
        t2va.validate_t2va_draft(job, draft_for(job), audio)


@pytest.mark.parametrize(
    "suffix,valid",
    [
        (" [Shot 2] At 00:03.000, the camera cuts to the doorway.", True),
        (" [Shot 3] At 00:03.000, the camera cuts.", False),
        (" [Shot 2] At 00:05.000, the camera cuts.", False),
        (" [Shot 2] At 00:03.000, cut. [Shot 3] At 00:02.000, cut.", False),
        (" [Shot 2] The camera cuts.", False),
    ],
)
def test_shot_order_and_boundaries(job, suffix, valid):
    draft = draft_for(job)
    draft.integrated_sequence.append(t2va.T2VAProsePart(kind="prose", text=suffix))
    if valid:
        materialize(job, draft)
    else:
        with pytest.raises(ValueError):
            materialize(job, draft)


def test_first_shot_timestamp_rejected(job):
    draft = draft_for(job)
    draft.integrated_sequence[0].text = draft.integrated_sequence[0].text.replace(
        "[Shot 1]", "[Shot 1] At 00:00.000,"
    )
    with pytest.raises(ValueError, match="first shot"):
        materialize(job, draft)


def test_selection_modes(tmp_path):
    ids = ["c", "a", "b", "d"]
    assert t2va.select_clip_uids(ids) == ids
    sampled = t2va.select_clip_uids(ids, sample_size=3, sample_seed=12)
    assert sampled == t2va.select_clip_uids(ids, sample_size=3, sample_seed=12)
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"clip_uids": ["b", "c"]}))
    assert t2va.select_clip_uids(ids, case_manifest=manifest) == ["b", "c"]
    for kwargs in (
        {"case_manifest": manifest, "sample_size": 2, "sample_seed": 1},
        {"sample_size": 3},
        {"sample_seed": 1},
        {"sample_size": 9, "sample_seed": 1},
    ):
        with pytest.raises(ValueError):
            t2va.select_clip_uids(ids, **kwargs)
    for clips in (["b", "b"], ["unknown"]):
        manifest.write_text(json.dumps({"clip_uids": clips}))
        with pytest.raises(ValueError):
            t2va.select_clip_uids(ids, case_manifest=manifest)


@pytest.fixture
def finalized(setup, tmp_path, ffmpeg):
    shots = shot_manifest(tmp_path)
    selection = select_t2va_shots(shots)
    production = prepare_t2va_audio(
        selection,
        output_root=tmp_path / "no-reference-audio",
        audio_backend=Audio(),
    )
    inventory = auk_tests.auk.build_auk_inventory(
        audio_production_root=production,
        shadow_run_id=setup.shadow_run_id,
        case_manifest_path=production / "case_manifest.json",
        configuration=setup.model_configuration,
    )
    root, _ = _resolve(inventory, tmp_path, ffmpeg, fail=selection.shots[0].clip_uid)
    run_stem_diarization_shadow(
        stem_root=root,
        production_diarization_root=production / "diarization",
        backend=_Diarization(),
        route="resolved",
        output_root=root.parent / "diarization",
        allow_unverified=True,
    )
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=root.parent / "diarization",
        source_visual_production_root=None,
        backend=_Qwen(),
        output_root=root.parent / "asr",
        route="resolved",
        allow_unverified=True,
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
    )
    return production, setup.shadow_run_id


def build(finalized, tmp_path, **kwargs):
    production, run_id = finalized
    return t2va.build_t2va_inventory(
        shot_manifest=tmp_path / "shots_f03_motion.jsonl",
        audio_production_root=production,
        audio_shadow_run_id=run_id,
        t2va_run_id="t2va-test",
        backend=config(tmp_path).provenance(),
        **kwargs,
    )


def test_finalized_source_projection_and_publication(finalized, tmp_path):
    inventory = build(finalized, tmp_path)
    assert inventory.clip_uids == shot_ids(tmp_path)
    assert inventory.jobs[0].upstream_failure
    facts_json = json.dumps([j.model_dump(mode="json") for j in inventory.jobs])
    for forbidden in (
        "entity_id",
        "Subject",
        "Picture",
        "Visual",
        "face_",
        "stem_path",
    ):
        assert forbidden not in facts_json
    production = finalized[0]
    before = {p: p.read_bytes() for p in production.rglob("*") if p.is_file()}
    ready = [j for j in inventory.jobs if not j.upstream_failure]
    client = Client([draft_for(j).model_dump_json() for j in ready])
    backend = T2VAMimoBackend(config(tmp_path), client=client)
    summary = t2va.run_t2va_shadow(inventory, backend)
    assert (
        summary.record_count,
        summary.ready_count,
        summary.skipped_count,
        summary.model_call_count,
    ) == (3, 2, 1, 4)
    assert len(client.calls) == 4
    root = t2va.t2va_root(production, "t2va-test")
    loaded, records, restored = t2va.load_t2va_shadow(root)
    assert loaded == inventory and restored == summary
    assert records[0].status == "skipped"
    assert not (root / "core" / f"{inventory.clip_uids[0]}.json").exists()
    page = build_t2va_qa(root)
    payload = json.loads(page.with_name("data.json").read_text())
    assert payload["cases"][1]["speech"][0]["assignment"]["speaker_id"] == "S1"
    assert payload["cases"][1]["video_url"].endswith("movie_2.mp4")
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not summary.production_artifacts_modified
    with pytest.raises(FileExistsError):
        t2va.run_t2va_shadow(inventory, backend)
    assert len(client.calls) == 4


def test_case_manifest_order_and_dry_run_no_client(finalized, tmp_path, monkeypatch):
    from tools import run_h3_t2va_shadow as cli

    manifest = tmp_path / "selected.json"
    ids = shot_ids(tmp_path)
    manifest.write_text(json.dumps({"clip_uids": [ids[2], ids[1]]}))
    monkeypatch.setattr(
        cli,
        "T2VAMimoBackend",
        lambda *a, **kw: pytest.fail("dry run constructed model client"),
    )
    report = cli.main(
        [
            "--shot-manifest",
            str(tmp_path / "shots_f03_motion.jsonl"),
            "--audio-production-root",
            str(finalized[0]),
            "--audio-shadow-run-id",
            finalized[1],
            "--t2va-run-id",
            "dry",
            "--case-manifest",
            str(manifest),
            "--media-root",
            str(tmp_path),
            "--base-url",
            "http://localhost/v1",
            "--dry-run",
        ]
    )
    assert report["clip_uids"] == [ids[2], ids[1]] and report["model_call_count"] == 0
    assert not Path(report["output_root"]).exists()


def test_one_failed_clip_does_not_stop_next(finalized, tmp_path):
    inventory = build(finalized, tmp_path)
    client = Client(["not JSON", draft_for(inventory.jobs[2]).model_dump_json()])
    result = t2va.run_t2va_shadow(
        inventory, T2VAMimoBackend(config(tmp_path), client=client)
    )
    assert (
        result.ready_count,
        result.failed_count,
        result.skipped_count,
        result.model_call_count,
    ) == (1, 1, 1, 3)
    root = t2va.t2va_root(finalized[0], "t2va-test")
    assert (
        json.loads((root / "raw" / f"{inventory.clip_uids[1]}.json").read_text())[
            "response"
        ]
        == "not JSON"
    )


def test_lineage_mismatch_before_call(finalized, tmp_path):
    inventory = build(finalized, tmp_path)
    source = next(
        Path(p) for p in inventory.source_hashes if p.endswith("asr/segments.jsonl")
    )
    source.write_text(source.read_text() + "\n")
    client = Client([])
    with pytest.raises(ValueError, match="provenance changed"):
        t2va.run_t2va_shadow(
            inventory, T2VAMimoBackend(config(tmp_path), client=client)
        )
    assert client.calls == []


def test_atomic_overwrite_keeps_old_stage_on_exception(
    finalized, tmp_path, monkeypatch
):
    inventory = build(finalized, tmp_path)
    backend = T2VAMimoBackend(
        config(tmp_path),
        client=Client(
            [
                draft_for(j).model_dump_json()
                for j in inventory.jobs
                if not j.upstream_failure
            ]
        ),
    )
    t2va.run_t2va_shadow(inventory, backend)
    root = t2va.t2va_root(finalized[0], "t2va-test")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    backend.client = Client(
        [
            draft_for(j).model_dump_json()
            for j in inventory.jobs
            if not j.upstream_failure
        ]
    )
    monkeypatch.setattr(
        t2va,
        "_publish_directory",
        lambda *a, **kw: (_ for _ in ()).throw(
            OSError("synthetic publication failure")
        ),
    )
    with pytest.raises(OSError, match="publication"):
        t2va.run_t2va_shadow(inventory, backend, overwrite=True)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not list(root.parent.glob(".*.tmp-*"))


@pytest.mark.parametrize("run_id", ["../audio", "/audio", "..", "a/b", ""])
def test_owned_namespace_rejects_traversal(tmp_path, run_id):
    with pytest.raises(ValueError):
        t2va.t2va_root(tmp_path, run_id)


def test_symlink_namespace_rejected(tmp_path):
    production = tmp_path / "production"
    production.mkdir()
    (production / "t2va_shadow_v1").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="redirect"):
        t2va.t2va_root(production, "x")


@pytest.mark.parametrize(
    "field,value",
    [
        ("speaker_cluster_id", "different_acoustic_cluster"),
        ("source_audio_path", "/wrong/source.wav"),
    ],
)
def test_projection_reconciliation_remains_strict(
    finalized, tmp_path, monkeypatch, field, value
):
    root = t2va.stem_shadow_root(finalized[0], finalized[1])
    validated = t2va.validate_stem_asr_lineage(root / "asr", expected_shadow_root=root)
    monkeypatch.setattr(t2va, "validate_stem_asr_lineage", lambda *a, **kw: validated)
    path = root / "asr/segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="reconciliation differs"):
        build(finalized, tmp_path)


def test_failed_asr_is_not_reinterpreted_as_empty_speech(
    finalized, tmp_path, monkeypatch
):
    root = t2va.stem_shadow_root(finalized[0], finalized[1])
    validated = t2va.validate_stem_asr_lineage(root / "asr", expected_shadow_root=root)
    monkeypatch.setattr(t2va, "validate_stem_asr_lineage", lambda *a, **kw: validated)
    path = root / "asr/segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    uid = rows[0]["clip_uid"]
    rows[0].update(
        status="failed",
        text=None,
        language=None,
        failure_reason="synthetic ASR failure",
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inventory = build(finalized, tmp_path)
    assert (
        next(j for j in inventory.jobs if j.clip_uid == uid).upstream_failure
        == "asr_failed"
    )


def test_legacy_speech_lineage_not_accepted(finalized, tmp_path, monkeypatch):
    root = t2va.stem_shadow_root(finalized[0], finalized[1])
    asr, diarization = t2va.validate_stem_asr_lineage(
        root / "asr", expected_shadow_root=root
    )
    monkeypatch.setattr(
        t2va,
        "validate_stem_asr_lineage",
        lambda *a, **kw: (
            asr,
            diarization.model_copy(
                update={
                    "route": "voice_first",
                    "source_stem_root": str(root / "separation"),
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="never legacy SAM"):
        build(finalized, tmp_path)


def test_source_binding_fields_are_dropped_not_transmitted(
    finalized, tmp_path, monkeypatch
):
    root = t2va.stem_shadow_root(finalized[0], finalized[1])
    validated = t2va.validate_stem_asr_lineage(root / "asr", expected_shadow_root=root)
    monkeypatch.setattr(t2va, "validate_stem_asr_lineage", lambda *a, **kw: validated)
    path = root / "asr/segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(entity_id="e99", entity_occurrence_id="private_entity_occurrence")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inventory = build(finalized, tmp_path)
    for item in inventory.jobs:
        request = T2VAMimoBackend(config(tmp_path), client=Client([])).build_request(
            item
        )
        assert '"e99"' not in request["messages"][1]["content"][1]["text"]
        assert "private_entity_occurrence" not in json.dumps(request)


@pytest.mark.parametrize("marker", ["S0", "S01", "S9"])
def test_unknown_speaker_markers_rejected(job, marker):
    draft = draft_for(job)
    draft.integrated_sequence[0].text += f" ({marker})"
    with pytest.raises(ValueError, match="speaker markers"):
        materialize(job, draft)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reorder", "unknown", "nontranscribed"]
)
def test_typed_speech_inventory_fails_closed(job, mutation):
    draft = draft_for(job)
    if mutation == "missing":
        draft.integrated_sequence.pop()
    elif mutation == "duplicate":
        draft.integrated_sequence.append(draft.integrated_sequence[-1])
    elif mutation == "reorder":
        draft.integrated_sequence[1:] = reversed(draft.integrated_sequence[1:])
    elif mutation == "unknown":
        draft.integrated_sequence[-1].segment_id = "unknown"
    else:
        job.speech_facts[-1].text = job.speech_facts[-1].language = None
    with pytest.raises(ValueError, match="speech placement inventory"):
        materialize(job, draft)


@pytest.mark.parametrize("prose", [
    "The man speaks in Chinese.",
    "The audio features a male voice speaking in Chinese.",
    "The man then speaks again in Chinese.",
    "A woman talks to the other person.",
])
def test_missing_slots_prose_substitution_is_diagnostic_only(job, prose):
    draft = draft_for(job)
    draft.integrated_sequence = [
        t2va.T2VAProsePart(kind="prose", text="[Shot 1] " + prose)
    ]
    with pytest.raises(ValueError, match="possible_prose_substitution"):
        materialize(job, draft)
    correct = draft_for(job)
    correct.integrated_sequence[0].text += " " + prose
    assert materialize(job, correct)
    for fact in job.speech_facts:
        fact.text = fact.language = None
    assert materialize(job, draft)


def test_zero_slots_without_speech_prose_keeps_inventory_failure(job):
    draft = draft_for(job)
    draft.integrated_sequence = draft.integrated_sequence[:1]
    with pytest.raises(ValueError, match="speech placement inventory") as exc:
        materialize(job, draft)
    assert "possible_prose_substitution" not in str(exc.value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "reorder"])
def test_assignment_inventory_still_exact(job, mutation):
    draft = draft_for(job)
    if mutation == "missing":
        draft.speaker_assignments.pop()
    elif mutation == "duplicate":
        draft.speaker_assignments.append(draft.speaker_assignments[-1])
    elif mutation == "unknown":
        draft.speaker_assignments[-1].segment_id = "unknown"
    else:
        draft.speaker_assignments.reverse()
    with pytest.raises(ValueError, match="speaker assignment inventory"):
        materialize(job, draft)


def test_asr_not_model_owned_and_placement_preserves_exact_bytes(job):
    draft = draft_for(job)
    raw = draft.model_dump_json()
    assert "<d>" not in raw
    assert all(fact.text not in raw for fact in job.speech_facts)
    draft.integrated_sequence.insert(
        2,
        t2va.T2VAProsePart(kind="prose", text="The listener turns toward the doorway."),
    )
    core = materialize(job, draft)
    prompt = t2va.render_t2va_prompt(core)
    assert (
        prompt.index("你好。</d>")
        < prompt.index("listener turns")
        < prompt.index("好的！</d>")
    )
    assert t2va._DIALOGUE.findall(prompt) == [
        f"[{s.language}] {s.text}" for s in job.speech_facts
    ]
    for field in ["text", "translation", "transliteration", "language", "asr_text"]:
        data = draft.model_dump()
        data["integrated_sequence"][1][field] = "Hello."
        with pytest.raises(ValueError, match="Extra inputs"):
            t2va.T2VAMimoDraft.model_validate(data)


@pytest.mark.parametrize(
    "value",
    [
        "<d>你好。</d>",
        "(S1) speaks",
        "<Audio 1>",
        "[[segment:segment_1]]",
        "[Shot 2]",
        "A person says 你好。",
    ],
)
def test_lead_in_cannot_own_dialogue_or_pipeline_syntax(job, value):
    draft = draft_for(job)
    draft.integrated_sequence[1].lead_in = value
    with pytest.raises(ValueError):
        materialize(job, draft)


def test_voice_over_rendering_outside_exact_dialogue(job):
    draft = draft_for(job)
    draft.speaker_assignments[0].speech_presentation = "voice_over"
    core = materialize(job, draft)
    assert (
        "says in an off-screen voiceover (S1) <d>[Chinese] 你好。</d>"
        in core.integrated_multimodal_description
    )
    assert (
        "voiceover"
        not in t2va._DIALOGUE.findall(core.integrated_multimodal_description)[0]
    )
    assert draft.integrated_sequence[1].lead_in == "A person says"


def test_core_cannot_substitute_its_own_asr_facts(job):
    core = materialize(job, draft_for(job))
    data = core.model_dump()
    data["speech_facts"][0]["text"] = "Hello."
    changed = t2va.H3NoReferenceAVCore.model_validate(data)
    with pytest.raises(ValueError, match="core speech facts differ"):
        materialize(job, changed)


@pytest.mark.parametrize("token", ["S1", "(S1)", "S2", "(S2)", "S123"])
@pytest.mark.parametrize(
    "field", ["prose", "lead_in", "overall_soundscape", "non_diegetic_music"]
)
def test_model_owned_speaker_tokens_forbidden(job, token, field):
    if field in t2va.SECTIONS[1:]:
        audio = audio_for()
        setattr(audio, field, f"{token} makes a sound.")
        with pytest.raises(ValueError, match="syntax"):
            t2va.validate_t2va_draft(job, draft_for(job), audio)
        return
    draft = draft_for(job)
    if field == "prose":
        draft.integrated_sequence[0].text += f" {token} speaks."
    elif field == "lead_in":
        draft.integrated_sequence[1].lead_in = f"{token} speaks"
    else:
        setattr(draft, field, f"{token} makes a sound.")
    with pytest.raises(ValueError, match="speaker markers"):
        materialize(job, draft)


def test_v2_run_requires_new_id_before_call(finalized, tmp_path, monkeypatch):
    inventory = build(finalized, tmp_path)
    destination = t2va.t2va_root(
        Path(inventory.audio_production_root), inventory.t2va_run_id
    )
    destination.mkdir(parents=True)
    values = inventory.model_dump(mode="json")
    values["backend"]["schema_version"] = "r2v.h3.t2va_mimo_backend.2"
    values["backend"]["prompt_version"] = "h3_t2va_joint_av_v2"
    values["inventory_fingerprint"] = t2va.fingerprint(
        {k: v for k, v in values.items() if k != "inventory_fingerprint"}
    )
    path = destination / "inventory.json"
    path.write_text(json.dumps(values))
    before = path.read_bytes()
    monkeypatch.setattr(
        t2va,
        "_check_sources",
        lambda _: pytest.fail("contract mismatch must fail before source checks"),
    )
    client = Client([])
    with pytest.raises(ValueError, match="new T2VA run ID"):
        t2va.run_t2va_shadow(
            inventory, T2VAMimoBackend(config(tmp_path), client=client), overwrite=True
        )
    assert not client.calls
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "lead", ["S1 speaks calmly", "(S1) speaks calmly", "A woman (S1) speaks calmly"]
)
def test_matching_lead_marker_cleanup_is_auditable(job, lead):
    raw = draft_for(job).model_dump()
    raw["integrated_sequence"][1]["lead_in"] = lead
    serialized = json.dumps(raw)
    draft, corrections = t2va.parse_t2va_semantic(job, serialized)
    assert "S1" not in draft.integrated_sequence[1].lead_in
    assert corrections == {"redundant_speech_lead_in_speaker_marker": 1}
    assert json.loads(serialized) == raw
    assert "(S1) <d>[Chinese] 你好。</d>" in t2va.render_t2va_prompt(
        materialize(job, draft)
    )


@pytest.mark.parametrize(
    "lead",
    [
        "S2 speaks",
        "S1 and S2 speak",
        "S99 speaks",
        "(S1)",
        "S1,",
        "( S1",
        "A voice, S1, speaks",
    ],
)
def test_lead_marker_cleanup_never_guesses_or_leaves_malformed(job, lead):
    raw = draft_for(job).model_dump()
    raw["integrated_sequence"][1]["lead_in"] = lead
    with pytest.raises(ValueError):
        t2va.parse_t2va_semantic(job, json.dumps(raw))


def test_cleanup_does_not_allow_sx_in_visual_prose(job):
    raw = draft_for(job).model_dump()
    raw["integrated_sequence"][0]["text"] += " S1 stands near the door."
    with pytest.raises(ValueError, match="speaker markers"):
        t2va.parse_t2va_semantic(job, json.dumps(raw))


def test_bounded_strings_and_completed_repetition(job):
    schema = t2va.T2VAMimoDraft.model_json_schema()["$defs"]
    assert schema["T2VAProsePart"]["properties"]["text"]["maxLength"] == 12000
    assert schema["T2VASpeechPart"]["properties"]["lead_in"]["maxLength"] == 512
    sentence = "The seated man holds his hands together while the standing man stays beside the doorway without changing his posture."
    raw = draft_for(job).model_dump()
    raw["integrated_sequence"][0]["text"] += f" {sentence} {sentence.upper()} {sentence}"
    with pytest.raises(ValueError, match="sentence loop"):
        t2va.T2VAMimoDraft.model_validate(raw)
    block = (
        sentence
        + " The camera remains fixed on the two men with the window and table still visible behind them."
    )
    raw["integrated_sequence"][0]["text"] = (
        "[Shot 1] " + block + " " + block + " " + block
    )
    with pytest.raises(ValueError, match="loop|block"):
        t2va.T2VAMimoDraft.model_validate(raw)
    raw["integrated_sequence"][0]["text"] = (
        "[Shot 1] A person waves. A person waves. A person waves."
    )
    t2va.T2VAMimoDraft.model_validate(raw)
    raw["integrated_sequence"][0]["text"] = "x" * 12001
    with pytest.raises(ValueError):
        t2va.T2VAMimoDraft.model_validate(raw)


def test_inventory_binds_resolved_audio_and_no_alternate_route(finalized, tmp_path):
    inventory = build(finalized, tmp_path)
    for job in inventory.jobs:
        if job.upstream_failure is None:
            assert job.audio_evidence.resolved_root.endswith("/resolved_stems_v1")
            assert Path(job.audio_evidence.music_path).is_file()
            t2va.check_audio_files(job.audio_evidence)
    t2va.validate_t2va_audio_lineage(inventory)
    good = next(j for j in inventory.jobs if j.upstream_failure is None)
    good.audio_evidence.music_path = good.audio_evidence.sfx_path
    good.audio_evidence.music_sha256 = good.audio_evidence.sfx_sha256
    with pytest.raises(ValueError, match="differs from resolved lineage"):
        t2va.validate_t2va_audio_lineage(inventory)
