from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3 import resolved_audio_stems as resolved
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    run_stem_diarization_shadow,
    run_stem_qwen3_asr_shadow,
    separation_skips,
    sha256_file,
    validate_stem_asr_lineage,
)
from tests import test_h3_auk_speech_shadow as auk_tests
from tests.test_h3_auk_speech_shadow import run, sam_baseline
from tests.test_h3_sam_audio_stem_shadow import (
    _Diarization,
    _production_diarization_inventory_for_records,
    _Qwen,
)

setup = auk_tests.setup
ffmpeg = auk_tests.ffmpeg


def _resolve(inventory, tmp_path, ffmpeg, *, fail=None):
    sam_baseline(inventory, tmp_path, route="music_first")
    run(inventory, ffmpeg, fail=fail)
    production = Path(inventory.audio_production_root)
    root = resolved.downstream_stem_root(production, inventory.shadow_run_id)
    before = {p: sha256_file(p) for p in production.rglob("*") if p.is_file()}
    summary = resolved.resolve_audio_stems(
        audio_production_root=production,
        shadow_run_id=inventory.shadow_run_id,
    )
    assert summary.model_call_count == 0
    assert not summary.production_artifacts_modified
    assert all(sha256_file(p) == h for p, h in before.items())
    return root, resolved.load_stem_source(root)


def test_fixed_sources_and_determinism(setup, tmp_path, ffmpeg):
    root, (inventory, records, summary) = _resolve(setup, tmp_path, ffmpeg)
    assert inventory.clip_uids == ["c", "a", "b"]
    sam_inventory, _, _ = resolved.load_stem_shadow(root.parent / "separation")
    assert sam_inventory.route == "music_first"
    assert not sam_inventory.run_both_routes
    assert summary.ready_count == 3
    for record in records:
        assert record.speech.source_backend == "auk"
        assert "/auk_speech_v1/canonical/" in record.speech.canonical_stem_path
        for stem in (record.music, record.sfx):
            assert stem.source_backend == "sam_audio"
            assert "/separation/" in stem.canonical_stem_path
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    downstream = root.parent / "diarization"
    downstream.mkdir()
    sentinel = downstream / "keep"
    sentinel.write_bytes(b"existing downstream stage")
    resolved.resolve_audio_stems(
        audio_production_root=Path(setup.audio_production_root),
        shadow_run_id=setup.shadow_run_id,
        overwrite=True,
    )
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    assert sentinel.read_bytes() == b"existing downstream stage"
    with pytest.raises(FileExistsError):
        resolved.resolve_audio_stems(
            audio_production_root=Path(setup.audio_production_root),
            shadow_run_id=setup.shadow_run_id,
        )


def test_failed_auk_has_no_fallback(setup, tmp_path, ffmpeg):
    _, (inventory, records, summary) = _resolve(setup, tmp_path, ffmpeg, fail="a")
    assert (summary.ready_count, summary.failed_count) == (2, 1)
    failure = records[1]
    assert failure.status == "failed"
    assert failure.stems == []
    skips = separation_skips(inventory=inventory, records=records, route="resolved")
    assert [s.clip_uid for s in skips] == ["a"]
    assert skips[0].source_stage == "resolved_stems_v1"
    assert skips[0].reason_code == "resolved_stems_failed"


@pytest.mark.parametrize("route,both", [("voice_first", False), ("music_first", True)])
def test_resolution_rejects_nonfixed_sam_route(setup, tmp_path, ffmpeg, route, both):
    sam_root = sam_baseline(setup, tmp_path, route=route, run_both_routes=both)
    run(setup, ffmpeg)
    # The old SAM artifacts remain readable, but cannot back a new resolved run.
    assert resolved.load_stem_shadow(sam_root)[0].route == route
    with pytest.raises(ValueError, match="requires SAM music_first only"):
        resolved.resolve_audio_stems(
            audio_production_root=Path(setup.audio_production_root),
            shadow_run_id=setup.shadow_run_id,
        )
    assert not (sam_root.parent / resolved.RESOLVED_STAGE).exists()


@pytest.mark.parametrize("sam_speech_change", ["removed", "changed"])
def test_sam_speech_is_never_requested_as_a_resolved_candidate(
    setup, tmp_path, ffmpeg, monkeypatch, sam_speech_change
):
    from r2v_data_v2.h3.sam_audio_stem_shadow import SAMAudioStemRecord

    sam_root = sam_baseline(setup, tmp_path, route="music_first")
    run(setup, ffmpeg, fail="a")
    _, sam_records, _ = resolved.load_stem_shadow(sam_root)
    for record in sam_records:
        # Only synthetic QA speech artifacts; music/SFX and signed records stay frozen.
        path = Path(record.stem("speech").canonical_stem_path)
        if sam_speech_change == "removed":
            path.unlink()
        else:
            path.write_bytes(b"changed QA-only SAM speech")
    original = SAMAudioStemRecord.stem
    requested = []

    def nonspeech_only(self, kind):
        requested.append(kind)
        assert kind != "speech", "SAM speech is QA-only, never a downstream candidate"
        return original(self, kind)

    monkeypatch.setattr(SAMAudioStemRecord, "stem", nonspeech_only)
    summary = resolved.resolve_audio_stems(
        audio_production_root=Path(setup.audio_production_root),
        shadow_run_id=setup.shadow_run_id,
    )
    # Atomic publication resolves twice to catch upstream changes.
    assert requested == ["music", "sfx", "music", "sfx"] * 2
    assert (summary.ready_count, summary.failed_count) == (2, 1)


def test_named_run_never_falls_back(setup, tmp_path):
    sam_baseline(setup, tmp_path, route="music_first")
    root = resolved.downstream_stem_root(
        Path(setup.audio_production_root), setup.shadow_run_id
    )
    with pytest.raises(FileNotFoundError):
        resolved.load_stem_source(root)
    assert resolved.downstream_stem_route(setup.shadow_run_id) == "resolved"
    assert resolved.downstream_stem_route(setup.shadow_run_id, "music_first") == "resolved"
    with pytest.raises(ValueError):
        resolved.downstream_stem_route(setup.shadow_run_id, "voice_first")
    for route in ("music_first", "voice_first"):
        assert resolved.downstream_stem_route(None, route) == route


def test_all_auk_failures_publish_only_skips(setup, tmp_path, ffmpeg, monkeypatch):
    monkeypatch.setattr(
        auk_tests.FakeBackend,
        "generate",
        lambda *a, **kw: {
            "status": "failed",
            "reason": "synthetic unavailable speech",
        },
    )
    root, (_, records, summary) = _resolve(setup, tmp_path, ffmpeg)
    assert summary.failed_count == 3 and summary.ready_count == 0
    assert all(not r.stems for r in records)
    production = Path(setup.audio_production_root)
    canonical = [
        CanonicalAudioClip.model_validate_json(line)
        for line in Path(setup.source_canonical_audio_manifest_path)
        .read_text()
        .splitlines()
    ]
    source = production / "diarization"
    source.mkdir()
    inventory = _production_diarization_inventory_for_records(production, canonical)
    (source / "inventory.json").write_text(inventory.model_dump_json())
    backend = _Diarization()
    provenance = run_stem_diarization_shadow(
        stem_root=root,
        production_diarization_root=source,
        backend=backend,
        route="resolved",
        output_root=root.parent / "diarization",
        allow_unverified=True,
    )
    assert not backend.paths and provenance.target_count == 0
    assert [s.clip_uid for s in provenance.skipped_clips] == ["c", "a", "b"]
    _, asr = run_stem_qwen3_asr_shadow(
        stem_diarization_root=root.parent / "diarization",
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=root.parent / "asr",
        route="resolved",
        allow_unverified=True,
    )
    assert asr.segment_count == 0 and asr.clip_uids == []
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow

    result = run_mimo25_stem_reconcile_shadow(
        jobs=[],
        stem_records=records,
        backend=object(),
        output_root=root.parent / "mimo_reconcile",
        source_clip_uids=["c", "a", "b"],
        skipped_clips=asr.skipped_clips,
        route="resolved",
        allow_unverified=True,
    )
    assert result.skipped_clip_count == 3
    assert result.model_call_count == result.processed_clip_count == 0


def test_upstream_changes_fail_closed_and_preserve_published_resolution(
    setup, tmp_path, ffmpeg
):
    root, (_, records, _) = _resolve(setup, tmp_path, ffmpeg)
    before = {p: p.read_bytes() for p in root.iterdir()}
    music = Path(records[0].music.canonical_stem_path)
    music.write_bytes(b"changed synthetic upstream")
    with pytest.raises(ValueError, match="lineage changed"):
        resolved.load_resolved_stems(root)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_resolved_diarization_asr_paths_and_failure_subset(
    setup, tmp_path, ffmpeg, monkeypatch
):
    root, (_, records, _) = _resolve(setup, tmp_path, ffmpeg, fail="a")
    production = Path(setup.audio_production_root)
    canonical = [
        CanonicalAudioClip.model_validate_json(line)
        for line in Path(setup.source_canonical_audio_manifest_path)
        .read_text()
        .splitlines()
    ]
    source = production / "diarization"
    source.mkdir()
    inventory = _production_diarization_inventory_for_records(production, canonical)
    (source / "inventory.json").write_text(inventory.model_dump_json())
    backend = _Diarization()
    provenance = run_stem_diarization_shadow(
        stem_root=root,
        production_diarization_root=source,
        backend=backend,
        route="resolved",
        output_root=root.parent / "diarization",
        allow_unverified=True,
    )
    expected = [
        Path(r.speech.canonical_stem_path) for r in records if r.status == "ready"
    ]
    assert backend.paths == expected
    assert provenance.diarization_source_kind == "resolved_speech_stem"
    assert provenance.usable_clip_uids == ["c", "b"]
    observed = []

    def loader(path, start, end):
        observed.append(path)
        return np.zeros(round((end - start) * 16000), dtype=np.float32), 16000

    summary, asr = run_stem_qwen3_asr_shadow(
        stem_diarization_root=root.parent / "diarization",
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=root.parent / "asr",
        segment_audio_loader=loader,
        route="resolved",
        allow_unverified=True,
    )
    assert observed == expected
    assert summary.transcribed_count == 2
    assert asr.asr_source_kind == "resolved_speech_stem_segments"
    assert [s.clip_uid for s in asr.skipped_clips] == ["a"]
    validate_stem_asr_lineage(root.parent / "asr")
    from tests.test_h3_sam_audio_stem_shadow import _mimo_inventory_for_clip_order
    from tools import run_h3_mimo25_stem_reconcile_shadow as reconcile_cli
    from tools import run_h3_stem_diarization_shadow as diarization_cli
    from tools import run_h3_stem_qwen3_asr_shadow as asr_cli

    common = [
        "--audio-production-root",
        str(production),
        "--shadow-run-id",
        setup.shadow_run_id,
        "--case-manifest",
        setup.source_case_manifest_path,
        "--dry-run",
        "--allow-unverified",
    ]
    dry = diarization_cli.main(common)
    assert dry["target_clip_uids"] == ["c", "b"]
    assert dry["diarization_source_kind"] == "resolved_speech_stem"
    assert asr_cli.main([*common, "--visual-production-root", "/visual/source"])[
        "clip_uids"
    ] == ["c", "b"]
    base = _mimo_inventory_for_clip_order(tmp_path / "mimo-base", ["c", "a", "b"])
    monkeypatch.setattr(reconcile_cli, "build_mimo25_inventory", lambda **kw: base)
    dry = reconcile_cli.main(
        [
            *common,
            "--visual-production-root",
            "/visual/source",
            "--visual-runs-root",
            "/visual/runs",
        ]
    )
    assert dry["clip_uids"] == ["c", "b"] and not dry["model_called"]


def test_mimo_uses_resolved_snippets_and_sam_nonspeech(setup, tmp_path, ffmpeg):
    from r2v_data_v2.h3.mimo25_stem_shadow import _validated_auxiliary_stems
    from tests.h3_mimo_two_turn_helpers import SnippetAudioBackend
    from tests.test_h3_mimo25_av_shadow import _backend
    from tests.test_h3_mimo25_speaker_snippets import _fixture

    _, (_, records, _) = _resolve(setup, tmp_path, ffmpeg)
    job, _, raws = _fixture(tmp_path)
    backend, calls = _backend(tmp_path, [(raw, 8) for raw in raws], transport="sglang")
    media = SnippetAudioBackend()
    backend._audio_media_backend = media
    stems = _validated_auxiliary_stems(records[0])
    backend._request(
        job,
        allowed_reference_labels={"<Subject 1>", "<Picture 1>"},
        auxiliary_audio_paths=stems,
    )
    assert len(calls.requests) == 4
    assert len(media.extractions) == 3
    assert all(e["source_audio_path"] == stems["speech"] for e in media.extractions)
    assert all(
        "/auk_speech_v1/" in str(e["source_audio_path"]) for e in media.extractions
    )
    for turn in (0, 1, 3):
        videos = [
            p
            for p in calls.requests[turn]["messages"][-1]["content"]
            if p["type"] == "video_url"
        ]
        assert len(videos) == 1
        assert videos[0]["video_url"]["url"] == backend.config.media_resolver.resolve(
            Path(job.target_video_path)
        )
    final = calls.requests[3]["messages"][-1]["content"]
    assert [p["audio_url"]["url"] for p in final if p["type"] == "audio_url"] == [
        backend.config.media_resolver.resolve(stems[k]) for k in ("music", "sfx")
    ]


@pytest.mark.parametrize(
    "offscreen,mixed", [(False, False), (True, False), (False, True)]
)
def test_reuse_resolved_pcm_preserves_ownership(tmp_path, offscreen, mixed):
    import soundfile as sf

    from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
    from tests.test_h3_audio_reuse import _fixture, _pcm

    args = _fixture(tmp_path, offscreen=offscreen, mixed=mixed, target_subtype="PCM_24")
    old = args["stem_record"]
    fields = {}
    for stem in old.stems:
        path = Path(stem.canonical_stem_path)
        if stem.stem_type == "speech":
            path = path.with_name("auk-speech.wav")
            sf.write(
                path, np.full((64000, 2), 4321, dtype=np.int16), 32000, subtype="PCM_16"
            )
        fields[stem.stem_type] = resolved.ResolvedStem(
            kind=stem.stem_type,
            source_backend="auk" if stem.stem_type == "speech" else "sam_audio",
            canonical_stem_path=str(path),
            canonical_stem_sha256=sha256_file(path),
            canonical_frame_count=64000,
            source_audio_path=stem.source_audio_path,
            source_audio_sha256=stem.source_audio_sha256,
            source_end_sample=64000,
        ).model_dump(mode="json")
    args["stem_record"] = resolved._signed(
        resolved.ResolvedStemRecord,
        {
            "clip_uid": old.clip_uid,
            "inventory_fingerprint": "a" * 64,
            "source_sam_record_fingerprint": old.record_fingerprint,
            "source_auk_record_fingerprint": "b" * 64,
            "status": "ready",
            **fields,
        },
        "record_fingerprint",
    )
    target = Path(args["job"].target_full_audio_path)
    before = target.read_bytes()
    result = build_audio_reuse_assets(**args)
    assert np.array_equal(
        _pcm(result.music.output_path),
        _pcm(old.stem("music").canonical_stem_path, container="WAV"),
    )
    assert target.read_bytes() == before
    assert sf.info(target).subtype == "PCM_24"
    for speaker in result.speakers:
        assert speaker.source_stem_path == fields["speech"]["canonical_stem_path"]
        pcm = _pcm(speaker.output_path)
        nonzero = pcm[pcm != 0]
        assert len(nonzero) and np.all(nonzero == 4321)
    if mixed:
        assert result.exclusions
    if offscreen:
        assert all(s.entity_id is None for s in result.speakers)


def test_resolved_run_finalizer_and_qa(setup, tmp_path, ffmpeg, monkeypatch):
    import json

    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3.audio_reuse import AudioReuseManifest
    from r2v_data_v2.h3.audio_reuse_finalizer import finalize_audio_reuse_shadow
    from r2v_data_v2.h3.mimo25_stem_shadow import (
        build_stem_reconcile_jobs,
        run_mimo25_stem_reconcile_shadow,
        usable_stem_reconcile_inventory,
    )
    from tests import test_h3_audio_shadow_qa as qa
    from tests import test_h3_sam_audio_stem_shadow as fixtures
    from tests.test_h3_audio_reuse_finalizer import _fixture

    work = tmp_path / "integration"
    work.mkdir()
    args, shadow = _fixture(work, monkeypatch)
    production = args["audio_production_root"]
    old, _, _ = sam.load_stem_shadow(shadow / "separation")
    inventory = sam.build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=Path(old.source_canonical_audio_manifest_path),
        model_configuration=old.model_configuration,
        route="music_first",
        case_manifest_path=args["case_manifest"],
    )
    sam.run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=shadow / "separation",
        backend=fixtures._SAM(inventory.model_configuration),
        canonicalizer=fixtures._Canonicalizer(),
        raw_probe_backend=fixtures._Media(),
        overwrite=True,
    )
    auk_inventory = auk.build_auk_inventory(
        audio_production_root=production,
        shadow_run_id=args["shadow_run_id"],
        case_manifest_path=args["case_manifest"],
        configuration=setup.model_configuration,
    )
    run(auk_inventory, ffmpeg)
    resolved.resolve_audio_stems(
        audio_production_root=production,
        shadow_run_id=args["shadow_run_id"],
    )
    root = shadow / resolved.RESOLVED_STAGE
    provenance = sam.run_stem_diarization_shadow(
        stem_root=root,
        production_diarization_root=production / "production-diarization",
        backend=fixtures._Diarization(),
        route="resolved",
        output_root=shadow / "diarization",
        allow_unverified=True,
        overwrite=True,
    )
    sam.run_stem_qwen3_asr_shadow(
        stem_diarization_root=shadow / "diarization",
        source_visual_production_root="/visual/source",
        backend=fixtures._Qwen(),
        output_root=shadow / "asr",
        route="resolved",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
        allow_unverified=True,
        overwrite=True,
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(
            qa.qa.build_mimo25_inventory(), provenance
        ),
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr",
        route="resolved",
    )
    _, stems, _ = resolved.load_stem_source(root)
    reconcile = run_mimo25_stem_reconcile_shadow(
        jobs=jobs,
        stem_records=stems,
        backend=qa._Reconcile(work),
        output_root=shadow / qa.MIMO25_STEM_RECONCILE_STAGE,
        source_clip_uids=inventory.clip_uids,
        route="resolved",
        allow_unverified=True,
        overwrite=True,
    )
    assert reconcile.schema_version.endswith(".17")
    assert reconcile.model_call_count == 12
    before = {p: sha256_file(p) for p in production.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fixed music_first SAM lineage"):
        finalize_audio_reuse_shadow(**{**args, "sam_route": "voice_first"})
    summary = finalize_audio_reuse_shadow(**{**args, "sam_route": "music_first"})
    assert summary.product_ready_count == 6
    assert summary.model_call_count == 0
    assert summary.schema_version.endswith(".2")
    manifest = AudioReuseManifest.model_validate_json(
        (shadow / "audio_reuse_assets_v1/clip-z/manifest.json").read_text()
    )
    assert all("/auk_speech_v1/" in s.source_stem_path for s in manifest.speakers)
    assert "/separation/" in manifest.music.source_stem_path
    qa.qa.build_audio_shadow_qa(
        **{k: v for k, v in args.items() if k != "allow_unverified"}
    )
    data = json.loads((shadow / "qa/data.json").read_text())
    assert data["clips"][0]["separation"]["speech"]["source_backend"] == "auk"
    assert data["clips"][0]["separation"]["music"]["source_backend"] == "sam_audio"
    assert all(sha256_file(p) == digest for p, digest in before.items())
