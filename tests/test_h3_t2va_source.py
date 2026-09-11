import json
from pathlib import Path

import pytest

from r2v_data_v2.h3 import t2va_shadow as t2va
from r2v_data_v2.h3.diarization_binding import DiarizationInventory
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.t2va_source import prepare_t2va_audio, select_t2va_shots
from r2v_data_v2.naming import parse_clip_identity
from tests import test_h3_t2va_shadow as fixtures

setup = fixtures.setup
ffmpeg = fixtures.ffmpeg
finalized = fixtures.finalized


def test_shot_population_without_audio_or_visual(tmp_path):
    manifest = fixtures.shot_manifest(tmp_path)
    selection = select_t2va_shots(manifest, sample_size=2, sample_seed=17)
    assert selection == select_t2va_shots(manifest, sample_size=2, sample_seed=17)
    assert selection.valid_row_count == 3
    assert [s.clip_uid for s in selection.shots] == t2va.select_clip_uids(
        fixtures.shot_ids(tmp_path), sample_size=2, sample_seed=17
    )
    for shot in selection.shots:
        assert shot.clip_uid == parse_clip_identity(shot.video_path).clip_uid
    production = tmp_path / "absent-audio"
    inventory = t2va.build_t2va_inventory(
        shot_manifest=manifest,
        audio_production_root=production,
        audio_shadow_run_id="audio",
        t2va_run_id="t2va",
        backend=fixtures.config(tmp_path).provenance(),
        sample_size=2,
        sample_seed=17,
    )
    assert inventory.clip_uids == [s.clip_uid for s in selection.shots]
    assert all(
        j.upstream_failure == "audio_preprocessing_required" for j in inventory.jobs
    )
    assert [j.target_video_path for j in inventory.jobs] == [
        s.video_path for s in selection.shots
    ]
    assert not production.exists()
    assert inventory.source_hashes[str(manifest)] == t2va.sha256_file(manifest)
    backend = fixtures.T2VAMimoBackend(
        fixtures.config(tmp_path), client=fixtures.Client([])
    )
    request = backend.build_request(inventory.jobs[0])
    media = [
        part
        for message in request["messages"]
        for part in (message["content"] if isinstance(message["content"], list) else [])
        if part.get("type") == "video_url"
    ]
    assert len(media) == 1
    assert media[0]["video_url"]["url"] == backend.config.media_resolver.resolve(
        Path(selection.shots[0].video_path)
    )


def test_new_no_reference_clip_survives_old_audio_population(finalized, tmp_path):
    manifest = tmp_path / "shots_f03_motion.jsonl"
    extra = tmp_path / "movie_4.mp4"
    extra.write_bytes(b"clip with no Pictures or Subjects")
    row = {
        "video_path": str(extra),
        "source_video_path": str(tmp_path / "movie.mp4"),
        "source_video_id": "movie",
        "shot_index": 4,
        "duration": 1,
    }
    with manifest.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    uid = parse_clip_identity(extra).clip_uid
    case = tmp_path / "extra-case.json"
    case.write_text(json.dumps({"clip_uids": [uid]}))
    inventory = fixtures.build(finalized, tmp_path, case_manifest=case)
    assert inventory.clip_uids == [uid]
    assert inventory.jobs[0].target_video_path == str(extra)
    assert inventory.jobs[0].upstream_failure == "audio_preprocessing_required"
    assert inventory.shot_selection.shots[0].source_index == 3
    assert uid not in (finalized[0] / "audio/canonical_clips.jsonl").read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_video_path", "/other/movie_1.mp4"),
        ("target_video_sha256", "a" * 64),
    ],
)
def test_cache_reuse_requires_exact_selected_identity(
    finalized, tmp_path, field, value
):
    path = finalized[0] / "audio/canonical_clips.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="selected JEA shot differs"):
        fixtures.build(finalized, tmp_path)


def test_selection_ignores_reference_fields_and_excludes_invalid_rows(tmp_path):
    manifest = fixtures.shot_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = select_t2va_shots(manifest)
    for row in rows:
        row.update(entities=[], references=[], reference_eligible=False)
    rows.insert(1, {"video_path": "missing.mp4"})
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = select_t2va_shots(manifest)
    assert [s.clip_uid for s in selected.shots] == [s.clip_uid for s in result.shots]
    assert len(result.excluded_rows) == 1
    assert [s.source_index for s in result.shots] == [0, 2, 3]


def test_bootstrap_reuses_audio_backend_without_visual(tmp_path):
    manifest = fixtures.shot_manifest(tmp_path)
    selection = select_t2va_shots(manifest)
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    root = prepare_t2va_audio(
        selection,
        output_root=tmp_path / "new-audio",
        audio_backend=fixtures.Audio(),
    )
    records = t2va.read_rows(root / "audio/canonical_clips.jsonl", CanonicalAudioClip)
    assert [c.clip_uid for c in records] == [s.clip_uid for s in selection.shots]
    assert all(
        c.subject_reference_count == 0 and c.target_audio_binding_path is None
        for c in records
    )
    assert all(Path(c.target_full_audio_path).is_file() for c in records)
    inventory = DiarizationInventory.model_validate_json(
        (root / "diarization/inventory.json").read_text()
    )
    assert inventory.source_inventory_kind == "jea_shot_manifest"
    assert inventory.source_visual_production_root is None
    assert all(
        not t.visual_references and t.target_audio_binding_path is None
        for t in inventory.targets
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        prepare_t2va_audio(selection, output_root=root, audio_backend=fixtures.Audio())


def test_audio_bootstrap_failure_is_atomic(tmp_path):
    selection = select_t2va_shots(fixtures.shot_manifest(tmp_path))

    class Failure:
        def materialize_full_audio(self, **kwargs):
            raise ValueError("synthetic extraction failure")

    root = tmp_path / "new-audio"
    with pytest.raises(ValueError, match="synthetic"):
        prepare_t2va_audio(selection, output_root=root, audio_backend=Failure())
    assert not root.exists()
    assert not list(tmp_path.glob(".new-audio.tmp-*"))


def test_prepare_cli_dry_run_requires_only_shots(tmp_path, monkeypatch):
    from tools import prepare_h3_t2va_audio as cli

    manifest = fixtures.shot_manifest(tmp_path)
    monkeypatch.setattr(
        cli,
        "FFmpegAudioMediaBackend",
        lambda **kw: pytest.fail("dry run extracted audio"),
    )
    result = cli.main(
        [
            "--shot-manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "audio"),
            "--sample-size",
            "1",
            "--sample-seed",
            "7",
            "--dry-run",
        ]
    )
    assert len(result["clip_uids"]) == 1 and result["model_call_count"] == 0
    assert not (tmp_path / "audio").exists()
