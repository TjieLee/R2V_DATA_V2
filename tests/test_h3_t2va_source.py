import json
import os
from pathlib import Path

import pytest

from r2v_data_v2.h3 import t2va_shadow as t2va
from r2v_data_v2.h3 import t2va_shot_index as shot_index
from r2v_data_v2.h3.diarization_binding import DiarizationInventory
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.t2va_source import prepare_t2va_audio, select_t2va_shots
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter
from tests import test_h3_t2va_shadow as fixtures

setup = fixtures.setup
ffmpeg = fixtures.ffmpeg
finalized = fixtures.finalized


def test_shot_population_without_audio_or_visual(tmp_path):
    manifest = fixtures.shot_manifest(tmp_path)
    selection = select_t2va_shots(manifest, sample_size=2, sample_seed=17)
    assert selection == select_t2va_shots(manifest, sample_size=2, sample_seed=17)
    assert selection.valid_row_count is None
    assert [s.source_index for s in selection.shots] == [
        *shot_index.random_indices(3, 17)
    ][:2]
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
            "--shot-index-root",
            str(tmp_path.parent / (tmp_path.name + "-index")),
        ]
    )
    assert len(result["clip_uids"]) == 1 and result["model_call_count"] == 0
    assert not (tmp_path / "audio").exists()
    assert list((tmp_path.parent / (tmp_path.name + "-index")).glob("*/metadata.json"))


def test_random_sampling_parses_only_selected_rows_and_reuses_index(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    template_path = fixtures.shot_manifest(source)
    template = json.loads(template_path.read_text().splitlines()[0])
    count, size, seed = 5000, 7, 83
    rows = []
    for index in range(count):
        video = source / f"movie_{index + 1}.mp4"
        video.write_bytes(b"synthetic video")
        rows.append({**template, "video_path": str(video), "shot_index": index + 1})
    template_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = template_path.read_bytes()
    original = JeaVideoMotionAdapter.parse
    calls = []

    def parse(self, raw, *, source_index):
        calls.append(source_index)
        return original(self, raw, source_index=source_index)

    monkeypatch.setattr(JeaVideoMotionAdapter, "parse", parse)
    cache = tmp_path / "cache"
    options = {"sample_size": size, "sample_seed": seed, "shot_index_root": cache}
    selected = select_t2va_shots(template_path, **options)
    assert len(calls) == size
    assert calls == [s.source_index for s in selected.shots]
    assert selected.shot_manifest_sha256 == t2va.sha256_file(template_path)
    for shot in selected.shots:
        assert shot.source_row_sha256 == t2va.fingerprint(rows[shot.source_index])
        assert shot.video_sha256 == t2va.sha256_file(Path(shot.video_path))
        assert shot.clip_uid == parse_clip_identity(shot.video_path).clip_uid

    # Warm selection must not hash/scan the manifest or rebuild the index.
    original_hash = shot_index.sha256_file

    def hash_file(path):
        assert path != template_path
        return original_hash(path)

    monkeypatch.setattr(shot_index, "sha256_file", hash_file)
    original_open = Path.open

    class SeekOnly:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, *args):
            return self.handle.seek(*args)

        def readline(self):
            return self.handle.readline()

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return SeekOnly(handle) if path == template_path else handle

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        again = select_t2va_shots(template_path, **options)
    assert again == selected
    assert len(calls) == 2 * size
    assert len(list(cache.glob("*/metadata.json"))) == 1
    assert template_path.read_bytes() == before


def test_indexed_invalid_and_duplicate_rows_have_deterministic_replacements(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = fixtures.shot_manifest(source)
    valid = manifest.read_text().splitlines()
    order = list(shot_index.random_indices(7, 17))
    by_draw = [
        "{broken",
        "[]",
        valid[0],
        valid[0],
        '{"duration": 1}',
        valid[1],
        valid[2],
    ]
    lines = [""] * len(order)
    for index, raw in zip(order, by_draw, strict=True):
        lines[index] = raw
    # Blank lines do not consume source_index, matching the existing adapter loop.
    manifest.write_text("\n\n".join(lines) + "\n")
    options = {"sample_size": 3, "sample_seed": 17, "shot_index_root": tmp_path / "cache"}
    result = select_t2va_shots(manifest, **options)
    assert [s.source_index for s in result.shots] == [order[2], order[5], order[6]]
    assert [r["source_index"] for r in result.excluded_rows] == [
        order[i] for i in (0, 1, 3, 4)
    ]
    assert result == select_t2va_shots(manifest, **options)
    with pytest.raises(ValueError, match="not enough valid unique"):
        select_t2va_shots(manifest, **{**options, "sample_size": 4})


def test_index_rebuilds_for_changed_manifest_and_damaged_offsets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = fixtures.shot_manifest(source)
    cache = tmp_path / "cache"
    options = {"sample_size": 3, "sample_seed": 11, "shot_index_root": cache}
    initial = select_t2va_shots(manifest, **options)
    offset = next(cache.glob("*/offsets.bin"))
    offset.write_bytes(b"corrupt")
    assert select_t2va_shots(manifest, **options) == initial
    assert len(list(cache.glob("*/metadata.json"))) == 1
    stat = manifest.stat()
    # Same byte size and restored mtime still changes ctime, invalidating the cache.
    manifest.write_text(manifest.read_text().replace('"duration": 1', '"duration": 2'))
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result = select_t2va_shots(manifest, **options)
    assert result.shot_manifest_sha256 != initial.shot_manifest_sha256
    assert result.shot_manifest_sha256 == t2va.sha256_file(manifest)
    assert all(s.duration_seconds == 2 for s in result.shots)
    assert len(list(cache.glob("*/metadata.json"))) == 2


def test_random_index_is_unique_and_detects_mid_selection_source_change(tmp_path):
    order = list(shot_index.random_indices(100, 2))
    assert sorted(order) == list(range(100))
    assert order == list(shot_index.random_indices(100, 2))
    manifest = fixtures.shot_manifest(tmp_path)
    with (
        pytest.raises(ValueError, match="changed during selection"),
        shot_index.indexed_rows(manifest, None) as (_, read),
    ):
        assert json.loads(read(0))["shot_index"] == 1
        with manifest.open("a") as handle:
            handle.write("{}\n")
    with pytest.raises(ValueError, match="separate writable cache"):
        select_t2va_shots(
            manifest, sample_size=1, sample_seed=1, shot_index_root=tmp_path / "index"
        )


def test_legacy_selection_shape_roundtrips(tmp_path):
    from r2v_data_v2.h3.t2va_source import T2VAShotSelection

    values = select_t2va_shots(fixtures.shot_manifest(tmp_path)).model_dump(mode="json")
    values["schema_version"] = "r2v.h3.t2va_shot_selection.1"
    assert T2VAShotSelection.model_validate(values).model_dump(mode="json") == values
