import json

from r2v_data_v2.h3 import t2va_full_production as full
from r2v_data_v2.h3 import t2va_production as production
from tests.test_h3_t2va_shadow import Audio, shot_manifest


def test_sam_full_inventory_publication(tmp_path):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from tests.test_h3_sam_audio_stem_shadow import (
        _SAM,
        _canonical_fixture,
        _Canonicalizer,
        _configuration,
        _Media,
    )

    manifest, _, _ = _canonical_fixture(tmp_path)
    config = _configuration(tmp_path)
    inventory = sam.build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest, model_configuration=config
    )

    def execute(stage_root, jobs, **kwargs):
        backend = _SAM(config)
        results = {}
        for job in jobs:
            folder = stage_root / job["job_id"]
            folder.mkdir(parents=True)
            record = sam._separate_one_route(
                inventory=inventory,
                job=sam.SAMAudioStemJob.model_validate(job["source"]),
                route="music_first",
                output_root=folder,
                backend=backend,
                canonicalizer=_Canonicalizer(),
                raw_probe_backend=_Media(),
            )
            results[job["job_id"]] = {
                "status": "ready",
                "result": {
                    "record": record.model_dump(mode="json"),
                    "output_root": str(folder),
                },
            }
        return results

    output = tmp_path / "shadow/separation"
    stems.run_sam(inventory, output, tmp_path / "work", ["0", "1"], execute=execute)
    loaded, records, summary = sam.load_stem_shadow(output)
    assert loaded == inventory
    assert summary.record_count == 1
    assert records[0].calls[0].prompt == "music soundtrack"
    assert records[0].calls[1].prompt == "human voices"
    assert all(
        __import__("pathlib").Path(s.canonical_stem_path).is_relative_to(output)
        for s in records[0].stems
    )


def test_bootstrap_from_index_and_resume(tmp_path):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    assert [s.source_index for s in selection.shots] == [0, 1, 2]
    calls = []

    class Media(Audio):
        def materialize_full_audio(self, **kwargs):
            calls.append(kwargs["clip_uid"])
            return super().materialize_full_audio(**kwargs)

    audio = full.bootstrap_audio(
        root / "shards" / production.shard_name(0), selection, Media()
    )
    assert len(calls) == 3
    rows = list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))
    assert [r["clip_uid"] for r in rows] == [s.clip_uid for s in selection.shots]
    assert all(r["subject_reference_count"] == 0 for r in rows)
    full.bootstrap_audio(root / "shards" / production.shard_name(0), selection, Media())
    assert len(calls) == 3
    inventory = json.loads((audio / "diarization/inventory.json").read_text())
    assert inventory["source_inventory_kind"] == "jea_shot_manifest"
    assert all(t["visual_references"] == [] for t in inventory["targets"])


def test_bootstrap_isolates_failures_and_retries(tmp_path):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    selection = full.shard_selection(
        root, production.build_source_index(manifest, root), 0, tmp_path, tmp_path
    )
    calls = []

    class Media(Audio):
        def materialize_full_audio(self, **kwargs):
            calls.append(kwargs["clip_uid"])
            if kwargs["clip_uid"] == selection.shots[1].clip_uid:
                raise ValueError("bad audio")
            super().materialize_full_audio(**kwargs)

    shard = root / "shards" / production.shard_name(0)
    audio = full.bootstrap_audio(shard, selection, Media())
    assert (
        len(list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))) == 2
    )
    full.bootstrap_audio(shard, selection, Media())
    assert len(calls) == 4
    full.bootstrap_audio(shard, selection, Audio())
    assert (
        len(list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))) == 3
    )
