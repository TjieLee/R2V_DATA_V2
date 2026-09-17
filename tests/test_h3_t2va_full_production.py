import json

from r2v_data_v2.h3 import t2va_full_production as full
from r2v_data_v2.h3 import t2va_production as production
from tests.test_h3_t2va_shadow import Audio, shot_manifest


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
