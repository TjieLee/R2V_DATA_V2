import json
from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3 import t2va_production as production
from tests import test_h3_t2va_shadow as shared
from tests.test_h3_ta2va_shadow import ProfileClient

setup = shared.setup
ffmpeg = shared.ffmpeg
finalized = shared.finalized


def invoke(tmp_path, finalized, backend, profiles):
    from r2v_data_v2.h3.t2va_full_downstream import run_downstream

    root = tmp_path / "downstream"
    index = production.build_source_index(tmp_path / "shots_f03_motion.jsonl", root)
    return run_downstream(
        root,
        index,
        0,
        finalized[0],
        tmp_path,
        tmp_path,
        backend,
        profiles,
        allow_unverified=True,
        run_id=finalized[1],
    )


def clients(tmp_path, finalized):
    inventory = shared.build(finalized, tmp_path)
    # These deterministic responses can also serve the initially skipped clip.
    drafts = [
        shared.draft_for(j).model_dump_json()
        for j in inventory.jobs
        if not j.upstream_failure
    ]
    client = shared.Client(drafts * 4)
    profile = ProfileClient()
    return (
        shared.T2VAMimoBackend(shared.config(tmp_path), client=client),
        _profiles(tmp_path, profile),
        client,
        profile,
    )


def _profiles(tmp_path, client):
    from r2v_data_v2.h3.ta2va_shadow import TA2VAProfileBackend

    return TA2VAProfileBackend(shared.config(tmp_path), client=client)


def refresh_upstream(tmp_path, finalized, ffmpeg, *, heal):
    """Republish real frozen lineage using fake backends only."""
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import diarization_binding as diar
    from r2v_data_v2.h3 import resolved_audio_stems as resolved
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from tests.test_h3_auk_speech_shadow import FakeBackend

    audio, run_id = finalized
    shadow = sam.stem_shadow_root(audio, run_id)
    canonical = audio / "audio/canonical_clips.jsonl"
    canonical.write_text(
        "".join(
            json.dumps(json.loads(line), sort_keys=True) + "\n"
            for line in canonical.read_text().splitlines()
        )
    )
    diar_path = audio / "diarization/inventory.json"
    d = diar.DiarizationInventory.model_validate_json(diar_path.read_text())
    values = d.model_dump(mode="json")
    values["source_canonical_audio_manifest_sha256"] = production._sha(canonical)
    values["inventory_fingerprint"] = diar._inventory_fingerprint(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode=d.mode,
        targets=d.targets,
        source_inventory_kind=d.source_inventory_kind,
        source_canonical_audio_manifest_sha256=values[
            "source_canonical_audio_manifest_sha256"
        ],
        source_shot_manifest_sha256=d.source_shot_manifest_sha256,
    )
    shared.t2va.write_json(diar_path, diar.DiarizationInventory.model_validate(values))
    old, old_records, _ = auk.load_auk_shadow(shadow / "auk_speech_v1")
    preserved = {
        Path(a.path): Path(a.path).read_bytes()
        for r in old_records
        if r.status == "ready"
        for a in (r.raw, r.canonical)
    }
    inventory = auk.build_auk_inventory(
        audio_production_root=audio,
        shadow_run_id=run_id,
        case_manifest_path=Path(old.source_case_manifest_path),
        configuration=old.model_configuration,
    )
    auk.run_auk_speech_shadow(
        inventory=inventory,
        backend=FakeBackend(
            inventory.model_configuration, fail=None if heal else inventory.clip_uids[0]
        ),
        ffmpeg=ffmpeg,
        overwrite=True,
    )
    auk_root = shadow / "auk_speech_v1"
    _, new_records, _ = auk.load_auk_shadow(auk_root)
    for path, contents in preserved.items():
        path.write_bytes(contents)
    records = []
    for old_record, new_record in zip(old_records, new_records, strict=True):
        record = old_record if old_record.status == "ready" else new_record
        values = record.model_dump(mode="json", exclude={"record_fingerprint"})
        values["inventory_fingerprint"] = inventory.inventory_fingerprint
        records.append(auk._signed(auk.AukRecord, values, "record_fingerprint"))
    sam._write_jsonl(auk_root / "records.jsonl", records)
    sam._write_json(auk_root / "summary.json", auk._summary(inventory, records))
    sam_root = shadow / "separation"
    inv, records, summary = sam.load_stem_shadow(sam_root)
    values = inv.model_dump(mode="json", exclude={"inventory_fingerprint"})
    values["source_canonical_audio_manifest_sha256"] = production._sha(canonical)
    signed = sam.SAMAudioStemInventory(
        **values, inventory_fingerprint=sam._sha256_text(sam._compact_json(values))
    )
    sam._write_json(sam_root / "inventory.json", signed)
    updated = []
    for record in records:
        values = record.model_dump(mode="json", exclude={"record_fingerprint"})
        values["inventory_fingerprint"] = signed.inventory_fingerprint
        updated.append(
            sam.SAMAudioStemRecord(
                **values, record_fingerprint=sam._sha256_text(sam._compact_json(values))
            )
        )
    sam._write_jsonl(sam_root / "records.jsonl", updated)
    sam._write_json(
        sam_root / "summary.json",
        summary.model_copy(
            update={"inventory_fingerprint": signed.inventory_fingerprint}
        ),
    )
    resolved.resolve_audio_stems(
        audio_production_root=audio, shadow_run_id=run_id, overwrite=True
    )
    sam.run_stem_diarization_shadow(
        stem_root=shadow / "resolved_stems_v1",
        production_diarization_root=audio / "diarization",
        backend=shared._Diarization(),
        route="resolved",
        output_root=shadow / "diarization",
        allow_unverified=True,
        overwrite=True,
    )
    sam.run_stem_qwen3_asr_shadow(
        stem_diarization_root=shadow / "diarization",
        source_visual_production_root=None,
        backend=shared._Qwen(),
        output_root=shadow / "asr",
        route="resolved",
        allow_unverified=True,
        overwrite=True,
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
    )


@pytest.mark.parametrize("heal", [False, True])
def test_upstream_republication_preserves_ready_and_reopens_skipped(
    tmp_path, finalized, ffmpeg, heal
):
    backend, profiles, client, profile = clients(tmp_path, finalized)
    first = invoke(tmp_path, finalized, backend, profiles)
    assert sum(s["t2va_status"] == "ready" for s in first.values()) == 2
    root = tmp_path / "downstream"
    shard = root / "shards" / production.shard_name(0)
    before = {
        p: p.read_bytes() for p in (shard / "artifacts").rglob("*") if p.is_file()
    }
    previous = list(production.complete_rows(shard / "exports/t2va.jsonl"))
    calls = len(client.calls), len(profile.calls)
    refresh_upstream(tmp_path, finalized, ffmpeg, heal=heal)
    # Generate the exact new clip's response, rather than borrowing its neighbors' facts.
    inventory = shared.build(finalized, tmp_path)
    if heal:
        client.responses = iter([shared.draft_for(inventory.jobs[0]).model_dump_json()])
    second = invoke(tmp_path, finalized, backend, profiles)
    assert sum(s["t2va_status"] == "ready" for s in second.values()) == 2 + int(heal)
    assert len(client.calls) == calls[0] + 2 * int(heal)
    assert all(path.read_bytes() == data for path, data in before.items())
    output = list(production.complete_rows(shard / "exports/t2va.jsonl"))
    assert len(output) == 2 + int(heal)
    assert all(row in output for row in previous)
    snapshot = production.build_snapshot(root, "resumed")
    assert len(list(production.complete_rows(snapshot / "t2va.jsonl"))) == len(output)


def test_ready_t2va_failed_ta_retries_only_ta(tmp_path, finalized, ffmpeg):
    backend, profiles, client, profile = clients(tmp_path, finalized)
    profile.fail = True
    first = invoke(tmp_path, finalized, backend, profiles)
    assert any(s["ta2va_status"] == "failed" for s in first.values())
    calls = len(client.calls)
    refresh_upstream(tmp_path, finalized, ffmpeg, heal=False)
    profile.fail = False
    invoke(tmp_path, finalized, backend, profiles)
    assert len(client.calls) == calls


@pytest.mark.parametrize("damage", ["video", "receipt", "sidecar"])
def test_ready_evidence_changes_fail_closed(tmp_path, finalized, damage):
    backend, profiles, client, profile = clients(tmp_path, finalized)
    states = invoke(tmp_path, finalized, backend, profiles)
    uid = next(uid for uid, state in states.items() if state["t2va_status"] == "ready")
    shard = tmp_path / "downstream/shards" / production.shard_name(0)
    if damage == "video":
        job = json.loads((shard / "artifacts" / uid / "t2va/job.json").read_text())
        Path(job["target_video_path"]).write_bytes(b"different video")
    elif damage == "receipt":
        import shutil

        shutil.rmtree(shard / "artifacts" / uid / "t2va")
    else:
        (shard / "downstream_dependencies" / f"{uid}.json").unlink()
    calls = len(client.calls), len(profile.calls)
    with pytest.raises(ValueError):
        invoke(tmp_path, finalized, backend, profiles)
    assert (len(client.calls), len(profile.calls)) == calls


@pytest.mark.parametrize("boundary", ["sidecar", "exports"])
def test_reopen_crash_is_idempotent(tmp_path, finalized, ffmpeg, monkeypatch, boundary):
    from r2v_data_v2.h3 import t2va_full_downstream as downstream

    backend, profiles, client, _profile = clients(tmp_path, finalized)
    invoke(tmp_path, finalized, backend, profiles)
    calls = len(client.calls)
    root = tmp_path / "downstream"
    shard = root / "shards" / production.shard_name(0)
    before = list(production.complete_rows(shard / "exports/t2va.jsonl"))
    refresh_upstream(tmp_path, finalized, ffmpeg, heal=True)
    inventory = shared.build(finalized, tmp_path)
    client.responses = iter([shared.draft_for(inventory.jobs[0]).model_dump_json()])
    if boundary == "sidecar":
        original = production.append_row

        def interrupted(path, value):
            if "reopened_from" in value:
                raise KeyboardInterrupt()
            return original(path, value)

        monkeypatch.setattr(production, "append_row", interrupted)
    else:
        original = downstream.os.replace

        def interrupted(source, target):
            result = original(source, target)
            if Path(target).name == "t2va.jsonl.partial":
                raise KeyboardInterrupt()
            return result

        monkeypatch.setattr(downstream.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(tmp_path, finalized, backend, profiles)
    monkeypatch.undo()
    invoke(tmp_path, finalized, backend, profiles)
    assert len(client.calls) == calls + 2
    output = list(production.complete_rows(shard / "exports/t2va.jsonl"))
    assert len(output) == 3
    assert all(row in output for row in before)


def test_receipt_without_journal_does_not_repeat_calls(tmp_path, finalized):
    backend, profiles, client, profile = clients(tmp_path, finalized)
    invoke(tmp_path, finalized, backend, profiles)
    shard = tmp_path / "downstream/shards" / production.shard_name(0)
    (shard / "state.jsonl.partial").write_bytes(b'{"truncated":')
    calls = len(client.calls), len(profile.calls)
    invoke(tmp_path, finalized, backend, profiles)
    assert (len(client.calls), len(profile.calls)) == calls


def test_conflicting_exports_fail_before_any_calls(tmp_path, finalized):
    backend, profiles, client, profile = clients(tmp_path, finalized)
    invoke(tmp_path, finalized, backend, profiles)
    shard = tmp_path / "downstream/shards" / production.shard_name(0)
    original = next(production.complete_rows(shard / "exports/t2va.jsonl"))
    production.append_row(
        shard / "exports/t2va.jsonl.partial", {**original, "caption": "conflicting"}
    )
    calls = len(client.calls), len(profile.calls)
    with pytest.raises(ValueError, match="conflicting"):
        invoke(tmp_path, finalized, backend, profiles)
    assert (len(client.calls), len(profile.calls)) == calls


def test_empty_diarization_backend_change_fails_closed(tmp_path, finalized):
    # Even a clip with no speech depends on the DiariZen configuration.
    backend, profiles, client, profile = clients(tmp_path, finalized)
    invoke(tmp_path, finalized, backend, profiles)
    shadow = shared.t2va.stem_shadow_root(*finalized)
    path = shadow / "diarization/summary.json"
    value = json.loads(path.read_text())
    value["backend_provenance"]["model_fingerprint"] = "f" * 64
    path.write_text(json.dumps(value))
    calls = len(client.calls), len(profile.calls)
    with pytest.raises(ValueError):
        invoke(tmp_path, finalized, backend, profiles)
    assert (len(client.calls), len(profile.calls)) == calls
