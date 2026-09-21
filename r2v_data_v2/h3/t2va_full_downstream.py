"""Full-supervisor resume policy around the unchanged target-side scheduler.

The caller holds invocation.lock across upstream publication and this call.
Existing downstream roots without these dependency receipts are not adopted.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from r2v_data_v2.h3 import t2va_production as production
from r2v_data_v2.h3 import t2va_shadow as frozen
from r2v_data_v2.h3.auk_speech_shadow import load_auk_shadow
from r2v_data_v2.h3.diarization_binding import DiarizationSummary
from r2v_data_v2.h3.sam_audio_stem_shadow import load_stem_shadow


def _by_uid(items):
    result = {item.clip_uid: item for item in items}
    if len(result) != len(items):
        raise ValueError("duplicate downstream dependency identity")
    return result


class _Processor(production.FrozenProductionProcessor):
    def __init__(self, contexts, backend, profiles, index, *, allow_unverified):
        super().__init__(
            contexts,
            backend,
            profiles,
            allow_unverified=allow_unverified,
            verify_sources_per_sample=False,
        )
        self.population = index["source_sha256"]
        self.policy = {
            "version": "full-downstream-per-clip-v1",
            "backend": backend.provenance().model_dump(mode="json"),
            "profiles": profiles.provenance(),
            "allow_unverified": allow_unverified,
        }
        self.dependencies = {}
        checked, loaded, diarization = set(), {}, {}
        for uid, (job, inventory) in contexts.items():
            if inventory.inventory_fingerprint not in checked:
                frozen._check_sources(inventory)
                checked.add(inventory.inventory_fingerprint)
            if job.upstream_failure:
                continue
            audio = Path(inventory.audio_production_root)
            shadow = frozen.stem_shadow_root(audio, inventory.audio_shadow_run_id)
            if shadow not in loaded:
                diarization[shadow] = DiarizationSummary.model_validate_json(
                    (shadow / "diarization/summary.json").read_text()
                ).backend_provenance.model_dump(mode="json")
                sam, sam_records, _ = load_stem_shadow(shadow / "separation")
                auk, auk_records, _ = load_auk_shadow(shadow / "auk_speech_v1")
                loaded[shadow] = (
                    _by_uid(
                        frozen.read_rows(
                            audio / "audio/canonical_clips.jsonl",
                            frozen.CanonicalAudioClip,
                        )
                    ),
                    sam,
                    _by_uid(sam.jobs),
                    _by_uid(sam_records),
                    auk,
                    _by_uid(auk.jobs),
                    _by_uid(auk_records),
                )
            canonical, sam, sam_jobs, sam_records, auk, auk_jobs, auk_records = loaded[
                shadow
            ]
            stems, raw, asr = self._sources[shadow]
            stem = stems[uid]
            if (
                stem.source_sam_record_fingerprint
                != sam_records[uid].record_fingerprint
                or stem.source_auk_record_fingerprint
                != auk_records[uid].record_fingerprint
            ):
                raise ValueError("resolved per-clip dependency differs")
            raw_keys = sorted(key for key in raw if key[0] == uid)
            if raw_keys != sorted(key for key in asr if key[0] == uid):
                raise ValueError("per-clip raw/ASR keys differ")
            # Full production already carries immutable upstream hashes in the
            # canonical/resolved records. Actual pending T2VA/TA2VA work verifies
            # the media at the sample boundary; do not hash every clip merely to
            # construct resume dependencies.
            job_value = job.model_dump(mode="json")
            # Each removed link is replaced by its validated concrete dependency
            # below. All media, model, ownership, and timeline fields remain exact.
            del job_value["audio_evidence"]["resolved_record_fingerprint"]
            self.dependencies[uid] = {
                "job": job_value,
                "canonical": canonical[uid].model_dump(mode="json"),
                "diarization_backend": diarization[shadow],
                "resolved": stem.model_dump(
                    mode="json",
                    exclude={
                        "record_fingerprint",
                        "inventory_fingerprint",
                        "source_sam_record_fingerprint",
                        "source_auk_record_fingerprint",
                    },
                ),
                "sam": {
                    "schema": sam.schema_version,
                    "configuration": sam.model_configuration.model_dump(mode="json"),
                    "job": sam_jobs[uid].model_dump(mode="json"),
                    "record": sam_records[uid].model_dump(
                        mode="json",
                        exclude={"record_fingerprint", "inventory_fingerprint"},
                    ),
                },
                "auk": {
                    "schema": auk.schema_version,
                    "configuration": auk.model_configuration.model_dump(mode="json"),
                    "job": auk_jobs[uid].model_dump(mode="json"),
                    "record": auk_records[uid].model_dump(
                        mode="json",
                        exclude={"record_fingerprint", "inventory_fingerprint"},
                    ),
                },
                "raw": [raw[key].model_dump(mode="json") for key in raw_keys],
                "asr": [asr[key].model_dump(mode="json") for key in raw_keys],
            }

    def evidence(self, row):
        return {
            "population": self.population,
            "source": {
                k: row[k]
                for k in ("source_index", "source_row_sha256", "clip_uid", "video")
            },
            "policy": self.policy,
            "dependencies": self.dependencies.get(row["clip_uid"]),
        }

    def identity(self, row):
        return frozen.fingerprint(self.evidence(row))


def _stage_receipts(shard, uid, identity, state):
    found = False
    for stage in ("t2va", "ta2va", "t2va_failed", "ta2va_failed"):
        directory = shard / "artifacts" / uid / stage
        if directory.is_symlink():
            raise ValueError("redirected downstream stage")
        if directory.exists():
            if not directory.is_dir() or any(
                p.is_symlink() for p in directory.rglob("*")
            ):
                raise ValueError("invalid downstream stage artifacts")
            receipt = json.loads((directory / "stage.json").read_text())
            for relative in receipt["files"]:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("invalid stage artifact path")
            production._recover_stage(directory, identity)
            found = True
        elif state and state.get(f"{stage}_status") == "ready":
            raise ValueError("ready downstream stage is missing; refusing model replay")
    return found


def _exports(shard):
    result = {}
    for task in production.TASKS:
        rows = {}
        for suffix in (".jsonl", ".jsonl.partial"):
            path = shard / "exports" / f"{task}{suffix}"
            for row in production.complete_rows(path):
                if set(row) != {"video", "images", "audios", "caption"}:
                    raise ValueError("invalid downstream export")
                video = row["video"]
                if video in rows and rows[video] != row:
                    raise ValueError("conflicting downstream exports")
                rows[video] = row
        result[task] = rows
    return result


def _reopen_exports(shard, exports):
    # Publish the union before removing a final. A crash leaving both versions
    # is harmless: both this function and snapshots deduplicate identical rows.
    (shard / "COMPLETE").unlink(missing_ok=True)
    production._sync_directory(shard)
    for task, rows in exports.items():
        partial = shard / "exports" / f"{task}.jsonl.partial"
        final = partial.with_suffix("")
        if not final.exists():
            continue
        temporary = partial.with_name(f".{partial.name}.{uuid.uuid4().hex}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows.values():
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, partial)
            production._sync_directory(partial.parent)
            final.unlink()
            production._sync_directory(partial.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _preflight(shard, rows, processor):
    journal = shard / "state.jsonl.partial"
    states = production.load_states(shard)
    sources = [{k: r[k] for k in ("source_index", "clip_uid", "video")} for r in rows]
    path = shard / "sources.json"
    if path.exists() and json.loads(path.read_text()) != sources:
        raise ValueError("downstream source population changed")
    if len({r["clip_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate downstream source identity")
    exports = _exports(shard)
    updates = []
    for row in rows:
        uid = row["clip_uid"]
        if Path(uid).name != uid or uid in {".", ".."}:
            raise ValueError("unsafe downstream identity")
        evidence = processor.evidence(row)
        identity = processor.identity(row)
        empty = {**evidence, "dependencies": None}
        path = shard / "downstream_dependencies" / f"{uid}.json"
        state = states.get(uid)
        if path.is_symlink():
            raise ValueError("redirected downstream dependencies")
        previous = json.loads(path.read_text()) if path.exists() else None
        artifacts = shard / "artifacts" / uid
        exported = any(row["video"] in task for task in exports.values())
        if previous is None and (state or artifacts.exists() or exported):
            raise ValueError(
                "missing downstream dependency evidence; refusing adoption"
            )
        if previous is not None and previous != evidence and previous != empty:
            raise ValueError("downstream per-clip source/config dependencies changed")
        found = _stage_receipts(shard, uid, identity, state)
        transition = None
        if state:
            retryable = (
                state.get("identity") == frozen.fingerprint(empty)
                and not found
                and not exported
                and (
                    state.get("preparation_failed")
                    or (
                        state.get("t2va_status") == "skipped"
                        and state.get("ta2va_status") == "skipped"
                        and state.get("failure_stage") == "upstream"
                        and state.get("failure_reason")
                        in {
                            "audio_preprocessing_required",
                            "resolved_speech_unavailable",
                            "resolved_audio_unavailable",
                            "asr_failed",
                        }
                    )
                )
            )
            if state["identity"] != identity and not retryable:
                raise ValueError("downstream checkpoint identity changed")
            if retryable and evidence["dependencies"] is not None:
                transition = {
                    **state,
                    "identity": identity,
                    "t2va_status": "pending",
                    "ta2va_status": "pending",
                    "failure_stage": None,
                    "failure_reason": None,
                    "preparation_failed": False,
                    "reopened_from": {
                        k: state.get(k)
                        for k in ("identity", "failure_stage", "failure_reason")
                    },
                }
        updates.append((path, evidence, transition))
    # No durable mutation until every clip's existing evidence has passed.
    production._repair_tail(journal)
    _reopen_exports(shard, exports)
    for path, evidence, transition in updates:
        if not path.exists() or json.loads(path.read_text()) != evidence:
            production.atomic_json(path, evidence)
        if transition:
            production.append_row(journal, transition)


def run_downstream(
    root,
    index,
    shard_id,
    audio_root,
    clips_root,
    source_videos_root,
    backend,
    profiles,
    allow_unverified=False,
    request_workers=1,
    run_id="production",
):
    """Run target stages; caller must hold this shard's invocation.lock."""
    root = Path(root).resolve()
    rows, contexts = production.prepare_shard(
        root,
        index,
        shard_id,
        audio_production_root=Path(audio_root).resolve(),
        audio_shadow_run_id=run_id,
        clips_root=Path(clips_root).resolve(),
        source_videos_root=Path(source_videos_root).resolve(),
        backend=backend.provenance(),
    )
    processor = _Processor(
        contexts, backend, profiles, index, allow_unverified=allow_unverified
    )
    shard = root / "shards" / production.shard_name(shard_id)
    with production.file_lock(shard / "shard.lock"):
        _preflight(shard, rows, processor)
    return production.process_shard(
        root, shard_id, rows, processor, request_workers=request_workers
    )
