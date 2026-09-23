"""Full-supervisor resume policy around the unchanged target-side scheduler.

The caller holds invocation.lock across upstream publication and this call.
Dependency sidecars are audit records; resume is not gated on runtime provenance.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from r2v_data_v2.h3 import t2va_production as production
from r2v_data_v2.h3 import t2va_shadow as frozen


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
            "version": "full-downstream-per-clip-v3-content-only",
            "allow_unverified": allow_unverified,
        }
        self.dependencies = {}
        for uid, (job, _inventory) in contexts.items():
            if job.upstream_failure:
                continue
            job_value = job.model_dump(mode="json")
            audio = job_value.get("audio_evidence")
            if isinstance(audio, dict):
                # This fingerprint is a publication/lineage identifier, not
                # downstream conditioning content. Actual media hashes remain.
                audio.pop("resolved_record_fingerprint", None)
            self.dependencies[uid] = {"job": job_value}

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
            if not directory.is_dir():
                raise ValueError("invalid downstream stage artifacts")
            receipt = json.loads((directory / "stage.json").read_text())
            for relative in receipt["files"]:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("invalid stage artifact path")
            # Historical stage identities may include runtime/model provenance.
            # Published artifact presence, not model metadata, is the resume gate.
            for relative in receipt["files"]:
                artifact = directory / relative
                if artifact.is_symlink() or not artifact.is_file():
                    raise ValueError("published stage artifact is missing")
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


def _model_agnostic_evidence(value):
    """Project historical dependency sidecars to stable downstream content."""
    if value is None:
        return None
    source = value.get("source")
    policy = value.get("policy") or {}
    dependencies = value.get("dependencies")
    stable_dependencies = None
    if isinstance(dependencies, dict) and isinstance(dependencies.get("job"), dict):
        job = json.loads(json.dumps(dependencies["job"]))
        audio = job.get("audio_evidence")
        if isinstance(audio, dict):
            audio.pop("resolved_record_fingerprint", None)
        stable_dependencies = {"job": job}
    return {
        "population": value.get("population"),
        "source": source,
        "policy": {"allow_unverified": policy.get("allow_unverified")},
        "dependencies": stable_dependencies,
    }


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
        exported = any(row["video"] in task for task in exports.values())
        # Historical dependency sidecars are audit records, never resume gates.
        # The immutable source population and published artifact checks below are
        # sufficient to protect ownership without coupling resume to upstream
        # implementation/provenance metadata.
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
            # Historical state identities may include MiMo backend/model
            # provenance. Full-production resume is model-agnostic once the
            # source/upstream evidence above has matched.
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
    # Publish current audit evidence only after structural preflight succeeds.
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
    print(f"shard={shard_id} downstream_prepare_start", flush=True)
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
    print(
        f"shard={shard_id} downstream_prepare_ready rows={len(rows)} "
        f"contexts={len(contexts)}",
        flush=True,
    )
    print(f"shard={shard_id} downstream_processor_start", flush=True)
    processor = _Processor(
        contexts, backend, profiles, index, allow_unverified=allow_unverified
    )
    print(f"shard={shard_id} downstream_processor_ready", flush=True)
    shard = root / "shards" / production.shard_name(shard_id)
    print(f"shard={shard_id} downstream_preflight_start", flush=True)
    with production.file_lock(shard / "shard.lock"):
        _preflight(shard, rows, processor)
    print(f"shard={shard_id} downstream_preflight_ready", flush=True)
    return production.process_shard(
        root,
        shard_id,
        rows,
        processor,
        request_workers=request_workers,
        allow_existing_identity_mismatch=True,
    )
