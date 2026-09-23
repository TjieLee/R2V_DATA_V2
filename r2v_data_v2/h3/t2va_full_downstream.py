"""Full-supervisor resume policy around the unchanged target-side scheduler.

The caller holds invocation.lock across upstream publication and this call.
Dependency sidecars are audit records; resume is not gated on runtime provenance.
"""

from __future__ import annotations

import json
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
    if (shard / "COMPLETE").exists():
        return
    states = production.load_states(shard)
    sources = [{k: r[k] for k in ("source_index", "clip_uid", "video")} for r in rows]
    path = shard / "sources.json"
    if path.exists() and json.loads(path.read_text()) != sources:
        raise ValueError("downstream source population changed")
    if len({r["clip_uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate downstream source identity")
    updates = []
    for row in rows:
        uid = row["clip_uid"]
        if Path(uid).name != uid or uid in {".", ".."}:
            raise ValueError("unsafe downstream identity")
        evidence = processor.evidence(row)
        identity = processor.identity(row)
        path = shard / "downstream_dependencies" / f"{uid}.json"
        state = states.get(uid)
        if path.is_symlink():
            raise ValueError("redirected downstream dependencies")
        # Historical dependency sidecars are audit records, never resume gates.
        # The immutable source population and published artifact checks below are
        # sufficient to protect ownership without coupling resume to upstream
        # implementation/provenance metadata.
        _stage_receipts(shard, uid, identity, state)
        updates.append((path, evidence))
    # Publish current audit evidence only after structural preflight succeeds.
    for path, evidence in updates:
        if not path.exists() or json.loads(path.read_text()) != evidence:
            production.atomic_json(path, evidence)


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
    shard = root / "shards" / production.shard_name(shard_id)
    with production.file_lock(shard / "shard.lock"):
        completed = production.seal_terminal_shard(shard)
    if completed is not None:
        return completed
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
