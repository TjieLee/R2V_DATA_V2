"""Fixed, model-free early stem resolution. No fallback or waveform rewriting."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal, Self

import soundfile as sf
from pydantic import Field, model_validator

from r2v_data_v2.h3.auk_speech_shadow import (
    AukJob,
    Hash,
    _check_fingerprint,
    _signed,
    check_source,
    load_auk_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemInventory,
    SAMAudioStemRecord,
    StemArtifact,
    _publish_directory,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    load_stem_shadow,
    sha256_file,
    stem_separation_root,
    stem_shadow_root,
)
from r2v_data_v2.h3.schemas import SchemaModel

RESOLVED_STAGE = "resolved_stems_v1"


class ResolvedStem(SchemaModel):
    kind: Literal["speech", "music", "sfx"]
    source_backend: Literal["auk", "sam_audio"]
    canonical_stem_path: str
    canonical_stem_sha256: Hash
    canonical_sample_rate_hz: Literal[32000] = 32000
    canonical_channels: Literal[2] = 2
    canonical_frame_count: int = Field(gt=0)
    source_audio_path: str
    source_audio_sha256: Hash
    source_start_sample: Literal[0] = 0
    source_end_sample: int = Field(gt=0)
    alignment_offset_seconds: Literal[0] = 0

    @property
    def stem_type(self) -> str:
        return self.kind

    @property
    def canonical_duration_seconds(self) -> float:
        return self.canonical_frame_count / 32000

    @property
    def source_duration_seconds(self) -> float:
        return self.source_end_sample / 32000

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        expected = "auk" if self.kind == "speech" else "sam_audio"
        if self.source_backend != expected:
            raise ValueError("resolved stem violates fixed AuK/SAM policy")
        if abs(self.canonical_frame_count - self.source_end_sample) > 3200:
            raise ValueError("resolved stem exceeds zero-offset timeline tolerance")
        if (
            self.kind == "speech"
            and self.canonical_frame_count != self.source_end_sample
        ):
            raise ValueError("resolved AuK speech extent differs from target")
        return self


class ResolvedStemRecord(SchemaModel):
    schema_version: Literal["r2v.h3.resolved_stem_record.1"] = (
        "r2v.h3.resolved_stem_record.1"
    )
    clip_uid: str
    inventory_fingerprint: Hash
    source_sam_record_fingerprint: Hash
    source_auk_record_fingerprint: Hash
    status: Literal["ready", "failed"]
    verification_state: Literal["unverified"] = "unverified"
    failure_reason: str | None = None
    speech: ResolvedStem | None = None
    music: ResolvedStem | None = None
    sfx: ResolvedStem | None = None
    record_fingerprint: Hash

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _check_fingerprint(self, "record_fingerprint")
        stems = [self.speech, self.music, self.sfx]
        if self.status == "ready":
            if self.failure_reason or any(s is None for s in stems):
                raise ValueError("ready resolved record requires all three stems")
            if [s.kind for s in stems] != ["speech", "music", "sfx"]:
                raise ValueError("resolved stem kinds differ")
            if (
                len(
                    {
                        (
                            s.source_audio_path,
                            s.source_audio_sha256,
                            s.source_end_sample,
                        )
                        for s in stems
                    }
                )
                != 1
            ):
                raise ValueError("resolved stems disagree on original target timeline")
        elif not self.failure_reason or any(s is not None for s in stems):
            raise ValueError(
                "failed resolution must not publish partial/fallback stems"
            )
        return self

    # Common downstream accessors; no SAM model/call provenance is fabricated.
    @property
    def route(self) -> Literal["resolved"]:
        return "resolved"

    @property
    def separation_state(self) -> str:
        return "failure" if self.status == "failed" else self.verification_state

    @property
    def stems(self) -> list[ResolvedStem]:
        return [s for s in (self.music, self.speech, self.sfx) if s is not None]

    def stem(self, kind: str) -> ResolvedStem:
        matches = [s for s in self.stems if s.kind == kind]
        if len(matches) != 1:
            raise ValueError(f"resolved record lacks one {kind} stem")
        return matches[0]


class ResolvedStemInventory(SchemaModel):
    schema_version: Literal["r2v.h3.resolved_stem_inventory.1"] = (
        "r2v.h3.resolved_stem_inventory.1"
    )
    speech_source: Literal["auk"] = "auk"
    music_source: Literal["sam_audio"] = "sam_audio"
    sfx_source: Literal["sam_audio"] = "sam_audio"
    source_sam_root: str
    source_auk_root: str
    source_sam_inventory_fingerprint: Hash
    source_auk_inventory_fingerprint: Hash
    source_sam_records_sha256: Hash
    source_auk_records_sha256: Hash
    source_canonical_audio_manifest_path: str
    source_canonical_audio_manifest_sha256: Hash
    clip_uids: list[str] = Field(min_length=1)
    jobs: list[AukJob] = Field(min_length=1)
    inventory_fingerprint: Hash

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        _check_fingerprint(self, "inventory_fingerprint")
        if self.clip_uids != [j.clip_uid for j in self.jobs] or len(
            set(self.clip_uids)
        ) != len(self.clip_uids):
            raise ValueError("resolved ordered inventory differs")
        return self


class ResolvedStemSummary(SchemaModel):
    schema_version: Literal["r2v.h3.resolved_stem_summary.1"] = (
        "r2v.h3.resolved_stem_summary.1"
    )
    inventory_fingerprint: Hash
    clip_uids: list[str]
    ready_count: int
    failed_count: int
    model_call_count: Literal[0] = 0
    production_artifacts_modified: Literal[False] = False


StemInventory = SAMAudioStemInventory | ResolvedStemInventory
StemRecord = SAMAudioStemRecord | ResolvedStemRecord
StemMedia = StemArtifact | ResolvedStem


def downstream_stem_root(audio_root: Path, run_id: str | None) -> Path:
    """Explicit legacy root remains readable; named runs never fall back."""
    if run_id is None:
        return stem_separation_root(audio_root)
    root = stem_shadow_root(audio_root, run_id) / RESOLVED_STAGE
    if root.resolve() != root:
        raise ValueError("resolved stem root must not redirect")
    return root


def downstream_stem_route(
    run_id: str | None, legacy_route: str | None = None
) -> Literal["resolved", "music_first", "voice_first"]:
    if run_id is not None:
        if legacy_route not in (None, "music_first"):
            raise ValueError(
                "named resolved runs require the fixed SAM music_first source"
            )
        return "resolved"
    if legacy_route not in (None, "music_first", "voice_first"):
        raise ValueError("unknown legacy stem route")
    return legacy_route or "music_first"


def _resolve(
    root: Path,
) -> tuple[ResolvedStemInventory, list[ResolvedStemRecord], ResolvedStemSummary]:
    sam_root, auk_root = root.parent / "separation", root.parent / "auk_speech_v1"
    sam, sam_records, _ = load_stem_shadow(sam_root)
    auk, auk_records, _ = load_auk_shadow(auk_root)
    if sam.route != "music_first" or sam.run_both_routes:
        raise ValueError("resolved policy requires SAM music_first only")
    if (
        sam.clip_uids != auk.clip_uids
        or [r.clip_uid for r in sam_records] != sam.clip_uids
    ):
        raise ValueError("resolved AuK/SAM ordered case inventory differs")
    if (
        sam.source_canonical_audio_manifest_path
        != auk.source_canonical_audio_manifest_path
        or sam.source_canonical_audio_manifest_sha256
        != auk.source_canonical_audio_manifest_sha256
    ):
        raise ValueError("resolved AuK/SAM canonical manifest lineage differs")
    inventory = _signed(
        ResolvedStemInventory,
        {
            "source_sam_root": str(sam_root),
            "source_auk_root": str(auk_root),
            "source_sam_inventory_fingerprint": sam.inventory_fingerprint,
            "source_auk_inventory_fingerprint": auk.inventory_fingerprint,
            "source_sam_records_sha256": sha256_file(sam_root / "records.jsonl"),
            "source_auk_records_sha256": sha256_file(auk_root / "records.jsonl"),
            "source_canonical_audio_manifest_path": auk.source_canonical_audio_manifest_path,
            "source_canonical_audio_manifest_sha256": auk.source_canonical_audio_manifest_sha256,
            "clip_uids": auk.clip_uids,
            "jobs": [j.model_dump(mode="json") for j in auk.jobs],
        },
        "inventory_fingerprint",
    )
    records = []
    for job, sam_job, a, s in zip(
        auk.jobs, sam.jobs, auk_records, sam_records, strict=True
    ):
        values = {
            "clip_uid": job.clip_uid,
            "inventory_fingerprint": inventory.inventory_fingerprint,
            "source_sam_record_fingerprint": s.record_fingerprint,
            "source_auk_record_fingerprint": a.record_fingerprint,
            "status": "ready",
        }
        try:
            check_source(job)
            if (
                sam_job.model_dump() != job.model_dump()
                or s.clip_uid != job.clip_uid
                or a.clip_uid != job.clip_uid
            ):
                raise ValueError("AuK/SAM canonical target lineage differs")
            if s.separation_state == "failure" or a.status != "ready":
                raise ValueError(
                    f"upstream failure: SAM={s.failure_reason}; AuK={a.failure_reason}"
                )
            for kind in ("speech", "music", "sfx"):
                if kind == "speech":
                    artifact = a.canonical
                    path, digest, frames = (
                        artifact.path,
                        artifact.sha256,
                        artifact.frame_count,
                    )
                else:
                    artifact = s.stem(kind)
                    if (
                        artifact.source_audio_path,
                        artifact.source_audio_sha256,
                        artifact.source_end_sample,
                    ) != (
                        job.source_audio_path,
                        job.source_audio_sha256,
                        job.source_frame_count,
                    ):
                        raise ValueError("SAM stem source target differs")
                    path, digest, frames = (
                        artifact.canonical_stem_path,
                        artifact.canonical_stem_sha256,
                        artifact.canonical_frame_count,
                    )
                info = sf.info(path)
                if sha256_file(Path(path)) != digest or (
                    info.format,
                    info.subtype,
                    info.samplerate,
                    info.channels,
                    info.frames,
                ) != (
                    "WAV",
                    "PCM_16",
                    32000,
                    2,
                    frames,
                ):
                    raise ValueError("resolved canonical stem media differs")
                values[kind] = ResolvedStem(
                    kind=kind,
                    source_backend="auk" if kind == "speech" else "sam_audio",
                    canonical_stem_path=path,
                    canonical_stem_sha256=digest,
                    canonical_frame_count=frames,
                    source_audio_path=job.source_audio_path,
                    source_audio_sha256=job.source_audio_sha256,
                    source_end_sample=job.source_frame_count,
                ).model_dump(mode="json")
        except (ValueError, OSError, sf.LibsndfileError) as exc:
            values.update(
                status="failed",
                failure_reason=str(exc),
                speech=None,
                music=None,
                sfx=None,
            )
        records.append(_signed(ResolvedStemRecord, values, "record_fingerprint"))
    summary = ResolvedStemSummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        clip_uids=inventory.clip_uids,
        ready_count=sum(r.status == "ready" for r in records),
        failed_count=sum(r.status == "failed" for r in records),
    )
    return inventory, records, summary


def resolve_audio_stems(
    *, audio_production_root: Path, shadow_run_id: str, overwrite: bool = False
):
    destination = downstream_stem_root(audio_production_root, shadow_run_id)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        # Ownership validation, without requiring stale upstream hashes to still match.
        old = ResolvedStemInventory.model_validate_json(
            (destination / "inventory.json").read_text()
        )
        if (
            Path(old.source_sam_root) != destination.parent / "separation"
            or Path(old.source_auk_root) != destination.parent / "auk_speech_v1"
        ):
            raise ValueError("resolved destination ownership differs")
    inventory, records, summary = _resolve(destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=".resolved-stems-", dir=destination.parent)
    )
    try:
        _write_json(temporary / "inventory.json", inventory)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        if (inventory, records, summary) != _resolve(destination):
            raise ValueError("resolved stem sources changed during publication")
        _publish_directory(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


def load_resolved_stems(root: Path):
    root = root.resolve(strict=True)
    if root.name != RESOLVED_STAGE:
        raise ValueError("resolved stems require their owned stage root")
    inventory = ResolvedStemInventory.model_validate_json(
        (root / "inventory.json").read_text()
    )
    records = [
        ResolvedStemRecord.model_validate(r)
        for r in _read_jsonl(root / "records.jsonl")
    ]
    summary = ResolvedStemSummary.model_validate_json(
        (root / "summary.json").read_text()
    )
    if (inventory, records, summary) != _resolve(root):
        raise ValueError("resolved stem source lineage changed; re-resolve explicitly")
    return inventory, records, summary


def load_stem_source(root: Path):
    if root.name == RESOLVED_STAGE:
        return load_resolved_stems(root)
    return load_stem_shadow(root)  # explicit legacy reads only, no automatic fallback


def validate_stem_record(record: StemRecord) -> StemRecord:
    model = (
        ResolvedStemRecord
        if record.schema_version == "r2v.h3.resolved_stem_record.1"
        else SAMAudioStemRecord
    )
    return model.model_validate(record.model_dump())
