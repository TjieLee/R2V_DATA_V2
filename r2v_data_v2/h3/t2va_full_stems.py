"""Production adapters for frozen SAM and AuK stem producers."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.t2va_production import atomic_json


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    return str(destination)


class AukWorker:
    def __init__(self, configuration):
        from r2v_data_v2.h3.auk_speech_shadow import (
            AukConfiguration,
            PersistentAukBackend,
        )

        self.configuration = configuration
        self.backend = PersistentAukBackend(
            AukConfiguration.model_validate(configuration["model"])
        )

    def __enter__(self):
        self.backend.__enter__()
        return self

    def __exit__(self, *args):
        self.backend.close()

    def infer(self, job, output_dir):
        from r2v_data_v2.h3 import auk_speech_shadow as auk

        source = auk.AukJob.model_validate(job["source"])
        auk.check_source(source)
        raw = output_dir / "speech.wav"
        try:
            response = self.backend.generate(source, raw)
        except Exception as exc:
            if (
                isinstance(exc, TimeoutError)
                or self.backend.process is None
                or self.backend.process.poll() is not None
            ):
                raise SystemExit(str(exc)) from exc
            raise
        if response["status"] != "ok":
            if "out of memory" in response.get("reason", "").lower():
                raise SystemExit(response["reason"])
            raise ValueError(response["reason"])
        return {"raw_path": str(raw), "response": response}

    def finalize(self, job, output_dir, result):
        from r2v_data_v2.h3 import auk_speech_shadow as auk

        source = auk.AukJob.model_validate(job["source"])
        raw = Path(result["raw_path"])
        import soundfile as sf

        info = sf.info(raw)
        delta = info.frames / info.samplerate - source.source_duration_seconds
        if abs(delta) > 0.10 + 1e-12:
            raise ValueError("AuK raw duration exceeds 0.10 second tolerance")
        canonical = output_dir / "canonical.wav"
        adjustment = auk.canonicalize_speech(
            raw,
            canonical,
            source.source_frame_count,
            ffmpeg=self.configuration["ffmpeg"],
        )
        return {
            **result,
            "canonical_path": str(canonical),
            "raw_duration_delta_seconds": delta,
            "canonical_adjustment_samples": adjustment,
        }

    def process(self, job, output_dir):
        return self.finalize(job, output_dir, self.infer(job, output_dir))


def run_auk(inventory, state_root, gpu_ids, *, eligible, ffmpeg="ffmpeg", execute=None):
    from r2v_data_v2.h3 import auk_speech_shadow as auk

    if execute is None:
        from r2v_data_v2.h3.t2va_full_workers import execute_stage

        execute = execute_stage
    results = execute(
        state_root,
        [
            {"job_id": j.clip_uid, "source": j.model_dump(mode="json")}
            for j in inventory.jobs
            if j.clip_uid in eligible
        ],
        gpu_ids=gpu_ids,
        factory=__name__ + ":AukWorker",
        log_root=state_root.parent.parent / "logs",
        configuration={
            "model": inventory.model_configuration.model_dump(mode="json"),
            "ffmpeg": ffmpeg,
        },
    )

    destination = auk.auk_stage_root(
        Path(inventory.audio_production_root), inventory.shadow_run_id
    )
    overwrite = destination.exists()
    auk.preflight(inventory, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{auk.AUK_STAGE}-", dir=destination.parent))
    records = []
    try:
        for job in inventory.jobs:
            raw_rel = Path("raw") / f"{job.clip_uid}.wav"
            canonical_rel = Path("canonical") / job.clip_uid / "speech.wav"
            raw, canonical = temporary / raw_rel, temporary / canonical_rel
            values = {
                "clip_uid": job.clip_uid,
                "inventory_fingerprint": inventory.inventory_fingerprint,
                "configuration_fingerprint": inventory.model_configuration.configuration_fingerprint,
                "source_audio_sha256": job.source_audio_sha256,
                "status": "failed",
                "expected_frame_count": round(job.source_duration_seconds * 32000),
                "model_runtime_seconds": 0.0,
                "model_call_count": 1,
            }
            started = time.monotonic()
            try:
                row = results.get(job.clip_uid)
                if row is None or row["status"] != "ready":
                    raise RuntimeError(
                        "upstream SAM unavailable" if row is None else row["failure_reason"]
                    )
                output = row["result"]
                raw.parent.mkdir(parents=True, exist_ok=True)
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output["raw_path"], raw)
                values["model_runtime_seconds"] = output["response"]["model_runtime_seconds"]
                values["raw"] = auk.audio_artifact(raw, destination / raw_rel).model_dump(mode="json")
                if "canonical_path" in output:
                    shutil.copyfile(output["canonical_path"], canonical)
                    values["raw_duration_delta_seconds"] = output["raw_duration_delta_seconds"]
                    values["canonical_adjustment_samples"] = output["canonical_adjustment_samples"]
                else:
                    # Older ready receipts contain the generated raw waveform only.
                    import soundfile as sf

                    info = sf.info(raw)
                    delta = info.frames / info.samplerate - job.source_duration_seconds
                    if abs(delta) > 0.10 + 1e-12:
                        raise ValueError("AuK raw duration exceeds 0.10 second tolerance")
                    values["raw_duration_delta_seconds"] = delta
                    values["canonical_adjustment_samples"] = auk.canonicalize_speech(
                        raw, canonical, values["expected_frame_count"], ffmpeg=ffmpeg
                    )
                values["canonical"] = auk.audio_artifact(
                    canonical, destination / canonical_rel
                ).model_dump(mode="json")
                values["status"] = "ready"
            except Exception as exc:  # noqa: BLE001 - preserve per-clip failure isolation
                values.update(
                    status="failed",
                    canonical=None,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
                if not values["model_runtime_seconds"]:
                    values["model_runtime_seconds"] = time.monotonic() - started
                canonical.unlink(missing_ok=True)
            records.append(auk._signed(auk.AukRecord, values, "record_fingerprint"))
        summary = auk._summary(inventory, records)
        auk._write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        auk._write_jsonl(temporary / "records.jsonl", records)
        auk._write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        auk._publish_directory(temporary, destination, overwrite=overwrite)
        sam_root = destination.parent / "separation"
        if sam_root.is_dir():
            prepare_resolved_stems(destination.parent, state_root / "resolved_prepared.json")
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_resolved_stems(shadow_root: Path, output_path: Path) -> None:
    """Prepare existing resolved rows from published source records, without media replay."""
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import resolved_audio_stems as resolved

    sam_root, auk_root = shadow_root / "separation", shadow_root / auk.AUK_STAGE
    sam_inventory = sam.SAMAudioStemInventory.model_validate_json(
        (sam_root / "inventory.json").read_text()
    )
    sam_records = [
        sam.SAMAudioStemRecord.model_validate(row)
        for row in sam._read_jsonl(sam_root / "records.jsonl")
    ]
    auk_inventory = auk.AukInventory.model_validate_json(
        (auk_root / "inventory.json").read_text()
    )
    auk_records = [
        auk.AukRecord.model_validate(row)
        for row in sam._read_jsonl(auk_root / "records.jsonl")
    ]
    inventory, records, summary = resolved._resolve_records(
        shadow_root / resolved.RESOLVED_STAGE,
        sam_inventory, sam_records, auk_inventory, auk_records,
        verify_media=False,
    )
    atomic_json(
        output_path,
        {
            "inventory": inventory.model_dump(mode="json"),
            "records": [record.model_dump(mode="json") for record in records],
            "summary": summary.model_dump(mode="json"),
        },
    )


class SAMWorker:
    def __init__(self, configuration):
        model = sam.SAMAudioModelConfiguration.model_validate(configuration["model"])
        self.backend = sam.OfficialSAMAudioBackend(model)
        self.inventory = None
        self.ffmpeg = configuration.get("ffmpeg", "ffmpeg")
        self.ffprobe = configuration.get("ffprobe", "ffprobe")
        if "inventory_path" in configuration:
            self.bind_request(configuration)

    def bind_request(self, configuration):
        inventory = sam.SAMAudioStemInventory.model_validate_json(
            Path(configuration["inventory_path"]).read_text()
        )
        self.inventory = inventory

    def __enter__(self):
        self.backend._load()
        return self

    def __exit__(self, *args):
        self.backend = None

    def infer(self, job, output_dir):
        source = sam.SAMAudioStemJob.model_validate(job["source"])
        path = Path(source.source_audio_path).expanduser().resolve(strict=True)
        if sam.sha256_file(path) != source.source_audio_sha256:
            raise ValueError("canonical source Audio hash changed before SAM separation")
        probe = FFmpegAudioMediaBackend(
            ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
        ).probe_audio_file(path)
        if (
            probe.sample_rate_hz,
            probe.channels,
            probe.frame_count,
        ) != (sam.STEM_SAMPLE_RATE_HZ, sam.STEM_CHANNELS, source.source_frame_count):
            raise ValueError("canonical source Audio format or sample extent changed")
        raw = output_dir / "stems" / source.clip_uid / "music_first"
        raw.mkdir(parents=True, exist_ok=True)
        first_target = raw / "music.raw.wav"
        residual = raw / "residual_1.raw.wav"
        second_target = raw / "speech.raw.wav"
        sfx = raw / "sfx.raw.wav"
        outcomes = []
        for prompt, input_path, target, remainder in (
            (self.inventory.model_configuration.music_prompt, path, first_target, residual),
            (self.inventory.model_configuration.speech_prompt, residual, second_target, sfx),
        ):
            try:
                result = self.backend.separate(
                    clip_uid=source.clip_uid,
                    source_audio_path=input_path,
                    prompt=prompt,
                    target_path=target,
                    residual_path=remainder,
                )
                sam._require_separation_result_paths(
                    result, target_path=target, residual_path=remainder
                )
                outcomes.append({"result": result.model_dump(mode="json")})
                if result.verification_state == "failure":
                    break
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise SystemExit(str(exc)) from exc
                outcomes.append({"error": f"{type(exc).__name__}: {exc}"})
                break
        return outcomes

    def finalize(self, job, output_dir, outcomes):
        from r2v_data_v2.h3.t2va_full_workers import SampleJobFailure

        class Replay:
            def __init__(self):
                self.index = 0

            def separate(self, **_kwargs):
                if self.index >= len(outcomes):
                    raise ValueError("SAM Audio separation reported failure")
                row = outcomes[self.index]
                self.index += 1
                if "error" in row:
                    raise RuntimeError(row["error"])
                return sam.SAMAudioSeparationResult.model_validate(row["result"])

        try:
            record = sam._separate_one_route(
                inventory=self.inventory,
                job=sam.SAMAudioStemJob.model_validate(job["source"]),
                route="music_first",
                output_root=output_dir,
                backend=Replay(),
                canonicalizer=sam.FFmpegStemCanonicalizer(
                    ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                ),
                raw_probe_backend=FFmpegAudioMediaBackend(
                    ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                ),
                source_validated=True,
            )
        except sam._SAMAudioRouteFailure as exc:
            if "out of memory" in str(exc).lower():
                raise SystemExit(str(exc)) from exc
            raise SampleJobFailure(
                str(exc),
                {
                    "calls": [c.model_dump(mode="json") for c in exc.calls],
                    "model_call_count": exc.model_call_count,
                },
            ) from exc
        return {
            "record": record.model_dump(mode="json"),
            "output_root": str(output_dir),
        }

    def process(self, job, output_dir):
        return self.finalize(job, output_dir, self.infer(job, output_dir))

    @staticmethod
    def output_digests(result):
        record = sam.SAMAudioStemRecord.model_validate(result["record"])
        root = Path(result["output_root"])
        digests = {}
        for call in record.calls:
            for path, digest in (
                (call.target_path, call.target_sha256),
                (call.residual_path, call.residual_sha256),
            ):
                digests[Path(path).relative_to(root).as_posix()] = digest
        for stem in record.stems:
            digests[Path(stem.canonical_stem_path).relative_to(root).as_posix()] = (
                stem.canonical_stem_sha256
            )
        return digests


def run_sam(
    inventory,
    destination,
    state_root,
    gpu_ids,
    *,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
    execute=None,
):
    if execute is None:
        from r2v_data_v2.h3.t2va_full_workers import execute_stage

        execute = execute_stage
    if inventory.route != "music_first" or inventory.run_both_routes:
        raise ValueError("full production requires frozen music_first route")
    inventory_path = state_root / "inventory.json"
    atomic_json(inventory_path, inventory.model_dump(mode="json"))
    results = execute(
        state_root,
        [
            {"job_id": j.clip_uid, "source": j.model_dump(mode="json")}
            for j in inventory.jobs
        ],
        gpu_ids=gpu_ids,
        factory=__name__ + ":SAMWorker",
        log_root=state_root.parent.parent / "logs",
        configuration={
            "inventory_path": str(inventory_path),
            "model": inventory.model_configuration.model_dump(mode="json"),
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
        },
        environment={"PYTHONPATH": os.environ.get("SAM_AUDIO_RUNTIME_PYTHONPATH", "")},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".separation-publish-", dir=destination.parent)
    )
    records = []
    try:
        for job in inventory.jobs:
            result = results[job.clip_uid]
            if result["status"] == "ready":
                record = sam.SAMAudioStemRecord.model_validate(
                    result["result"]["record"]
                )
                source = Path(result["result"]["output_root"])
                relative = Path("clips") / job.clip_uid
                shutil.copytree(
                    source, temporary / relative, copy_function=link_or_copy
                )
                record = sam._published_record_paths(
                    record,
                    temporary_root=source,
                    destination_root=destination / relative,
                )
                values = record.model_dump(mode="json", exclude={"record_fingerprint"})
                values["inventory_fingerprint"] = inventory.inventory_fingerprint
            else:
                detail = result.get("result") or {}
                values = {
                    "schema_version": sam.SAM_AUDIO_STEM_RECORD_VERSION,
                    "clip_uid": job.clip_uid,
                    "route": "music_first",
                    "inventory_fingerprint": inventory.inventory_fingerprint,
                    "model_configuration_fingerprint": inventory.model_configuration.configuration_fingerprint,
                    "separation_state": "failure",
                    "model_call_count": detail.get("model_call_count", 0),
                    "calls": detail.get("calls", []),
                    "stems": [],
                    "failure_reason": result.get("failure_reason", "worker job failed"),
                }
            records.append(
                sam.SAMAudioStemRecord(
                    **values,
                    record_fingerprint=sam._sha256_text(sam._compact_json(values)),
                )
            )
        summary = sam.SAMAudioStemSummary(
            inventory_fingerprint=inventory.inventory_fingerprint,
            clip_count=len(inventory.jobs),
            record_count=len(records),
            route_counts={"music_first": len(records)},
            verification_state_counts=dict(
                sorted(Counter(r.separation_state for r in records).items())
            ),
            model_call_count=sum(r.model_call_count for r in records),
        )
        sam._write_json(temporary / "inventory.json", inventory)
        sam._write_jsonl(temporary / "records.jsonl", records)
        sam._write_json(temporary / "summary.json", summary)
        sam._publish_directory(temporary, destination, overwrite=destination.exists())
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    sam.load_stem_shadow(destination)
    return summary
