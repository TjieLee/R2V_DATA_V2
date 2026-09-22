"""Production adapters for frozen SAM and AuK stem producers."""

from __future__ import annotations

import os
import shutil
import tempfile
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

    def process(self, job, output_dir):
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
        # Cache only an output that passes the same frozen media checks.
        import soundfile as sf

        info = sf.info(raw)
        if (
            abs(info.frames / info.samplerate - source.source_duration_seconds)
            > 0.10 + 1e-12
        ):
            raise ValueError("AuK raw duration exceeds 0.10 second tolerance")
        auk.canonicalize_speech(
            raw,
            output_dir / "canonical.wav",
            source.source_frame_count,
            ffmpeg=self.configuration["ffmpeg"],
        )
        return {"raw_path": str(raw), "response": response}


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

    class Replay:
        configuration = inventory.model_configuration

        def generate(self, job, raw_path):
            row = results.get(job.clip_uid)
            if row is None:
                return {
                    "status": "failed",
                    "reason": "upstream SAM unavailable",
                    "model_runtime_seconds": 0.0,
                }
            if row["status"] != "ready":
                return {
                    "status": "failed",
                    "reason": row["failure_reason"],
                    "model_runtime_seconds": 0.0,
                }
            shutil.copyfile(row["result"]["raw_path"], raw_path)
            return row["result"]["response"]

    result = auk.run_auk_speech_shadow(
        inventory=inventory,
        backend=Replay(),
        ffmpeg=ffmpeg,
        overwrite=auk.auk_stage_root(
            Path(inventory.audio_production_root), inventory.shadow_run_id
        ).exists(),
    )
    auk.load_auk_shadow(
        auk.auk_stage_root(
            Path(inventory.audio_production_root), inventory.shadow_run_id
        )
    )
    return result


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

    def process(self, job, output_dir):
        from r2v_data_v2.h3.t2va_full_workers import SampleJobFailure

        try:
            record = sam._separate_one_route(
                inventory=self.inventory,
                job=sam.SAMAudioStemJob.model_validate(job["source"]),
                route="music_first",
                output_root=output_dir,
                backend=self.backend,
                canonicalizer=sam.FFmpegStemCanonicalizer(
                    ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                ),
                raw_probe_backend=FFmpegAudioMediaBackend(
                    ffmpeg=self.ffmpeg, ffprobe=self.ffprobe
                ),
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
