"""Parallel inference with publication owned entirely by the frozen stem runners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from r2v_data_v2.h3 import diarization_binding as diar
from r2v_data_v2.h3 import qwen3_asr as asr
from r2v_data_v2.h3 import sam_audio_stem_shadow as frozen
from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path
from r2v_data_v2.h3.resolved_audio_stems import RESOLVED_STAGE, load_stem_source


def execute_stage(*args, **kwargs):
    # Keep CPU imports usable while the independently owned executor is absent.
    from r2v_data_v2.h3.t2va_full_workers import execute_stage as execute

    return execute(*args, **kwargs)


def _environment(prefix):
    names = {
        "DIARIZEN": (
            "PYTHON",
            "CODE_ROOT",
            "MODEL_PATH",
            "MODEL_IDENTIFIER",
            "TIMEOUT_SECONDS",
        ),
        "QWEN3_ASR": (
            "ENV",
            "MODEL_PATH",
            "DTYPE",
            "MAX_INFERENCE_BATCH_SIZE",
            "TIMEOUT_SECONDS",
        ),
    }[prefix]
    return {
        **{
            f"{prefix}_{name}": os.environ[f"{prefix}_{name}"]
            for name in names
            if f"{prefix}_{name}" in os.environ
        },
        f"{prefix}_DEVICE": "cuda:0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _diar_configuration():
    from tools.run_h3_diarization_binding import _configuration_fingerprint

    environment = _environment("DIARIZEN")
    model = fingerprint_local_model_path(Path(environment["DIARIZEN_MODEL_PATH"]))
    identifier = environment.get(
        "DIARIZEN_MODEL_IDENTIFIER", diar.DEFAULT_DIARIZEN_MODEL_IDENTIFIER
    )
    provenance = diar.DiarizationBackendProvenance(
        backend="diarizen_official_pipeline",
        model_identifier=identifier,
        model_fingerprint=model,
        configuration_fingerprint=_configuration_fingerprint(
            model_identifier=identifier,
            model_fingerprint=model,
            requested_device="cuda:0",
            input_profile="canonical_32k_stereo",
        ),
        input_profile="canonical_32k_stereo",
        input_preprocessing=diar.DIARIZATION_CANONICAL_PREPROCESSING_VERSION,
        source_sample_rate_hz=32000,
        source_channels=2,
    )
    return {
        "adapter_version": 1,
        "environment": environment,
        "provenance": provenance.model_dump(mode="json"),
    }


def _asr_configuration():
    configuration = asr.Qwen3ASRConfiguration.from_environment().model_copy(
        update={"device": "cuda:0"}
    )
    return {
        "adapter_version": 1,
        "environment": _environment("QWEN3_ASR"),
        "configuration": configuration.model_dump(mode="json"),
        "model_fingerprint": fingerprint_local_model_path(
            Path(configuration.local_model_path)
        ),
        "preprocessing": asr.QWEN3_ASR_PREPROCESSING_POLICY,
    }


def _inference(backend, call):
    """Do not let a dead persistent child become a durable sample failure."""
    process = getattr(backend, "_process", None)
    if process is None or process.poll() is not None:
        raise SystemExit("speech inference worker is not running")
    try:
        result = call()
    except Exception as exc:
        process = getattr(backend, "_process", None)
        if (
            isinstance(
                exc,
                (TimeoutError, subprocess.TimeoutExpired, BrokenPipeError, EOFError),
            )
            or process is None
            or process.poll() is not None
            or "worker exited" in str(exc).lower()
            or "out of memory" in str(exc).lower()
        ):
            raise SystemExit(f"speech inference worker failed: {exc}") from exc
        raise
    process = getattr(backend, "_process", None)
    if process is None or process.poll() is not None:
        raise SystemExit("speech inference worker exited after response")
    return result


def _diar_result(result, target, provenance):
    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise TypeError("invalid DiariZen result")
    segments = [
        diar.DiarizationBackendSegment.model_validate(row) for row in result["segments"]
    ]
    # Validate cache eligibility with the same canonical boundary rules as replay.
    diar._normalize_segments(target=target, segments=segments, provenance=provenance)
    return segments


def _asr_result(result):
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or "language" not in result
        or (result["language"] is not None and not isinstance(result["language"], str))
    ):
        raise ValueError("invalid Qwen3-ASR result")
    return result["text"], result["language"]


class _DiarWorker:
    def __init__(self, backend):
        self.backend = backend

    def process(self, job, output_dir):
        del output_dir
        target = diar.DiarizationTargetClip.model_validate(job["target"])
        path = Path(target.source_audio_path)
        if frozen.sha256_file(path) != target.source_audio_sha256:
            raise ValueError("DiariZen source audio changed")
        segments = _inference(
            self.backend,
            lambda: self.backend.diarize(
                clip_uid=target.target_clip_uid,
                audio_path=path,
            ),
        )
        result = {"segments": [item.model_dump(mode="json") for item in segments]}
        _diar_result(result, target, self.backend.provenance)
        return result


class _ASRWorker:
    def __init__(self, backend, ffmpeg):
        self.backend = backend
        self.ffmpeg = ffmpeg

    def process(self, job, output_dir):
        del output_dir
        row = asr._ReadableDiarizationSegment.model_validate(job["segment"])
        path = Path(row.source_audio_path)
        if frozen.sha256_file(path) != job["source_audio_sha256"]:
            raise ValueError("ASR source audio changed")
        waveform, rate = asr.load_qwen3_asr_model_input(
            path,
            row.start_time,
            row.end_time,
            ffmpeg=self.ffmpeg,
        )
        text, language = _inference(
            self.backend,
            lambda: self.backend.transcribe(
                waveform=waveform,
                sample_rate_hz=rate,
            ),
        )
        result = {"text": text, "language": language}
        _asr_result(result)
        return result


@contextmanager
def diarizen_worker(configuration):
    from tools.run_h3_diarization_binding import _runtime_backend

    backend = None
    with tempfile.TemporaryDirectory(prefix="t2va-diarizen-") as temporary:
        try:
            os.environ.update(configuration["environment"])
            os.environ["DIARIZEN_DEVICE"] = "cuda:0"
            backend, _ = _runtime_backend(
                output_root=Path(temporary) / "stage",
                input_profile="canonical_32k_stereo",
            )
            # The CLI helper normally remaps a physical device. Here the outer
            # executor already selected it, and the nested child must retain it.
            backend.environment["CUDA_VISIBLE_DEVICES"] = os.environ[
                "CUDA_VISIBLE_DEVICES"
            ]
            if (
                backend.provenance.model_dump(mode="json")
                != configuration["provenance"]
            ):
                raise ValueError("DiariZen worker configuration changed")
            backend.__enter__()
        except Exception as exc:
            if backend is not None:
                backend.close()
            raise SystemExit(f"DiariZen initialization failed: {exc}") from exc
        try:
            yield _DiarWorker(backend)
        finally:
            backend.close()


@contextmanager
def asr_worker(configuration):
    from tools.run_h3_stem_qwen3_asr_shadow import _isolated_backend

    backend = None
    try:
        os.environ.update(configuration["environment"])
        os.environ["QWEN3_ASR_DEVICE"] = "cuda:0"
        backend = _isolated_backend()
        if (
            backend.configuration.model_dump(mode="json")
            != configuration["configuration"]
        ):
            raise ValueError("Qwen3-ASR worker configuration changed")
        backend.__enter__()
    except Exception as exc:
        if backend is not None:
            backend.close(force=True)
        raise SystemExit(f"Qwen3-ASR initialization failed: {exc}") from exc
    try:
        yield _ASRWorker(backend, configuration["ffmpeg"])
    finally:
        backend.close()


def _ready(results, job_id):
    record = results[job_id]
    if record["status"] != "ready":
        raise RuntimeError(record.get("failure_reason") or "speech inference failed")
    return record["result"]


def run_diarizen(
    audio_root,
    run_id,
    stage_state,
    gpu_ids,
    allow_unverified,
    ffmpeg="ffmpeg",
    execute=execute_stage,
):
    del ffmpeg  # DiariZen owns its canonical preprocessing.
    audio_root = Path(audio_root)
    shadow = frozen.stem_shadow_root(audio_root, run_id)
    stem_root = shadow / RESOLVED_STAGE
    stems, records, _ = load_stem_source(stem_root)
    source_root = audio_root / "diarization"
    source = diar.DiarizationInventory.model_validate_json(
        (source_root / "inventory.json").read_text()
    )
    if source.source_inventory_kind != "jea_shot_manifest" or any(
        target.visual_references
        or target.target_audio_binding_path is not None
        or target.target_audio_binding_sha256 is not None
        for target in source.targets
    ):
        raise ValueError(
            "T2VA speech requires target-only JEA inventory without bindings"
        )
    inventory = frozen.build_stem_diarization_inventory(
        stem_inventory=stems,
        stem_records=records,
        production_diarization_inventory=source,
        route="resolved",
        allow_unverified=allow_unverified,
    )
    configuration = _diar_configuration()
    provenance = diar.DiarizationBackendProvenance.model_validate(
        configuration["provenance"]
    )
    jobs = [
        {
            "job_id": target.target_clip_uid,
            "target": target.model_dump(mode="json"),
            "inventory_fingerprint": inventory.inventory_fingerprint,
        }
        for target in inventory.targets
    ]
    results = execute(
        stage_root=Path(stage_state),
        jobs=jobs,
        gpu_ids=list(gpu_ids),
        factory=f"{__name__}:diarizen_worker",
        log_root=Path(stage_state).parent.parent / "logs",
        configuration=configuration,
        environment=configuration["environment"],
    )
    by_id = {target.target_clip_uid: target for target in inventory.targets}

    class Replay:
        def diarize(self, *, clip_uid, audio_path):
            target = by_id[clip_uid]
            if Path(target.source_audio_path) != audio_path:
                raise ValueError("DiariZen replay source differs")
            return _diar_result(_ready(results, clip_uid), target, provenance)

    replay = Replay()
    replay.provenance = provenance
    published = frozen.run_stem_diarization_shadow(
        stem_root=stem_root,
        production_diarization_root=source_root,
        backend=replay,
        route="resolved",
        output_root=shadow / "diarization",
        allow_unverified=allow_unverified,
        overwrite=True,
    )
    return {"provenance": published.model_dump(mode="json"), "job_count": len(jobs)}


def run_asr(
    audio_root,
    run_id,
    stage_state,
    gpu_ids,
    allow_unverified,
    ffmpeg="ffmpeg",
    execute=execute_stage,
):
    shadow = frozen.stem_shadow_root(Path(audio_root), run_id)
    root = shadow / "diarization"
    provenance, _, _ = frozen.validate_stem_diarization_lineage(
        root, expected_shadow_root=shadow
    )
    if (
        provenance.route != "resolved"
        or provenance.binding_evidence_mode != "legacy_lr_asd"
    ):
        raise ValueError("T2VA ASR requires resolved target-side speech lineage")
    if provenance.unverified_clip_uids and not allow_unverified:
        raise ValueError("unverified stems require allow_unverified")
    inputs = asr._load_inputs(root)
    configuration = {**_asr_configuration(), "ffmpeg": ffmpeg}
    jobs = []
    keys = {}
    for row in inputs.readable_segments:
        job_id = hashlib.sha256(
            json.dumps([row.clip_uid, row.segment_id]).encode()
        ).hexdigest()
        jobs.append(
            {
                "job_id": job_id,
                "segment": row.model_dump(mode="json"),
                "source_audio_sha256": provenance.speech_stem_hashes_by_clip[
                    row.clip_uid
                ],
            }
        )
        key = (row.source_audio_path, row.start_time, row.end_time)
        keys.setdefault(key, []).append(job_id)
    results = execute(
        stage_root=Path(stage_state),
        jobs=jobs,
        gpu_ids=list(gpu_ids),
        factory=f"{__name__}:asr_worker",
        log_root=Path(stage_state).parent.parent / "logs",
        configuration=configuration,
        environment=configuration["environment"],
    )

    configuration_data = configuration["configuration"]

    class Replay:
        current = None
        configuration = asr.Qwen3ASRConfiguration.model_validate(configuration_data)

        def load(self, path, start, end):
            self.current = keys[(str(path), start, end)].pop(0)
            # Only a shape-valid token is needed: the worker already used the
            # frozen crop loader, and transcribe below replays its exact result.
            return np.zeros(1, dtype=np.float32), 16000

        def transcribe(self, *, waveform, sample_rate_hz):
            return _asr_result(_ready(results, self.current))

    replay = Replay()
    summary, published = frozen.run_stem_qwen3_asr_shadow(
        stem_diarization_root=root,
        source_visual_production_root=None,
        backend=replay,
        output_root=shadow / "asr",
        segment_audio_loader=replay.load,
        ffmpeg=ffmpeg,
        route="resolved",
        allow_unverified=allow_unverified,
        overwrite=True,
    )
    return {
        "summary": summary.model_dump(mode="json"),
        "provenance": published.model_dump(mode="json"),
        "job_count": len(jobs),
    }
