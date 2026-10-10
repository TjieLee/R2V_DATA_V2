"""Group-native DiariZen bridge; reuse canonical preprocessing, no weight scans."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

from diarizen_worker import _all_inactive_reconstruct_guard, _prepare_analysis_audio


def run_worker(input_stream, output_stream):
    def emit(value):
        output_stream.write(json.dumps(value) + "\n")
        output_stream.flush()

    settings = json.loads(input_stream.readline())
    sys.path.insert(0, settings["code_root"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    import torchaudio
    from diarizen.pipelines.inference import DiariZenPipeline

    with contextlib.redirect_stdout(sys.stderr):
        pipeline = DiariZenPipeline.from_pretrained(settings["model_identifier"], cache_dir=settings["model_cache"])
        pipeline.to(torch.device("cuda:0"))
    emit({"status": "ready", "request_id": "startup"})
    for line in input_stream:
        request = json.loads(line)
        if request.get("operation") == "shutdown":
            return
        try:
            with tempfile.TemporaryDirectory(prefix="ra2va-diarizen-") as work:
                source = _prepare_analysis_audio(source=Path(request["audio_path"]),
                    destination=Path(work) / "analysis.wav", torch=torch, torchaudio=torchaudio,
                    input_profile="canonical_32k_stereo")
                with (_all_inactive_reconstruct_guard(pipeline, numpy=np), contextlib.redirect_stdout(sys.stderr)):
                    result = pipeline(str(source), sess_name=request["request_id"])
                segments = [{"start_time": float(turn.start), "end_time": float(turn.end), "speaker_label": str(speaker)}
                            for turn, _, speaker in result.itertracks(yield_label=True)]
            emit({"status": "ready", "request_id": request["request_id"], "segments": segments})
        except Exception as error:  # noqa: BLE001 - preserve per-clip business failures
            emit({"status": "failed", "request_id": request["request_id"],
                  "reason": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    run_worker(sys.stdin, sys.stdout)
