from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-uid", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import torch
    from silero_vad import get_speech_timestamps, read_audio

    audio_path = Path(args.audio).expanduser().resolve(strict=True)
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    waveform = read_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=16000,
        return_seconds=True,
    )
    intervals = [
        {
            "start_time": float(item["start"]),
            "end_time": min(float(item["end"]), args.duration_seconds),
            "confidence": None,
        }
        for item in timestamps
        if min(float(item["end"]), args.duration_seconds) > float(item["start"])
    ]
    payload = {
        "schema_version": "r2v.h3.speech_activity.1",
        "clip_uid": args.clip_uid,
        "backend": "silero_vad",
        "model_identifier": model_path.name,
        "source_audio_path": str(audio_path),
        "duration_seconds": args.duration_seconds,
        "intervals": intervals,
    }
    output = Path(args.output).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
