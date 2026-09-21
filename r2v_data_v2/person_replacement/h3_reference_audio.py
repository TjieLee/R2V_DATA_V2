"""Case-local stereo H3 reference copy for multichannel source audio.

H3 reference preparation rejects soundtracks with more than two channels, so a
visually valid clip with 5.1/6-channel audio would be lost. This helper builds a
case-local copy whose video stream is stream-copied and whose first audio stream
is downmixed to stereo. The original dataset file is never modified, and the
result is only an H3 conditioning artifact, never the training target.
"""

import json
import os
import subprocess
from pathlib import Path

REFERENCE_NAME = "h3_reference_stereo.mp4"
# ffmpeg infers the container from the final extension, so the temporary must
# still end in .mp4; ".mp4.partial" would fail with "Unable to find a suitable
# output format".
TEMPORARY_NAME = ".h3_reference_stereo.partial.mp4"
TARGET_CHANNELS = 2
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def _execute(command, *, capture=False):
    result = subprocess.run(command, check=True, capture_output=capture, text=True)
    return result.stdout if capture else ""


def _probe_json(path, entries):
    output = _execute([FFPROBE, "-v", "error", "-show_entries", entries, "-of", "json",
                       str(path)], capture=True)
    return json.loads(output)


def probe_audio_channels(path):
    """None means no audio stream; H3 accepts that unchanged."""
    output = _execute([FFPROBE, "-v", "error", "-select_streams", "a:0",
                       "-show_entries", "stream=channels", "-of", "json", str(path)], capture=True)
    streams = json.loads(output).get("streams", [])
    if not streams:
        return None
    channels = int(streams[0]["channels"])
    if channels < 1:
        raise ValueError(f"Invalid audio channel count for H3 reference: {path}")
    return channels


def stereo_reference_command(source, output):
    """Video is copied, never re-encoded; only the first audio track is downmixed."""
    return [FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac",
            "-ac", str(TARGET_CHANNELS), "-movflags", "+faststart", str(output)]


def validate_reference(path):
    """A usable reference has one video stream and at most stereo audio."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    streams = _probe_json(path, "stream=codec_type,channels").get("streams", [])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) > 1:
        return False
    return all(int(stream["channels"]) <= TARGET_CHANNELS for stream in audio)


def _provenance(channels, normalized):
    return {"reference_audio_normalized":normalized,
            "reference_audio_source_channels":channels,
            "reference_audio_target_channels":TARGET_CHANNELS if normalized else None}


def ensure_h3_reference(source, directory):
    """Return the path H3 must use plus provenance; mono/stereo/no-audio pass through."""
    source = Path(source)
    channels = probe_audio_channels(source)
    if channels is None or channels <= TARGET_CHANNELS:
        return source,_provenance(channels,False)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/REFERENCE_NAME
    if validate_reference(target):  # resume reuses a verified copy
        return target,_provenance(channels,True)
    temporary = directory/TEMPORARY_NAME
    temporary.unlink(missing_ok=True)  # a partial file from an interrupted run
    _execute(stereo_reference_command(source,temporary))
    if not validate_reference(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("Normalized H3 reference failed validation")
    os.replace(temporary,target)  # publish only after validation
    return target,_provenance(channels,True)
