"""H3 reference audio: >2ch sources get a stereo copy; everything else passes through."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.person_replacement import h3_reference_audio as module


class FakeFFmpeg:
    """Stands in for ffprobe/ffmpeg; ffmpeg invocations materialize their output."""

    def __init__(self, channels=None, produced=None):
        self.channels = channels  # None means no audio stream
        self.produced = produced  # channels of the file ffmpeg would write
        self.commands = []

    def __call__(self, command, *, capture=False):
        self.commands.append(command)
        if command[0] == "ffprobe":
            path = Path(command[-1])
            if not path.exists():
                raise RuntimeError("missing media")
            # A pre-existing or freshly written normalized file is stereo; only the
            # original source has the multichannel track.
            channels = self.channels if path.name == "source.mp4" else self.produced
            if "a:0" in command:  # audio-only probe
                return json.dumps({"streams":[] if not channels else
                                   [{"codec_type":"audio","channels":channels}]})
            streams = [{"codec_type":"video","channels":0}]
            if channels:
                streams.append({"codec_type":"audio","channels":channels})
            return json.dumps({"streams":streams})
        Path(command[-1]).write_bytes(b"mp4")
        return ""


def install(monkeypatch, fake):
    monkeypatch.setattr(module,"_execute",fake)


def test_command_copies_video_and_downmixes_audio_only():
    command = module.stereo_reference_command(Path("/data/a.mp4"),Path("/out/b.mp4"))
    assert command[:5] == ["ffmpeg","-nostdin","-v","error","-y"]
    assert "-c:v" in command and command[command.index("-c:v")+1] == "copy"
    assert command[command.index("-ac")+1] == "2"
    assert command[command.index("-c:a")+1] == "aac"
    assert "0:v:0" in command and "0:a:0?" in command
    assert str(Path("/out/b.mp4")) == command[-1]


def test_temporary_output_keeps_the_mp4_extension(tmp_path):
    """ffmpeg infers the container from the final extension."""
    assert module.TEMPORARY_NAME.endswith(".mp4")
    command = module.stereo_reference_command(tmp_path/"a.mp4",tmp_path/module.TEMPORARY_NAME)
    assert command[-1].endswith(".mp4")
    assert not command[-1].endswith(".partial")


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe unavailable")
def test_real_ffmpeg_downmixes_six_channels_to_stereo(tmp_path):
    source = tmp_path/"source.mp4"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y",
                    "-f","lavfi","-i","color=c=blue:s=320x240:r=24:d=1",
                    "-f","lavfi","-i","sine=frequency=440:duration=1",
                    "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-ac","6",
                    "-shortest",str(source)],check=True,capture_output=True)
    assert module.probe_audio_channels(source) == 6
    before = source.read_bytes()
    reference, provenance = module.ensure_h3_reference(source,tmp_path/"preparation")
    assert reference == tmp_path/"preparation"/module.REFERENCE_NAME
    assert provenance["reference_audio_source_channels"] == 6
    assert provenance["reference_audio_target_channels"] == 2
    assert module.probe_audio_channels(reference) == 2
    assert module.validate_reference(reference) is True
    assert source.read_bytes() == before  # original dataset file untouched
    assert not (tmp_path/"preparation"/module.TEMPORARY_NAME).exists()


@pytest.mark.parametrize("channels", [None,1,2])
def test_mono_stereo_and_silent_sources_pass_through(tmp_path, monkeypatch, channels):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"mp4")
    install(monkeypatch,FakeFFmpeg(channels=channels))
    reference, provenance = module.ensure_h3_reference(source,tmp_path/"preparation")
    assert reference == source
    assert provenance == {"reference_audio_normalized":False,
                          "reference_audio_source_channels":channels,
                          "reference_audio_target_channels":None}
    assert not (tmp_path/"preparation"/module.REFERENCE_NAME).exists()


def test_six_channel_source_is_normalized_to_case_local_stereo(tmp_path, monkeypatch):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"mp4")
    fake = FakeFFmpeg(channels=6,produced=2)
    install(monkeypatch,fake)
    reference, provenance = module.ensure_h3_reference(source,tmp_path/"preparation")
    target = tmp_path/"preparation"/module.REFERENCE_NAME
    assert reference == target and target.is_file()
    assert provenance == {"reference_audio_normalized":True,"reference_audio_source_channels":6,
                          "reference_audio_target_channels":2}
    ffmpeg = [command for command in fake.commands if command[0] == "ffmpeg"]
    assert len(ffmpeg) == 1 and "copy" in ffmpeg[0]
    assert not (tmp_path/"preparation"/module.TEMPORARY_NAME).exists()
    assert source.read_bytes() == b"mp4"  # original dataset file untouched


def test_verified_reference_is_reused_on_resume(tmp_path, monkeypatch):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"mp4")
    target = tmp_path/"preparation"/module.REFERENCE_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(b"mp4")
    fake = FakeFFmpeg(channels=6,produced=2)
    install(monkeypatch,fake)
    reference, _ = module.ensure_h3_reference(source,tmp_path/"preparation")
    assert reference == target
    assert not [command for command in fake.commands if command[0] == "ffmpeg"]


def test_partial_and_invalid_references_are_not_used(tmp_path, monkeypatch):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"mp4")
    directory = tmp_path/"preparation"
    directory.mkdir(parents=True)
    (directory/module.TEMPORARY_NAME).write_bytes(b"partial")
    install(monkeypatch,FakeFFmpeg(channels=6,produced=6))  # downmix validation fails
    with pytest.raises(ValueError,match="failed validation"):
        module.ensure_h3_reference(source,directory)
    assert not (directory/module.TEMPORARY_NAME).exists()
    assert not (directory/module.REFERENCE_NAME).exists()


def test_probe_reports_missing_audio_stream(tmp_path, monkeypatch):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"mp4")
    install(monkeypatch,FakeFFmpeg(channels=None))
    assert module.probe_audio_channels(source) is None
    assert module.validate_reference(source) is True
    assert not (tmp_path/"anything").exists()


def test_distributed_prepare_uses_the_normalized_reference(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.h3_pair_generation import h3_reference_source

    job = {"source":"/data/original.mp4","h3_reference_source":"/case/h3_reference_stereo.mp4"}
    assert h3_reference_source(job) == "/case/h3_reference_stereo.mp4"
    assert h3_reference_source({"source":"/data/original.mp4"}) == "/data/original.mp4"

    from r2v_data_v2.person_replacement.h3_pdd_distributed import PersistentPDD

    used = []

    def reference(kind):
        return lambda path:(kind,path)

    modules = __import__("sys").modules
    monkeypatch.setitem(modules,"diffusers",SimpleNamespace())
    monkeypatch.setitem(modules,"diffusers.modular_pipelines",SimpleNamespace())
    monkeypatch.setitem(modules,"diffusers.modular_pipelines.minimax_h3",
                        SimpleNamespace(MiniMaxH3VideoReference=SimpleNamespace(
                            from_file=lambda path:used.append(path) or ("video",path)),
                                        MiniMaxH3ImageReference=SimpleNamespace(
                            from_file=lambda path:("image",path))))
    backend = PersistentPDD.__new__(PersistentPDD)
    backend.prepare({"source":"/data/original.mp4","h3_reference_source":"/case/stereo.mp4"})
    assert used == ["/case/stereo.mp4"]
