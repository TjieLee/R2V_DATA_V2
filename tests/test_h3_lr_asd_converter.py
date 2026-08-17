from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from tools import convert_lr_asd_native as converter


def _convert_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int,
    bbox_count: int,
    score_count: int,
) -> dict[str, object]:
    vendor_output = tmp_path / "vendor-output"
    model_video = vendor_output / "pyavi" / "video.avi"
    audio = vendor_output / "pyavi" / "audio.wav"
    model_video.parent.mkdir(parents=True)
    model_video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    source_video = tmp_path / "source.mp4"
    model_path = tmp_path / "model.pth"
    source_video.write_bytes(b"source")
    model_path.write_bytes(b"model")
    bbox = [1.0, 2.0, 11.0, 12.0]
    artifacts = {
        "tracks.pckl": [
            {
                "track": {
                    "frame": list(range(frame_count)),
                    "bbox": [bbox] * bbox_count,
                }
            }
        ],
        "scores.pckl": [[0.25] * score_count],
        "faces.pckl": [
            [{"bbox": bbox, "conf": 0.9}] for _ in range(frame_count)
        ],
    }
    monkeypatch.setattr(converter, "_load_pickle", lambda path: artifacts[path.name])
    monkeypatch.setattr(
        converter,
        "_video_metadata",
        lambda path, *, model_fps: (100, 80, frame_count / model_fps),
    )
    return converter.convert(
        Namespace(
            vendor_output=vendor_output,
            clip_uid="clip-1",
            source_video=source_video,
            model_path=model_path,
            checkpoint_sha256="a" * 64,
        )
    )


@pytest.mark.parametrize(
    ("score_count", "expected_sample_count"),
    [(250, 250), (249, 249)],
)
def test_converter_accepts_full_or_one_short_score_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score_count: int,
    expected_sample_count: int,
) -> None:
    payload = _convert_fixture(
        tmp_path,
        monkeypatch,
        frame_count=250,
        bbox_count=250,
        score_count=score_count,
    )

    tracks = payload["tracks"]
    assert isinstance(tracks, list)
    samples = tracks[0]["samples"]
    assert len(samples) == expected_sample_count
    assert samples[-1]["frame_index"] == expected_sample_count - 1


@pytest.mark.parametrize(
    ("bbox_count", "score_count", "message"),
    [
        (249, 250, "frames and boxes differ"),
        (250, 248, "frames and scores differ"),
        (250, 251, "frames and scores differ"),
    ],
)
def test_converter_rejects_other_track_length_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bbox_count: int,
    score_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=250,
            bbox_count=bbox_count,
            score_count=score_count,
        )
