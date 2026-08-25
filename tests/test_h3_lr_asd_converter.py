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
    bbox: list[float] | None = None,
    detection_bbox: list[float] | None = None,
    video_size: tuple[int, int] = (100, 80),
    frame_indices: list[int] | None = None,
    track_bboxes: list[list[float]] | None = None,
    score_values: list[float] | None = None,
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
    track_bbox = bbox or [1.0, 2.0, 11.0, 12.0]
    face_bbox = detection_bbox or track_bbox
    frames = frame_indices if frame_indices is not None else list(range(frame_count))
    bboxes = track_bboxes if track_bboxes is not None else [track_bbox] * bbox_count
    logits = score_values if score_values is not None else [0.25] * score_count
    artifacts = {
        "tracks.pckl": [
            {
                "track": {
                    "frame": frames,
                    "bbox": bboxes,
                }
            }
        ],
        "scores.pckl": [logits],
        "faces.pckl": [
            [{"bbox": face_bbox, "conf": 0.9}] for _ in range(frame_count)
        ],
    }
    monkeypatch.setattr(converter, "_load_pickle", lambda path: artifacts[path.name])
    monkeypatch.setattr(
        converter,
        "_video_metadata",
        lambda path, *, model_fps: (*video_size, frame_count / model_fps),
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


def test_converter_accepts_shorter_scored_prefix_without_fabricating_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [10, 11, 12, 13, 14, 15]
    bboxes = [
        [float(index), 2.0, float(index + 10), 12.0]
        for index in range(1, 7)
    ]
    logits = [0.1, 0.2, 0.3]

    payload = _convert_fixture(
        tmp_path,
        monkeypatch,
        frame_count=6,
        bbox_count=6,
        score_count=3,
        frame_indices=frames,
        track_bboxes=bboxes,
        score_values=logits,
    )

    samples = payload["tracks"][0]["samples"]
    assert [sample["frame_index"] for sample in samples] == frames[:3]
    assert [sample["bbox_xyxy"] for sample in samples] == bboxes[:3]
    assert [sample["raw_class1_logit"] for sample in samples] == logits


def test_converter_rejects_frame_and_bbox_length_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="frames and boxes differ"):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=250,
            bbox_count=249,
            score_count=250,
        )


def test_converter_rejects_more_scores_than_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="more scores than tracked frames"):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=250,
            bbox_count=250,
            score_count=251,
        )


def test_converter_rejects_empty_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="track has no scores"):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=250,
            bbox_count=250,
            score_count=0,
            score_values=[],
        )


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([1.0, 2.0, 11.0, 12.0], [1.0, 2.0, 11.0, 12.0]),
        ([1.0, -2.0, 11.0, 12.0], [1.0, 0.0, 11.0, 12.0]),
        ([1.0, 2.0, 11.0, 82.0], [1.0, 2.0, 11.0, 80.0]),
        ([-2.0, 2.0, 102.0, 12.0], [0.0, 2.0, 100.0, 12.0]),
    ],
)
def test_converter_clips_only_published_bbox_to_model_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bbox: list[float],
    expected: list[float],
) -> None:
    payload = _convert_fixture(
        tmp_path,
        monkeypatch,
        frame_count=1,
        bbox_count=1,
        score_count=1,
        bbox=bbox,
    )

    assert payload["tracks"][0]["samples"][0]["bbox_xyxy"] == expected


@pytest.mark.parametrize(
    "bbox",
    [
        [5.0, 2.0, 5.0, 12.0],
        [6.0, 2.0, 5.0, 12.0],
        [1.0, 7.0, 11.0, 7.0],
        [1.0, 8.0, 11.0, 7.0],
        [1.0, float("inf"), 11.0, 12.0],
    ],
)
def test_converter_rejects_invalid_raw_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bbox: list[float],
) -> None:
    with pytest.raises(ValueError, match="invalid|degenerate"):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=1,
            bbox_count=1,
            score_count=1,
            bbox=bbox,
        )


def test_converter_rejects_bbox_that_clips_to_zero_area(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="outside the model video"):
        _convert_fixture(
            tmp_path,
            monkeypatch,
            frame_count=1,
            bbox_count=1,
            score_count=1,
            bbox=[-20.0, 2.0, -1.0, 12.0],
        )


def test_detection_confidence_matching_uses_raw_vendor_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_bbox = [-50.0, 2.0, 10.0, 12.0]
    payload = _convert_fixture(
        tmp_path,
        monkeypatch,
        frame_count=1,
        bbox_count=1,
        score_count=1,
        bbox=raw_bbox,
        detection_bbox=raw_bbox,
    )

    sample = payload["tracks"][0]["samples"][0]
    assert sample["bbox_xyxy"] == [0.0, 2.0, 10.0, 12.0]
    assert sample["detection_confidence"] == 0.9
