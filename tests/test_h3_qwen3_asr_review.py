from __future__ import annotations

import builtins
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import qwen3_asr_review as review_module
from r2v_data_v2.h3.jea_audio_production import JEAInPair
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration, Qwen3ASRSegment
from r2v_data_v2.h3.qwen3_asr_review import (
    FFmpegQwen3ASRReviewMediaBackend,
    generate_qwen3_asr_review,
)
from tools import run_h3_qwen3_asr_review as review_cli


class _FakeMediaBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def transcode_video(
        self,
        *,
        source_video_path: Path,
        destination_path: Path,
    ) -> None:
        self.calls.append((source_video_path, destination_path))
        destination_path.write_bytes(b"proxy:" + source_video_path.read_bytes())


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _pair(root: Path, clip_uid: str, display: str) -> JEAInPair:
    video = root / "source" / f"{clip_uid}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f"video-{clip_uid}".encode())
    return JEAInPair(
        pair_id=f"in_pair/{clip_uid}",
        target_clip_uid=clip_uid,
        target_clip_display_path=display,
        media_collection_relpath="show/collection",
        media_collection_name="collection",
        episode_name="episode",
        clip_name=clip_uid,
        shard_id=f"shard-{clip_uid}",
        target_video_path=str(video),
        target_full_audio_path=str(root / "audio" / f"{clip_uid}.flac"),
        target_audio_binding_path=str(root / "audio" / f"{clip_uid}.json"),
        subjects=[],
    )


def _segment(
    *,
    clip_uid: str,
    display: str,
    index: int,
    status: str = "transcribed",
) -> Qwen3ASRSegment:
    start_sample = index * 1600
    common = {
        "clip_uid": clip_uid,
        "clip_display_path": display,
        "media_collection_relpath": "show/collection",
        "media_collection_name": "collection",
        "episode_name": "episode",
        "clip_name": clip_uid,
        "shard_id": f"shard-{clip_uid}",
        "segment_id": f"segment_{index:04d}",
        "speaker_cluster_id": f"speaker_{index % 3}",
        "entity_id": "e1" if index % 2 == 0 else None,
        "entity_occurrence_id": f"{clip_uid}/e1" if index % 2 == 0 else None,
        "source_audio_path": f"/audio/{clip_uid}.wav",
        "source_start_sample": start_sample,
        "source_end_sample": start_sample + 1600,
        "source_sample_rate_hz": 16000,
        "start_time": index / 10,
        "end_time": index / 10 + 0.1,
        "status": status,
        "configuration": Qwen3ASRConfiguration(local_model_path="/local/qwen3"),
    }
    if status == "transcribed":
        return Qwen3ASRSegment(
            **common,
            text=f"text {clip_uid} {index}",
            language="Chinese",
        )
    if status == "empty":
        return Qwen3ASRSegment(**common)
    return Qwen3ASRSegment(**common, failure_reason="backend unavailable")


def _production_fixture(
    tmp_path: Path,
    *,
    segment_count: int = 92,
) -> tuple[Path, list[JEAInPair], list[Qwen3ASRSegment]]:
    root = tmp_path / "production"
    pairs = [
        _pair(root, "clip-b", "show/collection/episode_b"),
        _pair(root, "clip-a", "show/collection/episode_a"),
    ]
    rows = []
    for index in range(segment_count):
        clip_uid = "clip-a" if index % 2 == 0 else "clip-b"
        display = (
            "show/collection/episode_a"
            if clip_uid == "clip-a"
            else "show/collection/episode_b"
        )
        status = "transcribed"
        if index == segment_count - 2:
            status = "empty"
        elif index == segment_count - 1:
            status = "failed"
        rows.append(
            _segment(
                clip_uid=clip_uid,
                display=display,
                index=index,
                status=status,
            )
        )
    _write_jsonl(root / "pairs/in_pairs.jsonl", list(reversed(pairs)))
    _write_jsonl(root / "asr/segments.jsonl", list(reversed(rows)))
    return root, pairs, rows


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _review_data(review_html: str) -> dict[str, object]:
    prefix = "const reviewData = "
    start = review_html.index(prefix) + len(prefix)
    end = review_html.index(";\nconst allowedLabels", start)
    return json.loads(review_html[start:end])


def test_review_renders_all_rows_in_deterministic_order_without_source_changes(
    tmp_path: Path,
) -> None:
    root, pairs, _ = _production_fixture(tmp_path)
    before = _tree_bytes(root / "asr") | {
        f"pairs/{key}": value for key, value in _tree_bytes(root / "pairs").items()
    }
    backend = _FakeMediaBackend()

    manifest = generate_qwen3_asr_review(
        audio_production_root=root,
        media_backend=backend,
    )

    after = _tree_bytes(root / "asr") | {
        f"pairs/{key}": value for key, value in _tree_bytes(root / "pairs").items()
    }
    review = (root / "asr_review/review.html").read_text(encoding="utf-8")
    data = _review_data(review)
    assert before == after
    assert manifest.segment_count == 92
    assert manifest.transcribed_count == 90
    assert manifest.empty_count == 1
    assert manifest.failed_count == 1
    assert review.count("class='segment-row'") == 92
    assert "[EMPTY OUTPUT]" in review
    assert "backend unavailable" in review
    assert [item.clip_uid for item in manifest.clips] == ["clip-a", "clip-b"]
    for clip in manifest.clips:
        assert clip.segment_ids == sorted(
            clip.segment_ids,
            key=lambda value: int(value.removeprefix("segment_")),
        )
    pair_by_clip = {item.target_clip_uid: item for item in pairs}
    assert {
        item.clip_uid: item.target_video_path for item in manifest.clips
    } == {
        clip_uid: pair.target_video_path for clip_uid, pair in pair_by_clip.items()
    }
    assert {source for source, _ in backend.calls} == {
        Path(item.target_video_path).resolve() for item in pairs
    }
    assert len(data["rows"]) == 92
    first = data["rows"][0]
    assert first["clip_uid"] == "clip-a"
    assert first["start_time"] == 0.0
    assert first["end_time"] == 0.1
    assert {
        "clip_uid",
        "clip_display_path",
        "segment_id",
        "speaker_cluster_id",
        "entity_id",
        "start_time",
        "end_time",
        "status",
        "text",
        "language",
        "failure_reason",
    }.issubset(first)
    assert "video.currentTime=row.start_time" in review
    assert "stopAt=row.end_time" in review
    assert "Export QA JSON" in review
    assert "review_label" in review and "review_note" in review
    assert "JSON.stringify(payload,null,2)+'\\n'" in review


def test_generated_review_script_has_valid_javascript_syntax(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for JavaScript syntax validation")
    root, _, _ = _production_fixture(tmp_path, segment_count=2)
    generate_qwen3_asr_review(
        audio_production_root=root,
        media_backend=_FakeMediaBackend(),
    )
    review = (root / "asr_review/review.html").read_text(encoding="utf-8")
    script_match = re.search(r"<script>(.*?)</script>", review, flags=re.DOTALL)
    assert script_match is not None
    script_path = tmp_path / "review.js"
    script_path.write_text(script_match.group(1), encoding="utf-8")

    result = subprocess.run(
        [node, "--check", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_review_fails_closed_for_missing_video_or_duplicate_segment(
    tmp_path: Path,
) -> None:
    missing_root, pairs, rows = _production_fixture(tmp_path / "missing", segment_count=2)
    Path(pairs[0].target_video_path).unlink()
    with pytest.raises(FileNotFoundError, match="target video is missing"):
        generate_qwen3_asr_review(
            audio_production_root=missing_root,
            media_backend=_FakeMediaBackend(),
        )

    duplicate_root, _, duplicate_rows = _production_fixture(
        tmp_path / "duplicate",
        segment_count=2,
    )
    _write_jsonl(
        duplicate_root / "asr/segments.jsonl",
        duplicate_rows + [duplicate_rows[0]],
    )
    with pytest.raises(ValueError, match="duplicate Qwen3 ASR segment identity"):
        generate_qwen3_asr_review(
            audio_production_root=duplicate_root,
            media_backend=_FakeMediaBackend(),
        )
    assert rows


def test_review_generation_never_imports_external_qwen_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _ = _production_fixture(tmp_path, segment_count=2)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "qwen_asr" or name.startswith("qwen_asr."):
            raise AssertionError("review must not import qwen_asr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = generate_qwen3_asr_review(
        audio_production_root=root,
        media_backend=_FakeMediaBackend(),
    )

    assert manifest.segment_count == 2


def test_review_output_cannot_replace_production_stage(tmp_path: Path) -> None:
    root, _, _ = _production_fixture(tmp_path, segment_count=2)

    with pytest.raises(ValueError, match="cannot replace production stages"):
        generate_qwen3_asr_review(
            audio_production_root=root,
            output_root=root / "asr",
            media_backend=_FakeMediaBackend(),
            overwrite=True,
        )


def test_review_cli_uses_fixed_default_output_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _ = _production_fixture(tmp_path, segment_count=2)
    backend = _FakeMediaBackend()
    monkeypatch.setattr(
        review_cli,
        "FFmpegQwen3ASRReviewMediaBackend",
        lambda **_: backend,
    )

    result = review_cli.main(["--audio-production-root", str(root)])

    assert result["model_loaded"] is False
    assert result["model_calls"] == 0
    assert result["review_output_root"] == str((root / "asr_review").resolve())
    assert (root / "asr_review/review.html").is_file()


def test_ffmpeg_review_proxy_keeps_video_audio_and_full_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "review.mp4"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        Path(command[-1]).write_bytes(b"proxy")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(review_module.subprocess, "run", fake_run)
    FFmpegQwen3ASRReviewMediaBackend().transcode_video(
        source_video_path=source,
        destination_path=destination,
    )

    command = commands[0]
    assert ["-map", "0:v:0"] == command[command.index("0:v:0") - 1 : command.index("0:v:0") + 1]
    assert ["-map", "0:a:0"] == command[command.index("0:a:0") - 1 : command.index("0:a:0") + 1]
    assert "libx264" in command and "yuv420p" in command and "aac" in command
    assert "-ss" not in command and "-t" not in command
