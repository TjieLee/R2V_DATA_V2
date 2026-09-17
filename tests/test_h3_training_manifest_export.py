import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.training_manifest_export import TASK_ORDER, export_training_manifests


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_reference_products_split_visual_and_audio_variants(tmp_path: Path):
    shadow = tmp_path / "shadow"
    prepared = shadow / "audio_reuse_prepared_v1"
    products = shadow / "h3_audio_reuse_products_v1"
    video = "/data/clip-a.mp4"
    images = ["/data/p1.png", "/data/p2.png"]
    speech = "/data/S1.flac"
    music = "/data/music.flac"
    full = "/data/full.flac"

    write_json(
        prepared / "inventory.json",
        {
            "jobs": [
                {
                    "clip_uid": "clip-a",
                    "target_video_path": video,
                    "reference_images": [
                        {"image_index": 2, "image_artifact_path": images[1]},
                        {"image_index": 1, "image_artifact_path": images[0]},
                    ],
                }
            ]
        },
    )
    write_jsonl(
        products / "records.jsonl",
        [
            {
                "clip_uid": "clip-a",
                "status": "ready",
                "conditioning_variant": "visual_only",
                "audio_references": [],
                "rendered_h3_prompt": "reference visual caption",
            },
            {
                "clip_uid": "clip-a",
                "status": "ready",
                "conditioning_variant": "target_speech_reuse",
                "audio_references": [
                    {"contract": {"path": speech}},
                    {"contract": {"path": music}},
                ],
                "rendered_h3_prompt": "reference speech+bgm caption",
            },
            {
                "clip_uid": "clip-a",
                "status": "ready",
                "conditioning_variant": "full_audio_reuse",
                "audio_references": [{"contract": {"path": full}}],
                "rendered_h3_prompt": "reference full caption",
            },
            {
                "clip_uid": "clip-a",
                "status": "ready",
                "conditioning_variant": "cross_voice_reference",
                "audio_references": [{"contract": {"path": "/data/donor.flac"}}],
                "rendered_h3_prompt": "not exported here",
            },
        ],
    )

    output = tmp_path / "export"
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow)

    assert read_jsonl(output / "r2va_reference.jsonl") == [
        {"video": video, "images": images, "audios": [], "caption": "reference visual caption"}
    ]
    assert read_jsonl(output / "ra2va_reference_speech_bgm.jsonl") == [
        {
            "video": video,
            "images": images,
            "audios": [speech, music],
            "caption": "reference speech+bgm caption",
        }
    ]
    assert read_jsonl(output / "ra2va_reference_full_audio.jsonl") == [
        {
            "video": video,
            "images": images,
            "audios": [full],
            "caption": "reference full caption",
        }
    ]


def test_frame_products_split_visual_modes_and_audio_variants(tmp_path: Path):
    shadow = tmp_path / "shadow"
    prepared = shadow / "audio_reuse_prepared_v1"
    products = shadow / "h3_audio_reuse_products_v1"
    frames = shadow / "h3_frame_conditioned_products_v1"
    video = "/data/clip-a.mp4"
    write_json(
        prepared / "inventory.json",
        {"jobs": [{"clip_uid": "clip-a", "target_video_path": video, "reference_images": []}]},
    )
    write_jsonl(products / "records.jsonl", [])

    def frame_row(mode, variant, images, audios, caption):
        return {
            "clip_uid": "clip-a",
            "target_video_path": video,
            "visual_reference_mode": mode,
            "conditioning_variant": variant,
            "frame_references": [
                {"picture_index": index + 1, "image_path": path}
                for index, path in enumerate(images)
            ],
            "audio_references": [
                {"contract": {"path": path}} for path in audios
            ],
            "rendered_h3_prompt": caption,
        }

    write_jsonl(
        frames / "records.jsonl",
        [
            frame_row("first_frame", "visual_only", ["/frames/first.png"], [], "first visual"),
            frame_row("last_frame", "visual_only", ["/frames/last.png"], [], "last visual"),
            frame_row(
                "first_last_frame",
                "visual_only",
                ["/frames/first.png", "/frames/last.png"],
                [],
                "first last visual",
            ),
            frame_row(
                "first_frame",
                "full_audio_reuse",
                ["/frames/first.png"],
                ["/data/full.flac"],
                "first full",
            ),
            frame_row(
                "last_frame",
                "target_speech_reuse",
                ["/frames/last.png"],
                ["/data/S1.flac", "/data/music.flac"],
                "last speech bgm",
            ),
            frame_row(
                "first_last_frame",
                "target_speech_reuse",
                ["/frames/first.png", "/frames/last.png"],
                ["/data/S1.flac"],
                "first last speech",
            ),
        ],
    )

    output = tmp_path / "export"
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow)

    assert read_jsonl(output / "r2va_first_frame.jsonl")[0]["images"] == ["/frames/first.png"]
    assert read_jsonl(output / "r2va_last_frame.jsonl")[0]["caption"] == "last visual"
    assert read_jsonl(output / "r2va_first_last_frame.jsonl")[0]["images"] == [
        "/frames/first.png",
        "/frames/last.png",
    ]
    assert read_jsonl(output / "ra2va_first_frame_full_audio.jsonl")[0]["audios"] == [
        "/data/full.flac"
    ]
    assert read_jsonl(output / "ra2va_last_frame_speech_bgm.jsonl")[0]["audios"] == [
        "/data/S1.flac",
        "/data/music.flac",
    ]
    assert read_jsonl(output / "ra2va_first_last_frame_speech_bgm.jsonl")[0]["caption"] == (
        "first last speech"
    )


def test_ta2va_splits_full_audio_and_speech_bgm(tmp_path: Path):
    t2va = tmp_path / "t2va"
    ta2va = tmp_path / "ta2va"
    video = "/data/clip-b.mp4"

    write_json(t2va / "inventory.json", {"jobs": [{"clip_uid": "clip-b", "target_video_path": video}]})
    write_jsonl(t2va / "records.jsonl", [{"clip_uid": "clip-b", "status": "ready"}])
    (t2va / "prompts").mkdir(parents=True)
    (t2va / "prompts" / "clip-b.txt").write_text("t2va caption\n", encoding="utf-8")

    write_json(ta2va / "inventory.json", {"source_t2va_root": str(t2va)})
    write_jsonl(
        ta2va / "records.jsonl",
        [
            {
                "clip_uid": "clip-b",
                "variant": "full_audio_reuse",
                "status": "ready",
                "audio_references": [{"path": "/data/full.flac"}],
                "prompt": "ta2va full caption",
            },
            {
                "clip_uid": "clip-b",
                "variant": "target_speech_reuse",
                "status": "ready",
                "audio_references": [
                    {"path": "/data/S1.flac"},
                    {"path": "/data/music.flac"},
                ],
                "prompt": "ta2va speech caption",
            },
        ],
    )

    output = tmp_path / "export"
    export_training_manifests(output_root=output, t2va_root=t2va, ta2va_root=ta2va)

    assert read_jsonl(output / "t2va.jsonl") == [
        {"video": video, "images": [], "audios": [], "caption": "t2va caption\n"}
    ]
    assert read_jsonl(output / "ta2va_full_audio.jsonl") == [
        {"video": video, "images": [], "audios": ["/data/full.flac"], "caption": "ta2va full caption"}
    ]
    assert read_jsonl(output / "ta2va_speech_bgm.jsonl") == [
        {
            "video": video,
            "images": [],
            "audios": ["/data/S1.flac", "/data/music.flac"],
            "caption": "ta2va speech caption",
        }
    ]


def test_video_summary_lists_fine_grained_tasks(tmp_path: Path):
    shadow = tmp_path / "shadow"
    t2va = tmp_path / "t2va"
    video = "/data/shared.mp4"
    write_json(
        shadow / "audio_reuse_prepared_v1" / "inventory.json",
        {"jobs": [{"clip_uid": "shared", "target_video_path": video, "reference_images": []}]},
    )
    write_jsonl(
        shadow / "h3_audio_reuse_products_v1" / "records.jsonl",
        [
            {
                "clip_uid": "shared",
                "status": "ready",
                "conditioning_variant": "visual_only",
                "audio_references": [],
                "rendered_h3_prompt": "r",
            },
            {
                "clip_uid": "shared",
                "status": "ready",
                "conditioning_variant": "full_audio_reuse",
                "audio_references": [{"contract": {"path": "/data/a.flac"}}],
                "rendered_h3_prompt": "ra",
            },
        ],
    )
    write_json(t2va / "inventory.json", {"jobs": [{"clip_uid": "shared", "target_video_path": video}]})
    write_jsonl(t2va / "records.jsonl", [{"clip_uid": "shared", "status": "ready"}])
    (t2va / "prompts").mkdir(parents=True)
    (t2va / "prompts" / "shared.txt").write_text("t", encoding="utf-8")

    output = tmp_path / "export"
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow, t2va_root=t2va)

    assert read_jsonl(output / "videos.jsonl") == [
        {
            "video": video,
            "tasks": ["r2va_reference", "ra2va_reference_full_audio", "t2va"],
        }
    ]
    assert set(path.name for path in output.glob("*.jsonl")) == {
        "videos.jsonl",
        *(f"{task}.jsonl" for task in TASK_ORDER),
    }


def test_rejects_existing_output_directory(tmp_path: Path):
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_training_manifests(output_root=output)


def test_cli_exports_all_empty_task_files_with_only_output_root(tmp_path: Path):
    import os
    import subprocess
    import sys

    output = tmp_path / "cli-export"
    script = Path(__file__).resolve().parents[1] / "tools" / "export_h3_training_manifests.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--output-root", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert read_jsonl(output / "videos.jsonl") == []
    for task in TASK_ORDER:
        assert read_jsonl(output / f"{task}.jsonl") == []
