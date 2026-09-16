import json
from pathlib import Path

from r2v_data_v2.h3.training_manifest_export import export_training_manifests


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ra2va_export_reads_real_audio_reuse_prepared_stage(tmp_path: Path):
    shadow = tmp_path / "shadow"
    _write_json(
        shadow / "audio_reuse_prepared_v1" / "inventory.json",
        {
            "jobs": [
                {
                    "clip_uid": "clip-a",
                    "target_video_path": "/data/clip-a.mp4",
                    "reference_images": [
                        {"image_index": 1, "image_artifact_path": "/data/ref.png"}
                    ],
                }
            ]
        },
    )
    _write_jsonl(
        shadow / "h3_audio_reuse_products_v1" / "records.jsonl",
        [
            {
                "clip_uid": "clip-a",
                "status": "ready",
                "conditioning_variant": "visual_only",
                "audio_references": [],
                "rendered_h3_prompt": "caption",
            }
        ],
    )

    output = tmp_path / "export"
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow)

    assert _read_jsonl(output / "r2va_reference.jsonl") == [
        {
            "video": "/data/clip-a.mp4",
            "images": ["/data/ref.png"],
            "audios": [],
            "caption": "caption",
        }
    ]
