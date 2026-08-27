from __future__ import annotations

import argparse
import html
import json
import shutil
import uuid
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a model-free LR-ASD versus LASER shadow review",
    )
    parser.add_argument("--lr-asd-root", type=Path, required=True)
    parser.add_argument("--laser-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-id", action="append")
    return parser


def _review_clip_ids(
    lr_root: Path,
    laser_root: Path,
    explicit: list[str] | None,
) -> list[str]:
    common = {
        path.name for path in (lr_root / "review").iterdir() if path.is_dir()
    } & {path.name for path in (laser_root / "review").iterdir() if path.is_dir()}
    if explicit is None:
        return sorted(common)
    clip_ids = list(dict.fromkeys(explicit))
    missing = sorted(set(clip_ids) - common)
    if missing:
        raise ValueError(f"comparison clips are missing from one backend: {missing}")
    return clip_ids


def build_comparison_review(
    *,
    lr_asd_root: Path,
    laser_root: Path,
    output_root: Path,
    clip_ids: list[str] | None = None,
) -> dict[str, object]:
    lr_root = lr_asd_root.expanduser().resolve(strict=True)
    laser = laser_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    if any(output == root or root in output.parents or output in root.parents for root in (lr_root, laser)):
        raise ValueError("comparison output must be separate from both source roots")
    selected = _review_clip_ids(lr_root, laser, clip_ids)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        media_root = temporary / "media"
        media_root.mkdir(parents=True)
        rows = []
        for clip_uid in selected:
            lr_review = lr_root / "review" / clip_uid
            laser_review = laser / "review" / clip_uid
            source_name = f"{clip_uid}-source.mp4"
            shutil.copyfile(lr_review / "source.mp4", media_root / source_name)
            backend_media: dict[str, str | None] = {}
            for backend, review in (("lr_asd", lr_review), ("laser", laser_review)):
                source_visualization = review / "visualization.mp4"
                if source_visualization.is_file():
                    name = f"{clip_uid}-{backend}.mp4"
                    shutil.copyfile(source_visualization, media_root / name)
                    backend_media[backend] = f"media/{name}"
                else:
                    backend_media[backend] = None
            rows.append(
                {
                    "clip_uid": clip_uid,
                    "source_video": f"media/{source_name}",
                    "lr_asd_visualization": backend_media["lr_asd"],
                    "laser_visualization": backend_media["laser"],
                    "lr_asd_binding": json.loads(
                        (lr_review / "audio_binding.json").read_text(encoding="utf-8")
                    ),
                    "laser_binding": json.loads(
                        (laser_review / "audio_binding.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                }
            )
        summary = {
            "schema_version": "r2v.h3.asd_backend_comparison_review.1",
            "clip_count": len(rows),
            "clip_ids": selected,
            "automatic_accuracy_metric": None,
            "purpose": "manual LR-ASD versus LoCoNet+LASER shadow review",
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cards = []
        for row in rows:
            def video(value: object) -> str:
                if value is None:
                    return "<p class='missing'>visualization unavailable</p>"
                return (
                    "<video controls preload='metadata' src='"
                    + html.escape(str(value), quote=True)
                    + "'></video>"
                )

            cards.append(
                "<section><h2>"
                + html.escape(str(row["clip_uid"]))
                + "</h2><div class='grid'><article><h3>LR-ASD</h3>"
                + video(row["lr_asd_visualization"])
                + "</article><article><h3>LoCoNet + LASER</h3>"
                + video(row["laser_visualization"])
                + "</article></div></section>"
            )
        document = """<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>ASD backend comparison</title>
<style>body{font-family:system-ui;margin:24px;background:#f4f5f7;color:#17191c}
section{background:white;border:1px solid #ccd1d8;margin:18px 0;padding:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#111}
.missing{padding:40px;background:#eee}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
</head><body><h1>LR-ASD / LoCoNet + LASER shadow review</h1>
<p>Manual evidence only. No automatic accuracy metric is computed.</p>""" + "".join(cards) + "</body></html>\n"
        (temporary / "review.html").write_text(document, encoding="utf-8")
        temporary.replace(output)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    result = build_comparison_review(
        lr_asd_root=arguments.lr_asd_root,
        laser_root=arguments.laser_root,
        output_root=arguments.output_root,
        clip_ids=arguments.clip_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
