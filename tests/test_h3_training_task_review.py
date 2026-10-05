from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from r2v_data_v2.h3.training_manifest_export import (
    TASK_ORDER,
    build_ra2va_training_task_rows,
    export_training_manifests,
)
from r2v_data_v2.h3.training_task_review import (
    REVIEW_TASKS,
    TrainingTaskReviewAnnotation,
    TrainingTaskReviewStore,
    build_review_cases,
    export_reviewed_training_manifests,
    make_review_server,
)
from tools.serve_h3_training_task_review import main as review_main


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source(tmp_path: Path) -> Path:
    shadow = tmp_path / "shadow"
    video = tmp_path / "target.mp4"
    video.write_bytes(b"0123456789")
    image_paths = [tmp_path / "picture-1.png", tmp_path / "picture-2.png"]
    frame_paths = [tmp_path / "first.png", tmp_path / "last.png"]
    audio_paths = [
        tmp_path / "speech.flac",
        tmp_path / "music.flac",
        tmp_path / "full.flac",
    ]
    for path in [*image_paths, *frame_paths, *audio_paths]:
        path.write_bytes(path.name.encode())
    _write_json(
        shadow / "audio_reuse_prepared_v1/inventory.json",
        {
            "jobs": [
                {
                    "clip_uid": "clip-a",
                    "target_video_path": str(video),
                    "reference_images": [
                        {
                            "image_index": 2,
                            "picture_label": "<Picture 2>",
                            "image_artifact_path": str(image_paths[1]),
                        },
                        {
                            "image_index": 1,
                            "picture_label": "<Picture 1>",
                            "image_artifact_path": str(image_paths[0]),
                        },
                    ],
                    "reference_subjects": [
                        {
                            "subject_label": "<Subject 1>",
                            "source_picture_labels": ["<Picture 1>"],
                        }
                    ],
                }
            ]
        },
    )

    def audio(kind: str, path: Path) -> dict:
        return {"contract": {"kind": kind, "path": str(path)}}

    products = [
        {
            "clip_uid": "clip-a",
            "status": "ready",
            "conditioning_variant": "visual_only",
            "audio_references": [],
            "rendered_h3_prompt": "subject_definitions:\nA\n\nsummary:\nB\n\nretention_analysis:\nC\n\ndetailed_description:\nD\n\noverall_soundscape:\nE\n\nnon_diegetic_music:\nN/A",
        },
        {
            "clip_uid": "clip-a",
            "status": "ready",
            "conditioning_variant": "target_speech_reuse",
            "audio_references": [
                audio("speaker_speech_reuse", audio_paths[0]),
                audio("music_reuse", audio_paths[1]),
            ],
            "rendered_h3_prompt": "speech + music caption",
        },
        {
            "clip_uid": "clip-a",
            "status": "ready",
            "conditioning_variant": "full_audio_reuse",
            "audio_references": [audio("full_audio_reuse", audio_paths[2])],
            "rendered_h3_prompt": "full audio caption",
        },
        {
            "clip_uid": "clip-a",
            "status": "ready",
            "conditioning_variant": "cross_voice_reference",
            "audio_references": [audio("cross_voice", audio_paths[0])],
            "rendered_h3_prompt": "excluded cross-voice",
        },
    ]
    _write_jsonl(shadow / "h3_audio_reuse_products_v1/records.jsonl", products)

    frames = []
    for mode, indexes in (
        ("first_frame", [0]),
        ("last_frame", [1]),
        ("first_last_frame", [0, 1]),
    ):
        for variant, references in (
            ("visual_only", []),
            ("full_audio_reuse", [audio("full_audio_reuse", audio_paths[2])]),
            ("target_speech_reuse", [audio("speaker_speech_reuse", audio_paths[0])]),
        ):
            frames.append(
                {
                    "clip_uid": "clip-a",
                    "target_video_path": str(video),
                    "visual_reference_mode": mode,
                    "conditioning_variant": variant,
                    "frame_references": [
                        {
                            "picture_index": position + 1,
                            "picture_label": f"<Picture {position + 1}>",
                            "frame_role": "first_frame" if index == 0 else "last_frame",
                            "image_path": str(frame_paths[index]),
                        }
                        for position, index in enumerate(indexes)
                    ],
                    "audio_references": references,
                    "rendered_h3_prompt": f"{mode} {variant} caption",
                }
            )
    _write_jsonl(shadow / "h3_frame_conditioned_products_v1/records.jsonl", frames)
    return shadow


def _annotation(case, decision: str, *, tags: list[str] | None = None):
    return TrainingTaskReviewAnnotation(
        clip_uid=case.clip_uid,
        task=case.task,
        sample_fingerprint=case.sample_fingerprint,
        decision=decision,
        issue_tags=tags or [],
        notes="checked",
        reviewed_at="2026-10-06T00:00:00Z",
    )


def test_all_twelve_tasks_share_exact_export_projection(tmp_path: Path):
    shadow = _source(tmp_path)
    projection = build_ra2va_training_task_rows(shadow)
    cases = build_review_cases(shadow)
    original = tmp_path / "original-export"
    export_training_manifests(output_root=original, ra2va_shadow_root=shadow)

    assert REVIEW_TASKS == tuple(
        task for task in TASK_ORDER if task.startswith(("r2va_", "ra2va_"))
    )
    assert len(cases) == len(REVIEW_TASKS) == 12
    assert all(len(projection[task]) == 1 for task in REVIEW_TASKS)
    assert {(case.clip_uid, case.task) for case in cases} == {
        ("clip-a", task) for task in REVIEW_TASKS
    }
    for task in REVIEW_TASKS:
        assert projection[task][0].row == _read_jsonl(original / f"{task}.jsonl")[0]
        assert (
            next(case.row for case in cases if case.task == task)
            == projection[task][0].row
        )
    assert not any("cross_voice" in case.task for case in cases)

    reference = next(case for case in cases if case.task == "r2va_reference")
    assert [Path(path).name for path in reference.row["images"]] == [
        "picture-1.png",
        "picture-2.png",
    ]
    assert "<Picture 1>" in reference.image_labels[0]
    assert "<Subject 1>" in reference.image_labels[0]
    assert [
        Path(path).name
        for path in next(
            case for case in cases if case.task == "r2va_first_last_frame"
        ).row["images"]
    ] == ["first.png", "last.png"]
    assert next(
        case for case in cases if case.task == "r2va_first_frame"
    ).image_labels == ("<Picture 1> / First frame",)
    assert next(
        case for case in cases if case.task == "r2va_last_frame"
    ).image_labels == ("<Picture 1> / Last frame",)
    speech = next(case for case in cases if case.task == "ra2va_reference_speech_bgm")
    assert [Path(path).name for path in speech.row["audios"]] == [
        "speech.flac",
        "music.flac",
    ]
    assert speech.audio_kinds == ("speaker_speech_reuse", "music_reuse")
    assert next(
        case for case in cases if case.task == "ra2va_reference_full_audio"
    ).audio_kinds == ("full_audio_reuse",)


def test_speech_only_is_valid_and_per_task_reviews_are_independent(tmp_path: Path):
    shadow = _source(tmp_path)
    frame_records = shadow / "h3_frame_conditioned_products_v1/records.jsonl"
    rows = _read_jsonl(frame_records)
    cases = build_review_cases(shadow)
    store = TrainingTaskReviewStore(tmp_path / "reviews", cases)
    by_task = {case.task: case for case in cases}
    store.save(_annotation(by_task["r2va_reference"], "PASS"))
    store.save(
        _annotation(
            by_task["ra2va_reference_speech_bgm"], "ISSUE", tags=["music_issue"]
        )
    )
    store.save(_annotation(by_task["r2va_first_frame"], "SKIP"))
    assert len(store.current_annotations()) == 3
    assert store.current_annotations()[("clip-a", "r2va_reference")].decision == "PASS"
    assert (
        store.current_annotations()[("clip-a", "ra2va_reference_speech_bgm")].decision
        == "ISSUE"
    )
    assert store.publish_derived().reviewed_count == 3
    assert (tmp_path / "reviews/annotations.csv").is_file()

    speech = next(
        row
        for row in rows
        if row["visual_reference_mode"] == "first_last_frame"
        and row["conditioning_variant"] == "target_speech_reuse"
    )
    assert len(speech["audio_references"]) == 1
    case = next(
        case for case in cases if case.task == "ra2va_first_last_frame_speech_bgm"
    )
    assert case.audio_kinds == ("speaker_speech_reuse",)


def test_stale_review_is_not_exported_and_pass_only_has_four_keys(tmp_path: Path):
    shadow = _source(tmp_path)
    cases = build_review_cases(shadow)
    by_task = {case.task: case for case in cases}
    store = TrainingTaskReviewStore(tmp_path / "reviews", cases)
    store.save(_annotation(by_task["r2va_reference"], "PASS"))
    store.save(_annotation(by_task["ra2va_reference_full_audio"], "PASS"))
    store.save(
        _annotation(
            by_task["ra2va_reference_speech_bgm"], "ISSUE", tags=["music_issue"]
        )
    )
    store.save(_annotation(by_task["r2va_first_frame"], "SKIP"))

    products = shadow / "h3_audio_reuse_products_v1/records.jsonl"
    rows = _read_jsonl(products)
    rows[2]["rendered_h3_prompt"] = "revised full audio caption"
    _write_jsonl(products, rows)
    refreshed = build_review_cases(shadow)
    refreshed_store = TrainingTaskReviewStore(store.root, refreshed)
    assert refreshed_store.publish_derived().stale_annotation_count == 1
    assert (
        "clip-a",
        "ra2va_reference_full_audio",
    ) not in refreshed_store.current_annotations()
    with pytest.raises(ValueError, match="stale"):
        refreshed_store.save(_annotation(by_task["ra2va_reference_full_audio"], "PASS"))

    source_before = products.read_bytes()
    output = tmp_path / "reviewed-export"
    export_reviewed_training_manifests(
        cases=refreshed, store=refreshed_store, output_root=output
    )
    assert products.read_bytes() == source_before
    assert _read_jsonl(output / "r2va_reference.jsonl") == [
        by_task["r2va_reference"].row
    ]
    assert _read_jsonl(output / "ra2va_reference_full_audio.jsonl") == []
    assert _read_jsonl(output / "ra2va_reference_speech_bgm.jsonl") == []
    assert _read_jsonl(output / "r2va_first_frame.jsonl") == []
    assert _read_jsonl(output / "videos.jsonl") == [
        {"video": by_task["r2va_reference"].row["video"], "tasks": ["r2va_reference"]}
    ]
    assert all(
        set(row) == {"video", "images", "audios", "caption"}
        for task in REVIEW_TASKS
        for row in _read_jsonl(output / f"{task}.jsonl")
    )
    assert {path.name for path in output.glob("*.jsonl")} == {
        "videos.jsonl",
        *(f"{task}.jsonl" for task in REVIEW_TASKS),
    }
    with pytest.raises(FileExistsError):
        export_reviewed_training_manifests(
            cases=refreshed, store=refreshed_store, output_root=output
        )


def test_review_export_cli_reuses_saved_pass_decision(tmp_path: Path):
    shadow = _source(tmp_path)
    cases = build_review_cases(shadow)
    review_root = tmp_path / "reviews"
    store = TrainingTaskReviewStore(review_root, cases)
    store.save(_annotation(cases[0], "PASS"))
    output = tmp_path / "cli-export"
    assert (
        review_main(
            [
                "export",
                "--ra2va-shadow-root",
                str(shadow),
                "--review-root",
                str(review_root),
                "--output-root",
                str(output),
            ]
        )
        == output
    )
    assert _read_jsonl(output / f"{cases[0].task}.jsonl") == [cases[0].row]
    assert sum(len(_read_jsonl(output / f"{task}.jsonl")) for task in REVIEW_TASKS) == 1


def test_issue_requires_known_tag_and_pass_has_no_tags(tmp_path: Path):
    case = build_review_cases(_source(tmp_path))[0]
    with pytest.raises(ValueError, match="at least one issue tag"):
        _annotation(case, "ISSUE")
    with pytest.raises(ValueError, match="invalid training review issue tags"):
        _annotation(case, "ISSUE", tags=["invented_tag"])
    with pytest.raises(ValueError, match="only ISSUE review"):
        _annotation(case, "PASS", tags=["other"])


def test_cli_rejects_review_or_export_inside_frozen_shadow(tmp_path: Path):
    shadow = _source(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen shadow"):
        review_main(
            [
                "export",
                "--ra2va-shadow-root",
                str(shadow),
                "--review-root",
                str(shadow),
                "--output-root",
                str(tmp_path / "export"),
            ]
        )
    assert not (shadow / "summary.json").exists()
    with pytest.raises(ValueError, match="outside the frozen shadow"):
        review_main(
            [
                "export",
                "--ra2va-shadow-root",
                str(shadow),
                "--review-root",
                str(tmp_path / "reviews"),
                "--output-root",
                str(shadow / "unsafe-export"),
            ]
        )
    assert not (shadow / "unsafe-export").exists()


def test_review_server_rejects_non_loopback_bind(tmp_path: Path):
    cases = build_review_cases(_source(tmp_path))
    store = TrainingTaskReviewStore(tmp_path / "reviews", cases)
    with patch("r2v_data_v2.h3.training_task_review.ThreadingHTTPServer") as listener:
        with pytest.raises(ValueError, match="loopback"):
            make_review_server(host="0.0.0.0", port=0, cases=cases, store=store)
        listener.assert_not_called()


def test_review_server_serves_media_range_and_persists_annotation(tmp_path: Path):
    cases = build_review_cases(_source(tmp_path))
    store = TrainingTaskReviewStore(tmp_path / "reviews", cases)
    server = make_review_server(host="127.0.0.1", port=0, cases=cases, store=store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            html = response.read().decode()
        assert "Original target video" in html
        assert "Conditioning images" in html
        assert "Conditioning audio" in html
        assert "Final caption" in html
        assert "Training row preview" in html
        assert "<video" in html and "<audio" in html
        with urlopen(
            Request(base + "/media/m0", headers={"Range": "bytes=2-5"})
        ) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.read() == b"2345"
        body = _annotation(cases[0], "PASS").model_dump_json().encode()
        with urlopen(
            Request(
                base + "/api/annotation",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            assert response.status == 200
        assert (
            store.current_annotations()[(cases[0].clip_uid, cases[0].task)].decision
            == "PASS"
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(base + "/media/does-not-exist")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
