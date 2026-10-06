"""Read-only latest single-v7 caption review regressions."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from r2v_data_v2.h3.single_v7_caption_review import (
    build_caption_cases,
    make_caption_server,
)

OVERRIDE_UID = "469daba814e6d71f8f4dfa4a"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    base = tmp_path / "base"
    override = tmp_path / "override"
    base.mkdir()
    override.mkdir()
    uids = [f"clip_{index:02d}" for index in range(19)] + [OVERRIDE_UID]
    video = tmp_path / "target.mp4"
    video.write_bytes(b"0123456789")
    picture = tmp_path / "picture.png"
    picture.write_bytes(b"image")
    audio = tmp_path / "target.flac"
    audio.write_bytes(b"audio")
    jobs = [
        {
            "clip_uid": uid,
            "target_video_path": str(video),
            "target_full_audio_path": str(audio),
            "reference_images": [
                {
                    "picture_label": "<Picture 1>",
                    "kind": "object",
                    "entity_id": "e2",
                    "image_artifact_path": str(picture),
                }
            ],
            "reference_subjects": [
                {
                    "subject_label": "<Subject 1>",
                    "kind": "entity",
                    "entity_id": "e2",
                    "source_picture_labels": ["<Picture 1>"],
                }
            ],
            "segments": [
                {
                    "segment_id": "segment_0001",
                    "start_time": 0.1,
                    "end_time": 0.7,
                    "source_speaker_cluster_id": "cluster_a",
                    "asr_status": "transcribed",
                    "asr_text": "你好",
                    "asr_language": "Chinese",
                }
            ],
        }
        for uid in uids
    ]
    _write_json(base / "source_contract.json", {"clip_uids": uids, "jobs": jobs})

    def record(uid: str, caption: str) -> dict:
        semantics = {
            "subject_definitions": [
                {"subject_label": "<Subject 1>", "description": "A person."}
            ],
            "summary": "Visual summary.",
            "style_opening": "A steady shot.",
            "shot1_caption": caption,
            "overall_soundscape": "Soft ambience.",
            "non_diegetic_music": "N/A",
            "visual_retention_analysis": [
                {"subject_label": "<Subject 1>", "marker": "fully_preserved", "description": "Visible."}
            ],
        }
        return {
            "clip_uid": uid,
            "status": "ready",
            "annotation": {
                "h3_semantics": semantics,
                "av_grounding": {"segment_groundings": [
                    {"segment_id": "segment_0001", "primary_speaker_group": "g1", "speech_presentation": "onscreen_spoken"}
                ]},
            },
            "raw_responses": [json.dumps({"h3_semantics": semantics})],
        }

    base_records = [record(uid, f"BASE {uid}") for uid in uids]
    base_records[-1]["status"] = "failed"
    base_records[-1]["annotation"] = None
    base_records[-1]["failure_reason"] = "old validator failure"
    _write_jsonl(base / "records.jsonl", base_records)
    override_record = record(OVERRIDE_UID, "LATEST 469d caption")
    _write_jsonl(override / "records.jsonl", [override_record])
    return base, override, uids


def test_base_inventory_and_latest_override_keep_frozen_context(tmp_path: Path) -> None:
    base, override, uids = _fixture(tmp_path)
    cases = build_caption_cases(base, override)

    assert [case["clip_uid"] for case in cases] == uids
    assert len(cases) == 20
    assert cases[0]["caption"]["shot1_caption"] == "BASE clip_00"
    latest = cases[-1]
    assert latest["caption"]["shot1_caption"] == "LATEST 469d caption"
    assert latest["status"] == "ready"
    assert json.loads(latest["raw_compact"])["h3_semantics"]["shot1_caption"] == "LATEST 469d caption"
    assert latest["video_path"] == str(tmp_path / "target.mp4")
    assert latest["audio_path"] == str(tmp_path / "target.flac")
    assert latest["references"] == [
        {
            "picture_label": "<Picture 1>",
            "subject_labels": ["<Subject 1>"],
            "kind": "object",
            "path": str(tmp_path / "picture.png"),
        }
    ]
    assert latest["segments"][0]["segment_id"] == "segment_0001"
    assert latest["groundings"][0]["primary_speaker_group"] == "g1"
    assert build_caption_cases(base)[-1]["status"] == "failed"
    assert build_caption_cases(base)[-1]["caption"]["shot1_caption"] == f"BASE {OVERRIDE_UID}"


def test_http_media_range_and_read_only_page(tmp_path: Path) -> None:
    base, override, _ = _fixture(tmp_path)
    server = make_caption_server(host="127.0.0.1", port=0, cases=build_caption_cases(base, override))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert "LATEST 469d caption" in page
        assert "Raw compact JSON" in page
        connection.request("GET", "/media/m0", headers={"Range": "bytes=2-5"})
        response = connection.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 2-5/10"
        assert response.read() == b"2345"
        connection.request("POST", "/api/annotation", body=b"{}")
        response = connection.getresponse()
        assert response.status == 405
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_viewer_displays_subject_kind_and_owner_without_audio_materialization(tmp_path: Path) -> None:
    base, override, _ = _fixture(tmp_path)
    contract_path = base / "source_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["jobs"][0]["reference_subjects"].extend([
        {
            "subject_label": "<Subject 2>", "kind": "attribute", "owner_entity_id": "e2",
            "source_picture_labels": ["<Picture 2>"],
        },
        {
            "subject_label": "<Subject 3>", "kind": "background",
            "source_picture_labels": ["<Picture 3>"],
        },
    ])
    _write_json(contract_path, contract)
    subjects = build_caption_cases(base, override)[0]["subjects"]
    assert [subject["display_label"] for subject in subjects] == [
        "<Subject 1> [entity]",
        "<Subject 2> [attribute, owner=<Subject 1>]",
        "<Subject 3> [background]",
    ]
    html = (Path(__file__).parents[1] / "r2v_data_v2/h3/single_v7_caption_review.html").read_text(
        encoding="utf-8"
    )
    assert "Final <Audio N> definitions/relationships are pipeline-owned" in html
