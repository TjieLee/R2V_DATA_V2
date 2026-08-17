from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from r2v_data_v2.h3.audio_schemas import EmbeddingAsset, FileAsset
from r2v_data_v2.h3.embedding_pilot import (
    EmbeddingPilotModalityResult,
    EmbeddingPilotOccurrence,
)
from r2v_data_v2.h3.pairing_pilot import (
    build_pairing_pilot,
    report_accepted_pair_review,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _file_asset(root: Path, path: Path, media_type: str) -> FileAsset:
    return FileAsset(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        media_type=media_type,
    )


def _embedding_asset(
    root: Path,
    path: Path,
    vector: np.ndarray,
    model: str,
) -> EmbeddingAsset:
    values = np.asarray(vector, dtype=np.float32)
    values /= np.linalg.norm(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    return EmbeddingAsset(
        **_file_asset(root, path, "application/x-npy").model_dump(mode="python"),
        model_identifier=model,
        dimension=len(values),
    )


def _fixture(
    tmp_path: Path,
    *,
    face_unavailable: set[str] | None = None,
) -> tuple[Path, Path]:
    face_unavailable = face_unavailable or set()
    audio_root = tmp_path / "audio"
    embedding_root = tmp_path / "embedding"
    embedding_root.mkdir()
    occurrence_ids = ["clip-a/e1", "clip-b/e1", "clip-c/e1"]
    face_vectors = {
        "clip-a/e1": np.array([1.0, 0.0]),
        "clip-b/e1": np.array([0.9, np.sqrt(0.19)]),
        "clip-c/e1": np.array([0.8, 0.6]),
    }
    voice_vectors = {item: np.array([1.0, 0.0]) for item in occurrence_ids}
    rows: list[EmbeddingPilotOccurrence] = []
    for occurrence_id in occurrence_ids:
        clip_uid, entity_id = occurrence_id.split("/")
        source_video = tmp_path / "source_videos" / f"{clip_uid}.mp4"
        source_video.parent.mkdir(parents=True, exist_ok=True)
        source_video.write_bytes(f"source:{clip_uid}".encode())
        sidecar = audio_root / "clips" / clip_uid / "audio_binding.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "clip_uid": clip_uid,
                    "source_video_path": str(source_video),
                    "status": "ready",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        visual = tmp_path / "visual" / clip_uid / f"{entity_id}.png"
        visual.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (30, 60, 90)).save(visual)
        voice = tmp_path / "voice" / clip_uid / f"{entity_id}.flac"
        voice.parent.mkdir(parents=True, exist_ok=True)
        voice.write_bytes(f"voice:{occurrence_id}".encode())
        if occurrence_id in face_unavailable:
            face = EmbeddingPilotModalityResult(
                modality="face",
                status="unavailable",
                model_identifier="test/face",
                failure_reason="face_not_found_in_canonical_reference",
            )
        else:
            crop = embedding_root / "face_crops" / clip_uid / f"{entity_id}.png"
            crop.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (70, 80, 90)).save(crop)
            face = EmbeddingPilotModalityResult(
                modality="face",
                status="available",
                crop_asset=_file_asset(embedding_root, crop, "image/png"),
                embedding_asset=_embedding_asset(
                    embedding_root,
                    embedding_root
                    / "embeddings"
                    / "face"
                    / clip_uid
                    / f"{entity_id}.npy",
                    face_vectors[occurrence_id],
                    "test/face",
                ),
                model_identifier="test/face",
            )
        speaker = EmbeddingPilotModalityResult(
            modality="speaker",
            status="available",
            embedding_asset=_embedding_asset(
                embedding_root,
                embedding_root
                / "embeddings"
                / "voice"
                / clip_uid
                / f"{entity_id}.npy",
                voice_vectors[occurrence_id],
                "test/voice",
            ),
            model_identifier="test/voice",
        )
        rows.append(
            EmbeddingPilotOccurrence(
                entity_occurrence_id=occurrence_id,
                clip_uid=clip_uid,
                entity_id=entity_id,
                visual_reference_path=str(visual),
                visual_reference_sha256=_sha256(visual),
                primary_voice_reference_path=str(voice),
                primary_voice_reference_sha256=_sha256(voice),
                face=face,
                speaker=speaker,
            )
        )
    (embedding_root / "occurrences.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )
    return audio_root, embedding_root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_pairing_pilot_is_read_only_deterministic_and_reviews_every_cross_pair(
    tmp_path: Path,
) -> None:
    audio_root, embedding_root = _fixture(tmp_path)
    before_audio = _tree_hashes(audio_root)
    before_embedding = _tree_hashes(embedding_root)
    output = tmp_path / "pairing_pilot"

    summary = build_pairing_pilot(
        audio_pilot_root=audio_root,
        embedding_root=embedding_root,
        output_root=output,
    )

    assert _tree_hashes(audio_root) == before_audio
    assert _tree_hashes(embedding_root) == before_embedding
    assert {path.name for path in output.iterdir()} == {
        "in_pairs.jsonl",
        "cross_pairs.jsonl",
        "pair_evidence.jsonl",
        "summary.json",
        "review.html",
    }
    assert summary["eligible_in_pair_occurrence_count"] == 3
    assert summary["in_pair_count"] == 3
    assert summary["cross_pair_count"] == 3
    assert summary["accepted_target_donor_mapping_count"] == 3
    assert summary["thresholds_calibrated"] is True
    assert summary["parent_quota_applied"] is False
    assert summary["transitive_clustering_performed"] is False
    assert summary["human_calibration_labels_used_as_identity_truth"] is False
    cross_pairs = _read_jsonl(output / "cross_pairs.jsonl")
    assert [item["target_clip_uid"] for item in cross_pairs] == [
        "clip-a",
        "clip-b",
        "clip-c",
    ]
    first_mapping = cross_pairs[0]["mappings"][0]  # type: ignore[index]
    assert first_mapping["donor_occurrence_id"] == "clip-b/e1"  # type: ignore[index]
    html = (output / "review.html").read_text(encoding="utf-8")
    assert html.count('"accepted_pair_id"') == 3
    assert "1 CORRECT" in html
    assert "2 WRONG" in html
    assert "3 UNCERTAIN" in html


def test_pairing_pilot_preserves_in_pair_when_face_is_unavailable(tmp_path: Path) -> None:
    audio_root, embedding_root = _fixture(
        tmp_path,
        face_unavailable={"clip-c/e1"},
    )
    output = tmp_path / "pairing_pilot"

    summary = build_pairing_pilot(
        audio_pilot_root=audio_root,
        embedding_root=embedding_root,
        output_root=output,
    )

    assert summary["eligible_in_pair_occurrence_count"] == 3
    assert summary["in_pair_count"] == 3
    assert summary["cross_pair_eligible_target_count"] == 2
    assert [item["clip_uid"] for item in _read_jsonl(output / "in_pairs.jsonl")] == [
        "clip-a",
        "clip-b",
        "clip-c",
    ]


def test_accepted_pair_review_report_counts_wrong_and_uncertain(tmp_path: Path) -> None:
    audio_root, embedding_root = _fixture(tmp_path)
    output = tmp_path / "pairing_pilot"
    build_pairing_pilot(
        audio_pilot_root=audio_root,
        embedding_root=embedding_root,
        output_root=output,
    )
    mappings = [
        mapping
        for pair in _read_jsonl(output / "cross_pairs.jsonl")
        for mapping in pair["mappings"]  # type: ignore[index]
    ]
    labels = tmp_path / "labels.jsonl"
    decisions = ["WRONG", "UNCERTAIN", "CORRECT"]
    labels.write_text(
        "".join(
            json.dumps(
                {
                    "accepted_pair_id": item["accepted_pair_id"],
                    "target_occurrence_id": item["target_occurrence_id"],
                    "donor_occurrence_id": item["donor_occurrence_id"],
                    "label": decision,
                    "face_similarity": item["face_similarity"],
                    "voice_similarity": item["voice_similarity"],
                },
                sort_keys=True,
            )
            + "\n"
            for item, decision in zip(mappings, decisions, strict=True)
        ),
        encoding="utf-8",
    )

    report = report_accepted_pair_review(
        pairing_pilot_root=output,
        labels_path=labels,
        output_path=tmp_path / "accepted_pair_review_report.json",
    )

    assert report["accepted_cross_pair_count"] == 3
    assert report["reviewed_count"] == 3
    assert report["correct_count"] == 1
    assert report["wrong_count"] == 1
    assert report["uncertain_count"] == 1
    assert report["empirical_precision"] == 0.5
    assert len(report["wrong_pairs"]) == 1
    assert len(report["uncertain_pairs"]) == 1
    assert report["thresholds_modified"] is False
