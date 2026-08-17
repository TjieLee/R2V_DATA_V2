from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.h3.audio_backends import PrecomputedEmbeddingBackend
from r2v_data_v2.h3.audio_pairing import AudioPairingConfig
from r2v_data_v2.h3.audio_production import (
    H3ProductionInventory,
    ProductionVisualOccurrence,
    build_h3_production_pairs,
    enumerate_h3_production_inventory,
    orchestrate_production_stages,
    production_paths,
    run_production_embedding_stage,
)
from r2v_data_v2.h3.audio_schemas import FileAsset, VoiceReferenceArtifact
from r2v_data_v2.h3.pair_calibration import (
    EligibleVisualFaceOccurrence,
    VisualFaceOccurrenceLoadResult,
)
from r2v_data_v2.h3.pilot_schemas import H3AudioBindingPilotSummary
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
    VoiceReferenceQualityPolicy,
)
from tools.run_h3_audio_production import _parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(tmp_path: Path, occurrence_ids: list[str]) -> H3ProductionInventory:
    occurrences: list[ProductionVisualOccurrence] = []
    for occurrence_id in occurrence_ids:
        clip_uid, entity_id = occurrence_id.split("/")
        image = tmp_path / "visual" / clip_uid / f"{entity_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 18), (20, 40, 60)).save(image)
        occurrences.append(
            ProductionVisualOccurrence(
                entity_occurrence_id=occurrence_id,
                clip_uid=clip_uid,
                entity_id=entity_id,
                parent_video_id="shared-parent",
                clip_suffix=clip_uid,
                canonical_reference_path=str(image),
                canonical_reference_sha256=_sha256(image),
                canonical_reference_run_path=f"clips/{clip_uid}/selected/{entity_id}.png",
                visual_integrity_provenance={"entity_status": "accepted"},
            )
        )
    occurrences.sort(key=lambda item: item.entity_occurrence_id)
    clips = sorted({item.clip_uid for item in occurrences})
    return H3ProductionInventory(
        source_run_root=str(tmp_path / "visual_run"),
        scanned_clip_count=len(clips),
        eligible_clip_count=len(clips),
        eligible_occurrence_count=len(occurrences),
        eligible_clip_uids=clips,
        occurrences=occurrences,
        skip_reason_counts={},
    )


def _write_primary_voice(
    root: Path,
    inventory: H3ProductionInventory,
    *,
    unavailable: set[str] | None = None,
) -> None:
    unavailable = unavailable or set()
    root.mkdir(parents=True)
    policy = VoiceReferenceQualityPolicy()
    selections: list[PrimaryVoiceReferenceSelection] = []
    for item in inventory.occurrences:
        if item.entity_occurrence_id in unavailable:
            selections.append(
                PrimaryVoiceReferenceSelection(
                    clip_uid=item.clip_uid,
                    entity_id=item.entity_id,
                    entity_occurrence_id=item.entity_occurrence_id,
                    reason_codes=["no_voice_reference_passed_quality_gate"],
                    candidate_turn_ids=["turn_1"],
                    accepted_turn_ids=[],
                    policy_version=policy.version,
                    policy_fingerprint=policy.fingerprint(),
                )
            )
            continue
        path = root / "voice_refs" / item.clip_uid / item.entity_id / "voice_ref_1.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"voice:{item.entity_occurrence_id}".encode())
        asset = FileAsset(
            path=path.relative_to(root).as_posix(),
            sha256=_sha256(path),
            byte_size=path.stat().st_size,
            media_type="audio/flac",
        )
        selections.append(
            PrimaryVoiceReferenceSelection(
                clip_uid=item.clip_uid,
                entity_id=item.entity_id,
                entity_occurrence_id=item.entity_occurrence_id,
                primary_voice_reference=VoiceReferenceArtifact(
                    voice_reference_id=f"voice_ref/{item.entity_occurrence_id}",
                    entity_occurrence_id=item.entity_occurrence_id,
                    source_turn_id="turn_1",
                    source_start=0.0,
                    source_end=1.5,
                    source_start_sample=0,
                    source_end_sample=24000,
                    asset=asset,
                    quality_score=0.8,
                    quality_metadata={},
                ),
                candidate_turn_ids=["turn_1"],
                accepted_turn_ids=["turn_1"],
                policy_version=policy.version,
                policy_fingerprint=policy.fingerprint(),
            )
        )
    (root / "primary_voice_references.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in selections),
        encoding="utf-8",
    )
    available = len(selections) - len(unavailable)
    summary = PrimaryVoiceReferenceExportSummary(
        policy_version=policy.version,
        policy_fingerprint=policy.fingerprint(),
        candidate_turn_count=len(selections),
        accepted_turn_count=available,
        rejected_turn_count=len(unavailable),
        entity_occurrence_count=len(selections),
        occurrences_with_primary_voice_reference=available,
        occurrences_without_primary_voice_reference=len(unavailable),
        rejection_reason_counts=(
            {"voice_snr_too_low": len(unavailable)} if unavailable else {}
        ),
        selected_reference_rows=[],
    )
    (root / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _write_audio(root: Path, inventory: H3ProductionInventory) -> None:
    root.mkdir(parents=True)
    for clip_uid in inventory.eligible_clip_uids:
        source = root.parent / "source" / f"{clip_uid}.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source:{clip_uid}".encode())
        full_audio = root / "full_audio" / f"{clip_uid}.flac"
        full_audio.parent.mkdir(parents=True, exist_ok=True)
        full_audio.write_bytes(f"full-audio:{clip_uid}".encode())
        sidecar = root / "clips" / clip_uid / "audio_binding.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "clip_uid": clip_uid,
                    "source_run_root": inventory.source_run_root,
                    "source_video_path": str(source),
                    "status": "ready",
                    "evidence": {
                        "audio": {
                            "status": "ready",
                            "source_video_path": str(source),
                            "full_audio_path": str(full_audio),
                            "duration_seconds": 3.0,
                            "sample_rate_hz": 16000,
                            "channels": 1,
                            "quality_score": 1.0,
                            "reason": None,
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    summary = H3AudioBindingPilotSummary(
        source_run_root=inventory.source_run_root,
        output_root=str(root),
        clips_attempted=inventory.eligible_clip_count,
        clips_succeeded=inventory.eligible_clip_count,
        clips_failed=0,
        clips_with_speech=inventory.eligible_clip_count,
        bound_intervals=inventory.eligible_occurrence_count,
        overlap_intervals=0,
        offscreen_intervals=0,
        ambiguous_intervals=0,
        no_speech_intervals=0,
        face_entity_association_failures=0,
        asd_runtime_failures=0,
    )
    (root / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_production_inventory_enumerates_all_without_limit_or_parent_quota(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "clips").mkdir(parents=True)
    (run_root / "run.json").write_text("{}")
    source_occurrences: list[EligibleVisualFaceOccurrence] = []
    for index in range(4):
        clip_uid = f"clip-{index}"
        clip_dir = run_root / "clips" / clip_uid
        clip_dir.mkdir()
        (clip_dir / "clip.json").write_text("{}")
        image = tmp_path / f"{clip_uid}.png"
        Image.new("RGB", (8, 8), "white").save(image)
        source_occurrences.append(
            EligibleVisualFaceOccurrence(
                entity_occurrence_id=f"{clip_uid}/e1",
                clip_uid=clip_uid,
                entity_id="e1",
                parent_video_id="one-parent",
                clip_suffix=str(index),
                canonical_reference_path=image,
                canonical_reference_sha256=_sha256(image),
                canonical_reference_run_path=f"clips/{clip_uid}/selected/e1.png",
                existing_v3_cross_pair_provenance=None,
                visual_integrity_provenance={},
            )
        )
    monkeypatch.setattr(
        "r2v_data_v2.h3.audio_production.load_eligible_visual_face_occurrences",
        lambda root: VisualFaceOccurrenceLoadResult(
            run_root=root,
            occurrences=tuple(source_occurrences),
            skip_reason_counts={},
        ),
    )

    result = enumerate_h3_production_inventory(run_root)

    assert result.scanned_clip_count == 4
    assert result.eligible_clip_count == 4
    assert result.eligible_occurrence_count == 4
    assert result.bounded_limit_applied is False
    assert result.parent_quota_applied is False
    assert [item.entity_occurrence_id for item in result.occurrences] == [
        f"clip-{index}/e1" for index in range(4)
    ]
    assert "limit" not in {action.dest for action in _parser()._actions}
    assert {
        "maximum_timestamp_delta_seconds",
        "minimum_face_bbox_coverage",
        "minimum_matched_sampled_slots",
        "minimum_temporal_consistency",
        "minimum_top1_top2_margin",
    }.isdisjoint({action.dest for action in _parser()._actions})


def test_production_embedding_retains_all_and_gates_only_speaker_on_voice(
    tmp_path: Path,
) -> None:
    ids = ["clip-a/e1", "clip-b/e1", "clip-c/e1"]
    inventory = _inventory(tmp_path, ids)
    primary = tmp_path / "primary"
    _write_primary_voice(primary, inventory, unavailable={"clip-c/e1"})
    vectors = {
        "clip-a/e1": np.array([1.0, 0.0], dtype=np.float32),
        "clip-b/e1": np.array([0.8, 0.6], dtype=np.float32),
        "clip-c/e1": np.array([0.6, 0.8], dtype=np.float32),
    }
    output = tmp_path / "embedding"

    summary = run_production_embedding_stage(
        inventory=inventory,
        primary_voice_root=primary,
        output_root=output,
        face_backend=PrecomputedEmbeddingBackend(
            vectors,
            model_identifier="test/face",
        ),
        speaker_backend=PrecomputedEmbeddingBackend(
            vectors,
            model_identifier="test/voice",
        ),
    )

    rows = _read_jsonl(output / "occurrences.jsonl")
    assert [item["entity_occurrence_id"] for item in rows] == ids
    assert summary.input_occurrence_count == 3
    assert summary.face_embedding_available_count == 3
    assert summary.speaker_embedding_available_count == 2
    assert summary.primary_voice_unavailable_count == 1
    assert rows[2]["speaker"]["status"] == "unavailable"  # type: ignore[index]
    assert rows[2]["primary_voice_unavailable_reasons"]


def test_production_pairs_are_clip_level_exact_and_media_roles_are_canonical(
    tmp_path: Path,
) -> None:
    ids = ["clip-a/e1", "clip-b/e1", "clip-c/e1", "clip-d/e1", "clip-e/e1"]
    inventory = _inventory(tmp_path, ids)
    primary = tmp_path / "primary"
    _write_primary_voice(primary, inventory, unavailable={"clip-e/e1"})
    face_vectors = {
        "clip-a/e1": np.array([1.0, 0.0], dtype=np.float32),
        "clip-b/e1": np.array([0.8, 0.6], dtype=np.float32),
        "clip-c/e1": np.array([0.8, 0.6], dtype=np.float32),
        "clip-e/e1": np.array([0.0, 1.0], dtype=np.float32),
    }
    voice_vectors = {
        "clip-a/e1": np.array([1.0, 0.0], dtype=np.float32),
        "clip-b/e1": np.array([1.0, 0.0], dtype=np.float32),
        "clip-c/e1": np.array([1.0, 0.0], dtype=np.float32),
        "clip-d/e1": np.array([1.0, 0.0], dtype=np.float32),
    }
    embedding = tmp_path / "embedding"
    run_production_embedding_stage(
        inventory=inventory,
        primary_voice_root=primary,
        output_root=embedding,
        face_backend=PrecomputedEmbeddingBackend(
            face_vectors,
            model_identifier="test/face",
        ),
        speaker_backend=PrecomputedEmbeddingBackend(
            voice_vectors,
            model_identifier="test/voice",
        ),
    )
    audio = tmp_path / "audio"
    _write_audio(audio, inventory)
    output = tmp_path / "pairs"

    summary = build_h3_production_pairs(
        inventory=inventory,
        audio_root=audio,
        primary_voice_root=primary,
        embedding_root=embedding,
        output_root=output,
    )

    in_pairs = _read_jsonl(output / "in_pairs.jsonl")
    cross_pairs = _read_jsonl(output / "cross_pairs.jsonl")
    evidence = _read_jsonl(output / "pair_evidence.jsonl")
    assert len(in_pairs) == 4
    assert {item["target_clip_uid"] for item in in_pairs} == {
        item.split("/")[0] for item in ids[:4]
    }
    assert len({item["target_clip_uid"] for item in in_pairs}) == len(in_pairs)
    assert all(len(item["subjects"]) == 1 for item in in_pairs)

    target_in = next(item for item in in_pairs if item["target_clip_uid"] == "clip-a")
    donor_in = next(item for item in in_pairs if item["target_clip_uid"] == "clip-b")
    first = next(item for item in cross_pairs if item["target_clip_uid"] == "clip-a")
    mapping = first["mappings"][0]
    assert mapping["donor_occurrence_id"] == "clip-b/e1"
    assert (
        mapping["target_visual_reference_path"]
        == target_in["subjects"][0]["target_visual_reference_path"]
    )
    assert (
        mapping["donor_primary_voice_reference_path"]
        == donor_in["subjects"][0]["target_primary_voice_reference_path"]
    )
    assert (
        mapping["target_primary_voice_reference_path"]
        == target_in["subjects"][0]["target_primary_voice_reference_path"]
    )
    assert (
        mapping["donor_visual_reference_path"]
        == donor_in["subjects"][0]["target_visual_reference_path"]
    )
    assert first["target_video_path"] == target_in["target_video_path"]
    assert first["target_full_audio_path"] == target_in["target_full_audio_path"]
    assert Path(first["target_full_audio_path"]).is_file()
    assert len({item["target_clip_uid"] for item in cross_pairs}) == len(cross_pairs)

    selected_evidence = [item for item in evidence if item["selected"]]
    selected_mappings = [item for pair in cross_pairs for item in pair["mappings"]]
    assert len(evidence) == 3 * 2
    assert len(selected_evidence) == len(selected_mappings) == 3
    assert {
        (item["target_occurrence_id"], item["donor_occurrence_id"])
        for item in selected_evidence
    } == {
        (item["target_occurrence_id"], item["donor_occurrence_id"])
        for item in selected_mappings
    }

    html = (output / "review.html").read_text(encoding="utf-8")
    assert "Is this donor the same physical person as the target" in html
    assert all(item["mapping_id"] in html for item in selected_mappings)
    assert "1 CORRECT" in html
    assert "2 WRONG" in html
    assert "3 UNCERTAIN" in html
    assert "Export JSONL" in html
    assert len(list((output / "review_media" / "faces").iterdir())) == 3
    assert len(list((output / "review_media" / "voices").iterdir())) == 3

    assert summary.complete_visual_eligible_occurrence_count == 5
    assert summary.in_pair_clip_sample_count == 4
    assert summary.cross_pair_candidate_clip_count == 3
    assert summary.cross_pair_clip_sample_count == 3
    assert summary.selected_target_donor_subject_mapping_count == 3
    assert summary.clips_without_complete_cross_pair_mapping == 1
    assert summary.incomplete_cross_pair_reason_counts == {
        "target_missing_required_embedding": 1
    }
    assert summary.parent_quota_applied is False
    assert summary.transitive_clustering_performed is False
    assert summary.human_calibration_labels_used_as_identity_truth is False
    assert summary.exact_candidate_evaluation is True
    assert summary.thresholds_calibrated is True
    assert summary.face_threshold == 0.72
    assert summary.voice_threshold == 0.20
    assert summary.rank_gate_enabled is False
    assert summary.margin_gate_enabled is False
    assert summary.text_gate_enabled is False
    assert AudioPairingConfig().face_threshold == 0.72
    assert AudioPairingConfig().voice_threshold == 0.20


def test_multi_speaker_cross_uses_maximum_total_face_assignment(
    tmp_path: Path,
) -> None:
    ids = ["clip-a/e1", "clip-a/e2", "clip-b/e1", "clip-c/e1"]
    inventory = _inventory(tmp_path, ids)
    primary = tmp_path / "primary"
    _write_primary_voice(primary, inventory)

    def unit(degrees: float) -> np.ndarray:
        radians = math.radians(degrees)
        return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)

    face_vectors = {
        "clip-a/e1": unit(0.0),
        "clip-a/e2": unit(10.0),
        "clip-b/e1": unit(2.0),
        "clip-c/e1": unit(-10.0),
    }
    low_voice = np.array([0.2, math.sqrt(0.96)], dtype=np.float32)
    voice_vectors = {
        "clip-a/e1": unit(0.0),
        "clip-a/e2": unit(0.0),
        "clip-b/e1": unit(0.0),
        "clip-c/e1": low_voice,
    }
    embedding = tmp_path / "embedding"
    run_production_embedding_stage(
        inventory=inventory,
        primary_voice_root=primary,
        output_root=embedding,
        face_backend=PrecomputedEmbeddingBackend(
            face_vectors,
            model_identifier="test/face",
        ),
        speaker_backend=PrecomputedEmbeddingBackend(
            voice_vectors,
            model_identifier="test/voice",
        ),
    )
    audio = tmp_path / "audio"
    _write_audio(audio, inventory)
    output = tmp_path / "pairs"

    summary = build_h3_production_pairs(
        inventory=inventory,
        audio_root=audio,
        primary_voice_root=primary,
        embedding_root=embedding,
        output_root=output,
    )

    in_pairs = _read_jsonl(output / "in_pairs.jsonl")
    cross_pairs = _read_jsonl(output / "cross_pairs.jsonl")
    target_in = next(item for item in in_pairs if item["target_clip_uid"] == "clip-a")
    target_cross = next(
        item for item in cross_pairs if item["target_clip_uid"] == "clip-a"
    )
    assert len(target_in["subjects"]) == 2
    assert [item["target_occurrence_id"] for item in target_cross["mappings"]] == [
        "clip-a/e1",
        "clip-a/e2",
    ]
    assert [item["donor_occurrence_id"] for item in target_cross["mappings"]] == [
        "clip-c/e1",
        "clip-b/e1",
    ]
    assert len({item["donor_occurrence_id"] for item in target_cross["mappings"]}) == 2
    assert target_cross["mappings"][0]["voice_similarity"] == pytest.approx(0.2)
    assert target_cross["mappings"][1]["voice_similarity"] == pytest.approx(1.0)
    selected_face_total = sum(
        item["face_similarity"] for item in target_cross["mappings"]
    )
    greedy_face_total = float(
        np.dot(face_vectors["clip-a/e1"], face_vectors["clip-b/e1"])
        + np.dot(face_vectors["clip-a/e2"], face_vectors["clip-c/e1"])
    )
    assert selected_face_total > greedy_face_total
    assert len({item["target_clip_uid"] for item in in_pairs}) == len(in_pairs)
    assert len({item["target_clip_uid"] for item in cross_pairs}) == len(cross_pairs)
    assert summary.selected_target_donor_subject_mapping_count == sum(
        len(item["mappings"]) for item in cross_pairs
    )


def test_incomplete_multi_speaker_mapping_preserves_clip_in_pair(
    tmp_path: Path,
) -> None:
    ids = ["clip-a/e1", "clip-a/e2", "clip-b/e1"]
    inventory = _inventory(tmp_path, ids)
    primary = tmp_path / "primary"
    _write_primary_voice(primary, inventory)
    vectors = {
        occurrence_id: np.array([1.0, 0.0], dtype=np.float32) for occurrence_id in ids
    }
    embedding = tmp_path / "embedding"
    run_production_embedding_stage(
        inventory=inventory,
        primary_voice_root=primary,
        output_root=embedding,
        face_backend=PrecomputedEmbeddingBackend(vectors, model_identifier="face"),
        speaker_backend=PrecomputedEmbeddingBackend(vectors, model_identifier="voice"),
    )
    audio = tmp_path / "audio"
    _write_audio(audio, inventory)
    output = tmp_path / "pairs"

    summary = build_h3_production_pairs(
        inventory=inventory,
        audio_root=audio,
        primary_voice_root=primary,
        embedding_root=embedding,
        output_root=output,
    )

    in_pairs = _read_jsonl(output / "in_pairs.jsonl")
    cross_pairs = _read_jsonl(output / "cross_pairs.jsonl")
    evidence = _read_jsonl(output / "pair_evidence.jsonl")
    target_in = next(item for item in in_pairs if item["target_clip_uid"] == "clip-a")
    assert len(target_in["subjects"]) == 2
    assert not any(item["target_clip_uid"] == "clip-a" for item in cross_pairs)
    assert summary.clips_without_complete_cross_pair_mapping == 1
    assert summary.incomplete_cross_pair_reason_counts == {
        "no_complete_strict_donor_mapping": 1
    }
    assert len(evidence) == 4
    assert all(item["target_clip_uid"] != item["donor_clip_uid"] for item in evidence)


def test_no_cross_donor_keeps_valid_in_pair(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, ["clip-only/e1"])
    primary = tmp_path / "primary"
    _write_primary_voice(primary, inventory)
    vectors = {"clip-only/e1": np.array([1.0, 0.0], dtype=np.float32)}
    embedding = tmp_path / "embedding"
    run_production_embedding_stage(
        inventory=inventory,
        primary_voice_root=primary,
        output_root=embedding,
        face_backend=PrecomputedEmbeddingBackend(vectors, model_identifier="face"),
        speaker_backend=PrecomputedEmbeddingBackend(vectors, model_identifier="voice"),
    )
    audio = tmp_path / "audio"
    _write_audio(audio, inventory)

    summary = build_h3_production_pairs(
        inventory=inventory,
        audio_root=audio,
        primary_voice_root=primary,
        embedding_root=embedding,
        output_root=tmp_path / "pairs",
    )

    assert summary.in_pair_clip_sample_count == 1
    assert summary.cross_pair_candidate_clip_count == 1
    assert summary.cross_pair_clip_sample_count == 0
    assert summary.clips_without_complete_cross_pair_mapping == 1


def test_stage_reuse_never_reruns_completed_expensive_stage(tmp_path: Path) -> None:
    paths = production_paths(tmp_path / "audio_runs")
    roots = {
        "audio": paths.audio,
        "primary-voice": paths.primary_voice,
        "embedding": paths.embedding,
        "pair": paths.pairs,
    }
    required = {
        "audio": [
            "summary.json",
            "audio_bindings.jsonl",
            "failures.jsonl",
            "voice_reference_quality_summary.json",
        ],
        "primary-voice": [
            "summary.json",
            "primary_voice_references.jsonl",
            "voice_quality_assessments.jsonl",
        ],
        "embedding": ["summary.json", "occurrences.jsonl"],
        "pair": [
            "summary.json",
            "in_pairs.jsonl",
            "cross_pairs.jsonl",
            "pair_evidence.jsonl",
            "review.html",
        ],
    }
    for stage, names in required.items():
        roots[stage].mkdir(parents=True)
        for name in names:
            (roots[stage] / name).write_text("\n")

    status = orchestrate_production_stages(
        stages=("audio", "primary-voice", "embedding", "pair"),
        roots=roots,
        runners={
            stage: lambda overwrite: (_ for _ in ()).throw(AssertionError())
            for stage in roots
        },
    )

    assert status == {stage: "reused" for stage in roots}
    assert paths.root.name == "production"
    assert [
        paths.audio.name,
        paths.primary_voice.name,
        paths.embedding.name,
        paths.pairs.name,
    ] == [
        "audio",
        "primary_voice",
        "embedding",
        "pairs",
    ]


def test_overwrite_upstream_cannot_leave_completed_downstream_stale(
    tmp_path: Path,
) -> None:
    paths = production_paths(tmp_path)
    roots = {
        "audio": paths.audio,
        "primary-voice": paths.primary_voice,
        "embedding": paths.embedding,
        "pair": paths.pairs,
    }
    paths.pairs.mkdir(parents=True)
    for name in (
        "summary.json",
        "in_pairs.jsonl",
        "cross_pairs.jsonl",
        "pair_evidence.jsonl",
        "review.html",
    ):
        (paths.pairs / name).write_text("\n")

    with pytest.raises(ValueError, match="completed downstream stages"):
        orchestrate_production_stages(
            stages=("audio",),
            roots=roots,
            runners={stage: lambda overwrite: None for stage in roots},
            overwrite=True,
        )
