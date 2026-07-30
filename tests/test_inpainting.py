from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.config import (
    DinoEvaluatorConfig,
    InpaintingBackgroundConfig,
    InpaintingConfig,
    InpaintingConsistencyConfig,
    InpaintingEntityConfig,
    PipelineConfig,
    QwenConfig,
    QwenServicesConfig,
    RankingConfig,
    RankingEvaluatorsConfig,
    SiglipEvaluatorConfig,
)
from r2v_data_v2.inpainting import (
    Flux1FillBackend,
    InpaintingDependencyError,
    NoOpInpaintBackend,
    ProductionConsistencyValidator,
    QwenInpaintingConsistencyJudge,
    _background_generation_mask,
    _inpainting_prompt,
    run_inpainting,
)
from r2v_data_v2.schemas import (
    BackgroundInpaintingReview,
    InpaintingSemanticReview,
)


class _WholeImageBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def inpaint(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        del mask, seed
        self.prompts.append(prompt)
        return Image.new("RGB", image.size, (255, 20, 40))


def _raw_image() -> np.ndarray:
    y, x = np.indices((32, 32))
    return np.stack(
        (
            40 + (x * 3) % 120,
            50 + (y * 4) % 120,
            60 + ((x + y) * 2) % 120,
        ),
        axis=-1,
    ).astype(np.uint8)


def _write_reference(
    output_root: Path,
    *,
    reference_id: str,
    reference_type: str,
    mask: np.ndarray,
    needs_inpainting: bool,
) -> Path:
    destination = output_root / "references" / "clip-1" / reference_id
    destination.mkdir(parents=True)
    raw_path = destination / "canonical_raw.jpg"
    Image.fromarray(_raw_image()).save(raw_path, format="JPEG", quality=95)
    canonical_path = destination / "canonical.jpg"
    shutil.copyfile(raw_path, canonical_path)
    mask_path = (
        destination
        / (
            "foreground_mask.png"
            if reference_type == "background"
            else "mask.png"
        )
    )
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    foreground_rgba_path: Path | None = None
    neutral_background_path: Path | None = None
    dinov3_embedding_path: Path | None = None
    if reference_type == "entity":
        raw = np.asarray(Image.open(raw_path).convert("RGB"))
        foreground_rgba_path = destination / "foreground_rgba.png"
        Image.fromarray(
            np.dstack((raw, mask.astype(np.uint8) * 255))
        ).save(foreground_rgba_path)
        neutral = np.full_like(raw, 204)
        neutral[mask] = raw[mask]
        neutral_background_path = destination / "neutral_background.jpg"
        Image.fromarray(neutral).save(
            neutral_background_path,
            format="JPEG",
            quality=95,
        )
        dinov3_embedding_path = destination / "dinov3_embedding.npy"
        np.save(
            dinov3_embedding_path,
            np.asarray([1.0, 0.0], dtype=np.float16),
        )
    metadata = {
        "clip_uid": "clip-1",
        "reference_id": reference_id,
        "reference_type": reference_type,
        "entity_id": None if reference_type == "background" else reference_id,
        "phrase": (
            "a brick courtyard"
            if reference_type == "background"
            else "a red bicycle"
        ),
        "canonical_label": reference_type,
        "category": reference_type,
        "ref_token": (
            "<ref_bg_1>"
            if reference_type == "background"
            else "<ref_subject_1>"
        ),
        "raw_canonical_path": str(raw_path),
        "canonical_path": str(canonical_path),
        "mask_path": str(mask_path),
        "foreground_rgba_path": (
            str(foreground_rgba_path)
            if foreground_rgba_path is not None
            else None
        ),
        "neutral_background_path": (
            str(neutral_background_path)
            if neutral_background_path is not None
            else None
        ),
        "dinov3_embedding_path": (
            str(dinov3_embedding_path)
            if dinov3_embedding_path is not None
            else None
        ),
        "source_frame_index": 17,
        "needs_inpainting": needs_inpainting,
        "inpainted": False,
        "status": (
            "pending_inpainting"
            if reference_type == "background" and needs_inpainting
            else "ready"
        ),
        "rejected": False,
        "visual_review": {"completeness": 0.9},
    }
    artifact = destination / (
        "reference_metadata.json"
        if reference_type == "background"
        else "metadata.json"
    )
    artifact.write_text(json.dumps(metadata), encoding="utf-8")
    return artifact


def _config(
    tmp_path: Path,
    *,
    maximum_hole_area_ratio: float = 0.5,
    maximum_generation_mask_area_ratio: float = 0.35,
    entity_enabled: bool = False,
    enabled: bool = True,
) -> PipelineConfig:
    return PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        inpainting=InpaintingConfig(
            enabled=enabled,
            backend="noop",
            mask_dilation_pixels=1,
            feather_pixels=1,
            background=InpaintingBackgroundConfig(
                enabled=True,
                maximum_hole_area_ratio=maximum_hole_area_ratio,
                maximum_generation_mask_area_ratio=(
                    maximum_generation_mask_area_ratio
                ),
            ),
            entity=InpaintingEntityConfig(
                enabled=entity_enabled,
                maximum_repair_area_ratio=0.08,
                maximum_component_area_ratio=0.05,
            ),
            consistency=InpaintingConsistencyConfig(
                fallback_to_raw=True,
            ),
        ),
    )


def _accept_consistency(**kwargs: object) -> dict[str, object]:
    del kwargs
    return {
        "accepted": True,
        "dino_similarity": 0.99,
        "rejection_reasons": [],
    }


def _dino_siglip_ranking() -> RankingConfig:
    return RankingConfig(
        evaluators=RankingEvaluatorsConfig(
            dinov3=DinoEvaluatorConfig(enabled=True),
            siglip2=SiglipEvaluatorConfig(enabled=True),
        )
    )


def test_disabled_inpainting_is_complete_noop(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)

    stats = run_inpainting(config)

    assert stats.skipped_disabled == 1
    assert not config.output_root.exists()


def test_direct_entity_production_rejects_missing_semantic_configuration(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "flux"
    model_path.mkdir()
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        inpainting=InpaintingConfig(
            enabled=True,
            backend="flux1_fill",
            model_path=model_path,
            background=InpaintingBackgroundConfig(enabled=False),
            entity=InpaintingEntityConfig(enabled=True),
        ),
    )

    with pytest.raises(ValueError, match="production inpainting requires"):
        run_inpainting(
            config,
            backend=_WholeImageBackend(),
            validator=_accept_consistency,
        )

    assert not config.output_root.exists()


def test_background_flux_rejects_dino_siglip_without_repair_judge(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        ranking=_dino_siglip_ranking(),
        inpainting=InpaintingConfig(
            enabled=True,
            backend="flux1_fill",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"background_hole_fill requires qwen\.repair_judge",
    ):
        run_inpainting(
            config,
            backend=_WholeImageBackend(),
            validator=_accept_consistency,
        )

    assert not config.output_root.exists()


def test_background_flux_with_repair_judge_is_allowed(tmp_path: Path) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        qwen=QwenServicesConfig(
            repair_judge=QwenConfig(model="repair-model")
        ),
        inpainting=InpaintingConfig(
            enabled=True,
            backend="flux1_fill",
        ),
    )
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
    )

    assert stats.repaired == 1


def test_entity_only_flux_keeps_dino_siglip_validation_path(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        ranking=_dino_siglip_ranking(),
        inpainting=InpaintingConfig(
            enabled=True,
            backend="flux1_fill",
            mask_dilation_pixels=1,
            feather_pixels=1,
            background=InpaintingBackgroundConfig(enabled=False),
            entity=InpaintingEntityConfig(
                enabled=True,
                maximum_repair_area_ratio=0.08,
                maximum_component_area_ratio=0.05,
            ),
        ),
    )
    subject_mask = np.zeros((32, 32), dtype=bool)
    subject_mask[5:27, 5:27] = True
    subject_mask[14:17, 14:17] = False
    _write_reference(
        config.output_root,
        reference_id="e1",
        reference_type="entity",
        mask=subject_mask,
        needs_inpainting=False,
    )

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
    )

    assert stats.repaired == 1


def test_generated_whole_image_cannot_change_pixels_outside_repair_mask(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )
    backend = _WholeImageBackend()

    stats = run_inpainting(
        config,
        backend=backend,
        validator=_accept_consistency,
    )

    metadata = json.loads(artifact.read_text(encoding="utf-8"))
    raw = np.asarray(Image.open(metadata["raw_canonical_path"]).convert("RGB"))
    repaired = np.asarray(Image.open(metadata["canonical_path"]).convert("RGB"))
    repair_mask = (
        np.asarray(
            Image.open(artifact.parent / "repair_mask.png").convert("L")
        )
        >= 128
    )
    assert stats.repaired == 1
    assert metadata["inpainted"] is True
    assert metadata["status"] == "ready"
    assert Path(metadata["canonical_path"]).suffix == ".png"
    assert Image.open(metadata["canonical_path"]).format == "PNG"
    assert np.array_equal(raw[~repair_mask], repaired[~repair_mask])
    inpainting_metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert inpainting_metadata["unmasked_l1_diff"] == 0.0
    assert len(inpainting_metadata["source_image_sha256"]) == 64
    assert len(inpainting_metadata["mask_sha256"]) == 64
    assert inpainting_metadata["source_frame_index"] == 17
    assert len(inpainting_metadata["config_fingerprint"]) == 64
    assert inpainting_metadata["reference_phrase"] == "a brick courtyard"
    assert inpainting_metadata["canonical_label"] == "background"
    assert inpainting_metadata["reference_type"] == "background"
    assert len(inpainting_metadata["inpainting_prompt_sha256"]) == 64
    assert len(inpainting_metadata["consistency_prompt_sha256"]) == 64
    assert len(inpainting_metadata["prompt_fingerprint"]) == 64
    assert inpainting_metadata["source_metadata_version"] == "4"
    assert len(inpainting_metadata["version_fingerprint"]) == 64
    assert (
        len(inpainting_metadata["local_consistency_prompt_sha256"]) == 64
    )
    assert inpainting_metadata["source_foreground_area_ratio"] == pytest.approx(
        16 / 1024
    )
    assert inpainting_metadata[
        "generation_mask_area_ratio"
    ] == pytest.approx(inpainting_metadata["repair_area_ratio"])
    assert Path(inpainting_metadata["source_mask_path"]) == Path(
        metadata["mask_path"]
    )
    generation_mask_path = Path(
        inpainting_metadata["generation_mask_path"]
    )
    assert generation_mask_path == artifact.parent / "generation_mask.png"
    assert generation_mask_path.is_file()
    candidate_path = Path(inpainting_metadata["candidate_path"])
    assert candidate_path == (
        artifact.parent / "canonical_repaired_candidate.png"
    )
    assert candidate_path.is_file()


def test_reference_semantics_and_prompt_fingerprint_invalidate_inpainting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )
    backend = _WholeImageBackend()

    first = run_inpainting(
        config,
        backend=backend,
        validator=_accept_consistency,
    )
    first_metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    reference = json.loads(artifact.read_text(encoding="utf-8"))
    reference["phrase"] = "a covered brick arcade"
    reference["canonical_label"] = "covered arcade"
    artifact.write_text(json.dumps(reference), encoding="utf-8")

    second = run_inpainting(
        config,
        backend=backend,
        validator=_accept_consistency,
    )
    second_metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert first.repaired == 1
    assert second.repaired == 1
    assert len(backend.prompts) == 2
    assert first_metadata["prompt_fingerprint"] != second_metadata[
        "prompt_fingerprint"
    ]
    assert second_metadata["reference_phrase"] == "a covered brick arcade"
    assert second_metadata["canonical_label"] == "covered arcade"


def test_source_foreground_area_over_threshold_does_not_call_backend(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, maximum_hole_area_ratio=0.23)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )
    backend = _WholeImageBackend()

    stats = run_inpainting(config, backend=backend)

    assert backend.prompts == []
    assert stats.rejected == 1
    assert stats.fallback_to_raw == 0
    reference = json.loads(artifact.read_text(encoding="utf-8"))
    assert reference["status"] == "rejected"
    assert reference["rejected"] is True
    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["rejection_reasons"] == [
        "source_foreground_area_ratio"
    ]
    assert metadata["source_foreground_area_ratio"] == pytest.approx(0.25)
    assert metadata["generation_mask_area_ratio"] is None
    assert metadata["repair_area_ratio"] == 0.0
    assert Path(metadata["source_mask_path"]) == (
        artifact.parent / "foreground_mask.png"
    )
    assert metadata["generation_mask_path"] is None


def test_source_mask_ratio_is_checked_before_generation_dilation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, maximum_hole_area_ratio=0.23)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:23, 8:23] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )
    backend = _WholeImageBackend()

    stats = run_inpainting(
        config,
        backend=backend,
        validator=_accept_consistency,
    )

    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert stats.repaired == 1
    assert len(backend.prompts) == 1
    assert metadata["source_foreground_area_ratio"] == pytest.approx(
        225 / 1024
    )
    assert metadata["generation_mask_area_ratio"] == pytest.approx(
        metadata["repair_area_ratio"]
    )
    assert metadata["repair_area_ratio"] > metadata[
        "source_foreground_area_ratio"
    ]
    assert metadata["repair_area_ratio"] > 0.23
    assert metadata["repair_area_ratio"] <= 0.35
    assert Path(metadata["generation_mask_path"]).is_file()


def test_generation_mask_area_has_independent_pre_flux_limit(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        maximum_hole_area_ratio=0.23,
        maximum_generation_mask_area_ratio=0.20,
    )
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:23, 8:23] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )
    backend = _WholeImageBackend()

    stats = run_inpainting(config, backend=backend)

    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats.rejected == 1
    assert backend.prompts == []
    assert metadata["source_foreground_area_ratio"] == pytest.approx(
        225 / 1024
    )
    assert metadata["generation_mask_area_ratio"] > 0.20
    assert metadata["rejection_reasons"] == ["generation_mask_area_ratio"]
    assert Path(metadata["generation_mask_path"]).is_file()


def test_person_silhouette_becomes_hull_like_generation_mask() -> None:
    source = np.zeros((100, 100), dtype=bool)
    source[15:30, 44:56] = True
    source[30:64, 47:53] = True
    source[36:42, 34:66] = True
    source[64:88, 38:46] = True
    source[64:88, 54:62] = True

    generation = _background_generation_mask(
        source,
        mask_dilation_pixels=1,
    )

    assert generation.sum() > source.sum()
    assert generation[72, 50]
    assert not source[72, 50]


def test_nearby_ship_components_merge_before_hull_construction() -> None:
    source = np.zeros((100, 100), dtype=bool)
    source[35:43, 25:39] = True
    source[45:53, 44:58] = True

    generation = _background_generation_mask(
        source,
        mask_dilation_pixels=1,
    )

    assert generation[44, 41]


def test_distant_components_keep_separate_generation_hulls() -> None:
    source = np.zeros((100, 100), dtype=bool)
    source[15:25, 10:20] = True
    source[75:85, 80:90] = True

    generation = _background_generation_mask(
        source,
        mask_dilation_pixels=1,
    )

    assert generation[20, 15]
    assert generation[80, 85]
    assert not generation[50, 50]


def test_consistency_failure_rejects_background_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=lambda **kwargs: {
            "accepted": False,
            "dino_similarity": 0.5,
            "rejection_reasons": ["new_salient_object"],
        },
    )

    metadata = json.loads(artifact.read_text(encoding="utf-8"))
    inpainting_metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    candidate_path = Path(inpainting_metadata["candidate_path"])
    assert stats.fallback_to_raw == 0
    assert metadata["inpainted"] is False
    assert metadata["status"] == "rejected"
    assert metadata["rejected"] is True
    assert candidate_path == (
        artifact.parent / "canonical_repaired_candidate.png"
    )
    assert candidate_path.is_file()
    assert Path(metadata["canonical_path"]).read_bytes() == Path(
        metadata["raw_canonical_path"]
    ).read_bytes()
    assert candidate_path.read_bytes() != Path(
        metadata["raw_canonical_path"]
    ).read_bytes()


def test_run_accepts_low_dino_for_background_hole_fill(tmp_path: Path) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=lambda **kwargs: {
            "accepted": True,
            "dino_similarity": 0.5,
            "rejection_reasons": [],
        },
    )

    reference = json.loads(artifact.read_text(encoding="utf-8"))
    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert stats.repaired == 1
    assert reference["inpainted"] is True
    assert metadata["dino_similarity"] == pytest.approx(0.5)
    assert "dino_similarity" not in metadata["rejection_reasons"]


def test_run_still_rejects_low_dino_for_entity_local_repair(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, entity_enabled=True)
    subject_mask = np.zeros((32, 32), dtype=bool)
    subject_mask[5:27, 5:27] = True
    subject_mask[14:17, 14:17] = False
    artifact = _write_reference(
        config.output_root,
        reference_id="e1",
        reference_type="entity",
        mask=subject_mask,
        needs_inpainting=False,
    )

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=lambda **kwargs: {
            "accepted": True,
            "dino_similarity": 0.5,
            "rejection_reasons": [],
        },
    )

    reference = json.loads(artifact.read_text(encoding="utf-8"))
    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert stats.repaired == 0
    assert stats.fallback_to_raw == 1
    assert reference["inpainted"] is False
    assert reference["status"] == "ready"
    assert "dino_similarity" in metadata["rejection_reasons"]


def test_background_and_entity_use_distinct_prompts(tmp_path: Path) -> None:
    config = _config(tmp_path, entity_enabled=True)
    background_mask = np.zeros((32, 32), dtype=bool)
    background_mask[12:16, 12:16] = True
    _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=background_mask,
        needs_inpainting=True,
    )
    subject_mask = np.zeros((32, 32), dtype=bool)
    subject_mask[5:27, 5:27] = True
    subject_mask[14:17, 14:17] = False
    _write_reference(
        config.output_root,
        reference_id="e1",
        reference_type="entity",
        mask=subject_mask,
        needs_inpainting=False,
    )
    backend = _WholeImageBackend()

    stats = run_inpainting(
        config,
        backend=backend,
        validator=_accept_consistency,
    )

    assert stats.repaired == 2
    background_prompt = next(
        prompt for prompt in backend.prompts if "Scene description:" in prompt
    )
    lowered = background_prompt.casefold()
    assert background_prompt.startswith(
        "The masked region is filled entirely with continuous background "
        "scenery and textures matching the surrounding environment."
    )
    assert len(background_prompt.split()) < 60
    for prohibited in (
        "remove",
        "subject",
        "person",
        "animal",
        "boat",
        "foreground object",
    ):
        assert prohibited not in lowered
    assert any("exact same entity identity" in prompt for prompt in backend.prompts)


def test_background_generation_prompt_is_positive_and_clip_safe() -> None:
    prompt = _inpainting_prompt(
        {
            "phrase": " ".join(
                ["brick", "courtyard", "arches", "sunlight"] * 20
            )
        },
        "background_hole_fill",
    )
    lowered = prompt.casefold()

    assert len(prompt.split()) < 60
    assert lowered.count("brick") == 6
    for prohibited in (
        "remove",
        "subject",
        "person",
        "animal",
        "boat",
        "foreground object",
        "do not",
    ):
        assert prohibited not in lowered


def test_entity_repair_regenerates_mask_derived_artifacts_and_dino(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, entity_enabled=True)
    subject_mask = np.zeros((32, 32), dtype=bool)
    subject_mask[5:27, 5:27] = True
    subject_mask[14:17, 14:17] = False
    artifact = _write_reference(
        config.output_root,
        reference_id="e1",
        reference_type="entity",
        mask=subject_mask,
        needs_inpainting=False,
    )
    stale_rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    Image.fromarray(stale_rgba).save(artifact.parent / "foreground_rgba.png")
    Image.new("RGB", (32, 32), (0, 0, 0)).save(
        artifact.parent / "neutral_background.jpg"
    )
    stale_embedding = artifact.parent / "dinov3_embedding.npy"
    np.save(stale_embedding, np.asarray([1.0, 0.0], dtype=np.float16))
    reference = json.loads(artifact.read_text(encoding="utf-8"))
    reference.update(
        {
            "foreground_rgba_path": str(
                artifact.parent / "foreground_rgba.png"
            ),
            "neutral_background_path": str(
                artifact.parent / "neutral_background.jpg"
            ),
            "dinov3_embedding_path": str(stale_embedding),
        }
    )
    artifact.write_text(json.dumps(reference), encoding="utf-8")

    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 1
            return np.asarray([[0.0, 3.0]], dtype=np.float32)

    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
        dino_embedder=_Dino(),  # type: ignore[arg-type]
    )

    updated = json.loads(artifact.read_text(encoding="utf-8"))
    repaired_mask = (
        np.asarray(Image.open(artifact.parent / "repair_mask.png").convert("L"))
        >= 128
    )
    expected_mask = subject_mask | repaired_mask
    saved_mask = np.asarray(Image.open(updated["mask_path"]).convert("L")) >= 128
    rgba = np.asarray(Image.open(updated["foreground_rgba_path"]).convert("RGBA"))
    neutral = np.asarray(
        Image.open(updated["neutral_background_path"]).convert("RGB")
    )
    embedding = np.load(updated["dinov3_embedding_path"], allow_pickle=False)
    inpainting_metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert stats.repaired == 1
    assert repaired_mask[15, 15]
    assert not repaired_mask[10, 10]
    assert inpainting_metadata["mode"] == "entity_local_repair"
    assert inpainting_metadata["source_foreground_area_ratio"] is None
    assert inpainting_metadata["generation_mask_area_ratio"] is None
    assert inpainting_metadata["generation_mask_path"] is None
    assert np.array_equal(saved_mask, expected_mask)
    assert np.array_equal(rgba[..., 3] >= 128, expected_mask)
    assert np.allclose(embedding.astype(np.float32), [0.0, 1.0], atol=1e-3)
    assert float(neutral[~expected_mask].mean()) == pytest.approx(204, abs=3)
    assert rgba[expected_mask, :3].mean() > 0


def test_entity_inpaint_overwrite_restores_all_immutable_raw_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, entity_enabled=True)
    subject_mask = np.zeros((32, 32), dtype=bool)
    subject_mask[5:27, 5:27] = True
    subject_mask[14:17, 14:17] = False
    artifact = _write_reference(
        config.output_root,
        reference_id="e1",
        reference_type="entity",
        mask=subject_mask,
        needs_inpainting=False,
    )

    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 1
            return np.asarray([[0.0, 3.0]], dtype=np.float32)

    dino = _Dino()
    first = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
        dino_embedder=dino,  # type: ignore[arg-type]
    )
    raw_names = (
        "mask_raw.png",
        "foreground_rgba_raw.png",
        "neutral_background_raw.jpg",
        "dinov3_embedding_raw.npy",
    )
    raw_bytes = {
        name: (artifact.parent / name).read_bytes() for name in raw_names
    }
    Image.new("L", (32, 32), 255).save(artifact.parent / "mask.png")
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(
        artifact.parent / "foreground_rgba.png"
    )
    Image.new("RGB", (32, 32), (0, 0, 0)).save(
        artifact.parent / "neutral_background.jpg"
    )
    np.save(
        artifact.parent / "dinov3_embedding.npy",
        np.asarray([0.0, 1.0], dtype=np.float16),
    )

    second = run_inpainting(
        config,
        overwrite=True,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
        dino_embedder=dino,  # type: ignore[arg-type]
    )

    repair_mask = (
        np.asarray(Image.open(artifact.parent / "repair_mask.png").convert("L"))
        >= 128
    )
    saved_mask = (
        np.asarray(Image.open(artifact.parent / "mask.png").convert("L")) >= 128
    )
    assert first.repaired == 1
    assert second.repaired == 1
    assert np.array_equal(saved_mask, subject_mask | repair_mask)
    assert {
        name: (artifact.parent / name).read_bytes() for name in raw_names
    } == raw_bytes
    restored_embedding = np.load(
        artifact.parent / "dinov3_embedding.npy",
        allow_pickle=False,
    )
    assert np.allclose(restored_embedding.astype(np.float32), [0.0, 1.0])


def test_flux_pads_non_square_inputs_and_unpads_output() -> None:
    config = InpaintingConfig(enabled=True)
    backend = Flux1FillBackend(config)

    class _Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            self.seed = seed
            return self

    class _Torch:
        Generator = _Generator

    class _Pipeline:
        def __init__(self) -> None:
            self.call: dict[str, object] = {}

        def __call__(self, **kwargs: object) -> object:
            self.call = kwargs
            return SimpleNamespace(
                images=[
                    Image.new(
                        "RGB",
                        (int(kwargs["width"]), int(kwargs["height"])),
                        (20, 30, 40),
                    )
                ]
            )

    pipeline = _Pipeline()
    backend._pipeline = pipeline
    backend._torch = _Torch()

    result = backend.inpaint(
        image=Image.new("RGB", (37, 23), (1, 2, 3)),
        mask=Image.new("L", (37, 23), 255),
        prompt="repair",
        seed=17,
    )

    assert pipeline.call["width"] == 48
    assert pipeline.call["height"] == 32
    assert pipeline.call["image"].size == (48, 32)
    assert pipeline.call["mask_image"].size == (48, 32)
    assert result.size == (37, 23)


def test_production_default_rejects_without_semantic_models(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    stats = run_inpainting(config, backend=_WholeImageBackend())

    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert stats.repaired == 0
    assert "dinov3_validator_unavailable" in metadata["rejection_reasons"]
    assert "siglip_validator_unavailable" in metadata["rejection_reasons"]


def test_production_validator_uses_dino_and_siglip(tmp_path: Path) -> None:
    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 2
            return np.asarray([[1.0, 0.0], [0.999, 0.001]])

    class _Siglip:
        def score(
            self,
            images: list[Image.Image],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            assert len(images) == 2
            assert target_text == "a brick courtyard"
            assert distractor_texts == []
            return [
                SimpleNamespace(target_similarity=0.8),
                SimpleNamespace(target_similarity=0.79),
            ]

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
    )
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.new("L", (17, 13)),
        reference={"phrase": "a brick courtyard"},
        mode="background_hole_fill",
    )

    assert result["accepted"] is True
    assert result["dino_similarity"] > 0.99
    assert result["repaired_siglip_similarity"] == 0.79


def _background_review(
    **overrides: object,
) -> BackgroundInpaintingReview:
    values: dict[str, object] = {
        "background_continuity_preserved": True,
        "masked_foreground_removed": True,
        "reference_phrase_supported": True,
        "new_salient_objects": False,
        "visible_seam_or_artifact": False,
        "reason": "coherent background replacement",
    }
    values.update(overrides)
    return BackgroundInpaintingReview.model_validate(values)


def test_qwen_uses_dedicated_background_review_prompt_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = QwenInpaintingConsistencyJudge(QwenConfig())
    captured: list[dict[str, object]] = []

    def fake_request(**kwargs: object) -> str:
        captured.append(dict(kwargs))
        return _background_review().model_dump_json()

    monkeypatch.setattr(judge, "_request", fake_request)
    full_review = judge.review(
        original_path=tmp_path / "original.png",
        repaired_path=tmp_path / "repaired.png",
        repair_mask_path=tmp_path / "mask.png",
        reference_phrase="a brick courtyard",
        mode="background_hole_fill",
    )
    local_review = judge.review(
        original_path=tmp_path / "local-original.png",
        repaired_path=tmp_path / "local-repaired.png",
        repair_mask_path=tmp_path / "local-mask.png",
        reference_phrase="a brick courtyard",
        mode="background_hole_fill_local",
    )

    assert isinstance(full_review, BackgroundInpaintingReview)
    assert isinstance(local_review, BackgroundInpaintingReview)
    assert all(
        request["review_model"] is BackgroundInpaintingReview
        for request in captured
    )
    full_prompt = " ".join(str(captured[0]["prompt"]).split())
    assert "intentionally contains a foreground object" in full_prompt
    assert "not a semantic inconsistency" in full_prompt
    assert "original masked foreground remains" in full_prompt
    assert "another entity replaces it" in full_prompt
    assert "obvious seam or visual artifact" in full_prompt
    local_prompt = " ".join(str(captured[1]["prompt"]).split())
    for expected in (
        "person",
        "animal",
        "vehicle",
        "vessel",
        "product",
        "face",
        "body silhouette",
        "distinct foreground object",
        "blurred ghost",
        "repeated texture",
        "inset image",
        "artificial blob",
        "visible boundary",
    ):
        assert expected in local_prompt


def test_background_qwen_accepts_expected_masked_foreground_removal(
    tmp_path: Path,
) -> None:
    review = _background_review()

    class _Qwen:
        def review(self, **kwargs: object) -> BackgroundInpaintingReview:
            assert kwargs["mode"] in {
                "background_hole_fill",
                "background_hole_fill_local",
            }
            return review

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        qwen_judge=_Qwen(),  # type: ignore[arg-type]
    )
    mask = np.zeros((13, 17), dtype=np.uint8)
    mask[4:9, 6:11] = 255
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.fromarray(mask),
        reference={
            "phrase": "a brick courtyard",
            "raw_canonical_path": str(tmp_path / "canonical_raw.jpg"),
        },
        mode="background_hole_fill",
    )

    assert result["accepted"] is True
    assert result["qwen_review"] is None
    assert "same_semantic_content" not in result["qwen_full_review"]
    assert "identity_preserved" not in result["qwen_full_review"]
    assert "same_semantic_content" not in result["qwen_local_review"]
    assert "identity_preserved" not in result["qwen_local_review"]


@pytest.mark.parametrize(
    "review",
    [
        pytest.param(
            _background_review(
                masked_foreground_removed=False,
                reason="the original foreground remains",
            ),
            id="retained-masked-foreground",
        ),
        pytest.param(
            _background_review(
                new_salient_objects=True,
                reason="a replacement entity appears",
            ),
            id="replacement-entity",
        ),
        pytest.param(
            _background_review(
                new_salient_objects=True,
                reason="a new salient object was added",
            ),
            id="new-salient-object",
        ),
        pytest.param(
            _background_review(
                visible_seam_or_artifact=True,
                reason="a visible seam surrounds the fill",
            ),
            id="visible-seam",
        ),
    ],
)
def test_background_qwen_rejects_invalid_hole_fills(
    tmp_path: Path,
    review: BackgroundInpaintingReview,
) -> None:
    class _Qwen:
        def review(self, **kwargs: object) -> BackgroundInpaintingReview:
            assert kwargs["mode"] in {
                "background_hole_fill",
                "background_hole_fill_local",
            }
            return review

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        qwen_judge=_Qwen(),  # type: ignore[arg-type]
    )
    mask = np.zeros((13, 17), dtype=np.uint8)
    mask[4:9, 6:11] = 255
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.fromarray(mask),
        reference={
            "phrase": "a brick courtyard",
            "raw_canonical_path": str(tmp_path / "canonical_raw.jpg"),
        },
        mode="background_hole_fill",
    )

    assert result["accepted"] is False
    assert "qwen_background_consistency" in result["rejection_reasons"]


@pytest.mark.parametrize(
    ("full_passes", "local_passes", "expected_accepted"),
    [
        pytest.param(True, False, False, id="local-failure-rejects"),
        pytest.param(False, True, False, id="full-failure-rejects"),
        pytest.param(True, True, True, id="both-reviews-pass"),
    ],
)
def test_background_requires_full_and_local_qwen_reviews(
    tmp_path: Path,
    full_passes: bool,
    local_passes: bool,
    expected_accepted: bool,
) -> None:
    reviews = {
        "background_hole_fill": _background_review(
            visible_seam_or_artifact=not full_passes,
            reason="full pass" if full_passes else "full-frame seam",
        ),
        "background_hole_fill_local": _background_review(
            masked_foreground_removed=local_passes,
            reason="local pass" if local_passes else "local ghost remains",
        ),
    }

    class _Qwen:
        def __init__(self) -> None:
            self.image_sizes: dict[str, tuple[int, int]] = {}

        def review(self, **kwargs: object) -> BackgroundInpaintingReview:
            mode = str(kwargs["mode"])
            self.image_sizes[mode] = Image.open(
                Path(str(kwargs["original_path"]))
            ).size
            return reviews[mode]

    qwen = _Qwen()
    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        qwen_judge=qwen,  # type: ignore[arg-type]
    )
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 48:72] = 255

    result = validator(
        original=Image.new("RGB", (120, 80)),
        repaired=Image.new("RGB", (120, 80)),
        repair_mask=Image.fromarray(mask),
        reference={
            "phrase": "a brick courtyard",
            "raw_canonical_path": str(tmp_path / "canonical_raw.jpg"),
        },
        mode="background_hole_fill",
    )

    assert result["accepted"] is expected_accepted
    assert result["qwen_full_review"]["reason"] == reviews[
        "background_hole_fill"
    ].reason
    assert result["qwen_local_review"]["reason"] == reviews[
        "background_hole_fill_local"
    ].reason
    assert result["qwen_local_crop_box"] == (42, 25, 78, 55)
    assert qwen.image_sizes["background_hole_fill"] == (120, 80)
    assert max(qwen.image_sizes["background_hole_fill_local"]) >= 768
    if expected_accepted:
        assert "qwen_background_consistency" not in result[
            "rejection_reasons"
        ]
    else:
        assert "qwen_background_consistency" in result["rejection_reasons"]


def test_inpainting_metadata_stores_both_background_reviews(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    class _Qwen:
        def review(self, **kwargs: object) -> BackgroundInpaintingReview:
            return _background_review(reason=f"{kwargs['mode']} passed")

    validator = ProductionConsistencyValidator(
        config,
        qwen_judge=_Qwen(),  # type: ignore[arg-type]
    )
    stats = run_inpainting(
        config,
        backend=_WholeImageBackend(),
        validator=validator,
    )

    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats.repaired == 1
    assert metadata["validator"]["qwen_full_review"]["reason"] == (
        "background_hole_fill passed"
    )
    assert metadata["validator"]["qwen_local_review"]["reason"] == (
        "background_hole_fill_local passed"
    )


def test_low_dino_is_diagnostic_only_for_background_hole_fill(
    tmp_path: Path,
) -> None:
    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 2
            return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    class _Siglip:
        def score(
            self,
            images: list[Image.Image],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            del images, target_text, distractor_texts
            return [
                SimpleNamespace(target_similarity=0.8),
                SimpleNamespace(target_similarity=0.8),
            ]

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
    )
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.new("L", (17, 13)),
        reference={"phrase": "a brick courtyard"},
        mode="background_hole_fill",
    )

    assert result["accepted"] is True
    assert result["dino_similarity"] == pytest.approx(0.0)
    assert "dino_similarity" not in result["rejection_reasons"]


def test_low_dino_still_rejects_entity_local_repair(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    Image.new("L", (17, 13), 255).save(mask_path)

    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 2
            return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    class _Siglip:
        def score(
            self,
            images: list[Image.Image],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            del images, target_text, distractor_texts
            return [
                SimpleNamespace(target_similarity=0.8),
                SimpleNamespace(target_similarity=0.8),
            ]

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
    )
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.new("L", (17, 13)),
        reference={
            "phrase": "a red bicycle",
            "mask_path": str(mask_path),
        },
        mode="entity_local_repair",
    )

    assert result["accepted"] is False
    assert result["dino_similarity"] == pytest.approx(0.0)
    assert "dino_similarity" in result["rejection_reasons"]


def test_negative_siglip_absolute_score_does_not_reject_background(
    tmp_path: Path,
) -> None:
    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            assert len(images) == 2
            return np.asarray([[1.0, 0.0], [1.0, 0.0]])

    class _Siglip:
        def score(
            self,
            images: list[Image.Image],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            del images, target_text, distractor_texts
            return [
                SimpleNamespace(target_similarity=-0.50),
                SimpleNamespace(target_similarity=-0.51),
            ]

    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
    )
    result = validator(
        original=Image.new("RGB", (17, 13)),
        repaired=Image.new("RGB", (17, 13)),
        repair_mask=Image.new("L", (17, 13)),
        reference={"phrase": "a brick courtyard"},
        mode="background_hole_fill",
    )

    assert result["accepted"] is True
    assert result["repaired_siglip_similarity"] == pytest.approx(-0.51)
    assert "siglip_similarity" not in result["rejection_reasons"]


def test_entity_semantic_validation_uses_neutral_crops_and_qwen_mask(
    tmp_path: Path,
) -> None:
    subject_mask = np.zeros((13, 17), dtype=bool)
    subject_mask[3:11, 4:14] = True
    subject_mask[6:8, 8:10] = False
    repair_mask = np.zeros_like(subject_mask)
    repair_mask[6:8, 8:10] = True
    mask_path = tmp_path / "mask_raw.png"
    Image.fromarray(subject_mask.astype(np.uint8) * 255).save(mask_path)
    original = np.full((13, 17, 3), 20, dtype=np.uint8)
    original[subject_mask] = (90, 100, 110)
    repaired = original.copy()
    repaired[repair_mask] = (220, 30, 40)
    semantic_images: list[np.ndarray] = []

    class _Dino:
        def encode(self, images: list[Image.Image]) -> np.ndarray:
            semantic_images.extend(np.asarray(image) for image in images)
            return np.asarray([[1.0, 0.0], [0.999, 0.001]])

    class _Siglip:
        def score(
            self,
            images: list[Image.Image],
            target_text: str,
            distractor_texts: list[str],
        ) -> list[object]:
            assert all(
                np.array_equal(np.asarray(image), expected)
                for image, expected in zip(images, semantic_images)
            )
            assert target_text == "a red bicycle"
            assert distractor_texts == []
            return [
                SimpleNamespace(target_similarity=0.8),
                SimpleNamespace(target_similarity=0.8),
            ]

    class _Qwen:
        def __init__(self) -> None:
            self.mask: np.ndarray | None = None

        def review(
            self,
            *,
            original_path: Path,
            repaired_path: Path,
            repair_mask_path: Path,
            reference_phrase: str,
            mode: str,
        ) -> InpaintingSemanticReview:
            assert original_path.is_file()
            assert repaired_path.is_file()
            assert reference_phrase == "a red bicycle"
            assert mode == "entity_local_repair"
            self.mask = np.asarray(
                Image.open(repair_mask_path).convert("L")
            )
            return InpaintingSemanticReview(
                same_semantic_content=True,
                identity_preserved=True,
                reference_phrase_supported=True,
                new_salient_objects=False,
                reason="consistent",
            )

    qwen = _Qwen()
    validator = ProductionConsistencyValidator(
        _config(tmp_path),
        dino_embedder=_Dino(),  # type: ignore[arg-type]
        siglip_aligner=_Siglip(),  # type: ignore[arg-type]
        qwen_judge=qwen,  # type: ignore[arg-type]
    )
    result = validator(
        original=Image.fromarray(original),
        repaired=Image.fromarray(repaired),
        repair_mask=Image.fromarray(repair_mask.astype(np.uint8) * 255),
        reference={
            "phrase": "a red bicycle",
            "mask_raw_path": str(mask_path),
            "raw_canonical_path": str(tmp_path / "canonical_raw.jpg"),
        },
        mode="entity_local_repair",
    )

    expected_original = np.full_like(original, 204)
    expected_original[subject_mask] = original[subject_mask]
    expected_repaired = np.full_like(repaired, 204)
    expected_repaired[subject_mask | repair_mask] = repaired[
        subject_mask | repair_mask
    ]
    assert result["accepted"] is True
    assert result["semantic_input"] == "masked_neutral_crop"
    assert np.array_equal(semantic_images[0], expected_original)
    assert np.array_equal(semantic_images[1], expected_repaired)
    assert qwen.mask is not None
    assert np.array_equal(qwen.mask >= 128, repair_mask)


def test_noop_backend_never_marks_reference_inpainted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=True,
    )

    stats = run_inpainting(
        config,
        backend=NoOpInpaintBackend(),
        validator=_accept_consistency,
    )

    reference = json.loads(artifact.read_text(encoding="utf-8"))
    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert stats.repaired == 0
    assert reference["inpainted"] is False
    assert reference["status"] == "rejected"
    assert "noop_backend_test_only" in metadata["rejection_reasons"]


def test_overwrite_clears_stale_repair_and_restores_raw(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask = np.zeros((32, 32), dtype=bool)
    artifact = _write_reference(
        config.output_root,
        reference_id="bg1",
        reference_type="background",
        mask=mask,
        needs_inpainting=False,
    )
    stale_repaired = artifact.parent / "canonical_repaired.png"
    stale_candidate = artifact.parent / "canonical_repaired_candidate.png"
    stale_generation_mask = artifact.parent / "generation_mask.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(stale_repaired)
    Image.new("RGB", (32, 32), (0, 255, 0)).save(stale_candidate)
    Image.new("L", (32, 32), 255).save(stale_generation_mask)
    Image.new("L", (32, 32), 255).save(artifact.parent / "repair_mask.png")
    (artifact.parent / "inpainting_metadata.json").write_text(
        json.dumps({"accepted": True}),
        encoding="utf-8",
    )
    reference = json.loads(artifact.read_text(encoding="utf-8"))
    reference.update(
        {
            "canonical_path": str(stale_repaired),
            "inpainted": True,
            "status": "ready",
            "inpainting_metadata_path": str(
                artifact.parent / "inpainting_metadata.json"
            ),
        }
    )
    artifact.write_text(json.dumps(reference), encoding="utf-8")

    stats = run_inpainting(
        config,
        overwrite=True,
        backend=_WholeImageBackend(),
        validator=_accept_consistency,
    )

    restored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stats.skipped_no_repair_needed == 1
    assert not stale_repaired.exists()
    assert not stale_candidate.exists()
    assert not stale_generation_mask.exists()
    assert not (artifact.parent / "repair_mask.png").exists()
    assert not (artifact.parent / "inpainting_metadata.json").exists()
    assert restored["inpainted"] is False
    assert restored["status"] == "ready"
    assert Path(restored["canonical_path"]).read_bytes() == Path(
        restored["raw_canonical_path"]
    ).read_bytes()


def test_missing_flux_dependencies_report_optional_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "flux"
    model_path.mkdir()
    config = InpaintingConfig(enabled=True, model_path=model_path)
    backend = Flux1FillBackend(config)
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "diffusers":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(InpaintingDependencyError, match="requirements-inpaint.txt"):
        backend.inpaint(
            image=Image.new("RGB", (8, 8)),
            mask=Image.new("L", (8, 8)),
            prompt="repair",
            seed=1,
        )
