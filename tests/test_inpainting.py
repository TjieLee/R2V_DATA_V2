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
    InpaintingBackgroundConfig,
    InpaintingConfig,
    InpaintingConsistencyConfig,
    InpaintingEntityConfig,
    PipelineConfig,
)
from r2v_data_v2.inpainting import (
    Flux1FillBackend,
    InpaintingDependencyError,
    NoOpInpaintBackend,
    ProductionConsistencyValidator,
    run_inpainting,
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


def test_disabled_inpainting_is_complete_noop(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)

    stats = run_inpainting(config)

    assert stats.skipped_disabled == 1
    assert not config.output_root.exists()


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


def test_repair_area_over_threshold_does_not_call_backend(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, maximum_hole_area_ratio=0.02)
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
    assert metadata["rejection_reasons"] == ["repair_area_ratio"]


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
    assert stats.fallback_to_raw == 0
    assert metadata["inpainted"] is False
    assert metadata["status"] == "rejected"
    assert metadata["rejected"] is True
    assert Path(metadata["canonical_path"]).read_bytes() == Path(
        metadata["raw_canonical_path"]
    ).read_bytes()


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
    assert any("foreground subjects" in prompt for prompt in backend.prompts)
    assert any("exact same entity identity" in prompt for prompt in backend.prompts)


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
    Image.new("RGB", (32, 32), (255, 0, 0)).save(stale_repaired)
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
