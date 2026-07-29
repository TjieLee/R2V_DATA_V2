from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

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
    Image.fromarray(_raw_image()).save(raw_path, format="PNG")
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

    stats = run_inpainting(config, backend=backend)

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
    assert stats.fallback_to_raw == 1
    metadata = json.loads(
        (artifact.parent / "inpainting_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["rejection_reasons"] == ["repair_area_ratio"]


def test_consistency_failure_falls_back_to_raw(tmp_path: Path) -> None:
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
    assert stats.fallback_to_raw == 1
    assert metadata["inpainted"] is False
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

    stats = run_inpainting(config, backend=backend)

    assert stats.repaired == 2
    assert any("foreground subjects" in prompt for prompt in backend.prompts)
    assert any("exact same entity identity" in prompt for prompt in backend.prompts)


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
