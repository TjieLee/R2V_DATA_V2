from __future__ import annotations

import os
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image

from r2v_data_v2.v3.reference_edit_boogu import (
    BooguSubprocessBackend,
    BooguWorkerConfig,
    resolve_boogu_1k_size,
)

if TYPE_CHECKING:
    from r2v_data_v2.v3.config import V3Config
    from r2v_data_v2.v3.storage import RunStorage


def build_boogu_background_removal_prompt(
    removal_phrases: list[str],
) -> str:
    phrases = [phrase.strip() for phrase in removal_phrases if phrase.strip()]
    if not phrases:
        raise ValueError("removal_phrases must contain at least one phrase")
    return (
        "Remove the following foreground entities from the image: "
        f"{', '.join(phrases)}.\n"
        "Fill the removed areas naturally with the surrounding background."
    )


class BooguBackgroundRemovalBackend:
    """Thin BackgroundRemovalBackend adapter over the persistent Boogu worker."""

    def __init__(
        self,
        backend: BooguSubprocessBackend,
        *,
        target_area: int,
        alignment: int,
    ) -> None:
        self.backend = backend
        self.target_area = target_area
        self.alignment = alignment

    def generation_size(self, image: Image.Image) -> tuple[int, int]:
        return resolve_boogu_1k_size(
            image.width,
            image.height,
            target_area=self.target_area,
            alignment=self.alignment,
        )

    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        del background_phrase
        if image.mode != "RGB":
            raise ValueError("Boogu background removal input must be RGB")
        expected_prompt = build_boogu_background_removal_prompt(removal_phrases)
        if prompt != expected_prompt:
            raise ValueError("Boogu background removal prompt is not canonical")
        width, height = self.generation_size(image)
        output = self.backend.edit(
            source_rgb=image,
            instruction=prompt,
            width=width,
            height=height,
            thinking_enabled=False,
            seed=seed,
        )
        with Image.open(BytesIO(output.png_bytes)) as loaded:
            loaded.load()
            if loaded.mode != "RGB":
                raise ValueError("Boogu background removal output must be RGB")
            candidate = loaded.copy()
        if candidate.size != (width, height):
            raise ValueError("Boogu background removal output dimensions changed")
        if candidate.size != image.size:
            candidate = candidate.resize(image.size, Image.Resampling.LANCZOS)
        return candidate

    def close(self) -> None:
        self.backend.close()


def create_boogu_background_removal_backend(
    config: V3Config,
    storage: RunStorage,
) -> BooguBackgroundRemovalBackend:
    from r2v_data_v2.v3 import config as config_module

    physical_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical_device is None or not physical_device.isdigit():
        raise ValueError("Boogu remove worker requires one visible physical GPU")
    worker = BooguSubprocessBackend(
        BooguWorkerConfig(
            python_executable=config.reference_edit.python_executable,
            code_root=config.reference_edit.code_root,
            model_path=config.reference_edit.model_path,
            model_revision=config.reference_edit.model_revision,
            device="cuda:0",
            timeout_seconds=config.reference_edit.timeout_seconds,
            cuda_visible_devices=physical_device,
            allowed_server_root=config_module.ALLOWED_WRITABLE_ROOT,
            temporary_root=storage.boogu_remove_temporary_dir(),
        )
    )
    worker.start(stderr_log_path=storage.boogu_remove_worker_log_path())
    return BooguBackgroundRemovalBackend(
        worker,
        target_area=config.reference_edit.target_area,
        alignment=config.reference_edit.alignment,
    )
