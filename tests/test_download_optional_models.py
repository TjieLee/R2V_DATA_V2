from __future__ import annotations

from pathlib import Path

import pytest

import scripts.download_optional_models as downloader


def test_download_destination_rejects_public_and_arbitrary_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="destination must be inside"):
        downloader.validate_destination(
            Path("/mnt/workspace/public/pretrained/siglip2")
        )
    with pytest.raises(ValueError, match="destination must be inside"):
        downloader.validate_destination(tmp_path / "model")


def test_download_destination_accepts_user_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = (tmp_path / "models").resolve()
    cache_root = (tmp_path / "cache").resolve()
    monkeypatch.setattr(
        downloader,
        "ALLOWED_DESTINATION_ROOTS",
        (model_root, cache_root),
    )

    assert downloader.validate_destination(model_root / "siglip2") == (
        model_root / "siglip2"
    )
