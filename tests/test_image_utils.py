from __future__ import annotations

from pathlib import Path

from PIL import Image

from r2v_data_v2.image_utils import image_data_uri, image_mime_type


def test_data_uri_mime_comes_from_image_bytes(tmp_path: Path) -> None:
    misleading_path = tmp_path / "reference.jpg"
    Image.new("RGB", (8, 6), (20, 40, 60)).save(
        misleading_path,
        format="PNG",
    )

    assert image_mime_type(misleading_path) == "image/png"
    assert image_data_uri(misleading_path).startswith("data:image/png;base64,")
