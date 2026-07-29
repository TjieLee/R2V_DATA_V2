from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

_IMAGE_MIME_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def image_format(path: str | Path) -> str:
    image_path = Path(path)
    with Image.open(image_path) as image:
        detected = image.format
    if detected is None:
        raise ValueError(f"unable to detect image format: {image_path}")
    return detected.upper()


def image_mime_type(path: str | Path) -> str:
    detected = image_format(path)
    try:
        return _IMAGE_MIME_TYPES[detected]
    except KeyError as exc:
        raise ValueError(
            f"unsupported image format for data URI: {detected!r}"
        ) from exc


def image_extension(path: str | Path) -> str:
    detected = image_format(path)
    try:
        return _IMAGE_EXTENSIONS[detected]
    except KeyError as exc:
        raise ValueError(f"unsupported image format: {detected!r}") from exc


def image_data_uri(path: str | Path) -> str:
    image_path = Path(path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{image_mime_type(image_path)};base64,{encoded}"
