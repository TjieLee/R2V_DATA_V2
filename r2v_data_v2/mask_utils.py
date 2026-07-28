from __future__ import annotations

import base64
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    packed = np.packbits(binary.reshape(-1), bitorder="little")
    compressed = zlib.compress(packed.tobytes(), level=9)
    return {
        "shape": [int(binary.shape[0]), int(binary.shape[1])],
        "encoding": "packbits_zlib_base64",
        "data": base64.b64encode(compressed).decode(),
    }


def decode_mask(value: dict[str, object]) -> np.ndarray:
    if value.get("encoding") != "packbits_zlib_base64":
        raise ValueError("unsupported mask encoding")
    shape = value.get("shape")
    data = value.get("data")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(item, int) and item > 0 for item in shape)
        or not isinstance(data, str)
    ):
        raise ValueError("invalid encoded mask")
    height, width = shape
    packed = np.frombuffer(zlib.decompress(base64.b64decode(data)), dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")
    required = height * width
    if unpacked.size < required:
        raise ValueError("encoded mask data is truncated")
    return unpacked[:required].reshape(height, width).astype(bool)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(binary)
    if x.size == 0:
        raise ValueError("cannot calculate a bounding box for an empty mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def touches_border(mask: np.ndarray) -> bool:
    binary = np.asarray(mask, dtype=bool)
    return bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )


def fill_small_enclosed_holes(
    mask: np.ndarray,
    *,
    frame: np.ndarray | None = None,
    maximum_area_ratio: float = 0.002,
) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    foreground_area = int(binary.sum())
    if foreground_area == 0:
        return binary.copy()
    inverse = (~binary).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = binary.copy()
    gradient: np.ndarray | None = None
    gradient_limit = float("inf")
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        gradient = cv2.magnitude(gx, gy)
        gradient_limit = float(np.percentile(gradient, 75))
    height, width = binary.shape
    maximum_area = max(1, round(foreground_area * maximum_area_ratio))
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = stats[label]
        enclosed = (
            x > 0
            and y > 0
            and x + component_width < width
            and y + component_height < height
        )
        if not enclosed or area > maximum_area:
            continue
        component = labels == label
        if gradient is not None and float(gradient[component].mean()) > gradient_limit:
            continue
        result[component] = True
    return result


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    binary = np.asarray(mask, dtype=bool)
    Image.fromarray(binary.astype(np.uint8) * 255).save(path)


def save_mask_contact_sheet(
    *,
    frame_paths: list[Path],
    candidates: list[dict[str, object]],
    masks: dict[int, np.ndarray],
    destination: Path,
) -> None:
    tiles: list[Image.Image] = []
    by_slot = {int(candidate["frame_slot"]): candidate for candidate in candidates}
    for slot, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        candidate = by_slot.get(slot)
        if candidate is not None and slot in masks:
            mask = masks[slot]
            overlay = np.asarray(image).copy()
            overlay[mask] = (
                0.45 * overlay[mask] + 0.55 * np.array([255, 60, 60])
            ).astype(np.uint8)
            image = Image.fromarray(overlay)
            draw = ImageDraw.Draw(image)
            draw.rectangle(candidate["bbox_xyxy"], outline=(0, 255, 0), width=3)
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"slot {slot}", fill=(255, 255, 0))
        tiles.append(image)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    columns = min(5, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), color=(20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)
