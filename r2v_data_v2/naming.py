from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_CLIP_PATTERN = re.compile(r"^(?P<parent>.+?)_(?P<suffix>\d+(?:_\d+)*)$")


@dataclass(frozen=True)
class ClipIdentity:
    clip_uid: str
    parent_video_id: str
    clip_suffix: str
    clip_order: tuple[int, ...]


def clip_uid(video_path: str | Path) -> str:
    normalized = str(Path(video_path).expanduser().resolve(strict=False))
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


def parse_clip_identity(video_path: str | Path) -> ClipIdentity:
    path = Path(video_path).expanduser().resolve(strict=False)
    match = _CLIP_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"video filename must end in a numeric clip suffix: {path.name}"
        )
    suffix = match.group("suffix")
    return ClipIdentity(
        clip_uid=clip_uid(path),
        parent_video_id=match.group("parent"),
        clip_suffix=suffix,
        clip_order=tuple(int(part) for part in suffix.split("_")),
    )
