from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.naming import parse_clip_identity

JEA_VIDEO_MOTION_ADAPTER = "jea_video_motion_v1"

_PROVENANCE_FIELDS = (
    "source_video_id",
    "source_video_path",
    "shot_index",
    "start_frame",
    "end_frame",
    "num_frames",
    "start_time",
    "end_time",
    "duration",
    "aesthetic_score_siglip",
    "aesthetic_score_peak_end",
    "quality_score_qalign",
    "watermark_crop",
    "black_bar_detection",
    "subtitle_crop",
    "motion_score",
)


def _dataset_subdirectory(path: str | Path, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if not resolved.is_dir() or dataset_root not in resolved.parents:
        raise ValueError(f"{field_name} must be a directory below public dataset")
    return resolved


def _path_below_root(
    value: object,
    *,
    root: Path,
    field_name: str,
    require_file: bool,
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record.{field_name} must be a non-empty path")
    raw_path = Path(value).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    unresolved = candidate.resolve(strict=False)
    try:
        unresolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"record.{field_name} escapes its configured root") from exc
    resolved = candidate.resolve(strict=require_file)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"record.{field_name} escapes its configured root") from exc
    if not relative.parts:
        raise ValueError(f"record.{field_name} must identify a file below its root")
    if require_file and not resolved.is_file():
        raise FileNotFoundError(f"record.{field_name} is not a file: {resolved}")
    return resolved, relative.as_posix()


def _caption_raw(raw: dict[str, Any]) -> str:
    for field_name in ("caption_raw", "caption", "text"):
        value = raw.get(field_name)
        if isinstance(value, str):
            return value
    return ""


@dataclass(frozen=True)
class JeaVideoMotionAdapter:
    clips_root: Path
    source_videos_root: Path

    @classmethod
    def create(
        cls,
        *,
        clips_root: str | Path,
        source_videos_root: str | Path,
    ) -> JeaVideoMotionAdapter:
        return cls(
            clips_root=_dataset_subdirectory(clips_root, "clips_root"),
            source_videos_root=_dataset_subdirectory(
                source_videos_root,
                "source_videos_root",
            ),
        )

    def parse(
        self,
        raw: dict[str, Any],
        *,
        source_index: int,
    ) -> dict[str, object]:
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        video_path = self.resolve_clip_path(raw)
        _, source_relative_video_path = _path_below_root(
            raw.get("source_video_path"),
            root=self.source_videos_root,
            field_name="source_video_path",
            require_file=False,
        )

        source_video_id = raw.get("source_video_id")
        if not isinstance(source_video_id, str) or not source_video_id:
            raise ValueError("record.source_video_id must be a non-empty string")
        shot_index = raw.get("shot_index")
        if (
            not isinstance(shot_index, int)
            or isinstance(shot_index, bool)
            or shot_index < 0
        ):
            raise ValueError("record.shot_index must be a non-negative integer")

        identity = parse_clip_identity(video_path)
        if identity.parent_video_id != source_video_id:
            raise ValueError(
                "record.source_video_id does not match the clip filename parent"
            )
        if (
            not identity.clip_suffix.isdigit()
            or int(identity.clip_suffix) != shot_index
        ):
            raise ValueError(
                "record.shot_index does not match the clip filename suffix"
            )

        metadata = {
            field_name: raw.get(field_name)
            for field_name in _PROVENANCE_FIELDS
            if field_name in raw
        }
        metadata.update(
            {
                "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
                "source_relative_video_path": source_relative_video_path,
            }
        )
        return {
            "source_index": source_index,
            "clip_uid": identity.clip_uid,
            "parent_video_id": identity.parent_video_id,
            "video_path": str(video_path),
            "caption_raw": _caption_raw(raw),
            "metadata": metadata,
            "clip_suffix": identity.clip_suffix,
        }

    def resolve_clip_path(self, raw: dict[str, Any]) -> Path:
        video_path, _ = _path_below_root(
            raw.get("video_path"),
            root=self.clips_root,
            field_name="video_path",
            require_file=True,
        )
        return video_path


def parse_jea_video_motion_v1(
    raw: dict[str, Any],
    *,
    source_index: int,
    clips_root: str | Path,
    source_videos_root: str | Path,
) -> dict[str, object]:
    """Adapt one JEA motion record to the existing fixed_selection_v1 shape."""

    return JeaVideoMotionAdapter.create(
        clips_root=clips_root,
        source_videos_root=source_videos_root,
    ).parse(raw, source_index=source_index)
