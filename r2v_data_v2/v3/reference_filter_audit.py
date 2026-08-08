from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import select
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.schemas import AnnotationEntity, RawEntityReferenceDecision
from r2v_data_v2.v3.storage import RunStorage

ALLOWED_AUDIT_ROOT = Path("/mnt/workspace/litengjie/data/r2v_v3_audits")
MODEL_DISCOVERY_ROOTS = (
    Path("/mnt/workspace/public/pretrained"),
    Path("/mnt/workspace/litengjie/data/models"),
    Path("/mnt/workspace/litengjie/data/vendor"),
)
_QUALITY_NAMES = (
    "musiq",
    "maniqa",
    "topiq",
    "nima",
    "qalign",
    "q-align",
    "onealign",
    "one-align",
)
_EMBEDDING_NAMES = ("dinov2", "dino", "siglip", "clip")
_POSE_NAMES = (
    "insightface",
    "retinaface",
    "scrfd",
    "6drepnet",
    "6drepnet360",
    "face-landmark",
    "head-pose",
)
_FOCUS_CASES = frozenset(
    {
        ("15409fe27a23cb0a16bdd459", "e1"),
        ("251b44a75511156ff06222d0", "e1"),
        ("425f401670a4307b149f2420", "e1"),
        ("58c7d4523b65add330d71943", "e1"),
        ("82c20ca1c4f8e0c855a783de", "e1"),
        ("f374b11496cd99f988879a3e", "e1"),
        ("f374b11496cd99f988879a3e", "e2"),
    }
)

ArtifactScope = Literal["candidates", "final", "both"]


@dataclass(frozen=True)
class QualityObservation:
    quality_score: float
    quality_scale_min: float
    quality_scale_max: float
    aesthetic_score: float
    aesthetic_scale_min: float
    aesthetic_scale_max: float
    backend: str
    model_name: str
    runtime_seconds: float
    raw_metrics: Mapping[str, object]
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        values = (
            self.quality_score,
            self.quality_scale_min,
            self.quality_scale_max,
            self.aesthetic_score,
            self.aesthetic_scale_min,
            self.aesthetic_scale_max,
            self.runtime_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("quality observation values must be finite")
        if not self.quality_scale_min <= self.quality_score <= self.quality_scale_max:
            raise ValueError("quality score is outside its declared scale")
        if not (
            self.aesthetic_scale_min
            <= self.aesthetic_score
            <= self.aesthetic_scale_max
        ):
            raise ValueError("aesthetic score is outside its declared scale")
        if self.runtime_seconds < 0:
            raise ValueError("quality runtime must be non-negative")
        if not self.backend.strip() or not self.model_name.strip():
            raise ValueError("quality backend and model name must not be empty")


@dataclass(frozen=True)
class EmbeddingObservation:
    embedding: Sequence[float]
    backend: str
    model_name: str
    runtime_seconds: float
    raw_metrics: Mapping[str, object]

    def normalized(self) -> np.ndarray:
        vector = np.asarray(self.embedding, dtype=np.float64)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("visual embedding must be a finite non-empty vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("visual embedding norm must be positive")
        return vector / norm

    def __post_init__(self) -> None:
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("embedding runtime must be finite and non-negative")
        if not self.backend.strip() or not self.model_name.strip():
            raise ValueError("embedding backend and model name must not be empty")
        self.normalized()


@dataclass(frozen=True)
class SubjectPoseObservation:
    face_detected: bool
    face_detection_confidence: float | None
    face_bbox_area_ratio: float | None
    head_visible: bool | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    pose_backend: str
    model_name: str
    runtime_seconds: float
    subject_view_quality_score: float | None = None
    raw_metrics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.face_detection_confidence,
            self.face_bbox_area_ratio,
            self.yaw,
            self.pitch,
            self.roll,
            self.subject_view_quality_score,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("subject pose values must be finite when present")
        if self.face_detection_confidence is not None and not (
            0 <= self.face_detection_confidence <= 1
        ):
            raise ValueError("face confidence must be between zero and one")
        if self.face_bbox_area_ratio is not None and not (
            0 <= self.face_bbox_area_ratio <= 1
        ):
            raise ValueError("face area ratio must be between zero and one")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("subject pose runtime must be finite and non-negative")
        if not self.pose_backend.strip() or not self.model_name.strip():
            raise ValueError("pose backend and model name must not be empty")


class ReferenceQualityScorer(Protocol):
    backend: str
    model_name: str

    def score(self, image: Image.Image) -> QualityObservation: ...


class ReferenceDiscriminabilityScorer(Protocol):
    backend: str
    model_name: str
    fingerprint: str

    def embed(self, image: Image.Image) -> EmbeddingObservation: ...


class SubjectPoseScorer(Protocol):
    backend: str
    model_name: str

    def inspect(self, image: Image.Image) -> SubjectPoseObservation: ...


class ExternalReferenceFilterScorer:
    def __init__(
        self,
        *,
        kind: Literal["quality", "embedding", "subject_pose"],
        backend: str,
        python_executable: Path,
        code_root: Path,
        model_path: Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("external scorer timeout must be positive")
        self.kind = kind
        self.backend = backend
        self.model_name = model_path.name
        self.fingerprint = hashlib.sha256(
            f"{kind}:{backend}:{model_path.resolve(strict=False)}".encode()
        ).hexdigest()
        worker = Path(__file__).resolve().parents[2] / "tools" / (
            "run_v3_reference_filter_worker.py"
        )
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        self._timeout_seconds = timeout_seconds
        self._process = subprocess.Popen(
            [
                str(python_executable),
                str(worker),
                "--kind",
                kind,
                "--backend",
                backend,
                "--code-root",
                str(code_root),
                "--model-path",
                str(model_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )

    def _request(self, image: Image.Image) -> Mapping[str, object]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("external scorer streams are unavailable")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"external scorer exited with code {self._process.returncode}"
            )
        request_id = uuid.uuid4().hex
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        request = {
            "request_id": request_id,
            "image_png_hex": buffer.getvalue().hex(),
        }
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        readable, _, _ = select.select(
            [self._process.stdout],
            [],
            [],
            self._timeout_seconds,
        )
        if not readable:
            raise TimeoutError("external reference scorer timed out")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("external reference scorer closed stdout")
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise RuntimeError("external reference scorer returned an invalid response")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error", "external scorer failed")))
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise TypeError("external reference scorer result is not an object")
        return result

    def score(self, image: Image.Image) -> QualityObservation:
        if self.kind != "quality":
            raise TypeError("external scorer is not configured for quality")
        result = self._request(image)
        return QualityObservation(
            quality_score=float(result["quality_score"]),
            quality_scale_min=float(result["quality_scale_min"]),
            quality_scale_max=float(result["quality_scale_max"]),
            aesthetic_score=float(result["aesthetic_score"]),
            aesthetic_scale_min=float(result["aesthetic_scale_min"]),
            aesthetic_scale_max=float(result["aesthetic_scale_max"]),
            backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=float(result["runtime_seconds"]),
            raw_metrics=_mapping_value(result.get("raw_metrics")),
            higher_is_better=bool(result.get("higher_is_better", True)),
        )

    def embed(self, image: Image.Image) -> EmbeddingObservation:
        if self.kind != "embedding":
            raise TypeError("external scorer is not configured for embeddings")
        result = self._request(image)
        embedding = result.get("embedding")
        if not isinstance(embedding, Sequence) or isinstance(embedding, str):
            raise TypeError("external embedding result is invalid")
        return EmbeddingObservation(
            embedding=[float(value) for value in embedding],
            backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=float(result["runtime_seconds"]),
            raw_metrics=_mapping_value(result.get("raw_metrics")),
        )

    def inspect(self, image: Image.Image) -> SubjectPoseObservation:
        if self.kind != "subject_pose":
            raise TypeError("external scorer is not configured for subject pose")
        result = self._request(image)
        return SubjectPoseObservation(
            face_detected=bool(result["face_detected"]),
            face_detection_confidence=_optional_float(
                result.get("face_detection_confidence")
            ),
            face_bbox_area_ratio=_optional_float(result.get("face_bbox_area_ratio")),
            head_visible=_optional_bool(result.get("head_visible")),
            yaw=_optional_float(result.get("yaw")),
            pitch=_optional_float(result.get("pitch")),
            roll=_optional_float(result.get("roll")),
            pose_backend=self.backend,
            model_name=self.model_name,
            runtime_seconds=float(result["runtime_seconds"]),
            subject_view_quality_score=_optional_float(
                result.get("subject_view_quality_score")
            ),
            raw_metrics=_mapping_value(result.get("raw_metrics")),
        )

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.write(
                json.dumps({"request_id": uuid.uuid4().hex, "shutdown": True}) + "\n"
            )
            process.stdin.flush()
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def _mapping_value(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


class ReadOnlyAuditStorage(RunStorage):
    def frame_path(self, clip_uid: str, frame_slot: int) -> Path:
        self._require_clip(clip_uid)
        if not 0 <= frame_slot < self.config.frames.count:
            raise ValueError("frame_slot is outside configured frame range")
        return self.clip_dir(clip_uid) / "frames" / f"{frame_slot:02d}.jpg"


@dataclass
class _Artifact:
    record: dict[str, object]
    image: Image.Image
    image_sha256: str
    embedding_vector: np.ndarray | None = None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_run_files(run_root: Path) -> dict[str, str]:
    root = run_root.expanduser().resolve(strict=True)
    return {
        path.relative_to(root).as_posix(): (
            "<directory>" if path.is_dir() else _sha256_path(path)
        )
        for path in sorted(root.rglob("*"))
        if path.is_dir() or path.is_file()
    }


def _image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _white_composite(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _resolve_run_artifact(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("audit artifact is outside source run_root")
    if not path.is_file():
        raise ValueError("audit artifact is not a file")
    return path


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        return _white_composite(opened)


def _baseline_decision(
    storage: RunStorage,
    clip_uid: str,
    entity_id: str,
) -> dict[str, object] | None:
    path = (
        storage.clip_dir(clip_uid)
        / "debug"
        / "pair"
        / entity_id
        / "raw_responses.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    responses = payload.get("responses") if isinstance(payload, dict) else None
    if not isinstance(responses, list) or not responses:
        return {"valid": False, "reason": "missing_final_response"}
    raw = responses[-1]
    if not isinstance(raw, str):
        return {"valid": False, "reason": "non_text_final_response"}
    try:
        stripped = raw.strip()
        if stripped.startswith("```"):
            match = re.fullmatch(
                r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",
                stripped,
                flags=re.IGNORECASE,
            )
            if match is None:
                raise ValueError("final response has an incomplete JSON fence")
            stripped = match.group(1).strip()
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise TypeError("final response is not a JSON object")
        decision = RawEntityReferenceDecision.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        return {
            "valid": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    value = decision.model_dump(mode="json")
    value["valid"] = True
    return value


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    if left_vector.shape != right_vector.shape or left_vector.ndim != 1:
        raise ValueError("cosine vectors must have the same one-dimensional shape")
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("cosine vectors must have positive norms")
    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("distribution values must be finite")
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _runtime_summary(values: Sequence[float]) -> dict[str, object]:
    distribution = _distribution(values)
    mean = sum(values) / len(values) if values else None
    result: dict[str, object] = {
        "calls": len(values),
        "total_s": sum(values),
        "mean_s": mean,
        "mean_ms_per_image": mean * 1000 if mean is not None else None,
        "p50_s": distribution["p50"] if distribution is not None else None,
        "p95_s": (
            float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))
            if values
            else None
        ),
        "relative_to_qwen_candidate_mean": (
            mean / 11.0 if mean is not None else None
        ),
    }
    if mean is None:
        result["performance_band"] = "not_run"
    elif mean < 0.1:
        result["performance_band"] = "<0.1 sec/image"
    elif mean < 0.5:
        result["performance_band"] = "0.1-0.5 sec/image"
    elif mean < 1.0:
        result["performance_band"] = "0.5-1 sec/image"
    elif mean <= 2.0:
        result["performance_band"] = "1-2 sec/image"
    else:
        result["performance_band"] = ">2 sec/image"
        result["prefilter_attractiveness"] = "NOT_ATTRACTIVE_FOR_PREFILTER"
    return result


def _bounded_walk(root: Path, max_depth: int) -> list[Path]:
    if not root.is_dir():
        return []
    root = root.resolve(strict=True)
    results: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth >= max_depth:
            directories[:] = []
        results.extend(current_path / name for name in directories)
        results.extend(current_path / name for name in filenames)
    return results


def discover_local_models(
    roots: Sequence[Path] = MODEL_DISCOVERY_ROOTS,
    *,
    max_depth: int = 3,
) -> list[dict[str, object]]:
    if not 0 <= max_depth <= 3:
        raise ValueError("model discovery max_depth must be between zero and three")
    discoveries: dict[str, dict[str, object]] = {}
    categories = {
        "quality": _QUALITY_NAMES,
        "embedding": _EMBEDDING_NAMES,
        "subject_pose": _POSE_NAMES,
    }
    for root in roots:
        for path in _bounded_walk(root.expanduser(), max_depth):
            lowered = path.name.casefold().replace("_", "-").replace(" ", "-")
            matched = [
                category
                for category, names in categories.items()
                if any(name.replace("_", "-") in lowered for name in names)
            ]
            if not matched:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            discoveries[str(path.resolve(strict=False))] = {
                "path": str(path.resolve(strict=False)),
                "backend": next(
                    name
                    for names in categories.values()
                    for name in names
                    if name.replace("_", "-") in lowered
                ),
                "possible_model_types": sorted(matched),
                "size_bytes": size,
            }
    return [discoveries[key] for key in sorted(discoveries)]


def _candidate_record(
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    candidate: Any,
    baseline: Mapping[str, object] | None,
    detached_signal: bool,
) -> dict[str, object]:
    selected_id = baseline.get("selected_candidate_id") if baseline else None
    return {
        "clip_uid": clip_uid,
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "phrase": entity.phrase,
        "artifact_scope": "candidate",
        "candidate_id": candidate.candidate_id,
        "frame_slot": candidate.frame_slot,
        "source_frame_index": candidate.source_frame_index,
        "area_ratio": candidate.area_ratio,
        "bbox_fill_ratio": candidate.bbox_fill_ratio,
        "border_contact_count": candidate.border_contact_count,
        "sharpness_score": candidate.sharpness_score,
        "significant_component_count": candidate.significant_component_count,
        "largest_component_ratio": candidate.largest_component_ratio,
        "second_largest_component_ratio": (
            candidate.second_largest_component_ratio
        ),
        "nontrivial_detached_component_signal": detached_signal,
        "is_current_selected": candidate.candidate_id == selected_id,
        "production_baseline": dict(baseline) if baseline is not None else None,
    }


def _subject_detached_signal(candidate: Any, pair_module: Any) -> bool:
    existing_large_signal = (
        candidate.significant_component_count >= 2
        and 0.05 <= candidate.second_largest_component_ratio <= 0.20
        and candidate.largest_component_ratio >= 0.70
    )
    if existing_large_signal:
        return True
    components = pair_module._foreground_components(candidate.mask)
    if len(components) < 2:
        return False
    main, secondary = components[:2]
    total_foreground = sum(component.area_pixels for component in components)
    secondary_ratio = secondary.area_pixels / total_foreground
    if not (0.005 <= secondary_ratio < 0.05) or secondary.area_pixels < 64:
        return False
    main_x1, main_y1, main_x2, main_y2 = main.bbox_xyxy
    secondary_x1, secondary_y1, secondary_x2, secondary_y2 = secondary.bbox_xyxy
    main_long_side = max(main_x2 - main_x1, main_y2 - main_y1)
    secondary_long_side = max(
        secondary_x2 - secondary_x1,
        secondary_y2 - secondary_y1,
    )
    if secondary_long_side < max(12.0, 0.08 * main_long_side):
        return False
    gap_x = max(main_x1 - secondary_x2, secondary_x1 - main_x2, 0)
    gap_y = max(main_y1 - secondary_y2, secondary_y1 - main_y2, 0)
    if math.hypot(gap_x, gap_y) < max(4.0, 0.02 * main_long_side):
        return False
    union_x1 = min(main_x1, secondary_x1)
    union_y1 = min(main_y1, secondary_y1)
    union_x2 = max(main_x2, secondary_x2)
    union_y2 = max(main_y2, secondary_y2)
    main_bbox_area = (main_x2 - main_x1) * (main_y2 - main_y1)
    union_bbox_area = (union_x2 - union_x1) * (union_y2 - union_y1)
    maximum_extension = max(
        main_x1 - union_x1,
        main_y1 - union_y1,
        union_x2 - main_x2,
        union_y2 - main_y2,
    )
    return (
        union_bbox_area / main_bbox_area >= 1.15
        or maximum_extension >= 0.10 * main_long_side
    )


def _final_record(
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    reference: Any,
    image_sha256: str,
) -> dict[str, object]:
    return {
        "clip_uid": clip_uid,
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "phrase": entity.phrase,
        "artifact_scope": "final",
        "reference_scope": reference.reference_scope,
        "visible_region": reference.visible_region,
        "completeness": reference.completeness,
        "synthetic": reference.synthetic,
        "image_path": reference.image_path,
        "source_frame_index": reference.source_frame_index,
        "source_clip_uid": reference.source_clip_uid,
        "source_entity_id": reference.source_entity_id,
        "generation_metadata_path": reference.generation_metadata_path,
        "sha256": image_sha256,
        "production_baseline": {
            "selected_candidate_id": None,
            "image_quality": reference.image_quality,
            "completeness": reference.completeness,
            "reference_scope": reference.reference_scope,
            "visible_region": reference.visible_region,
            "viewpoint": reference.viewpoint,
            "identity_features_visible": reference.identity_features_visible,
            "primary_identity_region_visible": (
                reference.primary_identity_region_visible
            ),
            "major_structure_visible": reference.major_structure_visible,
            "truncation_severity": reference.truncation_severity,
            "completion_needed_for_reference_use": (
                reference.completion_needed_for_reference_use
            ),
            "detached_target_fragments_present": (
                reference.detached_target_fragments_present
            ),
            "valid": reference.status == "ready",
        },
    }


def _collect_artifacts(
    config: V3Config,
    storage: ReadOnlyAuditStorage,
    artifact_scope: ArtifactScope,
) -> list[_Artifact]:
    artifacts: list[_Artifact] = []
    include_candidates = artifact_scope in {"candidates", "both"}
    include_final = artifact_scope in {"final", "both"}
    root = storage.root.resolve(strict=True)
    pair_module: Any | None = None
    for initial_clip in storage.iter_clips():
        clip = storage.read_clip(initial_clip.clip_uid)
        if clip.annotation is None or clip.annotation.status != "ready":
            continue
        entities = {entity.entity_id: entity for entity in clip.annotation.entities}
        if include_candidates and clip.coverage is not None and clip.coverage.passed:
            if pair_module is None:
                from r2v_data_v2.v3 import pair as pair_module_value

                pair_module = pair_module_value
            frames = validate_sampled_frames(storage, clip.clip_uid)
            masks = storage.read_masks(clip.clip_uid)
            for entity in clip.annotation.entities:
                tracked = masks.entities.get(entity.entity_id)
                if tracked is None or tracked.status != "ready":
                    continue
                candidates = pair_module.build_entity_reference_candidates(
                    config,
                    storage,
                    clip_uid=clip.clip_uid,
                    entity=entity,
                    frames=frames,
                    masks=masks,
                )
                baseline = _baseline_decision(
                    storage,
                    clip.clip_uid,
                    entity.entity_id,
                )
                for candidate in candidates:
                    source_path = _resolve_run_artifact(root, candidate.image_path)
                    source = _load_rgb(source_path)
                    crop, _ = pair_module.build_reference_crop(
                        source,
                        candidate.mask,
                        crop_padding_ratio=config.pair.crop_padding_ratio,
                    )
                    image = _white_composite(crop)
                    detached_signal = bool(
                        _subject_detached_signal(candidate, pair_module)
                        if entity.reference_type == "subject"
                        else False
                    )
                    record = _candidate_record(
                        clip_uid=clip.clip_uid,
                        entity=entity,
                        candidate=candidate,
                        baseline=baseline,
                        detached_signal=detached_signal,
                    )
                    artifacts.append(
                        _Artifact(
                            record=record,
                            image=image,
                            image_sha256=_image_sha256(image),
                        )
                    )
        if include_final:
            for reference in clip.references.entities:
                if reference.status != "ready" or reference.image_path is None:
                    continue
                entity = entities.get(reference.entity_id)
                if entity is None:
                    raise ValueError("ready final reference has no annotation entity")
                path = _resolve_run_artifact(root, reference.image_path)
                image = _load_rgb(path)
                artifacts.append(
                    _Artifact(
                        record=_final_record(
                            clip_uid=clip.clip_uid,
                            entity=entity,
                            reference=reference,
                            image_sha256=_sha256_path(path),
                        ),
                        image=image,
                        image_sha256=_image_sha256(image),
                    )
                )
    return artifacts


def _failure(backend: str, exc: Exception, runtime: float) -> dict[str, object]:
    return {
        "status": "failed",
        "backend": backend,
        "failure_type": type(exc).__name__,
        "reason": str(exc),
        "runtime_seconds": runtime,
    }


def _quality_dict(observation: QualityObservation) -> dict[str, object]:
    return {
        "status": "succeeded",
        "backend": observation.backend,
        "model_name": observation.model_name,
        "quality_score": observation.quality_score,
        "quality_scale_min": observation.quality_scale_min,
        "quality_scale_max": observation.quality_scale_max,
        "aesthetic_score": observation.aesthetic_score,
        "aesthetic_scale_min": observation.aesthetic_scale_min,
        "aesthetic_scale_max": observation.aesthetic_scale_max,
        "runtime_seconds": observation.runtime_seconds,
        "raw_metrics": dict(observation.raw_metrics),
        "higher_is_better": observation.higher_is_better,
    }


def _pose_dict(observation: SubjectPoseObservation) -> dict[str, object]:
    return {
        "status": "succeeded",
        "face_detected": observation.face_detected,
        "face_detection_confidence": observation.face_detection_confidence,
        "face_bbox_area_ratio": observation.face_bbox_area_ratio,
        "head_visible": observation.head_visible,
        "yaw": observation.yaw,
        "pitch": observation.pitch,
        "roll": observation.roll,
        "pose_backend": observation.pose_backend,
        "model_name": observation.model_name,
        "runtime_seconds": observation.runtime_seconds,
        "subject_view_quality_score": observation.subject_view_quality_score,
        "raw_metrics": dict(observation.raw_metrics or {}),
    }


def _embedding_cache_path(
    cache_dir: Path,
    image_sha256: str,
    fingerprint: str,
) -> Path:
    key = hashlib.sha256(f"{image_sha256}:{fingerprint}".encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def _score_artifacts(
    artifacts: list[_Artifact],
    *,
    quality_backend: str,
    embedding_backend: str,
    subject_pose_backend: str,
    quality_scorer: ReferenceQualityScorer | None,
    embedding_scorer: ReferenceDiscriminabilityScorer | None,
    subject_pose_scorer: SubjectPoseScorer | None,
    cache_dir: Path,
    fail_fast: bool,
    clock: Callable[[], float],
) -> dict[str, list[float]]:
    runtimes = {"quality": [], "embedding": [], "subject_pose": []}
    for artifact in artifacts:
        record = artifact.record
        if quality_backend == "none":
            record["quality"] = {"status": "disabled", "backend": "none"}
        elif quality_scorer is None:
            record["quality"] = {
                "status": "unavailable",
                "backend": quality_backend,
                "reason": "quality_backend_unavailable",
            }
        else:
            started = clock()
            try:
                observation = quality_scorer.score(artifact.image.copy())
                elapsed = clock() - started
                if elapsed < 0:
                    raise ValueError("audit clock moved backwards")
                record["quality"] = _quality_dict(observation)
                runtimes["quality"].append(observation.runtime_seconds)
            except Exception as exc:
                elapsed = max(0.0, clock() - started)
                runtimes["quality"].append(elapsed)
                if fail_fast:
                    raise
                record["quality"] = _failure(quality_backend, exc, elapsed)

        if embedding_backend == "none":
            record["embedding"] = {"status": "disabled", "backend": "none"}
        elif embedding_scorer is None:
            record["embedding"] = {
                "status": "unavailable",
                "backend": embedding_backend,
                "reason": "embedding_backend_unavailable",
            }
        else:
            cache_path = _embedding_cache_path(
                cache_dir,
                artifact.image_sha256,
                embedding_scorer.fingerprint,
            )
            started = clock()
            try:
                if cache_path.is_file():
                    with cache_path.open("rb") as handle:
                        vector = np.load(handle, allow_pickle=False)
                    runtime_seconds = 0.0
                    raw_metrics: dict[str, object] = {"cache_hit": True}
                    backend = embedding_scorer.backend
                    model_name = embedding_scorer.model_name
                    cache_hit = True
                else:
                    observation = embedding_scorer.embed(artifact.image.copy())
                    vector = observation.normalized()
                    runtime_seconds = observation.runtime_seconds
                    raw_metrics = dict(observation.raw_metrics)
                    raw_metrics["cache_hit"] = False
                    backend = observation.backend
                    model_name = observation.model_name
                    cache_hit = False
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.save(handle, vector, allow_pickle=False)
                    temporary.replace(cache_path)
                artifact.embedding_vector = EmbeddingObservation(
                    embedding=vector,
                    backend=backend,
                    model_name=model_name,
                    runtime_seconds=runtime_seconds,
                    raw_metrics=raw_metrics,
                ).normalized()
                record["embedding"] = {
                    "status": "succeeded",
                    "backend": backend,
                    "model_name": model_name,
                    "runtime_seconds": runtime_seconds,
                    "embedding_dimension": int(artifact.embedding_vector.size),
                    "raw_metrics": raw_metrics,
                    "same_entity_mean_similarity": None,
                    "same_entity_min_similarity": None,
                    "same_entity_max_similarity": None,
                    "representativeness_score": None,
                    "max_other_entity_similarity": None,
                    "inter_entity_margin": None,
                }
                if not cache_hit:
                    runtimes["embedding"].append(runtime_seconds)
            except Exception as exc:
                elapsed = max(0.0, clock() - started)
                runtimes["embedding"].append(elapsed)
                if fail_fast:
                    raise
                record["embedding"] = _failure(embedding_backend, exc, elapsed)

        reference_type = record["reference_type"]
        if reference_type != "subject":
            record["subject_pose"] = {
                "status": "not_applicable",
                "reason": f"reference_type={reference_type}",
            }
        elif subject_pose_backend == "none":
            record["subject_pose"] = {"status": "disabled", "backend": "none"}
        elif subject_pose_scorer is None:
            record["subject_pose"] = {
                "status": "unavailable",
                "backend": subject_pose_backend,
                "reason": "subject_pose_backend_unavailable",
            }
        else:
            started = clock()
            try:
                observation = subject_pose_scorer.inspect(artifact.image.copy())
                elapsed = clock() - started
                if elapsed < 0:
                    raise ValueError("audit clock moved backwards")
                record["subject_pose"] = _pose_dict(observation)
                runtimes["subject_pose"].append(observation.runtime_seconds)
            except Exception as exc:
                elapsed = max(0.0, clock() - started)
                runtimes["subject_pose"].append(elapsed)
                if fail_fast:
                    raise
                record["subject_pose"] = _failure(
                    subject_pose_backend,
                    exc,
                    elapsed,
                )
    return runtimes


def _populate_embedding_metrics(artifacts: list[_Artifact]) -> None:
    candidates = [
        artifact
        for artifact in artifacts
        if artifact.record["artifact_scope"] == "candidate"
        and artifact.embedding_vector is not None
    ]
    for artifact in artifacts:
        vector = artifact.embedding_vector
        embedding = artifact.record.get("embedding")
        if vector is None or not isinstance(embedding, dict):
            continue
        same_entity = [
            other.embedding_vector
            for other in candidates
            if other is not artifact
            and other.record["clip_uid"] == artifact.record["clip_uid"]
            and other.record["entity_id"] == artifact.record["entity_id"]
            and other.embedding_vector is not None
        ]
        other_entities = [
            other.embedding_vector
            for other in candidates
            if other.record["clip_uid"] == artifact.record["clip_uid"]
            and other.record["entity_id"] != artifact.record["entity_id"]
            and other.embedding_vector is not None
        ]
        same_values = [cosine_similarity(vector, other) for other in same_entity]
        other_values = [cosine_similarity(vector, other) for other in other_entities]
        same_mean = sum(same_values) / len(same_values) if same_values else None
        max_other = max(other_values) if other_values else None
        embedding.update(
            {
                "same_entity_mean_similarity": same_mean,
                "same_entity_min_similarity": min(same_values)
                if same_values
                else None,
                "same_entity_max_similarity": max(same_values)
                if same_values
                else None,
                "representativeness_score": same_mean,
                "max_other_entity_similarity": max_other,
                "inter_entity_margin": (
                    same_mean - max_other
                    if same_mean is not None and max_other is not None
                    else None
                ),
            }
        )


def _successful_metric(
    record: Mapping[str, object],
    section: str,
    field: str,
) -> float | None:
    value = record.get(section)
    if not isinstance(value, Mapping) or value.get("status") != "succeeded":
        return None
    metric = value.get(field)
    return float(metric) if isinstance(metric, (int, float)) else None


def _distributions_by_type(
    records: list[dict[str, object]],
    section: str,
    fields: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for reference_type in ("subject", "object", "group"):
        typed = [
            record for record in records if record["reference_type"] == reference_type
        ]
        result[reference_type] = {
            field: _distribution(
                [
                    value
                    for record in typed
                    if (value := _successful_metric(record, section, field))
                    is not None
                ]
            )
            for field in fields
        }
    return result


def _selected_analysis(records: list[dict[str, object]]) -> dict[str, object]:
    candidates = [
        record for record in records if record["artifact_scope"] == "candidate"
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidates:
        groups.setdefault(
            (str(record["clip_uid"]), str(record["entity_id"])),
            [],
        ).append(record)
    metrics = {
        "selected_is_highest_quality": ("quality", "quality_score"),
        "selected_is_highest_aesthetic": ("quality", "aesthetic_score"),
        "selected_is_most_representative": (
            "embedding",
            "representativeness_score",
        ),
        "selected_is_highest_same_entity_similarity": (
            "embedding",
            "same_entity_mean_similarity",
        ),
        "selected_has_best_margin": ("embedding", "inter_entity_margin"),
        "selected_has_largest_face_area": (
            "subject_pose",
            "face_bbox_area_ratio",
        ),
        "selected_has_highest_subject_view_quality": (
            "subject_pose",
            "subject_view_quality_score",
        ),
    }
    result: dict[str, object] = {}
    for name, (section, field) in metrics.items():
        matches = 0
        comparisons = 0
        for group in groups.values():
            selected = next(
                (record for record in group if record["is_current_selected"]),
                None,
            )
            if selected is None:
                continue
            values = [
                (record, _successful_metric(record, section, field))
                for record in group
            ]
            if any(value is None for _, value in values):
                continue
            comparisons += 1
            selected_value = _successful_metric(selected, section, field)
            assert selected_value is not None
            matches += int(selected_value >= max(float(value) for _, value in values))
        result[f"{name}_count"] = matches
        result[f"{name}_denominator"] = comparisons
        result[f"{name}_ratio"] = matches / comparisons if comparisons else None
    comparison_fields = (
        ("quality", "quality_score"),
        ("quality", "aesthetic_score"),
        ("embedding", "same_entity_mean_similarity"),
        ("embedding", "representativeness_score"),
        ("embedding", "inter_entity_margin"),
        ("subject_pose", "face_bbox_area_ratio"),
        ("subject_pose", "yaw"),
        ("subject_pose", "pitch"),
        ("subject_pose", "subject_view_quality_score"),
    )
    result["metric_distributions"] = {
        population: {
            field: _distribution(
                [
                    value
                    for record in candidates
                    if bool(record["is_current_selected"]) is selected
                    and (value := _successful_metric(record, section, field))
                    is not None
                ]
            )
            for section, field in comparison_fields
        }
        for population, selected in (("selected", True), ("non_selected", False))
    }
    return result


def _selected_representativeness_rank_analysis(
    records: list[dict[str, object]],
) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        if record["artifact_scope"] != "candidate":
            continue
        groups.setdefault(
            (str(record["clip_uid"]), str(record["entity_id"])),
            [],
        ).append(record)
    cases: list[dict[str, object]] = []
    for (clip_uid, entity_id), group in sorted(groups.items()):
        selected = next(
            (record for record in group if record["is_current_selected"]),
            None,
        )
        scored = [
            (record, score)
            for record in group
            if (
                score := _successful_metric(
                    record,
                    "embedding",
                    "representativeness_score",
                )
            )
            is not None
        ]
        if selected is None or len(scored) != len(group):
            continue
        ranked = sorted(
            scored,
            key=lambda item: (-item[1], str(item[0].get("candidate_id"))),
        )
        selected_rank = next(
            index
            for index, (record, _) in enumerate(ranked, start=1)
            if record is selected
        )
        baseline = selected.get("production_baseline")
        cases.append(
            {
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "reference_type": selected["reference_type"],
                "production_selected_candidate_id": selected["candidate_id"],
                "selected_rank": selected_rank,
                "candidate_count": len(group),
                "representativeness_values": [
                    {
                        "candidate_id": record["candidate_id"],
                        "representativeness_score": score,
                    }
                    for record, score in sorted(
                        scored,
                        key=lambda item: str(item[0].get("candidate_id")),
                    )
                ],
                "production_completeness": (
                    baseline.get("completeness")
                    if isinstance(baseline, Mapping)
                    else None
                ),
                "production_reference_scope": (
                    baseline.get("reference_scope")
                    if isinstance(baseline, Mapping)
                    else None
                ),
                "production_viewpoint": (
                    baseline.get("viewpoint")
                    if isinstance(baseline, Mapping)
                    else None
                ),
            }
        )

    def summarize(values: list[dict[str, object]]) -> dict[str, object]:
        counts = {
            f"selected_rank_{rank}_count": sum(
                case["selected_rank"] == rank for case in values
            )
            for rank in (1, 2, 3)
        }
        denominator = len(values)
        return {
            "entity_count": denominator,
            **counts,
            "selected_rank_1_rate": (
                counts["selected_rank_1_count"] / denominator
                if denominator
                else None
            ),
        }

    by_reference_type = {
        reference_type: summarize(
            [case for case in cases if case["reference_type"] == reference_type]
        )
        for reference_type in ("subject", "object", "group")
    }
    return {
        **summarize(cases),
        "by_reference_type": by_reference_type,
        "cases": cases,
        "selected_rank_last_cases": [
            case
            for case in cases
            if case["selected_rank"] == case["candidate_count"]
        ],
    }


def _pose_by_production_viewpoint(
    records: list[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for viewpoint in ("front", "three_quarter", "side", "rear"):
        matching = [
            record
            for record in records
            if isinstance(record.get("production_baseline"), Mapping)
            and record["production_baseline"].get("viewpoint") == viewpoint
        ]
        result[viewpoint] = {
            "count": len(matching),
            "yaw": _distribution(
                [
                    value
                    for record in matching
                    if (
                        value := _successful_metric(record, "subject_pose", "yaw")
                    )
                    is not None
                ]
            ),
            "pitch": _distribution(
                [
                    value
                    for record in matching
                    if (
                        value := _successful_metric(record, "subject_pose", "pitch")
                    )
                    is not None
                ]
            ),
        }
    return result


def _route_analysis(records: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[dict[str, object]]] = {
        "complete_full": [],
        "local_usable": [],
        "repairable": [],
        "reject": [],
    }
    for record in records:
        baseline = record.get("production_baseline")
        if not isinstance(baseline, Mapping):
            continue
        if baseline.get("reference_scope") == "reject":
            bucket = "reject"
        elif baseline.get("completeness") == "repairable":
            bucket = "repairable"
        elif (
            baseline.get("completeness") == "local_usable"
            or baseline.get("reference_scope") == "local"
        ):
            bucket = "local_usable"
        else:
            bucket = "complete_full"
        buckets[bucket].append(record)
    fields = (
        ("quality", "quality_score"),
        ("embedding", "representativeness_score"),
        ("embedding", "inter_entity_margin"),
        ("subject_pose", "face_bbox_area_ratio"),
    )
    return {
        bucket: {
            field: _distribution(
                [
                    value
                    for record in bucket_records
                    if (value := _successful_metric(record, section, field))
                    is not None
                ]
            )
            for section, field in fields
        }
        for bucket, bucket_records in buckets.items()
    }


def _review_list(
    records: list[dict[str, object]],
    section: str,
    field: str,
    *,
    descending: bool = False,
    absolute: bool = False,
) -> list[dict[str, object]]:
    values = [
        (record, value)
        for record in records
        if (value := _successful_metric(record, section, field)) is not None
    ]
    if not values:
        return []
    count = max(1, math.ceil(len(values) * 0.10))
    values.sort(
        key=lambda item: abs(item[1]) if absolute else item[1],
        reverse=descending,
    )
    return [
        {
            "clip_uid": record["clip_uid"],
            "entity_id": record["entity_id"],
            "artifact_scope": record["artifact_scope"],
            "candidate_id": record.get("candidate_id"),
            "value": value,
        }
        for record, value in values[:count]
    ]


def _summary(
    records: list[dict[str, object]],
    *,
    artifact_scope: ArtifactScope,
    quality_backend: str,
    embedding_backend: str,
    subject_pose_backend: str,
    runtimes: Mapping[str, Sequence[float]],
    discoveries: list[dict[str, object]],
    source_unchanged: bool,
) -> dict[str, object]:
    candidates = [
        record for record in records if record["artifact_scope"] == "candidate"
    ]
    subjects = [record for record in records if record["reference_type"] == "subject"]
    face_detected = [
        record
        for record in subjects
        if isinstance(record.get("subject_pose"), Mapping)
        and record["subject_pose"].get("status") == "succeeded"
        and record["subject_pose"].get("face_detected") is True
    ]
    pose_succeeded = [
        record
        for record in subjects
        if isinstance(record.get("subject_pose"), Mapping)
        and record["subject_pose"].get("status") == "succeeded"
    ]
    focus_cases = [
        record
        for record in records
        if (str(record["clip_uid"]), str(record["entity_id"])) in _FOCUS_CASES
    ]
    visual_runtime = _runtime_summary(runtimes["embedding"])
    mean_embedding_seconds = visual_runtime["mean_s"]
    visual_runtime["estimated_seconds_per_three_candidate_entity"] = (
        float(mean_embedding_seconds) * 3
        if isinstance(mean_embedding_seconds, (int, float))
        else None
    )
    visual_runtime["embedding_seconds_per_entity"] = visual_runtime[
        "estimated_seconds_per_three_candidate_entity"
    ]
    visual_runtime["candidate_images_per_entity_assumption"] = 3
    return {
        "schema_version": 1,
        "audit_only": True,
        "qwen_calls_added": 0,
        "artifact_scope": artifact_scope,
        "record_count": len(records),
        "candidate_count": len(candidates),
        "final_reference_count": len(records) - len(candidates),
        "source_run_unchanged": source_unchanged,
        "backends": {
            "quality": quality_backend,
            "embedding": embedding_backend,
            "subject_pose": subject_pose_backend,
        },
        "model_discovery": discoveries,
        "quality_distributions": _distributions_by_type(
            records,
            "quality",
            ("quality_score", "aesthetic_score"),
        ),
        "embedding_distributions": _distributions_by_type(
            records,
            "embedding",
            (
                "same_entity_mean_similarity",
                "representativeness_score",
                "inter_entity_margin",
            ),
        ),
        "embedding_dimensions": sorted(
            {
                int(dimension)
                for record in records
                if (
                    dimension := _successful_metric(
                        record,
                        "embedding",
                        "embedding_dimension",
                    )
                )
                is not None
            }
        ),
        "representativeness_definition": {
            "formula": "mean cosine similarity to other same-entity candidates",
            "centroid_value_equivalent": False,
            "centroid_rank_equivalent": True,
            "centroid_rank_equivalence_scope": (
                "same normalized candidate set with a nonzero centroid"
            ),
        },
        "selected_representativeness_rank": (
            _selected_representativeness_rank_analysis(candidates)
        ),
        "subject_pose": {
            "face_detect_rate": (
                len(face_detected) / len(pose_succeeded)
                if pose_succeeded
                else None
            ),
            "face_area_ratio": _distribution(
                [
                    value
                    for record in subjects
                    if (
                        value := _successful_metric(
                            record,
                            "subject_pose",
                            "face_bbox_area_ratio",
                        )
                    )
                    is not None
                ]
            ),
            "yaw": _distribution(
                [
                    value
                    for record in subjects
                    if (
                        value := _successful_metric(record, "subject_pose", "yaw")
                    )
                    is not None
                ]
            ),
            "pitch": _distribution(
                [
                    value
                    for record in subjects
                    if (
                        value := _successful_metric(
                            record,
                            "subject_pose",
                            "pitch",
                        )
                    )
                    is not None
                ]
            ),
            "production_viewpoint_association": _pose_by_production_viewpoint(
                subjects
            ),
        },
        "selected_vs_non_selected": _selected_analysis(candidates),
        "production_route_analysis": _route_analysis(records),
        "focus_cases": focus_cases,
        "review_lists": {
            "bottom_10_percent_quality": _review_list(
                records,
                "quality",
                "quality_score",
            ),
            "bottom_10_percent_aesthetic": _review_list(
                records,
                "quality",
                "aesthetic_score",
            ),
            "bottom_10_percent_representativeness": _review_list(
                records,
                "embedding",
                "representativeness_score",
            ),
            "bottom_10_percent_inter_entity_margin": _review_list(
                records,
                "embedding",
                "inter_entity_margin",
            ),
            "largest_absolute_yaw": _review_list(
                subjects,
                "subject_pose",
                "yaw",
                descending=True,
                absolute=True,
            ),
            "smallest_face_area": _review_list(
                subjects,
                "subject_pose",
                "face_bbox_area_ratio",
            ),
        },
        "runtime": {
            "quality_model": _runtime_summary(runtimes["quality"]),
            "visual_encoder": visual_runtime,
            "subject_pose": _runtime_summary(runtimes["subject_pose"]),
            "qwen_candidate_reference_mean_seconds": "approximately 10-12",
        },
        "thresholds_applied": False,
        "production_filtering_applied": False,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _validated_output_root(output_root: Path, run_root: Path) -> Path:
    output = output_root.expanduser().resolve(strict=False)
    allowed = ALLOWED_AUDIT_ROOT.expanduser().resolve(strict=False)
    source = run_root.expanduser().resolve(strict=True)
    if output == allowed or allowed not in output.parents:
        raise ValueError("audit output must be below the allowed audit root")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("audit output must be separate from source run_root")
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    return output


def run_reference_filter_audit(
    config: V3Config,
    *,
    run_root: Path,
    output_root: Path,
    artifact_scope: ArtifactScope = "both",
    quality_backend: str = "none",
    embedding_backend: str = "none",
    subject_pose_backend: str = "none",
    quality_scorer: ReferenceQualityScorer | None = None,
    embedding_scorer: ReferenceDiscriminabilityScorer | None = None,
    subject_pose_scorer: SubjectPoseScorer | None = None,
    fail_fast: bool = False,
    discoveries: list[dict[str, object]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if artifact_scope not in {"candidates", "final", "both"}:
        raise ValueError("audit artifact_scope is invalid")
    source = run_root.expanduser().resolve(strict=True)
    destination = _validated_output_root(output_root, source)
    before = snapshot_run_files(source)
    replay_config = replace(config, run_root=source)
    replay_config.validate()
    storage = ReadOnlyAuditStorage(replay_config)
    storage.read_run()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"audit temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        artifacts = _collect_artifacts(replay_config, storage, artifact_scope)
        runtimes = _score_artifacts(
            artifacts,
            quality_backend=quality_backend,
            embedding_backend=embedding_backend,
            subject_pose_backend=subject_pose_backend,
            quality_scorer=quality_scorer,
            embedding_scorer=embedding_scorer,
            subject_pose_scorer=subject_pose_scorer,
            cache_dir=temporary / "cache",
            fail_fast=fail_fast,
            clock=clock,
        )
        _populate_embedding_metrics(artifacts)
        records = [artifact.record for artifact in artifacts]
        after = snapshot_run_files(source)
        if before != after:
            raise RuntimeError("source run changed during reference filter audit")
        discovery_values = discoveries if discoveries is not None else []
        summary = _summary(
            records,
            artifact_scope=artifact_scope,
            quality_backend=quality_backend,
            embedding_backend=embedding_backend,
            subject_pose_backend=subject_pose_backend,
            runtimes=runtimes,
            discoveries=discovery_values,
            source_unchanged=True,
        )
        _write_jsonl(temporary / "audit.jsonl", records)
        (temporary / "audit.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "model_discovery.json").write_text(
            json.dumps(discovery_values, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return summary
    except Exception as exc:
        source_changed = before != snapshot_run_files(source)
        if temporary.exists():
            shutil.rmtree(temporary)
        if source_changed:
            raise RuntimeError(
                "source run changed during failed reference filter audit"
            ) from exc
        raise
