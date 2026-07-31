from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

ALLOWED_WRITABLE_ROOT = Path("/mnt/workspace/litengjie/data").resolve()
ALLOWED_DATASET_ROOT = Path("/mnt/workspace/public/dataset").resolve()
ALLOWED_PRETRAINED_ROOT = Path("/mnt/workspace/public/pretrained").resolve()
ALLOWED_USER_MODEL_ROOT = Path("/mnt/workspace/litengjie/data/models").resolve()

ANNOTATION_MODEL_RELATIVE_PATH = Path("Qwen/Qwen3-VL-32B-Instruct")
REMOVE_MODEL_RELATIVE_PATH = Path("Qwen/Qwen-Image-Edit-2511")
REMOVE_ADAPTER_NAME = "Qwen-Image-Edit-2511-Object-Remover"
REMOVE_BACKEND = "qwen_image_edit_2511_object_remover"

_T = TypeVar("_T")


def _default_annotation_model() -> str:
    return str(ALLOWED_PRETRAINED_ROOT / ANNOTATION_MODEL_RELATIVE_PATH)


def _default_remove_model() -> Path:
    return ALLOWED_PRETRAINED_ROOT / REMOVE_MODEL_RELATIVE_PATH


def _default_remove_adapter() -> Path:
    return ALLOWED_USER_MODEL_ROOT / REMOVE_ADAPTER_NAME


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_strictly_below(path: Path, root: Path) -> bool:
    return root in path.parents


@dataclass(frozen=True)
class QwenVideoConfig:
    input_mode: str = "full_video"
    fps: float = 2.0
    do_sample_frames: bool = False


@dataclass(frozen=True)
class QwenServiceConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = field(default_factory=_default_annotation_model)
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 3600


@dataclass(frozen=True)
class QwenAnnotationConfig(QwenServiceConfig):
    max_tokens: int = 4096
    repair_retries: int = 1
    video: QwenVideoConfig = field(default_factory=QwenVideoConfig)


@dataclass(frozen=True)
class QwenServicesConfig:
    annotation: QwenAnnotationConfig = field(default_factory=QwenAnnotationConfig)
    instruction_writer: QwenServiceConfig = field(default_factory=QwenServiceConfig)
    candidate_judge: QwenServiceConfig | None = None
    background_remove_judge: QwenServiceConfig | None = None
    cross_pair_judge: QwenServiceConfig | None = None


@dataclass(frozen=True)
class SourceConfig:
    start_index: int = 0
    limit: int | None = None
    allow_full_run: bool = False


@dataclass(frozen=True)
class FramesConfig:
    count: int = 10


@dataclass(frozen=True)
class Sam3Config:
    backend: str = "sam3"
    model_path: Path | None = None
    device: str = "cuda"
    save_debug_overlays: bool = False


@dataclass(frozen=True)
class CoverageConfig:
    required_visible_frames: int = 7


@dataclass(frozen=True)
class ReferenceScopeConfig:
    enabled: bool = True
    allow_local: bool = True
    allow_synthetic_completion: bool = False


@dataclass(frozen=True)
class BackgroundConfig:
    enabled: bool = True
    raw_foreground_area_ratio: float = 0.0
    max_pending_remove_area_ratio: float = 0.50


@dataclass(frozen=True)
class RemoveConfig:
    enabled: bool = True
    backend: str = REMOVE_BACKEND
    base_model_path: Path = field(default_factory=_default_remove_model)
    adapter_path: Path | None = field(default_factory=_default_remove_adapter)
    candidate_seeds: tuple[int, ...] = (0, 17)
    fallback_to_raw: bool = False
    preserve_unmasked_pixels: bool = True


@dataclass(frozen=True)
class InstructionConfig:
    enabled: bool = True
    repair_retries: int = 1


@dataclass(frozen=True)
class DebugConfig:
    save_diagnostics: bool = False


@dataclass(frozen=True)
class V3Config:
    dataset_json: Path
    run_root: Path
    export_root: Path
    source: SourceConfig = field(default_factory=SourceConfig)
    qwen: QwenServicesConfig = field(default_factory=QwenServicesConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    reference_scope: ReferenceScopeConfig = field(
        default_factory=ReferenceScopeConfig
    )
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    remove: RemoveConfig = field(default_factory=RemoveConfig)
    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @property
    def resolved_run_root(self) -> Path:
        return self.run_root.expanduser().resolve(strict=False)

    @property
    def resolved_export_root(self) -> Path:
        return self.export_root.expanduser().resolve(strict=False)

    def validate(self) -> None:
        dataset = self.dataset_json.expanduser().resolve(strict=False)
        run_root = self.resolved_run_root
        export_root = self.resolved_export_root
        if not _is_at_or_below(dataset, ALLOWED_DATASET_ROOT):
            raise ValueError(
                "dataset_json must be inside /mnt/workspace/public/dataset"
            )
        if dataset.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("dataset_json must use .json or .jsonl")
        if (
            not isinstance(self.source.start_index, int)
            or isinstance(self.source.start_index, bool)
            or self.source.start_index < 0
        ):
            raise ValueError("source.start_index must be a non-negative integer")
        if self.source.limit is not None and (
            not isinstance(self.source.limit, int)
            or isinstance(self.source.limit, bool)
            or self.source.limit < 1
        ):
            raise ValueError("source.limit must be a positive integer")
        if not isinstance(self.source.allow_full_run, bool):
            raise TypeError("source.allow_full_run must be a boolean")
        if self.source.limit is None and not self.source.allow_full_run:
            raise ValueError(
                "source.limit is required unless source.allow_full_run is true"
            )
        for field_name, path in (
            ("run_root", run_root),
            ("export_root", export_root),
        ):
            if not _is_strictly_below(path, ALLOWED_WRITABLE_ROOT):
                raise ValueError(
                    f"{field_name} must be inside /mnt/workspace/litengjie/data"
                )
        if (
            run_root == export_root
            or run_root in export_root.parents
            or export_root in run_root.parents
        ):
            raise ValueError("run_root and export_root must be separate sibling trees")
        for service_name, service in self.qwen_services():
            model_path = Path(service.model).expanduser()
            if model_path.is_absolute() and not _is_at_or_below(
                model_path.resolve(strict=False),
                ALLOWED_PRETRAINED_ROOT,
            ):
                raise ValueError(
                    f"qwen.{service_name}.model must be inside "
                    "/mnt/workspace/public/pretrained"
                )
            if service.temperature != 0.0:
                raise ValueError(f"qwen.{service_name}.temperature must be 0.0")
            if service.max_tokens < 1:
                raise ValueError(f"qwen.{service_name}.max_tokens must be positive")
            if service.timeout_seconds < 1:
                raise ValueError(
                    f"qwen.{service_name}.timeout_seconds must be positive"
                )
        if self.qwen.annotation.video.input_mode != "full_video":
            raise ValueError("qwen.annotation.video.input_mode must be full_video")
        if self.qwen.annotation.video.fps != 2.0:
            raise ValueError("V3 annotation requires qwen video fps to be 2.0")
        if self.qwen.annotation.video.do_sample_frames:
            raise ValueError(
                "V3 annotation must disable HF video re-sampling because "
                "vLLM already samples the decoded video at the requested fps"
            )
        if (
            not isinstance(self.qwen.annotation.repair_retries, int)
            or isinstance(self.qwen.annotation.repair_retries, bool)
            or self.qwen.annotation.repair_retries < 0
        ):
            raise ValueError(
                "qwen.annotation.repair_retries must be a non-negative integer"
            )
        if self.frames.count != 10:
            raise ValueError("V3 requires exactly 10 sampled frames")
        if self.sam3.backend != "sam3":
            raise ValueError(f"unsupported V3 SAM3 backend: {self.sam3.backend}")
        if not isinstance(self.sam3.device, str) or not self.sam3.device.strip():
            raise ValueError("sam3.device must be a non-empty string")
        if not isinstance(self.sam3.save_debug_overlays, bool):
            raise TypeError("sam3.save_debug_overlays must be a boolean")
        if self.sam3.model_path is not None:
            sam3_model = self.sam3.model_path.expanduser().resolve(
                strict=False
            )
            if not (
                _is_at_or_below(sam3_model, ALLOWED_PRETRAINED_ROOT)
                or _is_at_or_below(sam3_model, ALLOWED_USER_MODEL_ROOT)
            ):
                raise ValueError(
                    "sam3.model_path must be inside an allowed model root"
                )
        if (
            not isinstance(self.coverage.required_visible_frames, int)
            or isinstance(self.coverage.required_visible_frames, bool)
            or not (
                1
                <= self.coverage.required_visible_frames
                <= self.frames.count
            )
        ):
            raise ValueError(
                "coverage.required_visible_frames must be between 1 and "
                "frames.count"
            )
        if (
            not isinstance(self.instruction.repair_retries, int)
            or isinstance(self.instruction.repair_retries, bool)
            or self.instruction.repair_retries < 0
        ):
            raise ValueError(
                "instruction.repair_retries must be a non-negative integer"
            )
        if self.reference_scope.allow_synthetic_completion:
            raise ValueError("V3 does not allow synthetic entity completion")
        if self.background.raw_foreground_area_ratio != 0.0:
            raise ValueError("V3 background.raw_foreground_area_ratio must be 0.0")
        max_pending_ratio = self.background.max_pending_remove_area_ratio
        if (
            not isinstance(max_pending_ratio, float)
            or not math.isfinite(max_pending_ratio)
            or not 0 < max_pending_ratio <= 1
        ):
            raise ValueError(
                "background.max_pending_remove_area_ratio must be a finite "
                "float greater than 0 and at most 1"
            )
        if self.remove.backend != REMOVE_BACKEND:
            raise ValueError(f"unsupported V3 remove backend: {self.remove.backend}")
        if self.remove.fallback_to_raw:
            raise ValueError("V3 remove.fallback_to_raw must be false")
        if not self.remove.preserve_unmasked_pixels:
            raise ValueError("V3 remove.preserve_unmasked_pixels must be true")
        if not self.remove.candidate_seeds:
            raise ValueError("remove.candidate_seeds must not be empty")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in self.remove.candidate_seeds
        ):
            raise ValueError("remove.candidate_seeds must contain integers")
        if len(self.remove.candidate_seeds) != len(set(self.remove.candidate_seeds)):
            raise ValueError("remove.candidate_seeds must be unique")
        if any(seed < 0 for seed in self.remove.candidate_seeds):
            raise ValueError("remove.candidate_seeds must be non-negative")
        remove_model = self.remove.base_model_path.expanduser().resolve(strict=False)
        if not _is_at_or_below(remove_model, ALLOWED_PRETRAINED_ROOT):
            raise ValueError(
                "remove.base_model_path must be inside "
                "/mnt/workspace/public/pretrained"
            )
        if self.remove.adapter_path is not None:
            adapter = self.remove.adapter_path.expanduser().resolve(strict=False)
            if not _is_at_or_below(adapter, ALLOWED_USER_MODEL_ROOT):
                raise ValueError(
                    "remove.adapter_path must be inside "
                    "/mnt/workspace/litengjie/data/models"
                )

    def qwen_services(self) -> list[tuple[str, QwenServiceConfig]]:
        services: list[tuple[str, QwenServiceConfig]] = [
            ("annotation", self.qwen.annotation),
            ("instruction_writer", self.qwen.instruction_writer),
        ]
        for name in (
            "candidate_judge",
            "background_remove_judge",
            "cross_pair_judge",
        ):
            service = getattr(self.qwen, name)
            if service is not None:
                services.append((name, service))
        return services

    def model_identifiers(self) -> dict[str, str | None]:
        return {
            **{
                f"qwen.{name}": service.model
                for name, service in self.qwen_services()
            },
            "remove.backend": self.remove.backend,
            "remove.base_model": str(self.remove.base_model_path),
            "remove.adapter": (
                str(self.remove.adapter_path)
                if self.remove.adapter_path is not None
                else None
            ),
            "sam3.backend": self.sam3.backend,
            "sam3.model": (
                str(self.sam3.model_path)
                if self.sam3.model_path is not None
                else None
            ),
            "sam3.device": self.sam3.device,
        }

    def fingerprint(self) -> str:
        value = _json_compatible(asdict(self))
        qwen = value.get("qwen")
        if isinstance(qwen, dict):
            for service in qwen.values():
                if isinstance(service, dict) and "api_key" in service:
                    service["api_key"] = "<redacted>"
        payload = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _json_compatible(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _mapping(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _build(cls: type[_T], values: dict[str, object], field_name: str) -> _T:
    try:
        return cls(**values)
    except TypeError as exc:
        raise TypeError(f"invalid {field_name} configuration: {exc}") from exc


def _parse_service(
    value: object,
    *,
    field_name: str,
    annotation: bool = False,
) -> QwenServiceConfig:
    values = _mapping(value, field_name)
    if annotation:
        video = _build(
            QwenVideoConfig,
            _mapping(values.pop("video", None), f"{field_name}.video"),
            f"{field_name}.video",
        )
        return _build(
            QwenAnnotationConfig,
            {**values, "video": video},
            field_name,
        )
    if "video" in values:
        raise ValueError(f"{field_name} is text/image-only and does not accept video")
    return _build(QwenServiceConfig, values, field_name)


def _parse_optional_service(
    value: object,
    *,
    field_name: str,
) -> QwenServiceConfig | None:
    if value is None:
        return None
    return _parse_service(value, field_name=field_name)


def load_config(path: str | Path) -> V3Config:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")
    allowed = {
        "dataset_json",
        "run_root",
        "export_root",
        "source",
        "qwen",
        "frames",
        "sam3",
        "coverage",
        "reference_scope",
        "background",
        "remove",
        "instruction",
        "debug",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown V3 configuration keys: {unknown}")
    missing = [
        name
        for name in ("dataset_json", "run_root", "export_root")
        if name not in raw
    ]
    if missing:
        raise ValueError(f"missing required V3 configuration keys: {missing}")

    qwen_values = _mapping(raw.get("qwen"), "qwen")
    qwen_unknown = sorted(
        set(qwen_values)
        - {
            "annotation",
            "instruction_writer",
            "candidate_judge",
            "background_remove_judge",
            "cross_pair_judge",
        }
    )
    if qwen_unknown:
        raise ValueError(f"unknown V3 qwen services: {qwen_unknown}")
    qwen = QwenServicesConfig(
        annotation=_parse_service(
            qwen_values.get("annotation"),
            field_name="qwen.annotation",
            annotation=True,
        ),
        instruction_writer=_parse_service(
            qwen_values.get("instruction_writer"),
            field_name="qwen.instruction_writer",
        ),
        candidate_judge=_parse_optional_service(
            qwen_values.get("candidate_judge"),
            field_name="qwen.candidate_judge",
        ),
        background_remove_judge=_parse_optional_service(
            qwen_values.get("background_remove_judge"),
            field_name="qwen.background_remove_judge",
        ),
        cross_pair_judge=_parse_optional_service(
            qwen_values.get("cross_pair_judge"),
            field_name="qwen.cross_pair_judge",
        ),
    )
    remove_values = _mapping(raw.get("remove"), "remove")
    sam3_values = _mapping(raw.get("sam3"), "sam3")
    if "model_path" in sam3_values:
        model_path = sam3_values["model_path"]
        sam3_values["model_path"] = (
            None
            if model_path in (None, "")
            else Path(str(model_path)).expanduser()
        )
    if "base_model_path" in remove_values:
        remove_values["base_model_path"] = Path(
            str(remove_values["base_model_path"])
        ).expanduser()
    if "adapter_path" in remove_values:
        adapter = remove_values["adapter_path"]
        remove_values["adapter_path"] = (
            None if adapter in (None, "") else Path(str(adapter)).expanduser()
        )
    if "candidate_seeds" in remove_values:
        seeds = remove_values["candidate_seeds"]
        if not isinstance(seeds, list):
            raise TypeError("remove.candidate_seeds must be a list")
        remove_values["candidate_seeds"] = tuple(seeds)

    config = V3Config(
        dataset_json=Path(str(raw["dataset_json"])).expanduser(),
        run_root=Path(str(raw["run_root"])).expanduser(),
        export_root=Path(str(raw["export_root"])).expanduser(),
        source=_build(
            SourceConfig,
            _mapping(raw.get("source"), "source"),
            "source",
        ),
        qwen=qwen,
        frames=_build(
            FramesConfig,
            _mapping(raw.get("frames"), "frames"),
            "frames",
        ),
        sam3=_build(
            Sam3Config,
            sam3_values,
            "sam3",
        ),
        coverage=_build(
            CoverageConfig,
            _mapping(raw.get("coverage"), "coverage"),
            "coverage",
        ),
        reference_scope=_build(
            ReferenceScopeConfig,
            _mapping(raw.get("reference_scope"), "reference_scope"),
            "reference_scope",
        ),
        background=_build(
            BackgroundConfig,
            _mapping(raw.get("background"), "background"),
            "background",
        ),
        remove=_build(RemoveConfig, remove_values, "remove"),
        instruction=_build(
            InstructionConfig,
            _mapping(raw.get("instruction"), "instruction"),
            "instruction",
        ),
        debug=_build(
            DebugConfig,
            _mapping(raw.get("debug"), "debug"),
            "debug",
        ),
    )
    config.validate()
    return config
