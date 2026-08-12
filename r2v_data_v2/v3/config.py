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
REFERENCE_EDIT_BACKEND = "boogu_image_0_1_edit_turbo"
REFERENCE_EDIT_MODEL_REVISION = "hotfix-1k-20260708"
DEFAULT_MAX_ANNOTATION_ENTITIES = 5
DENSE_MAX_ANNOTATION_ENTITIES = 8

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
    entity_selection_mode: str = "default"
    video: QwenVideoConfig = field(default_factory=QwenVideoConfig)


@dataclass(frozen=True)
class QwenServicesConfig:
    annotation: QwenAnnotationConfig = field(default_factory=QwenAnnotationConfig)
    instruction_writer: QwenServiceConfig = field(default_factory=QwenServiceConfig)
    candidate_judge: QwenServiceConfig | None = None
    background_remove_judge: QwenServiceConfig | None = None
    background_final_judge: QwenServiceConfig | None = None
    cross_pair_judge: QwenServiceConfig | None = None
    reference_edit_judge: QwenServiceConfig | None = None
    reference_integrity_judge: QwenServiceConfig | None = None


@dataclass(frozen=True)
class SourceConfig:
    start_index: int = 0
    limit: int | None = None
    allow_full_run: bool = False
    selection_mode: str = "sequential"
    random_seed: int | None = None
    max_clips_per_parent: int = 1


@dataclass(frozen=True)
class FramesConfig:
    count: int = 10


@dataclass(frozen=True)
class Sam3Config:
    backend: str = "sam3"
    model_path: Path | None = None
    device: str = "cuda"
    save_debug_overlays: bool = False
    object_rescue_mode: str = "off"
    anchor_search_mode: str = "legacy"


@dataclass(frozen=True)
class CoverageConfig:
    required_visible_frames: int = 7


@dataclass(frozen=True)
class ReferenceScopeConfig:
    enabled: bool = True
    allow_local: bool = True
    allow_synthetic_completion: bool = False


@dataclass(frozen=True)
class PairConfig:
    enabled: bool = True
    entity_geometry_mode: str = "legacy"
    reference_prefilter_mode: str = "off"
    background_final_guard_mode: str = "off"
    max_candidates_per_entity: int = 3
    crop_padding_ratio: float = 0.08
    repair_retries: int = 1
    same_parent_fallback_enabled: bool = False
    same_parent_max_donor_references: int = 8


@dataclass(frozen=True)
class BackgroundConfig:
    enabled: bool = True
    raw_foreground_area_ratio: float = 0.0
    max_pending_remove_area_ratio: float = 0.50


@dataclass(frozen=True)
class RemoveConfig:
    enabled: bool = True
    backend: str = REMOVE_BACKEND
    inference_profile: str = "object_remover_4step_v1"
    base_model_path: Path = field(default_factory=_default_remove_model)
    adapter_path: Path | None = field(default_factory=_default_remove_adapter)
    candidate_seeds: tuple[int, ...] = (0, 17)
    fallback_to_raw: bool = False
    preserve_unmasked_pixels: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int = 4
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: str = " "
    generation_mask_dilation_pixels: int = 16
    max_generation_mask_area_ratio: float = 0.65
    adapter_weight_name: str | None = None
    save_rejected_candidates: bool = False


@dataclass(frozen=True)
class ReferenceEditConfig:
    enabled: bool = False
    backend: str = REFERENCE_EDIT_BACKEND
    python_executable: Path = Path(
        "/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python"
    )
    code_root: Path = Path("/mnt/workspace/litengjie/data/vendor/Boogu-Image")
    model_path: Path = Path(
        "/mnt/workspace/litengjie/data/models/"
        "Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708"
    )
    model_revision: str = REFERENCE_EDIT_MODEL_REVISION
    cuda_visible_devices: str = "0"
    target_area: int = 1024 * 1024
    alignment: int = 16
    timeout_seconds: int = 3600
    add_background_to_complete: bool = True
    fallback_policy: str = "keep_source"
    scale_collapse_fallback_guard_mode: str = "off"
    sam_max_area_growth_ratio: float = 3.0
    sam_max_significant_components: int = 4
    min_source_content_area_pixels: int = 128 * 128
    min_source_content_long_side_pixels: int = 128
    min_candidate_scale_ratio: float = 0.60
    max_candidate_center_shift: float = 0.20


@dataclass(frozen=True)
class ReferenceIntegrityConfig:
    enabled: bool = False
    mode: str = "targeted_qwen_v1"


@dataclass(frozen=True)
class RuntimeStageWorkersConfig:
    annotate: int = 2
    frames: int = 4
    segment: int = 1
    rank: int = 4
    background: int = 4
    remove: int = 1
    pair: int = 2
    reference_edit: int = 1
    reference_integrity: int = 2
    instruct: int = 2


@dataclass(frozen=True)
class RuntimeGpuWorkersConfig:
    segment: str | None = None
    remove: str | None = None
    reference_edit: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "staged_legacy"
    qwen_max_inflight: int = 2
    cpu_workers: int = 8
    worker_timeout_seconds: int = 3600
    stage_workers: RuntimeStageWorkersConfig = field(
        default_factory=RuntimeStageWorkersConfig
    )
    gpu_workers: RuntimeGpuWorkersConfig = field(
        default_factory=RuntimeGpuWorkersConfig
    )


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
    reference_scope: ReferenceScopeConfig = field(default_factory=ReferenceScopeConfig)
    pair: PairConfig = field(default_factory=PairConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    remove: RemoveConfig = field(default_factory=RemoveConfig)
    reference_edit: ReferenceEditConfig = field(default_factory=ReferenceEditConfig)
    reference_integrity: ReferenceIntegrityConfig = field(
        default_factory=ReferenceIntegrityConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
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
        if self.source.selection_mode not in {
            "sequential",
            "parent_stratified_random_v1",
        }:
            raise ValueError(
                "source.selection_mode must be sequential or "
                "parent_stratified_random_v1"
            )
        if (
            not isinstance(self.source.max_clips_per_parent, int)
            or isinstance(self.source.max_clips_per_parent, bool)
            or self.source.max_clips_per_parent < 1
        ):
            raise ValueError("source.max_clips_per_parent must be a positive integer")
        if self.source.selection_mode == "parent_stratified_random_v1":
            if not isinstance(self.source.random_seed, int) or isinstance(
                self.source.random_seed, bool
            ):
                raise ValueError(
                    "source.random_seed must be an integer for "
                    "parent_stratified_random_v1"
                )
            if self.source.limit is None:
                raise ValueError(
                    "source.limit is required for parent_stratified_random_v1"
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
        if self.qwen.annotation.entity_selection_mode not in {
            "default",
            "composition_balanced_v1",
            "reference_dense_v1",
        }:
            raise ValueError(
                "qwen.annotation.entity_selection_mode must be default, "
                "composition_balanced_v1, or reference_dense_v1"
            )
        if self.frames.count != 10:
            raise ValueError("V3 requires exactly 10 sampled frames")
        if self.sam3.backend != "sam3":
            raise ValueError(f"unsupported V3 SAM3 backend: {self.sam3.backend}")
        if not isinstance(self.sam3.device, str) or not self.sam3.device.strip():
            raise ValueError("sam3.device must be a non-empty string")
        if not isinstance(self.sam3.save_debug_overlays, bool):
            raise TypeError("sam3.save_debug_overlays must be a boolean")
        if self.sam3.object_rescue_mode not in {"off", "phrase_retry_v1"}:
            raise ValueError("sam3.object_rescue_mode must be off or phrase_retry_v1")
        if self.sam3.anchor_search_mode not in {"legacy", "progressive_v1"}:
            raise ValueError("sam3.anchor_search_mode must be legacy or progressive_v1")
        if self.sam3.model_path is not None:
            sam3_model = self.sam3.model_path.expanduser().resolve(strict=False)
            if not (
                _is_at_or_below(sam3_model, ALLOWED_PRETRAINED_ROOT)
                or _is_at_or_below(sam3_model, ALLOWED_USER_MODEL_ROOT)
            ):
                raise ValueError("sam3.model_path must be inside an allowed model root")
        if (
            not isinstance(self.coverage.required_visible_frames, int)
            or isinstance(self.coverage.required_visible_frames, bool)
            or not (1 <= self.coverage.required_visible_frames <= self.frames.count)
        ):
            raise ValueError(
                "coverage.required_visible_frames must be between 1 and frames.count"
            )
        if (
            not isinstance(self.instruction.repair_retries, int)
            or isinstance(self.instruction.repair_retries, bool)
            or self.instruction.repair_retries < 0
        ):
            raise ValueError(
                "instruction.repair_retries must be a non-negative integer"
            )
        for name, value in (
            ("enabled", self.reference_scope.enabled),
            ("allow_local", self.reference_scope.allow_local),
            (
                "allow_synthetic_completion",
                self.reference_scope.allow_synthetic_completion,
            ),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"reference_scope.{name} must be a boolean")
        if self.reference_scope.allow_synthetic_completion and not self.pair.enabled:
            raise ValueError(
                "reference_scope.allow_synthetic_completion requires pair.enabled"
            )
        if not isinstance(self.pair.enabled, bool):
            raise TypeError("pair.enabled must be a boolean")
        if self.pair.enabled and self.qwen.candidate_judge is None:
            raise ValueError(
                "qwen.candidate_judge is required when pair.enabled is true"
            )
        if self.pair.entity_geometry_mode not in {"legacy", "type_aware_v1"}:
            raise ValueError(
                "pair.entity_geometry_mode must be legacy or type_aware_v1"
            )
        if self.pair.reference_prefilter_mode not in {"off", "conservative_v1"}:
            raise ValueError(
                "pair.reference_prefilter_mode must be off or conservative_v1"
            )
        if self.pair.background_final_guard_mode not in {"off", "qwen_v1"}:
            raise ValueError("pair.background_final_guard_mode must be off or qwen_v1")
        if (
            self.pair.background_final_guard_mode == "qwen_v1"
            and self.qwen.background_final_judge is None
        ):
            raise ValueError(
                "qwen.background_final_judge is required when "
                "pair.background_final_guard_mode is qwen_v1"
            )
        if (
            not isinstance(self.pair.max_candidates_per_entity, int)
            or isinstance(self.pair.max_candidates_per_entity, bool)
            or not 1 <= self.pair.max_candidates_per_entity <= 10
        ):
            raise ValueError(
                "pair.max_candidates_per_entity must be an integer between 1 and 10"
            )
        if (
            not isinstance(self.pair.crop_padding_ratio, float)
            or not math.isfinite(self.pair.crop_padding_ratio)
            or not 0 <= self.pair.crop_padding_ratio <= 0.5
        ):
            raise ValueError(
                "pair.crop_padding_ratio must be a finite float between 0 and 0.5"
            )
        if (
            not isinstance(self.pair.repair_retries, int)
            or isinstance(self.pair.repair_retries, bool)
            or self.pair.repair_retries < 0
        ):
            raise ValueError("pair.repair_retries must be a non-negative integer")
        if not isinstance(self.pair.same_parent_fallback_enabled, bool):
            raise TypeError("pair.same_parent_fallback_enabled must be a boolean")
        if (
            not isinstance(self.pair.same_parent_max_donor_references, int)
            or isinstance(
                self.pair.same_parent_max_donor_references,
                bool,
            )
            or not 1 <= self.pair.same_parent_max_donor_references <= 64
        ):
            raise ValueError(
                "pair.same_parent_max_donor_references must be an integer "
                "between 1 and 64"
            )
        if (
            self.pair.same_parent_fallback_enabled
            and self.qwen.cross_pair_judge is None
        ):
            raise ValueError(
                "qwen.cross_pair_judge is required when "
                "pair.same_parent_fallback_enabled is true"
            )
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
        if not isinstance(self.remove.enabled, bool):
            raise TypeError("remove.enabled must be a boolean")
        if self.remove.enabled and self.qwen.background_remove_judge is None:
            raise ValueError(
                "qwen.background_remove_judge is required when remove.enabled is true"
            )
        if self.remove.backend != REMOVE_BACKEND:
            raise ValueError(f"unsupported V3 remove backend: {self.remove.backend}")
        if self.remove.fallback_to_raw:
            raise ValueError("V3 remove.fallback_to_raw must be false")
        if not self.remove.preserve_unmasked_pixels:
            raise ValueError("V3 remove.preserve_unmasked_pixels must be true")
        if len(self.remove.candidate_seeds) not in {1, 2}:
            raise ValueError("remove.candidate_seeds must contain one or two seeds")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in self.remove.candidate_seeds
        ):
            raise ValueError("remove.candidate_seeds must contain integers")
        if len(self.remove.candidate_seeds) != len(set(self.remove.candidate_seeds)):
            raise ValueError("remove.candidate_seeds must be unique")
        if any(seed < 0 for seed in self.remove.candidate_seeds):
            raise ValueError("remove.candidate_seeds must be non-negative")
        if not isinstance(self.remove.device, str) or not self.remove.device.strip():
            raise ValueError("remove.device must be a non-empty string")
        if self.remove.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("remove.dtype must be bfloat16, float16, or float32")
        if self.remove.inference_profile not in {
            "object_remover_4step_v1",
            "experimental_override",
        }:
            raise ValueError(
                "remove.inference_profile must be object_remover_4step_v1 or "
                "experimental_override"
            )
        if (
            not isinstance(self.remove.num_inference_steps, int)
            or isinstance(self.remove.num_inference_steps, bool)
            or self.remove.num_inference_steps < 1
        ):
            raise ValueError("remove.num_inference_steps must be a positive integer")
        if (
            self.remove.inference_profile == "object_remover_4step_v1"
            and self.remove.num_inference_steps != 4
        ):
            raise ValueError(
                "remove.inference_profile=object_remover_4step_v1 requires "
                "num_inference_steps=4"
            )
        if self.remove.enabled and self.remove.adapter_path is None:
            raise ValueError(
                "remove.adapter_path is required when the Object-Remover stage is enabled"
            )
        for name, value in (
            ("true_cfg_scale", self.remove.true_cfg_scale),
            ("guidance_scale", self.remove.guidance_scale),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"remove.{name} must be finite and non-negative")
        dilation = self.remove.generation_mask_dilation_pixels
        if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation < 0:
            raise ValueError(
                "remove.generation_mask_dilation_pixels must be a non-negative integer"
            )
        maximum_ratio = self.remove.max_generation_mask_area_ratio
        if (
            not isinstance(maximum_ratio, float)
            or not math.isfinite(maximum_ratio)
            or not 0 < maximum_ratio <= 1
        ):
            raise ValueError(
                "remove.max_generation_mask_area_ratio must be a finite float "
                "greater than 0 and at most 1"
            )
        if not isinstance(self.remove.save_rejected_candidates, bool):
            raise TypeError("remove.save_rejected_candidates must be a boolean")
        if self.remove.adapter_weight_name is not None and (
            not isinstance(self.remove.adapter_weight_name, str)
            or not self.remove.adapter_weight_name.strip()
        ):
            raise ValueError(
                "remove.adapter_weight_name must be a non-empty string or null"
            )

        remove_model = self.remove.base_model_path.expanduser().resolve(strict=False)
        if not _is_at_or_below(remove_model, ALLOWED_PRETRAINED_ROOT):
            raise ValueError(
                "remove.base_model_path must be inside /mnt/workspace/public/pretrained"
            )
        if self.remove.adapter_path is not None:
            adapter = self.remove.adapter_path.expanduser().resolve(strict=False)
            if not _is_at_or_below(adapter, ALLOWED_USER_MODEL_ROOT):
                raise ValueError(
                    "remove.adapter_path must be inside "
                    "/mnt/workspace/litengjie/data/models"
                )
        if not isinstance(self.reference_edit.enabled, bool):
            raise TypeError("reference_edit.enabled must be a boolean")
        if self.reference_edit.backend != REFERENCE_EDIT_BACKEND:
            raise ValueError(
                f"unsupported V3 reference_edit backend: {self.reference_edit.backend}"
            )
        if self.reference_edit.enabled and self.qwen.reference_edit_judge is None:
            raise ValueError(
                "qwen.reference_edit_judge is required when "
                "reference_edit.enabled is true"
            )
        if not isinstance(self.reference_integrity.enabled, bool):
            raise TypeError("reference_integrity.enabled must be a boolean")
        if self.reference_integrity.mode != "targeted_qwen_v1":
            raise ValueError("reference_integrity.mode must be targeted_qwen_v1")
        if (
            self.reference_integrity.enabled
            and self.qwen.reference_integrity_judge is None
        ):
            raise ValueError(
                "qwen.reference_integrity_judge is required when "
                "reference_integrity.enabled is true"
            )
        if self.runtime.mode not in {"staged_legacy", "streaming_v1"}:
            raise ValueError("runtime.mode must be staged_legacy or streaming_v1")
        for name, value in (
            ("qwen_max_inflight", self.runtime.qwen_max_inflight),
            ("cpu_workers", self.runtime.cpu_workers),
            ("worker_timeout_seconds", self.runtime.worker_timeout_seconds),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"runtime.{name} must be a positive integer")
        for name, value in asdict(self.runtime.stage_workers).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"runtime.stage_workers.{name} must be a positive integer"
                )
        for stage in ("segment", "remove", "reference_edit"):
            if getattr(self.runtime.stage_workers, stage) != 1:
                raise ValueError(
                    f"runtime.stage_workers.{stage} must be 1 to avoid duplicate "
                    "GPU model copies"
                )
        for stage, visible in asdict(self.runtime.gpu_workers).items():
            if visible is not None and (
                not isinstance(visible, str)
                or not visible.isdigit()
                or int(visible) < 0
            ):
                raise ValueError(
                    f"runtime.gpu_workers.{stage} must be one physical CUDA index"
                )
        if (
            self.runtime.mode == "streaming_v1"
            and self.pair.same_parent_fallback_enabled
        ):
            raise ValueError(
                "runtime.mode=streaming_v1 does not support "
                "pair.same_parent_fallback_enabled=true"
            )
        if self.reference_edit.enabled:
            for name, path in (
                ("python_executable", self.reference_edit.python_executable),
                ("code_root", self.reference_edit.code_root),
                ("model_path", self.reference_edit.model_path),
            ):
                if not isinstance(path, Path):
                    raise TypeError(f"reference_edit.{name} must be a pathlib.Path")
                resolved = path.expanduser().resolve(strict=False)
                if not _is_at_or_below(resolved, ALLOWED_WRITABLE_ROOT):
                    raise ValueError(
                        f"reference_edit.{name} must be inside "
                        "/mnt/workspace/litengjie/data"
                    )
        if self.reference_edit.model_revision != REFERENCE_EDIT_MODEL_REVISION:
            raise ValueError("reference_edit.model_revision must be hotfix-1k-20260708")
        if (
            not isinstance(self.reference_edit.cuda_visible_devices, str)
            or not self.reference_edit.cuda_visible_devices.strip()
        ):
            raise ValueError("reference_edit.cuda_visible_devices must be non-empty")
        if (
            not isinstance(self.reference_edit.target_area, int)
            or isinstance(self.reference_edit.target_area, bool)
            or self.reference_edit.target_area < 1
        ):
            raise ValueError("reference_edit.target_area must be a positive integer")
        if self.reference_edit.alignment != 16:
            raise ValueError("reference_edit.alignment must be 16")
        if (
            not isinstance(self.reference_edit.timeout_seconds, int)
            or isinstance(self.reference_edit.timeout_seconds, bool)
            or self.reference_edit.timeout_seconds < 1
        ):
            raise ValueError(
                "reference_edit.timeout_seconds must be a positive integer"
            )
        if self.reference_edit.add_background_to_complete is not True:
            raise ValueError("reference_edit.add_background_to_complete must be true")
        if self.reference_edit.fallback_policy not in {
            "keep_source",
            "reject_entity",
        }:
            raise ValueError(
                "reference_edit.fallback_policy must be keep_source or reject_entity"
            )
        if self.reference_edit.scale_collapse_fallback_guard_mode not in {
            "off",
            "qwen_v1",
        }:
            raise ValueError(
                "reference_edit.scale_collapse_fallback_guard_mode must be "
                "off or qwen_v1"
            )
        if (
            not isinstance(self.reference_edit.sam_max_area_growth_ratio, float)
            or not math.isfinite(self.reference_edit.sam_max_area_growth_ratio)
            or self.reference_edit.sam_max_area_growth_ratio < 1
        ):
            raise ValueError(
                "reference_edit.sam_max_area_growth_ratio must be a finite "
                "float at least 1"
            )
        if (
            not isinstance(
                self.reference_edit.sam_max_significant_components,
                int,
            )
            or isinstance(
                self.reference_edit.sam_max_significant_components,
                bool,
            )
            or self.reference_edit.sam_max_significant_components < 1
        ):
            raise ValueError(
                "reference_edit.sam_max_significant_components must be a "
                "positive integer"
            )
        for name, value in (
            (
                "min_source_content_area_pixels",
                self.reference_edit.min_source_content_area_pixels,
            ),
            (
                "min_source_content_long_side_pixels",
                self.reference_edit.min_source_content_long_side_pixels,
            ),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"reference_edit.{name} must be a positive integer")
        minimum_scale = self.reference_edit.min_candidate_scale_ratio
        if (
            not isinstance(minimum_scale, float)
            or not math.isfinite(minimum_scale)
            or not 0 < minimum_scale <= 1
        ):
            raise ValueError(
                "reference_edit.min_candidate_scale_ratio must be a finite "
                "float greater than 0 and at most 1"
            )
        maximum_shift = self.reference_edit.max_candidate_center_shift
        if (
            not isinstance(maximum_shift, float)
            or not math.isfinite(maximum_shift)
            or not 0 <= maximum_shift <= math.sqrt(2)
        ):
            raise ValueError(
                "reference_edit.max_candidate_center_shift must be a finite "
                "float between 0 and sqrt(2)"
            )

    def qwen_services(self) -> list[tuple[str, QwenServiceConfig]]:
        services: list[tuple[str, QwenServiceConfig]] = [
            ("annotation", self.qwen.annotation),
            ("instruction_writer", self.qwen.instruction_writer),
        ]
        for name in (
            "candidate_judge",
            "background_remove_judge",
            "background_final_judge",
            "cross_pair_judge",
            "reference_edit_judge",
            "reference_integrity_judge",
        ):
            service = getattr(self.qwen, name)
            if service is not None:
                services.append((name, service))
        return services

    def model_identifiers(self) -> dict[str, str | None]:
        return {
            **{f"qwen.{name}": service.model for name, service in self.qwen_services()},
            "remove.backend": self.remove.backend,
            "remove.inference_profile": self.remove.inference_profile,
            "remove.base_model": str(self.remove.base_model_path),
            "remove.adapter": (
                str(self.remove.adapter_path)
                if self.remove.adapter_path is not None
                else None
            ),
            "remove.adapter_weight_name": self.remove.adapter_weight_name,
            "remove.device": self.remove.device,
            "remove.dtype": self.remove.dtype,
            "remove.num_inference_steps": str(self.remove.num_inference_steps),
            "remove.true_cfg_scale": str(self.remove.true_cfg_scale),
            "remove.guidance_scale": str(self.remove.guidance_scale),
            "remove.negative_prompt": self.remove.negative_prompt,
            "remove.generation_mask_dilation_pixels": str(
                self.remove.generation_mask_dilation_pixels
            ),
            "remove.max_generation_mask_area_ratio": str(
                self.remove.max_generation_mask_area_ratio
            ),
            "remove.save_rejected_candidates": str(
                self.remove.save_rejected_candidates
            ).lower(),
            "reference_edit.backend": self.reference_edit.backend,
            "reference_edit.model": str(self.reference_edit.model_path),
            "reference_edit.model_revision": self.reference_edit.model_revision,
            "sam3.backend": self.sam3.backend,
            "sam3.model": (
                str(self.sam3.model_path) if self.sam3.model_path is not None else None
            ),
            "sam3.device": self.sam3.device,
            "sam3.anchor_search_mode": self.sam3.anchor_search_mode,
            "reference_integrity.mode": self.reference_integrity.mode,
            "runtime.mode": self.runtime.mode,
            "runtime.qwen_max_inflight": str(self.runtime.qwen_max_inflight),
            "runtime.stage_workers": json.dumps(
                asdict(self.runtime.stage_workers), sort_keys=True
            ),
            "runtime.gpu_workers": json.dumps(
                asdict(self.runtime.gpu_workers), sort_keys=True
            ),
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
        "pair",
        "background",
        "remove",
        "reference_edit",
        "reference_integrity",
        "runtime",
        "instruction",
        "debug",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown V3 configuration keys: {unknown}")
    missing = [
        name for name in ("dataset_json", "run_root", "export_root") if name not in raw
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
            "background_final_judge",
            "cross_pair_judge",
            "reference_edit_judge",
            "reference_integrity_judge",
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
        background_final_judge=_parse_optional_service(
            qwen_values.get("background_final_judge"),
            field_name="qwen.background_final_judge",
        ),
        cross_pair_judge=_parse_optional_service(
            qwen_values.get("cross_pair_judge"),
            field_name="qwen.cross_pair_judge",
        ),
        reference_edit_judge=_parse_optional_service(
            qwen_values.get("reference_edit_judge"),
            field_name="qwen.reference_edit_judge",
        ),
        reference_integrity_judge=_parse_optional_service(
            qwen_values.get("reference_integrity_judge"),
            field_name="qwen.reference_integrity_judge",
        ),
    )
    remove_values = _mapping(raw.get("remove"), "remove")
    reference_edit_values = _mapping(
        raw.get("reference_edit"),
        "reference_edit",
    )
    runtime_values = dict(_mapping(raw.get("runtime"), "runtime"))
    runtime_stage_workers = _build(
        RuntimeStageWorkersConfig,
        _mapping(runtime_values.pop("stage_workers", None), "runtime.stage_workers"),
        "runtime.stage_workers",
    )
    runtime_gpu_workers = _build(
        RuntimeGpuWorkersConfig,
        _mapping(runtime_values.pop("gpu_workers", None), "runtime.gpu_workers"),
        "runtime.gpu_workers",
    )
    pair_values = _mapping(raw.get("pair"), "pair")
    if pair_values.get("reference_prefilter_mode") is False:
        pair_values["reference_prefilter_mode"] = "off"
    if pair_values.get("background_final_guard_mode") is False:
        pair_values["background_final_guard_mode"] = "off"
    sam3_values = _mapping(raw.get("sam3"), "sam3")
    if "model_path" in sam3_values:
        model_path = sam3_values["model_path"]
        sam3_values["model_path"] = (
            None if model_path in (None, "") else Path(str(model_path)).expanduser()
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
    for name in ("python_executable", "code_root", "model_path"):
        if name in reference_edit_values:
            reference_edit_values[name] = Path(
                str(reference_edit_values[name])
            ).expanduser()

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
        pair=_build(
            PairConfig,
            pair_values,
            "pair",
        ),
        background=_build(
            BackgroundConfig,
            _mapping(raw.get("background"), "background"),
            "background",
        ),
        remove=_build(RemoveConfig, remove_values, "remove"),
        reference_edit=_build(
            ReferenceEditConfig,
            reference_edit_values,
            "reference_edit",
        ),
        reference_integrity=_build(
            ReferenceIntegrityConfig,
            _mapping(raw.get("reference_integrity"), "reference_integrity"),
            "reference_integrity",
        ),
        runtime=_build(
            RuntimeConfig,
            {
                **runtime_values,
                "stage_workers": runtime_stage_workers,
                "gpu_workers": runtime_gpu_workers,
            },
            "runtime",
        ),
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
