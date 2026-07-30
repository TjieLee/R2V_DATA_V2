from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_OUTPUT_ROOT = Path("/mnt/workspace/litengjie/data").resolve()
ALLOWED_DATASET_ROOT = Path("/mnt/workspace/public/dataset").resolve()
ALLOWED_PRETRAINED_ROOT = Path("/mnt/workspace/public/pretrained").resolve()
ALLOWED_USER_MODEL_ROOT = Path("/mnt/workspace/litengjie/data/models").resolve()

PRESELECTION_METRICS = frozenset(
    {
        "dino_representativeness",
        "siglip_alignment",
        "sharpness",
        "exposure",
        "isolation",
        "sam_confidence",
    }
)
FINAL_METRICS = frozenset(
    {
        "dino_representativeness",
        "qwen_completeness",
        "qwen_recognizability",
        "siglip_alignment",
        "qwen_mask_quality",
        "mask_area_continuity",
        "sharpness_exposure",
        "qwen_visual_quality",
        "inverse_qwen_occlusion",
        "qwen_canonical_view",
        "isolation",
        "border_completeness",
        "crop_subject_ratio",
        "sam_confidence",
    }
)
NORMALIZATION_METRICS = frozenset(
    {
        "dino_representativeness",
        "siglip_alignment",
        "sharpness",
        "exposure",
        "isolation",
        "sam_confidence",
        "qwen_completeness",
        "qwen_recognizability",
        "qwen_mask_quality",
        "qwen_visual_quality",
        "inverse_qwen_occlusion",
        "qwen_canonical_view",
        "mask_area_continuity",
        "border_completeness",
        "crop_subject_ratio",
    }
)
NORMALIZATION_METHODS = frozenset(
    {"identity", "minmax", "robust_minmax", "fixed_range"}
)


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass(frozen=True)
class QwenVideoConfig:
    input_mode: str = "full_video"
    fps: float = 2.0
    do_sample_frames: bool = True
    max_pixels: int | None = None
    total_pixels: int | None = None


@dataclass(frozen=True)
class QwenImageConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 3600
    repair_retries: int = 1


@dataclass(frozen=True)
class QwenConfig(QwenImageConfig):
    video: QwenVideoConfig = field(default_factory=QwenVideoConfig)


def _image_config_from_annotation(config: QwenConfig) -> QwenImageConfig:
    return QwenImageConfig(
        **{
            name: getattr(config, name)
            for name in QwenImageConfig.__dataclass_fields__
        }
    )


@dataclass(frozen=True)
class QwenServicesConfig:
    annotation: QwenConfig = field(default_factory=QwenConfig)
    candidate_judge: QwenImageConfig = field(default_factory=QwenImageConfig)
    background_judge: QwenImageConfig = field(default_factory=QwenImageConfig)
    cross_pair_judge: QwenImageConfig = field(default_factory=QwenImageConfig)
    repair_judge: QwenConfig | None = None

    @classmethod
    def from_flat(cls, config: QwenConfig) -> QwenServicesConfig:
        image_config = _image_config_from_annotation(config)
        return cls(
            annotation=config,
            candidate_judge=image_config,
            background_judge=image_config,
            cross_pair_judge=image_config,
        )

    # Compatibility for callers that still read config.qwen.<flat field>.
    @property
    def base_url(self) -> str:
        return self.annotation.base_url

    @property
    def api_key(self) -> str:
        return self.annotation.api_key

    @property
    def model(self) -> str:
        return self.annotation.model

    @property
    def temperature(self) -> float:
        return self.annotation.temperature

    @property
    def max_tokens(self) -> int:
        return self.annotation.max_tokens

    @property
    def timeout_seconds(self) -> int:
        return self.annotation.timeout_seconds

    @property
    def repair_retries(self) -> int:
        return self.annotation.repair_retries

    @property
    def video(self) -> QwenVideoConfig:
        return self.annotation.video


@dataclass(frozen=True)
class FramesConfig:
    count: int = 10
    jpeg_quality: int = 92
    max_side: int = 1280


@dataclass(frozen=True)
class Sam3Config:
    code_root: Path = Path("/mnt/workspace/litengjie/data/vendor/sam3")
    checkpoint: Path | None = None
    minimum_confidence: float = 0.55
    minimum_visible_frames: int = 2
    minimum_entity_visible_ratio: float = 0.80


@dataclass(frozen=True)
class QwenVisualEvaluatorConfig:
    enabled: bool = True
    use_for_final_score: bool = True


@dataclass(frozen=True)
class DinoEvaluatorConfig:
    enabled: bool = False
    use_for_preselection: bool = True
    use_for_final_score: bool = True
    hard_reject_outlier: bool = False


@dataclass(frozen=True)
class SiglipEvaluatorConfig:
    enabled: bool = False
    use_for_preselection: bool = True
    use_for_final_score: bool = True
    hard_reject_wrong_entity: bool = False


@dataclass(frozen=True)
class RankingEvaluatorsConfig:
    qwen_visual: QwenVisualEvaluatorConfig = field(
        default_factory=QwenVisualEvaluatorConfig
    )
    dinov3: DinoEvaluatorConfig = field(default_factory=DinoEvaluatorConfig)
    siglip2: SiglipEvaluatorConfig = field(default_factory=SiglipEvaluatorConfig)


def _default_preselection_weights() -> dict[str, float]:
    return {
        "dino_representativeness": 0.35,
        "siglip_alignment": 0.20,
        "sharpness": 0.18,
        "exposure": 0.10,
        "isolation": 0.10,
        "sam_confidence": 0.07,
    }


def _default_final_weights() -> dict[str, float]:
    return {
        "dino_representativeness": 0.23,
        "qwen_completeness": 0.17,
        "qwen_recognizability": 0.14,
        "siglip_alignment": 0.10,
        "qwen_mask_quality": 0.08,
        "mask_area_continuity": 0.05,
        "sharpness_exposure": 0.06,
        "qwen_visual_quality": 0.04,
        "inverse_qwen_occlusion": 0.04,
        "qwen_canonical_view": 0.06,
        "isolation": 0.04,
        "border_completeness": 0.03,
        "crop_subject_ratio": 0.03,
        "sam_confidence": 0.02,
    }


@dataclass(frozen=True)
class MetricNormalizationConfig:
    method: str = "identity"
    minimum: float | None = None
    maximum: float | None = None


def _default_normalization() -> dict[str, MetricNormalizationConfig]:
    return {
        "dino_representativeness": MetricNormalizationConfig(
            method="fixed_range",
            minimum=0.60,
            maximum=0.95,
        ),
        "siglip_alignment": MetricNormalizationConfig(method="robust_minmax"),
        "sharpness": MetricNormalizationConfig(method="robust_minmax"),
        **{
            name: MetricNormalizationConfig()
            for name in NORMALIZATION_METRICS
            - {"dino_representativeness", "siglip_alignment", "sharpness"}
        },
    }


@dataclass(frozen=True)
class RankingConfig:
    minimum_effective_short_side: int = 128
    minimum_mask_area_ratio: float = 0.005
    maximum_mask_area_ratio: float = 0.75
    reject_border_touch: bool = False
    top_k_for_vlm_judge: int = 3
    save_top_k_mask_rle: int = 10
    minimum_exposure_score: float = 0.35
    minimum_crop_subject_ratio: float = 0.08
    maximum_crop_subject_ratio: float = 0.92
    maximum_other_mask_overlap: float = 0.50
    evaluators: RankingEvaluatorsConfig = field(default_factory=RankingEvaluatorsConfig)
    preselection_weights: dict[str, float] = field(
        default_factory=_default_preselection_weights
    )
    final_weights: dict[str, float] = field(default_factory=_default_final_weights)
    normalization: dict[str, MetricNormalizationConfig] = field(
        default_factory=_default_normalization
    )
    dinov3_repo_dir: Path = Path("/mnt/workspace/public/pretrained/dinov3")
    dinov3_model_path: Path | None = None
    dinov3_model_name: str = "dinov3_vits16"
    dinov3_batch_size: int = 16
    dinov3_cluster_similarity_threshold: float = 0.70
    siglip2_model_path: Path = Path(
        "/mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex"
    )
    siglip2_batch_size: int = 8


@dataclass(frozen=True)
class PairingConfig:
    require_entity_reference: bool = True
    enable_in_pair: bool = True
    enable_same_parent_cross_pair: bool = True
    cross_pair_minimum_confidence: float = 0.90
    cross_pair_fallback_to_in_pair: bool = True
    maximum_candidates_per_entity: int = 10


@dataclass(frozen=True)
class BackgroundConfig:
    enabled: bool = True
    allow_legacy_candidate_masks: bool = False
    raw_foreground_area_ratio: float = 0.05
    maximum_foreground_area_ratio: float = 0.25
    maximum_hole_area_ratio: float = 0.25
    minimum_exposure_score: float = 0.35
    minimum_sharpness: float = 0.0
    qwen_judge_enabled: bool = False
    top_k_for_vlm_judge: int = 3
    maximum_incomplete_mask_foreground_distraction: float = 0.10


@dataclass(frozen=True)
class InpaintingBackgroundConfig:
    enabled: bool = True
    maximum_hole_area_ratio: float = 0.23
    maximum_generation_mask_area_ratio: float = 0.35
    prompt_mode: str = "qwen_local_background"
    candidate_seeds: list[int] = field(
        default_factory=lambda: [0, 17, 42, 123]
    )
    stop_after_first_accepted: bool = True


@dataclass(frozen=True)
class InpaintingEntityConfig:
    enabled: bool = False
    maximum_repair_area_ratio: float = 0.08
    maximum_component_area_ratio: float = 0.05
    minimum_completeness_before_repair: float = 0.70
    require_reliable_repair_mask: bool = True


@dataclass(frozen=True)
class InpaintingConsistencyConfig:
    preserve_unmasked_pixels: bool = True
    minimum_dino_similarity: float = 0.92
    minimum_siglip_similarity: float = 0.0
    maximum_siglip_similarity_drop: float = 0.05
    maximum_unmasked_l1_diff: float = 0.0
    fallback_to_raw: bool = True


@dataclass(frozen=True)
class InpaintingConfig:
    enabled: bool = False
    backend: str = "flux1_fill"
    model_path: Path | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    num_inference_steps: int = 50
    guidance_scale: float = 30.0
    strength: float = 1.0
    max_sequence_length: int = 512
    mask_dilation_pixels: int = 16
    adaptive_mask_dilation_ratio: float = 0.04
    feather_pixels: int = 4
    top_k_per_reference: int = 2
    background: InpaintingBackgroundConfig = field(
        default_factory=InpaintingBackgroundConfig
    )
    entity: InpaintingEntityConfig = field(
        default_factory=InpaintingEntityConfig
    )
    consistency: InpaintingConsistencyConfig = field(
        default_factory=InpaintingConsistencyConfig
    )


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    save_rgba: bool = True
    save_neutral_background: bool = True
    generated_background_count: int = 0
    viewpoint_count: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    dataset_json: Path
    output_root: Path
    qwen: QwenServicesConfig | QwenConfig = field(default_factory=QwenServicesConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if isinstance(self.qwen, QwenConfig):
            object.__setattr__(self, "qwen", QwenServicesConfig.from_flat(self.qwen))

    def validate_paths(self) -> None:
        dataset = self.dataset_json.expanduser().resolve(strict=False)
        if not _is_at_or_below(dataset, ALLOWED_DATASET_ROOT):
            raise ValueError(
                "dataset_json must be inside /mnt/workspace/public/dataset"
            )
        qwen_services = _qwen_services(self.qwen)
        for service_name, service in _iter_qwen_services(qwen_services):
            model_path = Path(service.model).expanduser()
            if model_path.is_absolute() or model_path.exists():
                resolved_model = model_path.resolve(strict=False)
                if not _is_at_or_below(resolved_model, ALLOWED_PRETRAINED_ROOT):
                    raise ValueError(
                        "local qwen.model must be inside "
                        "/mnt/workspace/public/pretrained; "
                        f"invalid service: qwen.{service_name}"
                    )
        ranking_model_paths = []
        if self.ranking.evaluators.dinov3.enabled:
            ranking_model_paths.append(
                ("ranking.dinov3_repo_dir", self.ranking.dinov3_repo_dir)
            )
        if (
            self.ranking.evaluators.dinov3.enabled
            and self.ranking.dinov3_model_path is not None
        ):
            ranking_model_paths.append(
                ("ranking.dinov3_model_path", self.ranking.dinov3_model_path)
            )
        if self.ranking.evaluators.siglip2.enabled:
            ranking_model_paths.append(
                ("ranking.siglip2_model_path", self.ranking.siglip2_model_path)
            )
        for field_name, ranking_model_path in ranking_model_paths:
            resolved = ranking_model_path.expanduser().resolve(strict=False)
            if not (
                _is_at_or_below(resolved, ALLOWED_PRETRAINED_ROOT)
                or _is_at_or_below(resolved, ALLOWED_USER_MODEL_ROOT)
            ):
                raise ValueError(
                    f"{field_name} must be inside "
                    "/mnt/workspace/public/pretrained or "
                    "/mnt/workspace/litengjie/data/models"
                )
        if self.inpainting.model_path is not None:
            inpainting_model_path = (
                self.inpainting.model_path.expanduser().resolve(strict=False)
            )
            if not (
                _is_at_or_below(inpainting_model_path, ALLOWED_PRETRAINED_ROOT)
                or _is_at_or_below(inpainting_model_path, ALLOWED_USER_MODEL_ROOT)
            ):
                raise ValueError(
                    "inpainting.model_path must be inside "
                    "/mnt/workspace/public/pretrained or "
                    "/mnt/workspace/litengjie/data/models"
                )

    def ensure_output_root(self) -> Path:
        self.validate_paths()
        output = self.output_root.expanduser().resolve()
        if not _is_at_or_below(output, ALLOWED_OUTPUT_ROOT):
            raise ValueError("output_root must be inside /mnt/workspace/litengjie/data")
        output.mkdir(parents=True, exist_ok=True)
        return output


def _path_or_none(value: object) -> Path | None:
    return None if value in (None, "") else Path(str(value)).expanduser()


def _qwen_services(
    value: QwenServicesConfig | QwenConfig,
) -> QwenServicesConfig:
    return value if isinstance(value, QwenServicesConfig) else QwenServicesConfig.from_flat(value)


def _iter_qwen_services(
    config: QwenServicesConfig,
) -> list[tuple[str, QwenImageConfig]]:
    services = [
        ("annotation", config.annotation),
        ("candidate_judge", config.candidate_judge),
        ("background_judge", config.background_judge),
        ("cross_pair_judge", config.cross_pair_judge),
    ]
    if config.repair_judge is not None:
        services.append(("repair_judge", config.repair_judge))
    return services


def has_inpainting_semantic_validator(config: PipelineConfig) -> bool:
    qwen = _qwen_services(config.qwen)
    return (
        config.ranking.evaluators.dinov3.enabled
        and config.ranking.evaluators.siglip2.enabled
    ) or qwen.repair_judge is not None


def has_background_inpainting_repair_judge(config: PipelineConfig) -> bool:
    return _qwen_services(config.qwen).repair_judge is not None


def _parse_annotation_config(
    values: dict[str, object],
    *,
    fallback: QwenConfig | None = None,
    field_name: str = "annotation",
) -> QwenConfig:
    merged = vars(fallback).copy() if fallback is not None else {}
    merged.pop("video", None)
    fallback_video = (
        vars(fallback.video).copy() if fallback is not None else {}
    )
    provided = dict(values)
    video_values = provided.pop("video", {})
    merged.update(provided)
    if not isinstance(video_values, dict):
        raise TypeError(f"qwen.{field_name}.video must be a mapping")
    fallback_video.update(video_values)
    return QwenConfig(**merged, video=QwenVideoConfig(**fallback_video))


def _parse_image_config(
    values: dict[str, object] | None,
    *,
    fallback: QwenImageConfig,
    field_name: str,
) -> QwenImageConfig:
    if values is None:
        return fallback
    if not isinstance(values, dict):
        raise TypeError(f"qwen.{field_name} must be a mapping")
    if "video" in values:
        raise ValueError(f"qwen.{field_name} is image-only and does not accept video")
    merged = vars(fallback).copy()
    merged.update(values)
    return QwenImageConfig(**merged)


def _parse_qwen_services(raw_qwen: object) -> QwenServicesConfig:
    if not isinstance(raw_qwen, dict):
        raise TypeError("qwen configuration must be a mapping")
    service_keys = {
        "annotation",
        "candidate_judge",
        "background_judge",
        "cross_pair_judge",
        "repair_judge",
    }
    if not service_keys.intersection(raw_qwen):
        flat = dict(raw_qwen)
        video_values = flat.pop("video", {})
        if not isinstance(video_values, dict):
            raise TypeError("qwen.video must be a mapping")
        annotation = QwenConfig(
            **flat,
            video=QwenVideoConfig(**video_values),
        )
        return QwenServicesConfig.from_flat(annotation)

    unknown = sorted(set(raw_qwen) - service_keys)
    if unknown:
        raise ValueError(f"unknown qwen service configuration keys: {unknown}")
    annotation_values = raw_qwen.get("annotation", {})
    if not isinstance(annotation_values, dict):
        raise TypeError("qwen.annotation must be a mapping")
    annotation = _parse_annotation_config(dict(annotation_values))
    annotation_image = _image_config_from_annotation(annotation)
    candidate = _parse_image_config(
        raw_qwen.get("candidate_judge"),
        fallback=annotation_image,
        field_name="candidate_judge",
    )
    background = _parse_image_config(
        raw_qwen.get("background_judge"),
        fallback=candidate,
        field_name="background_judge",
    )
    cross_pair = _parse_image_config(
        raw_qwen.get("cross_pair_judge"),
        fallback=candidate,
        field_name="cross_pair_judge",
    )
    repair_values = raw_qwen.get("repair_judge")
    if repair_values is not None and not isinstance(repair_values, dict):
        raise TypeError("qwen.repair_judge must be a mapping")
    repair = (
        _parse_annotation_config(
            dict(repair_values),
            fallback=annotation,
            field_name="repair_judge",
        )
        if repair_values is not None
        else None
    )
    return QwenServicesConfig(
        annotation=annotation,
        candidate_judge=candidate,
        background_judge=background,
        cross_pair_judge=cross_pair,
        repair_judge=repair,
    )


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")

    qwen = _parse_qwen_services(raw.get("qwen", {}))
    frames = dict(raw.get("frames", {}))
    sam3 = dict(raw.get("sam3", {}))
    ranking = dict(raw.get("ranking", {}))
    evaluator_values = dict(ranking.pop("evaluators", {}))
    qwen_visual_evaluator = dict(evaluator_values.pop("qwen_visual", {}))
    dino_evaluator = dict(evaluator_values.pop("dinov3", {}))
    siglip_evaluator = dict(evaluator_values.pop("siglip2", {}))
    if evaluator_values:
        raise ValueError(
            f"unknown ranking evaluators: {sorted(evaluator_values)}"
        )
    normalization_values = dict(ranking.pop("normalization", {}))
    normalization = _default_normalization()
    for metric_name, metric_config in normalization_values.items():
        if not isinstance(metric_config, dict):
            raise TypeError(
                f"ranking.normalization.{metric_name} must be a mapping"
            )
        normalization[metric_name] = MetricNormalizationConfig(**metric_config)
    pairing = dict(raw.get("pairing", {}))
    background = dict(raw.get("background", {}))
    inpainting = dict(raw.get("inpainting", {}))
    inpainting_background = dict(inpainting.pop("background", {}))
    inpainting_entity = dict(inpainting.pop("entity", {}))
    inpainting_consistency = dict(inpainting.pop("consistency", {}))
    inpainting["model_path"] = _path_or_none(inpainting.get("model_path"))
    augmentation = dict(raw.get("augmentation", {}))
    sam3["code_root"] = Path(
        sam3.get("code_root", "/mnt/workspace/litengjie/data/vendor/sam3")
    ).expanduser()
    sam3["checkpoint"] = _path_or_none(sam3.get("checkpoint"))
    ranking["dinov3_repo_dir"] = Path(
        ranking.get("dinov3_repo_dir", "/mnt/workspace/public/pretrained/dinov3")
    ).expanduser()
    ranking["dinov3_model_path"] = _path_or_none(ranking.get("dinov3_model_path"))
    ranking["siglip2_model_path"] = Path(
        ranking.get(
            "siglip2_model_path",
            "/mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex",
        )
    ).expanduser()

    config = PipelineConfig(
        dataset_json=Path(str(raw["dataset_json"])).expanduser(),
        output_root=Path(str(raw["output_root"])).expanduser(),
        qwen=qwen,
        frames=FramesConfig(**frames),
        sam3=Sam3Config(**sam3),
        ranking=RankingConfig(
            **ranking,
            evaluators=RankingEvaluatorsConfig(
                qwen_visual=QwenVisualEvaluatorConfig(**qwen_visual_evaluator),
                dinov3=DinoEvaluatorConfig(**dino_evaluator),
                siglip2=SiglipEvaluatorConfig(**siglip_evaluator),
            ),
            normalization=normalization,
        ),
        background=BackgroundConfig(**background),
        inpainting=InpaintingConfig(
            **inpainting,
            background=InpaintingBackgroundConfig(**inpainting_background),
            entity=InpaintingEntityConfig(**inpainting_entity),
            consistency=InpaintingConsistencyConfig(
                **inpainting_consistency
            ),
        ),
        pairing=PairingConfig(**pairing),
        augmentation=AugmentationConfig(**augmentation),
    )
    _validate_config(config)
    return config


def _validate_config(config: PipelineConfig) -> None:
    config.validate_paths()
    if config.frames.count != 10:
        raise ValueError("the MVP requires exactly 10 sampled frames")
    if not 1 <= config.frames.jpeg_quality <= 100:
        raise ValueError("frames.jpeg_quality must be between 1 and 100")
    if config.frames.max_side < 1:
        raise ValueError("frames.max_side must be positive")
    qwen = _qwen_services(config.qwen)
    for service_name, service in _iter_qwen_services(qwen):
        if service.temperature != 0.0:
            raise ValueError(
                f"qwen.{service_name}.temperature must be 0.0 for deterministic use"
            )
        if service.max_tokens < 1 or service.max_tokens > 2048:
            raise ValueError(
                f"qwen.{service_name}.max_tokens must be between 1 and 2048"
            )
        if service.timeout_seconds < 1:
            raise ValueError(
                f"qwen.{service_name}.timeout_seconds must be positive"
            )
        if service.repair_retries < 0:
            raise ValueError(
                f"qwen.{service_name}.repair_retries must be non-negative"
            )
    video_services = [("annotation", qwen.annotation)]
    if qwen.repair_judge is not None:
        video_services.append(("repair_judge", qwen.repair_judge))
    for service_name, service in video_services:
        if service.video.input_mode != "full_video":
            raise ValueError(
                f"qwen.{service_name}.video.input_mode currently only supports "
                "full_video"
            )
        if service.video.fps <= 0:
            raise ValueError(f"qwen.{service_name}.video.fps must be positive")
        if (
            service.video.max_pixels is not None
            and service.video.max_pixels < 1
        ):
            raise ValueError(
                f"qwen.{service_name}.video.max_pixels must be positive "
                "when configured"
            )
        if (
            service.video.total_pixels is not None
            and service.video.total_pixels < 1
        ):
            raise ValueError(
                f"qwen.{service_name}.video.total_pixels must be positive "
                "when configured"
            )
    if config.sam3.minimum_visible_frames < 1:
        raise ValueError("sam3.minimum_visible_frames must be positive")
    if not 0.0 < config.sam3.minimum_entity_visible_ratio <= 1.0:
        raise ValueError(
            "sam3.minimum_entity_visible_ratio must be greater than 0 and at most 1"
        )
    if not 0.0 <= config.ranking.minimum_mask_area_ratio:
        raise ValueError("ranking.minimum_mask_area_ratio must be non-negative")
    if config.ranking.maximum_mask_area_ratio > 1.0:
        raise ValueError("ranking.maximum_mask_area_ratio must not exceed 1")
    if config.ranking.minimum_mask_area_ratio >= config.ranking.maximum_mask_area_ratio:
        raise ValueError("ranking mask area bounds are invalid")
    if not 1 <= config.ranking.top_k_for_vlm_judge <= 3:
        raise ValueError("ranking.top_k_for_vlm_judge must be between 1 and 3")
    if not 0.0 <= config.ranking.minimum_exposure_score <= 1.0:
        raise ValueError("ranking.minimum_exposure_score must be between 0 and 1")
    if not (
        0.0
        <= config.ranking.minimum_crop_subject_ratio
        < config.ranking.maximum_crop_subject_ratio
        <= 1.0
    ):
        raise ValueError("ranking crop subject ratio bounds are invalid")
    if not 0.0 <= config.ranking.maximum_other_mask_overlap <= 1.0:
        raise ValueError("ranking.maximum_other_mask_overlap must be between 0 and 1")
    if config.ranking.dinov3_batch_size < 1:
        raise ValueError("ranking.dinov3_batch_size must be positive")
    if not 0.0 <= config.ranking.dinov3_cluster_similarity_threshold <= 1.0:
        raise ValueError(
            "ranking.dinov3_cluster_similarity_threshold must be between 0 and 1"
        )
    if config.ranking.siglip2_batch_size < 1:
        raise ValueError("ranking.siglip2_batch_size must be positive")
    if not (
        0.0
        <= config.background.raw_foreground_area_ratio
        <= config.background.maximum_foreground_area_ratio
        <= 1.0
    ):
        raise ValueError("background foreground area thresholds are invalid")
    if not 0.0 <= config.background.maximum_hole_area_ratio <= 1.0:
        raise ValueError(
            "background.maximum_hole_area_ratio must be between 0 and 1"
        )
    if not 0.0 <= config.background.minimum_exposure_score <= 1.0:
        raise ValueError(
            "background.minimum_exposure_score must be between 0 and 1"
        )
    if config.background.minimum_sharpness < 0:
        raise ValueError("background.minimum_sharpness must be non-negative")
    if not 1 <= config.background.top_k_for_vlm_judge <= 3:
        raise ValueError(
            "background.top_k_for_vlm_judge must be between 1 and 3"
        )
    if not (
        0.0
        <= config.background.maximum_incomplete_mask_foreground_distraction
        <= 0.10
    ):
        raise ValueError(
            "background.maximum_incomplete_mask_foreground_distraction "
            "must be between 0 and 0.10"
        )
    if config.inpainting.backend not in {"flux1_fill", "noop"}:
        raise ValueError("inpainting.backend must be flux1_fill or noop")
    if config.inpainting.enabled and config.inpainting.backend == "flux1_fill":
        if (
            config.inpainting.background.enabled
            and not has_background_inpainting_repair_judge(config)
        ):
            raise ValueError(
                "production background_hole_fill requires qwen.repair_judge"
            )
        if not has_inpainting_semantic_validator(config):
            raise ValueError(
                "production inpainting requires DINOv3+SigLIP2 or an explicit "
                "qwen.repair_judge consistency validator"
            )
        model_path = config.inpainting.model_path
        if model_path is None:
            raise ValueError(
                "inpainting.model_path is required when FLUX inpainting is enabled"
            )
        if not model_path.expanduser().resolve(strict=False).exists():
            raise FileNotFoundError(
                "FLUX inpainting model path does not exist: "
                f"{model_path.expanduser().resolve(strict=False)}"
            )
    if config.inpainting.dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            "inpainting.dtype must be float16, bfloat16, or float32"
        )
    if config.inpainting.num_inference_steps < 1:
        raise ValueError("inpainting.num_inference_steps must be positive")
    if config.inpainting.guidance_scale < 0:
        raise ValueError("inpainting.guidance_scale must be non-negative")
    if not 0.0 < config.inpainting.strength <= 1.0:
        raise ValueError("inpainting.strength must be between 0 and 1")
    if not 1 <= config.inpainting.max_sequence_length <= 512:
        raise ValueError(
            "inpainting.max_sequence_length must be between 1 and 512"
        )
    if config.inpainting.mask_dilation_pixels < 0:
        raise ValueError("inpainting.mask_dilation_pixels must be non-negative")
    if not 0.0 <= config.inpainting.adaptive_mask_dilation_ratio <= 1.0:
        raise ValueError(
            "inpainting.adaptive_mask_dilation_ratio must be between 0 and 1"
        )
    if config.inpainting.feather_pixels < 0:
        raise ValueError("inpainting.feather_pixels must be non-negative")
    if config.inpainting.top_k_per_reference < 1:
        raise ValueError("inpainting.top_k_per_reference must be positive")
    if not config.inpainting.consistency.preserve_unmasked_pixels:
        raise ValueError(
            "inpainting.consistency.preserve_unmasked_pixels must remain true"
        )
    if config.inpainting.background.prompt_mode not in {
        "generic",
        "qwen_local_background",
    }:
        raise ValueError(
            "inpainting.background.prompt_mode must be generic or "
            "qwen_local_background"
        )
    if not config.inpainting.background.candidate_seeds:
        raise ValueError(
            "inpainting.background.candidate_seeds must not be empty"
        )
    if any(
        not isinstance(seed, int) or seed < 0
        for seed in config.inpainting.background.candidate_seeds
    ):
        raise ValueError(
            "inpainting.background.candidate_seeds must contain "
            "non-negative integers"
        )
    if len(set(config.inpainting.background.candidate_seeds)) != len(
        config.inpainting.background.candidate_seeds
    ):
        raise ValueError(
            "inpainting.background.candidate_seeds must be unique"
        )
    for field_name, value in (
        (
            "inpainting.background.maximum_hole_area_ratio",
            config.inpainting.background.maximum_hole_area_ratio,
        ),
        (
            "inpainting.background.maximum_generation_mask_area_ratio",
            config.inpainting.background.maximum_generation_mask_area_ratio,
        ),
        (
            "inpainting.entity.maximum_repair_area_ratio",
            config.inpainting.entity.maximum_repair_area_ratio,
        ),
        (
            "inpainting.entity.maximum_component_area_ratio",
            config.inpainting.entity.maximum_component_area_ratio,
        ),
        (
            "inpainting.entity.minimum_completeness_before_repair",
            config.inpainting.entity.minimum_completeness_before_repair,
        ),
        (
            "inpainting.consistency.minimum_dino_similarity",
            config.inpainting.consistency.minimum_dino_similarity,
        ),
        (
            "inpainting.consistency.maximum_siglip_similarity_drop",
            config.inpainting.consistency.maximum_siglip_similarity_drop,
        ),
        (
            "inpainting.consistency.maximum_unmasked_l1_diff",
            config.inpainting.consistency.maximum_unmasked_l1_diff,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be between 0 and 1")
    if not -1.0 <= config.inpainting.consistency.minimum_siglip_similarity <= 1.0:
        raise ValueError(
            "inpainting.consistency.minimum_siglip_similarity must be between -1 and 1"
        )
    _validate_metric_weights(
        name="ranking.preselection_weights",
        weights=config.ranking.preselection_weights,
        known_metrics=PRESELECTION_METRICS,
        enabled_metrics=_enabled_preselection_metrics(config.ranking),
    )
    _validate_metric_weights(
        name="ranking.final_weights",
        weights=config.ranking.final_weights,
        known_metrics=FINAL_METRICS,
        enabled_metrics=_enabled_final_metrics(config.ranking),
    )
    _validate_normalization(config.ranking.normalization)
    if config.ranking.evaluators.dinov3.enabled:
        model_path = config.ranking.dinov3_model_path
        if model_path is None:
            raise ValueError(
                "ranking.dinov3_model_path is required when DINOv3 is enabled"
            )
        resolved_model = model_path.expanduser().resolve(strict=False)
        resolved_repo = config.ranking.dinov3_repo_dir.expanduser().resolve(
            strict=False
        )
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"DINOv3 model path does not exist: {resolved_model}"
            )
        if resolved_model.is_dir():
            missing_hf_files = [
                filename
                for filename in ("config.json", "preprocessor_config.json")
                if not (resolved_model / filename).is_file()
            ]
            if missing_hf_files:
                raise FileNotFoundError(
                    "DINOv3 Hugging Face model directory is incomplete: "
                    f"{resolved_model}; missing {', '.join(missing_hf_files)}"
                )
        elif not (resolved_repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                "DINOv3 Torch Hub repository is incomplete: "
                f"{resolved_repo}; missing hubconf.py"
            )
    if (
        config.ranking.evaluators.siglip2.enabled
        and not config.ranking.siglip2_model_path.expanduser()
        .resolve(strict=False)
        .is_dir()
    ):
        raise FileNotFoundError(
            "SigLIP 2 model directory does not exist: "
            f"{config.ranking.siglip2_model_path.expanduser().resolve(strict=False)}"
        )


def _enabled_preselection_metrics(config: RankingConfig) -> set[str]:
    enabled = set(PRESELECTION_METRICS - {"dino_representativeness", "siglip_alignment"})
    if (
        config.evaluators.dinov3.enabled
        and config.evaluators.dinov3.use_for_preselection
    ):
        enabled.add("dino_representativeness")
    if (
        config.evaluators.siglip2.enabled
        and config.evaluators.siglip2.use_for_preselection
    ):
        enabled.add("siglip_alignment")
    return enabled


def _enabled_final_metrics(config: RankingConfig) -> set[str]:
    qwen_metrics = {
        "qwen_completeness",
        "qwen_recognizability",
        "qwen_mask_quality",
        "qwen_visual_quality",
        "inverse_qwen_occlusion",
        "qwen_canonical_view",
    }
    enabled = set(
        FINAL_METRICS
        - qwen_metrics
        - {"dino_representativeness", "siglip_alignment"}
    )
    if (
        config.evaluators.qwen_visual.enabled
        and config.evaluators.qwen_visual.use_for_final_score
    ):
        enabled.update(qwen_metrics)
    if (
        config.evaluators.dinov3.enabled
        and config.evaluators.dinov3.use_for_final_score
    ):
        enabled.add("dino_representativeness")
    if (
        config.evaluators.siglip2.enabled
        and config.evaluators.siglip2.use_for_final_score
    ):
        enabled.add("siglip_alignment")
    return enabled


def _validate_metric_weights(
    *,
    name: str,
    weights: dict[str, float],
    known_metrics: frozenset[str],
    enabled_metrics: set[str],
) -> None:
    unknown = sorted(set(weights) - known_metrics)
    if unknown:
        raise ValueError(f"{name} contains unknown metrics: {unknown}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"{name} weights must be non-negative")
    if not any(weights.get(metric, 0.0) > 0 for metric in enabled_metrics):
        raise ValueError(f"{name} must have at least one enabled positive weight")


def _validate_normalization(
    normalization: dict[str, MetricNormalizationConfig],
) -> None:
    unknown = sorted(set(normalization) - NORMALIZATION_METRICS)
    if unknown:
        raise ValueError(
            f"ranking.normalization contains unknown metrics: {unknown}"
        )
    missing = sorted(NORMALIZATION_METRICS - set(normalization))
    if missing:
        raise ValueError(
            f"ranking.normalization is missing metrics: {missing}"
        )
    for metric_name, policy in normalization.items():
        if policy.method not in NORMALIZATION_METHODS:
            raise ValueError(
                f"ranking.normalization.{metric_name}.method must be one of "
                f"{sorted(NORMALIZATION_METHODS)}"
            )
        if policy.method == "fixed_range" and (
            policy.minimum is None
            or policy.maximum is None
            or policy.minimum >= policy.maximum
        ):
            raise ValueError(
                f"ranking.normalization.{metric_name} fixed_range requires "
                "minimum < maximum"
            )


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    qwen = _qwen_services(config.qwen)

    def image_service(value: QwenImageConfig) -> dict[str, object]:
        return dict(vars(value))

    return {
        "dataset_json": str(config.dataset_json),
        "output_root": str(config.output_root),
        "qwen": {
            "annotation": {
                **vars(qwen.annotation),
                "video": vars(qwen.annotation.video),
            },
            "candidate_judge": image_service(qwen.candidate_judge),
            "background_judge": image_service(qwen.background_judge),
            "cross_pair_judge": image_service(qwen.cross_pair_judge),
            "repair_judge": (
                {
                    **vars(qwen.repair_judge),
                    "video": vars(qwen.repair_judge.video),
                }
                if qwen.repair_judge is not None
                else None
            ),
        },
        "frames": vars(config.frames),
        "sam3": {
            **vars(config.sam3),
            "code_root": str(config.sam3.code_root),
            "checkpoint": (
                str(config.sam3.checkpoint) if config.sam3.checkpoint else None
            ),
        },
        "ranking": {
            **vars(config.ranking),
            "evaluators": {
                "qwen_visual": vars(config.ranking.evaluators.qwen_visual),
                "dinov3": vars(config.ranking.evaluators.dinov3),
                "siglip2": vars(config.ranking.evaluators.siglip2),
            },
            "normalization": {
                name: vars(policy)
                for name, policy in config.ranking.normalization.items()
            },
            "dinov3_repo_dir": str(config.ranking.dinov3_repo_dir),
            "dinov3_model_path": (
                str(config.ranking.dinov3_model_path)
                if config.ranking.dinov3_model_path
                else None
            ),
            "siglip2_model_path": str(config.ranking.siglip2_model_path),
        },
        "background": vars(config.background),
        "inpainting": {
            **vars(config.inpainting),
            "model_path": (
                str(config.inpainting.model_path)
                if config.inpainting.model_path is not None
                else None
            ),
            "background": vars(config.inpainting.background),
            "entity": vars(config.inpainting.entity),
            "consistency": vars(config.inpainting.consistency),
        },
        "pairing": vars(config.pairing),
        "augmentation": vars(config.augmentation),
    }
