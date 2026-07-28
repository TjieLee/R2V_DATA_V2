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
        "isolation",
        "crop_subject_ratio",
        "sam_confidence",
    }
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
class QwenConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 3600
    repair_retries: int = 1
    video: QwenVideoConfig = field(default_factory=QwenVideoConfig)


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
        "isolation": 0.04,
        "crop_subject_ratio": 0.03,
        "sam_confidence": 0.02,
    }


@dataclass(frozen=True)
class RankingConfig:
    minimum_effective_short_side: int = 128
    minimum_mask_area_ratio: float = 0.005
    maximum_mask_area_ratio: float = 0.75
    reject_border_touch: bool = True
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
    enable_in_pair: bool = True
    enable_same_parent_cross_pair: bool = True
    cross_pair_minimum_confidence: float = 0.90
    cross_pair_fallback_to_in_pair: bool = True
    maximum_candidates_per_entity: int = 10


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
    qwen: QwenConfig = field(default_factory=QwenConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def validate_paths(self) -> None:
        dataset = self.dataset_json.expanduser().resolve(strict=False)
        if not _is_at_or_below(dataset, ALLOWED_DATASET_ROOT):
            raise ValueError(
                "dataset_json must be inside /mnt/workspace/public/dataset"
            )
        model_path = Path(self.qwen.model).expanduser()
        if model_path.is_absolute() or model_path.exists():
            resolved_model = model_path.resolve(strict=False)
            if not _is_at_or_below(resolved_model, ALLOWED_PRETRAINED_ROOT):
                raise ValueError(
                    "local qwen.model must be inside /mnt/workspace/public/pretrained"
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

    def ensure_output_root(self) -> Path:
        self.validate_paths()
        output = self.output_root.expanduser().resolve()
        if not _is_at_or_below(output, ALLOWED_OUTPUT_ROOT):
            raise ValueError("output_root must be inside /mnt/workspace/litengjie/data")
        output.mkdir(parents=True, exist_ok=True)
        return output


def _path_or_none(value: object) -> Path | None:
    return None if value in (None, "") else Path(str(value)).expanduser()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")

    qwen = dict(raw.get("qwen", {}))
    qwen_video = dict(qwen.pop("video", {}))
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
    pairing = dict(raw.get("pairing", {}))
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
        qwen=QwenConfig(**qwen, video=QwenVideoConfig(**qwen_video)),
        frames=FramesConfig(**frames),
        sam3=Sam3Config(**sam3),
        ranking=RankingConfig(
            **ranking,
            evaluators=RankingEvaluatorsConfig(
                qwen_visual=QwenVisualEvaluatorConfig(**qwen_visual_evaluator),
                dinov3=DinoEvaluatorConfig(**dino_evaluator),
                siglip2=SiglipEvaluatorConfig(**siglip_evaluator),
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
    if config.qwen.temperature != 0.0:
        raise ValueError("qwen.temperature must be 0.0 for deterministic annotation")
    if config.qwen.max_tokens > 2048:
        raise ValueError("qwen.max_tokens must not exceed 2048")
    if config.qwen.video.input_mode != "full_video":
        raise ValueError("qwen.video.input_mode currently only supports full_video")
    if config.qwen.video.fps <= 0:
        raise ValueError("qwen.video.fps must be positive")
    if config.qwen.video.max_pixels is not None and config.qwen.video.max_pixels < 1:
        raise ValueError("qwen.video.max_pixels must be positive when configured")
    if config.qwen.video.total_pixels is not None and config.qwen.video.total_pixels < 1:
        raise ValueError("qwen.video.total_pixels must be positive when configured")
    if config.sam3.minimum_visible_frames < 1:
        raise ValueError("sam3.minimum_visible_frames must be positive")
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


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "dataset_json": str(config.dataset_json),
        "output_root": str(config.output_root),
        "qwen": {
            **vars(config.qwen),
            "video": vars(config.qwen.video),
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
            "dinov3_repo_dir": str(config.ranking.dinov3_repo_dir),
            "dinov3_model_path": (
                str(config.ranking.dinov3_model_path)
                if config.ranking.dinov3_model_path
                else None
            ),
            "siglip2_model_path": str(config.ranking.siglip2_model_path),
        },
        "pairing": vars(config.pairing),
        "augmentation": vars(config.augmentation),
    }
