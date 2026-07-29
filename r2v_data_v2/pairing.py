from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import BadRequestError, OpenAI

from prompts.qwen_cross_pair_prompt import CROSS_PAIR_PROMPT
from r2v_data_v2.config import (
    PairingConfig,
    PipelineConfig,
    QwenImageConfig,
    _qwen_services,
)
from r2v_data_v2.image_utils import image_data_uri
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.phrase_alignment import resolve_background_caption_phrase
from r2v_data_v2.reconciliation import reconcile_final_samples, write_json_atomic
from r2v_data_v2.reference_binding import (
    ReferenceBindingError,
    bind_background_reference,
    rebuild_for_retained_entities,
    validate_final_reference_binding,
)
from r2v_data_v2.schemas import AnnotationResult, CrossPairJudgeResult
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    request_structured_output,
)
from r2v_data_v2.visual_embedding import (
    embedding_cosine_similarity,
    load_selected_dinov3_embedding,
)


@dataclass(frozen=True)
class PairingStats:
    processed: int = 0
    skipped_existing: int = 0
    in_pair_count: int = 0
    cross_pair_count: int = 0
    fallback_count: int = 0
    failed: int = 0
    skipped_no_ready_reference: int = 0
    skipped_no_bindable_reference: int = 0


CrossPairKey = tuple[str, str, str]


def cross_pair_index_key(reference: dict[str, Any]) -> CrossPairKey:
    return (
        str(reference["parent_video_id"]),
        str(reference["category"]).casefold(),
        str(reference["canonical_label"]).casefold(),
    )


def build_cross_pair_index(
    references_by_clip: dict[str, list[dict[str, Any]]],
) -> dict[CrossPairKey, list[dict[str, Any]]]:
    index: dict[CrossPairKey, list[dict[str, Any]]] = {}
    for references in references_by_clip.values():
        for reference in references:
            if (
                reference.get("status", "ready") != "ready"
                or reference.get("rejected") is True
            ):
                continue
            if reference.get("reference_type", "entity") == "background":
                continue
            index.setdefault(cross_pair_index_key(reference), []).append(reference)
    return index


def indexed_cross_pair_candidates(
    index: dict[CrossPairKey, list[dict[str, Any]]],
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in index.get(cross_pair_index_key(target), [])
        if candidate["clip_uid"] != target["clip_uid"]
        and candidate["clip_suffix"] != target["clip_suffix"]
    ]


def is_same_parent_cross_candidate(
    target: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if target["clip_uid"] == candidate["clip_uid"]:
        return False
    if target["parent_video_id"] != candidate["parent_video_id"]:
        return False
    if target["clip_suffix"] == candidate["clip_suffix"]:
        return False
    if str(target["category"]).lower() != str(candidate["category"]).lower():
        return False
    if (
        str(target["canonical_label"]).lower()
        != str(candidate["canonical_label"]).lower()
    ):
        return False
    target_named = target.get("genericity") == "named"
    candidate_named = candidate.get("genericity") == "named"
    if target_named or candidate_named:
        return (
            target_named
            and candidate_named
            and str(target["canonical_label"]).strip()
            == str(candidate["canonical_label"]).strip()
            and target.get("name_evidence") in {"draft_caption", "metadata"}
            and candidate.get("name_evidence") in {"draft_caption", "metadata"}
        )
    return True


def cross_pair_passes(
    result: CrossPairJudgeResult,
    *,
    minimum_confidence: float,
) -> bool:
    return (
        result.same_exact_instance == "yes"
        and result.confidence >= minimum_confidence
        and not result.near_duplicate
        and result.context_difference in {"moderate", "large"}
        and not result.conflicting_attributes
    )


def visual_histogram_similarity(left_path: str | Path, right_path: str | Path) -> float:
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise FileNotFoundError("cross-pair reference image is unreadable")
    left_hist = cv2.calcHist([left], [0, 1], None, [32, 32], [0, 256, 0, 256])
    right_hist = cv2.calcHist([right], [0, 1], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(left_hist, left_hist)
    cv2.normalize(right_hist, right_hist)
    correlation = float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _reference_dino_embedding(
    reference: dict[str, Any],
) -> np.ndarray | None:
    configured = reference.get("dinov3_embedding_path")
    path = (
        Path(configured)
        if isinstance(configured, str) and configured
        else Path(str(reference["canonical_path"])).parent / "dinov3_embedding.npy"
    )
    if not path.is_file():
        return None
    try:
        return load_selected_dinov3_embedding(path)
    except (OSError, ValueError):
        return None


def cross_pair_coarse_similarity(
    target: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    target_embedding = _reference_dino_embedding(target)
    candidate_embedding = _reference_dino_embedding(candidate)
    if (
        target_embedding is not None
        and candidate_embedding is not None
        and target_embedding.shape == candidate_embedding.shape
    ):
        return embedding_cosine_similarity(
            target_embedding,
            candidate_embedding,
        )
    return visual_histogram_similarity(
        target["canonical_path"],
        candidate["canonical_path"],
    )


class QwenCrossPairJudge:
    def __init__(self, config: QwenImageConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    def _image(path: str | Path) -> dict[str, object]:
        return {
            "type": "image_url",
            "image_url": {"url": image_data_uri(path)},
        }

    def _request(
        self,
        *,
        prompt: str,
        target: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        self._image(target["canonical_path"]),
                        self._image(candidate["canonical_path"]),
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": min(512, self.config.max_tokens),
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cross_pair_judge",
                        "strict": True,
                        "schema": CrossPairJudgeResult.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty cross-pair decision")
        return content

    def judge(
        self,
        *,
        target: dict[str, Any],
        candidate: dict[str, Any],
    ) -> CrossPairJudgeResult:
        prompt = CROSS_PAIR_PROMPT.format(
            target_phrase=target["phrase"],
            candidate_label=candidate["canonical_label"],
        )
        return request_structured_output(
            request=lambda request_text: self._request(
                prompt=request_text,
                target=target,
                candidate=candidate,
            ),
            original_request=prompt,
            model=CrossPairJudgeResult,
        )


def _annotations_by_clip(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["clip_uid"]): record for record in iter_source_records(path)}


def _references_by_clip(
    path: Path,
    annotations: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for reference in iter_source_records(path):
        reference.setdefault("reference_type", "entity")
        reference.setdefault(
            "reference_id",
            reference.get("entity_id"),
        )
        if "status" not in reference:
            if reference.get("rejected") is True:
                reference["status"] = "rejected"
            elif (
                reference.get("reference_type") == "background"
                and reference.get("needs_inpainting") is True
                and reference.get("inpainted") is not True
            ):
                reference["status"] = "pending_inpainting"
            else:
                reference["status"] = "ready"
        if (
            reference.get("status") != "ready"
            or reference.get("rejected") is True
        ):
            continue
        clip = str(reference["clip_uid"])
        annotation = annotations[clip]
        reference.update(
            {
                "parent_video_id": annotation["parent_video_id"],
                "clip_suffix": annotation["clip_suffix"],
                "video_path": annotation["video_path"],
            }
        )
        result.setdefault(clip, []).append(reference)
    return result


def choose_cross_pair(
    *,
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: PairingConfig,
    judge: QwenCrossPairJudge,
    structured_failures: list[dict[str, object]] | None = None,
) -> tuple[dict[str, Any], CrossPairJudgeResult, float] | None:
    coarse: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if not is_same_parent_cross_candidate(target, candidate):
            continue
        similarity = cross_pair_coarse_similarity(target, candidate)
        coarse.append((similarity, candidate))
    coarse.sort(key=lambda item: item[0], reverse=True)
    passing: list[tuple[float, float, dict[str, Any], CrossPairJudgeResult]] = []
    maximum_candidates = min(10, config.maximum_candidates_per_entity)
    for similarity, candidate in coarse[:maximum_candidates]:
        try:
            result = judge.judge(target=target, candidate=candidate)
        except StructuredOutputFailure as exc:
            if structured_failures is not None:
                structured_failures.append(
                    {
                        "target_entity_id": target["entity_id"],
                        "candidate_clip_uid": candidate["clip_uid"],
                        **exc.to_dict(),
                    }
                )
            continue
        if cross_pair_passes(
            result,
            minimum_confidence=config.cross_pair_minimum_confidence,
        ):
            passing.append((result.confidence, similarity, candidate, result))
    if not passing:
        return None
    confidence, similarity, candidate, result = max(
        passing,
        key=lambda item: (item[0], item[1]),
    )
    del confidence
    return candidate, result, similarity


def _existing_sample_uids(samples_dir: Path) -> set[str]:
    result: set[str] = set()
    for artifact in samples_dir.glob("*.json"):
        value = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "clip_uid" not in value:
            raise ValueError(f"invalid final sample artifact: {artifact}")
        result.add(str(value["clip_uid"]))
    return result


def _remove_samples_with_non_ready_references(
    samples_dir: Path,
    references_by_clip: dict[str, list[dict[str, Any]]],
) -> None:
    ready_paths = {
        str(reference["canonical_path"])
        for references in references_by_clip.values()
        for reference in references
    }
    for artifact in samples_dir.glob("*.json"):
        sample = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(sample, dict):
            raise TypeError(f"final sample artifact must be an object: {artifact}")
        sample_references = sample.get("references", [])
        if not isinstance(sample_references, list) or any(
            not isinstance(reference, dict)
            or str(reference.get("image_path")) not in ready_paths
            for reference in sample_references
        ):
            artifact.unlink()


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _append_background_binding_failure(
    output_root: Path,
    *,
    original_background_phrase: str,
    issue_codes: list[str],
    clip_uid: str,
    caption: str,
    entity_reference_count: int,
) -> None:
    _append_jsonl(
        output_root / "logs" / "background_binding_failed.jsonl",
        {
            "original_background_phrase": original_background_phrase,
            "issue_codes": issue_codes,
            "clip_uid": clip_uid,
            "caption": caption,
            "entity_reference_count": entity_reference_count,
        },
    )


def _annotation_from_record(record: dict[str, Any]) -> AnnotationResult:
    return AnnotationResult.model_validate(
        {
            key: record.get(key)
            for key in (
                "caption",
                "prompt_with_refs",
                "entities",
                "relations",
                "background",
            )
        }
    )


def _target_reference(
    *,
    target: dict[str, Any],
    selected: dict[str, Any],
    pair_type: str,
    judge_result: CrossPairJudgeResult | None = None,
    visual_similarity: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "reference_id": target.get("reference_id", target["entity_id"]),
        "reference_type": "entity",
        "entity_id": target["entity_id"],
        "phrase": target["phrase"],
        "ref_token": target["ref_token"],
        "category": target["category"],
        "image_path": selected["canonical_path"],
        "mask_path": selected["mask_path"],
        "pair_type": pair_type,
        "source_clip_uid": selected["clip_uid"],
        "source_frame_index": selected["source_frame_index"],
        "ranking_score": selected["ranking_score"],
    }
    if judge_result is not None:
        result["cross_pair_judgment"] = judge_result.model_dump(mode="json")
        result["cross_pair_visual_similarity"] = visual_similarity
    return result


def _background_target_reference(
    reference: dict[str, Any],
) -> dict[str, object]:
    return {
        "reference_id": reference.get("reference_id", "bg1"),
        "reference_type": "background",
        "entity_id": None,
        "phrase": reference["phrase"],
        "source_phrase": reference["source_phrase"],
        "ref_token": "<ref_bg_1>",
        "category": "background",
        "image_path": reference["canonical_path"],
        "mask_path": reference["mask_path"],
        "pair_type": "in_pair",
        "source_clip_uid": reference["clip_uid"],
        "source_frame_index": reference["source_frame_index"],
        "ranking_score": reference["ranking_score"],
    }


def build_pairs(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    judge: QwenCrossPairJudge | None = None,
) -> PairingStats:
    output_root = config.ensure_output_root()
    annotation_path = output_root / "manifests" / "annotations.jsonl"
    reference_path = output_root / "manifests" / "references.jsonl"
    final_path = output_root / "manifests" / "final_samples.jsonl"
    samples_dir = output_root / "samples"
    if not annotation_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("run Stages 02-04 before pairing")
    if overwrite:
        final_path.unlink(missing_ok=True)
        for artifact in samples_dir.glob("*.json"):
            artifact.unlink()
    annotations = _annotations_by_clip(annotation_path)
    references = _references_by_clip(reference_path, annotations)
    if not overwrite:
        _remove_samples_with_non_ready_references(samples_dir, references)
    existing = _existing_sample_uids(samples_dir)
    cross_pair_index = build_cross_pair_index(references)
    qwen = judge or QwenCrossPairJudge(
        _qwen_services(config.qwen).cross_pair_judge
    )
    processed = skipped = no_ready = in_pairs = cross_pairs = fallbacks = failed = 0
    no_bindable = 0
    for clip, annotation in annotations.items():
        if clip in existing:
            skipped += 1
            continue
        clip_references = references.get(clip, [])
        if not clip_references:
            no_ready += 1
            continue
        try:
            target_references = [
                reference
                for reference in clip_references
                if reference.get("reference_type", "entity") == "entity"
            ]
            background_references = [
                reference
                for reference in clip_references
                if reference.get("reference_type") == "background"
            ]
            retained_annotation = rebuild_for_retained_entities(
                _annotation_from_record(annotation),
                {str(reference["entity_id"]) for reference in target_references},
            )
            retained_by_id = {
                entity.entity_id: entity
                for entity in retained_annotation.entities
                if entity.reference_worthy
            }
            for target in target_references:
                retained = retained_by_id[str(target["entity_id"])]
                target["ref_token"] = retained.ref_token
                target["phrase"] = retained.phrase
            final_references: list[dict[str, object]] = []
            warnings = list(annotation.get("warnings", []))
            sample_in_pairs = sample_cross_pairs = sample_fallbacks = 0
            structured_failures: list[dict[str, object]] = []
            for target in target_references:
                selected = target
                pair_type = "in_pair"
                judgment = None
                visual_similarity = None
                if config.pairing.enable_same_parent_cross_pair:
                    cross = choose_cross_pair(
                        target=target,
                        candidates=indexed_cross_pair_candidates(
                            cross_pair_index,
                            target,
                        ),
                        config=config.pairing,
                        judge=qwen,
                        structured_failures=structured_failures,
                    )
                    if cross is not None:
                        selected, judgment, visual_similarity = cross
                        pair_type = "same_parent_cross_pair"
                        sample_cross_pairs += 1
                    elif config.pairing.cross_pair_fallback_to_in_pair:
                        sample_fallbacks += 1
                        warnings.append(
                            f"{target['entity_id']}: no verified cross-pair; used in-pair"
                        )
                if pair_type == "in_pair":
                    if not config.pairing.enable_in_pair:
                        warnings.append(
                            f"{target['entity_id']}: no permitted reference pair"
                        )
                        continue
                    sample_in_pairs += 1
                final_references.append(
                    _target_reference(
                        target=target,
                        selected=selected,
                        pair_type=pair_type,
                        judge_result=judgment,
                        visual_similarity=visual_similarity,
                    )
                )
            for structured_failure in structured_failures:
                _append_jsonl(
                    output_root / "logs" / "cross_pair_judge_failed.jsonl",
                    {
                        "clip_uid": clip,
                        **structured_failure,
                    },
                )
            background_reference = None
            if background_references:
                selected_background = min(
                    background_references,
                    key=lambda reference: str(
                        reference.get("reference_id", "bg1")
                    ),
                )
                if len(background_references) > 1:
                    warnings.append(
                        "multiple background references found; retained bg1 only"
                    )
                if not config.pairing.enable_in_pair:
                    warnings.append("bg1: no permitted in-pair reference")
                else:
                    source_phrase = str(selected_background["phrase"])
                    resolved_phrase = resolve_background_caption_phrase(
                        retained_annotation.caption,
                        source_phrase,
                    )
                    if resolved_phrase is None:
                        warning = (
                            "background_reference_dropped:"
                            "background_phrase_unresolvable"
                        )
                        warnings.append(warning)
                        _append_background_binding_failure(
                            output_root,
                            original_background_phrase=source_phrase,
                            issue_codes=["background_phrase_unresolvable"],
                            clip_uid=clip,
                            caption=retained_annotation.caption,
                            entity_reference_count=len(final_references),
                        )
                    else:
                        try:
                            retained_annotation = bind_background_reference(
                                retained_annotation,
                                phrase=resolved_phrase,
                            )
                        except ReferenceBindingError as exc:
                            issue_codes = [issue.code for issue in exc.issues]
                            warnings.append(
                                "background_reference_dropped:"
                                + ",".join(issue_codes)
                            )
                            _append_background_binding_failure(
                                output_root,
                                original_background_phrase=source_phrase,
                                issue_codes=issue_codes,
                                clip_uid=clip,
                                caption=retained_annotation.caption,
                                entity_reference_count=len(final_references),
                            )
                        else:
                            background_reference = _background_target_reference(
                                {
                                    **selected_background,
                                    "source_phrase": source_phrase,
                                    "phrase": resolved_phrase,
                                }
                            )
                            final_references.append(background_reference)
                            sample_in_pairs += 1
            if not final_references:
                no_bindable += 1
                continue
            sample = {
                "clip_uid": clip,
                "parent_video_id": annotation["parent_video_id"],
                "clip_suffix": annotation["clip_suffix"],
                "target_video": annotation["video_path"],
                "caption": annotation["caption"],
                "prompt_with_refs": retained_annotation.prompt_with_refs,
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "phrase": entity.phrase,
                        "grounding_prompt": entity.grounding_prompt,
                        "canonical_label": entity.canonical_label,
                        "category": entity.category,
                        "reference_worthy": entity.reference_worthy,
                        "ref_token": entity.ref_token,
                        "separability": entity.separability,
                    }
                    for entity in retained_annotation.entities
                ],
                "references": final_references,
                "relations": annotation.get("relations", []),
                "background_reference": background_reference,
                "augmentation_variants": [],
                "sampling_policy": {
                    "parent_only": 0.6,
                    "child_only": 0.1,
                    "both": 0.2,
                    "composite": 0.1,
                },
                "warnings": warnings,
            }
            binding_issues = validate_final_reference_binding(
                sample,
                retained_annotation,
            )
            if binding_issues:
                _append_jsonl(
                    output_root / "logs" / "pairing_failed.jsonl",
                    {
                        "clip_uid": clip,
                        "issues": [issue.to_dict() for issue in binding_issues],
                    },
                )
                failed += 1
                continue
            in_pairs += sample_in_pairs
            cross_pairs += sample_cross_pairs
            fallbacks += sample_fallbacks
            write_json_atomic(samples_dir / f"{clip}.json", sample)
            existing.add(clip)
            processed += 1
        except Exception as exc:  # noqa: BLE001 - one sample must not stop the batch
            _append_jsonl(
                output_root / "logs" / "pairing_failed.jsonl",
                {"clip_uid": clip, "error": str(exc)},
            )
            failed += 1
    reconcile_final_samples(output_root)
    return PairingStats(
        processed=processed,
        skipped_existing=skipped,
        skipped_no_ready_reference=no_ready,
        skipped_no_bindable_reference=no_bindable,
        in_pair_count=in_pairs,
        cross_pair_count=cross_pairs,
        fallback_count=fallbacks,
        failed=failed,
    )


def stats_dict(stats: PairingStats) -> dict[str, int]:
    return asdict(stats)
