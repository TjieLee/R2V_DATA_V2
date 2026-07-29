from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

from prompts.qwen_annotation_prompt import ICL_EXAMPLES, SYSTEM_PROMPT
from prompts.qwen_repair_prompt import build_repair_prompt
from r2v_data_v2.caption_validation import (
    annotation_warnings,
    exact_phrase_spans,
    validate_annotation,
)
from r2v_data_v2.config import (
    PipelineConfig,
    QwenConfig,
    _qwen_services,
)
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.phrase_alignment import resolve_reference_caption_phrase
from r2v_data_v2.reconciliation import reconcile_annotations, write_json_atomic
from r2v_data_v2.reference_binding import (
    ReferenceBindingError,
    assign_reference_tokens,
)
from r2v_data_v2.schemas import AnnotationResult, QwenAnnotationResult
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    parse_qwen_json_issues,
)


@dataclass(frozen=True)
class AnnotationStats:
    processed: int = 0
    skipped_existing: int = 0
    qwen_failed: int = 0
    no_reference_entity: int = 0
    generic_entity_labels: int = 0


class QwenAnnotationFailure(StructuredOutputFailure):
    pass


def _video_content(video_path: Path) -> list[dict[str, object]]:
    return [
        {
            "type": "video_url",
            "video_url": {"url": video_path.resolve().as_uri()},
        },
    ]


def _video_processor_extra_body(config: QwenConfig) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {
        "fps": config.video.fps,
        "do_sample_frames": config.video.do_sample_frames,
    }
    if config.video.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.video.max_pixels
    if config.video.total_pixels is not None:
        processor_kwargs["total_pixels"] = config.video.total_pixels
    return {"mm_processor_kwargs": processor_kwargs}


def _align_reference_phrases(
    annotation: QwenAnnotationResult,
) -> tuple[QwenAnnotationResult, list[str]]:
    """Align only missing reference phrases to unique caption spans."""
    entities = []
    warnings: list[str] = []

    for entity in annotation.entities:
        # Preserve valid phrases and multiple-occurrence failures.
        spans = exact_phrase_spans(annotation.caption, entity.phrase)
        if not entity.reference_worthy or spans:
            entities.append(entity)
            continue

        aligned = resolve_reference_caption_phrase(
            annotation.caption,
            entity.phrase,
        )
        if aligned is None:
            entities.append(entity)
            continue

        entities.append(
            entity.model_copy(
                update={"phrase": aligned},
            )
        )
        warnings.append(
            "aligned_reference_phrase:"
            f"{entity.entity_id}:"
            f"{entity.phrase!r}->{aligned!r}"
        )

    return (
        annotation.model_copy(update={"entities": entities}),
        warnings,
    )


def _sanitize_final_structure(
    annotation: QwenAnnotationResult,
) -> tuple[QwenAnnotationResult, list[str]]:
    sanitized, warnings = _align_reference_phrases(annotation)
    entities = []
    for entity in sanitized.entities:
        if (
            entity.reference_worthy
            and len(exact_phrase_spans(sanitized.caption, entity.phrase)) != 1
        ):
            entities.append(
                entity.model_copy(update={"reference_worthy": False})
            )
            warnings.append(
                f"demoted_unalignable_reference:{entity.entity_id}"
            )
        else:
            entities.append(entity)

    known_ids = {entity.entity_id for entity in entities}
    relations = []
    for index, relation in enumerate(sanitized.relations):
        if (
            relation.subject_id not in known_ids
            or relation.object_id not in known_ids
        ):
            warnings.append(f"dropped_invalid_relation:{index}")
            continue
        relations.append(relation)

    salience_rank = {"primary": 0, "secondary": 1, "incidental": 2}
    selected_indexes = {
        index
        for index, entity in sorted(
            enumerate(entities),
            key=lambda item: (
                salience_rank[item[1].salience],
                item[0],
            ),
        )
        if entity.reference_worthy
    }
    selected_indexes = set(
        sorted(
            selected_indexes,
            key=lambda index: (
                salience_rank[entities[index].salience],
                index,
            ),
        )[:3]
    )
    capped_entities = []
    for index, entity in enumerate(entities):
        if entity.reference_worthy and index not in selected_indexes:
            capped_entities.append(
                entity.model_copy(update={"reference_worthy": False})
            )
            warnings.append(f"demoted_reference_cap:{entity.entity_id}")
        else:
            capped_entities.append(entity)

    background = sanitized.background
    if (
        background is not None
        and background.reference_worthy
        and len(exact_phrase_spans(sanitized.caption, background.phrase)) != 1
    ):
        background = background.model_copy(update={"reference_worthy": False})
        warnings.append("demoted_unalignable_reference:background")

    return (
        sanitized.model_copy(
            update={
                "entities": capped_entities,
                "relations": relations,
                "background": background,
            }
        ),
        warnings,
    )


class QwenAnnotationClient:
    def __init__(
        self,
        config: QwenConfig,
        *,
        repair_config: QwenConfig | None = None,
    ) -> None:
        self.config = config
        self.repair_config = repair_config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self.repair_client = (
            OpenAI(
                base_url=repair_config.base_url,
                api_key=repair_config.api_key,
                timeout=repair_config.timeout_seconds,
            )
            if repair_config is not None
            else None
        )
        self._active_client = self.client
        self._active_config: QwenConfig = config

    def _messages(
        self,
        *,
        video_path: Path,
        caption_raw: str,
        metadata: dict[str, object],
        repair_prompt: str | None = None,
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for example in ICL_EXAMPLES:
            messages.extend(
                (
                    {
                        "role": "user",
                        "content": json.dumps(example["input"], ensure_ascii=False),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(example["output"], ensure_ascii=False),
                    },
                )
            )
        current = _video_content(video_path)
        current.append(
            {
                "type": "text",
                "text": (
                    repair_prompt
                    if repair_prompt is not None
                    else (
                        "Draft caption and explicit metadata follow. Treat the draft "
                        "as evidence, not as trusted prose:\n"
                        + json.dumps(
                            {"draft_caption": caption_raw, "metadata": metadata},
                            ensure_ascii=False,
                        )
                    )
                ),
            }
        )
        messages.append({"role": "user", "content": current})
        return messages

    def _request(self, messages: list[dict[str, object]]) -> str:
        active_config = getattr(self, "_active_config", self.config)
        active_client = getattr(self, "_active_client", self.client)
        parameters: dict[str, Any] = {
            "model": active_config.model,
            "messages": messages,
            "temperature": active_config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": active_config.max_tokens,
            "extra_body": _video_processor_extra_body(active_config),
        }
        try:
            response = active_client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "annotation_result",
                        "strict": True,
                        "schema": QwenAnnotationResult.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = active_client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty annotation response")
        return content

    def annotate(
        self,
        *,
        video_path: Path,
        caption_raw: str,
        metadata: dict[str, object],
    ) -> tuple[AnnotationResult, list[str]]:
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.config.repair_retries + 1):
            repair_prompt = None
            if attempt:
                repair_prompt = build_repair_prompt(
                    invalid_response=raw_responses[-1],
                    validation_issues=issues,
                    json_schema=QwenAnnotationResult.model_json_schema(),
                    draft_caption=caption_raw,
                    metadata=metadata,
                )
            try:
                repair_config = getattr(self, "repair_config", None)
                repair_client = getattr(self, "repair_client", None)
                if attempt and repair_config is not None and repair_client is not None:
                    self._active_config = repair_config
                    self._active_client = repair_client
                else:
                    self._active_config = self.config
                    self._active_client = getattr(self, "client", None)
                raw_response = self._request(
                    self._messages(
                        video_path=video_path,
                        caption_raw=caption_raw,
                        metadata=metadata,
                        repair_prompt=repair_prompt,
                    )
                )
            except Exception as exc:
                raise QwenAnnotationFailure(
                    raw_responses=raw_responses,
                    issues=[
                        ValidationIssue(
                            code="qwen_request_failed",
                            field=None,
                            message=str(exc),
                        )
                    ],
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw_response)
            semantic, issues = parse_qwen_json_issues(
                raw_response,
                QwenAnnotationResult,
            )
            alignment_warnings: list[str] = []
            if semantic is not None:
                semantic, alignment_warnings = _sanitize_final_structure(
                    semantic
                )
                issues = validate_annotation(
                    semantic,
                    caption_raw=caption_raw,
                    metadata=metadata,
                )
            result = None
            if semantic is not None and not issues:
                try:
                    result = assign_reference_tokens(semantic)
                except ReferenceBindingError as exc:
                    issues = exc.issues
            if result is not None and not issues:
                warnings = [
                    *alignment_warnings,
                    *annotation_warnings(semantic),
                ]
                if semantic.background and semantic.background.reference_worthy:
                    warnings.append("background_reference_deferred")
                return result, warnings
            if self.config.repair_retries < 1:
                break
        raise QwenAnnotationFailure(raw_responses=raw_responses, issues=issues)


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def annotate_manifest(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    client: QwenAnnotationClient | None = None,
) -> AnnotationStats:
    output_root = config.ensure_output_root()
    source_manifest = output_root / "manifests" / "source.jsonl"
    annotation_manifest = output_root / "manifests" / "annotations.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError("run Stage 00 before Qwen annotation")
    if overwrite:
        annotation_manifest.unlink(missing_ok=True)
        for artifact in (output_root / "annotations").glob("*.json"):
            artifact.unlink()
    services = _qwen_services(config.qwen)
    qwen = client or QwenAnnotationClient(
        services.annotation,
        repair_config=services.repair_judge,
    )
    processed = skipped = failed = no_ref = generic_count = 0
    for source in iter_source_records(source_manifest):
        clip = str(source["clip_uid"])
        destination = output_root / "annotations" / f"{clip}.json"
        if destination.is_file() and not overwrite:
            skipped += 1
            continue
        video_path = Path(str(source["video_path"]))
        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"source video does not exist: {video_path}")
            result, warnings = qwen.annotate(
                video_path=video_path,
                caption_raw=str(source.get("caption_raw", "")),
                metadata=(
                    source["metadata"]
                    if isinstance(source.get("metadata"), dict)
                    else {}
                ),
            )
        except QwenAnnotationFailure as exc:
            _append_jsonl(
                output_root / "logs" / "qwen_failed.jsonl",
                {
                    "clip_uid": clip,
                    "attempt_count": exc.attempt_count,
                    "raw_responses": exc.raw_responses,
                    "issues": [issue.to_dict() for issue in exc.issues],
                },
            )
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the batch
            _append_jsonl(
                output_root / "logs" / "qwen_failed.jsonl",
                {
                    "clip_uid": clip,
                    "attempt_count": 0,
                    "raw_responses": [],
                    "issues": [
                        ValidationIssue(
                            code="qwen_request_failed",
                            field=None,
                            message=str(exc),
                        ).to_dict()
                    ],
                },
            )
            failed += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **result.model_dump(mode="json"),
            "clip_uid": clip,
            "video_path": source["video_path"],
            "parent_video_id": source["parent_video_id"],
            "clip_suffix": source["clip_suffix"],
            "clip_order": source["clip_order"],
            "annotation_path": str(destination),
            "warnings": warnings,
        }
        write_json_atomic(destination, payload)
        reference_entities = sum(entity.reference_worthy for entity in result.entities)
        no_ref += reference_entities == 0 and not (
            result.background and result.background.reference_worthy
        )
        generic_count += sum(
            entity.canonical_label.casefold() in {"man", "woman", "person"}
            for entity in result.entities
        )
        processed += 1
    reconcile_annotations(output_root)
    return AnnotationStats(processed, skipped, failed, no_ref, generic_count)


def stats_dict(stats: AnnotationStats) -> dict[str, int]:
    return asdict(stats)
