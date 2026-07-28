from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

from prompts.qwen_annotation_prompt import ICL_EXAMPLES, SYSTEM_PROMPT
from prompts.qwen_repair_prompt import build_repair_prompt
from r2v_data_v2.caption_validation import (
    ValidationIssue,
    annotation_warnings,
    validate_annotation,
)
from r2v_data_v2.config import PipelineConfig, QwenConfig
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.reconciliation import reconcile_annotations, write_json_atomic
from r2v_data_v2.reference_binding import (
    ReferenceBindingError,
    assign_reference_tokens,
)
from r2v_data_v2.schemas import AnnotationResult, QwenAnnotationResult

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class AnnotationStats:
    processed: int = 0
    skipped_existing: int = 0
    qwen_failed: int = 0
    no_reference_entity: int = 0
    generic_entity_labels: int = 0


class QwenAnnotationFailure(ValueError):
    def __init__(
        self,
        *,
        raw_responses: list[str],
        issues: list[ValidationIssue],
        attempt_count: int | None = None,
    ) -> None:
        super().__init__("Qwen annotation failed parsing or semantic validation")
        self.raw_responses = raw_responses
        self.issues = issues
        self.attempt_count = (
            len(raw_responses) if attempt_count is None else attempt_count
        )


def _strip_complete_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",
        stripped,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("response must be one complete JSON object")
    return match.group(1).strip()


def parse_qwen_json_response(raw: str, model: type[ModelT]) -> ModelT:
    payload = json.loads(_strip_complete_json_fence(raw))
    if not isinstance(payload, dict):
        raise TypeError("Qwen response must be one JSON object")
    return model.model_validate(payload)


def _parse_issues(
    raw: str, model: type[ModelT]
) -> tuple[ModelT | None, list[ValidationIssue]]:
    try:
        return parse_qwen_json_response(raw, model), []
    except json.JSONDecodeError as exc:
        return None, [
            ValidationIssue(
                code="invalid_json",
                field=None,
                message=f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            )
        ]
    except ValidationError as exc:
        issues = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or None
            error_type = str(error["type"])
            code = {
                "extra_forbidden": "schema_extra_field",
                "missing": "schema_missing_field",
            }.get(error_type, "schema_validation")
            issues.append(
                ValidationIssue(
                    code=code,
                    field=location,
                    message=str(error["msg"]),
                )
            )
        return None, issues
    except (TypeError, ValueError) as exc:
        return None, [
            ValidationIssue(code="invalid_json_object", field=None, message=str(exc))
        ]


def _image_content(frame_paths: list[Path]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": "Inspect these eight frames in chronological order.",
        }
    ]
    for path in frame_paths:
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return content


class QwenAnnotationClient:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        frame_paths: list[Path],
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
        current = _image_content(frame_paths)
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
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(
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
            response = self.client.chat.completions.create(
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
        frame_paths: list[Path],
        caption_raw: str,
        metadata: dict[str, object],
    ) -> tuple[AnnotationResult, list[str]]:
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(2):
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
                raw_response = self._request(
                    self._messages(
                        frame_paths=frame_paths,
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
            semantic, issues = _parse_issues(raw_response, QwenAnnotationResult)
            if semantic is not None:
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
                warnings = annotation_warnings(semantic)
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
    qwen = client or QwenAnnotationClient(config.qwen)
    processed = skipped = failed = no_ref = generic_count = 0
    for source in iter_source_records(source_manifest):
        clip = str(source["clip_uid"])
        destination = output_root / "annotations" / f"{clip}.json"
        if destination.is_file() and not overwrite:
            skipped += 1
            continue
        frame_paths = [
            output_root / "frames" / clip / f"frame_{slot:02d}.jpg"
            for slot in range(config.frames.count)
        ]
        try:
            if not all(path.is_file() for path in frame_paths):
                raise FileNotFoundError("eight sampled frames are required")
            result, warnings = qwen.annotate(
                frame_paths=frame_paths,
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
