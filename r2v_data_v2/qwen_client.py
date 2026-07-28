from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from prompts.qwen_annotation_prompt import ICL_EXAMPLES, SYSTEM_PROMPT
from prompts.qwen_repair_prompt import build_repair_prompt
from r2v_data_v2.caption_validation import annotation_warnings, validate_annotation
from r2v_data_v2.config import PipelineConfig, QwenConfig
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.schemas import AnnotationResult


@dataclass(frozen=True)
class AnnotationStats:
    processed: int = 0
    skipped_existing: int = 0
    qwen_failed: int = 0
    no_reference_entity: int = 0
    generic_entity_labels: int = 0


def _json_from_response(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


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
                            {"draft_caption": caption_raw, "metadata": {}},
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
                        "schema": AnnotationResult.model_json_schema(),
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
        return _json_from_response(content)

    def annotate(
        self,
        *,
        frame_paths: list[Path],
        caption_raw: str,
    ) -> tuple[AnnotationResult, list[str]]:
        raw_response = self._request(
            self._messages(frame_paths=frame_paths, caption_raw=caption_raw)
        )
        attempts = 0
        while True:
            try:
                result = AnnotationResult.model_validate_json(raw_response)
                errors = validate_annotation(result)
            except (ValidationError, json.JSONDecodeError) as exc:
                result = None
                errors = [str(exc)]
            if result is not None and not errors:
                return result, annotation_warnings(result)
            if attempts >= min(1, self.config.repair_retries):
                raise ValueError(f"Qwen annotation validation failed: {errors}")
            raw_response = self._request(
                self._messages(
                    frame_paths=frame_paths,
                    caption_raw=caption_raw,
                    repair_prompt=build_repair_prompt(
                        invalid_response=raw_response,
                        validation_errors=errors,
                    ),
                )
            )
            attempts += 1


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
            )
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the batch
            _append_jsonl(
                output_root / "logs" / "qwen_failed.jsonl",
                {"clip_uid": clip, "error": str(exc)},
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
            "warnings": warnings,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _append_jsonl(
            annotation_manifest,
            {
                "clip_uid": clip,
                "video_path": source["video_path"],
                "parent_video_id": source["parent_video_id"],
                "clip_suffix": source["clip_suffix"],
                "clip_order": source["clip_order"],
                "annotation_path": str(destination),
                **result.model_dump(mode="json"),
                "warnings": warnings,
            },
        )
        reference_entities = sum(entity.reference_worthy for entity in result.entities)
        no_ref += reference_entities == 0 and not (
            result.background and result.background.reference_worthy
        )
        generic_count += sum(
            warning.startswith("generic_entity_label_count=") for warning in warnings
        )
        processed += 1
    return AnnotationStats(processed, skipped, failed, no_ref, generic_count)


def stats_dict(stats: AnnotationStats) -> dict[str, int]:
    return asdict(stats)
