from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from openai import BadRequestError, OpenAI

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenServiceConfig, V3Config
from r2v_data_v2.v3.schemas import (
    ClipRecord,
    InstructionBinding,
    InstructionLegendEntry,
    InstructionState,
    RawInstructionOutput,
    render_instruction_text,
)
from r2v_data_v2.v3.storage import RunStorage

_REFERENCE_TOKEN = re.compile(r"<ref_[^>]+>", flags=re.IGNORECASE)
_EXACT_PLACEHOLDER = re.compile(r"\{\{(image_[1-9]\d*)\}\}")
_ANY_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")
_DIRECT_CHINESE_IMAGE_LABEL = re.compile(r"\u56fe\s*\d+")
_DIRECT_ENGLISH_IMAGE_LABEL = re.compile(
    r"\bImage\s+[1-9]\d*\b",
    flags=re.IGNORECASE,
)
_CJK_TEXT = re.compile(
    r"[\u1100-\u11ff\u2e80-\u2fdf\u3040-\u30ff"
    r"\u3100-\u312f\u3130-\u318f\u31a0-\u31bf"
    r"\u31c0-\u31ff\u3200-\u33ff\u3400-\u4dbf"
    r"\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7ff"
    r"\uf900-\ufaff\U00020000-\U0002fa1f]"
)
_QUOTED_DIALOGUE = re.compile(
    r"(?:\u300c[^\u300d\n]+\u300d|"
    r"\u300e[^\u300f\n]+\u300f|"
    r"\u201c[^\u201d\n]+\u201d|"
    r'"[^"\n]+")'
)

SYSTEM_PROMPT = """You write an English reference-conditioned video instruction.

Return exactly one JSON object matching the supplied schema. The input bindings
are final and immutable. Do not add, remove, reorder, rename, or reinterpret any
binding. All schema field names, identifiers, and placeholders remain English.

Write instruction_body_template in English. Accurately describe the scene,
lighting, camera, composition, initial positions, spatial relationships, visible
actions, and shot changes in chronological order. Use each exact placeholder,
such as {{image_1}}, at least once. Placeholders may be repeated. In raw output,
never replace placeholders with rendered labels such as <Image 1>, Image 1, or
Chinese image-number labels. Do not output <ref_...> tokens. Do not invent shot
changes for a single continuous shot.

Without source_transcript, do not invent quoted dialogue. Visible speaking
motion may be described without supplying words. When source_transcript is
present, only dialogue supported by that transcript may be quoted.

reference_legend must contain exactly one entry for each binding in the same
order. Copy each binding image_id exactly. Each description must be in English
and summarize the stable visual appearance that the corresponding reference
image must preserve. A background description covers the environment and must
not invent subject actions. Return JSON only."""


@dataclass(frozen=True)
class InstructionAttempt:
    instruction: InstructionState
    raw_responses: tuple[str, ...]
    repair_attempts: int


@dataclass(frozen=True)
class InstructionStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    repaired: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class InstructionFailure(StructuredOutputFailure):
    pass


class InstructionClient(Protocol):
    def write(
        self,
        *,
        t2v_caption: str,
        bindings: list[InstructionBinding],
        source_transcript: str | None,
    ) -> InstructionAttempt: ...


def build_instruction_bindings(clip: ClipRecord) -> list[InstructionBinding]:
    if clip.annotation is None or clip.annotation.status != "ready":
        raise ValueError("instruction requires a ready annotation")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise ValueError("instruction requires ready pairing")

    entities = {
        entity.entity_id: entity for entity in clip.annotation.entities
    }
    bindings: list[InstructionBinding] = []
    for entity_id in clip.pairing.retained_entity_ids:
        entity = entities.get(entity_id)
        if entity is None:
            raise ValueError(
                f"paired entity is missing from annotation: {entity_id}"
            )
        image_index = len(bindings) + 1
        bindings.append(
            InstructionBinding(
                image_id=f"image_{image_index}",
                image_index=image_index,
                reference_type=entity.reference_type,
                entity_id=entity.entity_id,
                phrase=entity.phrase,
                grounding_prompt=entity.grounding_prompt,
            )
        )

    if clip.pairing.background_token is not None:
        background = clip.annotation.background
        if background is None:
            raise ValueError(
                "paired background is missing annotation semantics"
            )
        image_index = len(bindings) + 1
        bindings.append(
            InstructionBinding(
                image_id=f"image_{image_index}",
                image_index=image_index,
                reference_type="background",
                entity_id=None,
                phrase=background.phrase,
                grounding_prompt=background.grounding_prompt,
            )
        )
    return bindings


def source_transcript_from_metadata(
    metadata: dict[str, object],
) -> str | None:
    for key in ("source_transcript", "transcript", "dialogue"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def validate_instruction_output(
    output: RawInstructionOutput,
    *,
    t2v_caption: str,
    bindings: list[InstructionBinding],
    source_transcript: str | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    body = output.instruction_body_template.strip()
    expected_ids = [binding.image_id for binding in bindings]
    raw_text = json.dumps(output.model_dump(mode="json"), ensure_ascii=False)

    if not body:
        issues.append(
            ValidationIssue(
                code="empty_instruction_body_template",
                field="instruction_body_template",
                message="instruction_body_template must not be empty",
            )
        )
    elif _CJK_TEXT.search(body):
        issues.append(
            ValidationIssue(
                code="non_english_instruction_text",
                field="instruction_body_template",
                message=(
                    "instruction_body_template must not contain CJK characters"
                ),
            )
        )
    if _REFERENCE_TOKEN.search(raw_text):
        issues.append(
            ValidationIssue(
                code="reference_token_in_instruction",
                field=None,
                message="instruction output must not contain reference tokens",
            )
        )
    if _DIRECT_CHINESE_IMAGE_LABEL.search(raw_text):
        issues.append(
            ValidationIssue(
                code="direct_chinese_image_label",
                field=None,
                message=(
                    "raw instruction output must use English image placeholders"
                ),
            )
        )
    if _DIRECT_ENGLISH_IMAGE_LABEL.search(raw_text):
        issues.append(
            ValidationIssue(
                code="direct_english_image_label",
                field=None,
                message="raw instruction output must use {{image_N}} placeholders",
            )
        )

    exact_placeholders = _EXACT_PLACEHOLDER.findall(body)
    all_placeholders = _ANY_PLACEHOLDER.findall(body)
    remainder = _EXACT_PLACEHOLDER.sub("", body)
    if (
        len(exact_placeholders) != len(all_placeholders)
        or "{{" in remainder
        or "}}" in remainder
    ):
        issues.append(
            ValidationIssue(
                code="invalid_image_placeholder",
                field="instruction_body_template",
                message="all image placeholders must use {{image_N}}",
            )
        )
    unknown_ids = sorted(set(exact_placeholders) - set(expected_ids))
    if unknown_ids:
        issues.append(
            ValidationIssue(
                code="unknown_image_placeholder",
                field="instruction_body_template",
                message=f"unknown image placeholders: {unknown_ids}",
            )
        )
    missing_ids = [
        image_id
        for image_id in expected_ids
        if image_id not in exact_placeholders
    ]
    if missing_ids:
        issues.append(
            ValidationIssue(
                code="missing_image_placeholder",
                field="instruction_body_template",
                message=f"missing image placeholders: {missing_ids}",
            )
        )

    legend_ids = [entry.image_id for entry in output.reference_legend]
    if len(output.reference_legend) != len(bindings):
        issues.append(
            ValidationIssue(
                code="legend_count_mismatch",
                field="reference_legend",
                message="reference legend count must equal binding count",
            )
        )
    if legend_ids != expected_ids:
        issues.append(
            ValidationIssue(
                code="legend_order_mismatch",
                field="reference_legend",
                message="reference legend IDs must match binding order",
            )
        )
    for index, entry in enumerate(output.reference_legend):
        if not entry.description.strip():
            issues.append(
                ValidationIssue(
                    code="empty_legend_description",
                    field=f"reference_legend.{index}.description",
                    message="legend description must not be empty",
                )
            )
        elif _CJK_TEXT.search(entry.description):
            issues.append(
                ValidationIssue(
                    code="non_english_legend_description",
                    field=f"reference_legend.{index}.description",
                    message="legend description must not contain CJK characters",
                )
            )
        if _ANY_PLACEHOLDER.search(entry.description):
            issues.append(
                ValidationIssue(
                    code="placeholder_in_legend_description",
                    field=f"reference_legend.{index}.description",
                    message="legend descriptions must not contain placeholders",
                )
            )

    if source_transcript is None and _QUOTED_DIALOGUE.search(body):
        issues.append(
            ValidationIssue(
                code="quoted_dialogue_without_transcript",
                field="instruction_body_template",
                message="quoted dialogue requires source_transcript",
            )
        )
    if " ".join(body.split()) == " ".join(t2v_caption.split()):
        issues.append(
            ValidationIssue(
                code="instruction_copies_t2v_caption",
                field="instruction_body_template",
                message="instruction body must not copy t2v_caption verbatim",
            )
        )
    return issues


def _to_instruction_state(
    output: RawInstructionOutput,
) -> InstructionState:
    legend = [
        InstructionLegendEntry(
            image_id=entry.image_id,
            description=entry.description.strip(),
        )
        for entry in output.reference_legend
    ]
    body = output.instruction_body_template.strip()
    return InstructionState(
        status="ready",
        instruction_body_template=body,
        reference_legend=legend,
        r2v_instruction=render_instruction_text(body, legend),
    )


def _request_payload(
    *,
    t2v_caption: str,
    bindings: list[InstructionBinding],
    source_transcript: str | None,
) -> dict[str, object]:
    return {
        "t2v_caption": t2v_caption,
        "bindings": [
            binding.model_dump(mode="json") for binding in bindings
        ],
        "source_transcript": source_transcript,
    }


def _initial_request(
    *,
    t2v_caption: str,
    bindings: list[InstructionBinding],
    source_transcript: str | None,
) -> str:
    payload = _request_payload(
        t2v_caption=t2v_caption,
        bindings=bindings,
        source_transcript=source_transcript,
    )
    return (
        "Write the structured English instruction for this immutable input:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _repair_request(
    *,
    original_request: str,
    invalid_response: str,
    issues: list[ValidationIssue],
) -> str:
    return (
        "Repair only the structured-output problems listed below. Return the "
        "complete corrected JSON object with an English body and English legend "
        "descriptions. Preserve all image IDs and exact {{image_N}} placeholders; "
        "do not output rendered image-number labels in raw JSON.\n"
        f"Original request:\n{original_request}\n"
        "JSON Schema:\n"
        f"{json.dumps(RawInstructionOutput.model_json_schema(), ensure_ascii=False)}\n"
        "Validation issues:\n"
        f"{json.dumps([issue.to_dict() for issue in issues], ensure_ascii=False)}\n"
        f"Invalid response:\n{invalid_response}"
    )


class QwenInstructionClient:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        repair_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.repair_retries = repair_retries
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(self, request_text: str) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request_text},
            ],
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
                        "name": "v3_instruction",
                        "strict": True,
                        "schema": RawInstructionOutput.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty V3 instruction response")
        return str(content)

    def write(
        self,
        *,
        t2v_caption: str,
        bindings: list[InstructionBinding],
        source_transcript: str | None,
    ) -> InstructionAttempt:
        original_request = _initial_request(
            t2v_caption=t2v_caption,
            bindings=bindings,
            source_transcript=source_transcript,
        )
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.repair_retries + 1):
            request_text = (
                original_request
                if attempt == 0
                else _repair_request(
                    original_request=original_request,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            )
            try:
                raw = self._request(request_text)
            except Exception as exc:
                raise InstructionFailure(
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
            raw_responses.append(raw)
            output, issues = parse_qwen_json_issues(
                raw,
                RawInstructionOutput,
            )
            if output is not None:
                issues = validate_instruction_output(
                    output,
                    t2v_caption=t2v_caption,
                    bindings=bindings,
                    source_transcript=source_transcript,
                )
            if output is not None and not issues:
                return InstructionAttempt(
                    instruction=_to_instruction_state(output),
                    raw_responses=tuple(raw_responses),
                    repair_attempts=attempt,
                )
        raise InstructionFailure(
            raw_responses=raw_responses,
            issues=issues,
        )


def _write_debug_response(
    storage: RunStorage,
    *,
    clip_uid: str,
    raw_responses: tuple[str, ...] | list[str],
    issues: list[ValidationIssue] | None = None,
) -> None:
    if not storage.config.debug.save_diagnostics:
        return
    write_json_atomic(
        storage.debug_path(clip_uid, "instruction_raw.json"),
        {
            "raw_responses": list(raw_responses),
            "issues": [issue.to_dict() for issue in (issues or [])],
        },
    )


def instruct_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    client: InstructionClient | None = None,
) -> InstructionStats:
    if not config.instruction.enabled:
        raise ValueError("V3 instruction stage is disabled")
    clips = list(storage.iter_clips())
    if not clips:
        raise FileNotFoundError(
            "instruct stage requires manifest stage to create clip.json records first"
        )
    qwen = client or QwenInstructionClient(
        config.qwen.instruction_writer,
        repair_retries=config.instruction.repair_retries,
    )
    processed = skipped_existing = skipped_not_ready = failed = repaired = 0

    for clip in clips:
        if (
            clip.instruction is not None
            and clip.instruction.status == "ready"
            and not overwrite
        ):
            skipped_existing += 1
            continue
        if (
            clip.annotation is None
            or clip.annotation.status != "ready"
            or clip.pairing is None
            or clip.pairing.status != "ready"
        ):
            skipped_not_ready += 1
            continue
        try:
            bindings = build_instruction_bindings(clip)
            attempt = qwen.write(
                t2v_caption=clip.annotation.t2v_caption,
                bindings=bindings,
                source_transcript=source_transcript_from_metadata(
                    clip.source.metadata
                ),
            )
            storage.write_instruction(clip.clip_uid, attempt.instruction)
            _write_debug_response(
                storage,
                clip_uid=clip.clip_uid,
                raw_responses=attempt.raw_responses,
            )
            processed += 1
            repaired += int(attempt.repair_attempts > 0)
        except InstructionFailure as exc:
            reason = (
                exc.issues[0].code
                if exc.issues
                else "structured_output_failed"
            )
            storage.write_instruction(
                clip.clip_uid,
                InstructionState(status="failed", reason=reason),
            )
            _write_debug_response(
                storage,
                clip_uid=clip.clip_uid,
                raw_responses=exc.raw_responses,
                issues=exc.issues,
            )
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="instruct",
                reason=reason,
                details={
                    "attempt_count": exc.attempt_count,
                    "issues": [issue.to_dict() for issue in exc.issues],
                },
            )
            failed += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
            reason = str(exc)
            storage.write_instruction(
                clip.clip_uid,
                InstructionState(status="failed", reason=reason),
            )
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="instruct",
                reason=reason,
            )
            failed += 1

    stats = InstructionStats(
        processed=processed,
        skipped_existing=skipped_existing,
        skipped_not_ready=skipped_not_ready,
        failed=failed,
        repaired=repaired,
    )
    storage.update_stage_counts("instruct", stats.to_dict())
    return stats
