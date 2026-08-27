from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str | None
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


class StructuredOutputFailure(ValueError):
    def __init__(
        self,
        *,
        raw_responses: list[str],
        issues: list[ValidationIssue],
        attempt_count: int | None = None,
    ) -> None:
        super().__init__("Qwen structured output failed closed")
        self.raw_responses = raw_responses
        self.issues = issues
        self.attempt_count = (
            len(raw_responses) if attempt_count is None else attempt_count
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_count": self.attempt_count,
            "raw_responses": self.raw_responses,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def normalize_structured_json_envelope(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("Assistant:"):
        stripped = stripped[len("Assistant:") :].lstrip()
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


def parse_structured_json_response(raw: str, model: type[ModelT]) -> ModelT:
    payload = json.loads(normalize_structured_json_envelope(raw))
    if not isinstance(payload, dict):
        raise TypeError("structured response must be one JSON object")
    return model.model_validate(payload)


def parse_qwen_json_response(raw: str, model: type[ModelT]) -> ModelT:
    payload = json.loads(normalize_structured_json_envelope(raw))
    if not isinstance(payload, dict):
        raise TypeError("Qwen response must be one JSON object")
    return model.model_validate(payload)


def _parse_json_issues(
    raw: str,
    model: type[ModelT],
    parser: Callable[[str, type[ModelT]], ModelT],
) -> tuple[ModelT | None, list[ValidationIssue]]:
    try:
        return parser(raw, model), []
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
            code = {
                "extra_forbidden": "schema_extra_field",
                "missing": "schema_missing_field",
            }.get(str(error["type"]), "schema_validation")
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


def parse_structured_json_issues(
    raw: str,
    model: type[ModelT],
) -> tuple[ModelT | None, list[ValidationIssue]]:
    return _parse_json_issues(raw, model, parse_structured_json_response)


def parse_qwen_json_issues(
    raw: str,
    model: type[ModelT],
) -> tuple[ModelT | None, list[ValidationIssue]]:
    return _parse_json_issues(raw, model, parse_qwen_json_response)


def build_structured_repair_prompt(
    *,
    original_request: str,
    invalid_response: str,
    validation_issues: list[ValidationIssue],
    json_schema: dict[str, object],
) -> str:
    return (
        "Repair only the listed structured-output errors. Reinspect the attached "
        "original image or images and return one JSON object only.\n"
        f"Original request:\n{original_request}\n"
        f"JSON Schema:\n{json.dumps(json_schema, ensure_ascii=False)}\n"
        "Validation issues:\n"
        f"{json.dumps([issue.to_dict() for issue in validation_issues], indent=2)}\n"
        f"Original invalid response:\n{invalid_response}"
    )


def request_structured_output(
    *,
    request: Callable[[str], str],
    original_request: str,
    model: type[ModelT],
    validate: Callable[[ModelT], list[ValidationIssue]] | None = None,
) -> ModelT:
    raw_responses: list[str] = []
    issues: list[ValidationIssue] = []
    for attempt in range(2):
        request_text = original_request
        if attempt:
            request_text = build_structured_repair_prompt(
                original_request=original_request,
                invalid_response=raw_responses[-1],
                validation_issues=issues,
                json_schema=model.model_json_schema(),
            )
        try:
            raw = request(request_text)
        except Exception as exc:
            raise StructuredOutputFailure(
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
        result, issues = parse_structured_json_issues(raw, model)
        if result is not None and validate is not None:
            issues = validate(result)
        if result is not None and not issues:
            return result
    raise StructuredOutputFailure(raw_responses=raw_responses, issues=issues)
