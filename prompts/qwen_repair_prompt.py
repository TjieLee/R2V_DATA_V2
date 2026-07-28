from __future__ import annotations

import json

from r2v_data_v2.structured_output import ValidationIssue


def build_repair_prompt(
    *,
    invalid_response: str,
    validation_issues: list[ValidationIssue],
    json_schema: dict[str, object],
    draft_caption: str,
    metadata: dict[str, object],
) -> str:
    return (
        "Repair only the listed errors in the annotation. Reinspect the same ten "
        "frames, preserve supported visible facts, and return one JSON object only."
        "\nOriginal draft caption:\n"
        f"{draft_caption}\nAllowed metadata evidence:\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\nJSON Schema:\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}\nValidation issues:\n"
        f"{json.dumps([issue.to_dict() for issue in validation_issues], indent=2)}"
        f"\nOriginal invalid response:\n{invalid_response}"
    )
