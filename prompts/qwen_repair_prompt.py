from __future__ import annotations

import json


def build_repair_prompt(
    *,
    invalid_response: str,
    validation_errors: list[str],
) -> str:
    return (
        "Repair the annotation JSON. Preserve visible facts and return JSON only.\n"
        f"Validation errors:\n{json.dumps(validation_errors, indent=2)}\n"
        f"Invalid response:\n{invalid_response}"
    )
