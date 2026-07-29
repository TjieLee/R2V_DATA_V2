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
        "Repair the listed validation errors in the annotation. Reinspect the same "
        "complete video, preserve supported visible facts, and return one JSON object only."
        "\n\nCritical repair rules:"
        "\n1. For every entity with reference_worthy=true, entity.phrase must be copied "
        "verbatim from one contiguous span of caption, including capitalization."
        "\n2. Every reference-worthy phrase must occur exactly once in caption."
        "\n3. For phrase_missing_from_caption, either rewrite caption to contain the "
        "existing phrase exactly once, or replace entity.phrase with an exact phrase "
        "already present exactly once in caption. Preserve the same entity identity."
        "\n4. Do not invent a more specific entity phrase than the video supports."
        "\n5. If separability is attached_accessory, reference_worthy must be false."
        "\n6. After repairing the reported fields, recheck every reference-worthy entity "
        "against the final caption."
        "\n7. Return the complete corrected JSON object, not a patch or explanation."
        "\n\nOriginal draft caption:\n"
        f"{draft_caption}"
        "\n\nAllowed metadata evidence:\n"
        f"{json.dumps(metadata, ensure_ascii=False)}"
        "\n\nJSON Schema:\n"
        f"{json.dumps(json_schema, ensure_ascii=False)}"
        "\n\nValidation issues:\n"
        f"{json.dumps([issue.to_dict() for issue in validation_issues], indent=2)}"
        f"\n\nOriginal invalid response:\n{invalid_response}"
    )
