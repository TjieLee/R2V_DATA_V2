from __future__ import annotations

CANDIDATE_JUDGE_PROMPT = """You are selecting one canonical visual reference for
the entity {entity_phrase!r}. Inspect the numbered candidate contact sheet.

For every candidate, judge completeness, recognizability, occlusion,
segmentation-mask quality, overall visual quality, and whether identity-defining
features are visible. Reject truncation, severe occlusion, another salient
instance inside the mask, identity mismatch, or a visibly broken mask.

Return JSON only. Include exactly the supplied frame slots and choose
best_frame_slot from candidates without a hard visual failure."""
