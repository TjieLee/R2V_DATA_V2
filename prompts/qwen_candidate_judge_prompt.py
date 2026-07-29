from __future__ import annotations

CANDIDATE_JUDGE_PROMPT = """You are selecting one canonical visual reference for
the entity {entity_phrase!r}. Its category is {category!r} and its canonical
label is {canonical_label!r}. Inspect the numbered candidate contact sheet.

For every candidate, judge completeness, recognizability, occlusion,
segmentation-mask quality, overall visual quality, and whether identity-defining
features are visible. Also classify viewpoint and assign canonical_view_score.
For people, animals, and characters, prefer front or front-three-quarter views
that reveal identity. For vehicles, products, and objects, prefer the view that
best communicates the canonical shape and distinguishing features; a side or
rear view may be canonical when appropriate. Never reject a candidate merely
because it is not a front view.

Reject truncation, severe occlusion, another salient instance inside the mask,
identity mismatch, or a visibly broken mask.

Return JSON only. Include exactly the supplied frame slots and choose
best_frame_slot from candidates without a hard visual failure."""
