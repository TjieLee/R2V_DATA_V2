from __future__ import annotations

BACKGROUND_JUDGE_PROMPT = """You are selecting a reusable canonical background
reference for the scene {background_phrase!r}. Inspect the numbered full-frame
candidate contact sheet.

For every candidate, judge scene completeness, scene recognizability, foreground
distraction, visual quality, and whether the frame can be reused as a background.
Reject an unrecognizable scene, severe blur or exposure failure, a prominent
watermark, or foreground content that dominates the scene. A small masked
foreground region may be repaired later and is not by itself a rejection.

Return JSON only. Include exactly the supplied frame slots and choose
best_frame_slot from candidates without a hard visual failure."""
