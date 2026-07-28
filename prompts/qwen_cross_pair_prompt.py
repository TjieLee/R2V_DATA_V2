from __future__ import annotations

CROSS_PAIR_PROMPT = """Compare the two natural reference crops. The first crop is
the target entity {target_phrase!r}; the second is a same-parent sibling
candidate with label {candidate_label!r}.

Decide whether they show the same exact instance, not merely the same category.
Check face or shape, hair, clothing, markings, product geometry, color, texture,
logos, accessories, and any conflicting attributes. Also judge whether the
contexts differ enough that the pair is not a near duplicate.

Return JSON only. Use "uncertain" whenever exact identity cannot be established
from visible evidence."""
