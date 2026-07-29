INPAINTING_CONSISTENCY_PROMPT = """
Compare the original reference image and the repaired reference image.
The first image is original and the second image is repaired.

Decide whether the repaired image preserves the same scene or entity identity,
matches the supplied reference phrase, and introduces no new salient object or
semantic contradiction. Judge semantics, not exact pixels.

Reference phrase: {reference_phrase}
Repair mode: {repair_mode}

Return only the requested JSON structure.
""".strip()
