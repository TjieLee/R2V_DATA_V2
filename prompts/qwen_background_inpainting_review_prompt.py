BACKGROUND_INPAINTING_REVIEW_PROMPT = """
Review a background hole-fill result using the three supplied images.
The first image is the original image, which intentionally contains a
foreground object inside the white mask. The second image is the repaired
image, where that masked foreground is expected to be removed. The third image
is the exact repair mask.

Removal of the masked foreground is expected and is not a semantic
inconsistency. Inspect whether the repaired region coherently continues the
surrounding background, including perspective, lighting, color, depth, and
texture. The repaired background must still support this reference phrase:
{reference_phrase}

Reject the result if the original masked foreground remains, another entity
replaces it, any new salient object is introduced, or an obvious seam or visual
artifact is visible.

Return only the requested JSON structure.
""".strip()
