BACKGROUND_INPAINTING_REVIEW_PROMPT = """
Review a background hole-fill result using four supplied images:
1. the original full image;
2. the repaired full image;
3. the exact white generation mask;
4. a pixel-aligned comparison sheet with mask overlays, absolute-difference
   heatmap, and enlarged boundary-ring comparison.

The original intentionally contains foreground content inside the white mask.
The repaired image must replace that masked content with plausible background
that continues the surrounding geometry, perspective, texture, color,
lighting, depth, and atmosphere. The repaired full image must still support
this global reference phrase:
{reference_phrase}

A visually similar reconstruction of the original masked content is a failure,
even when it looks natural in the scene. Reconstructing the same mountain
slope, tree, person, animal, vehicle, vessel, product, or other salient
structure is foreground_reconstructed, not foreground removal. A different
salient entity in the mask is replaced_by_new_object.

Select exactly one masked_content_outcome. Use uncertain when the comparison is
ambiguous. Record every visible artifact in artifact_types. The categorical
outcome, artifact_types, and reason must agree with the visual evidence.

Return only the requested JSON structure.
""".strip()


BACKGROUND_INPAINTING_LOCAL_REVIEW_PROMPT = """
Review one connected-component crop from a background hole-fill using four
supplied images:
1. the original crop;
2. the repaired crop;
3. the component's white generation mask;
4. a pixel-aligned comparison sheet with mask overlays, absolute-difference
   heatmap, and enlarged boundary-ring comparison.

The full-frame review alone determines whether this global reference phrase is
supported:
{reference_phrase}
Do not reject the local crop merely because that phrase is not visible here.

The white component must become plausible background continuing the adjacent
pixels. A visually similar reconstruction of the original masked content is a
failure, even when natural-looking. Reconstructing the same mountain slope,
tree, person, animal, vehicle, vessel, product, or other salient structure is
foreground_reconstructed. A different salient entity is
replaced_by_new_object. Do not reject an unchanged object that appears only in
the surrounding context and does not meaningfully overlap the white mask.
Inside or meaningfully overlapping the white mask, inspect whether any person,
animal, vehicle, vessel, product, face, body silhouette, or other distinct
foreground content remains or newly appears.

Select exactly one masked_content_outcome. Use uncertain when evidence is
ambiguous. Record every seam, ghost, double exposure, inset image, texture
discontinuity, artificial blob, or color/exposure mismatch in artifact_types.
The outcome, artifact_types, and reason must agree with the comparison sheet.

Return only the requested JSON structure.
""".strip()
