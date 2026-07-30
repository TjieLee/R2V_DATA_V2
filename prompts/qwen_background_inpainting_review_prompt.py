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


BACKGROUND_INPAINTING_LOCAL_REVIEW_PROMPT = """
Review a local crop around a background hole-fill using the three supplied
images. The first image is the original crop, the second is the repaired crop,
and the third is the generation mask crop. White mask pixels identify the
region expected to become coherent background matching this reference phrase:
{reference_phrase}

Inspect the masked region and its immediate boundary closely. Reject if any
person, animal, vehicle, vessel, product, face, body silhouette, or other
distinct foreground object lies inside or meaningfully overlaps the white
repair mask, or if such an object is newly introduced in the repaired image.
Do not reject an object that appears only in the surrounding contextual pixels
and was already present in the original crop. Also reject any blurred ghost,
repeated texture, inset image, artificial blob, visible boundary, seam, or
other local artifact inside the repair mask or along its boundary.

Return only the requested JSON structure.
""".strip()
