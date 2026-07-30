FOREGROUND_REMOVAL_REVIEW_PROMPT = """
Review foreground removal for one connected component using three images:
1. the original crop;
2. the repaired crop;
3. the binary component mask, where white is the evaluated region.

Classify only foreground identity inside or meaningfully overlapping the white
mask. Uniform or continuous water, sky, grass, soil, rock, coral, cloud, wall,
pavement, and similar scene materials are background, not replacement objects.
A replacement salient entity must have a distinct bounded shape and a
recognizable semantic identity. Large pixel differences inside the mask are
expected after successful foreground removal and are not evidence of failure.
Do not reject an unchanged object that appears only in surrounding context.

Use removed when no foreground identity remains in the mask. Use remains when
the original foreground is still present, reconstructed when it was recreated,
and replaced_by_salient_entity only for a new distinct recognizable entity.
Use uncertain only when a foreground identity cannot be resolved.

Return only the requested JSON structure.
""".strip()


BACKGROUND_CONTINUITY_REVIEW_PROMPT = """
Review background continuity for one connected component using five images:
1. the context-only original crop, with mask pixels neutral gray;
2. the repaired crop;
3. the binary component mask, where white is the repaired region;
4. the raw original boundary crop with the foreground hidden;
5. the raw repaired boundary crop.

The original foreground content is intentionally hidden and must not be
inferred or classified. Judge only whether the repaired pixels continue the
surrounding background with coherent geometry, perspective, depth, texture,
color, exposure, and lighting. Inspect the raw boundary crops for seams,
ghosting, double exposure, artificial blobs, texture discontinuity, and
color/exposure mismatch. Neutral gray pixels are unavailable context, not an
artifact. Large changes inside the white mask are expected and are not by
themselves a failure.

Set uncertain only when continuity or an artifact cannot be resolved.

Return only the requested JSON structure.
""".strip()


FULL_SCENE_REVIEW_PROMPT = """
Review global scene semantics using:
1. the repaired full image;
2. the optional context-only original full image, with repaired-mask pixels
   neutral gray.

Reference phrase: {reference_phrase}

Decide only whether the repaired image supports the reference phrase and
whether the full scene remains globally coherent. Do not classify foreground
removal, local seams, local artifacts, masked-region pixel differences, or
object identity inside the repaired region. Those decisions belong to separate
component reviews.

Return only the requested JSON structure.
""".strip()
