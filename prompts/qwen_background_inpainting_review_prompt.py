FOREGROUND_REMOVAL_REVIEW_PROMPT = """
Review foreground removal for one connected component using four images:
1. original_mask_only, where pixels outside the component are neutral gray;
2. repaired_mask_only, where pixels outside the component are neutral gray;
3. repaired_context, the normal repaired crop;
4. the binary component mask, where white is the evaluated region.

Judge only content inside or meaningfully overlapping the white mask. Compare
the two mask-only images to determine whether original foreground remains or
was reconstructed. Use repaired_context only to resolve shapes at the mask
boundary.

Report each decision independently. List every recognizable person, people,
man, woman, child, animal, vehicle, car, boat, table, chair, product, face,
body silhouette, or other distinct bounded entity visible in the repaired
mask. If any entity is listed, new_salient_entity_visible must be true unless
it is clearly the original foreground, and background_only_inside_mask must be
false.

Uniform or continuous water, sky, grass, soil, rock, coral, cloud, wall,
pavement, and similar scene materials are background, not salient entities.
Clean water or coral texture may be background_only_inside_mask. However, a
mountain, forest, tree line, or other large original structure reconstructed
from original_mask_only must set original_foreground_reconstructed=true even
when it looks like natural scenery.

Large pixel differences inside the mask are expected after successful
foreground removal and are not evidence of failure. Use uncertain only when a
foreground identity cannot be resolved. The booleans, visible_entities, and
reason must describe the same visual conclusion.

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
