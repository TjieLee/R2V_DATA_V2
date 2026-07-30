BACKGROUND_FILL_PROMPT = """
You are writing a positive FLUX.1 Fill prompt for background completion.
Inspect the original image and the white generation mask. Describe only the
background that should plausibly occupy the white region by extrapolating from
adjacent unmasked pixels.

Return JSON only:
{"fill_prompt": "...", "visible_background_elements": [...], "reason": "..."}

Rules:
- Write 12-35 English words.
- Describe surfaces, materials, geometry, perspective, texture, color,
  lighting, depth, and atmospheric conditions that should continue through the
  white area.
- Do not mention the masked foreground or any person, animal, face, body,
  vehicle, vessel, product, logo, sign, or other salient foreground entity.
- Do not use: remove, replace, erase, without, no, empty, foreground, object,
  mask, hole.
- Do not copy the video caption or the background reference phrase.
- Do not invent hidden structures unsupported by adjacent pixels.
- Use positive natural-language description only.
""".strip()
