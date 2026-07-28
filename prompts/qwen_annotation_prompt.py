from __future__ import annotations

SYSTEM_PROMPT = """You are constructing annotations for reference-conditioned video generation.

Inspect all eight frames in chronological order and return one valid JSON object
containing:
1. one concise chronological video caption;
2. visually meaningful reference entities;
3. visible entity relations;
4. a copy of the caption with explicit reference-token bindings.

CAPTION:
Write one flowing English paragraph that starts directly with the main action.
Describe events chronologically like a cinematographer describing a shot.
Include visible movements and gestures, stable character or object appearances,
the environment, camera angle or motion, lighting, colors, and meaningful
changes. Keep the description literal and precise. Prefer 40-180 words and
never exceed 200 words. Do not repeat sentences, actions, entity descriptions,
or scene information. Do not use introductory phrases such as "the video
shows". Do not infer sound, dialogue, emotion, exact age, identity, or intent
unless supported by explicit metadata or visible evidence.

ENTITIES:
Select an entity as reference-worthy only when it is visually salient,
distinguishable, reasonably stable over time, likely segmentable, and useful
for preserving appearance during generation. Prefer specific visual phrases.
Generic phrases such as "man", "woman", "child", and "person" are allowed when
the video provides no reliable distinguishing attributes, but minimize them.
Exclude actions, body parts, vague attributes, tiny incidental objects, and
duplicate phrases referring to the same instance. A famous identity may be
used only when supported by supplied metadata, the draft caption, or visible
text; never identify a person from facial appearance alone.

BINDING:
Place each token immediately after its exact noun phrase:
<ref_subject_n>, <ref_object_n>, <ref_bg_n>, or <ref_group_n>.
Each selected phrase and token must occur exactly once. Use no token for an
entity that is not selected as reference-worthy.

RELATIONS:
Record visible relations such as holding, wearing, riding, carrying, inside,
attached to, standing beside, or interacting with. Mark an entity as
attached_accessory or composite_candidate when it cannot be cleanly separated
from its parent.

Return JSON only, following the provided schema."""


ICL_EXAMPLES: list[dict[str, object]] = [
    {
        "input": {
            "draft_caption": "Michael Jordan climbs a snowy ridge.",
            "metadata": {"identity_source": "draft_caption"},
        },
        "output": {
            "caption": (
                "Michael Jordan climbs steadily over a rocky snow-covered ridge, "
                "planting his boots between exposed stones as the camera follows "
                "from a low rear angle. His red insulated jacket and black backpack "
                "remain visible against a broad alpine valley under cold blue light."
            ),
            "prompt_with_refs": (
                "Michael Jordan <ref_subject_1> climbs steadily over a rocky "
                "snow-covered ridge, planting his boots between exposed stones as "
                "the camera follows from a low rear angle. His red insulated jacket "
                "and black backpack remain visible against a broad alpine valley "
                "<ref_bg_1> under cold blue light."
            ),
            "entities": [
                {
                    "entity_id": "e1",
                    "phrase": "Michael Jordan",
                    "grounding_prompt": "Michael Jordan in a red insulated jacket",
                    "canonical_label": "Michael Jordan",
                    "category": "person",
                    "ref_token": "<ref_subject_1>",
                    "reference_worthy": True,
                    "salience": "primary",
                    "genericity": "named",
                    "name_evidence": "draft_caption",
                    "separability": "independent",
                    "selection_reason": "named primary subject supported by metadata",
                }
            ],
            "relations": [],
            "background": {
                "phrase": "a broad alpine valley",
                "grounding_prompt": "empty broad snowy alpine valley",
                "ref_token": "<ref_bg_1>",
                "reference_worthy": True,
            },
        },
    },
    {
        "input": {"draft_caption": "", "metadata": {}},
        "output": {
            "caption": (
                "A woman in a yellow raincoat walks beside a man in a navy cap "
                "through a wet market lane, while both step around shallow puddles. "
                "The handheld camera moves backward at chest height as cool daylight "
                "reflects from blue awnings and damp pavement."
            ),
            "prompt_with_refs": (
                "A woman in a yellow raincoat <ref_subject_1> walks beside a man in "
                "a navy cap <ref_subject_2> through a wet market lane, while both "
                "step around shallow puddles. The handheld camera moves backward at "
                "chest height as cool daylight reflects from blue awnings and damp "
                "pavement."
            ),
            "entities": [
                {
                    "entity_id": "e1",
                    "phrase": "A woman in a yellow raincoat",
                    "grounding_prompt": "woman wearing a yellow raincoat",
                    "canonical_label": "woman",
                    "category": "person",
                    "ref_token": "<ref_subject_1>",
                    "reference_worthy": True,
                    "salience": "primary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "independent",
                    "selection_reason": "stable clothing distinguishes the subject",
                },
                {
                    "entity_id": "e2",
                    "phrase": "a man in a navy cap",
                    "grounding_prompt": "man wearing a navy cap",
                    "canonical_label": "man",
                    "category": "person",
                    "ref_token": "<ref_subject_2>",
                    "reference_worthy": True,
                    "salience": "secondary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "independent",
                    "selection_reason": "stable companion visible across the shot",
                },
            ],
            "relations": [
                {
                    "subject_id": "e1",
                    "predicate": "walking beside",
                    "object_id": "e2",
                }
            ],
            "background": None,
        },
    },
    {
        "input": {"draft_caption": "A man raises a wine glass.", "metadata": {}},
        "output": {
            "caption": (
                "A gray-haired man in a dark suit raises a small wine glass near his "
                "shoulder, pauses, and lowers it toward a white table. The camera "
                "holds a steady medium view inside a softly lit reception hall with "
                "warm highlights along the glass and his jacket."
            ),
            "prompt_with_refs": (
                "A gray-haired man in a dark suit <ref_subject_1> raises a small wine "
                "glass near his shoulder, pauses, and lowers it toward a white table. "
                "The camera holds a steady medium view inside a softly lit reception "
                "hall with warm highlights along the glass and his jacket."
            ),
            "entities": [
                {
                    "entity_id": "e1",
                    "phrase": "A gray-haired man in a dark suit",
                    "grounding_prompt": "gray-haired man wearing a dark suit",
                    "canonical_label": "man",
                    "category": "person",
                    "ref_token": "<ref_subject_1>",
                    "reference_worthy": True,
                    "salience": "primary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "independent",
                    "selection_reason": "primary stable subject",
                },
                {
                    "entity_id": "e2",
                    "phrase": "a small wine glass",
                    "grounding_prompt": "small stemmed wine glass in his hand",
                    "canonical_label": "wine glass",
                    "category": "object",
                    "ref_token": None,
                    "reference_worthy": False,
                    "salience": "secondary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "attached_accessory",
                    "selection_reason": "generic accessory attached to the subject",
                },
            ],
            "relations": [
                {"subject_id": "e1", "predicate": "holding", "object_id": "e2"}
            ],
            "background": None,
        },
    },
    {
        "input": {"draft_caption": "A watch turns inside its case.", "metadata": {}},
        "output": {
            "caption": (
                "A silver dive watch rotates slowly inside an open black presentation "
                "case, bringing its blue dial and metal bracelet into view. A hand "
                "briefly steadies the case before leaving the frame, while a locked "
                "close camera and soft white studio lights reveal crisp reflections."
            ),
            "prompt_with_refs": (
                "A silver dive watch <ref_object_1> rotates slowly inside an open "
                "black presentation case <ref_object_2>, bringing its blue dial and "
                "metal bracelet into view. A hand briefly steadies the case before "
                "leaving the frame, while a locked close camera and soft white studio "
                "lights reveal crisp reflections."
            ),
            "entities": [
                {
                    "entity_id": "e1",
                    "phrase": "A silver dive watch",
                    "grounding_prompt": "silver dive watch with blue dial",
                    "canonical_label": "watch",
                    "category": "product",
                    "ref_token": "<ref_object_1>",
                    "reference_worthy": True,
                    "salience": "primary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "independent",
                    "selection_reason": "primary product with stable details",
                },
                {
                    "entity_id": "e2",
                    "phrase": "an open black presentation case",
                    "grounding_prompt": "open black watch presentation case",
                    "canonical_label": "presentation case",
                    "category": "object",
                    "ref_token": "<ref_object_2>",
                    "reference_worthy": True,
                    "salience": "secondary",
                    "genericity": "descriptive",
                    "name_evidence": "none",
                    "separability": "important_independent_object",
                    "selection_reason": "distinct product container",
                },
            ],
            "relations": [
                {"subject_id": "e1", "predicate": "inside", "object_id": "e2"}
            ],
            "background": None,
        },
    },
]
