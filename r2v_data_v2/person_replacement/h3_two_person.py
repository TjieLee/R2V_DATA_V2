"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False):
    definitions = []
    for index, source, replacement in ((1, source1, replacement1), (2, source2, replacement2)):
        definitions.append(
            f"<Subject {index}> is the physical performer in <Video 1> identified by {source}. "
            f"Its intended target appearance is Replacement {index}: {replacement}. "
            "The label follows that same physical performer, not current screen position; "
            "<Video 1> supplies its complete temporal motion/performance track."
            + (" <Picture 1> additionally supplies a concrete edited first-frame visual anchor "
               "where visible; replacement text defines the intended target identity." if frame0 else "")
        )
    definitions.append("<Video 1> is the source video for the target video edit.")
    if frame0:
        definitions.append("<Picture 1> is the edited first-frame target appearance and composition anchor, "
                           "not the sole identity source and not evidence that both people are visible.")
    retention = [f"<Subject {i}> (throughout its source performance): partially_preserved - "
                 "identity/appearance replaced; exact source performance retained." for i in (1, 2)]
    retention.append("<Video 1> (whole single-shot temporal structure): partially_preserved - "
                     "only the two target appearances change; all other source content is retained.")
    if frame0:
        retention.append("<Picture 1> (first-frame anchor): fully_preserved - retain its edited "
                         "visible target appearances and composition at the start of the shot.")
    # Preservation is prompt intent, not a claim of byte-copied audio: retain
    # the existing PDD video-reference/audio-generation behavior unchanged.
    task = "video editing + keyframe completion" if frame0 else "video editing"
    anchor = "The shot begins from the edited first-frame anchor <Picture 1>. " if frame0 else ""
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        (f"summary:\n[{task}] The target video is an edited version of <Video 1>. "
        "<Subject 1> -> Replacement 1; <Subject 2> -> Replacement 2. Change only their identities/visible "
        "appearances; never cross-bind or swap the two replacements. Preserve source motion, action, "
        "timing, interaction, camera and background."),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\nKeep the source visual style and lighting unchanged.\n[Shot 1] " + anchor + shot +
        " <Subject 1> follows its exact original source trajectory/performance in <Video 1>; "
        "<Subject 2> follows its own exact original trajectory/performance. Preserve body pose/orientation, "
        "limbs, hands, head pose, expression, gaze, mouth state, position, scale, depth, occlusion and timing. "
        "Preserve hand-body, hand-object and person-person contact, held objects and all interactions. "
        "During crossing, overlap or exchanged screen order, Subject 1 remains Replacement 1 and "
        "Subject 2 remains Replacement 2: never cross-bind. Keep camera motion, crop, framing, background, "
        "objects, lighting and composition unchanged. Follow the complete single-shot source timeline "
        "without adding cuts, actions or new dialogue; preserve source synchronized sound.",
        ("overall_soundscape:\nPreserve source synchronized sound, ambience and physical sounds; "
        "do not invent new effects or voices because the appearances change."),
        "non_diegetic_music:\nPreserve existing source music; do not add new music.",
    ])


def boogu_prompt(source1, source2, replacement1, replacement2):
    return (
        f"In one edit replace the visible person matching {source1} -> {replacement1}; "
        f"replace the visible person matching {source2} -> {replacement2}. "
        "Keep each binding fixed; never swap the two target identities. Preserve exact pose, hands, "
        "hand-body and hand-object contact, person-person contact and object interactions, facial expression, "
        "gaze, mouth state, position, scale and occlusion. Preserve camera, crop, background, objects "
        "and lighting. Do not add a missing person merely because it appears in the text. "
        "Change only the visible target people's identity and appearance."
    )
