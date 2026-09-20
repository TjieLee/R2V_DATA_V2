"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""


def _person_fragment(text):
    value = text.strip().rstrip(".").rstrip()
    for article in ("A ", "An ", "The "):
        if value.startswith(article):
            return article.lower() + value[len(article):]
    return value


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False):
    # Qwen's shot description remains preparation provenance. H3 motion is
    # driven directly by <Video 1> so a text re-description cannot retime it.
    source1, source2, replacement1, replacement2 = map(
        _person_fragment, (source1, source2, replacement1, replacement2)
    )

    definitions = [
        (
            f"<Subject 1> is the source performer in <Video 1> identified by {source1}. "
            "This description is only for identifying the performer; <Video 1> determines "
            "<Subject 1>'s action, pose, held-object state and timing."
        ),
        (
            f"<Subject 2> is the source performer in <Video 1> identified by {source2}. "
            "This description is only for identifying the performer; <Video 1> determines "
            "<Subject 2>'s action, pose, held-object state and timing."
        ),
        (
            f"<Subject 3> is {replacement1}. In the target video, <Subject 3> replaces only "
            "<Subject 1>'s visible human identity and appearance. <Subject 3> is pose-driven "
            "and motion-driven by <Subject 1> in <Video 1> and performs the same source "
            "performance with the same timing."
            if not frame0 else
            f"<Subject 3> is {replacement1}. <Subject 3> is the edited target counterpart of "
            "<Subject 1> and replaces only <Subject 1>'s visible human identity and appearance. "
            "<Subject 3> takes its visible appearance from <Picture 1>, while its pose, body motion, "
            "facial performance, hand-object interactions, position and timing come from "
            "<Subject 1> in <Video 1>."
        ),
        (
            f"<Subject 4> is {replacement2}. In the target video, <Subject 4> replaces only "
            "<Subject 2>'s visible human identity and appearance. <Subject 4> is pose-driven "
            "and motion-driven by <Subject 2> in <Video 1> and performs the same source "
            "performance with the same timing."
            if not frame0 else
            f"<Subject 4> is {replacement2}. <Subject 4> is the edited target counterpart of "
            "<Subject 2> and replaces only <Subject 2>'s visible human identity and appearance. "
            "<Subject 4> takes its visible appearance from <Picture 1>, while its pose, body motion, "
            "facial performance, hand-object interactions, position and timing come from "
            "<Subject 2> in <Video 1>."
        ),
        "<Video 1> is the source video for the target video edit and the pose, motion and timing driver for the entire shot.",
    ]
    if frame0:
        definitions.append(
            "<Picture 1> is the edited first frame of [Shot 1] and the appearance and composition "
            "anchor of the target video. <Picture 1> controls the two target people's visible "
            "appearance; it never defines pose, motion or timing."
        )

    retention = [
        "<Subject 1> (appears in [Shot 1]): attribute_transfer - <Subject 1>'s pose, motion, "
        "facial performance, hand and object interactions, position and timing from <Video 1> "
        "are transferred to <Subject 3>; <Subject 1>'s visible identity and appearance are not retained.",
        "<Subject 2> (appears in [Shot 1]): attribute_transfer - <Subject 2>'s pose, motion, "
        "facial performance, hand and object interactions, position and timing from <Video 1> "
        "are transferred to <Subject 4>; <Subject 2>'s visible identity and appearance are not retained.",
        "<Subject 3> (appears in [Shot 1]): fully_preserved - the defined replacement appearance "
        "is retained while pose and motion are driven by <Subject 1> in <Video 1>.",
        "<Subject 4> (appears in [Shot 1]): fully_preserved - the defined replacement appearance "
        "is retained while pose and motion are driven by <Subject 2> in <Video 1>.",
        "<Video 1> (pose, motion, timing and source video editing): fully_preserved - use <Video 1> "
        "as the motion source for the complete performance and preserve its action timing, body and "
        "head pose, facial expression, gaze, mouth movement, hand pose, object contacts, positions, "
        "scale, orientation, occlusion, camera, background, lighting and non-person objects.",
    ]
    if frame0:
        retention.append(
            "<Picture 1> ([Shot 1] first frame): fully_preserved - retain its edited target "
            "appearances and composition at the start of the shot."
        )

    task = "video editing + keyframe completion" if frame0 else "video editing"
    anchor = (
        "Use <Picture 1> as the first-frame appearance authority. Use <Video 1> as the pose, motion "
        "and timing driver. The shot begins from the edited first-frame anchor <Picture 1>. "
        "Only human appearance changes; no action, object state or timing may change. "
        if frame0 else ""
    )
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        (
            f"summary:\n[{task}] The target video is an edited version of <Video 1>. "
            + ("<Picture 1> is the edited first frame of [Shot 1] and defines the edited first-frame "
               "appearance of <Subject 3> and <Subject 4> where visible; <Video 1> provides the "
               "complete source performance, including every pose, motion and timing. " if frame0 else "")
            + "This is a pose-driven human appearance replacement. <Subject 3> replaces only "
            "<Subject 1>'s visible identity and appearance, and <Subject 4> replaces only "
            "<Subject 2>'s visible identity and appearance. Their pose, motion, actions and "
            "timing come directly from <Video 1>; the camera, scene and non-person objects remain unchanged."
        ),
        "retention_analysis:\n" + "\n".join(retention),
        (
            "detailed_description:\nKeep the visual style and lighting of <Video 1>.\n\n"
            "[Shot 1] " + anchor +
            "This is a pose-driven and motion-driven human appearance replacement. Use <Video 1> "
            "as the pose and motion driver for the entire shot. A video of <Subject 3> and <Subject 4> "
            "performing the same specific actions as <Subject 1> and <Subject 2> in <Video 1>. "
            "<Subject 3> follows <Subject 1>'s source pose, body motion, head motion, facial expression, "
            "gaze, mouth movement, hand pose, limb motion, object interactions, position and timing. "
            "<Subject 4> follows the corresponding source performance of <Subject 2>. Copy the source "
            "performance from <Video 1> rather than generating or reinterpreting a new performance. "
            "Only the visible human identity, appearance and clothing of the two performers may change. "
            "Do not retime, invent, omit, merge or reinterpret actions. Preserve the camera, background, "
            "lighting and every non-person object. If any text description conflicts with <Video 1>, "
            "follow <Video 1>."
        ),
        (
            "overall_soundscape:\nPreserve source synchronized sound, ambience and physical "
            "sounds; do not invent new effects or voices because the appearances change."
        ),
        "non_diegetic_music:\nPreserve existing source music; do not add new music.",
    ])


def boogu_frame0_prompt(source1, source2, replacement1, replacement2):
    """Still-image first-frame edit instruction; no video timing or new people."""
    return (
        "Edit only the two visible people in this frame.\n\n"
        f"Replace the visible appearance of the person matching {source1} with {replacement1}.\n\n"
        f"Replace the visible appearance of the person matching {source2} with {replacement2}.\n\n"
        "Preserve each person's exact pose, body orientation, facial expression, gaze, mouth state, "
        "hand pose, held or touched objects, hand-object contacts, position, scale and occlusion.\n\n"
        "Preserve every non-person object, the background, framing, lighting and composition.\n\n"
        "Do not add a person who is not visible in the source frame. Do not change any action or "
        "object state. Only the two target people's visible identity, hair, body appearance and "
        "clothing may change."
    )


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
