"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""


def _person_fragment(text):
    value = text.strip().rstrip(".").rstrip()
    for article in ("A ", "An ", "The "):
        if value.startswith(article):
            return article.lower() + value[len(article):]
    return value


PERFORMANCE_NONE = "NONE"
_SENTENCE_PUNCTUATION = ".!?;:,"


def normalize_performance_anchor(value):
    """Strip Qwen's own sentence punctuation; NONE means no object to protect."""
    text = (value or "").strip().rstrip(_SENTENCE_PUNCTUATION).rstrip()
    if not text or text.casefold() == PERFORMANCE_NONE.casefold():
        return None
    return text


# Generic fallback: Qwen may miss an object, so this rule is always present.
PERFORMANCE_OBJECT_RULE = (
    "Any source object held, carried, touched, or otherwise interacted with in <Video 1> "
    "must remain the same object after replacement, with the same hand-object contact and "
    "source timing."
)


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False,
                 performance1=None, performance2=None):
    # Qwen's shot description remains preparation provenance. H3 motion is
    # driven directly by <Video 1> so a text re-description cannot retime it.
    source1, source2, replacement1, replacement2 = map(
        _person_fragment, (source1, source2, replacement1, replacement2)
    )
    # Object-only anchors. Re-describing pose, clothing or general action in text
    # makes H3 follow the prose instead of <Video 1>, so NONE is dropped entirely.
    anchor1, anchor2 = (normalize_performance_anchor(performance1),
                        normalize_performance_anchor(performance2))

    definitions = [
        (
            f"<Subject 1> is a source-only performer identity in <Video 1> identified by {source1}. "
            "This description is only for identifying the performer being replaced; <Video 1> "
            "determines <Subject 1>'s action, pose, held-object state and timing. "
            "<Subject 1> must not appear as an additional person in the target video."
        ),
        (
            f"<Subject 2> is a source-only performer identity in <Video 1> identified by {source2}. "
            "This description is only for identifying the performer being replaced; <Video 1> "
            "determines <Subject 2>'s action, pose, held-object state and timing. "
            "<Subject 2> must not appear as an additional person in the target video."
        ),
        (
            f"<Subject 3> is {replacement1}. In the target video, <Subject 3> replaces "
            "<Subject 1> in place and takes over <Subject 1>'s position in the shot. "
            "<Subject 3> is pose-driven and motion-driven by <Video 1> and performs the same "
            "source performance with the same timing. <Subject 3> and <Subject 1> must never "
            "coexist as separate people in the target video."
            if not frame0 else
            f"<Subject 3> is {replacement1}. <Subject 3> is the edited target counterpart of "
            "<Subject 1> and replaces <Subject 1> in place, taking over <Subject 1>'s position "
            "in the shot. <Subject 3> takes its visible appearance from <Picture 1>, while its "
            "pose, body motion, facial performance, hand-object interactions, position and "
            "timing come from <Video 1>. <Subject 3> and <Subject 1> must never coexist as "
            "separate people in the target video."
        ),
        (
            f"<Subject 4> is {replacement2}. In the target video, <Subject 4> replaces "
            "<Subject 2> in place and takes over <Subject 2>'s position in the shot. "
            "<Subject 4> is pose-driven and motion-driven by <Video 1> and performs the same "
            "source performance with the same timing. <Subject 4> and <Subject 2> must never "
            "coexist as separate people in the target video."
            if not frame0 else
            f"<Subject 4> is {replacement2}. <Subject 4> is the edited target counterpart of "
            "<Subject 2> and replaces <Subject 2> in place, taking over <Subject 2>'s position "
            "in the shot. <Subject 4> takes its visible appearance from <Picture 1>, while its "
            "pose, body motion, facial performance, hand-object interactions, position and "
            "timing come from <Video 1>. <Subject 4> and <Subject 2> must never coexist as "
            "separate people in the target video."
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
    reminders = [f"<Subject {index}> must preserve this source object interaction: {anchor}."
                 for index, anchor in ((3,anchor1),(4,anchor2)) if anchor]
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
            "timing come directly from <Video 1>; the camera, scene and non-person objects remain unchanged. "
            "The replacement happens in place. Do not retain a source performer beside its replacement, "
            "duplicate a replacement, or add any new person."
        ),
        "retention_analysis:\n" + "\n".join(retention),
        (
            "detailed_description:\nKeep the visual style and lighting of <Video 1>.\n\n"
            "[Shot 1] " + anchor +
            # Target-centric: the Subject1->3 / Subject2->4 mapping is already
            # defined above, so this section never re-activates the source labels.
            "Edit <Video 1> in place. Replace the two designated source performers with "
            "<Subject 3> and <Subject 4> according to the mappings already defined above. "
            "Do not add any new person, duplicate either replacement, or keep a source identity "
            "as an additional person. Preserve the same visible-person count and occupancy as "
            "<Video 1> at every moment.\n\n"
            "<Subject 3> and <Subject 4> inherit their corresponding source performers' pose, "
            "body and head motion, facial expression, gaze, mouth movement, hand motion, position, "
            "occlusion and timing directly from <Video 1>. Only human identity, appearance and "
            "clothing may change.\n\n"
            + (" ".join(reminders) + " " if reminders else "")
            + PERFORMANCE_OBJECT_RULE
            + "\n\nDo not reinterpret facial emotion, pose or action from text. "
            "Do not retime, invent, omit, merge or reinterpret actions. Preserve the camera, "
            "background, lighting and every non-person object. If any text conflicts with "
            "<Video 1>, follow <Video 1>."
        ),
        (
            "overall_soundscape:\nPreserve source synchronized sound, ambience and physical "
            "sounds; do not invent new effects or voices because the appearances change."
        ),
        "non_diegetic_music:\nPreserve existing source music; do not add new music.",
    ])


def boogu_frame0_prompt(source1, source2, replacement1, replacement2):
    """Still-image first-frame edit instruction; no video timing or extra terms."""
    return (
        f"Replace the person matching {source1} with {replacement1}.\n"
        f"Replace the person matching {source2} with {replacement2}.\n\n"
        "Keep the original pose, expression, hands, object interactions, positions, "
        "background and composition unchanged.\n"
        "Do not add a person who is not visible in the image."
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
