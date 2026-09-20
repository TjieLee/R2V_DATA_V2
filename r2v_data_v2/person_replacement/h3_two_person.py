"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""


def _person_fragment(text):
    value = text.strip().rstrip(".").rstrip()
    for article in ("A ", "An ", "The "):
        if value.startswith(article):
            return article.lower() + value[len(article):]
    return value


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False):
    """Two-subject in-place appearance edit of <Video 1>.

    <Subject 1> and <Subject 2> are the same source performers seen in
    <Video 1>; replacement1/replacement2 are appearance-only modification
    descriptions for those same subjects, never additional entities. There is
    deliberately no <Subject 3> / <Subject 4> and no attribute_transfer.
    """
    # Qwen's shot description remains preparation provenance. <Video 1> is the
    # only authority for motion, events, camera and scene structure.
    source1, source2, replacement1, replacement2 = map(
        _person_fragment, (source1, source2, replacement1, replacement2)
    )

    definitions = [
        (
            f"<Subject 1> is the target performer in <Video 1> identified by {source1}. "
            "This description identifies the performer to edit. <Video 1> remains the "
            "authority for this performer's pose, motion, facial performance, interactions, "
            "position and timing."
        ),
        (
            f"<Subject 2> is the target performer in <Video 1> identified by {source2}. "
            "This description identifies the performer to edit. <Video 1> remains the "
            "authority for this performer's pose, motion, facial performance, interactions, "
            "position and timing."
        ),
        (
            "<Video 1> is the source video being directly edited. Its original shot, events, "
            "camera behavior, scene content and all non-target people remain the source for "
            "the target video."
        ),
    ]
    if frame0:
        definitions.append(
            "<Picture 1> is the edited first-frame appearance anchor for <Subject 1> and "
            "<Subject 2>. It does not replace <Video 1> as the authority for motion, events, "
            "camera or scene geometry."
        )

    retention = [
        (
            "<Subject 1> (appears in [Shot 1]): partially_preserved - preserve the performer's "
            "original pose, body motion, head motion, facial performance, gaze, mouth movement, "
            "hand motion, object interactions, position, scale, orientation, occlusion and timing "
            f"from <Video 1>; modify only the visible human identity, facial appearance, hair and "
            f"clothing to match {replacement1}."
        ),
        (
            "<Subject 2> (appears in [Shot 1]): partially_preserved - preserve the performer's "
            "original pose, body motion, head motion, facial performance, gaze, mouth movement, "
            "hand motion, object interactions, position, scale, orientation, occlusion and timing "
            f"from <Video 1>; modify only the visible human identity, facial appearance, hair and "
            f"clothing to match {replacement2}."
        ),
        (
            "<Video 1> (source video editing): fully_preserved - preserve the original shot "
            "structure, camera framing, camera motion, scene geometry, depth relationships, "
            "background, lighting, all non-target people, all non-person objects, actions, "
            "interactions and event timing. Only the specified appearance edits to <Subject 1> "
            "and <Subject 2> are made."
        ),
    ]
    if frame0:
        retention.append(
            "<Picture 1> ([Shot 1] first frame): fully_preserved - its edited first-frame "
            "appearance and composition anchor the two target performers' edited appearance only."
        )

    task = "video editing + keyframe completion" if frame0 else "video editing"
    picture = (
        "<Picture 1> supplies the edited first-frame appearance anchor; <Video 1> still defines "
        "motion, events, camera and scene. " if frame0 else ""
    )
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        (
            f"summary:\n[{task}] The target video is an edited version of <Video 1>. "
            + picture
            + "Only the visible human identity and appearance of <Subject 1> and <Subject 2> "
            "are modified. Their original performance remains from <Video 1>. Preserve the "
            "original events, camera behavior, scene, all non-target people and all "
            "non-person objects."
        ),
        "retention_analysis:\n" + "\n".join(retention),
        (
            "detailed_description:\nThe target video preserves the original visual content and "
            "complete event sequence of <Video 1>. Preserve the original camera framing and "
            "camera motion, scene geometry and depth relationships, background, lighting, all "
            "non-target people, all non-person objects, actions, interactions, positions, "
            "occlusions and timing. Only the visible human identity and appearance of "
            "<Subject 1> and <Subject 2> are modified.\n\n"
            "[Shot 1] Use <Video 1> directly as the source video.\n\n"
            "Keep <Subject 1>'s original pose, body motion, head motion, facial expression, gaze, "
            "mouth movement, hand motion, object interactions, position, scale, orientation, "
            f"occlusion and timing exactly as shown in <Video 1>. Modify only <Subject 1>'s "
            f"visible human identity, facial appearance, hair and clothing to match: "
            f"{replacement1}.\n\n"
            "Keep <Subject 2>'s original pose, body motion, head motion, facial expression, gaze, "
            "mouth movement, hand motion, object interactions, position, scale, orientation, "
            f"occlusion and timing exactly as shown in <Video 1>. Modify only <Subject 2>'s "
            f"visible human identity, facial appearance, hair and clothing to match: "
            f"{replacement2}.\n\n"
            "All other visible people and all other scene content remain those of <Video 1>. "
            "Do not create replacement people as additional subjects. Do not reinterpret or "
            "regenerate the shot as a new scene.\n\n"
            "If any appearance description conflicts with the source performance or scene "
            "structure in <Video 1>, preserve <Video 1> and apply only the compatible "
            "appearance change."
        ),
        "overall_soundscape:\nN/A",
        "non_diegetic_music:\nN/A",
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
