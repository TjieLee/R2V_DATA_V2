"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""

import re


def _appearance_fragment(text):
    return text.strip().rstrip(".").rstrip()


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False):
    # Source anchors remain preparation provenance; never inject them into H3.
    shot = re.sub(r"(?<!<)\bSubject ([12])\b(?!>)", r"<Subject \1>", shot)
    replacement1, replacement2 = map(_appearance_fragment, (replacement1, replacement2))
    definitions = []
    for index, appearance in ((1, replacement1), (2, replacement2)):
        definitions.append(
            f"<Subject {index}> is the edited version of the source performer whose actions are tracked "
            f"as <Subject {index}> in [Shot 1] of <Video 1>. In the target video, <Subject {index}> "
            f"appears as {appearance}. The corresponding source performer in <Video 1> provides "
            f"<Subject {index}>'s motion, pose, expression, hand state, object interactions, position and timing."
            + (" <Picture 1> additionally supplies an edited first-frame visual anchor where visible." if frame0 else "")
        )
    definitions.append("<Video 1> is the source video for the target video edit and provides the original "
                       "action sequence, interactions, camera, background, lighting and composition.")
    if frame0:
        definitions.append("<Picture 1> is the edited first-frame target appearance and composition anchor, "
                           "not the sole identity source and not evidence that both people are visible.")
    retention = [f"<Subject {i}> (throughout [Shot 1]): partially_preserved - the corresponding source "
                 "performer's motion, pose, expression, hand state, object interactions, position and timing "
                 "are retained from <Video 1>, while the visible identity, hair and clothing follow the "
                 "target appearance defined above." for i in (1, 2)]
    retention.append("<Video 1> (motion, interaction, camera and scene structure): fully_preserved - "
                     "retain the original action order and timing, poses, hand-object contacts, non-person "
                     "props, camera, background, lighting and composition.")
    if frame0:
        retention.append("<Picture 1> (first-frame anchor): fully_preserved - retain its edited "
                         "visible target appearances and composition at the start of the shot.")
    # Preservation is prompt intent, not a claim of byte-copied audio.
    task = "video editing + keyframe completion" if frame0 else "video editing"
    anchor = "The shot begins from the edited first-frame anchor <Picture 1>. " if frame0 else ""
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        (f"summary:\n[{task}] The target video is an edited version of <Video 1>. "
         "The two tracked performers keep their original performances and interactions from <Video 1>, "
         "while their visible identities and clothing are changed to the target appearances defined for "
         "<Subject 1> and <Subject 2>. Preserve the original action order and timing, non-person props, camera and scene."),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\nKeep the visual style and lighting of <Video 1>.\n\n[Shot 1] " + anchor +
        f"In the target video, <Subject 1> appears as {replacement1}, while <Subject 2> appears as {replacement2}. " + shot +
        " These actions and interactions follow <Video 1> in the same order and timing. "
        "Keep all non-person props unchanged, including held or touched objects, and preserve the original "
        "hand-object contact, camera, background, lighting and composition. The source performers' original "
        "clothing is replaced by the clothing described for <Subject 1> and <Subject 2>. "
        "Keep each subject on its corresponding source performer's track throughout crossings, overlap or "
        "occlusion; do not swap the two subjects.",
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
