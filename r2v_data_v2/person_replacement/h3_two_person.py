"""Deterministic paired two-person prompts following MiniMax's Ref2VA guide."""

import re


def _person_fragment(text):
    value = text.strip().rstrip(".").rstrip()
    for article in ("A ", "An ", "The "):
        if value.startswith(article):
            return article.lower() + value[len(article):]
    return value


_SUBJECT_OR_PRONOUN = re.compile(
    r"<Subject ([34])>|\b(he|she|him|her|his|hers)\b",
    re.IGNORECASE,
)
_HER_POSSESSIVE_NEXT = re.compile(
    r"\s+(?:right|left|own|head|face|mouth|eye|eyes|gaze|hair|hand|hands|arm|arms|"
    r"shoulder|shoulders|body|torso|leg|legs|foot|feet|clothing|shirt|jacket|robe|"
    r"tunic|trousers|pants|cup|chair|object)\b",
    re.IGNORECASE,
)


def _replace_person_pronouns(text):
    current_subject = None

    def replace(match):
        nonlocal current_subject
        if match.group(1):
            current_subject = match.group(1)
            return match.group(0)
        if current_subject is None:
            return match.group(0)

        label = f"<Subject {current_subject}>"
        pronoun = match.group(2).lower()
        if pronoun in {"his", "hers"}:
            return label + "'s"
        if pronoun == "her" and _HER_POSSESSIVE_NEXT.match(text[match.end():]):
            return label + "'s"
        return label

    return _SUBJECT_OR_PRONOUN.sub(replace, text)


def build_prompt(source1, source2, shot, replacement1, replacement2, *, frame0=False):
    shot = re.sub(r"(?<!<)\bSubject ([12])\b(?!>)", r"<Subject \1>", shot)
    shot = re.sub(
        r"<Subject ([12])>",
        lambda match: f"<Subject {int(match.group(1)) + 2}>",
        shot,
    )
    shot = _replace_person_pronouns(shot)
    source1, source2, replacement1, replacement2 = map(
        _person_fragment, (source1, source2, replacement1, replacement2)
    )

    definitions = [
        f"<Subject 1> is {source1} in <Video 1>.",
        f"<Subject 2> is {source2} in <Video 1>.",
        (
            f"<Subject 3> is {replacement1}. In the target video, <Subject 3> replaces "
            "<Subject 1>'s visible appearance and follows <Subject 1>'s original performance "
            "from <Video 1>, including motion, pose, facial expression, gaze, mouth state, "
            "hand movements, object interactions, position and timing."
            + (" <Picture 1> additionally supplies an edited first-frame visual anchor where visible." if frame0 else "")
        ),
        (
            f"<Subject 4> is {replacement2}. In the target video, <Subject 4> replaces "
            "<Subject 2>'s visible appearance and follows <Subject 2>'s original performance "
            "from <Video 1>, including motion, pose, facial expression, gaze, mouth state, "
            "hand movements, object interactions, position and timing."
            + (" <Picture 1> additionally supplies an edited first-frame visual anchor where visible." if frame0 else "")
        ),
        "<Video 1> is the source video for the target video edit.",
    ]
    if frame0:
        definitions.append(
            "<Picture 1> is the edited first-frame target appearance and composition anchor."
        )

    retention = [
        "<Subject 1> (appears in [Shot 1]): attribute_transfer - its original performance "
        "from <Video 1>, including motion, pose, facial expression, gaze, mouth state, hand "
        "movements, object interactions, position and timing, is transferred to <Subject 3>; "
        "<Subject 1>'s original visible appearance is not retained.",
        "<Subject 2> (appears in [Shot 1]): attribute_transfer - its original performance "
        "from <Video 1>, including motion, pose, facial expression, gaze, mouth state, hand "
        "movements, object interactions, position and timing, is transferred to <Subject 4>; "
        "<Subject 2>'s original visible appearance is not retained.",
        "<Subject 3> (appears in [Shot 1]): fully_preserved - the defined replacement "
        "appearance is retained while <Subject 3> follows <Subject 1>'s original performance.",
        "<Subject 4> (appears in [Shot 1]): fully_preserved - the defined replacement "
        "appearance is retained while <Subject 4> follows <Subject 2>'s original performance.",
        "<Video 1> (source video editing): fully_preserved - retain the original action order "
        "and timing, poses, hand-object contacts, non-person props, camera, background, "
        "lighting and composition.",
    ]
    if frame0:
        retention.append(
            "<Picture 1> ([Shot 1] first frame): fully_preserved - retain its edited target "
            "appearances and composition at the start of the shot."
        )

    task = "video editing + keyframe completion" if frame0 else "video editing"
    anchor = "The shot begins from the edited first-frame anchor <Picture 1>. " if frame0 else ""
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        (
            f"summary:\n[{task}] The target video is an edited version of <Video 1>. "
            "<Subject 3> replaces the visible appearance of <Subject 1>, and <Subject 4> "
            "replaces the visible appearance of <Subject 2>. <Subject 3> and <Subject 4> "
            "follow the original performances of <Subject 1> and <Subject 2> from <Video 1>. "
            "Preserve the original action order and timing, non-person props, camera and scene."
        ),
        "retention_analysis:\n" + "\n".join(retention),
        (
            "detailed_description:\nKeep the visual style and lighting of <Video 1>.\n\n"
            "[Shot 1] " + anchor +
            "<Subject 3> appears in place of <Subject 1> and follows <Subject 1>'s original "
            "performance from <Video 1>. <Subject 4> appears in place of <Subject 2> and follows "
            "<Subject 2>'s original performance from <Video 1>. " + shot +
            " These actions and interactions reproduce the corresponding source performances "
            "from <Video 1> in the same order and timing. Keep all non-person props unchanged, "
            "including held or touched objects, and preserve the original hand-object contact, "
            "camera, background, lighting and composition. Keep <Subject 3> on <Subject 1>'s "
            "source track and <Subject 4> on <Subject 2>'s source track throughout crossings, "
            "overlap or occlusion; do not swap the two target subjects."
        ),
        (
            "overall_soundscape:\nPreserve source synchronized sound, ambience and physical "
            "sounds; do not invent new effects or voices because the appearances change."
        ),
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
