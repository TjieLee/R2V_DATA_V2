"""Deterministic local Ref2VA prompt; no additional model call."""


def build_h3_ref2va_replacement_prompt(source_person: str, replacement_person: str) -> str:
    if not source_person.strip() or not replacement_person.strip():
        raise ValueError("Both person descriptions must be nonempty")
    return f"""subject_definitions:
<Subject 1> is the original person in <Video 1>, described as {source_person}.
<Video 1> is the source video for the target video edit.

summary:
[video editing] The target video is an edited version of <Video 1>.
Replace <Subject 1> with {replacement_person}.
Only the person's identity and visible appearance may change.

retention_analysis:
<Video 1> - fully preserve complete temporal structure, cuts, camera motion, framing,
crop, background, foreground objects, lighting, shadows, reflections, scene geometry,
composition, and non-human regions.
<Subject 1> - replace identity and visible appearance, but fully preserve frame-by-frame
pose sequence, body orientation, limb positions, hand placement, hand-body contact,
hand-object contact, holding / pocket / touching states, foot placement, head pose,
expression, gaze, mouth state, spatial position, scale, occlusion relationships,
and exact timing of movement.

detailed_description:
[Shot 1] Treat <Video 1> as the direct source video.
Reproduce the same scene and performance along the same temporal path.
Replace only <Subject 1>'s identity and visible appearance with {replacement_person}.
Do not reinterpret or simplify the action. Do not redesign clothing/pose in a way
that changes motion/contact. All background/object/camera content remains unchanged.

overall_soundscape:
N/A

non_diegetic_music:
N/A"""
