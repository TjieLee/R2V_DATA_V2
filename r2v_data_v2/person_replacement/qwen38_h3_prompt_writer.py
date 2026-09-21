"""Single-instance Qwen3.8-27B writer for the text V19 production path.

One worker loads the 27B model once and reuses it for two calls per case:
Call 1 plans replacement identities, Call 2 writes the complete MiniMax-H3
six-section prompt straight from <Video 1>. No 8B model, no service, no
ASR/audio input, and no fallback to the old template prompt.
"""

import re
from pathlib import Path

from .qwen3vl import _labelled_fields

VIDEO_FPS = 4.0
REPLACEMENT_MAX_NEW_TOKENS = 512
PROMPT_MAX_NEW_TOKENS = 4096

DETAIL_PRESERVATION_SENTENCE = (
    "Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head "
    "orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, "
    "hand-object and person-person interactions, positions, scale, occlusion, and "
    "timing exactly the same as in <Video 1>, and keep the source camera movement, "
    "framing, background, lighting, and scene structure unchanged; only their "
    "identities and appearances are replaced."
)

CALL1_LABELS = ("SOURCE_PERFORMER_1","SOURCE_PERFORMER_2",
                "REPLACEMENT_SUBJECT_1","REPLACEMENT_SUBJECT_2")

REPLACEMENT_PLANNING_PROMPT = """You are preparing two replacement identities for a two-person
MiniMax-H3 video-editing task.

The attached source video contains exactly two physical human performers.

First identify and keep track of the same two physical performers throughout
the entire video, including crossings, occlusions and changes in screen order.

Assign performer identity deterministically:
- performer 1 is the performer who appears first in time;
- if both first appear at the same time, performer 1 is the leftmost one in
  the earliest frame where both are visible;
- performer 2 is the other physical performer.
Once assigned, never swap them when they move across the frame.

For SOURCE_PERFORMER_1 and SOURCE_PERFORMER_2:
write one concise STATIC locator only.

A locator may contain:
- stable clothing
- hair
- approximate apparent age
- stable appearance
- initial foreground/background or left/right location

Do not put:
- actions
- pose changes
- gaze
- expression/emotion
- speaking
- temporal sequence
- then/later/turning/smiling/etc.

Now invent one clearly different realistic replacement person for each source
performer.

Replacement 1 diversity cue:
{cue1}

Replacement 2 diversity cue:
{cue2}

Each diversity cue affects appearance and clothing only.

Do not introduce:
- props
- tools
- weapons
- handheld objects
- actions
- gestures
- pose
- interactions

Each replacement should be noticeably different from its corresponding source
performer, not just a lightly modified lookalike.

CASTING VARIATION

The source performer's apparent gender presentation and approximate age are
identification facts, not attributes that need to be preserved in the
replacement.

Do not default to the same gender presentation or a nearby age band merely
because the source performer has them. Same-gender or similar-age casting is
allowed, but it should be only one possible outcome rather than the default.

For the pair of two replacements, when physically plausible and not contradicted
by an explicitly gendered diversity cue, prefer:
- at least one replacement whose apparent gender presentation differs from its
  corresponding source performer;
- at least one replacement whose apparent adult age band differs clearly from
  its corresponding source performer.

These two differences may occur in the same replacement or in different
replacements. Keep age changes realistic: do not introduce a child merely to
create contrast.

Body-scale compatibility means approximate occupied height/build volume for
motion transfer. It does NOT mean preserving gender presentation, age, face,
hair, grooming or clothing.

When reasonable, create difference across at least two or three of the
following aspects:
- apparent age band,
- hairstyle, hair texture or hair colour,
- face shape or facial-structure impression,
- grooming or facial hair,
- clothing style,
- overall colour palette,
- general visual styling archetype.

However, keep the result realistic and plausible.
Do not make the replacement bizarre, caricatured or unnaturally extreme.
Do not request a radically different body size or silhouette.
Keep the replacement person broadly compatible with the source performer's body
scale and spatial occupancy so the source motion can be preserved cleanly.

The diversity cue must not be satisfied only by a clothing change.
The replacement should also differ in facial appearance and overall styling
from the source performer.

Each REPLACEMENT_SUBJECT line should be compact but identity-bearing. When
visually plausible, include:
- an apparent age band,
- a hair/head appearance,
- a face-shape or grooming detail,
- and a clothing/style detail with at least one useful colour or material cue.

Optional diversity dimensions may also include a broad country/region or
ethnic-background cue, apparent skin tone, or profession-inspired styling.
These are optional appearance cues, not required fields. Do not force all of
them into every replacement.

Do not use empty identity phrases such as "a generic man", "a generic woman",
"a generic person", or "an ordinary person" as the replacement description.
The replacement must read like a specific new casting choice, not a vague
placeholder.

FEW-SHOT IDENTITY EXAMPLES

These examples demonstrate the desired identity distance and description
density only. Do not copy their attributes unless the current video and
diversity cue support them.

Example A — gender, age and skin-tone contrast:
SOURCE_PERFORMER_1: a young adult man with short black hair and a light grey hoodie, initially on the left
SOURCE_PERFORMER_2: a young adult woman with long straight dark hair and a pale blouse, initially on the right
REPLACEMENT_SUBJECT_1: a Black British woman in her forties with deep brown skin, short natural coily hair, high cheekbones, and a dark teal office-professional knit top
REPLACEMENT_SUBJECT_2: an older East Asian man in his sixties with light-medium skin, swept-back silver hair, a rectangular face, a neatly trimmed moustache, and a burgundy casual button-up shirt

Example B — regional background and profession styling:
SOURCE_PERFORMER_1: an older man with thinning grey hair and a dark jacket, initially nearer the camera
SOURCE_PERFORMER_2: a young woman with shoulder-length dark hair and a plain top, initially farther back
REPLACEMENT_SUBJECT_1: a young South Asian woman with medium-brown skin, a cropped black pixie cut, a round face, and a fitted navy pharmacist-style work jacket
REPLACEMENT_SUBJECT_2: a middle-aged white Eastern European man with fair skin, wavy dark-brown hair, a broad oval face, a short beard, and a muted rust polo shirt

Example C — keep some dimensions ordinary while changing others:
SOURCE_PERFORMER_1: a middle-aged woman with straight dark hair and a beige sweater, initially in the foreground
SOURCE_PERFORMER_2: a young adult man with short dark hair and a black t-shirt, initially behind her
REPLACEMENT_SUBJECT_1: a young Southeast Asian man with warm tan skin, short textured hair, a lean oval face, and simple dark barista-style workwear
REPLACEMENT_SUBJECT_2: a Latina woman in her fifties with medium olive skin, shoulder-length wavy brown hair, a square face, and a muted blue everyday blouse

The examples intentionally use different subsets of age, gender presentation,
broad regional or ethnic-background cue, skin tone, profession-inspired styling,
hair, face and clothing. Do not force every replacement to mention every one of
these dimensions. Use whichever subset makes the new identity distinct,
plausible and concise.

Keep the replacement person's overall height/body scale and silhouette broadly
compatible with the corresponding source performer so source pose and spatial
occupancy can be preserved.

The two replacements must:
- be clearly different from their corresponding source performers
- be clearly distinguishable from each other
- remain realistic humans
- not be celebrities

Return exactly four nonempty single-line fields:

SOURCE_PERFORMER_1: ...
SOURCE_PERFORMER_2: ...
REPLACEMENT_SUBJECT_1: ...
REPLACEMENT_SUBJECT_2: ...

No explanation.
No markdown."""

H3_PROMPT_SYSTEM_PROMPT = """You are a production prompt writer for MiniMax-H3 Ref2VA video editing.

You receive:
1. the complete source video <Video 1>;
2. a static locator for source performer 1;
3. a static locator for source performer 2;
4. the required replacement appearance for <Subject 1>;
5. the required replacement appearance for <Subject 2>.

The source video contains exactly two physical human performers.

Your task is to watch the complete source video and write the FINAL
production-ready MiniMax-H3 prompt.

SUBJECT SEMANTICS

<Subject 1> is the replacement person for source performer 1.

<Subject 2> is the replacement person for source performer 2.

<Subject 1> and <Subject 2> are the two human subjects that must appear in the
final generated video.

The original source performers are NOT additional Subject labels.

Never define <Subject 3> or <Subject 4>.

<Video 1> is the source video providing the complete motion, facial performance,
camera, scene, object and temporal structure.

Use the supplied source-performer locators only to bind each replacement to the
correct physical source performer.

Keep this mapping fixed through crossings, occlusions and screen-order changes.

Do not swap identities when the performers move left or right.

APPEARANCE REPLACEMENT

Replace each source performer's visible human identity and appearance from the
first visible frame through the final visible frame.

The source performer contributes performance and spatial behavior, not identity.

The replacement identity must inherit the source performer's:
- body pose
- body motion
- head motion
- facial expression
- gaze
- mouth movement
- hand motion
- hand-object interactions
- person-person interactions
- position
- scale
- orientation
- occlusion
- timing

These motions and expressions must be performed by the replacement identity.

Do not retain the original source face, hair or clothing when they conflict with
the replacement appearance.

Do not create additional replacement people.

Do not add props, tools or objects that are not already present in <Video 1>,
except ordinary non-interactive parts of clothing.

GLOBAL DETAILED-DESCRIPTION PREFACE

The first sentence immediately after the `detailed_description:` heading is a
mandatory preservation statement. Write this exact sentence first:

Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, hand-object and person-person interactions, positions, scale, occlusion, and timing exactly the same as in <Video 1>, and keep the source camera movement, framing, background, lighting, and scene structure unchanged; only their identities and appearances are replaced.

After that fixed preservation sentence and before `[Shot 1]`, write exactly one
source-grounded dynamic overview sentence.

The dynamic overview must summarize the dominant visible motion of <Subject 1>
and <Subject 2>, their relative movement or most important interaction when one
is visible, and the dominant camera behavior across the source clip.

The preservation sentence states the editing constraint. The following dynamic
overview and [Shot] narration describe the observed source performance in
concrete target-video terms. Do not reinterpret, embellish, intensify, simplify,
or creatively restage that performance.

Prioritize concrete dynamic information such as subject trajectories,
body/head/hand movement, interaction progression, camera tracking, panning,
tilting, zooming, reframing, handheld movement, or cuts when those are actually
visible. If the source camera is genuinely static, say that the framing remains
static instead of inventing camera movement.

Do not use the dynamic overview for static appearance, clothing, background,
lighting, or style description.

FIRST-SENTENCE PRIORITY

The first sentence inside [Shot 1] is especially important.

It must immediately describe the dominant visible motion and the dominant camera
behavior at the start of the clip.

Whenever supported by <Video 1>, the first sentence should jointly state:
- what <Subject 1> and <Subject 2> are doing,
- how they are moving relative to each other,
- the most important interaction between them,
- and the main camera behavior (static, panning, tracking, tilting, zooming,
  reframing, handheld, or cutting).

Do not begin [Shot 1] with static clothing, static background description,
or generic wording such as "preserve the original motion".

OBSERVATION-ONLY LANGUAGE

Describe visible motion, pose, facial configuration, gaze, mouth movement,
interaction, spatial relations and camera behavior. Do not infer invisible
causes, intentions, powers, hidden objects or psychological explanations.

Avoid speculative phrases such as:
- "as if"
- "seems to"
- "appears to be trying to"
- "suggesting that"

For example, if a hand extends into empty space, describe the hand extension and
finger configuration. Do not invent an invisible force, unseen object or motive.
Prefer visible facial facts such as a smile, lowered gaze, tightened lips or
furrowed brows over inferred inner states.

FACIAL AND HEAD MICRO-MOTION PRIORITY

For close-up, medium-close, dialogue-like, reaction, or otherwise face-dominant
shots, facial and head motion is higher priority than static background,
lighting, clothing, or generic emotion labels.

For each clearly visible subject, describe the observable sequence when it is
supported by <Video 1>:
- initial torso and shoulder orientation when relevant,
- initial head orientation: frontal, three-quarter left/right, near-profile, or profile,
- whether the head turns and in which direction,
- chin raised, level, or lowered and any visible change,
- gaze direction and any gaze shift,
- eyelid/blink or brow change when salient,
- initial mouth state: closed, slightly parted, or open,
- visible lip and lower-jaw movement over time,
- whether the mouth closes, opens, parts briefly, or returns to its earlier state,
- the final visible head/gaze/mouth state,
- the timing of these changes relative to the other subject and to body/hand motion.

Do not collapse these observations into only "shocked", "sad", "concerned",
"angry", "stunned", "crying", "speaking" or another emotion/action label.
If an emotion label is useful, first state the visible facial configuration that
supports it.

Do not infer spoken words or phoneme-level lip sync. The sampled video may not
resolve every rapid mouth movement, so stay conservative when a transition is
not clearly visible. But when a mouth, jaw, gaze, or head transition is visible,
state its initial state, transition, and resulting state explicitly.

FEW-SHOT DETAILED-DESCRIPTION STYLE EXAMPLES

These examples demonstrate structure and emphasis only. Never copy their
motion, camera behavior, scene or interaction unless it is actually visible in
the current <Video 1>.

Example A — face-dominant static close-up:

detailed_description:
Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, hand-object and person-person interactions, positions, scale, occlusion, and timing exactly the same as in <Video 1>, and keep the source camera movement, framing, background, lighting, and scene structure unchanged; only their identities and appearances are replaced.
<Subject 1> holds a three-quarter-right head orientation toward <Subject 2>, keeping her eyes on him as her initially parted lips gradually come together, while <Subject 2> remains in near left profile, briefly lowers his gaze and opens then closes his mouth; the camera holds a static tight two-shot throughout.

[Shot 1]
<Subject 1> sits on the left with her shoulders mostly forward and her head rotated to the right into a three-quarter view. Her chin begins slightly lowered, her eyes stay directed toward <Subject 2>, and her lips are initially separated with the lower jaw slightly dropped. She keeps the same head direction while the jaw rises gradually and the lips meet, with her gaze remaining on <Subject 2>. <Subject 2> sits on the right in near left profile, with his nose and chin pointing toward the left side of frame. His head position stays nearly fixed while his eyes shift slightly downward. His mouth begins closed, the lips part briefly as the lower jaw drops, and the mouth returns to a closed state. The camera remains completely static with no pan, tilt, zoom, reframing, or cut.

Example B — subtle head, gaze and mouth transitions:

detailed_description:
Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, hand-object and person-person interactions, positions, scale, occlusion, and timing exactly the same as in <Video 1>, and keep the source camera movement, framing, background, lighting, and scene structure unchanged; only their identities and appearances are replaced.
<Subject 1> starts nearly frontal, shifts her gaze toward <Subject 2> before turning her head slightly right and parting her lips, while <Subject 2> raises his chin and closes his previously open mouth; the camera remains static.

[Shot 1]
<Subject 1> begins with her face almost frontal and her mouth closed. Her eyes move first toward <Subject 2>; after the gaze shift, her head rotates slightly to the right into a shallow three-quarter view. Her lips then separate and the lower jaw lowers a little while the head holds the new angle. <Subject 2> begins facing slightly left with his chin lowered and mouth open. He gradually raises his chin, keeps his eyes on <Subject 1>, and brings the lower jaw upward until the lips meet. Neither subject changes seat or body position, and the camera keeps the same framing throughout.

Example C — body motion with a moving camera:

detailed_description:
Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, hand-object and person-person interactions, positions, scale, occlusion, and timing exactly the same as in <Video 1>, and keep the source camera movement, framing, background, lighting, and scene structure unchanged; only their identities and appearances are replaced.
<Subject 1> and <Subject 2> walk forward side by side while briefly turning their heads toward each other and exchanging hand gestures, as the camera tracks backward with them and gradually tightens the two-shot.

[Shot 1]
<Subject 1> and <Subject 2> advance together from the mid-ground toward the foreground while the camera tracks backward at their pace, keeping both subjects centered. <Subject 1> turns his head toward <Subject 2>, briefly parts his lips and lifts one hand during the exchange; <Subject 2> shifts her gaze toward <Subject 1> before making a small hand motion while continuing forward. Their heads return closer to the walking direction as the camera slowly reframes them more tightly without introducing a cut.

DETAILED DESCRIPTION

The detailed_description is the most important section.

It must be a COMPLETE POSITIVE DESCRIPTION OF THE TARGET VIDEO.

It must NOT merely be a list of preservation constraints.

Watch the complete <Video 1> and narrate what the final edited video visibly
contains from beginning to end.

Use <Subject 1> and <Subject 2> in place of the original performers when
describing the source performance.

Actually describe the visible action and event sequence.

When relevant, explicitly describe:
- opening composition
- relative placement of <Subject 1> and <Subject 2>
- foreground/background placement
- body pose and orientation
- action progression
- facial-expression changes
- gaze changes
- visible mouth movement / visible speaking behavior
- hand and arm motion
- hand-object interactions
- person-person interactions
- important source objects
- crossings
- occlusions
- scene state changes
- camera framing
- camera movement
- camera cuts
- background
- scene structure
- event progression

Do not replace a real visible sequence with vague text such as:
"preserve the original actions"
or
"follow the source motion".

Describe what visibly happens.

Do not describe the original source performers' face, hair or clothing as part
of the final target appearance.

Use the required replacement appearance instead.

Do not invent unsupported:
- actions
- events
- people
- objects
- cinematic styling
- lens type
- focal length
- depth-of-field changes
- camera movement
- lighting changes
- background details

Preserve the actual visual world and camera behavior in <Video 1> rather than
creatively redesigning it.

If uncertain about a visual detail, stay conservative.

SHOTS

If <Video 1> is one continuous source shot, use only:

[Shot 1]

If actual source camera cuts exist, use:

[Shot 1]
[Shot 2]
...

in playback order.

Do not invent cuts.

Only include exact timestamps when clearly reliable.
Do not fabricate precise timestamps.

AUDIO

Ignore source audio completely.

Do not:
- transcribe dialogue
- quote speech
- infer dialogue text from mouth motion
- identify speakers
- describe voices
- describe sound effects
- describe music

Visible mouth movement may be described only as visual behavior.

OUTPUT FORMAT

Output exactly these six sections and nothing else:

subject_definitions:
...

summary:
...

retention_analysis:
...

detailed_description:
...

overall_soundscape:
N/A

non_diegetic_music:
N/A

subject_definitions style:

<Subject 1>: The replacement principal subject corresponding to source performer
1. [replacement appearance]

<Subject 2>: The replacement principal subject corresponding to source performer
2. [replacement appearance]

<Video 1>: The source video providing the complete motion, facial performance,
camera, scene, object and temporal structure.

summary style:

[video editing + reference generation] Replace the two original performers
throughout <Video 1> with <Subject 1> and <Subject 2>, beginning from their first
visible frames and continuing through the final frame. Follow the complete
source performance and shot structure while changing their identities and
appearances.

retention_analysis style:

<Subject 1>: replace - use the replacement identity and appearance defined above
throughout the clip while inheriting the corresponding source performer's
performance and timing.

<Subject 2>: replace - use the replacement identity and appearance defined above
throughout the clip while inheriting the corresponding source performer's
performance and timing.

<Video 1>: preserve - preserve the complete action sequence, positions, object
interactions, shot timing, camera movement, framing, background, lighting and
scene structure.

Do NOT use:
<Subject 1> (appears in [Shot 1])
or attribute_transfer syntax.

Do not define source performers as separate Subjects.

No markdown fences.
No explanation before or after the prompt."""

H3_PROMPT_USER_TEMPLATE = """Generate the final MiniMax-H3 Ref2VA prompt for the attached <Video 1>.

Source performer 1 locator:
{source_performer_1}

Required appearance for <Subject 1>:
{replacement_subject_1}

Source performer 2 locator:
{source_performer_2}

Required appearance for <Subject 2>:
{replacement_subject_2}

The locator descriptions are only for identifying and binding the corresponding
physical source performer.

Do not preserve source appearance attributes from the locator when they conflict
with the required replacement appearance.

Watch the complete <Video 1> and describe the target video fully from beginning
to end.

Output only the final six-section MiniMax-H3 prompt."""

SECTION_HEADINGS = ("subject_definitions","summary","retention_analysis",
                    "detailed_description","overall_soundscape","non_diegetic_music")
_HEADING = re.compile(r"(?m)^([a-z_]+):[ \t]*$")
_EXTRA_SUBJECT = re.compile(r"<Subject\s+(?:[3-9]|\d{2,})\s*>")
_ANY_AUDIO_REFERENCE = re.compile(r"<Audio\s*\d*\s*>")


def split_sections(text):
    """Ordered (heading, body) pairs for top-level `heading:` lines."""
    matches = list(_HEADING.finditer(text))
    sections = []
    for index, match in enumerate(matches):
        end = matches[index+1].start() if index+1 < len(matches) else len(text)
        sections.append((match.group(1), text[match.end():end]))
    return sections


def enforce_detail_preservation_preface(text):
    """Guarantee the H3 detailed description starts with the canonical constraint."""
    if not isinstance(text,str):
        return text
    match = re.search(r"(?m)^detailed_description:[ \t]*$",text)
    if match is None:
        return text
    body_start = match.end()
    next_match = re.search(r"(?m)^overall_soundscape:[ \t]*$",text[body_start:])
    if next_match is None:
        return text
    body_end = body_start + next_match.start()
    body = text[body_start:body_end].strip()
    if body.startswith(DETAIL_PRESERVATION_SENTENCE):
        return text.strip()
    replacement = (
        "\n" + DETAIL_PRESERVATION_SENTENCE + "\n\n" + body + "\n\n"
    )
    return (text[:body_start] + replacement + text[body_end:]).strip()


def validate_h3_prompt_writer_output(text):
    """Fail-closed structural check; never a semantic parser."""
    if not isinstance(text,str) or not text.strip():
        raise ValueError("H3 prompt writer returned no text")
    text = text.strip()
    # A reasoning prefix must never reach H3; fail closed instead.
    if "<think>" in text or "</think>" in text:
        raise ValueError("H3 prompt contains thinking markers; enable_thinking must be off")
    if not text.startswith("subject_definitions:"):
        raise ValueError("H3 prompt must start with subject_definitions:, got leading prose")
    sections = split_sections(text)
    headings = [heading for heading,_ in sections]
    if headings != list(SECTION_HEADINGS):
        raise ValueError(f"H3 prompt must have exactly these ordered sections "
                         f"{list(SECTION_HEADINGS)}, got {headings}")
    bodies = {heading: body.strip() for heading, body in sections}
    for heading in SECTION_HEADINGS[:4]:
        if not bodies[heading]:
            raise ValueError(f"H3 prompt section is empty: {heading}")
    for heading in SECTION_HEADINGS[4:]:
        if bodies[heading] != "N/A":
            raise ValueError(f"H3 prompt section {heading} must be exactly N/A, "
                             f"got {bodies[heading]!r}")
    for token in ("<Subject 1>","<Subject 2>","<Video 1>"):
        if token not in text:
            raise ValueError(f"H3 prompt is missing {token}")
    if _EXTRA_SUBJECT.search(text):
        raise ValueError("H3 prompt must not define <Subject 3> or higher")
    if _ANY_AUDIO_REFERENCE.search(text):
        raise ValueError("H3 prompt must not reference <Audio 1> or <Audio 2>")
    if "attribute_transfer" in text:
        raise ValueError("H3 prompt must not use attribute_transfer syntax")
    if "(appears in [Shot" in text:
        raise ValueError("H3 prompt must not use '(appears in [Shot ...])' syntax")
    detailed = bodies["detailed_description"]
    # The detailed description is target-video narration, so it must name both
    # replacement subjects. A literal <Video 1> token is not required here:
    # the source-video binding is already enforced globally by subject_definitions
    # / summary / retention_analysis, and forcing the token into narration is a
    # formatting preference rather than a semantic correctness condition.
    for token in ("<Subject 1>","<Subject 2>"):
        if token not in detailed:
            raise ValueError(f"detailed_description is missing {token}")
    if "[Shot 1]" not in detailed:
        raise ValueError("detailed_description must contain [Shot 1]")
    preface = detailed.split("[Shot 1]",1)[0].strip()
    if not preface.startswith(DETAIL_PRESERVATION_SENTENCE):
        raise ValueError("detailed_description must start with the canonical preservation sentence")
    dynamic_preface = preface[len(DETAIL_PRESERVATION_SENTENCE):].strip()
    if not dynamic_preface:
        raise ValueError("detailed_description must contain a dynamic overview after the preservation sentence")
    for token in ("<Subject 1>","<Subject 2>"):
        if token not in dynamic_preface:
            raise ValueError(f"dynamic overview is missing {token}")
    if not re.search(r"\b(?:camera|framing|shot|view)\b",dynamic_preface,re.IGNORECASE):
        raise ValueError("dynamic overview must describe camera or framing behavior")
    speculative = re.search(
        r"\b(?:as if|seems to|appears to be trying to|suggesting that|invisible force)\b",
        detailed,
        re.IGNORECASE,
    )
    if speculative:
        raise ValueError(
            f"detailed_description uses speculative language: {speculative.group(0)!r}"
        )
    return text.strip()


class LocalQwen38H3PromptWriter:
    """One resident Qwen3.8-27B instance for a whole worker partition."""

    def __init__(self, model_path, *, video_fps=VIDEO_FPS,
                 replacement_max_new_tokens=REPLACEMENT_MAX_NEW_TOKENS,
                 prompt_max_new_tokens=PROMPT_MAX_NEW_TOKENS):
        self.model_path = Path(model_path).expanduser()
        self.video_fps = video_fps
        self.replacement_max_new_tokens = replacement_max_new_tokens
        self.prompt_max_new_tokens = prompt_max_new_tokens
        self.model = self.processor = None
        self.thinking_disabled = True  # flipped when the template cannot disable it

    def _load(self):
        if self.model is not None:
            return
        if not self.model_path.is_dir():
            raise ValueError(f"Qwen3.8 checkpoint must be a local directory: {self.model_path}")
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Text V19 needs a Qwen3.8-capable Transformers runtime with "
                "AutoModelForMultimodalLM. Use the isolated server environment; "
                "no automatic installation and no fallback model."
            ) from exc
        # local_files_only: never download. dtype is left to the FP8 checkpoint.
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True,
        )
        # Qwen video processors carry a default fps (commonly 2). Transformers
        # treats that default as an active sampling argument, so passing our
        # explicit num_frames at chat-template time otherwise triggers the
        # mutually-exclusive num_frames+fps guard. Disable only the processor
        # default; num_frames remains the sole bounded sampling control.
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None and hasattr(video_processor, "fps"):
            video_processor.fps = None
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(self.model_path), local_files_only=True, device_map="auto",
            # Fine-grained FP8 on Hopper may lazily load the official
            # kernels-community/deep-gemm kernel package. Transformers gates
            # that executable kernel code behind the remote-code trust flag.
            trust_remote_code=True,
        ).eval()

    def _apply_template(self, messages, *, num_frames):
        """num_frames sampling is mandatory; only enable_thinking may be dropped."""
        kwargs = {"tokenize":True, "add_generation_prompt":True,
                  "return_dict":True, "return_tensors":"pt"}
        try:
            return self.processor.apply_chat_template(
                messages, **kwargs, num_frames=num_frames, enable_thinking=False)
        except TypeError as exc:
            if "num_frames" in str(exc):
                # Never silently decode the whole video.
                raise TypeError(
                    "Qwen3.8 processor does not support num_frames video sampling; "
                    "refusing to run without bounded frame sampling"
                ) from exc
            self.thinking_disabled = False  # honest; the validator then guards output
            return self.processor.apply_chat_template(messages, **kwargs, num_frames=num_frames)

    def _generate(self, messages, max_new_tokens, *, num_frames):
        self._load()
        import torch

        inputs = self._apply_template(messages, num_frames=num_frames).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated, strict=True)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise ValueError("Qwen3.8 returned an empty completion")
        return text

    def _video(self, video):
        # Transformers multimodal chat templates take a local file in `path`;
        # `video` is reserved for an already decoded video object/array.
        return {"type":"video", "path":str(Path(video).resolve(strict=True))}

    def invent_replacements(self, video, cue1, cue2, *, num_frames):
        """Call 1: bind the two performers and plan replacement identities."""
        text = self._generate([
            {"role":"user", "content":[
                self._video(video),
                {"type":"text", "text":REPLACEMENT_PLANNING_PROMPT.format(cue1=cue1, cue2=cue2)},
            ]},
        ], self.replacement_max_new_tokens, num_frames=num_frames)
        return _labelled_fields(text, CALL1_LABELS)

    def write_h3_prompt(self, video, source_performer_1, source_performer_2,
                        replacement_subject_1, replacement_subject_2, *, num_frames):
        """Call 2: watch <Video 1> again and write the whole six-section prompt."""
        raw = self._generate([
            {"role":"system", "content":H3_PROMPT_SYSTEM_PROMPT},
            {"role":"user", "content":[
                self._video(video),
                {"type":"text", "text":H3_PROMPT_USER_TEMPLATE.format(
                    source_performer_1=source_performer_1,
                    replacement_subject_1=replacement_subject_1,
                    source_performer_2=source_performer_2,
                    replacement_subject_2=replacement_subject_2)},
            ]},
        ], self.prompt_max_new_tokens, num_frames=num_frames)
        return enforce_detail_preservation_preface(raw)

    def close(self):
        """Release the 27B before the H3 launcher consumes GPU memory."""
        if self.model is None:
            return
        import gc

        import torch

        self.model = self.processor = None
        gc.collect()
        if hasattr(torch,"cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
