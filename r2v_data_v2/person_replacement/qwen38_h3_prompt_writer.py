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
    for token in ("<Subject 1>","<Subject 2>","<Video 1>"):
        if token not in detailed:
            raise ValueError(f"detailed_description is missing {token}")
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
        return self._generate([
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
