"""Single-instance Qwen3.8-27B writer for the text V19 production path.

One worker loads the 27B model once and reuses it for two calls per case:
Call 1 plans replacement identities, Call 2 writes the complete MiniMax-H3
six-section prompt straight from <Video 1>. No 8B model, no service, no
ASR/audio input, and no fallback to the old template prompt.
"""

import re
from pathlib import Path

from .qwen3vl import _labelled_fields

VIDEO_FPS = 8.0
REPLACEMENT_MAX_NEW_TOKENS = 512
PROMPT_MAX_NEW_TOKENS = 4096

QWEN38_SGLANG_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN38_SGLANG_SERVED_MODEL = "Qwen/Qwen3.8-Flash-Next"
QWEN38_SGLANG_CHECKPOINT = "/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next"
QWEN38_TEMPERATURE = 0.7
QWEN38_TOP_P = 0.8
QWEN38_TOP_K = 20
QWEN38_MIN_P = 0.0
QWEN38_PRESENCE_PENALTY = 1.5
QWEN38_REPETITION_PENALTY = 1.0

MIMO_SGLANG_BASE_URL = "http://127.0.0.1:8092/v1"
MIMO_SGLANG_SERVED_MODEL = "mimo-v2.5"
MIMO_CHECKPOINT = "/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5"
MIMO_VIDEO_FPS = 8.0
MIMO_MEDIA_RESOLUTION = "default"
MIMO_TEMPERATURE = 0.0

DETAIL_PRESERVATION_SENTENCE = (
    "Only replace <Subject 1> and <Subject 2>'s identities and appearances; keep "
    "<Video 1>'s motion, facial performance, interactions, and timing unchanged."
)
DETAIL_CAMERA_SENTENCE = (
    "Keep <Video 1>'s camera, framing, scene, lighting, and non-person objects unchanged."
)
DETAIL_PRESERVATION_PREFIX = DETAIL_PRESERVATION_SENTENCE + " " + DETAIL_CAMERA_SENTENCE

CALL1_LABELS = ("SOURCE_PERFORMER_1","SOURCE_PERFORMER_2",
                "REPLACEMENT_SUBJECT_1","REPLACEMENT_SUBJECT_2")

REPLACEMENT_PLANNING_PROMPT = """You are preparing two replacement identities for a two-person
MiniMax-H3 video-editing task.

Track exactly two physical performers through the whole source video. Assign
performer 1 to the person who appears first; if both first appear together,
performer 1 is the leftmost person in the earliest frame where both are visible.
Performer 2 is the other person. Never swap them after crossings or occlusion.

For SOURCE_PERFORMER_1 and SOURCE_PERFORMER_2, output one concise STATIC locator
using stable appearance such as clothing, hair, approximate age, and initial
left/right or foreground/background position. Do not include action, pose, gaze,
expression, speaking, or temporal wording.

Invent one realistic replacement identity for each source person.

Replacement 1 diversity cue:
{cue1}

Replacement 2 diversity cue:
{cue2}

The replacement identity changes appearance only. Do not add props, actions,
gestures, pose, or interactions.

Make each replacement clearly different from its source and from the other
replacement. Source gender presentation and approximate age are identification
facts, not preservation targets. When plausible for the pair, include at least
one cross-gender replacement and at least one clearly different adult age band;
these may occur in the same replacement.

Create meaningful identity distance using several of: age, gender presentation,
face shape, hair, grooming, skin tone, clothing/style, or overall visual
archetype. Optional broad regional/ethnic-background or profession-inspired
styling may be used when it helps diversity. Do not satisfy diversity only by
changing clothes.

Keep body scale and occupied silhouette broadly compatible with the source so
motion transfer remains clean. Keep every replacement realistic, non-celebrity,
and compact. A good REPLACEMENT_SUBJECT line usually contains age, hair/head,
one face/grooming trait, and one clothing colour/material cue.

Return exactly four nonempty single-line fields:

SOURCE_PERFORMER_1: ...
SOURCE_PERFORMER_2: ...
REPLACEMENT_SUBJECT_1: ...
REPLACEMENT_SUBJECT_2: ...

No explanation. No markdown."""

H3_PROMPT_SYSTEM_PROMPT = """You write the final MiniMax-H3 Ref2VA prompt for a two-person identity-replacement edit.

CORE TASK
- <Subject 1> and <Subject 2> are replacement identities.
- Each is bound to exactly one source person by the supplied static locator.
- Keep those two bindings fixed through movement, overlap, crossing, and occlusion.
- <Video 1> is the authority for all motion, facial performance, interaction,
  timing, camera behavior, scene, lighting, objects, and shot structure.
- Change who the two people are; do not change what happens.

APPEARANCE
Use the requested replacement appearance from each subject's first visible frame
through the end. Do not carry over conflicting source face, hair, grooming, or
clothing. Do not add people, props, or objects.

DETAILED DESCRIPTION
Keep this section concise and source-grounded. Do not repeat replacement
appearance, static clothing, or unchanged background here; those details already
belong in subject_definitions or <Video 1>.

After the fixed preservation prefix, write [Shot 1] directly (and later shots
only if real source cuts exist). Do not add a separate overview paragraph unless
it contributes information not already stated in the shot narration.

The [Shot 1] narration must explicitly use both labels <Subject 1> and
<Subject 2>. Do not replace these labels with "the man", "the woman", "the left
person", "the other person", or other unnamed references when first describing
their visible actions.

Describe only meaningful visible action/state changes in playback order. Include
head direction, gaze, mouth/jaw motion, hand motion, interaction, or occlusion
when they are salient to the source performance. Do not enumerate every blink,
micro-expression, frame, or opening/middle/final state just to make the caption
longer.

If a body/head/gaze state visibly changes, describe the change. Use words such
as "throughout", "maintains", or "remains" only when the state truly stays
constant across the observed interval.

Describe camera movement or cuts when they occur; say the camera is static when
it is genuinely static. Do not invent intent, psychology, hidden causes,
unsupported objects, cinematic styling, lens details, or camera behavior.
If uncertain, stay conservative.

COMPACT STYLE EXAMPLE

This example shows naming and density only. Do not copy its motion unless it is
actually visible in the current video.

detailed_description:
[Shot 1]
<Subject 1> sits on the left and turns the head from <Subject 2> toward the
front while <Subject 2> remains seated on the right with only a small gaze
change; the camera stays static. <Subject 1> then holds the new orientation while
<Subject 2> keeps nearly the same pose.

Do not transcribe or describe source audio. Visible mouth movement may be
described only as visual behavior.

OUTPUT
Return exactly these six sections in this order and nothing else:

subject_definitions:
<Subject 1>: [replacement appearance]. Replaces only the source person identified by [locator].
<Subject 2>: [replacement appearance]. Replaces only the source person identified by [locator].
<Video 1>: source video and authority for non-identity content.

summary:
[one concise editing summary]

retention_analysis:
[three concise lines for Subject 1, Subject 2, and Video 1]

detailed_description:
[Shot 1]
[concise chronological visible action/camera description]

overall_soundscape:
N/A

non_diegetic_music:
N/A

Put replacement appearance before its locator. Never define <Subject 3> or
higher. Never use <Audio N>, attribute_transfer, or "(appears in [Shot ...])".
The final response must end with the two exact N/A sections above.
No markdown fences and no explanation."""

H3_PROMPT_USER_TEMPLATE = """Generate the final six-section MiniMax-H3 prompt for the attached <Video 1>.

<Subject 1> appearance:
{replacement_subject_1}
Source-person locator:
{source_performer_1}

<Subject 2> appearance:
{replacement_subject_2}
Source-person locator:
{source_performer_2}

Keep each binding fixed. Watch the complete video. In detailed_description,
write [Shot 1] directly after the fixed preservation prefix. In the [Shot 1]
narration explicitly use both <Subject 1> and <Subject 2>, then describe only
salient visible motion/state changes and camera behavior in temporal order; do
not repeat appearance or pad the caption with micro-events.

End exactly with:
overall_soundscape:
N/A

non_diegetic_music:
N/A"""

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


def _clean_binding_text(value):
    return " ".join(str(value).split()).strip().rstrip(".")


def source_binding_sentence(source_performer_1, source_performer_2):
    first = _clean_binding_text(source_performer_1)
    second = _clean_binding_text(source_performer_2)
    return (
        "Binding is fixed throughout <Video 1>: <Subject 1> replaces only the source "
        f"person identified by {first}; <Subject 2> replaces only the source person "
        f"identified by {second}. Never swap these mappings when the people move, "
        "cross, overlap, become occluded, or change screen order."
    )


def _rebuild_sections(text, replacements):
    sections = split_sections(text)
    if [heading for heading,_ in sections] != list(SECTION_HEADINGS):
        return text
    bodies = {heading: body.strip() for heading,body in sections}
    bodies.update(replacements)
    return "\n\n".join(
        f"{heading}:\n{bodies[heading].strip()}" for heading in SECTION_HEADINGS
    ).strip()


def enforce_source_binding_sections(
    text,
    source_performer_1,
    source_performer_2,
    replacement_subject_1,
    replacement_subject_2,
):
    """Publish one strong appearance/binding statement without repeating it everywhere."""
    locator1 = _clean_binding_text(source_performer_1)
    locator2 = _clean_binding_text(source_performer_2)
    appearance1 = _clean_binding_text(replacement_subject_1)
    appearance2 = _clean_binding_text(replacement_subject_2)
    subject_definitions = (
        f"<Subject 1>: {appearance1}. Replaces only the source person identified by: {locator1}.\n"
        f"<Subject 2>: {appearance2}. Replaces only the source person identified by: {locator2}.\n"
        "<Video 1>: Source video; authority for motion, facial performance, interactions, "
        "timing, camera, scene, lighting, and non-person objects."
    )
    summary = (
        "[video editing + reference generation] Replace only the two bound source people "
        "with <Subject 1> and <Subject 2>; keep both mappings fixed and preserve the rest "
        "of <Video 1>."
    )
    retention = (
        "<Subject 1>: replace appearance only; inherit the bound source person's performance and timing.\n"
        "<Subject 2>: replace appearance only; inherit the bound source person's performance and timing.\n"
        "<Video 1>: preserve motion, interactions, timing, camera, framing, scene, lighting, "
        "and non-person objects."
    )
    return _rebuild_sections(text,{
        "subject_definitions":subject_definitions,
        "summary":summary,
        "retention_analysis":retention,
    })


def _strip_leading_preservation_prefixes(body):
    """Remove model-emitted canonical prefix copies regardless of whitespace."""
    pattern = re.compile(
        r"^" + re.escape(DETAIL_PRESERVATION_SENTENCE)
        + r"\s*" + re.escape(DETAIL_CAMERA_SENTENCE)
        + r"(?:\s*\n\s*)*",
        re.DOTALL,
    )
    value = body.strip()
    while True:
        match = pattern.match(value)
        if match is None:
            return value
        value = value[match.end():].lstrip()


def enforce_detail_preservation_preface(text, source_performer_1=None, source_performer_2=None):
    """Insert one concise preservation prefix; bindings live in subject_definitions."""
    del source_performer_1, source_performer_2
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
    tail = _strip_leading_preservation_prefixes(body)
    pieces = [DETAIL_PRESERVATION_PREFIX]
    if tail:
        pieces.append(tail)
    replacement = "\n" + "\n\n".join(pieces) + "\n\n"
    return (text[:body_start] + replacement + text[body_end:]).strip()


def validate_h3_prompt_writer_output(text):
    """Fail-closed structural check without forcing verbose narration."""
    if not isinstance(text,str) or not text.strip():
        raise ValueError("H3 prompt writer returned no text")
    text = text.strip()
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
            raise ValueError(f"H3 prompt section {heading} must be exactly N/A, got {bodies[heading]!r}")
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
    if "[Shot 1]" not in detailed:
        raise ValueError("detailed_description must contain [Shot 1]")
    preface, shot_text = detailed.split("[Shot 1]",1)
    preface = preface.strip()
    if not preface.startswith(DETAIL_PRESERVATION_PREFIX):
        raise ValueError("detailed_description must start with the canonical preservation prefix")
    # V38-slim deliberately does not require a separate dynamic-overview paragraph.
    # The actual shot narration is the useful conditioning signal.
    for token in ("<Subject 1>","<Subject 2>"):
        if token not in shot_text:
            raise ValueError(f"[Shot 1] narration is missing {token}")
    return text.strip()


class OpenAIQwen38H3PromptWriter:
    """Qwen3.8 SGLang client using the repository's validated non-thinking contract."""

    uses_local_frame_sampling = False
    thinking_disabled = True

    def __init__(
        self,
        checkpoint_id=QWEN38_SGLANG_CHECKPOINT,
        *,
        base_url=QWEN38_SGLANG_BASE_URL,
        served_model=QWEN38_SGLANG_SERVED_MODEL,
        api_key="EMPTY",
        timeout_seconds=900.0,
        replacement_max_new_tokens=REPLACEMENT_MAX_NEW_TOKENS,
        prompt_max_new_tokens=PROMPT_MAX_NEW_TOKENS,
        client=None,
    ):
        self.model_path = Path(checkpoint_id)
        self.checkpoint_id = str(checkpoint_id)
        self.base_url = str(base_url)
        self.served_model = str(served_model)
        self.api_key = str(api_key)
        self.timeout_seconds = float(timeout_seconds)
        self.replacement_max_new_tokens = replacement_max_new_tokens
        self.prompt_max_new_tokens = prompt_max_new_tokens
        self.client = client

    def _load(self):
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.8 SGLang writer needs the OpenAI client in the isolated "
                "prompt-writer environment"
            ) from exc
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _video(video):
        path = Path(video).expanduser().resolve(strict=True)
        return {"type":"video_url","video_url":{"url":path.as_uri()}}

    def _generate(self, messages, max_new_tokens, *, num_frames=None):
        # SGLang/Qwen3.8 uses its validated default video processing. The
        # repository's validated server path intentionally sends no former
        # vLLM-only fps/mm_processor override.
        del num_frames
        self._load()
        completion = self.client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            temperature=QWEN38_TEMPERATURE,
            top_p=QWEN38_TOP_P,
            presence_penalty=QWEN38_PRESENCE_PENALTY,
            max_tokens=max_new_tokens,
            stream=False,
            modalities=["text"],
            extra_body={
                "top_k":QWEN38_TOP_K,
                "min_p":QWEN38_MIN_P,
                "repetition_penalty":QWEN38_REPETITION_PENALTY,
                "chat_template_kwargs":{"enable_thinking":False},
            },
        )
        choices = getattr(completion,"choices",None)
        if not choices:
            raise ValueError("Qwen3.8 SGLang returned no choices")
        message = choices[0].message
        reasoning = getattr(message,"reasoning_content",None)
        if reasoning not in (None,""):
            raise ValueError("Qwen3.8 SGLang emitted reasoning despite enable_thinking=false")
        text = getattr(message,"content",None)
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Qwen3.8 SGLang returned an empty completion")
        return text.strip()

    def invent_replacements(self, video, cue1, cue2, *, num_frames):
        text = self._generate([
            {"role":"user","content":[
                self._video(video),
                {"type":"text","text":REPLACEMENT_PLANNING_PROMPT.format(cue1=cue1,cue2=cue2)},
            ]},
        ], self.replacement_max_new_tokens, num_frames=num_frames)
        return _labelled_fields(text,CALL1_LABELS)

    def write_h3_prompt(self, video, source_performer_1, source_performer_2,
                        replacement_subject_1, replacement_subject_2, *, num_frames):
        raw = self._generate([
            {"role":"system","content":H3_PROMPT_SYSTEM_PROMPT},
            {"role":"user","content":[
                self._video(video),
                {"type":"text","text":H3_PROMPT_USER_TEMPLATE.format(
                    source_performer_1=source_performer_1,
                    replacement_subject_1=replacement_subject_1,
                    source_performer_2=source_performer_2,
                    replacement_subject_2=replacement_subject_2)},
            ]},
        ], self.prompt_max_new_tokens, num_frames=num_frames)
        prompt = enforce_detail_preservation_preface(
            raw,source_performer_1,source_performer_2)
        return enforce_source_binding_sections(
            prompt,source_performer_1,source_performer_2,
            replacement_subject_1,replacement_subject_2)

    def close(self):
        self.client = None


class OpenAIMimoH3PromptWriter(OpenAIQwen38H3PromptWriter):
    """MiMo-V2.5 SGLang writer with explicit 8 FPS visual sampling."""

    uses_local_frame_sampling = False
    thinking_disabled = True

    def __init__(
        self,
        checkpoint_id=MIMO_CHECKPOINT,
        *,
        base_url=MIMO_SGLANG_BASE_URL,
        served_model=MIMO_SGLANG_SERVED_MODEL,
        api_key="EMPTY",
        timeout_seconds=900.0,
        video_fps=MIMO_VIDEO_FPS,
        replacement_max_new_tokens=REPLACEMENT_MAX_NEW_TOKENS,
        prompt_max_new_tokens=PROMPT_MAX_NEW_TOKENS,
        client=None,
    ):
        super().__init__(
            checkpoint_id,
            base_url=base_url,
            served_model=served_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            replacement_max_new_tokens=replacement_max_new_tokens,
            prompt_max_new_tokens=prompt_max_new_tokens,
            client=client,
        )
        self.video_fps = float(video_fps)
        if self.video_fps != MIMO_VIDEO_FPS:
            raise ValueError(f"MiMo person-replacement writer requires {MIMO_VIDEO_FPS:g} FPS")

    def _video(self, video):
        path = Path(video).expanduser().resolve(strict=True)
        return {
            "type":"video_url",
            "video_url":{"url":path.as_uri()},
            "fps":self.video_fps,
            "media_resolution":MIMO_MEDIA_RESOLUTION,
        }

    def _generate(self, messages, max_new_tokens, *, num_frames=None):
        del num_frames
        self._load()
        completion = self.client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            temperature=MIMO_TEMPERATURE,
            max_completion_tokens=max_new_tokens,
            stream=False,
            reasoning_effort="none",
            extra_body={
                "use_audio_in_video":False,
                "chat_template_kwargs":{"thinking":False,"enable_thinking":False},
            },
        )
        choices = getattr(completion,"choices",None)
        if not choices:
            raise ValueError("MiMo SGLang returned no choices")
        message = choices[0].message
        text = getattr(message,"content",None)
        if not isinstance(text,str) or not text.strip():
            raise ValueError("MiMo SGLang returned an empty completion")
        return text.strip()


class LocalQwen38H3PromptWriter:
    """One resident local Qwen3.5-27B instance for a whole worker partition."""

    uses_local_frame_sampling = True

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
        prompt = enforce_detail_preservation_preface(
            raw,source_performer_1,source_performer_2)
        return enforce_source_binding_sections(
            prompt,source_performer_1,source_performer_2,
            replacement_subject_1,replacement_subject_2)

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
