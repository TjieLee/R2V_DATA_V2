"""Two plain-text calls sharing one lazy, local-only Qwen3-VL instance."""

from pathlib import Path

TWO_PERSON_VIDEO_FPS = 4.0

TWO_SOURCE_BODY = """Watch this whole single-shot video. Select the two main physical performers to edit.
For SOURCE_SUBJECT_1 and SOURCE_SUBJECT_2, write one concise STATIC IDENTIFICATION PHRASE
per performer so that the person is easy to identify in <Video 1>.
A static identification phrase may include only: apparent age, gender presentation, hair,
facial appearance, body build, clothing, other stable visible characteristics, and an
approximate initial location such as foreground/background or initial left/right when useful.
A static identification phrase must NEVER contain an action, motion, pose progression, gaze
behavior, expression or emotion, speaking, mouth state, held-object action, or any temporal
sequence, and must never use words such as then, later, at first, turning, looking, smiling
or speaking. All of that belongs in SHOT_DESCRIPTION only.
BAD: "a woman in a yellow shirt looking down then turning around"
GOOD: "a younger woman in a yellow graphic T-shirt with dark hair in a high ponytail and
bangs, initially in the foreground"
BAD: "an older woman smiling and then speaking seriously"
GOOD: "an older woman in a light-colored button-up shirt with dark hair tied back"
The video may contain other visible people. Do not assume the two selected performers are the
only people in the shot. Describe only the two main performers selected for replacement; other
visible people are non-target scene content and must not be folded into either target identity.
Keep the descriptions natural rather than filling a checklist.
Each Subject label must follow the same physical performer throughout crossings, overlap,
occlusion and screen-order changes.
Put the complete action sequence in SHOT_DESCRIPTION. Describe the visible progression in
playback order and preserve every important action/state transition and hand-object interaction.
Use source-video timestamps for significant action/state transitions, especially when an
action starts after a period of stillness. State the preceding interval explicitly when useful,
for example: "<Subject 1> remains seated without the cup until 00:04.000. At 00:04.000,
<Subject 1> raises <Subject 1>'s right hand toward the cup." Keep these timestamps tied to the
source video; do not shift an action earlier or later. Do not omit, merge, reorder or invent
actions. Track held/touched props through the sequence
and state when they remain in the person's hand/contact. Include relevant starting positions,
hand-body/person-person contact, occlusions, crossings and camera movement/framing changes.
Pay particular attention to brief hand, arm, head and object movements.
Do not replace a sequence of visible actions with a static summary such as "remains still throughout"
unless the performer is genuinely motionless for the entire source sequence.
Use English and the exact reference labels <Subject 1> and <Subject 2> in SHOT_DESCRIPTION.
Never write bare "Subject 1" or "Subject 2" there. Do not use person pronouns such as he, she,
him, her, his or hers in SHOT_DESCRIPTION. Repeat the exact Subject label instead, including
possessives, for example: "<Subject 1> raises <Subject 1>'s right hand."
No shot headers, markdown or explanation. Do not infer race, ethnicity or nationality."""

TWO_SOURCE_PROMPT = TWO_SOURCE_BODY + """
Return exactly three nonempty single-line labelled fields:
SOURCE_SUBJECT_1: ...
SOURCE_SUBJECT_2: ...
SHOT_DESCRIPTION: ..."""

TWO_SOURCE_WITH_PERFORMANCE_PROMPT = TWO_SOURCE_BODY + """

For the two extra performance fields, write object-only non-temporal anchors.
They may name only the visible held, carried, touched or clearly interacted object and
the hand-object relation, for example: holding a black clipboard with both hands, carrying
a bag in the right hand, holding and handling a hockey stick, touching the table with the
left hand. They must never describe standing, sitting, standing still, walking, facing
direction, general pose, facial expression, gaze, emotion, mouth state, clothing, uniform,
hair, body appearance, background, camera or a temporal action sequence; <Video 1> alone is
the authority for pose, motion, facial performance and action timing. If the performer has
no object that needs protection, output exactly NONE and nothing else.
The three field families have different jobs, and they are not in conflict:
SOURCE_SUBJECT_1 and SOURCE_SUBJECT_2 stay about identity and stable visual role and
must not contain temporary props or temporary actions. SOURCE_PERFORMANCE_1 and
SOURCE_PERFORMANCE_2 are the non-temporal record of the visible held, carried, touched
and interacted objects and their hand-object relation, so a replacement performer can
inherit them. SHOT_DESCRIPTION still keeps the complete temporal sequence in playback
order. The same prop may therefore be named as a non-temporal anchor in
SOURCE_PERFORMANCE and also described with its timing in SHOT_DESCRIPTION; that is
intentional, not a contradiction.
Keep each anchor limited to the object and its hand-object relation. Never write timestamps
or ordering words such as then, before, after, initially or later, never write a full action
sequence, background, camera, replacement appearance, race, ethnicity or nationality, and
never repeat the Subject labels inside the value. If an object is handled during only part
of the shot, do not claim it is held throughout; write it as interacting with that object,
as in: interacting with the same black clipboard, holding it with both hands when handled.
When there is no object to protect, write exactly NONE.
Return exactly five nonempty single-line labelled fields:
SOURCE_SUBJECT_1: ...
SOURCE_SUBJECT_2: ...
SOURCE_PERFORMANCE_1: ...
SOURCE_PERFORMANCE_2: ...
SHOT_DESCRIPTION: ..."""

TWO_REPLACEMENT_BODY = """Invent two ordinary realistic replacement people.
Source Subject 1: {subject1}
Source Subject 2: {subject2}
Replacement 1 should look clearly different from Source 1, and Replacement 2 should look
clearly different from Source 2. Change several visible identity characteristics so each
replacement reads as a different person. You may vary apparent age, gender presentation,
facial appearance or facial hair, hairstyle and hair color, skin appearance, body build,
and clothing, but choose a coherent realistic combination rather than forcing every
category to differ. The two replacements should also be easy to distinguish from each other.
Keep each description concise and natural. No celebrities, fantasy characters, costume
characters or exaggerated bodies. Describe only the replacement person's appearance and
clothing. Do not invent held/carried/touched objects or props, pose or gesture,
standing/sitting state, hand state, action or motion, expression, gaze, mouth state,
screen position or interactions; those come from <Video 1>. Do not assign race, ethnicity
or nationality."""

TWO_REPLACEMENT_TAIL = """
Return English, exactly two nonempty single-line labelled fields, no markdown/explanation:
REPLACEMENT_SUBJECT_1: ...
REPLACEMENT_SUBJECT_2: ..."""

TWO_REPLACEMENT_PROMPT = TWO_REPLACEMENT_BODY + TWO_REPLACEMENT_TAIL

TWO_REPLACEMENT_WITH_DIVERSITY_PROMPT = TWO_REPLACEMENT_BODY + """
Replacement 1 diversity cue: {cue1}
Replacement 2 diversity cue: {cue2}
Each cue is an appearance-only style hint. Use it only through clothing, hairstyle,
grooming, colour palette and general visual styling. A generic cue means ordinary
everyday clothing with no particular role. A profession cue means that profession's
clothing and grooming only. A period or distinctive-attire cue means that historical or
stylistic clothing and hairstyle only. Never introduce any object, prop, tool, equipment,
weapon, accessory or item that would be held, worn as an interactive device or interacted
with, and never introduce an action, pose, gesture, hand state or interaction implied by
the occupation or historical style. Clothing and grooming alone must express the cue.
Do not request a substantially different height, extreme body build, or exaggerated body
silhouette. Keep each replacement person's overall body scale and silhouette compatible with
that performer's source body so the source pose and spatial occupancy can be preserved.
The two replacements must remain clearly distinguishable from each other and from their
own source performer.""" + TWO_REPLACEMENT_TAIL


def _labelled_fields(text: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != len(labels):
        raise ValueError("Qwen must return exactly the requested labelled fields")
    values = []
    for line, label in zip(lines, labels, strict=True):
        prefix = label + ":"
        if not line.startswith(prefix):
            raise ValueError(f"Missing or out-of-order Qwen field: {label}")
        value = line[len(prefix):].strip()
        if not value or "[shot" in value.lower():
            raise ValueError(f"Empty or shot-header-containing Qwen field: {label}")
        values.append(value)
    return tuple(values)

SOURCE_PROMPT = """Watch this video and briefly describe the single main person so that a video-editing model can unambiguously identify which person should be replaced.
Use stable visible characteristics such as clothing, hair, overall appearance, and approximate location in the frame.
Also include any stable pose or action and any hand-body or hand-object contact that is clearly visible and useful for preserving the original performance, for example both hands inside trouser pockets, holding an object, or resting a hand on the body.
Keep it concise. Do not describe the background or explain your reasoning. Do not infer ethnicity or race.
Output only one short natural-language description of the person."""

REPLACEMENT_PROMPT = """The source video contains this person:
"{description}"
Invent one realistic person who looks clearly different and could plausibly replace this person in the same video.
Change several strong visible characteristics. You may vary apparent gender presentation, age, hairstyle, hair color, skin tone, facial appearance, body build, clothing style, and clothing colors.
The replacement should remain an ordinary realistic human, not a fantasy character, costume character, celebrity, or exaggerated body type.
Describe only the replacement person's appearance; do not invent a different pose or action.
Keep the description concise and suitable for a video-editing prompt.
Output only the replacement-person description. Do not explain your choices."""


class LocalQwen:
    def __init__(self, model_path, *, max_new_tokens=128, dtype="auto"):
        self.model_path = Path(model_path).expanduser()
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.model = self.processor = None

    def _load(self):
        if self.model is not None:
            return
        if not self.model_path.is_dir():
            raise ValueError(f"Qwen checkpoint must be a local directory: {self.model_path}")
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Person-replacement needs a Qwen3-VL-capable Transformers runtime, "
                "torch/accelerate and a Transformers video decoder (e.g. PyAV). "
                "Use an isolated environment; do not upgrade shared pipeline dependencies."
            ) from exc
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path), local_files_only=True, device_map="auto", dtype=self.dtype,
        ).eval()

    def _text(self, content):
        self._load()
        import torch

        try:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(self.model.device)
        except (ImportError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "Qwen3-VL video/chat preprocessing failed. Check the isolated Transformers "
                "version and its video dependencies (PyAV/torchcodec); no fallback model is used."
            ) from exc
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated, strict=True)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise ValueError("Qwen returned an empty person description")
        return text

    def describe(self, video: Path) -> str:
        return self._text([
            {"type": "video", "path": str(video.resolve(strict=True))},
            {"type": "text", "text": SOURCE_PROMPT},
        ])

    def invent(self, description: str) -> str:
        return self._text([{
            "type": "text", "text": REPLACEMENT_PROMPT.format(description=description),
        }])

    def close(self):
        """Release Qwen before the separate Bernini launcher consumes GPU memory."""
        if self.model is None:
            return
        import gc

        import torch

        self.model = self.processor = None
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe_two(self, video: Path) -> tuple[str, str, str]:
        return _labelled_fields(self._text([
            {"type":"video", "path":str(video.resolve(strict=True)), "fps":TWO_PERSON_VIDEO_FPS},
            {"type":"text", "text":TWO_SOURCE_PROMPT},
        ]), ("SOURCE_SUBJECT_1", "SOURCE_SUBJECT_2", "SHOT_DESCRIPTION"))

    def describe_two_with_performance(self, video: Path) -> tuple[str, str, str, str, str]:
        """Pair-executor source call: two subjects, two non-temporal performance anchors, shot."""
        return _labelled_fields(self._text([
            {"type":"video", "path":str(video.resolve(strict=True)), "fps":TWO_PERSON_VIDEO_FPS},
            {"type":"text", "text":TWO_SOURCE_WITH_PERFORMANCE_PROMPT},
        ]), ("SOURCE_SUBJECT_1", "SOURCE_SUBJECT_2", "SOURCE_PERFORMANCE_1",
             "SOURCE_PERFORMANCE_2", "SHOT_DESCRIPTION"))

    def invent_two(self, subject1: str, subject2: str) -> tuple[str, str]:
        return _labelled_fields(self._text([{
            "type":"text", "text":TWO_REPLACEMENT_PROMPT.format(subject1=subject1, subject2=subject2),
        }]), ("REPLACEMENT_SUBJECT_1", "REPLACEMENT_SUBJECT_2"))

    def invent_two_with_diversity(self, subject1: str, subject2: str,
                                  cue1: str, cue2: str) -> tuple[str, str]:
        """Pair-executor replacement call with deterministic appearance-only cues."""
        return _labelled_fields(self._text([{
            "type":"text",
            "text":TWO_REPLACEMENT_WITH_DIVERSITY_PROMPT.format(subject1=subject1, subject2=subject2,
                                                                cue1=cue1, cue2=cue2),
        }]), ("REPLACEMENT_SUBJECT_1", "REPLACEMENT_SUBJECT_2"))
