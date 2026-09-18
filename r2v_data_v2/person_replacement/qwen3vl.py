"""Two plain-text calls sharing one lazy, local-only Qwen3-VL instance."""

from pathlib import Path

TWO_PERSON_VIDEO_FPS = 4.0

TWO_SOURCE_PROMPT = """Watch this whole single-shot video. Identify the two main physical performers.
For SOURCE_SUBJECT_1 and SOURCE_SUBJECT_2, write one concise natural description of each
source performer so that the person can be identified in <Video 1>. Describe the person's
visible appearance and, when useful, a simple locator such as foreground/background or the
initial left/right position. Keep these fields about who the person is, not what the person does.
Put actions, poses, held objects, hand state, expression, gaze, mouth state and interactions
in SHOT_DESCRIPTION instead.
Each Subject label must follow the same physical performer throughout crossings, overlap,
occlusion and screen-order changes.
Describe the complete visible single-shot progression in playback order.
Preserve every visible action/state transition and important hand-object interaction.
Do not omit, merge, reorder or invent actions. Track held/touched props through the sequence
and state when they remain in the person's hand/contact. Include starting positions,
hand-body/person-person contact, occlusions, crossings and camera movement/framing changes.
Pay particular attention to brief hand, arm, head and object movements.
Describe every visible motion/state transition, including small movements that occur over a short interval.
Do not replace a sequence of visible actions with a static summary such as "remains still throughout"
unless the performer is genuinely motionless for the entire source sequence.
For an interacted prop, preserve the visible progression of its state/contact/motion in playback order.
Do not infer race, ethnicity or nationality.
Use English and the exact reference labels <Subject 1> and <Subject 2> in SHOT_DESCRIPTION.
Never write bare "Subject 1" or "Subject 2" there. No shot headers, markdown or explanation.
In SHOT_DESCRIPTION, refer to the two performers with <Subject 1> and <Subject 2>
rather than gendered pronouns such as he, she, his or her, so the source action
description stays independent of the target appearance.
Return exactly three nonempty single-line labelled fields:
SOURCE_SUBJECT_1: ...
SOURCE_SUBJECT_2: ...
SHOT_DESCRIPTION: ..."""

TWO_REPLACEMENT_PROMPT = """Invent two ordinary realistic replacement people.
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
or nationality.
Return English, exactly two nonempty single-line labelled fields, no markdown/explanation:
REPLACEMENT_SUBJECT_1: ...
REPLACEMENT_SUBJECT_2: ..."""


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

    def invent_two(self, subject1: str, subject2: str) -> tuple[str, str]:
        return _labelled_fields(self._text([{
            "type":"text", "text":TWO_REPLACEMENT_PROMPT.format(subject1=subject1, subject2=subject2),
        }]), ("REPLACEMENT_SUBJECT_1", "REPLACEMENT_SUBJECT_2"))
