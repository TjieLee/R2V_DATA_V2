"""Two plain-text calls sharing one lazy, local-only Qwen3-VL instance."""

from pathlib import Path

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
