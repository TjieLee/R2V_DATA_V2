"""Qwen3.8-27B writer: lazy single load, two structured calls, fail-closed output."""

import sys
from types import SimpleNamespace

import pytest

CALL1_RESPONSE = ("SOURCE_PERFORMER_1: a younger woman in a yellow shirt\n"
               "SOURCE_PERFORMER_2: an older woman in a light shirt\n"
               "REPLACEMENT_SUBJECT_1: a middle-aged man\n"
               "REPLACEMENT_SUBJECT_2: a young man in denim")

LEGAL_PROMPT = """subject_definitions:
<Subject 1>: The replacement principal subject corresponding to source performer 1. an older man
<Subject 2>: The replacement principal subject corresponding to source performer 2. a young woman
<Video 1>: The source video providing the complete motion, facial performance, camera, scene, object
and temporal structure.

summary:
[video editing + reference generation] Replace the two original performers throughout <Video 1> with
<Subject 1> and <Subject 2>.

retention_analysis:
<Subject 1>: replace - use the replacement identity and appearance defined above throughout the clip.
<Subject 2>: replace - use the replacement identity and appearance defined above throughout the clip.
<Video 1>: preserve - preserve the complete action sequence and shot structure.

detailed_description:
[Shot 1] <Subject 1> stands at the left of frame facing <Subject 2> in <Video 1> and lifts a hand.
The camera holds a static wide framing throughout.

overall_soundscape:
N/A

non_diegetic_music:
N/A"""


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    shared = None

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls.shared if cls.shared is not None else cls()
        instance.path = path
        instance.kwargs = kwargs
        return instance

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages,dict(kwargs)))
        for key, flag in (("enable_thinking","reject_thinking"),("num_frames","reject_num_frames")):
            if key in kwargs and getattr(self,flag,False):
                raise TypeError(f"unexpected keyword argument {key}")
        return FakeInputs({"input_ids":[[1,2,3]]})

    def batch_decode(self, trimmed, **kwargs):
        return [self.text]

    def __call__(self, *args, **kwargs):
        return self


class FakeModel:
    def __init__(self):
        self.generate_calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path = path
        instance.kwargs = kwargs
        return instance

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[1,2,3,4,5]]

    @property
    def device(self):
        return "cuda:0"


def install(monkeypatch, processor=None, model=None):
    processor = processor or FakeProcessor()
    model = model or FakeModel()
    processor.text = LEGAL_PROMPT
    FakeProcessor.shared = processor
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        AutoProcessor=FakeProcessor, AutoModelForMultimodalLM=FakeModel))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(
        inference_mode=lambda: _null(), cuda=SimpleNamespace(is_available=lambda:True,
                                                             empty_cache=lambda:None)))
    return processor, model


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def writer(tmp_path, monkeypatch, **kwargs):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        LocalQwen38H3PromptWriter,
    )

    install(monkeypatch)
    instance = LocalQwen38H3PromptWriter(tmp_path, **kwargs)
    instance._load()
    instance.processor.text = LEGAL_PROMPT
    return instance


def test_lazy_single_load_uses_local_files_only(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        LocalQwen38H3PromptWriter,
    )

    install(monkeypatch)
    instance = LocalQwen38H3PromptWriter(tmp_path)
    assert instance.model is None  # nothing is loaded until the first call
    instance._load()
    instance._load()
    assert instance.model.path == str(tmp_path)
    assert instance.model.kwargs["local_files_only"] is True
    assert instance.model.kwargs["device_map"] == "auto"
    assert "dtype" not in instance.model.kwargs  # FP8 checkpoint keeps its dtype
    assert instance.processor.kwargs["local_files_only"] is True


def test_missing_transformers_fails_loudly(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        LocalQwen38H3PromptWriter,
    )

    monkeypatch.delitem(sys.modules,"transformers",raising=False)
    with pytest.raises(RuntimeError,match="no automatic installation"):
        LocalQwen38H3PromptWriter(tmp_path)._load()


def test_video_input_num_frames_and_generation_settings(tmp_path, monkeypatch):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = CALL1_RESPONSE
    video = tmp_path/"source.mp4"
    video.touch()
    instance.invent_replacements(video,"doctor","generic",num_frames=40)
    messages, kwargs = instance.processor.calls[-1]
    assert "fps" not in kwargs  # no vLLM-only fps sampling parameter
    assert kwargs["num_frames"] == 40 and kwargs["enable_thinking"] is False
    assert messages[0]["content"][0] == {"type":"video","video":str(video.resolve())}
    assert instance.model.generate_calls[-1]["do_sample"] is False
    assert instance.model.generate_calls[-1]["max_new_tokens"] == 512
    instance.write_h3_prompt(video,"p1","p2","r1","r2",num_frames=40)
    assert instance.model.generate_calls[-1]["max_new_tokens"] == 4096
    assert [kwargs["num_frames"] for _, kwargs in instance.processor.calls] == [40,40]
    assert len(instance.processor.calls) == 2  # one model instance, two calls


def test_template_without_thinking_support_is_reported(tmp_path, monkeypatch):
    processor = FakeProcessor()
    processor.reject_thinking = True
    processor.text = LEGAL_PROMPT
    install(monkeypatch,processor=processor)
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        LocalQwen38H3PromptWriter,
    )

    instance = LocalQwen38H3PromptWriter(tmp_path)
    instance._load()
    instance.processor.text = CALL1_RESPONSE
    video = tmp_path/"source.mp4"
    video.touch()
    instance.invent_replacements(video,"generic","generic",num_frames=20)
    assert instance.thinking_disabled is False  # honest, not silently assumed
    assert instance.processor.calls[-1][1]["num_frames"] == 20


def test_close_releases_model_and_processor(tmp_path, monkeypatch):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = CALL1_RESPONSE
    video = tmp_path/"source.mp4"
    video.touch()
    instance.invent_replacements(video,"generic","generic",num_frames=20)
    instance.close()
    assert instance.model is None and instance.processor is None


@pytest.mark.parametrize("text", [
    "SOURCE_PERFORMER_1: a\nSOURCE_PERFORMER_2: b\nREPLACEMENT_SUBJECT_1: c",
    ("SOURCE_PERFORMER_1: a\nSOURCE_PERFORMER_2: b\nREPLACEMENT_SUBJECT_1: c\n"
     "REPLACEMENT_SUBJECT_2: d\nEXTRA: e"),
    ("SOURCE_PERFORMER_2: b\nSOURCE_PERFORMER_1: a\nREPLACEMENT_SUBJECT_1: c\n"
     "REPLACEMENT_SUBJECT_2: d"),
    ("SOURCE_PERFORMER_1: \nSOURCE_PERFORMER_2: b\nREPLACEMENT_SUBJECT_1: c\n"
     "REPLACEMENT_SUBJECT_2: d"),
])
def test_call1_parser_is_strict(tmp_path, monkeypatch, text):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = text
    video = tmp_path/"source.mp4"
    video.touch()
    with pytest.raises(ValueError):
        instance.invent_replacements(video,"generic","generic",num_frames=20)


def test_call1_parser_accepts_four_ordered_fields(tmp_path, monkeypatch):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = ("SOURCE_PERFORMER_1: a younger woman in a yellow shirt\n"
                               "SOURCE_PERFORMER_2: an older woman in a light shirt\n"
                               "REPLACEMENT_SUBJECT_1: a middle-aged man\n"
                               "REPLACEMENT_SUBJECT_2: a young man in denim")
    video = tmp_path/"source.mp4"
    video.touch()
    assert instance.invent_replacements(video,"doctor","generic",num_frames=20) == (
        "a younger woman in a yellow shirt","an older woman in a light shirt",
        "a middle-aged man","a young man in denim")


def test_validator_accepts_legal_prompt():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        validate_h3_prompt_writer_output,
    )

    assert validate_h3_prompt_writer_output(LEGAL_PROMPT).startswith("subject_definitions:")


@pytest.mark.parametrize("broken", [
    LEGAL_PROMPT.replace("summary:","summaries:"),                       # missing section
    LEGAL_PROMPT.replace("non_diegetic_music:\nN/A",""),                 # missing section
    LEGAL_PROMPT.replace("overall_soundscape:\nN/A","overall_soundscape:\npreserve sound"),
    LEGAL_PROMPT.replace("<Subject 2>:","<Subject 3>:"),                 # forbidden subject
    LEGAL_PROMPT + "\n<Audio 1> is the source audio.\n",
    LEGAL_PROMPT.replace("replace -","attribute_transfer -"),
    LEGAL_PROMPT.replace("<Subject 1>: replace","<Subject 1> (appears in [Shot 1]): replace"),
    "",
])
def test_validator_rejects_broken_prompts(broken):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        validate_h3_prompt_writer_output,
    )

    with pytest.raises(ValueError):
        validate_h3_prompt_writer_output(broken)


def test_validator_requires_subjects_in_detailed_description():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        validate_h3_prompt_writer_output,
    )

    with pytest.raises(ValueError,match="detailed_description"):
        validate_h3_prompt_writer_output(
            LEGAL_PROMPT.replace("[Shot 1] <Subject 1> stands","[Shot 1] the performer stands"))


def test_num_frames_is_never_dropped(tmp_path, monkeypatch):
    processor = FakeProcessor()
    processor.reject_num_frames = True
    processor.text = LEGAL_PROMPT
    install(monkeypatch,processor=processor)
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        LocalQwen38H3PromptWriter,
    )

    instance = LocalQwen38H3PromptWriter(tmp_path)
    instance._load()
    video = tmp_path/"source.mp4"
    video.touch()
    with pytest.raises(TypeError,match="num_frames"):
        instance.invent_replacements(video,"generic","generic",num_frames=40)


def test_validator_rejects_leading_prose_and_thinking():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        validate_h3_prompt_writer_output,
    )

    for broken in ("Here is the prompt you asked for.\n\n" + LEGAL_PROMPT,
                   "subject definitions first\n" + LEGAL_PROMPT,
                   "<think>plan the sections</think>\n" + LEGAL_PROMPT,
                   LEGAL_PROMPT + "\n</think>"):
        with pytest.raises(ValueError):
            validate_h3_prompt_writer_output(broken)
    assert validate_h3_prompt_writer_output("\n\n" + LEGAL_PROMPT + "\n").startswith(
        "subject_definitions:")
