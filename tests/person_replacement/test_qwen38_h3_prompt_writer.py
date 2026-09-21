"""Qwen3.8-27B writer: lazy single load, two structured calls, fail-closed output."""

import sys
from types import SimpleNamespace

import pytest

CALL1_RESPONSE = ("SOURCE_PERFORMER_1: a younger woman in a yellow shirt\n"
               "SOURCE_PERFORMER_2: an older woman in a light shirt\n"
               "REPLACEMENT_SUBJECT_1: a middle-aged man\n"
               "REPLACEMENT_SUBJECT_2: a young man in denim")

LEGAL_PROMPT = """subject_definitions:
<Subject 1>: an older man. This replacement applies only to the source person in <Video 1> identified by the static visual locator: a younger woman in a yellow shirt.
<Subject 2>: a young woman. This replacement applies only to the source person in <Video 1> identified by the static visual locator: an older woman in a light shirt.
<Video 1>: The source video providing the complete motion, facial performance, camera, scene, object and temporal structure.

summary:
[video editing + reference generation] <Subject 1> replaces only the source person identified by a younger woman in a yellow shirt. <Subject 2> replaces only the source person identified by an older woman in a light shirt. Keep both mappings fixed.

retention_analysis:
<Subject 1>: replace - appearance first; replace only the source person identified by a younger woman in a yellow shirt.
<Subject 2>: replace - appearance first; replace only the source person identified by an older woman in a light shirt.
<Video 1>: preserve - preserve the complete action sequence and shot structure. Binding is fixed and must never swap.

detailed_description:
Only replace <Subject 1> and <Subject 2>'s identities and appearances; keep <Video 1>'s motion, facial performance, interactions, and timing unchanged.
Keep <Video 1>'s camera, framing, scene, lighting, and non-person objects unchanged.

<Subject 1> raises one hand toward <Subject 2> while <Subject 2> remains in place, with the camera holding a static wide framing throughout.

[Shot 1] <Subject 1> stands at the left of frame facing <Subject 2> in <Video 1> and lifts a hand while the camera remains static.
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
        # Match the real Qwen video processor, whose default fps would conflict
        # with an explicit num_frames unless the writer disables it after load.
        self.video_processor = SimpleNamespace(fps=2)

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


def test_qwen38_sglang_writer_uses_video_url_and_non_thinking_contract(tmp_path):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        OpenAIQwen38H3PromptWriter,
    )

    calls = []
    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = CALL1_RESPONSE if len(calls) == 1 else LEGAL_PROMPT
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content, reasoning_content=None))]
            )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    writer = OpenAIQwen38H3PromptWriter(
        "/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next",
        client=client,
    )
    video = tmp_path/"source.mp4"
    video.touch()
    result = writer.invent_replacements(video,"generic","doctor",num_frames=40)
    assert result[0].startswith("a younger woman")
    writer.write_h3_prompt(video,*result,num_frames=40)
    assert calls[0]["model"] == "Qwen/Qwen3.8-Flash-Next"
    media = calls[0]["messages"][0]["content"][0]
    assert media["type"] == "video_url" and media["video_url"]["url"] == video.resolve().as_uri()
    assert calls[0]["temperature"] == 0.7 and calls[0]["top_p"] == 0.8
    assert calls[0]["extra_body"]["top_k"] == 20
    assert calls[0]["presence_penalty"] == 1.5
    assert calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert calls[0]["max_tokens"] == 512 and calls[1]["max_tokens"] == 4096
    assert "mm_processor_kwargs" not in calls[0]["extra_body"]


def test_mimo_writer_uses_explicit_fps8_and_visual_only_contract(tmp_path):
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        OpenAIMimoH3PromptWriter,
    )

    calls = []
    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = CALL1_RESPONSE if len(calls) == 1 else LEGAL_PROMPT
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    writer = OpenAIMimoH3PromptWriter(
        "/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5",
        client=client,
    )
    video = tmp_path/"source.mp4"
    video.touch()
    result = writer.invent_replacements(video,"generic","doctor",num_frames=40)
    writer.write_h3_prompt(video,*result,num_frames=40)
    media = calls[0]["messages"][0]["content"][0]
    assert calls[0]["model"] == "mimo-v2.5"
    assert media == {
        "type":"video_url",
        "video_url":{"url":video.resolve().as_uri()},
        "fps":8.0,
        "media_resolution":"default",
    }
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_completion_tokens"] == 512
    assert calls[1]["max_completion_tokens"] == 4096
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["extra_body"]["use_audio_in_video"] is False
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {
        "thinking":False,"enable_thinking":False,
    }


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
    assert instance.model.kwargs["trust_remote_code"] is True
    assert "dtype" not in instance.model.kwargs  # FP8 checkpoint keeps its dtype
    assert instance.processor.kwargs["local_files_only"] is True
    assert instance.processor.video_processor.fps is None


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
    # Local files use the `path` locator; `video` is for decoded media objects.
    assert messages[0]["content"][0] == {"type":"video","path":str(video.resolve())}
    assert "video" not in messages[0]["content"][0]  # no decoded-object key
    assert str(video.resolve()).startswith("/")  # local path, never a URL
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


def test_validator_allows_detailed_description_without_literal_video_token():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        validate_h3_prompt_writer_output,
    )

    prompt = LEGAL_PROMPT.replace(
        " stands at the left of frame facing <Subject 2> in <Video 1> and lifts a hand.",
        " stands at the left of frame facing <Subject 2> and lifts a hand.",
    )
    assert "<Video 1>" not in dict(
        __import__(
            "r2v_data_v2.person_replacement.qwen38_h3_prompt_writer",
            fromlist=["split_sections"],
        ).split_sections(prompt)
    )["detailed_description"]
    assert validate_h3_prompt_writer_output(prompt) == prompt.strip()


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


def test_both_calls_use_the_same_path_style_video_part(tmp_path, monkeypatch):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = CALL1_RESPONSE
    video = tmp_path/"source.mp4"
    video.touch()
    instance.invent_replacements(video,"doctor","generic",num_frames=40)
    instance.write_h3_prompt(video,"p1","p2","r1","r2",num_frames=40)
    parts = []
    for messages, _ in instance.processor.calls:
        for message in messages:
            content = message["content"]
            if isinstance(content,list):
                parts.extend(item for item in content if item.get("type") == "video")
    assert parts == [{"type":"video","path":str(video.resolve())}]*2
    assert all(not part.keys() - {"type","path"} for part in parts)
    assert [kwargs["num_frames"] for _, kwargs in instance.processor.calls] == [40,40]


def test_replacement_planning_prompt_keeps_identity_distance_without_prompt_bloat():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        REPLACEMENT_PLANNING_PROMPT,
    )

    prompt = REPLACEMENT_PLANNING_PROMPT
    assert "clearly different from its source" in prompt
    assert "at least one cross-gender replacement" in prompt
    assert "at least one clearly different adult age band" in prompt
    assert "face shape, hair, grooming, skin tone, clothing/style" in prompt
    assert "Do not satisfy diversity only by\nchanging clothes." in prompt
    assert "body scale and occupied silhouette broadly compatible" in prompt
    assert "Do not add props, actions,\ngestures, pose, or interactions." in prompt
    assert "{cue1}" in prompt and "{cue2}" in prompt
    assert len(prompt.split()) < 500


def test_system_prompt_is_slim_and_prioritizes_salient_motion():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        H3_PROMPT_SYSTEM_PROMPT,
    )

    prompt = H3_PROMPT_SYSTEM_PROMPT
    assert "Change who the two people are; do not change what happens." in prompt
    assert "Do not repeat replacement\nappearance" in prompt
    assert "Describe only meaningful visible action/state changes" in prompt
    assert "write [Shot 1] directly" in prompt
    assert "one short dynamic overview sentence naming both" not in prompt
    assert "Do not enumerate every blink" in prompt
    assert 'Use words such\nas "throughout", "maintains", or "remains" only' in prompt
    assert "If uncertain, stay conservative." in prompt
    assert "TEMPORAL VERIFICATION BEFORE WRITING" not in prompt
    assert "FACIAL AND HEAD MICRO-MOTION PRIORITY" not in prompt
    assert "FEW-SHOT DETAILED-DESCRIPTION STYLE EXAMPLES" not in prompt
    assert "early-to-middle-to-late progression" not in prompt
    assert "Treat a visible change as a micro-event" not in prompt
    assert len(prompt.split()) < 650


def test_final_prompt_uses_appearance_first_and_single_source_bindings(tmp_path, monkeypatch):
    instance = writer(tmp_path,monkeypatch)
    instance.processor.text = LEGAL_PROMPT
    video = tmp_path/"source.mp4"
    video.touch()
    locator1 = "the younger woman in a yellow shirt initially on the left"
    locator2 = "the older woman in a light shirt initially on the right"
    prompt = instance.write_h3_prompt(
        video,
        locator1,
        locator2,
        "a young South Asian woman with long dark wavy hair and an olive sweater",
        "an older white man with short curly white hair and a navy shirt",
        num_frames=40,
    )
    subject1 = next(line for line in prompt.splitlines() if line.startswith("<Subject 1>:"))
    assert subject1.startswith(
        "<Subject 1>: a young South Asian woman with long dark wavy hair and an olive sweater."
    )
    assert f"Replaces only the source person identified by: {locator1}." in subject1
    assert prompt.count(locator1) == 1
    assert prompt.count(locator2) == 1
    detailed = dict(__import__(
        "r2v_data_v2.person_replacement.qwen38_h3_prompt_writer",
        fromlist=["split_sections"],
    ).split_sections(prompt))["detailed_description"]
    assert "Binding is fixed throughout" not in detailed
    assert "replace appearance only" in prompt


def test_preservation_prefix_is_not_duplicated_when_model_uses_newline_between_fixed_sentences():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        DETAIL_CAMERA_SENTENCE,
        DETAIL_PRESERVATION_PREFIX,
        DETAIL_PRESERVATION_SENTENCE,
        enforce_detail_preservation_preface,
        split_sections,
    )

    raw = LEGAL_PROMPT.replace(
        DETAIL_PRESERVATION_PREFIX,
        DETAIL_PRESERVATION_SENTENCE + "\n" + DETAIL_CAMERA_SENTENCE,
        1,
    )
    fixed = enforce_detail_preservation_preface(
        raw,
        "a younger woman in a yellow shirt",
        "an older woman in a light shirt",
    )
    detailed = dict(split_sections(fixed))["detailed_description"].strip()
    assert detailed.startswith(DETAIL_PRESERVATION_PREFIX)
    assert detailed.count(DETAIL_PRESERVATION_SENTENCE) == 1
    assert detailed.count(DETAIL_CAMERA_SENTENCE) == 1


def test_preservation_sentence_is_deterministically_prepended():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        DETAIL_PRESERVATION_PREFIX,
        enforce_detail_preservation_preface,
        split_sections,
        validate_h3_prompt_writer_output,
    )

    without = LEGAL_PROMPT.replace(DETAIL_PRESERVATION_PREFIX + "\n\n","",1)
    fixed = enforce_detail_preservation_preface(
        without,"a younger woman in a yellow shirt","an older woman in a light shirt")
    detailed = dict(split_sections(fixed))["detailed_description"].strip()
    assert detailed.startswith(DETAIL_PRESERVATION_PREFIX)
    assert detailed.count(DETAIL_PRESERVATION_PREFIX) == 1
    assert validate_h3_prompt_writer_output(fixed) == fixed.strip()

    unchanged = enforce_detail_preservation_preface(
        LEGAL_PROMPT,"a younger woman in a yellow shirt","an older woman in a light shirt")
    assert unchanged.count(DETAIL_PRESERVATION_PREFIX) == 1


def test_validator_allows_shot1_directly_after_prefix_and_allows_speculative_phrasing():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        DETAIL_PRESERVATION_PREFIX,
        validate_h3_prompt_writer_output,
    )

    missing_preface = LEGAL_PROMPT.replace(
        DETAIL_PRESERVATION_PREFIX + "\n\n",
        "",
        1,
    )
    with pytest.raises(ValueError,match="canonical preservation prefix"):
        validate_h3_prompt_writer_output(missing_preface)

    # No separate dynamic overview is required in V38-slim.
    direct = LEGAL_PROMPT.replace(
        "<Subject 1> raises one hand toward <Subject 2> while <Subject 2> remains in place, "
        "with the camera holding a static wide framing throughout.\n\n",
        "",
        1,
    )
    assert validate_h3_prompt_writer_output(direct) == direct.strip()

    speculative = LEGAL_PROMPT.replace(
        "and lifts a hand while the camera remains static.",
        "and lifts a hand as if controlling an invisible force while the camera remains static.",
    )
    assert validate_h3_prompt_writer_output(speculative) == speculative.strip()


def test_prompt_explicitly_requires_the_fixed_audio_tail():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        H3_PROMPT_SYSTEM_PROMPT,
        H3_PROMPT_USER_TEMPLATE,
    )

    tail = "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    assert tail in H3_PROMPT_SYSTEM_PROMPT
    assert tail in H3_PROMPT_USER_TEMPLATE
    assert "The final response must end with the two exact N/A sections above." in H3_PROMPT_SYSTEM_PROMPT


def test_system_prompt_still_enforces_the_six_section_contract():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        H3_PROMPT_SYSTEM_PROMPT,
    )

    prompt = H3_PROMPT_SYSTEM_PROMPT
    for heading in ("subject_definitions:","summary:","retention_analysis:",
                    "detailed_description:","overall_soundscape:","non_diegetic_music:"):
        assert heading in prompt
    assert "Never define <Subject 3> or <Subject 4>." in prompt
    assert "N/A" in prompt
