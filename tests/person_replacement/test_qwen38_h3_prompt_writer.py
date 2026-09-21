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
Keep <Subject 1> and <Subject 2>'s actions, body poses and motion, head orientation and motion, facial expressions, gaze, mouth/lip/jaw movement, hand-object and person-person interactions, positions, scale, occlusion, and timing exactly the same as in <Video 1>, and keep the source camera movement, framing, background, lighting, and scene structure unchanged; only their identities and appearances are replaced.

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


def test_replacement_planning_prompt_pushes_real_identity_distance():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        REPLACEMENT_PLANNING_PROMPT,
    )

    prompt = REPLACEMENT_PLANNING_PROMPT
    assert "noticeably different from its corresponding source" in prompt
    assert "not just a lightly modified lookalike" in prompt
    assert "CASTING VARIATION" in prompt
    assert "gender presentation and approximate age are\nidentification facts" in prompt
    assert "Do not default to the same gender presentation or a nearby age band" in prompt
    assert "at least one replacement whose apparent gender presentation differs" in prompt
    assert "at least one replacement whose apparent adult age band differs clearly" in prompt
    assert "do not introduce a child merely to\ncreate contrast" in prompt
    assert "It does NOT mean preserving gender presentation, age, face," in prompt
    assert "at least two or three of the\nfollowing aspects" in prompt
    assert "- apparent age band," in prompt
    assert "- hairstyle, hair texture or hair colour," in prompt
    assert "- face shape or facial-structure impression," in prompt
    assert "- grooming or facial hair," in prompt
    assert "- overall colour palette," in prompt
    assert ("The diversity cue must not be satisfied only by a clothing change." in prompt)
    assert "Do not make the replacement bizarre, caricatured or unnaturally extreme." in prompt
    assert "Do not request a radically different body size or silhouette." in prompt
    assert "broadly compatible with the source performer's body" in prompt
    # Existing constraints must survive
    assert "not be celebrities" in prompt
    assert "- props" in prompt and "- weapons" in prompt
    assert "{cue1}" in prompt and "{cue2}" in prompt


def test_system_prompt_adds_global_preface_and_first_sentence_priority():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        H3_PROMPT_SYSTEM_PROMPT,
    )

    prompt = H3_PROMPT_SYSTEM_PROMPT
    assert "GLOBAL DETAILED-DESCRIPTION PREFACE" in prompt
    assert "mandatory preservation statement" in prompt
    assert "actions, body poses and motion, head orientation and motion" in prompt
    assert "facial expressions, gaze, mouth/lip/jaw movement" in prompt
    assert "only their identities and appearances are replaced" in prompt
    assert "Immediately after the `detailed_description:` heading and before `[Shot 1]`" in prompt
    assert "exactly one source-grounded dynamic overview sentence" in prompt
    assert "dominant visible\nmotion of <Subject 1> and <Subject 2>" in prompt
    assert "their relative movement or most important\ninteraction" in prompt
    assert "dominant camera behavior across the\nsource clip" in prompt
    assert "Write concrete target-video content, not editing instructions." in prompt
    assert "Describe what\nvisibly happens rather than saying to preserve it." in prompt
    assert "subject trajectories" in prompt
    assert "body/head/hand movement" in prompt
    assert "camera tracking, panning," in prompt
    assert "If the source camera is genuinely static" in prompt
    assert '"The target video is a direct edit of <Video 1>."' in prompt
    assert '"Preserve the complete source motion."' in prompt
    assert "Do not use this sentence for static appearance, clothing, background, lighting," in prompt
    assert "FIRST-SENTENCE PRIORITY" in prompt
    assert "The first sentence inside [Shot 1] is especially important." in prompt
    assert "dominant visible motion and the dominant camera\nbehavior" in prompt
    assert "- how they are moving relative to each other," in prompt
    assert ("- and the main camera behavior (static, panning, tracking, tilting, zooming,\n"
            "  reframing, handheld, or cutting)." in prompt)
    assert ("Do not begin [Shot 1] with static clothing, static background description,"
            in prompt)
    assert 'generic wording such as "preserve the original motion".' in prompt


def test_prompt_includes_identity_and_motion_icl_examples():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        H3_PROMPT_SYSTEM_PROMPT,
        REPLACEMENT_PLANNING_PROMPT,
    )

    replacement = REPLACEMENT_PLANNING_PROMPT
    assert "FEW-SHOT IDENTITY EXAMPLES" in replacement
    assert "Do not use empty identity phrases" in replacement
    assert '"a generic man"' in replacement and '"a generic woman"' in replacement
    assert "woman in her forties with short textured auburn hair" in replacement
    assert "older man in his sixties with swept-back silver hair" in replacement
    assert "young adult woman with a cropped black pixie cut" in replacement
    assert "middle-aged man with wavy dark-brown hair" in replacement
    assert "a face-shape or grooming detail" in replacement
    assert "Optional diversity dimensions may also include a broad country/region" in replacement
    assert "skin tone" in replacement and "profession-inspired styling" in replacement
    assert "Black British woman in her forties with deep brown skin" in replacement
    assert "young South Asian woman with medium-brown skin" in replacement
    assert "white Eastern European man with fair skin" in replacement
    assert "young Southeast Asian man with warm tan skin" in replacement
    assert "Latina woman in her fifties with medium olive skin" in replacement
    assert "Do not force every replacement to mention every one" in replacement

    system = H3_PROMPT_SYSTEM_PROMPT
    assert "OBSERVATION-ONLY LANGUAGE" in system
    assert "FACIAL AND HEAD MICRO-MOTION PRIORITY" in system
    assert "initial head orientation: frontal, three-quarter left/right" in system
    assert "initial mouth state: closed, slightly parted, or open" in system
    assert "visible lip and lower-jaw movement over time" in system
    assert "final visible head/gaze/mouth state" in system
    assert "Do not collapse these observations into only" in system
    assert "Do not infer spoken words or phoneme-level lip sync." in system
    assert "FEW-SHOT DETAILED-DESCRIPTION STYLE EXAMPLES" in system
    assert "Example A — face-dominant static close-up:" in system
    assert "Example B — subtle head, gaze and mouth transitions:" in system
    assert "Example C — body motion with a moving camera:" in system
    assert "the lips meet" in system
    assert "eyes move first toward <Subject 2>" in system
    assert "camera tracks backward with them" in system
    assert "Do not invent an invisible force" in system


def test_preservation_sentence_is_deterministically_prepended():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        DETAIL_PRESERVATION_SENTENCE,
        enforce_detail_preservation_preface,
        split_sections,
        validate_h3_prompt_writer_output,
    )

    without = LEGAL_PROMPT.replace(DETAIL_PRESERVATION_SENTENCE + "\n\n","",1)
    fixed = enforce_detail_preservation_preface(without)
    detailed = dict(split_sections(fixed))["detailed_description"].strip()
    assert detailed.startswith(DETAIL_PRESERVATION_SENTENCE)
    assert detailed.count(DETAIL_PRESERVATION_SENTENCE) == 1
    assert validate_h3_prompt_writer_output(fixed) == fixed.strip()

    unchanged = enforce_detail_preservation_preface(LEGAL_PROMPT)
    assert unchanged.count(DETAIL_PRESERVATION_SENTENCE) == 1


def test_validator_requires_dynamic_preface_and_rejects_speculation():
    from r2v_data_v2.person_replacement.qwen38_h3_prompt_writer import (
        DETAIL_PRESERVATION_SENTENCE,
        validate_h3_prompt_writer_output,
    )

    missing_preface = LEGAL_PROMPT.replace(
        DETAIL_PRESERVATION_SENTENCE + "\n\n"
        "<Subject 1> raises one hand toward <Subject 2> while <Subject 2> remains in place, "
        "with the camera holding a static wide framing throughout.\n\n",
        "",
    )
    with pytest.raises(ValueError,match="canonical preservation sentence"):
        validate_h3_prompt_writer_output(missing_preface)

    missing_camera = LEGAL_PROMPT.replace(
        "with the camera holding a static wide framing throughout.",
        "and both remain visible throughout.",
        1,
    )
    with pytest.raises(ValueError,match="camera or framing"):
        validate_h3_prompt_writer_output(missing_camera)

    speculative = LEGAL_PROMPT.replace(
        "and lifts a hand while the camera remains static.",
        "and lifts a hand as if controlling an invisible force while the camera remains static.",
    )
    with pytest.raises(ValueError,match="speculative language"):
        validate_h3_prompt_writer_output(speculative)


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
