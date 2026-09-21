"""frame0 variant semantic invariants: Qwen -> Boogu -> H3(Video1 + Picture1)."""

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import (
    atomic_json,
    failure_count,
    phase,
    publish_qwen,
    qwen_prepared,
    read_json,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline

TEXTS = {"source_subject_1":"a man in a blue jacket","source_subject_2":"a woman with a cup",
         "shot_description":"They talk at a table.","replacement_subject_1":"an old man",
         "replacement_subject_2":"a young woman"}
BOOGU_ROOT = "/mnt/workspace/litengjie/data"


def case_fixture(root, count=4):
    return [{"case_id":f"case{i}","directory":str(root/f"case{i}"),"row":{"video_path":f"{i}.mp4"},
             "row_sha256":str(i),"source_index":i} for i in range(count)]


def pair_config(cases, clips, variant="frame0", prepare_budget=2):
    return {"cases":cases,"clips_root":str(clips),"qwen_model":str(clips),"seed":42,"pair_id":0,
            "identity":"identity","variant":variant,
            "limits":{case["case_id"]:{"prepare":prepare_budget,"generate":2} for case in cases},
            "boogu_python":f"{BOOGU_ROOT}/venvs/boogu-image/bin/python",
            "boogu_code_root":f"{BOOGU_ROOT}/vendor/Boogu-Image",
            "boogu_model_root":f"{BOOGU_ROOT}/models/Boogu"}


def qwen_marker(case, video):
    publish_qwen(case,{**TEXTS,"case_id":case["case_id"],"row_sha256":case["row_sha256"],
                       "identity":"identity","source":str(video),"seed":42,"frames":124,
                       "width":1376,"height":768,"timeline":{},"aspect_ratio":"16:9"},TEXTS)


class QwenSpy:
    calls: ClassVar = []

    def __init__(self, path):
        QwenSpy.calls.append("load")

    def _load(self):
        pass

    def describe_two(self, video):
        QwenSpy.calls.append("describe")
        return TEXTS["source_subject_1"],TEXTS["source_subject_2"],TEXTS["shot_description"]

    def describe_two_with_performance(self, video):
        pytest.fail("production pair path must not use the 5-field source call")

    def invent_two(self, *args):
        return TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"]

    def invent_two_with_diversity(self, subject1, subject2, cue1, cue2):
        QwenSpy.calls.append("invent")
        return TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"]

    def close(self):
        QwenSpy.calls.append("close")


class FakeBoogu:
    """Minimal stand-in for BooguSubprocessBackend with its real start/edit lifecycle."""

    instances: ClassVar = []

    def __init__(self, settings, *, fail_edit=(), fail_start=False, effective=None):
        self.settings = settings
        self.fail_edit = fail_edit
        self.fail_start = fail_start
        self.effective_instruction = effective
        self.edits = 0
        self.started = 0
        self.closed = 0
        FakeBoogu.instances.append(self)

    def start(self, stderr_log_path):
        self.started += 1
        self.log = Path(stderr_log_path)
        self.env_tmpdir = os.environ.get("TMPDIR")  # proves the cache namespace
        if self.fail_start:
            raise RuntimeError("boogu worker failed to load")

    def edit(self, source_rgb, instruction, width, height, thinking_enabled,
             instruction_rewrite_enabled, seed):
        self.edits += 1
        if self.edits in self.fail_edit:
            raise RuntimeError("boogu edit failed")
        return SimpleNamespace(png_bytes=b"\x89PNG-repainted",
                               original_instruction=instruction,
                               rewritten_instruction=None,
                               effective_instruction=self.effective_instruction or instruction,
                               worker_metadata={"backend":"fake"})

    def close(self):
        self.closed += 1


@contextmanager
def nullcontext():
    yield


def decode_stub(video, directory, width, height):
    """Exact frame 0 decode stand-in; writes the real source_frame0.png artifact."""
    path = Path(directory)/"source_frame0.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b"\x89PNG-source")
    return SimpleNamespace(), path


@pytest.fixture(autouse=True)
def reset_spies(monkeypatch):
    """No ffmpeg/GPU here: mono/stereo passthrough, audio tests live in their own file."""
    from r2v_data_v2.person_replacement import h3_pair_prepare

    monkeypatch.setattr(h3_pair_prepare,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    QwenSpy.calls.clear()
    FakeBoogu.instances.clear()
    yield


# ---------------------------------------------------------------- CLI / identity

def test_cli_variant_choices_and_default(tmp_path):
    from tools.person_replacement.run_h3_pdd_pair_executor import arguments

    clips = tmp_path/"clips"
    clips.mkdir()
    source = tmp_path/"source.jsonl"
    source.write_text('{"video_path":"a.mp4"}\n')
    base = ["--input-jsonl",str(source),"--clips-root",str(clips),"--output-root",str(tmp_path/"out")]
    assert arguments(base).variant == "text"
    assert arguments(base+["--variant","frame0"]).variant == "frame0"
    with pytest.raises(SystemExit):
        arguments(base+["--variant","picture"])
    frame0 = arguments(base+["--variant","frame0"])
    for name in ("boogu_python","boogu_code_root","boogu_model_root"):
        assert str(getattr(frame0,name)).startswith("/mnt/workspace/")  # server defaults, unvalidated


def test_variant_contracts_are_distinct_and_group_size_is_not_semantic(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_executor as module
    from tools.person_replacement.run_h3_pdd_pair_executor import arguments

    clips = tmp_path/"clips"
    clips.mkdir()
    source = tmp_path/"source.jsonl"
    source.write_text('{"video_path":"a.mp4"}\n')
    base = ["--input-jsonl",str(source),"--clips-root",str(clips),"--output-root",str(tmp_path/"out")]
    _,text = module.make_config(arguments(base+["--variant","text"]))
    _,frame0 = module.make_config(arguments(base+["--variant","frame0"]))
    assert text["identity"] != frame0["identity"]
    assert text["identity_details"]["contract"] == "text_two_person_pdd_fsdp2_pair_v33"
    assert frame0["identity_details"]["contract"] == "frame0_two_person_pdd_fsdp2_pair_v18"
    for key in ("boogu_python","boogu_code_root","boogu_model_root"):
        assert key in frame0["identity_details"]["resources"] and key not in text["identity_details"]["resources"]
    _,bigger = module.make_config(arguments(base+["--variant","frame0","--group-size","4","--ulysses-degree","2"]))
    assert bigger["identity"] == frame0["identity"]  # execution strategy is not semantic identity


# ------------------------------------------------------------ text variant path

def test_text_variant_still_publishes_prepared_without_picture(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    (clips/"0.mp4").touch()
    cases = case_fixture(tmp_path,count=1)
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_partition(pair_config(cases,clips,variant="text"),0,lambda _:QwenSpy("model"))
    directory = Path(cases[0]["directory"])/"preparation"
    prepared = read_json(directory/"prepared.json")
    assert prepared["variant"] == "text_two_person"
    assert "reference_image" not in prepared and "Picture 1" not in prepared["prompt"]
    assert qwen_prepared(cases[0]) is None  # text needs no intermediate marker
    assert phase(cases[0]) == "generate"


def test_frame0_qwen_phase_is_durable_but_not_prepared(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    (clips/"0.mp4").touch()
    cases = case_fixture(tmp_path,count=1)
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_partition(pair_config(cases,clips),0,lambda _:QwenSpy("model"))
    directory = Path(cases[0]["directory"])/"preparation"
    assert not (directory/"prepared.json").exists()
    assert phase(cases[0]) == "prepare"
    marker = qwen_prepared(cases[0])
    for name, value in TEXTS.items():
        assert marker[name] == value
        assert (directory/f"{name}.txt").read_text(encoding="utf-8").strip() == value
    for key in ("source","width","height","frames","timeline","aspect_ratio","identity","seed"):
        assert key in marker
    QwenSpy.calls.clear()
    module.prepare_partition(pair_config(cases,clips),0,lambda _:QwenSpy("model"))
    assert QwenSpy.calls == []  # resume must not repeat Qwen


# ------------------------------------------------------------ frame0 preparation

def test_frame0_boogu_publishes_prepared_last_with_provenance(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=1)
    video = tmp_path/"video.mp4"
    video.touch()
    qwen_marker(cases[0],video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    module.prepare_frame0_partition(pair_config(cases,tmp_path),0,lambda settings:FakeBoogu(settings))
    directory = Path(cases[0]["directory"])/"preparation"
    assert (directory/"source_frame0.png").read_bytes() == b"\x89PNG-source"
    repainted = directory/"repainted_frame0.png"
    assert repainted.read_bytes() == b"\x89PNG-repainted"
    prepared = read_json(directory/"prepared.json")
    assert prepared["variant"] == "frame0_two_person"
    assert prepared["reference_mode"] == "frame0_boogu"
    assert prepared["reference_image"] == str(repainted)
    assert phase(cases[0]) == "generate"
    job = prepared["boogu"]
    assert job["model_name"] == "Boogu-Image-0.1-Edit-Turbo"
    assert job["model_revision"]
    assert job["python"] and job["code_root"] and job["model_path"]
    assert job["instruction"] == job["original_instruction"] == job["effective_instruction"]
    assert job["instruction_rewritten"] is False  # rewrite disabled, never silently rewritten
    assert job["num_inference_steps"] == 4 and job["seed"] == 42
    assert (directory/"boogu_prompt.txt").read_text(encoding="utf-8").strip() == job["instruction"]


def test_frame0_provenance_records_effective_instruction(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=1)
    video = tmp_path/"video.mp4"
    video.touch()
    qwen_marker(cases[0],video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    module.prepare_frame0_partition(pair_config(cases,tmp_path),0,
                                    lambda settings:FakeBoogu(settings,effective="rewritten by boogu"))
    job = read_json(Path(cases[0]["directory"])/"preparation/prepared.json")["boogu"]
    assert job["effective_instruction"] == "rewritten by boogu"
    assert job["instruction"] != job["effective_instruction"]
    assert job["instruction_rewritten"] is True


def test_frame0_resume_after_boogu_crash_without_rerunning_qwen(tmp_path, monkeypatch):
    """A Boogu crash leaves nothing committed; the relaunched worker resumes."""
    from r2v_data_v2.person_replacement import h3_pair_frame0 as frame0
    from r2v_data_v2.person_replacement import h3_pair_prepare as prepare

    clips = tmp_path/"clips"
    clips.mkdir()
    (clips/"0.mp4").touch()
    cases = case_fixture(tmp_path,count=1)
    monkeypatch.setattr(prepare,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    prepare.prepare_partition(pair_config(cases,clips),0,lambda _:QwenSpy("model"))
    assert qwen_prepared(cases[0]) is not None
    monkeypatch.setattr(frame0,"_decode_frame_zero",decode_stub)
    directory = Path(cases[0]["directory"])/"preparation"

    class CrashingBoogu(FakeBoogu):
        def edit(self, source_rgb, instruction, width, height, thinking_enabled,
                 instruction_rewrite_enabled, seed):
            raise KeyboardInterrupt("frame0 worker died mid-edit")

    with pytest.raises(KeyboardInterrupt):
        frame0.prepare_frame0_partition(pair_config(cases,tmp_path),0,CrashingBoogu)
    assert not (directory/"prepared.json").exists()
    assert not (directory/"repainted_frame0.png").exists()
    assert qwen_prepared(cases[0]) is not None
    QwenSpy.calls.clear()
    frame0.prepare_frame0_partition(pair_config(cases,tmp_path),0,lambda settings:FakeBoogu(settings))
    assert QwenSpy.calls == []  # Qwen never runs again
    assert (directory/"repainted_frame0.png").read_bytes() == b"\x89PNG-repainted"
    prepared = read_json(directory/"prepared.json")
    assert prepared["variant"] == "frame0_two_person" and phase(cases[0]) == "generate"


def test_frame0_case_failure_records_attempt_and_resumes_with_budget(tmp_path, monkeypatch):
    """Real failures consume attempts durably; nothing is committed before Boogu succeeds."""
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=1)
    video = tmp_path/"video.mp4"
    video.touch()
    qwen_marker(cases[0],video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    directory = Path(cases[0]["directory"])/"preparation"
    module.prepare_frame0_partition(pair_config(cases,tmp_path),0,
                                    lambda settings:FakeBoogu(settings,fail_edit=(1,)))
    assert failure_count(cases[0],"prepare") == 2
    assert phase(cases[0]) == "prepare" and qwen_prepared(cases[0]) is not None
    assert not (directory/"prepared.json").exists()
    module.prepare_frame0_partition(pair_config(cases,tmp_path,prepare_budget=3),0,
                                    lambda settings:FakeBoogu(settings))
    assert (directory/"prepared.json").is_file() and phase(cases[0]) == "generate"


def test_frame0_start_failure_records_attempt_and_resumes_with_budget(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=1)
    video = tmp_path/"video.mp4"
    video.touch()
    qwen_marker(cases[0],video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    config = pair_config(cases,tmp_path)
    module.prepare_frame0_partition(config,0,lambda settings:FakeBoogu(settings,fail_start=True))
    assert failure_count(cases[0],"prepare") == 2 and phase(cases[0]) == "prepare"
    module.prepare_frame0_partition(pair_config(cases,tmp_path,prepare_budget=3),0,
                                    lambda settings:FakeBoogu(settings))
    assert phase(cases[0]) == "generate"


def test_persistent_backend_starts_once_for_many_cases(tmp_path, monkeypatch):
    """Worker 0 owns case0 and case2 and must load Boogu exactly once."""
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    video = tmp_path/"video.mp4"
    video.touch()
    for case in cases:
        qwen_marker(case,video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    config = pair_config(cases,tmp_path)
    config["group_size"] = 2  # worker 0 -> case0, case2; worker 1 -> case1, case3
    module.prepare_frame0_partition(config,0,lambda settings:FakeBoogu(settings))
    backend, = FakeBoogu.instances  # one subprocess, not one per case
    assert backend.started == 1 and backend.edits == 2 and backend.closed == 1
    assert phase(cases[0]) == "generate" and phase(cases[2]) == "generate"
    assert phase(cases[1]) == phase(cases[3]) == "prepare"  # other partition untouched
    for case in (cases[0],cases[2]):
        assert (Path(case["directory"])/"preparation/repainted_frame0.png").is_file()


def test_worker_directories_are_isolated(tmp_path, monkeypatch):
    """Concurrent GPU workers must not share runtime, temp or log namespaces."""
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    video = tmp_path/"video.mp4"
    video.touch()
    for case in cases:
        qwen_marker(case,video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    config = pair_config(cases,tmp_path)
    config["group_size"] = 2
    for worker in (0,1):
        module.prepare_frame0_partition(config,worker,lambda settings:FakeBoogu(settings))
    first, second = FakeBoogu.instances
    assert first.log.name == "boogu_worker-0.log" and second.log.name == "boogu_worker-1.log"
    assert first.log.parent != second.log.parent
    assert first.settings.temporary_root != second.settings.temporary_root
    assert first.env_tmpdir != second.env_tmpdir
    for worker, backend in ((0,first),(1,second)):
        root = tmp_path/f"frame0-worker-{worker}"
        assert (root/"runtime").is_dir()
        assert backend.settings.temporary_root == root/"boogu_tmp"
    assert (tmp_path/"frame0-worker-0"/"runtime") != (tmp_path/"frame0-worker-1"/"runtime")
    # case-level artifacts keep their original location
    for case in cases:
        directory = Path(case["directory"])/"preparation"
        assert (directory/"source_frame0.png").is_file()
        assert (directory/"repainted_frame0.png").is_file()
        assert (directory/"boogu_job.json").is_file()
        assert (directory/"prepared.json").is_file()


def test_resume_reuses_the_same_worker_namespace(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=2)
    video = tmp_path/"video.mp4"
    video.touch()
    for case in cases:
        qwen_marker(case,video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    config = pair_config(cases,tmp_path)
    config["group_size"] = 2

    class CrashingBoogu(FakeBoogu):
        def edit(self, source_rgb, instruction, width, height, thinking_enabled,
                 instruction_rewrite_enabled, seed):
            raise KeyboardInterrupt("frame0 worker died mid-edit")

    with pytest.raises(KeyboardInterrupt):
        module.prepare_frame0_partition(config,0,CrashingBoogu)
    first = tmp_path/"frame0-worker-0"/"runtime"
    assert first.is_dir()
    module.prepare_frame0_partition(config,0,lambda settings:FakeBoogu(settings))
    assert first.is_dir()  # stable namespace, never a fresh random directory
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith("frame0-worker-")] == [
        "frame0-worker-0"]
    assert len(FakeBoogu.instances) == 2 and phase(cases[0]) == "generate"


def test_backend_is_rebuilt_after_it_breaks_mid_partition(tmp_path, monkeypatch):
    """A dead subprocess must not permanently disable the rest of the partition."""
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    video = tmp_path/"video.mp4"
    video.touch()
    for case in cases:
        qwen_marker(case,video)
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    config = pair_config(cases,tmp_path,prepare_budget=1)  # one attempt per case
    config["group_size"] = 2  # worker 0 owns case0 and case2
    backends = []

    def factory(settings):
        # The first backend breaks on its only edit, like a subprocess that died.
        broken = not backends
        backend = FakeBoogu(settings,fail_edit=(1,) if broken else ())
        backends.append(backend)
        return backend

    module.prepare_frame0_partition(config,0,factory)
    assert len(backends) == 2  # rebuilt for the next eligible case
    assert phase(cases[0]) == "prepare"  # broken case stays case-local
    assert phase(cases[2]) == "generate"
    assert backends[0].closed == 1 and backends[1].edits == 1


def test_frame0_worker_keeps_fixed_partition_and_loads_nothing_when_idle(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    video = tmp_path/"video.mp4"
    video.touch()
    for case in cases:
        qwen_marker(case,video)
    seen = []

    def fake_prepare(case, config, backend):
        assert isinstance(backend,module.Frame0BooguSession)
        seen.append(case["case_id"])
        directory = Path(case["directory"])/"preparation"
        directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/"prepared.json",{"variant":"frame0_two_person"})

    monkeypatch.setattr(module,"prepare_case",fake_prepare)
    config = pair_config(cases,tmp_path,prepare_budget=1)
    config["group_size"] = 4
    for worker in range(4):
        module.prepare_frame0_partition(config,worker,lambda settings:FakeBoogu(settings))
    assert seen == [f"case{i}" for i in range(4)]
    assert FakeBoogu.instances == []  # nothing to edit means no Boogu subprocess


def test_shared_worker_directory_and_log_are_partition_local():
    from r2v_data_v2.person_replacement.h3_pair_frame0 import (
        Frame0BooguSession,
        shared_worker_directory,
    )

    cases = case_fixture(Path("/tmp/shard-000000"),count=2)
    assert shared_worker_directory(cases) == Path("/tmp/shard-000000")
    assert Frame0BooguSession({},Path("/tmp/shard-000000"),worker=2).log_path.name == "boogu_worker-2.log"


def test_boogu_worker_keeps_partition_gpu_not_physical_zero(monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3")
    settings = module.boogu_config({"boogu_python":"/a","boogu_code_root":"/b","boogu_model_root":"/c"},
                                   Path("/tmp/frame0"))
    assert settings.cuda_visible_devices == "3"  # never remap every worker onto GPU 0
    assert settings.device == "cuda:0"  # subprocess-local index stays canonical


def test_runtime_cache_namespace_is_retry_safe(tmp_path):
    """The Boogu cache namespace is re-entered on every resume attempt."""
    from r2v_data_v2.person_replacement.h3_pair_frame0 import (
        boogu_runtime,
        frame0_runtime_environment,
    )

    cache = tmp_path/"frame0_runtime"
    frame0_runtime_environment(cache)
    frame0_runtime_environment(cache)  # second attempt after a failed edit must not raise
    with boogu_runtime(cache):
        pass


# -------------------------------------------------------------------- references

def test_distributed_references_video_then_image(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.h3_pdd_distributed import PersistentPDD

    video = tmp_path/"video.mp4"
    image = tmp_path/"frame0.png"
    video.touch()
    image.touch()

    def reference(kind):
        return SimpleNamespace(from_file=lambda path:(kind,path))

    modules = __import__("sys").modules
    monkeypatch.setitem(modules,"diffusers",SimpleNamespace())
    monkeypatch.setitem(modules,"diffusers.modular_pipelines",SimpleNamespace())
    monkeypatch.setitem(modules,"diffusers.modular_pipelines.minimax_h3",
                        SimpleNamespace(MiniMaxH3VideoReference=reference("video"),
                                        MiniMaxH3ImageReference=reference("image")))
    backend = PersistentPDD.__new__(PersistentPDD)
    text = backend.prepare({"source":str(video)})
    assert [kind for kind,_ in text] == ["video"]
    both = backend.prepare({"source":str(video),"reference_image":str(image)})
    assert [kind for kind,_ in both] == ["video","image"]
    assert [path for _,path in both] == [str(video),str(image)]
    arm = SimpleNamespace(index=1,arm=lambda value:None)
    backend.device = "cuda:0"
    backend.transformer_ref = SimpleNamespace(_pdd_step_arm=arm)
    monkeypatch.setitem(modules,"torch",SimpleNamespace(
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda _:None),
        Generator=lambda:SimpleNamespace(manual_seed=lambda seed:seed),
        no_grad=nullcontext))
    infer = {}

    class Pipeline:
        transformer_ref = SimpleNamespace(_pdd_step_arm=arm)
        def __call__(self, **kwargs):
            infer.update(kwargs)
            return "result"

    backend.pipeline = Pipeline()
    backend.infer({"prompt":"p","frames":124,"width":1376,"height":768,"seed":42},both)
    assert infer["references"] == both and arm.index == 0


def test_generation_manifest_variant_and_fail_closed(tmp_path):
    from r2v_data_v2.person_replacement.h3_pair_generation import reference_manifest

    video = tmp_path/"video.mp4"
    video.touch()
    variant, references = reference_manifest({"variant":"text_two_person","source":str(video)})
    assert variant == "text_two_person" and references == [
        {"type":"video","label":"Video 1","path":str(video)}]
    image = tmp_path/"repainted.png"
    image.touch()
    variant, references = reference_manifest({"variant":"frame0_two_person","source":str(video),
                                              "reference_mode":"frame0_boogu","reference_image":str(image)})
    assert variant == "frame0_two_person" and references == [
        {"type":"video","label":"Video 1","path":str(video)},
        {"type":"image","label":"Picture 1","path":str(image)}]
    with pytest.raises(ValueError,match="reference image"):
        reference_manifest({"variant":"frame0_two_person","source":str(video),
                            "reference_mode":"frame0_boogu","reference_image":None})
    with pytest.raises(ValueError,match="reference image"):
        reference_manifest({"variant":"frame0_two_person","source":str(video),
                            "reference_mode":"frame0_boogu","reference_image":str(tmp_path/"missing.png")})
    with pytest.raises(ValueError,match="reference_mode"):
        reference_manifest({"variant":"frame0_two_person","source":str(video),"reference_image":str(image)})
    with pytest.raises(ValueError,match="variant"):
        reference_manifest({"source":str(video)})


def test_frame0_generation_fails_closed_instead_of_text_fallback(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_generation as module

    case = {"case_id":"case0","directory":str(tmp_path/"case0"),"row_sha256":"0"}
    atomic_json(Path(case["directory"])/"preparation/prepared.json",{
        "case_id":"case0","row_sha256":"0","identity":"same","source":str(tmp_path/"v.mp4"),
        "variant":"frame0_two_person","reference_mode":"frame0_boogu",
        "reference_image":str(tmp_path/"missing.png"),"prompt":"p","frames":124,
        "width":1376,"height":768,"seed":42})
    config = {"cases":[case],"identity":"same","pair_id":0,"limits":{"case0":{"prepare":2,"generate":2}}}
    with pytest.raises(ValueError,match="reference image"):
        module._next_job(config)
    assert phase(case) == "generate"  # never silently downgraded to text-only


# ------------------------------------------------------------------------ prompt

def test_frame0_prompt_semantics_and_no_shot_description():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    shot = "They raise the cups in exactly this order."
    prompt = build_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],shot,
                          TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"],frame0=True)
    assert "<Picture 1>" in prompt  # frame0 adds an appearance anchor only
    assert "<Picture 1> is the edited first-frame appearance anchor for <Subject 1>" in prompt
    assert "It does not replace <Video 1> as the authority" in prompt
    assert "<Video 1>" in prompt
    assert "<Subject 3>" not in prompt and "<Subject 4>" not in prompt
    assert "partially_preserved" in prompt and "fully_preserved" in prompt
    assert "attribute_transfer" not in prompt
    assert shot not in prompt  # Qwen shot description is provenance only
    assert "[video editing + keyframe completion]" in prompt
    assert "Use <Video 1> directly as the source video" in prompt
    text = build_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],shot,
                        TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"])
    assert "<Picture 1>" not in text


def test_boogu_frame0_prompt_is_short_and_binding_exact():
    from r2v_data_v2.person_replacement.h3_two_person import boogu_frame0_prompt

    instruction = boogu_frame0_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],
                                      TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"])
    lines = [line for line in instruction.splitlines() if line.strip()]
    assert lines[0] == f"Replace the person matching {TEXTS['source_subject_1']} " \
                       f"with {TEXTS['replacement_subject_1']}."
    assert lines[1] == f"Replace the person matching {TEXTS['source_subject_2']} " \
                       f"with {TEXTS['replacement_subject_2']}."
    assert "Do not add a person who is not visible in the image." in instruction
    assert "two visible people" not in instruction
    assert "->" not in instruction  # no sequential arrow form
    for banned in ("body orientation","gaze","mouth state","scale","occlusion","lighting",
                   "framing","object state","identity","hair","clothing"):
        assert banned not in instruction.lower()
    assert len(instruction.split()) <= 55  # do not re-inflate it later


# --------------------------------------------------------------------- execution

def test_execute_phases_runs_boogu_only_after_qwen_exits(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_executor as module

    case = {"case_id":"case0","directory":str(tmp_path/"shard-000000/case0"),"row_sha256":"0"}
    config = {"cases":[case],"limits":{"case0":{"prepare":2,"generate":2}},"pair_id":0,
              "identity":"identity","h3_python":"python","variant":"frame0","group_size":4}
    calls = []
    monkeypatch.setattr(module,"run_children",
                        lambda specs,lock_fd,**kwargs:calls.append([spec["command"] for spec in specs])
                              or [0]*len(specs))
    module.execute_phases(config,tmp_path/"shard-000000","0,1,2,3",123,prepare_only=True)
    assert len(calls) == 2 and len(calls[0]) == len(calls[1]) == 4
    assert all("h3_pair_prepare_worker.py" in command[1] for command in calls[0])
    assert all("h3_pair_frame0_worker.py" in command[1] for command in calls[1])
    assert [command[command.index("--worker")+1] for command in calls[1]] == ["0","1","2","3"]
