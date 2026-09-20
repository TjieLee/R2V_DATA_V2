"""frame0 variant semantic invariants: Qwen -> Boogu -> H3(Video1 + Picture1)."""

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import (
    atomic_json,
    phase,
    publish_qwen,
    qwen_prepared,
    read_json,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline

TEXTS = {"source_subject_1":"a man in a blue jacket","source_subject_2":"a woman with a cup",
         "shot_description":"They talk at a table.","replacement_subject_1":"an old man",
         "replacement_subject_2":"a young woman"}


def case_fixture(tmp_path, name="case0", count=4):
    cases = []
    for i in range(count):
        cases.append({"case_id":f"case{i}","directory":str(tmp_path/f"case{i}"),
                      "row":{"video_path":f"{i}.mp4"},"row_sha256":str(i),"source_index":i})
    return cases


def qwen_config(cases, clips, variant="frame0"):
    return {"cases":cases,"clips_root":str(clips),"qwen_model":str(clips),"seed":42,"pair_id":0,
            "identity":"identity","variant":variant,
            "limits":{case["case_id"]:{"prepare":2,"generate":2} for case in cases}}


class FakeQwen:
    def __init__(self, path):
        self.calls = []
    def _load(self):
        pass
    def describe_two(self, video):
        return TEXTS["source_subject_1"],TEXTS["source_subject_2"],TEXTS["shot_description"]
    def invent_two(self, *args):
        return TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"]
    def close(self):
        pass


class FakeBoogu:
    """Stands in for BooguSubprocessBackend; still exercises publish ordering."""

    instances: ClassVar = []

    def __init__(self, config, directory=None):
        self.config = config
        self.directory = directory
        self.started = False
        FakeBoogu.instances.append(self)

    def start(self, stderr_log_path):
        self.started = True
        self.log = stderr_log_path

    def edit(self, source_rgb, instruction, width, height, thinking_enabled,
             instruction_rewrite_enabled, seed):
        self.instruction = instruction
        self.width, self.height, self.seed = width, height, seed
        return SimpleNamespace(png_bytes=b"\x89PNG-repainted", worker_metadata={"backend":"fake"})

    def close(self):
        self.closed = True


def decode_stub(video, directory, width, height):
    """Exact frame 0 decode stand-in; writes the real source_frame0.png artifact."""
    path = Path(directory)/"source_frame0.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b"\x89PNG-source")
    return SimpleNamespace(), path


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
    assert text["identity_details"]["contract"] == "text_two_person_pdd_fsdp2_pair_v14"
    assert frame0["identity_details"]["contract"] == "frame0_two_person_pdd_fsdp2_pair_v14"
    for key in ("boogu_python","boogu_code_root","boogu_model_root"):
        assert key in frame0["identity_details"]["resources"] and key not in text["identity_details"]["resources"]
    _,bigger = module.make_config(arguments(base+["--variant","frame0","--group-size","4","--ulysses-degree","2"]))
    assert bigger["identity"] == frame0["identity"]  # execution strategy is not semantic identity


def test_text_variant_still_publishes_prepared_without_picture(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    (clips/"0.mp4").touch()
    cases = case_fixture(tmp_path,count=1)
    config = qwen_config(cases,clips,variant="text")
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_partition(config,0,lambda _:FakeQwen("model"))
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
    config = qwen_config(cases,clips,variant="frame0")
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    calls = []
    def factory(path):
        calls.append(path)
        return FakeQwen(path)
    module.prepare_partition(config,0,factory)
    directory = Path(cases[0]["directory"])/"preparation"
    assert not (directory/"prepared.json").exists()
    assert phase(cases[0]) == "prepare"
    marker = qwen_prepared(cases[0])
    assert marker is not None
    for name, value in TEXTS.items():
        assert marker[name] == value
        assert (directory/f"{name}.txt").read_text(encoding="utf-8").strip() == value
    for key in ("source","width","height","frames","timeline","aspect_ratio","identity","seed"):
        assert key in marker
    calls.clear()
    module.prepare_partition(config,0,factory)  # resume must not repeat Qwen
    assert calls == []


def test_frame0_boogu_phase_publishes_prepared_last(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    for case in cases:
        publish_qwen(case,{**TEXTS,"case_id":case["case_id"],"row_sha256":case["row_sha256"],
                           "identity":"identity","source":str(tmp_path/"video.mp4"),"seed":42,
                           "frames":124,"width":1376,"height":768,"timeline":{},"aspect_ratio":"16:9"},TEXTS)
    (tmp_path/"video.mp4").touch()
    config = qwen_config(cases,tmp_path,variant="frame0")
    config.update(group_size=4,
                  boogu_python="/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python",
                  boogu_code_root="/mnt/workspace/litengjie/data/vendor/Boogu-Image",
                  boogu_model_root="/mnt/workspace/litengjie/data/models/Boogu")
    FakeBoogu.instances = []
    monkeypatch.setattr(module,"_decode_frame_zero",decode_stub)
    monkeypatch.setattr(module,"boogu_config",lambda config,directory:SimpleNamespace(
        python_executable=Path(config["boogu_python"]),code_root=Path(config["boogu_code_root"]),
        model_path=Path(config["boogu_model_root"]),cuda_visible_devices="0",
        temporary_root=directory/"boogu_tmp"))
    monkeypatch.setattr(module,"boogu_runtime",lambda directory:nullcontext())
    for worker in range(4):  # one Boogu worker per GPU, fixed partition
        module.prepare_frame0_partition(config,worker,
                                        lambda settings,directory=None:FakeBoogu(settings,directory))
    for case in cases:
        directory = Path(case["directory"])/"preparation"
        assert (directory/"source_frame0.png").exists()
        repainted = directory/"repainted_frame0.png"
        assert repainted.read_bytes() == b"\x89PNG-repainted"
        prepared = read_json(directory/"prepared.json")
        assert prepared["variant"] == "frame0_two_person"
        assert prepared["reference_mode"] == "frame0_boogu"
        assert prepared["reference_image"] == str(repainted)
        assert prepared["boogu"]["num_inference_steps"] == 4
        assert phase(case) == "generate"
    assert len(FakeBoogu.instances) == 4


def test_frame0_resume_reruns_only_boogu_and_skips_prepared(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as frame0
    from r2v_data_v2.person_replacement import h3_pair_prepare as prepare

    clips = tmp_path/"clips"
    clips.mkdir()
    (clips/"0.mp4").touch()
    cases = case_fixture(tmp_path,count=1)
    config = qwen_config(cases,clips,variant="frame0")
    config.update(boogu_python="/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python",
                  boogu_code_root="/mnt/workspace/litengjie/data/vendor/Boogu-Image",
                  boogu_model_root="/mnt/workspace/litengjie/data/models/Boogu")
    monkeypatch.setattr(prepare,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    qwen_calls = []
    monkeypatch.setattr(frame0,"_decode_frame_zero",decode_stub)
    monkeypatch.setattr(frame0,"boogu_config",lambda config,directory:SimpleNamespace(
        python_executable=Path(config["boogu_python"]),code_root=Path(config["boogu_code_root"]),
        model_path=Path(config["boogu_model_root"]),cuda_visible_devices="0",
        temporary_root=directory/"boogu_tmp"))
    monkeypatch.setattr(frame0,"boogu_runtime",lambda directory:nullcontext())
    backend = lambda settings,directory=None:FakeBoogu(settings,directory)
    prepare.prepare_partition(config,0,lambda _:FakeQwen("model"))  # Qwen succeeded
    frame0.prepare_frame0_partition(config,0,backend)
    assert phase(cases[0]) == "generate"
    qwen_calls.clear()
    prepare.prepare_partition(config,0,lambda _:FakeQwen("model"))
    frame0.prepare_frame0_partition(config,0,backend)  # prepared: neither phase reruns
    assert phase(cases[0]) == "generate"


def test_frame0_worker_uses_fixed_partition_one_gpu_each(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    cases = case_fixture(tmp_path,count=4)
    seen = []
    def fake_prepare(case, config, backend_factory=None):
        seen.append(case["case_id"])
        directory = Path(case["directory"])/"preparation"
        directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/"prepared.json",{"variant":"frame0_two_person"})
    monkeypatch.setattr(module,"prepare_case",fake_prepare)
    monkeypatch.setattr(module,"qwen_prepared",lambda case:{"source":"v.mp4","width":1376,"height":768})
    config = {"cases":cases,"group_size":4,"pair_id":0,
              "limits":{case["case_id"]:{"prepare":1,"generate":2} for case in cases}}
    for worker in range(4):
        module.prepare_frame0_partition(config,worker)
    assert seen == [f"case{i}" for i in range(4)]  # fixed partition, one case per GPU


def test_distributed_references_video_then_image(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.h3_pdd_distributed import PersistentPDD

    created = []
    video = tmp_path/"video.mp4"
    image = tmp_path/"frame0.png"
    video.touch()
    image.touch()

    def reference(kind):
        def from_file(path):
            created.append((kind,path))
            return (kind,path)
        return SimpleNamespace(from_file=from_file)

    monkeypatch.setitem(__import__("sys").modules,"diffusers",SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules,"diffusers.modular_pipelines",SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules,"diffusers.modular_pipelines.minimax_h3",
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
    monkeypatch.setitem(__import__("sys").modules,"torch",SimpleNamespace(
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


def test_frame0_generation_fails_closed_instead_of_text_fallback(tmp_path, monkeypatch):
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


def test_frame0_prompt_semantics_and_no_shot_description():
    from r2v_data_v2.person_replacement.h3_two_person import (
        boogu_frame0_prompt,
        build_prompt,
    )

    shot = "They raise the cups in exactly this order."
    prompt = build_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],shot,
                          TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"],frame0=True)
    assert "<Picture 1>" in prompt
    assert "appearance from <Picture 1>" in prompt
    assert "pose, motion and timing driver" in prompt
    assert "<Video 1>" in prompt
    assert "attribute_transfer" in prompt and "fully_preserved" in prompt
    assert shot not in prompt  # Qwen shot description is provenance only
    assert "[video editing + keyframe completion]" in prompt
    text = build_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],shot,
                        TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"])
    assert "<Picture 1>" not in text
    instruction = boogu_frame0_prompt(TEXTS["source_subject_1"],TEXTS["source_subject_2"],
                                      TEXTS["replacement_subject_1"],TEXTS["replacement_subject_2"])
    assert "->" not in instruction  # natural language, not the sequential arrow form
    assert "Do not add a person who is not visible in the source frame." in instruction
    assert "timing" not in instruction.lower()  # still-image edit only


def test_execute_phases_runs_boogu_only_after_qwen_exits(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_executor as module

    case = {"case_id":"case0","directory":str(tmp_path/"shard-000000/case0"),"row_sha256":"0"}
    config = {"cases":[case],"limits":{"case0":{"prepare":2,"generate":2}},"pair_id":0,
              "identity":"identity","h3_python":"python","variant":"frame0","group_size":4}
    calls = []
    def children(specs, lock_fd, **kwargs):
        commands = [spec["command"] for spec in specs]
        calls.append(commands)
        if len(calls) == 1:
            for case_dir in (Path(case["directory"]),):
                (case_dir/"preparation").mkdir(parents=True,exist_ok=True)
        return [0]*len(specs)
    monkeypatch.setattr(module,"run_children",children)
    module.execute_phases(config,tmp_path/"shard-000000","0,1,2,3",123,prepare_only=True)
    assert len(calls) == 2 and len(calls[0]) == len(calls[1]) == 4
    assert all("h3_pair_prepare_worker.py" in command[1] for command in calls[0])
    assert all("h3_pair_frame0_worker.py" in command[1] for command in calls[1])
    assert [spec_env for spec_env in []] == []
    # Every Boogu worker gets exactly one GPU, same fixed set as Qwen.
    assert [command[command.index("--worker")+1] for command in calls[1]] == ["0","1","2","3"]


def test_boogu_worker_keeps_partition_gpu_not_physical_zero(monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_frame0 as module

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3")
    captured = {}
    class Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setitem(__import__("sys").modules,"r2v_data_v2.v3.reference_edit_boogu",
                        SimpleNamespace(BooguWorkerConfig=Config))
    module.boogu_config({"boogu_python":"/a","boogu_code_root":"/b","boogu_model_root":"/c"},
                        Path("/tmp/frame0"))
    assert captured["cuda_visible_devices"] == "3"  # never remap every worker onto GPU 0


class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False
