"""Boogu first-frame preparation phase for the parallel pair executor.

Runs only after every Qwen worker has exited, one worker per GPU, and is the
only stage that publishes preparation/prepared.json for the frame0 variant.
One Boogu subprocess serves every case in a worker's partition.
Heavy imports stay inside functions so the parent process stays lightweight.
"""

import os
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from .h3_pair_state import (
    atomic_json,
    begin_attempt,
    eligible,
    fail_attempt,
    partition,
    publish_prepared,
    qwen_prepared,
)
from .h3_two_person import boogu_frame0_prompt, build_prompt

VARIANT = "frame0_two_person"
REFERENCE_MODE = "frame0_boogu"
SOURCE_FRAME = "source_frame0.png"
REPAINTED_FRAME = "repainted_frame0.png"
BOOGU_CACHE_FOLDERS = {"HF_HOME":"hf", "HF_HUB_CACHE":"hf/hub", "HUGGINGFACE_HUB_CACHE":"hf/hub",
                       "TRANSFORMERS_CACHE":"hf/transformers", "TORCH_HOME":"torch",
                       "TORCHINDUCTOR_CACHE_DIR":"inductor", "TRITON_CACHE_DIR":"triton",
                       "XDG_CACHE_HOME":"cache", "TMPDIR":"tmp"}


def frame0_runtime_environment(directory):
    """Local retry-safe cache namespace for Boogu.

    The shared helper cannot be reused here: it creates the runtime root with a
    strict mkdir, so a resumed frame0 attempt after a Boogu failure would crash
    instead of retrying. This local copy is idempotent by design.
    """
    cache = Path(directory)
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV"):
        env.pop(key,None)
    env.update({"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                "HF_HUB_DISABLE_TELEMETRY":"1", "PYTHONDONTWRITEBYTECODE":"1"})
    for key, folder in BOOGU_CACHE_FOLDERS.items():
        path = cache/folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


@contextmanager
def boogu_runtime(directory):
    """Boogu owns its own cache namespace; restore the pair worker env afterwards."""
    previous = os.environ.copy()
    environment = frame0_runtime_environment(directory)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def shared_worker_directory(cases):
    """Cases of one pair always live under a single shard root."""
    return Path(cases[0]["directory"]).parent


def boogu_config(config, directory):
    from r2v_data_v2.v3.reference_edit_boogu import BooguWorkerConfig

    # CUDA_VISIBLE_DEVICES already names this worker's single physical GPU.
    # Overwriting it with "0" would remap every worker onto physical GPU 0.
    return BooguWorkerConfig(
        python_executable=Path(config["boogu_python"]),
        code_root=Path(config["boogu_code_root"]),
        model_path=Path(config["boogu_model_root"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") or "0",
        temporary_root=Path(directory)/"boogu_tmp",
    )


class Frame0BooguSession:
    """One persistent Boogu subprocess per GPU worker, never shared across GPUs.

    Boogu starts lazily on the first real edit and is reused for every later
    case in the same partition. If an edit breaks the subprocess, the session
    drops it so the next eligible case rebuilds a fresh one.
    """

    def __init__(self, config, directory, worker=None, backend_factory=None):
        self.config = config
        self.directory = Path(directory)
        self.worker = worker
        self.backend_factory = backend_factory
        self.start_count = 0
        self.model_name = None
        self.settings = None
        self._backend = None

    @property
    def log_path(self):
        suffix = "" if self.worker is None else f"-{self.worker}"
        return self.directory/f"boogu_worker{suffix}.log"

    def _build(self):
        from r2v_data_v2.v3.reference_edit_boogu import (
            BOOGU_MODEL_NAME,
            BooguSubprocessBackend,
        )

        self.settings = boogu_config(self.config,self.directory)
        backend = self.backend_factory(self.settings) if self.backend_factory else BooguSubprocessBackend(self.settings)
        backend.start(stderr_log_path=self.log_path)
        self.model_name = BOOGU_MODEL_NAME
        self.start_count += 1
        self._backend = backend
        return backend

    def start(self):
        return self._backend if self._backend is not None else self._build()

    def _discard(self):
        backend, self._backend = self._backend, None
        if backend is None:
            return
        # A failed subprocess usually terminated itself already: discarding is
        # the intended outcome, and the next case rebuilds when needed.
        with suppress(Exception):
            backend.close()

    def edit(self, **request):
        backend = self.start()
        try:
            return backend.edit(**request)
        except Exception:
            self._discard()  # a case-local failure must not poison later cases
            raise

    def close(self):
        self._discard()

    def provenance(self):
        settings = self.settings
        return {"model_name":self.model_name,
                "model_revision":getattr(settings,"model_revision",None),
                "python":str(getattr(settings,"python_executable","")),
                "code_root":str(getattr(settings,"code_root","")),
                "model_path":str(getattr(settings,"model_path","")),
                "cuda_visible_devices":getattr(settings,"cuda_visible_devices",None),
                "worker_log":str(self.log_path)}


def _decode_frame_zero(video, directory, width, height):
    from PIL import Image, ImageOps

    from r2v_data_v2.v3.frames import OpenCvFrameDecoder

    # Exact decoded index 0. No search, visibility gate or source modification.
    frame = OpenCvFrameDecoder().decode_indices(video,[0])[0].image
    image = ImageOps.fit(frame.convert("RGB"),(width,height),method=Image.Resampling.LANCZOS,
                         centering=(0.5,0.5))
    path = directory/SOURCE_FRAME
    image.save(path,format="PNG")
    return image, path


def edit_frame_zero(video, directory, width, height, texts, session):
    """Boogu replaces only the described people; pose, layout and props stay exact."""
    started = time.monotonic()
    image, source_path = _decode_frame_zero(video,directory,width,height)
    instruction = boogu_frame0_prompt(texts["source_subject_1"],texts["source_subject_2"],
                                      texts["replacement_subject_1"],texts["replacement_subject_2"])
    (directory/"boogu_prompt.txt").write_text(instruction+"\n",encoding="utf-8")
    result = session.edit(source_rgb=image,instruction=instruction,width=width,height=height,
                          thinking_enabled=False,instruction_rewrite_enabled=False,seed=42)
    output = directory/REPAINTED_FRAME
    output.write_bytes(result.png_bytes)
    original = getattr(result,"original_instruction",instruction)
    rewritten = getattr(result,"rewritten_instruction",None)
    effective = getattr(result,"effective_instruction",original)
    job = {**session.provenance(),"source_frame_index":0,"source_video":str(video),
           "source_frame":str(source_path),"repainted_frame":str(output),
           "width":width,"height":height,"seed":42,
           "thinking_enabled":False,"instruction_rewrite_enabled":False,"num_inference_steps":4,
           "text_guidance_scale":1.0,"image_guidance_scale":1.0,"use_dmd_student_inference":True,
           "frame0_transform":"cover_resize_center_crop","instruction":instruction,
           "original_instruction":original,"rewritten_instruction":rewritten,
           "effective_instruction":effective,
           # With rewrite disabled the effective instruction must stay untouched.
           "instruction_rewritten":bool(effective != original),
           "worker_metadata":getattr(result,"worker_metadata",None),
           "wall_seconds":time.monotonic()-started}
    atomic_json(directory/"boogu_job.json",job)
    return output, job


def prepare_case(case, config, session):
    marker = qwen_prepared(case)
    if marker is None:
        raise ValueError("frame0 preparation requires the durable Qwen marker")
    directory = Path(case["directory"])/"preparation"
    video = Path(marker["source"])
    width, height = int(marker["width"]), int(marker["height"])
    names = ("source_subject_1","source_subject_2","shot_description",
             "replacement_subject_1","replacement_subject_2")
    texts = {name:marker[name] for name in names}
    for name, value in texts.items():
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Qwen marker missing field: {name}")
    reference, job = edit_frame_zero(video,directory,width,height,texts,session)
    if not reference.is_file() or reference.stat().st_size == 0:
        raise ValueError("Boogu did not produce a usable repainted first frame")
    prompt = build_prompt(texts["source_subject_1"],texts["source_subject_2"],texts["shot_description"],
                          texts["replacement_subject_1"],texts["replacement_subject_2"],frame0=True)
    publish_prepared(case,{**marker,"prompt":prompt,"variant":VARIANT,"reference_mode":REFERENCE_MODE,
                           "reference_image":str(reference),"boogu":job},
                     {**texts,"h3_prompt":prompt})
    return reference


def prepare_frame0_partition(config, worker, backend_factory=None):
    cases = partition(config["cases"],worker,config.get("group_size",2))
    limits = config["limits"]
    # Only Qwen-marked cases that are not yet committed; Qwen has already exited.
    cases = [case for case in cases if qwen_prepared(case) is not None
             and eligible(case,"prepare",limits[case["case_id"]]["prepare"])]
    if not cases:
        return
    directory = shared_worker_directory(cases)
    with boogu_runtime(directory/"frame0_runtime"):
        session = Frame0BooguSession(config,directory,worker=worker,backend_factory=backend_factory)
        try:
            for case in cases:
                while eligible(case,"prepare",limits[case["case_id"]]["prepare"]):
                    attempt = begin_attempt(case,"prepare",{"pair_id":config["pair_id"],"frame0_worker":worker})
                    try:
                        prepare_case(case,config,session)
                    except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                        fail_attempt(case,attempt,exc)
        finally:
            session.close()
