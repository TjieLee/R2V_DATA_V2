from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from r2v_data_v2.v3.qwen_image21_backend import (
    QwenImage21SubprocessBackend,
    QwenImage21WorkerConfig,
)
from tools import run_v3_qwen_image21_worker as worker


class _FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _FakeGenerator:
        self.seed = seed
        return self


class _FakeTorch:
    Generator = _FakeGenerator

    @staticmethod
    def manual_seed(seed: int) -> None:
        assert seed >= 0

    @staticmethod
    def no_grad():
        from contextlib import nullcontext

        return nullcontext()


class _FakeInputs(dict):
    def to(self, device: str) -> _FakeInputs:
        self["device"] = device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None
        self.tokenizer = SimpleNamespace(decode=self.decode, eos_token_id=7)

    @staticmethod
    def create_mm_token_type_ids(input_ids: object) -> list[int]:
        assert input_ids == [[1, 2]]
        return [0, 1]

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object):
        self.messages = messages
        assert kwargs["enable_thinking"] is True
        return _FakeInputs(input_ids=[[1, 2]])

    @staticmethod
    def decode(tokens: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert tokens == [3, 4]
        return '<think>reason</think>{"rewritten_prompt":"add soft light","wh_ratio":"16:9","ratio_follow":""}'


class _FakePE:
    device = "cuda:0"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.calls += 1
        assert kwargs["device"] == "cuda:0"
        assert kwargs["mm_token_type_ids"] == [0, 1]
        assert kwargs["pad_token_id"] == 7
        return [[1, 2, 3, 4]]


class _FakePipe:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(
            images=[
                Image.new("RGBA", (kwargs["width"], kwargs["height"]), (3, 4, 5, 255))
            ]
        )


def _request(
    tmp_path: Path, *, request_id: str, rewrite: bool, seed: int
) -> dict[str, object]:
    input_path = tmp_path / f"{request_id}-in.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(input_path)
    return {
        "schema_version": 1,
        "type": "edit",
        "request_id": request_id,
        "input_image_path": str(input_path),
        "output_image_path": str(tmp_path / f"{request_id}-out.png"),
        "instruction": "change the light",
        "thinking_enabled": False,
        "instruction_rewrite_enabled": rewrite,
        "width": 64,
        "height": 32,
        "seed": seed,
    }


def test_loaded_worker_skips_pe_when_rewrite_disabled(tmp_path: Path) -> None:
    pipe, pe, processor = _FakePipe(), _FakePE(), _FakeProcessor()
    runtime = worker.LoadedModels(pipe, pe, processor, "official system", _FakeTorch())

    response = worker.run_loaded_request(
        _request(tmp_path, request_id="a", rewrite=False, seed=19),
        runtime=runtime,
        device="cuda:0",
        default_seed=0,
        num_inference_steps=40,
    )

    assert pe.calls == 0
    assert processor.messages is None
    assert response["effective_instruction"] == "change the light"
    assert response["rewritten_instruction"] is None
    assert response["returned_size"] == [64, 32]
    assert response["seed"] == 19
    assert pipe.calls[0]["prompt"] == "change the light"
    assert pipe.calls[0]["image"].mode == "RGB"
    assert pipe.calls[0]["generator"].seed == 19
    assert pipe.calls[0]["num_inference_steps"] == 40
    with Image.open(tmp_path / "a-out.png") as image:
        assert image.mode == "RGB"
        assert image.size == (64, 32)


def test_loaded_worker_rewrites_with_source_image_before_generation(
    tmp_path: Path,
) -> None:
    pipe, pe, processor = _FakePipe(), _FakePE(), _FakeProcessor()
    runtime = worker.LoadedModels(pipe, pe, processor, "official system", _FakeTorch())

    response = worker.run_loaded_request(
        _request(tmp_path, request_id="b", rewrite=True, seed=27),
        runtime=runtime,
        device="cuda:0",
        default_seed=0,
        num_inference_steps=40,
    )

    assert pe.calls == 1
    assert processor.messages[0]["content"][0]["text"] == "official system"
    user_content = processor.messages[1]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[0]["image"].mode == "RGB"
    assert user_content[1]["text"] == "change the light"
    assert pipe.calls[0]["prompt"] == "add soft light"
    assert pipe.calls[0]["width"] == 64
    assert pipe.calls[0]["height"] == 32
    assert pipe.calls[0]["generator"].seed == 27
    assert response["original_instruction"] == "change the light"
    assert response["rewritten_instruction"] == "add soft light"
    assert response["effective_instruction"] == "add soft light"
    assert response["pe_wh_ratio"] == "16:9"


def test_pe_source_is_downscaled_without_changing_base_source(tmp_path: Path) -> None:
    pipe, pe, processor = _FakePipe(), _FakePE(), _FakeProcessor()
    runtime = worker.LoadedModels(pipe, pe, processor, "official system", _FakeTorch())
    request = _request(tmp_path, request_id="large", rewrite=True, seed=2)
    Image.new("RGB", (2048, 1024)).save(request["input_image_path"])

    worker.run_loaded_request(
        request,
        runtime=runtime,
        device="cuda:0",
        default_seed=0,
        num_inference_steps=40,
    )

    pe_image = processor.messages[1]["content"][0]["image"]
    assert pe_image.width * pe_image.height <= 1024 * 1024
    assert pipe.calls[0]["image"].size == (2048, 1024)


def test_pe_answer_parses_json_after_thinking_and_trailing_prose() -> None:
    assert worker.parse_pe_answer(
        '<think>reason</think>Result: {"rewrited_prompt":"clear edit","wh_ratio":"","ratio_follow":"<image1>"} done'
    ) == ("clear edit", {"pe_wh_ratio": "", "pe_ratio_follow": "<image1>"})


def test_serve_loads_once_for_two_requests(tmp_path: Path, monkeypatch) -> None:
    load_count = 0
    pipe, pe, processor = _FakePipe(), _FakePE(), _FakeProcessor()

    def fake_load_models(**kwargs: object):
        nonlocal load_count
        load_count += 1
        return worker.LoadedModels(pipe, pe, processor, "official system", _FakeTorch())

    monkeypatch.setattr(worker, "load_models", fake_load_models)
    requests = [
        _request(tmp_path, request_id="one", rewrite=False, seed=3),
        _request(tmp_path, request_id="two", rewrite=True, seed=4),
        {"schema_version": 1, "type": "shutdown", "request_id": "stop"},
    ]
    incoming = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    outgoing = io.StringIO()
    args = argparse.Namespace(
        model_path=tmp_path / "base",
        prompt_enhancer_i2i_path=tmp_path / "pe",
        device="cuda:0",
        seed=0,
        num_inference_steps=40,
    )

    assert worker.serve(args, incoming=incoming, outgoing=outgoing) == 0
    responses = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [item["type"] for item in responses] == [
        "ready",
        "response",
        "response",
        "shutdown",
    ]
    assert load_count == 1
    assert len(pipe.calls) == 2
    assert pe.calls == 1


def test_startup_loads_both_local_models_once(tmp_path: Path, monkeypatch) -> None:
    pe_path = tmp_path / "pe"
    pe_path.mkdir()
    (pe_path / "system_prompt.txt").write_text(" shipped prompt \n", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, object]]] = []
    fake_pe = _FakePE()
    fake_pipe = _FakePipe()

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls.append(("processor", path, kwargs))
            return _FakeProcessor()

    class _PELoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls.append(("pe", path, kwargs))
            return SimpleNamespace(
                to=lambda device: SimpleNamespace(eval=lambda: fake_pe)
            )

    class _PipeLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object):
            calls.append(("base", path, kwargs))
            return SimpleNamespace(to=lambda device: fake_pipe)

    fake_torch = _FakeTorch()
    fake_torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=_ProcessorLoader,
            AutoModelForImageTextToText=_PELoader,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(QwenImage21Pipeline=_PipeLoader)
    )

    runtime = worker.load_models(
        model_path=tmp_path / "base",
        prompt_enhancer_i2i_path=pe_path,
        device="cuda:0",
    )

    assert runtime.pipeline is fake_pipe
    assert runtime.prompt_enhancer is fake_pe
    assert runtime.system_prompt == "shipped prompt"
    assert [name for name, _, _ in calls] == ["processor", "pe", "base"]
    assert [path for _, path, _ in calls] == [
        str(pe_path),
        str(pe_path),
        str(tmp_path / "base"),
    ]
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in calls)
    assert calls[1][2]["dtype"] is fake_torch.bfloat16
    assert calls[1][2]["low_cpu_mem_usage"] is True
    assert "device_map" not in calls[1][2]
    assert calls[2][2]["dtype"] is fake_torch.bfloat16


def test_subprocess_backend_reuses_one_process_and_preserves_contract(
    tmp_path: Path,
) -> None:
    fake_script = tmp_path / "fake_worker.py"
    fake_script.write_text(
        "import json, os, sys\n"
        "from PIL import Image\n"
        "print(json.dumps({'schema_version':1,'type':'ready','status':'ok'}), flush=True)\n"
        "count=0\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if r['type']=='shutdown':\n"
        "  print(json.dumps({'schema_version':1,'type':'shutdown','request_id':r['request_id'],'status':'ok'}), flush=True); break\n"
        " count+=1\n"
        " Image.new('RGB',(r['width'],r['height']),(count,2,3)).save(r['output_image_path'])\n"
        " print(json.dumps({'schema_version':1,'type':'response','request_id':r['request_id'],'status':'ok','original_instruction':r['instruction'],'rewritten_instruction':None,'effective_instruction':r['instruction'],'returned_size':[r['width'],r['height']],'seed':r['seed'],'pid':os.getpid(),'request_index':count}), flush=True)\n",
        encoding="utf-8",
    )
    config = QwenImage21WorkerConfig(
        python_executable=Path(sys.executable),
        model_path=tmp_path / "nonexistent-base",
        prompt_enhancer_i2i_path=tmp_path / "nonexistent-pe",
        cuda_visible_devices="7",
        temporary_root=tmp_path,
        worker_script=fake_script,
    )
    backend = QwenImage21SubprocessBackend(config)
    assert not backend.started
    backend.start(stderr_log_path=tmp_path / "worker.log")
    try:
        assert backend.started
        outputs = [
            backend.edit(
                source_rgb=Image.new("RGB", (32, 32)),
                instruction="keep the subject",
                width=32,
                height=64,
                thinking_enabled=False,
                instruction_rewrite_enabled=False,
                seed=seed,
            )
            for seed in (11, 12)
        ]
        assert outputs[0].worker_metadata["pid"] == outputs[1].worker_metadata["pid"]
        assert [item.worker_metadata["request_index"] for item in outputs] == [1, 2]
        assert outputs[0].original_instruction == "keep the subject"
        assert outputs[0].effective_instruction == "keep the subject"
        assert outputs[0].returned_size == (32, 64)
        with Image.open(io.BytesIO(outputs[0].png_bytes)) as image:
            assert image.mode == "RGB"
            assert image.size == (32, 64)
    finally:
        backend.close()
    assert not backend.started
