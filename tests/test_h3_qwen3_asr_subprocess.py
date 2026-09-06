from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration
from r2v_data_v2.h3.qwen3_asr_subprocess import PersistentQwen3ASRBackend


def _fake_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "fake_qwen_worker.py"
    worker.write_text(
        """from __future__ import annotations
import argparse
import base64
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model-path')
parser.add_argument('--device')
parser.add_argument('--dtype')
parser.add_argument('--max-inference-batch-size')
parser.add_argument('--max-new-tokens')
parser.parse_args()

def emit(value):
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')
    sys.stdout.flush()

emit({'request_id': 'startup', 'status': 'ready'})
for line in sys.stdin:
    request = json.loads(line)
    request_id = request['request_id']
    if request['operation'] == 'shutdown':
        emit({'request_id': request_id, 'status': 'shutdown'})
        raise SystemExit(0)
    raw = base64.b64decode(request['audio_f32le_base64'])
    emit({
        'request_id': request_id,
        'status': 'ok',
        'text': f"samples={len(raw) // 4};pythonpath={'set' if os.environ.get('PYTHONPATH') else 'missing'}",
        'language': 'English',
    })
""",
        encoding="utf-8",
    )
    return worker


def _configuration() -> Qwen3ASRConfiguration:
    return Qwen3ASRConfiguration(
        local_model_path="/local/qwen3-asr",
        device="cpu",
        dtype="float32",
        max_inference_batch_size=1,
    )


def test_persistent_qwen_backend_preserves_venv_python_symlink(tmp_path: Path) -> None:
    base_python = tmp_path / "base-python"
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)
    backend = PersistentQwen3ASRBackend(
        _configuration(),
        python_path=venv_python,
        worker_path=_fake_worker(tmp_path),
        timeout_seconds=5,
    )
    assert backend.python_path == venv_python.absolute()
    assert backend.python_path != venv_python.resolve()


def test_persistent_qwen_backend_uses_isolated_worker_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/poisoned/parent/pythonpath")
    backend = PersistentQwen3ASRBackend(
        _configuration(),
        python_path=Path(sys.executable),
        worker_path=_fake_worker(tmp_path),
        timeout_seconds=5,
    )
    with backend:
        text, language = backend.transcribe(
            waveform=np.asarray([0.0, 0.25, -0.25, 0.5], dtype=np.float32),
            sample_rate_hz=16000,
        )
    assert text == "samples=4;pythonpath=missing"
    assert language == "English"


def test_persistent_qwen_backend_rejects_noncanonical_model_input(
    tmp_path: Path,
) -> None:
    backend = PersistentQwen3ASRBackend(
        _configuration(),
        python_path=Path(sys.executable),
        worker_path=_fake_worker(tmp_path),
        timeout_seconds=5,
    )
    with pytest.raises(ValueError, match="16 kHz"):
        backend.transcribe(
            waveform=np.asarray([0.0, 0.1], dtype=np.float32),
            sample_rate_hz=32000,
        )
