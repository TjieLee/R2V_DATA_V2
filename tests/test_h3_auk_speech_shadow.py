from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from r2v_data_v2.h3 import auk_speech_shadow as auk
from r2v_data_v2.h3.auk_speech_qa import build_auk_speech_qa
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioSeparationResult,
    build_sam_audio_stem_inventory,
    run_sam_audio_stem_shadow,
    stem_separation_root,
)
from tools import run_h3_auk_speech_shadow as cli
from tools.run_h3_auk_speech_worker import validate_dependencies

FAKE_AUK = """import json, os, wave
from pathlib import Path
class AukInfer:
    def __init__(self, *, config_path, ckpt_path, device, dtype, qwen_path):
        self.log = Path(config_path).with_name("worker_calls.jsonl")
        self.write({"init": True, "config_path": config_path, "ckpt_path": ckpt_path,
                    "device": device, "dtype": dtype, "qwen_path": qwen_path,
                    "offline": [os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]]})
        print("model initialization log")
    def write(self, value):
        with self.log.open("a") as f: f.write(json.dumps(value)+"\\n")
    def generate(self, messages, *, gen_seconds, nfe, cfg_strength, sway_sampling_coef, seed):
        self.write({"messages": messages, "gen_seconds": gen_seconds, "nfe": nfe,
                    "cfg_strength": cfg_strength, "sway_sampling_coef": sway_sampling_coef, "seed": seed})
        return round(gen_seconds * 24000), 24000
def save_audio(frames, sample_rate, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as f:
        f.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        f.writeframes(b"\\x01\\x00" * frames)
"""


@pytest.fixture
def setup(tmp_path):
    code = tmp_path / "auk-code"
    infer = code / "src/auk/infer/infer_auk.py"
    infer.parent.mkdir(parents=True)
    infer.write_text(FAKE_AUK)
    (code / "pyproject.toml").write_text('[project]\nname="auk"\n')
    weights = tmp_path / "auk-model"
    weights.mkdir()
    for name in ("config.yaml", "auk_base.safetensors", "vae.safetensors"):
        (weights / name).write_bytes(b"local mock dependency")
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (qwen / name).write_bytes(b"{}")
    config = auk.auk_configuration(
        python_path=Path(sys.executable),
        code_root=code,
        checkpoint=weights / "auk_base.safetensors",
        qwen_path=qwen,
    )
    root = tmp_path / "production"
    audio = root / "audio"
    audio.mkdir(parents=True)
    clips = []
    for uid in ("a", "b", "c"):
        path = audio / f"{uid}.flac"
        sf.write(path, np.full((32000, 2), 0.125), 32000, subtype="PCM_24")
        video = tmp_path / f"{uid}.mp4"
        video.write_bytes(b"frozen mock video " + uid.encode())
        clips.append(
            CanonicalAudioClip(
                clip_uid=uid,
                clip_display_path=f"collection/episode/{uid}",
                media_collection_relpath="collection",
                media_collection_name="collection",
                episode_name="episode",
                clip_name=uid,
                shard_id="0",
                target_video_path=str(video),
                target_video_sha256=auk.sha256_file(video),
                target_full_audio_path=str(path),
                target_full_audio_sha256=auk.sha256_file(path),
                frame_count=32000,
                target_duration_seconds=1.0,
                subject_reference_count=0,
            )
        )
    manifest = audio / "canonical_clips.jsonl"
    manifest.write_text("".join(c.model_dump_json() + "\n" for c in clips))
    cases = tmp_path / "cases.json"
    cases.write_text(MimoCaseManifest(clip_uids=["c", "a", "b"]).model_dump_json())
    inventory = auk.build_auk_inventory(
        audio_production_root=root,
        shadow_run_id="pilot-1",
        case_manifest_path=cases,
        configuration=config,
    )
    return inventory


@pytest.fixture
def ffmpeg():
    binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not binary:
        pytest.skip("CPU ffmpeg unavailable")
    return binary


class FakeBackend:
    def __init__(self, configuration, *, fail=None, delta=0.0):
        self.configuration = configuration
        self.fail = fail
        self.delta = delta
        self.calls = []
        self.raw_bytes = {}

    def generate(self, job, raw):
        self.calls.append(job.clip_uid)
        if job.clip_uid == self.fail:
            return {
                "status": "failed",
                "reason": "synthetic model failure",
                "model_runtime_seconds": 0.1,
            }
        frames = round((job.source_duration_seconds + self.delta) * 24000)
        sf.write(raw, np.full(frames, 0.25), 24000, format="WAV", subtype="FLOAT")
        self.raw_bytes[job.clip_uid] = raw.read_bytes()
        return {"status": "ok", "model_runtime_seconds": 0.1}


def stage(inventory):
    return auk.auk_stage_root(
        Path(inventory.audio_production_root), inventory.shadow_run_id
    )


def run(inventory, ffmpeg, **kwargs):
    backend = FakeBackend(inventory.model_configuration, **kwargs)
    summary = auk.run_auk_speech_shadow(
        inventory=inventory, backend=backend, ffmpeg=ffmpeg
    )
    return backend, summary


def test_order_and_fingerprints(setup):
    assert setup.clip_uids == ["c", "a", "b"]
    assert auk.AukInventory.model_validate_json(setup.model_dump_json()) == setup
    changed = setup.model_dump(mode="json")
    changed["jobs"][0]["source_audio_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        auk.AukInventory.model_validate(changed)
    changed = setup.model_configuration.model_dump(mode="json")
    changed["dependency_files"][changed["vae_path"]] = "e" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        auk.AukConfiguration.model_validate(changed)


@pytest.mark.parametrize("ids", [["a", "a"], ["unknown"]])
def test_invalid_case_inventory(setup, ids):
    path = Path(setup.source_case_manifest_path)
    path.write_text(json.dumps({"clip_uids": ids}))
    with pytest.raises(ValueError):
        auk.build_auk_inventory(
            audio_production_root=Path(setup.audio_production_root),
            shadow_run_id="pilot-1",
            case_manifest_path=path,
            configuration=setup.model_configuration,
        )


@pytest.mark.parametrize("dependency", ["config_path", "checkpoint_path", "vae_path"])
def test_dependency_changes_fingerprint(setup, dependency):
    c = setup.model_configuration
    Path(getattr(c, dependency)).write_bytes(b"different local model")
    new = auk.auk_configuration(
        python_path=Path(c.python_path),
        code_root=Path(c.code_root),
        checkpoint=Path(c.checkpoint_path),
        qwen_path=Path(c.qwen_path),
    )
    assert new.configuration_fingerprint != c.configuration_fingerprint
    inventory = auk.build_auk_inventory(
        audio_production_root=Path(setup.audio_production_root),
        shadow_run_id="pilot-1",
        case_manifest_path=Path(setup.source_case_manifest_path),
        configuration=new,
    )
    assert inventory.inventory_fingerprint != setup.inventory_fingerprint


def test_local_checkpoint_symlink_keeps_sibling_assets(setup, tmp_path):
    c = setup.model_configuration
    checkpoint = Path(c.checkpoint_path)
    cache = tmp_path / "checkpoint-cache-blob"
    checkpoint.rename(cache)
    checkpoint.symlink_to(cache)
    new = auk.auk_configuration(
        python_path=Path(c.python_path),
        code_root=Path(c.code_root),
        checkpoint=checkpoint,
        qwen_path=Path(c.qwen_path),
    )
    assert new.checkpoint_path == c.checkpoint_path
    assert new.config_path == c.config_path
    assert new.vae_path == c.vae_path


def test_qwen_checkpoint_fingerprint_and_worker_preflight(setup):
    c = setup.model_configuration
    weights = Path(c.qwen_path) / "model.safetensors"
    weights.write_bytes(b"changed Qwen checkpoint")
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        auk.PersistentAukBackend(c),
    ):
        pytest.fail("changed dependencies must fail before model construction")
    new = auk.auk_configuration(
        python_path=Path(c.python_path),
        code_root=Path(c.code_root),
        checkpoint=Path(c.checkpoint_path),
        qwen_path=Path(c.qwen_path),
    )
    assert new.configuration_fingerprint != c.configuration_fingerprint


@pytest.mark.parametrize(
    "missing",
    [
        "vae.safetensors",
        "tokenizer_config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ],
)
def test_missing_local_dependencies_fail_before_model(setup, missing):
    c = setup.model_configuration
    parent = (
        Path(c.checkpoint_path).parent
        if missing == "vae.safetensors"
        else Path(c.qwen_path)
    )
    (parent / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        auk.auk_configuration(
            python_path=Path(c.python_path),
            code_root=Path(c.code_root),
            checkpoint=Path(c.checkpoint_path),
            qwen_path=Path(c.qwen_path),
        )
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        auk.PersistentAukBackend(c),
    ):
        pytest.fail("must not reach model startup")
    assert not Path(c.config_path).with_name("worker_calls.jsonl").exists()


def test_empty_python_sources_are_hashed_and_worker_startup_accepts(setup):
    c = setup.model_configuration
    paths = [
        Path(c.code_root) / name
        for name in (
            "src/auk/__init__.py",
            "src/auk/model/__init__.py",
            "src/auk/model/vae/modules/__init__.py",
        )
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    configuration = auk.auk_configuration(
        python_path=Path(c.python_path),
        code_root=Path(c.code_root),
        checkpoint=Path(c.checkpoint_path),
        qwen_path=Path(c.qwen_path),
    )
    for path in paths:
        assert (
            configuration.dependency_files[str(path)] == hashlib.sha256(b"").hexdigest()
        )
    validate_dependencies(configuration.model_dump(mode="json"))
    with auk.PersistentAukBackend(configuration) as backend:
        assert backend.process.poll() is None  # fake engine startup only, no generation
    assert all(path.read_bytes() == b"" for path in paths)
    paths[-1].write_text("# changed source\n")
    with pytest.raises(ValueError, match="hash differs"):
        validate_dependencies(configuration.model_dump(mode="json"))


@pytest.mark.parametrize(
    "name",
    [
        "auk_base.safetensors",
        "config.yaml",
        "vae.safetensors",
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model-00001-of-00001.safetensors",
    ],
)
def test_empty_runtime_artifacts_rejected_by_parent_and_worker(setup, name):
    c = setup.model_configuration
    if name.startswith("model-00001"):
        root = Path(c.qwen_path)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": name}})
        )
        (root / name).write_bytes(b"mock shard")
        c = auk.auk_configuration(
            python_path=Path(c.python_path),
            code_root=Path(c.code_root),
            checkpoint=Path(c.checkpoint_path),
            qwen_path=root,
        )
    parent = (
        Path(c.checkpoint_path).parent
        if name
        in {
            "auk_base.safetensors",
            "config.yaml",
            "vae.safetensors",
        }
        else Path(c.qwen_path)
    )
    path = parent / name
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="non-empty"):
        auk.auk_configuration(
            python_path=Path(c.python_path),
            code_root=Path(c.code_root),
            checkpoint=Path(c.checkpoint_path),
            qwen_path=Path(c.qwen_path),
        )
    payload = c.model_dump(mode="json")
    payload["dependency_files"][str(path)] = hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError, match="non-empty"):
        validate_dependencies(payload)  # reject even when the empty-file hash matches


def test_persistent_worker_offline_exact_request_once(setup, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    c = setup.model_configuration
    with auk.PersistentAukBackend(c) as backend:
        process = backend.process
        for job in setup.jobs:
            result = backend.generate(job, tmp_path / f"{job.clip_uid}.wav")
            assert result["status"] == "ok"
            assert backend.process is process
    assert process.poll() == 0
    calls = [
        json.loads(line)
        for line in Path(c.config_path)
        .with_name("worker_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(calls) == 4 and sum(bool(call.get("init")) for call in calls) == 1
    assert calls[0]["offline"] == ["1", "1"]
    assert calls[0]["qwen_path"] == c.qwen_path
    for call, job in zip(calls[1:], setup.jobs, strict=True):
        assert call == {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": auk.FIXED_INSTRUCTION},
                        {"type": "audio", "audio": job.source_audio_path},
                    ],
                }
            ],
            "gen_seconds": job.source_duration_seconds,
            "nfe": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "seed": 0,
        }


@pytest.mark.parametrize(
    "delta,ready",
    [
        (0, True),
        (0.032, True),
        (-0.025, True),
        (0.08, True),
        (0.12, False),
        (-0.12, False),
    ],
)
def test_raw_preservation_extent_and_bounded_drift(setup, ffmpeg, delta, ready):
    frozen = {
        Path(j.source_audio_path): Path(j.source_audio_path).read_bytes()
        for j in setup.jobs
    }
    backend, summary = run(setup, ffmpeg, delta=delta)
    assert summary.ready_count == (3 if ready else 0)
    _, records, loaded_summary = auk.load_auk_shadow(stage(setup))
    assert loaded_summary == summary
    for record in records:
        assert Path(record.raw.path).read_bytes() == backend.raw_bytes[record.clip_uid]
        assert record.raw_duration_delta_seconds == pytest.approx(delta)
        if ready:
            info = sf.info(record.canonical.path)
            assert (
                info.format,
                info.subtype,
                info.samplerate,
                info.channels,
                info.frames,
            ) == ("WAV", "PCM_16", 32000, 2, 32000)
            assert record.canonical_adjustment_samples == -round(delta * 32000)
            pcm, _ = sf.read(record.canonical.path, dtype="int16", always_2d=True)
            assert np.all(pcm[0] != 0)  # no beginning shift/padding
            if delta < 0:
                assert np.all(pcm[-round(-delta * 32000) :] == 0)
    assert all(path.read_bytes() == content for path, content in frozen.items())
    assert summary.production_artifacts_modified is False


def test_failure_isolation_and_owned_overwrite(setup, ffmpeg):
    sibling = stage(setup).parent / "separation" / "sentinel"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"frozen SAM baseline")
    backend, summary = run(setup, ffmpeg, fail="a")
    assert backend.calls == ["c", "a", "b"]
    assert (summary.ready_count, summary.failed_count, summary.model_call_count) == (
        2,
        1,
        3,
    )
    assert auk.load_auk_shadow(stage(setup))[1][1].failure_reason.endswith(
        "synthetic model failure"
    )
    with pytest.raises(FileExistsError):
        run(setup, ffmpeg)
    summary = auk.run_auk_speech_shadow(
        inventory=setup,
        backend=FakeBackend(setup.model_configuration),
        ffmpeg=ffmpeg,
        overwrite=True,
    )
    assert summary.ready_count == 3
    assert sibling.read_bytes() == b"frozen SAM baseline"


def test_atomic_interruption_leaves_published_stage(setup, ffmpeg, monkeypatch):
    run(setup, ffmpeg)
    frozen = {p: p.read_bytes() for p in stage(setup).rglob("*") if p.is_file()}

    def fail(*args, **kwargs):
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr(auk, "_publish_directory", fail)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        auk.run_auk_speech_shadow(
            inventory=setup,
            backend=FakeBackend(setup.model_configuration),
            ffmpeg=ffmpeg,
            overwrite=True,
        )
    assert all(path.read_bytes() == content for path, content in frozen.items())
    assert not list(stage(setup).parent.glob(".auk_speech_v1-*"))


def test_unknown_destination_rejected_before_model(setup, ffmpeg):
    stage(setup).mkdir(parents=True)
    (stage(setup) / "sentinel").write_text("unknown")
    backend = FakeBackend(setup.model_configuration)
    with pytest.raises((ValueError, FileNotFoundError)):
        auk.run_auk_speech_shadow(
            inventory=setup, backend=backend, ffmpeg=ffmpeg, overwrite=True
        )
    assert backend.calls == []


def test_source_tamper_rejected_before_model(setup, ffmpeg):
    Path(setup.jobs[0].source_audio_path).write_bytes(b"tampered")
    backend = FakeBackend(setup.model_configuration)
    with pytest.raises(ValueError, match="hash"):
        auk.run_auk_speech_shadow(inventory=setup, backend=backend, ffmpeg=ffmpeg)
    assert backend.calls == []


def test_cli_preflight_before_worker_startup(setup, monkeypatch):
    stage(setup).mkdir(parents=True)

    def forbidden(*args, **kwargs):
        pytest.fail("worker started despite occupied destination")

    monkeypatch.setattr(cli, "PersistentAukBackend", forbidden)
    c = setup.model_configuration
    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--audio-production-root",
                setup.audio_production_root,
                "--shadow-run-id",
                setup.shadow_run_id,
                "--case-manifest",
                setup.source_case_manifest_path,
                "--auk-python",
                c.python_path,
                "--auk-code-root",
                c.code_root,
                "--auk-checkpoint",
                c.checkpoint_path,
                "--auk-qwen-path",
                c.qwen_path,
            ]
        )


def sam_baseline(inventory, tmp_path):
    from r2v_data_v2.h3.audio_backends import AudioFileProbe
    from tests.test_h3_sam_audio_stem_shadow import (
        _Canonicalizer,
        _configuration,
        _Media,
    )

    class Media(_Media):
        def probe_audio_file(self, path):
            info = sf.info(path)
            return AudioFileProbe(
                sample_rate_hz=info.samplerate,
                channels=info.channels,
                frame_count=info.frames,
                duration_seconds=info.duration,
                format_name=info.format.lower(),
            )

    class FakeSAM:
        def __init__(self, config):
            self.configuration = config

        def separate(
            self, *, clip_uid, source_audio_path, prompt, target_path, residual_path
        ):
            info = sf.info(source_audio_path)
            for path in (target_path, residual_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    path,
                    np.zeros(info.frames),
                    info.samplerate,
                    format="WAV",
                    subtype="PCM_16",
                )
            return SAMAudioSeparationResult(
                target_path=str(target_path),
                residual_path=str(residual_path),
                verification_state="unverified",
                runtime_seconds=0.1,
                backend_metadata={"fake": True},
            )

    config = _configuration(tmp_path)
    sam = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=Path(
            inventory.source_canonical_audio_manifest_path
        ),
        model_configuration=config,
        route="voice_first",
        case_manifest_path=Path(inventory.source_case_manifest_path),
    )
    root = stem_separation_root(
        Path(inventory.audio_production_root), inventory.shadow_run_id
    )
    run_sam_audio_stem_shadow(
        inventory=sam,
        output_root=root,
        backend=FakeSAM(config),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=Media(),
    )
    return root


def test_static_qa_provenance_media_and_js(setup, ffmpeg, tmp_path):
    run(setup, ffmpeg)
    sam_root = sam_baseline(setup, tmp_path)
    frozen = {p: p.read_bytes() for p in sam_root.rglob("*") if p.is_file()}
    result = build_auk_speech_qa(
        audio_production_root=Path(setup.audio_production_root),
        shadow_run_id=setup.shadow_run_id,
    )
    output = Path(result["output_root"])
    data = json.loads((output / "data.json").read_text())
    assert [c["clip_uid"] for c in data["cases"]] == ["c", "a", "b"]
    for case in data["cases"]:
        for key in ("original", "sam_speech", "auk", "sam_sfx"):
            assert (output / case[key]).is_symlink()
            assert (output / case[key]).is_file()
    html = (output / "index.html").read_text()
    for name in (
        "Original target video",
        "SAM speech",
        "AuK speech",
        "SAM SFX",
        "Previous",
        "Next",
    ):
        assert name in html
    assert "https://" not in html
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node:
        script = tmp_path / "qa.js"
        script.write_text(re.findall(r"<script>(.*?)</script>", html, re.DOTALL)[0])
        subprocess.run([node, "--check", str(script)], check=True, capture_output=True)
    assert all(p.read_bytes() == content for p, content in frozen.items())
    with pytest.raises(FileExistsError):
        build_auk_speech_qa(
            audio_production_root=Path(setup.audio_production_root),
            shadow_run_id=setup.shadow_run_id,
        )


def test_qa_rejects_baseline_prompt_mismatch(setup, ffmpeg, tmp_path, monkeypatch):
    from r2v_data_v2.h3 import auk_speech_qa as qa

    run(setup, ffmpeg)
    root = sam_baseline(setup, tmp_path)
    inventory, records, summary = qa.load_stem_shadow(root)
    config = inventory.model_configuration.model_copy(
        update={"speech_prompt": "man speaking"}
    )
    inventory = inventory.model_copy(update={"model_configuration": config})
    monkeypatch.setattr(qa, "load_stem_shadow", lambda _: (inventory, records, summary))
    with pytest.raises(ValueError, match="baseline"):
        qa.build_auk_speech_qa(
            audio_production_root=Path(setup.audio_production_root),
            shadow_run_id=setup.shadow_run_id,
        )


def test_publication_rollback_after_old_stage_moved(setup, ffmpeg, monkeypatch):
    run(setup, ffmpeg)
    original = (stage(setup) / "records.jsonl").read_bytes()
    replace = Path.replace

    def fail_new_publish(self, target):
        if self.name.startswith(".auk_speech_v1-"):
            raise OSError("synthetic rename failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_publish)
    with pytest.raises(OSError, match="rename failure"):
        auk.run_auk_speech_shadow(
            inventory=setup,
            backend=FakeBackend(setup.model_configuration),
            ffmpeg=ffmpeg,
            overwrite=True,
        )
    assert (stage(setup) / "records.jsonl").read_bytes() == original
    assert auk.load_auk_shadow(stage(setup))[2].ready_count == 3


def test_qa_browser_navigation_and_media(setup, ffmpeg, tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional preinstalled Playwright not configured")
    # Only synthetic CPU media; requests below are all fulfilled in-process.
    for job in setup.jobs:
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=320x240:r=25",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-y",
                job.target_video_path,
            ],
            check=True,
            capture_output=True,
        )
    rows = auk._read_jsonl(Path(setup.source_canonical_audio_manifest_path))
    for row in rows:
        row["target_video_sha256"] = auk.sha256_file(Path(row["target_video_path"]))
    Path(setup.source_canonical_audio_manifest_path).write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    inventory = auk.build_auk_inventory(
        audio_production_root=Path(setup.audio_production_root),
        shadow_run_id=setup.shadow_run_id,
        case_manifest_path=Path(setup.source_case_manifest_path),
        configuration=setup.model_configuration,
    )
    run(inventory, ffmpeg)
    sam_baseline(inventory, tmp_path)
    result = build_auk_speech_qa(
        audio_production_root=Path(inventory.audio_production_root),
        shadow_run_id=inventory.shadow_run_id,
    )
    script = tmp_path / "browser.cjs"
    script.write_text(r"""
const fs=require("fs"),path=require("path"),assert=require("assert");
const {chromium}=require(process.argv[2]);
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.QA_BROWSER_CHANNEL});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on("pageerror",e=>errors.push(e.message));
  await page.route("**/*",route=>{
   const url=new URL(route.request().url());assert.strictEqual(url.hostname,"qa.invalid");
   const name=path.join(process.argv[3],url.pathname);
   return route.fulfill({body:fs.readFileSync(name),contentType:name.endsWith(".html")?"text/html":name.endsWith(".wav")?"audio/wav":"video/mp4"});
  });
  await page.goto("http://qa.invalid/index.html");
  assert.strictEqual(await page.locator("#position").textContent(),"1 / 3");
  await page.waitForFunction(()=>[...document.querySelectorAll("audio,video")].every(p=>p.readyState>=1));
  assert.strictEqual(await page.locator("#original").evaluate(v=>v.videoWidth),320);
  const first=await page.locator("#auk").getAttribute("src");
  await page.locator("#next").click();
  assert.strictEqual(await page.locator("#position").textContent(),"2 / 3");
  assert.notStrictEqual(await page.locator("#auk").getAttribute("src"),first);
  await page.locator("#clips").selectOption("2");
  assert.strictEqual(await page.locator("#position").textContent(),"3 / 3");
  await page.locator("#prev").click();
  assert.strictEqual(await page.locator("#position").textContent(),"2 / 3");
  await page.waitForFunction(()=>[...document.querySelectorAll("audio,video")].every(p=>p.readyState>=2));
  await page.screenshot({path:path.join(process.argv[4],"auk-desktop.png")});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:path.join(process.argv[4],"auk-mobile.png"),fullPage:true});
  assert.deepStrictEqual(errors,[]);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
""")
    subprocess.run(
        [node, str(script), playwright, result["output_root"], str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
