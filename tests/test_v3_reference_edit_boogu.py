from __future__ import annotations

import importlib
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import r2v_data_v2.v3.reference_edit_boogu as boogu_module
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguBackgroundReview,
    BooguCompletionReview,
    BooguEditOutput,
    BooguSamReview,
    BooguSubprocessBackend,
    BooguWorkerConfig,
    resolve_boogu_1k_size,
    run_boogu_reference_edit,
)


def _png_bytes(
    size: tuple[int, int],
    *,
    color: tuple[int, ...] = (27, 44, 91),
    mode: str = "RGB",
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _completion_review(*, accept: bool = True) -> BooguCompletionReview:
    values = {
        "same_physical_entity": accept,
        "identity_preserved": accept,
        "original_visible_attributes_preserved": accept,
        "exactly_one_entity": accept,
        "missing_parts_plausibly_completed": accept,
        "no_duplicate_entity": accept,
        "no_unrelated_entity": accept,
        "no_severe_structure_artifact": accept,
        "style_coherent": accept,
        "resolution_usable": accept,
        "reference_usable": accept,
        "certain": accept,
    }
    return BooguCompletionReview(
        verdict="accept" if accept else "reject",
        reason="usable" if accept else "identity drift",
        **values,
    )


def _background_review(*, accept: bool = True) -> BooguBackgroundReview:
    values = {
        "exactly_one_target_entity": accept,
        "identity_preserved": accept,
        "entity_appearance_consistent": accept,
        "no_duplicate_entity": accept,
        "no_added_salient_entity": accept,
        "no_unintended_completion_or_extension": accept,
        "background_coherent": accept,
        "background_style_consistent": accept,
        "no_halo_or_seam": accept,
        "subject_not_severely_redrawn": accept,
        "reference_usable": accept,
        "certain": accept,
    }
    return BooguBackgroundReview(
        verdict="accept" if accept else "reject",
        reason="usable" if accept else "new entity",
        **values,
    )


class _Backend:
    def __init__(self, *, returned_size: tuple[int, int] | None = None) -> None:
        self.returned_size = returned_size
        self.calls: list[dict[str, object]] = []
        self.output_bytes: bytes | None = None

    def edit(self, **kwargs: object) -> BooguEditOutput:
        self.calls.append(kwargs)
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        output_size = self.returned_size or (width, height)
        self.output_bytes = _png_bytes(output_size, color=(191, 22, 43))
        thinking = bool(kwargs["thinking_enabled"])
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=self.output_bytes,
            original_instruction=instruction,
            rewritten_instruction="rewritten" if thinking else None,
            effective_instruction="rewritten" if thinking else instruction,
            worker_metadata={"returned_size": list(output_size)},
        )


class _Judge:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> BooguCompletionReview | BooguBackgroundReview:
        self.calls.append(kwargs)
        if kwargs["operation"] == "complete_entity":
            return _completion_review(accept=self.accept)
        return _background_review(accept=self.accept)


class _SamReviewer:
    def __init__(self, *, passed: bool = True) -> None:
        self.passed = passed
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> BooguSamReview:
        self.calls.append(kwargs)
        return BooguSamReview(
            passed=self.passed,
            target_entity_present=self.passed,
            exactly_one_target_instance=self.passed,
            area_growth_acceptable=self.passed,
            fragmentation_acceptable=self.passed,
            reason="valid" if self.passed else "fragmented",
            diagnostics={"mask_path": "review-only.png"},
        )


def _environment(
    tmp_path: Path,
    *,
    size: tuple[int, int] = (48, 32),
) -> tuple[Path, Path, bytes]:
    run_root = tmp_path / "run"
    canonical = run_root / "clips" / "clip-1" / "selected" / "e1.png"
    canonical.parent.mkdir(parents=True)
    image = Image.new("RGBA", size, (20, 180, 70, 255))
    image.putpixel((0, 0), (10, 20, 30, 0))
    image.save(canonical, format="PNG")
    payload = canonical.read_bytes()
    return run_root, canonical, payload


@pytest.mark.parametrize(
    ("source_size", "expected"),
    [
        ((100, 100), (1024, 1024)),
        ((160, 90), (1360, 768)),
        ((90, 160), (768, 1360)),
        ((150, 100), (1248, 832)),
    ],
)
def test_resolve_boogu_1k_size_preserves_ratio_and_alignment(
    source_size: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    resolved = resolve_boogu_1k_size(*source_size)

    assert resolved == expected
    assert resolved[0] % 16 == 0
    assert resolved[1] % 16 == 0
    assert abs(resolved[0] * resolved[1] - 1024 * 1024) / (1024 * 1024) < 0.02
    source_ratio = source_size[0] / source_size[1]
    assert abs(resolved[0] / resolved[1] - source_ratio) / source_ratio < 0.01


@pytest.mark.parametrize(
    "values",
    [(0, 1, 1024, 16), (1, -1, 1024, 16), (1, 1, 0, 16), (1, 1, 1024, 0)],
)
def test_resolve_boogu_1k_size_rejects_invalid_values(
    values: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_boogu_1k_size(
            values[0],
            values[1],
            target_area=values[2],
            alignment=values[3],
        )


def test_completion_publishes_native_1k_output_without_paste_back(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)
    backend = _Backend()
    judge = _Judge()
    sam = _SamReviewer()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete the same entity.",
        entity_phrase="green object",
        reference_type="object",
        backend=backend,
        judge=judge,
        sam_reviewer=sam,
    )

    assert result.status == "accepted"
    assert backend.calls[0]["thinking_enabled"] is True
    assert "alpha_mask" not in backend.calls[0]
    assert "mask" not in backend.calls[0]
    assert canonical.read_bytes() == canonical_bytes
    assert result.candidate_path is not None
    assert result.final_reference_path is not None
    assert result.candidate_path.read_bytes() == backend.output_bytes
    assert result.final_reference_path.read_bytes() == backend.output_bytes
    with Image.open(result.final_reference_path) as final:
        assert final.mode == "RGB"
        assert final.size == (1248, 832)
        assert final.getpixel((10, 10)) == (191, 22, 43)
    assert sam.calls
    assert "candidate_rgb" in sam.calls[0]


def test_white_rgb_input_preserves_opaque_pixels_and_whitens_alpha(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    backend = _Backend()

    run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete the entity.",
        entity_phrase="object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
    )

    source_rgb = backend.calls[0]["source_rgb"]
    assert isinstance(source_rgb, Image.Image)
    assert source_rgb.mode == "RGB"
    assert source_rgb.getpixel((0, 0)) == (255, 255, 255)
    assert source_rgb.getpixel((1, 1)) == (20, 180, 70)


def test_background_uses_thinking_false_and_no_source_restoration(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)
    backend = _Backend()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a quiet studio background.",
        entity_phrase="green object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
    )

    assert result.status == "accepted"
    assert backend.calls[0]["thinking_enabled"] is False
    assert canonical.read_bytes() == canonical_bytes
    assert result.final_reference_path is not None
    assert result.final_reference_path.read_bytes() == backend.output_bytes


def test_wrong_native_output_size_fails_closed_and_preserves_canonical(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(returned_size=(640, 640)),
        judge=_Judge(),
    )

    assert result.status == "rejected"
    assert result.final_reference_path is None
    assert canonical.read_bytes() == canonical_bytes
    rejection = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    assert "returned_size=(640, 640)" in rejection["reason"]


def test_rejected_candidate_uses_legacy_canonical_fallback(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="add_entity_background",
        instruction="Add a background.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(accept=False),
        fallback_status="canonical_local_fallback",
    )

    assert result.status == "rejected"
    assert result.fallback_status == "canonical_local_fallback"
    assert result.candidate_path is not None
    assert result.final_reference_path is None
    assert canonical.read_bytes() == canonical_bytes
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["fallback_status"] == "canonical_local_fallback"


def test_failed_sam_review_rejects_but_never_changes_candidate_pixels(
    tmp_path: Path,
) -> None:
    run_root, _, _ = _environment(tmp_path)
    backend = _Backend()

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=backend,
        judge=_Judge(),
        sam_reviewer=_SamReviewer(passed=False),
    )

    assert result.status == "rejected"
    assert result.candidate_path is not None
    assert result.candidate_path.read_bytes() == backend.output_bytes
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["sam_mask_usage"] == "review_only"


def test_metadata_records_source_output_dimensions_and_provenance(
    tmp_path: Path,
) -> None:
    run_root, canonical, canonical_bytes = _environment(tmp_path)

    result = run_boogu_reference_edit(
        run_root=run_root,
        clip_uid="clip-1",
        entity_id="e1",
        operation="complete_entity",
        instruction="Complete it.",
        entity_phrase="object",
        reference_type="object",
        backend=_Backend(),
        judge=_Judge(),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_dimensions"] == {"width": 48, "height": 32}
    assert metadata["resolved_output_dimensions"] == {"width": 1248, "height": 832}
    assert metadata["target_area"] == 1024 * 1024
    assert metadata["alignment"] == 16
    assert metadata["output_pixel_count"] == 1248 * 832
    assert metadata["canonical_source_sha256"] == boogu_module._sha256_bytes(
        canonical_bytes
    )
    assert metadata["output_sha256"] == boogu_module._sha256_bytes(
        result.candidate_path.read_bytes()
    )
    assert canonical.read_bytes() == canonical_bytes


def test_subprocess_backend_invokes_configured_python_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "venvs" / "boogu" / "bin" / "python"
    code_root = tmp_path / "vendor" / "Boogu-Image"
    model = tmp_path / "models" / "boogu"
    worker = tmp_path / "repo" / "worker.py"
    for path in (python, worker):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    code_root.mkdir(parents=True)
    model.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        Path(request["output_image_path"]).write_bytes(
            _png_bytes((request["width"], request["height"]))
        )
        Path(request["result_path"]).write_text(
            json.dumps(
                {
                    "original_instruction": request["instruction"],
                    "rewritten_instruction": None,
                    "effective_instruction": request["instruction"],
                    "returned_size": [request["width"], request["height"]],
                }
            ),
            encoding="utf-8",
        )
        assert request["thinking_enabled"] is False
        assert request["width"] == 1360
        assert request["height"] == 768
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(boogu_module.subprocess, "run", fake_run)
    backend = BooguSubprocessBackend(
        BooguWorkerConfig(
            python_executable=python.resolve(),
            code_root=code_root.resolve(),
            model_path=model.resolve(),
            allowed_server_root=tmp_path.resolve(),
            worker_script=worker.resolve(),
        )
    )

    output = backend.edit(
        source_rgb=Image.new("RGB", (160, 90)),
        instruction="Add a background.",
        width=1360,
        height=768,
        thinking_enabled=False,
    )

    assert output.effective_instruction == "Add a background."
    assert calls[0]["command"][:2] == [str(python.resolve()), str(worker.resolve())]
    assert calls[0]["cwd"] == code_root.resolve()
    assert calls[0]["check"] is True
    assert "shell" not in calls[0]
    assert all("conda" not in part for part in calls[0]["command"])


def test_worker_module_import_does_not_import_torch_or_boogu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if name == "torch" or name == "boogu" or name.startswith("boogu."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("tools.run_v3_boogu_reference_edit_worker")

    assert module is not None
    assert "torch" not in sys.modules
    assert not any(name == "boogu" or name.startswith("boogu.") for name in sys.modules)


def test_worker_passes_explicit_size_and_thinking_to_fake_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module = importlib.import_module("tools.run_v3_boogu_reference_edit_worker")
    code_root = tmp_path / "vendor"
    model = tmp_path / "model"
    code_root.mkdir()
    model.mkdir()
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    result_path = tmp_path / "result.json"
    Image.new("RGB", (19, 23), (1, 2, 3)).save(input_path)
    captured: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            captured["generator_device"] = device

        def manual_seed(self, seed: int) -> FakeGenerator:
            captured["seed"] = seed
            return self

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakePipeline:
            captured["model_path"] = path
            captured["load_kwargs"] = kwargs
            return cls()

        def to(self, device: str) -> None:
            captured["device"] = device

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            captured["call"] = kwargs
            rewrite_path = Path(str(kwargs["save_rewritten_instruction_path"]))
            rewrite_path.write_text(
                json.dumps(
                    {
                        "ori_instruction": ["Complete it."],
                        "rewritten_instruction": ["Complete the same object."],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(images=[Image.new("RGB", (1360, 768))])

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    torch_module.Generator = FakeGenerator
    pipeline_module = types.ModuleType("boogu.pipelines.boogu.pipeline_boogu_turbo")
    pipeline_module.BooguImageTurboPipeline = FakePipeline
    for name in ("boogu", "boogu.pipelines", "boogu.pipelines.boogu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "boogu.pipelines.boogu.pipeline_boogu_turbo",
        pipeline_module,
    )
    payload = {
        "schema_version": 1,
        "code_root": str(code_root.resolve()),
        "model_path": str(model.resolve()),
        "model_name": "Boogu-Image-0.1-Edit-Turbo",
        "model_revision": "hotfix-1k-20260708",
        "device": "cuda:0",
        "seed": 17,
        "input_image_path": str(input_path.resolve()),
        "output_image_path": str(output_path.resolve()),
        "result_path": str(result_path.resolve()),
        "instruction": "Complete it.",
        "thinking_enabled": True,
        "width": 1360,
        "height": 768,
    }

    response = worker_module.run_request(payload)

    call = captured["call"]
    assert isinstance(call, dict)
    assert call["width"] == 1360
    assert call["height"] == 768
    assert call["use_rewrite_text_instruction"] is True
    assert call["input_images"][0][0].size == (19, 23)
    assert call["input_images"][0][0].mode == "RGB"
    assert response["rewritten_instruction"] == "Complete the same object."
    assert response["effective_instruction"] == "Complete the same object."
    with Image.open(output_path) as output:
        assert output.size == (1360, 768)
        assert output.mode == "RGB"
