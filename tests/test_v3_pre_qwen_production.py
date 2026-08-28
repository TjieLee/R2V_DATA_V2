from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import yaml
from PIL import Image

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.pre_qwen_production as production
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.frames import DecodedFrameInfo, DecodedVideoFrame
from r2v_data_v2.v3.pre_qwen_production import (
    ConfigIdentity,
    ShardBusyError,
    canary_summary,
    check_stage2_candidate_judge_health,
    inspect_stage2_shard,
    inventory,
    load_annotation_shard,
    prepare_canary,
    process_ready_clip,
    process_shard,
    run_canary_worker,
    select_canary_rows,
    shard_lock,
    validate_stage2_preflight,
)
from r2v_data_v2.v3.sam3_anchor_selector import QwenSam3AnchorSelector
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
    Sam3SegmentationBackend,
)
from tools import run_v3_pre_qwen_batch as batch_tool
from tools import run_v3_pre_qwen_canary as canary_tool


@dataclass
class FakeDecoder:
    calls: list[str] = field(default_factory=list)

    def inspect(self, video_path: Path) -> list[DecodedFrameInfo]:
        self.calls.append(f"inspect:{video_path.name}")
        return [
            DecodedFrameInfo(
                source_frame_index=index,
                timestamp_seconds=index * 0.1,
                width=16,
                height=12,
            )
            for index in range(20)
        ]

    def decode_indices(
        self,
        video_path: Path,
        source_frame_indices: list[int],
    ) -> list[DecodedVideoFrame]:
        self.calls.append(f"decode:{video_path.name}")
        return [
            DecodedVideoFrame(
                source_frame_index=index,
                image=Image.new("RGB", (16, 12), (index, 20, 30)),
            )
            for index in source_frame_indices
        ]


@dataclass
class FakeBackend:
    visible_slots: int = 10
    failure: Exception | None = None
    fail_on_call: int | None = None
    calls: list[str] = field(default_factory=list)
    close_calls: int = 0

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
        **kwargs: object,
    ) -> EntityTrackResult:
        del reference_type, grounding_prompt, kwargs
        self.calls.append(entity_id)
        if self.failure is not None and (
            self.fail_on_call is None or len(self.calls) == self.fail_on_call
        ):
            raise self.failure
        observations = []
        for slot in range(self.visible_slots):
            mask = np.zeros((12, 16), dtype=bool)
            mask[4:8, 6:10] = True
            observations.append(
                BackendMaskObservation(
                    slot=slot,
                    mask=mask,
                    confidence=0.9,
                    object_id=f"track-{entity_id}",
                )
            )
        assert len(frame_paths) == 10
        return EntityTrackResult(status="ready", observations=tuple(observations))

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class CountingBackend(FakeBackend):
    def anchor_search_counters(self) -> dict[str, int]:
        return {
            "anchor_probe_calls": len(self.calls) * 2,
            "anchor_fast_path_hits": len(self.calls),
            "anchor_fallback_attempted": 0,
            "anchor_fallback_hits": 0,
            "anchor_all_frames_not_found": 0,
        }

    def recall_rescue_counters(self) -> dict[str, int]:
        return {
            "multi_instance_rescue_attempted": len(self.calls),
            "multi_instance_rescue_selected": len(self.calls),
            "multi_instance_rescue_rejected": 0,
        }


@dataclass(frozen=True)
class Fixture:
    writable: Path
    dataset: Path
    input_root: Path
    output_root: Path
    base_path: Path
    config: V3Config
    identity: ConfigIdentity


def _paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    models = (writable / "models").resolve()
    for path in (writable, dataset, pretrained, models):
        path.mkdir(parents=True, exist_ok=True)
    for module in (config_module, production):
        monkeypatch.setattr(module, "ALLOWED_WRITABLE_ROOT", writable)
        monkeypatch.setattr(module, "ALLOWED_DATASET_ROOT", dataset)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", models)
    dataset_json = dataset / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    qwen_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "base-run",
        export_root=writable / "base-export",
        source=SourceConfig(limit=100),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(qwen_model)),
            instruction_writer=QwenServiceConfig(model=str(qwen_model)),
            candidate_judge=QwenServiceConfig(model=str(qwen_model)),
            background_remove_judge=QwenServiceConfig(model=str(qwen_model)),
        ),
        sam3=Sam3Config(model_path=models / "sam3.pt"),
        remove=RemoveConfig(
            enabled=False,
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=models / "object-remover",
            generation_mask_dilation_pixels=0,
        ),
    )
    config.validate()
    base_path = writable / "base.yaml"
    base_path.write_text("base: fixture\n", encoding="utf-8")
    identity = ConfigIdentity(
        path=base_path,
        sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        fingerprint=config.fingerprint(),
        config=config,
    )
    input_root = dataset / "entity_annotations"
    (input_root / "parts").mkdir(parents=True)
    return Fixture(
        writable=writable,
        dataset=dataset,
        input_root=input_root,
        output_root=writable / "pre-qwen-v1",
        base_path=base_path,
        config=config,
        identity=identity,
    )


def _write_loadable_config(fixture: Fixture, *, qwen_rescue: bool = False) -> Path:
    path = fixture.writable / ("qwen.yaml" if qwen_rescue else "off.yaml")
    payload = {
        "dataset_json": str(fixture.config.dataset_json),
        "run_root": str(fixture.writable / "yaml-run"),
        "export_root": str(fixture.writable / "yaml-export"),
        "source": {"limit": 100},
        "qwen": {
            "candidate_judge": {
                "model": str(
                    fixture.dataset.parent
                    / "pretrained"
                    / "Qwen"
                    / "Qwen3-VL-32B-Instruct"
                )
            },
            "background_remove_judge": {
                "model": str(
                    fixture.dataset.parent
                    / "pretrained"
                    / "Qwen"
                    / "Qwen3-VL-32B-Instruct"
                )
            },
        },
        "sam3": {
            "model_path": str(fixture.writable / "models" / "sam3.pt"),
            "multi_instance_rescue_mode": (
                "qwen_anchor_select_v1" if qwen_rescue else "off"
            ),
        },
        "remove": {
            "enabled": False,
            "base_model_path": str(
                fixture.dataset.parent / "pretrained" / "Qwen" / "Qwen-Image-Edit-2511"
            ),
            "adapter_path": str(fixture.writable / "models" / "object-remover"),
            "generation_mask_dilation_pixels": 0,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _entity(index: int = 1) -> dict[str, object]:
    return {
        "entity_id": f"e{index}",
        "reference_type": "subject" if index == 1 else "object",
        "phrase": f"entity {index}",
        "grounding_prompt": f"distinct entity {index}",
    }


def _row(
    fixture: Fixture,
    source_index: int,
    *,
    status: str = "ready",
    entities: int = 1,
    background: bool = False,
) -> dict[str, object]:
    video = fixture.dataset / "clips" / f"clip-{source_index}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-mp4")
    if status == "failed":
        return {
            "source_index": source_index,
            "clip_uid": f"clip-{source_index}",
            "video_path": str(video),
            "source_video_id": "movie",
            "source_video_path": "movie.mkv",
            "shot_index": source_index,
            "status": "failed",
            "entities": [],
            "background": None,
            "instruction_template": "",
            "reason": "annotation failed",
            "repair_attempts": 0,
            "warnings": [],
        }
    return {
        "source_index": source_index,
        "clip_uid": f"clip-{source_index}",
        "video_path": str(video),
        "source_video_id": "movie",
        "source_video_path": "movie.mkv",
        "shot_index": source_index,
        "status": "ready",
        "entities": [_entity(index + 1) for index in range(entities)],
        "background": (
            {"phrase": "a room", "grounding_prompt": "empty room"}
            if background
            else None
        ),
        "instruction_template": "An entity moves in the scene.",
        "reason": None,
        "repair_attempts": 0,
        "warnings": [],
    }


def _write_shard(
    fixture: Fixture,
    rows: list[dict[str, object]],
    *,
    nominal_end: int | None = None,
) -> Path:
    start = int(rows[0]["source_index"])
    end = nominal_end if nominal_end is not None else int(rows[-1]["source_index"])
    path = fixture.input_root / "parts" / f"shard-{start:09d}-{end:09d}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_canary_shards(
    fixture: Fixture,
    *,
    shard_count: int = 16,
    eligible_per_shard: int = 5,
) -> list[Path]:
    paths = []
    for shard_index in range(shard_count):
        start = shard_index * 10_000
        rows = [
            _row(fixture, start + offset) for offset in range(eligible_per_shard)
        ]
        paths.append(_write_shard(fixture, rows, nominal_end=start + 9_999))
    return paths


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_qwen_anchor_mode_without_candidate_judge_fails_preflight() -> None:
    config = V3Config(
        dataset_json=Path("/mnt/workspace/public/dataset/source.jsonl"),
        run_root=Path("/mnt/workspace/litengjie/data/run"),
        export_root=Path("/mnt/workspace/litengjie/data/export"),
        source=SourceConfig(limit=1),
        sam3=Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
    )

    with pytest.raises(ValueError, match="qwen.candidate_judge is required"):
        validate_stage2_preflight(config)


def test_qwen_anchor_mode_passes_preflight_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    config = replace(
        fixture.config,
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )

    policy = validate_stage2_preflight(config)

    assert config.sam3.multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert policy["sam3_multi_instance_rescue_mode"] == "qwen_anchor_select_v1"
    assert policy["sam3_anchor_qwen_enabled"] is True
    assert policy["candidate_judge_base_url"] == "http://127.0.0.1:8000/v1"
    assert policy["candidate_judge_model"] == config.qwen.candidate_judge.model


def test_candidate_judge_health_uses_models_endpoint_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    service = replace(
        fixture.config.qwen.candidate_judge,
        base_url="http://6.167.57.88:8000/v1",
    )
    config = replace(
        fixture.config,
        qwen=replace(fixture.config.qwen, candidate_judge=service),
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )
    requests: list[tuple[str, float]] = []

    class Response:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def opener(request: object, *, timeout: float) -> Response:
        requests.append((request.full_url, timeout))
        return Response()

    health = check_stage2_candidate_judge_health(config, opener=opener)

    assert requests == [("http://6.167.57.88:8000/v1/models", 5.0)]
    assert health["candidate_judge_health"] == "available"


def test_candidate_judge_health_failure_prevents_sam3_backend_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    config = replace(
        fixture.config,
        sam3=replace(
            fixture.config.sam3,
            multi_instance_rescue_mode="qwen_anchor_select_v1",
        ),
    )
    identity = replace(fixture.identity, config=config)
    monkeypatch.setattr(batch_tool, "load_config_identity", lambda path: identity)
    monkeypatch.setattr(batch_tool, "enumerate_annotation_shards", lambda root: [shard])
    monkeypatch.setattr(
        batch_tool,
        "check_stage2_candidate_judge_health",
        lambda current: (_ for _ in ()).throw(RuntimeError("gateway unavailable")),
    )
    monkeypatch.setattr(
        batch_tool,
        "default_backend_factory",
        lambda current: pytest.fail("SAM3 must not load before Qwen health passes"),
    )

    with pytest.raises(RuntimeError, match="gateway unavailable"):
        batch_tool.run_claim_loop(
            input_root=fixture.input_root,
            base_config=fixture.base_path,
            output_root=fixture.output_root,
            scan_offset=0,
        )


def test_full_stage2_shard_is_one_to_one_and_skips_ineligible_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(
        fixture,
        [
            _row(fixture, 0),
            _row(fixture, 1, status="failed"),
            _row(fixture, 2, entities=0),
        ],
        nominal_end=9999,
    )
    backend = FakeBackend()
    decoder = FakeDecoder()

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=decoder,
    )

    rows = _read_jsonl(Path(result["path"]))
    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert [row["status"] for row in rows] == [
        "ready_no_background",
        "skipped_annotation_failed",
        "skipped_no_entities",
    ]
    assert backend.calls == ["e1"]
    assert decoder.calls == ["inspect:clip-0.mp4", "decode:clip-0.mp4"]
    assert Path(result["path"]).name == shard.name


def test_annotation_shard_rejects_duplicate_ready_clip_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    rows = [_row(fixture, 0), _row(fixture, 1)]
    rows[1]["clip_uid"] = rows[0]["clip_uid"]
    shard = _write_shard(fixture, rows)

    with pytest.raises(ValueError, match="duplicate ready clip_uid"):
        load_annotation_shard(shard)


def test_completed_shard_validates_and_skips_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    backend = FakeBackend(failure=RuntimeError("must not run"))

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=FakeDecoder(),
    )

    assert result["skipped"] is True
    assert backend.calls == []


def test_artifacts_published_before_row_are_reused_without_sam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    row = _row(fixture, 0)
    shard_path = _write_shard(fixture, [row])
    shard = load_annotation_shard(shard_path)
    workspace = fixture.output_root / "artifacts" / shard_path.stem / "clip-0"
    first = FakeBackend()
    process_ready_clip(
        row=row,
        shard=shard,
        output_root=fixture.output_root,
        workspace=workspace,
        config_identity=fixture.identity,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend(failure=RuntimeError("must not run"))

    process_shard(
        shard_path,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=second,
        decoder=FakeDecoder(),
    )

    assert first.calls == ["e1"]
    assert second.calls == []


def test_retryable_sam_failure_does_not_commit_terminal_row_and_resumes_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    first_decoder = FakeDecoder()
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("CUDA OOM")),
        decoder=first_decoder,
    )

    assert result["retryable"] is True
    assert not (fixture.output_root / "parts" / shard.name).exists()
    assert not (fixture.output_root / "parts" / f"{shard.name}.partial").exists()
    second_decoder = FakeDecoder()
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=second_decoder,
    )
    assert first_decoder.calls
    assert second_decoder.calls == []


def test_coverage_rejection_is_business_terminal_and_skips_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(visible_slots=6),
        decoder=FakeDecoder(),
    )

    rows = _read_jsonl(Path(result["path"]))
    assert rows[0]["status"] == "coverage_rejected"
    assert rows[0]["background_status"] is None


def test_background_pending_remove_is_published_after_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    row = _read_jsonl(Path(result["path"]))[0]
    assert row["status"] == "ready_background_pending_remove"
    mask_paths = list((fixture.output_root / row["artifact_root"]).rglob("source_mask_*.png"))
    assert len(mask_paths) == 1


def test_torn_partial_tail_is_truncated_and_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    completed = Path(result["path"])
    first = completed.read_bytes().splitlines(keepends=True)[0]
    partial = completed.with_name(f"{completed.name}.partial")
    completed.unlink()
    partial.write_bytes(first + b'{"source_index":1')

    resumed = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("artifacts must be reused")),
        decoder=FakeDecoder(),
    )

    assert len(_read_jsonl(Path(resumed["path"]))) == 2


def test_changed_stage1_shard_sha_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    changed = _row(fixture, 0)
    changed["warnings"] = ["changed"]
    shard.write_text(json.dumps(changed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata mismatch"):
        process_shard(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=FakeBackend(),
            decoder=FakeDecoder(),
        )


@pytest.mark.parametrize("artifact", ["frames", "masks", "background"])
def test_completed_shard_fails_closed_on_corrupt_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0, background=True)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    row = _read_jsonl(Path(result["path"]))[0]
    workspace = fixture.output_root / row["artifact_root"] / "run" / "clips" / "clip-0"
    if artifact == "frames":
        (workspace / "frames" / "00.jpg").write_bytes(b"corrupt")
    elif artifact == "masks":
        (workspace / "masks.rle.json").write_text("{}", encoding="utf-8")
    else:
        mask = next((workspace / "background").glob("source_mask_*.png"))
        mask.write_bytes(b"corrupt")

    with pytest.raises((ValueError, OSError)):
        process_shard(
            shard,
            output_root=fixture.output_root,
            config_identity=fixture.identity,
            backend=FakeBackend(failure=RuntimeError("must not rerun")),
            decoder=FakeDecoder(),
        )


def test_invalid_source_index_and_clip_uid_prefix_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    bad = _row(fixture, 1)
    shard = fixture.input_root / "parts" / "shard-000000000-000000009.jsonl"
    shard.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not contiguous"):
        load_annotation_shard(shard)


def test_flock_claim_is_nonblocking_and_released(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.lock"
    with shard_lock(path), pytest.raises(ShardBusyError), shard_lock(path):
        pass
    with shard_lock(path):
        pass


def test_inspect_and_inventory_report_stage_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    inspected = inspect_stage2_shard(result["path"])
    report = inventory(fixture.input_root, fixture.output_root)
    assert inspected["rows"] == 1
    assert inspected["sam3_entity_ready"] == 1
    assert report["stage2_completed_shards"] == 1
    assert report["outstanding_shards"] == 0


def test_canary_selection_is_deterministic_and_counts_only_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    first_rows = [
        _row(fixture, 0, status="failed"),
        _row(fixture, 1, entities=0),
        *[_row(fixture, index) for index in range(2, 7)],
    ]
    _write_shard(fixture, first_rows, nominal_end=9_999)
    for shard_index in range(1, 17):
        start = shard_index * 10_000
        _write_shard(
            fixture,
            [_row(fixture, start + offset) for offset in range(5)],
            nominal_end=start + 9_999,
        )

    first = select_canary_rows(
        fixture.input_root,
        gpu_count=8,
        canary_shards=16,
        samples_per_shard=5,
    )
    second = select_canary_rows(
        fixture.input_root,
        gpu_count=8,
        canary_shards=16,
        samples_per_shard=5,
    )

    assert first == second
    assert len(first) == 80
    assert [sum(item.gpu_slot == slot for item in first) for slot in range(8)] == [10] * 8
    assert [item.source_index for item in first[:5]] == [2, 3, 4, 5, 6]
    assert {Path(item.input_annotation_shard).name for item in first}.isdisjoint(
        {"shard-000160000-000169999.jsonl"}
    )


def test_canary_selection_rejects_short_shard_without_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture, shard_count=2, eligible_per_shard=4)
    start = 20_000
    _write_shard(
        fixture,
        [_row(fixture, start + offset) for offset in range(5)],
        nominal_end=start + 9_999,
    )

    with pytest.raises(ValueError, match=r"shard-000000000-000009999.*4 eligible.*5 required"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=1,
            canary_shards=2,
            samples_per_shard=5,
        )


def test_canary_selection_rejects_cross_shard_duplicate_clip_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture, shard_count=2, eligible_per_shard=5)
    second = fixture.input_root / "parts" / "shard-000010000-000019999.jsonl"
    rows = _read_jsonl(second)
    rows[0]["clip_uid"] = "clip-0"
    second.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate clip_uid across Stage1 shards"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=1,
            canary_shards=2,
            samples_per_shard=5,
        )


def test_canary_selection_requires_divisible_gpu_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="must be divisible"):
        select_canary_rows(
            fixture.input_root,
            gpu_count=8,
            canary_shards=3,
            samples_per_shard=5,
        )


def test_canary_output_cannot_enter_production_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    production_root = fixture.writable / "formal" / "pre-qwen-v1"
    monkeypatch.setattr(production, "DEFAULT_PRODUCTION_OUTPUT_ROOT", production_root)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)

    for output_root in (
        production_root,
        production_root / "fake-canary",
        production_root.parent,
    ):
        with pytest.raises(ValueError, match="fully disjoint"):
            prepare_canary(
                input_root=fixture.input_root,
                base_config=base,
                output_root=output_root,
                gpus=["0"],
                canary_shards=1,
                samples_per_shard=1,
            )


def test_prepare_canary_persists_and_reuses_exact_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_canary_shards(fixture)
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "run-1"

    manifest, first = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )
    monkeypatch.setattr(
        production,
        "select_canary_rows",
        lambda *args, **kwargs: pytest.fail("persisted selection must be reused"),
    )
    resumed_manifest, second = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=[str(index) for index in range(8)],
        canary_shards=16,
        samples_per_shard=5,
    )

    assert manifest == resumed_manifest
    assert first == second
    with pytest.raises(ValueError, match="identity"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=base,
            output_root=output,
            gpus=["0", "1"],
            canary_shards=16,
            samples_per_shard=5,
        )
    assert manifest.selection_policy == "first_n_shards_first_k_eligible_v1"
    assert manifest.selected_shard_names == [
        f"shard-{index * 10_000:09d}-{index * 10_000 + 9_999:09d}.jsonl"
        for index in range(16)
    ]
    assert manifest.duplicate_clip_uid_skipped == 0


def test_canary_resume_rejects_changed_input_shard_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "sha"
    prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    changed = _row(fixture, 0)
    changed["warnings"] = ["mutated"]
    shard.write_text(json.dumps(changed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA changed"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=base,
            output_root=output,
            gpus=["0"],
            canary_shards=1,
            samples_per_shard=1,
        )


def test_qwen_anchor_canary_uses_original_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture, qwen_rescue=True)
    output = fixture.writable / "canary" / "qwen-anchor"

    manifest, _selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    assert manifest.sam3_multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert manifest.sam3_anchor_qwen_enabled is True
    assert manifest.candidate_judge_base_url == "http://127.0.0.1:8000/v1"
    assert "qwen_required" not in manifest.model_dump()
    assert "qwen_calls" not in manifest.model_dump()


def test_off_mode_canary_cannot_resume_with_qwen_anchor_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    output = fixture.writable / "canary" / "off-mode-history"
    prepare_canary(
        input_root=fixture.input_root,
        base_config=_write_loadable_config(fixture),
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    with pytest.raises(ValueError, match="identity"):
        prepare_canary(
            input_root=fixture.input_root,
            base_config=_write_loadable_config(fixture, qwen_rescue=True),
            output_root=output,
            gpus=["0"],
            canary_shards=1,
            samples_per_shard=1,
        )


def test_canary_worker_resume_reuses_completed_masks_and_summary_counts_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "run"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    first = FakeBackend()
    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend(failure=RuntimeError("must not run"))
    resumed = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=second,
        decoder=FakeDecoder(),
    )
    summary = canary_summary(output, manifest)

    assert result["completed"] == 1
    assert resumed["skipped"] is True
    assert second.calls == []
    assert summary["status"] == "complete"
    assert summary["selection_complete"] is True
    assert summary["functional_status"] == "pass"
    assert summary["failed_input"] == 0
    assert summary["failed_frames"] == 0
    assert summary["terminal_failures"] == 0
    assert summary["duplicate_clip_uid_skipped"] == 0
    assert canary_tool._canary_exit_code([], summary) == 0
    assert summary["frames_bytes"] > 0
    assert summary["masks_bytes"] > 0
    assert summary["annotation_qwen_calls"] == 0
    assert summary["background_remove_qwen_calls"] == 0
    assert "qwen_calls" not in summary


def test_canary_sam3_counters_use_backend_deltas_not_per_clip_cumulative_sum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "counter-deltas"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=2,
    )
    backend = CountingBackend()

    run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=backend,
        decoder=FakeDecoder(),
    )
    summary = canary_summary(output, manifest)

    assert summary["anchor_probe_calls"] == 4
    assert summary["anchor_fast_path_hits"] == 2
    assert summary["multi_instance_rescue_attempted"] == 2
    assert summary["multi_instance_rescue_selected"] == 2


def test_canary_retryable_sam_failure_keeps_worker_partial_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "retry"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )

    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=FakeBackend(failure=RuntimeError("CUDA context error")),
        decoder=FakeDecoder(),
    )

    assert result["retryable"] is True
    assert not (output / "workers" / "gpu-0.jsonl").exists()
    summary = canary_summary(output, manifest)
    assert summary["status"] == "partial"
    assert summary["selection_complete"] is False
    assert summary["functional_status"] == "fail"
    assert summary["outstanding_retryable_work"] == 1
    assert canary_tool._canary_exit_code([], summary) == 2


@pytest.mark.parametrize("terminal_status", ["failed_input", "failed_frames"])
def test_canary_terminal_row_completes_selection_but_fails_functionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "terminal"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    item = selection[0]
    terminal = production.Stage2Row(
        source_index=item.source_index,
        clip_uid=item.clip_uid,
        input_annotation_shard=item.input_annotation_shard,
        input_annotation_shard_sha256=item.input_annotation_shard_sha256,
        input_row_sha256=item.input_row_sha256,
        status=terminal_status,
        reason="invalid input or frame stream",
    )
    worker_path = output / "workers" / "gpu-0.jsonl"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(
        json.dumps(terminal.model_dump(mode="json")) + "\n", encoding="utf-8"
    )

    summary = canary_summary(output, manifest)

    assert summary["status"] == "complete"
    assert summary["selection_complete"] is True
    assert summary["functional_status"] == "fail"
    assert summary[terminal_status] == 1
    assert summary["terminal_failures"] == 1
    assert canary_tool._canary_exit_code([], summary) == 2


def test_retryable_canary_worker_cli_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "worker-exit"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=1,
    )
    monkeypatch.setattr(
        canary_tool, "_load_identity_files", lambda root: (manifest, selection)
    )
    monkeypatch.setattr(canary_tool, "default_backend_factory", lambda config: object())
    monkeypatch.setattr(
        canary_tool,
        "run_canary_worker",
        lambda **kwargs: {
            "gpu_slot": 0,
            "selected": 1,
            "completed": 0,
            "retryable": True,
        },
    )

    with pytest.raises(SystemExit) as raised:
        canary_tool.main(["--output-root", str(output), "--worker-slot", "0"])

    assert raised.value.code == 2


def test_canary_partial_worker_resumes_from_exact_next_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture)
    output = fixture.writable / "canary" / "partial"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["0"],
        canary_shards=1,
        samples_per_shard=2,
    )
    first = FakeBackend(
        failure=RuntimeError("CUDA OOM"),
        fail_on_call=2,
    )
    result = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=first,
        decoder=FakeDecoder(),
    )
    second = FakeBackend()
    second_decoder = FakeDecoder()

    resumed = run_canary_worker(
        output_root=output,
        manifest=manifest,
        selection=selection,
        gpu_slot=0,
        backend=second,
        decoder=second_decoder,
    )

    assert result["completed"] == 1
    assert resumed["completed"] == 2
    assert second.calls == ["e1"]
    assert second_decoder.calls == []


def test_stage2_backend_factory_uses_builder_default_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    candidate_judge = replace(
        fixture.config.qwen.candidate_judge,
        base_url="http://6.167.57.88:8000/v1",
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    physical_index_config = replace(
        fixture.config,
        qwen=replace(fixture.config.qwen, candidate_judge=candidate_judge),
        sam3=replace(
            fixture.config.sam3,
            device="cuda:7",
            object_rescue_mode="phrase_retry_v1",
            not_found_rescue_mode="entity_phrase_retry_v1",
            multi_instance_rescue_mode="qwen_anchor_select_v1",
            anchor_search_mode="progressive_v1",
        ),
    )

    backend = production.default_backend_factory(physical_index_config)

    assert isinstance(backend, Sam3SegmentationBackend)
    assert backend.config.device == "cuda"
    assert backend.config.object_rescue_mode == "phrase_retry_v1"
    assert backend.config.not_found_rescue_mode == "entity_phrase_retry_v1"
    assert backend.config.multi_instance_rescue_mode == "qwen_anchor_select_v1"
    assert backend.config.anchor_search_mode == "progressive_v1"
    assert isinstance(backend._anchor_selector, QwenSam3AnchorSelector)
    assert backend._anchor_selector.config is candidate_judge
    assert backend._anchor_selector.config.base_url == "http://6.167.57.88:8000/v1"
    assert physical_index_config.sam3.device == "cuda:7"
    assert production._workspace_config(
        physical_index_config, fixture.writable / "workspace"
    ).sam3.device == "cuda:0"


def test_stage2_backend_lazy_init_supports_builder_without_device_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    assert fixture.config.sam3.model_path is not None
    fixture.config.sam3.model_path.write_bytes(b"sam3-checkpoint")
    backend = production.default_backend_factory(
        replace(
            fixture.config,
            sam3=replace(fixture.config.sam3, device="cuda:5"),
        )
    )
    assert isinstance(backend, Sam3SegmentationBackend)
    calls: list[str] = []
    predictor = object()

    def installed_builder_without_device(checkpoint_path: str) -> object:
        calls.append(checkpoint_path)
        return predictor

    backend._builder = installed_builder_without_device

    assert backend._load_predictor() is predictor
    assert calls == [str(fixture.config.sam3.model_path.resolve())]


def test_frames_ready_checkpoint_resumes_with_fixed_runtime_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    first_decoder = FakeDecoder()
    first = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("SAM3 startup failed")),
        decoder=first_decoder,
    )
    workspace = fixture.output_root / "artifacts" / shard.stem / "clip-0"
    checkpoint = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
    fixed_backend = FakeBackend()
    effective_devices: list[str] = []

    def build_backend(config: V3Config) -> FakeBackend:
        effective_devices.append(config.sam3.device)
        return fixed_backend

    monkeypatch.setattr(production, "build_sam3_segment_backend", build_backend)
    backend = production.default_backend_factory(fixture.config)
    resumed_decoder = FakeDecoder()
    resumed = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=backend,
        decoder=resumed_decoder,
    )

    assert first["retryable"] is True
    assert checkpoint["stage"] == "frames_ready"
    assert first_decoder.calls
    assert resumed["rows"] == 1
    assert effective_devices == ["cuda"]
    assert fixed_backend.calls == ["e1"]
    assert resumed_decoder.calls == []


def test_canary_launcher_isolates_physical_gpus_with_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    _write_shard(fixture, [_row(fixture, 0), _row(fixture, 1)])
    base = _write_loadable_config(fixture, qwen_rescue=True)
    payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    payload["qwen"]["candidate_judge"]["base_url"] = (
        "http://6.167.57.88:8000/v1"
    )
    base.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = fixture.writable / "canary" / "gpu-isolation"
    manifest, selection = prepare_canary(
        input_root=fixture.input_root,
        base_config=base,
        output_root=output,
        gpus=["5", "7"],
        canary_shards=1,
        samples_per_shard=2,
    )
    launched: list[dict[str, object]] = []

    class Process:
        def wait(self) -> int:
            return 0

    def popen(command: list[str], **kwargs: object) -> Process:
        launched.append({"command": command, **kwargs})
        return Process()

    monkeypatch.setattr(
        canary_tool,
        "prepare_canary",
        lambda **kwargs: (manifest, selection),
    )
    monkeypatch.setattr(
        canary_tool,
        "canary_summary",
        lambda output_root, current_manifest: {
            "status": "complete",
            "selection_complete": True,
            "functional_status": "pass",
        },
    )
    monkeypatch.setattr(
        canary_tool,
        "check_stage2_candidate_judge_health",
        lambda config: {"candidate_judge_health": "available"},
    )
    monkeypatch.setattr(canary_tool.subprocess, "Popen", popen)

    canary_tool.main(
        [
            "--input-root",
            str(fixture.input_root),
            "--base-config",
            str(base),
            "--output-root",
            str(fixture.output_root),
            "--gpus",
            "5,7",
            "--canary-shards",
            "1",
            "--samples-per-shard",
            "2",
        ]
    )

    assert [item["env"]["CUDA_VISIBLE_DEVICES"] for item in launched] == ["5", "7"]
    assert manifest.candidate_judge_base_url == "http://6.167.57.88:8000/v1"
    assert all("8001" not in " ".join(item["command"]) for item in launched)
    assert all("8008" not in " ".join(item["command"]) for item in launched)


def test_new_stage1_shard_is_discovered_without_topology_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    first = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        first,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    _write_shard(fixture, [_row(fixture, 1)])

    report = inventory(fixture.input_root, fixture.output_root)

    assert report["stage1_completed_shards"] == 2
    assert report["stage2_completed_shards"] == 1
    assert report["outstanding_shards"] == 1


def test_claim_worker_builds_one_persistent_backend_for_multiple_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shards = [
        _write_shard(fixture, [_row(fixture, index)]) for index in range(2)
    ]
    backend = FakeBackend()
    factory_calls: list[V3Config] = []
    process_backends: list[object] = []
    monkeypatch.setattr(
        batch_tool,
        "load_config_identity",
        lambda path: fixture.identity,
    )
    monkeypatch.setattr(
        batch_tool,
        "enumerate_annotation_shards",
        lambda root: shards,
    )

    def factory(config: V3Config) -> FakeBackend:
        factory_calls.append(config)
        return backend

    def process(path: Path, **kwargs: object) -> dict[str, object]:
        process_backends.append(kwargs["backend"])
        destination = fixture.output_root / "parts" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("done\n", encoding="utf-8")
        return {"path": str(destination), "rows": 1, "skipped": False}

    monkeypatch.setattr(batch_tool, "default_backend_factory", factory)
    monkeypatch.setattr(batch_tool, "process_shard", process)

    batch_tool.run_claim_loop(
        input_root=fixture.input_root,
        base_config=fixture.base_path,
        output_root=fixture.output_root,
        scan_offset=1,
    )

    assert len(factory_calls) == 1
    assert process_backends == [backend, backend]
    assert backend.close_calls == 1


def test_final_partial_sized_stage1_shard_keeps_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(
        fixture,
        [_row(fixture, 10000), _row(fixture, 10001)],
        nominal_end=19999,
    )

    result = process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )

    assert Path(result["path"]).name == "shard-000010000-000019999.jsonl"


def test_machine_topology_is_absent_from_shard_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _paths(tmp_path, monkeypatch)
    shard = _write_shard(fixture, [_row(fixture, 0)])
    process_shard(
        shard,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    meta = json.loads(
        (fixture.output_root / "parts" / f"{shard.stem}.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert not {"rank", "world_size", "gpu", "hostname"}.intersection(meta)


def test_no_remove_or_qwen_modules_are_constructed_by_stage2_source() -> None:
    source = Path(production.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "QwenAnnotationClient",
        "QwenBackgroundRemovalJudge",
        "QwenBackgroundFinalGuard",
        "QwenCrossPairJudge",
        "QwenReferenceEditJudge",
        "QwenReferenceIntegrityJudge",
        "QwenInstructionWriter",
        "QwenSubjectAttributeClient",
        "Boogu",
    ):
        assert forbidden not in source
    assert "QwenSam3AnchorSelector" not in source
    assert "build_sam3_segment_backend" in source
    assert "build_background_candidates" in source
