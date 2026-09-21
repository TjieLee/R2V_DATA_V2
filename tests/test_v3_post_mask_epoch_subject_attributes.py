"""Subject Attributes resource epoch (8a): durable owner + discovery foundation.

Every fixture here compares the epoch against the legacy authority
``subject_attributes.process_subject_attribute_clip`` on the same real clip: the
legacy owner artifact bytes, the legacy clip counts and the legacy terminal
semantics. SAM, review and completion are forbidden in these tests, because 8a
must never reach them.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import tests.test_v3_reference_integrity as legacy_integrity
from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    JobExecution,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
    SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
    SubjectAttributeDurableError,
    SubjectAttributeEpochError,
    SubjectAttributeEpochRunner,
)
from r2v_data_v2.v3.subject_attributes import (
    DiscoveredSubjectAttribute,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    SubjectAttributeDiscovery,
    _owner_artifact_path,
    process_subject_attribute_clip,
)

SHARD = "shard-000000000-000000000"
CLIP_UID = "clip-1"
OWNER = "e1"
OUTPUT_DIRNAME = "subject_attributes"


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


def _config_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str) -> Any:
    config = legacy_integrity._config(tmp_path, monkeypatch)
    config = replace(
        config,
        run_root=config.run_root.parent / run_name,
        reference_edit=replace(
            config.reference_edit,
            enabled=False,
            # The fixture clip is small; keep the legacy candidate pool usable.
            min_source_content_area_pixels=1,
            min_source_content_long_side_pixels=1,
        ),
        reference_integrity=replace(config.reference_integrity, enabled=False),
    )
    config.validate()
    return config


def _storage_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """The legacy ready-pair clip on its own run root."""
    config = _config_for(tmp_path, monkeypatch, run_name)
    monkeypatch.setattr(legacy_integrity, "_config", lambda *args, **kw: config)
    storage = legacy_integrity._storage_with_ready_pair(tmp_path, monkeypatch, **kwargs)
    return config, storage


def _runner(config: Any, storage: Any, tmp_path: Path) -> Any:
    return SubjectAttributeEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: [CLIP_UID]},
    )


def _output_root(storage: Any) -> Path:
    return Path(storage.root) / OUTPUT_DIRNAME


def _owner_artifact_file(storage: Any, owner: str = OWNER) -> Path:
    return _owner_artifact_path(
        _output_root(storage), sample_id=CLIP_UID, owner_entity_id=owner
    )


class _DiscoveryClient:
    """The legacy ``SubjectAttributeDiscoveryClient`` protocol, scripted."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def discover(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.requests.append(kwargs)
        result = self.results[self.calls - 1]
        if isinstance(result, Exception):
            raise result
        return result


class _ForbiddenClient:
    """Any call proves 8a leaked into SAM / review / completion."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        def _call(**kwargs: Any) -> Any:
            del kwargs
            self.calls += 1
            raise AssertionError(f"{self.label} must never be called in 8a")

        return _call


class _SerialQwenExecutor:
    """Serial executor for the real scheduler; one resource, no real model."""

    def __init__(self, runner: Any, handle: Any) -> None:
        self.runner = runner
        self.handle = handle
        self.batches = 0

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        self.batches += 1
        outcomes: dict[str, Any] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner.run(job, self.handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


def _scheduler(runner: Any, executor: Any, finalize: Any = None) -> Any:
    return ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=finalize or runner.finalize,
        executors={RESOURCE_QWEN: executor},
    )


def _drain(
    runner: Any,
    client: Any,
    *,
    finalize: Any = None,
) -> tuple[Any, list[Any]]:
    executor = _SerialQwenExecutor(runner, client)
    jobs = runner.seed_jobs()
    if jobs:
        _scheduler(runner, executor, finalize).run(jobs)
    return executor, jobs


def _legacy_run(
    config: Any, storage: Any, client: Any
) -> tuple[Any, _ForbiddenClient, _ForbiddenClient]:
    """The legacy authority on the same clip, with 8b surfaces forbidden."""
    review = _ForbiddenClient("review")
    segmentation = _ForbiddenClient("segmentation")
    result = process_subject_attribute_clip(
        config,
        storage=storage,
        output_root=_output_root(storage),
        clip=storage.read_clip(CLIP_UID),
        discovery_client=client,
        review_client=review,
        segmentation_backend=segmentation,
    )
    assert review.calls == 0
    assert segmentation.calls == 0
    return result, review, segmentation


def _human_discovery(owner: str = OWNER) -> SubjectAttributeDiscovery:
    return SubjectAttributeDiscovery(
        owner_entity_id=owner,
        owner_is_human=True,
        attributes=[
            DiscoveredSubjectAttribute(
                attribute_type="headwear",
                phrase="a red cap",
                grounding_prompt="red cap on the head",
            )
        ],
    )


def _nonhuman_discovery(owner: str = OWNER) -> SubjectAttributeDiscovery:
    return SubjectAttributeDiscovery(
        owner_entity_id=owner, owner_is_human=False, attributes=[]
    )


def _read_artifact(storage: Any, owner: str = OWNER) -> Any:
    return OwnerEnrichmentArtifact.model_validate_json(
        _owner_artifact_file(storage, owner).read_text(encoding="utf-8")
    )


def _preexisting_nonhuman_artifact(storage: Any) -> OwnerEnrichmentArtifact:
    clip = storage.read_clip(CLIP_UID)
    owner = next(item for item in clip.annotation.entities if item.entity_id == OWNER)
    artifact = OwnerEnrichmentArtifact(
        sample_id=CLIP_UID,
        owner_entity_id=OWNER,
        owner_is_human=False,
        attribute_id_start=1,
        owner_phrase=owner.phrase,
        owner_grounding_prompt=owner.grounding_prompt,
        records=[],
        metrics=OwnerEnrichmentMetrics(discovery_calls=1),
        gme_screen_mode=None,
        completion_mode=None,
    )
    path = _owner_artifact_file(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return artifact


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_preexisting_owner_artifact_is_reused_without_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid legacy owner artifact is terminal with zero model calls.

    8a also refuses the GME production path outright: GME is not implemented.
    """
    gme_config, gme_storage = _storage_variant(tmp_path, monkeypatch, "run-gme")
    with pytest.raises(SubjectAttributeEpochError, match="GME disabled"):
        SubjectAttributeEpochRunner(
            replace(
                gme_config,
                subject_attribute_gme=replace(
                    gme_config.subject_attribute_gme, enabled=True
                ),
            ),
            {SHARD: gme_storage},
            GroupLedger(tmp_path / "ledger-gme"),
            eligible_clip_uids_by_shard={SHARD: [CLIP_UID]},
        )

    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    _preexisting_nonhuman_artifact(legacy_storage)
    legacy_client = _DiscoveryClient()
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 0, "the cached path pays nothing"
    assert legacy_result.totals.skipped_existing_owners == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    artifact = _preexisting_nonhuman_artifact(storage)
    artifact_bytes = _owner_artifact_file(storage).read_bytes()
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient()
    executor, jobs = _drain(runner, client)

    assert jobs == [], "a valid cache needs no discovery job"
    assert executor.batches == 0
    assert client.calls == 0
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["skipped_existing_owners"] == 1
    assert stats.to_dict()["discovery_calls"] == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1
    assert stats.terminal_clips == 1

    marker = json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "outcome.json"
        ).read_text(encoding="utf-8")
    )
    assert marker["source"] == "preexisting"
    assert marker["discovery_job_id"] is None
    assert marker["record_count"] == 0
    # The reused artifact is never rewritten.
    assert _owner_artifact_file(storage).read_bytes() == artifact_bytes
    assert _read_artifact(storage) == artifact


def test_discovery_nonhuman_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed non-human owner publishes exactly the legacy artifact."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    legacy_client = _DiscoveryClient(_nonhuman_discovery())
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert jobs[0].job_type == SUBJECT_ATTRIBUTE_DISCOVERY_JOB
    assert jobs[0].resource == RESOURCE_QWEN
    assert dict(jobs[0].target) == {"owner_entity_id": OWNER}
    assert jobs[0].model_identity == f"qwen:{config.qwen.candidate_judge.model}"
    assert executor.batches == 1
    assert client.calls == 1, "exactly one discovery call"

    assert _owner_artifact_file(storage).read_bytes() == (
        _owner_artifact_file(legacy_storage).read_bytes()
    ), "the epoch reuses the legacy OwnerEnrichmentArtifact schema and bytes"
    epoch_artifact = _read_artifact(storage)
    assert epoch_artifact.owner_is_human is False
    assert epoch_artifact.failure_reason is None
    assert epoch_artifact.metrics.discovery_calls == 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["discovery_calls"] == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1
    assert stats.to_dict()["failures"] == 0
    assert not (_output_root(storage) / "samples" / f"{CLIP_UID}.json").exists()


def test_discovery_failure_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discovery exception is the legacy terminal failure, not a retry."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    legacy_client = _DiscoveryClient(ValueError("discovery exploded"))
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(ValueError("discovery exploded"))
    _executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert client.calls == 1
    assert _owner_artifact_file(storage).read_bytes() == (
        _owner_artifact_file(legacy_storage).read_bytes()
    )
    artifact = _read_artifact(storage)
    assert artifact.owner_is_human is None
    assert artifact.records == []
    assert artifact.failure_reason == "discovery_failed:ValueError:discovery exploded"
    assert artifact.metrics.failures == 1
    assert artifact.metrics.discovery_calls == 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["failures"] == 1
    assert stats.to_dict()["eligible_human_owners"] == 0


def test_discovery_receipt_before_finalizer_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt committed before the CPU finalizer replays without new calls."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    _executor, jobs = _drain(runner, client, finalize=lambda job, result: ())

    assert len(jobs) == 1
    assert client.calls == 1
    assert not _owner_artifact_file(storage).exists()
    assert not (_output_root(storage) / "samples").exists()

    fresh = _runner(config, storage, tmp_path)
    fresh_executor, fresh_jobs = _drain(fresh, client)

    assert len(fresh_jobs) == 1, "the same frozen job is re-seeded"
    assert fresh_jobs[0].job_id() == jobs[0].job_id()
    assert fresh_executor.batches == 0, "the committed receipt is reused"
    assert client.calls == 1, "restart must not pay another discovery call"
    assert _owner_artifact_file(storage).is_file()
    stats = fresh.reconcile_stats(SHARD)
    assert stats.terminal_clips == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1

    # The marker is never its own authority: a tampered outcome must fail closed
    # against the committed receipt and the artifact it claims.
    outcome_path = (
        Path(fresh.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "outcome.json"
    )
    marker = json.loads(outcome_path.read_text(encoding="utf-8"))
    marker["record_count"] = 7
    outcome_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(SubjectAttributeDurableError, match="owner outcome"):
        _runner(config, storage, tmp_path).reconcile_stats(SHARD)


def test_human_discovery_receipt_defers_to_8b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed human owner keeps only its receipt and stays unresolved."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_human_discovery())
    executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert client.calls == 1
    assert executor.batches == 1
    # No artifact and no epoch outcome: 8a must not fake a human terminal.
    assert not _owner_artifact_file(storage).exists()
    outcome_path = (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "outcome.json"
    )
    assert not outcome_path.exists()
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        runner.reconcile_stats(SHARD)

    fresh = _runner(config, storage, tmp_path)
    fresh_executor, fresh_jobs = _drain(fresh, client)
    assert len(fresh_jobs) == 1
    assert fresh_jobs[0].job_id() == jobs[0].job_id()
    assert fresh_executor.batches == 0, "the human receipt is reused"
    assert client.calls == 1, "0 fresh Qwen on the human defer path"
    assert not _owner_artifact_file(storage).exists()
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        fresh.reconcile_stats(SHARD)


def test_frozen_discovery_input_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered frozen candidate source image is durable corruption."""
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    jobs = runner.seed_jobs()

    assert len(jobs) == 1
    assert client.calls == 0
    owner_plan = json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "plan.json"
        ).read_text(encoding="utf-8")
    )
    candidate = owner_plan["candidates"][0]
    source = Path(storage.root) / candidate["image_path"]
    assert source.is_file()
    original = source.read_bytes()
    # Trailing bytes keep the JPEG decodable, so every derived metric (and the
    # discovery context the model sees) is unchanged: only the frozen source
    # image digest can catch this.
    source.write_bytes(original + b"trailing-tamper")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(
        SubjectAttributeDurableError, match="clip plan"
    ):
        fresh.seed_jobs()

    assert client.calls == 0, "no model call may be paid for corruption"
    assert not _owner_artifact_file(storage).exists(), (
        "corruption must never be written as a legacy owner failure"
    )
    assert not (_output_root(storage) / "samples").exists()
    failures_path = Path(storage.root) / "failures.jsonl"
    assert not failures_path.is_file() or not failures_path.read_text(
        encoding="utf-8"
    ).strip()
