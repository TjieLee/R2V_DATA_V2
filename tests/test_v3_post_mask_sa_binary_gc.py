"""Only terminal Subject Attribute intermediates may be garbage-collected."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_online_worker_is_lazy_bounded_and_scan_does_not_block_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from r2v_data_v2.v3 import post_mask_epoch_subject_attributes as sa

    entered, release = threading.Event(), threading.Event()
    cleaned = []
    scans = []

    def scan(root):
        scans.append(root)
        entered.set()
        assert release.wait(5)
        return {(SHARD, "clip-a"): sa.BinaryJobIds(sam={SAM_A})}

    monkeypatch.setattr(sa, "load_binary_job_index", scan)
    worker = sa._TerminalBinaryGC(Path("unused"), lambda *args: cleaned.append(args), capacity=1)
    assert not scans
    assert worker.submit(SHARD, "clip-a", frozenset(), frozenset())
    assert entered.wait(5)
    assert worker.submit(SHARD, "clip-b", frozenset({SAM_B}), frozenset())
    assert not worker.submit(SHARD, "clip-c", frozenset(), frozenset())
    release.set()
    worker.close()
    assert len(scans) == 1
    assert cleaned[0][2].sam == {SAM_A}
    assert cleaned[1][2].sam == {SAM_B}
    assert not worker.thread.is_alive()


def test_online_worker_cleanup_does_not_block_next_snapshot(monkeypatch) -> None:
    from r2v_data_v2.v3 import post_mask_epoch_subject_attributes as sa

    entered, release = threading.Event(), threading.Event()
    seen = []
    monkeypatch.setattr(sa, "load_binary_job_index", lambda root: {})

    def cleanup(shard, clip, jobs):
        seen.append(jobs.sam)
        entered.set()
        assert release.wait(5)

    worker = sa._TerminalBinaryGC(Path("unused"), cleanup, capacity=1)
    worker.submit(SHARD, "clip-a", frozenset({SAM_A}), frozenset())
    assert entered.wait(5)
    assert worker.submit(SHARD, "clip-b", frozenset({SAM_B}), frozenset())
    release.set()
    worker.close()
    assert seen == [{SAM_A}, {SAM_B}]


def test_current_commit_during_historic_scan_is_in_terminal_snapshot(monkeypatch):
    from r2v_data_v2.v3 import post_mask_epoch_subject_attributes as sa

    entered, release = threading.Event(), threading.Event()
    seen = []

    def scan(root):
        entered.set()
        assert release.wait(5)
        return {}

    monkeypatch.setattr(sa, "load_binary_job_index", scan)
    runner = sa.SubjectAttributeEpochRunner.__new__(sa.SubjectAttributeEpochRunner)
    runner.ledger = SimpleNamespace(root="unused")
    runner._binary_job_index = {}
    runner._binary_job_index_lock = threading.Lock()
    runner._terminal_binary_gc = None
    runner._committed_payload_cache = {}
    runner._committed_payload_cache_lock = threading.Lock()
    runner._cleanup_terminal_clip_binary = lambda *args: seen.append(args)
    runner._gc_new_terminal_clip_binary(SHARD, "clip-a")
    assert entered.wait(5)
    job = SimpleNamespace(
        job_type=sa.SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB,
        canonical_shard=SHARD, clip_uid="clip-b", job_id=lambda: COMPLETION_B,
    )
    runner._cache_committed_result(job, SimpleNamespace(payload={}))
    runner._gc_new_terminal_clip_binary(SHARD, "clip-b")
    release.set()
    runner.close_online_binary_gc()
    assert seen[1][2].completion == {COMPLETION_B}


@pytest.mark.parametrize("failure", ["seed", "reconcile"])
def test_sa_stage_exception_joins_gc_worker(tmp_path, monkeypatch, failure) -> None:
    from r2v_data_v2.v3 import post_mask_epoch_pipeline as pipeline
    from r2v_data_v2.v3 import post_mask_epoch_subject_attributes as sa
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    cleaned = []
    monkeypatch.setattr(sa, "load_binary_job_index", lambda root: {})
    worker = sa._TerminalBinaryGC(tmp_path, lambda *args: cleaned.append(args))

    class Runner:
        def seed_job_batches(self, size):
            worker.submit(SHARD, "clip-a", frozenset({SAM_A}), frozenset())
            if failure == "seed":
                raise RuntimeError("seed failed")
            return iter(())

        def close_online_binary_gc(self):
            worker.close()

    class Scheduler:
        def run_batches(self, batches):
            list(batches)
            return {"unresolved_job_ids": ()}

    def reconcile(**kwargs):
        raise RuntimeError("reconcile failed")

    monkeypatch.setattr(pipeline, "_reconcile_and_publish_subject_attribute_stats", reconcile)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        pipeline.run_subject_attributes_stage(
            config=object(), storages={}, eligible={}, ledger=GroupLedger(tmp_path / "ledger"),
            result={}, subject_attributes_runner_factory=lambda **kwargs: Runner(),
            subject_attributes_scheduler_factory=lambda runner: Scheduler(),
        )
    assert cleaned
    assert not worker.thread.is_alive()

SHARD = "shard-000000000-000009999"
SAM_A = "a" * 64
COMPLETION_A = "b" * 64
SAM_B = "c" * 64
COMPLETION_B = "d" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _group(state_root: Path) -> Path:
    group = state_root / "group-000000"
    _write_json(
        group / "group.json",
        {
            "schema_version": "post_mask_resource_epoch_group/1",
            "group_size": 8,
            "campaign_identity": "existing-campaign",
            "canonical_shards": [SHARD],
            "group_id": group.name,
            "group_index": 0,
            "group_identity": "existing-group-identity",
        },
    )
    return group


def _job(job_id: str, job_type: str, clip_uid: str) -> dict:
    return {
        "schema_version": "post_mask_resource_epoch_v3/2",
        "job_type": job_type,
        "resource": "sam" if "sam" in job_type else "boogu",
        "canonical_shard": SHARD,
        "clip_uid": clip_uid,
        "target": [["owner_entity_id", "e1"], ["attribute_id", "a1"]],
        "attempt_index": 0,
        "seed": None,
        "input_digest": "existing-input-digest",
        "dependency_digests": [],
        "model_identity": "existing-model",
        "job_id": job_id,
        "job_identity": job_id,
    }


def _plan(group: Path) -> None:
    jobs = [
        _job(SAM_A, "subject_attribute_sam_probe", "clip-a"),
        _job(COMPLETION_A, "subject_attribute_completion_generate", "clip-a"),
        _job(SAM_B, "subject_attribute_completion_sam", "clip-b"),
        _job(COMPLETION_B, "subject_attribute_completion_generate", "clip-b"),
        _job("e" * 64, "pair_review", "clip-a"),
    ]
    _write_json(
        group / "phases/phase-000001/plan.json",
        {"plan_hash": "existing-plan-hash", "job_count": len(jobs), "jobs": jobs},
    )


def _terminal(group: Path, clip_uid: str) -> None:
    from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
        SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA,
        SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA,
    )
    from r2v_data_v2.v3.subject_attributes import ClipEnrichmentResult, EnrichmentTotals

    counts = ClipEnrichmentResult(
        clip_uid=clip_uid,
        totals=EnrichmentTotals(eligible_human_owners=1),
        owner_limit_reached=False,
        enriched_sample=None,
    ).to_counts()

    _write_json(
        group / "semantic/subject_attributes/plans" / SHARD / f"{clip_uid}.json",
        {
            "schema": SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA,
            "clip_uid": clip_uid,
            "policy": {},
            "clip_digest": "existing-clip-digest",
            "effective_export": {},
            "frames_sha256": "existing-frames-digest",
            "masks_sha256": "existing-masks-digest",
            "classification": "owners",
            "subject_owners": [{"owner_entity_id": "e1", "eligible": True}],
        },
    )
    _write_json(
        group / "semantic/subject_attributes/outcomes" / SHARD / f"{clip_uid}.json",
        {
            "schema": SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "terminal": "ready",
            "counts": counts,
            "owner_outcome_ids": ["e1"],
            "enriched_sample_path": None,
            "enriched_sample_sha256": None,
        },
    )


def _binary(group: Path, clip_uid: str) -> tuple[Path, Path]:
    sam_id, completion_id = (
        (SAM_A, COMPLETION_A) if clip_uid == "clip-a" else (SAM_B, COMPLETION_B)
    )
    sam = group / "semantic/subject_attributes/binary/sam" / sam_id / "mask-000.npy"
    completion = (
        group
        / "semantic/subject_attributes/binary/completion"
        / completion_id
        / "generated.png"
    )
    sam.parent.mkdir(parents=True, exist_ok=True)
    completion.parent.mkdir(parents=True, exist_ok=True)
    sam.write_bytes(b"intermediate mask")
    completion.write_bytes(b"intermediate generation")
    return sam, completion


def test_index_uses_only_exact_sa_job_types_and_clip_provenance(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import load_binary_job_index

    group = _group(tmp_path)
    _plan(group)

    indexed = load_binary_job_index(group)

    assert indexed[(SHARD, "clip-a")].sam == {SAM_A}
    assert indexed[(SHARD, "clip-a")].completion == {COMPLETION_A}
    assert indexed[(SHARD, "clip-b")].sam == {SAM_B}
    assert indexed[(SHARD, "clip-b")].completion == {COMPLETION_B}
    assert all("e" * 64 not in ids.sam | ids.completion for ids in indexed.values())


def test_backfill_dry_run_is_read_only_and_apply_preserves_pending(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root

    state_root = tmp_path / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    a_sam, a_png = _binary(group, "clip-a")
    b_sam, b_png = _binary(group, "clip-b")
    before = {path: path.read_bytes() for path in group.rglob("*") if path.is_file()}

    preview = gc_state_root(state_root, apply=False)

    assert preview["groups_seen"] == 1
    assert preview["terminal_clips_seen"] == 1
    assert {path: path.read_bytes() for path in group.rglob("*") if path.is_file()} == before
    assert not (state_root / "group-000000.lock").exists()

    applied = gc_state_root(state_root, apply=True)

    assert applied["sam_files_cleaned"] == 1
    assert applied["completion_pngs_cleaned"] == 1
    assert not a_sam.exists()
    assert not a_png.exists()
    assert b_sam.read_bytes() == b"intermediate mask"
    assert b_png.read_bytes() == b"intermediate generation"
    assert (group / "semantic/subject_attributes/plans" / SHARD / "clip-a.json").is_file()
    assert (group / "semantic/subject_attributes/outcomes" / SHARD / "clip-a.json").is_file()
    assert (group / "phases/phase-000001/plan.json").is_file()
    assert gc_state_root(state_root, apply=True)["sam_files_cleaned"] == 0


def test_outcome_without_matching_plan_cannot_authorize_deletion(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root

    state_root = tmp_path / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    mask, generated = _binary(group, "clip-a")
    (group / "semantic/subject_attributes/plans" / SHARD / "clip-a.json").unlink()

    result = gc_state_root(state_root, apply=True)

    assert result["failures"] >= 1
    assert mask.is_file()
    assert generated.is_file()


def test_inconsistent_ready_outcome_cannot_authorize_deletion(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root

    state_root = tmp_path / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    mask, generated = _binary(group, "clip-a")
    outcome_path = group / "semantic/subject_attributes/outcomes" / SHARD / "clip-a.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["owner_outcome_ids"] = ["other-owner"]
    _write_json(outcome_path, outcome)

    result = gc_state_root(state_root, apply=True)

    assert result["failures"] >= 1
    assert mask.is_file()
    assert generated.is_file()


def test_invalid_terminal_counter_shape_cannot_authorize_deletion(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root

    state_root = tmp_path / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    mask, generated = _binary(group, "clip-a")
    outcome_path = group / "semantic/subject_attributes/outcomes" / SHARD / "clip-a.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["counts"]["made_up_counter"] = 1
    _write_json(outcome_path, outcome)

    result = gc_state_root(state_root, apply=True)

    assert result["failures"] >= 1
    assert mask.is_file()
    assert generated.is_file()


def test_symlinked_job_directory_is_never_followed(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import gc_state_root

    state_root = tmp_path / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    external = tmp_path / "unrelated"
    external.mkdir()
    protected = external / "mask-000.npy"
    protected.write_bytes(b"keep")
    binary_root = group / "semantic/subject_attributes/binary/sam"
    binary_root.mkdir(parents=True)
    (binary_root / SAM_A).symlink_to(external, target_is_directory=True)

    result = gc_state_root(state_root, apply=True)

    assert result["failures"] >= 1
    assert protected.read_bytes() == b"keep"


def test_unsafe_job_id_is_rejected_without_touching_binary(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import load_binary_job_index

    group = _group(tmp_path)
    bad_job = _job("../../outside", "subject_attribute_sam_probe", "clip-a")
    _write_json(
        group / "phases/phase-000001/plan.json",
        {"plan_hash": "existing", "job_count": 1, "jobs": [bad_job]},
    )

    with pytest.raises(ValueError, match="job_id"):
        load_binary_job_index(group)


def test_same_binary_job_id_cannot_belong_to_two_clips(tmp_path: Path) -> None:
    from r2v_data_v2.v3.post_mask_epoch_sa_binary_gc import load_binary_job_index

    group = _group(tmp_path)
    jobs = [
        _job(SAM_A, "subject_attribute_sam_probe", "clip-a"),
        _job(SAM_A, "subject_attribute_sam_probe", "clip-b"),
    ]
    _write_json(
        group / "phases/phase-000001/plan.json",
        {"plan_hash": "existing", "job_count": len(jobs), "jobs": jobs},
    )

    with pytest.raises(ValueError, match="belongs to multiple clips"):
        load_binary_job_index(group)


def test_cli_uses_fixed_official_root_and_requires_apply_for_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from r2v_data_v2.v3 import config as config_module
    from tools import gc_v3_post_mask_sa_binary as cli

    official = tmp_path / "official"
    state_root = official / "state" / "resource_epochs"
    group = _group(state_root)
    _plan(group)
    _terminal(group, "clip-a")
    mask, generated = _binary(group, "clip-a")
    monkeypatch.setattr(config_module, "OFFICIAL_POST_MASK_EXPORT_ROOT", official)

    assert cli.main([]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["apply"] is False
    assert preview["terminal_clips_seen"] == 1
    assert mask.is_file() and generated.is_file()

    assert cli.main(["--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["apply"] is True
    assert applied["sam_files_cleaned"] == 1
    assert not mask.exists() and not generated.exists()
