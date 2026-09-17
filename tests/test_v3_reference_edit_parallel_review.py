from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from r2v_data_v2.v3 import reference_edit_boogu as module
from r2v_data_v2.v3.profiling import (
    V3Profiler,
    active_profiler,
    get_model_profile_context,
    model_profile_context,
    profiled_openai_call,
    qwen_concurrency_gate,
)
from tests.test_v3_reference_edit_boogu import (
    _Backend,
    _environment,
    _Judge,
    _SamReviewer,
)


def _run(root, judge, sam, **kwargs):
    return module.run_boogu_reference_edit(
        run_root=root, clip_uid="clip-1", entity_id="e1",
        operation=kwargs.pop("operation", "complete_entity"),
        instruction="complete the entity", entity_phrase="green entity",
        reference_type="object", backend=_Backend(), judge=judge,
        sam_reviewer=sam, target_area=256 * 256, **kwargs,
    )


def test_parallel_reviews_start_after_written_candidate_and_before_either_finishes(tmp_path):
    root, _, _ = _environment(tmp_path)
    entered = {"qwen": Event(), "sam": Event()}
    release = Event()
    metrics = []

    def review(kind, delegate):
        def call(**kwargs):
            candidate = root / "clips/clip-1/reference_edit/e1/completion_candidate_1k.png"
            assert candidate.is_file()
            entered[kind].set()
            assert release.wait(5)
            return delegate.review(**kwargs)
        return call

    judge, sam = _Judge(), _SamReviewer()
    judge.review = review("qwen", _Judge())
    sam.review = review("sam", _SamReviewer())
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run, root, judge, sam, review_execution="parallel_independent",
            review_observer=metrics.append,
        )
        try:
            assert entered["qwen"].wait(3)
            assert entered["sam"].wait(3)
            assert not future.done()
        finally:
            release.set()
        assert future.result().status == "accepted"
    assert metrics[0]["reference_complete_parallel_attempts"] == 1
    assert metrics[0]["reference_complete_parallel_both_success"] == 1
    for part in ("qwen", "sam", "wall"):
        assert metrics[0][f"reference_complete_parallel_{part}_seconds"] >= 0


@pytest.mark.parametrize("outcome", ["accept", "qwen_reject", "sam_reject", "qwen_error", "sam_error"])
def test_parallel_preserves_semantic_artifacts_and_inputs(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(module, "new_boogu_seed", lambda: 12345)
    outputs, calls, observations = [], [], []
    for mode in ("sequential", "parallel_independent"):
        root, _, _ = _environment(tmp_path / mode)
        judge = _Judge(accept=outcome != "qwen_reject")
        sam = _SamReviewer(passed=outcome != "sam_reject")
        for kind, reviewer in (("qwen", judge), ("sam", sam)):
            if outcome == f"{kind}_error":
                def fail(_reviewer=reviewer, **kwargs):
                    _reviewer.calls.append(kwargs)
                    raise RuntimeError("review failed")
                reviewer.review = fail
        result = _run(root, judge, sam, review_execution=mode, review_observer=observations.append)
        outputs.append({
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()
        })
        calls.append((judge.calls, sam.calls))
        assert (result.status == "accepted") == (outcome == "accept")
        metadata = json.loads(result.metadata_path.read_text())
        if outcome == "qwen_error":
            assert metadata["qwen_review"] is None
            assert metadata["sam_review"] is None
        elif outcome == "sam_error":
            assert metadata["qwen_review"] is not None
            assert metadata["sam_review"] is None
    assert outputs[0] == outputs[1]
    assert len(calls[0][0]) == len(calls[1][0]) == 1
    assert len(calls[1][1]) == 1
    assert len(calls[0][1]) == (0 if outcome == "qwen_error" else 1)
    for sequential, parallel in zip(calls[0], calls[1]):
        if not sequential:
            continue
        for key, value in sequential[0].items():
            other = parallel[0][key]
            if hasattr(value, "tobytes"):
                assert value.size == other.size
                assert value.tobytes() == other.tobytes()
            else:
                assert value == other
    assert observations[0]["reference_complete_parallel_both_success"] == int("error" not in outcome)


@pytest.mark.parametrize("mode", [None, "sequential", "parallel_independent"])
@pytest.mark.parametrize("hard_failure", [False, True])
def test_background_keeps_sam_then_conditional_qwen(tmp_path, mode, hard_failure):
    root, _, _ = _environment(tmp_path)
    events, metrics = [], []
    judge = _Judge(events=events)
    sam = _SamReviewer(failure_kind="multiple_instances" if hard_failure else "none", events=events)
    kwargs = {} if mode is None else {"review_execution": mode}
    _run(root, judge, sam, operation="add_entity_background", review_observer=metrics.append, **kwargs)
    assert events == (["sam"] if hard_failure else ["sam", "qwen"])
    assert len(judge.calls) == (0 if hard_failure else 1)
    assert metrics == []


def test_default_complete_review_order_is_sequential(tmp_path):
    root, _, _ = _environment(tmp_path)
    events = []
    _run(root, _Judge(events=events), _SamReviewer(events=events))
    assert events == ["qwen", "sam"]


def test_parallel_inherits_existing_qwen_gate_without_double_acquisition(tmp_path):
    root, _, _ = _environment(tmp_path)

    class Gate:
        acquisitions = 0

        @contextmanager
        def acquire(self):
            self.acquisitions += 1
            yield None

    gate = Gate()
    delegate = _Judge()

    class Judge:
        def review(self, **kwargs):
            return profiled_openai_call(
                lambda: delegate.review(**kwargs), component="qwen_reference_edit",
                operation="complete_entity", retry_index=0, model="fake", messages=[],
            )

    with qwen_concurrency_gate(gate):
        assert _run(root, Judge(), _SamReviewer(), review_execution="parallel_independent").status == "accepted"
    assert gate.acquisitions == 1


def test_failed_qwen_drains_sam_before_rejection_publication(tmp_path):
    root, _, _ = _environment(tmp_path)
    sam_started, qwen_failed, release_sam, sam_finished = (Event() for _ in range(4))

    class Judge:
        def review(self, **kwargs):
            assert sam_started.wait(3)
            qwen_failed.set()
            raise RuntimeError("qwen failed")

    class Sam:
        def review(self, **kwargs):
            sam_started.set()
            assert release_sam.wait(5)
            sam_finished.set()
            return _SamReviewer().review(**kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, root, Judge(), Sam(), review_execution="parallel_independent")
        try:
            assert qwen_failed.wait(3)
            assert not future.done()
            assert not (root / "clips/clip-1/reference_edit/e1/completion_metadata.json").exists()
            assert not (root / "clips/clip-1/reference_edit/e1/final_reference_1k.png").exists()
        finally:
            release_sam.set()
        assert future.result().status == "rejected"
    assert sam_finished.is_set()


def test_review_observer_failure_cannot_change_acceptance(tmp_path):
    root, _, _ = _environment(tmp_path)

    def broken_observer(metrics):
        raise RuntimeError("metrics sink unavailable")

    result = _run(root, _Judge(), _SamReviewer(), review_execution="parallel_independent",
                  review_observer=broken_observer)
    assert result.status == "accepted"


def test_both_parallel_reviews_inherit_model_profile_context(tmp_path):
    root, _, _ = _environment(tmp_path)
    profiler = V3Profiler(root, git_commit="test")
    contexts = []

    class Judge(_Judge):
        def review(self, **kwargs):
            contexts.append(get_model_profile_context())
            return super().review(**kwargs)

    class Sam(_SamReviewer):
        def review(self, **kwargs):
            contexts.append(get_model_profile_context())
            return super().review(**kwargs)

    with active_profiler(profiler), model_profile_context(retry_index=2, metadata={"clip_uid": "clip-1"}):
        result = _run(root, Judge(), Sam(), review_execution="parallel_independent")
    assert result.status == "accepted"
    assert len(contexts) == 2
    assert all(context.retry_index == 2 and context.metadata == {"clip_uid": "clip-1"} for context in contexts)
    events = [json.loads(line) for line in profiler.events_path.read_text().splitlines()]
    assert any(event["component"] == "sam3_boogu_review" for event in events)
