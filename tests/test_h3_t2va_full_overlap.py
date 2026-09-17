import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from r2v_data_v2.h3 import t2va_full_production as full


def test_first_canonical_starts_before_pools_and_closes_on_startup_failure(
    tmp_path, monkeypatch
):
    import pytest

    from r2v_data_v2.h3 import t2va_full_prefetch as prefetch_module
    from r2v_data_v2.h3 import t2va_full_worker_pool as pools

    events = []

    class Prefetch:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def start(self, shard):
            events.append(("canonical", shard))

        def __exit__(self, *args):
            events.append("canonical_close")

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            events.append("pools")
            return self

        def start(self, *args, **kwargs):
            raise RuntimeError("startup failed")

        def __exit__(self, *args):
            events.append("pools_close")

    monkeypatch.setattr(prefetch_module, "CanonicalPrefetch", Prefetch)
    monkeypatch.setattr(pools, "PersistentPoolManager", Manager)
    config = SimpleNamespace(model_dump=lambda **kw: {})
    pipeline = full.FullPipeline(
        root=tmp_path,
        index={},
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0"],
        sam_configuration=config,
        auk_configuration=config,
        backend=None,
        profiles=None,
    )
    with pytest.raises(RuntimeError, match="startup failed"), pipeline.node([3, 1]):
        raise AssertionError("startup should fail")
    assert events == [("canonical", 3), "pools", "pools_close", "canonical_close"]
    assert pipeline.prefetch is None
    assert pipeline.pools is None


def test_one_canonical_lookahead_and_serial_gpu_barriers(tmp_path):
    events = []
    next_started = threading.Event()
    release_next = threading.Event()
    config = SimpleNamespace(model_dump=lambda **kw: {})

    class Prefetch:
        def __init__(self):
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.future = None
            self.pending = None

        def start(self, shard):
            assert self.future is None
            self.pending = shard

            def prepare():
                events.append((shard, "prepare"))
                if shard == 1:
                    next_started.set()
                    assert release_next.wait(5)
                return {
                    "audio_root": str(tmp_path / str(shard)),
                    "summary": {"ready": 1},
                }

            self.future = self.executor.submit(prepare)

        def wait(self, shard):
            assert self.pending == shard
            result = self.future.result(timeout=5)
            self.future = None
            self.pending = None
            return result

    class Pipeline(full.FullPipeline):
        def stage(self, name, shard):
            if name == "canonical":
                return self.prepared[shard].summary
            assert self.prepared[shard].audio_root == tmp_path / str(shard)
            events.append((shard, name))
            if shard == 0 and name == "sam":
                assert next_started.wait(5)
                assert (2, "prepare") not in events
                release_next.set()
            return {}

    pipeline = Pipeline(
        root=tmp_path,
        index={},
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0"],
        sam_configuration=config,
        auk_configuration=config,
        backend=None,
        profiles=None,
    )
    prefetch = Prefetch()
    pipeline.prefetch = prefetch
    prefetch.start(0)
    try:
        full.run_assigned_shards(tmp_path, [0, 1, 2], pipeline)
    finally:
        release_next.set()
        prefetch.executor.shutdown()
    assert events.index((1, "prepare")) < events.index((0, "mimo"))
    assert events.index((0, "mimo")) < events.index((1, "sam"))
    assert events.index((1, "mimo")) < events.index((2, "sam"))
    assert events.index((2, "prepare")) > events.index((0, "mimo"))
