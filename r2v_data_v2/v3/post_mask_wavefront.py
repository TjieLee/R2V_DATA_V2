"""Bounded execution primitives only; no Visual policy or durable queue."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def bounded_results(function, items, maximum, *, on_interrupt):
    """Yield completed results, with at most maximum submitted tasks at a time.

    Callers isolate ordinary item failures. External interruption stops admission
    before waiting for active tasks, whose committed outputs remain durable.
    """
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=maximum) as pool:
        pending = set()
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < maximum:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                    else:
                        pending.add(pool.submit(function, item))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
        except BaseException as exc:
            # Ordinary shard/clip infrastructure errors stay local. Only an
            # external interruption may stop the shared worker model resources.
            if not isinstance(exc, Exception):
                on_interrupt()
            for future in pending:
                future.cancel()
            raise
