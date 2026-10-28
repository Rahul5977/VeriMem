"""Run claims concurrently, because almost all of the wall-clock is waiting on a socket.

A verification spends a fraction of a second on local work -- BM25, the T5 evaluator,
string handling -- and then several seconds blocked on an HTTP call to the teacher. Doing
those one at a time leaves the endpoint idle most of the time. A 100-claim cell at ~10s
each is 17 minutes sequentially and about 4 with eight workers, and the C1 sweep is
twenty such cells.

**Why threads and not processes.** The blocking call releases the GIL, which is the only
parallelism this needs; processes would mean a separate copy of the T5 evaluator per
worker, and on a 16 GB T4 that is the one thing there is no room for.

**What this does not make safe.** Threads share the caches, the token ledger, the
evaluator and the experience store's counters. Those are guarded at their own definitions
rather than here -- see `DiskCache._lock`, `LLMClient._usage_lock`, `T5Scorer._lock` and
`ExperienceStore._stats_lock`. This module only guarantees that results come back in
input order and that one claim's failure does not vanish silently.

**Determinism is preserved** because every model call is cached by a hash of its input
and nothing in the pipeline depends on the order claims are processed in. Two runs with
different worker counts produce the same verdicts; only the wall-clock differs. The one
thing that does change is which worker *wrote* a given cache entry, and that is invisible.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Eight saturates a local Ollama server with `OLLAMA_NUM_PARALLEL` at its default without
# queueing so deeply that a failure takes a long time to surface. Raising it past the
# server's own parallelism buys nothing: the requests simply queue inside the server.
DEFAULT_WORKERS = 8


def resolve_workers(requested: int | None) -> int:
    """How many workers to use. `VERIMEM_WORKERS` is the environment override."""
    if requested and requested > 0:
        return requested
    raw = os.environ.get("VERIMEM_WORKERS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 1  # sequential unless asked: the default must not change existing behaviour


def map_ordered(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int = 1,
    on_done: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Apply `fn` to every item, returning results in input order.

    `on_done(n_completed, result)` fires as each finishes, for progress reporting. It is
    called from the worker threads, so it must not touch unguarded shared state; printing
    is fine.

    One worker means no pool at all rather than a pool of one: a stack trace from a
    sequential run points at the actual failure, which matters more than consistency
    while debugging.
    """
    if workers <= 1 or len(items) <= 1:
        out: list[R] = []
        for i, item in enumerate(items, start=1):
            result = fn(item)
            out.append(result)
            if on_done:
                on_done(i, result)
        return out

    results: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for n, future in enumerate(as_completed(futures), start=1):
            # Let the exception propagate: a run that quietly drops a claim reports a
            # metric over a different denominator than it claims.
            result = future.result()
            results[futures[future]] = result
            if on_done:
                on_done(n, result)
    return [results[i] for i in range(len(items))]


class Progress:
    """Progress with a projected finish, so a doomed run says so in the first minutes.

    A Kaggle session is capped at 12 hours and nothing survives a kill, so the expensive
    mistake is not a slow run -- it is discovering at hour eleven that a run needed
    fourteen. The cost per claim is near-constant and measurable after a handful, so a
    projection from the observed rate is reliable long before the commitment is sunk.

    The projection deliberately uses the rate over *recent* items rather than the whole
    run: the first few claims in a cell are slower (a cold evaluator, a model still
    loading), and averaging those in over-estimates a long run by enough to look alarming.
    """

    def __init__(self, total: int, label: str = "", every: int = 25, window: int = 50) -> None:
        self.total = total
        self.label = label
        self.every = max(1, every)
        self.window = max(2, window)
        self._start = time.monotonic()
        self._marks: deque[tuple[int, float]] = deque(maxlen=self.window)
        self._lock = threading.Lock()

    def tick(self, n: int, _result: object = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._marks.append((n, now))
            due = n % self.every == 0 or n == self.total
            if not due:
                return
            elapsed = now - self._start
            remaining = self._project(n)
        head = (
            f"    [{n}/{self.total}]" if not self.label else f"    {self.label} [{n}/{self.total}]"
        )
        if remaining is None:
            print(f"{head} {_hms(elapsed)} elapsed", flush=True)
        else:
            print(
                f"{head} {_hms(elapsed)} elapsed, ~{_hms(remaining)} left "
                f"(finish ~{_hms(elapsed + remaining)} total)",
                flush=True,
            )

    def _project(self, n: int) -> float | None:
        if n >= self.total or len(self._marks) < 2:
            return None
        (n0, t0), (n1, t1) = self._marks[0], self._marks[-1]
        done = n1 - n0
        if done <= 0 or t1 <= t0:
            return None
        per_item = (t1 - t0) / done
        return per_item * (self.total - n)


def _hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
