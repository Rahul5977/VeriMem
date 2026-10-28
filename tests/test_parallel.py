"""Concurrency must buy wall-clock without changing a single number.

Every model call is cached by a hash of its input and nothing in the pipeline depends on
the order claims are processed in, so running claims in parallel is safe in principle.
These tests hold the two things that make it safe in practice: results come back in input
order, and shared counters do not lose increments.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.parallel import DEFAULT_WORKERS, Progress, map_ordered, resolve_workers
from experience.format import Evidence, Step, Trajectory
from experience.store import ExperienceStore

# --- worker resolution ----------------------------------------------------------------


def test_sequential_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default must not change existing behaviour or reproduce old runs differently."""
    monkeypatch.delenv("VERIMEM_WORKERS", raising=False)
    assert resolve_workers(None) == 1
    assert resolve_workers(0) == 1


def test_an_explicit_count_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIMEM_WORKERS", "4")
    assert resolve_workers(16) == 16


def test_the_environment_sets_it_for_a_whole_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERIMEM_WORKERS", "6")
    assert resolve_workers(None) == 6


def test_nonsense_in_the_environment_falls_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must not silently serialise *or* silently fan out."""
    monkeypatch.setenv("VERIMEM_WORKERS", "lots")
    assert resolve_workers(None) == 1


# --- ordering -------------------------------------------------------------------------


def test_results_come_back_in_input_order() -> None:
    """Verdicts are zipped against the claim list, so a reordering would mislabel every
    one of them."""
    items = list(range(50))

    def slow_square(n: int) -> int:
        # Deliberately inverted sleep: later items finish first, so a function that
        # returned completion order would fail this.
        time.sleep((50 - n) / 2000)
        return n * n

    assert map_ordered(slow_square, items, workers=8) == [n * n for n in items]


def test_one_worker_and_eight_workers_agree() -> None:
    items = [f"claim-{i}" for i in range(30)]
    assert map_ordered(str.upper, items, workers=1) == map_ordered(str.upper, items, workers=8)


def test_every_item_is_processed_exactly_once() -> None:
    seen: list[int] = []
    lock = threading.Lock()

    def record(n: int) -> int:
        with lock:
            seen.append(n)
        return n

    map_ordered(record, list(range(40)), workers=8)
    assert sorted(seen) == list(range(40))


def test_a_failure_is_raised_rather_than_dropped() -> None:
    """A run that quietly loses a claim reports its metric over the wrong denominator."""

    def boom(n: int) -> int:
        if n == 17:
            raise ValueError("claim 17 failed")
        return n

    with pytest.raises(ValueError, match="claim 17 failed"):
        map_ordered(boom, list(range(40)), workers=8)


def test_progress_fires_once_per_item() -> None:
    calls: list[int] = []
    lock = threading.Lock()

    def on_done(n: int, _r: int) -> None:
        with lock:
            calls.append(n)

    map_ordered(lambda n: n, list(range(20)), workers=4, on_done=on_done)
    assert sorted(calls) == list(range(1, 21))


def test_an_empty_input_is_not_an_error() -> None:
    assert map_ordered(lambda n: n, [], workers=8) == []


# --- shared state ---------------------------------------------------------------------


def _traj(i: int, poisoned: bool = False) -> Trajectory:
    t = Trajectory(
        claim=f"A claim about topic {i} and several shared words.",
        verdict="Refuted",
        rationale="r",
        evidence=[Evidence(url=f"https://example.com/{i}", text="a passage")],
        steps=[Step("retrieve", "top-5")],
        claim_id=f"c{i}",
        gold_label="Refuted",
        poisoned=poisoned,
    )
    return t


def test_the_stores_counters_survive_concurrent_retrieval() -> None:
    """`poison_pass_rate` is a reported number in the C1 table, so a lost increment
    under --workers would be a wrong result rather than untidy bookkeeping."""
    store = ExperienceStore([_traj(i, poisoned=i % 2 == 0) for i in range(40)])
    queries = [f"A claim about topic {i} and several shared words." for i in range(40)]

    map_ordered(lambda q: store.retrieve(q, k=3), queries, workers=8)

    s = store.stats
    assert s.considered == s.kept + s.dropped_by_gate + s.dropped_by_trust
    assert s.poisoned_kept <= s.poisoned_considered
    assert s.considered > 0


def test_concurrent_retrieval_returns_what_sequential_does() -> None:
    """The same trajectory objects in both stores: `Trajectory.id` is per-instance, so
    two separately built stores would differ for a reason that has nothing to do with
    concurrency."""
    trajectories = [_traj(i) for i in range(20)]
    queries = [f"A claim about topic {i} and several shared words." for i in range(20)]
    a = ExperienceStore(trajectories)
    b = ExperienceStore(trajectories)

    def ids(store, q):
        return [r.trajectory.claim_id for r in store.retrieve(q, k=3)]

    seq = map_ordered(lambda q: ids(a, q), queries, workers=1)
    par = map_ordered(lambda q: ids(b, q), queries, workers=8)
    assert seq == par
    assert a.stats.as_dict() == b.stats.as_dict()


def test_the_default_worker_count_is_sane() -> None:
    assert 2 <= DEFAULT_WORKERS <= 16


# --- the ETA projection ---------------------------------------------------------------


def test_progress_projects_a_finish_time(capsys: pytest.CaptureFixture[str]) -> None:
    """The point is to surface a doomed run in minutes rather than at hour eleven."""
    p = Progress(100, every=10)
    for n in range(1, 21):
        p.tick(n)
    out = capsys.readouterr().out
    assert "[10/100]" in out
    assert "left" in out, "a mid-run tick must project, not just report elapsed"


def test_progress_stops_projecting_at_the_end(capsys: pytest.CaptureFixture[str]) -> None:
    """ "~0m left" on the final line is noise; the run is over."""
    p = Progress(5, every=5)
    for n in range(1, 6):
        p.tick(n)
    final = capsys.readouterr().out.strip().splitlines()[-1]
    assert "[5/5]" in final
    assert "left" not in final


def test_progress_is_quiet_between_intervals(capsys: pytest.CaptureFixture[str]) -> None:
    """A line per claim buries the result in a 2,000-claim run."""
    p = Progress(100, every=25)
    for n in range(1, 25):
        p.tick(n)
    assert capsys.readouterr().out == ""


def test_progress_survives_concurrent_ticks() -> None:
    """It is called from the worker threads."""
    p = Progress(200, every=50)
    map_ordered(lambda n: n, list(range(200)), workers=8, on_done=p.tick)
    assert len(p._marks) > 0
