"""Check everything before committing 2.5 hours to the runs.

Each check is here because getting it wrong costs real time rather than failing fast:
a missing data file surfaces ninety minutes in, an evaluator that loads but scores
backwards produces numbers that look plausible and are wrong, and a pipeline that cannot
reach the endpoint fails on claim one of two thousand.

    python handoff/c1-part2/preflight.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

OK, BAD = "  ok   ", "  FAIL "
failures: list[str] = []


def check(label: str, fn) -> object:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a preflight reports, it does not crash
        print(f"{BAD}{label}\n         {type(exc).__name__}: {exc}")
        failures.append(label)
        return None
    print(f"{OK}{label}" + (f"  —  {detail}" if detail else ""))
    return detail


# --- inputs ---------------------------------------------------------------------------

print("\n--- input files ---")

EXPECTED = {
    "data/averitec/dev.json": 1_000_000,
    "data/averitec/pool_dev.jsonl": 5_000_000,
    "data/averitec/misinfo_list.txt": 10_000,
    "data/trajectories/trajectories.jsonl": 1_000_000,
}


def _file(path: str, min_bytes: int):
    def inner():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{path} is missing — see the README's table")
        size = p.stat().st_size
        if size < min_bytes:
            raise ValueError(
                f"{path} is only {size / 1e6:.1f} MB, expected at least "
                f"{min_bytes / 1e6:.1f} MB — the transfer was probably truncated"
            )
        return f"{size / 1e6:.1f} MB"

    return inner


for path, min_bytes in EXPECTED.items():
    check(path, _file(path, min_bytes))


def _trajectory_count():
    from experience.format import read_jsonl

    traj = read_jsonl(Path("data/trajectories/trajectories.jsonl"))
    if len(traj) < 200:
        raise ValueError(f"only {len(traj)} trajectories; too few to fit trust on")
    labelled = sum(1 for t in traj if t.gold_label)
    domains = {d for t in traj for d in t.domains}
    return f"{len(traj)} trajectories, {labelled} labelled, {len(domains)} publishers"


check("trajectories parse and carry publishers", _trajectory_count)


# --- the evaluator --------------------------------------------------------------------

print("\n--- relevance evaluator ---")


def _find_evaluator():
    hits = sorted(Path("results").glob("*_t5-evaluator/model/config.json"))
    if not hits:
        raise FileNotFoundError("no trained evaluator — run setup.sh first")
    return hits[-1].parent


EVALUATOR = check("evaluator directory", _find_evaluator)


def _evaluator_separates():
    from evaluator.score import T5Scorer

    scorer = T5Scorer(EVALUATOR)
    relevant, irrelevant = scorer.score(
        "The Kochi metro opened to the public in 2017.",
        [
            "The Kochi Metro began commercial operations in June 2017 after several delays.",
            "Penguins are flightless birds found mainly in the southern hemisphere.",
        ],
    )
    if relevant <= irrelevant:
        raise ValueError(
            f"scored the relevant passage {relevant:+.3f} and the irrelevant one "
            f"{irrelevant:+.3f}. A backwards evaluator produces plausible-looking "
            "numbers that are wrong, so this is fatal rather than a warning."
        )
    return f"relevant {relevant:+.3f} > irrelevant {irrelevant:+.3f}"


if EVALUATOR:
    check("evaluator separates relevant from irrelevant", _evaluator_separates)


# --- the endpoint ---------------------------------------------------------------------

print("\n--- teacher endpoint ---")


def _endpoint():
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
        import json

        names = [m["name"] for m in json.load(r).get("models", [])]
    if not any(n.startswith("qwen2.5:7b") for n in names):
        raise ValueError(f"qwen2.5:7b is not pulled. Available: {names or 'none'}")
    return ", ".join(names)


check("ollama serving qwen2.5:7b", _endpoint)


def _concurrency():
    conf = Path("handoff/c1-part2/.concurrency")
    if not conf.exists():
        return "not set (run.sh will default to 1 worker — slow but correct)"
    return conf.read_text(encoding="utf-8").strip().replace("\n", ", ")


check("concurrency setting", _concurrency)


# --- two claims end to end ------------------------------------------------------------

print("\n--- two claims through the real pipeline ---")


def _smoke():
    os.environ.setdefault("VERIMEM_API_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("VERIMEM_API_MODEL", "qwen2.5:7b")
    os.environ.setdefault("VERIMEM_API_KEY", "ollama")

    from agent.crag import crag
    from core.config import RunConfig
    from core.llm import LLMClient
    from eval.datasets import load_averitec, sample
    from evaluator.score import load_scorer
    from retrieval.store import open_store

    claims = sample(load_averitec(Path("data/averitec/dev.json"), "dev"), 2, seed=13)
    llm = LLMClient()
    scorer = load_scorer(EVALUATOR)
    source = open_store(Path("data/averitec/pool_dev.jsonl"))
    cfg = RunConfig(split="dev", seed=13)

    out = []
    for claim in claims:
        t = crag(claim, llm, source, scorer, cfg)
        out.append(f"{t.verdict} (gold {t.gold_label})")
    return " | ".join(out)


if EVALUATOR and not failures:
    check("end-to-end verification", _smoke)
elif failures:
    print("  skipped — fix the failures above first")


# --- verdict --------------------------------------------------------------------------

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f in failures:
        print(f"  - {f}")
    print("\nFix these before starting the runs. The README has a troubleshooting table.")
    raise SystemExit(1)

print("All checks passed. Start the runs with:\n")
print("    bash handoff/c1-part2/run.sh\n")
print("It takes about 2.5 hours and prints a projected finish every 25 claims.")
