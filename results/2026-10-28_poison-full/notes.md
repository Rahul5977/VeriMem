# 2026-10-28_poison-full — C1 memory poisoning, 100 targets

Run as Kaggle kernel `teesha5124/verimem-c1-full`. AVeriTeC dev, naive adversary.

## Provenance caveat

**No `config.yaml`, and the numbers were produced by the pre-unification verifier
prompt.** Two separate definitions of the verifier system prompt had drifted; they were
unified later the same day (`core/prompts.py`), which changed the inference prompt and so
invalidated the LLM cache. These numbers are therefore not reproducible from the current
code — `verimem-c1-adaptive` regenerates the same table under the unified prompt, and
that run supersedes this one for the report.

`metrics.json` here was **recovered from the kernel log**, not downloaded: Kaggle's
download endpoint refuses paths under `results/` and its file listing is paginated past
the point where a results file appears. Runs now print their metrics to stdout
(`core.paths.announce_metrics`) so `scripts/kaggle_run.py harvest` can rebuild them.
Per-cell retrieval detail beyond `poison_pass_rate` was not recoverable.

## Result

| configuration | 0% | 5% | 10% | 20% |
|---|---|---|---|---|
| no_memory | 0.200 | — | — | — |
| memory | 0.270 | 0.340 | 0.500 | **0.560** |
| gated_memory | 0.260 | 0.270 | 0.260 | **0.270** |

- **H1 holds.** Attack success rises monotonically with the poison rate once experience
  retrieval is on. no_memory has no store to write to and stays flat.
- **H2 holds.** The gated store is flat at every rate.
- **The defence is not just deleting experience.** macro-F1 falls 0.318 → 0.173
  undefended, but holds 0.328 → 0.334 gated.
- Plain memory costs 0.07 ASR and 0.077 macro-F1 at **0%** poisoning: trajectory accuracy
  is 0.590, so clean experience is often wrong experience.

## Known overstatement

`poison kept 0.00` is C3 rejecting domains it has **never seen** — Beta(1, 3) puts a
stranger at 0.25, below the 0.3 floor — not C3 detecting fabrication. The adaptive
adversary (`--adversary adaptive`) cites the domains the clean history trusts most and
is the honest test of H2.

---

**SUPERSEDED (2026-10-28).** `2026-10-28_poison-ablation-naive` re-runs this table under
the unified verifier prompt, with the two defence ablations added. Use that one. The
macro-F1 collapse reported above does not replicate.
