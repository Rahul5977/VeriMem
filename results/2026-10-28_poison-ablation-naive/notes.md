# 2026-10-28_poison-ablation-naive — C1, naive adversary, each defence alone

Kernel `teesha5124/verimem-c1-adaptive`, run 1. AVeriTeC dev, 100 targets, 100 claims per
cell. Five configurations x four poison rates.

**Provenance.** `config.yaml` carries no `verifier_prompt_sha` because the kernel cloned
the repo before that stamp was added. It *does* use the unified verifier prompt
(`core/prompts.py`, commit c28b90c), which was committed before this kernel launched.
These numbers supersede `2026-10-28_poison-full`, which used the pre-unification prompt.

## Attack success rate

| configuration | 0% | 5% | 10% | 20% | poison kept @20% |
|---|---|---|---|---|---|
| no memory | 0.160 | — | — | — | — |
| memory (ExpRAG as published) | 0.220 | 0.340 | 0.420 | **0.430** | 1.000 |
| gate only (C2) | 0.220 | 0.250 | 0.270 | **0.320** | 0.361 |
| trust only (C3) | 0.220 | 0.230 | 0.220 | **0.230** | 0.000 |
| gated memory (C2 + C3) | 0.220 | 0.230 | 0.220 | **0.230** | 0.000 |

## Reading it

- **H1 holds.** Attack success climbs monotonically with the poison rate once experience
  retrieval is on: 0.220 → 0.430. No-memory has no store to write to and sits at 0.160.
- **Against this adversary C3 does all the work.** `trust_only` is flat at 0.230 and keeps
  **zero** poisoned records; `gated_memory` matches it exactly, so C2 adds nothing here.
- **C2 alone is only partial** (0.320 at 20%), and gets *weaker* as the rate rises — its
  poison pass rate goes 0.255 → 0.273 → 0.361. More fabricated records means more chances
  that one of them scores above the gate.
- **Why C3 wins so easily, and why that is not as strong as it looks:** these fakes cite
  domains the system has never seen, and Beta(1, 3) puts a stranger at 0.25, below the 0.3
  floor. C3 is rejecting them for being *new*, not for lying. See
  `2026-10-28_poison-adaptive` for what happens when the adversary cites trusted sources.

## Correction to an earlier claim

The superseded `poison-full` run showed macro-F1 collapsing from 0.318 to 0.173 under
attack, which was read as "the attack also causes broad collateral damage". **That does
not replicate.** Here macro-F1 under the undefended store is 0.344 / 0.312 / 0.322 / 0.365
across the four rates — roughly flat. The attack is **surgical**: it flips its targets
without degrading the rest of the evaluation set. That is arguably the more concerning
property and the report should say so rather than repeating the collapse figure.
