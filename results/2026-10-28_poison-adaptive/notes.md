# 2026-10-28_poison-adaptive — C1, an adversary that defeats source trust

Kernel `teesha5124/verimem-c1-adaptive`, run 2. Same 100 targets and same claims as
`2026-10-28_poison-ablation-naive`, so the two tables are directly comparable. Rates 0%
and 20% only: the dose curve is established by run 1, and this asks whether the defences
survive a stronger attacker at full strength.

**Provenance.** No `verifier_prompt_sha` (the kernel predates the stamp), but it uses the
unified verifier prompt. Same prompt as the naive run above.

## The adversary

Instead of known misinformation sites, the fakes cite the five domains the clean history
trusts most:

```
en.wikipedia.org, twitter.com, cdc.gov, who.int, ncbi.nlm.nih.gov
```

An attacker who reads the system's own history and impersonates the CDC, the WHO and
Wikipedia. Every one of these clears any usable trust floor, so C3 is defeated **by
construction** — not by a weakness in the implementation, but because trust in a publisher
cannot distinguish a real page from a fabricated one attributed to that publisher.

## Attack success rate

| configuration | 0% | 20% | poison kept |
|---|---|---|---|
| memory (undefended) | 0.220 | **0.460** | 1.000 |
| gate only (C2) | 0.220 | **0.310** | 0.373 |
| trust only (C3) | 0.220 | **0.460** | **1.000** |
| gated memory (C2 + C3) | 0.220 | **0.310** | 0.373 |

## The finding: the two defences are complementary, not redundant

Put beside the naive table, which defence carries H2 **inverts**:

| | naive adversary | adaptive adversary |
|---|---|---|
| memory (undefended) | 0.430 | 0.460 |
| gate only (C2) | 0.320 | **0.310** ← carries it |
| trust only (C3) | **0.230** ← carries it | 0.460 ← fully defeated |
| gated (C2 + C3) | 0.230 | 0.310 |

Each defence alone fails against one of the two adversaries and holds against the other,
and the combination tracks whichever is working. This is a stronger result than "the
defences help": it is a direct argument that C2 and C3 are both necessary, which is what
having them as separate contributions requires.

C3 fails here because trust is about *who published*, and the adversary chose who to
claim to be. C2 survives because it asks a question the adversary cannot dodge by
relabelling: does this trajectory's evidence actually support the verdict it reached?

## Limitation, stated plainly

**Gated memory under the adaptive attack is 0.310 — still nearly double no-memory's
0.160.** The defences reduce the cost of having a memory; they do not remove it. Against
an adversary who studies the system's history, experience retrieval remains more
attackable than having no experience at all. H1's concern is mitigated, not answered, and
the report should not claim otherwise.
