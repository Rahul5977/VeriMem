# 2026-10-26_smoke

Flan-T5 relevance evaluator. Base `google/flan-t5-base`, 1.0 epochs,
batch 8 x accum 2, lr 0.0003, seed 13.

| metric | value |
|---|---|
| dev accuracy | 0.53 |
| dev AUC | 0.536 |
| mean score, relevant | 0.5272 |
| mean score, irrelevant | 0.5045 |
| dev pairs | 751 |

AUC matters more than accuracy here: CRAG picks its action from two thresholds on the
score, so what counts is whether relevant passages are ranked above irrelevant ones,
not whether they land on the right side of zero.

Next: P1.4 compares this against a cross-encoder baseline, and P1.7 tunes the two
thresholds on dev and logs the resulting action shares.
