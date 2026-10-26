# 2026-10-26_t5-evaluator

Flan-T5 relevance evaluator. Base `google/flan-t5-base`, 3.0 epochs,
batch 8 x accum 2, lr 0.0003, seed 13.

| metric | value |
|---|---|
| dev accuracy | 0.7284 |
| dev AUC | 0.8107 |
| mean score, relevant | 0.2833 |
| mean score, irrelevant | -0.5363 |
| dev pairs | 751 |

AUC matters more than accuracy here: CRAG picks its action from two thresholds on the
score, so what counts is whether relevant passages are ranked above irrelevant ones,
not whether they land on the right side of zero.

Next: P1.4 compares this against a cross-encoder baseline, and P1.7 tunes the two
thresholds on dev and logs the resulting action shares.
