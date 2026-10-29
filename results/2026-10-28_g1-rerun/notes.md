# 2026-10-28_g1-rerun

Split **dev**, None claims, seed 13, k=5, evidence from
`store`. Teacher `qwen2.5:7b` at `http://localhost:11434/v1`.

| system | n | accuracy | macro-F1 | unparseable |
|---|---|---|---|---|
| no_retrieval | 500 | 0.576 | 0.324 | 0 |
| plain_rag | 500 | 0.600 | 0.354 | 0 |
| crag | 500 | 0.546 | 0.363 | 0 |

Usage: {'calls': 1705, 'cached_calls': 140, 'prompt_tokens': 797160, 'completion_tokens': 72281, 'cost_usd': 0.0}

To fill in: do the failures cluster on one verdict? Is the retriever or the verifier at
fault -- re-run plain_rag with `--evidence gold` to separate them.
