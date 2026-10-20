# 2026-10-20_G0-baselines

Split **dev**, 20 claims, seed 13, k=5, evidence from
`store`. Teacher `qwen2.5:7b` at `http://localhost:11434/v1`.

| system | n | accuracy | macro-F1 | unparseable |
|---|---|---|---|---|
| no_retrieval | 20 | 0.450 | 0.216 | 0 |
| plain_rag | 20 | 0.500 | 0.244 | 1 |

Usage: {'calls': 40, 'cached_calls': 0, 'prompt_tokens': 22576, 'completion_tokens': 2121, 'cost_usd': 0.0}

To fill in: do the failures cluster on one verdict? Is the retriever or the verifier at
fault -- re-run plain_rag with `--evidence gold` to separate them.
