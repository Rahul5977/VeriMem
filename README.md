# VeriMem

A fact-checking agent that grades its evidence and learns from its own past verifications.

**In:** a claim. **Out:** a verdict (Supported / Refuted / Conflicting evidence / Not enough
evidence) with an evidence trail — sources, relevance grades, source trust scores, and the past
verifications it drew on.

It combines two papers, both reproduced from scratch and transferred to fact-checking: **CRAG**
(a fine-tuned T5 evaluator grades retrieved evidence and routes to refine / rewrite+search / both)
and **ExpRAG** (a store of past trajectories feeds retrieval-augmented LoRA fine-tuning). On top of
those we add evaluator-gated experience retrieval, Bayesian per-domain source trust, and a
memory-poisoning study.

NLP course project, IIT Bhilai. The schedule and task breakdown are kept outside this
repo; ask Teesha for the current plan.

## Setup

```bash
make setup          # python3.11 venv + CPU requirements + .env from the template
$EDITOR .env        # API key, search key, spend cap
make test
```

Python 3.11. The Mac is a dev machine only. GPU work runs on the two laptops under Linux or WSL2
(`pip install -r requirements-gpu.txt`), with Kaggle/Colab as backup:

| Machine | Role |
|---|---|
| Mac M3, 8 GB | code, agent graph, caching, search calls, FAISS, tests on <= 10 claims, report, demo from cache |
| GPU-A: RTX 4070, 8 GB VRAM | LoRA/QLoRA on Qwen2.5-3B, bulk evaluation of the fine-tuned verifier |
| GPU-B: RTX 4050, 6 GB VRAM | Flan-T5 evaluator training, embeddings, baselines, prompted 3B in 4-bit |

The teacher/judge is one hosted Qwen model over an OpenAI-compatible endpoint, fixed in P0.13.

## Layout

| Path | What lives there |
|---|---|
| `core/` | paths, device selection, run config, the hashed cache |
| `retrieval/` | web search, page fetch, strip refinement, source tagging, domain blocklist |
| `evaluator/` | T5 relevance evaluator: data building, training, thresholds |
| `experience/` | trajectory format, FAISS store, query/selection options |
| `training/` | LoRA SFT and retrieval-augmented SFT (GPU only) |
| `agent/` | LangGraph state machine and config flags |
| `attacks/` | source + memory poisoning sets (fixed seed), propagation pairs |
| `eval/` | metrics and experiment runners |
| `demo/` | Gradio UI |
| `cache/`, `data/`, `results/` | gitignored working dirs; `results/` keeps config + metrics |

## Caching

Every LLM call, search and page fetch goes through `core.cache`, keyed by a SHA-256 of its inputs.
A repeat call with identical inputs must never hit the network.

```python
from core.cache import get_cache

cache = get_cache("search")
hits = cache.get_or_compute({"q": claim, "k": 5}, lambda: provider.search(claim, k=5))
```

`make cache-stats` shows how many records each namespace holds; `core.cache.STATS` reports hits,
misses and hit rate for the current process.
