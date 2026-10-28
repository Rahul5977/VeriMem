# VeriMem C1 part 2 — run this on the RTX 4070 laptop

You are running the last two experiments of VeriMem's headline contribution. Everything
here is **inference only** — no fine-tuning, no QLoRA, nothing that needs more than 8 GB.
Total machine time is about **2.5 hours**, most of it unattended.

You do not need to understand the research to run this. If you want the one-paragraph
version, it is at the bottom.

---

## What you need from Teesha

Four things. Three are small; one she has to download from Kaggle for you.

| what | where it goes | size |
|---|---|---|
| `dev.json` | `data/averitec/` | 1.7 MB |
| `pool_dev.jsonl` | `data/averitec/` | 8.2 MB |
| `misinfo_list.txt` | `data/averitec/` | 28 KB |
| `trajectories.jsonl` | `data/trajectories/` | 3.4 MB |

All four are on her laptop, so this is one small zip (about 13 MB total).

You do **not** need the trained evaluator. `setup.sh` trains it here in about 25 minutes
from data that is in the repo, which is faster than transferring a 1 GB model.

---

## Steps

### 0. Clone and check out

```bash
git clone https://github.com/TERABYTE05/VeriMem.git
cd VeriMem
```

### 1. Drop in the data

```bash
mkdir -p data/averitec data/trajectories
# copy the four files from Teesha into those two directories
```

### 2. One-time setup (~30 min, mostly the evaluator training)

```bash
bash handoff/c1-part2/setup.sh
```

This installs Python dependencies, installs Ollama, pulls `qwen2.5:7b`, and trains the
relevance evaluator. It prints what it did at each step and stops on the first failure.

### 3. Preflight (~2 min) — **do not skip this**

```bash
python handoff/c1-part2/preflight.py
```

It checks every input, loads the evaluator and confirms it separates relevant from
irrelevant text, then runs **two claims end to end** through the real pipeline. Two
minutes here beats discovering a missing file ninety minutes into the run.

### 4. The runs (~2.5 h, unattended)

```bash
bash handoff/c1-part2/run.sh
```

You can leave it. It prints a projected finish time every 25 claims, like
`[50/100] 12m30s elapsed, ~38m left`. If the projection says something wildly more than
three hours, stop it and tell Teesha — something is wrong and waiting will not fix it.

### 5. Send back

```bash
bash handoff/c1-part2/run.sh --package
```

That writes `verimem-c1-part2-results.zip` (a few MB). Send her that. It contains the
`results/` folders and the console logs, nothing else.

---

## Laptop notes

**Plug it in, set the power mode to maximum performance, and raise it for airflow.** This
is 2.5 hours of sustained GPU load.

**VRAM is the thing to watch.** `qwen2.5:7b` in 4-bit is about 4.7 GB and the evaluator
another 1 GB, which fits 8 GB — but only with limited request concurrency. `setup.sh`
picks the concurrency from your detected VRAM and `run.sh` uses it. If you override it
upward, Ollama will quietly move layers to the CPU and the run gets several times slower
without failing. If `nvidia-smi` shows VRAM pinned at the limit and the projected finish
keeps climbing, that is what happened.

**Windows is fine.** This run needs no `bitsandbytes`, so WSL2 is not required (the
project's Linux rule is for the fine-tuning jobs, which are not these). Native Windows
with Python 3.11 works; use `bash` from Git Bash, or run the commands inside each script
by hand.

**It is resumable.** Every model call is cached on disk under `cache/`. If the run dies
or you stop it, run the same command again — completed work replays instantly and it
picks up where it stopped. Do not delete `cache/`.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ollama: command not found` after setup | PATH not reloaded | open a new shell, or `export PATH=$PATH:/usr/local/bin` |
| `connection refused` on port 11434 | the server is not running | `ollama serve &` then retry |
| `no --evaluator` warning | the evaluator did not train | re-run `setup.sh`; check `results/*_t5-evaluator/model/` exists |
| `evaluator is not separating` | wrong or corrupt model directory | delete `results/*_t5-evaluator/` and re-run `setup.sh` |
| run much slower than projected | VRAM spilled to CPU | lower `OLLAMA_NUM_PARALLEL`, restart `ollama serve` |
| `no pairs cleared --min-similarity` | `trajectories.jsonl` missing or truncated | confirm the file has ~900 lines (`wc -l`) |

Anything else: send Teesha the console output. Every run prints its own metrics to the
log, so even a failed run is diagnosable from what it printed.

---

## What this is actually measuring

VeriMem is a fact-checking agent that keeps a memory of its past verifications and
retrieves them as context for new claims. The project's headline question is what an
attacker can do to that memory. Two experiments remain:

**Source poisoning** adds fabricated web pages to the evidence corpus. The interesting
part is that these pages are *genuinely about* the claim they target, so the relevance
evaluator scores them highly — and is right to. Relevance is not truthfulness. The run
reports the evaluator's average score on fabricated text beside real text to show that
the gap is small, which is the evidence that relevance grading cannot defend this attack.
The only defence left is tracking which publishers have been reliable.

**Propagation** asks whether the damage outlasts the fakes. A fabricated record makes the
agent misjudge claim A; that verdict gets stored as experience; it is then retrieved when
a related claim B comes along. By then the harmful record is not fabricated at all — it
is the system's own output citing real sources. So deleting every fake does not undo it.
Each pair runs three times with stores that differ *only* in A's verdict, which is what
makes any change in B attributable to the poison rather than to retrieval in general.

---

## For Teesha: the four files to send

All four are on your laptop:

```
data/averitec/dev.json
data/averitec/pool_dev.jsonl
data/averitec/misinfo_list.txt
data/trajectories/trajectories.jsonl
```

```bash
zip -r verimem-c1-inputs.zip \
  data/averitec/dev.json data/averitec/pool_dev.jsonl \
  data/averitec/misinfo_list.txt data/trajectories/trajectories.jsonl
```

About 13 MB. Nothing else is needed — the
evaluator trains on this end from data already in the repo, and `cache/` must not be
copied (it is gigabytes and the runs rebuild what they need).

Do **not** send anything from `attacks/`, and nothing fabricated comes back: the
`--package` step deliberately excludes the corrupted trajectories, which are
reproducible from the fixed seed anyway.
