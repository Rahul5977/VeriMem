#!/usr/bin/env bash
# One-time setup for the C1 part 2 runs. Safe to re-run: every step is skipped if
# already done. Stops on the first real failure rather than carrying on and failing
# two hours later with a confusing error.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
echo "[setup] repo: $REPO"

step() { printf '\n=== %s ===\n' "$1"; }

# --- Python ---------------------------------------------------------------------------
step "Python"
PY="${PYTHON:-python3}"
VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "using $PY ($VER)"
case "$VER" in
  3.11|3.12) ;;
  *) echo "WARNING: the project is pinned to 3.11 (torch/transformers wheels). $VER may work; if imports fail, that is why." ;;
esac

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "created .venv"
fi
VPY="$REPO/.venv/bin/python"
[ -x "$VPY" ] || VPY="$REPO/.venv/Scripts/python.exe"   # Windows layout
"$VPY" -m pip install -q --upgrade pip

# This run is inference only: no bitsandbytes, no unsloth, no trl.
step "dependencies"
"$VPY" -m pip install -q torch transformers accelerate openai pyyaml python-dotenv
echo "installed"
"$VPY" -c "import torch; print('cuda:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# --- Ollama ---------------------------------------------------------------------------
step "Ollama (serves the teacher model)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
  export PATH="$PATH:/usr/local/bin"
fi
command -v ollama >/dev/null 2>&1 || {
  echo "ollama is still not on PATH. Open a new shell and re-run, or install it from"
  echo "https://ollama.com/download and re-run."; exit 1; }

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "starting ollama serve in the background"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 || {
  echo "ollama did not start; see /tmp/ollama.log"; exit 1; }
echo "server up"
ollama pull qwen2.5:7b

# --- concurrency, from the actual card ------------------------------------------------
step "request concurrency"
VRAM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
fi
# qwen2.5:7b in 4-bit is ~4.7 GB and the evaluator ~1 GB. Each concurrent request adds
# its own KV cache on top, so concurrency is what decides whether this fits. Too high
# and Ollama silently moves layers to the CPU: no error, several times slower.
if   [ "${VRAM_MB:-0}" -ge 15000 ]; then PAR=8
elif [ "${VRAM_MB:-0}" -ge 7000  ]; then PAR=4
elif [ "${VRAM_MB:-0}" -ge 5000  ]; then PAR=2
else PAR=1; fi
echo "detected ${VRAM_MB} MB VRAM -> OLLAMA_NUM_PARALLEL=$PAR"
printf 'OLLAMA_NUM_PARALLEL=%s\nVERIMEM_WORKERS=%s\n' "$PAR" "$PAR" > handoff/c1-part2/.concurrency
echo "wrote handoff/c1-part2/.concurrency (run.sh reads this)"

if [ "$PAR" -gt 1 ]; then
  echo "restarting ollama so the server picks up the concurrency setting"
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
  OLLAMA_NUM_PARALLEL="$PAR" OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

# --- the relevance evaluator ----------------------------------------------------------
step "relevance evaluator"
if ls results/*_t5-evaluator/model/config.json >/dev/null 2>&1; then
  echo "already trained: $(ls -d results/*_t5-evaluator/model | head -1)"
else
  [ -f data/evaluator/relevance_train.jsonl ] || {
    echo "data/evaluator/relevance_train.jsonl is missing. It ships with the repo --"
    echo "check the clone completed."; exit 1; }
  echo "training (~25 min on a 4070). This is a one-off."
  "$VPY" -u -m evaluator.train --data data/evaluator --name t5-evaluator
fi

printf '\n=== setup complete ===\n'
echo "next: python handoff/c1-part2/preflight.py"
