#!/usr/bin/env bash
# The two C1 part 2 experiments. About 2.5 hours, unattended.
#
# Resumable: every model call is cached under cache/, so re-running the same command
# replays completed work instantly and continues from where it stopped. Do not delete
# cache/ between attempts.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
HANDOFF="$REPO/handoff/c1-part2"
LOGS="$HANDOFF/logs"
mkdir -p "$LOGS"

VPY="$REPO/.venv/bin/python"
[ -x "$VPY" ] || VPY="$REPO/.venv/Scripts/python.exe"
[ -x "$VPY" ] || { echo "no .venv — run setup.sh first"; exit 1; }

# --- packaging mode -------------------------------------------------------------------
if [ "${1:-}" = "--package" ]; then
  ZIP="$REPO/verimem-c1-part2-results.zip"
  rm -f "$ZIP"
  # Only results and logs. No cache (gigabytes), no data, no fabricated attack content.
  "$VPY" - "$ZIP" <<'PYEOF'
import sys, zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
wanted = []
for d in Path("results").glob("*source-poison*"):
    wanted += [p for p in d.rglob("*") if p.is_file()]
for d in Path("results").glob("*propagation*"):
    wanted += [p for p in d.rglob("*") if p.is_file()]
wanted += [p for p in Path("handoff/c1-part2/logs").glob("*.log")]

# Fabricated attack content never leaves the machine (project ethics rule 10). The
# metrics and configs carry the findings; the fakes themselves are reproducible from
# the fixed seed if anyone needs them.
skip = ("corrupted_trajectories.jsonl",)
wanted = [p for p in wanted if p.name not in skip]

if not wanted:
    raise SystemExit("nothing to package — did the runs produce results/?")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(wanted)):
        z.write(p, p.as_posix())
print(f"{len(set(wanted))} files -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
for p in sorted(set(wanted)):
    print(f"  {p}")
PYEOF
  echo
  echo "Send that zip to Teesha."
  exit 0
fi

# --- environment ----------------------------------------------------------------------
export VERIMEM_API_BASE_URL="http://localhost:11434/v1"
export VERIMEM_API_MODEL="qwen2.5:7b"
export VERIMEM_API_KEY="${VERIMEM_API_KEY:-ollama}"

# setup.sh wrote the concurrency that fits this card. Without it, one worker: slower
# but correct, which is the right way round to fail.
if [ -f "$HANDOFF/.concurrency" ]; then
  # shellcheck disable=SC1090
  set -a; . "$HANDOFF/.concurrency"; set +a
fi
WORKERS="${VERIMEM_WORKERS:-1}"
echo "[run] workers: $WORKERS"
[ "$WORKERS" = "1" ] && echo "[run] NOTE: 1 worker is ~4x slower. Run setup.sh to set this from your VRAM."

EVALUATOR="$(ls -d "$REPO"/results/*_t5-evaluator/model 2>/dev/null | tail -1 || true)"
[ -n "$EVALUATOR" ] || { echo "no trained evaluator — run setup.sh first"; exit 1; }
echo "[run] evaluator: $EVALUATOR"

COMMON=(
  --data "$REPO/data/averitec/dev.json"
  --store "$REPO/data/averitec/pool_dev.jsonl"
  --trajectories "$REPO/data/trajectories/trajectories.jsonl"
  --evaluator "$EVALUATOR"
  --misinfo "$REPO/data/averitec/misinfo_list.txt"
  --targets 100
  --workers "$WORKERS"
)

stage() {
  local name="$1"; shift
  printf '\n################ %s ################\n' "$name"
  date -u "+started %Y-%m-%dT%H:%M:%SZ"
  # tee so a crash still leaves the log, and because every runner prints its own
  # metrics to stdout -- that is how the numbers get back even if a file is lost.
  if ! "$@" 2>&1 | tee "$LOGS/$name.log"; then
    echo "!! $name FAILED -- see $LOGS/$name.log"
    echo "!! Completed calls are cached; re-running this script resumes."
    exit 1
  fi
  date -u "+finished %Y-%m-%dT%H:%M:%SZ"
}

# P3.6 -- source poisoning. Four page counts give the dose curve; the naive adversary
# cites known misinformation domains.
stage "source-poison-naive" \
  "$VPY" -u -m eval.run_source_poisoning "${COMMON[@]}" \
  --adversary naive --pages 0 1 3 5 --name source-poison-naive

# The adaptive adversary impersonates the publishers the history trusts most, which
# defeats the trust defence by construction. Two page counts: the curve is already
# established, this asks whether the defence survives a stronger attacker.
stage "source-poison-adaptive" \
  "$VPY" -u -m eval.run_source_poisoning "${COMMON[@]}" \
  --adversary adaptive --pages 0 5 --name source-poison-adaptive

# P3.8 -- propagation, run twice: neighbours undefended, then with both defences on.
stage "propagation-undefended" \
  "$VPY" -u -m eval.run_propagation "${COMMON[@]}" \
  --rate 0.2 --per-target 1 --name propagation-undefended

stage "propagation-defended" \
  "$VPY" -u -m eval.run_propagation "${COMMON[@]}" \
  --rate 0.2 --per-target 1 --defences --name propagation-defended

printf '\n################ all four runs finished ################\n'
echo "Package the results and send them back:"
echo
echo "    bash handoff/c1-part2/run.sh --package"
