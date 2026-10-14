#!/usr/bin/env bash
# Drive the full anchor ablation for several models, sequentially.
#
# Sequential by design: the provider's quota is per account, roughly 26
# requests/minute, so running models in parallel does not finish sooner - it
# only multiplies 429s and backoff. Measured: raising in-flight requests from 8
# to 24 left wall clock unchanged (135s per 60 calls) and nearly doubled the
# throttle hits.
#
# Resumable: every model call is cached, so re-running after an interruption
# replays from cache and continues where it stopped. Baseline first, so the
# later arms reuse cycle 1 and pass@1 is identical across arms by construction.
#
#   ./scripts/run_overnight.sh                 # default model set
#   MODELS="a b" ./scripts/run_overnight.sh    # override
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-../.venv-gen/bin/python}"
MODELS="${MODELS:-gemma-4-26b-a4b-it gemini-3.5-flash-lite}"
ANCHORS="${ANCHORS:-assert,masked_assert,signature}"
MAX_CYCLES="${MAX_CYCLES:-10}"
OUT_DIR="${OUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# Enough in-flight requests to saturate the per-minute quota without piling on
# retries. gemma saturates at 8; the slower gemini needs more.
concurrency_for() {
  case "$1" in
    gemma-*) echo 8 ;;
    *)       echo 24 ;;
  esac
}

echo "=== anchor ablation: $(date) ==="
echo "models  : $MODELS"
echo "anchors : $ANCHORS"
echo "cycles  : $MAX_CYCLES"
echo

for model in $MODELS; do
  conc="$(concurrency_for "$model")"
  log="$LOG_DIR/${model}.log"
  echo "--- $model (concurrency $conc) -> $log"
  start=$(date +%s)

  "$PYTHON" -u scripts/run_anchored_experiment.py \
      --model "$model" \
      --anchor "$ANCHORS" \
      --max-cycles "$MAX_CYCLES" \
      --thinking-level minimal \
      --concurrency "$conc" \
      --tester docker \
      --output-dir "$OUT_DIR" \
      --log-level WARNING \
      >> "$log" 2>&1
  status=$?

  elapsed=$(( $(date +%s) - start ))
  if [ $status -eq 0 ]; then
    echo "    done in $((elapsed / 60))m"
    tail -6 "$log" | sed 's/^/    /'
  else
    # Keep going: one model failing should not lose the others' results.
    echo "    FAILED (exit $status) after $((elapsed / 60))m; see $log"
    tail -15 "$log" | sed 's/^/    /'
  fi
  echo
done

echo "=== finished: $(date) ==="
ls -la "$OUT_DIR"/*.json 2>/dev/null | sed 's/^/  /'
