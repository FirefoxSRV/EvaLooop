#!/usr/bin/env bash
# Drive the full anchor ablation: models in parallel, anchors sequential.
#
# The free-tier quota is per model, not per account - the 429 payload says
# "limit: 30, model: gemma-4-26b" with quotaDimensions {model: ...}. So two
# models run concurrently get 30 requests/minute each and finish in roughly half
# the wall clock. Within a model the anchors stay sequential and the baseline
# runs first, because all three arms share cycle 1: running them in order lets
# the later arms replay those calls from cache, which is what makes pass@1
# identical across arms by construction rather than by assumption.
#
# Raising in-flight requests per model does not raise its ceiling (measured:
# 8 vs 24 in-flight gave identical wall clock and nearly double the 429s), so
# concurrency is set just high enough to keep the quota saturated.
#
# Resumable: every successful model call is cached, so re-running after an
# interruption replays from cache and continues where it stopped.
#
#   ./scripts/run_overnight.sh                 # default model set
#   MODELS="a b" ./scripts/run_overnight.sh    # override
#   SEQUENTIAL=1 ./scripts/run_overnight.sh    # one model at a time
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

run_model() {
  local model="$1"
  local conc; conc="$(concurrency_for "$model")"
  local log="$LOG_DIR/${model}.log"
  local start; start=$(date +%s)

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
  local status=$?
  local elapsed=$(( $(date +%s) - start ))

  if [ $status -eq 0 ]; then
    echo "[$model] done in $((elapsed / 60))m"
  else
    echo "[$model] FAILED (exit $status) after $((elapsed / 60))m; see $log"
  fi
  return $status
}

echo "=== anchor ablation: $(date) ==="
echo "models  : $MODELS"
echo "anchors : $ANCHORS"
echo "cycles  : $MAX_CYCLES"
echo "mode    : ${SEQUENTIAL:+sequential}${SEQUENTIAL:-parallel across models}"
echo

if [ -n "${SEQUENTIAL:-}" ]; then
  for model in $MODELS; do
    echo "--- $model -> $LOG_DIR/${model}.log"
    run_model "$model"
    echo
  done
else
  pids=""
  for model in $MODELS; do
    echo "--- $model -> $LOG_DIR/${model}.log (background)"
    run_model "$model" &
    pids="$pids $!"
  done
  # One model failing must not lose the others' results.
  for pid in $pids; do wait "$pid" || true; done
fi

echo
echo "=== finished: $(date) ==="
for model in $MODELS; do
  echo "--- $model"
  tail -7 "$LOG_DIR/${model}.log" 2>/dev/null | sed 's/^/    /'
done
ls -la "$OUT_DIR"/*.json 2>/dev/null | sed 's/^/  /'
