#!/usr/bin/env bash
# Assemble the Gradescope ZIP and check it against the assignment's list.
#
#   ./scripts/package_submission.sh [output.zip]
#
# Deliverables required: the report, a fork link plus a patch against upstream,
# the results JSON for every (model, configuration), summary.csv with an exact
# header, and a README. The data package is deliberately NOT included; records
# are referenced by model_id/task_id/cycle.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ZIP="${1:-submission.zip}"
STAGE="$(mktemp -d)"
UPSTREAM_BASE="${UPSTREAM_BASE:-upstream/dev}"
PYTHON="${PYTHON:-../.venv-gen/bin/python}"
EXPECTED_HEADER="model,configuration,n_tasks,success_rate,asl_score"
problems=0

note()  { echo "  [ok]   $*"; }
warn()  { echo "  [WARN] $*"; problems=$((problems + 1)); }

echo "==> Refreshing summary.csv and metrics.csv"
if "$PYTHON" scripts/summarize_runs.py --results-dir results --full >/dev/null 2>&1; then
  note "summary.csv regenerated"
else
  warn "summarize_runs.py failed; are there runs in results/ yet?"
fi

echo "==> Generating the patch against $UPSTREAM_BASE"
if git rev-parse --verify "$UPSTREAM_BASE" >/dev/null 2>&1; then
  git diff "$UPSTREAM_BASE"...HEAD > "$STAGE/evaloop-anchor-ablation.patch"
  lines=$(wc -l < "$STAGE/evaloop-anchor-ablation.patch" | tr -d ' ')
  note "patch written ($lines lines)"
  git diff --stat "$UPSTREAM_BASE"...HEAD | tail -1 | sed 's/^/         /'
else
  warn "$UPSTREAM_BASE not found; add the upstream remote and fetch it"
fi

echo "==> Collecting deliverables"
for f in README.md PREREGISTRATION.md; do
  [ -f "$f" ] && { cp "$f" "$STAGE/"; note "$f"; } || warn "missing $f"
done

if [ -f report.pdf ]; then
  cp report.pdf "$STAGE/"; note "report.pdf"
else
  warn "report.pdf missing (export the write-up before submitting)"
fi

mkdir -p "$STAGE/results"
runs=0
for f in results/*__*.json; do
  [ -e "$f" ] || continue
  cp "$f" "$STAGE/results/"; runs=$((runs + 1))
done
[ "$runs" -gt 0 ] && note "$runs run file(s)" || warn "no run files in results/"
for f in results/summary.csv results/metrics.csv results/stats.json results/tester_validation.json; do
  [ -f "$f" ] && { cp "$f" "$STAGE/results/"; note "$(basename "$f")"; } \
               || warn "missing $(basename "$f")"
done

if [ -d analysis ]; then
  mkdir -p "$STAGE/analysis"
  cp -R analysis/. "$STAGE/analysis/" 2>/dev/null
  note "manual failure coding"
else
  warn "analysis/ missing (run scripts/failure_sampler.py and code the sample)"
fi

echo "==> Checking summary.csv"
if [ -f results/summary.csv ]; then
  header="$(head -1 results/summary.csv)"
  if [ "$header" = "$EXPECTED_HEADER" ]; then
    note "header exact"
  else
    warn "header is '$header', expected '$EXPECTED_HEADER'"
  fi
  rows=$(( $(wc -l < results/summary.csv) - 1 ))
  note "$rows data row(s)"
fi

echo "==> Checking the data package is not bundled"
if find "$STAGE" -iname "leaderboard.json" -o -iname "loops" -type d | grep -q .; then
  warn "the data package appears to be inside the ZIP; remove it"
else
  note "data package not included"
fi

echo "==> Reminders"
grep -q "add your fork URL here" "$STAGE/README.md" 2>/dev/null \
  && warn "README still has the fork-URL placeholder" \
  || note "fork URL filled in"

rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$OLDPWD/$ZIP" .)
rm -rf "$STAGE"

echo
echo "==> $ZIP"
unzip -l "$ZIP" | tail -n +4 | head -40
if [ "$problems" -gt 0 ]; then
  echo
  echo "$problems item(s) need attention before submitting."
  exit 1
fi
echo
echo "All deliverables present."
