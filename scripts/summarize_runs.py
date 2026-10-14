"""Build summary.csv and the metrics table from the anchor-ablation runs.

Every number is computed with the framework's own ``evaloop.analysis.metrics``,
which ``scripts/validate_leaderboard.py`` checks against the published
leaderboard. Nothing is reimplemented here.

The CSV header is the one the assignment specifies:

    model,configuration,n_tasks,success_rate,asl_score

``--full`` additionally writes metrics.csv with the columns the analysis needs:
pass@1, ASL, ASL_bias (robustness independent of one-shot accuracy), pass-drop
as a task count, mid-loop failures, the sustained-cycle histogram, and the
assert-echo rate that mediates the ablation's effect.

Usage:
    python scripts/summarize_runs.py --results-dir results
    python scripts/summarize_runs.py --results-dir results --full --baseline assert
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_bias, asl_pow, pass_at_k
from evaloop.tasks.stripped_code_generation import count_module_level_asserts

# Configuration labels used in the deliverables, keyed by anchor name.
CONFIG_LABELS = {
    "assert": "standard-mbpp-plus",          # upstream loop = the baseline arm
    "masked_assert": "anchor-masked-oracle",
    "signature": "anchor-signature-only",
}


def load_runs(results_dir: Path):
    """Read every run payload, newest wins on duplicate (model, anchor)."""
    runs = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "tasks" not in payload or "configuration" not in payload:
            continue  # not a run file (e.g. tester_validation.json)
        runs[(payload["model_id"], payload["configuration"])] = (path, payload)
    return runs


def metrics_for(payload):
    """Compute every reported metric for one run."""
    tasks = payload["tasks"]
    max_cycles = payload["max_cycles"]
    cycles = [t["successful_cycles"] for t in tasks]
    n = len(cycles)

    total_cycles = echo_cycles = 0
    for task in tasks:
        for cycle in task["cycles"]:
            total_cycles += 1
            if count_module_level_asserts(cycle.get("code") or ""):
                echo_cycles += 1

    histogram = [0] * (max_cycles + 1)
    for c in cycles:
        histogram[c] += 1

    return {
        "model": payload["model_id"],
        "anchor": payload["configuration"],
        "configuration": CONFIG_LABELS.get(payload["configuration"], payload["configuration"]),
        "n_tasks": n,
        "success_rate": pass_at_k(cycles, 1) / n,
        "asl_score": asl_pow(cycles, max_cycles),
        "asl_bias": asl_bias(cycles, max_cycles),
        "pass_drop_count": pass_at_k(cycles, max_cycles) - pass_at_k(cycles, 1),
        "mean_cycles": sum(cycles) / n,
        "survived_all": histogram[max_cycles],
        "mid_failures": sum(histogram[1:max_cycles]),
        "failed_at_1": histogram[0],
        "assert_echo_rate": echo_cycles / total_cycles if total_cycles else 0.0,
        "total_cycles_run": total_cycles,
        "histogram": histogram,
        "wall_seconds": payload.get("run_meta", {}).get("wall_seconds"),
        "api_calls": payload.get("run_meta", {}).get("usage", {}).get("calls"),
        "anchor_fallbacks": payload.get("run_meta", {}).get("anchor_fallbacks", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument("--full", action="store_true", help="Also write metrics.csv")
    parser.add_argument("--baseline", default="assert", help="Anchor to compare against")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        print(f"No run files in {results_dir}", file=sys.stderr)
        return 2

    rows = [metrics_for(payload) for _path, payload in runs.values()]
    # Group by model, baseline arm first.
    order = {"assert": 0, "masked_assert": 1, "signature": 2}
    rows.sort(key=lambda r: (r["model"], order.get(r["anchor"], 9)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "configuration", "n_tasks", "success_rate", "asl_score"])
        for row in rows:
            writer.writerow([row["model"], row["configuration"], row["n_tasks"],
                             f"{row['success_rate']:.4f}", f"{row['asl_score']:.4f}"])
    print(f"Wrote {out} ({len(rows)} rows)")

    if args.full:
        full_out = out.with_name("metrics.csv")
        columns = ["model", "anchor", "configuration", "n_tasks", "success_rate",
                   "asl_score", "asl_bias", "pass_drop_count", "mean_cycles",
                   "survived_all", "mid_failures", "failed_at_1",
                   "assert_echo_rate", "total_cycles_run", "wall_seconds", "api_calls"]
        with full_out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote {full_out}")

    # Console table.
    print(f"\n{'model':<24}{'anchor':<15}{'pass@1':>8}{'ASL':>8}{'ASL_bias':>10}"
          f"{'mid-fail':>9}{'surv10':>8}{'echo%':>7}")
    for row in rows:
        print(f"{row['model']:<24}{row['anchor']:<15}{row['success_rate']:>8.3f}"
              f"{row['asl_score']:>8.3f}{row['asl_bias']:>10.3f}"
              f"{row['mid_failures']:>9}{row['survived_all']:>8}"
              f"{100 * row['assert_echo_rate']:>6.1f}%")

    # Per-model deltas against the baseline arm: the headline of the ablation.
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["anchor"]] = row
    deltas = [(m, arms) for m, arms in by_model.items() if args.baseline in arms]
    if deltas:
        print(f"\nChange from the '{args.baseline}' arm:")
        print(f"{'model':<24}{'anchor':<15}{'dASL':>8}{'d mid-fail':>12}{'d surv10':>10}")
        for model, arms in deltas:
            base = arms[args.baseline]
            for anchor, row in arms.items():
                if anchor == args.baseline:
                    continue
                print(f"{model:<24}{anchor:<15}"
                      f"{row['asl_score'] - base['asl_score']:>+8.3f}"
                      f"{row['mid_failures'] - base['mid_failures']:>+12d}"
                      f"{row['survived_all'] - base['survived_all']:>+10d}")

    incomplete = [f"{m}/{a}" for m, arms in by_model.items()
                  for a in ("assert", "masked_assert", "signature") if a not in arms]
    if incomplete:
        print(f"\nNote: missing arms (run still in progress?): {', '.join(incomplete)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
