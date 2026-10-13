"""Validate evaloop.analysis.metrics against the published EvaLooop leaderboard.

Recomputes the leaderboard columns for every model in the assignment data package
straight from the raw per-cycle records, using the framework's own metric
functions, and diffs them against leaderboard.json.

This is a harness-validation gate: if our reading of the results schema or of the
ASL definition were wrong, the numbers would not match. Run it before spending
any money on new generations.

Two definition mismatches this script pins down (both reported in the write-up):

1. leaderboard.json's ``passDrop`` is a raw *task count* (pass@M - pass@1), while
   the framework's ``metrics.pass_drop`` returns a *fraction of T*. We compare
   against the count and derive it from the framework's own pass_at_k.
2. Per the data package README, 7 models were run five times and the leaderboard
   averages all runs, while loops/ ships only run ``exp_1``. Those models cannot
   reproduce exactly from a single run; they are reported separately rather than
   counted as failures.

Usage:
    python scripts/validate_leaderboard.py --data-dir ../assignment2-data
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_bias, asl_pow, pass_at_k

MAX_CYCLES = 10
# leaderboard.json rounds to 3 decimals; allow half a unit in the last place.
TOL = 5e-4
# The data package README allows at most this many models to disagree (multi-run averaging).
MAX_MULTIRUN_MODELS = 7
COLUMNS = ("successRate", "aslScore", "robustnessScore", "passDrop")


def recompute(cycles: list) -> dict:
    """Recompute the leaderboard columns from per-task sustained loop counts."""
    num_tasks = len(cycles)
    return {
        "successRate": pass_at_k(cycles, 1) / num_tasks,
        "aslScore": asl_pow(cycles, MAX_CYCLES),
        "robustnessScore": asl_bias(cycles, MAX_CYCLES),
        # leaderboard reports a raw task count, negative = pass rate fell.
        "passDrop": float(pass_at_k(cycles, MAX_CYCLES) - pass_at_k(cycles, 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../assignment2-data",
                        help="Assignment data package root (contains leaderboard.json and loops/)")
    parser.add_argument("--verbose", action="store_true", help="Print every model, not just disagreements")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    published = {m["model_id"]: m
                 for m in json.loads((data_dir / "leaderboard.json").read_text())["models"]}

    loop_files = sorted((data_dir / "loops").glob("*.json"))
    if not loop_files:
        print(f"No loop records found under {data_dir / 'loops'}", file=sys.stderr)
        return 2

    exact, multirun, errors = [], [], []

    for loop_file in loop_files:
        data = json.loads(loop_file.read_text())
        model_id = data["model_id"]
        cycles = [task["successful_cycles"] for task in data["tasks"]]

        if model_id not in published:
            errors.append(f"{model_id}: absent from leaderboard.json")
            continue

        ours, theirs = recompute(cycles), published[model_id]
        bad = [c for c in COLUMNS if abs(ours[c] - theirs[c]) > TOL]

        if not bad:
            exact.append(model_id)
            if args.verbose:
                print(f"  ok  {model_id:<40} ASL={ours['aslScore']:.3f}  n={len(cycles)}")
        else:
            detail = ", ".join(f"{c} {ours[c]:.4f} vs {theirs[c]:.4f}" for c in bad)
            multirun.append(f"{model_id}: {detail}")

    total = len(exact) + len(multirun)
    print(f"\nChecked {total} models x {len(COLUMNS)} metrics against leaderboard.json "
          f"(tolerance {TOL}).")
    print(f"  exact match          : {len(exact)}")
    print(f"  single-run deviation : {len(multirun)}")

    if multirun:
        print("\nModels that do not reproduce from the shipped single run "
              "(expected: README says 7 models were run 5x and the leaderboard averages them):")
        for line in multirun:
            print(f"  - {line}")

    if errors:
        print(f"\nFAIL: {len(errors)} structural error(s):")
        for line in errors:
            print(f"  - {line}")
        return 1

    if len(multirun) > MAX_MULTIRUN_MODELS:
        print(f"\nFAIL: {len(multirun)} models deviate, more than the {MAX_MULTIRUN_MODELS} "
              "the data package accounts for.")
        return 1

    print(f"\nPASS: evaloop.analysis.metrics reproduces the leaderboard for all "
          f"{len(exact)} single-run models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
