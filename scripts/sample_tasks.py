"""Draw the stratified MBPP+ task subset used by the anchor-ablation runs.

Sampling policy, fixed before any run (see PREREGISTRATION.md):

* Universe: the MBPP+ tasks whose assert line states a comparison, so all three
  anchors are defined on every sampled task. 362 of 378 qualify; the other 16
  assert bare truthiness (``assert is_woodall(383)``) and have no oracle that
  can be masked without destroying the call.
* Strata: 5 equal-width quantile strata over each task's *mean sustained loops
  across the 60 models in the data package*, so the subset spans loops that are
  easy and hard to sustain rather than over-representing either.
* Allocation: equal, ``--n / 5`` tasks per stratum, drawn without replacement
  with a fixed seed.

The script reports the resulting subset's ranking fidelity: the Spearman
correlation between each of the 60 models' ASL on the subset and on the full 378
tasks. That number is what justifies subsetting at all, so it is printed and
stored alongside the ids.

Usage:
    python scripts/sample_tasks.py --data-dir ../assignment2-data --n 150 --seed 20251013
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_pow
from evaloop.anchors import mask_assert_oracle

MAX_CYCLES = 10
NUM_STRATA = 5


def spearman(xs, ys) -> float:
    """Spearman rank correlation of two equal-length sequences (no ties expected)."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for rank, idx in enumerate(order):
            out[idx] = rank
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../assignment2-data")
    parser.add_argument("--n", type=int, default=150, help="Subset size (must divide by 5)")
    parser.add_argument("--seed", type=int, default=20251013)
    parser.add_argument("--out", default="configs/task_ids.json")
    args = parser.parse_args()

    if args.n % NUM_STRATA:
        print(f"--n must be divisible by {NUM_STRATA}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir).resolve()
    loop_files = sorted((data_dir / "loops").glob("*.json"))
    if not loop_files:
        print(f"No loop records under {data_dir / 'loops'}", file=sys.stderr)
        return 2

    # Per-model, per-task sustained loops, plus each task's assert line.
    per_model = {}
    assert_lines = {}
    for loop_file in loop_files:
        data = json.loads(loop_file.read_text())
        per_model[data["model_id"]] = {t["task_id"]: t["successful_cycles"] for t in data["tasks"]}
        for task in data["tasks"]:
            assert_lines.setdefault(task["task_id"], task["initial_prompt"].splitlines()[-2])

    all_ids = sorted(assert_lines)
    eligible = [tid for tid in all_ids if mask_assert_oracle(assert_lines[tid]) is not None]
    excluded = [tid for tid in all_ids if tid not in set(eligible)]

    if args.n > len(eligible):
        print(f"--n {args.n} exceeds the {len(eligible)} eligible tasks", file=sys.stderr)
        return 2

    # Difficulty signal: mean sustained loops across the package's models.
    difficulty = {tid: statistics.mean(m[tid] for m in per_model.values()) for tid in eligible}
    ordered = sorted(eligible, key=lambda tid: (difficulty[tid], tid))

    rng = random.Random(args.seed)
    per_stratum = args.n // NUM_STRATA
    chosen, strata_report = [], []
    for s in range(NUM_STRATA):
        lo = s * len(ordered) // NUM_STRATA
        hi = (s + 1) * len(ordered) // NUM_STRATA
        stratum = ordered[lo:hi]
        picked = rng.sample(stratum, per_stratum)
        chosen.extend(picked)
        strata_report.append({
            "stratum": s + 1,
            "difficulty_range": [round(difficulty[stratum[0]], 3), round(difficulty[stratum[-1]], 3)],
            "stratum_size": len(stratum),
            "sampled": per_stratum,
        })

    chosen.sort(key=lambda tid: int(tid.split("/")[-1]))

    # Ranking fidelity: subset ASL vs full-benchmark ASL across the 60 models.
    models = sorted(per_model)
    full = [asl_pow([per_model[m][t] for t in all_ids], MAX_CYCLES) for m in models]
    subset = [asl_pow([per_model[m][t] for t in chosen], MAX_CYCLES) for m in models]
    rho = spearman(full, subset)

    payload = {
        "task_ids": chosen,
        "provenance": {
            "n": args.n,
            "seed": args.seed,
            "num_strata": NUM_STRATA,
            "universe": "MBPP+ tasks whose assert line states a comparison (oracle is maskable)",
            "universe_size": len(eligible),
            "excluded_unmaskable": excluded,
            "difficulty_signal": f"mean sustained loops over the {len(models)} models in the data package",
            "strata": strata_report,
            "ranking_fidelity_spearman": round(rho, 4),
            "ranking_fidelity_note": (
                f"Spearman correlation between each of the {len(models)} package models' "
                "ASL on this subset and on all 378 tasks."
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nSampled {len(chosen)} of {len(eligible)} eligible tasks "
          f"(excluded {len(excluded)} unmaskable) with seed {args.seed}.")
    for s in strata_report:
        print(f"  stratum {s['stratum']}: difficulty {s['difficulty_range'][0]:>5.2f}-"
              f"{s['difficulty_range'][1]:>5.2f} loops, {s['sampled']}/{s['stratum_size']} sampled")
    print(f"\nRanking fidelity over {len(models)} package models: Spearman rho = {rho:.4f}")
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
