"""Paired statistics for the anchor ablation.

Every arm runs the same tasks for the same model, and cycle 1 is a shared,
cached generation, so the arms are paired task by task. Paired tests are both
more sensitive than independent ones here and the only honest choice: comparing
arms as independent samples would ignore that the hard tasks are hard in all
three.

Reported per (model, baseline arm -> other arm):

* dASL with a bootstrap confidence interval, resampling *tasks* (the unit of
  randomisation) rather than cycles.
* Wilcoxon signed-rank on per-task sustained cycles, computed here rather than
  pulled from scipy so the result is reproducible from the deliverables alone.
* The discordance table behind the test: how many tasks got worse, better or
  stayed level.

Across models it also reports whether each arm reorders them, and the spread of
ASL_bias, which is the pre-registered test of whether removing the leaked oracle
restores discriminative power.

Usage:
    python scripts/stats.py --results-dir results
    python scripts/stats.py --results-dir results --bootstrap 20000
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_bias, asl_pow

BASELINE = "assert"
ARM_ORDER = ("assert", "masked_assert", "signature")


def load_runs(results_dir: Path) -> dict:
    """Return {(model, anchor): {task_id: sustained_cycles}} plus max_cycles."""
    runs, max_cycles = {}, None
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "tasks" not in payload or "configuration" not in payload:
            continue
        key = (payload["model_id"], payload["configuration"])
        runs[key] = {t["task_id"]: t["successful_cycles"] for t in payload["tasks"]}
        max_cycles = payload["max_cycles"]
    return runs, max_cycles


def paired_bootstrap_delta(base, other, max_cycles, draws, seed):
    """Bootstrap CI for ASL(other) - ASL(base), resampling tasks together."""
    task_ids = sorted(set(base) & set(other))
    n = len(task_ids)
    observed = (asl_pow([other[t] for t in task_ids], max_cycles)
                - asl_pow([base[t] for t in task_ids], max_cycles))

    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = [task_ids[rng.randrange(n)] for _ in range(n)]
        deltas.append(asl_pow([other[t] for t in sample], max_cycles)
                      - asl_pow([base[t] for t in sample], max_cycles))
    deltas.sort()
    lo = deltas[int(0.025 * draws)]
    hi = deltas[min(int(0.975 * draws), draws - 1)]
    return observed, lo, hi, n


def wilcoxon_signed_rank(base, other):
    """Two-sided Wilcoxon signed-rank test on per-task sustained cycles.

    Uses the normal approximation with a tie correction and a continuity
    correction, which is appropriate at these sample sizes (n of non-zero
    differences well above 20). Returns (W, z, p, counts).
    """
    task_ids = sorted(set(base) & set(other))
    diffs = [other[t] - base[t] for t in task_ids]
    nonzero = [d for d in diffs if d != 0]
    worse = sum(1 for d in diffs if d < 0)
    better = sum(1 for d in diffs if d > 0)
    level = len(diffs) - worse - better

    counts = {"worse_under_other": worse, "better_under_other": better, "unchanged": level}
    n = len(nonzero)
    if n == 0:
        return 0.0, 0.0, 1.0, counts

    # Rank absolute differences, averaging ranks within ties.
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1

    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(nonzero, ranks) if d < 0)
    w = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4
    # Tie correction on the variance.
    tie_groups = {}
    for d in nonzero:
        tie_groups[abs(d)] = tie_groups.get(abs(d), 0) + 1
    tie_term = sum(t ** 3 - t for t in tie_groups.values())
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2) / 24
    if var_w <= 0:
        return w, 0.0, 1.0, counts

    z = (w - mean_w + 0.5) / math.sqrt(var_w)  # continuity correction
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return w, z, min(p, 1.0), counts


def spearman(xs, ys) -> float:
    """Spearman rank correlation (no tie handling; inputs are distinct scores)."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for rank, idx in enumerate(order):
            out[idx] = rank
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20251013)
    parser.add_argument("--out", default="results/stats.json")
    args = parser.parse_args()

    runs, max_cycles = load_runs(Path(args.results_dir))
    if not runs:
        print(f"No run files in {args.results_dir}", file=sys.stderr)
        return 2

    models = sorted({model for model, _ in runs})
    report = {"max_cycles": max_cycles, "bootstrap_draws": args.bootstrap,
              "seed": args.seed, "comparisons": [], "across_models": {}}

    print(f"\n=== Paired comparisons against the '{BASELINE}' arm "
          f"({args.bootstrap} bootstrap draws, seed {args.seed}) ===\n")
    for model in models:
        base = runs.get((model, BASELINE))
        if base is None:
            print(f"{model}: no '{BASELINE}' arm yet, skipping\n")
            continue
        for arm in ARM_ORDER:
            if arm == BASELINE or (model, arm) not in runs:
                continue
            other = runs[(model, arm)]
            delta, lo, hi, n = paired_bootstrap_delta(
                base, other, max_cycles, args.bootstrap, args.seed)
            w, z, p, counts = wilcoxon_signed_rank(base, other)
            significant = p < 0.05

            print(f"{model}  {BASELINE} -> {arm}   (n={n} paired tasks)")
            print(f"  dASL            {delta:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
            print(f"  Wilcoxon        W={w:.1f}  z={z:.2f}  p={p:.2e}"
                  f"   {'significant' if significant else 'not significant'} at 0.05")
            print(f"  per task        {counts['worse_under_other']} worse, "
                  f"{counts['better_under_other']} better, "
                  f"{counts['unchanged']} unchanged")
            print()

            report["comparisons"].append({
                "model": model, "baseline": BASELINE, "arm": arm, "n_paired": n,
                "delta_asl": delta, "ci95_low": lo, "ci95_high": hi,
                "wilcoxon_W": w, "wilcoxon_z": z, "p_value": p,
                "significant_at_0.05": significant, **counts,
            })

    # Across models: does an arm reorder them, and does it spread them out?
    print("=== Across models ===\n")
    for arm in ARM_ORDER:
        present = [m for m in models if (m, arm) in runs]
        if len(present) < 2:
            continue
        asl = [asl_pow(list(runs[(m, arm)].values()), max_cycles) for m in present]
        bias = [asl_bias(list(runs[(m, arm)].values()), max_cycles) for m in present]
        ranking = [m for _, m in sorted(zip(asl, present), reverse=True)]
        spread = statistics.pstdev(bias) if len(bias) > 1 else 0.0

        print(f"{arm}")
        print(f"  ASL        {', '.join(f'{m}={a:.3f}' for m, a in zip(present, asl))}")
        print(f"  ranking    {' > '.join(ranking)}")
        print(f"  sd(ASL_bias) {spread:.4f}")
        report["across_models"][arm] = {
            "models": present, "asl": asl, "asl_bias": bias,
            "ranking": ranking, "sd_asl_bias": spread,
        }

        if arm != BASELINE and all((m, BASELINE) in runs for m in present):
            base_asl = [asl_pow(list(runs[(m, BASELINE)].values()), max_cycles)
                        for m in present]
            rho = spearman(base_asl, asl)
            base_ranking = [m for _, m in sorted(zip(base_asl, present), reverse=True)]
            print(f"  vs {BASELINE}: Spearman rho={rho:.3f}  "
                  f"{'same order' if ranking == base_ranking else 'REORDERED'}")
            report["across_models"][arm]["spearman_vs_baseline"] = rho
            report["across_models"][arm]["reordered"] = ranking != base_ranking
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
