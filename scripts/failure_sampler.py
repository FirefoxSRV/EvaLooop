"""Draw a blind, stratified sample of failed loops for manual coding.

The assignment asks for at least 20 failed loops to be inspected by hand. Two
things make that inspection worth trusting:

* **Stratification.** Failures are sampled across the depth at which they happen
  (early 2-3, mid 4-6, late 7-10) and across arms, so the sample is not dominated
  by whatever failure mode happens to be most common.
* **Blinding.** The CSV the coder fills in does not say which arm a failure came
  from, and the rows are shuffled. The arm labels live in a separate key file.
  Without that, knowing a failure came from the ablated arm would make it easy to
  code it as oracle dependence - exactly the finding the experiment is testing.

Writes two files:
  <out>/failures_to_code.csv   shuffled, unlabelled, one row per failure
  <out>/failures_key.json      row id -> model, arm, task id, cycle

The coder's reading copy is built separately by scripts/make_coding_worksheet.py,
which adds the mechanical facts that are tedious to check by eye.

The codebook is in PREREGISTRATION.md and is reproduced in the CSV header
comment, so the coder does not have to hold eight codes in their head.

Usage:
    python scripts/failure_sampler.py --results-dir results --per-arm 10
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CODEBOOK = {
    "C1": "summary drops or garbles a requirement (spec loss)",
    "C2": "summary describes a different but coherent task (semantic drift)",
    "C3": "summary is faithful, the code does not implement it",
    "C4": "signature/anchor failure: wrong name, arity or parameter semantics",
    "C5": "extraction or formatting failure (no code block, prose as code, truncation)",
    "C6": "oracle dependence: survived longer under assert, and the failing code is wrong on the leaked example input",
    "C7": "plus-only failure: base tests pass, extended EvalPlus tests do not",
    "C8": "harness or other (timeout, exception, unparseable)",
}

# Failure depth strata; a failure at cycle k means k-1 sustained loops.
STRATA = [("early", 2, 3), ("mid", 4, 6), ("late", 7, 10)]


def collect_failures(results_dir: Path):
    """Gather every mid-loop failure, with the context a coder needs."""
    failures = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "tasks" not in payload or "configuration" not in payload:
            continue
        max_cycles = payload["max_cycles"]
        for task in payload["tasks"]:
            sustained = task["successful_cycles"]
            # Only mid-loop failures: a cycle-1 failure never produced a summary,
            # so there is no drift to inspect.
            if not 1 <= sustained < max_cycles:
                continue
            cycles = task["cycles"]
            if len(cycles) <= sustained:
                continue
            failing = cycles[sustained]
            if failing.get("passed") is not False:
                continue
            failures.append({
                "model": payload["model_id"],
                "arm": payload["configuration"],
                "task_id": task["task_id"],
                "failed_at_cycle": sustained + 1,
                "sustained": sustained,
                "initial_prompt": task["initial_prompt"],
                "last_summary": cycles[sustained - 1].get("summary") or "",
                "last_passing_code": cycles[sustained - 1].get("code") or "",
                "failing_code": failing.get("code") or "",
                "base_passed": failing.get("base_passed"),
                "plus_passed": failing.get("plus_passed"),
                "error": failing.get("error"),
                "boundary_similarity": task.get("boundary_similarity"),
            })
    return failures


def stratum_of(failed_at_cycle: int):
    """Return the depth-stratum name for a failure cycle."""
    for name, lo, hi in STRATA:
        if lo <= failed_at_cycle <= hi:
            return name
    return "late"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--per-arm", type=int, default=10,
                        help="Failures to sample per arm (>=20 total is required)")
    parser.add_argument("--seed", type=int, default=20251013)
    args = parser.parse_args()

    failures = collect_failures(Path(args.results_dir))
    if not failures:
        print("No mid-loop failures found (runs still in progress?)", file=sys.stderr)
        return 1

    by_arm = {}
    for failure in failures:
        by_arm.setdefault(failure["arm"], []).append(failure)

    print(f"\nMid-loop failures available:")
    for arm, items in sorted(by_arm.items()):
        depths = {name: sum(1 for i in items if stratum_of(i["failed_at_cycle"]) == name)
                  for name, _, _ in STRATA}
        print(f"  {arm:<18} {len(items):>4}   " +
              "  ".join(f"{k}={v}" for k, v in depths.items()))

    rng = random.Random(args.seed)
    chosen = []
    for arm, items in sorted(by_arm.items()):
        # Spread the per-arm quota over depth strata, then top up from whatever
        # is left if a stratum is too small to fill its share.
        per_stratum = max(1, args.per_arm // len(STRATA))
        picked, used = [], set()
        for name, _, _ in STRATA:
            pool = [i for i in items if stratum_of(i["failed_at_cycle"]) == name]
            take = rng.sample(pool, min(per_stratum, len(pool)))
            picked.extend(take)
            used.update(id(t) for t in take)
        remaining = [i for i in items if id(i) not in used]
        shortfall = args.per_arm - len(picked)
        if shortfall > 0 and remaining:
            picked.extend(rng.sample(remaining, min(shortfall, len(remaining))))
        chosen.extend(picked[: args.per_arm])

    rng.shuffle(chosen)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unlabelled CSV for coding.
    csv_path = out_dir / "failures_to_code.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([f"# codebook: " + "; ".join(f"{k}={v}" for k, v in CODEBOOK.items())])
        writer.writerow(["row_id", "failed_at_cycle", "base_passed", "plus_passed",
                         "primary_code", "secondary_code", "notes"])
        for index, _failure in enumerate(chosen, start=1):
            writer.writerow([index, _failure["failed_at_cycle"],
                             _failure["base_passed"], _failure["plus_passed"], "", "", ""])

    # The key, kept separate so coding stays blind to the arm.
    key_path = out_dir / "failures_key.json"
    key_path.write_text(json.dumps({
        "seed": args.seed,
        "note": "Do not open until coding is complete; the arm label is the thing being tested.",
        "rows": {str(i): {"model": f["model"], "arm": f["arm"], "task_id": f["task_id"],
                          "failed_at_cycle": f["failed_at_cycle"],
                          "boundary_similarity": f["boundary_similarity"]}
                 for i, f in enumerate(chosen, start=1)},
    }, indent=2) + "\n")

    print(f"\nSampled {len(chosen)} failures "
          f"({args.per_arm} per arm x {len(by_arm)} arm(s)), shuffled, seed {args.seed}")
    print(f"  code here   : {csv_path}")
    print(f"  arm labels  : {key_path}  (open only after coding)")
    print(f"  next        : python scripts/make_coding_worksheet.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
