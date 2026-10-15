"""Score the failure coding: unblind, derive C6, and measure agreement.

Run after every row in ``failures_to_code.csv`` carries a primary code. Three
things happen, in this order:

1. **Unblind.** Join the codes to the arm each failure came from.
2. **Derive C6.** Oracle dependence cannot be coded from a single loop - it
   means the same task sustained strictly longer under the unablated anchor - so
   it is computed from the run data rather than judged.
3. **Agreement.** Compare the primary coding against the sealed LLM second pass
   and report Cohen's kappa. The second coder is reported under the assignment's
   LLM-usage rule; it is a reliability check, not the analysis.

Usage:
    python scripts/score_coding.py --results-dir results
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"C1", "C2", "C3", "C4", "C5", "C7", "C8"}  # C6 is derived, not coded.


def cohens_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two coders over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in set(a) | set(b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--coding", default="analysis/failures_to_code.csv")
    parser.add_argument("--key", default="analysis/failures_key.json")
    parser.add_argument("--sealed", default="analysis/coder_llm_sealed.json")
    parser.add_argument("--out", default="analysis/coding_summary.json")
    parser.add_argument("--no-agreement", action="store_true",
                        help="Skip the second-coder comparison (use when the\n                              primary coding and the sealed pass are the same work)")
    args = parser.parse_args()

    rows = list(csv.reader(Path(args.coding).open()))
    coded = {r[0]: r[4].strip().upper() for r in rows[2:] if len(r) > 4}
    missing = [k for k, v in coded.items() if not v]
    if missing:
        print(f"{len(missing)} row(s) not yet coded: {', '.join(sorted(missing, key=int))}",
              file=sys.stderr)
        print("Fill the primary_code column in "
              f"{args.coding} (worksheet: analysis/CODING_WORKSHEET.md)", file=sys.stderr)
        return 1
    bad = {k: v for k, v in coded.items() if v not in VALID}
    if bad:
        print(f"Unrecognised code(s): {bad}. Valid: {', '.join(sorted(VALID))} "
              "(C6 is derived, not coded).", file=sys.stderr)
        return 1

    key = json.loads(Path(args.key).read_text())["rows"]
    sustained = {}
    for path in Path(args.results_dir).glob("*__*.json"):
        payload = json.loads(path.read_text())
        for task in payload["tasks"]:
            sustained[(payload["model_id"], payload["configuration"],
                       task["task_id"])] = task["successful_cycles"]

    by_arm, oracle_dependent = defaultdict(Counter), []
    for row_id, code in coded.items():
        meta = key[row_id]
        model, arm, task = meta["model"], meta["arm"], meta["task_id"]
        base = sustained.get((model, "assert", task))
        here = sustained.get((model, arm, task))
        if arm != "assert" and base is not None and here is not None and base > here:
            oracle_dependent.append({"row": row_id, "model": model, "arm": arm,
                                     "task_id": task, "mechanism": code,
                                     "sustained_assert": base, "sustained_here": here})
        by_arm[arm][code] += 1

    print(f"\n=== primary code distribution by arm ({len(coded)} failures) ===")
    codes = sorted({c for ctr in by_arm.values() for c in ctr})
    print(f"{'arm':<16}" + "".join(f"{c:>6}" for c in codes) + f"{'n':>6}")
    for arm in ["assert", "masked_assert", "signature"]:
        if arm in by_arm:
            ctr = by_arm[arm]
            print(f"{arm:<16}" + "".join(f"{ctr.get(c, 0):>6}" for c in codes)
                  + f"{sum(ctr.values()):>6}")

    print(f"\n=== C6 oracle dependence (derived, not coded) ===")
    print(f"  {len(oracle_dependent)}/{len(coded)} sampled failures are on tasks that "
          f"sustained strictly\n  more cycles under the unablated 'assert' anchor.")
    collapsed = [o for o in oracle_dependent if o["sustained_assert"] == 10
                 and o["sustained_here"] <= 2]
    if collapsed:
        print(f"  {len(collapsed)} of them collapsed from ten cycles to <=2.")

    agreement = None
    sealed_path = Path(args.sealed)
    if args.no_agreement:
        print("\n(agreement against a second coder not computed: --no-agreement)")
    elif sealed_path.exists():
        try:
            sealed = json.loads(sealed_path.read_text())
            theirs = sealed["codes"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"\nSkipping agreement: {sealed_path} is not a readable "
                  f"second-coder file ({exc}).")
            theirs = None
        if theirs is not None:
            shared = sorted(set(coded) & set(theirs), key=int)
            mine = [coded[r] for r in shared]
            llm = [theirs[r]["primary_code"] for r in shared]
            exact = sum(1 for x, y in zip(mine, llm) if x == y)
            kappa = cohens_kappa(mine, llm)
            agreement = {"n": len(shared), "exact": exact,
                         "percent": round(100 * exact / len(shared), 1),
                         "cohens_kappa": round(kappa, 3),
                         "second_coder": sealed.get("coder")}
            print(f"\n=== agreement with the sealed second coder ===")
            print(f"  second coder : {sealed.get('coder')}")
            print(f"  exact        : {exact}/{len(shared)} ({agreement['percent']}%)")
            print(f"  Cohen's kappa: {kappa:.3f}")
            disagreements = [(r, coded[r], theirs[r]["primary_code"]) for r in shared
                             if coded[r] != theirs[r]["primary_code"]]
            if disagreements:
                print(f"  disagreements ({len(disagreements)}):")
                for row_id, a, b in disagreements:
                    print(f"    row {row_id:>2}: primary={a}  second={b}  "
                          f"[{key[row_id]['arm']}, {key[row_id]['task_id']}]")

    Path(args.out).write_text(json.dumps({
        "n_coded": len(coded),
        "by_arm": {k: dict(v) for k, v in by_arm.items()},
        "oracle_dependent": oracle_dependent,
        "oracle_dependent_count": len(oracle_dependent),
        "collapsed_from_ten": len(collapsed),
        "second_coder_agreement": agreement,
    }, indent=2) + "\n")
    print(f"\nWritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
