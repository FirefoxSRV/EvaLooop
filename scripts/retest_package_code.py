"""Validate our tester by re-executing code from the published records.

The data package stores, for every cycle, the code a model generated and whether
it passed. Re-running that stored code through our sandbox and comparing against
the recorded verdict checks the testing half of the pipeline end to end -
dataset, oracles, EvalPlus invocation, container, result plumbing - independently
of anything our own runs produce. If this disagrees, our pass/fail outcomes are
not comparable to the published ones and nothing downstream can be trusted.

It also measures the cost of the assert-stripping rule. Stripping is applied to
every arm of the ablation, so it must not change baseline outcomes; this script
tests each sample twice, as stored and as stripped, and reports any verdict that
changes.

Usage:
    python scripts/retest_package_code.py --n 500
    python scripts/retest_package_code.py --n 200 --models gpt-5,gemini-2.5-pro
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.evaluation.sandbox_tester import (
    DockerSandboxTester,
    LocalTester,
    SandboxUnavailable,
)
from evaloop.tasks.stripped_code_generation import strip_module_level_asserts


def collect_samples(data_dir: Path, models, limit: int, seed: int):
    """Draw (model_id, task_id, cycle, code, recorded_passed) tuples at random."""
    loop_files = sorted((data_dir / "loops").glob("*.json"))
    if models:
        wanted = {m.strip() for m in models.split(",")}
        loop_files = [f for f in loop_files if f.stem in wanted]

    samples = []
    for loop_file in loop_files:
        data = json.loads(loop_file.read_text())
        for task in data["tasks"]:
            for cycle in task["cycles"]:
                code = cycle.get("code")
                if not code or "passed" not in cycle:
                    continue
                samples.append({
                    "model_id": data["model_id"],
                    "task_id": task["task_id"],
                    "cycle": cycle["cycle"],
                    "code": code,
                    "recorded": bool(cycle["passed"]),
                })

    rng = random.Random(seed)
    rng.shuffle(samples)
    # Keep passes and failures both represented rather than whatever the shuffle
    # happens to give: a tester that said "fail" to everything would otherwise
    # look fine on a mostly-failing sample.
    passes = [s for s in samples if s["recorded"]][: limit // 2]
    fails = [s for s in samples if not s["recorded"]][: limit - len(passes)]
    chosen = passes + fails
    rng.shuffle(chosen)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../assignment2-data")
    parser.add_argument("--models", help="Comma-separated model ids (default: all)")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20251013)
    parser.add_argument("--tester", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="evaloop-test:latest")
    parser.add_argument("--out", default="results/tester_validation.json")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    samples = collect_samples(Path(args.data_dir).resolve(), args.models, args.n, args.seed)
    if not samples:
        print("No samples found", file=sys.stderr)
        return 2

    if args.tester == "local":
        tester = LocalTester()
    else:
        tester = DockerSandboxTester(image=args.image,
                                     repo_root=Path(__file__).resolve().parents[1])
        try:
            tester.preflight()
        except SandboxUnavailable as exc:
            raise SystemExit(f"\nSandbox unavailable:\n  {exc}\n")

    recorded_pass = sum(1 for s in samples if s["recorded"])
    print(f"\nRe-testing {len(samples)} stored samples "
          f"({recorded_pass} recorded pass, {len(samples) - recorded_pass} recorded fail)")
    print(f"Tester: {tester}\n")

    # As stored.
    as_stored = tester.test_batch([(s["task_id"], s["code"]) for s in samples])
    # As the ablation would test it.
    stripped_code, strip_fired = [], 0
    for sample in samples:
        code, removed = strip_module_level_asserts(sample["code"])
        strip_fired += 1 if removed else 0
        stripped_code.append((sample["task_id"], code))
    as_stripped = tester.test_batch(stripped_code)

    agree = disagree = 0
    strip_changed = []
    mismatches = []
    for sample, stored_result, stripped_result in zip(samples, as_stored, as_stripped):
        ours = bool(stored_result.get("passed"))
        if ours == sample["recorded"]:
            agree += 1
        else:
            disagree += 1
            if len(mismatches) < 10:
                mismatches.append({
                    "model_id": sample["model_id"], "task_id": sample["task_id"],
                    "cycle": sample["cycle"], "recorded": sample["recorded"], "ours": ours,
                    "error": stored_result.get("error"),
                })
        if bool(stripped_result.get("passed")) != ours:
            strip_changed.append({
                "model_id": sample["model_id"], "task_id": sample["task_id"],
                "cycle": sample["cycle"], "as_stored": ours,
                "as_stripped": bool(stripped_result.get("passed")),
            })

    rate = agree / len(samples)
    print(f"Agreement with the recorded verdicts : {agree}/{len(samples)} ({rate:.2%})")
    print(f"Disagreements                        : {disagree}")
    print(f"\nAssert-stripping:")
    print(f"  samples containing a module-level assert : {strip_fired}/{len(samples)}")
    print(f"  verdicts changed by stripping            : {len(strip_changed)}")

    if mismatches:
        print("\nDisagreement examples (referenced by model_id/task_id/cycle):")
        for m in mismatches:
            print(f"  - {m['model_id']} {m['task_id']} cycle {m['cycle']}: "
                  f"recorded={m['recorded']} ours={m['ours']}"
                  + (f" error={m['error'][:60]}" if m.get("error") else ""))
    if strip_changed:
        print("\nStripping changed these verdicts:")
        for c in strip_changed[:10]:
            print(f"  - {c['model_id']} {c['task_id']} cycle {c['cycle']}: "
                  f"{c['as_stored']} -> {c['as_stripped']}")

    payload = {
        "n_samples": len(samples), "seed": args.seed,
        "agreement": agree, "disagreement": disagree, "agreement_rate": rate,
        "strip_fired": strip_fired, "strip_changed_verdict": len(strip_changed),
        "strip_changed_examples": strip_changed[:50],
        "mismatch_examples": mismatches,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
