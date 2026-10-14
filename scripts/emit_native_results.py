"""Also emit each run in the framework's native results shape.

The runs are stored in the assignment data package's schema, which keeps one
analysis path for both the published records and ours. The deliverables ask for
"the framework's results JSON", so this writes the same runs in the shape
``EvaLoopEvaluator._save_results`` produces and ``evaloop.analysis.analyzer``
consumes - a list of model results, each with ``prompt_results`` whose cycles
carry ``task_1_output`` (the code) and ``task_2_output`` (the summary).

Nothing is recomputed: this is a pure re-serialisation of the same run, so the
two files always agree. ``--verify`` proves it by loading the output through the
framework's own ResultAnalyzer and checking its ASL against ours.

Usage:
    python scripts/emit_native_results.py --results-dir results --verify
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.metrics import asl_pow


def to_native(payload: dict) -> list:
    """Re-serialise a package-schema run into the framework's native shape."""
    prompt_results = []
    for task in payload["tasks"]:
        cycles = []
        for cycle in task["cycles"]:
            native = {
                "cycle": cycle["cycle"],
                "task_id": task["task_id"],
                "task_1_output": cycle.get("code", ""),
                "success": bool(cycle.get("passed", False)),
                "test_results": {
                    "success": bool(cycle.get("passed", False)),
                    "base_success": bool(cycle.get("base_passed", False)),
                    "plus_success": bool(cycle.get("plus_passed", False)),
                },
            }
            # A failing final cycle has no summary; omit the key rather than
            # writing null, which is what the framework itself does.
            if cycle.get("summary") is not None:
                native["task_2_output"] = cycle["summary"]
            if cycle.get("error"):
                native["test_results"]["error"] = cycle["error"]
            cycles.append(native)

        prompt_results.append({
            "task_id": task["task_id"],
            "initial_prompt": task["initial_prompt"],
            "successful_cycles": task["successful_cycles"],
            "max_cycles_reached": task["max_cycles_reached"],
            "cycles": cycles,
        })

    sustained = [p["successful_cycles"] for p in prompt_results]
    return [{
        "model": payload["model_id"],
        "configuration": payload["configuration"],
        "max_cycles": payload["max_cycles"],
        "average_successful_cycles": sum(sustained) / len(sustained) if sustained else 0.0,
        "prompt_results": prompt_results,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/native")
    parser.add_argument("--verify", action="store_true",
                        help="Load each output through the framework's ResultAnalyzer")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(results_dir.glob("*__*.json")):
        payload = json.loads(path.read_text())
        if "tasks" not in payload or "configuration" not in payload:
            continue
        native = to_native(payload)
        out_path = out_dir / f"{path.stem}_results.json"
        out_path.write_text(json.dumps(native, indent=2) + "\n")
        written += 1

        ours = asl_pow([t["successful_cycles"] for t in payload["tasks"]],
                       payload["max_cycles"])
        line = f"  {out_path.name:<52} ASL={ours:.4f}"

        if args.verify:
            # The framework's analyzer, run on the file we just wrote.
            from evaloop.analysis.analyzer import ResultAnalyzer
            analyzer = ResultAnalyzer(str(out_path), output_dir=str(out_dir / "_verify"))
            # ASL_pow is the unweighted quadratic ASL, i.e. the leaderboard's
            # aslScore; the judge-weighted "ASL" would need API calls.
            analysis = analyzer.analyze(metrics=["ASL_pow"], generate_plots=False)
            theirs = analysis["ASL_pow"]["by_model"][payload["model_id"]]
            agree = abs(theirs - ours) < 1e-9
            line += f"  framework={theirs:.4f}  {'OK' if agree else 'MISMATCH'}"
            if not agree:
                print(line)
                print(f"\nFAIL: {out_path.name} disagrees with our own metric", file=sys.stderr)
                return 1
        print(line)

    print(f"\nWrote {written} native-shape file(s) to {out_dir}")
    if args.verify and written:
        print("The framework's own ResultAnalyzer reproduces our ASL on every run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
