"""Functional-test worker. Runs *inside* the sandbox container.

Reads a JSON batch of candidate solutions on stdin, executes each against the
MBPP+ test suites with the framework's own EvalPlusCodeTester, and writes JSON
results to stdout. Nothing else in the pipeline executes model-generated code.

Input (stdin):
    {"items": [{"task_id": "Mbpp/2", "code": "def f(): ..."}, ...]}

Output (stdout):
    {"results": [{"task_id": "Mbpp/2", "passed": true,
                  "base_passed": true, "plus_passed": true}, ...]}

Field names follow the assignment data package's schema (``base_passed`` /
``plus_passed``), not the tester's internal ``*_success`` keys, so one analysis
pipeline reads both the published records and ours.

Stdout carries only the JSON payload; logs and EvalPlus chatter go to stderr.
"""

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    items = payload.get("items", [])

    # Imported here so a missing evalplus fails with a clear message on stderr
    # rather than at module import inside the container entrypoint.
    from evaloop.evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset

    print(f"[worker] loading MBPP+ and ground truth...", file=sys.stderr)
    problems, ground_truth = load_mbpp_dataset()
    tester = EvalPlusCodeTester(problems, ground_truth)
    print(f"[worker] testing {len(items)} candidate(s)", file=sys.stderr)

    results = []
    for item in items:
        task_id = item["task_id"]
        problem = problems.get(task_id)
        if problem is None:
            results.append({
                "task_id": task_id, "passed": False,
                "base_passed": False, "plus_passed": False,
                "error": f"unknown task_id {task_id}",
            })
            continue

        try:
            outcome = tester.test_code(item["code"], task_id, problem["entry_point"])
        except Exception as exc:  # noqa: BLE001 - untrusted code, any failure is a fail
            results.append({
                "task_id": task_id, "passed": False,
                "base_passed": False, "plus_passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        record = {
            "task_id": task_id,
            "passed": bool(outcome.get("success", False)),
            "base_passed": bool(outcome.get("base_success", False)),
            "plus_passed": bool(outcome.get("plus_success", False)),
        }
        if "error" in outcome:
            record["error"] = str(outcome["error"])
        results.append(record)

    json.dump({"results": results}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
