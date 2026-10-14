"""Export the MBPP+ metadata the generation stage needs, so it needs no evalplus.

The two stages of the pipeline have different dependencies by design. Testing
needs ``evalplus`` and executes untrusted code, so it runs in a container.
Generation only needs each task's starting prompt and the name of the function
the tests will call - two strings per task. Exporting them once means the
generation stage runs in a plain virtualenv with no evalplus and no execution of
model output, on a host whose Python ``evalplus`` does not even support.

Two sources, recorded in the output so a reader knows which was used:

``--source evalplus``
    Authoritative. Reads ``get_mbpp_plus()`` directly, so ``entry_point`` is the
    dataset's own. Needs evalplus; intended to run inside the test container.

``--source package``
    Offline fallback from the assignment data package, whose records carry every
    task's ``initial_prompt``. ``entry_point`` is recovered from the prompt's
    assert line by AST and then *validated*: for each task the script checks
    that the recovered name is actually defined as a function in a cycle of
    stored code that passed the tests. A name that passes that check is the
    name the tests called.

Usage:
    python scripts/export_dataset_meta.py --source package --data-dir ../assignment2-data
    python scripts/export_dataset_meta.py --source evalplus     # inside the container
"""

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.anchors import entry_point_from_anchor


def from_evalplus() -> dict:
    """Read prompts and entry points straight from the benchmark."""
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    return {
        task_id: {"prompt": problem["prompt"], "entry_point": problem["entry_point"]}
        for task_id, problem in problems.items()
    }


def from_package(data_dir: Path) -> tuple:
    """Recover prompts and entry points from the data package, and validate them.

    Returns:
        ``(meta, validation)`` where validation reports how many recovered entry
        points were confirmed against stored passing code.
    """
    loop_files = sorted((data_dir / "loops").glob("*.json"))
    if not loop_files:
        raise SystemExit(f"No loop records under {data_dir / 'loops'}")

    prompts, passing_code = {}, {}
    for loop_file in loop_files:
        data = json.loads(loop_file.read_text())
        for task in data["tasks"]:
            task_id = task["task_id"]
            prompts.setdefault(task_id, task["initial_prompt"])
            if task_id not in passing_code:
                for cycle in task["cycles"]:
                    if cycle.get("passed") and cycle.get("code"):
                        passing_code[task_id] = cycle["code"]
                        break

    meta, confirmed, unconfirmed, no_sample = {}, 0, [], []
    for task_id, prompt in prompts.items():
        anchor = prompt.splitlines()[-2]
        entry_point = entry_point_from_anchor(anchor)
        if entry_point is None:
            unconfirmed.append(f"{task_id}: no name recoverable from {anchor!r}")
            continue
        meta[task_id] = {"prompt": prompt, "entry_point": entry_point}

        sample = passing_code.get(task_id)
        if sample is None:
            no_sample.append(task_id)
            continue
        if _defines_function(sample, entry_point):
            confirmed += 1
        else:
            unconfirmed.append(f"{task_id}: {entry_point!r} not defined in passing code")

    return meta, {
        "confirmed_against_passing_code": confirmed,
        "no_passing_sample": len(no_sample),
        "unconfirmed": unconfirmed,
    }


def _defines_function(code: str, name: str) -> bool:
    """True if ``code`` defines a function called ``name`` anywhere."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["evalplus", "package"], default="package")
    parser.add_argument("--data-dir", default="../assignment2-data")
    parser.add_argument("--out", default="configs/mbpp_plus_meta.json")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    validation = None
    if args.source == "evalplus":
        meta = from_evalplus()
    else:
        meta, validation = from_package(Path(args.data_dir).resolve())

    payload = {
        "source": args.source,
        "num_tasks": len(meta),
        "validation": validation,
        "tasks": dict(sorted(meta.items(), key=lambda kv: int(kv[0].split("/")[-1]))),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nExported {len(meta)} task(s) from '{args.source}' to {out_path}")
    if validation:
        print(f"  entry points confirmed against stored passing code : "
              f"{validation['confirmed_against_passing_code']}/{len(meta)}")
        if validation["no_passing_sample"]:
            print(f"  tasks with no passing sample to check against     : "
                  f"{validation['no_passing_sample']}")
        if validation["unconfirmed"]:
            print(f"  UNCONFIRMED ({len(validation['unconfirmed'])}):")
            for line in validation["unconfirmed"][:10]:
                print(f"    - {line}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
