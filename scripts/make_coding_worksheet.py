"""Build the human coder's worksheet for the failure analysis.

Produces one self-contained Markdown file with every sampled failure laid out in
the order a coder needs it: the original task, the summary that fed the failing
cycle, the code that passed the cycle before, and the code that failed. The arm
label is withheld - it lives in failures_key.json - because knowing a failure
came from an ablated arm would make it trivial to code as oracle dependence,
which is the finding under test.

Mechanical facts that are tedious to check by eye are precomputed (does the code
parse, is the entry point defined, how many functions, base vs plus outcome).
They are deliberately neutral: they say what is true of the artifact, never which
code to assign.

Usage:
    python scripts/make_coding_worksheet.py --out analysis/CODING_WORKSHEET.md
"""

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaloop.anchors import entry_point_from_anchor
from failure_sampler import CODEBOOK, collect_failures


def diagnostics(failure: dict) -> list:
    """Neutral, mechanical facts about the failing artifact."""
    code = failure["failing_code"] or ""
    entry = entry_point_from_anchor(failure["initial_prompt"].splitlines()[-2])
    facts = []

    try:
        tree = ast.parse(code)
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        facts.append("parses as Python: yes")
        facts.append(f"functions defined: {len(funcs)}"
                     + (f" ({', '.join(f.name for f in funcs[:4])})" if funcs else ""))
        names = {f.name for f in funcs}
        facts.append(f"defines `{entry}`: {'yes' if entry in names else 'NO'}")
        if entry in names:
            node = next(f for f in funcs if f.name == entry)
            facts.append(f"its signature: `def {entry}({ast.unparse(node.args)}):`")
    except SyntaxError as exc:
        facts.append(f"parses as Python: **NO** ({exc.msg})")
    if code.lstrip().startswith("```"):
        facts.append("begins with a markdown fence: yes")

    facts.append(f"base tests passed: {failure['base_passed']}, "
                 f"plus tests passed: {failure['plus_passed']}")
    if failure.get("error"):
        facts.append(f"harness error: `{str(failure['error'])[:160]}`")
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--key", default="analysis/failures_key.json")
    parser.add_argument("--out", default="analysis/CODING_WORKSHEET.md")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    key = json.loads(Path(args.key).read_text())["rows"]
    by_id = {(f["model"], f["arm"], f["task_id"]): f
             for f in collect_failures(Path(args.results_dir))}

    lines = [
        "# Failure coding worksheet", "",
        "Twenty loops that failed. For each, decide **one primary code** and write it",
        "into the `primary_code` column of `failures_to_code.csv`, matching `row_id`.",
        "", "The arm each failure came from is deliberately withheld. Do not open",
        "`failures_key.json` or `coder_llm_sealed.json` until every row is coded.", "",
        "**C6 is not yours to assign.** It needs cross-arm information, so it is derived",
        "afterwards from the run data. Code the mechanism you can see in the loop itself.",
        "", "## Codebook", "",
    ]
    lines += [f"- **{c}** — {t}" for c, t in CODEBOOK.items() if c != "C6"]
    lines += ["", "_(C6 — oracle dependence — is derived, not coded.)_", "",
              "## How to read each entry", "",
              "The loop ran: original task → code → summary → code → … The **summary** shown",
              "is what the model wrote about its own passing code, and is the only",
              "description the failing cycle received. Ask: did the summary lose something",
              "the original said? Did it describe a different operation? Or was the summary",
              "faithful and the code simply wrong?", "", "---", ""]

    for row_id in sorted(key, key=int):
        meta = key[row_id]
        f = by_id[(meta["model"], meta["arm"], meta["task_id"])]
        lines += [f"## Row {row_id}", "",
                  f"Failed at cycle **{f['failed_at_cycle']}** (sustained "
                  f"{f['sustained']} cycle(s) before this one).", ""]
        lines += [f"- {d}" for d in diagnostics(f)]
        lines += [
            "", "**Original task**", "", "```python",
            f["initial_prompt"].strip(), "```", "",
            "**Summary the failing cycle received** (the model's own words)", "",
            "```python", (f["last_summary"].strip() or "(none)"), "```", "",
            "**Code that passed the previous cycle**", "", "```python",
            (f["last_passing_code"].strip() or "(none)"), "```", "",
            "**Code that failed**", "", "```python",
            (f["failing_code"].strip() or "(none)"), "```", "",
            f"> Row {row_id} primary code: ______", "", "---", "",
        ]

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(key)} failures, {out.stat().st_size // 1024} KB)")
    print("Code into analysis/failures_to_code.csv, then run:")
    print("  python scripts/score_coding.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
