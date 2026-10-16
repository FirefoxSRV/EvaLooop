"""Check the submission against the assignment text, mechanically.

Every item below is a requirement from Instruction.docx, checked against the
files that would actually be submitted rather than against memory. Exits
non-zero if anything required is missing, so this can gate the ZIP build.

Usage:
    python scripts/verify_submission.py
    python scripts/verify_submission.py --zip ../sviswes_A3.zip
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEADER = ["model", "configuration", "n_tasks", "success_rate", "asl_score"]

results = []


def check(section, requirement, ok, detail=""):
    results.append((section, requirement, bool(ok), detail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="Also verify the contents of a built ZIP")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    report = (ROOT / "report.md").read_text() if (ROOT / "report.md").exists() else ""
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    res_dir = ROOT / args.results_dir

    # ---- 1. Read the code -------------------------------------------------
    pipeline_bits = {
        "where prompts are built": r"create_prompt|renders",
        "how output is extracted": r"triple-backtick",
        "how code is tested": r"untrusted_check",
        "summary -> next prompt": r"becomes the next cycle's description|re-appended",
        "how ASL is computed": r"ASL = ",
    }
    for name, pattern in pipeline_bits.items():
        check("1 Read the code", f"report explains {name}",
              re.search(pattern, report) is not None)
    check("1 Read the code", "report says what was modified and why",
          "What I changed" in report and "five existing source files" in report.lower())

    # ---- 2. Fork and extend ----------------------------------------------
    check("2 Fork/extend", "public fork linked in README",
          "github.com/FirefoxSRV/EvaLooop" in readme)
    for name, pattern in {
        "prompts specified": r"Generate Python code for the following task",
        "what is carried between cycles": r"Carries|carried",
        "failure condition": r"passing only if both pass|a cycle passes only if both",
        "signature stability": r"[Ss]ignature stability",
    }.items():
        check("2 Fork/extend", f"loop spec: {name}", re.search(pattern, report) is not None)
    check("2 Fork/extend", "new code lives in new files",
          (ROOT / "evaloop/anchors").is_dir())
    check("2 Fork/extend", "README explains upstream modifications",
          "upstream" in readme.lower() and "cycle.py" in readme)

    # ---- 3. Run the experiment -------------------------------------------
    runs = {}
    for path in sorted(res_dir.glob("*__*.json")):
        payload = json.loads(path.read_text())
        runs[(payload["model_id"], payload["configuration"])] = payload
    models = {m for m, _ in runs}
    check("3 Experiment", "at least 2 models evaluated", len(models) >= 2,
          f"{len(models)} models")
    check("3 Experiment", "max 10 cycles",
          all(p["max_cycles"] == 10 for p in runs.values()))
    check("3 Experiment", "at least 100 tasks",
          all(p["n_tasks"] >= 100 for p in runs.values()),
          f"n_tasks={sorted({p['n_tasks'] for p in runs.values()})}")
    check("3 Experiment", "greedy decoding recorded in the runs",
          any((p.get("run_meta", {}).get("usage", {}).get("settings", {}) or {})
              .get("temperature") == 0.0 for p in runs.values()))
    check("3 Experiment", "sampling documented",
          (ROOT / "configs/task_ids.json").exists()
          and "provenance" in json.loads((ROOT / "configs/task_ids.json").read_text()))
    check("3 Experiment", "baseline arm present for every model",
          all((m, "assert") in runs for m in models))

    # ---- 4. Analyze -------------------------------------------------------
    check("4 Analysis", "ASL and success rate reported",
          "ASL" in report and "pass@1" in report)
    check("4 Analysis", "ranking question answered",
          "Does the ranking change" in report)
    coding = ROOT / "analysis/coding_summary.json"
    n_coded = json.loads(coding.read_text())["n_coded"] if coding.exists() else 0
    check("4 Analysis", "at least 20 failed loops inspected", n_coded >= 20,
          f"{n_coded} coded")
    check("4 Analysis", "failure modes compared against the baseline",
          "modes differ" in report.lower())
    check("4 Analysis", "says what the extension reveals",
          "reveals that the standard loop does not" in report)
    check("4 Analysis", "representative case the extension broke",
          "loop the extension broke" in report.lower())
    check("4 Analysis", "representative case it did not",
          "loop it did not" in report.lower())
    for threat in ["contaminat", "[Tt]est-suite strength", "subset", "famil"]:
        check("4 Analysis", f"threat covered: {threat.strip('[]Tt')}",
              re.search(threat, report) is not None)

    # ---- Common requirements ---------------------------------------------
    check("Common", "LLM usage disclosed with model name",
          "Claude Opus 4" in report)
    check("Common", "LLM usage: conversion to a decision",
          "Conversion to a decision" in report or "conversion to a decision" in report)
    check("Common", "LLM usage: cost reported", "Cost." in report or "*Cost.*" in report)
    check("Common", "sandboxed execution described",
          "--network none" in report)

    # ---- Deliverables ------------------------------------------------------
    pdf = ROOT / "report.pdf"
    pages = None
    if pdf.exists():
        raw = pdf.read_bytes()
        counts = [int(m.group(1)) for m in re.finditer(rb"/Count\s+(\d+)", raw)]
        pages = max(counts) if counts else None
    check("Deliverables", "report.pdf exists and is at most 3 pages",
          pdf.exists() and (pages or 99) <= 3, f"{pages} pages")
    check("Deliverables", "patch against upstream can be produced",
          subprocess.run(["git", "rev-parse", "--verify", "upstream/dev"],
                         cwd=ROOT, capture_output=True).returncode == 0)
    check("Deliverables", "results JSON for every (model, configuration)",
          len(runs) >= 2, f"{len(runs)} runs")
    check("Deliverables", "results also in the framework's native shape",
          (res_dir / "native").is_dir()
          and len(list((res_dir / "native").glob("*.json"))) == len(runs))
    summary = res_dir / "summary.csv"
    header_ok = rows = 0
    if summary.exists():
        with summary.open() as handle:
            reader = list(csv.reader(handle))
        header_ok = reader[0] == EXPECTED_HEADER
        rows = len(reader) - 1
    check("Deliverables", "summary.csv header is byte-exact", header_ok)
    check("Deliverables", "summary.csv has one row per run", rows == len(runs),
          f"{rows} rows for {len(runs)} runs")
    check("Deliverables", "README explains how to reproduce each number",
          "number to the command" in readme or "Which command produces" in readme)
    check("Deliverables", "data package NOT redistributed",
          not (ROOT / "assignment2-data").exists()
          and not list(ROOT.glob("**/leaderboard.json")))

    # ---- Style requested by the author ------------------------------------
    check("Style", "no em dashes in the report", "—" not in report)
    check("Style", "written in the first person",
          len(re.findall(r"\bI [a-z]", report)) >= 10,
          f"{len(re.findall(r'\bI [a-z]', report))} first-person verbs")

    # ---- ZIP ---------------------------------------------------------------
    if args.zip:
        zpath = Path(args.zip)
        if zpath.exists():
            names = zipfile.ZipFile(zpath).namelist()
            check("ZIP", "report.pdf included", any(n.endswith("report.pdf") for n in names))
            check("ZIP", "summary.csv included", any(n.endswith("summary.csv") for n in names))
            check("ZIP", "patch included", any(n.endswith(".patch") for n in names))
            check("ZIP", "README included", any(n.endswith("README.md") for n in names))
            check("ZIP", "all run files included",
                  sum(1 for n in names if re.search(r"results/[^/]+__[^/]+\.json$", n)) == len(runs))
            check("ZIP", "data package not bundled",
                  not any("leaderboard.json" in n or "/loops/" in n for n in names))
        else:
            check("ZIP", "zip exists", False, str(zpath))

    # ---- Report ------------------------------------------------------------
    width = max(len(r[1]) for r in results) + 2
    current = None
    failed = 0
    for section, requirement, ok, detail in results:
        if section != current:
            print(f"\n{section}")
            current = section
        mark = "ok  " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {requirement:<{width}}{detail}")

    total = len(results)
    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        print(f"{failed} requirement(s) unmet.")
        return 1
    print("Every checked requirement from the assignment is satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
