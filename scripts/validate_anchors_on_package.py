"""Validate the A0 anchor path against the published per-cycle records.

For every cycle in the data package this replays the summarization wrapper:
given the prompt the cycle started from and the model's raw summary text, the
``assert`` anchor must reproduce the ``summary`` string the package recorded -
i.e. exactly the text that became the next cycle's generation input.

That is a stronger check than comparing against upstream's source, because it
validates our anchor logic against 165k records of what the framework actually
produced when the leaderboard was generated.

The script also reports the two rates the ablation design depends on:
  * how often models echo a module-level assertion into their code
    (the behaviour ``stripped_code_generation`` neutralises), and
  * how many of the dataset's assert lines can have their oracle masked.

Usage:
    python scripts/validate_anchors_on_package.py --data-dir ../assignment2-data
    python scripts/validate_anchors_on_package.py --models gemini-2.0-flash-lite,gpt-5
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.anchors import mask_assert_oracle
from evaloop.tasks import AnchoredSummarizationTask, CodeSummarizationTask
from evaloop.tasks.stripped_code_generation import count_module_level_asserts

WRAPPER_OPEN = '"""'


def unwrap_summary(wrapped: str):
    """Recover the model's own summary text from a stored wrapped summary.

    Stored form is ``\"\"\"\\n{summary}\\n{anchor}\\n\"\"\"\\n``; the summary itself
    may span several lines.

    Returns:
        ``(summary_text, anchor_line)``, or None if the string is not wrapped.
    """
    lines = wrapped.splitlines()
    if len(lines) < 3 or lines[0].strip() != WRAPPER_OPEN or lines[-1].strip() != WRAPPER_OPEN:
        return None
    return "\n".join(lines[1:-2]), lines[-2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../assignment2-data")
    parser.add_argument("--models", help="Comma-separated model ids (default: all)")
    parser.add_argument("--max-mismatch-examples", type=int, default=5)
    args = parser.parse_args()

    # Parsing untrusted model code raises SyntaxWarning for stray escapes in
    # string literals; it is noise here, not a finding.
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    data_dir = Path(args.data_dir).resolve()
    loop_files = sorted((data_dir / "loops").glob("*.json"))
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        loop_files = [f for f in loop_files if f.stem in wanted]
    if not loop_files:
        print("No matching loop records found", file=sys.stderr)
        return 2

    upstream = CodeSummarizationTask({})
    anchored = AnchoredSummarizationTask({"anchor": "assert"})

    checked = mismatched = unwrappable = 0
    echo_cycles = total_cycles = 0
    examples = []
    per_model = []

    for loop_file in loop_files:
        data = json.loads(loop_file.read_text())
        m_checked = m_bad = m_echo = m_cycles = 0

        for task in data["tasks"]:
            cycle_input = task["initial_prompt"]
            for cycle in task["cycles"]:
                code = cycle.get("code") or ""
                m_cycles += 1
                if count_module_level_asserts(code):
                    m_echo += 1

                stored = cycle.get("summary")
                if not stored:
                    # Failing final cycles carry no summary; nothing to replay.
                    cycle_input = None
                    continue

                parts = unwrap_summary(stored)
                if parts is None:
                    unwrappable += 1
                    cycle_input = stored
                    continue
                summary_text, _ = parts

                context = {
                    "cycle_input": cycle_input,
                    "task_input": code,
                    "task_id": task["task_id"],
                    "cycle_number": cycle["cycle"],
                }
                ours = anchored.postprocess(summary_text, context)
                theirs = upstream.postprocess(summary_text, context)
                m_checked += 1

                if ours != stored or ours != theirs:
                    m_bad += 1
                    if len(examples) < args.max_mismatch_examples:
                        examples.append(
                            f"{data['model_id']} {task['task_id']} cycle {cycle['cycle']}\n"
                            f"    ours   : {ours!r}\n    stored : {stored!r}"
                        )
                # The recorded summary becomes the next cycle's input.
                cycle_input = stored

        checked += m_checked
        mismatched += m_bad
        echo_cycles += m_echo
        total_cycles += m_cycles
        per_model.append((data["model_id"], m_checked, m_bad, m_echo, m_cycles))

    # Maskability of the dataset's own assert lines.
    first = json.loads(loop_files[0].read_text())
    asserts = [t["initial_prompt"].splitlines()[-2] for t in first["tasks"]]
    maskable = sum(1 for a in asserts if mask_assert_oracle(a) is not None)

    print(f"\nReplayed {checked} recorded cycle transitions across {len(loop_files)} model(s).")
    print(f"  byte-identical to the recorded prompt : {checked - mismatched}")
    print(f"  mismatched                            : {mismatched}")
    if unwrappable:
        print(f"  summaries not in wrapper form (skipped): {unwrappable}")

    print(f"\nAssert echoing (module-level assert in generated code):")
    print(f"  {echo_cycles}/{total_cycles} cycles ({100 * echo_cycles / max(total_cycles, 1):.1f}%)")
    worst = sorted(per_model, key=lambda r: -r[3] / max(r[4], 1))[:5]
    for model_id, _, _, echo, cycles in worst:
        print(f"    {model_id:<36} {100 * echo / max(cycles, 1):5.1f}%")

    print(f"\nOracle maskability over the {len(asserts)} MBPP+ assert lines:")
    print(f"  maskable (assert with a comparison) : {maskable}")
    print(f"  unmaskable (bare truthiness)        : {len(asserts) - maskable}")

    if examples:
        print("\nMismatch examples:")
        for ex in examples:
            print(f"  - {ex}")

    if mismatched:
        print(f"\nFAIL: {mismatched} transition(s) did not reproduce.")
        return 1
    print("\nPASS: the assert anchor reproduces every recorded cycle transition byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
