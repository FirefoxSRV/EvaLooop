"""Score the failure boundaries of the ablation runs with an LLM judge.

Reuses the framework's own judge (``evaloop.analysis.similarity``), which already
carries the paper's byte-identical prompt, the score parser, the boundary-pair
definition and a hash-validated score cache. This script only adapts our
results to that machinery and swaps in an accessible judge model.

Why a judge at all: the ablation's pre-registered H3 says that if removing the
leaked oracle is what breaks loops, then failures under the ablated anchors are
preceded by *less* semantic drift than baseline failures - the specification was
still faithful, the model simply no longer had the answer. Boundary similarity
is how that gets measured, and it is the prediction that separates "the summary
wandered" from "the oracle was load-bearing".

Judge substitution is reported, not hidden: the published scores were produced
with gpt-4-turbo-2024-04-09, which this project cannot reach, so absolute scores
are **not** comparable to the data package's. Within this experiment all arms are
judged by the same model with the same prompt, so the arm-to-arm comparison -
which is what H3 needs - is sound.

Usage:
    python scripts/judge_similarity.py --results-dir results --judge gemma-4-26b-a4b-it
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.analysis.similarity import (
    DEFAULT_JUDGE_PROMPT,
    BoundaryPair,
    SemanticSimilarityJudge,
    collect_similarity_scores,
)
from evaloop.models.gemini import GeminiLLM

# Judge used for the published scores; recorded for the write-up's comparability note.
PUBLISHED_JUDGE = "gpt-4-turbo-2024-04-09"


def boundary_pairs(payload: dict):
    """Build judge inputs from a package-schema run.

    Mirrors ``similarity.extract_boundary_pair``, which reads the framework's
    internal ``prompt_results`` shape; our runs are stored in the data package's
    schema, where a cycle carries ``code`` and ``summary`` directly.

    A pair exists only for a task that sustained 1..M-1 loops: one that failed at
    cycle 1 produced no summary, and one that never failed has no boundary.
    """
    max_cycles = payload["max_cycles"]
    pairs = []
    for task in payload["tasks"]:
        sustained = task["successful_cycles"]
        if not 1 <= sustained < max_cycles:
            continue
        cycles = task["cycles"]
        if len(cycles) <= sustained or cycles[sustained].get("passed") is not False:
            continue

        last_success = cycles[sustained - 1]
        prompt1 = (task["initial_prompt"] if sustained == 1
                   else cycles[sustained - 2].get("summary") or "")
        pairs.append(BoundaryPair(
            task_id=str(task["task_id"]),
            sustained_loops=sustained,
            prompt1=prompt1,
            code1=last_success.get("code") or "",
            prompt2=last_success.get("summary") or "",
            code2=cycles[sustained].get("code") or "",
        ))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--judge", default="gemma-4-26b-a4b-it",
                        help="Judge model id (an accessible substitute for "
                             f"{PUBLISHED_JUDGE})")
    parser.add_argument("--thinking-level", default="minimal")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count the pairs that would be judged and stop")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "tasks" in payload and "configuration" in payload:
            runs.append((path, payload))
    if not runs:
        print(f"No run files in {results_dir}", file=sys.stderr)
        return 2

    all_pairs = {path: boundary_pairs(payload) for path, payload in runs}
    total = sum(len(p) for p in all_pairs.values())
    print(f"\nFound {total} failure boundar{'y' if total == 1 else 'ies'} "
          f"across {len(runs)} run(s):")
    for path, pairs in all_pairs.items():
        print(f"  {path.name:<48} {len(pairs):>4}")
    if args.dry_run or total == 0:
        return 0

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("No GEMINI_API_KEY in the environment.")

    judge_llm = GeminiLLM({
        "name": args.judge, "type": "gemini", "model_id": args.judge,
        "temperature": 0.0, "top_p": 1.0,
        # The paper's judge answers with a bare number; 10 tokens is its budget.
        "max_tokens": 10,
        "thinking_level": args.thinking_level,
        "seed": 20251013,
        "concurrency": args.concurrency,
    })
    judge = SemanticSimilarityJudge(judge_llm, name=args.judge,
                                    prompt_template=DEFAULT_JUDGE_PROMPT)
    print(f"\nJudge: {args.judge} (temperature 0, max_tokens 10, "
          f"thinking={args.thinking_level})")
    print(f"Prompt: the framework's DEFAULT_JUDGE_PROMPT, byte-identical to the paper's\n")

    for path, payload in runs:
        pairs = all_pairs[path]
        if not pairs:
            continue
        cache_path = Path(args.cache_dir) / f"{path.stem}__judge_{args.judge}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        scores = collect_similarity_scores(
            pairs=pairs, cache_path=cache_path,
            judge_provider=lambda: judge, prompt_template=DEFAULT_JUDGE_PROMPT,
        )

        # Write the scores into the run file, in the data package's field names.
        by_task = {p.task_id: (scores.get(p.key), p.sustained_loops) for p in pairs}
        for task in payload["tasks"]:
            score, cycle = by_task.get(str(task["task_id"]), (None, None))
            task["boundary_similarity"] = score
            task["boundary_similarity_cycle"] = cycle
        payload.setdefault("run_meta", {})["judge"] = {
            "model": args.judge,
            "published_judge": PUBLISHED_JUDGE,
            "comparable_to_published": False,
            "note": ("Absolute scores are not comparable to the data package, which "
                     f"used {PUBLISHED_JUDGE}. All arms here share one judge and one "
                     "prompt, so arm-to-arm comparisons are valid."),
            "temperature": 0.0, "max_tokens": 10,
            "thinking_level": args.thinking_level,
            "usage": judge_llm.usage_report(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

        usable = [s for s in scores.values() if s is not None]
        mean = sum(usable) / len(usable) if usable else float("nan")
        print(f"{path.name:<48} scored {len(usable)}/{len(pairs)}  mean={mean:.3f}")

    usage = judge_llm.usage_report()
    print(f"\nJudge usage: calls={usage['calls']}  in={usage['prompt_tokens']} tok  "
          f"out={usage['output_tokens']} tok  throttled={usage['rate_limited']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
