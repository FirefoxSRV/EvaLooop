"""Run one or more (model, anchor) configurations of the anchor-leak ablation.

Pipeline, two stages with different trust levels:

  generation  talks to the model API; never executes model output
  testing     executes model output in a container with --network none

Every model call is cached to ``cache/<model>.jsonl``, keyed by prompt plus all
decoding settings. Because cycle 1 is identical across the three anchors, the
second and third configuration of a model replay its cycle-1 calls from cache:
pass@1 is then equal across arms by construction, not by assumption. Runs are
resumable for the same reason.

Usage:
    # one configuration
    python scripts/run_anchored_experiment.py --model gemini-3.5-flash-lite --anchor assert

    # the full ablation for one model, baseline first so later arms reuse cycle 1
    python scripts/run_anchored_experiment.py --model gemini-3.5-flash-lite \
        --anchor assert,masked_assert,signature

    # smoke test without a container (untrusted code runs in-process)
    python scripts/run_anchored_experiment.py --model gemini-3.5-flash-lite \
        --anchor assert --limit 5 --tester local
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaloop.core.config import EvaluationConfig
from evaloop.evaluation.anchored_evaluator import AnchoredEvaluator
from evaloop.evaluation.sandbox_tester import (
    DockerSandboxTester,
    LocalTester,
    SandboxUnavailable,
)
from evaloop.models.cached import CachingLLM
from evaloop.models.factory import ModelFactory

ANCHORS = ("assert", "masked_assert", "signature")


def build_model(model_id: str, args) -> CachingLLM:
    """Create the model client and wrap it in the shared per-model call cache."""
    model_config = {
        "name": model_id,
        "type": "gemini",
        "model_id": model_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "requests_per_minute": args.requests_per_minute or None,
    }
    # Only one thinking knob is valid per model generation; see models/gemini.py.
    if args.thinking_level:
        model_config["thinking_level"] = args.thinking_level
    elif args.thinking_budget is not None:
        model_config["thinking_budget"] = args.thinking_budget

    if not os.environ.get("GEMINI_API_KEY") and not args.use_vertex:
        raise SystemExit(
            "No GEMINI_API_KEY in the environment. Export it, or pass --use-vertex "
            "to authenticate to Vertex AI with application default credentials."
        )
    if args.use_vertex:
        model_config["use_vertex"] = True

    inner = ModelFactory().create_model(model_config)
    # Cache is keyed per model, not per configuration: that is what lets the
    # three anchors share cycle 1.
    return CachingLLM(inner, str(Path(args.cache_dir) / f"{model_id}.jsonl"))


def build_tester(args):
    """Create the sandboxed tester, or the in-process one when asked."""
    if args.tester == "local":
        return LocalTester()
    tester = DockerSandboxTester(
        image=args.image,
        repo_root=Path(__file__).resolve().parents[1],
        memory=args.memory,
        cpus=args.cpus,
    )
    try:
        tester.preflight()
    except SandboxUnavailable as exc:
        raise SystemExit(f"\nSandbox unavailable:\n  {exc}\n")
    return tester


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model id (one per invocation)")
    parser.add_argument("--anchor", default="assert",
                        help=f"Comma-separated anchors to run, in order. Choices: {', '.join(ANCHORS)}")
    parser.add_argument("--task-ids", default="configs/task_ids.json")
    parser.add_argument("--dataset-meta", default="configs/mbpp_plus_meta.json")
    parser.add_argument("--limit", type=int, help="Use only the first N tasks (smoke tests)")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cache-dir", default="cache")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20251013)
    parser.add_argument("--thinking-level", help="Gemini 3.x: minimal|low|medium|high")
    parser.add_argument("--thinking-budget", type=int, help="Gemini 2.x: 0 disables thinking")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel in-flight requests. Raising this past the "
                             "provider's requests-per-minute cap only adds retries.")
    parser.add_argument("--requests-per-minute", type=int, default=28,
                        help="Pace requests under the provider's per-minute quota "
                             "instead of discovering it through 429s. The free tier "
                             "allows 30/min per model; 28 leaves headroom. 0 disables.")
    parser.add_argument("--use-vertex", action="store_true",
                        help="Authenticate to Vertex AI with application default "
                             "credentials instead of using GEMINI_API_KEY")

    parser.add_argument("--tester", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="evaloop-test:latest")
    parser.add_argument("--memory", default="2g")
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    anchors = [a.strip() for a in args.anchor.split(",") if a.strip()]
    unknown = [a for a in anchors if a not in ANCHORS]
    if unknown:
        raise SystemExit(f"Unknown anchor(s): {unknown}. Choices: {', '.join(ANCHORS)}")

    meta_path, ids_path = Path(args.dataset_meta), Path(args.task_ids)
    for path, hint in [(meta_path, "scripts/export_dataset_meta.py"),
                       (ids_path, "scripts/sample_tasks.py")]:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Generate it with {hint}.")

    dataset_meta = json.loads(meta_path.read_text())["tasks"]
    task_ids = json.loads(ids_path.read_text())["task_ids"]
    if args.limit:
        task_ids = task_ids[:args.limit]

    model = build_model(args.model, args)
    tester = build_tester(args)
    print(f"\nModel   : {args.model}\nTester  : {tester}\n"
          f"Tasks   : {len(task_ids)}\nAnchors : {', '.join(anchors)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for anchor in anchors:
        config = EvaluationConfig(
            model=args.model,
            max_cycles=args.max_cycles,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            seed=args.seed,
            output_dir=str(output_dir),
            experiment_name=f"{args.model}__{anchor}",
        )
        evaluator = AnchoredEvaluator(
            config=config, model=model, tester=tester,
            dataset_meta=dataset_meta, task_ids=task_ids, anchor=anchor,
        )

        print(f"=== {args.model} / anchor={anchor} ===")
        started = time.time()
        payload = evaluator.run(args.model)
        payload["run_meta"]["wall_seconds"] = round(time.time() - started, 1)
        payload["run_meta"]["cache"] = model.cache_report()

        out_file = output_dir / f"{args.model}__{anchor}.json"
        out_file.write_text(json.dumps(payload, indent=2) + "\n")

        cycles = [t["successful_cycles"] for t in payload["tasks"]]
        n = len(cycles)
        line = {
            "anchor": anchor,
            "pass@1": sum(1 for c in cycles if c >= 1) / n,
            "mean_cycles": sum(cycles) / n,
            "survived_all": sum(1 for c in cycles if c == args.max_cycles),
            "mid_failures": sum(1 for c in cycles if 1 <= c < args.max_cycles),
            "wall_s": payload["run_meta"]["wall_seconds"],
            "cache_hit_rate": payload["run_meta"]["cache"]["hit_rate"],
        }
        summary.append(line)
        print(f"  pass@1={line['pass@1']:.3f}  mean_cycles={line['mean_cycles']:.2f}  "
              f"survived_all={line['survived_all']}  mid_failures={line['mid_failures']}  "
              f"cache_hits={line['cache_hit_rate']:.1%}  {line['wall_s']:.0f}s")
        print(f"  -> {out_file}\n")

    print("=== run summary ===")
    for line in summary:
        print(f"  {line['anchor']:<15} pass@1={line['pass@1']:.3f}  "
              f"mean_cycles={line['mean_cycles']:.2f}  mid_failures={line['mid_failures']}")
    usage = model.usage_report()
    print(f"\nAPI calls={usage.get('calls')}  in={usage.get('prompt_tokens')} tok  "
          f"out={usage.get('output_tokens')} tok  retries={usage.get('retries')}  "
          f"throttled={usage.get('rate_limited')}")
    print(f"Cache: {usage.get('cache')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
