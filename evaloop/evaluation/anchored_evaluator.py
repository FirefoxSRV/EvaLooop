"""Evaluator for the anchor-ablation runs.

Subclasses the framework's :class:`~evaloop.core.evaluator.EvaLoopEvaluator` and
overrides four seams, leaving the loop control flow upstream:

* ``_create_model``   - use an injected (cached) client rather than the factory.
* ``_create_tasks``   - the anchored task pair, carrying the chosen anchor and
                        the entry-point map the signature anchor needs.
* ``_load_dataset``   - the sampled task subset, read from exported metadata so
                        the generation stage needs no evalplus.
* ``_process_cycle_results`` - test a whole cycle's candidates in **one**
                        sandbox launch instead of one launch per task.

Results are written in the assignment data package's schema, so the same
analysis code reads both the published records and ours.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..core.evaluator import EvaLoopEvaluator

logger = logging.getLogger(__name__)

GENERATION_PROMPT = (
    "Generate Python code for the following task: \n{description}\n"
    "Return only the code, without explanations or comments."
)
SUMMARIZATION_PROMPT = (
    "Use one sentence to summarize the following code and start with "
    "'write a python function to':\n\n```\n{code}\n```\n\n```\n"
    "write a python function to\n```"
)


class AnchoredEvaluator(EvaLoopEvaluator):
    """Runs one (model, anchor) configuration over a fixed task subset."""

    def __init__(self, config, model, tester, dataset_meta: Dict[str, Any],
                 task_ids: List[str], anchor: str):
        """
        Args:
            config: An EvaluationConfig (supplies max_cycles, output_dir, name).
            model: A ready model client, normally wrapped in CachingLLM.
            tester: Anything with ``test_batch([(task_id, code)])``.
            dataset_meta: ``{task_id: {"prompt": ..., "entry_point": ...}}``.
            task_ids: The sampled subset, in evaluation order.
            anchor: Registered anchor name - ``assert``, ``masked_assert`` or
                ``signature``.
        """
        super().__init__(config)
        self._model = model
        self._tester = tester
        self._meta = dataset_meta
        self._task_ids = list(task_ids)
        self.anchor = anchor
        #: Per-cycle sandbox batch sizes, for the run report.
        self.cycle_sizes: List[int] = []

        missing = [t for t in self._task_ids if t not in self._meta]
        if missing:
            raise ValueError(
                f"{len(missing)} sampled task(s) absent from the dataset metadata, "
                f"first few: {missing[:5]}"
            )

    # --- injected components -------------------------------------------------

    def _create_model(self):
        """Return the injected client instead of building one from config."""
        return self._model

    def _create_tester(self, py_problems):
        """Return the injected (sandboxed) tester."""
        return self._tester

    def _create_tasks(self):
        """Build the generation/summarization pair for this configuration."""
        entry_points = {t: self._meta[t]["entry_point"] for t in self._task_ids}
        return [
            self.task_factory.create_task({
                "type": "stripped_code_generation",
                "language": "python",
                "prompt_template": GENERATION_PROMPT,
            }),
            self.task_factory.create_task({
                "type": "anchored_summarization",
                "prompt_template": SUMMARIZATION_PROMPT,
                "anchor": self.anchor,
                "entry_points": entry_points,
            }),
        ]

    def _load_dataset(self):
        """Return the sampled subset's prompts and a problem map keyed like MBPP+."""
        prompts = [{"task_id": t, "prompt": self._meta[t]["prompt"]} for t in self._task_ids]
        problems = {t: {"entry_point": self._meta[t]["entry_point"],
                        "prompt": self._meta[t]["prompt"]}
                    for t in self._task_ids}
        logger.info("Prepared %d sampled task(s) for anchor '%s'", len(prompts), self.anchor)
        return prompts, problems

    # --- batched testing -----------------------------------------------------

    def _process_cycle_results(self, cycle_output, model_result, tasks, py_problems,
                               tester, current_cycle) -> List[Dict[str, Any]]:
        """Test every candidate of this cycle in one sandbox launch.

        Upstream tests one task per call, which would mean one container launch
        per task per cycle. Semantics are otherwise unchanged: a cycle passes
        only if both the base and the extended tests pass, and only passing
        tasks continue.
        """
        results = cycle_output["cycle_results"]
        code_index = self._find_code_generation_task_index(tasks)
        if code_index is None:
            raise RuntimeError("No task in the loop produces testable code")
        code_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(code_index + 1)
        summary_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(len(tasks))

        candidates = [(r["task_id"], r[code_key]) for r in results if code_key in r]
        self.cycle_sizes.append(len(candidates))
        outcomes = {o["task_id"]: o for o in tester.test_batch(candidates)}

        next_prompts = []
        for result in results:
            task_id = result["task_id"]
            outcome = outcomes.get(task_id)

            if outcome is None:
                # No code was produced at all (extraction or API failure).
                result["test_results"] = {
                    "passed": False, "base_passed": False, "plus_passed": False,
                    "error": result.get(f"task_{code_index + 1}_error", "no code produced"),
                }
                result["success"] = False
            else:
                result["test_results"] = outcome
                result["success"] = bool(outcome.get("passed", False))

            prompt_result = next(r for r in model_result["prompt_results"]
                                 if r["task_id"] == task_id)
            prompt_result["cycles"].append(result)

            if result["success"]:
                prompt_result["successful_cycles"] = current_cycle
                if summary_key in result:
                    next_prompts.append({"task_id": task_id, "prompt": result[summary_key]})

        logger.info("Cycle %d: %d/%d passed", current_cycle, len(next_prompts), len(results))
        return next_prompts

    # --- results in the data package's schema --------------------------------

    def to_package_schema(self, model_result: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Convert a run's results to the shape of ``loops/<model_id>.json``.

        Keeps the full per-cycle code and summary, so failure analysis and the
        similarity judge read our runs exactly as they read the published ones.
        """
        tasks = []
        for prompt_result in model_result["prompt_results"]:
            cycles = []
            for cycle in prompt_result["cycles"]:
                test = cycle.get("test_results", {})
                cycles.append({
                    "cycle": cycle["cycle"],
                    "code": cycle.get("task_1_output", ""),
                    "summary": cycle.get("task_2_output"),
                    "passed": bool(test.get("passed", False)),
                    "base_passed": bool(test.get("base_passed", False)),
                    "plus_passed": bool(test.get("plus_passed", False)),
                    **({"error": test["error"]} if test.get("error") else {}),
                })
            tasks.append({
                "task_id": prompt_result["task_id"],
                "initial_prompt": prompt_result["initial_prompt"],
                "successful_cycles": prompt_result["successful_cycles"],
                "max_cycles_reached": prompt_result["max_cycles_reached"],
                "boundary_similarity": None,
                "boundary_similarity_cycle": None,
                "cycles": cycles,
            })

        return {
            "model_id": model_id,
            "name": model_id,
            "configuration": self.anchor,
            "max_cycles": self.config.max_cycles,
            "n_tasks": len(tasks),
            "tasks": tasks,
        }

    def run(self, model_id: str) -> Dict[str, Any]:
        """Run the configuration and return results in the package schema."""
        model = self._create_model()
        tasks = self._create_tasks()
        prompts, problems = self._load_dataset()
        tester = self._create_tester(problems)

        model_result = self._run_evaluation_cycles(model, tasks, prompts, problems, tester)
        payload = self.to_package_schema(model_result, model_id)

        payload["run_meta"] = {
            "anchor": self.anchor,
            "max_cycles": self.config.max_cycles,
            "cycle_batch_sizes": self.cycle_sizes,
            "sandbox_launches": getattr(tester, "launches", None),
            "anchor_fallbacks": getattr(tasks[-1].anchor, "fallbacks", {}),
            "usage": model.usage_report() if hasattr(model, "usage_report") else {},
        }
        return payload
