"""End-to-end loop tests for AnchoredEvaluator, with no API and no execution.

A scripted model and a scripted tester let the whole generation <-> summarization
loop run offline, so the parts that are easy to get wrong - what crosses the
cycle boundary, when the loop stops, how results are shaped - are checked
without spending a call or executing untrusted code.
"""

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.evaluation.anchored_evaluator import AnchoredEvaluator
from evaloop.models.base import BaseLLM

TASK_ID = "Mbpp/11"
PROMPT = ('"""\nWrite a function to remove first and last occurrence of a given character.\n'
          'assert remove_Occ("hello","l") == "heo"\n"""\n')
META = {TASK_ID: {"prompt": PROMPT, "entry_point": "remove_Occ"}}
CODE = "def remove_Occ(s, ch):\n    return s\n"


class ScriptedModel(BaseLLM):
    """Returns code for generation prompts and a summary for summarization ones."""

    def __init__(self, code=CODE, summary="write a python function to remove occurrences."):
        super().__init__({"name": "scripted"})
        self.code, self.summary = code, summary
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("Generate Python code"):
            return f"```python\n{self.code}\n```"
        return self.summary

    def batch_generate(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]


class ScriptedTester:
    """Passes a fixed number of cycles, then fails."""

    def __init__(self, pass_cycles=3):
        self.pass_cycles = pass_cycles
        self.calls = 0
        self.launches = 0
        self.seen_code = []

    def test_batch(self, items):
        self.calls += 1
        self.launches += 1
        passed = self.calls <= self.pass_cycles
        self.seen_code.append([code for _tid, code in items])
        return [{"task_id": tid, "passed": passed,
                 "base_passed": passed, "plus_passed": passed} for tid, _c in items]


def build(anchor, model=None, tester=None, max_cycles=10, meta=None, task_ids=None, tmp_path=None):
    config = EvaluationConfig(
        model="scripted", max_cycles=max_cycles,
        output_dir=str(tmp_path), experiment_name=f"scripted__{anchor}",
    )
    return AnchoredEvaluator(
        config=config,
        model=model or ScriptedModel(),
        tester=tester or ScriptedTester(),
        dataset_meta=meta or META,
        task_ids=task_ids or [TASK_ID],
        anchor=anchor,
    )


class TestLoopControl:
    def test_loop_stops_at_the_first_failing_cycle(self, tmp_path):
        tester = ScriptedTester(pass_cycles=3)
        payload = build("assert", tester=tester, tmp_path=tmp_path).run("scripted")
        task = payload["tasks"][0]
        assert task["successful_cycles"] == 3
        assert len(task["cycles"]) == 4           # three passes plus the failure
        assert [c["passed"] for c in task["cycles"]] == [True, True, True, False]
        assert task["max_cycles_reached"] is False

    def test_a_task_that_never_fails_reaches_the_budget(self, tmp_path):
        tester = ScriptedTester(pass_cycles=99)
        payload = build("assert", tester=tester, max_cycles=4, tmp_path=tmp_path).run("scripted")
        task = payload["tasks"][0]
        assert task["successful_cycles"] == 4
        assert task["max_cycles_reached"] is True
        assert len(task["cycles"]) == 4

    def test_failing_cycle_one_yields_zero_sustained_cycles(self, tmp_path):
        tester = ScriptedTester(pass_cycles=0)
        payload = build("assert", tester=tester, tmp_path=tmp_path).run("scripted")
        assert payload["tasks"][0]["successful_cycles"] == 0
        assert len(payload["tasks"][0]["cycles"]) == 1

    def test_one_sandbox_launch_per_cycle_not_per_task(self, tmp_path):
        meta = {f"Mbpp/{i}": {"prompt": PROMPT, "entry_point": "remove_Occ"} for i in range(1, 6)}
        tester = ScriptedTester(pass_cycles=2)
        evaluator = build("assert", tester=tester, meta=meta,
                          task_ids=list(meta), tmp_path=tmp_path)
        evaluator.run("scripted")
        # 5 tasks x 3 cycles would be 15 launches if tested one at a time.
        assert tester.launches == 3
        assert evaluator.cycle_sizes == [5, 5, 5]


class TestAnchorPropagation:
    """What actually crosses the cycle boundary, cycle after cycle."""

    @pytest.mark.parametrize("anchor,expected", [
        ("assert", 'assert remove_Occ("hello","l") == "heo"'),
        ("masked_assert", "assert remove_Occ('hello', 'l') == ..."),
        ("signature", "def remove_Occ(s, ch):"),
    ])
    def test_each_anchor_appears_in_every_later_prompt(self, anchor, expected, tmp_path):
        model = ScriptedModel()
        evaluator = build(anchor, model=model, tester=ScriptedTester(pass_cycles=4),
                          tmp_path=tmp_path)
        payload = evaluator.run("scripted")

        summaries = [c["summary"] for c in payload["tasks"][0]["cycles"] if c["summary"]]
        assert summaries, "the loop produced no summaries"
        for summary in summaries:
            assert summary.splitlines()[-2] == expected

        # And the generation prompts of cycles 2+ carry it too.
        gen_prompts = [p for p in model.prompts if p.startswith("Generate Python code")]
        for prompt in gen_prompts[1:]:
            assert expected in prompt

    def test_the_oracle_only_survives_under_the_assert_anchor(self, tmp_path):
        for anchor, leaks in [("assert", True), ("masked_assert", False), ("signature", False)]:
            model = ScriptedModel()
            build(anchor, model=model, tester=ScriptedTester(pass_cycles=4),
                  tmp_path=tmp_path).run("scripted")
            later = [p for p in model.prompts if p.startswith("Generate Python code")][1:]
            assert later, anchor
            assert all(('== "heo"' in p) == leaks for p in later), anchor

    def test_the_signature_anchor_follows_the_models_own_renaming(self, tmp_path):
        renamed = "def remove_Occ(text, char):\n    return text\n"
        model = ScriptedModel(code=renamed)
        evaluator = build("signature", model=model, tester=ScriptedTester(pass_cycles=3),
                          tmp_path=tmp_path)
        payload = evaluator.run("scripted")
        summaries = [c["summary"] for c in payload["tasks"][0]["cycles"] if c["summary"]]
        assert all(s.splitlines()[-2] == "def remove_Occ(text, char):" for s in summaries)


class TestEchoStripping:
    def test_an_echoed_assertion_never_reaches_the_tester(self, tmp_path):
        echoing = CODE + '\nassert remove_Occ("hello","l") == "heo"\n'
        tester = ScriptedTester(pass_cycles=2)
        build("masked_assert", model=ScriptedModel(code=echoing), tester=tester,
              tmp_path=tmp_path).run("scripted")
        for batch in tester.seen_code:
            for code in batch:
                assert "assert" not in code


class TestResultsSchema:
    def test_payload_matches_the_data_package_shape(self, tmp_path):
        payload = build("signature", tester=ScriptedTester(pass_cycles=2),
                        tmp_path=tmp_path).run("scripted")
        assert {"model_id", "configuration", "max_cycles", "n_tasks", "tasks"} <= set(payload)
        assert payload["configuration"] == "signature"

        task = payload["tasks"][0]
        assert {"task_id", "initial_prompt", "successful_cycles", "max_cycles_reached",
                "boundary_similarity", "cycles"} <= set(task)
        cycle = task["cycles"][0]
        assert {"cycle", "code", "summary", "passed", "base_passed", "plus_passed"} <= set(cycle)

    def test_run_meta_records_what_the_report_needs(self, tmp_path):
        payload = build("assert", tester=ScriptedTester(pass_cycles=2),
                        tmp_path=tmp_path).run("scripted")
        meta = payload["run_meta"]
        assert meta["anchor"] == "assert"
        assert meta["cycle_batch_sizes"] == [1, 1, 1]
        assert meta["sandbox_launches"] == 3

    def test_metrics_module_consumes_the_result_directly(self, tmp_path):
        """The framework's own metrics must read our results with no adapter."""
        from evaloop.analysis.metrics import asl_pow, pass_at_k
        payload = build("assert", tester=ScriptedTester(pass_cycles=3),
                        tmp_path=tmp_path).run("scripted")
        cycles = [t["successful_cycles"] for t in payload["tasks"]]
        assert pass_at_k(cycles, 1) == 1
        assert asl_pow(cycles, payload["max_cycles"]) == pytest.approx(0.9)


class TestGuards:
    def test_a_task_missing_from_the_metadata_is_rejected_up_front(self, tmp_path):
        with pytest.raises(ValueError, match="absent from the dataset metadata"):
            build("assert", task_ids=[TASK_ID, "Mbpp/9999"], tmp_path=tmp_path)


class TestGenerationFailureGuard:
    """A failed API call must never be scored as a model loop failure."""

    def test_an_api_error_string_aborts_instead_of_being_tested(self, tmp_path):
        from evaloop.evaluation.anchored_evaluator import GenerationFailure
        from evaloop.models.gemini import ERROR_PREFIX

        class ThrottledModel(ScriptedModel):
            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                if prompt.startswith("Generate Python code"):
                    return f"{ERROR_PREFIX}: 429 RESOURCE_EXHAUSTED"
                return self.summary

        tester = ScriptedTester(pass_cycles=99)
        evaluator = build("assert", model=ThrottledModel(), tester=tester, tmp_path=tmp_path)
        with pytest.raises(GenerationFailure, match="exhausting retries"):
            evaluator.run("scripted")
        # The point: the error string never reached the tester.
        assert tester.calls == 0

    def test_the_abort_message_says_how_to_resume(self, tmp_path):
        from evaloop.evaluation.anchored_evaluator import GenerationFailure
        from evaloop.models.gemini import ERROR_PREFIX

        class ThrottledModel(ScriptedModel):
            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return (f"{ERROR_PREFIX}: quota"
                        if prompt.startswith("Generate Python code") else self.summary)

        with pytest.raises(GenerationFailure) as excinfo:
            build("assert", model=ThrottledModel(), tmp_path=tmp_path).run("scripted")
        assert "cache resumes" in str(excinfo.value)
        assert "NOT written" in str(excinfo.value)


class TestBrokenGenerationTolerance:
    """A few unrecoverable calls should cost those tasks, not the whole run."""

    def _model_failing(self, failing_ids):
        from evaloop.models.gemini import ERROR_PREFIX

        class PartialModel(ScriptedModel):
            def generate(self, prompt, **kwargs):
                # Overridden rather than only batch_generate, because a
                # single-task cycle takes the sequential path.
                self.prompts.append(prompt)
                if not prompt.startswith("Generate Python code"):
                    return self.summary
                if any(t in prompt for t in failing_ids):
                    return f"{ERROR_PREFIX}: 429 exhausted"
                return f"```python\n{self.code}\n```"

            def batch_generate(self, prompts, **kwargs):
                return [self.generate(p, **kwargs) for p in prompts]
        return PartialModel()

    def test_a_small_share_is_dropped_and_the_run_continues(self, tmp_path):
        # 100 tasks, one of which never generates: 1% is under the 2% tolerance.
        meta = {f"Mbpp/{i}": {"prompt": PROMPT.replace("remove_Occ", f"remove_Occ"),
                              "entry_point": "remove_Occ"} for i in range(1, 101)}
        evaluator = build("assert", model=self._model_failing(["Mbpp/7\n"]),
                          tester=ScriptedTester(pass_cycles=2), meta=meta,
                          task_ids=list(meta), tmp_path=tmp_path)
        payload = evaluator.run("scripted")
        # The run completed rather than raising.
        assert payload["n_tasks"] == 100

    def test_dropped_tasks_are_recorded_for_the_analysis(self, tmp_path):
        meta = {f"Mbpp/{i}": {"prompt": PROMPT, "entry_point": "remove_Occ"}
                for i in range(1, 101)}
        evaluator = build("assert", model=self._model_failing([]),
                          tester=ScriptedTester(pass_cycles=2), meta=meta,
                          task_ids=list(meta), tmp_path=tmp_path)
        payload = evaluator.run("scripted")
        assert "dropped_tasks" in payload["run_meta"]

    def test_above_the_tolerance_the_run_still_aborts(self, tmp_path):
        from evaloop.evaluation.anchored_evaluator import GenerationFailure
        # One task, one failure: 100% is far above tolerance.
        with pytest.raises(GenerationFailure, match="tolerance"):
            build("assert", model=self._model_failing(["assert"]),
                  tester=ScriptedTester(pass_cycles=5), tmp_path=tmp_path).run("scripted")
