"""Tests for the analysis scripts.

Two kinds of check:

* Known-answer tests for the paired statistics, built from synthetic runs where
  the right answer is known by construction. These always run.
* A reproduction test against the published leaderboard, which needs the
  assignment data package. It skips when the package is absent, since the
  package is deliberately not redistributed with this repo.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stats import load_runs, paired_bootstrap_delta, spearman, wilcoxon_signed_rank  # noqa: E402
from summarize_runs import metrics_for  # noqa: E402

DATA_DIR = REPO_ROOT.parent / "assignment2-data"
needs_package = pytest.mark.skipif(
    not (DATA_DIR / "leaderboard.json").exists(),
    reason="assignment data package not present (it is not redistributed)",
)


def make_run(tmp_path, model, arm, sustained_by_task, max_cycles=10):
    """Write a minimal package-schema run file."""
    tasks = []
    for task_id, sustained in sustained_by_task.items():
        cycles = [{"cycle": i + 1, "code": "def f():\n    return 1\n",
                   "summary": '"""\ns\nassert f() == 1\n"""\n',
                   "passed": True, "base_passed": True, "plus_passed": True}
                  for i in range(sustained)]
        if sustained < max_cycles:
            cycles.append({"cycle": sustained + 1, "code": "def f():\n    return 2\n",
                           "summary": None, "passed": False,
                           "base_passed": False, "plus_passed": False})
        tasks.append({"task_id": task_id, "initial_prompt": '"""\nt\nassert f() == 1\n"""\n',
                      "successful_cycles": sustained,
                      "max_cycles_reached": sustained == max_cycles,
                      "boundary_similarity": None, "cycles": cycles})
    payload = {"model_id": model, "name": model, "configuration": arm,
               "max_cycles": max_cycles, "n_tasks": len(tasks), "tasks": tasks}
    (tmp_path / f"{model}__{arm}.json").write_text(json.dumps(payload))
    return payload


class TestMetrics:
    def test_all_tasks_surviving_gives_the_maximum_asl(self, tmp_path):
        payload = make_run(tmp_path, "m", "assert", {f"t{i}": 10 for i in range(20)})
        row = metrics_for(payload)
        assert row["success_rate"] == 1.0
        assert row["asl_score"] == pytest.approx(10.0)
        assert row["survived_all"] == 20
        assert row["mid_failures"] == 0

    def test_all_tasks_failing_at_cycle_one_gives_zero(self, tmp_path):
        payload = make_run(tmp_path, "m", "assert", {f"t{i}": 0 for i in range(20)})
        row = metrics_for(payload)
        assert row["success_rate"] == 0.0
        assert row["asl_score"] == 0.0
        assert row["failed_at_1"] == 20

    def test_asl_is_quadratic_in_sustained_cycles(self, tmp_path):
        # 10 tasks at 5 cycles: sum(i^2 * n) / (T*M) = 25*10 / (10*10) = 2.5
        payload = make_run(tmp_path, "m", "assert", {f"t{i}": 5 for i in range(10)})
        assert metrics_for(payload)["asl_score"] == pytest.approx(2.5)

    def test_asl_bias_divides_out_one_shot_accuracy(self, tmp_path):
        # Half the tasks never start; the survivors all reach the budget.
        sustained = {f"t{i}": (10 if i < 10 else 0) for i in range(20)}
        row = metrics_for(make_run(tmp_path, "m", "assert", sustained))
        assert row["success_rate"] == pytest.approx(0.5)
        assert row["asl_score"] == pytest.approx(5.0)
        # Among tasks that passed cycle 1, robustness is perfect.
        assert row["asl_bias"] == pytest.approx(10.0)

    def test_echo_rate_counts_module_level_asserts_only(self, tmp_path):
        payload = make_run(tmp_path, "m", "assert", {"t0": 2})
        payload["tasks"][0]["cycles"][0]["code"] = "def f():\n    return 1\nassert f() == 1\n"
        payload["tasks"][0]["cycles"][1]["code"] = "def f():\n    assert True\n    return 1\n"
        row = metrics_for(payload)
        # One of the three recorded cycles has a module-level assert.
        assert row["assert_echo_rate"] == pytest.approx(1 / 3)


class TestPairedStatistics:
    def test_a_uniform_degradation_is_detected_in_the_right_direction(self):
        base = {f"t{i}": 10 for i in range(100)}
        worse = {f"t{i}": 4 for i in range(100)}
        delta, lo, hi, n = paired_bootstrap_delta(base, worse, 10, 2000, 1)
        assert n == 100
        # 10 -> 4 sustained: ASL falls from 10.0 to 1.6
        assert delta == pytest.approx(-8.4, abs=1e-6)
        assert hi < 0, "the CI must exclude zero for a uniform degradation"

        _w, _z, p, counts = wilcoxon_signed_rank(base, worse)
        assert p < 1e-10
        assert counts == {"worse_under_other": 100, "better_under_other": 0, "unchanged": 0}

    def test_identical_arms_are_not_significant_and_have_a_zero_delta(self):
        base = {f"t{i}": (i % 11) for i in range(100)}
        delta, lo, hi, _n = paired_bootstrap_delta(base, dict(base), 10, 2000, 1)
        assert delta == pytest.approx(0.0)
        assert lo == hi == pytest.approx(0.0)
        _w, _z, p, counts = wilcoxon_signed_rank(base, dict(base))
        assert p == 1.0
        assert counts["unchanged"] == 100

    def test_a_bigger_degradation_gives_a_bigger_drop(self):
        base = {f"t{i}": 10 for i in range(60)}
        mild = {f"t{i}": 7 for i in range(60)}
        severe = {f"t{i}": 3 for i in range(60)}
        mild_delta, *_ = paired_bootstrap_delta(base, mild, 10, 1000, 1)
        severe_delta, *_ = paired_bootstrap_delta(base, severe, 10, 1000, 1)
        assert severe_delta < mild_delta < 0

    def test_an_improvement_is_reported_as_an_improvement(self):
        base = {f"t{i}": 3 for i in range(50)}
        better = {f"t{i}": 8 for i in range(50)}
        delta, lo, _hi, _n = paired_bootstrap_delta(base, better, 10, 1000, 1)
        assert delta > 0 and lo > 0
        _w, _z, _p, counts = wilcoxon_signed_rank(base, better)
        assert counts["better_under_other"] == 50

    def test_only_tasks_present_in_both_arms_are_paired(self):
        base = {"a": 10, "b": 10, "c": 10}
        other = {"a": 5, "b": 5}
        _delta, _lo, _hi, n = paired_bootstrap_delta(base, other, 10, 500, 1)
        assert n == 2

    def test_load_runs_reads_our_schema_and_ignores_other_json(self, tmp_path):
        make_run(tmp_path, "m", "assert", {"t0": 3})
        make_run(tmp_path, "m", "signature", {"t0": 1})
        (tmp_path / "tester_validation.json").write_text(json.dumps({"agreement": 1}))
        runs, max_cycles = load_runs(tmp_path)
        assert set(runs) == {("m", "assert"), ("m", "signature")}
        assert max_cycles == 10

    @pytest.mark.parametrize("xs,ys,expected", [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
    ])
    def test_spearman_endpoints(self, xs, ys, expected):
        assert spearman(xs, ys) == pytest.approx(expected)


@needs_package
class TestReproducesLeaderboard:
    """The analysis path must reproduce published numbers from published records."""

    @pytest.mark.parametrize("model_id", ["gemini-2.0-flash-lite", "gpt-5", "gemma-3-12b-it"])
    def test_metrics_match_the_leaderboard(self, model_id, tmp_path):
        published = {m["model_id"]: m for m in
                     json.loads((DATA_DIR / "leaderboard.json").read_text())["models"]}
        record = json.loads((DATA_DIR / "loops" / f"{model_id}.json").read_text())
        payload = {"model_id": model_id, "name": model_id, "configuration": "assert",
                   "max_cycles": 10, "n_tasks": len(record["tasks"]),
                   "tasks": record["tasks"]}
        row = metrics_for(payload)
        expected = published[model_id]
        assert row["success_rate"] == pytest.approx(expected["successRate"], abs=5e-4)
        assert row["asl_score"] == pytest.approx(expected["aslScore"], abs=5e-4)
        assert row["asl_bias"] == pytest.approx(expected["robustnessScore"], abs=5e-4)
        # The leaderboard's passDrop is a raw task count, not a fraction of T.
        assert row["pass_drop_count"] == pytest.approx(expected["passDrop"], abs=0.5)
