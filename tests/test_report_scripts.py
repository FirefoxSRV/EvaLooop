"""Tests for the reporting scripts: native re-serialisation and cost accounting."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cost_report import cost_of, decumulate  # noqa: E402
from emit_native_results import to_native  # noqa: E402


def package_run(configuration="assert", max_cycles=10):
    """A minimal package-schema run with one passing and one failing task."""
    return {
        "model_id": "m", "name": "m", "configuration": configuration,
        "max_cycles": max_cycles, "n_tasks": 2,
        "tasks": [
            {"task_id": "Mbpp/1", "initial_prompt": '"""\nt\nassert f() == 1\n"""\n',
             "successful_cycles": 2, "max_cycles_reached": False,
             "boundary_similarity": None,
             "cycles": [
                 {"cycle": 1, "code": "def f():\n    return 1\n",
                  "summary": '"""\ns1\nassert f() == 1\n"""\n',
                  "passed": True, "base_passed": True, "plus_passed": True},
                 {"cycle": 2, "code": "def f():\n    return 1\n",
                  "summary": '"""\ns2\nassert f() == 1\n"""\n',
                  "passed": True, "base_passed": True, "plus_passed": True},
                 {"cycle": 3, "code": "def f():\n    return 2\n", "summary": None,
                  "passed": False, "base_passed": False, "plus_passed": False},
             ]},
            {"task_id": "Mbpp/2", "initial_prompt": '"""\nu\nassert g() == 1\n"""\n',
             "successful_cycles": 10, "max_cycles_reached": True,
             "boundary_similarity": None,
             "cycles": [{"cycle": i + 1, "code": "def g():\n    return 1\n",
                         "summary": '"""\ns\nassert g() == 1\n"""\n',
                         "passed": True, "base_passed": True, "plus_passed": True}
                        for i in range(10)]},
        ],
    }


class TestNativeSerialisation:
    def test_shape_matches_what_the_framework_consumes(self):
        native = to_native(package_run())
        assert isinstance(native, list) and len(native) == 1
        result = native[0]
        assert {"model", "max_cycles", "prompt_results"} <= set(result)
        prompt_result = result["prompt_results"][0]
        assert {"task_id", "initial_prompt", "successful_cycles",
                "max_cycles_reached", "cycles"} <= set(prompt_result)

    def test_code_and_summary_become_the_numbered_task_outputs(self):
        cycle = to_native(package_run())[0]["prompt_results"][0]["cycles"][0]
        assert cycle["task_1_output"] == "def f():\n    return 1\n"
        assert cycle["task_2_output"].endswith('"""\n')

    def test_a_failing_final_cycle_has_no_summary_key(self):
        # The framework omits the key rather than writing null.
        cycles = to_native(package_run())[0]["prompt_results"][0]["cycles"]
        assert "task_2_output" not in cycles[-1]
        assert cycles[-1]["success"] is False

    def test_sustained_cycles_are_preserved_exactly(self):
        native = to_native(package_run())[0]
        assert [p["successful_cycles"] for p in native["prompt_results"]] == [2, 10]
        assert native["average_successful_cycles"] == pytest.approx(6.0)

    def test_base_and_plus_outcomes_survive_the_round_trip(self):
        cycle = to_native(package_run())[0]["prompt_results"][0]["cycles"][2]
        assert cycle["test_results"] == {"success": False, "base_success": False,
                                         "plus_success": False}


class TestCostAccounting:
    def test_rates_absent_means_no_invented_number(self):
        assert cost_of(1_000_000, 1_000_000, {}) is None
        assert cost_of(1_000_000, 1_000_000,
                       {"input_per_mtok": None, "output_per_mtok": None}) is None

    def test_cost_is_per_million_tokens(self):
        cost = cost_of(2_000_000, 500_000,
                       {"input_per_mtok": 1.0, "output_per_mtok": 4.0})
        assert cost == pytest.approx(2.0 + 2.0)

    def test_decumulate_recovers_per_arm_figures(self):
        # The arms share a client, so counters accumulate across them.
        rows = [
            {"model": "m", "configuration": "assert", "api_calls": 300, "cache_hits": 100,
             "prompt_tokens": 1000, "output_tokens": 500, "thoughts_tokens": 0,
             "retries": 10, "rate_limited": 10, "paid_equivalent_usd": None},
            {"model": "m", "configuration": "masked_assert", "api_calls": 450,
             "cache_hits": 400, "prompt_tokens": 1400, "output_tokens": 700,
             "thoughts_tokens": 0, "retries": 15, "rate_limited": 15,
             "paid_equivalent_usd": None},
            {"model": "m", "configuration": "signature", "api_calls": 600,
             "cache_hits": 900, "prompt_tokens": 1800, "output_tokens": 900,
             "thoughts_tokens": 0, "retries": 20, "rate_limited": 20,
             "paid_equivalent_usd": None},
        ]
        out = {r["configuration"]: r for r in decumulate(rows)}
        assert out["assert"]["api_calls"] == 300
        assert out["masked_assert"]["api_calls"] == 150
        assert out["signature"]["api_calls"] == 150
        # The model's true total is the last arm's cumulative value.
        assert sum(r["api_calls"] for r in out.values()) == 600
        assert all(r["counters_decumulated"] for r in out.values())

    def test_non_monotonic_counters_are_left_alone(self):
        # A resumed run does not share a client, so differencing would be wrong.
        rows = [
            {"model": "m", "configuration": "assert", "api_calls": 500, "cache_hits": 0,
             "prompt_tokens": 100, "output_tokens": 50, "thoughts_tokens": 0,
             "retries": 0, "rate_limited": 0, "paid_equivalent_usd": None},
            {"model": "m", "configuration": "signature", "api_calls": 10, "cache_hits": 0,
             "prompt_tokens": 10, "output_tokens": 5, "thoughts_tokens": 0,
             "retries": 0, "rate_limited": 0, "paid_equivalent_usd": None},
        ]
        out = {r["configuration"]: r for r in decumulate(rows)}
        assert out["assert"]["api_calls"] == 500
        assert out["signature"]["api_calls"] == 10
        assert "counters_decumulated" not in out["assert"]

    def test_models_are_decumulated_independently(self):
        rows = []
        for model, scale in [("a", 1), ("b", 10)]:
            for i, arm in enumerate(["assert", "masked_assert", "signature"], start=1):
                rows.append({"model": model, "configuration": arm,
                             "api_calls": 100 * i * scale, "cache_hits": 0,
                             "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
                             "retries": 0, "rate_limited": 0, "paid_equivalent_usd": None})
        out = {(r["model"], r["configuration"]): r for r in decumulate(rows)}
        assert out[("a", "signature")]["api_calls"] == 100
        assert out[("b", "signature")]["api_calls"] == 1000
