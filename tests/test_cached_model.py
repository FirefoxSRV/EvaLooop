"""Tests for the disk-backed call cache.

The guarantee that matters for the experiment: identical prompts under identical
settings are served from cache, which is what makes cycle 1 shared across the
three anchor arms and pass@1 equal by construction.
"""

import json

import pytest

from evaloop.models.base import BaseLLM
from evaloop.models.cached import CachingLLM


class FakeLLM(BaseLLM):
    """Counts calls and returns a deterministic, distinguishable response."""

    def __init__(self, config=None):
        super().__init__(config or {"name": "fake"})
        self.model_id = "fake-1"
        self.temperature = 0.0
        self.top_p = 1.0
        self.max_tokens = 128
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return f"response::{prompt}::{len(self.calls)}"


class FakeBatchLLM(FakeLLM):
    """A client that exposes batch_generate, like the real API backends."""

    def __init__(self, config=None):
        super().__init__(config)
        self.batches = []

    def batch_generate(self, prompts, **kwargs):
        self.batches.append(list(prompts))
        return [self.generate(p, **kwargs) for p in prompts]


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "cache" / "fake.jsonl"


class TestSingleGenerate:
    def test_second_identical_call_is_served_from_cache(self, cache_file):
        inner = FakeLLM()
        model = CachingLLM(inner, str(cache_file))
        first = model.generate("hello")
        second = model.generate("hello")
        assert first == second
        assert len(inner.calls) == 1
        assert (model.hits, model.misses) == (1, 1)

    def test_different_prompts_are_separate_entries(self, cache_file):
        inner = FakeLLM()
        model = CachingLLM(inner, str(cache_file))
        model.generate("a")
        model.generate("b")
        assert len(inner.calls) == 2
        assert model.hits == 0

    def test_changed_settings_are_a_different_key(self, cache_file):
        inner = FakeLLM()
        model = CachingLLM(inner, str(cache_file))
        model.generate("a")
        model.generate("a", temperature=0.7)
        # A cached greedy completion must never be served for a sampled request.
        assert len(inner.calls) == 2

    def test_failures_are_not_cached(self, cache_file):
        class FailingLLM(FakeLLM):
            def generate(self, prompt, **kwargs):
                self.calls.append(prompt)
                return "Error generating response: boom"

        inner = FailingLLM()
        model = CachingLLM(inner, str(cache_file))
        model.generate("a")
        model.generate("a")
        # Retried, not replayed from cache forever.
        assert len(inner.calls) == 2


class TestPersistence:
    def test_cache_survives_a_new_wrapper(self, cache_file):
        first_inner = FakeLLM()
        CachingLLM(first_inner, str(cache_file)).generate("hello")

        second_inner = FakeLLM()
        reloaded = CachingLLM(second_inner, str(cache_file))
        reloaded.generate("hello")
        assert second_inner.calls == []
        assert reloaded.hits == 1

    def test_records_are_json_lines(self, cache_file):
        model = CachingLLM(FakeLLM(), str(cache_file))
        model.generate("hello")
        records = [json.loads(line) for line in cache_file.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert {"key", "model_id", "prompt", "response", "created_at"} <= set(records[0])

    def test_a_truncated_final_line_is_skipped(self, cache_file):
        model = CachingLLM(FakeLLM(), str(cache_file))
        model.generate("hello")
        with cache_file.open("a") as handle:
            handle.write('{"key": "partial", "resp')  # crash mid-write

        inner = FakeLLM()
        reloaded = CachingLLM(inner, str(cache_file))
        reloaded.generate("hello")
        assert inner.calls == []

    def test_a_different_model_id_does_not_collide(self, cache_file):
        a = FakeLLM()
        CachingLLM(a, str(cache_file), settings={"model_id": "m-a"}).generate("hello")
        b = FakeLLM()
        CachingLLM(b, str(cache_file), settings={"model_id": "m-b"}).generate("hello")
        assert len(b.calls) == 1


class TestBatchGenerate:
    def test_order_is_preserved(self, cache_file):
        model = CachingLLM(FakeBatchLLM(), str(cache_file))
        prompts = ["a", "b", "c"]
        assert [r.split("::")[1] for r in model.batch_generate(prompts)] == prompts

    def test_only_misses_reach_the_model(self, cache_file):
        inner = FakeBatchLLM()
        model = CachingLLM(inner, str(cache_file))
        model.batch_generate(["a", "b"])
        inner.batches.clear()
        results = model.batch_generate(["a", "b", "c"])
        # Only the new prompt is fanned out; the two cached ones are served locally.
        assert inner.batches == [["c"]]
        assert [r.split("::")[1] for r in results] == ["a", "b", "c"]

    def test_a_fully_cached_batch_makes_no_call(self, cache_file):
        inner = FakeBatchLLM()
        model = CachingLLM(inner, str(cache_file))
        model.batch_generate(["a", "b"])
        inner.batches.clear()
        model.batch_generate(["a", "b"])
        assert inner.batches == []
        assert model.cache_report()["hit_rate"] > 0

    def test_empty_batch(self, cache_file):
        assert CachingLLM(FakeBatchLLM(), str(cache_file)).batch_generate([]) == []

    def test_falls_back_to_generate_when_inner_has_no_batch(self, cache_file):
        inner = FakeLLM()
        model = CachingLLM(inner, str(cache_file))
        assert len(model.batch_generate(["a", "b"])) == 2
        assert len(inner.calls) == 2


class TestCycleOneSharingGuarantee:
    """The property the experimental design relies on."""

    def test_three_arms_share_cycle_one(self, cache_file):
        """Three runs over one shared cache must call the model only once."""
        inner = FakeBatchLLM()
        cycle_one_prompts = ["task-1 prompt", "task-2 prompt"]

        hit_rates = []
        for _arm in ("assert", "masked_assert", "signature"):
            arm_model = CachingLLM(inner, str(cache_file))
            arm_model.batch_generate(cycle_one_prompts)
            hit_rates.append(arm_model.cache_report()["hit_rate"])

        assert inner.calls == cycle_one_prompts        # one call per task, total
        assert hit_rates == [0.0, 1.0, 1.0]            # arms 2 and 3 fully cached


class TestDelegation:
    def test_unknown_attributes_reach_the_inner_client(self, cache_file):
        model = CachingLLM(FakeLLM(), str(cache_file))
        assert model.model_id == "fake-1"

    def test_usage_report_is_annotated_with_cache_stats(self, cache_file):
        model = CachingLLM(FakeLLM(), str(cache_file))
        model.generate("a")
        assert "cache" in model.usage_report()
