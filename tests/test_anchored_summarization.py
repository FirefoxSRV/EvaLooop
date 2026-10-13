"""Tests for AnchoredSummarizationTask.

The most important test here is the equivalence test: with ``anchor: assert`` the
new task must be byte-for-byte identical to the upstream summarization task. The
anchor ablation's baseline arm is run with that config, so if the two ever
diverge the experiment would be comparing against something that is not the
published loop.
"""

import pytest

from evaloop.tasks import AnchoredSummarizationTask, CodeSummarizationTask

PROMPT_TEMPLATE = (
    "Use one sentence to summarize the following code and start with "
    "'write a python function to':\n\n```\n{code}\n```\n\n```\n"
    "write a python function to\n```"
)

# (cycle_input, code, summary) triples covering the prompt shapes that occur in
# MBPP+ and the model outputs that occur in the published records.
CASES = [
    # plain equality assert
    ('"""\nFind the max difference.\nassert find_length("11000010001") == 6\n"""\n',
     "def find_length(s):\n    return 6\n",
     "write a python function to find the maximum difference."),
    # assert whose call is wrapped in a builtin
    ('"""\nFind shared elements.\nassert set(similar_elements((3, 4),(5, 7))) == set((4, 5))\n"""\n',
     "def similar_elements(a, b):\n    return tuple(set(a) & set(b))\n",
     "write a python function to find the shared elements of two lists."),
    # bare truthiness assert
    ('"""\nCheck Woodall.\nassert is_woodall(383)\n"""\n',
     "def is_woodall(x):\n    return True\n",
     "write a python function to check whether a number is a Woodall number."),
    # multi-line summary (some models ignore the one-sentence instruction)
    ('"""\nSort the list.\nassert sort_list([3, 1]) == [1, 3]\n"""\n',
     "def sort_list(xs):\n    return sorted(xs)\n",
     "write a python function to sort a list.\nIt uses the builtin sorted()."),
    # empty summary (observed when a model returns nothing usable)
    ('"""\nDo it.\nassert f(1) == 1\n"""\n', "def f(x):\n    return x\n", ""),
]


def _upstream(code_config=None):
    return CodeSummarizationTask(code_config or {"prompt_template": PROMPT_TEMPLATE})


def _anchored(anchor="assert"):
    return AnchoredSummarizationTask(
        {"prompt_template": PROMPT_TEMPLATE, "anchor": anchor}
    )


def _context(cycle_input, code, cycle=3):
    return {
        "cycle_input": cycle_input,
        "task_input": code,
        "task_id": "Mbpp/1",
        "cycle_number": cycle,
    }


class TestUpstreamEquivalence:
    """anchor='assert' must reproduce upstream exactly."""

    @pytest.mark.parametrize("cycle_input,code,summary", CASES)
    def test_postprocess_is_byte_identical_to_upstream(self, cycle_input, code, summary):
        context = _context(cycle_input, code)
        assert _anchored("assert").postprocess(summary, context) == \
            _upstream().postprocess(summary, context)

    def test_create_prompt_is_inherited_unchanged(self):
        code = "def f(x):\n    return x\n"
        assert _anchored().create_prompt(code) == _upstream().create_prompt(code)

    def test_extract_output_is_inherited_unchanged(self):
        raw = "  write a python function to do a thing.  \n"
        assert _anchored().extract_output(raw) == _upstream().extract_output(raw)

    def test_the_default_anchor_is_the_upstream_one(self):
        # A config that names no anchor must not silently change the loop.
        task = AnchoredSummarizationTask({"prompt_template": PROMPT_TEMPLATE})
        assert task.anchor.name == "assert"

    def test_summarization_is_not_marked_as_testable_code(self):
        # The evaluator picks which output to run by this flag; a summary is not code.
        assert _anchored().produces_testable_code is False


class TestWrapperInvariant:
    """Every anchor must leave the anchor line second-to-last."""

    @pytest.mark.parametrize("anchor", ["assert", "masked_assert", "signature"])
    @pytest.mark.parametrize("cycle_input,code,summary", CASES)
    def test_anchor_is_the_second_to_last_line(self, anchor, cycle_input, code, summary):
        wrapped = _anchored(anchor).postprocess(summary, _context(cycle_input, code))
        lines = wrapped.splitlines()
        assert lines[0] == '"""'
        assert lines[-1] == '"""'
        expected = _anchored(anchor).anchor.build(cycle_input, code, _context(cycle_input, code))
        assert lines[-2] == expected

    @pytest.mark.parametrize("anchor", ["assert", "masked_assert", "signature"])
    def test_the_wrapper_round_trips_across_cycles(self, anchor, ):
        """Feeding a wrapped prompt back in must keep the anchor recoverable."""
        task = _anchored(anchor)
        cycle_input, code, summary = CASES[0]
        for cycle in range(1, 5):
            wrapped = task.postprocess(summary, _context(cycle_input, code, cycle))
            # The next cycle's anchor is read off this output.
            assert task.anchor.previous_anchor(wrapped) == wrapped.splitlines()[-2]
            cycle_input = wrapped


class TestAblationActuallyAblates:
    """The three anchors must differ in what they carry, or there is no experiment."""

    def test_the_oracle_is_present_only_under_the_assert_anchor(self):
        cycle_input, code, summary = CASES[0]
        context = _context(cycle_input, code)
        assert "== 6" in _anchored("assert").postprocess(summary, context)
        assert "== 6" not in _anchored("masked_assert").postprocess(summary, context)
        assert "== 6" not in _anchored("signature").postprocess(summary, context)

    def test_the_test_input_is_dropped_only_by_the_signature_anchor(self):
        cycle_input, code, summary = CASES[0]
        context = _context(cycle_input, code)
        assert "11000010001" in _anchored("assert").postprocess(summary, context)
        assert "11000010001" in _anchored("masked_assert").postprocess(summary, context)
        assert "11000010001" not in _anchored("signature").postprocess(summary, context)

    @pytest.mark.parametrize("anchor", ["assert", "masked_assert", "signature"])
    def test_the_function_name_survives_every_anchor(self, anchor):
        """Signature stability is the whole reason an anchor exists."""
        cycle_input, code, summary = CASES[0]
        wrapped = _anchored(anchor).postprocess(summary, _context(cycle_input, code))
        assert "find_length" in wrapped
