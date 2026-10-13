"""Tests for the cross-cycle anchor strategies."""

import ast

import pytest

from evaloop.anchors import (
    AssertAnchor,
    MaskedAssertAnchor,
    SignatureAnchor,
    create_anchor,
    entry_point_from_anchor,
    list_anchors,
    mask_assert_oracle,
    signature_from_code,
)

# A real MBPP+ prompt: the anchor is the second-to-last line by construction.
MBPP_PROMPT = '"""\nWrite a function to find the shared elements from the given two lists.\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"""\n'
SIMPLE_PROMPT = '"""\nWrite a function to find the maximum difference.\nassert find_length("11000010001") == 6\n"""\n'
CODE = "def find_length(s):\n    return max(len(s), 0)\n"


class TestRegistry:
    def test_all_three_anchors_are_registered(self):
        assert list_anchors() == ["assert", "masked_assert", "signature"]

    def test_create_anchor_returns_the_right_class(self):
        assert isinstance(create_anchor("assert"), AssertAnchor)
        assert isinstance(create_anchor("masked_assert"), MaskedAssertAnchor)
        assert isinstance(create_anchor("signature"), SignatureAnchor)

    def test_unknown_anchor_is_rejected_and_lists_the_options(self):
        with pytest.raises(ValueError, match="masked_assert"):
            create_anchor("no_such_anchor")


class TestMaskAssertOracle:
    def test_masks_a_plain_equality(self):
        assert mask_assert_oracle('assert find_length("11000010001") == 6') == \
            "assert find_length('11000010001') == ..."

    def test_masks_a_call_wrapped_in_a_builtin(self):
        masked = mask_assert_oracle("assert set(similar_elements((3, 4),(5, 7))) == set((4, 5))")
        assert masked == "assert set(similar_elements((3, 4), (5, 7))) == ..."
        # The arguments must survive: they are what A1 still carries.
        assert "(3, 4)" in masked

    def test_is_idempotent_so_later_cycles_are_stable(self):
        once = mask_assert_oracle('assert find_length("11") == 6')
        assert mask_assert_oracle(once) == once

    def test_returns_none_for_a_bare_truthiness_assert(self):
        # The oracle *is* the truthiness, so there is nothing to mask.
        assert mask_assert_oracle("assert is_woodall(383)") is None

    def test_returns_none_for_non_assert_and_unparseable_input(self):
        assert mask_assert_oracle("x = 1") is None
        assert mask_assert_oracle("def f(:") is None
        assert mask_assert_oracle("") is None

    def test_drops_an_assert_message_that_could_restate_the_answer(self):
        masked = mask_assert_oracle('assert f(1) == 2, "should be 2"')
        assert masked == "assert f(1) == ..."

    def test_masked_output_is_still_valid_python(self):
        ast.parse(mask_assert_oracle("assert f(1) == 2"))


class TestSignatureFromCode:
    def test_extracts_a_module_level_def(self):
        assert signature_from_code(CODE, "find_length") == "def find_length(s):"

    def test_keeps_the_models_own_annotations_and_defaults(self):
        code = "def f(a: int, b: str = 'x', *args, **kw) -> bool:\n    return True\n"
        assert signature_from_code(code, "f") == "def f(a: int, b: str='x', *args, **kw):"

    def test_prefers_module_level_over_a_nested_shadow(self):
        code = "def outer():\n    def f(wrong):\n        return wrong\n\ndef f(right):\n    return right\n"
        assert signature_from_code(code, "f") == "def f(right):"

    def test_falls_back_to_a_nested_def_when_there_is_no_module_level_one(self):
        code = "def outer():\n    def f(inner):\n        return inner\n    return f\n"
        assert signature_from_code(code, "f") == "def f(inner):"

    def test_handles_async_def(self):
        assert signature_from_code("async def f(x):\n    return x\n", "f") == "async def f(x):"

    def test_returns_none_when_the_name_is_absent_or_code_is_broken(self):
        assert signature_from_code(CODE, "other_name") is None
        assert signature_from_code("def f(:\n", "f") is None


class TestEntryPointFromAnchor:
    @pytest.mark.parametrize("line,expected", [
        ('assert find_length("11") == 6', "find_length"),
        ("assert set(similar_elements((3, 4),(5, 7))) == set((4, 5))", "similar_elements"),
        ("assert round(area(3), 2) == 28.27", "area"),
        ("assert is_woodall(383)", "is_woodall"),
        ("assert find_length('11') == ...", "find_length"),
        ("def find_length(s):", "find_length"),
        ("async def f(x, y=2):", "f"),
    ])
    def test_recovers_the_name_from_both_anchor_shapes(self, line, expected):
        assert entry_point_from_anchor(line) == expected

    @pytest.mark.parametrize("line", ["", "   ", "garbage ((", "x = 1"])
    def test_returns_none_when_there_is_no_call_or_def(self, line):
        assert entry_point_from_anchor(line) is None


class TestAnchorBuild:
    CONTEXT = {"task_id": "Mbpp/2", "cycle_number": 1}

    def test_assert_anchor_returns_the_line_verbatim(self):
        anchor = create_anchor("assert")
        assert anchor.build(SIMPLE_PROMPT, CODE, self.CONTEXT) == \
            'assert find_length("11000010001") == 6'

    def test_assert_anchor_survives_a_degenerate_prompt(self):
        assert create_anchor("assert").build("", CODE, self.CONTEXT) == ""

    def test_masked_anchor_removes_the_oracle(self):
        anchor = create_anchor("masked_assert")
        assert anchor.build(SIMPLE_PROMPT, CODE, self.CONTEXT) == \
            "assert find_length('11000010001') == ..."

    def test_masked_anchor_passes_an_unmaskable_assert_through_and_records_it(self):
        prompt = '"""\nCheck it.\nassert is_woodall(383)\n"""\n'
        anchor = create_anchor("masked_assert")
        assert anchor.build(prompt, CODE, self.CONTEXT) == "assert is_woodall(383)"
        assert anchor.fallbacks == {"unmaskable_assert": 1}

    def test_signature_anchor_reads_the_models_own_code(self):
        anchor = create_anchor("signature", {"entry_points": {"Mbpp/2": "find_length"}})
        assert anchor.build(SIMPLE_PROMPT, CODE, self.CONTEXT) == "def find_length(s):"
        assert anchor.fallbacks == {}

    def test_signature_anchor_bootstraps_the_name_from_the_anchor_line(self):
        # No entry_points mapping: the name must come from the previous anchor.
        anchor = create_anchor("signature")
        assert anchor.build(SIMPLE_PROMPT, CODE, self.CONTEXT) == "def find_length(s):"

    def test_signature_anchor_tracks_the_models_parameter_renaming(self):
        anchor = create_anchor("signature", {"entry_points": {"Mbpp/2": "find_length"}})
        renamed = "def find_length(binary_string):\n    return 0\n"
        assert anchor.build(SIMPLE_PROMPT, renamed, self.CONTEXT) == \
            "def find_length(binary_string):"

    def test_signature_anchor_falls_back_when_there_is_no_functiondef(self):
        anchor = create_anchor("signature", {"entry_points": {"Mbpp/2": "find_length"}})
        lambda_code = "find_length = lambda s: 0\n"
        assert anchor.build(SIMPLE_PROMPT, lambda_code, self.CONTEXT) == "def find_length(*args):"
        assert anchor.fallbacks == {"no_functiondef": 1}

    def test_signature_anchor_falls_back_when_the_name_is_unrecoverable(self):
        anchor = create_anchor("signature")
        assert anchor.build("", CODE, self.CONTEXT) == ""
        assert anchor.fallbacks == {"no_entry_point": 1}

    def test_every_anchor_leaks_nothing_wider_than_a_single_line(self):
        for name in list_anchors():
            line = create_anchor(name).build(MBPP_PROMPT, CODE, self.CONTEXT)
            assert "\n" not in line, name
