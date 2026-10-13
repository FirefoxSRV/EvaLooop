"""Tests for assert-stripping code generation."""

import pytest

from evaloop.tasks import CodeGenerationTask, StrippedCodeGenerationTask
from evaloop.tasks.stripped_code_generation import (
    count_module_level_asserts,
    strip_module_level_asserts,
)

CONTEXT = {"task_id": "Mbpp/1", "cycle_number": 2}


class TestStripModuleLevelAsserts:
    def test_removes_a_module_level_assert(self):
        code, removed = strip_module_level_asserts("def f(x):\n    return x\nassert f(1) == 1\n")
        assert removed == 1
        assert "assert" not in code
        assert "def f(x):" in code

    def test_removes_several(self):
        src = "def f(x):\n    return x\nassert f(1) == 1\nassert f(2) == 2\n"
        code, removed = strip_module_level_asserts(src)
        assert removed == 2
        assert "assert" not in code

    def test_keeps_assertions_inside_a_function(self):
        # An in-function assertion is part of the implementation, not an echo.
        src = "def f(s):\n    assert len(s) > 0\n    return 1\n"
        code, removed = strip_module_level_asserts(src)
        assert removed == 0
        assert "assert len(s) > 0" in code

    def test_strips_the_echo_but_keeps_the_guard(self):
        src = "def f(s):\n    assert len(s) > 0\n    return 1\nassert f('x') == 1\n"
        code, removed = strip_module_level_asserts(src)
        assert removed == 1
        assert "assert len(s) > 0" in code
        assert "f('x')" not in code

    def test_leaves_code_without_assertions_untouched(self):
        src = "def f(x):\n    return x\n"
        assert strip_module_level_asserts(src) == (src, 0)

    def test_returns_unparseable_code_unchanged_so_the_tester_sees_the_error(self):
        assert strip_module_level_asserts("def f(:") == ("def f(:", 0)

    def test_preserves_imports_and_other_top_level_statements(self):
        src = "import math\nCONST = 2\ndef f(x):\n    return math.sqrt(x) * CONST\nassert f(4) == 4\n"
        code, removed = strip_module_level_asserts(src)
        assert removed == 1
        assert "import math" in code and "CONST = 2" in code


class TestCountModuleLevelAsserts:
    @pytest.mark.parametrize("src,expected", [
        ("def f(x):\n    return x\n", 0),
        ("def f(x):\n    return x\nassert f(1) == 1\n", 1),
        ("assert a\nassert b\n", 2),
        ("def f(s):\n    assert s\n    return s\n", 0),
        ("def f(:", 0),
    ])
    def test_counts_only_module_level(self, src, expected):
        assert count_module_level_asserts(src) == expected


class TestTaskBehaviour:
    def test_postprocess_strips(self):
        task = StrippedCodeGenerationTask({})
        out = task.postprocess("def f(x):\n    return x\nassert f(1) == 1\n", CONTEXT)
        assert "assert" not in out

    def test_output_is_still_marked_testable(self):
        # The evaluator finds the code step via this flag.
        assert StrippedCodeGenerationTask({}).produces_testable_code is True

    def test_extraction_is_inherited_from_upstream(self):
        raw = "Here you go:\n```python\ndef f(x):\n    return x\n```\nDone."
        plain, stripped = CodeGenerationTask({}), StrippedCodeGenerationTask({})
        assert stripped.extract_output(raw) == plain.extract_output(raw)

    def test_upstream_generation_does_not_strip(self):
        # Guards against the rule leaking into the unmodified task.
        src = "def f(x):\n    return x\nassert f(1) == 1\n"
        assert CodeGenerationTask({}).postprocess(src, CONTEXT) == src
