"""Code generation that removes echoed test assertions before testing.

Why this exists
---------------
Models routinely copy the prompt's example assertion into their answer. In the
published records the rate ranges from 0% (gemini-2.0-flash) to 86%
(gemini-2.5-pro) of cycles, so it is not a fringe behaviour.

Under the upstream ``assert`` anchor an echoed assertion is harmless: it is the
dataset's own ground truth, so it holds. Under an ablated anchor it is actively
misleading in two ways:

1. ``masked_assert`` carries ``assert f(x) == ...``. A model that echoes it
   raises AssertionError at import, so the cycle fails for a *formatting* reason
   rather than for the information reason the experiment is about. Models that
   echo more would look less robust purely as an artefact.
2. An echoed assertion is part of the code that gets summarized next. A summary
   of code containing ``assert f("11") == 6`` can restate that expected output
   in prose, re-injecting through the summary exactly the oracle the ablation
   removed from the anchor.

So module-level assertions are stripped from generated code in *every*
configuration, baseline included, before the code is tested and before it is
summarized. Applying the rule uniformly keeps the arms comparable; applying it
only to the ablated arms would not.

This is a deliberate deviation from upstream, reported in the README and the
write-up, and its effect on the baseline is measured directly by
``scripts/retest_package_code.py``.
"""

import ast
import logging
from typing import Any, Dict, Tuple

from .code_generation import CodeGenerationTask
from .factory import register_task

logger = logging.getLogger(__name__)


@register_task("stripped_code_generation")
class StrippedCodeGenerationTask(CodeGenerationTask):
    """Code generation whose output has module-level assertions removed."""

    produces_testable_code = True

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Strip module-level assertions from the extracted code.

        Falls back to the unmodified code when it does not parse, so a syntax
        error still reaches the tester and is recorded as a genuine failure
        rather than being masked here.
        """
        stripped, removed = strip_module_level_asserts(output)
        if removed:
            logger.debug(
                "Stripped %d echoed assertion(s) from %s cycle %s",
                removed, context.get("task_id"), context.get("cycle_number"),
            )
        return stripped


def strip_module_level_asserts(code: str) -> Tuple[str, int]:
    """Remove top-level ``assert`` statements from source code.

    Only module-level assertions are removed. Assertions *inside* a function are
    part of the implementation (a guard clause, say) and are left alone.

    Returns:
        ``(code, number_of_assertions_removed)``. If the code does not parse it
        is returned unchanged with a count of 0.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0

    kept = [node for node in tree.body if not isinstance(node, ast.Assert)]
    removed = len(tree.body) - len(kept)
    if not removed:
        return code, 0

    tree.body = kept
    return ast.unparse(ast.fix_missing_locations(tree)), removed


def count_module_level_asserts(code: str) -> int:
    """Count top-level ``assert`` statements, for per-model echo-rate reporting."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(1 for node in tree.body if isinstance(node, ast.Assert))
