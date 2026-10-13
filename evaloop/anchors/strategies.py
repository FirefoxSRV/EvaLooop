"""The three anchor strategies of the anchor-leak ablation.

See :mod:`evaloop.anchors.base` for why an anchor exists at all and what each
strategy is meant to isolate.
"""

import ast
from typing import Any, Dict, Optional

from .base import BaseAnchor, register_anchor


@register_anchor("assert")
class AssertAnchor(BaseAnchor):
    """A0 - upstream behaviour: the original prompt's assert line, verbatim.

    Carries the function name, the arity, one concrete input *and* its
    ground-truth expected output. Because every cycle re-appends the line taken
    from the cycle's own input, and that input's anchor came from the cycle
    before it, the original dataset assert propagates unchanged through the whole
    loop. This is the baseline the ablation is measured against; it must stay
    byte-identical to ``CodeSummarizationTask.postprocess``.
    """

    def build(self, cycle_input: str, code: str, context: Dict[str, Any]) -> str:
        """Return the previous cycle's assert line unchanged."""
        return self.previous_anchor(cycle_input)


@register_anchor("masked_assert")
class MaskedAssertAnchor(BaseAnchor):
    """A1 - the assert with its expected output replaced by ``...``.

    Keeps the function name, the arity and the concrete argument values, and
    removes only the oracle. Isolates the contribution of the leaked expected
    output: anything A0 sustains but A1 does not was being held up by knowing
    the answer to one test case.

    Masking is idempotent, so applying it to an already-masked anchor from a
    later cycle is a no-op (modulo ``ast.unparse`` normalisation, which reaches a
    fixed point after the first cycle).
    """

    def build(self, cycle_input: str, code: str, context: Dict[str, Any]) -> str:
        """Return the previous anchor with the comparison's right side masked."""
        previous = self.previous_anchor(cycle_input)
        masked = mask_assert_oracle(previous)
        if masked is None:
            self._note_fallback("unmaskable_assert")
            return previous
        return masked


@register_anchor("signature")
class SignatureAnchor(BaseAnchor):
    """A2 - a ``def`` line reconstructed from the model's own code.

    Carries the function name and its parameter names, and nothing about any
    test case. The signature is read out of the code the model generated this
    cycle, so under this anchor the model's own output is the only thing crossing
    the cycle boundary - which is what EvaLoop's self-consistency design claims
    to do.

    The code this reads has already passed the benchmark's tests (the loop only
    summarizes passing code), so a function with the expected name and a
    compatible arity is present by construction. ``entry_points`` maps task ids
    to the expected function name; without it the name is recovered from the
    previous anchor.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.entry_points: Dict[str, str] = self.config.get("entry_points") or {}

    def build(self, cycle_input: str, code: str, context: Dict[str, Any]) -> str:
        """Return ``def <entry_point>(<params>):`` taken from this cycle's code."""
        task_id = context.get("task_id", "")
        entry_point = self.entry_points.get(task_id) or entry_point_from_anchor(
            self.previous_anchor(cycle_input)
        )

        if not entry_point:
            self._note_fallback("no_entry_point")
            return self.previous_anchor(cycle_input)

        signature = signature_from_code(code, entry_point)
        if signature is None:
            # The loop only summarizes code that passed, so the function is
            # callable; it may still be bound by something other than a def
            # (a lambda, a functools.partial, a class attribute).
            self._note_fallback("no_functiondef")
            return f"def {entry_point}(*args):"
        return signature


def mask_assert_oracle(assert_line: str) -> Optional[str]:
    """Replace the expected value in an assert with ``...``.

    ``assert f(1) == 2``                -> ``assert f(1) == ...``
    ``assert set(g(1)) == set((1, 2))`` -> ``assert set(g(1)) == ...``
    ``assert h(1)``                     -> None (the oracle *is* the truthiness)

    Returns:
        The masked line, or None if the statement is not an assert whose test is
        a comparison (such asserts cannot be masked without destroying the call).
    """
    try:
        tree = ast.parse(assert_line.strip())
    except SyntaxError:
        return None

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None

    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return None

    test.comparators[0] = ast.Constant(value=Ellipsis)
    # Drop any assert message: it could restate the expected value.
    tree.body[0].msg = None
    return ast.unparse(ast.fix_missing_locations(tree))


def signature_from_code(code: str, entry_point: str) -> Optional[str]:
    """Reconstruct the ``def`` line of ``entry_point`` from source code.

    Prefers a module-level definition, then any nested one, so a helper that
    shadows the name inside another function cannot win over the real entry
    point.

    Returns:
        ``def name(params):`` with the model's own parameter names, annotations
        and defaults, or None if no matching function definition exists.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    def matches(node) -> bool:
        return isinstance(node, func_types) and node.name == entry_point

    node = next((n for n in tree.body if matches(n)), None)
    if node is None:
        node = next((n for n in ast.walk(tree) if matches(n)), None)
    if node is None:
        return None

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({ast.unparse(node.args)}):"


def entry_point_from_anchor(anchor_line: str) -> Optional[str]:
    """Recover the tested function's name from an anchor line.

    Handles both anchor shapes - an assert (``assert f(1) == 2``, including calls
    wrapped in a builtin such as ``set(f(1))``) and a def line (``def f(x):``) -
    so a signature anchor can bootstrap from cycle 1's dataset prompt and keep
    working from its own output afterwards.

    Returns:
        The function name, or None if the line yields no call or definition.
    """
    tree = _parse_anchor(anchor_line)
    if tree is None:
        return None

    if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return tree.body[0].name

    # Innermost-first: the entry point is the call nested inside any wrapper
    # such as set(...), sorted(...) or round(...).
    names = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    if not names:
        return None
    non_builtin = [n for n in names if not _is_builtin(n)]
    return (non_builtin or names)[-1]


def _parse_anchor(anchor_line: str) -> Optional[ast.Module]:
    """Parse an anchor line, tolerating a bodyless ``def`` line.

    A signature anchor such as ``def f(x):`` is not a parseable module on its
    own, so a placeholder body is supplied before reparsing.
    """
    line = anchor_line.strip()
    if not line:
        return None
    try:
        return ast.parse(line)
    except SyntaxError:
        pass
    try:
        return ast.parse(line + "\n    ...")
    except SyntaxError:
        return None


def _is_builtin(name: str) -> bool:
    """True if ``name`` is a Python builtin, which is never a task's entry point."""
    import builtins
    return hasattr(builtins, name)
