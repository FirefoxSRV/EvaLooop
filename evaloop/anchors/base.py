"""Cross-cycle anchors: what holds a task's identity steady across loop cycles.

EvaLoop's generation <-> summarization loop has a problem to solve that is easy
to miss. A summary of code is not a function specification: it does not fix the
function's *name*, so code generated from it would be called something else and
the benchmark's tests could not invoke it. Upstream solves this by appending the
final assert line of the *original* dataset prompt to every summary
(``CodeSummarizationTask.postprocess``).

That solution carries a cost. The assert line is re-injected verbatim from
outside the loop on every cycle, so it never degrades, and it carries more than
a name: it carries one concrete input together with its ground-truth expected
output. EvaLoop's stated premise is that the model's own output is the only
source of perturbation, yet this one line is dataset ground truth entering the
loop afresh each cycle.

An :class:`BaseAnchor` makes that choice explicit and swappable, so the
contribution of the leaked oracle can be measured instead of assumed:

======================  =========================================  ===============================
anchor                  carries                                    source
======================  =========================================  ===============================
``assert``              name, arity, inputs, **expected output**   original prompt (constant)
``masked_assert``       name, arity, inputs                        original prompt, oracle masked
``signature``           name, parameter names                      the model's own code (per cycle)
======================  =========================================  ===============================

``assert`` reproduces upstream exactly and is the baseline. ``signature`` is the
only configuration in which nothing but model output crosses the cycle boundary.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type

_ANCHOR_REGISTRY: Dict[str, Type["BaseAnchor"]] = {}


def register_anchor(name: str) -> Callable[[Type["BaseAnchor"]], Type["BaseAnchor"]]:
    """Class decorator registering a BaseAnchor subclass under a config name.

    Mirrors ``evaloop.tasks.factory.register_task`` so anchors are discovered the
    same way task types are.

    Raises:
        TypeError: If the decorated class is not a BaseAnchor subclass.
        ValueError: If the anchor name is already registered.
    """
    def decorator(cls: Type["BaseAnchor"]) -> Type["BaseAnchor"]:
        key = name.lower()
        if not (isinstance(cls, type) and issubclass(cls, BaseAnchor)):
            raise TypeError(f"{cls!r} must subclass BaseAnchor to be registered")
        if key in _ANCHOR_REGISTRY:
            raise ValueError(f"Anchor '{key}' is already registered")
        _ANCHOR_REGISTRY[key] = cls
        cls.name = key
        return cls
    return decorator


def create_anchor(name: str, config: Dict[str, Any] = None) -> "BaseAnchor":
    """Instantiate a registered anchor by name.

    Raises:
        ValueError: If the anchor name is not registered.
    """
    key = (name or "").lower()
    anchor_cls = _ANCHOR_REGISTRY.get(key)
    if anchor_cls is None:
        raise ValueError(
            f"Unknown anchor: '{name}'. Available anchors: {', '.join(sorted(_ANCHOR_REGISTRY))}"
        )
    return anchor_cls(config or {})


def list_anchors() -> list:
    """Return the sorted names of all registered anchors."""
    return sorted(_ANCHOR_REGISTRY)


class BaseAnchor(ABC):
    """Builds the line that pins a task's identity for the next cycle.

    The anchored summarization task wraps a summary as::

        \"\"\"
        {summary}
        {anchor}
        \"\"\"

    which keeps the upstream invariant that the anchor is the second-to-last
    line of a cycle's input prompt, so an anchor can always recover the previous
    cycle's anchor from ``cycle_input``.
    """

    #: Set by @register_anchor to the registered config name.
    name: str = "base"

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the anchor with its (optional) config block."""
        self.config = config or {}
        #: Counts of degraded paths taken, reported with the run for transparency.
        self.fallbacks: Dict[str, int] = {}

    @abstractmethod
    def build(self, cycle_input: str, code: str, context: Dict[str, Any]) -> str:
        """Return the anchor line to append to this cycle's summary.

        Args:
            cycle_input: The prompt this cycle started from. Its second-to-last
                line is the previous cycle's anchor (for cycle 1, the dataset
                prompt's assert line).
            code: The code the model generated this cycle, i.e. the code that
                was just summarized. Anchors derived from model output use this.
            context: Loop context; carries at least ``task_id`` and
                ``cycle_number``.

        Returns:
            A single line of Python, without a trailing newline.
        """

    def _note_fallback(self, reason: str) -> None:
        """Record that a degraded path was taken, for run-level reporting."""
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    @staticmethod
    def previous_anchor(cycle_input: str) -> str:
        """Extract the anchor line from a cycle's input prompt.

        Upstream's ``[-2]`` convention: the prompt is a docstring whose last line
        is the closing quotes and whose second-to-last line is the anchor.
        """
        try:
            return cycle_input.splitlines()[-2]
        except IndexError:
            return ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
