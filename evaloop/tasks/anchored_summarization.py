"""Summarization whose cross-cycle anchor is configurable (the anchor ablation).

Upstream's :class:`~evaloop.tasks.code_summarization.CodeSummarizationTask`
hardcodes one anchor: the assert line of the prompt the cycle started from. This
task keeps everything else about that step identical - same summarization prompt,
same docstring wrapper, same ``[-2]`` layout - and makes only the anchor
pluggable, so the loop can be run with the oracle removed from the anchor or with
an anchor derived from the model's own code.

With ``anchor: assert`` it is byte-for-byte equivalent to the upstream task; that
equivalence is asserted in ``tests/test_anchored_summarization.py`` so the
baseline arm of the experiment is provably unchanged.

Config::

    {"type": "anchored_summarization",
     "anchor": "assert" | "masked_assert" | "signature",
     "prompt_template": "...",            # same meaning as upstream
     "entry_points": {"Mbpp/2": "similar_elements", ...}}   # signature anchor only
"""

from typing import Any, Dict

from ..anchors import create_anchor
from .code_summarization import CodeSummarizationTask
from .factory import register_task


@register_task("anchored_summarization")
class AnchoredSummarizationTask(CodeSummarizationTask):
    """Code summarization with a swappable cross-cycle anchor."""

    #: Anchor used when the config does not name one; reproduces upstream.
    DEFAULT_ANCHOR = "assert"

    def __init__(self, task_config: Dict[str, Any]):
        """Initialize the task and its anchor strategy."""
        super().__init__(task_config)
        self.anchor = create_anchor(
            task_config.get("anchor", self.DEFAULT_ANCHOR), task_config
        )

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Wrap the summary as a docstring and append the configured anchor.

        The wrapper keeps the anchor as the second-to-last line, which is the
        layout both the upstream task and every anchor strategy rely on to
        recover the previous cycle's anchor.

        Args:
            output: The model's summary of the code it just generated.
            context: Loop context. ``cycle_input`` is the prompt this cycle
                started from; ``task_input`` is the code that was summarized,
                which model-derived anchors read.
        """
        anchor_line = self.anchor.build(
            cycle_input=context.get("cycle_input", ""),
            code=context.get("task_input", ""),
            context=context,
        )
        return f"\"\"\"\n{output}\n{anchor_line}\n\"\"\"\n"
