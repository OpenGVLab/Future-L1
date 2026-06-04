"""FutureL1 LLM-as-judge helpers (ported from legacy future_l1_rl judge tools).

Use via::

    from future_l1_rl.judge.api_judge import api_batch_judge
"""

from .api_judge import api_batch_judge  # noqa: F401

__all__ = ["api_batch_judge"]
