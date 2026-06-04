"""FutureL1 RL runtime package.

Importing this module does not patch anything; call ``future_l1_rl_patch.patch``
(at the repo root) to apply transformers + vLLM patches before
``verl.trainer.main`` constructs models.
"""

__all__ = []
