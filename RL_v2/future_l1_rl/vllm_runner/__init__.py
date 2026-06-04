"""FutureL1 vLLM runner package.

This package hosts:
  * ``future_l1_gpu_model_runner``: drop-in replacement for
    ``vllm.v1.worker.gpu_model_runner`` (sys.modules-level monkey patch).
  * ``latent_hook`` / ``latent_recorder``: per-step latent emission and the
    driver-side recorder, copied from the HyLar implementation.
"""

__all__ = []
