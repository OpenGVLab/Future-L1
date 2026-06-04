"""FutureL1 transformers monkey-patch dispatcher.

Re-exports the per-backbone forward replacements from
``VideoL1/src/train/monkey_patch_forward.py`` and applies them based on the
``FUTURE_L1_BACKBONE_MODEL_TYPE`` env var.

Layout assumption: ``FUTURE_L1_CODE_ROOT`` (or the auto-detected VideoL1 root)
must be on ``sys.path`` before this module is imported. The top-level
``future_l1_rl_patch.py`` handles that.
"""

from __future__ import annotations

import os
import sys


def apply(model_type: str | None = None) -> None:
    """Apply FutureL1 transformers patches for the given backbone.

    ``model_type`` is one of {"qwen2_5_vl", "qwen3_vl", "qwen3_5"}. If None,
    falls back to env var ``FUTURE_L1_BACKBONE_MODEL_TYPE``. If still unknown,
    patches all supported backbones (cheap; only one will be used at runtime).
    """
    if model_type is None:
        model_type = os.environ.get("FUTURE_L1_BACKBONE_MODEL_TYPE")

    # Import here so an unrelated import failure does not break package loading.
    from src.train.monkey_patch_forward import (  # noqa: PLC0415
        replace_qwen2_5_vl_generation_forward,
        replace_qwen2_5_with_mixed_modality_forward,
        replace_qwen3_vl_generation_forward,
        replace_qwen3_with_mixed_modality_forward,
    )

    def _patch_qwen3_vl() -> None:
        replace_qwen3_with_mixed_modality_forward()
        replace_qwen3_vl_generation_forward()
        print("[FutureL1 patch] qwen3_vl forwards replaced.", file=sys.stderr)

    def _patch_qwen2_5_vl() -> None:
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vl_generation_forward()
        print("[FutureL1 patch] qwen2_5_vl forwards replaced.", file=sys.stderr)

    def _patch_qwen3_5() -> None:
        # Both import-time *and* call-time can fail when transformers is too
        # old for Qwen3.5 (VideoL1's monkey patch raises ImportError lazily
        # inside the replace_qwen3_5_* helpers as well).
        try:
            from src.train.monkey_patch_forward import (  # noqa: PLC0415
                replace_qwen3_5_generation_forward,
                replace_qwen3_5_with_mixed_modality_forward,
            )
            replace_qwen3_5_with_mixed_modality_forward()
            replace_qwen3_5_generation_forward()
            print("[FutureL1 patch] qwen3_5 forwards replaced.", file=sys.stderr)
        except ImportError as e:
            print(
                "[FutureL1 patch] qwen3_5 backbone not available in this "
                f"transformers install ({e}); skipping.",
                file=sys.stderr,
            )

    if model_type == "qwen3_vl":
        _patch_qwen3_vl()
    elif model_type == "qwen2_5_vl":
        _patch_qwen2_5_vl()
    elif model_type == "qwen3_5":
        _patch_qwen3_5()
    else:
        # Unknown / not specified: patch the families we know about.
        _patch_qwen3_vl()
        _patch_qwen2_5_vl()
        _patch_qwen3_5()
