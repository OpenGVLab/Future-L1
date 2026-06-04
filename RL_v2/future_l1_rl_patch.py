"""FutureL1 RL runtime patch entrypoint.

Apply transformers + vLLM patches required by FutureL1's interleaved
latent/text generation before vLLM workers and HF models are instantiated.

Usage (imported at the top of ``verl.trainer.main``)::

    import future_l1_rl_patch  # side-effect: applies patches

Required env vars:
    FUTURE_L1_LATENT_START_ID   -- token id of <|latent_start|>
    FUTURE_L1_LATENT_END_ID     -- token id of <|latent_end|>
    FUTURE_L1_LATENT_ID         -- token id of <|latent|>
    FUTURE_L1_LATENT_SIZE       -- max number of latent tokens per span

Optional:
    FUTURE_L1_BACKBONE_MODEL_TYPE  -- one of qwen3_vl, qwen2_5_vl, qwen3_5.
        Auto-patches all if unset.
    FUTURE_L1_CODE_ROOT         -- absolute path to the Future-L1 repo root.
        Defaults to the parent of this file's parent (i.e. ``Future-L1/``).
    FUTURE_L1_APPLY_PROJ_HEAD   -- "1" to load and apply the projection head
        inside the vLLM runner during latent rollout (matches HF generate).
"""

from __future__ import annotations

import importlib
import os
import sys


def _default_code_root() -> str:
    # This file lives in Future-L1/RL_v2/. Code root is the Future-L1 directory.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _ensure_paths() -> str:
    legacy_root = os.environ.get("SWIMBIRD_CODE_ROOT", "").strip()
    code_root = os.environ.get("FUTURE_L1_CODE_ROOT") or legacy_root or _default_code_root()
    rl_v2_root = os.path.abspath(os.path.dirname(__file__))
    for path in (rl_v2_root, code_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["FUTURE_L1_CODE_ROOT"] = code_root
    return code_root


def patch_transformers() -> None:
    _ensure_paths()
    from future_l1_rl import transformers_patch  # noqa: PLC0415

    transformers_patch.apply(os.environ.get("FUTURE_L1_BACKBONE_MODEL_TYPE"))


def _cuda_visible_to_this_process() -> bool:
    """Cheap, side-effect-free check whether the current process has a CUDA GPU.

    Importing the FutureL1 vLLM runner transitively imports
    ``vllm.v1.worker.gpu_model_runner``, which eagerly probes CUDA at module
    load time (FA3 capability check). That raises
    ``RuntimeError: No CUDA GPUs are available`` on legitimate CPU-only Ray
    processes — most notably the driver-side ``TemporaryActor`` that Ray
    spawns to validate the ``Runner`` class can be imported. Those CPU-only
    actors do not need the FutureL1 vLLM patch (they never run vLLM); the
    real FSDPWorker actors, each with a GPU, will replace the runner on
    their own. We therefore probe for GPU visibility here and skip the
    runner replacement (without raising) when none is found.

    We do *not* call ``torch.cuda.is_available()`` directly because under
    PyTorch >= 2.4 it still triggers a partial init that prints scary
    warnings on CPU-only nodes. ``CUDA_VISIBLE_DEVICES`` is the same signal
    Ray uses internally to scope GPU access, so checking it is sufficient.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        # Variable unset — defer to torch. Importing torch here is cheap; it
        # was already imported via transformers in patch_transformers().
        try:
            import torch  # noqa: PLC0415

            return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
        except Exception:  # noqa: BLE001
            return False
    stripped = cvd.strip()
    if stripped == "" or stripped == "-1":
        return False
    return True


def patch_vllm() -> bool:
    """Substitute vLLM's v1 GPU model runner with the FutureL1-aware one.

    The FutureL1 runner is REQUIRED for every HyLar-equivalent baseline
    (GRPO / DAPO / DePO / LDPO / CAVA — all using
    ``sampling_strategy=future_l1_depo``). It is what makes the latent state
    machine execute correctly during rollout (force `<|latent|>` token,
    feed projected hidden states back as the next-step input embedding) AND
    what streams per-step latent vectors back to the driver via the
    `LatentRecorder` TCP hook.

    Returns ``True`` if the runner replacement actually happened, ``False``
    if it was deliberately skipped on a CPU-only process (see
    ``_cuda_visible_to_this_process`` for why this is the right thing to do
    inside Ray's TemporaryActor / driver). Any *other* failure (e.g. vLLM
    API drift) is still escalated when ``FUTURE_L1_REQUIRE_VLLM_RUNNER=1``.
    """
    # Use vLLM v1 worker so the latent runner can take over.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    # Always use binary TCP hook (more efficient than JSON).
    os.environ.setdefault("AVT_LATENT_HOOK_BIN", "1")

    # Propagate FutureL1 latent ids to LATENT_* aliases so any legacy code path
    # that reads those still works.
    for src, dst in (
        ("FUTURE_L1_LATENT_START_ID", "LATENT_START_ID"),
        ("FUTURE_L1_LATENT_END_ID", "LATENT_END_ID"),
        ("FUTURE_L1_LATENT_ID", "LATENT_ID"),
        ("FUTURE_L1_LATENT_SIZE", "LATENT_SIZE"),
    ):
        if os.environ.get(src) and not os.environ.get(dst):
            os.environ[dst] = os.environ[src]

    if not _cuda_visible_to_this_process():
        # This is the expected path for Ray's CPU-only driver / TemporaryActor.
        # It is NOT a degradation: the actual GPU FSDPWorker actors will run
        # this same patch on their own processes (with their GPUs visible) and
        # do the substitution there. Surface a clear log line so anyone
        # reading the output can tell driver-skip from real-failure.
        print(
            "[FutureL1 RL patch] no GPU visible to this process "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
            "skipping vLLM GPU runner replacement here. The patch will be "
            "applied independently inside each GPU FSDPWorker actor.",
            file=sys.stderr,
        )
        return False

    require_runner = os.environ.get("FUTURE_L1_REQUIRE_VLLM_RUNNER", "0") == "1"
    try:
        sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
            "future_l1_rl.vllm_runner.future_l1_gpu_model_runner"
        )
        print(
            "[FutureL1 RL patch] vLLM GPU model runner replaced.",
            file=sys.stderr,
        )
        return True
    except Exception as e:  # noqa: BLE001
        msg = (
            "[FutureL1 RL patch] FutureL1 GPU model runner failed to load "
            f"({type(e).__name__}: {e}). Per-step latent capture is "
            "UNAVAILABLE; HyLar-equivalent GRPO/DAPO/DePO baselines (which all "
            "use sampling_strategy=future_l1_depo) will silently DEGRADE to "
            "stock-vLLM rollout without latent recording. Only the ablation-only "
            "future_l1_text / default strategies are safe in this state. Set "
            "FUTURE_L1_REQUIRE_VLLM_RUNNER=1 to escalate this to a hard error "
            "(required for any HyLar baseline run)."
        )
        if require_runner:
            raise RuntimeError(msg) from e
        print(msg, file=sys.stderr)
        return False


def patch() -> None:
    code_root = _ensure_paths()
    print(
        f"[FutureL1 RL patch] code_root={code_root} "
        f"backbone={os.environ.get('FUTURE_L1_BACKBONE_MODEL_TYPE')}",
        file=sys.stderr,
    )
    patch_transformers()
    vllm_applied = patch_vllm()
    if vllm_applied:
        print(
            "[FutureL1 RL patch] transformers + vLLM patches applied.",
            file=sys.stderr,
        )
    else:
        # Either no GPU on this process (driver / TemporaryActor — normal) or
        # the future_l1 runner module failed to import in non-strict mode.
        # patch_vllm() has already printed the precise reason.
        print(
            "[FutureL1 RL patch] transformers patches applied; vLLM runner "
            "replacement skipped on this process (see preceding log line).",
            file=sys.stderr,
        )


# Apply patches on import for legacy launchers that just `import future_l1_rl_patch`.
def _normalize_legacy_env() -> None:
    aliases = {
        "SWIMBIRD_CODE_ROOT": "FUTURE_L1_CODE_ROOT",
        "SWIMBIRD_RL_PATCH": "FUTURE_L1_RL_PATCH",
        "SWIMBIRD_RL_PATCH_REQUIRED": "FUTURE_L1_RL_PATCH_REQUIRED",
        "SWIMBIRD_LATENT_START_ID": "FUTURE_L1_LATENT_START_ID",
        "SWIMBIRD_LATENT_END_ID": "FUTURE_L1_LATENT_END_ID",
        "SWIMBIRD_LATENT_ID": "FUTURE_L1_LATENT_ID",
    }
    for legacy, new in aliases.items():
        if not os.environ.get(new) and os.environ.get(legacy):
            os.environ[new] = os.environ[legacy]


_normalize_legacy_env()
patch()
