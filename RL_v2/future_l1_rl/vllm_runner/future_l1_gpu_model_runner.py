# SPDX-License-Identifier: Apache-2.0
"""FutureL1 vLLM v1 GPUModelRunner.

This module is a *drop-in replacement* for ``vllm.v1.worker.gpu_model_runner``:
``future_l1_rl_patch.patch_vllm`` does ``sys.modules[<that name>] = this``.

Implementation strategy (changed from the older HyLar-derived fork):
    * Inherit from the upstream ``GPUModelRunner`` and **only** override the
      three places where FutureL1 semantics differ from vanilla generation.
      That avoids tracking the ~4000-line upstream runner across vLLM
      versions; only the surgical hook points need to stay in sync.
    * Re-export upstream's public surface (via ``from ... import *``) so
      any downstream code that does ``from vllm.v1.worker.gpu_model_runner
      import X`` keeps working.

Three FutureL1 hooks injected into ``execute_model``:
    A. Before ``self.model(...)``: for every request whose latent state is
       "active", replace the last input embedding with the pending hidden
       state captured at the previous step. This implements the
       teacher-forced latent loop, mirroring FutureL1's HF ``_future_l1_sample``.
    B. After ``_bookkeeping_sync`` returns ``valid_sampled_token_ids``: run
       the per-request latent state machine. Detect ``<|latent_start|>`` from
       the model's sampling, force-emit ``<|latent|>`` while active, and
       force-emit ``<|latent_end|>`` when the latent budget is exhausted.
       Also capture ``sample_hidden_states[i]`` (optionally projected) into
       the request's ``pending`` slot for the next step's Hook A.
    C. Emit the per-step latents over the binary-TCP hook so the
       driver-side ``LatentRecorder`` can build the per-trajectory ``z``
       tensors consumed by DePO/LDPO.

Env-controlled state (read once on ``__init__``):
    FUTURE_L1_LATENT_START_ID    : token id of <|latent_start|>
    FUTURE_L1_LATENT_END_ID      : token id of <|latent_end|>
    FUTURE_L1_LATENT_ID          : token id of <|latent|>          (the filler)
    FUTURE_L1_LATENT_SIZE        : max latent positions per span (int)
    FUTURE_L1_APPLY_PROJ_HEAD    : 1 to load the FutureL1 projection head and
                                  apply it to the hidden state before
                                  feeding it back (default off; raw hidden
                                  state still produces a stable rollout)
    FUTURE_L1_LATENT_DEBUG       : 1 to print per-step state transitions
    LATENT_*                    : back-compat aliases for the FUTURE_L1_* vars
"""

print("[FutureL1 RL vllm patch] Initializing future_l1_gpu_model_runner...")

import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Re-export the upstream module so any other vLLM code that imports things
# from vllm.v1.worker.gpu_model_runner keeps working after the sys.modules
# substitution.
# --------------------------------------------------------------------------
import vllm.v1.worker.gpu_model_runner as _upstream
from vllm.v1.worker.gpu_model_runner import *  # noqa: F401, F403

# Explicit upstream symbols we use inside our override.
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner as _UpstreamGPUModelRunner,
    AsyncGPUModelRunnerOutput,
)

# Helpers used inside the replicated ``execute_model`` body.
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,  # noqa: F401  (kept for re-export shape)
    ModelRunnerOutput,
)
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .latent_hook import emit_latents_step


logger = init_logger(__name__)


# ==========================================================================
# Projection-head helpers (kept in this module so the runner is a single
# drop-in replacement). Mirrors ``VideoL1/src/model/projection_head.py``.
# ==========================================================================
class _FutureL1SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        gate = torch.nn.functional.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)


def _build_future_l1_head(kind: str, hidden_dim: int, proj_hidden: int):
    """Build a FutureL1 projection head module by ``kind`` ("lvr" / "swiglu")."""
    if kind == "lvr":
        class _LVRWrap(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.ln_q = nn.LayerNorm(hidden_dim, eps=1e-6)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            def forward(self, x):
                return self.mlp(self.ln_q(x))

        return _LVRWrap(hidden_dim)
    if kind == "swiglu":
        class _SwiGLUWrap(nn.Module):
            def __init__(self, hidden_dim, proj_hidden):
                super().__init__()
                self.projection = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-6),
                    nn.Linear(hidden_dim, proj_hidden),
                    _FutureL1SwiGLU(input_dim=proj_hidden, hidden_dim=proj_hidden),
                    nn.Linear(proj_hidden, hidden_dim),
                )

            def forward(self, x):
                return self.projection(x)

        return _SwiGLUWrap(hidden_dim, proj_hidden)
    return None


def _load_future_l1_head_from_dir(head: nn.Module, model_dir: str, prefix: str) -> int:
    """Scan safetensors / pytorch_model*.bin in ``model_dir`` and load tensors
    whose names start with ``prefix`` into ``head``. Returns number loaded."""
    import glob

    state_dict = {}
    try:
        from safetensors import safe_open  # type: ignore
    except Exception:  # noqa: BLE001
        safe_open = None  # type: ignore

    if os.path.isdir(model_dir):
        if safe_open is not None:
            for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                try:
                    with safe_open(path, framework="pt") as f:
                        for k in f.keys():
                            if k.startswith(prefix):
                                state_dict[k[len(prefix):]] = f.get_tensor(k)
                except Exception as _e:  # noqa: BLE001
                    logger.warning("safetensors load failed for %s: %s", path, _e)
        if not state_dict:
            for path in sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin"))):
                try:
                    blob = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:  # noqa: BLE001
                    continue
                for k, v in blob.items():
                    if k.startswith(prefix):
                        state_dict[k[len(prefix):]] = v
                if state_dict:
                    break

    if not state_dict:
        return 0

    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning("Unexpected FutureL1 head keys: %s", unexpected[:8])
    if missing:
        logger.warning("Missing FutureL1 head keys: %s", missing[:8])
    return len(state_dict)


def _read_int_env(*names: str) -> Optional[int]:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            try:
                return int(v)
            except ValueError:
                continue
    return None


def _debug_enabled() -> bool:
    return (
        os.environ.get("FUTURE_L1_LATENT_DEBUG", os.environ.get("LATENT_DEBUG", "0"))
        == "1"
    )


# ==========================================================================
# FutureL1-aware GPUModelRunner.
# ==========================================================================
class GPUModelRunner(_UpstreamGPUModelRunner):
    """vLLM v1 GPUModelRunner extended with FutureL1's latent state machine.

    Drop-in for ``vllm.v1.worker.gpu_model_runner.GPUModelRunner``: every
    request acquires a small piece of latent state on first sight, and on
    each step we (1) override its last input embedding when active and
    (2) post-process the sampled token to enforce the
    ``<|latent_start|> -> N x <|latent|> -> <|latent_end|>`` token grammar.
    """

    # ------------------------------------------------------------------
    # Initialisation: parse env vars + latent state container.
    # ------------------------------------------------------------------
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self._init_future_l1_state()

    def _init_future_l1_state(self) -> None:
        start_id = _read_int_env("FUTURE_L1_LATENT_START_ID", "LATENT_START_ID")
        end_id = _read_int_env("FUTURE_L1_LATENT_END_ID", "LATENT_END_ID")
        latent_id = _read_int_env("FUTURE_L1_LATENT_ID", "LATENT_ID")

        # ------------------------------------------------------------------
        # Adaptive latent budget, mirroring VideoL1/src/model/future_l1.py:_future_l1_sample.
        #   * fixed_latent_budget=True  -> exactly max_latent_token, ignore the
        #                                   model's own <|latent_end|> sampling
        #   * fixed_latent_budget=False -> early-exit allowed; upper cap is
        #                                   loose_latent_budget if set, else
        #                                   round(max_latent_token * infer_latent_multiplier)
        # Env vars (``FUTURE_L1_FIXED_LATENT_BUDGET`` / ``FUTURE_L1_LATENT_SIZE``)
        # still override hf_config for ablation studies.
        # ------------------------------------------------------------------
        hf_config = getattr(self.model_config, "hf_config", None)
        cfg_fixed = bool(getattr(hf_config, "fixed_latent_budget", False)) if hf_config is not None else False
        cfg_max_token = int(getattr(hf_config, "max_latent_token", 4)) if hf_config is not None else 4
        cfg_loose = getattr(hf_config, "loose_latent_budget", None) if hf_config is not None else None
        cfg_mult = float(getattr(hf_config, "infer_latent_multiplier", 2.0)) if hf_config is not None else 2.0

        env_fixed_raw = os.environ.get("FUTURE_L1_FIXED_LATENT_BUDGET")
        env_fixed: Optional[bool] = None
        if env_fixed_raw is not None and env_fixed_raw != "":
            env_fixed = env_fixed_raw.strip().lower() in ("1", "true", "yes", "y")
        self.fixed_latent_budget: bool = (
            env_fixed if env_fixed is not None else cfg_fixed
        )

        env_size = _read_int_env("FUTURE_L1_LATENT_SIZE", "LATENT_SIZE")
        if env_size is not None and env_size > 0:
            # User pinned a specific size (typically for ablation).
            self.latent_size: int = int(env_size)
            cap_source = f"env (FUTURE_L1_LATENT_SIZE={env_size})"
        elif self.fixed_latent_budget:
            self.latent_size = max(1, int(cfg_max_token))
            cap_source = f"hf_config.max_latent_token={cfg_max_token} (fixed mode)"
        elif cfg_loose is not None and int(cfg_loose) > 0:
            self.latent_size = int(cfg_loose)
            cap_source = f"hf_config.loose_latent_budget={cfg_loose}"
        else:
            self.latent_size = max(1, int(round(cfg_max_token * cfg_mult)))
            cap_source = (
                f"max_latent_token={cfg_max_token} * "
                f"infer_latent_multiplier={cfg_mult} = {self.latent_size}"
            )

        print(
            f"[FutureL1 runner] latent_start_id={start_id} "
            f"latent_end_id={end_id} latent_id={latent_id} "
            f"latent_size={self.latent_size} "
            f"fixed_latent_budget={self.fixed_latent_budget} "
            f"(cap from: {cap_source})",
            flush=True,
        )

        self.latent_start_id: Optional[int] = start_id
        self.latent_end_id: Optional[int] = end_id
        self.latent_id: Optional[int] = latent_id
        self.latent_enabled: bool = (
            self.latent_start_id is not None
            and self.latent_end_id is not None
            and self.latent_id is not None
            and self.latent_size > 0
        )

        # req_id -> {"active": bool, "pending": Tensor|None, "current_len":
        #            int, "just_saw_start": bool}
        self.latent_state: dict = {}

        # Optional projection head plumbing.
        self.apply_proj_head = (
            os.environ.get("FUTURE_L1_APPLY_PROJ_HEAD", "0") == "1"
        )
        self.future_l1_proj_head: Optional[nn.Module] = None

        # Step counter for the TCP latent emitter.
        self._future_l1_step_idx: int = 0

    # ------------------------------------------------------------------
    # Model loading: defer to upstream, then optionally build the
    # FutureL1 projection head from the same checkpoint.
    # ------------------------------------------------------------------
    def load_model(self, eep_scale_up: bool = False) -> None:
        super().load_model(eep_scale_up=eep_scale_up)
        try:
            self._maybe_build_future_l1_proj_head()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "FutureL1 projection head not loaded (using raw "
                "last_hidden_state during latent rollout): %s", e
            )
            self.future_l1_proj_head = None

    def _maybe_build_future_l1_proj_head(self) -> None:
        if not self.apply_proj_head:
            self.future_l1_proj_head = None
            return
        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config is None:
            self.future_l1_proj_head = None
            return
        if not bool(getattr(hf_config, "use_projection_head", False)):
            self.future_l1_proj_head = None
            return

        try:
            hidden_dim = hf_config.text_config.hidden_size
        except AttributeError:
            hidden_dim = int(getattr(hf_config, "hidden_size", 3584))
        proj_hidden = int(getattr(hf_config, "projection_hidden_dim", 2048))
        head_type = str(getattr(hf_config, "projection_head_type", "swiglu")).lower()
        use_dual = bool(getattr(hf_config, "use_dual_projection_heads", False))
        prefer = str(getattr(hf_config, "infer_head_prefer", "render")).lower()

        head = _build_future_l1_head(head_type, hidden_dim, proj_hidden)
        if head is None:
            self.future_l1_proj_head = None
            return

        prefix = (
            ("projection_head_render." if prefer == "render"
             else "projection_head_orig.")
            if use_dual
            else "projection_head."
        )
        loaded = _load_future_l1_head_from_dir(head, self.model_config.model, prefix)
        if loaded == 0:
            logger.warning(
                "FutureL1 projection head weights not found under prefix %r "
                "in %s; using raw hidden state.", prefix, self.model_config.model
            )
            self.future_l1_proj_head = None
            return

        head.to(self.device, dtype=self.dtype).eval()
        for p in head.parameters():
            p.requires_grad = False
        self.future_l1_proj_head = head
        logger.info(
            "Loaded FutureL1 projection head: kind=%s dual=%s prefer=%s "
            "loaded_keys=%d device=%s",
            head_type, use_dual, prefer, loaded, self.device,
        )

    @torch.no_grad()
    def _future_l1_project(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.future_l1_proj_head is None:
            return hidden
        h = hidden.to(self.device, dtype=self.dtype, non_blocking=True)
        return self.future_l1_proj_head(h).detach()

    # ------------------------------------------------------------------
    # FutureL1 Hook A: per-step inputs_embeds override.
    # ------------------------------------------------------------------
    def _future_l1_apply_inputs_embeds_override(
        self, inputs_embeds: Optional[torch.Tensor]
    ) -> None:
        if inputs_embeds is None or not self.latent_enabled:
            return
        if self.speculative_config:
            return
        pp = get_pp_group()
        ranks = getattr(pp, "ranks", [])
        # PP > 1 path is rare for FutureL1 RL and the cross-PP "inbox"
        # forwarding adds significant complexity. We restrict the latent
        # override to the single-PP case (which covers our 8x TP setup).
        if not (pp.is_last_rank and len(ranks) == 1):
            return

        num_reqs = self.input_batch.num_reqs
        query_start_loc_cpu = (
            self.query_start_loc_cpu
            if hasattr(self, "query_start_loc_cpu")
            else self.query_start_loc.cpu
        )
        rows = query_start_loc_cpu[: num_reqs + 1][1:] - 1
        rows_cpu = rows.to(device="cpu", dtype=torch.int64)
        override_indices: list[int] = []
        override_embeds: list[torch.Tensor] = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            st = self.latent_state.get(req_id)
            if st and st.get("active") and st.get("pending") is not None:
                override_indices.append(int(rows_cpu[i].item()))
                override_embeds.append(st["pending"])
        if not override_indices:
            return

        idx = torch.tensor(override_indices, device=self.device, dtype=torch.int64)
        embeds = torch.stack(override_embeds, dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        inputs_embeds.index_copy_(0, idx, embeds)
        # Pending consumed; will be refreshed in Hook B if still active.
        for req_id in self.input_batch.req_ids:
            st = self.latent_state.get(req_id)
            if st and st.get("active"):
                st["pending"] = None
        if _debug_enabled():
            print(f"[FUTURE_L1][A] override {len(override_indices)} rows", flush=True)

    # ------------------------------------------------------------------
    # FutureL1 Hook B: state machine + token forcing post-sample.
    # ------------------------------------------------------------------
    def _future_l1_step_state_machine(
        self,
        valid_sampled_token_ids: list,
        sample_hidden_states: Optional[torch.Tensor],
    ) -> list:
        latents_step: list = [None] * len(self.input_batch.req_ids)
        if not (self.latent_enabled and not self.speculative_config):
            return latents_step
        if not get_pp_group().is_last_rank:
            return latents_step

        last_h = sample_hidden_states  # (num_reqs, H)
        for i, req_id in enumerate(self.input_batch.req_ids):
            st = self.latent_state.setdefault(
                req_id,
                {"active": False, "pending": None, "current_len": 0,
                 "just_saw_start": False},
            )
            gen_ids = (
                valid_sampled_token_ids[i]
                if i < len(valid_sampled_token_ids) else []
            )

            if st["active"]:
                sampled_tid = gen_ids[0] if gen_ids else None
                # Mirror VideoL1/src/model/future_l1.py:_future_l1_sample's
                # latent-exit rule:
                #   * fixed_latent_budget=True  -> only force-exit on budget
                #     exhaustion (model's <|latent_end|> sampling is ignored)
                #   * fixed_latent_budget=False -> early exit on model emitting
                #     <|latent_end|>, AND force-exit on budget
                force_end_on_budget = st["current_len"] >= self.latent_size
                model_emitted_end = (
                    not self.fixed_latent_budget
                    and sampled_tid == self.latent_end_id
                )
                if force_end_on_budget or model_emitted_end:
                    # End the latent span; force-emit <|latent_end|>.
                    old_len = st["current_len"]
                    st["active"] = False
                    st["pending"] = None
                    st["current_len"] = 0
                    st["just_saw_start"] = False
                    valid_sampled_token_ids[i] = [self.latent_end_id]
                    if _debug_enabled():
                        reason = "budget" if force_end_on_budget else "model_end"
                        print(f"[FUTURE_L1][B] end req={req_id} len={old_len} reason={reason}", flush=True)
                else:
                    # Continue: capture h, force-emit <|latent|>.
                    if last_h is not None and i < last_h.shape[0]:
                        mu = self._future_l1_project(last_h[i].detach())
                        st["pending"] = mu
                        st["current_len"] += 1
                        latents_step[i] = mu.detach().to(device="cpu", dtype=torch.float16)
                        valid_sampled_token_ids[i] = [self.latent_id]
                        if _debug_enabled():
                            print(f"[FUTURE_L1][B] continue req={req_id} len={st['current_len']}", flush=True)
            elif st["just_saw_start"]:
                # The previous step sampled <|latent_start|>; activate now.
                st["active"] = True
                st["just_saw_start"] = False
                if last_h is not None and i < last_h.shape[0]:
                    mu = self._future_l1_project(last_h[i].detach())
                    st["pending"] = mu
                    st["current_len"] = 1
                    latents_step[i] = mu.detach().to(device="cpu", dtype=torch.float16)
                    valid_sampled_token_ids[i] = [self.latent_id]
                    if _debug_enabled():
                        print(f"[FUTURE_L1][B] activate req={req_id}", flush=True)
            else:
                # Watch for the model freely sampling <|latent_start|>.
                for tid in gen_ids:
                    if tid == self.latent_start_id:
                        st["just_saw_start"] = True
                        if _debug_enabled():
                            print(f"[FUTURE_L1][B] saw start req={req_id}", flush=True)
                        break
        return latents_step

    # ------------------------------------------------------------------
    # FutureL1 Hook C: emit per-step latents to the TCP listener so the
    # driver-side ``LatentRecorder`` can stitch per-trajectory ``z``.
    # ------------------------------------------------------------------
    def _future_l1_emit_latents(self, latents_step: list) -> None:
        if not get_pp_group().is_last_rank:
            return
        try:
            emit_latents_step(
                req_ids=list(self.input_batch.req_ids),
                latents=latents_step,
                extra={"step": int(self._future_l1_step_idx)},
            )
        except Exception as _e:  # noqa: BLE001
            if _debug_enabled():
                print(f"[FUTURE_L1][C] emit failed: {_e}", flush=True)
        self._future_l1_step_idx += 1

    # ------------------------------------------------------------------
    # ``execute_model`` re-implementation: verbatim from vLLM 0.11 with
    # three FutureL1 hooks inserted (see file docstring).
    # ------------------------------------------------------------------
    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncGPUModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                self._update_states(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, please disable it "
                        "when the requests need prompt logprobs"
                    )

                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len, ubatch_slices, num_tokens_after_padding
                 ) = self._prepare_inputs(scheduler_output)

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(
                scheduler_output, intermediate_tensors,
                ubatch_slices, num_tokens_after_padding,
            )

            uniform_decode = (
                max_query_len == self.uniform_decode_query_len
            ) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode
            )
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(batch_descriptor)
            )

        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        # === FutureL1 Hook A: override last-position embed for active reqs ===
        self._future_l1_apply_inputs_embeds_override(inputs_embeds)

        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("Postprocess"):
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                if not get_pp_group().is_last_rank:
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states
                if self.is_pooling_model:
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                assert not self.is_pooling_model
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                model_output_broadcast_data = (
                    get_pp_group().broadcast_tensor_dict(
                        model_output_broadcast_data,
                        src=len(get_pp_group().ranks) - 1,
                    )
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            if scheduler_output.grammar_bitmask is not None:
                apply_grammar_bitmask(
                    scheduler_output, self.input_batch, logits, self.device
                )

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        use_padded_batch_for_eagle = (
            self.speculative_config
            and self.speculative_config.use_eagle()
            and not self.speculative_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            self.speculative_config
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len
            )
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.seq_lens.max()
            + self.speculative_config.num_speculative_tokens
            <= effective_drafter_max_model_len
        )
        if use_padded_batch_for_eagle and input_fits_in_drafter:
            propose_draft_token_ids(sampler_output.sampled_token_ids)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output, sampler_output, logits, hidden_states,
                num_scheduled_tokens,
            )

        # === FutureL1 Hook B: latent state machine + token forcing ===
        try:
            latents_step = self._future_l1_step_state_machine(
                valid_sampled_token_ids, sample_hidden_states
            )
        except Exception as _e:  # noqa: BLE001
            # Never break sampling on bookkeeping errors; latent will degrade.
            logger.warning("FutureL1 state machine error: %s", _e)
            latents_step = [None] * len(self.input_batch.req_ids)

        if (self.speculative_config and not use_padded_batch_for_eagle
                and input_fits_in_drafter):
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        # === FutureL1 Hook C: emit per-step latents to LatentRecorder ===
        self._future_l1_emit_latents(latents_step)

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )


print("[FutureL1 RL vllm patch] future_l1_gpu_model_runner ready.", flush=True)
