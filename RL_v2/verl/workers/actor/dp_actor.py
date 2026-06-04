# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

# Belt-and-suspenders: in case fsdp_workers wasn't the first thing imported in
# this worker process, also apply the FutureL1 vLLM runner patch here. Same
# pattern as HyLar/RL/verl/workers/actor/dp_actor.py:17.
import os
import sys as _sys

if os.environ.get("FUTURE_L1_RL_PATCH", "1") != "0":
    _rl_v2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    if _rl_v2_root not in _sys.path:
        _sys.path.insert(0, _rl_v2_root)
    try:
        import future_l1_rl_patch  # noqa: F401  (side-effect: monkey patches)
    except Exception as _e:  # noqa: BLE001
        _msg = f"[verl.workers.actor.dp_actor] FutureL1 patch skipped: {_e}"
        # See verl/trainer/main.py for the rationale: any silent fallback to
        # stock vLLM breaks the latent-recording + vMF substitution that all
        # HyLar-equivalent baselines depend on. Escalate when the launcher set
        # FUTURE_L1_RL_PATCH_REQUIRED=1.
        if os.environ.get("FUTURE_L1_RL_PATCH_REQUIRED", "0") == "1":
            raise RuntimeError(_msg) from _e
        print(_msg, file=_sys.stderr)

from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import (
    average_loss,
    compute_kl,
    compute_latent_vmf_kl,
    compute_policy_loss,
)
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


# ----------------------------------------------------------------------
# FutureL1 helpers: locate latent-token positions inside a response and
# build masks aligned with rollout `z` vectors.
# ----------------------------------------------------------------------
def _future_l1_collect_varlen_segments(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_id: int,
    end_id: int,
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Collect *open-interval* latent positions ``(s_pos, e_pos)`` and map them
    to indices on the unpadded varlen stream (i.e. the ``[1, total_nnz]``
    sequence after ``unpad_input`` + transpose).

    Returns:
        ``(concat_indices, per_batch_segments)`` where ``concat_indices`` is a
        ``(K,)`` LongTensor of all kept varlen positions (or empty), and
        ``per_batch_segments[b]`` is the list of per-segment LongTensors for
        sample ``b`` (each tensor is the inner positions of one latent span).
    """
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    mask = attention_mask.to(dtype=torch.long)
    mask_flat = mask.reshape(-1)
    prefix = torch.cumsum(mask_flat, dim=0)

    flat_indices: list[torch.Tensor] = []
    per_batch_segments: list[list[torch.Tensor]] = []
    for b in range(bsz):
        segments: list[torch.Tensor] = []
        row_ids = input_ids[b]
        row_mask = mask[b]

        starts = (row_ids == start_id).nonzero(as_tuple=False).squeeze(-1)
        ends = (row_ids == end_id).nonzero(as_tuple=False).squeeze(-1)
        if starts.numel() == 0 or ends.numel() == 0:
            per_batch_segments.append(segments)
            continue

        i_ptr, j_ptr = 0, 0
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = int(starts[i_ptr].item())
            while j_ptr < ends.numel() and int(ends[j_ptr].item()) <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = int(ends[j_ptr].item())
            i_ptr += 1
            j_ptr += 1

            if e_pos <= s_pos + 1:
                continue
            inner = torch.arange(s_pos + 1, e_pos, device=device, dtype=torch.long)
            inner_valid = inner[row_mask[inner] == 1]
            if inner_valid.numel() == 0:
                continue
            flat_pos = b * seqlen + inner_valid
            var_idx = prefix[flat_pos] - 1
            flat_indices.append(var_idx)
            segments.append(var_idx)
        per_batch_segments.append(segments)

    if not flat_indices:
        return torch.empty(0, dtype=torch.long, device=device), per_batch_segments
    return torch.cat(flat_indices, dim=0), per_batch_segments


def _future_l1_build_latent_mask(
    input_ids: torch.Tensor,
    response_length: int,
    start_id: int,
    end_id: int,
) -> torch.Tensor:
    """Build a ``(bsz, response_length)`` bool mask: True at positions strictly
    inside a ``<|latent_start|>...<|latent_end|>`` span.

    The marker positions themselves (``latent_start`` and ``latent_end``) are
    treated as regular tokens whose log-probs participate in the token PPO
    surrogate, matching HyLar's build_latent_mask.
    """
    bsz, seqlen = input_ids.shape
    device = input_ids.device
    response_ids = input_ids[:, -response_length:]
    mask = torch.zeros(bsz, response_length, dtype=torch.bool, device=device)

    for b in range(bsz):
        row = response_ids[b]
        starts = (row == start_id).nonzero(as_tuple=False).squeeze(-1)
        ends = (row == end_id).nonzero(as_tuple=False).squeeze(-1)
        if starts.numel() == 0 or ends.numel() == 0:
            continue
        i_ptr, j_ptr = 0, 0
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = int(starts[i_ptr].item())
            while j_ptr < ends.numel() and int(ends[j_ptr].item()) <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = int(ends[j_ptr].item())
            if e_pos > s_pos + 1:
                mask[b, s_pos + 1 : e_pos] = True
            i_ptr += 1
            j_ptr += 1
    return mask


def _future_l1_latent_log_probs(
    latent_poss: torch.Tensor,
    latents: torch.Tensor,
    last_hidden: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Closed-form vMF log-prob at latent positions: ``kappa * <mu, z>``."""
    if last_hidden.dim() == 2:
        mu = last_hidden[latent_poss, :]
    else:
        mu = last_hidden[0, latent_poss, :]
    z = latents.to(mu.device, mu.dtype)
    return (kappa * (mu.float() * z.float()).sum(dim=-1)).to(mu.dtype)




def _future_l1_filter_latents_by_sample_gate(
    latent_mu: torch.Tensor,
    z_aligned: torch.Tensor,
    segment_groups: Optional[list[list[int]]],
    sample_gate: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[list[list[int]]]]:
    """Keep latent rows whose sample-level gate is true.

    ``latent_mu`` / ``z_aligned`` are concatenated in the same sample/span order
    as ``segment_groups``. This helper lets format-gated DePO remove malformed
    samples from latent-only auxiliary terms while keeping the token PPO update.
    """
    if sample_gate is None or not segment_groups:
        return latent_mu, z_aligned, segment_groups

    kept_mu: list[torch.Tensor] = []
    kept_z: list[torch.Tensor] = []
    kept_groups: list[list[int]] = []
    offset = 0
    gate = sample_gate.to(device=latent_mu.device, dtype=torch.bool)
    for sample_idx, group in enumerate(segment_groups):
        group_len = sum(int(v) for v in group)
        next_offset = offset + group_len
        if sample_idx < gate.numel() and bool(gate[sample_idx].item()) and group_len > 0:
            kept_mu.append(latent_mu[offset:next_offset])
            kept_z.append(z_aligned[offset:next_offset])
            kept_groups.append([int(v) for v in group])
        offset = next_offset

    if offset != latent_mu.shape[0] or not kept_mu:
        empty_mu = latent_mu[:0]
        empty_z = z_aligned[:0]
        return empty_mu, empty_z, []
    return torch.cat(kept_mu, dim=0), torch.cat(kept_z, dim=0), kept_groups


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _is_future_l1_depo(self) -> bool:
        strategy = getattr(self.config, "sampling_strategy", "default")
        return strategy in ("future_l1_depo", "swimbird_depo")

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
        return_latent_mu: bool = False,
    ):
        """Run a single actor forward, optionally with FutureL1 latent log-probs.

        Returns:
            * ``log_probs``: ``(bs, response_len)``.
            * If ``return_latent_mu=True``: ``(log_probs, latent_mu, z_aligned)``
              where ``latent_mu`` is the actor's hidden states at latent
              positions ``(L, D)`` (gradient-enabled if called inside an
              autograd context) and ``z_aligned`` is the rolled-out z vectors
              concatenated in the same order; both ``None`` if no latents.
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        # --- FutureL1: prepare aligned latent positions + z vectors -------
        latent_poss: Optional[torch.Tensor] = None
        latents_concat: Optional[torch.Tensor] = None
        latent_segment_groups: Optional[list[list[int]]] = None
        is_future_l1 = self._is_future_l1_depo()
        latent_start_env = (
            os.environ.get("FUTURE_L1_LATENT_START_ID")
            or os.environ.get("SWIMBIRD_LATENT_START_ID")
            or os.environ.get("LATENT_START_ID")
        )
        latent_end_env = (
            os.environ.get("FUTURE_L1_LATENT_END_ID")
            or os.environ.get("SWIMBIRD_LATENT_END_ID")
            or os.environ.get("LATENT_END_ID")
        )
        if is_future_l1 and "latents" in micro_batch and latent_start_env and latent_end_env:
            try:
                start_id = int(latent_start_env)
                end_id = int(latent_end_env)
                flat_indices, per_batch_segments = _future_l1_collect_varlen_segments(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_id=start_id,
                    end_id=end_id,
                )
                latents_list: list[torch.Tensor] = []
                kept_segment_indices: list[torch.Tensor] = []
                kept_segment_groups: list[list[int]] = []
                for i, lat in enumerate(micro_batch["latents"]):
                    if lat is None:
                        continue
                    t = torch.as_tensor(lat)  # (steps, D)
                    segments = per_batch_segments[i] if i < len(per_batch_segments) else []
                    poss_cnt = sum(int(v.numel()) for v in segments)
                    if poss_cnt == 0:
                        continue
                    if t.shape[0] != poss_cnt:
                        # Token / latent count mismatch: skip this sample's
                        # latent gradients but keep its token PPO term.
                        continue
                    latents_list.append(t)
                    kept_segment_indices.extend(segments)
                    kept_segment_groups.append([int(seg.numel()) for seg in segments])
                if latents_list and kept_segment_indices:
                    latent_poss = torch.cat(kept_segment_indices, dim=0)
                    latents_concat = torch.cat(latents_list, dim=0).to(input_ids.device)
                    if latents_concat.shape[0] != latent_poss.shape[0]:
                        latent_poss, latents_concat, latent_segment_groups = None, None, None
                    else:
                        latent_segment_groups = kept_segment_groups
            except Exception:  # noqa: BLE001
                latent_poss, latents_concat, latent_segment_groups = None, None, None

        # When DePO is on we need hidden states from the last layer to compute
        # the vMF latent log-prob (and optionally its KL).
        output_hidden_states = latent_poss is not None and latents_concat is not None

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # --- FutureL1 latent log-prob (varlen, padding_free path) ----
            latent_mu = None
            if output_hidden_states and latent_poss is not None and latents_concat is not None:
                if output.hidden_states is None:
                    raise RuntimeError(
                        "FutureL1 latent log-prob requires hidden states, but actor forward returned None. "
                        "Check the Qwen-VL monkey patch in verl/models/transformers/*_vl.py."
                    )
                last_hs = output.hidden_states[-1]
                vmf_lp = _future_l1_latent_log_probs(
                    latent_poss=latent_poss,
                    latents=latents_concat,
                    last_hidden=last_hs,
                    kappa=float(getattr(self.config, "future_l1_rl_kappa", 0.01)),
                )
                log_probs[latent_poss] = vmf_lp.to(log_probs.dtype)
                if return_latent_mu:
                    if last_hs.dim() == 2:
                        mu_raw = last_hs[latent_poss, :]
                    else:
                        mu_raw = last_hs[0, latent_poss, :]
                    latent_mu = mu_raw if torch.is_grad_enabled() else mu_raw.detach().clone()

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            latent_mu = None
            # In non-padding_free we re-locate latent positions on the *response*
            # part for the vMF substitution (the rest of the path mirrors above).
            if output_hidden_states and latent_poss is not None and latents_concat is not None:
                # latent_poss live on the varlen stream; in dense mode we use
                # response-local positions instead.
                start_id = int(os.environ.get("FUTURE_L1_LATENT_START_ID") or os.environ["LATENT_START_ID"])
                end_id = int(os.environ.get("FUTURE_L1_LATENT_END_ID") or os.environ["LATENT_END_ID"])
                lat_mask = _future_l1_build_latent_mask(input_ids, response_length, start_id, end_id)
                if lat_mask.any():
                    flat_mask = lat_mask.reshape(-1)
                    if output.hidden_states is None:
                        raise RuntimeError(
                            "FutureL1 latent log-prob requires hidden states, but actor forward returned None. "
                            "Check the Qwen-VL monkey patch in verl/models/transformers/*_vl.py."
                        )
                    last_hs = output.hidden_states[-1]
                    if last_hs.dim() == 2:
                        # Should not happen in dense path; guard anyway.
                        flat_hs = last_hs
                    else:
                        flat_hs = last_hs.reshape(-1, last_hs.shape[-1])
                    mu_raw = flat_hs[flat_mask.nonzero(as_tuple=False).squeeze(-1), :]
                    z = latents_concat.to(mu_raw.device, mu_raw.dtype)
                    if mu_raw.shape[0] == z.shape[0] and mu_raw.shape[0] > 0:
                        kappa = float(getattr(self.config, "future_l1_rl_kappa", 0.01))
                        vmf_lp = kappa * (mu_raw.float() * z.float()).sum(dim=-1)
                        log_probs.view(-1)[flat_mask] = vmf_lp.to(log_probs.dtype)
                        if return_latent_mu:
                            latent_mu = mu_raw if torch.is_grad_enabled() else mu_raw.detach().clone()

        if return_latent_mu:
            return (
                log_probs,
                latent_mu,
                (latents_concat.detach() if latents_concat is not None else None),
                latent_segment_groups,
            )
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        if self._is_future_l1_depo() and "latents" in data.non_tensor_batch:
            non_tensor_select_keys.append("latents")

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0 and not getattr(self.config, "disable_tqdm", False):
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        if "reward_format_scores" in data.batch.keys():
            select_keys.append("reward_format_scores")
        non_tensor_select_keys = ["multi_modal_inputs"]
        if self._is_future_l1_depo() and "latents" in data.non_tensor_batch:
            non_tensor_select_keys.append("latents")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        # Resolve FutureL1 ids for the optional decoupled-PPO latent mask.
        is_depo = (
            self._is_future_l1_depo()
            and bool(getattr(self.config, "enable_decoupled_hybrid_ppo", False))
        )
        latent_start_id = None
        latent_end_id = None
        if is_depo:
            try:
                latent_start_id = int(os.environ.get("FUTURE_L1_LATENT_START_ID") or os.environ["LATENT_START_ID"])
                latent_end_id = int(os.environ.get("FUTURE_L1_LATENT_END_ID") or os.environ["LATENT_END_ID"])
            except (KeyError, ValueError, TypeError):
                # Missing env vars: silently fall back to vanilla PPO loss.
                is_depo = False
        use_vmf_kl = bool(getattr(self.config, "enable_latent_vmf_kl", False))
        use_format_gate = bool(getattr(self.config, "enable_format_gated_latent_loss", False))

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0 and not getattr(self.config, "disable_tqdm", False):
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0 and not getattr(self.config, "disable_tqdm", False):
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    response_length = model_inputs["responses"].size(-1)
                    format_gate = None
                    if use_format_gate and "reward_format_scores" in model_inputs:
                        format_scores = model_inputs["reward_format_scores"].to(
                            device=response_mask.device, dtype=torch.float32
                        )
                        threshold = float(getattr(self.config, "format_gate_threshold", 1.0))
                        format_gate = format_scores >= threshold

                    # Forward (optionally returns latent_mu for vMF KL).
                    if use_vmf_kl and is_depo:
                        log_probs, latent_mu, z_aligned, latent_segment_groups = self._forward_micro_batch(
                            model_inputs, temperature=temperature, return_latent_mu=True
                        )
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        latent_mu, z_aligned, latent_segment_groups = None, None, None

                    # ---- DePO: decoupled token / latent PPO loss ----
                    if is_depo:
                        lat_mask_full = _future_l1_build_latent_mask(
                            model_inputs["input_ids"], response_length, latent_start_id, latent_end_id
                        ).to(response_mask.dtype)
                        token_mask = response_mask * (1.0 - lat_mask_full)
                        lat_mask = response_mask * lat_mask_full
                        lat_mask_pre_gate_sum = lat_mask.sum()
                        if format_gate is not None:
                            lat_mask = lat_mask * format_gate.to(lat_mask.dtype).unsqueeze(-1)

                        pg_loss_tok, pg_metrics_tok = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=token_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            tau_positive=self.config.tau_positive,
                            tau_negative=self.config.tau_negative,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        if lat_mask.sum() > 0:
                            pg_loss_lat, pg_metrics_lat = compute_policy_loss(
                                old_log_probs=old_log_probs,
                                log_probs=log_probs,
                                advantages=advantages,
                                response_mask=lat_mask,
                                clip_ratio_low=self.config.latent_clip_ratio_low,
                                clip_ratio_high=self.config.latent_clip_ratio_high,
                                clip_ratio_dual=self.config.latent_clip_ratio_dual,
                                tau_positive=self.config.tau_positive,
                                tau_negative=self.config.tau_negative,
                                loss_type=self.config.loss_type,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                        else:
                            pg_loss_lat = torch.tensor(0.0, device=log_probs.device)
                            pg_metrics_lat = {}

                        pg_loss = pg_loss_tok + self.config.latent_loss_alpha * pg_loss_lat
                        pg_metrics = {f"{k}_tok": v for k, v in pg_metrics_tok.items()}
                        pg_metrics.update({f"{k}_lat": v for k, v in pg_metrics_lat.items()})
                        pg_metrics["pg_loss_tok"] = float(pg_loss_tok.detach().item())
                        pg_metrics["pg_loss_lat"] = float(pg_loss_lat.detach().item()) if isinstance(pg_loss_lat, torch.Tensor) else 0.0
                        pg_metrics["latent_ratio"] = float(lat_mask.sum().item() / max(response_mask.sum().item(), 1.0))
                        if format_gate is not None:
                            pg_metrics["format_gate_rate"] = float(format_gate.float().mean().item())
                            pg_metrics["latent_gate_keep_ratio"] = float(
                                lat_mask.sum().item() / max(lat_mask_pre_gate_sum.item(), 1.0)
                            )
                    else:
                        pg_loss, pg_metrics = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            tau_positive=self.config.tau_positive,
                            tau_negative=self.config.tau_negative,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )

                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        # In DePO mode, restrict sample-based KL to non-latent tokens
                        # (latents are constrained by the closed-form vMF KL below).
                        kl_mask = (response_mask * (1.0 - lat_mask_full)) if is_depo else response_mask
                        kl_loss = average_loss(kld, kl_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    # ---- DePO: closed-form vMF KL at latent positions ----
                    if use_vmf_kl and is_depo and latent_mu is not None and z_aligned is not None:
                        z = z_aligned.to(latent_mu.device, latent_mu.dtype)
                        if format_gate is not None:
                            latent_mu_kl, z, latent_segment_groups_kl = _future_l1_filter_latents_by_sample_gate(
                                latent_mu, z, latent_segment_groups, format_gate
                            )
                        else:
                            latent_mu_kl, latent_segment_groups_kl = latent_mu, latent_segment_groups
                        if z.shape[0] == latent_mu_kl.shape[0] and z.shape[0] > 0:
                            vmf_kl = compute_latent_vmf_kl(
                                mu_actor=latent_mu_kl,
                                mu_ref=z.detach(),
                                kappa=float(getattr(self.config, "future_l1_rl_kappa", 0.01)),
                            )
                            loss = loss + vmf_kl * float(self.config.latent_kl_coef)
                            metrics["actor/latent_vmf_kl"] = vmf_kl.detach().item()
                            metrics["actor/latent_kl_coef"] = float(self.config.latent_kl_coef)

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
