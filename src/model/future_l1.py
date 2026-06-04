import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None  # type: ignore[misc, assignment]

QWEN3_5_BACKBONE_AVAILABLE = Qwen3_5ForConditionalGeneration is not None

import os
from typing import Optional, Union, Tuple

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList


from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)

from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache
from transformers.utils import logging

from .projection_head import ProjectionHead, LVRHead

logger = logging.get_logger(__name__)


def _merge_initial_cache_position(model, input_ids: torch.Tensor, model_kwargs: dict) -> dict:
    """Compatibility wrapper for transformers cache init across versions."""
    mk = dict(model_kwargs) if model_kwargs else {}
    fn = getattr(model, "_get_initial_cache_position", None)
    if fn is None:
        return mk

    seq_length = int(input_ids.shape[1])
    device = input_ids.device
    try:
        return fn(seq_length, device, mk)
    except TypeError:
        try:
            return fn(input_ids, mk)
        except TypeError:
            return mk


def _maybe_add_projection_head(model, config, dtype=None):
    """Add projection head(s) to model if enabled.

    When use_dual_projection_heads=True (dual-MSE): creates projection_head_orig and
    projection_head_render (weights not shared). projection_head is set to None.
    """
    use_ph = getattr(config, "use_projection_head", False)
    use_dual = getattr(config, "use_dual_projection_heads", False)

    if not use_ph:
        model.projection_head = None
        model.projection_head_orig = None
        model.projection_head_render = None
        return

    try:
        hidden_dim = config.text_config.hidden_size
    except AttributeError:
        hidden_dim = getattr(config, "hidden_size", 3584)
    proj_hidden = getattr(config, "projection_hidden_dim", 2048)
    head_type = getattr(config, "projection_head_type", "swiglu")

    if use_dual:
        # Dual-MSE: orig and render each have their own head (weights not shared).
        model.projection_head = None
        head_type = (head_type or "lvr").lower()
        if head_type == "lvr":
            model.projection_head_orig = LVRHead(hidden_size=hidden_dim)
            model.projection_head_render = LVRHead(hidden_size=hidden_dim)
        else:
            # RoT-style: SwiGLU projection head
            model.projection_head_orig = ProjectionHead(
                hidden_dim=hidden_dim, projection_hidden_dim=proj_hidden
            )
            model.projection_head_render = ProjectionHead(
                hidden_dim=hidden_dim, projection_hidden_dim=proj_hidden
            )
        if dtype is not None:
            model.projection_head_orig = model.projection_head_orig.to(dtype=dtype)
            model.projection_head_render = model.projection_head_render.to(dtype=dtype)
    else:
        model.projection_head_orig = None
        model.projection_head_render = None
        if head_type == "lvr":
            ph = LVRHead(hidden_size=hidden_dim)
        else:
            ph = ProjectionHead(hidden_dim=hidden_dim, projection_hidden_dim=proj_hidden)
        model.projection_head = ph
        if dtype is not None:
            model.projection_head = model.projection_head.to(dtype=dtype)


def _future_l1_sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    enable_text_mode: bool = False,
    force_initial_latent: bool = False,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = _merge_initial_cache_position(model, input_ids, model_kwargs)

    model_forward = model.__call__
    compile_forward = model._valid_auto_compile_criteria(model_kwargs, generation_config)

    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        # If we use FA2 and a static cache, we cannot compile with fullgraph
        if model.config._attn_implementation == "flash_attention_2":
            # only raise warning if the user passed an explicit compile-config
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = model.get_compiled_call(generation_config.compile_config)

    # Latent / text mode init (DUAL: orig=latent, render=text)
    in_latent_mode = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    has_text_mode = (
        enable_text_mode
        and hasattr(model.config, "text_start_id")
        and getattr(model.config, "text_start_id", None) is not None
    )
    in_text_mode = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device) if has_text_mode else None

    latent_hidden_state = None
    latent_hidden_states = None
    latent_states_buffer = []
    latent_mask_buffer = []

    # Keep fixed_latent_budget exact; allow loose cap in non-fixed mode;
    # otherwise default to 2x config max_latent_token with early latent_end allowed.
    if getattr(model.config, "fixed_latent_budget", False):
        max_latent_num = model.config.max_latent_token
    else:
        loose_budget = getattr(model.config, "loose_latent_budget", None)
        if loose_budget is not None and int(loose_budget) > 0:
            max_latent_num = int(loose_budget)
        else:
            latent_multiplier = float(getattr(model.config, "infer_latent_multiplier", 2.0))
            max_latent_num = max(1, int(round(model.config.max_latent_token * latent_multiplier)))

    max_latent_len = [max_latent_num] * batch_size
    latent_steps_orig = torch.tensor(max_latent_len, dtype=torch.int, device=input_ids.device)
    latent_remaining_steps = latent_steps_orig.clone()
    text_remaining_steps = latent_steps_orig.clone() if has_text_mode else None
    pending_forced_latent = force_initial_latent
    if pending_forced_latent and getattr(model.config, "latent_start_id", None) is None:
        raise ValueError("force_initial_latent=True requires config.latent_start_id to be set.")

    is_prefill = True

    while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        in_special_mode = in_latent_mode | (in_text_mode if in_text_mode is not None else torch.zeros_like(in_latent_mode))
        model_inputs.update({"in_latent_mode": in_special_mode})
        model_inputs.update({"latent_hidden_state": latent_hidden_state})

        if is_prefill:
            outputs = model(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if pending_forced_latent:
            next_tokens = torch.full_like(next_tokens, model.config.latent_start_id)
            pending_forced_latent = False

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        last_tokens = input_ids[:, -1]
        latent_start = (last_tokens == model.config.latent_start_id).to(device=input_ids.device)
        latent_end_predicted = (next_tokens == model.config.latent_end_id).to(device=input_ids.device)
        temp_latent = in_latent_mode | latent_start
        just_entered_latent = (~in_latent_mode) & temp_latent
        latent_remaining_steps = torch.where(just_entered_latent, latent_steps_orig, latent_remaining_steps)
        latent_remaining_steps = latent_remaining_steps - in_latent_mode.long()
        force_end_latent = in_latent_mode & (latent_remaining_steps <= 0)
        # fixed_latent_budget mode: ignore model's <|latent_end|> prediction,
        # only exit when the token budget is fully exhausted.
        if getattr(model.config, "fixed_latent_budget", False):
            latent_end = force_end_latent
        else:
            latent_end = latent_end_predicted | force_end_latent
        in_latent_mode = temp_latent & (~latent_end)

        if has_text_mode and in_text_mode is not None:
            text_start = (last_tokens == model.config.text_start_id).to(device=input_ids.device)
            text_end = (next_tokens == model.config.text_end_id).to(device=input_ids.device)
            temp_text = in_text_mode | text_start
            just_entered_text = (~in_text_mode) & temp_text
            text_remaining_steps = torch.where(just_entered_text, latent_steps_orig, text_remaining_steps)
            text_remaining_steps = text_remaining_steps - in_text_mode.long()
            force_end_text = in_text_mode & (text_remaining_steps <= 0)
            text_end = text_end | force_end_text
            in_text_mode = temp_text & (~text_end)
            next_tokens[in_text_mode, None] = model.config.text_id
            next_tokens[text_end, None] = model.config.text_end_id

        latent_hidden_state = outputs.latent_hidden_state
        next_tokens[in_latent_mode, None] = model.config.latent_id
        next_tokens[latent_end, None] = model.config.latent_end_id
        if return_dict_in_generate and latent_hidden_state is not None:
            latent_mask = in_latent_mode & (next_tokens == model.config.latent_id)
            if latent_mask.any():
                state_step = torch.zeros_like(latent_hidden_state)
                state_step[latent_mask] = latent_hidden_state[latent_mask].detach()
                latent_states_buffer.append(state_step)
                latent_mask_buffer.append(latent_mask.detach())

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        output = GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
        if latent_states_buffer:
            output.latent_states = torch.stack(latent_states_buffer, dim=1)
            output.latent_mask = torch.stack(latent_mask_buffer, dim=1)
        return output

    return input_ids


class FutureL1_Qwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        _maybe_add_projection_head(self, config)
        if getattr(self, "model", None) is not None:
            # Avoid nn.Module child registration cycle (self.model -> self).
            self.model.__dict__["_future_l1_lm_wrapper"] = self

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        return _future_l1_sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            enable_text_mode=False,
            **model_kwargs,
        )


class FutureL1_Qwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        _maybe_add_projection_head(self, config)
        if getattr(self, "model", None) is not None:
            # Avoid nn.Module child registration cycle (self.model -> self).
            self.model.__dict__["_future_l1_lm_wrapper"] = self

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        return _future_l1_sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            enable_text_mode=False,
            **model_kwargs,
        )


class RICE_Qwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        _maybe_add_projection_head(self, config)
        self.config.force_initial_latent_mode = True
        if getattr(self, "model", None) is not None:
            # Avoid nn.Module child registration cycle (self.model -> self).
            self.model.__dict__["_future_l1_lm_wrapper"] = self

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        return _future_l1_sample(
            self,
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            enable_text_mode=False,
            force_initial_latent=True,
            **model_kwargs,
        )


_QWEN3_5_IMPORT_HELP = (
    "当前环境的 transformers 未提供 Qwen3_5ForConditionalGeneration（或版本过旧）。"
    "请升级到包含 Qwen3.5 的 transformers（例如 requirements_sft.txt 中的版本），"
    "或改用 qwen2_5_vl / qwen3_vl 作为 backbone。"
)


if QWEN3_5_BACKBONE_AVAILABLE:

    class FutureL1_Qwen3_5_VL(Qwen3_5ForConditionalGeneration):  # type: ignore[misc]
        def __init__(self, config):
            super().__init__(config)
            _maybe_add_projection_head(self, config)
            if getattr(self, "model", None) is not None:
                # Avoid nn.Module child registration cycle (self.model -> self).
                self.model.__dict__["_future_l1_lm_wrapper"] = self

        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            return _future_l1_sample(
                self,
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                enable_text_mode=False,
                **model_kwargs,
            )

    class RICE_Qwen3_5_VL(Qwen3_5ForConditionalGeneration):  # type: ignore[misc]
        """Same as RICE_Qwen3VL but for Qwen3.5 (qwen3_5) backbone checkpoints."""

        def __init__(self, config):
            super().__init__(config)
            _maybe_add_projection_head(self, config)
            self.config.force_initial_latent_mode = True
            if getattr(self, "model", None) is not None:
                self.model.__dict__["_future_l1_lm_wrapper"] = self

        def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
        ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
            return _future_l1_sample(
                self,
                input_ids=input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                enable_text_mode=False,
                force_initial_latent=True,
                **model_kwargs,
            )

else:

    class FutureL1_Qwen3_5_VL:  # type: ignore[too-many-ancestors]
        """占位类：仅在不支持 Qwen3.5 的 transformers 下存在，避免 import 阶段失败。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(_QWEN3_5_IMPORT_HELP)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError(_QWEN3_5_IMPORT_HELP)

    class RICE_Qwen3_5_VL:  # type: ignore[too-many-ancestors]
        """占位类，与 FutureL1_Qwen3_5_VL 相同。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(_QWEN3_5_IMPORT_HELP)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError(_QWEN3_5_IMPORT_HELP)