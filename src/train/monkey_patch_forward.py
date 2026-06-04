import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe

try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as _modeling_qwen3_5
except ImportError:
    _modeling_qwen3_5 = None

_QWEN3_5_PATCH_HELP = (
    "当前环境无法导入 transformers.models.qwen3_5.modeling_qwen3_5（可能缺少 Qwen3_5ForConditionalGeneration）。"
    "请升级 transformers 后再使用 qwen3_5，或改用 qwen2_5_vl / qwen3_vl。"
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling


def _compute_cot_recon_loss(
    wrapper,
    shift_predict_embeddings: torch.Tensor,
    shift_image_mask: torch.Tensor,
    recon_cot_input_ids: torch.Tensor,
    recon_cot_attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Auxiliary CoT-reconstruction CE loss (SIM-CoT style).

    Given flattened predicted latent embeddings (post projection head), reshape
    them per-sample and feed them as a prefix into the base text LM; supervise
    the model to autoregressively generate the original CoT tokens. The
    auxiliary decoder is the existing `wrapper.model.language_model` (no new
    parameters). At inference the whole decoder path is skipped.

    Args:
        wrapper: the outer ForConditionalGeneration module.
        shift_predict_embeddings: (num_latent_total, H) flattened predicted
            latents, already gone through the projection head (same tensor the
            MSE is computed on).
        shift_image_mask: (B, seq_len-1) 0/1 mask marking latent positions
            used to flatten shift_predict_embeddings.
        recon_cot_input_ids: (B, T) tokenized CoT text (pad with pad_token_id
            or 0; masked by attention mask).
        recon_cot_attention_mask: (B, T) 1 on real CoT tokens, 0 on padding.

    Returns:
        Scalar CE loss (torch.Tensor) or None if there is nothing to supervise.
    """
    if recon_cot_input_ids is None or shift_predict_embeddings is None:
        return None

    device = shift_predict_embeddings.device
    dtype = shift_predict_embeddings.dtype

    per_sample_latents = shift_image_mask.sum(dim=1).detach().to("cpu").tolist()
    bsz = len(per_sample_latents)
    if bsz == 0 or min(per_sample_latents) <= 0:
        # Some sample in the batch has no latent prefix -> skip recon loss
        # rather than tripping reshape / mask logic.
        return None

    latent_chunks = torch.split(shift_predict_embeddings, per_sample_latents)

    if recon_cot_attention_mask is None:
        recon_cot_attention_mask = torch.ones_like(recon_cot_input_ids, dtype=torch.long)
    recon_cot_attention_mask = recon_cot_attention_mask.to(device=device, dtype=torch.long)
    recon_cot_input_ids = recon_cot_input_ids.to(device=device, dtype=torch.long)
    cot_lens = recon_cot_attention_mask.sum(dim=1).tolist()

    if sum(cot_lens) == 0:
        return None

    # Embed CoT text tokens via the base LLM's input embedding layer.
    text_embed_layer = wrapper.get_input_embeddings()
    cot_embeds_full = text_embed_layer(recon_cot_input_ids).to(dtype=dtype)

    max_total_len = max(per_sample_latents[i] + cot_lens[i] for i in range(bsz))
    hidden_dim = cot_embeds_full.size(-1)

    input_embs = torch.zeros(bsz, max_total_len, hidden_dim, device=device, dtype=dtype)
    attn_mask = torch.zeros(bsz, max_total_len, dtype=torch.long, device=device)
    labels_recon = torch.full((bsz, max_total_len), -100, dtype=torch.long, device=device)

    for i in range(bsz):
        K = per_sample_latents[i]
        T = cot_lens[i]
        if T == 0:
            # Sample has latent prefix but no CoT target -> leave row masked.
            input_embs[i, :K, :] = latent_chunks[i].to(dtype=dtype)
            attn_mask[i, :K] = 1
            continue
        input_embs[i, :K, :] = latent_chunks[i].to(dtype=dtype)
        input_embs[i, K : K + T, :] = cot_embeds_full[i, :T, :]
        attn_mask[i, : K + T] = 1
        labels_recon[i, K : K + T] = recon_cot_input_ids[i, :T]

    # Use the pure text LM for a clean, vision-free forward pass.
    text_lm = getattr(wrapper.model, "language_model", None) or wrapper.model
    position_ids = torch.arange(max_total_len, device=device).unsqueeze(0).expand(bsz, -1)
    lm_out = text_lm(
        inputs_embeds=input_embs,
        attention_mask=attn_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    lm_hidden = lm_out.last_hidden_state if hasattr(lm_out, "last_hidden_state") else lm_out[0]
    recon_logits = wrapper.lm_head(lm_hidden)

    shift_logits = recon_logits[:, :-1, :].contiguous()
    shift_labels = labels_recon[:, 1:].contiguous()
    recon_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return recon_loss


def _select_projection_head(module, prefer: str = "render"):
    """Select projection head with explicit preference in dual-head mode.

    prefer:
      - "render": prefer projection_head_render
      - "orig": prefer projection_head_orig
      - "auto": render -> orig -> single
    """
    prefer = (prefer or "render").lower()
    head_render = getattr(module, "projection_head_render", None)
    head_orig = getattr(module, "projection_head_orig", None)
    head_single = getattr(module, "projection_head", None)

    if prefer == "orig":
        return head_orig or head_single or head_render
    if prefer == "auto":
        return head_render or head_orig or head_single
    # default "render"
    return head_render or head_single or head_orig


def _projection_head_for_latent_image(wrapper, kind: Optional[str]):
    """Pick orig/render/single projection head for pooled-latent GT path (dual-MSE aware)."""
    if wrapper is None:
        return None
    if kind is not None:
        k = str(kind).lower()
        if k == "render" and getattr(wrapper, "projection_head_render", None) is not None:
            return wrapper.projection_head_render
        if k in ("orig", "original", "thinking") and getattr(wrapper, "projection_head_orig", None) is not None:
            return wrapper.projection_head_orig
    prefer = getattr(wrapper.config, "infer_head_prefer", "render")
    return _select_projection_head(wrapper, prefer=prefer)


def _pool_latent_vision_tokens_per_image(
    latent_image_embeds: torch.Tensor,
    image_grid_thw_latent: torch.LongTensor,
    num_out_tokens: int,
    wrapper: Optional[torch.nn.Module],
    pool_after_proj: bool,
    spatial_merge_size: int = 1,
    latent_pool_head_kinds: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Average-pool each image's vision tokens to num_out_tokens vectors.

    If no projection head exists on wrapper, pool_after_proj is ignored.
    Otherwise, per-image head may be chosen via latent_pool_head_kinds ('orig' / 'render').
      - pool_after_proj True: apply head per vision token, then pool.
      - pool_after_proj False: pool first, then apply head per pooled token.
    """
    n_images = int(image_grid_thw_latent.shape[0])
    if latent_pool_head_kinds is not None and len(latent_pool_head_kinds) != n_images:
        raise ValueError(
            f"latent_pool_head_kinds length ({len(latent_pool_head_kinds)}) must equal "
            f"image_grid_thw_latent rows ({n_images})"
        )
    out_list = []
    offset = 0
    rows = int(latent_image_embeds.shape[0])
    if spatial_merge_size <= 0:
        spatial_merge_size = 1
    merge_div = int(spatial_merge_size) ** 2

    for i in range(n_images):
        t = int(image_grid_thw_latent[i, 0].item())
        h = int(image_grid_thw_latent[i, 1].item())
        w = int(image_grid_thw_latent[i, 2].item())
        raw_tokens = t * h * w
        ti = raw_tokens // merge_div
        chunk = latent_image_embeds[offset : offset + ti]
        offset += ti
        if ti == 0:
            continue
        kind = latent_pool_head_kinds[i] if latent_pool_head_kinds is not None else None
        head = _projection_head_for_latent_image(wrapper, kind)
        if head is not None and pool_after_proj:
            chunk = head(chunk)
        x = chunk.unsqueeze(0).transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, num_out_tokens)
        x = x.transpose(1, 2).squeeze(0)
        if head is not None and not pool_after_proj:
            x = head(x)
        out_list.append(x)
    if offset != rows:
        raise ValueError(
            f"image_grid_thw_latent implied {offset} vision tokens but got {rows} rows in latent_image_embeds"
        )
    return torch.cat(out_list, dim=0)


def _maybe_pool_latent_for_training(
    inner_model,
    latent_image_embeds: torch.Tensor,
    image_grid_thw_latent: Optional[torch.LongTensor],
    latent_pool_head_kinds: Optional[List[str]] = None,
) -> torch.Tensor:
    cfg = inner_model.config
    n_fixed = getattr(cfg, "train_fixed_latent_budget", None)
    if n_fixed is None or int(n_fixed) <= 0:
        return latent_image_embeds
    if image_grid_thw_latent is None:
        raise ValueError("train_fixed_latent_budget is set but image_grid_thw_latent is None")
    n_fixed = int(n_fixed)
    pool_after = bool(getattr(cfg, "pool_after_proj", True))
    wrapper = getattr(inner_model, "_future_l1_lm_wrapper", None)
    visual = getattr(inner_model, "visual", None)
    spatial_merge_size = int(getattr(visual, "spatial_merge_size", 1)) if visual is not None else 1
    return _pool_latent_vision_tokens_per_image(
        latent_image_embeds,
        image_grid_thw_latent,
        n_fixed,
        wrapper,
        pool_after,
        spatial_merge_size,
        latent_pool_head_kinds,
    )


def replace_qwen2_5_with_mixed_modality_forward():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen2_5_mixed_modality_forward

def replace_qwen2_5_vl_generation_forward():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_vl_generation_forward

def replace_qwen3_with_mixed_modality_forward():
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.forward = qwen3_vl_mixed_modality_forward

def replace_qwen3_vl_generation_forward():
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = qwen3_vl_generation_forward

def replace_qwen3_5_with_mixed_modality_forward():
    if _modeling_qwen3_5 is None:
        raise ImportError(_QWEN3_5_PATCH_HELP)
    _modeling_qwen3_5.Qwen3_5Model.forward = qwen3_5_mixed_modality_forward


def replace_qwen3_5_generation_forward():
    if _modeling_qwen3_5 is None:
        raise ImportError(_QWEN3_5_PATCH_HELP)
    _modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = qwen3_5_generation_forward


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

@dataclass
class Qwen2_5_VLModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    latent_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    ce_loss: Optional[torch.FloatTensor] = None
    latent_loss: Optional[torch.FloatTensor] = None

@dataclass
class Qwen3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None

@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    latent_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    ce_loss: Optional[torch.FloatTensor] = None
    latent_loss: Optional[torch.FloatTensor] = None
    # Auxiliary CoT-reconstruction CE loss (SIM-CoT style): decode latents back to CoT text.
    recon_loss: Optional[torch.FloatTensor] = None
    # LVR replay / GRPO: last token hidden before lm_head (same convention as lvr monkey_patch_forward_lvr_rl)
    last_position_hidden_state: Optional[torch.FloatTensor] = None

@dataclass
class Qwen3_5ModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None

@dataclass
class Qwen3_5CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    latent_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    ce_loss: Optional[torch.FloatTensor] = None
    latent_loss: Optional[torch.FloatTensor] = None
    last_position_hidden_state: Optional[torch.FloatTensor] = None


def _flatten_qwen3_5_vision_features(vision_outputs):
    pooled = getattr(vision_outputs, "pooler_output", vision_outputs)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(pooled, (tuple, list)):
        return torch.cat(list(pooled), dim=0)
    raise TypeError(f"Unsupported vision output type: {type(vision_outputs)!r}")


def _qwen3_5_image_grid_thw_for_rope(
    image_grid_thw: Optional[torch.Tensor],
    image_grid_thw_latent: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Merge user-side and assistant latent ``image_grid_thw`` for M-RoPE / ``get_rope_index``.

    Qwen3.5 consumes one grid row per contiguous image-modality run in ``mm_token_type_ids``.
    FutureL1 passes user images as ``image_grid_thw`` and reasoning images as
    ``image_grid_thw_latent``; both must be concatenated in **dialogue order** (user segments
    first, then assistant latent segments) or ``get_rope_index`` exhausts the iterator
    (``StopIteration``).
    """
    if image_grid_thw is not None and image_grid_thw_latent is not None:
        return torch.cat([image_grid_thw, image_grid_thw_latent], dim=0)
    if image_grid_thw is not None:
        return image_grid_thw
    return image_grid_thw_latent


def _qwen3_5_latent_grid_thw_for_fixed_budget_rope(
    image_grid_thw_latent: torch.Tensor,
    fixed_n: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Rewrite latent ``image_grid_thw`` rows for RoPE only when using fixed latent token budget.

    ``get_rope_index`` assigns ``get_vision_position_ids(grid)`` width = ``T * llm_h * llm_w``
    (after merge), not the span length in ``mm_token_type_ids``. After
    ``replace_latent_fixed`` the sequence has exactly ``fixed_n`` <|latent|> tokens per
    reasoning image while ``image_grid_thw_latent`` still describes native resolution →
    RoPE length ≠ token count. Use a synthetic grid ``(1, fixed_n * sm, sm)`` so
    ``llm_h = fixed_n``, ``llm_w = 1``, matching ``_maybe_pool_latent_for_training``.
    """
    sm = max(1, int(spatial_merge_size))
    n = int(fixed_n)
    if n <= 0:
        return image_grid_thw_latent
    out = image_grid_thw_latent.clone()
    out[:, 0] = 1
    out[:, 1] = n * sm
    out[:, 2] = sm
    return out


def qwen2_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,  # =============================
    pixel_values_latent: Optional[torch.Tensor] = None, # latent images
    image_grid_thw_latent: Optional[torch.LongTensor] = None, # latent images grid
    latent_target_embeds: Optional[torch.Tensor] = None,  # [N, H] offline GT; mutually exclusive with pixel_values_latent
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    latent_pool_head_kinds = kwargs.pop("latent_pool_head_kinds", None)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if in_latent_mode is not None and in_latent_mode.any():
            latent_hidden_state = latent_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds[in_latent_mode, -1, :] = latent_hidden_state[in_latent_mode]

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if latent_target_embeds is not None and pixel_values_latent is not None:
        raise ValueError("Use only one of latent_target_embeds or pixel_values_latent for latent supervision.")

    if latent_target_embeds is not None:
        latent_image_embeds = latent_target_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent offline features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        mask = input_ids == self.config.latent_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, latent_image_embeds)

    elif pixel_values_latent is not None:
        latent_image_embeds = self.get_image_features(pixel_values_latent, image_grid_thw_latent)
        latent_image_embeds = torch.cat(latent_image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        latent_image_embeds = _maybe_pool_latent_for_training(
            self, latent_image_embeds, image_grid_thw_latent, latent_pool_head_kinds
        )
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]
        # reasoning image token num != latent token num 
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent image features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        
        mask = input_ids == self.config.latent_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)
        
        latent_image_embeds = latent_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, latent_image_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
        inputs_embeds=inputs_embeds
    )
    return output if return_dict else output.to_tuple()

def qwen2_5_vl_generation_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    pixel_values_latent: Optional[torch.Tensor] = None, # latent images
    image_grid_thw_latent: Optional[torch.LongTensor] = None, # latent images grid
    latent_target_embeds: Optional[torch.Tensor] = None,
    image_out_mask: Optional[torch.Tensor] = None, # latent token mask
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        pixel_values_latent=pixel_values_latent,
        image_grid_thw_latent=image_grid_thw_latent,
        latent_target_embeds=latent_target_embeds,
        in_latent_mode=in_latent_mode,
        latent_hidden_state=latent_hidden_state,
        **kwargs,
    )

    hidden_states = outputs[0]
    last_hidden = outputs.last_hidden_state[:, -1, :]
    # Apply projection head for latent token prediction (training & generation).
    prefer_head = getattr(self.config, "infer_head_prefer", "render")
    selected_head = _select_projection_head(self, prefer=prefer_head)
    if selected_head is not None:
        latent_hidden_state = selected_head(last_hidden)
    else:
        latent_hidden_state = last_hidden

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    ce_loss = None
    latent_loss = None
    if labels is not None:
        ce_loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
        )
        loss = ce_loss

    if image_out_mask is not None and (pixel_values_latent is not None or latent_target_embeds is not None):
        predict_embeddings = hidden_states
        prefer_head = getattr(self.config, "infer_head_prefer", "render")
        selected_head = _select_projection_head(self, prefer=prefer_head)
        if selected_head is not None:
            predict_embeddings = selected_head(predict_embeddings)

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()

        if self.config.latent_loss == 'mse':
            mse_loss = torch.nn.functional.mse_loss(shift_predict_embeddings, gt_embeddings)
            latent_loss = mse_loss
        elif self.config.latent_loss == 'sim':
            sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
            latent_loss = 1 - sim_loss
        
        if loss is None:
            loss = self.config.latent_lambda * latent_loss
        else:
            loss = loss + self.config.latent_lambda * latent_loss


    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        latent_hidden_state = latent_hidden_state,
        ce_loss=ce_loss,
        latent_loss=latent_loss,
    )


def qwen3_vl_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    pixel_values_latent: Optional[torch.Tensor] = None, # latent images
    image_grid_thw_latent: Optional[torch.LongTensor] = None, # latent images grid
    latent_target_embeds: Optional[torch.Tensor] = None,
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    latent_pool_head_kinds = kwargs.pop("latent_pool_head_kinds", None)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if in_latent_mode is not None and in_latent_mode.any():
            latent_hidden_state = latent_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds[in_latent_mode, -1, :] = latent_hidden_state[in_latent_mode]

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if latent_target_embeds is not None and pixel_values_latent is not None:
        raise ValueError("Use only one of latent_target_embeds or pixel_values_latent for latent supervision.")

    if latent_target_embeds is not None:
        latent_image_embeds = latent_target_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent offline features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        mask = input_ids == self.config.latent_id
        latent_image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(latent_image_mask, latent_image_embeds)

    elif pixel_values_latent is not None:
        latent_image_embeds, _ = self.get_image_features(pixel_values_latent, image_grid_thw_latent)

        latent_image_embeds = torch.cat(latent_image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        latent_image_embeds = _maybe_pool_latent_for_training(
            self, latent_image_embeds, image_grid_thw_latent, latent_pool_head_kinds
        )
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent image features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        
        mask = input_ids == self.config.latent_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        latent_image_mask = mask_expanded.to(inputs_embeds.device)
        
        latent_image_embeds = latent_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.masked_scatter(latent_image_mask, latent_image_embeds)

    # LVR generation / GRPO teacher forcing (consumed here; must not reach language_model **kwargs)
    lvr_mode_switch = kwargs.pop("lvr_mode_switch", None)
    last_position_hs_in = kwargs.pop("last_position_hidden_state", None)
    lvr_mask_tf = kwargs.pop("lvr_mask", None)
    lvr_states_tf = kwargs.pop("lvr_states", None)
    prompt_length_tf = kwargs.pop("prompt_length", None)

    if last_position_hs_in is not None and lvr_mode_switch is not None:
        last_position_hs_in = last_position_hs_in.to(inputs_embeds.device, inputs_embeds.dtype)
        lvr_mask = lvr_mode_switch.to(device=inputs_embeds.device)
        inputs_embeds[lvr_mask, -1, :] = last_position_hs_in[lvr_mask]

    if lvr_states_tf is not None and lvr_mask_tf is not None and prompt_length_tf is not None:
        comp_embeds = inputs_embeds[:, prompt_length_tf:, :]
        lvr_mask_tf = lvr_mask_tf.to(comp_embeds.device)
        lvr_states_tf = lvr_states_tf.to(comp_embeds.device, comp_embeds.dtype)
        comp_embeds = torch.where(lvr_mask_tf.unsqueeze(-1), lvr_states_tf, comp_embeds)
        inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length_tf, :], comp_embeds], dim=1)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # Decode steps can carry full attention_mask with single-token input_ids;
        # calling get_rope_index there will cause shape mismatch.
        current_input_len = 0
        if input_ids is not None:
            current_input_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            current_input_len = inputs_embeds.shape[1]
        is_prefill_step = current_input_len != 1

        # When compiling, we can't check tensor values thus we check only input length.
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding.
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and is_prefill_step and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        should_compute_rope = (prefill_compiled_stage or prefill_noncompiled_stage) or (
            self.rope_deltas is None and is_prefill_step
        )
        if should_compute_rope:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            if self.rope_deltas is None:
                self.rope_deltas = torch.zeros(batch_size, dtype=torch.long, device=inputs_embeds.device)
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
        inputs_embeds=inputs_embeds,
    )


def qwen3_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    pixel_values_latent: Optional[torch.Tensor] = None,
    image_grid_thw_latent: Optional[torch.LongTensor] = None,
    latent_target_embeds: Optional[torch.Tensor] = None,
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3_5ModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    latent_pool_head_kinds = kwargs.pop("latent_pool_head_kinds", None)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if in_latent_mode is not None and in_latent_mode.any():
            latent_hidden_state = latent_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds[in_latent_mode, -1, :] = latent_hidden_state[in_latent_mode]

    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = _flatten_qwen3_5_vision_features(image_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        video_embeds = _flatten_qwen3_5_vision_features(video_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if latent_target_embeds is not None and pixel_values_latent is not None:
        raise ValueError("Use only one of latent_target_embeds or pixel_values_latent for latent supervision.")

    if latent_target_embeds is not None:
        latent_image_embeds = latent_target_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent offline features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        latent_mask = (input_ids == self.config.latent_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(latent_mask, latent_image_embeds)
    elif pixel_values_latent is not None:
        latent_outputs = self.get_image_features(pixel_values_latent, image_grid_thw_latent, return_dict=True)
        latent_image_embeds = _flatten_qwen3_5_vision_features(latent_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        latent_image_embeds = _maybe_pool_latent_for_training(
            self, latent_image_embeds, image_grid_thw_latent, latent_pool_head_kinds
        )
        n_image_tokens = (input_ids == self.config.latent_id).sum().item()
        n_image_features = latent_image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Latent image features and latent tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        latent_mask = (input_ids == self.config.latent_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(latent_mask, latent_image_embeds)

    if position_ids is None:
        latent_grid_for_rope = image_grid_thw_latent
        fixed_n = getattr(self.config, "train_fixed_latent_budget", None)
        if (
            latent_grid_for_rope is not None
            and fixed_n is not None
            and int(fixed_n) > 0
            and pixel_values_latent is not None
        ):
            sm = int(getattr(self.config.vision_config, "spatial_merge_size", 1))
            latent_grid_for_rope = _qwen3_5_latent_grid_thw_for_fixed_budget_rope(
                latent_grid_for_rope, int(fixed_n), sm
            )
        rope_image_grid_thw = _qwen3_5_image_grid_thw_for_rope(image_grid_thw, latent_grid_for_rope)
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=rope_image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen3_5ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self, "rope_deltas", None),
        inputs_embeds=inputs_embeds,
    )
    return output if return_dict else output.to_tuple()


def qwen3_vl_generation_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    pixel_values_latent: Optional[torch.Tensor] = None, # latent images
    image_grid_thw_latent: Optional[torch.LongTensor] = None, # latent images grid
    latent_target_embeds: Optional[torch.Tensor] = None,
    image_out_mask: Optional[torch.Tensor] = None, # latent token mask
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    recon_cot_input_ids: Optional[torch.LongTensor] = None,
    recon_cot_attention_mask: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.

    Example:
        TODO: Add example
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        pixel_values_latent=pixel_values_latent,
        image_grid_thw_latent=image_grid_thw_latent,
        latent_target_embeds=latent_target_embeds,
        in_latent_mode=in_latent_mode,
        latent_hidden_state=latent_hidden_state,
        **kwargs,
    )

    hidden_states = outputs[0]
    last_hidden = outputs.last_hidden_state[:, -1, :]
    # Apply projection head for latent token prediction (training & generation).
    prefer_head = getattr(self.config, "infer_head_prefer", "render")
    selected_head = _select_projection_head(self, prefer=prefer_head)
    if selected_head is not None:
        latent_hidden_state = selected_head(last_hidden)
    else:
        latent_hidden_state = last_hidden

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    ce_loss = None
    latent_loss = None
    recon_loss = None
    if labels is not None:
        ce_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        loss = ce_loss

    if image_out_mask is not None and (pixel_values_latent is not None or latent_target_embeds is not None):
        predict_embeddings = hidden_states
        prefer_head = getattr(self.config, "infer_head_prefer", "render")
        selected_head = _select_projection_head(self, prefer=prefer_head)
        if selected_head is not None:
            predict_embeddings = selected_head(predict_embeddings)

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()

        if self.config.latent_loss == 'mse':
            mse_loss = torch.nn.functional.mse_loss(shift_predict_embeddings, gt_embeddings)
            latent_loss = mse_loss
        elif self.config.latent_loss == 'sim':
            sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
            latent_loss = 1 - sim_loss

        if loss is None:
            loss = self.config.latent_lambda * latent_loss
        else:
            loss = loss + self.config.latent_lambda * latent_loss

        # ---- Optional: CoT reconstruction loss (SIM-CoT style aux decoder) ----
        # Only active when decoder_recon_lambda > 0 AND CoT tokens are provided.
        recon_lambda = float(getattr(self.config, "decoder_recon_lambda", 0.0) or 0.0)
        if recon_lambda > 0.0 and recon_cot_input_ids is not None:
            # When decoder_recon_use_pre_proj=True, use raw hidden states as the latent
            # prefix instead of post-proj embeddings, avoiding gradient conflict between
            # MSE (image space) and CE (text space) objectives on the projection head.
            use_pre_proj = bool(getattr(self.config, "decoder_recon_use_pre_proj", False))
            if use_pre_proj and selected_head is not None:
                shift_recon_embeddings = hidden_states[..., :-1, :][shift_image_mask.to(hidden_states.device) != 0].contiguous()
            else:
                shift_recon_embeddings = shift_predict_embeddings
            recon_loss = _compute_cot_recon_loss(
                wrapper=self,
                shift_predict_embeddings=shift_recon_embeddings,
                shift_image_mask=shift_image_mask,
                recon_cot_input_ids=recon_cot_input_ids,
                recon_cot_attention_mask=recon_cot_attention_mask,
            )
            if recon_loss is not None:
                loss = (recon_lambda * recon_loss) if loss is None else (loss + recon_lambda * recon_loss)

    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
        latent_hidden_state=latent_hidden_state,
        ce_loss=ce_loss,
        latent_loss=latent_loss,
        recon_loss=recon_loss,
        last_position_hidden_state=last_position_hidden_state,
    )


def qwen3_5_generation_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    pixel_values_latent: Optional[torch.Tensor] = None,
    image_grid_thw_latent: Optional[torch.LongTensor] = None,
    latent_target_embeds: Optional[torch.Tensor] = None,
    image_out_mask: Optional[torch.Tensor] = None,
    in_latent_mode: Optional[torch.Tensor] = None,
    latent_hidden_state: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        pixel_values_latent=pixel_values_latent,
        image_grid_thw_latent=image_grid_thw_latent,
        latent_target_embeds=latent_target_embeds,
        in_latent_mode=in_latent_mode,
        latent_hidden_state=latent_hidden_state,
        **kwargs,
    )

    hidden_states = outputs[0]
    last_hidden = outputs.last_hidden_state[:, -1, :]
    prefer_head = getattr(self.config, "infer_head_prefer", "render")
    selected_head = _select_projection_head(self, prefer=prefer_head)
    if selected_head is not None:
        latent_hidden_state = selected_head(last_hidden)
    else:
        latent_hidden_state = last_hidden

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    ce_loss = None
    latent_loss = None
    if labels is not None:
        ce_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        loss = ce_loss

    if image_out_mask is not None and (pixel_values_latent is not None or latent_target_embeds is not None):
        predict_embeddings = hidden_states
        prefer_head = getattr(self.config, "infer_head_prefer", "render")
        selected_head = _select_projection_head(self, prefer=prefer_head)
        if selected_head is not None:
            predict_embeddings = selected_head(predict_embeddings)

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()

        if self.config.latent_loss == "mse":
            latent_loss = torch.nn.functional.mse_loss(shift_predict_embeddings, gt_embeddings)
        elif self.config.latent_loss == "sim":
            sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
            latent_loss = 1 - sim_loss

        if loss is None:
            loss = self.config.latent_lambda * latent_loss
        else:
            loss = loss + self.config.latent_lambda * latent_loss

    last_position_hidden_state = outputs.last_hidden_state[:, -1, :]

    return Qwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        latent_hidden_state=latent_hidden_state,
        ce_loss=ce_loss,
        latent_loss=latent_loss,
        last_position_hidden_state=last_position_hidden_state,
    )