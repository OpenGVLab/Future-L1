import os
import pathlib
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser

from src.trainer import FutureL1SFTTrainer
from src.dataset import (
    make_supervised_data_module,
    make_supervised_data_module_twiff,
    make_supervised_data_module_mixed,
)
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.model.future_l1 import (
    RICE_Qwen3VL,
    FutureL1_Qwen2_5_VL,
    FutureL1_Qwen3VL,
    FutureL1_Qwen3_5_VL,
    QWEN3_5_BACKBONE_AVAILABLE,
)

from train_utils import get_vision_tower, safe_save_model_for_hf_trainer
from src.train.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen2_5_vl_generation_forward,
    replace_qwen3_5_with_mixed_modality_forward,
    replace_qwen3_5_generation_forward,
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen3_vl_generation_forward
)

local_rank = None

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = get_vision_tower(model)
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    # Handle merger specifically
    merger = getattr(vision_tower, "merger", None)
    if merger is not None:
        merger_params = merger.parameters()
        set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_projection_head(model, training_args, compute_dtype):
    """Configure projection_head: move to dtype and set requires_grad."""
    if not hasattr(model, "projection_head") or model.projection_head is None:
        return
    model.projection_head.to(dtype=compute_dtype)
    for p in model.projection_head.parameters():
        p.requires_grad = not training_args.freeze_projection_head


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def get_model_vocab_size(config):
    """Safely retrieves the vocabulary size from a potentially nested model configuration."""
    try:
        return config.text_config.vocab_size
    except AttributeError:
        return getattr(config, 'vocab_size', None)

import math
from qwen_vl_utils import vision_process

def smart_resize_fixed(height: int, width: int, factor: int, min_pixels=None, max_pixels=None) -> tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else (vision_process.IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (vision_process.IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > vision_process.MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {vision_process.MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, vision_process.round_by_factor(height, factor))
    w_bar = max(factor, vision_process.round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        # Ensure the result is at least factor
        h_bar = max(factor, vision_process.floor_by_factor(height / beta, factor))
        w_bar = max(factor, vision_process.floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = vision_process.ceil_by_factor(height * beta, factor)
        w_bar = vision_process.ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth)
    config.force_initial_latent_mode = getattr(model_args, "force_initial_latent_mode", False)
    # Projection head config (RoT-style)
    config.use_projection_head = getattr(training_args, "use_projection_head", False)
    config.projection_hidden_dim = getattr(training_args, "projection_hidden_dim", 2048)
    config.projection_head_type = getattr(training_args, "projection_head_type", "swiglu")

    vision_process.smart_resize = smart_resize_fixed
    
    # print(model_pth)

    # Patch the forward function
    if config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vl_generation_forward()
        model = FutureL1_Qwen2_5_VL.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        replace_qwen3_vl_generation_forward()
        qwen3_cls = RICE_Qwen3VL if getattr(config, "force_initial_latent_mode", False) else FutureL1_Qwen3VL
        model = qwen3_cls.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    elif config.model_type == "qwen3_5":
        if not QWEN3_5_BACKBONE_AVAILABLE:
            raise ImportError(
                "config.model_type 为 qwen3_5，但当前 transformers 未提供 Qwen3_5ForConditionalGeneration。"
                "请升级 transformers（可参考 FutureL1/requirements_sft.txt），或改用 qwen2_5_vl / qwen3_vl 权重。"
            )
        replace_qwen3_5_with_mixed_modality_forward()
        replace_qwen3_5_generation_forward()
        model = FutureL1_Qwen3_5_VL.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")
    
    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    configure_projection_head(model_to_configure, training_args, compute_dtype)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)
    latent_tokens = ["<|latent|>", "<|latent_start|>", "<|latent_end|>"]
    processor.tokenizer.add_tokens(latent_tokens, special_tokens=False)

    latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
    latent_start_id = processor.tokenizer.convert_tokens_to_ids("<|latent_start|>")
    latent_end_id = processor.tokenizer.convert_tokens_to_ids("<|latent_end|>")
 
    model.config.latent_id = latent_id
    model.config.latent_start_id = latent_start_id
    model.config.latent_end_id = latent_end_id
    model.config.max_latent_token = data_args.max_latent_token
    model.config.train_fixed_latent_budget = getattr(data_args, "fixed_latent_budget", None)
    model.config.pool_after_proj = bool(getattr(data_args, "pool_after_proj", True))

    # there are some dummy tokens in newer hf version
    if get_model_vocab_size(model.config) < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    # configure latent loss type
    model.config.latent_loss = training_args.latent_loss
    model.config.latent_lambda = training_args.latent_lambda
    # Auxiliary CoT-reconstruction decoder (SIM-CoT style): 0.0 disables it.
    model.config.decoder_recon_lambda = getattr(training_args, "decoder_recon_lambda", 0.0)
    model.config.decoder_recon_max_text_len = getattr(data_args, "decoder_recon_max_text_len", 0)
    model.config.decoder_recon_use_pre_proj = getattr(training_args, "decoder_recon_use_pre_proj", False)

    if getattr(data_args, "use_mixed_dataset", False):
        data_module = make_supervised_data_module_mixed(processor=processor, args=data_args)
    elif data_args.use_twiff_dataset:
        data_module = make_supervised_data_module_twiff(processor=processor, args=data_args)
    else:
        data_module = make_supervised_data_module(processor=processor, args=data_args)
    
    
    trainer = FutureL1SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()