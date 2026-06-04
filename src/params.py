from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    max_latent_tokens: int = field(default=None)
    force_initial_latent_mode: bool = field(
        default=False,
        metadata={
            "help": "For qwen3_vl only: use RICE_Qwen3VL generation behavior that forces the first assistant token to be <|latent_start|>.",
        },
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    model_init_kwargs: Optional[dict] = field(default_factory=dict)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, 
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})

    latent_loss: str = field(default="mse")
    latent_lambda: float = field(default=0.1)

    # Projection head (RoT-style): maps LLM hidden states to latent embedding space
    use_projection_head: bool = field(
        default=False,
        metadata={"help": "Use projection_head (SwiGLU MLP) for latent token prediction. Same as RoT CoTCompressor."},
    )
    projection_hidden_dim: int = field(
        default=2048,
        metadata={"help": "Hidden dim for projection_head intermediate layer (only when use_projection_head=True)."},
    )
    projection_head_type: str = field(
        default="swiglu",
        metadata={"help": "Projection head type: swiglu (RoT-style) or lvr (simpler MLP)."},
    )
    freeze_projection_head: bool = field(
        default=False,
        metadata={"help": "Freeze projection_head (e.g. when loading pretrained and only finetuning LLM)."},
    )
    use_dual_projection_heads: bool = field(
        default=False,
        metadata={
            "help": "Use separate LVR heads for orig and render branches (dual-MSE only). "
                    "Weights are not shared between the two heads.",
        },
    )

    # Auxiliary CoT-reconstruction CE loss (SIM-CoT style).
    # Decode predicted latent embeddings back to the original CoT text with the
    # base LLM as an auxiliary decoder. Set lambda > 0 to enable.
    decoder_recon_lambda: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the auxiliary CoT-reconstruction CE loss. 0.0 disables it "
                    "(default; original MSE training unchanged). Typical values: 0.1~0.5.",
        },
    )
    decoder_recon_use_pre_proj: bool = field(
        default=False,
        metadata={
            "help": "If True, use raw LLM hidden states (pre-projection-head) as the latent "
                    "prefix for the CoT reconstruction decoder, instead of post-proj embeddings. "
                    "Recommended when a projection head is enabled to avoid gradient conflict "
                    "between MSE (image space) and CE (text space) objectives.",
        },
    )

    checkpoint_name: Optional[str] = None


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default=None, metadata={"help": "Path to the training data.", "nargs": "+"}
    )
    use_twiff_dataset: bool = field(
        default=False,
        metadata={
            "help": "TwiFF video-frame format: extract frames from video by index rather than loading "
                    "pre-saved image files. Expects each JSON sample to have a 'video' field (absolute "
                    "path) and integer lists in 'image'/'reasoning_image' (1-based frame indices into a "
                    "uniformly sampled pool of max(indices) frames, same convention as TwiFF)."
        },
    )
    use_mixed_dataset: bool = field(
        default=False,
        metadata={
            "help": "Mixed TwiFF + FutureL1 training: each JSON file under --data_path is auto-classified "
                    "as TwiFF (frame-index based or chat_video_distill) or FutureL1 (path-based images "
                    "with conversations), then concatenated into a single dataset. The collator dispatches "
                    "each sample to the matching sub-collator. Overrides --use_twiff_dataset when True."
        },
    )
    add_answer_tag: bool = field(
        default=False,
        metadata={
            "help": "If True, wrap assistant final answer with <answer>...</answer>. "
                    "Primarily used by Vanilla SFT Stage 3 for consistent answer boundary."
        },
    )
    cot_response_mode: str = field(
        default="tagged",
        metadata={
            "help": "Assistant response format for vanilla CoT data. "
                    "Choices: tagged (use <reason>...</reason><answer>...</answer>) "
                    "or raw (use only cot/gpt value directly)."
        },
    )
    data_exclude: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated keywords to exclude subdirs (only used for OG dataset)."
        },
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})
    random_seed: Optional[int] = field(default=None)
    shuffle_latent_images: bool = field(
        default=False,
        metadata={
            "help": "If True, randomly shuffle the order of assistant-side latent images (reasoning_image) per sample."
        },
    )
    use_dual_latent_tokens: bool = field(
        default=False,
        metadata={
            "help": "DUAL mode: orig uses <|latent_start|><|latent|><|latent_end|>, "
                    "render uses <|text_start|><|text|><|text_end|>. Requires train_render_dual_mse."
        },
    )
    max_latent_token: int = field(default=32)
    twiff_cot_ratio: float = field(
        default=0.0,
        metadata={
            "help": "TwiFF mix training only: fraction of samples that should use vanilla CoT text-only format (no latent images). 0.0 keeps all samples interleaved; 1.0 makes all samples text-only.",
        },
    )
    max_latent_token_orig: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max latent tokens for thinking (orig) images in dual-MSE. "
                    "If None, uses max_latent_token. Only used by train_render_dual_mse."
        },
    )
    max_latent_token_render: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max latent tokens for render (reasoning_text_*.png) images in dual-MSE. "
                    "If None, uses max_latent_token. Only used by train_render_dual_mse."
        },
    )
    fixed_latent_budget: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set (e.g. 32), each reasoning image uses exactly this many <|latent|> tokens; "
                    "vision features are average-pooled in the forward to match. "
                    "If None, use dynamic token count (max_latent_token caps image resolution only)."
        },
    )
    pool_after_proj: bool = field(
        default=True,
        metadata={
            "help": "When fixed_latent_budget is set and a latent projection head exists: "
                    "True = apply projection to each vision token, then average-pool; "
                    "False = average-pool vision tokens first, then apply projection per pooled token. "
                    "No effect when use_projection_head is false."
        },
    )
    system_message_mode: str = field(
        default="default",
        metadata={
            "help": "System message mode used in data preprocessing. Choices: default, rot, none, custom."
        },
    )
    custom_system_message_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a txt/md file used when system_message_mode=custom."
        },
    )
    start_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Use only the tail part of the dataset. "
                    "0.0 means use all data; 0.6 means skip the first 60% samples and train on the remaining 40%."
        },
    )
    # OCR auxiliary task (task_type=ocr)
    ocr_prompt: str = field(
        default="What is written in the image?",
        metadata={"help": "User prompt for OCR auxiliary samples."},
    )
    ocr_system_message: Optional[str] = field(
        default=None,
        metadata={"help": "System message for OCR samples. If None, uses the same as system_message_mode."},
    )
    # Max # of CoT tokens retained for the auxiliary reconstruction decoder.
    # 0 disables CoT tokenization entirely (default; keeps original pipeline).
    decoder_recon_max_text_len: int = field(
        default=0,
        metadata={
            "help": "Max token length of CoT text used for the aux-decoder reconstruction "
                    "loss. 0 disables it (default). Typical values: 256~768.",
        },
    )
    # Multi-image layout controls (both default to preserve existing behaviour).
    merge_latent_segments: bool = field(
        default=False,
        metadata={
            "help": "Third-path: when a sample has multiple reasoning_image entries, "
                    "merge their consecutive <|latent_start|>...<|latent_end|> envelopes "
                    "into ONE envelope in the assistant sequence (each image still "
                    "goes through ViT separately). No-op on single-image samples. "
                    "Used by train.py / FutureL1DataCollator.",
        },
    )
    single_segment_concat_direction: str = field(
        default="horizontal",
        metadata={
            "help": "For SingleSegmentCollator only: how to stitch multiple "
                    "reasoning_image pages into one image. 'horizontal' (default, "
                    "original behaviour) or 'vertical' (top-to-bottom, better for "
                    "rendered text). No effect when a sample has only one image.",
        },
    )