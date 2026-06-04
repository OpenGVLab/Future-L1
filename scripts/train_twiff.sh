#!/bin/bash

# We use W&B in offline mode and store logs under OUTPUT_DIR.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export OPENCV_FFMPEG_LOGLEVEL=-8                                                                                                               
export AV_LOG_LEVEL=quiet     

# Debug: print text tokens that participate in CE loss.
# Set to 0 to disable; set to N (int) to print first N batches.
# export FUTURE_L1_DEBUG_CE_TOKENS=1
# export FUTURE_L1_DEBUG_CE_TOKEN_BATCHES=2

unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY




NPROC_PER_NODE=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=0.0.0.0
MASTER_PORT=29502

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# model configs
MODEL_NAME="/path/to/ckpt/Qwen3-VL-8B-Instruct"



DATA_PATH=(
    "/path/to/your/data/TwiFF-2.7M/interleave_top50K.json"
)

RANDOM_SEED=42
GRAD_CHECK=True

GLOBAL_BATCH_SIZE=128      
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NPROC_PER_NODE * WORLD_SIZE)))

# LLM-related params
LR=1e-5

# Latent-related params
LATENT_LOSS=mse
LATENT_LAMBDA=0.2
MAX_LATENT_TOKEN=32

MAX_TOKEN=16384
MIN_TOKEN=2
VIDEO_MAX_FRAMES=16
VIDEO_MAX_TOKEN=128

RUN_NAME="TwiFF-interleave-top50K-lambda$LATENT_LAMBDA-max$MAX_LATENT_TOKEN"
# RUN_NAME="DEBUG"
OUTPUT_DIR="/path/to/your/data/VideoL1/$RUN_NAME"
LOG_DIR="${VIDEO_L1_ROOT}/logs"
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

export PYTHONPATH=$(pwd)
# W&B offline mode, log directory aligned with model output_dir
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=VideoL1
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
torchrun $DISTRIBUTED_ARGS \
    src/train/train.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --latent_loss $LATENT_LOSS\
    --model_id $MODEL_NAME \
    --data_path "${DATA_PATH[@]}" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --latent_lambda $LATENT_LAMBDA \
    --max_latent_token $MAX_LATENT_TOKEN \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 32 * 32)) \
    --image_max_pixels $((MAX_TOKEN * 32 * 32)) \
    --video_max_pixels $((VIDEO_MAX_TOKEN * 32 * 32)) \
    --nframes $VIDEO_MAX_FRAMES \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 False \
    --gradient_checkpointing $GRAD_CHECK \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --random_seed $RANDOM_SEED \
    --use_mixed_dataset True \
    --report_to wandb 2>&1 | tee -a "$LOG_DIR/${RUN_NAME}-${TIMESTAMP}.log"