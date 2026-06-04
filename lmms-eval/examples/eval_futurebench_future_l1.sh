#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUTURE_L1_ROOT="${FUTURE_L1_ROOT:-${VIDEO_L1_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
export FUTURE_L1_ROOT VIDEO_L1_ROOT="${FUTURE_L1_ROOT}" FUTURE_L1_CODE_ROOT="${FUTURE_L1_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# === FutureBench ===
task='futurebench_future_l1'
output_path="./logs_futurebench_future_l1/"
mkdir -p "${output_path}"

model_path="${MODEL_PATH:-ckpt}"
model_basename="$(basename "${model_path}")"

cd "${SCRIPT_DIR}/.."
accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model future_l1 \
        --model_args pretrained=${model_path},code_root=${FUTURE_L1_ROOT},attn_implementation=flash_attention_2,max_num_frames=32 \
        --tasks $task \
        --batch_size 1 \
        --gen_kwargs max_new_tokens=2048,temperature=0,top_p=1,do_sample=false \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'future_l1' \
        --output_path $output_path \
        2>&1 | tee "${output_path}/${model_basename}_$(date +%Y%m%d_%H%M%S).log"
