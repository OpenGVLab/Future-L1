#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# export TWIFF_SKIP_JUDGE=1

export ROT_CODE_ROOT="./Future-L1"
model_path="ckpt"
model_basename="$(basename "${model_path}")"

output_path="./logs_twiffbench_future_l1/"
mkdir -p "${output_path}"

accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model future_l1 \
        --model_args pretrained=${model_path},attn_implementation=flash_attention_2,max_num_frames=8 \
        --tasks twiffbench_future_l1 \
        --batch_size 1 \
        --gen_kwargs max_new_tokens=4096,temperature=0,top_p=1,do_sample=false \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'twiffbench_future_l1' \
        --output_path "${output_path}" \
        2>&1 | tee "${output_path}/${model_basename}_$(date +%Y%m%d_%H%M%S).log"
