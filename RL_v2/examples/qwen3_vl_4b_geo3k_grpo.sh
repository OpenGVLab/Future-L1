#!/bin/bash

set -x

export WANDB_MODE=${WANDB_MODE:-offline}

MODEL_PATH=/path/to/ckpt/Qwen3-VL-8B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/path/to/EasyR1/geometry3k/data/train-00000-of-00001.parquet \
    data.val_files=/path/to/EasyR1/geometry3k/data/validation-00000-of-00001.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_8b_geo_grpo \
    trainer.n_gpus_per_node=8 \
    worker.rollout.tensor_parallel_size=1
