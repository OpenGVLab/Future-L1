# Future-L1: Interleaved Latent Visual Reasoning for Video Event Prediction

<p align="center">
  <b>Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction</b>
</p>

<p align="center">
  <b>Future-L1</b> teaches multimodal LLMs to alternate between language tokens and continuous latent visual spans, enabling compact future-state imagination before answering video event prediction questions.
</p>

<p align="center">
  <a href="#highlights">Highlights</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#sft-training">SFT Training</a> •
  <a href="#rl-training-la-dapo">RL Training</a> •
  <a href="#lmms-eval-evaluation">Evaluation</a>
</p>

---

## Overview

Video event prediction requires a model to infer unobserved future states from a partially observed video. Existing video MLLMs usually verbalize intermediate future reasoning in text, which can discard fine-grained motion, geometry, relative position, and interaction cues. **Future-L1** keeps those intermediate dynamics in a continuous visual latent space.

Future-L1 augments a Qwen3-VL backbone with three special tokens:

- `<|latent_start|>`: enter latent visual reasoning mode;
- `<|latent|>`: emit a continuous latent visual thought that is fed back as the next input embedding;
- `<|latent_end|>`: return to textual reasoning.

The model therefore generates responses as interleaved trajectories:

```text
<reason> textual reasoning step 0 </reason>
<|latent_start|> <|latent|> ... <|latent|> <|latent_end|>
<reason> textual reasoning step 1 </reason>
...
<answer> predicted future event </answer>
```

Training is organized in two stages:

1. **SFT** on visual-gain selected interleaved traces aligns latent hidden states to future-frame embeddings.
2. **LA-DAPO RL** further optimizes sampled latent trajectories with answer, format, outcome-contrastive, and temporal-diversity rewards.

All reported checkpoints are evaluated with the repository-integrated **lmms-eval** pipeline.

---

## Highlights

- **Interleaved latent visual reasoning.** Future-L1 alternates between textual reasoning and bounded continuous latent spans during autoregressive decoding.
- **Future-L1-50K SFT data.** We curate 50K high-utility examples from TwiFF-style trajectories by selecting cases where future visual hints measurably improve prediction.
- **Latent-aware RL.** LA-DAPO extends DAPO with trajectory-level rewards that align successful latent rollouts and discourage repeated visual thoughts.
- **State-of-the-art VEP performance.** Future-L1-RL reaches **85.4%** on FutureBench and **3.04** average score on TwiFF-Bench.
- **Compact inference.** The final model improves accuracy through latent visual computation rather than long text-only chains or multi-turn search.

---

## Method

### 1. Interleaved Latent Visual Spans

Given an observed video prefix and a forecasting question, Future-L1 starts in textual mode. When it emits `<|latent_start|>`, the following `<|latent|>` positions produce hidden states that are fed back into the model as continuous embeddings rather than decoded into vocabulary tokens. These latent states remain in the KV cache and condition later textual reasoning. The span terminates when `<|latent_end|>` is emitted or when the configured latent budget is reached.

This design lets the model preserve intermediate visual semantics while still using language to structure the reasoning process.

### 2. Visual-Gain SFT

Future-L1-50K is selected from TwiFF-style visual chain-of-thought data. For each candidate sample, the backbone is evaluated under two settings:

- **Text-only condition:** observed video prefix + question;
- **Visual-hint condition:** observed prefix + question + intermediate future reasoning frames.

A sample is retained when the text-only setting is not saturated and the visual hint provides a measurable accuracy lift. The retained examples are ranked by visual gain, and the top 50K are converted into interleaved text/latent trajectories.

The SFT objective combines next-token prediction with latent MSE alignment:

```text
L_SFT = L_CE + lambda * L_Latent
```

where latent positions are aligned to Qwen3-VL vision-encoder embeddings of the corresponding future reasoning frames.

### 3. LA-DAPO RL

SFT provides a grounded initialization, but teacher-forced latent states are not directly optimized for sampled prediction success. Future-L1 therefore uses **LA-DAPO**: Latent-Aware Direct Advantage Policy Optimization.

The total reward is:

```text
R = lambda_a * R_acc + lambda_f * R_fmt + lambda_c * R_ctr + lambda_d * R_div
```

- `R_acc`: final-answer correctness, using rule-based matching with LLM-judge fallback;
- `R_fmt`: valid interleaved format with reason, latent, and answer tags;
- `R_ctr`: outcome-contrastive latent reward that pulls correct latent trajectories together and treats incorrect rollouts as negatives;
- `R_div`: temporal-diversity reward that discourages adjacent latent spans from collapsing to the same visual thought.

During RL, intermediate future-frame annotations are no longer required; the latent rewards are computed from generated trajectories and rollout outcomes.

---

## Results

### FutureBench

Accuracy (%) on FutureBench. Future-L1 is built on Qwen3-VL-8B-Instruct and uses 32 input frames for evaluation.

| Model | Size | Training | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---|---:|---:|---:|---:|---:|
| GPT-4o | -- | Zero-shot | 61.9 | 61.7 | 72.1 | 51.6 | 59.0 |
| GPT-5 | -- | Zero-shot | 59.6 | 57.3 | 62.6 | 55.6 | 57.9 |
| Qwen3-VL | 30B-A3B | Zero-shot | 65.3 | 70.5 | 76.1 | 62.2 | 66.9 |
| Video-o3 | 7B | SFT+RL | 68.2 | 73.6 | 63.2 | 69.7 | 68.9 |
| Video-CoE | 7B | SFT+RL | 80.9 | 83.9 | 71.6 | 71.4 | 75.0 |
| Qwen3-VL-Instruct | 8B | Zero-shot | 64.2 | 65.8 | 66.2 | 55.8 | 61.0 |
| Text-only SFT on Future-L1-50K | 8B | SFT | 67.6 | 66.8 | 68.2 | 62.0 | 65.0 |
| **Future-L1-SFT** | 8B | SFT | 70.5 | 73.1 | 77.6 | 72.2 | **73.2** |
| **Future-L1-RL** | 8B | SFT+RL | **83.2** | **86.5** | **86.6** | **85.1** | **85.4** |

### TwiFF-Bench

TwiFF-Bench evaluates both reasoning trajectory quality and final answer quality on a 0--5 scale.

| Model | Size | CoT | Answer | Avg. |
|---|---:|---:|---:|---:|
| Qwen2.5-VL | 7B | 2.46 | 1.63 | 2.05 |
| InternVL3.5 | 8B | 2.35 | 1.85 | 2.10 |
| DeepEyes | 7B | 2.54 | 2.20 | 2.37 |
| TwiFF-300K | 7B | 2.90 | 2.55 | 2.73 |
| TwiFF-2.7M | 7B | 2.95 | 2.62 | 2.79 |
| Qwen3-VL-Instruct | 8B | 2.75 | 2.14 | 2.44 |
| **Future-L1-SFT** | 8B | 2.62 | 2.42 | 2.52 |
| **Future-L1-RL** | 8B | **3.11** | **2.97** | **3.04** |

### Ablation Snapshot

| Study | Best setting | Finding |
|---|---|---|
| Latent MSE weight | `lambda=0.1` | Explicit but not dominant latent supervision is best. |
| Maximum latent budget | `L_max=4` | Short supervised spans outperform long latent spans. |
| RL objective | LA-DAPO | Latent-aware rewards improve over GRPO, DePO, and DAPO baselines. |
| RL data scale | 20K | LA-DAPO benefits monotonically from more retained visual-gain data. |

---

## Repository Structure

```text
VideoL1/
├── src/
│   ├── train/                  # SFT and ablation training entry points
│   ├── dataset/                # Interleaved video/image datasets and collators
│   ├── model/                  # Future-L1 model wrapper and latent decoding logic
│   ├── trainer/                # Trainer implementations
│   └── params.py               # Training/data argument definitions
├── scripts/                    # SFT launch scripts and DeepSpeed configs
├── RL_v2/                      # Easy-R1-based LA-DAPO / GRPO / DAPO RL pipeline
├── lmms-eval/                  # Evaluation fork with Future-L1 model/task adapters
├── prompts/                    # System prompts for interleaved latent reasoning
├── plot/                       # Analysis and visualization scripts/assets
├── case_study/                 # Qualitative video examples
├── logs/                       # Training and evaluation logs
├── requirements_sft.txt        # SFT dependencies
└── requirements_rl.txt         # RL dependencies
```

---

## Installation

The project is intended for multi-GPU CUDA environments. The paper experiments use 8×NVIDIA H200 GPUs, bf16 training, and Qwen3-VL-8B-Instruct as the backbone.

### SFT environment

```bash
pip install -r requirements_sft.txt
```

### RL environment

```bash
pip install -r requirements_rl.txt
cd RL_v2
pip install -e .
cd ..
```

### lmms-eval environment

```bash
cd lmms-eval
pip install -e .
cd ..
```

---

## Data Format

SFT examples are JSON objects with multimodal conversations and optional future reasoning images. The important fields are:

```json
{
  "conversations": [
    {"from": "human", "value": "Question ... <video> or <image>"},
    {"from": "gpt", "value": "<reason> ... </reason> <|latent_start|> ... <|latent_end|> ... <answer> ... </answer>"}
  ],
  "image": ["path/to/input_or_prefix_frame.png"],
  "video": ["path/to/video.mp4"],
  "reasoning_image": ["path/to/future_reasoning_frame.png"],
  "answer": "final answer"
}
```

Notes:

- `data_path` accepts one JSON file, a directory of JSON files, or multiple paths.
- Relative media paths are resolved against the JSON file location.
- Future-L1 SFT uses `<|latent|>`, `<|latent_start|>`, and `<|latent_end|>` special tokens.
- TwiFF-style training should enable the TwiFF dataset path in the training script or pass `--use_twiff_dataset True`.

---

## SFT Training

The main SFT entry point is:

```text
src/train/train.py
```

A representative launch script is:

```bash
bash scripts/train_seed6k+twiff50K.sh
```

For a portable run, set the backbone, data, and output paths in the script, then launch with 8 GPUs:

```bash
torchrun \
  --nproc_per_node 8 \
  --master_port 29502 \
  src/train/train.py \
  --deepspeed scripts/zero2.json \
  --model_id /path/to/Qwen3-VL-8B-Instruct \
  --data_path /path/to/Future-L1-50K.json \
  --output_dir /path/to/output/Future-L1-SFT \
  --latent_loss mse \
  --latent_lambda 0.1 \
  --max_latent_token 4 \
  --freeze_vision_tower True \
  --freeze_merger True \
  --freeze_llm False \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16 True \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --use_twiff_dataset True
```

Paper SFT defaults:

| Hyperparameter | Value |
|---|---:|
| Backbone | Qwen3-VL-8B-Instruct |
| Trainable modules | LLM full tuning |
| Frozen modules | vision tower and merger |
| Precision | bf16 |
| Engine | DeepSpeed ZeRO-2 |
| Optimizer | AdamW |
| Peak LR | `1e-5` |
| Global batch size | `128` |
| Sequence length | `16384` |
| Video frames | `16` |
| Latent loss | MSE |
| Latent MSE weight | `0.1` |
| Maximum latent budget | `4` |

Additional SFT and ablation entry points are available under `src/train/`, including text-only SFT, vanilla CoT SFT, dual-MSE variants, and projection-head variants.

---

## RL Training: LA-DAPO

The RL code lives in `RL_v2/` and is implemented on top of an Easy-R1-style pipeline. The default Future-L1 RL config is:

```text
RL_v2/examples/config_future_l1.yaml
```

The common launcher is:

```bash
cd RL_v2
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
RUN_NAME=Future-L1-LA-DAPO \
OUTPUT_DIR=/path/to/output/Future-L1-RL \
bash rjob/train.sh dapo
```

To enable the latent-aware LA-DAPO rewards used by the paper:

```bash
cd RL_v2
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
RUN_NAME=Future-L1-LA-DAPO-colvr020-div010 \
OUTPUT_DIR=/path/to/output/Future-L1-RL \
FUTURE_L1_COLVR_LATENT_COEF=0.2 \
FUTURE_L1_DIVERSITY_PENALTY_BETA=0.1 \
bash rjob/train.sh dapo
```

Paper RL defaults:

| Hyperparameter | Value |
|---|---:|
| Initialization | Future-L1-SFT checkpoint |
| RL framework | Easy-R1-style pipeline in `RL_v2/` |
| Training data | FutureBench 2K / TwiFF-Bench 20K setting |
| Rollout batch | `64` |
| Group size | `8` |
| Max prompt length | `8192` |
| Max response length | `2048` |
| Sampling | temperature `0.9`, top-p `0.99` |
| Accuracy weight | `lambda_a=0.9` |
| Format weight | `lambda_f=0.1` |
| Outcome-contrastive weight | `lambda_c=0.2` |
| Temporal-diversity weight | `lambda_d=0.1` |
| DAPO clip | `epsilon_l=0.2`, `epsilon_h=0.28` |
| KL coefficient | `1e-2` |
| Online group filter | mean accuracy in `[0.1, 0.9]` |
| Judge model | Qwen3.6-27B |

The RL launcher supports GRPO, DAPO, DePO, and several latent-ablation modes through the first argument to `rjob/train.sh`.

---

## lmms-eval Evaluation

Future-L1 evaluation is integrated into the local `lmms-eval/` fork through the `future_l1` model wrapper.

### FutureBench

```bash
cd lmms-eval
export ROT_CODE_ROOT=/path/to/VideoL1

accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
  --model future_l1 \
  --model_args pretrained=/path/to/Future-L1-RL,code_root=${ROT_CODE_ROOT},attn_implementation=flash_attention_2,max_num_frames=32 \
  --tasks futurebench_future_l1 \
  --batch_size 1 \
  --gen_kwargs max_new_tokens=2048,temperature=0,top_p=1,do_sample=false \
  --log_samples \
  --log_samples_suffix future_l1 \
  --output_path ./logs_futurebench_future_l1/
```

Equivalent script:

```bash
bash examples/eval_futurebench_future_l1.sh
```

### TwiFF-Bench

```bash
cd lmms-eval
export ROT_CODE_ROOT=/path/to/VideoL1

accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
  --model future_l1 \
  --model_args pretrained=/path/to/Future-L1-RL,code_root=${ROT_CODE_ROOT},attn_implementation=flash_attention_2,max_num_frames=8 \
  --tasks twiffbench_future_l1 \
  --batch_size 1 \
  --gen_kwargs max_new_tokens=4096,temperature=0,top_p=1,do_sample=false \
  --log_samples \
  --log_samples_suffix twiffbench_future_l1 \
  --output_path ./logs_twiffbench_future_l1/
```

Equivalent script:

```bash
bash examples/eval_twiffbench_future_l1.sh
```

Paper evaluation settings:

| Benchmark | Frames | Decoding | Max new tokens | Metric |
|---|---:|---|---:|---|
| FutureBench | 32 | greedy | 2048 | multiple-choice accuracy |
| TwiFF-Bench | 8 | greedy | 4096 | CoT, answer, average score |

---

## Practical Notes

- Many provided launch scripts contain cluster-specific absolute paths. Replace `MODEL_PATH`, `DATA_PATH`, `TRAIN_FILES`, `OUTPUT_DIR`, and log directories before running on a new machine.
- Use different `MASTER_PORT` or `main_process_port` values when running multiple jobs on the same node.
- Keep `WANDB_MODE=offline` if running in restricted or air-gapped environments.
- For reproducible evaluation, use deterministic lmms-eval decoding: `temperature=0`, `top_p=1`, and `do_sample=false`.
- If a checkpoint does not include Future-L1 special tokens, the RL launcher will reject it unless an explicit plain-Qwen ablation mode is enabled.

---

## Citation

If you find Future-L1 useful, please cite:

```bibtex
@article{futurel1,
  title   = {Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction},
  author  = {Anonymous},
  journal = {EMNLP},
  year    = {2026}
}
```

---

## License

This repository follows the license in `LICENSE`. Third-party datasets, benchmarks, base models, and evaluation frameworks should be used under their respective licenses and terms.
