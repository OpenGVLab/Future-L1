<h1 align="center">Imagine Before You Predict</h1>
<h3 align="center">Interleaved Latent Visual Reasoning for Video Event Prediction</h3>

<p align="center">
  <a href="#highlights"><b>Highlights</b></a> •
  <a href="#getting-started"><b>Getting Started</b></a> •
  <a href="#acknowledgements"><b>Acknowledgements</b></a> •
  <a href="#previous-work"><b>Previous Work</b></a> •
  <a href="#citation"><b>Citation</b></a>
</p>

<p align="center">
  <b>Future-L1</b> teaches multimodal LLMs to alternate between language tokens and continuous latent visual spans, enabling compact future-state imagination before answering video event prediction questions.
</p>

<p align="center">
  <img src="asset/figure1.png" width="96%" alt="Motivation of interleaved latent visual reasoning"/>
</p>
<p align="center"><em><b>Figure 1.</b> Text-CoT can be verbose and visually lossy, while pixel-space future simulation is computationally heavy. Future-L1 inserts compact latent visual spans that preserve dynamic future semantics without generating full frames.</em></p>

---

## TL;DR

| | FutureBench (Acc.) | TwiFF-Bench (Avg.) |
|---|---:|---:|
| Qwen3-VL-8B (zero-shot) | 61.0 | 2.44 |
| Text-only SFT on Future-L1-50K | 65.0 | — |
| **Future-L1-SFT** | **73.2** | 2.52 |
| **Future-L1-RL (LA-DAPO)** | **85.4** | **3.04** |

> **+24.4** points on FutureBench over the Qwen3-VL-8B backbone · **+10.4** over the previous best Video-CoE · **+0.60** average score on TwiFF-Bench

---

## Highlights

- **Interleaved latent visual reasoning.** Future-L1 alternates between `<reason>` text and bounded `<|latent_start|>…<|latent_end|>` spans during autoregressive decoding, keeping dynamic visual structure in a continuous channel instead of verbalizing every intermediate hypothesis.
- **Future-L1-50K.** We curate 50K high-utility examples from TwiFF-style trajectories by **visual-gain selection**: retain samples where intermediate future visual hints measurably improve prediction over a text-only baseline.
- **LA-DAPO RL.** A latent-aware extension of DAPO with **outcome-contrastive** (`R_ctr`) and **temporal-diversity** (`R_div`) rewards that optimize sampled latent trajectories without intermediate-frame annotations at RL time.
- **State-of-the-art VEP performance.** Future-L1-RL reaches **85.4%** on FutureBench and **3.04** average score on TwiFF-Bench, with especially strong gains on multi-hop and non-consecutive future-event splits.
- **Compact inference.** Accuracy improves through latent visual computation rather than long text-only chains or multi-turn search.

---

## Getting Started

### Installation

Paper experiments use **8× NVIDIA H200 GPUs**, **bf16** training, and **Qwen3-VL-8B-Instruct** as the backbone.

```bash
# SFT
pip install -r requirements_sft.txt

# RL
pip install -r requirements_rl.txt
cd RL_v2 && pip install -e . && cd ..

# Evaluation
cd lmms-eval && pip install -e . && cd ..
```

### Prepare the backbone checkpoint

Before SFT or RL training, replace the Qwen3-VL weight directory's `chat_template.json` with the one shipped in this repository. This registers Future-L1 special tokens (`<|latent_start|>`, `<|latent|>`, `<|latent_end|>`, `<reason>`, etc.) in the chat template, following the same practice as [SwimBird](https://github.com/Accio-Lab/SwimBird).

```bash
cp chat_template.json /path/to/Qwen3-VL-8B-Instruct/chat_template.json
```

Run this once on the base backbone before `scripts/train.sh` / `scripts/train_twiff.sh`. Checkpoints saved from that run will carry the updated template forward to RL and evaluation.

### SFT Training

Main entry point: `src/train/train.py`

```bash
# Future-L1 format
bash scripts/train.sh

# TwiFF-style mixed dataset
bash scripts/train_twiff.sh
```

Representative command:

```bash
torchrun --nproc_per_node 8 --master_port 29502 src/train/train.py \
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
  --bf16 True \
  --lazy_preprocess True \
  --use_mixed_dataset True
```

| Hyperparameter | Value |
|---|---:|
| Backbone | Qwen3-VL-8B-Instruct |
| Trainable modules | LLM (vision tower + merger frozen) |
| Precision / engine | bf16 + DeepSpeed ZeRO-2 |
| Global batch size | 128 |
| Peak LR | 1e-5 |
| Latent MSE weight `λ` | 0.1 |
| Max latent budget | 4 |

### RL Training (LA-DAPO)

RL code lives in `RL_v2/`. Config: `RL_v2/examples/config_future_l1.yaml`.

```bash
cd RL_v2

# DAPO baseline (latent-aware rollout + vMF log-prob)
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
bash train.sh dapo

# DePO (decoupled token/latent PPO + closed-form vMF KL)
bash train.sh depo

# Full LA-DAPO with paper reward weights
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
FUTURE_L1_LATENT_CTR_LAMBDA=0.2 \
FUTURE_L1_LATENT_DIV_LAMBDA=0.1 \
FUTURE_L1_LATENT_CTR_TEMPERATURE=0.5 \
bash train.sh depo
```

Supported modes: `grpo`, `dapo`, `depo`, and `grpo_ctr` / `dapo_ctr` / `depo_ctr` (base mode + outcome-contrastive `R_ctr`).

| Hyperparameter | Value |
|---|---:|
| Rollout batch / group size | 64 / 8 |
| Max prompt / response length | 8192 / 2048 |
| Sampling | temperature 0.9, top-p 0.99 |
| DAPO clip | ε_l=0.2, ε_h=0.28 |
| Online group filter | mean accuracy ∈ [0.1, 0.9] |
| `λ_c` / `λ_d` / `τ` | 0.2 / 0.1 / 0.5 |

### Evaluation

Future-L1 evaluation is integrated into the local `lmms-eval/` fork via the `future_l1` model wrapper.

**FutureBench** (32 frames, greedy decoding):

```bash
cd lmms-eval
export ROT_CODE_ROOT=/path/to/Future-L1

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
  --model future_l1 \
  --model_args pretrained=/path/to/Future-L1-RL,code_root=${ROT_CODE_ROOT},attn_implementation=flash_attention_2,max_num_frames=32 \
  --tasks futurebench_future_l1 \
  --batch_size 1 \
  --gen_kwargs max_new_tokens=2048,temperature=0,top_p=1,do_sample=false \
  --output_path ./logs_futurebench_future_l1/

# or
bash examples/eval_futurebench_future_l1.sh
```

**TwiFF-Bench** (8 frames, greedy decoding):

```bash
bash examples/eval_twiffbench_future_l1.sh
```

---

## Data Format

SFT examples are JSON objects with multimodal conversations and optional future reasoning images:

```json
{
  "conversations": [
    {"from": "human", "value": "Question ... <video>"},
    {"from": "gpt", "value": "<reason> ... </reason> <|latent_start|> ... <|latent_end|> ... <answer> ... </answer>"}
  ],
  "video": ["path/to/video.mp4"],
  "reasoning_image": ["path/to/future_reasoning_frame.png"],
  "answer": "final answer"
}
```

Relative media paths are resolved against the JSON file location.

---

## Acknowledgements

We gratefully acknowledge the contributions of the open-source community, particularly:

- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- [Latent Visual Reasoning (LVR)](https://github.com/VincentLeebang/lvr)
- [SwimBird](https://github.com/Accio-Lab/SwimBird)
- [EasyR1](https://github.com/hiyouga/easyr1)

---

## Previous Work

- [LaViT](https://github.com/Svardfox/LaViT) — Aligning latent visual thoughts for multi-modal reasoning via teacher-extracted visual thought trajectories.

---

## Citation

```bibtex
@article{tbd,
  title   = {TBD},
  author  = {TBD},
  year    = {TBD}
}
```

Citation will be updated upon publication.
