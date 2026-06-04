<h1 align="center">Imagine Before You Predict</h1>
<h3 align="center">Interleaved Latent Visual Reasoning for Video Event Prediction</h3>

<p align="center">
  <a href="#-highlights"><b>Highlights</b></a> •
  <a href="#-method"><b>Method</b></a> •
  <a href="#-results"><b>Results</b></a> •
  <a href="#-getting-started"><b>Getting Started</b></a> •
  <a href="#-acknowledgements"><b>Acknowledgements</b></a> •
  <a href="#-previous-work"><b>Previous Work</b></a> •
  <a href="#-citation"><b>Citation</b></a>
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

## Method

<p align="center">
  <img src="asset/figure2.png" width="96%" alt="Future-L1 pipeline"/>
</p>
<p align="center"><em><b>Figure 2.</b> (Left) Future-L1-50K is built by ranking TwiFF candidates by visual gain <i>p<sub>v</sub> − p<sub>t</sub></i>. (Center) SFT trains interleaved text–latent trajectories, aligning latent spans with future visual states. (Right) LA-DAPO further optimizes sampled trajectories with outcome-contrastive and temporal-diversity rewards.</em></p>

### Interleaved decoding

Given an observed video prefix and a forecasting question, Future-L1 generates responses as interleaved trajectories:

```text
<reason> textual reasoning step 0 </reason>
<|latent_start|> <|latent|> ... <|latent|> <|latent_end|>
<reason> textual reasoning step 1 </reason>
...
<answer> predicted future event </answer>
```

Three special tokens control the latent channel:

| Token | Role |
|---|---|
| `<\|latent_start\|>` | Enter latent visual reasoning mode |
| `<\|latent\|>` | Emit a continuous latent thought fed back as the next input embedding |
| `<\|latent_end\|>` | Return to textual reasoning |

### Stage 1: Visual-gain SFT

Future-L1-50K is selected from TwiFF-style visual chain-of-thought data. For each candidate, the backbone is evaluated under:

- **Text-only:** observed prefix + question
- **Visual-hint:** prefix + question + intermediate future reasoning frames

Samples are retained when text-only accuracy is not saturated and the visual hint provides a measurable lift. SFT combines next-token prediction with latent MSE alignment:

```text
L_SFT = L_CE + λ · L_Latent
```

where latent positions are aligned to Qwen3-VL vision-encoder embeddings of the corresponding future reasoning frames.

### Stage 2: LA-DAPO RL

SFT provides a grounded initialization, but teacher-forced latents are not directly optimized for sampled prediction success. **LA-DAPO** (Latent-Aware Direct Advantage Policy Optimization) keeps DAPO's answer and format rewards and adds two trajectory-level latent terms:

```text
R = λ_a · R_acc + λ_f · R_fmt + λ_c · R_ctr + λ_d · R_div
```

| Reward | Meaning |
|---|---|
| `R_acc` | Final-answer correctness (rule-based + LLM-judge fallback) |
| `R_fmt` | Valid interleaved `<reason>` / latent / `<answer>` format |
| `R_ctr` | Outcome-contrastive latent reward: pull correct trajectories together, treat incorrect rollouts as negatives |
| `R_div` | Temporal-diversity reward: discourage adjacent latent spans from collapsing to the same visual thought |

Paper defaults: `λ_a=0.9`, `λ_f=0.1`, `λ_c=0.2`, `λ_d=0.1`, contrastive temperature `τ=0.5`.

---

## Results

### Table 1: Main results on FutureBench

Accuracy (%); best results in **bold**. Future-L1 uses **32 input frames** at evaluation.

| Model | Size | Method | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---|---:|---:|---:|---:|---:|
| GPT-4o | — | Zero-Shot | 61.9 | 61.7 | 72.1 | 51.6 | 59.0 |
| GPT-5 | — | Zero-Shot | 59.6 | 57.3 | 62.6 | 55.6 | 57.9 |
| Qwen3-VL | 30B-A3B | Zero-Shot | 65.3 | 70.5 | 76.1 | 62.2 | 66.9 |
| Video-R1 | 7B | SFT+RL | 67.6 | 65.3 | 61.2 | 61.8 | 63.3 |
| Video-o3 | 7B | SFT+RL | 68.2 | 73.6 | 63.2 | 69.7 | 68.9 |
| NEP | 7B | SFT+RL | 66.2 | 69.9 | 63.7 | 68.1 | 67.3 |
| Video-CoE | 7B | SFT+RL | 80.9 | 83.9 | 71.6 | 71.4 | 75.0 |
| LVR | 7B | SFT+RL | 22.5 | 26.4 | 22.9 | 17.6 | 21.0 |
| Monet | 7B | SFT+RL | 46.8 | 47.2 | 45.3 | 49.7 | 47.9 |
| SwimBird | 8B | SFT | 59.0 | 66.8 | 64.7 | 61.8 | 62.8 |
| Qwen3-VL-Instruct | 8B | Zero-Shot | 64.2 | 65.8 | 66.2 | 55.8 | 61.0 |
| Text-Only SFT (on Future-L1-50K) | 8B | SFT | 67.6 | 66.8 | 68.2 | 62.0 | 65.0 |
| **Future-L1-SFT** | 8B | SFT | 70.5 | 73.1 | 77.6 | 72.2 | **73.2** |
| **Future-L1-RL** | 8B | SFT+RL | **83.2** | **86.5** | **86.6** | **85.1** | **85.4** |

### Table 2: Main results on TwiFF-Bench

Avg. = (CoT + Ans) / 2; best results in **bold**.

| Model | Size | CoT | Answer | Avg. |
|---|---:|---:|---:|---:|
| Qwen2.5-VL | 7B | 2.46 | 1.63 | 2.05 |
| InternVL3.5 | 8B | 2.35 | 1.85 | 2.10 |
| DeepEyes | 7B | 2.54 | 2.20 | 2.37 |
| Janus-Pro | 7B | 2.04 | 1.04 | 1.54 |
| Bagel | 7B | 2.29 | 1.85 | 2.07 |
| TwiFF-300K | 7B | 2.90 | 2.55 | 2.73 |
| TwiFF-2.7M | 7B | 2.95 | 2.62 | 2.79 |
| Qwen3-VL-Instruct (Zero-Shot) | 8B | 2.75 | 2.14 | 2.44 |
| **Future-L1-SFT** | 8B | 2.62 | 2.42 | 2.52 |
| **Future-L1-RL** | 8B | **3.11** | **2.97** | **3.04** |

### Table 3: SFT hyperparameter ablation on FutureBench

| Setting | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---:|---:|---:|---:|
| *Latent MSE weight λ* | | | | | |
| 0.01 | 68.2 | 69.9 | 73.1 | 67.5 | 69.1 |
| 0.05 | 71.1 | 72.0 | 73.6 | 69.3 | 70.9 |
| **0.10** | 70.5 | 73.1 | **77.6** | **72.2** | **73.2** |
| 0.20 | 69.9 | **76.7** | 74.6 | 70.1 | 72.2 |
| 0.50 | **73.4** | 71.0 | 71.6 | 69.3 | 70.7 |
| 1.00 | **73.4** | 73.1 | 68.7 | 67.1 | 69.5 |
| *Maximum latent budget L_max* | | | | | |
| 2 | 66.5 | 74.1 | 74.6 | 69.3 | 70.7 |
| **4** | 70.5 | 73.1 | **77.6** | 72.2 | **73.2** |
| 8 | 65.9 | **75.1** | 73.6 | **72.4** | 72.1 |
| 16 | 69.9 | 72.5 | 71.1 | 70.8 | 71.0 |
| 32 | 69.4 | 72.0 | 71.1 | 69.5 | 70.3 |
| 64 | 67.1 | 68.9 | 70.6 | 65.6 | 67.4 |

### Table 4: RL objective ablation on FutureBench

| Method | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---:|---:|---:|---:|
| Text-Only SFT | 67.6 | 66.8 | 68.2 | 62.0 | 65.0 |
| &nbsp;&nbsp;+ GRPO | 77.5 | 78.8 | 78.1 | 77.1 | 77.7 |
| &nbsp;&nbsp;+ DAPO | **83.2** | 81.3 | 78.1 | 71.2 | 76.3 |
| Future-L1-SFT | 70.5 | 73.1 | 77.6 | 72.2 | 73.2 |
| &nbsp;&nbsp;+ GRPO | 82.7 | 84.5 | 85.1 | 81.2 | 82.8 |
| &nbsp;&nbsp;+ DePO | 78.0 | 80.3 | 86.6 | 80.2 | 81.1 |
| &nbsp;&nbsp;+ DAPO | **83.2** | 85.5 | 86.6 | 82.4 | 83.8 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ R_ctr | **83.2** | 86.0 | 87.1 | 83.2 | 84.5 |
| &nbsp;&nbsp;&nbsp;&nbsp;+ R_div | 82.7 | **87.0** | **87.6** | 83.4 | 84.8 |
| **Future-L1-RL** | **83.2** | 86.5 | 86.6 | **85.1** | **85.4** |

### Table 5: LA-DAPO reward coefficient ablation on FutureBench

| Setting | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---:|---:|---:|---:|
| *Outcome-contrastive weight λ_c* | | | | | |
| 0.01 | 81.5 | 84.5 | 86.1 | 83.4 | 83.8 |
| 0.05 | 82.7 | **87.0** | 86.1 | 83.0 | 84.3 |
| 0.10 | **84.4** | 86.5 | 87.1 | 84.0 | 85.1 |
| **0.20** | 83.2 | 86.5 | 86.6 | **85.1** | **85.4** |
| 0.50 | 82.1 | 86.0 | 86.1 | 83.0 | 84.0 |
| 1.00 | 83.8 | 86.5 | 86.6 | 84.5 | 85.1 |
| *Temporal diversity weight λ_d* | | | | | |
| 0.01 | 83.2 | 86.5 | 86.6 | 83.8 | 84.8 |
| 0.05 | **83.8** | **87.0** | 86.6 | 84.3 | 85.1 |
| **0.10** | 83.2 | 86.5 | 86.6 | **85.1** | **85.4** |
| 0.20 | 80.9 | 82.9 | **87.1** | 83.2 | 83.5 |
| 0.50 | 79.8 | 83.4 | 85.6 | 81.6 | 82.4 |
| 1.00 | 78.0 | 82.4 | 85.6 | 81.0 | 81.6 |

### Table 6: Effect of visual-gain filtering

| Training Set | 1-Hop | 2-Hop | 3-Hop | Interp. | Avg. |
|---|---:|---:|---:|---:|---:|
| Zero-Shot | 64.2 | 65.8 | 66.2 | 55.8 | 61.0 |
| Random 50K | 67.6 | 68.9 | 70.1 | 67.7 | 68.4 |
| **Future-L1-50K** | **70.5** | **73.1** | **77.6** | **72.2** | **73.2** |

<p align="center">
  <img src="asset/futurebench_latent_span_donut.png" width="88%" alt="Latent-span usage by reasoning depth"/>
</p>
<p align="center"><em><b>Figure 3.</b> Latent-span usage by reasoning depth. Donuts show span-count distributions; values report mean spans over six RL settings.</em></p>

<p align="center">
  <img src="asset/data_volume_combined.png" width="88%" alt="RL data scaling on TwiFF-Bench"/>
</p>
<p align="center"><em><b>Figure 4.</b> RL data scaling on TwiFF-Bench. Scores improve as LA-DAPO uses 5K, 10K, and 20K retained visual-gain samples.</em></p>

### Table 7: Inference cost on FutureBench

| Model | Tokens ↓ | Acc. ↑ | Latency (s) ↓ | Acc./s ↑ |
|---|---:|---:|---:|---:|
| Video-R1 | 398.5 | 63.3 | 3.28 | 19.3 |
| Video-o3 | 348.6 | 68.9 | 25.90 | 2.7 |
| Qwen3-VL-8B | 288.8 | 61.0 | 1.18 | 51.7 |
| **Future-L1-SFT** | 205.3 | 73.1 | 0.96 | 76.1 |
| **Future-L1-RL** | **195.3** | **85.4** | **0.91** | **93.8** |

---

## Repository Structure

```text
Future-L1/
├── asset/                      # Paper figures for README / project page
├── src/
│   ├── train/                  # SFT training entry point
│   ├── dataset/                # Future-L1 / TwiFF / mixed SFT datasets
│   ├── model/                  # Future-L1 model wrapper and latent decoding
│   └── trainer/                # FutureL1SFTTrainer
├── scripts/                    # SFT launch scripts (train.sh, train_twiff.sh)
├── RL_v2/                      # EasyR1-based GRPO / DAPO / DePO / LA-DAPO RL
├── lmms-eval/                  # Evaluation fork with Future-L1 adapters
├── prompts/                    # System prompts for interleaved reasoning
├── requirements_sft.txt
└── requirements_rl.txt
```

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

## Practical Notes

- Replace cluster-specific paths (`MODEL_PATH`, `DATA_PATH`, `TRAIN_FILES`, `OUTPUT_DIR`) before running on a new machine.
- Use different `MASTER_PORT` / `main_process_port` values when launching multiple jobs on one node.
- For reproducible evaluation, use `temperature=0`, `top_p=1`, `do_sample=false` in lmms-eval.
- RL requires Future-L1 special tokens in the checkpoint; the launcher auto-detects `<|latent_start|>`, `<|latent|>`, `<|latent_end|>` from the tokenizer.

---

## Acknowledgements

We gratefully acknowledge the contributions of the open-source community, particularly:

- [**Qwen-VL-Series-Finetune**](https://github.com/2U1/Qwen-VL-Series-Finetune) — SFT training infrastructure for Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL models.
- [**Latent Visual Reasoning (LVR)**](https://github.com/VincentLeebang/lvr) — latent visual reasoning formulation and training recipes that informed our continuous latent-span design.
- [**SwimBird**](https://github.com/Accio-Lab/SwimBird) — hybrid autoregressive MLLM with switchable text / vision / interleaved reasoning modes; our SFT codebase builds on this design.
- [**EasyR1**](https://github.com/hiyouga/easyr1) — efficient multi-modality RL training framework; our `RL_v2/` pipeline is built on top of EasyR1 / veRL.

---

## Previous Work

- [**LaViT**](https://github.com/Svardfox/LaViT)

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
