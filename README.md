<h1 align="center">💭 Imagine Before You Predict</h1>
<h3 align="center">Interleaved Latent Visual Reasoning for Video Event Prediction</h3>

<p align="center">
  <a href="#highlights"><b>Highlights</b></a> •
  <a href="#getting-started"><b>Getting Started</b></a> •
  <a href="#acknowledgements"><b>Acknowledgements</b></a> •
  <a href="#citation"><b>Citation</b></a>
  <!-- <a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b" alt="arXiv"/></a> -->
</p>

<p align="center">
  <b>Future-L1</b> teaches multimodal LLMs to alternate between language tokens and continuous latent visual spans, enabling compact future-state imagination before answering video event prediction questions.
</p>

<p align="center">
  <img src="asset/figure1.png" width="96%" alt="Motivation of interleaved latent visual reasoning"/>
</p>
<p align="center"><em><b>Figure 1.</b> Text-CoT can be verbose and visually lossy, while pixel-space future simulation is computationally heavy. Future-L1 inserts compact latent visual spans that preserve dynamic future semantics without generating full frames.</em></p>

---

## ✨ Highlights

- **Interleaved latent visual reasoning.** Future-L1 alternates between `<reason>` text and bounded `<|latent_start|>…<|latent_end|>` spans during autoregressive decoding, keeping dynamic visual structure in a continuous channel instead of verbalizing every intermediate hypothesis.
- **Future-L1-50K.** We curate 50K high-utility examples from TwiFF-style trajectories by **visual-gain selection**: retain samples where intermediate future visual hints measurably improve prediction over a text-only baseline.
- **LA-DAPO RL.** A latent-aware extension of DAPO with **outcome-contrastive** (`R_ctr`) and **temporal-diversity** (`R_div`) rewards that optimize sampled latent trajectories without intermediate-frame annotations at RL time.
- **State-of-the-art VEP performance.** Future-L1-RL reaches **85.4%** on FutureBench and **3.04** average score on TwiFF-Bench, with especially strong gains on multi-hop and non-consecutive future-event splits.
- **Compact inference.** Accuracy improves through latent visual computation rather than long text-only chains or multi-turn search.

<p align="center">
  <img src="asset/figure2.png" width="96%" alt="Future-L1 pipeline"/>
</p>
<p align="center"><em><b>Figure 2.</b> (Left) Future-L1-50K is built by ranking TwiFF candidates by visual gain <i>p<sub>v</sub> − p<sub>t</sub></i>. (Center) SFT trains interleaved text–latent trajectories, aligning latent spans with future visual states. (Right) LA-DAPO further optimizes sampled trajectories with outcome-contrastive and temporal-diversity rewards.</em></p>

<p align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="asset/futurebench_latent_span_donut.png" width="100%" alt="Latent-span usage by reasoning depth"/>
<br/><br/>
<em><b>Figure 4.</b> Latent-span usage by reasoning depth. Donuts show span-count distributions; values report mean spans over six RL settings.</em>
</td>
<td align="center" width="50%">
<img src="asset/data_volume_combined.png" width="100%" alt="RL data scaling on TwiFF-Bench"/>
<br/><br/>
<em><b>Figure 5.</b> RL data scaling on TwiFF-Bench. Scores improve as LA-DAPO uses 5K, 10K, and 20K retained visual-gain samples.</em>
</td>
</tr>
</table>
</p>

**Table 1. Main results on FutureBench.** Accuracy (%); best results are in **bold**.

| Model | Size | Method | Frames | 1-Hop | 2-Hop | 3-Hop | Interp. | AVG |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| *Open-source and Proprietary Models* |||||||||
| Qwen2.5-VL-Instruct | 72B | Zero-Shot | 32 | 55.5 | 68.4 | 63.7 | 53.2 | 58.3 |
| Qwen3-VL | 30B-A3B | Zero-Shot | 32 | 65.3 | 70.5 | 76.1 | 62.2 | 66.9 |
| GPT-4o | — | Zero-Shot | 32 | 61.9 | 61.7 | 72.1 | 51.6 | 59.0 |
| GPT-5 | — | Zero-Shot | 32 | 59.6 | 57.3 | 62.6 | 55.6 | 57.9 |
| *Video Reasoning Models* |||||||||
| Video-RFT | 7B | SFT+RL | 32 | 62.4 | 53.9 | 50.7 | 53.8 | 54.6 |
| Video-R1 | 7B | SFT+RL | 32 | 67.6 | 65.3 | 61.2 | 61.8 | 63.3 |
| VideoAuto-R1 | 8B | SFT+RL | 32 | 63.6 | 69.4 | 67.7 | 59.3 | 63.4 |
| Video-o3 | 7B | SFT+RL | 32 | 68.2 | 73.6 | 63.2 | 69.7 | 68.9 |
| NEP | 7B | SFT+RL | 32 | 66.2 | 69.9 | 63.7 | 68.1 | 67.3 |
| Video-CoE | 7B | SFT+RL | 32 | 80.9 | 83.9 | 71.6 | 71.4 | 75.0 |
| *Latent Visual Reasoning Models* |||||||||
| LVR | 7B | SFT+RL | 32 | 22.5 | 26.4 | 22.9 | 17.6 | 21.0† |
| Monet | 7B | SFT+RL | 32 | 46.8 | 47.2 | 45.3 | 49.7 | 47.9 |
| SwimBird | 8B | SFT | 32 | 59.0 | 66.8 | 64.7 | 61.8 | 62.8 |
| *Ours* |||||||||
| Qwen3-VL-Instruct | 8B | Zero-Shot | 32 | 64.2 | 65.8 | 66.2 | 55.8 | 61.0 |
| Text-Only SFT (on Future-L1-50K) | 8B | SFT | 32 | 67.6 | 66.8 | 68.2 | 62.0 | 65.0 |
| **Future-L1-SFT** | 8B | SFT | 32 | 70.5 | 73.1 | 77.6 | 72.2 | **73.2** |
| **Future-L1-RL** | 8B | SFT+RL | 32 | **83.2** | **86.5** | **86.6** | **85.1** | **85.4** |

<p align="center">
<table>
<tr>
<td valign="top" width="52%">

**Table 2. Main results on TwiFF-Bench.** Avg. = (CoT + Ans) / 2; best results are in **bold**.

| Model | Size | CoT | Answer | Avg. |
|---|---:|---:|---:|---:|
| *Multimodal Large Language Models* |||||
| Qwen2.5-VL | 7B | 2.46 | 1.63 | 2.05 |
| InternVL3.5 | 8B | 2.35 | 1.85 | 2.10 |
| DeepEyes | 7B | 2.54 | 2.20 | 2.37 |
| *Unified Models* |||||
| Janus-Pro | 7B | 2.04 | 1.04 | 1.54 |
| Bagel | 7B | 2.29 | 1.85 | 2.07 |
| TwiFF-300K | 7B | 2.90 | 2.55 | 2.73 |
| TwiFF-2.7M | 7B | 2.95 | 2.62 | 2.79 |
| *Ours* |||||
| Zero-Shot | 8B | 2.75 | 2.14 | 2.44 |
| **Future-L1-SFT** | 8B | 2.62 | 2.42 | 2.52 |
| **Future-L1-RL** | 8B | **3.11** | **2.97** | **3.04** |

</td>
<td valign="top" width="48%">

**Table 7. Inference cost on FutureBench.** Average tokens, accuracy, latency, and accuracy per second.

| Model | Tokens ↓ | Acc. ↑ | Latency (s) ↓ | Acc./s ↑ |
|---|---:|---:|---:|---:|
| Video-R1 | 398.5 | 63.3 | 3.28 | 19.3 |
| Video-o3 | 348.6 | 68.9 | 25.90 | 2.7 |
| Qwen3-VL-8B | 288.8 | 61.0 | 1.18 | 51.7 |
| **Future-L1-SFT** | 205.3 | 73.1 | 0.96 | 76.1 |
| **Future-L1-RL** | **195.3** | **85.4** | **0.91** | **93.8** |

</td>
</tr>
</table>
</p>

---

## 🚀 Getting Started

```bash
# Install
pip install -r requirements_sft.txt
pip install -r requirements_rl.txt
cd RL_v2 && pip install -e . && cd ..
cd lmms-eval && pip install -e . && cd ..

# Replace chat_template.json before training (once on the base Qwen3-VL checkpoint)
cp chat_template.json /path/to/Qwen3-VL-8B-Instruct/chat_template.json

# SFT — edit MODEL_NAME / DATA_PATH / OUTPUT_DIR in scripts/train_twiff.sh
bash scripts/train_twiff.sh

# RL — set checkpoint, data, and LLM-as-judge API (OpenAI-compatible, e.g. Qwen3.6-27B)
cd RL_v2
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
JUDGE_API_URL=http://localhost:8000/v1 \
JUDGE_API_NAME=your-judge-model \
JUDGE_API_KEY=your-api-key \
FUTURE_L1_LATENT_CTR_LAMBDA=0.2 \
FUTURE_L1_LATENT_DIV_LAMBDA=0.1 \
bash train.sh dapo

# Evaluation — edit model_path in the eval scripts; TwiFF-Bench also needs lmms-eval/.env
cd lmms-eval
cp .env.example .env   # fill OPENAI_API_KEY, OPENAI_API_BASE, LOCAL_LLM
bash examples/eval_futurebench_future_l1.sh
bash examples/eval_twiffbench_future_l1.sh
```

Before running, configure paths and services in the launch scripts / environment:

| Stage | Required | Notes |
|---|---|---|
| **SFT** | `MODEL_NAME`, `DATA_PATH`, `OUTPUT_DIR` | `MODEL_NAME` = base **Qwen3-VL-8B-Instruct**; copy `chat_template.json` into that checkpoint once |
| **RL** | `MODEL_PATH`, `TRAIN_FILES` | `MODEL_PATH` = **Future-L1-SFT** checkpoint (not the raw base model) |
| **RL judge** | `JUDGE_API_URL`, `JUDGE_API_NAME`, `JUDGE_API_KEY` | OpenAI-compatible endpoint for accuracy reward (`USE_LLM_JUDGE=1` by default) |
| **RL (LA-DAPO)** | `FUTURE_L1_LATENT_CTR_LAMBDA`, `FUTURE_L1_LATENT_DIV_LAMBDA` | Paper defaults: `0.2` / `0.1` |
| **Eval** | `model_path` in eval scripts | FutureBench is rule-based; TwiFF-Bench reads `OPENAI_API_KEY`, `OPENAI_API_BASE`, `LOCAL_LLM` from `lmms-eval/.env` |

---

## 🙏 Acknowledgements

We gratefully acknowledge the contributions of the open-source community, particularly:

- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [Latent Visual Reasoning (LVR)](https://github.com/VincentLeebang/lvr), [SwimBird](https://github.com/Accio-Lab/SwimBird), [EasyR1](https://github.com/hiyouga/easyr1)
- Previous Work: [LaViT](https://github.com/Svardfox/LaViT) — Aligning latent visual thoughts for multi-modal reasoning via teacher-extracted visual thought trajectories.

---

## 📖 Citation

```bibtex
@article{tbd,
  title   = {TBD},
  author  = {TBD},
  year    = {TBD}
}
```