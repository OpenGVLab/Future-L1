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
<em><b>Figure 4.</b> Latent-span usage by reasoning depth. Donuts show span-count distributions.</em>
</td>
<td align="center" width="50%">
<img src="asset/data_volume_combined.png" width="100%" alt="RL data scaling on TwiFF-Bench"/>
<br/><br/>
<em><b>Figure 5.</b> RL data scaling on TwiFF-Bench. Scores improve as LA-DAPO uses 5K, 10K, and 20K retained visual-gain samples.</em>
</td>
</tr>
</table>
</p>

<p align="center">
<table align="center" width="72%">
<caption align="center"><b>Table 1. Main results on FutureBench.</b> Accuracy (%); best results are in <b>bold</b>.</caption>
<thead>
<tr>
<th align="left">Model</th><th align="left">Method</th>
<th align="right">1-Hop</th><th align="right">2-Hop</th><th align="right">3-Hop</th><th align="right">Interp.</th><th align="right">AVG</th>
</tr>
</thead>
<tbody>
<tr><td>GPT-5</td><td>Zero-Shot</td><td align="right">59.6</td><td align="right">57.3</td><td align="right">62.6</td><td align="right">55.6</td><td align="right">57.9</td></tr>
<tr><td>Video-R1</td><td>SFT+RL</td><td align="right">67.6</td><td align="right">65.3</td><td align="right">61.2</td><td align="right">61.8</td><td align="right">63.3</td></tr>
<tr><td>VideoAuto-R1</td><td>SFT+RL</td><td align="right">63.6</td><td align="right">69.4</td><td align="right">67.7</td><td align="right">59.3</td><td align="right">63.4</td></tr>
<tr><td>Video-o3</td><td>SFT+RL</td><td align="right">68.2</td><td align="right">73.6</td><td align="right">63.2</td><td align="right">69.7</td><td align="right">68.9</td></tr>
<tr><td>NEP</td><td>SFT+RL</td><td align="right">66.2</td><td align="right">69.9</td><td align="right">63.7</td><td align="right">68.1</td><td align="right">67.3</td></tr>
<tr><td>Video-CoE</td><td>SFT+RL</td><td align="right">80.9</td><td align="right">83.9</td><td align="right">71.6</td><td align="right">71.4</td><td align="right">75.0</td></tr>
<tr><td>Monet</td><td>SFT+RL</td><td align="right">46.8</td><td align="right">47.2</td><td align="right">45.3</td><td align="right">49.7</td><td align="right">47.9</td></tr>
<tr><td>SwimBird</td><td>SFT</td><td align="right">59.0</td><td align="right">66.8</td><td align="right">64.7</td><td align="right">61.8</td><td align="right">62.8</td></tr>
<tr><td>Qwen3-VL-Instruct</td><td>Zero-Shot</td><td align="right">64.2</td><td align="right">65.8</td><td align="right">66.2</td><td align="right">55.8</td><td align="right">61.0</td></tr>
<tr><td><b>Future-L1-SFT</b></td><td>SFT</td><td align="right">70.5</td><td align="right">73.1</td><td align="right">77.6</td><td align="right">72.2</td><td align="right"><b>73.2</b></td></tr>
<tr><td><b>Future-L1-RL</b></td><td>SFT+RL</td><td align="right"><b>83.2</b></td><td align="right"><b>86.5</b></td><td align="right"><b>86.6</b></td><td align="right"><b>85.1</b></td><td align="right"><b>85.4</b></td></tr>
</tbody>
</table>
</p>

<p align="center">
<table align="center" width="82%">
<tr>
<td valign="top" width="52%" align="center">
<table align="center" width="95%">
<caption align="center"><b>Table 2. Main results on TwiFF-Bench.</b> Avg. = (CoT + Ans) / 2; best results are in <b>bold</b>.</caption>
<thead>
<tr><th align="left">Model</th><th align="right">CoT</th><th align="right">Answer</th><th align="right">Avg.</th></tr>
</thead>
<tbody>
<tr><td>Bagel</td><td align="right">2.29</td><td align="right">1.85</td><td align="right">2.07</td></tr>
<tr><td>TwiFF-2.7M</td><td align="right">2.95</td><td align="right">2.62</td><td align="right">2.79</td></tr>
<tr><td>Zero-Shot</td><td align="right">2.75</td><td align="right">2.14</td><td align="right">2.44</td></tr>
<tr><td><b>Future-L1-SFT</b></td><td align="right">2.62</td><td align="right">2.42</td><td align="right">2.52</td></tr>
<tr><td><b>Future-L1-RL</b></td><td align="right"><b>3.11</b></td><td align="right"><b>2.97</b></td><td align="right"><b>3.04</b></td></tr>
</tbody>
</table>
</td>
<td valign="top" width="48%" align="center">
<table align="center" width="95%">
<caption align="center"><b>Table 7. Inference cost on FutureBench.</b> Average tokens, accuracy, latency, and accuracy per second.</caption>
<thead>
<tr><th align="left">Model</th><th align="right">Tokens ↓</th><th align="right">Acc. ↑</th><th align="right">Latency (s) ↓</th><th align="right">Acc./s ↑</th></tr>
</thead>
<tbody>
<tr><td>Video-R1</td><td align="right">398.5</td><td align="right">63.3</td><td align="right">3.28</td><td align="right">19.3</td></tr>
<tr><td>Video-o3</td><td align="right">348.6</td><td align="right">68.9</td><td align="right">25.90</td><td align="right">2.7</td></tr>
<tr><td>Qwen3-VL-8B</td><td align="right">288.8</td><td align="right">61.0</td><td align="right">1.18</td><td align="right">51.7</td></tr>
<tr><td><b>Future-L1-SFT</b></td><td align="right">205.3</td><td align="right">73.1</td><td align="right">0.96</td><td align="right">76.1</td></tr>
<tr><td><b>Future-L1-RL</b></td><td align="right"><b>195.3</b></td><td align="right"><b>85.4</b></td><td align="right"><b>0.91</b></td><td align="right"><b>93.8</b></td></tr>
</tbody>
</table>
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