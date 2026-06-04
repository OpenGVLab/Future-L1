<h1 align="center">💭 Imagine Before You Predict</h1>
<h3 align="center">Interleaved Latent Visual Reasoning for Video Event Prediction</h3>

<p align="center">
  <a href="#highlights"><b>Highlights</b></a> •
  <a href="#getting-started"><b>Getting Started</b></a> •
  <a href="#acknowledgements"><b>Acknowledgements</b></a> •
  <a href="#citation"><b>Citation</b></a> •
  <a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b" alt="arXiv"/></a>
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
  <img src="asset/futurebench_latent_span_donut.png" width="88%" alt="Latent-span usage by reasoning depth"/>
</p>
<p align="center"><em><b>Figure 4.</b> Latent-span usage by reasoning depth. Donuts show span-count distributions; values report mean spans over six RL settings.</em></p>

<p align="center">
  <img src="asset/data_volume_combined.png" width="88%" alt="RL data scaling on TwiFF-Bench"/>
</p>
<p align="center"><em><b>Figure 5.</b> RL data scaling on TwiFF-Bench. Scores improve as LA-DAPO uses 5K, 10K, and 20K retained visual-gain samples.</em></p>

**Table 7. Inference cost on FutureBench.** Average tokens, accuracy, latency, and accuracy per second.

| Model | Tokens ↓ | Acc. ↑ | Latency (s) ↓ | Acc./s ↑ |
|---|---:|---:|---:|---:|
| Video-R1 | 398.5 | 63.3 | 3.28 | 19.3 |
| Video-o3 | 348.6 | 68.9 | 25.90 | 2.7 |
| Qwen3-VL-8B | 288.8 | 61.0 | 1.18 | 51.7 |
| **Future-L1-SFT** | 205.3 | 73.1 | 0.96 | 76.1 |
| **Future-L1-RL** | **195.3** | **85.4** | **0.91** | **93.8** |

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

# SFT
bash scripts/train_twiff.sh    # TwiFF-style mixed dataset

# RL (LA-DAPO: DAPO + R_ctr + R_div)
cd RL_v2
MODEL_PATH=/path/to/Future-L1-SFT \
TRAIN_FILES=/path/to/RL_20K.json \
FUTURE_L1_LATENT_CTR_LAMBDA=0.2 \
FUTURE_L1_LATENT_DIV_LAMBDA=0.1 \
bash train.sh dapo

# Evaluation
cd lmms-eval
bash examples/eval_futurebench_future_l1.sh
bash examples/eval_twiffbench_future_l1.sh
```

Set `MODEL_PATH`, `DATA_PATH`, `TRAIN_FILES`, and `OUTPUT_DIR` in the launch scripts before running.

---

## 🙏 Acknowledgements

We gratefully acknowledge the contributions of the open-source community, particularly:

- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [Latent Visual Reasoning (LVR)](https://github.com/VincentLeebang/lvr), [SwimBird](https://github.com/Accio-Lab/SwimBird), [EasyR1](https://github.com/hiyouga/easyr1)
- [LaViT](https://github.com/Svardfox/LaViT) — Aligning latent visual thoughts for multi-modal reasoning via teacher-extracted visual thought trajectories.

---

## 📖 Citation

```bibtex
@article{tbd,
  title   = {TBD},
  author  = {TBD},
  year    = {TBD}
}
```

Citation will be updated upon publication.
