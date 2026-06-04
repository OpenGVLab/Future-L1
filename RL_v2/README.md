# RL_v2 — FutureL1 × EasyR1 with GRPO / DAPO / DePO

This package implements RL training for FutureL1 (`VideoL1/src/model/future_l1.py`)
on top of EasyR1, with HyLar/RL-aligned config and a single-flag switch between
**GRPO**, **DAPO**, and **DePO** (Decoupled Hybrid PPO + closed-form vMF KL).

It is a clean fork of `EasyR1/verl` with the minimum surgical changes needed to:

1. Apply FutureL1's transformers monkey patches (`qwen3_vl_generation_forward` /
   `qwen3_vl_mixed_modality_forward`) before model construction.
2. Swap vLLM's v1 `GPUModelRunner` for `future_l1_gpu_model_runner` that
   reproduces FutureL1's `_future_l1_sample` semantics during rollout:
   * Detects `<|latent_start|>` and activates latent mode on the next step.
   * Forces `<|latent|>` for every step inside the latent span; feeds the
     previous step's projected hidden state back as the input embedding.
   * Forces `<|latent_end|>` when the budget (`FUTURE_L1_LATENT_SIZE`) is hit.
   * Streams per-step latent vectors via a binary-TCP hook to a driver-side
     `LatentRecorder` (only when `sampling_strategy=future_l1_depo`).
3. Extend `dp_actor.py` to score latent positions with a closed-form vMF
   log-prob (`kappa * cos(actor_hidden, z_rollout)`), and to optionally split
   the PPO surrogate into separate token / latent terms with different clip
   ratios (DePO D-1) and a closed-form vMF KL constraint (DePO D-2).


## LA-DAPO latent rewards (R_ctr + R_div)

Implements LA-DAPO latent rewards from the paper as **outcome rewards** in
`future_l1_reward_function.py`:

* **Outcome-contrastive** ``R_ctr``: ``overall += latent_ctr_lambda * R_ctr``,
  grouped by rollout ``uid``, using recorded per-step latents (hardest-positive
  InfoNCE, paper Eq. ctr). Default off (`latent_ctr_lambda=0`); `depo_ctr` sets
  ``lambda_c=1.0`` and ``tau=0.5``.
* **Temporal diversity** ``R_div``: ``overall -= latent_div_lambda * mean(cos^2)``
  (equivalent to ``+ latent_div_lambda * R_div``). Default off
  (`latent_div_lambda=0`); set ``FUTURE_L1_LATENT_DIV_LAMBDA=0.1`` to enable.

Both are compatible with DePO (shared GRPO/DAPO advantage on text and latent PPO).
Requires `worker.rollout.sampling_strategy=future_l1_depo` and `rollout.n >= 2`.

## Layout

```
RL_v2/
  future_l1_rl_patch.py            # runtime patch entry (transformers + vLLM)
  future_l1_rl/
    transformers_patch.py         # dispatches to VideoL1/src/train/monkey_patch_forward
    vllm_runner/
      future_l1_gpu_model_runner.py  # drop-in for vllm.v1.worker.gpu_model_runner
      latent_hook.py / latent_recorder.py  # per-step latent IPC
  verl/                            # forked EasyR1 (minimal-diff extensions)
  examples/
    config_future_l1.yaml           # HyLar-aligned config
    format_prompt/future_l1_format.jinja
    reward_function/future_l1_reward_function.py
  train.sh                         # ./train.sh grpo | dapo | depo
```

## Usage

```bash
cd VideoL1/RL_v2

# Required env (or pass as CLI overrides):
export MODEL_PATH=/path/to/FutureL1_Qwen3VL/checkpoint
export TRAIN_FILES=/path/to/train.parquet
# Optional: export VAL_FILES=/path/to/val.parquet  # reward-based val (legacy)
# FutureBench lmms_eval (default: once on final checkpoint after training):
# export SKIP_POST_TRAIN_EVAL=0
# export POST_TRAIN_EVAL_LAST_ONLY=1
# Mid-training FutureBench every N steps (optional):
# export FUTUREBENCH_EVAL_DURING_TRAIN=1
# export FUTUREBENCH_EVAL_FREQ=50
# Optional: export FUTURE_L1_LATENT_SIZE=32

# GRPO baseline
bash train.sh grpo

# DAPO (online-filter pass rates + asymmetric clip)
bash train.sh dapo

# DePO (decoupled PPO + closed-form vMF KL on latents)
bash train.sh depo

# DePO + LA-DAPO outcome-contrastive reward R_ctr (paper tau=0.5)
bash train.sh depo_ctr
# Or enable on any mode:
# FUTURE_L1_LATENT_CTR_LAMBDA=0.2 FUTURE_L1_LATENT_CTR_TEMPERATURE=0.5 bash train.sh depo
# Full LA-DAPO (R_ctr + R_div):
# FUTURE_L1_LATENT_CTR_LAMBDA=0.2 FUTURE_L1_LATENT_DIV_LAMBDA=0.1 bash train.sh depo
```

The launcher auto-detects:
* The backbone `model_type` (only `qwen3_vl` supported in this drop; other
  Qwen backbones are wired through `future_l1_rl/transformers_patch.py`).
* The three FutureL1 special-token ids (`<|latent_start|>`, `<|latent|>`,
  `<|latent_end|>`) from the tokenizer and exports them as
  `FUTURE_L1_LATENT_*` env vars consumed by both the transformers patch and
  the vLLM GPU runner.

## Mode summary

All three baselines are **HyLar-equivalent** (HyLar/RL `hylar` strategy) and
differ only in the toggles HyLar itself flips between GRPO / DAPO / DePO. The
single mechanical divergence from HyLar/RL is the base model
(`FutureL1_Qwen3VL` instead of `FutureL1_Qwen2_5_VL`) and the corresponding
chat-format prompt.

> **Audit trail.** The full equivalence proof, diagnosis of the historical
> silent-degrade bug, and validation checklist live in
> [`docs/HYLAR_ALIGNMENT.md`](docs/HYLAR_ALIGNMENT.md). Cite that file from
> the paper appendix / reviewer rebuttal.

| Mode   | adv_estimator | online_filtering | decoupled PPO | vMF KL | sampling_strategy |
|--------|---------------|------------------|---------------|--------|-------------------|
| grpo   | grpo          | off              | off           | off    | future_l1_depo     |
| dapo   | dapo          | on (accuracy)    | off           | off    | future_l1_depo     |
| depo   | dapo          | on (accuracy)    | on            | on     | future_l1_depo     |

`future_l1_depo` (the HyLar-equivalent path) does three things together:

1. **Latent-aware vLLM rollout** — FutureL1's `future_l1_gpu_model_runner`
   detects `<|latent_start|>`, forces `<|latent|>` for every step of the
   span, feeds the previous step's projected hidden state back as the next
   step's input embedding, and forces `<|latent_end|>` when the budget is hit.
2. **Per-step latent recording** — a TCP `LatentRecorder` snapshots each
   step's latent vector `z` and writes it to `non_tensor_batch["latents"]`
   for the actor to consume.
3. **vMF log-prob substitution** — inside `dp_actor._forward_micro_batch`,
   the per-token categorical log-prob at every latent position is replaced
   by `kappa * cos(actor_hidden, z_rollout)`, so PPO's ratio at latent
   positions optimises the *direction of the actor's hidden state*, not the
   probability of the placeholder `<|latent|>` token id.

The DePO-specific extras (decoupled token/latent PPO surrogate and closed-form
vMF KL) are *separately* gated by their own `algorithm.enable_*` flags.

> **Ablation-only modes.** `future_l1_text` (latent-aware rollout, no latent
> recording, no vMF substitution) and `default` (stock EasyR1 rollout, no
> FutureL1 patch) are kept as ablation handles to isolate "rollout semantics"
> from "latent-aware actor". They are **not** HyLar-equivalent and must not
> be used for headline GRPO/DAPO numbers.

## Projection head

By default the vLLM rollout feeds the **raw** last-layer hidden state back as
the pending latent embedding. FutureL1's HF path applies a learned projection
head (`projection_head` or `projection_head_render`) first. To match the HF
path exactly during rollout, set `FUTURE_L1_APPLY_PROJ_HEAD=1`; the runner will
build the head module from `config.use_projection_head` / `use_dual_projection_heads` /
`projection_head_type` and load its weights from the checkpoint's safetensors.
The actor's training-time forward already uses the model's full forward (with
its existing projection-head integration via FutureL1's monkey patches), so
this only affects the vMF target `z_rollout`.

## Algorithm notes

* `dapo` advantage is registered as an explicit alias of `grpo` for clarity in
  configs and logs (see `verl/trainer/core_algos.py`); the actual DAPO
  behaviour (asymmetric clip + online filtering) is driven by
  `worker.actor.clip_ratio_high` + `algorithm.online_filtering`.
* `compute_latent_vmf_kl(mu_actor, z_rollout, kappa)` returns the mean
  `kappa * (1 - <mu_actor, z_rollout>)` over latent positions; the actor adds
  it to the policy loss when `enable_latent_vmf_kl=True` and FutureL1 latents
  are present in the batch.
* `algorithm.enable_format_gated_latent_loss=true` gates latent-only objectives
  by per-sample `reward_format_scores`: malformed responses still get token PPO
  / format-reward pressure, but their latent PPO and latent vMF KL losses are zeroed.
* Sample-based KL (the standard `compute_kl` term) is masked away at latent
  positions in DePO mode so it does not double-count against the vMF term.

## Known caveats

* The vLLM runner is a copy of vLLM's `v1` GPU runner. Upgrading vLLM
  requires re-syncing changes; the only diffs to this baseline are inside
  `__init__` (FutureL1 env vars + state), the latent override / emission
  blocks of `execute_model`, and `load_model` (optional projection head).
* Multi-image / video processing follows the EasyR1 baseline; FutureL1's
  on-the-fly latent-image emission (`pixel_values_latent`) is **not** active
  during RL rollout — only its text-mode latent stream is.
