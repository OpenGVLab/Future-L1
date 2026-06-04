#!/bin/bash
# FutureL1 RL trainer entry point (RL_v2).
# Modes: grpo | dapo | depo
#   depo_ctr | dapo_ctr | grpo_ctr = base mode + outcome-contrastive R_ctr (lambda_c=1, tau=0.5)
set -euo pipefail
set -x

unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

# --------------------------------------------------------------------------
# Datasets / weights — override via environment variables before running.
# --------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/path/to/your/sft/checkpoint}"
TRAIN_FILES="${TRAIN_FILES:-/path/to/your/RL_20K.json}"

# Skip-all-val is the RL_v2 default; leave VAL_FILES empty so the dataloader
# never even constructs the val dataset. Set VAL_FILES to a real path to
# re-enable validation (together with `trainer.val_freq > 0`).
VAL_FILES=${VAL_FILES:-""}

MODE=${1:-grpo}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUTURE_L1_ROOT="${FUTURE_L1_ROOT:-${VIDEO_L1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
export FUTURE_L1_ROOT VIDEO_L1_ROOT="${FUTURE_L1_ROOT}" FUTURE_L1_CODE_ROOT="${FUTURE_L1_ROOT}"
RL_V2_ROOT="${RL_V2_ROOT:-${SCRIPT_DIR}}"

case "${MODE}" in
  grpo) DEFAULT_RUN_NAME="GRPO" ;;
  dapo) DEFAULT_RUN_NAME="DAPO" ;;
  depo) DEFAULT_RUN_NAME="DePO" ;;
  depo_ctr) DEFAULT_RUN_NAME="DePO_LA-DAPO" ;;
  dapo_ctr) DEFAULT_RUN_NAME="DAPO_LA-DAPO" ;;
  grpo_ctr) DEFAULT_RUN_NAME="GRPO_LA-DAPO" ;;
  *) DEFAULT_RUN_NAME="${MODE}" ;;
esac
RUN_NAME=${RUN_NAME:-${EXPERIMENT_NAME:-${DEFAULT_RUN_NAME}}}
# Save path aligned with upstream: gpfs2-shared training output.
OUTPUT_DIR="${OUTPUT_DIR:-${RL_V2_ROOT}/outputs/${RUN_NAME}}"
LOG_DIR=${LOG_DIR:-"${FUTURE_L1_ROOT}/logs"}
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

export FUTURE_L1_CODE_ROOT="${FUTURE_L1_ROOT}"
export FUTURE_L1_RL_V2_ROOT="${RL_V2_ROOT}"
export FUTURE_L1_RL_PATCH=1
# Refuse silent fallback when the FutureL1 patch (transformers monkey patches +
# vLLM `future_l1_gpu_model_runner`) fails to load. With both flags = 1, ANY
# import / runner-replacement failure raises a hard error instead of letting
# the workers degrade to stock vLLM — which would silently invalidate every
# HyLar-equivalent GRPO/DAPO/DePO baseline by skipping latent recording + vMF
# log-prob substitution. Override with `FUTURE_L1_RL_PATCH_REQUIRED=0
# FUTURE_L1_REQUIRE_VLLM_RUNNER=0` only for explicit `default` / `future_l1_text`
# ablation runs.
export FUTURE_L1_RL_PATCH_REQUIRED=${FUTURE_L1_RL_PATCH_REQUIRED:-1}
export FUTURE_L1_REQUIRE_VLLM_RUNNER=${FUTURE_L1_REQUIRE_VLLM_RUNNER:-1}

# --------------------------------------------------------------------------
# LLM-as-judge (OpenAI-compatible, e.g. local vLLM gateway).
# Aligned with upstream: judge ON by default; cluster gateway hard-wired.
# --------------------------------------------------------------------------
export USE_LLM_JUDGE=${USE_LLM_JUDGE:-1}
export JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:8000/v1}"
export JUDGE_API_NAME="${JUDGE_API_NAME:-your-judge-model}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-test}"
export API_JUDGE_WORKERS=${API_JUDGE_WORKERS:-32}
# LLM_JUDGE_ONLY=1 -> skip the rule-based pre-check (rule-first + LLM fallback
# is the default behavior).
export LLM_JUDGE_ONLY=${LLM_JUDGE_ONLY:-0}
# FUTURE_L1_REWARD_DEBUG=1 -> on every reward call, print the first
# FUTURE_L1_REWARD_DEBUG_N (default 3) decoded responses with their format /
# accuracy scores. Use it for fast diagnosis when reward looks 0 / suspicious.
export FUTURE_L1_REWARD_DEBUG=${FUTURE_L1_REWARD_DEBUG:-0}
export FUTURE_L1_REWARD_DEBUG_N=${FUTURE_L1_REWARD_DEBUG_N:-3}
if [[ "${USE_LLM_JUDGE}" == "1" ]]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${JUDGE_API_URL}}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${JUDGE_API_KEY}}"
  export API_KEY="${API_KEY:-${JUDGE_API_KEY}}"
fi

# --------------------------------------------------------------------------
# Ray / vLLM / WandB env (offline by default per user request).
# --------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export USE_RAY_LOCAL=${USE_RAY_LOCAL:-1}
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-128}
export RAY_NUM_GPUS=${RAY_NUM_GPUS:-8}
export RAY_SPILL_DIR=${RAY_SPILL_DIR:-/tmp/ray_spill}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
export WANDB_PROJECT=${WANDB_PROJECT:-Future-L1-RL}
export WANDB_MODE=${WANDB_MODE:-offline}
# W&B offline data: repo-local wandb/0518/<RUN_NAME> (override with WANDB_DIR / WANDB_RUN_ROOT).
WANDB_RUN_ROOT="${WANDB_RUN_ROOT:-${RL_V2_ROOT}/wandb}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_RUN_ROOT}/${RUN_NAME}}"

mkdir -p "${RAY_SPILL_DIR}" "${RAY_TMPDIR}" "${OUTPUT_DIR}" "${LOG_DIR}" "${WANDB_DIR}"

# --------------------------------------------------------------------------
# Resolve interpreter. For rjob-style launchers the env may not be activated,
# so allow PYTHON= to point at the env's interpreter explicitly.
# --------------------------------------------------------------------------
PYTHON=${PYTHON:-python}

if [[ -z "${MODEL_PATH}" || -z "${TRAIN_FILES}" ]]; then
  echo "MODEL_PATH and TRAIN_FILES must be set." >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Auto-detect backbone + FutureL1 special token ids from the tokenizer.
# Mirrors upstream behaviour: refuse non-qwen3_vl unless ALLOW_NON_QWEN3VL=1.
# --------------------------------------------------------------------------
eval "$(
"${PYTHON}" - "${MODEL_PATH}" <<'PY'
import os
import sys
from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
model_type = getattr(cfg, "model_type", None)
if model_type != "qwen3_vl" and os.environ.get("ALLOW_NON_QWEN3VL", "0") != "1":
    raise SystemExit(
        f"Expected a Qwen3-VL FutureL1 checkpoint with config.model_type='qwen3_vl', got {model_type!r}. "
        "Set ALLOW_NON_QWEN3VL=1 only if you intentionally want another backbone."
    )
print(f"[future_l1_rl] backbone model_type={model_type}", file=sys.stderr)
print(f"export FUTURE_L1_BACKBONE_MODEL_TYPE={model_type}")

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
for name, token in [
    ("FUTURE_L1_LATENT_ID", "<|latent|>"),
    ("FUTURE_L1_LATENT_START_ID", "<|latent_start|>"),
    ("FUTURE_L1_LATENT_END_ID", "<|latent_end|>"),
]:
    tid = tok.convert_tokens_to_ids(token)
    if tid is None or tid == tok.unk_token_id:
        raise SystemExit(f"Missing FutureL1 special token: {token}")
    print(f"export {name}={int(tid)}")
PY
)"

# Back-compat aliases (HyLar runner reads LATENT_*).
export LATENT_START_ID="${FUTURE_L1_LATENT_START_ID}"
export LATENT_END_ID="${FUTURE_L1_LATENT_END_ID}"
export LATENT_ID="${FUTURE_L1_LATENT_ID}"
# Latent budget per span - upstream default is 4.
# Don't force a latent budget by default. The FutureL1 vLLM runner reads
# ``max_latent_token`` / ``loose_latent_budget`` / ``infer_latent_multiplier``
# / ``fixed_latent_budget`` from the checkpoint's ``hf_config`` (same logic
# as ``_future_l1_sample`` in ``Future-L1/src/model/future_l1.py``). Export
# ``FUTURE_L1_LATENT_SIZE=<n>`` to pin a hard cap for ablation.
if [[ -n "${FUTURE_L1_LATENT_SIZE:-}" ]]; then
  export LATENT_SIZE="${FUTURE_L1_LATENT_SIZE}"
fi
export MODEL_PATH

# --------------------------------------------------------------------------
# Common overrides (parameter values aligned with upstream train.sh).
# --------------------------------------------------------------------------
COMMON_OVERRIDES=(
  "config=examples/config_future_l1.yaml"
  "data.train_files=${TRAIN_FILES}"
  "worker.actor.model.model_path=${MODEL_PATH}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-${RAY_NUM_GPUS}}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.save_checkpoint_path=${OUTPUT_DIR}/checkpoints"
  "trainer.save_samples=${SAVE_SAMPLES:-true}"
  "trainer.samples_save_dir=${SAMPLES_SAVE_DIR:-${OUTPUT_DIR}/rollout_samples}"
  "trainer.samples_save_interval=${SAMPLES_SAVE_INTERVAL:-50}"
  "trainer.max_try_make_batch=${MAX_TRY_MAKE_BATCH:-20}"
  "algorithm.answer_tag_filtering=${ANSWER_TAG_FILTERING:-false}"
  "worker.actor.fsdp.torch_dtype=bf16"
  "worker.actor.optim.strategy=adamw_bf16"
  "worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}"
  "worker.rollout.n=${ROLLOUT_N:-8}"
  "worker.rollout.temperature=${TEMPERATURE:-0.9}"
  "worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION:-0.9}"
  "worker.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-true}"
  "worker.rollout.max_num_seqs=${MAX_NUM_SEQS:-128}"
  "worker.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-65536}"
  "worker.actor.future_l1_rl_kappa=${FUTURE_L1_RL_KAPPA:-0.01}"
  "worker.ref.future_l1_rl_kappa=${FUTURE_L1_RL_KAPPA:-0.01}"
  "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE:-64}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH:-8192}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH:-2048}"
)
# Only forward `data.val_files` when the user actually set one - an empty
# value would be parsed as `None` by OmegaConf and rejected against the
# `str` field. Skipping the override falls back to the YAML default ("")
# which the dataloader interprets as "no validation".
if [[ -n "${VAL_FILES}" ]]; then
  COMMON_OVERRIDES+=("data.val_files=${VAL_FILES}")
fi

# --------------------------------------------------------------------------
# Mode-specific overrides.
# --------------------------------------------------------------------------
case "${MODE}" in
  grpo)
    # HyLar-equivalent GRPO baseline: sampling_strategy=future_l1_depo turns on
    # latent-aware vLLM rollout + per-step latent recording + vMF log-prob
    # substitution at latent positions inside dp_actor (mirrors HyLar's single
    # `hylar` strategy). decoupled-PPO / vMF-KL are off, matching HyLar's GRPO
    # config (`enable_decoupled_hybrid_ppo=false`, `enable_latent_vmf_kl=false`).
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=grpo"
      "algorithm.online_filtering=false"
      "algorithm.enable_decoupled_hybrid_ppo=false"
      "algorithm.enable_latent_vmf_kl=false"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
    )
    ;;
  grpo_ctr)
    # GRPO + outcome-contrastive latent reward R_ctr (LA-DAPO).
    # HyLar-equivalent GRPO baseline: sampling_strategy=future_l1_depo turns on
    # latent-aware vLLM rollout + per-step latent recording + vMF log-prob
    # substitution at latent positions inside dp_actor (mirrors HyLar's single
    # `hylar` strategy). decoupled-PPO / vMF-KL are off, matching HyLar's GRPO
    # config (`enable_decoupled_hybrid_ppo=false`, `enable_latent_vmf_kl=false`).
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=grpo"
      "algorithm.online_filtering=false"
      "algorithm.enable_decoupled_hybrid_ppo=false"
      "algorithm.enable_latent_vmf_kl=false"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
      "worker.reward.reward_function_kwargs.latent_ctr_lambda=${FUTURE_L1_LATENT_CTR_LAMBDA:-1.0}"
      "worker.reward.reward_function_kwargs.latent_ctr_temperature=${FUTURE_L1_LATENT_CTR_TEMPERATURE:-0.5}"
    )
    ;;
  dapo)
    # HyLar-equivalent DAPO baseline: same latent-aware path as GRPO above, but
    # with the DAPO advantage estimator, online accuracy filtering, and the
    # asymmetric clip ratios that DAPO introduces.
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=dapo"
      "algorithm.online_filtering=${ONLINE_FILTERING:-true}"
      "algorithm.filter_key=${FILTER_KEY:-accuracy}"
      "algorithm.filter_low=${FILTER_LOW:-0.1}"
      "algorithm.filter_high=${FILTER_HIGH:-0.9}"
      "algorithm.enable_decoupled_hybrid_ppo=false"
      "algorithm.enable_latent_vmf_kl=false"
      "worker.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}"
      "worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
    )
    ;;
  dapo_ctr)
    # DAPO + outcome-contrastive latent reward R_ctr (LA-DAPO).
    # HyLar-equivalent DAPO baseline: same latent-aware path as GRPO above, but
    # with the DAPO advantage estimator, online accuracy filtering, and the
    # asymmetric clip ratios that DAPO introduces.
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=dapo"
      "algorithm.online_filtering=${ONLINE_FILTERING:-true}"
      "algorithm.filter_key=${FILTER_KEY:-accuracy}"
      "algorithm.filter_low=${FILTER_LOW:-0.1}"
      "algorithm.filter_high=${FILTER_HIGH:-0.9}"
      "algorithm.enable_decoupled_hybrid_ppo=false"
      "algorithm.enable_latent_vmf_kl=false"
      "worker.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}"
      "worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
      "worker.reward.reward_function_kwargs.latent_ctr_lambda=${FUTURE_L1_LATENT_CTR_LAMBDA:-1.0}"
      "worker.reward.reward_function_kwargs.latent_ctr_temperature=${FUTURE_L1_LATENT_CTR_TEMPERATURE:-0.5}"
    )
    ;;
  depo)
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=${DEPO_ADV_ESTIMATOR:-dapo}"
      "algorithm.online_filtering=${ONLINE_FILTERING:-true}"
      "algorithm.filter_key=${FILTER_KEY:-accuracy}"
      "algorithm.filter_low=${FILTER_LOW:-0.1}"
      "algorithm.filter_high=${FILTER_HIGH:-0.9}"
      "algorithm.enable_decoupled_hybrid_ppo=true"
      "algorithm.latent_clip_ratio_low=${LATENT_CLIP_LOW:-0.1}"
      "algorithm.latent_clip_ratio_high=${LATENT_CLIP_HIGH:-0.1}"
      "algorithm.latent_clip_ratio_dual=${LATENT_CLIP_DUAL:-3.0}"
      "algorithm.latent_loss_alpha=${LATENT_LOSS_ALPHA:-0.5}"
      "algorithm.enable_latent_vmf_kl=${ENABLE_LATENT_VMF_KL:-true}"
      "algorithm.latent_kl_coef=${LATENT_KL_COEF:-1e-2}"
      "algorithm.enable_format_gated_latent_loss=${FORMAT_GATED_LATENT_LOSS:-false}"
      "algorithm.format_gate_threshold=${FORMAT_GATE_THRESHOLD:-1.0}"
      "worker.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}"
      "worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
    )
    ;;
  depo_ctr)
    # DePO + outcome-contrastive latent reward R_ctr (LA-DAPO, paper tau=0.5).
    ALG_OVERRIDES=(
      "algorithm.adv_estimator=${DEPO_ADV_ESTIMATOR:-dapo}"
      "algorithm.online_filtering=${ONLINE_FILTERING:-true}"
      "algorithm.filter_key=${FILTER_KEY:-accuracy}"
      "algorithm.filter_low=${FILTER_LOW:-0.1}"
      "algorithm.filter_high=${FILTER_HIGH:-0.9}"
      "algorithm.enable_decoupled_hybrid_ppo=true"
      "algorithm.latent_clip_ratio_low=${LATENT_CLIP_LOW:-0.1}"
      "algorithm.latent_clip_ratio_high=${LATENT_CLIP_HIGH:-0.1}"
      "algorithm.latent_clip_ratio_dual=${LATENT_CLIP_DUAL:-3.0}"
      "algorithm.latent_loss_alpha=${LATENT_LOSS_ALPHA:-0.5}"
      "algorithm.enable_latent_vmf_kl=${ENABLE_LATENT_VMF_KL:-true}"
      "algorithm.latent_kl_coef=${LATENT_KL_COEF:-1e-2}"
      "algorithm.enable_format_gated_latent_loss=${FORMAT_GATED_LATENT_LOSS:-false}"
      "algorithm.format_gate_threshold=${FORMAT_GATE_THRESHOLD:-1.0}"
      "worker.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}"
      "worker.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}"
      "worker.rollout.sampling_strategy=future_l1_depo"
      "trainer.experiment_name=${RUN_NAME}"
      "worker.reward.reward_function_kwargs.latent_ctr_lambda=${FUTURE_L1_LATENT_CTR_LAMBDA:-1.0}"
      "worker.reward.reward_function_kwargs.latent_ctr_temperature=${FUTURE_L1_LATENT_CTR_TEMPERATURE:-0.5}"
    )
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use grpo|dapo|depo|depo_ctr|dapo_ctr|grpo_ctr." >&2
    exit 2
    ;;
esac

cd "${RL_V2_ROOT}"
EXTRA_OVERRIDES=()
if [[ -n "${V133K2K_EXTRA_OVERRIDES+x}" ]]; then
  EXTRA_OVERRIDES=("${V133K2K_EXTRA_OVERRIDES[@]}")
fi
"${PYTHON}" -m verl.trainer.main \
    "${COMMON_OVERRIDES[@]}" "${ALG_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}-${TIMESTAMP}.log"
