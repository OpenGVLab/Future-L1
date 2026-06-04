#!/usr/bin/env bash
# Local 8-GPU DAPO: V1-33K-RL-2K + 0519-TwiFF-interleave SFT checkpoint.
# Run inside an rlaunch / interactive GPU node (same mounts as train.sh).
#
# Usage:
#   cd ${VIDEO_L1_ROOT}/RL_v2
#   bash examples/run_dapo_v133k2k_0519_local.sh
set -euo pipefail

RL_V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../rjob/defaults_v133k2k_0519.sh
source "${RL_V2_ROOT}/rjob/defaults_v133k2k_0519.sh"

apply_v133k2k_run_output_defaults dapo

export MODE=dapo
export N_GPUS_PER_NODE="${GPUS_PER_NODE}"
export RAY_NUM_GPUS="${GPUS_PER_NODE}"
unset FUTURE_L1_LATENT_CTR_LAMBDA
unset FUTURE_L1_LATENT_DIV_LAMBDA

exec bash "${RL_V2_ROOT}/train.sh" dapo
