#!/usr/bin/env bash
# Sync all W&B offline runs under RL_v2/wandb/0518 to the remote project.
#
# Usage:
#   bash scripts/sync_wandb_offline_0518.sh
#   WANDB_ROOT=.../wandb/0518 bash scripts/sync_wandb_offline_0518.sh
#   DRY_RUN=1 bash scripts/sync_wandb_offline_0518.sh          # print only
#   SYNC_JOBS=8 bash scripts/sync_wandb_offline_0518.sh       # parallel (default 8)
#
set -euo pipefail

WANDB_ROOT="${WANDB_ROOT:-${RL_V2_ROOT:-./RL_v2}/wandb/tf}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_JOBS="${SYNC_JOBS:-8}"

if [[ ! -d "${WANDB_ROOT}" ]]; then
  echo "WANDB_ROOT not found: ${WANDB_ROOT}" >&2
  exit 1
fi

if ! [[ "${SYNC_JOBS}" =~ ^[0-9]+$ ]] || [[ "${SYNC_JOBS}" -lt 1 ]]; then
  echo "SYNC_JOBS must be a positive integer, got: ${SYNC_JOBS}" >&2
  exit 1
fi

mapfile -t runs < <(find "${WANDB_ROOT}" -type d -name 'offline-run-*' | LC_ALL=C sort)

if [[ "${#runs[@]}" -eq 0 ]]; then
  echo "No offline-run-* directories under ${WANDB_ROOT}" >&2
  exit 0
fi

echo "Found ${#runs[@]} offline run(s) under ${WANDB_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 (parallel would use SYNC_JOBS=${SYNC_JOBS})"
  for run in "${runs[@]}"; do
    echo "------------------------------------------------------------"
    echo "${run}"
    echo "  (dry-run) would run: wandb sync \"${run}\""
  done
  echo "------------------------------------------------------------"
  echo "Done."
  exit 0
fi

echo "Syncing with SYNC_JOBS=${SYNC_JOBS} (set SYNC_JOBS=1 to force serial)"
# Note: GNU xargs -I ignores -P; use -n1 + bash -c for true parallelism.
printf '%s\0' "${runs[@]}" | xargs -0 -r -n1 -P"${SYNC_JOBS}" bash -c 'echo ">>> wandb sync \"$1\""; wandb sync "$1"' _

echo "------------------------------------------------------------"
echo "Done."
