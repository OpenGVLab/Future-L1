# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward FutureL1 driver env vars to Ray workers on every node."""

from __future__ import annotations

import os
from typing import Dict

_LEGACY_ENV_ALIASES = {
    "SWIMBIRD_CODE_ROOT": "FUTURE_L1_CODE_ROOT",
    "SWIMBIRD_RL_V2_ROOT": "FUTURE_L1_RL_V2_ROOT",
    "SWIMBIRD_RL_PATCH": "FUTURE_L1_RL_PATCH",
    "SWIMBIRD_RL_PATCH_REQUIRED": "FUTURE_L1_RL_PATCH_REQUIRED",
    "SWIMBIRD_REQUIRE_VLLM_RUNNER": "FUTURE_L1_REQUIRE_VLLM_RUNNER",
    "SWIMBIRD_BACKBONE_MODEL_TYPE": "FUTURE_L1_BACKBONE_MODEL_TYPE",
    "SWIMBIRD_LATENT_ID": "FUTURE_L1_LATENT_ID",
    "SWIMBIRD_LATENT_START_ID": "FUTURE_L1_LATENT_START_ID",
    "SWIMBIRD_LATENT_END_ID": "FUTURE_L1_LATENT_END_ID",
    "SWIMBIRD_LATENT_SIZE": "FUTURE_L1_LATENT_SIZE",
    "SWIMBIRD_FIXED_LATENT_BUDGET": "FUTURE_L1_FIXED_LATENT_BUDGET",
    "SWIMBIRD_LATENT_DEBUG": "FUTURE_L1_LATENT_DEBUG",
    "SWIMBIRD_INITIAL_MODEL_PATH": "FUTURE_L1_INITIAL_MODEL_PATH",
    "SWIMBIRD_MODEL_PATH": "FUTURE_L1_MODEL_PATH",
    "SWIMBIRD_FORMAT_MODE": "FUTURE_L1_FORMAT_MODE",
    "SWIMBIRD_FORMAT_WEIGHT": "FUTURE_L1_FORMAT_WEIGHT",
    "SWIMBIRD_LENGTH_PENALTY_WEIGHT": "FUTURE_L1_LENGTH_PENALTY_WEIGHT",
    "SWIMBIRD_DIVERSITY_PENALTY_BETA": "FUTURE_L1_LATENT_DIV_LAMBDA",
    "SWIMBIRD_COLVR_LATENT_COEF": "FUTURE_L1_LATENT_CTR_LAMBDA",
    "SWIMBIRD_COLVR_TEMPERATURE": "FUTURE_L1_LATENT_CTR_TEMPERATURE",
    "FUTURE_L1_DIVERSITY_PENALTY_BETA": "FUTURE_L1_LATENT_DIV_LAMBDA",
    "FUTURE_L1_COLVR_LATENT_COEF": "FUTURE_L1_LATENT_CTR_LAMBDA",
    "FUTURE_L1_COLVR_TEMPERATURE": "FUTURE_L1_LATENT_CTR_TEMPERATURE",
    "SWIMBIRD_REWARD_DEBUG": "FUTURE_L1_REWARD_DEBUG",
    "SWIMBIRD_REWARD_DEBUG_N": "FUTURE_L1_REWARD_DEBUG_N",
    "SWIMBIRD_RL_KAPPA": "FUTURE_L1_RL_KAPPA",
    "SWIMBIRD_RL_PLAIN_SYSTEM": "FUTURE_L1_RL_PLAIN_SYSTEM",
    "SWIMBIRD_RL_STRICT_SYSTEM": "FUTURE_L1_RL_STRICT_SYSTEM",
}

_FUTURE_L1_RAY_ENV_KEYS = (
    "FUTURE_L1_CODE_ROOT",
    "FUTURE_L1_RL_V2_ROOT",
    "FUTURE_L1_RL_PATCH",
    "FUTURE_L1_RL_PATCH_REQUIRED",
    "FUTURE_L1_REQUIRE_VLLM_RUNNER",
    "FUTURE_L1_BACKBONE_MODEL_TYPE",
    "FUTURE_L1_LATENT_ID",
    "FUTURE_L1_LATENT_START_ID",
    "FUTURE_L1_LATENT_END_ID",
    "LATENT_ID",
    "LATENT_START_ID",
    "LATENT_END_ID",
    "FUTURE_L1_LATENT_SIZE",
    "LATENT_SIZE",
    "FUTURE_L1_FIXED_LATENT_BUDGET",
    "FUTURE_L1_LATENT_DEBUG",
    "LATENT_DEBUG",
    "FUTURE_L1_INITIAL_MODEL_PATH",
    "MODEL_PATH",
    "FUTURE_L1_FORMAT_MODE",
    "FUTURE_L1_FORMAT_WEIGHT",
    "FUTURE_L1_LENGTH_PENALTY_WEIGHT",
    "FUTURE_L1_LATENT_DIV_LAMBDA",
    "FUTURE_L1_LATENT_CTR_LAMBDA",
    "FUTURE_L1_LATENT_CTR_TEMPERATURE",
    "USE_LLM_JUDGE",
    "LLM_JUDGE_ONLY",
    "JUDGE_API_URL",
    "JUDGE_API_NAME",
    "JUDGE_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "API_KEY",
    "FUTURE_L1_REWARD_DEBUG",
    "FUTURE_L1_REWARD_DEBUG_N",
    "RL_PLAIN_QWEN3VL",
)


def normalize_legacy_env() -> None:
    """Map legacy SWIMBIRD_* exports to FUTURE_L1_* when the new key is unset."""
    for legacy, new in _LEGACY_ENV_ALIASES.items():
        if not os.environ.get(new) and os.environ.get(legacy):
            os.environ[new] = os.environ[legacy]


def collect_future_l1_ray_env_vars() -> Dict[str, str]:
    """Return non-empty FutureL1-related env vars from the current process."""
    normalize_legacy_env()
    out: Dict[str, str] = {}
    for key in _FUTURE_L1_RAY_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None and str(value) != "":
            out[key] = str(value)
    return out


collect_swimbird_ray_env_vars = collect_future_l1_ray_env_vars
