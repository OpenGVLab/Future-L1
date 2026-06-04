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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    repetition_penalty: float = 1.0
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # ---- FutureL1-specific knobs ----
    sampling_strategy: str = "default"
    """One of {"default", "future_l1_text", "future_l1_depo"}.

    - "default"        : stock EasyR1 rollout, no FutureL1 patch, no latent
      emission. Ablation-only — NOT HyLar-equivalent.
    - "future_l1_text"  : FutureL1-aware vLLM rollout (latent state machine
      runs, `<|latent_start|>...<|latent_end|>` spans are emitted correctly)
      but per-step latent vectors are NOT recorded and the actor does NOT
      perform vMF log-prob substitution at latent positions. Ablation-only —
      NOT HyLar-equivalent (HyLar always records and substitutes).
    - "future_l1_depo"  : HyLar-equivalent path. Latent-aware vLLM rollout +
      per-step latent recording via `LatentRecorder` + vMF log-prob
      substitution `kappa * cos(actor_hidden, z_rollout)` at every latent
      position inside `dp_actor`. Use this for ALL HyLar baselines
      (GRPO / DAPO / DePO). DePO-specific extras
      (decoupled token/latent PPO, closed-form vMF KL) are gated separately by
      `algorithm.enable_*`."""
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
