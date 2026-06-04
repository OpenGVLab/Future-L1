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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: Optional[str] = ""
    """Path to validation shards. Empty string OR null skips validation
    entirely (the dataloader returns None and the trainer drops the val
    reward manager)."""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_video_frames: Optional[int] = 32
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    # Legacy shared pixel limits. Keep these for old configs and command-line
    # overrides; the modality-specific fields below take precedence when set.
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    image_min_pixels: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    # ---- HyLar-aligned subsampling caps (-1 means no cap) ----
    train_max_samples: int = -1
    val_max_samples: int = -1

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        if self.image_min_pixels is None:
            self.image_min_pixels = self.min_pixels
        if self.image_max_pixels is None:
            self.image_max_pixels = self.max_pixels
        if self.video_min_pixels is None:
            self.video_min_pixels = self.min_pixels
        if self.video_max_pixels is None:
            self.video_max_pixels = self.max_pixels


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator: `gae`, `grpo`, `grpo_passk`, `reinforce_plus_plus`,
    `remax`, `rloo`, `dapo` (alias of GRPO outcome adv, kept for clarity)."""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    answer_tag_filtering: bool = False
    """drop a group when any of its responses misses or duplicates `<answer>` tags."""
    loss_avg_mode: str = "token"
    """mirror to ``worker.actor.loss_avg_mode``: 'token' | 'seq'."""
    # ===== DePO (Decoupled Hybrid PPO) + vMF KL =====
    # See HyLar/RL/verl/trainer/config.py for the original definitions.
    enable_decoupled_hybrid_ppo: bool = False
    """D-1: enable separate token/latent surrogate loss with per-group clip."""
    latent_clip_ratio_low: float = 0.1
    """PPO clip ratio lower bound for latent positions."""
    latent_clip_ratio_high: float = 0.1
    """PPO clip ratio upper bound for latent positions."""
    latent_clip_ratio_dual: float = 3.0
    """Dual-clip constant C for latent positions."""
    latent_loss_alpha: float = 0.5
    """Weight alpha applied to the latent PPO loss term."""
    enable_latent_vmf_kl: bool = False
    """D-2: enable closed-form vMF KL constraint at latent positions."""
    latent_kl_coef: float = 1e-2
    """KL coefficient for the latent vMF term."""
    enable_format_gated_latent_loss: bool = False
    """Only apply latent PPO / latent vMF KL losses to samples whose
    format reward passes ``format_gate_threshold``."""
    format_gate_threshold: float = 1.0
    """Minimum per-sample format reward required for latent-only objectives."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    futurebench_eval_freq: int = -1
    """Run FutureBench lmms_eval after checkpoint save every N steps (replaces
    reward-based val when > 0). Requires ``futurebench_eval_script``."""
    futurebench_eval_script: Optional[str] = None
    """Shell script invoked with CHECKPOINT_DIR=... after offloading models."""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    save_samples: bool = True
    """whether to save rollout / training sample JSON dumps (debug; large files)"""
    samples_save_dir: str = "./rollout_samples"
    """directory for saving rollout samples"""
    samples_save_interval: int = 100
    """save sample JSON every N training steps when ``save_samples`` is true; ``<= 0`` disables writes"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        # Propagate DePO algorithm flags to the actor / rollout so that workers
        # see them through the worker config tree.
        self.worker.actor.enable_decoupled_hybrid_ppo = self.algorithm.enable_decoupled_hybrid_ppo
        self.worker.actor.latent_clip_ratio_low = self.algorithm.latent_clip_ratio_low
        self.worker.actor.latent_clip_ratio_high = self.algorithm.latent_clip_ratio_high
        self.worker.actor.latent_clip_ratio_dual = self.algorithm.latent_clip_ratio_dual
        self.worker.actor.latent_loss_alpha = self.algorithm.latent_loss_alpha
        self.worker.actor.enable_latent_vmf_kl = self.algorithm.enable_latent_vmf_kl
        self.worker.actor.latent_kl_coef = self.algorithm.latent_kl_coef
        self.worker.actor.enable_format_gated_latent_loss = self.algorithm.enable_format_gated_latent_loss
        self.worker.actor.format_gate_threshold = self.algorithm.format_gate_threshold
        # HyLar-style alignment: keep actor.loss_avg_mode in sync with algorithm.
        self.worker.actor.loss_avg_mode = self.algorithm.loss_avg_mode
        # Mirror sampling strategy across actor/ref so they know how to handle latents.
        self.worker.actor.sampling_strategy = self.worker.rollout.sampling_strategy
        self.worker.ref.sampling_strategy = self.worker.rollout.sampling_strategy

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
