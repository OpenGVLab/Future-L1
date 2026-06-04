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

import json
import os
import sys

# ----------------------------------------------------------------------
# Apply FutureL1 runtime patches (transformers forwards + vLLM GPU model
# runner) before importing anything that pulls in vLLM. The patch module is
# a no-op when FutureL1 env vars are not set, so this is safe for stock
# EasyR1 runs as well, with the caveat that vLLM v1 will be forced.
# ----------------------------------------------------------------------
if os.environ.get("FUTURE_L1_RL_PATCH", "1") != "0":
    _rl_v2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if _rl_v2_root not in sys.path:
        sys.path.insert(0, _rl_v2_root)
    try:
        import future_l1_rl_patch  # noqa: F401  (side-effect: monkey patches)
    except Exception as _e:  # noqa: BLE001
        _msg = f"[verl.trainer.main] FutureL1 patch skipped: {_e}"
        # `FUTURE_L1_RL_PATCH_REQUIRED=1` (default for HyLar-equivalent baselines,
        # set by the launcher scripts in rjob/) escalates ANY patch import
        # failure to a hard error so we never silently fall back to stock vLLM,
        # which would invalidate every GRPO/DAPO/DePO baseline by skipping
        # latent recording and vMF log-prob substitution.
        if os.environ.get("FUTURE_L1_RL_PATCH_REQUIRED", "0") == "1":
            raise RuntimeError(_msg) from _e
        print(_msg, file=sys.stderr)

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.future_l1_ray_env import collect_future_l1_ray_env_vars
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        # Skip the val reward manager entirely when there is no val dataloader
        # (matches the "skip all val" mode driven by data.val_files="" or
        # trainer.val_freq <= 0 + val_before_train: false in the YAML).
        if val_dataloader is None:
            val_reward_fn = None
        else:
            val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
            # Forward W&B settings into Ray workers (Runner + FSDP workers).
            "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
        }

        if os.environ.get("WANDB_DIR"):
            env_vars["WANDB_DIR"] = os.environ["WANDB_DIR"]
        if os.environ.get("WANDB_PROJECT"):
            env_vars["WANDB_PROJECT"] = os.environ["WANDB_PROJECT"]

        # Multi-node: follower-node Ray workers do not inherit train.sh exports.
        # Without latent token ids the vLLM runner disables the latent FSM.
        future_l1_env = collect_future_l1_ray_env_vars()
        env_vars.update(future_l1_env)
        if future_l1_env.get("FUTURE_L1_LATENT_START_ID"):
            print(
                "[verl.trainer.main] Forwarding FutureL1 latent env to Ray workers: "
                f"latent_start_id={future_l1_env.get('FUTURE_L1_LATENT_START_ID')} "
                f"latent_end_id={future_l1_env.get('FUTURE_L1_LATENT_END_ID')} "
                f"latent_id={future_l1_env.get('FUTURE_L1_LATENT_ID')}",
                flush=True,
            )
        elif os.environ.get("FUTURE_L1_RL_PATCH_REQUIRED", "0") == "1":
            print(
                "[verl.trainer.main] WARNING: FUTURE_L1_LATENT_* not set on driver; "
                "multi-node rollouts may sample garbage after <|latent_start|>.",
                flush=True,
            )

        use_ray_local = os.environ.get("USE_RAY_LOCAL", "1").lower() in ("1", "true", "yes")
        ray_address = None if use_ray_local else os.environ.get("RAY_ADDRESS")
        ray_init_kwargs: dict = {"runtime_env": {"env_vars": env_vars}}
        if ray_address:
            ray_init_kwargs["address"] = ray_address
        elif use_ray_local:
            ray_init_kwargs["address"] = "local"

        ray.init(**ray_init_kwargs)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
