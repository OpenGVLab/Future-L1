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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


# FutureL1 wraps Qwen-VL backbones (`FutureL1_Qwen2_5_VL`, `FutureL1_Qwen3VL`,
# `RICE_Qwen3VL`, ...) by subclassing the HF generation class and attaching
# extra `projection_head*` modules. Their checkpoints are saved with the
# wrapper class name in ``config.architectures``, which vLLM cannot find.
# The actual weights are 100% compatible with the HF base class - the
# projection head is loaded separately by the FutureL1 vLLM runner
# (see future_l1_rl/vllm_runner/future_l1_gpu_model_runner.py) - so we just
# remap the architecture name at LLM-construction time via ``hf_overrides``.
_FUTURE_L1_ARCH_TO_HF_BASE = {
    "FutureL1_Qwen3VL": "Qwen3VLForConditionalGeneration",
    "SwimBird_Qwen3VL": "Qwen3VLForConditionalGeneration",
    "RICE_Qwen3VL": "Qwen3VLForConditionalGeneration",
    "FutureL1_Qwen3_5_VL": "Qwen3VLForConditionalGeneration",
    "SwimBird_Qwen3_5_VL": "Qwen3VLForConditionalGeneration",
    "RICE_Qwen3_5_VL": "Qwen3VLForConditionalGeneration",
    "FutureL1_Qwen2_5_VL": "Qwen2_5_VLForConditionalGeneration",
    "SwimBird_Qwen2_5_VL": "Qwen2_5_VLForConditionalGeneration",
}


_resolve_swimbird_arch_override = _resolve_future_l1_arch_override


def _resolve_future_l1_arch_override(model_path: str) -> Optional[dict]:
    """Best-effort: read ``config.architectures`` from ``model_path`` and return
    a ``hf_overrides`` dict that points at the HF base class when the
    checkpoint is a FutureL1/RICE wrapper. Returns ``None`` when no override
    is needed (vanilla HF arch) or when the file is unreadable.
    """
    import json as _json

    cfg_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if cfg_path is None or not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = _json.load(f)
    except Exception:  # noqa: BLE001
        return None
    archs = cfg.get("architectures") or []
    if not isinstance(archs, list) or not archs:
        return None
    mapped: list = []
    touched = False
    for a in archs:
        if a in _FUTURE_L1_ARCH_TO_HF_BASE:
            mapped.append(_FUTURE_L1_ARCH_TO_HF_BASE[a])
            touched = True
        else:
            mapped.append(a)
    if not touched:
        return None
    print(
        f"[vLLMRollout] Detected FutureL1 checkpoint at {model_path}; "
        f"remapping architectures {archs} -> {mapped} via hf_overrides.",
        flush=True,
    )
    return {"architectures": mapped}


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
    video_min_pixels: Optional[int],
    video_max_pixels: Optional[int],
    video_fps: float,
    max_video_frames: Optional[int] = None,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, image_min_pixels, image_max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    video_min_pixels,
                    video_max_pixels,
                    video_fps,
                    max_video_frames,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0 and len(videos) != 0:
        return {"image": images, "video": videos}

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        # Remap FutureL1/RICE architectures back to the vanilla HF base so
        # vLLM picks the right model loader. The actor's FSDP weights are
        # also Qwen3VL-only (FutureL1's projection_head weights are dropped
        # by the HF auto-loader when loading into Qwen3VLForConditionalGeneration);
        # the FSDP -> vLLM sharding manager therefore syncs cleanly. Set
        # FUTURE_L1_APPLY_PROJ_HEAD=1 to also load projection_head into the
        # FutureL1 vLLM runner from disk for HF-faithful latent rollout.
        hf_overrides = _resolve_future_l1_arch_override(model_path)
        if hf_overrides is not None:
            engine_kwargs["hf_overrides"] = hf_overrides

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        suppress_token_ids = kwargs.pop("suppress_token_ids", None)
        extra_logit_bias = kwargs.pop("extra_logit_bias", None)
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        if suppress_token_ids or extra_logit_bias:
            old_sampling_params_args.setdefault("logit_bias", getattr(self.sampling_params, "logit_bias", None))
            merged_logit_bias = dict(getattr(self.sampling_params, "logit_bias", None) or {})
            if extra_logit_bias:
                merged_logit_bias.update({int(k): float(v) for k, v in extra_logit_bias.items()})
            for token_id in suppress_token_ids or []:
                merged_logit_bias[int(token_id)] = -100.0
            setattr(self.sampling_params, "logit_bias", merged_logit_bias)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info.get("image_min_pixels", prompts.meta_info.get("min_pixels")),
                            prompts.meta_info.get("image_max_pixels", prompts.meta_info.get("max_pixels")),
                            prompts.meta_info.get("video_min_pixels", prompts.meta_info.get("min_pixels")),
                            prompts.meta_info.get("video_max_pixels", prompts.meta_info.get("max_pixels")),
                            prompts.meta_info["video_fps"],
                            prompts.meta_info.get("max_video_frames", None),
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # ---- FutureL1: optional per-step latent recorder ----
        # Only enabled when sampling_strategy=="future_l1_depo" so we don't pay
        # the IPC cost during pure-text rollouts (GRPO / DAPO).
        future_l1_recorder = None
        record_latents = (
            getattr(self.config, "sampling_strategy", "default") in ("future_l1_depo", "swimbird_depo")
            and not bool(prompts.meta_info.get("disable_latent_recorder", False))
        )
        if record_latents:
            try:
                from future_l1_rl.vllm_runner.latent_recorder import LatentRecorder  # noqa: PLC0415

                # The recorder spins up a local TCP listener and exports the
                # address via env vars; the FutureL1 vLLM runner picks it up
                # and streams binary frames per decode step.
                future_l1_recorder = LatentRecorder(
                    set_env=True, prefer_tcp=True, filter_rank=self.rank
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[FutureL1 rollout] LatentRecorder unavailable, falling back: {_e}")
                future_l1_recorder = None

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            if future_l1_recorder is not None:
                # Use the recorder as a context manager so the TCP listener is
                # always torn down even on exceptions.
                with future_l1_recorder as rec:
                    completions: list[RequestOutput] = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=self.use_tqdm,
                    )
                    # Snapshot parent-major sample-minor latent trajectories.
                    min_req_id = min(int(c.request_id) for c in completions) if completions else 0
                    latents_array = rec.to_object_array_auto(
                        bsz=batch_size,
                        rollout_n=self.sampling_params.n,
                        min_req_id=min_req_id,
                    )
            else:
                completions = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=self.use_tqdm,
                )
                latents_array = None
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}
        if latents_array is not None:
            non_tensor_batch["latents"] = latents_array

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
