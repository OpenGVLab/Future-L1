import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _strict_bool(v):
    """Allow `verbose=False` / `True` from CLI string."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return bool(v)
    s = str(v).strip()
    if s.lower() in {"false", "none", "no", "off", "0", ""}:
        return False
    if s.lower() in {"true", "yes", "on", "1"}:
        return True
    return v


def _resolve_model_class(pretrained: str, is_moe: bool):
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
        dtype_key = "torch_dtype"
    else:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        model_cls = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration
        dtype_key = "dtype"

    return model_cls, dtype_key


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        use_flash_attention_2: Optional[bool] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for qwen3_vl. Install with `pip install qwen-vl-utils`.")

        # Backward-compat: accept use_flash_attention_2 and translate to attn_implementation.
        if use_flash_attention_2 is not None:
            if attn_implementation is not None and attn_implementation != ("flash_attention_2" if use_flash_attention_2 else attn_implementation):
                raise ValueError("Specify either `use_flash_attention_2` or `attn_implementation`, not both with conflicting values.")
            if use_flash_attention_2:
                attn_implementation = "flash_attention_2"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        is_moe = bool(re.search(r"A\d+B", pretrained))
        model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)

        model_kwargs = {
            dtype_key: "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.enable_thinking = enable_thinking

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # verbose: print model outputs to stdout while running. Also honors
        # the env var QWEN3_VL_VERBOSE=1 for shell-side toggling.
        self.verbose = bool(_strict_bool(verbose)) or os.environ.get("QWEN3_VL_VERBOSE", "0") in ("1", "true", "True", "yes")

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3 VL models")

    def flatten(self, data):
        """Flatten nested list/tuple visuals into a single list of leaves.

        Handles both ``[[img_a, img_b], [img_c]]`` (legacy grouping) and a flat
        ``[img_a, img_b, img_c]`` from tasks like TwiFF-Bench that return many
        PIL frames in one list (each PIL must not be iterated as a sequence).
        """
        if data is None:
            return None
        out = []
        for item in data:
            if isinstance(item, (list, tuple)):
                out.extend(self.flatten(item))
            else:
                out.append(item)
        return out

    def _build_video_kwargs(self):
        video_kwargs = {"min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels
        return video_kwargs

    @staticmethod
    def _floor_to_frame_factor(nframes: int, factor: int = 2) -> int:
        return max(factor, (int(nframes) // factor) * factor)

    @staticmethod
    def _get_video_total_frames(video_path: str) -> Optional[int]:
        if video_path.startswith("file://"):
            video_path = video_path[len("file://") :]
        try:
            from decord import VideoReader

            return len(VideoReader(video_path))
        except Exception as exc:
            eval_logger.debug(f"Could not inspect video frame count for {video_path}: {exc}")
            return None

    def _cap_video_kwargs_to_available_frames(self, video_path: str, video_kwargs: dict) -> dict:
        total_frames = self._get_video_total_frames(video_path)
        if total_frames is None:
            return video_kwargs

        capped_kwargs = dict(video_kwargs)
        capped_frames = self._floor_to_frame_factor(min(self.max_num_frames, total_frames))

        if "nframes" in capped_kwargs:
            capped_kwargs["nframes"] = capped_frames
        if "max_frames" in capped_kwargs:
            capped_kwargs["max_frames"] = capped_frames

        return capped_kwargs

    def _apply_chat_template(self, batched_messages, **kwargs):
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

    def _build_generate_kwargs(self, gen_kwargs):
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val
        return generate_kwargs

    def _strip_thinking(self, answer):
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def _get_system_prompt_for_instance(self, task_name: str) -> str:
        """Hook for subclasses (e.g. FutureL1) to vary chat ``system`` by task."""
        return self.system_prompt

    def _preprocess_chunk(self, chunk):
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        visual_list = [self.flatten(v) if v else v for v in visual_list]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)
        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        video_kwargs = self._build_video_kwargs()
        batched_messages = []

        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self._get_system_prompt_for_instance(task[i])}]
            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.lower().endswith(
                        (".mp4", ".avi", ".mov", ".mkv", ".webm")
                    ):
                        processed_visuals.append({"type": "video", "video": visual, **self._cap_video_kwargs_to_available_frames(visual, video_kwargs)})
                    elif isinstance(visual, str) and os.path.isfile(visual):
                        processed_visuals.append(
                            {"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
                        )

            if self.interleave_visuals is False:
                message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]})
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})
                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})
                message.append({"role": "user", "content": content_parts})

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        return inputs, contexts, gen_kwargs, until

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until = future.result()
                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # Keep raw decoded outputs for verbose printing (before `until` stripping).
                raw_answers = list(answers)
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for raw, ans, context in zip(raw_answers, answers, contexts):
                    ans = self._strip_thinking(ans)
                    if self.verbose and self.rank == 0:
                        # ANSI 颜色:prompt 红色,raw 黄色,extracted 绿色;只在 rank 0 打印
                        print("\033[31m[qwen3_vl][prompt]\033[0m", context)
                        print("\033[33m[qwen3_vl][raw   ]\033[0m", raw)
                        print("\033[32m[qwen3_vl][answer]\033[0m", ans)
                        print("\033[90m" + "-" * 80 + "\033[0m", flush=True)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
