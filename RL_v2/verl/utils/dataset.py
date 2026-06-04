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

import math
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import re
from . import torch_functional as VF


# ----------------------------------------------------------------------
# FutureL1-specific helpers (system prompt + TwiFF frame loader)
# ----------------------------------------------------------------------
# Mirrors `VideoL1/src/constants.SYSTEM_MESSAGE`; used when the SFT package is
# not importable (e.g. FUTURE_L1_CODE_ROOT unset).
_RL_SYSTEM_MESSAGE_FALLBACK = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.


Use the following tags to switch your thinking mode:

1.  **Textual Mode**: `<reason>Your textual reasoning process</reason>`
    *   For logical analysis, planning, and verbal thought.

2.  **Visual Mode**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
    *   For mental visualization, imagination and simulation.


**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""

_RL_PLAIN_SYSTEM_MESSAGE = "You are a helpful assistant."

_RL_COT_THINKING_SYSTEM_MESSAGE = (
    "You are a multimodal reasoning assistant. "
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <reason> </reason> tags. "
    "After all thinking is complete, place the final answer inside "
    "`<answer>option's letter from the given choices</answer>`."
)

_RL_STRICT_RFIRST_SYSTEM_MESSAGE = """You are a multimodal reasoning assistant.

You must strictly follow this output format:
<reason>Your first textual reasoning step</reason>
<|latent_start|><|latent|><|latent_end|>
<reason>Your next textual reasoning step after latent visual thinking</reason>
<answer>Your final answer</answer>

Rules:
* Always start with one `<reason>...</reason>` block.
* Use at least one `<|latent_start|>...<|latent_end|>` block.
* Every latent block must be followed by a `<reason>...</reason>` block.
* You may repeat latent + reason blocks if needed.
* Finish with exactly one `<answer>...</answer>` block.
* Do not output any text outside these tags.
"""

_RL_SYSTEM_MESSAGE_CACHE: Optional[str] = None


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y"}


def _future_l1_rl_system_prompt() -> str:
    """Return FutureL1's SFT system message.

    Resolution order:
        1. ``FUTURE_L1_CODE_ROOT`` -> ``src.constants.SYSTEM_MESSAGE``
        2. Inferred VideoL1 root (this file lives at ``VideoL1/RL_v2/verl/utils``)
        3. Built-in fallback string (kept in sync with VideoL1/src/constants.py).
    """
    global _RL_SYSTEM_MESSAGE_CACHE
    if _truthy_env("RL_PLAIN_QWEN3VL"):
        return _RL_COT_THINKING_SYSTEM_MESSAGE
    if _truthy_env("FUTURE_L1_RL_PLAIN_SYSTEM"):
        return _RL_PLAIN_SYSTEM_MESSAGE
    if _truthy_env("FUTURE_L1_RL_STRICT_SYSTEM"):
        return _RL_STRICT_RFIRST_SYSTEM_MESSAGE

    if _RL_SYSTEM_MESSAGE_CACHE is not None:
        return _RL_SYSTEM_MESSAGE_CACHE

    roots: List[str] = []
    env_root = os.environ.get("FUTURE_L1_CODE_ROOT", "").strip()
    if env_root:
        roots.append(env_root)
    try:
        inferred = str(Path(__file__).resolve().parents[3])
        if inferred not in roots:
            roots.append(inferred)
    except (IndexError, ValueError):
        pass
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        if r not in sys.path:
            sys.path.insert(0, r)
        try:
            from src.constants import SYSTEM_MESSAGE  # type: ignore[import-not-found]

            _RL_SYSTEM_MESSAGE_CACHE = SYSTEM_MESSAGE
            return _RL_SYSTEM_MESSAGE_CACHE
        except ImportError:
            continue

    _RL_SYSTEM_MESSAGE_CACHE = _RL_SYSTEM_MESSAGE_FALLBACK
    return _RL_SYSTEM_MESSAGE_CACHE


def _get_frame_indices_uni(num_frames: int, vlen: int, start_frames: int = 1, end_frames: int = 1) -> List[int]:
    if vlen <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        return []
    if vlen <= num_frames:
        indices = list(range(vlen))
        if len(indices) < num_frames:
            indices.extend([indices[-1]] * (num_frames - len(indices)))
        return indices
    start = min(start_frames, vlen)
    end = max(vlen - end_frames, start)
    if num_frames == 1:
        return [(start + end - 1) // 2]
    return np.linspace(start, end - 1, num_frames).round().astype(int).tolist()


def read_twiff_frames_by_clip(video_path: str, clip_indices: List[int]) -> List[ImageObject]:
    """Read TwiFF 1-based frame indices from the fixed 8-frame uniform pool.

    Matches ``VideoL1/RL/future_l1_rl/verl/utils/dataset.py:read_twiff_frames_by_clip``
    so RL_v2 can train on the same TwiFF JSON shards without re-encoding.
    """
    if not clip_indices:
        return []
    try:
        import decord  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TwiFF RL data requires the `decord` package to decode videos. "
            "Install in your env or rjob image, e.g.: pip install decord"
        ) from exc

    video_reader = decord.VideoReader(video_path, num_threads=1)
    pool = _get_frame_indices_uni(8, len(video_reader), start_frames=1, end_frames=1)
    selected = [pool[int(i) - 1] for i in clip_indices]
    raw = video_reader.get_batch(selected).asnumpy()
    return [Image.fromarray(raw[i]) for i in range(raw.shape[0])]


def _coerce_conversations(value: Any) -> List[dict]:
    """Normalize TwiFF's ``conversations`` field across shard variants."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        froms = value.get("from") or []
        values = value.get("value") or []
        return [{"from": f, "value": v} for f, v in zip(froms, values)]
    return []


def _normalize_ground_truth_answer(value: Any) -> str:
    """Normalize GT answer text for reward matching.

    If the label is wrapped as ``<answer>...</answer>``, return the inner text
    (e.g. ``<answer>B</answer>`` -> ``B``). Otherwise return stripped text.
    """
    text = str(value if value is not None else "").strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


_MCQ_ANSWER_INSTRUCTION = "Please provide only the single option letter within the <answer> </answer> tags."


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch features with mixed/non-uniform metadata keys safely.

    Mixed datasets may not share all non-tensor fields (e.g. only some samples
    carry ``problem_id``). We align every key to batch size by filling missing
    non-tensor entries with ``None`` so ``DataProto`` consistency checks pass.
    """
    all_keys = set()
    for feature in features:
        all_keys.update(feature.keys())

    tensors: dict[str, torch.Tensor] = {}
    non_tensors: dict[str, np.ndarray] = {}

    for key in all_keys:
        values = [feature.get(key, None) for feature in features]
        first_non_none = next((v for v in values if v is not None), None)

        if isinstance(first_non_none, torch.Tensor):
            if any(v is None for v in values):
                raise RuntimeError(
                    f"Tensor key {key!r} missing in part of the batch; "
                    "all tensor fields must exist for every sample."
                )
            tensors[key] = torch.stack(values, dim=0)
        else:
            # Keep metadata as a batch-length object vector. Direct
            # np.array(values, dtype=object) may expand same-length lists such
            # as source_options into shape (B, num_options), then later concat
            # fails when another batch has different option counts.
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            non_tensors[key] = arr

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    max_video_frames: Optional[int] = None,
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Any:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    maxf = None
    if max_video_frames is not None and int(max_video_frames) > 0:
        maxf = int(max_video_frames)
        # qwen_vl_utils accepts `fps` + `max_frames` and computes
        # min(max_frames, frames selected by fps, actual frames) internally.
        # Do not pass `nframes` together with `fps`: that is an assertion error.
        vision_info["max_frames"] = maxf
    video_obj = fetch_video(vision_info, return_video_sample_fps=return_fps, return_video_metadata=return_metadata)
    if maxf is None:
        return video_obj

    def _cap_frames(x):
        # Enforce n_frames = min(actual_frames, max_video_frames) post decode.
        # Handles common containers used by qwen_vl_utils across versions.
        try:
            if isinstance(x, list):
                return x[:maxf]
            if isinstance(x, tuple):
                return x[:maxf]
            if isinstance(x, np.ndarray):
                return x[:maxf]
            if torch.is_tensor(x):
                return x[:maxf]
        except Exception:
            return x
        return x

    def _num_frames(x) -> Optional[int]:
        try:
            if isinstance(x, (list, tuple)):
                return len(x)
            if isinstance(x, np.ndarray) or torch.is_tensor(x):
                return int(x.shape[0])
        except Exception:
            return None
        return None

    def _cap_metadata(metadata: dict, n_frames: Optional[int]) -> dict:
        if n_frames is None:
            return metadata
        metadata = dict(metadata)
        indices = metadata.get("frames_indices")
        if isinstance(indices, list):
            metadata["frames_indices"] = indices[:n_frames]
        elif torch.is_tensor(indices):
            metadata["frames_indices"] = indices[:n_frames]
        elif isinstance(indices, np.ndarray):
            metadata["frames_indices"] = indices[:n_frames]
        return metadata

    def _cap_video_obj(obj):
        # return_video_metadata=True returns (video, metadata).
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
            capped_video = _cap_frames(obj[0])
            return capped_video, _cap_metadata(obj[1], _num_frames(capped_video))

        # return_video_sample_fps=True wraps the video object as (video_obj, sample_fps).
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], (float, int)):
            return _cap_video_obj(obj[0]), obj[1]

        return _cap_frames(obj)

    return _cap_video_obj(video_obj)


_LEADING_VIDEO_IMAGE_BLOCK_RE = re.compile(
    r"^\s*<video>\s*(?:\r?\n)\s*<image>\s*(?:\r?\n)?",
    re.IGNORECASE,
)


def _strip_leading_video_image_tags(prompt: str) -> str:
    """Strip leading <video>/<image> lines so HF chat media slots bind without duplicate markers."""
    return _LEADING_VIDEO_IMAGE_BLOCK_RE.sub("", prompt, count=1).lstrip()


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_video_frames: Optional[int] = 16,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_video_frames = max_video_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_min_pixels = image_min_pixels if image_min_pixels is not None else min_pixels
        self.image_max_pixels = image_max_pixels if image_max_pixels is not None else max_pixels
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else min_pixels
        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # Bail out fast if TwiFF JSON was asked for but decord isn't installed.
        self._is_twiff_path = "TwiFF" in data_path or "twiff" in data_path.lower()
        if self._is_twiff_path:
            try:
                import decord  # noqa: F401, PLC0415
            except ImportError as exc:
                raise ImportError(
                    "TwiFF RL data requires the `decord` package to decode videos. "
                    "Install in your env or rjob image, e.g.: pip install decord"
                ) from exc

        if filter_overlong_prompts:
            # TwiFF JSON shards trigger video decode per filter call which is
            # expensive; keep this off by default for TwiFF (matches the
            # upstream future_l1_rl behaviour).
            if self._is_twiff_path and filter_overlong_prompts_workers > 0:
                print(
                    "[RLHFDataset] TwiFF detected; skipping pre-filter to "
                    "avoid full-video decode of every shard.",
                    flush=True,
                )
            else:
                self.dataset = self.dataset.filter(
                    self._filter_overlong_prompts,
                    desc="Filtering overlong prompts",
                    num_proc=filter_overlong_prompts_workers,
                )

    def _joint_video_image_example(self, example: dict[str, Any]) -> bool:
        """True when both modalities are present (e.g. SEED-Bench-R1 video + current frame)."""
        if self.video_key not in example or self.image_key not in example:
            return False
        v = example.get(self.video_key)
        i = example.get(self.image_key)
        return isinstance(v, list) and isinstance(i, list) and len(v) > 0 and len(i) > 0

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        problem_type = str(example.get("problem_type", "")).strip().lower()
        if problem_type == "multiple choice" and _MCQ_ANSWER_INSTRUCTION not in prompt_str:
            prompt_str = f"{prompt_str}\n\n{_MCQ_ANSWER_INSTRUCTION}"

        system_prompt = _future_l1_rl_system_prompt()

        if self._joint_video_image_example(example):
            clean = _strip_leading_video_image_tags(prompt_str)
            videos = example.get(self.video_key) or []
            images = example.get(self.image_key) or []
            content_list: list[dict[str, Any]] = []
            content_list.extend({"type": "video"} for _ in videos)
            content_list.extend({"type": "image"} for _ in images)
            if clean.strip():
                content_list.append({"type": "text", "text": clean})
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            images = example.get(self.image_key) or []
            num_images = len(images) if isinstance(images, list) else 1
            content_list = []
            if "<image>" not in prompt_str and num_images > 0:
                # Some standard VLM datasets store media paths separately and
                # keep the question text marker-free. Add explicit placeholders
                # so HF/vLLM can bind each media item to prompt tokens.
                content_list.extend({"type": "image"} for _ in range(num_images))
                if prompt_str:
                    content_list.append({"type": "text", "text": prompt_str})
            else:
                for i, content in enumerate(prompt_str.split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})

                    if content:
                        content_list.append({"type": "text", "text": content})

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        elif self.video_key in example:
            videos = example.get(self.video_key) or []
            num_videos = len(videos) if isinstance(videos, list) else 1
            content_list = []
            if "<video>" not in prompt_str and num_videos > 0:
                content_list.extend({"type": "video"} for _ in range(num_videos))
                if prompt_str:
                    content_list.append({"type": "text", "text": prompt_str})
            else:
                for i, content in enumerate(prompt_str.split("<video>")):
                    if i != 0:
                        content_list.append({"type": "video"})

                    if content:
                        content_list.append({"type": "text", "text": content})

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_str},
            ]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        # Filter is invoked by HF datasets via num_proc workers; skipping the
        # TwiFF branch above guarantees we never try to spin up decord here.
        if self._is_twiff_example(example):
            return True
        messages = self._build_messages(example)
        if self._joint_video_image_example(example):
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_images = []
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))
            processed_videos = []
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.video_min_pixels,
                    self.video_max_pixels,
                    self.video_fps,
                    self.max_video_frames,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            proc_kw: dict[str, Any] = dict(
                images=processed_images,
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                proc_kw["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            model_inputs = self.processor(**proc_kw)
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(
                    process_video(
                        video,
                        self.video_min_pixels,
                        self.video_max_pixels,
                        self.video_fps,
                        self.max_video_frames,
                    )
                )

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    # ------------------------------------------------------------------
    # FutureL1 / TwiFF-aware row materialization
    # ------------------------------------------------------------------
    def _is_twiff_example(self, example: dict[str, Any]) -> bool:
        return (
            self._is_twiff_path
            or ("conversations" in example and "video" in example and "image" in example)
            or "reasoning_image" in example
        )

    def _materialize_twiff(self, example: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Convert a raw TwiFF row into the standard ``{problem,answer,images}``.

        Returns ``None`` when the row should be skipped (e.g. unreadable video).
        Mirrors ``RL/future_l1_rl/verl/utils/dataset.py:RLHFDataset._build_example``
        for the TwiFF branch.
        """
        video_path = str(example.get("video", "") or "")
        conversations = _coerce_conversations(example.get("conversations"))
        human_turn = None
        gpt_turn = None
        for turn in conversations:
            role = str(turn.get("from", ""))
            if role == "human":
                human_turn = turn
            elif role == "gpt":
                gpt_turn = turn
        if human_turn is None:
            return None

        try:
            image_indices = [int(x) for x in (example.get("image") or [])]
        except (TypeError, ValueError):
            image_indices = []

        try:
            images = read_twiff_frames_by_clip(video_path, image_indices) if image_indices else []
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[RLHFDataset] failed to load TwiFF frames from {video_path}: {exc}", flush=True)
            return None

        problem = str(human_turn.get("value", ""))
        answer = str(example.get("answer", "") or "")
        if not answer and gpt_turn is not None:
            answer = str(gpt_turn.get("value", ""))

        materialized: dict = {}
        if images:
            materialized[self.image_key] = images
            materialized[self.prompt_key] = problem
        else:
            materialized[self.prompt_key] = problem.replace("<image>", "")
        materialized[self.answer_key] = answer
        return materialized

    def _maybe_materialize(self, example: dict[str, Any]) -> Optional[dict[str, Any]]:
        """If the row needs TwiFF-style conversion, do it; else return as-is."""
        if self._is_twiff_example(example):
            return self._materialize_twiff(example)
        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw_example: dict = self.dataset[index]
        materialized = self._maybe_materialize(raw_example)
        if materialized is None:
            raise RuntimeError(
                f"TwiFF row at index {index} could not be materialized "
                f"(missing video or unreadable). Filter the shard upstream "
                "or install decord."
            )
        example: dict = materialized
        messages = self._build_messages(example)
        raw_problem_text = example.get(self.prompt_key)
        example.pop(self.prompt_key, None)

        if self._joint_video_image_example(example):
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_images = []
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))
            processed_videos = []
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.video_min_pixels,
                    self.video_max_pixels,
                    self.video_fps,
                    self.max_video_frames,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            proc_kw: dict[str, Any] = dict(
                images=processed_images,
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                proc_kw["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            model_inputs = self.processor(**proc_kw)
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos, "images": images}
        elif self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.video_min_pixels,
                    self.video_max_pixels,
                    self.video_fps,
                    self.max_video_frames,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = _normalize_ground_truth_answer(example.pop(self.answer_key))
        # Keep the raw question text so the LLM-as-judge reward (and any
        # other reward function that wants task context) can access it via
        # `non_tensor_batch["problem"]`.
        if raw_problem_text is not None:
            example["problem"] = raw_problem_text
        return example
