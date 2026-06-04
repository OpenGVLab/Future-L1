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
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def _maybe_subset(dataset, max_samples: int):
    """Subsample a HF datasets-backed RLHFDataset in-place.

    Mirrors HyLar/RL/examples/config_hylar.yaml's ``train_max_samples`` /
    ``val_max_samples``: ``-1`` (or any non-positive value) means no cap.
    """
    if not max_samples or max_samples < 0:
        return dataset
    inner = getattr(dataset, "dataset", None)
    if inner is None:
        return dataset
    if hasattr(inner, "__len__") and len(inner) > int(max_samples):
        dataset.dataset = inner.select(range(int(max_samples)))
    return dataset


def _parse_data_paths(value: Optional[str]) -> list[str]:
    """Parse `data.*_files` into a list of paths/dataset ids.

    Supported forms:
      - single path/id: "/path/a.json"
      - comma-separated: "/path/a.json,/path/b.json"
      - JSON list string: '["/path/a.json", "/path/b.json"]'
    """
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def _build_dataset_from_paths(
    data_paths: list[str],
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
):
    datasets = []
    for data_path in data_paths:
        ds = RLHFDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_video_frames=getattr(config, "max_video_frames", None),
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            video_min_pixels=config.video_min_pixels,
            video_max_pixels=config.video_max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    concat = ConcatDataset(datasets)
    print(f"[create_dataloader] ConcatDataset enabled with {len(datasets)} sources.")
    return concat


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_paths = _parse_data_paths(config.train_files)
    if not train_paths:
        raise ValueError("data.train_files is empty after parsing.")

    train_dataset = _build_dataset_from_paths(train_paths, config, tokenizer, processor)
    train_max_samples = getattr(config, "train_max_samples", -1)
    if len(train_paths) == 1:
        _maybe_subset(train_dataset, train_max_samples)
    elif train_max_samples and train_max_samples > 0 and len(train_dataset) > int(train_max_samples):
        train_dataset = Subset(train_dataset, range(int(train_max_samples)))

    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # `val_files=""` OR `val_files=null` means "skip validation entirely"
    # - we return `None` for the val dataloader and the trainer handles the rest.
    val_files = config.val_files or ""
    val_paths = _parse_data_paths(val_files)
    if val_paths:
        val_dataset = _build_dataset_from_paths(val_paths, config, tokenizer, processor)
        val_max_samples = getattr(config, "val_max_samples", -1)
        if len(val_paths) == 1:
            _maybe_subset(val_dataset, val_max_samples)
        elif val_max_samples and val_max_samples > 0 and len(val_dataset) > int(val_max_samples):
            val_dataset = Subset(val_dataset, range(int(val_max_samples)))

        if config.val_batch_size == -1:
            val_batch_size = len(val_dataset)
        else:
            val_batch_size = config.val_batch_size

        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        assert len(val_dataloader) >= 1
        print(f"Size of val dataloader: {len(val_dataloader)}")
    else:
        val_dataloader = None
        print("[create_dataloader] data.val_files is empty -> validation is disabled.")

    assert len(train_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader, val_dataloader
