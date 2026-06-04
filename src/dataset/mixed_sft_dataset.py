"""
Mixed SFT dataset that concatenates TwiFF-format JSON files (frame-index based,
plus chat_video_distill format) with FutureL1-format JSON files (image-path
based, conversations with from/value).

Each sample retains its original schema; the dataset wrapper records its source
in a private ``__source__`` key, and the collator dispatches to the matching
sub-collator (TwiFFDataCollator / FutureL1DataCollator). Mixed-source batches
are processed sub-batch by sub-batch and merged with right-padding.

Auto-classification of each input JSON file is performed by peeking at the
first non-empty record:
  - has ``messages`` and ``videos``                                    -> twiff
  - has ``reasoning_image`` whose entries are ints (or empty + has
    ``video`` key)                                                     -> twiff
  - otherwise (``conversations`` + path-list ``image``/``reasoning_image``)
                                                                       -> future_l1
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from .future_l1_dataset import FutureL1DataCollator, FutureL1SFTDataset, make_supervised_data_module
from .twiff_sft_dataset import TwiFFDataCollator, TwiFFSFTDataset, make_supervised_data_module_twiff

# --------------------------------------------------------------------------- #
# JSON file classifier                                                         #
# --------------------------------------------------------------------------- #

_TWIFF_KIND = "twiff"
_FUTURE_L1_KIND = "future_l1"
_VALID_KINDS = (_TWIFF_KIND, _FUTURE_L1_KIND)


def _peek_first_record(json_path: str):
    """Load first record from a JSON file (which may be a list or single dict)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0]
    return None


def _classify_record(record) -> str:
    if not isinstance(record, dict):
        return _FUTURE_L1_KIND

    # chat_video_distill
    if "messages" in record and "videos" in record:
        return _TWIFF_KIND

    # twiff frame-index format: reasoning_image / image are int lists
    ri = record.get("reasoning_image", None)
    img = record.get("image", None)
    if ri is not None or img is not None:
        for arr in (ri, img):
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, bool):
                    pass
                elif isinstance(first, int):
                    return _TWIFF_KIND
                elif isinstance(first, str):
                    return _FUTURE_L1_KIND

    # Fallback heuristics
    if "video" in record and (
        "reasoning_image" in record or "image" in record
    ):
        return _TWIFF_KIND
    if "conversations" in record:
        return _FUTURE_L1_KIND
    return _FUTURE_L1_KIND


def _collect_json_files(root: Union[str, Path]) -> List[str]:
    p = Path(root)
    if not p.exists():
        logging.warning(f"[mixed-sft] path does not exist, skipping: {p}")
        return []
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.json") if x.is_file())
        if not files:
            logging.warning(f"[mixed-sft] no .json files in directory: {p}")
        return files
    if p.is_file() and p.suffix == ".json":
        return [str(p)]
    logging.warning(f"[mixed-sft] not a valid .json file or directory: {p}")
    return []


def _normalize_data_paths(data_path: Union[str, List[str]]) -> List[str]:
    if isinstance(data_path, str):
        if "," in data_path and not os.path.exists(data_path):
            return [s.strip() for s in data_path.split(",") if s.strip()]
        return [data_path]
    if isinstance(data_path, list):
        return [str(p).strip() for p in data_path if str(p).strip()]
    raise TypeError(f"Unsupported data_path type: {type(data_path)}")


def classify_data_paths(
    data_path: Union[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """Return (twiff_json_files, future_l1_json_files) by peeking the first
    record in each JSON file."""
    twiff_files: List[str] = []
    future_l1_files: List[str] = []

    seen = set()
    for src in _normalize_data_paths(data_path):
        for jf in _collect_json_files(src):
            if jf in seen:
                continue
            seen.add(jf)
            try:
                first = _peek_first_record(jf)
            except Exception as e:
                logging.warning(f"[mixed-sft] failed to peek {jf}: {e}; treating as future_l1")
                future_l1_files.append(jf)
                continue
            kind = _classify_record(first)
            if kind == _TWIFF_KIND:
                twiff_files.append(jf)
            else:
                future_l1_files.append(jf)

    twiff_files = sorted(set(twiff_files))
    future_l1_files = sorted(set(future_l1_files))
    logging.info(
        f"[mixed-sft] classified {len(twiff_files)} TwiFF JSON file(s) "
        f"and {len(future_l1_files)} FutureL1 JSON file(s)."
    )
    if twiff_files:
        logging.info("[mixed-sft] twiff files: " + ", ".join(twiff_files))
    if future_l1_files:
        logging.info("[mixed-sft] future_l1 files: " + ", ".join(future_l1_files))
    return twiff_files, future_l1_files


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

_SOURCE_KEY = "__source__"


class _AnnotatedSubset(Dataset):
    """Wraps an underlying dataset and stamps each item with a source kind."""

    def __init__(self, base: Dataset, kind: str):
        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind}")
        self.base = base
        self.kind = kind

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        item = self.base[i]
        if isinstance(item, dict):
            # Avoid mutating the underlying HFDataset row object: shallow-copy.
            item = dict(item)
            item[_SOURCE_KEY] = self.kind
        else:
            item = {_SOURCE_KEY: self.kind, "raw": item}
        return item


class MixedSFTDataset(Dataset):
    """Concatenates TwiFF and FutureL1 datasets while preserving per-sample kind."""

    def __init__(
        self,
        twiff_paths: Sequence[str],
        future_l1_paths: Sequence[str],
    ):
        super().__init__()
        self.parts: List[_AnnotatedSubset] = []
        self.lengths: List[int] = []

        if twiff_paths:
            twiff_ds = TwiFFSFTDataset(data_root=list(twiff_paths))
            self.parts.append(_AnnotatedSubset(twiff_ds, _TWIFF_KIND))
            self.lengths.append(len(twiff_ds))
        if future_l1_paths:
            sb_ds = FutureL1SFTDataset(data_root=list(future_l1_paths))
            self.parts.append(_AnnotatedSubset(sb_ds, _FUTURE_L1_KIND))
            self.lengths.append(len(sb_ds))

        if not self.parts:
            raise ValueError("MixedSFTDataset received no data files.")

        self._cum: List[int] = []
        s = 0
        for n in self.lengths:
            s += n
            self._cum.append(s)

        logging.info(
            f"[mixed-sft] composed dataset: "
            + ", ".join(f"{p.kind}={n}" for p, n in zip(self.parts, self.lengths))
            + f" (total={self._cum[-1]})"
        )

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, i: int):
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        for cum, part in zip(self._cum, self.parts):
            if i < cum:
                local = i - (cum - len(part))
                return part[local]
        raise IndexError(i)


# --------------------------------------------------------------------------- #
# Batch-merge helpers                                                          #
# --------------------------------------------------------------------------- #

# Tensor keys whose first dim is the batch dim and second dim is the sequence
# length — they need right-padding before concatenation.
_SEQ_PAD_KEYS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids", "image_out_mask")
# Tensor keys whose entries are concatenated along dim 0 directly (no batch dim
# on the sample side; they are flat feature stacks across the whole batch).
_CONCAT_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_latent",
    "image_grid_thw_latent",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
)


def _pad_value_for_key(key: str, pad_token_idx) -> int:
    if key == "labels":
        return -100
    if key == "attention_mask":
        return 0
    if key == "image_out_mask":
        return 0
    if key == "mm_token_type_ids":
        return 0
    if key == "input_ids":
        if isinstance(pad_token_idx, torch.Tensor):
            return int(pad_token_idx.flatten()[0].item())
        return int(pad_token_idx) if pad_token_idx is not None else 0
    return 0


def _right_pad_2d(tensor: torch.Tensor, target_len: int, pad_value) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    cur = tensor.size(-1)
    if cur >= target_len:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-1] = target_len - cur
    pad = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)


def _merge_batches(batches: List[dict], pad_token_idx) -> dict:
    """Merge batches produced by sub-collators. Right-pads sequence dims and
    concats along batch / feature dims."""
    batches = [b for b in batches if b]
    if not batches:
        return {}
    if len(batches) == 1:
        return batches[0]

    # Determine target sequence length across all sub-batches.
    max_seq = 0
    for b in batches:
        if "input_ids" in b and isinstance(b["input_ids"], torch.Tensor):
            t = b["input_ids"]
            if t.dim() >= 2:
                max_seq = max(max_seq, t.size(-1))
            else:
                max_seq = max(max_seq, t.size(0))

    merged: dict = {}
    all_keys = set()
    for b in batches:
        all_keys.update(b.keys())

    for key in all_keys:
        present = [b for b in batches if key in b and b[key] is not None]
        if not present:
            merged[key] = None
            continue
        values = [b[key] for b in present]
        if all(isinstance(v, torch.Tensor) for v in values):
            if key in _SEQ_PAD_KEYS and max_seq > 0:
                pad_v = _pad_value_for_key(key, pad_token_idx)
                padded = [_right_pad_2d(v, max_seq, pad_v) for v in values]
                merged[key] = torch.cat(padded, dim=0)
            elif key in _CONCAT_KEYS:
                merged[key] = torch.cat(values, dim=0)
            else:
                # Best-effort: try concat on dim 0; otherwise keep the first.
                try:
                    merged[key] = torch.cat(values, dim=0)
                except Exception:
                    merged[key] = values[0]
        else:
            # Non-tensor (lists, strings, etc.): flatten lists, otherwise keep first.
            if all(isinstance(v, list) for v in values):
                flat: list = []
                for v in values:
                    flat.extend(v)
                merged[key] = flat
            else:
                merged[key] = values[0]
    return merged


# --------------------------------------------------------------------------- #
# Collator                                                                     #
# --------------------------------------------------------------------------- #


class MixedSFTDataCollator:
    """Per-batch dispatcher that hands each sub-group of samples to its
    matching sub-collator (TwiFF / FutureL1) and merges the outputs."""

    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.twiff_collator = TwiFFDataCollator(processor=processor, args=args)
        self.future_l1_collator = FutureL1DataCollator(processor=processor, args=args)
        self._pad_token_idx = self.twiff_collator.pad_token_idx

    @staticmethod
    def _strip_source(samples: Iterable[dict]) -> List[dict]:
        out = []
        for ex in samples:
            if isinstance(ex, dict) and _SOURCE_KEY in ex:
                ex = {k: v for k, v in ex.items() if k != _SOURCE_KEY}
            out.append(ex)
        return out

    def __call__(self, raw_examples):
        twiff_samples: List[dict] = []
        future_l1_samples: List[dict] = []
        for ex in raw_examples:
            kind = ex.get(_SOURCE_KEY) if isinstance(ex, dict) else None
            if kind == _FUTURE_L1_KIND:
                future_l1_samples.append(ex)
            else:
                # Default to twiff for legacy / unannotated samples (preserves
                # behavior when this collator is used standalone with a pure
                # TwiFFSFTDataset).
                twiff_samples.append(ex)

        twiff_clean = self._strip_source(twiff_samples)
        future_l1_clean = self._strip_source(future_l1_samples)

        sub_batches: List[dict] = []
        if twiff_clean:
            sub_batches.append(self.twiff_collator(twiff_clean))
        if future_l1_clean:
            sub_batches.append(self.future_l1_collator(future_l1_clean))

        return _merge_batches(sub_batches, self._pad_token_idx)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def make_supervised_data_module_mixed(processor, args):
    """Build a mixed TwiFF + FutureL1 data module from ``args.data_path``.

    Each JSON file is auto-classified by peeking at its first record. Files of
    each kind are routed to the matching dataset class, then concatenated.
    """
    twiff_files, future_l1_files = classify_data_paths(args.data_path)

    if not twiff_files and not future_l1_files:
        raise ValueError(
            "No JSON files were found under args.data_path while building the mixed dataset."
        )

    # Single-kind shortcut: defer to the existing data modules so we keep all
    # of their batch-sampler / sub-selection behavior intact.
    if twiff_files and not future_l1_files:
        from copy import copy as _copy
        sub_args = _copy(args)
        sub_args.data_path = twiff_files
        return make_supervised_data_module_twiff(processor=processor, args=sub_args)
    if future_l1_files and not twiff_files:
        from copy import copy as _copy
        sub_args = _copy(args)
        sub_args.data_path = future_l1_files
        return make_supervised_data_module(processor=processor, args=sub_args)

    dataset = MixedSFTDataset(twiff_paths=twiff_files, future_l1_paths=future_l1_files)
    data_collator = MixedSFTDataCollator(processor=processor, args=args)
    return dict(train_dataset=dataset, eval_dataset=None, data_collator=data_collator)
