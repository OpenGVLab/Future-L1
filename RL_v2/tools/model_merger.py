# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# SPDX-License-Identifier: Apache-2.0
#
# FutureL1: merge FSDP actor shards to HuggingFace weights. Extends verl model_merger
# to resolve custom config.architectures (e.g. FutureL1_Qwen3VL).

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)

# Future-L1 repo root so `from src.model.future_l1 import …` works
_FUTURE_L1_ROOT = Path(__file__).resolve().parents[2]
if str(_FUTURE_L1_ROOT) not in sys.path:
    sys.path.insert(0, str(_FUTURE_L1_ROOT))

from src.model.future_l1 import (  # noqa: E402
    RICE_Qwen3VL,
    FutureL1_Qwen2_5_VL,
    FutureL1_Qwen3VL,
)

try:
    from src.model.future_l1 import RICE_Qwen3_5_VL, FutureL1_Qwen3_5_VL  # noqa: E402
except ImportError:
    FutureL1_Qwen3_5_VL = None  # type: ignore[misc, assignment]
    RICE_Qwen3_5_VL = None  # type: ignore[misc, assignment]


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def remove_shard_weight_files(local_dir: str, *, hf_subdir: str = "huggingface") -> list[str]:
    """Remove FSDP shard .pt files in *local_dir* after merge; keep *hf_subdir* untouched."""
    hf_path = os.path.join(local_dir, hf_subdir)
    hf_resolved = os.path.abspath(hf_path)
    removed: list[str] = []
    for name in os.listdir(local_dir):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path):
            continue
        os.remove(path)
        removed.append(path)
    if removed:
        print(f"Removed {len(removed)} shard checkpoint file(s) under {local_dir} (kept {hf_resolved}/).")
        for path in sorted(removed):
            print(f"  deleted: {path}")
    else:
        print(f"No .pt shard files to remove in {local_dir}.")
    return removed


def upload_model_to_huggingface(local_path: str, remote_path: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def _future_l1_architecture_table():
    m = {
        "FutureL1_Qwen2_5_VL": FutureL1_Qwen2_5_VL,
        "FutureL1_Qwen3VL": FutureL1_Qwen3VL,
        "RICE_Qwen3VL": RICE_Qwen3VL,
        # Legacy checkpoint architecture strings.
        "SwimBird_Qwen2_5_VL": FutureL1_Qwen2_5_VL,
        "SwimBird_Qwen3VL": FutureL1_Qwen3VL,
    }
    if FutureL1_Qwen3_5_VL is not None:
        m["FutureL1_Qwen3_5_VL"] = FutureL1_Qwen3_5_VL
        m["SwimBird_Qwen3_5_VL"] = FutureL1_Qwen3_5_VL
    if RICE_Qwen3_5_VL is not None:
        m["RICE_Qwen3_5_VL"] = RICE_Qwen3_5_VL
    return m


_swimbird_architecture_table = _future_l1_architecture_table


def resolve_model_class(config: PretrainedConfig):
    architectures: list[str] = getattr(config, "architectures", []) or []
    arch0 = architectures[0] if architectures else ""
    table = _future_l1_architecture_table()
    if arch0 in table:
        return table[arch0]
    if "ForTokenClassification" in arch0:
        return AutoModelForTokenClassification
    if "ForConditionalGeneration" in arch0:
        return AutoModelForImageTextToText
    if "ForCausalLM" in arch0:
        return AutoModelForCausalLM
    raise NotImplementedError(
        f"Unknown architecture {architectures!r}. "
        f"Add it to tools/model_merger.py or use a standard HF architecture string in config."
    )


def build_model_on_meta(model_cls: type[PreTrainedModel], config: PretrainedConfig) -> PreTrainedModel:
    """Construct an uninitialized model on the meta device for save_pretrained(state_dict=...).

    Qwen3 VL / some transformers builds expose only ``_from_config`` on the concrete subclass,
    so we fall back when ``from_config`` is missing.
    """
    from_config = getattr(model_cls, "from_config", None)
    if callable(from_config):
        with torch.device("meta"):
            return from_config(config, torch_dtype=torch.bfloat16)
    _from_config = getattr(model_cls, "_from_config", None)
    if callable(_from_config):
        with torch.device("meta"):
            return _from_config(config, torch_dtype=torch.bfloat16)
    raise RuntimeError(
        f"{model_cls.__name__} has neither from_config nor _from_config; "
        "extend tools/model_merger.py or adjust your transformers version."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="Actor checkpoint dir (contains model_world_size_* shards)")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="Optional Hugging Face repo id to upload")
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep model/optim/extra_state *.pt shards after merge (default: delete them)",
    )
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    world_size = ""
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_0.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = [state_dict]
    for rank in range(1, total_shards):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        model_state_dict_lst.append(torch.load(model_path, map_location="cpu", weights_only=False))

    merge_dict: dict[str, list] = {}
    param_placements: dict[str, list[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        merge_dict[key] = []
        for rank_idx, model_state_dict in enumerate(model_state_dict_lst):
            try:
                tensor = model_state_dict.pop(key)
            except KeyError:
                print(f"Cannot find key {key} in rank {rank_idx}.")
                raise

            if isinstance(tensor, DTensor):
                merge_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                merge_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(merge_dict):
        if not isinstance(merge_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            placements: tuple[Shard, ...] = param_placements[key]
            if len(mesh_shape) == 1:
                assert len(placements) == 1
                shards = merge_dict[key]
                merge_dict[key] = merge_by_placement(shards, placements[0])
            else:
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            merge_dict[key] = torch.cat(merge_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    ModelCls = resolve_model_class(config)

    model = build_model_on_meta(ModelCls, config)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path} (class {ModelCls.__name__})...")
    model.save_pretrained(hf_path, state_dict=merge_dict)
    del merge_dict, model

    if not args.keep_shards:
        remove_shard_weight_files(local_dir)

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
