import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
import warnings
import os
import json
import importlib
import inspect
from types import ModuleType
from typing import Callable, List

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    if "Qwen2_5" in config["architectures"][0]:
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def load_reward_funcs(
    module_path: str = "train.reward_funcs",
    *,
    name_pred = lambda n: n.endswith("_reward"),
    obj_pred  = lambda o: callable(o),
    keep_order: bool = True
) -> List[Callable]:

    mod: ModuleType = importlib.import_module(module_path)
    
    members = inspect.getmembers(mod, predicate=obj_pred)

    reward_funcs = [(n, o) for n, o in members if name_pred(n)]

    if keep_order:
        reward_funcs.sort(key=lambda pair: inspect.getsourcelines(pair[1])[1])

    return [o for _, o in reward_funcs]