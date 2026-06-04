"""FutureL1 model integration for lmms-eval.

FutureL1 wraps Qwen3-VL / Qwen3.5-VL backbones with a custom mixed-modality
forward and latent-token-aware generation path. This file reuses the
preprocessing in ``Qwen3_VL`` and overrides only the model loading + decoding
so that the FutureL1-specific monkey patches are applied and special tokens
(e.g. ``<|latent_start|>``, ``<|latent_end|>``, ``<answer>...</answer>``) are
handled identically to the VLMEvalKit reference implementation in
``vlmeval/vlm/future_l1/model.py``.

Required environment:
    Set one of these to the FutureL1 repo root that contains ``src/model/future_l1``
    and ``src/train/monkey_patch_forward.py``:

        export ROT_CODE_ROOT=/path/to/FutureL1          # preferred
        export FUTURE_L1_CODE_ROOT=/path/to/FutureL1     # alias

    Or pass it via ``--model_args code_root=/path/to/FutureL1``.

Example:
    python -m lmms_eval \
        --model future_l1 \
        --model_args pretrained=/path/to/future_l1/ckpt,code_root=/path/to/FutureL1,attn_implementation=flash_attention_2 \
        --tasks mmstar --batch_size 1
"""

from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_vl import Qwen3_VL


_ANSWER_REGEX = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
_LATENT_BLOCK_RE = re.compile(r"(<\|latent_start\|>)(.*?)(<\|latent_end\|>)", re.DOTALL)


def _normalize_latent_tag_spacing(text: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", str(text))


def _latent_block_token_counts(text: str) -> List[int]:
    text = _normalize_latent_tag_spacing(text)
    return [match.group(2).count("<|latent|>") for match in _LATENT_BLOCK_RE.finditer(text)]


def _latent_trajectory_from_output(states: Any, mask: Any, batch_idx: int) -> Optional[np.ndarray]:
    if states is None or mask is None:
        return None
    try:
        sample_states = states[batch_idx]
        sample_mask = mask[batch_idx].to(dtype=torch.bool)
        selected = sample_states[sample_mask]
        if selected.numel() == 0:
            return None
        return selected.detach().float().cpu().numpy()
    except Exception:  # noqa: BLE001
        return None


def _adjacent_pair_cos_mse_stats(sequence: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    """Mean cos / cos^2 / per-dim MSE over consecutive rows in ``sequence`` (N, D)."""
    if sequence.shape[0] < 2:
        return None, None, None, 0
    cos_values: List[float] = []
    mse_values: List[float] = []
    for idx in range(sequence.shape[0] - 1):
        left, right = sequence[idx], sequence[idx + 1]
        if float(np.linalg.norm(left)) <= 1e-12 or float(np.linalg.norm(right)) <= 1e-12:
            continue
        mse_values.append(float(np.mean((left - right) ** 2)))
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        cos_values.append(float(np.dot(left, right) / denom))
    if not cos_values:
        return None, None, None, 0
    cos_arr = np.asarray(cos_values, dtype=np.float32)
    mse_arr = np.asarray(mse_values, dtype=np.float32)
    return (
        float(cos_arr.mean()),
        float((cos_arr * cos_arr).mean()),
        float(mse_arr.mean()),
        int(cos_arr.size),
    )


def _latent_token_metadata(raw_text: str, vector_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = _latent_block_token_counts(raw_text)
    block_ids: List[int] = []
    token_pos: List[int] = []
    global_pos: List[int] = []
    cursor = 0
    for block_idx, count in enumerate(counts):
        for pos in range(int(max(0, count))):
            block_ids.append(block_idx)
            token_pos.append(pos)
            global_pos.append(cursor)
            cursor += 1
    if len(block_ids) < vector_count:
        for pos in range(len(block_ids), vector_count):
            block_ids.append(-1)
            token_pos.append(-1)
            global_pos.append(pos)
    return (
        np.asarray(block_ids[:vector_count], dtype=np.int32),
        np.asarray(token_pos[:vector_count], dtype=np.int32),
        np.asarray(global_pos[:vector_count], dtype=np.int32),
    )


def _token_ids_for_strings(tokenizer: Any, strings: List[str]) -> set[int]:
    ids: set[int] = set()
    for text in strings:
        try:
            tid = tokenizer.convert_tokens_to_ids(text)
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.add(int(tid))
        except Exception:  # noqa: BLE001
            pass
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
        except Exception:  # noqa: BLE001
            pass
    return ids


def _vision_token_ids(config: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for attr in ("image_token_id", "video_token_id", "vision_token_id"):
        value = getattr(config, attr, None)
        if value is not None:
            ids.add(int(value))
    ids.update(_token_ids_for_strings(tokenizer, ["<|image_pad|>", "<|video_pad|>", "<image>", "<video>"]))
    return ids


def _special_token_ids(config: Any, tokenizer: Any) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in (
        "latent_id",
        "latent_start_id",
        "latent_end_id",
        "text_id",
        "text_start_id",
        "text_end_id",
        "image_token_id",
        "video_token_id",
        "vision_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        value = getattr(config, attr, None)
        if value is not None:
            ids.add(int(value))
    ids.update(
        _token_ids_for_strings(
            tokenizer,
            [
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|latent_start|>",
                "<|latent|>",
                "<|latent_end|>",
            ],
        )
    )
    return ids


def _compute_latent_similarity_stats(raw_text: str, latents: Optional[np.ndarray]) -> Dict[str, Any]:
    counts = _latent_block_token_counts(raw_text)
    stats: Dict[str, Any] = {
        # LASER original: adjacent <|latent|> steps (T-1 pairs, mean cos^2).
        "latent_adjacent_cos2_mean": None,
        "latent_adjacent_cos_mean": None,
        "latent_adjacent_mse_mean": None,
        "latent_adjacent_pair_count": 0,
        # FutureL1: adjacent keyframe blocks (mean-pool inside block first).
        "latent_block_cos2_mean": None,
        "latent_block_cos_mean": None,
        "latent_block_mse_mean": None,
        "latent_block_pair_count": 0,
        "latent_block_count": len(counts),
        "latent_vector_count": 0 if latents is None else int(latents.shape[0]),
        "latent_block_token_counts": counts,
    }
    if latents is not None:
        adj_cos, adj_cos2, adj_mse, adj_pairs = _adjacent_pair_cos_mse_stats(latents)
        stats["latent_adjacent_cos_mean"] = adj_cos
        stats["latent_adjacent_cos2_mean"] = adj_cos2
        stats["latent_adjacent_mse_mean"] = adj_mse
        stats["latent_adjacent_pair_count"] = adj_pairs

    if latents is None or not counts:
        return stats

    offset = 0
    block_reps: List[np.ndarray] = []
    for count in counts:
        count = int(max(0, count))
        next_offset = offset + count
        if count > 0 and offset < latents.shape[0]:
            block = latents[offset : min(next_offset, latents.shape[0])]
            if block.shape[0] > 0:
                block_reps.append(block.mean(axis=0))
        offset = next_offset

    stats["latent_block_count"] = len(block_reps)
    if len(block_reps) < 2:
        return stats

    block_cos, block_cos2, block_mse, block_pairs = _adjacent_pair_cos_mse_stats(np.stack(block_reps, axis=0))
    stats["latent_block_cos_mean"] = block_cos
    stats["latent_block_cos2_mean"] = block_cos2
    stats["latent_block_mse_mean"] = block_mse
    stats["latent_block_pair_count"] = block_pairs
    return stats


FUTURE_L1_SYSTEM_PROMPT = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.


Use the following tags to switch your thinking mode:

1.  **Textual Mode**: `<reason>Your textual reasoning process</reason>`
    *   For logical analysis, planning, and verbal thought.

2.  **Visual Mode**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
    *   For mental visualization, imagination and simulation.


**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""


# TwiFF-Bench + FutureL1: same instructions without the MC-only last line so
# open-ended forecasting answers are not steered toward a single letter.
FUTURE_L1_SYSTEM_PROMPT_TWIFFBENCH = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.


Use the following tags to switch your thinking mode:

1.  **Textual Mode**: `<reason>Your textual reasoning process</reason>`
    *   For logical analysis, planning, and verbal thought.

2.  **Visual Mode**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
    *   For mental visualization, imagination and simulation.


**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""


def _resolve_code_root(code_root: Optional[str]) -> str:
    candidates = [
        code_root,
        os.environ.get("ROT_CODE_ROOT"),
        os.environ.get("FUTURE_L1_CODE_ROOT"),
    ]
    for c in candidates:
        if not c:
            continue
        # `src/model/future_l1` 既可能是目录(包),也可能是单文件 future_l1.py;两者都接受
        future_l1_pkg = os.path.join(c, "src", "model", "future_l1")
        future_l1_mod = os.path.join(c, "src", "model", "future_l1.py")
        monkey_patch = os.path.join(c, "src", "train", "monkey_patch_forward.py")
        if (os.path.isdir(future_l1_pkg) or os.path.isfile(future_l1_mod)) and os.path.isfile(monkey_patch):
            return c
    raise ValueError(
        "FutureL1 code_root not resolved. Pass `code_root=/path/to/FutureL1` "
        "via --model_args, or set env var ROT_CODE_ROOT / FUTURE_L1_CODE_ROOT. "
        "The directory must contain `src/model/future_l1(.py)` and `src/train/monkey_patch_forward.py`."
    )


def _strict_bool(v):
    """Allow `fixed_latent_budget=False` / `True` / int from CLI string."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s.lower() in {"false", "none", "no", "off", "0", ""}:
        return False
    if s.lower() in {"true", "yes", "on"}:
        return True
    try:
        return int(s)
    except ValueError:
        return v


@register_model("future_l1")
class FutureL1(Qwen3_VL):
    """FutureL1 (Qwen3-VL / Qwen3.5-VL backbone) model wrapper for lmms-eval.

    This subclass reuses ``Qwen3_VL`` preprocessing & batching, but:
      * applies FutureL1's mixed-modality / generation monkey patches before
        loading,
      * loads ``FutureL1_Qwen3VL`` / ``FutureL1_Qwen3_5_VL`` / ``RICE_Qwen3VL``
        instead of HF native ``Qwen3VLForConditionalGeneration``,
      * exposes inference-time config flags (``infer_head_prefer``,
        ``force_initial_latent_mode``, ``loose_latent_budget``,
        ``infer_latent_multiplier``, ``fixed_latent_budget``),
      * decodes with ``skip_special_tokens=False`` so latent markers survive,
      * extracts ``<answer>...</answer>`` from the raw output (and optionally
        the last ``\\boxed{...}`` group).
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 8192,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = FUTURE_L1_SYSTEM_PROMPT,
        interleave_visuals: Optional[bool] = False,
        # ---- FutureL1-specific ----
        code_root: Optional[str] = None,
        infer_head_prefer: str = "render",
        force_initial_latent_mode: bool = False,
        loose_latent_budget: Optional[int] = None,
        infer_latent_multiplier: float = 2.0,
        fixed_latent_budget=False,
        measure_latent_similarity: bool = False,
        post_process: bool = False,
        extract_answer: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        # Note: we deliberately skip Qwen3_VL.__init__ (it would load the wrong
        # model class) and call lmms.__init__ directly.
        lmms.__init__(self)

        # Backward-compat aliases.
        if "loss_latent_budget" in kwargs and loose_latent_budget is None:
            loose_latent_budget = kwargs.pop("loss_latent_budget")
        # Drop any kwargs that VLMEvalKit accepted but lmms-eval doesn't need.
        for ignored in ("use_vllm", "use_lmdeploy", "use_audio_in_video",
                        "save_raw_output", "max_new_tokens",
                        "use_custom_prompt", "nframe", "top_p", "top_k",
                        "temperature", "repetition_penalty", "gpu_utils"):
            kwargs.pop(ignored, None)
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn}, got {attn_implementation}"
            )

        # Coerce CLI strings.
        fixed_latent_budget = _strict_bool(fixed_latent_budget)
        force_initial_latent_mode = bool(_strict_bool(force_initial_latent_mode))
        measure_latent_similarity = bool(_strict_bool(measure_latent_similarity))
        infer_latent_multiplier = float(infer_latent_multiplier)
        if loose_latent_budget is not None:
            loose_latent_budget = int(loose_latent_budget)

        # Apply min/max pixels defaults that match Qwen3_VL when unset.
        if min_pixels is None:
            min_pixels = 256 * 28 * 28
        if max_pixels is None:
            max_pixels = 1605632

        # Distributed setup.
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Locate FutureL1 source tree and import classes / monkey patches.
        code_root = _resolve_code_root(code_root)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        eval_logger.info(f"[future_l1] using code_root={code_root}")

        from src.model.future_l1 import (
            RICE_Qwen3VL,
            FutureL1_Qwen3VL,
            FutureL1_Qwen3_5_VL,
        )
        from src.train.monkey_patch_forward import (
            replace_qwen3_with_mixed_modality_forward,
            replace_qwen3_vl_generation_forward,
            replace_qwen3_5_with_mixed_modality_forward,
            replace_qwen3_5_generation_forward,
        )

        backbone_cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        model_type = getattr(backbone_cfg, "model_type", None)

        if model_type == "qwen3_5":
            replace_qwen3_5_with_mixed_modality_forward()
            replace_qwen3_5_generation_forward()
            MODEL_CLS = FutureL1_Qwen3_5_VL
            dtype_key = "torch_dtype"
        else:
            replace_qwen3_with_mixed_modality_forward()
            replace_qwen3_vl_generation_forward()
            MODEL_CLS = RICE_Qwen3VL if force_initial_latent_mode else FutureL1_Qwen3VL
            dtype_key = "dtype"

        model_kwargs = {
            dtype_key: "bfloat16",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = MODEL_CLS.from_pretrained(pretrained, **model_kwargs).eval()

        # Inference-time config flags consumed by the monkey-patched forward.
        self._model.config.infer_head_prefer = infer_head_prefer
        self._model.config.force_initial_latent_mode = force_initial_latent_mode
        self._model.config.loose_latent_budget = loose_latent_budget
        self._model.config.infer_latent_multiplier = infer_latent_multiplier
        if fixed_latent_budget:
            self._model.config.fixed_latent_budget = True
            self._model.config.max_latent_token = int(fixed_latent_budget)
        else:
            self._model.config.fixed_latent_budget = False

        # Mirror Qwen3_VL attributes so its preprocessing helpers work as-is.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = None
        self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        # FutureL1-specific post-processing flags.
        self.post_process = post_process
        self.extract_answer = extract_answer
        self.measure_latent_similarity = measure_latent_similarity or os.environ.get("FUTURE_L1_EVAL_LATENT_SIM", "0") in ("1", "true", "True", "yes")
        self.export_latents = os.environ.get("FUTURE_L1_EXPORT_LATENTS", "0") in ("1", "true", "True", "yes")
        self.export_mirage_embeddings = os.environ.get("FUTURE_L1_EXPORT_MIRAGE_EMBEDDINGS", "0") in ("1", "true", "True", "yes")
        self.latent_export_dir = os.environ.get("FUTURE_L1_LATENT_EXPORT_DIR")
        self.latent_export_max_per_rank = int(os.environ.get("FUTURE_L1_LATENT_EXPORT_MAX_PER_RANK", "20000"))
        self.mirage_export_max_per_type_per_rank = int(os.environ.get("FUTURE_L1_MIRAGE_EXPORT_MAX_PER_TYPE_PER_RANK", "5000"))
        if self.export_mirage_embeddings:
            self.export_latents = True
        if self.export_latents:
            self.measure_latent_similarity = True
            if self.latent_export_dir is None:
                self.latent_export_dir = os.path.join(os.getcwd(), "future_l1_latent_exports")
            os.makedirs(self.latent_export_dir, exist_ok=True)
        self._latent_export_counter = 0
        self._mirage_export_counter = 0
        self._vision_token_ids = _vision_token_ids(self._model.config, self._tokenizer)
        self._special_token_ids = _special_token_ids(self._model.config, self._tokenizer)
        self.latent_similarity_stats: list[Optional[Dict[str, Any]]] = []
        # verbose: print model outputs to stdout while running. Also honors
        # the env var FUTURE_L1_VERBOSE=1 for shell-side toggling.
        self.verbose = bool(_strict_bool(verbose)) or os.environ.get("FUTURE_L1_VERBOSE", "0") in ("1", "true", "True", "yes")

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _get_system_prompt_for_instance(self, task_name: str) -> str:
        """Use a TwiFF-specific default system when the task is ``twiffbench_future_l1``.

        If ``system_prompt`` was overridden via ``--model_args``, keep the user's
        string for all tasks.
        """
        if str(task_name) not in ("twiffbench_future_l1", "twiffbench_swimbird"):
            return self.system_prompt
        if self.system_prompt == FUTURE_L1_SYSTEM_PROMPT:
            return FUTURE_L1_SYSTEM_PROMPT_TWIFFBENCH
        return self.system_prompt

    # ---- post-processing helpers ----
    @staticmethod
    def _strip_boxed(text: str) -> str:
        """Extract content of the last ``\\boxed{...}`` (matching braces)."""
        if "\\boxed{" not in text:
            return text
        resp = text.split("\\boxed{")[-1]
        counter, end = 1, None
        for i, c in enumerate(resp):
            if c == "{":
                counter += 1
            elif c == "}":
                counter -= 1
            if counter == 0:
                end = i
                break
        if end is None:
            end = len(resp)
        return resp[:end]

    # ---- generation ----
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Same as ``Qwen3_VL.generate_until`` but:
        * pops ``mm_token_type_ids`` (some processor variants emit it but the
          patched generate signature does not accept it),
        * decodes with ``skip_special_tokens=False`` to keep latent markers,
        * extracts ``<answer>...</answer>`` (and optionally ``\\boxed{...}``).
        """
        res: list[str] = []
        raw_res: list[str] = []
        latent_similarity_stats: list[Optional[Dict[str, Any]]] = []
        export_embeddings: list[np.ndarray] = []
        export_request_pos: list[np.ndarray] = []
        export_block_id: list[np.ndarray] = []
        export_token_pos: list[np.ndarray] = []
        export_global_pos: list[np.ndarray] = []
        mirage_embeddings: list[np.ndarray] = []
        mirage_request_pos: list[np.ndarray] = []
        mirage_type_id: list[np.ndarray] = []
        mirage_source_pos: list[np.ndarray] = []

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

                # Compatibility guard for processor/model signature mismatch.
                if "mm_token_type_ids" in inputs:
                    inputs.pop("mm_token_type_ids")

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                if self.measure_latent_similarity:
                    generate_kwargs.update(
                        {
                            "return_dict_in_generate": True,
                            "output_scores": False,
                            "output_hidden_states": False,
                        }
                    )
                if self.export_mirage_embeddings:
                    input_ids_cpu = inputs.input_ids.detach().cpu().numpy()
                    prompt_hidden = None
                    try:
                        export_model = self.accelerator.unwrap_model(self.model)
                    except Exception:  # noqa: BLE001
                        export_model = self.model
                    inner_model = getattr(export_model, "model", None)
                    if inner_model is not None:
                        with torch.no_grad():
                            prompt_outputs = inner_model(**inputs, return_dict=True)
                            prompt_hidden = prompt_outputs.last_hidden_state.detach().float().cpu().numpy()
                    else:
                        eval_logger.warning("[future_l1] Mirage-style export skipped prompt hidden states: no inner model found")
                else:
                    prompt_hidden = None
                    input_ids_cpu = None

                cont = self.model.generate(**inputs, **generate_kwargs)
                generated_sequences = cont.sequences if hasattr(cont, "sequences") else cont
                latent_states = getattr(cont, "latent_states", None)
                latent_mask = getattr(cont, "latent_mask", None)

                trimmed = [out[len(in_):] for in_, out in zip(inputs.input_ids, generated_sequences)]
                # IMPORTANT: keep special tokens so latent markers + <answer> tags survive.
                raw_answers = self.processor.tokenizer.batch_decode(
                    trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                for batch_idx, (raw, context) in enumerate(zip(raw_answers, contexts)):
                    if self.export_mirage_embeddings and prompt_hidden is not None and input_ids_cpu is not None:
                        ids = input_ids_cpu[batch_idx]
                        hidden = prompt_hidden[batch_idx]
                        text_mask = np.asarray(
                            [(int(tid) not in self._special_token_ids) for tid in ids],
                            dtype=bool,
                        )
                        vision_mask = np.asarray(
                            [(int(tid) in self._vision_token_ids) for tid in ids],
                            dtype=bool,
                        )
                        for type_value, mask in ((0, text_mask), (1, vision_mask)):
                            current_count = sum(int((types == type_value).sum()) for types in mirage_type_id)
                            remaining_type = self.mirage_export_max_per_type_per_rank - current_count
                            if remaining_type <= 0:
                                continue
                            pos = np.flatnonzero(mask)[:remaining_type]
                            if pos.size == 0:
                                continue
                            mirage_embeddings.append(hidden[pos].astype(np.float16))
                            mirage_request_pos.append(np.full(pos.size, len(res), dtype=np.int32))
                            mirage_type_id.append(np.full(pos.size, type_value, dtype=np.int8))
                            mirage_source_pos.append(pos.astype(np.int32))

                    response = raw
                    for term in until:
                        if term:
                            response = response.split(term)[0]
                    if self.measure_latent_similarity:
                        latents = _latent_trajectory_from_output(latent_states, latent_mask, batch_idx)
                        latent_stats = _compute_latent_similarity_stats(raw, latents)
                        if self.export_latents and latents is not None and latents.shape[0] > 0:
                            remaining = self.latent_export_max_per_rank - sum(arr.shape[0] for arr in export_embeddings)
                            if remaining > 0:
                                latents_to_save = latents[:remaining]
                                block_id, token_pos, global_pos = _latent_token_metadata(raw, latents_to_save.shape[0])
                                export_embeddings.append(latents_to_save.astype(np.float16))
                                export_request_pos.append(np.full(latents_to_save.shape[0], len(res), dtype=np.int32))
                                export_block_id.append(block_id)
                                export_token_pos.append(token_pos)
                                export_global_pos.append(global_pos)
                            if self.export_mirage_embeddings:
                                remaining_latent = self.mirage_export_max_per_type_per_rank - sum(
                                    int((types == 2).sum()) for types in mirage_type_id
                                )
                                if remaining_latent > 0:
                                    latent_m = latents[:remaining_latent].astype(np.float16)
                                    mirage_embeddings.append(latent_m)
                                    mirage_request_pos.append(np.full(latent_m.shape[0], len(res), dtype=np.int32))
                                    mirage_type_id.append(np.full(latent_m.shape[0], 2, dtype=np.int8))
                                    mirage_source_pos.append(np.arange(latent_m.shape[0], dtype=np.int32))
                    else:
                        latent_stats = None
                    if self.post_process:
                        response = self._strip_boxed(response)
                    if self.extract_answer:
                        m = _ANSWER_REGEX.search(response)
                        if m:
                            response = m.group(1).strip()
                        else:
                            response = response.strip()
                    if self.verbose:
                        print(f"\033[31m[future_l1][prompt rank{self.rank}]\033[0m", context)
                        print(f"\033[33m[future_l1][raw    rank{self.rank}]\033[0m", raw)
                        print(f"\033[32m[future_l1][answer rank{self.rank}]\033[0m", response)
                        print("\033[90m" + "-" * 80 + "\033[0m", flush=True)
                    res.append(response)
                    raw_res.append(raw)
                    latent_similarity_stats.append(latent_stats)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
                    pbar.update(1)

        if self.export_latents and export_embeddings:
            # Map generated-order request positions back to the original request order,
            # so metadata can be joined with samples_*.jsonl by doc/order after eval.
            request_pos_to_original = np.full(len(res), -1, dtype=np.int32)
            for original_idx, generated_idx in enumerate(re_ords.reorder_indices):
                if generated_idx < request_pos_to_original.shape[0]:
                    request_pos_to_original[generated_idx] = original_idx
            request_pos = np.concatenate(export_request_pos, axis=0)
            original_request_pos = request_pos_to_original[request_pos]
            export_path = os.path.join(
                self.latent_export_dir,
                f"latent_embeddings_rank{self.rank}_part{self._latent_export_counter}.npz",
            )
            np.savez_compressed(
                export_path,
                embeddings=np.concatenate(export_embeddings, axis=0),
                request_pos=request_pos,
                original_request_pos=original_request_pos,
                block_id=np.concatenate(export_block_id, axis=0),
                token_pos=np.concatenate(export_token_pos, axis=0),
                global_pos=np.concatenate(export_global_pos, axis=0),
            )
            self._latent_export_counter += 1
            eval_logger.info(f"[future_l1] exported latent embeddings to {export_path}")

        if self.export_mirage_embeddings and mirage_embeddings:
            request_pos_to_original = np.full(len(res), -1, dtype=np.int32)
            for original_idx, generated_idx in enumerate(re_ords.reorder_indices):
                if generated_idx < request_pos_to_original.shape[0]:
                    request_pos_to_original[generated_idx] = original_idx
            mirage_request = np.concatenate(mirage_request_pos, axis=0)
            mirage_original_request = request_pos_to_original[mirage_request]
            mirage_path = os.path.join(
                self.latent_export_dir,
                f"mirage_embeddings_rank{self.rank}_part{self._mirage_export_counter}.npz",
            )
            np.savez_compressed(
                mirage_path,
                embeddings=np.concatenate(mirage_embeddings, axis=0),
                request_pos=mirage_request,
                original_request_pos=mirage_original_request,
                type_id=np.concatenate(mirage_type_id, axis=0),
                source_pos=np.concatenate(mirage_source_pos, axis=0),
                type_names=np.asarray(["text", "vision", "latent"], dtype="<U8"),
            )
            self._mirage_export_counter += 1
            eval_logger.info(f"[future_l1] exported Mirage-style embeddings to {mirage_path}")

        res = re_ords.get_original(res)
        raw_res = re_ords.get_original(raw_res)
        latent_similarity_stats = re_ords.get_original(latent_similarity_stats)
        self.raw_responses = raw_res
        self.latent_similarity_stats = latent_similarity_stats
        pbar.close()
        return res
