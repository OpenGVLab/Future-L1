r"""CoT / thinking Qwen3-VL reward (no FutureL1 latent tokens).

Format: ``<reason>...</reason>`` then ``<answer>...</answer>`` (FutureBench-thinking style).
Reuses rule/LLM accuracy helpers from ``future_l1_reward_function``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import importlib.util
from pathlib import Path as _Path

_sb_spec = importlib.util.spec_from_file_location(
    "future_l1_reward_function",
    _Path(__file__).with_name("future_l1_reward_function.py"),
)
_sb = importlib.util.module_from_spec(_sb_spec)
assert _sb_spec.loader is not None
_sb_spec.loader.exec_module(_sb)

REWARD_TYPE = _sb.REWARD_TYPE
_api_batch_judge_safe = _sb._api_batch_judge_safe
_resolve_int = _sb._resolve_int
_truthy_env = _sb._truthy_env
compute_repetition_penalty = _sb.compute_repetition_penalty
extract_and_check = _sb.extract_and_check

REWARD_NAME = "cot_thinking"

FORMAT_RE = re.compile(
    r"^\s*<reason>.*?</reason>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)


def format_reward(predict: str) -> float:
    return 1.0 if FORMAT_RE.fullmatch(predict.strip()) else 0.0


def compute_score(
    items: List[Dict[str, Any]],
    *,
    format_weight: Optional[float] = None,
    length_penalty_weight: Optional[float] = None,
    latent_div_lambda: Optional[float] = None,
    latent_ctr_lambda: Optional[float] = None,
    latent_ctr_temperature: Optional[float] = None,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    # FutureL1-only kwargs from config_future_l1.yaml are accepted but ignored.
    del latent_div_lambda, latent_ctr_lambda, latent_ctr_temperature, kwargs
    if format_weight is None:
        try:
            format_weight = float(os.environ.get("COT_FORMAT_WEIGHT", "0.1"))
        except ValueError:
            format_weight = 0.1
    if length_penalty_weight is None:
        try:
            length_penalty_weight = float(os.environ.get("COT_LENGTH_PENALTY_WEIGHT", "0.001"))
        except ValueError:
            length_penalty_weight = 0.001

    use_llm_judge = _truthy_env("USE_LLM_JUDGE")
    llm_only = _truthy_env("LLM_JUDGE_ONLY")

    predicts = [str(it.get("response", "")) for it in items]
    ground_truths = [str(it.get("ground_truth", "")) for it in items]
    questions = [str(it.get("problem", "")) for it in items]
    response_lengths = [_resolve_int(it.get("response_length"), 0) for it in items]
    ref_lengths = [_resolve_int(it.get("ref_resp_length"), 0) for it in items]

    format_scores = [format_reward(p) for p in predicts]
    repetition_scores = [compute_repetition_penalty(p) for p in predicts]
    rule_accuracies = [extract_and_check(p, gt) for p, gt in zip(predicts, ground_truths)]

    if use_llm_judge:
        if llm_only:
            final_accuracies = [float(c) for c in _api_batch_judge_safe(questions, predicts, ground_truths)]
        else:
            unresolved_idx = [i for i, a in enumerate(rule_accuracies) if a == 0.0]
            final_accuracies = list(rule_accuracies)
            if unresolved_idx:
                sub_q = [questions[i] for i in unresolved_idx]
                sub_p = [predicts[i] for i in unresolved_idx]
                sub_g = [ground_truths[i] for i in unresolved_idx]
                llm_correct = _api_batch_judge_safe(sub_q, sub_p, sub_g)
                for j, idx in enumerate(unresolved_idx):
                    final_accuracies[idx] = float(llm_correct[j])
    else:
        final_accuracies = rule_accuracies

    out: List[Dict[str, float]] = []
    for i, item in enumerate(items):
        accuracy_score = final_accuracies[i]
        if repetition_scores[i] > 0.5:
            accuracy_score = -1.0
        ref_len = ref_lengths[i] if ref_lengths[i] > 0 else response_lengths[i]
        length_part = length_penalty_weight * max(0, response_lengths[i] - ref_len)
        accuracy_part = (1.0 - format_weight) * accuracy_score
        format_part = format_weight * format_scores[i]
        out.append(
            {
                "overall": accuracy_part + format_part - length_part,
                "format": format_scores[i],
                "accuracy": accuracy_score,
                "accuracy_reward_part": accuracy_part,
                "format_reward_part": format_part,
                "length_penalty_part": length_part,
                "repetition_score": repetition_scores[i],
                "is_repetitive": float(repetition_scores[i] > 0.5),
            }
        )
    return out
