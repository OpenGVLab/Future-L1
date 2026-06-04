r"""FutureL1 RL reward function (RL_v2 port).

Based on ``Future-L1/RL_v2/examples/reward_function/future_l1_reward_function.py``
with **format rules tightened**: RL_v2 accepts L-first ``(L R)+ A`` and R-first
``R (L R)+ A`` (requiring at least one latent), switches to R-first only when
the initial checkpoint path contains ``rfirst``, and uses a ``<reason>``
pattern that cannot span two adjacent reason blocks.

* **Format reward**: HyLar-aligned format with FutureL1 token names,
  i.e. ``<think>`` -> ``<reason>`` and
  ``<|canvas_start|><canvas><|canvas_end|>`` -> ``<|latent_start|>...<|latent_end|>``.
  Valid completions are **L-first** ``(L R)+ A`` or **R-first** ``R (L R)+ A``
  by default (each ``L`` is a latent span, each ``R`` a reason span), with at
  least one ``L R`` pair so ``RA`` / ``RRA``-style skips are not format-perfect.
  ``FUTURE_L1_FORMAT_MODE=auto`` changes this to R-first only if
  ``FUTURE_L1_INITIAL_MODEL_PATH`` / ``MODEL_PATH`` contains ``rfirst``.
* **Accuracy reward**: rule-based ``extract_and_check`` (mathruler when
  available, otherwise strict string equality of the inner-``<answer>``
  text). Optionally upgraded by an LLM-as-judge fallback when the env var
  ``USE_LLM_JUDGE=1`` is set; the rule check is run first, and the LLM is
  only consulted for samples it failed (i.e. ``rule_then_api_batch_judge``
  semantics) - flip to LLM-only by setting ``LLM_JUDGE_ONLY=1``.
* **Repetition penalty**: same n-gram / consecutive-window / line-dup
  heuristic; if the repetition score exceeds 0.5 the accuracy is set to
  ``-1.0`` (matches the upstream behavior).
* **Length penalty**: ``length_penalty_weight * max(0, len - ref_len)``;
  ``ref_resp_lengths`` is consumed when the batch carries it (otherwise the
  per-sample response length is used and the penalty is 0).
* **Temporal diversity reward** ``R_div``: optional ``latent_div_lambda *
  mean(cos(block_t, block_{t+1})^2)`` (equivalent to ``+ latent_div_lambda *
  R_div`` with ``R_div = -mean(cos^2)``). Each latent block is mean-pooled
  before comparing adjacent keyframe blocks.
* **Outcome-contrastive latent reward** ``R_ctr`` (LA-DAPO): optional
  ``latent_ctr_lambda * R_ctr`` grouped by rollout ``uid``, using recorded
  per-step latents and hardest-positive InfoNCE (paper Eq. ctr).

Final composition:
    overall = (1 - format_weight) * accuracy  +  format_weight * format  -  length_penalty_weight * length_penalty  -  latent_div_part  +  latent_ctr_part

Interface (EasyR1 ``BatchFunctionRewardManagerMixin``)::

    def compute_score(items: list[RewardInput]) -> list[RewardScore]

where ``RewardInput`` has ``response``, ``response_length``, ``ground_truth``
and (when ``BatchFunctionRewardManagerMixin`` is patched) ``problem`` and
``ref_resp_lengths``.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import importlib.util
from pathlib import Path as _Path

_ctr_spec = importlib.util.spec_from_file_location(
    "outcome_contrastive_latent_reward",
    _Path(__file__).with_name("outcome_contrastive_latent_reward.py"),
)
outcome_contrastive_latent_reward = importlib.util.module_from_spec(_ctr_spec)
assert _ctr_spec.loader is not None
_ctr_spec.loader.exec_module(outcome_contrastive_latent_reward)


REWARD_NAME = "future_l1"
REWARD_TYPE = "batch"


# --------------------------- Regex / token helpers ----------------------
# Disallow a second ``<reason>`` inside the span; otherwise ``.*?`` + DOTALL
# can backtrack so one ``REASON`` match consumes ``R+R`` (invalid ``LRRA``).
REASON_BLOCK_RE = r"<reason>((?:(?!<reason>).)*?)</reason>"
# HyLar-style canonical latent block. ``_normalize_predict`` first replaces
# arbitrary content inside a complete latent span with exactly this block; any
# extra boundary marker (e.g. a second ``<|latent_end|>``) remains outside the
# canonical span and therefore breaks the strict format regex.
LATENT_BLOCK_CANONICAL = "<|latent_start|><|latent|><|latent_end|>"
LATENT_BLOCK_RE = r"<\|latent_start\|><\|latent\|><\|latent_end\|>"
RAW_LATENT_BLOCK_RE = re.compile(
    r"(<\|latent_start\|>)(.*?)(<\|latent_end\|>)",
    re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
# L-first: (L R)+ A ; R-first: R (L R)+ A .  Require >=1 latent span (the (L R)+
# tail is non-empty in both branches) so plain ``RA`` does not get format=1.
# ``FUTURE_L1_FORMAT_MODE=auto`` switches to R-first only when the initial SFT
# checkpoint path contains ``rfirst``; set ``both`` / ``rfirst`` to override.
FORMAT_BOTH_RE = re.compile(
    rf"^\s*(?:"
    rf"(?:{LATENT_BLOCK_RE}\s*{REASON_BLOCK_RE}\s*)+"
    rf"|"
    rf"{REASON_BLOCK_RE}(?:\s*{LATENT_BLOCK_RE}\s*{REASON_BLOCK_RE})+"
    rf")\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
FORMAT_RFIRST_RE = re.compile(
    rf"^\s*{REASON_BLOCK_RE}(?:\s*{LATENT_BLOCK_RE}\s*{REASON_BLOCK_RE})+"
    rf"\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)

try:
    from mathruler.grader import grade_answer
except Exception:  # pragma: no cover  - mathruler is optional

    def grade_answer(answer, ground_truth):  # type: ignore[no-redef]
        return str(answer).strip().lower() == str(ground_truth).strip().lower()


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y"}


def _format_mode() -> str:
    """Resolve format mode: ``both`` or ``rfirst``.

    ``auto`` preserves old behavior unless the RL initial checkpoint path name
    contains ``rfirst`` (e.g. ``TwiFF-interleave-top50K-rfirst-lambda0.2-max4``).
    """
    mode = str(os.environ.get("FUTURE_L1_FORMAT_MODE", "auto")).strip().lower()
    if mode in {"both", "rfirst"}:
        return mode
    if mode != "auto":
        print(
            f"[future_l1 reward] unknown FUTURE_L1_FORMAT_MODE={mode!r}; using auto",
            file=sys.stderr,
        )

    init_path = (
        os.environ.get("FUTURE_L1_INITIAL_MODEL_PATH")
        or os.environ.get("MODEL_PATH")
        or ""
    )
    return "rfirst" if "rfirst" in init_path.lower() else "both"


def extract_answer_text_for_judge(pred: Optional[str]) -> str:
    """First ``<answer>...</answer>`` inner text; empty string if missing."""
    if pred is None:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", str(pred), re.DOTALL)
    return m.group(1).strip() if m else ""


def format_reward(predict: str) -> float:
    # Accept both raw decoded responses and already-normalized strings. The
    # training path calls this on normalized text, but direct debugging often
    # passes raw rollout samples.
    predict = _normalize_predict(str(predict))
    has_answer = ANSWER_RE.search(predict) is not None
    pattern = FORMAT_RFIRST_RE if _format_mode() == "rfirst" else FORMAT_BOTH_RE
    return 1.0 if has_answer and pattern.match(predict) is not None else 0.0


def extract_and_check(predict: str, ground_truth: str) -> float:
    answer_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else "None"
    return float(bool(grade_answer(answer, ground_truth)))


def accuracy_reward(predict: str, ground_truth: str) -> float:
    return 1.0 if extract_and_check(predict, ground_truth) else 0.0


def compute_repetition_penalty(text: str, ngram_size: int = 4, window_size: int = 50) -> float:
    clean_text = re.sub(r"<\|.*?\|>", " ", text)
    clean_text = re.sub(r"</?(?:think|reason|answer)>", " ", clean_text)
    tokens = clean_text.split()
    if len(tokens) < ngram_size + 1:
        return 0.0

    ngrams = [tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    ngram_rep_rate = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    max_consecutive_ratio = 0.0
    if len(tokens) >= window_size * 2:
        for ws in (window_size, window_size // 2):
            if ws < 5:
                continue
            for i in range(len(tokens) - ws * 2 + 1):
                block = " ".join(tokens[i : i + ws])
                if block == " ".join(tokens[i + ws : i + ws * 2]):
                    repeat_count = 2
                    pos = i + ws * 2
                    while pos + ws <= len(tokens) and " ".join(tokens[pos : pos + ws]) == block:
                        repeat_count += 1
                        pos += ws
                    max_consecutive_ratio = max(max_consecutive_ratio, repeat_count * ws / len(tokens))

    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 10]
    line_rep_rate = 1.0 - len(set(lines)) / len(lines) if len(lines) > 2 else 0.0
    return min(max(ngram_rep_rate, max_consecutive_ratio, line_rep_rate), 1.0)


# --------------------------- LLM judge wrapper --------------------------
_API_JUDGE_FN = None


def _get_api_batch_judge():
    """Lazily import the FutureL1 api_batch_judge wrapper.

    The function is only required when ``USE_LLM_JUDGE=1`` is set, so we
    delay the import to keep the reward fn usable in environments without
    openai installed.
    """
    global _API_JUDGE_FN
    if _API_JUDGE_FN is not None:
        return _API_JUDGE_FN

    # Make sure the RL_v2 root is on sys.path so we can import future_l1_rl.
    rl_v2_root = os.environ.get("FUTURE_L1_RL_V2_ROOT")
    if not rl_v2_root:
        rl_v2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if rl_v2_root not in sys.path:
        sys.path.insert(0, rl_v2_root)

    from future_l1_rl.judge.api_judge import api_batch_judge  # noqa: PLC0415

    _API_JUDGE_FN = api_batch_judge
    return _API_JUDGE_FN


def _api_batch_judge_safe(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
) -> List[float]:
    """LLM-as-judge call that never propagates exceptions to the trainer.

    Returns 0.0 for every sample on failure (so a momentary API outage
    degrades gracefully into 'incorrect' rather than crashing the run).
    """
    api_name = os.environ.get("JUDGE_API_NAME", "gpt-4o-mini")
    api_url = os.environ.get("JUDGE_API_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    try:
        workers = int(os.environ.get("API_JUDGE_WORKERS", "32"))
    except ValueError:
        workers = 32

    try:
        fn = _get_api_batch_judge()
        return list(
            fn(
                questions,
                [extract_answer_text_for_judge(p) for p in preds],
                gts,
                api_name=api_name,
                api_max_workers=workers,
                api_url=api_url,
                api_key=api_key,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[future_l1 reward] LLM judge unavailable, falling back to rule-only: {exc}", file=sys.stderr)
        return [0.0] * len(preds)


# --------------------------- Public callables ---------------------------
def _resolve_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _length_penalty_for(response_length: int, ref_response_length: Optional[int]) -> float:
    if not ref_response_length or ref_response_length <= 0:
        return 0.0
    return float(max(0, response_length - ref_response_length))


# Trailing chat / generation terminators that vLLM may decode into the
# response when ``skip_special_tokens=False``. ``get_response_mask`` keeps
# the first EOS position in the mask -> the decoder emits the literal
# ``<|im_end|>`` (or ``<|endoftext|>``) string at the end of the response,
# which breaks the format regex's ``$`` anchor. Strip them up-front so the
# downstream regex / answer extraction sees only the model's payload.
# This mirrors Future-L1/RL_v2/verl/workers/reward/function.py:
# normalize_reward_response_text.
_TRAILING_SPECIALS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|eot|>",
    "<|eot_id|>",
)


def _strip_terminators(text: str) -> str:
    for tok in _TRAILING_SPECIALS:
        text = text.replace(tok, "")
    return text


def _normalize_tag_spacing(text: str) -> str:
    text = _strip_terminators(text)
    return re.sub(r"\s*(<|>|/)\s*", r"\1", text)


def _replace_latent_token_content(text: str) -> str:
    """HyLar-style canonicalization of complete latent spans.

    The non-greedy match consumes only the first closing marker, so malformed
    output such as ``<|latent_end|><|latent_end|>`` keeps the second marker in
    the text and fails the strict format regex.
    """
    return RAW_LATENT_BLOCK_RE.sub(lambda _m: LATENT_BLOCK_CANONICAL, text)


def _normalize_predict(text: str) -> str:
    """Strip chat terminators, then collapse ``< / >`` whitespace artefacts so
    the strict HyLar-aligned format regex matches. Matches upstream
    future_l1_rl behaviour."""
    return _replace_latent_token_content(_normalize_tag_spacing(text))


def _as_float_latents(value: Any) -> Optional[np.ndarray]:
    """Convert one rollout's latent trajectory to ``(steps, dim)`` float32."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(x) for x in arr if x is not None], axis=0)
        except Exception:  # noqa: BLE001
            return None
    try:
        arr = arr.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    return arr


def _latent_block_token_counts(raw_predict: str) -> List[int]:
    """Number of latent tokens in each complete latent block."""
    text = _normalize_tag_spacing(str(raw_predict))
    counts: List[int] = []
    for match in RAW_LATENT_BLOCK_RE.finditer(text):
        counts.append(match.group(2).count("<|latent|>"))
    return counts


def _temporal_latent_diversity_stats(raw_predict: str, latents: Any) -> Tuple[float, int, int, int]:
    """Return ``(mean_cos2, pair_count, block_count, latent_count)``.

    Each latent block corresponds to one keyframe in the current SFT design, so
    we mean-pool within a block and compare only adjacent block representatives.
    """
    arr = _as_float_latents(latents)
    counts = _latent_block_token_counts(raw_predict)
    if arr is None or not counts:
        return 0.0, 0, len(counts), 0 if arr is None else int(arr.shape[0])

    offset = 0
    block_reps: List[np.ndarray] = []
    for count in counts:
        count = int(max(0, count))
        next_offset = offset + count
        if count > 0 and offset < arr.shape[0]:
            block = arr[offset : min(next_offset, arr.shape[0])]
            if block.shape[0] > 0:
                block_reps.append(block.mean(axis=0))
        offset = next_offset

    if len(block_reps) < 2:
        return 0.0, 0, len(block_reps), int(arr.shape[0])

    cos2_values: List[float] = []
    for left, right in zip(block_reps, block_reps[1:]):
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-12:
            continue
        cos = float(np.dot(left, right) / denom)
        cos2_values.append(cos * cos)
    if not cos2_values:
        return 0.0, 0, len(block_reps), int(arr.shape[0])
    return float(np.mean(cos2_values)), len(cos2_values), len(block_reps), int(arr.shape[0])


def compute_score(
    items: List[Dict[str, Any]],
    *,
    format_weight: Optional[float] = None,
    length_penalty_weight: Optional[float] = None,
    latent_div_lambda: Optional[float] = None,
    latent_ctr_lambda: Optional[float] = None,
    latent_ctr_temperature: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Batch reward used by EasyR1's BatchFunctionRewardManager.

    Each ``item`` follows EasyR1's ``RewardInput`` shape with the small
    additions made by ``RL_v2/verl/workers/reward/function.py``:

        * ``response``         (str) - decoded model output
        * ``response_length``  (int) - per-sample response length
        * ``ground_truth``     (str) - reference answer
        * ``problem``          (str, optional) - the user question; required
          when ``USE_LLM_JUDGE=1`` so the judge sees task context
        * ``ref_resp_length``  (int, optional) - for the length penalty
        * ``latents``          (array-like, optional) - per-step rollout latent
          vectors recorded by ``future_l1_depo`` rollout

    Returns one ``RewardScore`` dict per item with keys::

        overall, format, accuracy,
        accuracy_reward_part, format_reward_part, length_penalty_part,
        latent_div_part, latent_block_cos2,
        latent_block_pair_count, latent_block_count, latent_vector_count,
        repetition_score, is_repetitive
    """
    if format_weight is None:
        try:
            format_weight = float(os.environ.get("FUTURE_L1_FORMAT_WEIGHT", "0.1"))
        except ValueError:
            format_weight = 0.1
    if length_penalty_weight is None:
        try:
            length_penalty_weight = float(os.environ.get("FUTURE_L1_LENGTH_PENALTY_WEIGHT", "0.001"))
        except ValueError:
            length_penalty_weight = 0.001
    if latent_div_lambda is None:
        try:
            latent_div_lambda = float(
                os.environ.get(
                    "FUTURE_L1_LATENT_DIV_LAMBDA",
                    os.environ.get("FUTURE_L1_DIVERSITY_PENALTY_BETA", "0.0"),
                )
            )
        except ValueError:
            latent_div_lambda = 0.0
    if latent_ctr_lambda is None:
        try:
            latent_ctr_lambda = float(
                os.environ.get(
                    "FUTURE_L1_LATENT_CTR_LAMBDA",
                    os.environ.get("FUTURE_L1_COLVR_LATENT_COEF", "0.0"),
                )
            )
        except ValueError:
            latent_ctr_lambda = 0.0
    if latent_ctr_temperature is None:
        try:
            latent_ctr_temperature = float(
                os.environ.get(
                    "FUTURE_L1_LATENT_CTR_TEMPERATURE",
                    os.environ.get("FUTURE_L1_COLVR_TEMPERATURE", "0.5"),
                )
            )
        except ValueError:
            latent_ctr_temperature = 0.5

    use_llm_judge = _truthy_env("USE_LLM_JUDGE")
    llm_only = _truthy_env("LLM_JUDGE_ONLY")

    raw_predicts = [str(it.get("response", "")) for it in items]
    predicts = [_normalize_predict(p) for p in raw_predicts]
    ground_truths = [str(it.get("ground_truth", "")) for it in items]
    questions = [str(it.get("problem", "")) for it in items]
    response_lengths = [_resolve_int(it.get("response_length"), 0) for it in items]
    ref_lengths = [_resolve_int(it.get("ref_resp_length"), 0) for it in items]

    # 1) format + repetition + initial rule-based accuracy.
    format_scores: List[float] = [format_reward(p) for p in predicts]
    repetition_scores: List[float] = [compute_repetition_penalty(p) for p in predicts]
    rule_accuracies: List[float] = [
        extract_and_check(p, gt) for p, gt in zip(predicts, ground_truths)
    ]

    # 2) optional LLM judge.
    if use_llm_judge:
        if llm_only:
            llm_correct = _api_batch_judge_safe(questions, predicts, ground_truths)
            final_accuracies = [float(c) for c in llm_correct]
        else:
            # rule-first; only call the LLM on unresolved (=0) samples.
            unresolved_idx = [i for i, a in enumerate(rule_accuracies) if a == 0.0]
            final_accuracies = list(rule_accuracies)
            if unresolved_idx:
                sub_q = [questions[i] for i in unresolved_idx]
                sub_p = [predicts[i] for i in unresolved_idx]
                sub_g = [ground_truths[i] for i in unresolved_idx]
                llm_correct = _api_batch_judge_safe(sub_q, sub_p, sub_g)
                for k, idx in enumerate(unresolved_idx):
                    final_accuracies[idx] = float(llm_correct[k])
    else:
        final_accuracies = rule_accuracies

    repetition_flags_pre = [compute_repetition_penalty(p) > 0.5 for p in predicts]
    ctr_rewards, ctr_cases, ctr_case_ids = outcome_contrastive_latent_reward.latent_rewards_batch(
        items,
        final_accuracies,
        repetition_flags_pre,
        temperature=latent_ctr_temperature,
        as_float_latents=_as_float_latents,
    )

    # 3) compose final reward.
    scores: List[Dict[str, float]] = []
    for i in range(len(items)):
        accuracy_score = final_accuracies[i]
        repetition_score = repetition_scores[i]
        is_repetitive = repetition_score > 0.5
        if is_repetitive:
            accuracy_score = -1.0

        length_penalty = _length_penalty_for(response_lengths[i], ref_lengths[i])
        accuracy_part = (1.0 - format_weight) * accuracy_score
        format_part = format_weight * format_scores[i]
        length_part = length_penalty_weight * length_penalty
        latent_cos2, latent_pair_count, latent_block_count, latent_vector_count = _temporal_latent_diversity_stats(
            raw_predicts[i],
            items[i].get("latents"),
        )
        latent_div_part = float(latent_div_lambda or 0.0) * latent_cos2
        latent_ctr_part = float(latent_ctr_lambda or 0.0) * ctr_rewards[i]
        scores.append(
            {
                "overall": accuracy_part + format_part - length_part - latent_div_part + latent_ctr_part,
                "format": format_scores[i],
                "accuracy": accuracy_score,
                "accuracy_reward_part": accuracy_part,
                "format_reward_part": format_part,
                "length_penalty_part": length_part,
                "latent_div_part": latent_div_part,
                "latent_block_cos2": latent_cos2,
                "latent_block_pair_count": float(latent_pair_count),
                "latent_block_count": float(latent_block_count),
                "latent_vector_count": float(latent_vector_count),
                "latent_ctr_reward": ctr_rewards[i],
                "latent_ctr_part": latent_ctr_part,
                "latent_ctr_valid": float(ctr_rewards[i] != 0.0 or ctr_cases[i] != "none"),
                "latent_ctr_case_id": ctr_case_ids[i],
                "repetition_score": repetition_score,
                "is_repetitive": float(is_repetitive),
            }
        )

    # Optional debug dump: ``FUTURE_L1_REWARD_DEBUG=1`` prints up to N decoded
    # responses + their scores on every reward call. Use this when format
    # reward is suspiciously 0 across the board - 99% of the time the model
    # output simply doesn't match the HyLar-aligned format regex (e.g. L-first
    # under rfirst mode, R-first with no latent block, ``RA``/``RRA``, or
    # trailing chat-template tokens).
    if _truthy_env("FUTURE_L1_REWARD_DEBUG"):
        try:
            n_show = int(os.environ.get("FUTURE_L1_REWARD_DEBUG_N", "3"))
        except ValueError:
            n_show = 3
        for i in range(min(n_show, len(items))):
            s = scores[i]
            preview = predicts[i][:400].replace("\n", "\\n")
            print(
                f"[future_l1 reward DEBUG] fmt={s['format']:.1f} acc={s['accuracy']:+.1f} "
                f"r_ctr={s.get('latent_ctr_reward', 0):.3f} rep={s['repetition_score']:.2f} | resp[:400]={preview!r}",
                flush=True,
            )

    return scores


# Keep the upstream alias name so launchers / configs that reference it still work.
compute_score_w_prev_correctness = compute_score
