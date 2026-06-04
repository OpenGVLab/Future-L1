from typing import List, Optional, Dict, Tuple
from .custom_api import get_api_response
import traceback
import time


def judge_wrap_fn(pred: Optional[str], gt: Optional[str], question: Optional[str], repetition_penalty: bool = False) -> Tuple[str, str]:
    if not repetition_penalty:
        sys_prompt = (
            "You are a strict and objective answer judge. Your sole task is to determine if the model's predicted answer matches the ground-truth answer based on the question provided.\n"
            "Important Rules:\n"
            "1. **Absolute Truth**: The Ground Truth Answer is the ONLY standard. Even if you think it is factually incorrect, you must judge based on it. Do not introduce your own knowledge.\n"
            "2. **Multiple Choice**: If the ground truth is an option (e.g., 'A' or 'A. Text'), the prediction is correct if it contains the same option letter OR the exact content of that option.\n"
            "3. **Numeric/Format**: Ignore case, punctuation, and minor formatting differences. Numeric values must be equivalent (e.g., 1.0 = 1).\n"
            "4. **Key Information**: If the ground truth is a long sentence or phrase, the prediction is correct if it captures the essential key information required by the question and ground truth.\n"
            "5. **Open-ended / rubric labels**: If the ground truth is a short rubric, criterion, or reference wording rather than a single token, treat the prediction as correct when it satisfies that rubric to the same standard a careful human grader would use, still anchored to the ground truth text.\n"
            "Output only 'yes' or 'no'."
        )
        user_prompt = (
            f"Question: {question if question is not None else ''}\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n"
            f"Predicted Answer: {pred if pred is not None else ''}\n"
            "Judge: Does the predicted answer match the ground truth? Reply 'yes' or 'no'."
        )
    else:
        sys_prompt = (
            "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, you should:\n"
            "1. Determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
            "and numeric equivalence if applicable. If the prediction is correct, reply with '1'.\n"
            "2. If the prediction is incorrect, then determine if the prediction contains repeatedly illogical contents. Here are two examples:\n"
            "Example (1) 'First, observe the pattern in the top row of the image.  The pattern in the top row is  increasing by one row each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time. ...'\n"
            "Example (2) 'First, observe the pattern in the top row of the provided image.  The pattern in the top row is  \\boxed{A}.  The pattern in the bottom row is  \\boxed{D}.  The pattern in the middle row is  \\boxed{B}.  The pattern in the bottom row is  \\boxed{C}.  The pattern in the middle row is  \\boxed{A}.  The pattern in the bottom row is  \\boxed{D}.  The pattern in the middle row is  \\boxed{B}. ...'\n"
            "If the prediction doesn't contain such contents, reply with '0'. Else, reply with '-1'.\n"
            "Remember, you are only allowed to output '1', '0', or '-1', do not output anything else."
        )
        user_prompt = (
            f"Question: {question if question is not None else ''}\n\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n\n"
            f"Predicted Answer (Check for garbled text/repetition FIRST, then judge correctness):\n{pred if pred is not None else ''}\n\n"
            "Your output (only '1' or '0'): "
        )
    return sys_prompt, user_prompt


def _api_call_wrapper(
    api_name: str,
    pred: Optional[str],
    gt: Optional[str],
    question: Optional[str],
    dataset_name: str,
    client=None,
    api_kwargs: Optional[dict] = None,
    repetition_penalty: bool = False,
    sample_idx: int = -1,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> Optional[bool]:
    """
    Execute API-based judging with retries.

    Returns True (1.0) / False (0.0) / None (all retries failed).
    """
    if pred is None or gt is None or str(pred).strip() == "":
        if verbose:
            print(f"[Judge Sample {sample_idx}] INCORRECT - Empty prediction or ground truth")
        return False

    attempts = 500
    for atpt in range(attempts):
        try:
            sys_prompt, user_prompt = judge_wrap_fn(pred, gt, question, repetition_penalty)

            final_api_kwargs = api_kwargs.copy() if api_kwargs else {}
            if 'temperature' not in final_api_kwargs:
                final_api_kwargs['temperature'] = 0.1

            responses = get_api_response(
                api_name, sys_prompt, [user_prompt],
                client=client, max_retries=500,
                api_url=api_url, api_key=api_key,
                **final_api_kwargs
            )

            if responses and isinstance(responses[0], str) and responses[0].strip():
                t = responses[0].strip().lower()
                if not repetition_penalty:
                    if "yes" in t and "no" not in t:
                        if verbose:
                            print(f"[Judge Sample {sample_idx}] CORRECT - API response: {responses[0][:100]}")
                        return 1.0
                    if "no" in t and "yes" not in t:
                        if verbose:
                            print(f"[Judge Sample {sample_idx}] INCORRECT - API response: {responses[0][:100]}")
                        return 0.0
                    if verbose:
                        print(f"[Judge Sample {sample_idx}] INCORRECT - Ambiguous API response: {responses[0][:200]}")
                    return 0.0
                else:
                    if "1" in t and "0" not in t:
                        if verbose:
                            print(f"[Judge Sample {sample_idx}] CORRECT - API response: {responses[0][:100]}")
                        return 1.0
                    if "0" in t and "1" not in t:
                        if verbose:
                            print(f"[Judge Sample {sample_idx}] INCORRECT - API response: {responses[0][:100]}")
                        return 0.0
                    if verbose:
                        print(f"[Judge Sample {sample_idx}] INCORRECT - Invalid API response: {responses[0][:200]}")
                    return 0.0

            if verbose:
                print(f"[Judge Sample {sample_idx}] Retry {atpt+1}/{attempts} - Empty/invalid response")
                if responses and isinstance(responses[0], str):
                    print(f"  API response was: {responses[0][:200]}")
            continue

        except Exception as e:
            if verbose:
                print(f"[Judge Sample {sample_idx}] Retry {atpt+1}/{attempts} - Exception: {e}")
                if atpt == attempts - 1:
                    traceback.print_exc()
            elif atpt == attempts - 1:
                traceback.print_exc()
            continue

    if verbose:
        print(f"[Judge Sample {sample_idx}] FAILED - All {attempts} attempts exhausted")
    return None


def _strip_boxed_instruction(q: str) -> str:
    if not isinstance(q, str):
        return q
    return (
        q.replace("Put the letter of your choice within \\boxed{}.", "")
        .replace("Put your final answer within \\boxed{}.", "")
        .replace("Given the answer in a single word and put it within \\boxed{}.", "")
        .strip()
    )


def api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gpt-5',
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client=None,
    dataset_name: str = "",
    repetition_penalty: bool = False,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose_per_sample: Optional[bool] = None,
) -> List[int]:
    """
    Batch API-based judging with parallel workers.

    For each (question, pred, gt) triple, calls an external LLM to decide
    whether the prediction matches the ground truth. Falls back to 0
    (incorrect) on API failure or ambiguity.

    Args:
        questions:        List of questions (items can be None).
        preds:            List of model predictions.
        gts:              List of ground-truth answers.
        api_name:         Judge model name.
        api_max_workers:  Max parallel workers (overridable via API_JUDGE_WORKERS env).
        api_kwargs:       Extra kwargs passed to the API caller.
        client:           Optional pre-built OpenAI client.
        dataset_name:     Optional dataset name for logging.
        repetition_penalty: Whether to check for repetitive content.
        api_url:          Custom API base URL.
        api_key:          Custom API key.
        verbose_per_sample: If True, log each sample's judge outcome. If None, use env
        API_JUDGE_VERBOSE=1 to enable; default is batch-level summary only.

    Returns:
        List of floats (1.0 = correct, 0.0 = incorrect).
    """
    import os
    import concurrent.futures as cf

    start_time = time.time()

    if not (len(questions) == len(preds) == len(gts)):
        raise ValueError("Length mismatch: `questions`, `preds`, and `gts` must have the same length.")

    n = len(preds)
    results: List[float] = [0.0] * n

    try:
        questions_wo_inst = [_strip_boxed_instruction(q) for q in questions]
    except NameError:
        questions_wo_inst = questions

    try:
        max_workers = int(os.environ.get("API_JUDGE_WORKERS", api_max_workers))
    except Exception:
        max_workers = api_max_workers

    if verbose_per_sample is None:
        verbose_per_sample = os.environ.get("API_JUDGE_VERBOSE", "").strip().lower() in ("1", "true", "yes")

    empty_pred = sum(1 for p in preds if p is None or (isinstance(p, str) and str(p).strip() == ""))

    if verbose_per_sample:
        print(f"\n{'='*80}")
        print(f"[API Batch Judge] Starting judgment for {n} samples using API '{api_name}'")
        print(f"[API Batch Judge] Max workers: {max_workers}, Repetition penalty: {repetition_penalty}")
        print(f"{'='*80}\n")
    else:
        print(
            f"[API Batch Judge] start n={n} api={api_name!r} workers={max_workers} "
            f"(per-sample logs: API_JUDGE_VERBOSE=1)"
        )

    unexpected_type = 0
    worker_exceptions = 0

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i in range(n):
            fut = ex.submit(
                _api_call_wrapper,
                api_name,
                preds[i],
                gts[i],
                questions_wo_inst[i],
                dataset_name,
                client,
                api_kwargs,
                repetition_penalty,
                i,
                api_url,
                api_key,
                verbose_per_sample,
            )
            futs.append((i, fut))

        for i, fut in futs:
            try:
                r = fut.result()
                if r is None:
                    results[i] = 0.0
                elif isinstance(r, (int, float)):
                    results[i] = float(r)
                else:
                    unexpected_type += 1
                    if verbose_per_sample:
                        print(f"[Judge Sample {i}] WARNING - Unexpected result type {type(r)}: {r}, treating as incorrect")
                    results[i] = 0.0
            except Exception as e:
                worker_exceptions += 1
                if verbose_per_sample:
                    traceback.print_exc()
                    print(f"[Judge Sample {i}] EXCEPTION - Setting to incorrect: {e}")
                results[i] = 0.0

    end_time = time.time()

    correct_count = sum(1 for r in results if r == 1.0)
    incorrect_count = sum(1 for r in results if r == 0.0)
    accuracy = correct_count / n if n > 0 else 0.0

    if verbose_per_sample:
        print(f"\n{'='*80}")
        print(f"[API Batch Judge] Completed in {end_time - start_time:.2f} seconds")
        print(f"[API Batch Judge] Results: {correct_count} correct | {incorrect_count} incorrect")
        print(f"[API Batch Judge] Accuracy: {accuracy:.2%} ({correct_count}/{n})")
        print(f"{'='*80}\n")
    else:
        extra = []
        if empty_pred:
            extra.append(f"empty_pred={empty_pred}")
        if unexpected_type:
            extra.append(f"unexpected_type={unexpected_type}")
        if worker_exceptions:
            extra.append(f"worker_exceptions={worker_exceptions}")
        tail = (" | " + ", ".join(extra)) if extra else ""
        print(
            f"[API Batch Judge] done n={n} in {end_time - start_time:.2f}s | "
            f"correct={correct_count} incorrect={incorrect_count} acc={accuracy:.2%}{tail}"
        )

    return results
