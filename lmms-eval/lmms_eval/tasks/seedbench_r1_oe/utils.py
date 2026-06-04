"""SEED-Bench-R1 open-ended (OE) evaluation aligned with
``TwiFF/eval/SeedBenchR1``:

- Inference text matches ``inference_SeedBenchR1_TwiFF_mp.py`` with
  ``--disable_options`` (``COT_SYSTEM_PROMPT``, strip ``Considering ...``
  prefixes, no MC options).
- Judge matches ``gencot_SeedBenchR1_eval.py`` (system prompt, user message
  layout, ``<ans>`` / ``</think>`` parsing, ``score`` as a
  one-element list).

API / ``.env`` mirror ``twiffbench`` (``OPENAI_API_KEY``, ``OPENAI_API_BASE``,
``LOCAL_LLM`` / ``JUDGE_MODEL``).
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger as eval_logger
from PIL import Image

_CACHE_DIR: Optional[str] = None

# ``inference_SeedBenchR1_TwiFF_mp.py`` — ``COT_SYSTEM_PROMPT`` (verbatim).
COT_SYSTEM_PROMPT = (
    "You are an AI assistant capable of reasoning with visual imagery. You should conduct a detailed analysis of the question. Consider "
    "different angles, potential solutions, and reason through the problem step-by-step with image. After fully reasoning through the problem—"
    "potentially using image-based thinking—provide only a clear, concise, and direct answer to the user's question."
).strip()

# ``gencot_SeedBenchR1_eval.py`` — ``system_prompt_template`` (verbatim).
JUDGE_SYSTEM_PROMPT = """You are a evaluator. You will have to evaluate the model answer based on the question and ground truth answer.
Given:
    Question: The original forecasting question with image originates from the first video frame.
    Ground Truth Answer: The ground truth of the question.
    Model Response Reasoning Chain: The model's reasoning chain. Some models may generate images as part of their reasoning chain. 
    Model Response Answer: The model's answer.
The rating should base on the following rules:
    Answer Accuracy: Score 0-5 based on how well the final answer matches the question requirement and ground truth answer. Full credit requires correctness and completeness; partial or incorrect answers receive lower scores. For answers that differ from the ground truth, appropriately evaluate their reasonableness and assign a score, rather than treating them as entirely incorrect.
Put the score in a list such that output score = [score1], where 'score1' evaluates the Answer.
You will have to give your output in the JSON format (Keep your reasoning concise and short.):
{{
"reasoning": str #the score reasoning
"score": List[int]
}}""".strip()

_QUESTION_PREFIX_STRIPS = (
    "Considering my current observation shown in the image, ",
    "Considering the progress shown in the video and my current observation shown in the image, ",
)


def seedbench_r1_oe_normalize_question_for_model(question: str) -> str:
    """Same prefix removal as ``inference_SeedBenchR1_TwiFF_mp.py`` when ``--disable_options``."""
    q = question or ""
    for p in _QUESTION_PREFIX_STRIPS:
        q = q.replace(p, "")
    return q.strip()


def _resolve_cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is not None:
        return _CACHE_DIR
    yaml_dir = Path(__file__).parent
    for name in (
        "seedbench_r1_l1_oe.yaml",
        "seedbench_r1_l2_oe.yaml",
        "seedbench_r1_l3_oe.yaml",
    ):
        p = yaml_dir / name
        if not p.exists():
            continue
        with open(p, "r") as f:
            raw = f.readlines()
        safe = [ln for ln in raw if "!function" not in ln]
        cfg = yaml.safe_load("".join(safe))
        _CACHE_DIR = os.path.expanduser(cfg["dataset_kwargs"]["cache_dir"])
        return _CACHE_DIR
    raise RuntimeError("seedbench_r1_oe: no task yaml found to resolve cache_dir")


def seedbench_r1_oe_doc_to_visual(doc):
    cache_dir = _resolve_cache_dir()
    video_path = ""
    if len(doc.get("task_progress_metadata") or []) > 0:
        video_path = os.path.join(cache_dir, doc["video_basename"])
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video path:{video_path} does not exist, please check")

    image_path = os.path.join(
        cache_dir, "images", doc["video_source"], doc["current_observation_basename"]
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image path:{image_path} does not exist, please check")

    return [(video_path, image_path)]


def seedbench_r1_oe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """User prompt aligned with ``inference_SeedBenchR1_TwiFF_mp.py`` + ``--disable_options``."""
    cfg = lmms_eval_specific_kwargs or {}
    if "pre_prompt" in cfg:
        head = cfg["pre_prompt"]
    else:
        head = COT_SYSTEM_PROMPT
    question = seedbench_r1_oe_normalize_question_for_model(str(doc.get("question") or ""))
    if "post_prompt" in cfg:
        post_prompt = cfg["post_prompt"]
    else:
        post_prompt = ""
    parts = [str(head).strip(), "", question]
    if str(post_prompt).strip():
        parts.extend(["", str(post_prompt).strip()])
    return "\n".join(parts)


def _golden_letter(doc) -> str:
    return str(doc["golden_choice_idx"]).strip().upper()


def _golden_answer_text(doc) -> str:
    letter = _golden_letter(doc)
    key = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}.get(letter)
    if not key:
        return ""
    return str(doc.get(key) or "").strip()


def _ground_truth_answer_for_judge(doc) -> str:
    """``gencot_SeedBenchR1_eval.py`` uses jsonl ``answer``; HF rows may omit it — fall back to gold option text."""
    a = doc.get("answer")
    if a is not None and str(a).strip() and str(a).strip().lower() != "nan":
        return str(a).strip()
    return _golden_answer_text(doc)


def seedbench_r1_oe_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    """Same gold string family as TwiFF QA jsonl ``answer`` when present."""
    return _ground_truth_answer_for_judge(doc)


def seedbench_r1_oe_split_model_for_judge(model_answer: str) -> Tuple[str, str]:
    """Mirror ``prepare_prompt`` in ``gencot_SeedBenchR1_eval.py`` (``<ans>``, ``</think>``)."""
    if not model_answer:
        return "", ""
    reasoning = re.split(r"<ans>", model_answer)[0].strip()
    answer_match = re.search(r"<ans>(.*?)</ans>", model_answer, re.DOTALL)
    if answer_match:
        final = answer_match.group(1).strip()
        return reasoning, final
    if "</think>" in model_answer:
        tmp = model_answer.split("</think>", 1)
        reasoning = tmp[0]
        tail = tmp[1].strip() if len(tmp) > 1 else ""
        if tail:
            return reasoning.strip(), tail
        return reasoning.strip(), ""
    # BAGEL jsonl path ends with empty tagged answer; lmms models often omit tags — use full text as answer.
    return "", model_answer.strip()


def seedbench_r1_oe_extract_final_answer(s: str) -> str:
    """Final answer span: ``<ans>`` (SeedBenchR1), else ``<answer>`` (common in LMMS)."""
    _, final = seedbench_r1_oe_split_model_for_judge(s)
    if final:
        return final
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return (s or "").strip()


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    image = image.convert("RGB") if image.mode != "RGB" else image
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _serialize_observation_image(doc) -> Optional[Dict[str, str]]:
    cache_dir = _resolve_cache_dir()
    image_path = os.path.join(
        cache_dir, "images", doc["video_source"], doc["current_observation_basename"]
    )
    if not os.path.exists(image_path):
        return None
    pil = Image.open(image_path)
    return {"format": "JPEG", "b64": _pil_to_base64(pil, "JPEG")}


def seedbench_r1_oe_process_results(doc, results):
    pred = results[0] if results else ""
    img = _serialize_observation_image(doc)
    reasoning, final_ans = seedbench_r1_oe_split_model_for_judge(pred)
    # LMMS-style ``<answer>`` (no BAGEL ``<ans>``): split like gencot would for tagged answers.
    if re.search(r"<answer>", pred or "", re.IGNORECASE) and not re.search(r"<ans>", pred or ""):
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", pred, re.DOTALL | re.IGNORECASE)
        if m:
            final_ans = m.group(1).strip()
            reasoning = re.split(r"<answer>", pred, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    elif not final_ans:
        final_ans = (pred or "").strip()
        reasoning = ""
    display_reasoning = reasoning if reasoning else (pred or "").strip()
    gt = _ground_truth_answer_for_judge(doc)
    record = {
        "video_id": doc.get("video_id"),
        "sample_id": doc.get("sample_id"),
        "question": doc.get("question"),
        "answer": gt,
        "golden_letter": _golden_letter(doc),
        "golden_answer_text": _golden_answer_text(doc),
        "raw_model_response": pred,
        "model_reasoning_for_judge": display_reasoning,
        "model_answer_for_judge": final_ans,
        "observation_image": img,
    }
    return {"seedbench_r1_oe_score": record}


def _is_text_only_judge() -> bool:
    """Shared switch: ``JUDGE_TEXT_ONLY=1`` disables image payloads for both
    twiffbench and seedbench_r1_oe, allowing a plain LLM as judge."""
    return os.environ.get("JUDGE_TEXT_ONLY", "").lower() in {"1", "true", "yes"}


def _load_env_file() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _normalize_api_base(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _build_judge_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same structure as ``prepare_prompt`` in ``gencot_SeedBenchR1_eval.py``.

    When ``JUDGE_TEXT_ONLY=1`` the observation image is omitted so a plain
    LLM can serve as judge (shared switch with twiffbench).
    """
    question = record.get("question") or ""
    gt_text = record.get("answer") or ""
    reasoning = record.get("model_reasoning_for_judge") or ""
    model_ans = record.get("model_answer_for_judge") or ""
    img = record.get("observation_image")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"Question: {question}"}]
    if img and img.get("b64") and not _is_text_only_judge():
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{str(img.get('format') or 'jpeg').lower()};base64,{img['b64']}"
            },
        })
    user_content.append({"type": "text", "text": f"\nGround Truth Answer: {gt_text}"})
    user_content.append({"type": "text", "text": "\nModel Response Reasoning Chain: "})
    if reasoning:
        user_content.append({"type": "text", "text": f"{reasoning}"})
    user_content.append({"type": "text", "text": f"\nModel Response Answer: {model_ans}"})

    return [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def _query_judge(messages: List[Dict[str, Any]], max_retries: int = 3):
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_base = _normalize_api_base(os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    model_name = (
        os.environ.get("SEEDBENCH_R1_OE_JUDGE_MODEL")
        or os.environ.get("LOCAL_LLM")
        or os.environ.get("JUDGE_MODEL")
        or "gpt-4o"
    )
    max_tokens = int(os.environ.get("SEEDBENCH_R1_OE_JUDGE_MAX_TOKENS", "1024"))
    temperature = float(os.environ.get("SEEDBENCH_R1_OE_JUDGE_TEMPERATURE", "0.0"))

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        eval_logger.error(f"openai SDK not available, skipping judge: {exc}")
        return None

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            last_err = exc
            eval_logger.warning(
                f"[seedbench_r1_oe judge] attempt {attempt + 1}/{max_retries} failed: {exc}"
            )
    eval_logger.error(f"[seedbench_r1_oe judge] giving up after {max_retries} retries: {last_err}")
    return None


def seedbench_r1_oe_aggregate_results(results, args=None):
    skip = os.environ.get("SEEDBENCH_R1_OE_SKIP_JUDGE", "").lower() in {"1", "true", "yes"}
    if skip:
        eval_logger.warning(
            "[seedbench_r1_oe] SEEDBENCH_R1_OE_SKIP_JUDGE set — returning 0.0; "
            "re-run without it to call the judge."
        )
        return 0.0

    scores: List[int] = []
    failures = 0
    for rec in results:
        try:
            messages = _build_judge_messages(rec)
        except Exception as exc:
            eval_logger.warning(f"[seedbench_r1_oe] prompt build failed: {exc}")
            failures += 1
            continue
        verdict = _query_judge(messages)
        if not verdict or "score" not in verdict:
            failures += 1
            continue
        try:
            raw_sc = verdict["score"]
            if isinstance(raw_sc, list) and raw_sc:
                sc = int(raw_sc[0])
            else:
                sc = int(raw_sc)
            sc = max(0, min(5, sc))
            scores.append(sc)
        except (TypeError, ValueError):
            failures += 1
            continue

    n_ok = len(scores)
    n_total = len(results)
    if n_ok == 0:
        eval_logger.error(
            f"[seedbench_r1_oe] judge produced no valid scores (failures={failures}/{n_total})"
        )
        return 0.0

    mean_score = sum(scores) / n_ok
    eval_logger.info(
        f"[seedbench_r1_oe] scored {n_ok}/{n_total} (failed={failures}); "
        f"mean_answer_score={mean_score:.3f}/5"
    )
    return mean_score
