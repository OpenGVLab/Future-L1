"""TwiFF-Bench task utils for lmms-eval.

This file ports the inference + GPT-judge pipeline from
``/path/to/your/TwiFFBench`` so the
benchmark can be evaluated against Qwen-VL series (or any lmms-eval model)
inside ``VideoL1/lmms-eval``.

Inference path (lmms-eval):
    doc_to_visual    -> question_images (PIL list, position-faithful to the
                        original clip = question_images_index)
    doc_to_text      -> COT_SYSTEM_PROMPT + question (with <image> tokens
                        stripped because Qwen-VL prepends visuals; the
                        original frame_<i> labels are kept) + post_prompt
    process_results  -> packs the raw model response together with all the
                        fields the GPT judge needs.
    aggregate_results-> Replicates ``gencot_vqa_eval.py``: builds the same
                        evaluation prompt (verbatim system prompt, same
                        message structure with question/reasoning images and
                        ground-truth answer) and queries an OpenAI-compatible
                        chat completion API loaded from
                        ``VideoL1/lmms-eval/.env``. Returns the macro-average
                        of (reasoning_score + answer_score) on a 0-5 scale.

The reasoning_images / reasoning_images_index fields are kept on the doc but
NOT fed to the model under test (Qwen-VL has no image-generation head). They
are only consumed by the judge as in the original pipeline.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger as eval_logger
from PIL import Image

# ---------------------------------------------------------------------------
# Constants copied verbatim from the upstream TwiFFBench scripts so the
# evaluation protocol stays bit-identical to the BAGEL reference run.
# ---------------------------------------------------------------------------

# COT_SYSTEM_PROMPT — inference_benchmark_TwiFF_mp.py
COT_SYSTEM_PROMPT = (
    "You are an AI assistant capable of reasoning with visual imagery. "
    "You should conduct a detailed analysis of the question. Consider "
    "different angles, potential solutions, and reason through the problem "
    "step-by-step with image. After fully reasoning through the problem—"
    "potentially using image-based thinking—provide only a clear, concise, "
    "and direct answer to the user's question."
).strip()

# Judge system prompt — gencot_vqa_eval.py
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. You will have to evaluate the model "
    "response reasoning chain and answer based on the reference reasoning "
    "chain and ground truth answer.\n"
    "Given:\n"
    "    Question: The original forecasting question with image originates "
    "from the first video frame.\n"
    "    Reference Reasoning Chain: What actually happened, as a reference "
    "for the rationality of the reasoning chain.\n"
    "    Ground Truth Answer: The ground truth of the question.\n"
    "    Model Response Reasoning Chain: The model's reasoning chain.\n"
    "    Model Response Answer: The model's answer.\n"
    "The rating should base on the following rules:\n"
    "    Reasoning Chain Quality: Score 0-5 based on the logical coherence, "
    "completeness, and relevance of the reasoning (including appropriate "
    "use of multimodal information if present). The chain need not match "
    "the reference exactly but must be valid and support the final answer.\n"
    "    Answer Accuracy: Score 0-5 based on how well the final answer "
    "matches the ground truth answer. Full credit requires correctness and "
    "completeness; partial or incorrect answers receive lower scores.\n"
    "Put the score in a list such that output score = [score1, score2], "
    "where 'score1' evaluates the\n"
    "Reasoning Chain and 'score2' evaluates the Answer.\n"
    "You will have to give your output in the JSON format (Keep your "
    "reasoning concise and short.):\n"
    "{\n"
    '"reasoning": str #the score reasoning\n'
    '"score": List[int]\n'
    "}"
).strip()


# Cached config dict (resolved once on first call).
_TASK_CONFIG: Dict[str, Any] | None = None


def _load_task_config() -> Dict[str, Any]:
    global _TASK_CONFIG
    if _TASK_CONFIG is not None:
        return _TASK_CONFIG
    yaml_path = Path(__file__).parent / "twiffbench.yaml"
    with open(yaml_path, "r") as f:
        raw = f.readlines()
    safe = [ln for ln in raw if "!function" not in ln]
    _TASK_CONFIG = yaml.safe_load("".join(safe))
    return _TASK_CONFIG


# ---------------------------------------------------------------------------
# Helpers shared with the upstream pipeline.
# ---------------------------------------------------------------------------


def _to_pil(img: Any) -> Image.Image:
    """Coerce a HF datasets Image cell into a PIL.Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"]))
        if img.get("path"):
            return Image.open(img["path"])
    raise TypeError(f"Cannot decode image cell of type {type(img)!r}")


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    image = image.convert("RGB") if image.mode != "RGB" else image
    buffered = BytesIO()
    image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _coerce_twiff_judge_score_pair(score: Any) -> Optional[List[int]]:
    """Flatten judge ``score`` to ``[reasoning_int, answer_int]``.

    Some OpenAI-compatible models return nested lists (e.g. ``[[3, 4]]`` or
    ``[3, [4, 5]]``) despite ``response_format=json_object``. We walk scalars
    in depth-first order and take the first two integers.
    """
    out: List[int] = []

    def walk(x: Any) -> None:
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
            return
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            out.append(x)
            return
        if isinstance(x, float):
            out.append(int(round(x)))
            return
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return
            try:
                out.append(int(round(float(s))))
            except ValueError:
                return

    walk(score)
    if len(out) >= 2:
        return [out[0], out[1]]
    return None


def _strip_image_placeholders(text: str) -> str:
    """Remove ``<image>`` markers (Qwen-VL prepends visuals automatically)."""
    return re.sub(r"<image>\s*", "", text)


# ---------------------------------------------------------------------------
# FutureL1 / TwiFF-RL model output — normalize before GPT judge.
# ---------------------------------------------------------------------------

_LATENT_TOK_RE = re.compile(r"<\|latent(?:_start|_end)?\|>|<\|latent\|>", re.IGNORECASE)
_JUDGE_TEMPLATE_TOK_RE = re.compile(
    r"<\|im_(?:start|end)\|>|<\|endoftext\|>|<\|redacted_im_end\|>",
    re.IGNORECASE,
)
_REASON_XML_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
_ANSWER_SPAN_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
# Any leftover ``<reason>`` / ``</reason>`` after merge (unclosed tags, attrs, spacing).
_REASON_OPEN_RE = re.compile(r"<\s*reason\b[^>]*>", re.IGNORECASE)
_REASON_CLOSE_RE = re.compile(r"</\s*reason\s*>", re.IGNORECASE)


def _legacy_ans_tags_to_answer(text: str) -> str:
    """TwiFF-Bench releases still use ``<ans>...</ans>``; unify to ``<answer>``."""
    if not text:
        return text
    t = re.sub(r"</ans\s*>", "</answer>", text, flags=re.IGNORECASE)
    t = re.sub(r"<ans\s*>", "<answer>", t, flags=re.IGNORECASE)
    return t


def twiffbench_extract_answer_tag_span(text: str) -> str:
    """Return inner text of the first ``<answer>...</answer>`` span (``<ans>`` OK).

    Used by rule-based answer-accuracy metrics. Does **not** fall back to
    ``</think>`` tails — only explicit answer tags count.
    """
    if not text:
        return ""
    normalized = _legacy_ans_tags_to_answer(text)
    m = _ANSWER_SPAN_RE.search(normalized)
    return m.group(1).strip() if m else ""


def _scrub_remaining_reason_tags(text: str) -> str:
    """Remove all ``<reason>...</reason>`` markup so the GPT judge sees plain text."""
    if not text:
        return text
    t = _REASON_OPEN_RE.sub("", text)
    t = _REASON_CLOSE_RE.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def twiffbench_sanitize_model_response_for_judge(text: str) -> str:
    """Strip FutureL1 latent / template tokens, merge consecutive ``<reason>``
    blocks into one reasoning paragraph, normalize the final span to
    ``<answer>...</answer>`` (legacy ``<ans>`` is rewritten first), then drop
    any remaining ``<reason>`` / ``</reason>`` so the judge payload is tag-free."""
    if not text:
        return text
    cleaned = _LATENT_TOK_RE.sub("", text)
    cleaned = _JUDGE_TEMPLATE_TOK_RE.sub("", cleaned)
    cleaned = _legacy_ans_tags_to_answer(cleaned)

    answ_m = _ANSWER_SPAN_RE.search(cleaned)
    if answ_m:
        head = cleaned[: answ_m.start()].rstrip()
        ans_inner = answ_m.group(1).strip()
    else:
        head = cleaned.strip()
        ans_inner = ""

    head = _merge_reason_xml_blocks(head)
    head = _scrub_remaining_reason_tags(head)
    ans_inner = _scrub_remaining_reason_tags(ans_inner)

    if ans_inner:
        out = f"{head}\n<answer>{ans_inner}</answer>".strip() if head else f"<answer>{ans_inner}</answer>"
        return _scrub_remaining_reason_tags(out)
    return head


def _merge_reason_xml_blocks(fragment: str) -> str:
    """Concatenate all ``<reason>...</reason>`` inner texts; keep any leftover
    non–reason-tag text after the last closing tag (e.g. stray whitespace)."""
    if not fragment:
        return ""
    parts = [m.group(1).strip() for m in _REASON_XML_RE.finditer(fragment) if m.group(1).strip()]
    rest = _REASON_XML_RE.sub("", fragment).strip()
    merged = "\n\n".join(parts)
    if rest:
        merged = f"{merged}\n\n{rest}".strip() if merged else rest
    return merged.strip()


def _sanitize_record_for_judge(rec: Dict[str, Any]) -> None:
    """In-place: normalize ``model_response`` tail for judge consumption."""
    mr = rec.get("model_response")
    if isinstance(mr, list) and mr:
        tail = mr[-1] if mr else ""
        cleaned = twiffbench_sanitize_model_response_for_judge(tail or "")
        rec["model_response"] = list(mr[:-1]) + [cleaned]
        rec["raw_response_clean"] = cleaned
    elif isinstance(rec.get("raw_response"), str):
        rec["raw_response_clean"] = twiffbench_sanitize_model_response_for_judge(rec["raw_response"])


# ---------------------------------------------------------------------------
# lmms-eval task hooks.
# ---------------------------------------------------------------------------


def twiffbench_doc_to_visual(doc):
    """Return the list of question images as PIL.

    These correspond, in the original BAGEL code, to the frames at
    ``question_images_index`` (i.e. ``frames`` in the upstream jsonl) sampled
    from the source mp4 — but they are pre-rendered in the parquet, so we
    use them directly.
    """
    images = doc.get("question_images") or []
    return [_to_pil(im) for im in images]


def twiffbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    cfg = lmms_eval_specific_kwargs or {}
    system_prompt = cfg.get("system_prompt", COT_SYSTEM_PROMPT)
    pre_prompt = (cfg.get("pre_prompt") or "").strip()
    post_prompt = (cfg.get("post_prompt") or "").strip()

    question = _strip_image_placeholders(doc["question"]).strip()
    parts = [system_prompt, ""]
    if pre_prompt:
        parts += [pre_prompt, ""]
    parts.append(question)
    if post_prompt:
        parts += ["", post_prompt]
    return "\n".join(parts)


def _serialize_images(images) -> List[Dict[str, str]]:
    """Pack PIL images as base64-jpeg dicts so they survive json round-trips
    when lmms-eval stores ``process_results`` payloads (e.g. log_samples)."""
    out = []
    for im in images or []:
        pil = _to_pil(im)
        out.append({"format": "JPEG", "b64": _pil_to_base64(pil, "JPEG")})
    return out


def twiffbench_process_results(doc, results):
    """Pack everything the judge needs into a single dict per sample."""
    pred = results[0] if results else ""
    record = {
        "video": doc.get("video"),
        "question": doc.get("question"),
        "answer": doc.get("answer"),
        # Keep upstream field names (frames / recon_frames) as aliases.
        "frames": list(doc.get("question_images_index") or []),
        "recon_frames": list(doc.get("reasoning_images_index") or []),
        "question_images": _serialize_images(doc.get("question_images")),
        "reasoning_images": _serialize_images(doc.get("reasoning_images")),
        # The original gencot_vqa_eval.py expects model_response to be a
        # list — emulate that shape with a single text element.
        "model_response": [pred],
        "raw_response": pred,
    }
    return {"twiffbench_score": record}


# ---------------------------------------------------------------------------
# Judge — replicates gencot_vqa_eval.py against the .env API.
# ---------------------------------------------------------------------------


def _load_env_file() -> None:
    """Load VideoL1/lmms-eval/.env into os.environ if not already set."""
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


def _is_text_only_judge() -> bool:
    """Shared switch: ``JUDGE_TEXT_ONLY=1`` disables image payloads for both
    twiffbench and seedbench_r1_oe, allowing a plain LLM as judge."""
    return os.environ.get("JUDGE_TEXT_ONLY", "").lower() in {"1", "true", "yes"}


def twiffbench_log_judge_mode() -> None:
    """Print a visible banner when ``JUDGE_TEXT_ONLY=1`` (call after ``_load_env_file``)."""
    if not _is_text_only_judge():
        return
    banner = (
        "\n"
        + "=" * 72
        + "\n"
        + "  TWIFFBENCH JUDGE: TEXT-ONLY MODE (JUDGE_TEXT_ONLY=1)\n"
        + "  Question / reference-reasoning images are NOT sent to the judge API.\n"
        + "  Only text (with <image>/<rimage> tokens stripped) is used for scoring.\n"
        + "=" * 72
        + "\n"
    )
    print(banner, flush=True)
    eval_logger.warning(
        "[twiffbench] JUDGE_TEXT_ONLY=1 — text-only judge (no images sent to the API)"
    )


def _build_judge_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reimplements ``prepare_prompt`` in gencot_vqa_eval.py.

    The original code reads frames from a video using the union
    ``clip = frames + recon_frames`` of indices. Here we already have the
    matching images pre-decoded; concatenating ``question_images`` and
    ``reasoning_images`` reproduces the exact same sequence.

    When ``JUDGE_TEXT_ONLY=1`` all ``image_url`` blocks are omitted and
    ``<image>`` / ``<rimage>`` special tokens are stripped from the text
    so a plain LLM can serve as judge without receiving any images.
    """
    text_only = _is_text_only_judge()

    question = record["question"]
    answer = _legacy_ans_tags_to_answer(record.get("answer") or "")
    frames = record["frames"]
    recon_frames = record["recon_frames"]
    question_imgs_b64 = record["question_images"]
    reasoning_imgs_b64 = record["reasoning_images"]
    model_response = list(record["model_response"])

    # Build a flat list of {clip-index -> base64 image} matching the upstream
    # ``question_images = read_frames_decord(..., clip=frames+recon_frames)``.
    all_images_b64 = list(question_imgs_b64) + list(reasoning_imgs_b64)
    num_input_images = len(frames)
    num_output_images = len(recon_frames)

    if text_only:
        # Strip all <image> / <rimage> tokens and build a single text block.
        clean_question = re.sub(r"<image>\s*", "", question).strip()
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Question: {clean_question}"}
        ]
        image_index = num_input_images  # skip image loop below
    else:
        user_content = [{"type": "text", "text": "Question: "}]
        image_index = 0

    if not text_only:
        prompt_parts = question
        image_cnt = question.count("<image>")
        for i in range(image_cnt):
            if image_index == num_input_images:
                break
            pre, prompt_parts = prompt_parts.split("<image>", 1)
            if pre.strip():
                user_content.append({"type": "text", "text": pre})
            img = all_images_b64[image_index]
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{img['format'].lower()};base64,{img['b64']}"},
            })
            image_index += 1
        if prompt_parts.strip():
            user_content.append({"type": "text", "text": prompt_parts})

    # --- Reference reasoning chain (split on <answer>...</answer>) ---
    reasoning = re.split(r"(?i)<answer>", answer, maxsplit=1)[0].strip()
    answer_match = _ANSWER_SPAN_RE.search(answer)
    if answer_match:
        gt_answer = answer_match.group(1).strip()
    else:
        gt_answer = answer

    user_content.append({"type": "text", "text": "\nReference Reasoning Chain: "})

    if text_only:
        clean_reasoning = re.sub(r"<rimage>\s*", "", reasoning).strip()
        if clean_reasoning:
            user_content.append({"type": "text", "text": clean_reasoning})
    else:
        prompt_parts = reasoning
        rimg_cnt = reasoning.count("<rimage>")
        for i in range(rimg_cnt):
            if image_index == num_input_images + num_output_images:
                break
            pre, prompt_parts = prompt_parts.split("<rimage>", 1)
            if pre.strip():
                user_content.append({"type": "text", "text": pre})
            img = all_images_b64[image_index]
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{img['format'].lower()};base64,{img['b64']}"},
            })
            image_index += 1
        if prompt_parts.strip():
            user_content.append({"type": "text", "text": prompt_parts})
    user_content.append({"type": "text", "text": f"\nGround Truth Answer: {answer}"})

    # --- Model response (text-only for Qwen-VL) ---
    raw_model_answer = _legacy_ans_tags_to_answer(model_response[-1] if model_response else "")
    reasoning_part = re.split(r"(?i)<answer>", raw_model_answer, maxsplit=1)[0].strip()
    m_match = _ANSWER_SPAN_RE.search(raw_model_answer)
    if m_match:
        model_answer = m_match.group(1).strip()
    elif "</think>" in raw_model_answer:
        a, b = raw_model_answer.split("</think>", 1)
        reasoning_part = a
        model_answer = b.strip()
    else:
        model_answer = ""

    reasoning_part = _scrub_remaining_reason_tags(reasoning_part)

    user_content.append({"type": "text", "text": "\nModel Response Reasoning Chain: "})
    user_content.append({"type": "text", "text": reasoning_part})
    user_content.append({"type": "text", "text": f"\nModel Response Answer: {model_answer}"})

    return [
        {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def _normalize_api_base(base: str) -> str:
    """The .env may point at the chat-completions endpoint directly. The
    OpenAI SDK expects the v1 root, so trim a trailing /chat/completions."""
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _query_judge(messages, max_retries: int = 3):
    """Call the OpenAI-compatible API loaded from the repo .env file."""
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_base = _normalize_api_base(os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    model_name = os.environ.get("LOCAL_LLM") or os.environ.get("JUDGE_MODEL") or "gpt-4o"
    max_tokens = int(os.environ.get("TWIFF_JUDGE_MAX_TOKENS", "1024"))
    temperature = float(os.environ.get("TWIFF_JUDGE_TEMPERATURE", "0.0"))

    try:
        from openai import OpenAI  # lazy import — only needed for aggregate
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
                f"[twiffbench judge] attempt {attempt + 1}/{max_retries} failed: {exc}"
            )
    eval_logger.error(f"[twiffbench judge] giving up after {max_retries} retries: {last_err}")
    return None


def twiffbench_aggregate_results(results, args=None):
    """Aggregate per-sample records into final scores using the GPT judge.

    Returns the mean of (reasoning_score + answer_score) / 2 across all
    successfully scored samples (0-5 scale).
    """
    skip_judge = os.environ.get("TWIFF_SKIP_JUDGE", "").lower() in {"1", "true", "yes"}
    if skip_judge:
        eval_logger.warning(
            "[twiffbench] TWIFF_SKIP_JUDGE set — emitting placeholder score 0.0; "
            "use the standalone tools/twiffbench/post_score.py to score later."
        )
        return 0.0

    _load_env_file()
    twiffbench_log_judge_mode()

    reasoning_scores: List[int] = []
    answer_scores: List[int] = []
    per_class: Dict[str, List[int]] = defaultdict(list)
    failures = 0

    for rec in results:
        try:
            mr = rec.get("model_response")
            if isinstance(mr, list) and mr:
                rec_for_judge = dict(rec)
                rec_for_judge["model_response"] = list(mr[:-1]) + [
                    twiffbench_sanitize_model_response_for_judge(mr[-1] or "")
                ]
            else:
                rec_for_judge = rec
            messages = _build_judge_messages(rec_for_judge)
        except Exception as exc:
            eval_logger.warning(f"[twiffbench] prompt build failed: {exc}")
            failures += 1
            continue

        verdict = _query_judge(messages)
        if not verdict or "score" not in verdict:
            failures += 1
            continue
        pair = _coerce_twiff_judge_score_pair(verdict.get("score"))
        if pair is None:
            failures += 1
            continue
        reasoning_scores.append(pair[0])
        answer_scores.append(pair[1])
        # Optional per-classification breakdown if meta_data carries it.
        # (Not always populated; safe to ignore.)
        meta = rec.get("meta_data") or {}
        cls = meta.get("classification") if isinstance(meta, dict) else None
        if cls:
            per_class[cls].append((pair[0] + pair[1]) / 2.0)

    n_ok = len(reasoning_scores)
    n_total = len(results)
    if n_ok == 0:
        eval_logger.error(
            f"[twiffbench] judge produced no valid scores (failures={failures}/{n_total})"
        )
        return 0.0

    mean_reason = sum(reasoning_scores) / n_ok
    mean_answer = sum(answer_scores) / n_ok
    overall = (mean_reason + mean_answer) / 2.0

    eval_logger.info(
        f"[twiffbench] scored {n_ok}/{n_total} (failed={failures}); "
        f"reasoning={mean_reason:.3f}/5  answer={mean_answer:.3f}/5  overall={overall:.3f}/5"
    )
    for cls, scs in per_class.items():
        eval_logger.info(f"[twiffbench]   class={cls}: {sum(scs) / len(scs):.3f}/5  (n={len(scs)})")

    return overall
