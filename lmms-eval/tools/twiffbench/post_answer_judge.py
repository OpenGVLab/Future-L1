#!/usr/bin/env python3
"""GPT-judge TwiFF-Bench answer accuracy on ``<answer>`` tag spans only.

Unlike ``post_score.py`` (full reasoning-chain + answer + images), this script
extracts the inner text of GT ``<answer>``/``<ans>`` and model ``<answer>``,
then asks the judge to score **only** those two sentences (0–5). This aligns
with TwiFF ``score[1]`` / Answer Accuracy but removes reasoning-chain context.

Reads API config from ``Future-L1/lmms-eval/.env`` (same as post_score.py).

Usage:
    python tools/twiffbench/post_answer_judge.py \\
        --input logs_twiffbench/.../samples_twiffbench_*.jsonl \\
        --concurrency 32

Writes ``<input_stem>.answer_judge.jsonl`` and ``*.answer_judge.summary.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOOLS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "twiff_answer_acc", _TOOLS_DIR / "post_answer_acc.py"
)
_acc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_acc)

extract_answer_tag_span = _acc.extract_answer_tag_span
sanitize_model_response_for_answer = _acc.sanitize_model_response_for_answer
_record_from_log_line = _acc._record_from_log_line
_model_response_text = _acc._model_response_text

REPO_ROOT = Path(__file__).resolve().parents[2]

ANSWER_JUDGE_SYSTEM = """You are a strict evaluator for answer accuracy only.
You will compare the model's final answer to the ground truth answer.
Do not evaluate any reasoning chain — score only whether the model answer
matches the ground truth in correctness and completeness.

Rating rule (Answer Accuracy):
    Score 0-5 based on how well the model answer matches the ground truth.
    Full credit (5) requires the answer to be correct and complete;
    partially correct or incorrect answers receive lower scores.

Output JSON (keep reasoning concise):
{
  "reasoning": str,
  "score": int
}
""".strip()


def _load_env_file() -> None:
    env_path = REPO_ROOT / ".env"
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


def _coerce_score(val: Any) -> Optional[int]:
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(round(val))
    if isinstance(val, list):
        for x in val:
            s = _coerce_score(x)
            if s is not None:
                return s
        return None
    if isinstance(val, str):
        try:
            return int(round(float(val.strip())))
        except ValueError:
            return None
    return None


def _build_answer_judge_messages(gt_span: str, pred_span: str) -> List[Dict[str, Any]]:
    user_text = (
        f"Ground Truth Answer: {gt_span}\n\n"
        f"Model Response Answer: {pred_span}"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": ANSWER_JUDGE_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def _prepare_spans(
    rec: Dict[str, Any],
    *,
    strip_latent: bool,
) -> Tuple[str, str, bool, bool]:
    gt_raw = rec.get("answer") or ""
    pred_raw = _model_response_text(rec)
    if strip_latent:
        pred_raw = sanitize_model_response_for_answer(pred_raw)
    gt_span = extract_answer_tag_span(gt_raw)
    pred_span = extract_answer_tag_span(pred_raw)
    return gt_span, pred_span, bool(gt_span), bool(pred_span)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TwiFF-Bench GPT judge on <answer> spans only")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=int(os.environ.get("TWIFF_JUDGE_MAX_TOKENS", 512)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TWIFF_JUDGE_TEMPERATURE", 0.0)))
    p.add_argument(
        "--strip-latent",
        dest="strip_latent",
        action="store_true",
        default=True,
    )
    p.add_argument("--no-strip-latent", dest="strip_latent", action="store_false")
    p.add_argument("--suffix", default="", help="e.g. dpsk -> <stem>.answer_judge.dpsk.jsonl")
    return p.parse_args()


async def _judge_one(
    client,
    sem: asyncio.Semaphore,
    model: str,
    args: argparse.Namespace,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    async with sem:
        gt_span = payload["gt_answer_span"]
        pred_span = payload["pred_answer_span"]
        if not gt_span or not pred_span:
            payload["judge_error"] = "missing_gt_or_pred_answer_tag"
            return payload

        messages = _build_answer_judge_messages(gt_span, pred_span)
        for attempt in range(args.max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                )
                verdict = json.loads(resp.choices[0].message.content)
                payload["judge_reasoning"] = verdict.get("reasoning")
                score = _coerce_score(verdict.get("score"))
                if score is None:
                    raise ValueError(f"invalid judge score: {verdict.get('score')!r}")
                payload["judge_answer_score"] = score
                return payload
            except Exception as exc:
                if attempt == args.max_retries - 1:
                    payload["judge_error"] = str(exc)
                    return payload
    return payload


async def _amain() -> None:
    args = parse_args()
    inp = Path(args.input).expanduser().resolve()
    suffix_part = f".{args.suffix}" if args.suffix else ""
    out_path = (
        Path(args.output)
        if args.output
        else inp.with_name(f"{inp.stem}.answer_judge{suffix_part}.jsonl")
    )
    summary_path = (
        Path(args.summary)
        if args.summary
        else out_path.with_name(f"{out_path.stem}.summary.json")
    )

    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_base = _normalize_api_base(os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    model = os.environ.get("LOCAL_LLM") or os.environ.get("JUDGE_MODEL") or "gpt-4o"

    from openai import AsyncOpenAI  # lazy

    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=600)
    sem = asyncio.Semaphore(args.concurrency)

    payloads: List[Dict[str, Any]] = []
    n_skip = 0
    with open(inp, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec, doc_id = _record_from_log_line(line)
            if rec is None:
                n_skip += 1
                continue
            gt_span, pred_span, has_gt, has_pred = _prepare_spans(
                rec, strip_latent=args.strip_latent
            )
            row: Dict[str, Any] = {
                "video": rec.get("video"),
                "gt_answer_span": gt_span,
                "pred_answer_span": pred_span,
                "has_gt_answer_tag": has_gt,
                "has_pred_answer_tag": has_pred,
            }
            if doc_id is not None:
                row["doc_id"] = doc_id
            meta = rec.get("meta_data")
            if isinstance(meta, dict) and meta.get("classification"):
                row["classification"] = meta["classification"]
            payloads.append(row)

    print(
        f"[twiffbench/post_answer_judge] {len(payloads)} samples, "
        f"judge={model} @ {api_base}, concurrency={args.concurrency}"
    )
    print(f"[twiffbench/post_answer_judge] output -> {out_path}")

    tasks = [_judge_one(client, sem, model, args, p) for p in payloads]
    results: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm  # type: ignore

        pbar = tqdm(total=len(tasks), desc="answer_judge", dynamic_ncols=True)
    except ImportError:
        pbar = None

    n_ok_run = 0
    scores_run: List[int] = []
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        results.append(rec)
        sc = rec.get("judge_answer_score")
        if sc is not None:
            n_ok_run += 1
            scores_run.append(int(sc))
        elif rec.get("judge_error") and pbar is not None:
            tqdm.write(
                f"[answer_judge] doc_id={rec.get('doc_id', '?')}: {rec['judge_error']}"
            )
        if pbar is not None:
            if n_ok_run:
                pbar.set_postfix(
                    ok=n_ok_run,
                    mean=f"{sum(scores_run) / n_ok_run:.2f}/5",
                )
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores: List[int] = []
    n_ok = 0
    n_err = 0
    n_missing_tag = 0
    with open(out_path, "w") as fout:
        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("judge_answer_score") is not None:
                n_ok += 1
                scores.append(int(rec["judge_answer_score"]))
            elif rec.get("judge_error") == "missing_gt_or_pred_answer_tag":
                n_missing_tag += 1
            else:
                n_err += 1

    mean_score = sum(scores) / n_ok if n_ok else None
    per_class: Dict[str, List[int]] = defaultdict(list)
    for rec in results:
        sc = rec.get("judge_answer_score")
        if sc is not None and rec.get("classification"):
            per_class[rec["classification"]].append(int(sc))

    summary: Dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "input": str(inp),
        "output": str(out_path.resolve()),
        "judge_model": model,
        "api_base": api_base,
        "metric": "answer_accuracy_on_answer_tag_spans_only",
        "score_scale": "0-5",
        "n_records": len(results),
        "n_skipped_input_lines": n_skip,
        "n_scored_ok": n_ok,
        "n_scored_err": n_err,
        "n_missing_answer_tag": n_missing_tag,
        "answer_score_mean": mean_score,
    }
    if per_class:
        summary["per_classification"] = {
            cls: {"n": len(v), "mean": sum(v) / len(v)}
            for cls, v in sorted(per_class.items())
        }

    with open(summary_path, "w") as fsum:
        json.dump(summary, fsum, indent=2, ensure_ascii=False)
        fsum.write("\n")

    print(f"[twiffbench/post_answer_judge] summary -> {summary_path}")
    if mean_score is not None:
        print(
            f"[twiffbench/post_answer_judge] answer_score_mean: "
            f"{n_ok}/{len(results)} = {mean_score:.3f}/5"
        )
    else:
        print("[twiffbench/post_answer_judge] no valid scores")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
